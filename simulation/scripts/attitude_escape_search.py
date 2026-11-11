"""Refine weighted robot paths by explicitly turning before high-risk intervals.

An RRT* tree need not discover a useful wrist posture in a finite run. This
postprocessor locates expensive segments, inserts wrist and whole-arm posture
candidates, checks their complete changed edges, and compares the original
time/persistence objective. Source weights and endpoints remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from plan_watering_path import ARM_JOINTS, CollisionChecker, timed_path_score
from plant_scene import read_json, write_json
from trajectory_tools import time_parameterize


def ranked_segments(checker, path, samples=21):
    """Find costly transitions, including the final approach if it is costly."""
    rows = []
    for segment, (a, b) in enumerate(zip(path[:-1], path[1:])):
        risk = [checker.check((1-u)*a + u*b)[1] for u in np.linspace(0, 1, samples)]
        rows.append(dict(segment=segment, peak_risk=float(max(risk)),
                         mean_risk=float(np.trapezoid(risk, dx=1/(samples-1))),
                         peak_fraction=float(np.argmax(risk)/(samples-1))))
    return sorted(rows, key=lambda row: row["mean_risk"], reverse=True)


def changed_edge_cost(checker, a, q, b, step=.04):
    """Score both continuous joint edges; a safe inserted pose alone is insufficient."""
    first = checker.edge_cost(a, q, max_step_rad=step)
    if not first[0]:
        return float("inf")
    second = checker.edge_cost(q, b, max_step_rad=step)
    return first[1] + second[1] if second[0] else float("inf")


def pose_candidates(a, b, lower, upper, peak_fraction):
    """Deterministic wrist sweeps plus whole-arm escape proposals.

    The wrist sweep runs first. Pan/lift/elbow proposals supplement it so this
    works when rotating the tool alone cannot clear a plant. Each candidate is
    an arm posture, not an independent rotation of a rigidly held nozzle.
    """
    fractions = np.unique(np.clip([.35, .5, .65, .8, peak_fraction, peak_fraction+.2], .15, .85))
    offsets = []
    for flex in (-45, -30, -15, 0, 15, 30):
        for roll in (-45, -20, 0, 20, 45):
            if flex or roll:
                offsets.append(np.deg2rad([0, 0, 0, flex, roll]))
    # All five arm joints are available; the sixth joint holds the nozzle closed.
    for joint in (0, 1, 2):
        for delta in (-20, 20):
            for flex, roll in ((-15, 0), (0, -30), (0, 30), (15, 0)):
                offset = np.zeros(5)
                offset[joint] = delta
                offset[3:] = [flex, roll]
                offsets.append(np.deg2rad(offset))
    for fraction in fractions:
        anchor = (1-fraction)*a + fraction*b
        for offset in offsets:
            q = anchor + offset
            # Reject out-of-range proposals instead of changing their meaning by clipping.
            if np.any(q < lower) or np.any(q > upper):
                continue
            yield float(fraction), q


def refine_candidates(checker, base, budget_seconds=180., top_k=8, progress=None):
    ranked = ranked_segments(checker, base)
    started = time.perf_counter()
    candidates = []
    attempts = 0
    # Only expensive segments need refinement. The exact endpoints are fixed.
    peak = max((row["mean_risk"] for row in ranked), default=0.)
    for row in ranked[:2]:
        if row["mean_risk"] <= 1e-8 or row["mean_risk"] < .15*peak:
            continue
        segment = row["segment"]
        a, b = base[segment:segment+2]
        for fraction, q in pose_candidates(a, b, checker.q_lo, checker.q_hi, row["peak_fraction"]):
            if attempts and time.perf_counter()-started >= budget_seconds:
                break
            attempts += 1
            cost = changed_edge_cost(checker, a, q, b, step=.04)
            if np.isfinite(cost):
                path = np.vstack([base[:segment+1], q, base[segment+1:]])
                candidates.append(dict(path=path, segment=segment, fraction=fraction,
                                       inserted_q_rad=q.tolist(), changed_edge_cost=float(cost)))
            if progress and (attempts % 20 == 0):
                progress(dict(attempts=attempts, feasible=len(candidates),
                              best_changed_cost=min((c["changed_edge_cost"] for c in candidates), default=None),
                              elapsed_s=time.perf_counter()-started, segment=segment))
        if time.perf_counter()-started >= budget_seconds:
            break
    candidates.sort(key=lambda item: item["changed_edge_cost"])
    selected = []
    for item in candidates:
        if any(np.linalg.norm(np.asarray(item["inserted_q_rad"])-old["inserted_q_rad"]) < .03
               for old in selected):
            continue
        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected, dict(segment_ranking=ranked, attempts=attempts, feasible=len(candidates),
                          elapsed_s=time.perf_counter()-started)


def export_path(root, out, checker, source, source_report, selected, scored, diagnostics, config, config_path):
    path = selected["path"]
    if (not np.array_equal(path[0], source_report["start_q_rad"])
            or not np.array_equal(path[-1], source_report["goal_q_rad"])):
        raise ValueError("Refinement changed an endpoint")
    for a, b in zip(path[:-1], path[1:]):
        if not checker.edge_cost(a, b, max_step_rad=.008)[0]:
            raise ValueError("Final dense hard check failed")
    np.savez_compressed(out/"trajectory_q.npz", q_arm_rad=path,
                        q_gripper_rad=np.full(len(path), checker.gripper_rad),
                        goal_id=np.int32(source_report["goal_id"]))
    with (out/"trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*ARM_JOINTS, "gripper"])
        writer.writerows([*q, checker.gripper_rad] for q in path)
    timing_plan=selected.get('timing_plan')
    trajectory = time_parameterize(path, config, .02,timing_plan=timing_plan)
    if timing_plan is not None:write_json(out/'timing_plan.json',timing_plan)
    tips = []
    for q in trajectory["q_ref"]:
        checker.set_q(q)
        tips.append(checker.data.site_xpos[checker.model.site("nozzle_tip").id].copy())
    np.savetxt(out/"nozzle_path_line.csv", np.c_[trajectory["time_s"], tips], delimiter=",",
               header="time_s,x_m,y_m,z_m", comments="")
    np.savez_compressed(out/"nozzle_path_line.npz", time_s=trajectory["time_s"],
                        tip_m=np.asarray(tips), q_arm_rad=trajectory["q_ref"])
    report = dict(source_report)
    # Replace inherited diagnostics: they describe a different selected trajectory.
    for key in ("stats", "candidate_scores", "attitude_escape", "timing_plan_relative", "timing_plan_sha256",
                "natural_acceptance", "natural_search", "selected_motion"):
        report.pop(key, None)
    if timing_plan is not None:
        report.update(timing_plan_relative='timing_plan.json',timing_plan_sha256=hashlib.sha256((out/'timing_plan.json').read_bytes()).hexdigest())
    report.update(created_local=datetime.now().astimezone().isoformat(),
                  planner="five-joint weighted RRT* followed by risk-focused joint-posture refinement",
                  source_path=source.relative_to(root).as_posix(),
                  source_path_report_sha256=hashlib.sha256((source/"path_report.json").read_bytes()).hexdigest(),
                  waypoint_count=len(path), waypoints_q_rad=path.tolist(),
                  selected_score={key:selected[key] for key in
                                  ("persistence_cost", "exposure_integral", "duration_s", "objective", "source")},
                  timing_config_relative=config_path.relative_to(root).as_posix(),
                  timing_config=config, soft_contact=checker.contact_trees.report(),
                  attitude_escape=dict(**diagnostics, weights_changed=False,
                                       endpoints_changed=False, extra_contact_penalties=False),
                  objectives=dict(local_search="same weighted joint-distance edge cost as RRT*",
                                  final="original per-organ time/persistence score + 0.05 * duration_s"))
    summaries = [{key:value for key,value in item.items() if key != "path"} for item in scored]
    report["candidate_scores"] = summaries
    write_json(out/"path_report.json", report)
    write_json(out/"candidate_scores.json", summaries)
    np.savez_compressed(out/"candidate_paths.npz", **{f"q_{i}":item["path"] for i,item in enumerate(scored)})
    write_json(out/"config_snapshot.json", config)
    (out/"PATH_PLANNING_SUCCESS").write_text("Weighted joint-posture refinement; dynamic checks follow.\n",
                                           encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root/"configs/weighted_trajectory_validation_v2.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--dt", type=float, default=.04)
    args = parser.parse_args()
    if args.top_k < 1 or args.seconds <= 0 or args.dt <= 0 or args.dt > .1:
        raise ValueError("Positive search settings and scoring step <= 0.1 s required")
    source, out, config_path = args.path.resolve(), args.output_dir.resolve(), args.config.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    source_report = read_json(source/"path_report.json")
    if source_report["status"] != "PATH_PLANNING_COMPLETE":
        raise ValueError("Input must be a completed path")
    config = read_json(config_path)
    with np.load(source/"trajectory_q.npz", allow_pickle=False) as data:
        base = data["q_arm_rad"].copy()
        if not np.allclose(data["q_gripper_rad"], config["gripper_rad"]):
            raise ValueError("Gripper differs from saved path")
    checker = CollisionChecker(root, root/source_report["scene_relative"])

    def progress(row):
        write_json(out/"search_progress.json", row)
        print(f"POSTURE_SEARCH {row}", flush=True)

    candidates, diagnostics = refine_candidates(checker, base, args.seconds, args.top_k, progress)
    pool = [dict(path=base, source="unchanged_baseline")]
    pool += [dict(**item, source="explicit_joint_posture") for item in candidates]
    scored = []
    for i, item in enumerate(pool):
        score = timed_path_score(checker, item["path"], config, dt=args.dt)
        item.update(score, goal_id=source_report["goal_id"])
        scored.append(item)
        print(f"TIMED_POSTURE {i+1}/{len(pool)} objective={score['objective']:.6f}", flush=True)
        write_json(out/"score_progress.json", [
            {key:value for key,value in entry.items() if key != "path"} for entry in scored])
    baseline_score = {key:scored[0][key] for key in ("objective", "persistence_cost", "duration_s")}
    scored = [item for item in scored if np.isfinite(item['objective'])]
    if not scored:
        raise RuntimeError('No valid interpolated posture candidates')
    selected = min(scored, key=lambda item:item["objective"])
    diagnostics["baseline_score"] = baseline_score
    export_path(root, out, checker, source, source_report, selected, scored, diagnostics, config, config_path)
    print(f"POSTURE_REFINEMENT_COMPLETE: {out}; objective={selected['objective']:.6f}", flush=True)


if __name__ == "__main__":
    main()
