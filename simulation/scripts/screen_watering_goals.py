"""Multistart SO-101 IK screening with hard collisions and soft plant risk."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from plant_scene import read_json, write_json
from plan_watering_path import ARM_JOINTS, CollisionChecker


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def ik_residual(q_arm, model, data, joint_ids, site_id, target, down_target, gripper_rad=.35):
    data.qpos[:] = 0
    for j, q in zip(joint_ids, q_arm):
        data.qpos[model.jnt_qposadr[j]] = q
    # Keep the configured tool grasp fixed, leaving its actuator out of arm IK.
    gripper = model.joint("gripper")
    data.qpos[gripper.qposadr] = gripper_rad
    mujoco.mj_forward(model, data)
    tip = data.site_xpos[site_id].copy()
    direction = unit(data.site_xmat[site_id].reshape(3, 3) @ np.array([0.0, 0.0, -1.0]))
    # Position is in metres. The direction residual leaves yaw free, matching
    # the 5-DOF arm, while rejecting an upward pointing spray axis.
    return np.r_[(tip - target) / 0.003, (direction - down_target) / 0.15]


def solve_ik(model, data, target, seeds, q_lo, q_hi, joint_ids, site_id, gripper_rad=.35):
    down = np.array([0.0, 0.0, -1.0])
    best = None
    solutions = []
    for seed in seeds:
        result = least_squares(
            ik_residual, seed, bounds=(q_lo, q_hi),
            args=(model, data, joint_ids, site_id, target, down, gripper_rad),
            max_nfev=350, xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        err = float(np.linalg.norm(ik_residual(result.x, model, data, joint_ids, site_id, target, down, gripper_rad)))
        if best is None or err < best["residual"]:
            best = {"result": result, "residual": err}
        same = next((s for s in solutions if np.max(abs(s["result"].x - result.x)) < 1e-4), None)
        if same is None:
            solutions.append({"result": result, "residual": err})
        elif err < same["residual"]:
            same.update(result=result, residual=err)
    return {**best, "solutions": solutions}


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=root / read_json(root / "configs/latest_scene.json")["scene"])
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--seed-count", type=int, default=24)
    args = ap.parse_args()
    scene = args.scene.resolve()
    scene_dir = scene.parent
    out = (args.output_dir or root / "outputs" / ("watering_screen_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh output directory; existing results are preserved")
    out.mkdir(parents=True, exist_ok=True)
    config = read_json(scene_dir / "config_snapshot.json")
    candidates = read_json(scene_dir / "watering_candidates.json")["entries"]
    print("Loading full triangles, weights and scene-specific policy", flush=True)
    checker = CollisionChecker(root, scene)
    model, data = checker.model, checker.data
    tolerance = float(config["target_candidates"].get("position_tolerance_mm", 4.0)) / 1000
    bounds = checker.policy["watering_target"]["approach_height_range_mm"]
    site_id = int(model.site("nozzle_tip").id)
    joint_ids = [int(model.joint(n).id) for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
    q_lo = np.array([model.jnt_range[j, 0] for j in joint_ids])
    q_hi = np.array([model.jnt_range[j, 1] for j in joint_ids])
    rng = np.random.default_rng(20261022)
    seeds = [np.zeros(5), np.array([0, .5, -.8, .4, 0]), np.array([0, -.5, .8, -.4, 0])]
    seeds += [rng.uniform(q_lo, q_hi) for _ in range(max(0, args.seed_count - len(seeds)))]
    reports = []
    for entry in candidates:
        target = np.asarray(entry["nozzle_tip_candidate_world_m"], dtype=float)
        solved = solve_ik(model, data, target, seeds, q_lo, q_hi, joint_ids, site_id, checker.gripper_rad)
        branches = []
        for solution in solved["solutions"]:
            result = solution["result"]
            valid, detail = checker.check(result.x, details=True)
            tip = data.site_xpos[site_id].copy()
            error = float(np.linalg.norm(tip - target))
            down = float(data.site_xmat[site_id].reshape(3, 3)[2, 2])
            height = float(1000 * (tip[2] - entry["soil_surface_world_m"][2]))
            accepted = bool(valid and solution["residual"] < 1.2 and error <= tolerance
                            and down > .94 and bounds[0] <= height <= bounds[1])
            branches.append(dict(accepted_screening=accepted, ik_success=bool(result.success),
                                 ik_residual_scaled=solution["residual"], q_arm_rad=result.x.tolist(),
                                 tip_m=tip.tolist(), position_error_m=error, spray_down_cosine=down,
                                 height_above_soil_mm=height, **detail))
        branches.sort(key=lambda b: (not b["accepted_screening"], b["risk_proxy"], b["ik_residual_scaled"]))
        pose = branches[0]
        reports.append({"id": int(entry["id"]), "source_face_index": int(entry["source_face_index"]),
                        "target_m": target.tolist(), "soil_surface_world_m": entry["soil_surface_world_m"],
                        "joint_names": list(ARM_JOINTS), "q_gripper_rad": checker.gripper_rad,
                        "ik_branch_count": len(branches), "checked_branches": branches, **pose})
        print(f"GOAL {entry['id']}: accepted={pose['accepted_screening']} "
              f"height={pose['height_above_soil_mm']:.3f}mm "
              f"hard lower bound={1000*pose['hard_clearance_min_m']:.3f}mm", flush=True)
    report = {"status": "GOAL_SCREENING_COMPLETE", "scene": str(scene), "scene_layout_is_virtual": True,
              "scene_relative": scene.relative_to(root).as_posix(),
              "full_resolution_vertices": len(checker.xyz), "full_resolution_faces": len(checker.faces),
              "robot_collision_representation": "Hard pot/soil/cart collision checks; soft plant proximity is scored",
              "robot_collision_geom_count": len(checker.geom_ids), "hard_face_count": len(checker.hard_faces),
              "hard_margin_mm": 1000*checker.margin_m, "position_tolerance_mm": 1000*tolerance,
              "effective_watering_target": checker.policy["watering_target"],
              "plant_query": "libigl exact triangle AABB nearest distance",
              "hard_collision_rule": "Pot, soil and cart are forbidden; stems, flowers and leaves are soft-risk classes",
              "not_a_trajectory": True, "open_mesh_limitation": "surface-distance screening cannot prove enclosure safety on open reconstructed shells",
              "entries": reports}
    write_json(out / "goal_screening_report.json", report)
    np.savez_compressed(out / "screened_poses.npz", q_arm_rad=np.asarray([r["q_arm_rad"] for r in reports]),
                        targets_m=np.asarray([r["target_m"] for r in reports]), accepted=np.asarray([r["accepted_screening"] for r in reports]))
    (out / "GOAL_SCREENING_SUCCESS").write_text("goal screening complete; not a continuous trajectory\n", encoding="utf-8")
    write_json(root / "configs/latest_screening.json", dict(output=out.relative_to(root).as_posix(),
               report=(out / "goal_screening_report.json").relative_to(root).as_posix(), status=report["status"]))
    print(json.dumps({"status": report["status"], "output": str(out),
                      "accepted": int(sum(r["accepted_screening"] for r in reports)), "total": len(reports)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
