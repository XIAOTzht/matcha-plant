"""Time parameterize a path, simulate its actuators, and query every sampled pose.

Full plant triangles are externally checked; plant display meshes are not physics
colliders. All limits and duration parameters are explicit simulation settings.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import mujoco
import igl
import numpy as np

from plan_watering_path import ARM_JOINTS, CollisionChecker
from plant_scene import read_json, write_json
from trajectory_tools import score_exposure, time_parameterize,actuator_command


def state_metrics(checker, q, grip):
    valid, detail = checker.check(q, details=True, gripper_rad=grip)
    detail["valid"] = valid
    tip_id = checker.model.site("nozzle_tip").id
    detail["tip_m"] = checker.data.site_xpos[tip_id].copy()
    detail["spray_down_cosine"] = float(checker.data.site_xmat[tip_id].reshape(3, 3)[2, 2])
    # The nozzle extends along local -Z: its alignment with world -Z equals Rzz.
    trace = checker.contact_trace()
    detail["soft_classes"] = np.asarray([r[0] for r in trace], dtype=np.int32)
    detail["soft_face_ids"] = np.asarray([r[1] for r in trace], dtype=np.int32)
    detail["soft_clear_m"] = np.asarray([r[2] for r in trace])
    detail["soft_local_weight"] = np.asarray([r[3] for r in trace])
    detail["soft_instances"] = np.asarray([r[4] for r in trace], dtype=np.int32)
    detail['soft_responsible_geom'] = checker.contact_trees.last_responsible_geom.copy()
    return detail


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path)
    ap.add_argument("--scene", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument('--no-promote',action='store_true',help='Validate a trial without replacing the current replay')
    args = ap.parse_args()
    path_dir = (args.path or root / read_json(root / "configs/latest_path.json")["output"]).resolve()
    path_report = read_json(path_dir / "path_report.json")
    if path_report["status"] != "PATH_PLANNING_COMPLETE":
        raise ValueError("Source path did not complete")
    # A new latest layout must not silently change an old saved path's scene.
    default_scene = str(args.scene) if args.scene else path_report.get("scene_relative")
    if not default_scene:
        source_screening = read_json(root / "configs/latest_screening.json")
        screening_report = read_json(root / source_screening["report"])
        matched = [entry for entry in screening_report["entries"]
                   if entry["id"] == path_report["goal_id"]
                   and np.allclose(entry["q_arm_rad"], path_report["goal_q_rad"], atol=1e-9)]
        if not matched:
            raise ValueError("Legacy path scene cannot be inferred; specify --scene")
        default_scene = screening_report["scene"]
    scene = (args.scene or root / default_scene).resolve()
    default_config = "weighted_trajectory_validation.json" if path_report.get("risk_optimized") else "trajectory_validation.json"
    # A refined path was ranked with a particular timing profile. Preserve it
    # on default validation rather than silently returning to an older speed.
    saved_config = path_report.get("timing_config_relative")
    config = read_json(args.config or (root / saved_config if saved_config
                                      else root / "configs" / default_config))
    scene_report = read_json(scene.parent / "scene_report.json")
    if path_report.get("mesh_fingerprint") and path_report["mesh_fingerprint"] != scene_report["mesh_fingerprint"]:
        raise ValueError("Path and scene mesh fingerprints disagree")
    with np.load(path_dir / "trajectory_q.npz", allow_pickle=False) as z:
        waypoints = z["q_arm_rad"].copy()
        gripper = z["q_gripper_rad"].copy()
    if (not np.allclose(waypoints[0], path_report["start_q_rad"], atol=1e-9)
            or not np.allclose(waypoints[-1], path_report["goal_q_rad"], atol=1e-9)
            or not np.allclose(gripper, config["gripper_rad"], atol=1e-9)):
        raise ValueError("Path endpoints or gripper disagree with saved report/config")
    out = (args.output_dir or root / "outputs" / ("watering_replay_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    print("Loading full triangle collision queries and original contact policy", flush=True)
    checker = CollisionChecker(root, scene, margin_m=config["hard_margin_mm"] / 1000)
    model = mujoco.MjModel.from_xml_path(str(scene)); sim = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    timing_profile = str(config.get("timing_profile", "rest_to_rest_quintic"))
    timing_plan=None
    if path_report.get('timing_plan_relative'):
        timing_file=path_dir/path_report['timing_plan_relative']
        if sha256(timing_file)!=path_report['timing_plan_sha256']:raise ValueError('Saved timing checksum mismatch')
        timing_plan=read_json(timing_file)
    trajectory = time_parameterize(waypoints, config, dt,timing_plan=timing_plan)
    feedforward_mode=(timing_plan or {}).get('actuator_feedforward_mode','legacy')
    if feedforward_mode not in ('legacy','inverse_dynamics'):raise ValueError('Unknown feedforward mode')
    inverse=mujoco.MjData(model)if feedforward_mode=='inverse_dynamics'else None
    time = trajectory["time_s"]; q_ref = trajectory["q_ref"]
    joint_names = (*ARM_JOINTS, "gripper")
    qadr = np.asarray([model.joint(name).qposadr[0] for name in joint_names], dtype=int)
    vadr = np.asarray([model.joint(name).dofadr[0] for name in joint_names], dtype=int)
    actuator_ids = np.asarray([model.actuator(name).id for name in joint_names], dtype=int)
    kp = model.actuator_gainprm[actuator_ids, 0]
    if np.any(kp <= 0):
        raise ValueError("Expected position actuators with positive gain")
    sim.qpos[qadr] = np.r_[q_ref[0], config["gripper_rad"]]
    sim.ctrl[actuator_ids] = sim.qpos[qadr]
    mujoco.mj_forward(model, sim)
    q_actual, qvel_actual, qacc_actual, ctrl_trace, force_trace = [], [], [], [], []
    ref_metrics, sim_metrics = [], []
    controls_ok = True
    print(f"Timing: {time[-1]:.3f} s, {len(time)} samples, physics/query step {dt:.3f} s", flush=True)
    for k, t in enumerate(time):
        # Position actuators have no explicit velocity command.  A small,
        # configurable position lead reduces tracking lag on continuous timing
        # profiles without changing the planned geometric path.  The lead is
        # zero by default and is disabled during holds because qd_ref is zero.
        lead_s = 0. if inverse is not None else float(config.get("position_lead_s", 0.0))
        mujoco.mj_forward(model, sim)
        command=actuator_command(model,sim,config,np.r_[q_ref[k],config['gripper_rad']],
            np.r_[trajectory['qd_ref'][k],0.],np.r_[trajectory['qdd_ref'][k],0.],qadr,vadr,actuator_ids,inverse)
        sim.ctrl[actuator_ids] = command
        ranges = model.actuator_ctrlrange[actuator_ids]
        controls_ok = controls_ok and bool(np.all(command >= ranges[:, 0]) and np.all(command <= ranges[:, 1]))
        mujoco.mj_forward(model, sim)
        q_actual.append(sim.qpos[qadr].copy()); qvel_actual.append(sim.qvel[vadr].copy())
        qacc_actual.append(sim.qacc[vadr].copy()); ctrl_trace.append(command.copy())
        force_trace.append(sim.actuator_force[actuator_ids].copy())
        ref_metrics.append(state_metrics(checker, q_ref[k], config["gripper_rad"]))
        sim_metrics.append(state_metrics(checker, sim.qpos[qadr[:5]], float(sim.qpos[qadr[5]])))
        if k % 200 == 0 or k == len(time) - 1:
            print(f"REPLAY_CHECK {k+1}/{len(time)} t={t:.3f}s hard gap={1000*sim_metrics[-1]['hard_clearance_min_m']:.3f}mm", flush=True)
        if k != len(time) - 1:
            mujoco.mj_step(model, sim)
    q_actual, qvel_actual, qacc_actual = map(np.asarray, (q_actual, qvel_actual, qacc_actual))
    ctrl_trace, force_trace = np.asarray(ctrl_trace), np.asarray(force_trace)
    target = np.asarray(path_report["goal_target_m"])
    final_tip = sim_metrics[-1]["tip_m"]
    final_error = float(np.linalg.norm(final_tip - target))
    candidates = read_json(scene.parent / "watering_candidates.json")["entries"]
    same_target = [entry for entry in candidates if entry["id"] == path_report["goal_id"]
                   and np.allclose(entry["nozzle_tip_candidate_world_m"], target, atol=1e-9)]
    if len(same_target) != 1:
        raise ValueError("Path goal does not match the selected scene's soil target")
    soil = np.asarray(same_target[0]["soil_surface_world_m"])
    height_mm = float((final_tip[2] - soil[2]) * 1000)
    # Check the complete final hold as well as its last sample. A centroid's
    # vertical offset differs slightly from distance to uneven soil triangles.
    hold = trajectory["phase"] == 2
    hold_tips = np.asarray([m["tip_m"] for m in sim_metrics])[hold]
    hold_heights_mm = 1000 * (hold_tips[:, 2] - soil[2])
    soil_faces = checker.faces[checker.face_labels == checker.policy["watering_target"]["class_id"]]
    soil_tree = igl.AABB(); soil_tree.init(checker.xyz, soil_faces)
    soil_sq, _, _ = soil_tree.squared_distance(checker.xyz, soil_faces, hold_tips)
    hold_soil_distance_mm = 1000 * np.sqrt(np.maximum(soil_sq, 0))
    height_bounds = checker.policy["watering_target"]["approach_height_range_mm"]
    classes = ref_metrics[0]["soft_classes"]
    data_arrays = dict(time_s=time, joint_names=np.asarray(joint_names),
                       q_ref_rad=np.column_stack([q_ref, np.full(len(time), config["gripper_rad"])]),
                       qd_ref_rad_s=np.column_stack([trajectory["qd_ref"], np.zeros(len(time))]),
                       qdd_ref_rad_s2=np.column_stack([trajectory["qdd_ref"], np.zeros(len(time))]),
                       q_actual_rad=q_actual, qd_actual_rad_s=qvel_actual, qdd_actual_rad_s2=qacc_actual,
                       actuator_ctrl_rad=ctrl_trace, actuator_force_nm=force_trace, phase=trajectory["phase"],
                       soft_class_ids=classes, soft_channels=checker.contact_trees.pairs, robot_geom_ids=checker.geom_ids,
                       robot_geom_names=np.asarray(checker.geom_names), robot_cover_radius_m=checker.radii,
                       goal_target_m=target, mesh_fingerprint=np.asarray(scene_report["mesh_fingerprint"]))
    risk_reports = {}
    for prefix, metrics in (("reference", ref_metrics), ("simulated", sim_metrics)):
        for name in ("tip_m", "spray_down_cosine", "hard_clearance_min_m", "nozzle_hard_clearance_m",
                     "hard_clearance_by_geom_m", "soft_clear_m", "soft_local_weight", "soft_instances", "soft_face_ids", "soft_responsible_geom"):
            data_arrays[f"{prefix}_{name}"] = np.asarray([m[name] for m in metrics])
        costs = score_exposure(time, classes, data_arrays[f"{prefix}_soft_clear_m"],
                               data_arrays[f"{prefix}_soft_local_weight"],
                               data_arrays[f"{prefix}_soft_instances"], checker.policy)
        for name in ("instantaneous_exposure", "active_episode_age_s", "overlap_any", "proximity_any", "episode_weighted_series"):
            data_arrays[f"{prefix}_{name}"] = costs[name]
        risk_reports[prefix] = {key: costs[key] for key in ("exposure_integral", "persistence_cost", "episodes")}
        risk_reports[prefix]["overlap_duration_s"] = float(np.trapezoid(costs["overlap_any"].astype(float), time))
        risk_reports[prefix]["proximity_duration_s"] = float(np.trapezoid(costs["proximity_any"].astype(float), time))
        risk_reports[prefix]["interpretation"] = "Native robot volumes vs source triangle covers; geometric proxy, not force or measured damage."
    velocity = np.asarray(config["velocity_limit_rad_s"]); acceleration = np.asarray(config["acceleration_limit_rad_s2"])
    force_bounds = model.actuator_forcerange[actuator_ids]
    checks = dict(reference_joint_limits=bool(np.all(q_ref >= checker.q_lo) and np.all(q_ref <= checker.q_hi)),
                  reference_velocity=bool(np.all(np.max(abs(trajectory["qd_ref"]), axis=0) <= velocity + 1e-8)),
                  reference_acceleration=bool(np.all(np.max(abs(trajectory["qdd_ref"]), axis=0) <= acceleration + 1e-8)),
                  simulated_velocity=bool(np.all(np.max(abs(qvel_actual[:, :5]), axis=0) <= velocity + 1e-8)),
                  simulated_acceleration=bool(np.all(np.max(abs(qacc_actual[:, :5]), axis=0) <= acceleration + 1e-8)),
                  actuator_control_range=controls_ok,
                  actuator_force_range=bool(np.all(force_trace >= force_bounds[:, 0] - 1e-8)
                                            and np.all(force_trace <= force_bounds[:, 1] + 1e-8)),
                  finite_dynamics=bool(np.isfinite(q_actual).all() and np.isfinite(qvel_actual).all()
                                       and np.isfinite(qacc_actual).all()),
                  reference_pose_checks=all(m["valid"] for m in ref_metrics),
                  simulated_pose_checks=all(m["valid"] for m in sim_metrics),
                  no_numerical_warnings=not bool(np.any(sim.warning.number)),
                  joint_tracking=bool(np.max(abs(q_actual[:, :5] - q_ref)) <= np.deg2rad(config["joint_tracking_tolerance_deg"])),
                  final_tip_position=final_error <= config["goal_position_tolerance_mm"] / 1000,
                  final_height_above_soil=checker.policy["watering_target"]["approach_height_range_mm"][0]
                      <= height_mm <= checker.policy["watering_target"]["approach_height_range_mm"][1],
                  target_hold_height=bool(np.all(hold_heights_mm >= height_bounds[0])
                                         and np.all(hold_heights_mm <= height_bounds[1])),
                  target_hold_nearest_soil_distance=bool(np.all(hold_soil_distance_mm > config["hard_margin_mm"])
                                                        and np.all(hold_soil_distance_mm <= height_bounds[1])),
                  final_spray_direction=sim_metrics[-1]["spray_down_cosine"] >= np.cos(np.deg2rad(config["down_axis_tolerance_deg"])))
    checks = {key: bool(value) for key, value in checks.items()}
    # With a tilted nozzle, proximity to soil alone is insufficient: its spray
    # centerline must hit soil before pot or foliage throughout the final hold.
    from spray_target import SprayRayQuery
    spray=SprayRayQuery(checker.xyz,checker.faces,checker.face_labels)
    hold_ids=np.flatnonzero(hold)
    spray_rays=[]
    for k in hold_ids:
        checker.set_q(q_actual[k,:5],q_actual[k,5])
        sid=checker.model.site('nozzle_tip').id
        ray=spray.first_hit(checker.data.site_xpos[sid],-checker.data.site_xmat[sid].reshape(3,3)[:,2])
        spray_rays.append(dict(index=int(k),**ray))
    checks['target_hold_spray_hits_soil']=all(r['first_hit_class']==checker.policy['watering_target']['class_id']
        and .5<r['range_mm']<40 for r in spray_rays)
    status = "SAMPLED_REPLAY_CHECKS_PASSED" if all(checks.values()) else "REPLAY_CHECKS_FAILED"
    natural_result=None
    if path_report.get('natural_acceptance'):
        from natural_motion import natural_metrics,motion_passes
        rule=path_report['natural_acceptance'];baseline=rule['baseline']
        natural_result=natural_metrics(dict(time_s=time,q_ref=q_actual[:,:5],qd_ref=qvel_actual[:,:5],
            qdd_ref=qacc_actual[:,:5],phase=trajectory['phase']),rule['settings'])
        checks['natural_motion_improved']=motion_passes(natural_result,rule['settings'],baseline['motion'])
        checks['no_increase_in_actual_exposure']=risk_reports['simulated']['persistence_cost']<=baseline['persistence_cost']
        status='SAMPLED_REPLAY_CHECKS_PASSED'if all(checks.values())else 'REPLAY_CHECKS_FAILED'
    report = dict(status=status, created_local=datetime.now().astimezone().isoformat(),
                  scene=str(scene), scene_relative=scene.relative_to(root).as_posix(),
                  source_path=path_dir.relative_to(root).as_posix(), source_path_report_sha256=sha256(path_dir / "path_report.json"),
                  source_trajectory_sha256=sha256(path_dir / "trajectory_q.npz"),
                  mesh_fingerprint=scene_report["mesh_fingerprint"], scene_sha256=sha256(scene),
                  sample_count=len(time), dt_s=dt, duration_s=float(time[-1]), goal_id=path_report["goal_id"],
                  timing=timing_profile, position_lead_s=lead_s,
                  segments=trajectory["segments"], checks=checks,
                  reference_peak_velocity_rad_s=np.max(abs(trajectory["qd_ref"]), axis=0).tolist(),
                  reference_peak_acceleration_rad_s2=np.max(abs(trajectory["qdd_ref"]), axis=0).tolist(),
                  simulated_peak_velocity_rad_s=np.max(abs(qvel_actual), axis=0).tolist(),
                  simulated_peak_acceleration_rad_s2=np.max(abs(qacc_actual), axis=0).tolist(),
                  simulated_peak_actuator_force_nm=np.max(abs(force_trace), axis=0).tolist(),
                  peak_joint_tracking_error_deg=float(np.rad2deg(np.max(abs(q_actual[:, :5]-q_ref)))),
                  reference_min_hard_clearance_mm=1000*min(m["hard_clearance_min_m"] for m in ref_metrics),
                  simulated_min_hard_clearance_mm=1000*min(m["hard_clearance_min_m"] for m in sim_metrics),
                  simulated_nozzle_min_hard_clearance_mm=1000*min(m["nozzle_hard_clearance_m"] for m in sim_metrics),
                  final_tip_world_m=final_tip.tolist(), target_tip_world_m=target.tolist(), final_tip_error_mm=final_error*1000,
                  final_height_above_soil_mm=height_mm,
                  final_tip_to_nearest_soil_mm=float(hold_soil_distance_mm[-1]),
                  target_hold_height_range_mm=[float(hold_heights_mm.min()), float(hold_heights_mm.max())],
                  target_hold_soil_distance_range_mm=[float(hold_soil_distance_mm.min()), float(hold_soil_distance_mm.max())],
                  effective_watering_target=checker.policy["watering_target"],
                  hard_collision_refinement=checker.config.get("hard_collision_refinement", {"enabled": False}),
                  final_down_axis_error_deg=float(np.rad2deg(np.arccos(np.clip(sim_metrics[-1]["spray_down_cosine"], -1, 1)))),
                  numerical_warnings=sim.warning.number.tolist(), contact_risk=risk_reports,
                  plant_physics_contacts_enabled=False, real_robot_ready=False, risk_optimized=bool(path_report.get("risk_optimized", False)),
                  collision_claim="Time-sampled full-triangle checks using conservative spheres and adaptive native-volume box bounds when enabled; not continuous certification.",
                  limitations=["Open reconstructed surfaces cannot certify solid enclosure or hidden geometry.",
                               "Plant display colliders are disabled: no leaf bending, fluid flow, or plant contact dynamics are simulated.",
                               "Soft overlap flags describe conservative triangle-cover exposure, not verified physical contacts.",
                               "Finite weighted search and time-sampled ranking do not prove a global minimum.",
                               "Robot base registration, nozzle geometry and hardware velocity/acceleration limits need physical calibration."],
                  soft_contact_query=checker.contact_trees.report(),
                  grasp_contact_rule=checker.config["nozzle"].get("grasp_contact", {}),
                  target_hold_spray_rays=spray_rays,
                  invalid_samples={prefix:[dict(index=i,time_s=float(time[i]),
                      self_contacts=m["self_contacts"],hard_clearance_mm=1000*m["hard_clearance_min_m"],
                      workspace_ok=m["workspace_ok"],joint_limits_ok=m["joint_limits_ok"],floor_collision=m["floor_collision"])
                      for i,m in enumerate(metrics) if not m["valid"]]
                      for prefix,metrics in (("reference",ref_metrics),("simulated",sim_metrics))},
                  config=config)
    for filename in ("validate_watering_trajectory.py", "trajectory_tools.py", "plan_watering_path.py", "plant_scene.py", "robot_volume_query.py", "soft_contact_field.py", "spray_target.py"):
        report.setdefault("script_sha256", {})[filename] = sha256(root / "scripts" / filename)
    if timing_plan is not None:report['timing_plan_sha256']=path_report['timing_plan_sha256']
    report['actuator_feedforward_mode']=feedforward_mode
    if natural_result is not None:report['natural_motion']=natural_result
    np.savez_compressed(out / "timed_trajectory.npz", **data_arrays)
    with (out / "timed_trajectory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "phase", *[f"ref_{n}_rad" for n in joint_names],
                         *[f"actual_{n}_rad" for n in joint_names],
                         *[f"velocity_{n}_rad_s" for n in joint_names],
                         *[f"acceleration_{n}_rad_s2" for n in joint_names],
                         "hard_clearance_mm", "nozzle_hard_clearance_mm", "soft_exposure_proxy", "active_episode_age_s"])
        for k, t in enumerate(time):
            writer.writerow([t, int(trajectory["phase"][k]), *data_arrays["q_ref_rad"][k], *q_actual[k],
                             *qvel_actual[k], *qacc_actual[k], 1000*sim_metrics[k]["hard_clearance_min_m"],
                             1000*sim_metrics[k]["nozzle_hard_clearance_m"],
                             data_arrays["simulated_instantaneous_exposure"][k],
                             data_arrays["simulated_active_episode_age_s"][k]])
    write_json(out / "validation_report.json", report)
    write_json(out / "config_snapshot.json", config)
    marker = "SAMPLED_REPLAY_SUCCESS" if all(checks.values()) else "REPLAY_CHECKS_FAILED"
    (out / marker).write_text(status + "\n", encoding="utf-8")
    # An unsuccessful trial must never replace a previously usable replay.
    if all(checks.values())and not args.no_promote:
        write_json(root / "configs/latest_replay.json", dict(output=out.relative_to(root).as_posix(),
                   report=(out / "validation_report.json").relative_to(root).as_posix(), status=status))
    print(json.dumps(dict(status=status, output=str(out), checks=checks,
                         min_hard_clearance_mm=report["simulated_min_hard_clearance_mm"],
                         final_tip_error_mm=report["final_tip_error_mm"]), ensure_ascii=False), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
