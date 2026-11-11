"""Replay saved MuJoCo dynamics states; no real robot commands are sent."""
from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from plant_scene import read_json


def load_replay(root, directory=None):
    directory = Path(directory) if directory else root / read_json(root / "configs/latest_replay.json")["output"]
    directory = directory.resolve()
    report = read_json(directory / "validation_report.json")
    # Prefer a root-relative scene path so the saved replay survives deployment.
    scene = root / report["scene_relative"]
    if not scene.is_file():
        raise ValueError(f"Scene missing: {scene}")
    with np.load(directory / "timed_trajectory.npz", allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    if str(arrays["mesh_fingerprint"]) != report["mesh_fingerprint"]:
        raise ValueError("Replay arrays and report geometry disagree")
    return directory, report, scene.resolve(), arrays


def apply_frame(model, data, arrays, index, reference=False):
    q = arrays["q_ref_rad"] if reference else arrays["q_actual_rad"]
    qd = arrays["qd_ref_rad_s"] if reference else arrays["qd_actual_rad_s"]
    for col, name in enumerate(arrays["joint_names"]):
        joint = model.joint(str(name))
        data.qpos[joint.qposadr] = q[index, col]
        data.qvel[joint.dofadr] = qd[index, col]
        actuator = model.actuator(str(name))
        data.ctrl[actuator.id] = arrays["actuator_ctrl_rad"][index, col]
    data.time = arrays["time_s"][index]
    mujoco.mj_forward(model, data)


def add_path_line(scene, points, rgba=(.05,.65,1.,1.), radius=.0011, segments=180):
    indices=np.unique(np.linspace(0,len(points)-1,min(len(points),segments+1)).astype(int))
    for a,b in zip(points[indices[:-1]],points[indices[1:]]):
        if np.linalg.norm(b-a)<1e-6 or scene.ngeom>=scene.maxgeom:continue
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(rgba,dtype=np.float32))
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,a,b)
        scene.ngeom+=1


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--speed", type=float, default=1.)
    ap.add_argument("--reference", action="store_true")
    args = ap.parse_args()
    if not 0 < args.speed <= 10:
        raise ValueError("Playback speed must be in (0, 10]")
    directory, report, scene, arrays = load_replay(root, args.replay)
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    frames = arrays["time_s"]
    controls = dict(paused=False, restart=False, reference=args.reference)
    lock = threading.Lock()

    def key(code):
        with lock:
            if code == 32:
                controls["paused"] = not controls["paused"]
            elif code in (ord("R"), ord("r")):
                controls["restart"] = True
            elif code in (ord("T"), ord("t")):
                controls["reference"] = not controls["reference"]

    print(f"Saved simulation replay: {directory}")
    print(f"Status: {report['status']}; duration: {report['duration_s']:.3f}s")
    print("SPACE pause/resume | R restart | T reference/actual. Close the window to exit.")
    print("This viewer displays saved states. Plant deformation and water are not simulated.")
    apply_frame(model, data, arrays, 0, args.reference)
    with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
        viewer.cam.lookat[:] = [.23, .10, .17]
        viewer.cam.distance = .85; viewer.cam.azimuth = 135; viewer.cam.elevation = -25
        viewer.opt.geomgroup[3:5] = 0
        viewer.opt.sitegroup[5] = 0
        play_time = 0.; last = time.perf_counter(); ended = False
        while viewer.is_running():
            now = time.perf_counter(); delta = now - last; last = now
            with lock:
                if controls["restart"]:
                    play_time = 0.; controls["restart"] = False; ended = False
                if not controls["paused"]:
                    play_time = min(float(frames[-1]), play_time + delta*args.speed)
                reference = controls["reference"]
            index = min(len(frames)-1, int(np.searchsorted(frames, play_time, side="right")-1))
            with viewer.lock():
                apply_frame(model, data, arrays, max(0, index), reference)
                viewer.user_scn.ngeom=0
                add_path_line(viewer.user_scn,arrays["simulated_tip_m"],rgba=(.1,.65,1.,.8))
            viewer.sync()
            if play_time >= frames[-1] and not ended:
                print("Replay complete; holding final pose. Press R to replay.")
                ended = True
            time.sleep(.01)


if __name__ == "__main__":
    main()
