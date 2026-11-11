"""Render the IK poses from a goal-screening report for visual inspection."""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import zlib

import mujoco
import numpy as np

from plant_scene import read_json


def save_png(path, rgb):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    h, w, _ = rgb.shape
    scan = b"".join(b"\0" + row.tobytes() for row in rgb)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(scan, 6)) + chunk(b"IEND", b""))


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("screening", type=Path, help="goal_screening_report.json or its output directory")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    report_path = args.screening / "goal_screening_report.json" if args.screening.is_dir() else args.screening
    report = read_json(report_path)
    scene = Path(report["scene"]).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    entries = report["entries"]
    width, height = 720, 520
    cols, rows = 3, 2
    canvas = np.full((rows * height, cols * width, 3), 238, dtype=np.uint8)
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    for k, entry in enumerate(entries):
        data.qpos[:] = 0
        for name, q in zip(joints, entry["q_arm_rad"]):
            j = model.joint(name)
            data.qpos[j.qposadr] = q
        data.qpos[model.joint("gripper").qposadr] = entry.get("q_gripper_rad", 0.35)
        mujoco.mj_forward(model, data)
        image = np.empty((height, width, 3), dtype=np.uint8)
        camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [.27, .20, .16]; camera.distance = .62; camera.azimuth = 135; camera.elevation = -27
        option = mujoco.MjvOption(); mujoco.mjv_defaultOption(option)
        option.geomgroup[3:5] = 0
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            image[:] = renderer.render()
        col, row = k % cols, k // cols
        canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = image
    output = args.output or report_path.parent / "goal_screening_preview.png"
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); save_png(output, canvas)
    print(f"GOAL_SCREENING_PREVIEW_RENDERED: {output}")


if __name__ == "__main__":
    main()
