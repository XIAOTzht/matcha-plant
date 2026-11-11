"""Render the latest watering layout without opening a viewer window."""
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
    scanlines = b"".join(b"\0" + row.tobytes() for row in rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(scanlines, 6)) + chunk(b"IEND", b""))


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path)
    ap.add_argument("--output", type=Path, default=root / "outputs/watering_scene_preview.png")
    args = ap.parse_args()
    scene = args.scene or root / read_json(root / "configs/latest_scene.json")["scene"]
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    image = np.empty((660, 1760, 3), dtype=np.uint8)
    with mujoco.Renderer(model, height=660, width=880) as renderer:
        for i, azimuth in enumerate((135, -40)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [.23, .09, .17]
            camera.distance = .9
            camera.azimuth = azimuth
            camera.elevation = -25
            option = mujoco.MjvOption()
            mujoco.mjv_defaultOption(option)
            option.geomgroup[3:5] = 0  # Hide robot collision proxies, keep visual meshes.
            option.sitegroup[5] = 0    # Workspace edges can be shown interactively.
            renderer.update_scene(data, camera=camera, scene_option=option)
            image[:, i * 880:(i + 1) * 880] = renderer.render()
    if image.std() < 3:
        raise ValueError("Rendered image is unexpectedly uniform")
    save_png(args.output, image)
    print(f"WATERING_PREVIEW_RENDERED: {args.output}")


if __name__ == "__main__":
    main()
