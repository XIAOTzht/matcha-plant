#!/usr/bin/env python3
"""Render a lightweight multi-view point preview for a weighted PLY."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from build_contact_weights import _read_ply


def render(points: np.ndarray, colors: np.ndarray, width: int, height: int,
           title: str, axes: tuple[int, int, int], flip: tuple[bool, bool]) -> Image.Image:
    p = points[:, list(axes)].astype(np.float64)
    # Bounding-box center, rather than the median, keeps the entire model
    # visible even though the cart accounts for most vertices.
    p -= (p.min(axis=0) + p.max(axis=0)) / 2
    xy = p[:, :2]
    if flip[0]: xy[:, 0] *= -1
    if flip[1]: xy[:, 1] *= -1
    span = np.maximum(np.ptp(xy, axis=0), 1e-6)
    scale = 0.88 * min(width / span[0], height / span[1])
    x = (xy[:, 0] * scale + width * 0.5).astype(np.int32)
    y = (-xy[:, 1] * scale + height * 0.53).astype(np.int32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, c, depth = x[valid], y[valid], colors[valid], p[valid, 2]
    order = np.argsort(depth)
    im = Image.new("RGB", (width, height), (238, 240, 244))
    d = ImageDraw.Draw(im)
    for i in order[::1]:
        r = 2 if len(x) < 250000 else 1
        d.ellipse((int(x[i]-r), int(y[i]-r), int(x[i]+r), int(y[i]+r)), fill=tuple(map(int, c[i])))
    d.rectangle((8, 8, width - 8, height - 8), outline=(70, 75, 85), width=2)
    d.text((18, 16), title, fill=(20, 20, 25))
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-points", type=int, default=220000)
    ap.add_argument("--coordinate-system", type=Path,
                    help="Board frame for an upright preview; mesh coordinates stay unchanged")
    ap.add_argument("--single-view", action="store_true")
    args = ap.parse_args()
    vertices, _ = _read_ply(args.input)
    rng = np.random.default_rng(20261022)
    take = np.arange(len(vertices)) if len(vertices) <= args.max_points else rng.choice(len(vertices), args.max_points, replace=False)
    pts = np.stack([vertices["x"][take], vertices["y"][take], vertices["z"][take]], axis=1)
    coord = args.coordinate_system
    if coord is None:
        candidate = args.input.parent.parent / "complete_hardware_v2" / "coordinate_system.json"
        if candidate.is_file():
            coord = candidate
    if coord is not None:
        cs = json.loads(coord.read_text(encoding="utf-8"))
        frame = np.array(cs["board_frame_axes"], dtype=float)
        if np.linalg.det(frame) < 0:
            frame[:, 2] *= -1
        pts = (pts - np.array(cs["board_center_mm"])) @ frame
    rgb = np.stack([vertices["red"][take], vertices["green"][take], vertices["blue"][take]], axis=1)
    if args.single_view:
        # Mild azimuth shows the board/pot without rotating the plant sideways.
        angle = np.deg2rad(25)
        rotation = np.array([[np.cos(angle), 0, -np.sin(angle)],
                             [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
        pts = pts @ rotation
        render(pts, rgb, 1000, 1150, "Contact risk weights", (0, 1, 2), (False, False)).save(args.output)
        print(args.output)
        return 0
    tile_w, tile_h = 720, 650
    canvas = Image.new("RGB", (tile_w * 2, tile_h * 2), (250, 250, 252))
    views = [
        ("Front", (0, 1, 2), (False, False)),
        ("Side", (2, 1, 0), (False, False)),
        ("Back", (0, 1, 2), (True, False)),
        ("Top", (0, 2, 1), (False, False)),
    ]
    for i, (title, axes, flip) in enumerate(views):
        tile = render(pts, rgb, tile_w, tile_h, title, axes, flip)
        canvas.paste(tile, ((i % 2) * tile_w, (i // 2) * tile_h))
    canvas.save(args.output, quality=94)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
