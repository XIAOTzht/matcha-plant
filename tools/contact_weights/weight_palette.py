"""Shared colors for the weighted mesh and its standalone legend."""
from __future__ import annotations

import numpy as np

PALETTE_VERSION = "weight-colors-0.4.2-smooth-green-yellow"
# The earlier palette jumped from dark green at 4.8 to pale yellow at 5.
# These closely spaced anchors keep the semantic ranges while making the
# green -> yellow transition continuous and easier to read on the mesh.
ANCHORS = np.array([0, 1, 2, 3, 4, 4.5, 4.8, 5, 5.8, 6.7, 7.5, 10, 12.5, 15, 17.5, 20, 24], dtype=np.float32)
COLORS = np.array([
    [220, 248, 220],  # free space / left end of legend: pale green
    [190, 242, 175],  # ordinary leaf: pale edge
    [155, 230, 135],  # low leaf / pale green
    [115, 210, 105],  # leaf interior
    [95, 195, 95],    # high leaf: still green, no abrupt jump
    [120, 205, 92],   # first chartreuse bridge
    [150, 215, 90],   # second chartreuse bridge
    [180, 222, 88],   # ordinary flower: yellow-green edge
    [200, 222, 82],   # smooth green-yellow bridge
    [220, 218, 78],   # smooth green-yellow bridge
    [240, 215, 75],   # flower interior: yellow
    [246, 180, 60],   # amber
    [238, 135, 70],   # strongest configured flower: orange
    [240, 145, 90],   # orange-red bridge
    [245, 155, 130],  # pale-red bridge
    [248, 170, 170],  # ordinary stem: light red
    [245, 125, 135],  # uncertain stem: slightly deeper, still clearly light
], dtype=np.float32)
HARD_COLOR = (215, 40, 55)  # same bright deep red for pot, soil, cart, hardware


def soft_colors(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Finite soft weights are required for color mapping")
    rgb = np.stack([np.interp(values, ANCHORS, COLORS[:, c]) for c in range(3)], axis=-1)
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def risk_colors(weights: np.ndarray, hard: np.ndarray) -> np.ndarray:
    rgb = soft_colors(weights)
    rgb[np.asarray(hard, dtype=bool)] = HARD_COLOR
    return rgb


def palette_description() -> dict:
    return {
        "version": PALETTE_VERSION,
        "soft_anchors": [{"weight": float(w), "rgb": list(map(int, c))}
                         for w, c in zip(ANCHORS, COLORS)],
        "hard_collision_rgb": list(HARD_COLOR),
        "note": "All forbidden classes share a color. Color does not change collision permissions.",
    }
