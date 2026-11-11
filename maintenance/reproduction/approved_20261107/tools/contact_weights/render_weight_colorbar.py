#!/usr/bin/env python3
"""Render the standalone light-green to deep-red weight color bar."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from weight_palette import soft_colors, HARD_COLOR
from preview_fonts import preview_font


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    width, height = 1600, 290
    left, right, top, bottom = 65, 1190, 85, 155
    values = np.linspace(0, 24, right - left + 1)
    rgb = soft_colors(values)
    image = Image.new("RGB", (width, height), (248, 249, 252))
    draw = ImageDraw.Draw(image)
    font, chinese = preview_font(27)
    small_font, _ = preview_font(20)
    for x, color in enumerate(rgb, start=left):
        draw.line((x, top, x, bottom), fill=tuple(map(int, color)), width=1)
    draw.rectangle((left, top, right, bottom), outline=(55, 55, 65), width=2)
    draw.text((left, 23), "接触风险权重：绿色 → 黄色 → 浅红；越往右代价越高" if chinese else
              "Contact risk: green -> yellow -> pale red; higher cost to the right", fill=(25, 25, 30), font=font)
    for value in [0, 1, 5, 20, 24]:
        x = int(left + (right - left) * value / 24.0)
        draw.line((x, bottom + 2, x, bottom + 13), fill=(45, 45, 50), width=2)
        label = str(value)
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 18), label, fill=(25, 25, 30), font=small_font)
    draw.text((left, 224), "叶 1–10.8（大叶片中心更高）   花 5–15（中心更高）   枝干 20 / 24；范围含不确定 ×1.2" if chinese else
              "Leaf 1-10.8 (large-leaf centers higher)   Flower 5-15 (centers higher)   Stem 20/24; ranges include uncertainty x1.2", fill=(25, 25, 30), font=small_font)
    # A separate swatch is necessary: forbidden contact is not a finite 24.
    hard_left, hard_right = 1270, 1530
    draw.rectangle((hard_left, top, hard_right, bottom), fill=HARD_COLOR, outline=(55, 55, 65), width=2)
    draw.text((hard_left, bottom + 17), "不可触碰 / 硬约束" if chinese else "Forbidden / hard constraint", fill=(25, 25, 30), font=small_font)
    draw.text((hard_left, 224), "花盆 · 泥土 · 小车 · 螺母" if chinese else "Pot / soil / cart / hardware", fill=(25, 25, 30), font=small_font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=95)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
