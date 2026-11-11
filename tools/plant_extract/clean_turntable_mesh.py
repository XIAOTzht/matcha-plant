#!/usr/bin/env python3
"""Align input masks to MAtCha's rectified/cropped images, then crop an existing mesh.

For the zero-distortion PINHOLE turntable experiment. No training or mesh synthesis.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from auto_extract_plant import read_colmap, image_file


def intrinsic(view):
    return np.array([[view.fx, 0., view.cx], [0., view.fy, view.cy], [0., 0., 1.]])


def align_mask(mask, source, target):
    if mask.shape != (source.height, source.width):
        raise ValueError(f'Original mask dimensions do not match input camera: {source.name}')
    if not np.allclose(source.R, target.R, atol=2e-4) or not np.allclose(source.t, target.t, atol=2e-4):
        raise ValueError(f'Input/output poses differ: {source.name}; this helper requires the fixed-pose experiment')
    # All stages assume zero lens distortion. The full-resolution image mapping
    # used by run_mast3r.py is K_output @ inv(K_input), not a plain resize.
    H = intrinsic(target) @ np.linalg.inv(intrinsic(source))
    aligned = cv2.warpPerspective(mask, H, (target.width, target.height),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned, H


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='.../plus or .../minus')
    parser.add_argument('--input-dir', type=Path, required=True, help='Matching .../input_plus or input_minus')
    parser.add_argument('--mesh', type=Path, help='Defaults to tetra_meshes_nointerp final PLY')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.input_dir = args.input_dir.resolve()
    mesh = args.mesh or args.run_dir / 'tetra_meshes_nointerp' / 'tetra_mesh_binary_search_7.ply'
    if not mesh.is_file():
        raise FileNotFoundError(f'Missing extracted mesh: {mesh}')
    output = args.output_dir or args.run_dir / ('plant_cleanup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    source_views, _ = read_colmap(args.input_dir)
    target_views, _ = read_colmap(args.run_dir / 'mast3r_sfm')
    source_by_name = {v.name: v for v in source_views}
    if set(source_by_name) != {v.name for v in target_views}:
        raise ValueError('Input and reconstructed camera image names differ')
    aligned_dir = output / 'aligned_masks'
    aligned_dir.mkdir(parents=True)
    cards = []
    mappings = []
    for target in target_views:
        source = source_by_name[target.name]
        mask_path = image_file(args.input_dir / 'masks', source.name, is_mask=True)
        with Image.open(mask_path) as im:
            mask = np.array(im.convert('L'))
        mask, H = align_mask(mask, source, target)
        if np.count_nonzero(mask >= 100) < 100:
            raise ValueError(f'Empty/small transformed mask: {target.name}')
        Image.fromarray(mask).save(aligned_dir / (Path(target.name).stem + '_mask.png'))
        mappings.append({'image': target.name, 'source_size': [source.width, source.height],
                         'target_size': [target.width, target.height], 'homography': H.tolist()})
        # Overlay ALL masks on the actual images used in reconstruction.
        image_path = image_file(args.run_dir / 'mast3r_sfm' / 'images', target.name)
        with Image.open(image_path) as im:
            if im.size != (target.width, target.height):
                raise ValueError(f'Exported image dimensions differ from camera: {image_path}')
            im = im.convert('RGB')
            im.thumbnail((280, 340), Image.Resampling.LANCZOS)
            pix = np.array(im).copy()
        small = cv2.resize(mask, (pix.shape[1], pix.shape[0]), interpolation=cv2.INTER_NEAREST) >= 100
        # Foreground keeps original color; rejected background is red-tinted.
        pix[~small] = (.35 * pix[~small] + .65 * np.array([255, 40, 40])).astype(np.uint8)
        tile = Image.new('RGB', (300, 380), '#303640')
        tile.paste(Image.fromarray(pix), ((300-pix.shape[1])//2, 12))
        ImageDraw.Draw(tile).text((10, 357), Path(target.name).name, fill='white')
        cards.append(tile)
        print(f'Aligned mask: {target.name} {source.width}x{source.height} -> {target.width}x{target.height}', flush=True)
    sheet = Image.new('RGB', (1200, 380*((len(cards)+3)//4)), '#303640')
    for i, tile in enumerate(cards):
        sheet.paste(tile, ((i%4)*300, (i//4)*380))
    sheet.save(output / 'mask_alignment.jpg')
    (output / 'mask_mapping.json').write_text(json.dumps(mappings, indent=2), encoding='utf-8')
    command = [sys.executable, str(Path(__file__).with_name('auto_extract_plant.py')),
               '--mesh', str(mesh), '--sfm-dir', str(args.run_dir / 'mast3r_sfm'),
               '--masks-dir', str(aligned_dir), '--output-dir', str(output)]
    subprocess.run(command, check=True)
    print(f'TURNTABLE_CLEANUP_SUCCESS: {output}', flush=True)


if __name__ == '__main__':
    main()
