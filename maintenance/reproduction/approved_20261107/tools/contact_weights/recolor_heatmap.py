#!/usr/bin/env python3
"""Recolor an existing weighted PLY without changing geometry or weights."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData
from weight_palette import risk_colors, PALETTE_VERSION


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--arrays', type=Path, required=True,
                    help='contact_weights.npz matching the PLY')
    args = ap.parse_args()
    if args.output.resolve() == args.mesh.resolve():
        raise SystemExit('Use a separate output file; source heatmap is preserved.')
    ply = PlyData.read(str(args.mesh), mmap=False, known_list_len={'face': {'vertex_indices': 3}})
    if 'vertex' not in ply or not all(k in ply['vertex'].data.dtype.names for k in ('red','green','blue')):
        raise SystemExit('PLY has no RGB vertex fields.')
    with np.load(args.arrays, allow_pickle=False) as z:
        weights = np.asarray(z['vertex_effective_weight'], np.float32)
        hard = np.asarray(z['vertex_hard_collision'], bool)
        if len(weights) != len(ply['vertex'].data):
            raise SystemExit('Weights and PLY vertex count differ.')
        vertex = ply['vertex'].data
        for name, expected in [('contact_weight', weights), ('hard_collision', hard)]:
            if name not in vertex.dtype.names or not np.array_equal(vertex[name], expected):
                raise SystemExit(f'Weights and PLY {name} do not match; use the original matching NPZ.')
        if 'vertex_labels' in z and 'label' in vertex.dtype.names and not np.array_equal(vertex['label'], z['vertex_labels']):
            raise SystemExit('Weights and PLY semantic labels differ.')
        rgb = risk_colors(weights, hard)
    data = ply['vertex'].data.copy()
    data['red'], data['green'], data['blue'] = rgb.T
    from plyfile import PlyElement
    elements = []
    for element in ply.elements:
        if element.name == 'vertex':
            elements.append(PlyElement.describe(data, 'vertex', comments=list(element.comments)))
        else:
            elements.append(element)
    out = args.output.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=False, byte_order='<', comments=list(ply.comments) + [f'recolored with {PALETTE_VERSION}; weights unchanged']).write(str(out))
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
