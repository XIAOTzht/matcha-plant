"""Heuristic vertical-axis estimation for plant meshes.

The default assumes the largest coordinate extent is the vertical extent. This
is convenient for upright reconstructions; pass -UpAxis explicitly for rotated
or unusually wide plants.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import plantseg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    xyz, _, _ = plantseg.mesh_data(args.input)
    extent = xyz.max(0) - xyz.min(0)
    axis = int(extent.argmax())
    print('xyz'[axis])
    print({'axis': 'xyz'[axis], 'extent': extent.tolist()}, file=sys.stderr)


if __name__ == '__main__':
    main()
