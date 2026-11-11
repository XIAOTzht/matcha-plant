"""Behavioral checks for narrow rim repair and nearby genuine stems."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from refine_leaf_rim import complete_rim


def scene():
    # A leaf sheet with a one-row misclassified rim.
    x, y = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, .5, 101))
    xyz = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    grid = np.arange(x.size).reshape(x.shape)
    a, b, c, d = (v.ravel() for v in (grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]))
    tri = np.vstack([np.column_stack([a, b, c]), np.column_stack([b, d, c])])
    labels = np.full(len(xyz), 2, np.uint8)
    rim = grid[-1]
    labels[rim] = 1
    # A real tube starts at the leaf base and extends away from it. Another
    # identical tube almost touches the sheet but has no mesh connection.
    angle, height = np.meshgrid(np.arange(12)*np.pi/6, np.linspace(0, .15, 31))
    tube = np.column_stack([.5+.006*np.cos(angle.ravel()), .5+height.ravel(), .006*np.sin(angle.ravel())])
    rings = np.arange(len(tube)).reshape(31, 12)
    u, v = rings[:-1].ravel(), np.roll(rings[:-1], -1, axis=1).ravel()
    w, q = rings[1:].ravel(), np.roll(rings[1:], -1, axis=1).ravel()
    tube_tri = np.vstack([np.column_stack([u,v,w]), np.column_stack([v,q,w])])
    n = len(xyz)
    attached = np.arange(n, n+len(tube))
    disconnected = np.arange(n+len(tube), n+2*len(tube))
    tri = np.vstack([tri, tube_tri+n, [[int(rim[100]),n,n+6]], tube_tri+n+len(tube)])
    near = tube + np.array([.15, -.15, .008])
    xyz = np.vstack([xyz, tube, near])
    labels = np.r_[labels, np.ones(2*len(tube), np.uint8)]
    bud = np.zeros(len(xyz), bool)
    return xyz, tri, labels, bud, rim, attached, disconnected


class LeafRimTests(unittest.TestCase):
    def run_scene(self, xyz, tri, labels, bud, region=None):
        if region is None:
            region = labels == 2
        return complete_rim(xyz, tri, labels, bud, region, max_distance=.012,
                            support_radius=.04, voxel_size=.004, threads=2)

    def test_sheet_rim_repaired_without_jumping_to_overlapping_tube(self):
        xyz, tri, labels, bud, rim, attached, disconnected = scene()
        changed, diag = self.run_scene(xyz, tri, labels, bud)
        self.assertGreater(changed[rim].mean(), .95)
        self.assertFalse(changed[disconnected].any())
        # Only the ambiguous attachment band may change, never the stem body.
        self.assertFalse(changed[attached[xyz[attached, 1] > .52]].any())
        self.assertFalse(changed[labels == 2].any())
        self.assertLessEqual(float(diag['distance'].max()), .012001)

    def test_flower_pot_soil_and_protected_buds_remain_unchanged(self):
        xyz, tri, labels, bud, rim, _, _ = scene()
        labels[rim[:10]] = 3
        labels[rim[10:20]] = 4
        labels[rim[20:30]] = 5
        bud[rim[30:40]] = True
        changed, _ = self.run_scene(xyz, tri, labels, bud)
        self.assertFalse(changed[rim[:40]].any())
        self.assertTrue(changed[rim[60:90]].all())

    def test_same_decision_after_scale_and_translation(self):
        xyz, tri, labels, bud, _, _, _ = scene()
        first, _ = self.run_scene(xyz, tri, labels, bud)
        second, _ = self.run_scene(xyz*3.25+[7,-4,2], tri, labels, bud)
        np.testing.assert_array_equal(first, second)

    def test_no_reliable_leaf_is_a_noop(self):
        xyz, tri, labels, bud, _, _, _ = scene()
        changed, diag = self.run_scene(xyz, tri, labels, bud, np.zeros(len(xyz), bool))
        self.assertFalse(changed.any())
        self.assertEqual(len(diag['indices']), 0)


if __name__ == '__main__':
    unittest.main()
