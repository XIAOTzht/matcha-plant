"""Small geometric invariants, independent of the current plant."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from plyfile import PlyData, PlyElement
from organ_geometry import split_organs, surface_areas, smooth_center_field
from split_plant_organs import read_mesh, write_fields


def lobes(connected=True, step=1.):
    xx, yy = np.meshgrid(np.arange(-23, 24, step), np.arange(-11, 12, step))
    keep = ((xx+12)**2 + yy**2 <= 100) | ((xx-12)**2 + yy**2 <= 100)
    if connected:
        keep |= (abs(xx) <= 12) & (abs(yy) <= 1.5)
    ids = np.full(xx.shape, -1, int)
    ids[keep] = np.arange(keep.sum())
    xyz = np.column_stack((xx[keep], yy[keep], np.zeros(keep.sum())))
    tri = []
    for row in range(ids.shape[0]-1):
        for col in range(ids.shape[1]-1):
            a,b,c,d = ids[row,col],ids[row,col+1],ids[row+1,col],ids[row+1,col+1]
            if min(a,b,c) >= 0:
                tri.append([a,b,c])
            if min(b,c,d) >= 0:
                tri.append([b,d,c])
    return xyz, np.array(tri, np.int32)


class OrganTests(unittest.TestCase):
    def test_single_leaf_center_field_has_gradual_edge_to_center_profile(self):
        xx,yy=np.meshgrid(np.linspace(-1,1,51),np.linspace(-1,1,51))
        xyz=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        tri=[]
        for y in range(50):
            for x in range(50):
                a=y*51+x;tri.extend([[a,a+1,a+51],[a+1,a+52,a+51]])
        field,_,_=smooth_center_field(xyz,np.asarray(tri),200)
        line=field.reshape(51,51)[25,:26]
        self.assertTrue(np.all(np.diff(line)>=-1e-5))
        self.assertLess(line[0],.05)
        self.assertGreater(line[-1],.98)
        self.assertLess(np.diff(line).max(),.09)

    def test_connected_lobes_split_and_center_rim(self):
        xyz, f = lobes()
        a, r = split_organs(xyz, f, np.full(len(xyz), 2), min_area_mm2=2., max_pixels=180)
        self.assertEqual(len(r), 2)
        self.assertTrue(np.all(a['vertex_instance_id'] > 0))
        for region in r:
            m = a['vertex_instance_id'] == region['instance_id']
            self.assertAlmostEqual(float(a['vertex_center_score'][m].max()), 1.)
            self.assertLess(float(np.quantile(a['vertex_center_score'][m], .1)), .3)
        self.assertAlmostEqual(sum(t['reconstructed_surface_area_mm2'] for t in r), surface_areas(xyz, f).sum())

    def test_disconnected_neighbors_do_not_merge(self):
        xyz, f = lobes(connected=False)
        a, _ = split_organs(xyz, f, np.full(len(xyz), 2), min_area_mm2=2., max_pixels=180)
        left = set(a['vertex_instance_id'][xyz[:, 0] < 0])
        right = set(a['vertex_instance_id'][xyz[:, 0] > 0])
        self.assertFalse(left & right)

    def test_rigid_transform_and_scale_area(self):
        xyz, f = lobes()
        a, r = split_organs(xyz, f, np.full(len(xyz), 2), min_area_mm2=2., max_pixels=180)
        rng = np.random.default_rng(99)
        rotation, _ = np.linalg.qr(rng.normal(size=(3,3)))
        b, s = split_organs(xyz @ rotation + [11, 52, -33], f, np.full(len(xyz), 2), min_area_mm2=2., max_pixels=180)
        self.assertEqual(len(r), len(s))
        np.testing.assert_allclose(sorted(t['reconstructed_surface_area_mm2'] for t in r),
                                   sorted(t['reconstructed_surface_area_mm2'] for t in s), rtol=.03)
        np.testing.assert_allclose(surface_areas(2*xyz, f), 4*surface_areas(xyz, f))

    def test_area_independent_of_tessellation(self):
        x = np.array([[0,0,0], [2,0,0], [0,3,0]], float)
        coarse = surface_areas(x, np.array([[0,1,2]])).sum()
        fine_x = np.vstack((x, [1,0,0], [1,1.5,0], [0,1.5,0]))
        fine = surface_areas(fine_x, np.array([[0,3,5], [3,1,4], [5,4,2], [3,4,5]])).sum()
        self.assertEqual(coarse, fine)

    def test_user_markers_replace_automatic_centers(self):
        xyz, f = lobes()
        center = int(np.argmin(np.linalg.norm(xyz - [12,0,0], axis=1)))
        a, r = split_organs(xyz, f, np.full(len(xyz), 17), class_ids=(17, 22),
                            min_area_mm2=2., max_pixels=180, explicit_seeds=[{'vertex_index':center}])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]['class_id'], 17)
        self.assertTrue(np.all(a['vertex_instance_review']))

    def test_small_fragments_preserved_as_unresolved(self):
        xyz = np.array([[0,0,0], [1,0,0], [0,1,0]], float)
        a, r = split_organs(xyz, np.array([[0,1,2]]), np.full(3,2), min_area_mm2=10.)
        self.assertEqual(len(r), 0)
        np.testing.assert_array_equal(a['vertex_instance_id'], [-1,-1,-1])

    def test_preserves_fields_and_non_target_class(self):
        v = np.zeros(4, dtype=[('x','f4'),('y','f4'),('z','f4'),('label','u1'),('custom','f4')])
        v['x'] = [0,1,0,0]; v['y'] = [0,0,1,0]; v['z'] = [0,0,0,1]
        v['label'] = [2,2,2,6]; v['custom'] = np.arange(4)
        f = np.zeros(2, dtype=[('vertex_indices','i4',(3,)), ('label','u1')])
        f['vertex_indices'] = [[0,1,2],[0,1,3]]; f['label'] = [2,6]
        with TemporaryDirectory() as tmp:
            src, out = Path(tmp)/'src.ply', Path(tmp)/'out.ply'
            PlyData([PlyElement.describe(v,'vertex'),PlyElement.describe(f,'face')], text=True).write(str(src))
            ply, x, t, lab = read_mesh(src, mm_per_unit=1000)
            self.assertEqual(x[1,0], 1000.)
            a, _ = split_organs(x, t, lab, min_area_mm2=1., max_pixels=100)
            self.assertEqual(a['vertex_instance_id'][-1], 0)
            write_fields(out, ply, {'organ_id': a['vertex_instance_id']})
            q = PlyData.read(str(out), mmap=False)
            for name in v.dtype.names:
                np.testing.assert_array_equal(q['vertex'].data[name], v[name])
            np.testing.assert_array_equal(q['face'].data['label'], f['label'])
            np.testing.assert_array_equal(np.stack(q['face'].data['vertex_indices']), f['vertex_indices'])


if __name__ == '__main__':
    unittest.main()
