"""Tiny visual color bins must compile without changing the plant surface."""
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from build_watering_scene import display_meshes, write_stl


RECORD = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])


class DisplayImportTests(unittest.TestCase):
    def compile_stl(self, path):
        return mujoco.MjModel.from_xml_string(
            '<mujoco><asset><mesh name="visual" file="'+path.as_posix()+'" inertia="shell" maxhullvert="64"/>'
            '</asset><worldbody><geom type="mesh" mesh="visual" contype="0" conaffinity="0" mass="0"/>'
            '</worldbody></mujoco>')

    def test_single_triangle_preserves_plane_area_orientation_and_compiles(self):
        for size in (.01, .0001):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'one.stl'
                points=np.array([[.1,.2,.3],[.1+size,.2,.3],[.1,.2+size,.3]],dtype='f4')
                before=points.copy()
                count=write_stl(p,points,np.array([[0,1,2]]))
                self.assertEqual(count,3)
                triangles=np.fromfile(p,dtype=RECORD,offset=84)['vertices']
                self.assertEqual(len(np.unique(triangles.reshape(-1,3),axis=0)),4)
                np.testing.assert_array_equal(points,before)
                np.testing.assert_array_equal(triangles[:,:,2],np.full((3,3),points[0,2]))
                normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                self.assertTrue((normals[:,2]>0).all())
                original=np.cross(points[1]-points[0],points[2]-points[0])
                np.testing.assert_allclose(normals.sum(0),original,rtol=2e-5,atol=1e-15)
                model=self.compile_stl(p)
                self.assertEqual(model.nmeshvert,4)
                self.assertFalse(model.geom_contype.any());self.assertFalse(model.geom_conaffinity.any())

    def test_regular_quad_unchanged_and_degenerate_triangle_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'quad.stl'
            points=np.array([[0,0,0],[.01,0,0],[.01,.01,0],[0,.01,0]],float)
            faces=np.array([[0,1,2],[0,2,3]])
            self.assertEqual(write_stl(p,points,faces),2)
            np.testing.assert_array_equal(np.fromfile(p,dtype=RECORD,offset=84)['vertices'],points[faces].astype('f4'))
            self.compile_stl(p)
            self.assertEqual(write_stl(root/'degenerate.stl',points,np.array([[0,0,1]])),0)
            self.assertFalse((root/'degenerate.stl').exists())

    def test_extreme_leaf_color_bin_can_be_just_one_triangle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            # Two disconnected leaves with separate weight bins. Both contain
            # a single triangle, exactly the server's failing display case.
            points=np.array([[0,0,0],[.01,0,0],[0,.01,0],[.03,0,0],[.04,0,0],[.03,.01,0]],float)
            faces=np.array([[0,1,2],[3,4,5]])
            weights=dict(face_labels=np.array([2,2]),vertex_hard_collision=np.zeros(6,bool),
                face_hard_collision=np.zeros(2,bool),vertex_effective_weight=np.array([1.,1,1,10,10,10]))
            saved={k:a.copy() for k,a in weights.items()}
            records=display_meshes(root,points,faces,np.full(6,2),weights,
                dict(default_max_faces=20,max_faces_by_class={},soft_color_bins_per_class=6),
                dict(soft_anchors=[dict(weight=1,rgb=[0,255,0]),dict(weight=10,rgb=[255,255,0])],hard_collision_rgb=[255,0,0]))
            self.assertEqual([r['name'] for r in records],['plant_class_2_color_1','plant_class_2_color_6'])
            for r in records:self.compile_stl(r['file'])
            for k,a in saved.items():np.testing.assert_array_equal(weights[k],a)


if __name__ == '__main__':unittest.main()
