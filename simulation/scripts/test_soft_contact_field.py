import unittest
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
import igl
from soft_contact_field import SoftContactField


class SoftFieldTests(unittest.TestCase):
    def test_native_box_avoids_large_sphere_false_contact(self):
        size=np.array([.1,.005,.005])
        gap=SoftContactField.volume_distance(np.array([[0,.04,0],[0,0,0],[.12,0,0]]),
                                             (int(mujoco.mjtGeom.mjGEOM_BOX),size,None))
        np.testing.assert_allclose(gap,[.035,0,.02])

    def test_convex_mesh_inside_and_outside(self):
        v=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])
        h=ConvexHull(v);f=np.ascontiguousarray(h.simplices,dtype=np.int64)
        tree=igl.AABB();tree.init(v,f)
        distance=SoftContactField.volume_distance(np.array([[0,0,0],[2,0,0],[2,2,0],[1,0,0]]),
            (int(mujoco.mjtGeom.mjGEOM_MESH),np.zeros(3),(v,f,tree,h.equations)))
        np.testing.assert_allclose(distance,[0,1,np.sqrt(2),0],atol=1e-10)

    def test_triangle_cover_contains_source_corners(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".01"/></worldbody></mujoco>')
        v=np.array([[.02,0,0],[.02,.0006,0],[.02,0,.0006]])
        f=np.array([[0,1,2]],dtype=np.int64)
        field=SoftContactField(v,f,np.array([2]),np.array([False]),np.array([7]),np.array([1.,2.,3.]),m,[0])
        self.assertTrue(np.all(np.linalg.norm(v-field.centers[0],axis=1)<=field.radii[0]+1e-12))
        self.assertEqual(field.weights[0],3.)

    def test_visual_triangle_surface_is_scored_without_native_proxy(self):
        m=mujoco.MjModel.from_xml_string('''<mujoco><asset><mesh name="jaw"
          vertex="0 0 0 .01 0 0 0 .01 0 0 0 .01"
          face="0 1 2 0 3 1 0 2 3 1 3 2"/></asset><worldbody>
          <geom type="sphere" pos="1 0 0" size=".01"/>
          <geom name="visual_jaw" type="mesh" mesh="jaw" contype="0" conaffinity="0"/>
        </worldbody></mujoco>''')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        v=np.array([[.002,.002,.0001],[.0022,.002,.0001],[.002,.0022,.0001]])
        f=np.array([[0,1,2]],dtype=np.int64)
        field=SoftContactField(v,f,np.array([3]),np.array([False]),np.array([7]),np.full(3,5.),
                               m,[0],surface_geom_ids=[1])
        rows,risk=field.trace(d)
        self.assertGreater(risk,0)
        self.assertEqual(field.last_responsible_geom[0],1)
        self.assertIsNone(field.shapes[1][2][3])


if __name__=='__main__':unittest.main()
