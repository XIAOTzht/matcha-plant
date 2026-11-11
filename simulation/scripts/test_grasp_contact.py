import unittest
import mujoco
from plan_watering_path import allowed_grasp_contact


class GraspContactTests(unittest.TestCase):
    def setUp(self):
        self.model=mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <geom name="nozzle_grasp_handle" type="sphere" size=".01"/>
          <geom name="placeholder_nozzle_cylinder" type="sphere" size=".01"/>
          <geom name="fixed_jaw_sph_tip1" type="sphere" size=".01"/>
          <geom name="moving_jaw_sph_tip1" type="sphere" size=".01"/>
          <geom name="other_link" type="sphere" size=".01"/>
        </worldbody></mujoco>""")
        self.config=dict(grasp_handle=True,gripper_rad=-.08,grasp_contact=dict(
            enabled=True,pad_geoms=['fixed_jaw_sph_tip1','moving_jaw_sph_tip1'],
            max_proxy_overlap_mm=.6,angle_tolerance_rad=.002))

    def allowed(self,a,b,gap=-.0005,angle=-.08):
        return allowed_grasp_contact(self.model,self.model.geom(a).id,self.model.geom(b).id,
                                     gap,angle,self.config)

    def test_only_handle_pad_is_permitted(self):
        self.assertTrue(self.allowed('nozzle_grasp_handle','fixed_jaw_sph_tip1'))
        self.assertTrue(self.allowed('moving_jaw_sph_tip1','nozzle_grasp_handle'))
        self.assertFalse(self.allowed('placeholder_nozzle_cylinder','fixed_jaw_sph_tip1'))
        self.assertFalse(self.allowed('nozzle_grasp_handle','other_link'))
        self.assertFalse(self.allowed('other_link','fixed_jaw_sph_tip1'))
        self.assertFalse(self.allowed('nozzle_grasp_handle','nozzle_grasp_handle'))

    def test_excess_overlap_or_wrong_angle_is_rejected(self):
        self.assertFalse(self.allowed('nozzle_grasp_handle','fixed_jaw_sph_tip1',gap=-.001))
        self.assertFalse(self.allowed('nozzle_grasp_handle','fixed_jaw_sph_tip1',angle=.35))
        self.config['grasp_contact']['enabled']=False
        self.assertFalse(self.allowed('nozzle_grasp_handle','fixed_jaw_sph_tip1'))


if __name__=='__main__':unittest.main()
