import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
from explore_watering_poses import position_cone_residual
from build_weighted_corridors import screening_report


class PostureSearchTests(unittest.TestCase):
    def test_down_cone_allows_distinct_roll_and_pitch(self):
        # A tip-centered wrist can change attitude without changing position.
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
          <joint name="tilt" axis="0 1 0" range="-1 1"/>
          <geom type="sphere" size=".01" mass=".01"/>
          <body><joint name="roll" axis="0 0 1" range="-3 3"/>
          <geom type="sphere" size=".01" mass=".01"/><site name="nozzle_tip"/></body>
        </body></worldbody></mujoco>''')
        data=mujoco.MjData(model)
        checker=SimpleNamespace(model=model,data=data,q_lo=np.array([-1.,-3.]),q_hi=np.array([1.,3.]))
        def set_q(q):
            data.qpos[:]=q;mujoco.mj_forward(model,data)
        checker.set_q=set_q
        cosine=np.cos(np.deg2rad(20))
        for q in (np.array([.2,1.]),np.array([-.2,-1.])):
            r=position_cone_residual(q,checker,np.zeros(3),q,cosine)
            np.testing.assert_allclose(r,0,atol=1e-12)
        q=np.array([.5,0.]);r=position_cone_residual(q,checker,np.zeros(3),q,cosine)
        self.assertGreater(r[3],0)

    def test_same_soil_target_keeps_multiple_arm_branches(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);scene=root/'scene.xml'
            checker=SimpleNamespace(scene=scene,root=root,gripper_rad=-.08)
            base=dict(id=0,target_id=7,target_m=[0,0,.1],soil_surface_world_m=[0,0,.09],
                source_face_index=10,risk_proxy=1.,q_arm_rad=[0]*5)
            other={**base,'q_arm_rad':[0,0,0,1,2]}
            report=screening_report(checker,[base,other],root)
            self.assertEqual(len(report['entries']),1)
            self.assertEqual(report['entries'][0]['id'],7)
            self.assertEqual(len(report['entries'][0]['checked_branches']),2)


if __name__=='__main__':unittest.main()
