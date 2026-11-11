import json
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from trajectory_tools import make_timing_plan,time_parameterize,_septic_hermite_coefficients,_evaluate_septic,actuator_command
from natural_motion import significant_reversals,natural_metrics,motion_passes,validate_natural_settings
from plan_watering_path import timed_path_score
from attitude_escape_search import export_path

ROOT=Path(__file__).resolve().parents[1]


class NaturalMotionTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/weighted_trajectory_validation_v5.json').read_text())
        self.settings=json.loads((ROOT/'configs/natural_motion.json').read_text())
        self.path=np.array([[0]*5,[.2,.1,-.2,.1,.3],[.3,.2,-.1,.25,.4],[.4,.3,.2,.3,.1]])

    def test_fixed_plan_sampling_and_json_roundtrip(self):
        plan=make_timing_plan(self.path,self.config)
        plan=json.loads(json.dumps(plan))
        a=time_parameterize(self.path,self.config,.04,plan)
        b=time_parameterize(self.path,self.config,.005,plan)
        common=np.arange(0,int(b['time_s'][-1]/.04))*8
        np.testing.assert_allclose(a['q_ref'][:len(common)],b['q_ref'][common],atol=1e-10)
        np.testing.assert_allclose(a['q_ref'][[0,-1]],self.path[[0,-1]],atol=1e-10)
        self.assertEqual(a['time_s'][-1],b['time_s'][-1])

    def test_shared_derivatives_c3_and_analytic_limits(self):
        plan=make_timing_plan(self.path,self.config)
        h=plan['segment_duration_s'];v=np.array(plan['knot_velocity_rad_s']);a=np.array(plan['knot_acceleration_rad_s2']);j=np.array(plan['knot_jerk_rad_s3'])
        end=None
        for i,duration in enumerate(h):
            c=_septic_hermite_coefficients(self.path[i],self.path[i+1],v[i],v[i+1],a[i],a[i+1],j[i],j[i+1],duration)
            values=_evaluate_septic(c,duration,np.array([0.,1.]))
            if end is not None:
                for prev,now in zip(end,values):np.testing.assert_allclose(prev,now[0],atol=2e-9)
            end=[x[-1]for x in values]
        trajectory=time_parameterize(self.path,self.config,.005,plan)
        self.assertTrue(np.all(abs(trajectory['qd_ref'])<=np.array(self.config['velocity_limit_rad_s'])+1e-8))
        self.assertTrue(np.all(abs(trajectory['qdd_ref'])<=np.array(self.config['acceleration_limit_rad_s2'])+1e-8))

    def test_mismatched_or_over_limit_plan_rejected(self):
        plan=make_timing_plan(self.path,self.config)
        with self.assertRaises(ValueError):time_parameterize(self.path+.1,self.config,.01,plan)
        plan['knot_velocity_rad_s'][1]=[100]*5
        with self.assertRaises(ValueError):time_parameterize(self.path,self.config,.01,plan)

    def test_hysteresis_ignores_tiny_reverse_but_counts_full_reversals(self):
        deg=np.deg2rad
        self.assertEqual(significant_reversals(deg([0,20,19,21,40,39,41])),0)
        self.assertEqual(significant_reversals(deg([0,20,10,0])),1)
        self.assertEqual(significant_reversals(deg([0,20,0,20])),2)

    def test_normalized_jerk_not_improved_by_uniform_slowdown(self):
        plan=make_timing_plan(self.path,self.config);a=time_parameterize(self.path,self.config,.005,plan)
        b={k:(v.copy()if isinstance(v,np.ndarray)else v)for k,v in a.items()}
        b['time_s']*=2;b['qd_ref']/=2;b['qdd_ref']/=4;b['qddd_ref']/=8
        self.assertAlmostEqual(natural_metrics(a,self.settings)['normalized_jerk'],natural_metrics(b,self.settings)['normalized_jerk'],places=6)

    def test_timed_score_rejects_hard_interpolated_collision_before_contact(self):
        class Checker:
            def check(self,q,evaluate_risk=False):return (False,{})
            def contact_trace(self):raise AssertionError('Collision must reject first')
        score=timed_path_score(Checker(),self.path,self.config,timing_plan=make_timing_plan(self.path,self.config))
        self.assertFalse(np.isfinite(score['objective']))

    def test_natural_acceptance_keeps_activity_and_time_constraints(self):
        row=dict(duration_s=19.,significant_reversals=[0,1,1,1,1],joint_span_deg=[40,50,33,38,43],
                 longest_interior_low_speed_s=.4,normalized_jerk=100.)
        self.assertTrue(motion_passes(row,self.settings,dict(normalized_jerk=120.)))
        row['duration_s']=20.1
        self.assertFalse(motion_passes(row,self.settings))

    def test_invalid_natural_budget_rejected(self):
        self.settings['search_seconds']=float('nan')
        with self.assertRaises(ValueError):validate_natural_settings(self.settings)

    def test_export_preserves_grasp_endpoints_and_frozen_timing(self):
        class Checker:
            gripper_rad=-.08
            contact_trees=SimpleNamespace(report=lambda:{})
            data=SimpleNamespace(site_xpos=np.zeros((1,3)))
            model=SimpleNamespace(site=lambda name:SimpleNamespace(id=0))
            def edge_cost(self,*a,**k):return (True,0)
            def set_q(self,q):self.data.site_xpos[0]=q[:3]
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);source=root/'source';source.mkdir();out=root/'out';out.mkdir()
            (source/'path_report.json').write_text('{}');config_path=root/'config.json'
            config_path.write_text(json.dumps(self.config))
            plan=make_timing_plan(self.path,self.config)
            item=dict(path=self.path,timing_plan=plan,persistence_cost=1.,exposure_integral=1.,duration_s=10.,objective=1.5,source='test')
            report=dict(start_q_rad=self.path[0].tolist(),goal_q_rad=self.path[-1].tolist(),goal_id=1)
            export_path(root,out,Checker(),source,report,item,[item],{},self.config,config_path)
            with np.load(out/'trajectory_q.npz')as z:
                np.testing.assert_array_equal(z['q_arm_rad'][[0,-1]],self.path[[0,-1]])
                np.testing.assert_array_equal(z['q_gripper_rad'],np.full(len(self.path),-.08))
            saved=json.loads((out/'timing_plan.json').read_text())
            self.assertEqual(saved,plan)

    def test_inverse_feedforward_and_legacy_commands_are_explicit(self):
        import mujoco
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j" type="hinge" damping=".2"/><geom type="sphere" size=".1" mass="1"/></body></worldbody><actuator><position joint="j" kp="100" kv="2"/></actuator></mujoco>')
        data=mujoco.MjData(model);inverse=mujoco.MjData(model)
        q=np.array([.2]);v=np.array([.1]);a=np.array([.3]);adr=np.array([0])
        cfg=dict(position_lead_s=.01,bias_feedforward=True)
        data.qpos[:]=q;data.qvel[:]=v;mujoco.mj_forward(model,data)
        legacy=actuator_command(model,data,cfg,q,v,a,adr,adr,adr)
        np.testing.assert_allclose(legacy,q+.01*v+data.qfrc_bias/100)
        command=actuator_command(model,data,cfg,q,v,a,adr,adr,adr,inverse)
        np.testing.assert_allclose(command,q+(inverse.qfrc_inverse+2*v)/100)


if __name__=='__main__':unittest.main()
