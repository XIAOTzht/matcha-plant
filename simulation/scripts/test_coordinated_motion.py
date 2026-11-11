import unittest
from pathlib import Path
import numpy as np
from coordinated_motion import controls,path_proposals,motion_metrics,select_candidate,validate_settings
from plant_scene import read_json


class CoordinationTests(unittest.TestCase):
    def setUp(self):
        self.s=read_json(Path(__file__).resolve().parents[1]/'configs/coordinated_motion.json')
        self.path=np.array([[0,0,0,0,1.4],[-.3,-.5,.2,.3,1.6],[-.6,-.4,.4,.5,1.5]])

    def test_all_joints_can_change_with_exact_endpoints(self):
        proposals=list(path_proposals(self.path,np.full(5,-3.),np.full(5,3.),self.s))
        phase,anchor=controls(self.path)
        extent=np.zeros(5)
        for row in proposals:
            p=row['path'];np.testing.assert_array_equal(p[[0,-1]],self.path[[0,-1]])
            extent=np.maximum(extent,np.max(abs(p-anchor),axis=0))
            self.assertTrue(np.all(abs(p)<=3))
        self.assertTrue(np.all(extent>.1))

    def test_out_of_bounds_rejected_not_clipped(self):
        lo=self.path.min(0)-.01;hi=self.path.max(0)+.01
        for row in path_proposals(self.path,lo,hi,self.s):
            self.assertTrue(np.all(row['path']>=lo));self.assertTrue(np.all(row['path']<=hi))

    def test_extra_loops_do_not_raise_motion_reward(self):
        p=np.array([[0]*5,[1]*5,[0]*5],float)
        a=motion_metrics(p,self.s);b=motion_metrics(np.vstack([p,p[1:],p[1:]]),self.s)
        self.assertLessEqual(b['coordination_score'],a['coordination_score'])

    def candidate(self,name,cost,seconds,quality,travel=3.):
        return dict(source=name,objective=cost+.05*seconds,persistence_cost=cost,duration_s=seconds,
                    motion=dict(coordination_score=quality,joint_travel_rad=travel))

    def test_presentation_preference_cannot_hide_extra_contact_or_time(self):
        rows=[self.candidate('efficient',100,10,.3),self.candidate('coordinated',101,14,.7),
              self.candidate('too_much_contact',116,13,.9),self.candidate('too_slow',100,17,1.)]
        selected,report=select_candidate(rows,self.s)
        self.assertEqual(selected['source'],'coordinated')
        self.assertGreater(report['selected_objective_delta'],0)
        self.assertEqual(report['selected_persistence_delta'],1)
        self.s['selection_mode']='efficient'
        self.assertEqual(select_candidate(rows,self.s)[0]['source'],'efficient')

    def test_zero_contact_does_not_allow_manufactured_contact(self):
        rows=[self.candidate('clean',0,10,.2),self.candidate('contact',.001,10,.8)]
        self.assertEqual(select_candidate(rows,self.s)[0]['source'],'clean')

    def test_more_expressive_cannot_increase_baseline_exposure(self):
        rows=[self.candidate('unchanged_baseline',100,10,.3),
              self.candidate('more_expressive',110,14,.9)]
        self.assertEqual(select_candidate(rows,self.s)[0]['source'],'unchanged_baseline')

    def test_nan_and_limits_rejected(self):
        self.s['max_duration_ratio']=float('nan')
        with self.assertRaises(ValueError):validate_settings(self.s)


if __name__=='__main__':unittest.main()
