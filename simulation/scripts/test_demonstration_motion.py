import json
from pathlib import Path
import unittest
import numpy as np
from demonstration_motion import proposals,route_metrics,presentation_passes,presentation_score,validate_presentation_settings


class DemonstrationTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]
        self.s=json.loads((root/'configs/demonstration_motion.json').read_text())['presentation']

    def test_proposals_repeat_with_fixed_endpoints_and_seed(self):
        base=np.array([[0]*5,[.1]*5,[.2]*5,[.3]*5,[.4]*5])
        a=list(proposals(base,42));b=list(proposals(base,42))
        self.assertEqual(len(a),74)
        for (p,label),(q,other)in zip(a,b):
            np.testing.assert_array_equal(p[[0,-1]],base[[0,-1]])
            np.testing.assert_array_equal(p,q);self.assertEqual(label,other)

    def test_longer_same_line_is_not_enough(self):
        tips=np.array([[0.,0,0],[.1,0,0],[.2,0,0]])
        q=np.zeros((3,5));q[:,0]=[0,1,2]
        r=route_metrics(tips,q,tips,q)
        self.assertEqual(r['route_departure_mm'],0)
        self.assertEqual(r['added_lift_mm'],0)
        self.assertFalse(presentation_passes(r,dict(joint_span_deg=[60]*5),self.s))

    def test_more_loops_rejected_and_reward_is_bounded(self):
        route=dict(route_departure_mm=100,added_lift_mm=70,tip_length_ratio=1.3,joint_travel_ratio=1.2)
        motion=dict(joint_span_deg=[40,50,45,40,70])
        self.assertTrue(presentation_passes(route,motion,self.s))
        self.assertEqual(presentation_score(route,motion,self.s),1.)
        route['route_departure_mm']=1000
        self.assertEqual(presentation_score(route,motion,self.s),1.)
        route['tip_length_ratio']=2
        self.assertFalse(presentation_passes(route,motion,self.s))

    def test_invalid_presentation_targets_rejected(self):
        validate_presentation_settings(self.s)
        self.s['desired_roll_span_deg']=float('nan')
        with self.assertRaises(ValueError):validate_presentation_settings(self.s)


if __name__=='__main__':unittest.main()
