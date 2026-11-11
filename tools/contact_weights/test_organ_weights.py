import copy
import json
import math
from pathlib import Path
import unittest
import numpy as np
from build_organ_contact_weights import build_arrays, center_multiplier, leaf_size_multiplier, uncertainty_factors
from contact_cost import ContactSample, instantaneous_cost, distance_factor, score_trace


HERE = Path(__file__).parent


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((HERE/'contact_policy.json').read_text(encoding='utf-8'))
        self.gp = json.loads((HERE/'organ_weight_policy.json').read_text(encoding='utf-8'))

    def test_center_and_leaf_size_monotone_bounded(self):
        centers = center_multiplier(np.linspace(0,1,51))
        sizes = leaf_size_multiplier(np.geomspace(1,1e9,51))
        self.assertTrue(np.all(np.diff(centers)>0))
        self.assertTrue(np.all(np.diff(sizes)>0))
        self.assertEqual(centers[0],1)
        self.assertEqual(centers[-1],2)
        self.assertTrue(np.all((sizes>=1)&(sizes<2)))
        self.assertGreater(leaf_size_multiplier(5000),leaf_size_multiplier(500))

    def test_hard_boundaries_and_geometry_only_on_leaf_flower(self):
        self.gp['uncertainty_scope']='vertex' # Legacy field compatibility.
        labels = np.array([2,2,3,1,4,6])
        uncertain = np.array([0,1,0,1,0,0],bool)
        faces = np.array([[0,1,2],[0,3,4],[3,4,5]])
        organ = dict(vertex_instance_id=np.array([1,1,2,0,0,0]),vertex_center_score=np.array([0,1,1,0,0,0]),
            vertex_organ_area_mm2=np.array([2000,2000,3000,0,0,0]),vertex_instance_review=np.ones(6,bool))
        a = build_arrays(labels,uncertain,faces,np.array([2,1,6]),np.zeros(3,bool),organ,self.policy,self.gp)
        self.assertAlmostEqual(a['vertex_effective_weight'][0],1.8888889,places=5)
        self.assertAlmostEqual(a['vertex_effective_weight'][1],6.8,places=5)
        self.assertEqual(a['vertex_effective_weight'][2],12.5)
        self.assertEqual(a['vertex_effective_weight'][3],24.)
        np.testing.assert_array_equal(a['vertex_hard_collision'],[0,0,0,0,1,1])
        np.testing.assert_array_equal(a['face_hard_collision'],[0,1,1])
        self.assertTrue(np.all(a['face_effective_weight'][1:]==0))
        np.testing.assert_array_equal(a['vertex_size_multiplier'][2:], [1,1,1,1])

    def test_uncertain_speckles_do_not_create_organ_weight_speckles(self):
        labels=np.array([2,2,2,3,3,1,4])
        ids=np.array([1,1,1,2,2,0,0])
        uncertain=np.array([0,1,0,1,0,1,0],bool)
        area=np.array([1.,2.,1.,1.,3.,1.,1.])
        f=uncertainty_factors(labels,uncertain,ids,dict(vertex_surface_area_mm2=area),
            np.full(7,1.2),'organ_area_fraction',2,3)
        np.testing.assert_allclose(f,[1.1,1.1,1.1,1.05,1.05,1.2,1.])
        # Different uncertainty flags on two otherwise equal faces cannot make
        # the same continuous leaf change weight at every triangle boundary.
        organ=dict(vertex_instance_id=ids[:3],vertex_center_score=np.full(3,.5),
            vertex_organ_area_mm2=np.full(3,2000.),vertex_instance_review=np.ones(3,bool),
            vertex_surface_area_mm2=area[:3])
        result=build_arrays(labels[:3],uncertain[:3],np.array([[0,1,2],[0,1,2]]),
            np.array([2,2]),np.array([0,1],bool),organ,self.policy,self.gp)
        self.assertAlmostEqual(result['face_effective_weight'][0],result['face_effective_weight'][1])

    def test_local_weight_used_once_and_required_when_enabled(self):
        self.policy['spatial_weighting'] = dict(require_local_weight=True,class_ids=[2,3])
        with self.assertRaises(ValueError):
            instantaneous_cost(ContactSample(0,2,0,0,1), self.policy)
        s = ContactSample(0,2,0,0,1,True,3.5,1)
        self.assertEqual(instantaneous_cost(s,self.policy),3.5)
        self.assertTrue(math.isinf(instantaneous_cost(ContactSample(0,6,0,0,1,False,0,1), self.policy)))

    def test_distance_curve_is_bounded_and_convex(self):
        values = [distance_factor(c, 0, 5, 2) for c in [5,4,3,2,1,0,-1]]
        np.testing.assert_allclose(values, [0., .04, .16, .36, .64, 1., 1.])
        self.assertEqual(distance_factor(0, 99, 5, 2), 1.)

    def test_long_continuous_contact_and_distinct_organs(self):
        short = [ContactSample(i/10,2,0,0,1,False,2.,1) for i in range(3)]
        long = [ContactSample(i/10,2,0,0,1,False,2.,1) for i in range(21)]
        a,b = score_trace(short,self.policy),score_trace(long,self.policy)
        self.assertGreater(b['total_cost']/b['exposure_integral'], a['total_cost']/a['exposure_integral'])
        parts = [ContactSample(0,2,0,0,1,False,2.,1),ContactSample(.1,2,0,0,1,False,2.,2)]
        self.assertEqual(len(score_trace(parts,self.policy)['episodes']),2)


if __name__ == '__main__':
    unittest.main()
