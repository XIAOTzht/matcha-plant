import unittest
import numpy as np
from spray_target import SprayRayQuery


class SprayRayTests(unittest.TestCase):
    def layers(self,labels):
        xyz=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.],
                      [-1.,-1.,.004],[1.,-1.,.004],[0.,1.,.004]])
        faces=np.array([[0,1,2],[3,4,5]],dtype=np.int64)
        return SprayRayQuery(xyz,faces,np.array(labels))

    def test_first_surface_blocks_soil(self):
        query=self.layers([5,2]);hit=query.first_hit([0,0,.01],[0,0,-1])
        self.assertEqual(hit['first_hit_class'],2)
        self.assertAlmostEqual(hit['range_mm'],6.)
        query=self.layers([5,4]);self.assertEqual(query.first_hit([0,0,.01],[0,0,-1])['first_hit_class'],4)

    def test_two_sided_and_nearest_nonzero_hit(self):
        query=self.layers([5,5]);hit=query.first_hit([0,0,-.01],[0,0,2])
        self.assertEqual(hit['source_face_index'],0)
        self.assertAlmostEqual(hit['range_mm'],10.)

    def test_parallel_outside_and_behind_are_misses(self):
        query=self.layers([5,5])
        for origin,direction in (([0,0,.01],[1,0,0]),([2,0,.01],[0,0,-1]),([0,0,.01],[0,0,1])):
            self.assertIsNone(query.first_hit(origin,direction)['first_hit_class'])


if __name__=='__main__':unittest.main()
