"""Checks that crop geometry is inferred from input rather than fixed coordinates."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import auto_crop_mesh as crop


class CropTests(unittest.TestCase):
    def test_support_plane_at_different_pose_and_scale(self):
        rng=np.random.default_rng(10)
        # An arbitrarily transformed flat table and a colored canopy above it.
        table=np.column_stack((rng.uniform(-1.5,1.5,12000),rng.normal(0,.001,12000),
                               rng.uniform(-1.5,1.5,12000)))
        seeds=rng.normal(size=(2000,3))*[.2,.18,.12]+[0,.8,0]
        R=Rotation.from_euler("xyz",[35,24,70],degrees=True).as_matrix()
        for scale,offset in [(1,np.zeros(3)),(7,np.array([10,-18,40]))]:
            with self.subTest(scale=scale):
                points=np.concatenate((table,seeds))@R.T*scale+offset
                canopy=seeds@R.T*scale+offset
                span=np.linalg.norm(np.diff(np.percentile(canopy,[2,98],axis=0),axis=0)[0])
                plane,_=crop.fit_support_plane(points,canopy,span,np.random.default_rng(1),200)
                self.assertGreater(plane['normal']@R[:,1],.999)
                origin,frame,lo,hi=crop.estimate_frame(canopy,plane,.14,.012)
                q=(canopy-origin)@frame
                self.assertGreater(((q>=lo)&(q<=hi)).all(axis=1).mean(),.97)
                table_q=(table@R.T*scale+offset-origin)@frame
                self.assertLess((table_q[:,1]>=lo[1]).mean(),.01)

    def test_color_detection_includes_green_and_pink(self):
        rgb=np.array([[20,120,30],[200,70,160],[180,180,180],[50,50,50]])
        np.testing.assert_array_equal(crop.color_seeds(rgb,"auto"),[True,True,False,False])
        np.testing.assert_array_equal(crop.color_seeds(rgb,"green"),[True,False,False,False])
        np.testing.assert_array_equal(crop.color_seeds(rgb,"flowers"),[False,True,False,False])

    def test_gray_input_refused(self):
        rng=np.random.default_rng(2)
        xyz=rng.normal(size=(10000,3))
        with self.assertRaisesRegex(ValueError,"Too few botanical"):
            crop.detect_seed_cluster(xyz,np.full((10000,3),150),"auto",.006,1)


if __name__=="__main__":
    unittest.main()
