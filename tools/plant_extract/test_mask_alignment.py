import unittest
import numpy as np
from auto_extract_plant import View, project
from clean_turntable_mesh import align_mask


def camera(width, height, f, cx, cy):
    return View('img_000.jpg', width, height, f, f, cx, cy, np.eye(3), np.zeros(3))


class MaskAlignmentTests(unittest.TestCase):
    def test_center_crop_matches_projection(self):
        src = camera(120, 160, 100., 60., 80.)
        dst = camera(120, 120, 100., 60., 60.)
        mask = np.zeros((160,120), np.uint8)
        mask[48:93, 40:80] = 255
        out, H = align_mask(mask, src, dst)
        np.testing.assert_array_equal(out, mask[20:140])
        points = np.array([[0.,0.,1.], [.1,-.1,1.],[-.1,.2,1.]])
        src.mask = mask > 0; dst.mask = out > 0
        u,y,_ = project(points, src)
        uv = (H @ np.c_[u,y,np.ones(3)].T).T
        a,b,_ = project(points, dst)
        np.testing.assert_allclose(uv[:,:2]/uv[:,2:], np.c_[a,b], atol=1e-10)

    def test_focal_remap_and_crop(self):
        src = camera(120,160,100.,60.,80.)
        dst = camera(120,120,110.,60.,60.)
        mask = np.zeros((160,120), np.uint8); mask[80,70]=255
        out,H = align_mask(mask,src,dst)
        self.assertEqual(out[60,71],255)
        np.testing.assert_allclose(H@np.array([70.,80.,1.]),[71.,60.,1.])

    def test_different_pose_refused(self):
        src = camera(120,160,100.,60.,80.)
        dst = camera(120,120,100.,60.,60.); dst.t=np.array([1.,0.,0.])
        with self.assertRaisesRegex(ValueError,'poses differ'):
            align_mask(np.zeros((160,120),np.uint8),src,dst)


if __name__ == '__main__': unittest.main()
