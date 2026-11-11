"""Cart anchoring must not depend on botanical color or world orientation."""
import unittest
import numpy as np

from cart_geometry import detect_geometry, grow_above_board


def scene(plant_color):
    rng = np.random.default_rng(71)
    # Lower chassis and the same 135:110 upper board, arbitrary model units.
    x,z = np.meshgrid(np.linspace(-.5,.5,201),np.linspace(-.3,.3,121))
    chassis = np.column_stack((x.ravel(),np.zeros(x.size),z.ravel()))
    x,z = np.meshgrid(np.linspace(-.2025,.2025,101),np.linspace(-.165,.165,83))
    upper = np.column_stack((x.ravel(),np.full(x.size,.2),z.ravel()))
    angle = rng.uniform(0,2*np.pi,3000); height = rng.uniform(.2,.65,3000)
    pot = np.column_stack((.14*np.cos(angle),height,.12*np.sin(angle)))
    leaves = np.column_stack((rng.uniform(-.30,.30,4000),rng.uniform(.64,1.25,4000),rng.uniform(-.18,.18,4000)))
    points = np.vstack((chassis,upper,pot,leaves))
    colors = np.vstack((np.tile([20,55,200],(len(chassis)+len(upper),1)),np.tile([140,140,140],(3000,1)),np.tile(plant_color,(4000,1))))
    return points, colors


class CartGeometryTests(unittest.TestCase):
    def test_gray_yellow_orange_white_plants_all_preserved(self):
        for color in ([110,110,110],[230,210,20],[230,90,10],[240,240,240]):
            with self.subTest(color=color):
                points, rgb = scene(color)
                result = detect_geometry(points, rgb, np.random.default_rng(42))
                self.assertFalse(result['growth']['color_filter_used'])
                self.assertLess(result['upper_plate']['scale_disagreement_fraction'],.05)
                upper = (result['canopy']-result['origin'])@result['frame']
                self.assertGreater(upper[:,1].max(),1.20)
                self.assertGreater(len(result['canopy']),7000)

    def test_rotation_translation_uniform_scale_do_not_change_selection(self):
        points, rgb = scene([220,170,30])
        rng = np.random.default_rng(78)
        rotation, _ = np.linalg.qr(rng.normal(size=(3,3)))
        transformed = points@rotation*2.7+np.array([5,-3,8])
        result = detect_geometry(transformed, rgb, np.random.default_rng(42))
        self.assertGreater(len(result['canopy']),7000)
        np.testing.assert_allclose(result['frame'].T@result['frame'],np.eye(3),atol=1e-8)

    def test_detached_near_leaf_retained_remote_background_excluded(self):
        rng = np.random.default_rng(60)
        points, rgb = scene([120,120,120])
        fragment = rng.uniform([.31,.85,-.03],[.33,.94,.03],size=(80,3))
        wall = rng.uniform([1.7,.4,-.2],[1.75,1.2,.2],size=(3000,3))
        # Direct growth uses a known board frame (XYZ with Y up).
        selected, _ = grow_above_board(np.vstack((points,fragment,wall)),np.array([0,.2,0]),np.eye(3),
                                      np.array([.405,.33]),1.)
        self.assertLess(selected[:,0].max(),.4)
        self.assertGreater(selected[:,0].max(),.31)

    def test_envelope_contact_refuses_truncated_geometry(self):
        rng = np.random.default_rng(32)
        column = rng.uniform([-.08,.18,-.08],[.08,1.99,.08],size=(8000,3))
        with self.assertRaisesRegex(ValueError,'search envelope'):
            grow_above_board(column,np.array([0,.2,0]),np.eye(3),np.array([.405,.33]),1.,height_limit=1.8)

    def test_missing_upper_board_is_rejected(self):
        points, rgb = scene([120,120,120])
        # Remove upper board; a blue floor alone must not anchor a plant crop.
        keep = np.ones(len(points),bool); keep[201*121:201*121+101*83] = False
        with self.assertRaisesRegex(ValueError,'verified upper plate'):
            detect_geometry(points[keep],rgb[keep],np.random.default_rng(42))


if __name__ == '__main__':
    unittest.main()
