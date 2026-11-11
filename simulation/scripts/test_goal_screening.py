import unittest
import numpy as np
import igl


class TriangleQueryTests(unittest.TestCase):
    def test_triangle_interior_is_not_nearest_vertex(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        tree = igl.AABB(); tree.init(vertices, faces)
        sq, tri, closest = tree.squared_distance(vertices, faces, np.array([[.5, .25, .1]]))
        np.testing.assert_array_equal(tri, [0])
        np.testing.assert_allclose(closest[0], [.5, .25, 0.], atol=1e-12)
        self.assertAlmostEqual(float(sq[0]), .01, places=12)

    def test_hard_subset_is_explicit(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [0., 0., 1.], [1., 0., 1.], [0., 1., 1.]], dtype=float)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        hard = np.array([False, True])
        self.assertEqual(int(np.flatnonzero(hard)[0]), 1)
        tree = igl.AABB(); tree.init(vertices, faces[hard])
        sq, tri, _ = tree.squared_distance(vertices, faces[hard], np.array([[.2, .2, .9]]))
        self.assertEqual(int(tri[0]), 0)
        self.assertGreater(float(sq[0]), 0.)


if __name__ == "__main__":
    unittest.main()
