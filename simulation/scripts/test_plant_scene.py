import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

from plant_scene import board_rotation, effective_policy, make_transform, transform_points, soil_candidates


class TransformTests(unittest.TestCase):
    def test_height_axis_handedness_and_length(self):
        cs = dict(board_center_mm=[100, 200, 300], board_frame_axes=[[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        center, axes = board_rotation(cs)
        self.assertAlmostEqual(np.linalg.det(axes), 1)
        # Source second column, rather than source Z, is board up.
        np.testing.assert_allclose((center + 20 * np.array(cs['board_frame_axes'])[:, 1] - center) @ axes, [0, 0, 20])
        layout = dict(yaw_deg=35, board_center_xy_m=[.28, .2], ground_clearance_mm=2)
        t, board, yaw = make_transform(center, axes, -.1, layout)
        p = transform_points(np.array([center, center + [135, 0, 0]]), t)
        np.testing.assert_allclose(p[0], board)
        self.assertAlmostEqual(np.linalg.norm(p[1] - p[0]), .135)
        np.testing.assert_allclose(transform_points(p, np.linalg.inv(t)), [center, center + [135, 0, 0]], atol=1e-10)

    def test_bad_frame_rejected(self):
        with self.assertRaises(ValueError):
            board_rotation(dict(board_center_mm=[0, 0, 0], board_frame_axes=np.ones((3, 3))))

    def test_candidates_are_above_real_soil_faces(self):
        xyz = np.array([[0., 0, .1], [.1, 0, .1], [0, .1, .1], [1, 1, 1]])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        goals, surfaces, face_ids = soil_candidates(xyz, faces, dict(face_labels=np.array([5, 4])), 5,
                            dict(count=5, separation_mm=10, height_above_soil_mm=20))
        np.testing.assert_array_equal(face_ids, [0])
        np.testing.assert_allclose(goals - surfaces, [[0, 0, .02]])
        with self.assertRaises(ValueError):
            soil_candidates(xyz, faces, dict(face_labels=np.array([4, 4])), 5, dict())

    def test_close_soil_target_keeps_original_source_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = dict(hard_collision=dict(classes=[4, 5, 6]), watering_target=dict(
                class_id=5, approach_height_range_mm=[10, 30], default_height_mm=20))
            source = json.dumps(policy)
            (root / 'policy.json').write_text(source)
            config = dict(source_root='.', policy='policy.json', watering_target_override=dict(
                approach_height_range_mm=[1, 2.25], default_height_mm=2))
            result = effective_policy(root, config)
            self.assertEqual(result['hard_collision']['classes'], [4, 5, 6])
            self.assertEqual(result['watering_target']['default_height_mm'], 2)
            self.assertEqual((root / 'policy.json').read_text(), source)
            config['watering_target_override']['class_id'] = 4
            with self.assertRaises(ValueError):
                effective_policy(root, config)


if __name__ == '__main__':
    unittest.main()
