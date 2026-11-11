"""Flags must be released by independent evidence, never by scores alone."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from reassess_uncertainty import reassess


def sample(n=6):
    label = np.full(n, 2, np.uint8)
    data = dict(vertex_labels=label, uncertain=np.ones(n, bool),
                class_scores=np.tile([.02, .9, .03, .03, .02], (n, 1)),
                view_count=np.full(n, 4, np.uint16),
                leaf_mask_positive_views=np.zeros(n, np.uint16),
                leaf_mask_observed_views=np.zeros(n, np.uint16),
                leaf_rim_refined=np.zeros(n, bool), bud_protected=np.zeros(n, bool))
    support = np.zeros((n, 6), np.uint16)
    support[:, 2] = 4
    total = np.zeros((n, 6), np.float32)
    total[:, 2] = 1.6
    instances = dict(view_count=data['view_count'].copy(), support_count=support, total=total)
    return data, instances


class UncertaintyTests(unittest.TestCase):
    def test_consensus_releases_flag_but_sparse_or_conflicting_evidence_does_not(self):
        data, instances = sample()
        data['view_count'][1] = 1
        instances['view_count'][1] = 1
        instances['support_count'][1, 2] = 1
        # A competing class has full support: signed subtraction is essential.
        instances['support_count'][2, 2] = 3
        instances['support_count'][2, 3] = 4
        data['class_scores'][3] = [.05, .48, .45, .01, .01]
        data['class_scores'][4] = [.01, .3, .67, .01, .01]
        data['uncertain'][5] = False
        original_label = data['vertex_labels'].copy()
        result = reassess(data, instances)
        np.testing.assert_array_equal(result['uncertainty_released'], [1,0,0,0,0,0])
        np.testing.assert_array_equal(result['uncertain'], [0,1,1,1,1,0])
        np.testing.assert_array_equal(data['vertex_labels'], original_label)
        self.assertTrue(data['uncertain'][0])

    def test_independent_leaf_masks_can_resolve_old_detector_disagreement(self):
        data, instances = sample(3)
        instances['support_count'][:] = 0
        instances['total'][:] = 0
        data['leaf_mask_positive_views'][:] = [3, 2, 3]
        data['leaf_mask_observed_views'][:] = 4
        data['bud_protected'][2] = True
        result = reassess(data, instances)
        np.testing.assert_array_equal(result['uncertainty_released'], [1,0,0])

    def test_geometry_only_rim_stays_flagged_until_leaf_masks_confirm_it(self):
        data, instances = sample(2)
        data['leaf_rim_refined'][:] = True
        data['leaf_mask_positive_views'][1] = 3
        data['leaf_mask_observed_views'][1] = 4
        result = reassess(data, instances)
        np.testing.assert_array_equal(result['uncertainty_released'], [0,1])
        self.assertTrue(result['uncertainty_reason'][0] & 16)

    def test_rejects_mismatched_camera_counts_and_impossible_votes(self):
        data, instances = sample(1)
        instances['view_count'][0] = 3
        with self.assertRaises(ValueError):
            reassess(data, instances)
        instances['view_count'][0] = 4
        instances['support_count'][0, 2] = 5
        with self.assertRaises(ValueError):
            reassess(data, instances)


if __name__ == '__main__':
    unittest.main()
