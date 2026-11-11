"""New camera evidence must justify every uncertainty update."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from extra_view_review import review_flags, select_views


def sample(n=1):
    data = dict(vertex_labels=np.full(n, 2, np.uint8),
                uncertain=np.ones(n, bool),
                class_scores=np.tile([.02, .9, .03, .03, .02], (n, 1)))
    votes = np.zeros((n, 6), np.uint16)
    votes[:, 2] = 4
    total = np.zeros((n, 6), np.float32)
    total[:, 2] = 1.6
    return data, dict(view_count=np.full(n, 4, np.uint16), votes=votes, total=total)


def cameras(angles):
    return [dict(direction=[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a)), 0])
            for a in angles]


class ExtraViewTests(unittest.TestCase):
    def test_clear_consensus_releases_only_existing_flags_without_changing_input(self):
        data, extra = sample(2)
        data['uncertain'][1] = False
        before = {key: value.copy() for key, value in data.items()}
        result = review_flags(data, extra)
        np.testing.assert_array_equal(result['extra_view_released'], [True, False])
        self.assertFalse(result['uncertain'].any())
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])

    def test_sparse_ambiguous_unknown_and_high_score_only_evidence_stays_flagged(self):
        data, extra = sample(6)
        extra['view_count'][0] = 2
        extra['votes'][0, 2] = 2
        extra['view_count'][1] = 5
        extra['votes'][1] = [0, 0, 3, 2, 0, 0]
        extra['votes'][2] = [2, 0, 2, 0, 0, 0]
        extra['view_count'][3] = 0
        extra['votes'][3] = 0
        extra['total'][3] = 0
        data['class_scores'][4] = [.01, .5, .47, .01, .01]
        data['class_scores'][5] = [.01, .2, .77, .01, .01]
        self.assertTrue(review_flags(data, extra)['uncertain'].all())

    def test_competing_consensus_adds_flags_and_does_not_underflow_unsigned_counts(self):
        data, extra = sample(2)
        data['uncertain'][0] = False
        extra['votes'][:] = [0, 0, 1, 3, 0, 0]
        result = review_flags(data, extra)
        self.assertTrue(result['extra_view_conflict'].all())
        self.assertTrue(result['uncertain'].all())
        self.assertFalse(result['extra_view_released'].any())

    def test_abstentions_stay_in_denominator_and_low_mean_cannot_confirm(self):
        data, extra = sample(2)
        extra['votes'][:] = [1, 0, 3, 0, 0, 0]
        extra['total'][1, 2] = .79
        result = review_flags(data, extra)
        np.testing.assert_array_equal(result['extra_view_released'], [True, False])
        np.testing.assert_allclose(result['extra_view_agreement'], [.75, .75])

    def test_invalid_mesh_shapes_or_vote_counts_are_rejected(self):
        data, extra = sample()
        extra['votes'][0, 2] = 5
        with self.assertRaises(ValueError):
            review_flags(data, extra)
        extra['votes'] = np.zeros((1, 5), np.uint16)
        with self.assertRaises(ValueError):
            review_flags(data, extra)

    def test_invalid_labels_nan_scores_and_fractional_counts_are_rejected(self):
        for kind in ['label', 'score', 'evidence', 'count']:
            data, extra = sample()
            if kind == 'label':
                data['vertex_labels'][0] = 0
            elif kind == 'score':
                data['class_scores'][0, 1] = np.nan
            elif kind == 'evidence':
                extra['total'][0, 2] = np.nan
            else:
                extra['view_count'] = np.array([4.5])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                review_flags(data, extra)

    def test_selection_spreads_cameras_and_honors_weighted_coverage(self):
        visible = np.array([[1, 0], [1, 0], [0, 1]], bool)
        chosen = select_views(visible, np.array([2., 1.]), cameras([0, 5, 90]), budget=3)
        self.assertEqual(chosen, [0, 2])
        chosen = select_views(visible, np.array([1., 3.]), cameras([0, 5, 90]), budget=1)
        self.assertEqual(chosen, [2])

    def test_diminishing_returns_prefer_uncovered_surface_and_empty_input_stops(self):
        visible = np.array([[1, 0], [1, 0], [0, 1]], bool)
        chosen = select_views(visible, np.array([1.5, 1.]), cameras([0, 90, 180]), budget=2)
        self.assertEqual(chosen, [0, 2])
        self.assertEqual(select_views(np.zeros((3, 0), bool), np.array([]), cameras([0, 90, 180])), [])


if __name__ == '__main__':
    unittest.main()
