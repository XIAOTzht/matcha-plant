"""Regression checks for wrist-first refinement of an expensive soft crossing."""
import unittest

import numpy as np

from attitude_escape_search import changed_edge_cost, pose_candidates, ranked_segments


class WristRefinementTests(unittest.TestCase):
    def test_wrist_can_avoid_costly_crossing_without_endpoint_change(self):
        class Checker:
            def edge_cost(self, a, b, max_step_rad=.04):
                q = np.linspace(a, b, 101)
                risk = ((abs(q[:, 0]) < .15) & (abs(q[:, 3]) < .25)).astype(float)*30
                length = float(np.linalg.norm(b-a))
                cost = length*(1+float(np.trapezoid(risk, dx=.01)))
                return True, cost, 0., length
        checker = Checker()
        a = np.array([-1., 0, 0, 0, 0])
        b = np.array([1., 0, 0, 0, 0])
        candidates = list(pose_candidates(a, b, np.full(5, -3.), np.full(5, 3.), .5))
        scored = [(changed_edge_cost(checker, a, q, b), q) for _, q in candidates]
        best, q = min(scored, key=lambda row:row[0])
        self.assertLess(best, checker.edge_cost(a, b)[1])
        self.assertGreater(abs(q[3]), .25)
        np.testing.assert_array_equal(a, [-1, 0, 0, 0, 0])
        np.testing.assert_array_equal(b, [1, 0, 0, 0, 0])

    def test_both_new_edges_are_checked_even_when_inserted_pose_is_safe(self):
        class Checker:
            def __init__(self):
                self.calls = []
            def edge_cost(self, a, b, max_step_rad=.04):
                self.calls.append((a.copy(), b.copy()))
                valid = len(self.calls) != 2
                return valid, 1., 0., 1.
        checker = Checker()
        score = changed_edge_cost(checker, np.zeros(5), np.ones(5), np.full(5, 2.))
        self.assertFalse(np.isfinite(score))
        self.assertEqual(len(checker.calls), 2)

    def test_whole_arm_and_independent_wrist_proposals_respect_limits(self):
        lo, hi = np.full(5, -1.), np.full(5, 1.)
        candidates = np.array([q for _,q in pose_candidates(np.zeros(5), np.zeros(5), lo, hi, .5)])
        self.assertTrue(np.all(candidates >= lo))
        self.assertTrue(np.all(candidates <= hi))
        for joint in range(5):
            self.assertGreater(np.ptp(candidates[:,joint]), .1)

    def test_final_segment_is_not_silently_excluded_from_risk_search(self):
        class Checker:
            def check(self, q):
                return True, 30. if q[0] > 1.5 else 0.
        path = np.zeros((3,5))
        path[:,0] = [0, 1, 2]
        ranking = ranked_segments(Checker(), path)
        self.assertEqual(ranking[0]["segment"], 1)


if __name__ == "__main__":
    unittest.main()
