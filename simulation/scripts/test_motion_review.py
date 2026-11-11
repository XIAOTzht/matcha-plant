import unittest
from motion_review import review_motion


class MotionReviewTests(unittest.TestCase):
    def report(self, source, spans):
        return dict(selected_score=dict(source=source),
                    selected_motion=dict(joint_span_deg=spans))

    def test_approved_v2_style_passes(self):
        r = review_motion(self.report('whole_arm_smooth_excursion', [41.8, 51.5, 33.2, 38.3, 43.2]))
        self.assertTrue(r['accepted'])

    def test_new_server_fallback_is_not_accepted_style(self):
        r = review_motion(self.report('unchanged_baseline', [33.5, 11.5, 46.3, 35.2, 6.4]))
        self.assertFalse(r['accepted'])
        self.assertFalse(r['checks']['wrist_roll_span'])
        self.assertFalse(r['checks']['coordination_not_fallback'])

    def test_missing_or_nonfinite_metrics_never_pass(self):
        self.assertFalse(review_motion({})['accepted'])
        self.assertFalse(review_motion(self.report('candidate', [30, 30, 35, 40, float('nan')]))['accepted'])


if __name__ == '__main__':
    unittest.main()
