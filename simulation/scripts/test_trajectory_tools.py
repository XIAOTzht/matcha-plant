import json
from pathlib import Path
import unittest
import numpy as np

from trajectory_tools import (distance_intensity, score_exposure, timed_quintic,
                              timed_continuous_quintic, timed_continuous_septic,
                              _septic_hermite_coefficients, _evaluate_septic,
                              _polynomial_motion_peaks)
from trajectory_tools import time_parameterize
from plan_watering_path import rrt_connect, rrt_star, shortcut, path_cost


ROOT = Path(__file__).resolve().parents[1]


class TimingTests(unittest.TestCase):
    def test_common_dispatcher_uses_current_continuous_profile(self):
        cfg = json.loads((ROOT / 'configs/weighted_trajectory_validation_v5.json').read_text())
        points = np.array([[0]*5, [.2]*5, [.5]*5])
        result = time_parameterize(points, cfg, .005)
        expected = timed_continuous_septic(points, cfg, .005)
        np.testing.assert_array_equal(result['q_ref'], expected['q_ref'])
        self.assertGreater(float(np.max(abs(result['qd_ref']))), 0)
        with self.assertRaises(ValueError):
            time_parameterize(points, dict(cfg, timing_profile='unknown'), .005)

    def test_endpoints_continuity_and_limits(self):
        cfg = json.loads((ROOT / "configs/trajectory_validation.json").read_text())
        points = np.array([[0, 0, 0, 0, 0], [1, -.2, .4, -.1, .8], [.2, -.5, .3, .2, .4]])
        result = timed_quintic(points, cfg, .005)
        np.testing.assert_allclose(result["q_ref"][0], points[0])
        np.testing.assert_allclose(result["q_ref"][-1], points[-1])
        np.testing.assert_allclose(result["qd_ref"][[0, -1]], 0)
        np.testing.assert_allclose(result["qdd_ref"][[0, -1]], 0)
        np.testing.assert_allclose(np.diff(result["time_s"]), .005, atol=1e-12)
        for segment in result["segments"]:
            self.assertTrue(np.all(np.asarray(segment["analytic_peak_velocity_rad_s"]) <= cfg["velocity_limit_rad_s"]))
            self.assertTrue(np.all(np.asarray(segment["analytic_peak_acceleration_rad_s2"]) <= cfg["acceleration_limit_rad_s2"]))
            t = segment["start_s"] + segment["duration_s"]
            idx = np.argmin(abs(result["time_s"] - t))
            np.testing.assert_allclose(result["qd_ref"][idx], 0, atol=1e-12)
            np.testing.assert_allclose(result["qdd_ref"][idx], 0, atol=1e-12)

    def test_continuous_quintic_shared_knot_velocity(self):
        cfg = json.loads((ROOT / "configs/trajectory_validation.json").read_text())
        cfg.update(timing_profile="continuous_quintic", waypoint_velocity_tension=0.4)
        points = np.array([[0, 0, 0, 0, 0], [1, -.2, .4, -.1, .8], [.2, -.5, .3, .2, .4]])
        result = timed_continuous_quintic(points, cfg, .005)
        np.testing.assert_allclose(result["q_ref"][0], points[0])
        np.testing.assert_allclose(result["q_ref"][ -1], points[-1])
        np.testing.assert_allclose(result["qd_ref"][0], 0, atol=1e-12)
        np.testing.assert_allclose(result["qd_ref"][ -1], 0, atol=1e-12)
        np.testing.assert_allclose(result["qdd_ref"][0], 0, atol=1e-12)
        np.testing.assert_allclose(result["qdd_ref"][ -1], 0, atol=1e-12)
        self.assertTrue(np.max(np.abs(result["qd_ref"])) <= max(cfg["velocity_limit_rad_s"]) + 1e-8)
        self.assertTrue(np.max(np.abs(result["qdd_ref"])) <= max(cfg["acceleration_limit_rad_s2"]) + 1e-8)
        self.assertTrue(all(np.any(np.abs(s["end_velocity_rad_s"]) > 1e-8)
                            for s in result["segments"][:-1]))

    def test_continuous_septic_has_zero_knot_acceleration(self):
        cfg = json.loads((ROOT / "configs/trajectory_validation.json").read_text())
        cfg.update(timing_profile="continuous_septic", waypoint_velocity_tension=0.4)
        points = np.array([[0, 0, 0, 0, 0], [1, -.2, .4, -.1, .8], [.2, -.5, .3, .2, .4]])
        result = timed_continuous_septic(points, cfg, .005)
        np.testing.assert_allclose(result["q_ref"][0], points[0])
        np.testing.assert_allclose(result["q_ref"][-1], points[-1])
        np.testing.assert_allclose(result["qd_ref"][[0, -1]], 0, atol=1e-12)
        np.testing.assert_allclose(result["qdd_ref"][[0, -1]], 0, atol=1e-12)
        self.assertTrue(all(np.allclose(s["end_acceleration_rad_s2"], 0, atol=1e-12)
                            for s in result["segments"]))

    def test_septic_matches_all_endpoint_derivatives(self):
        endpoints = np.array([[.1, -.2, .3, -.4, .5], [.8, .2, -.1, .6, -.2]])
        velocity = np.array([[.1, .2, -.3, 0, .1], [-.2, .1, .1, 0, .3]])
        acceleration = velocity * .37
        jerk = velocity * -.51
        c = _septic_hermite_coefficients(*endpoints, *velocity, *acceleration, *jerk, 2.3)
        for actual, expected in zip(_evaluate_septic(c, 2.3, np.array([0., 1.])),
                                    (endpoints, velocity, acceleration, jerk)):
            np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_reversal_limiter_prevents_overshoot_without_whole_arm_stop(self):
        cfg = json.loads((ROOT / "configs/weighted_trajectory_validation_v5.json").read_text())
        points = np.array([[0, 0, 0, 0, 0], [.5, .3, -.4, .2, .3],
                           [-.2, .5, -.6, -.1, .4]])
        result = timed_continuous_septic(points, cfg, .005)
        knot_velocity = np.array(result["segments"][0]["end_velocity_rad_s"])
        np.testing.assert_allclose(knot_velocity[[0, 3]], 0)
        self.assertTrue(np.all(abs(knot_velocity[[1, 2, 4]]) > 0))
        for s in result["segments"]:
            i = s["index"]
            mask = ((result["time_s"] >= s["start_s"] - 1e-10)
                    & (result["time_s"] <= s["start_s"] + s["duration_s"] + 1e-10))
            self.assertTrue(np.all(result["q_ref"][mask] >= np.min(points[i:i+2], axis=0) - 1e-10))
            self.assertTrue(np.all(result["q_ref"][mask] <= np.max(points[i:i+2], axis=0) + 1e-10))
            self.assertTrue(np.all(np.array(s["analytic_peak_velocity_rad_s"]) <= cfg["velocity_limit_rad_s"]))
            self.assertTrue(np.all(np.array(s["analytic_peak_acceleration_rad_s2"]) <= cfg["acceleration_limit_rad_s2"]))
        np.testing.assert_allclose(np.diff(result["time_s"]), .005, atol=1e-12)

    def test_peak_roots_bound_dense_derivative_samples(self):
        a = np.zeros(5); b = np.array([.8, -.4, .6, 0, -.1])
        va = np.array([.1, 0, -.2, 0, .1]); vb = -va
        c = _septic_hermite_coefficients(a, b, va, vb, a, a, a, a, 1.7)
        _, v, acc, _ = _evaluate_septic(c, 1.7, np.linspace(0, 1, 20001))
        peak_v, peak_a = _polynomial_motion_peaks(c, 1.7)
        self.assertTrue(np.all(np.max(abs(v), axis=0) <= peak_v + 1e-10))
        self.assertTrue(np.all(np.max(abs(acc), axis=0) <= peak_a + 1e-10))


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT.parent / "tools/contact_weights/contact_policy.json").read_text())

    def test_distance_and_local_weight_are_not_multiplied_twice(self):
        values = distance_intensity(np.array([.006, .005, .004, .003, .002, .001, 0, -.001]))
        np.testing.assert_allclose(values, [0, 0, .04, .16, .36, .64, 1, 1], atol=1e-12)
        t = np.linspace(0, 1, 101)
        r = score_exposure(t, [1], np.zeros((101, 1, 2)), np.full((101, 1, 2), 24.),
                           np.full((101, 1, 2), -1), self.policy)
        # Two geometry covers for one organ do not double its exposure.
        per_second = float(distance_intensity(np.array([0.0]))[0])
        self.assertAlmostEqual(r["exposure_integral"], 24 * per_second)
        self.assertAlmostEqual(r["persistence_cost"], 24 * per_second * 1.5)

    def test_longer_contact_costs_more_and_gap_splits_episode(self):
        t = np.linspace(0, 2, 201)
        long = score_exposure(t, [2], np.zeros((201, 1, 1)), np.ones((201, 1, 1)),
                              np.zeros((201, 1, 1), dtype=int), self.policy)
        clear = np.zeros((201, 1, 1)); clear[80:121] = .006
        gap = score_exposure(t, [2], clear, np.ones_like(clear), np.zeros_like(clear, dtype=int), self.policy)
        self.assertEqual(len(long["episodes"]), 1)
        self.assertEqual(len(gap["episodes"]), 2)
        self.assertGreater(long["persistence_cost"], gap["persistence_cost"])

    def test_short_gaps_preserve_episode_identity(self):
        t = np.linspace(0, 1, 101)
        clear = np.zeros((101, 1, 1)); clear[45:50] = .006
        r = score_exposure(t, [2], clear, np.ones_like(clear), np.zeros_like(clear, dtype=int), self.policy)
        self.assertEqual(len(r["episodes"]), 1)
        self.assertAlmostEqual(r["episodes"][0]["duration_s"], 1.)

    def test_distinct_organs_accumulate_but_duplicate_covers_do_not(self):
        t = np.linspace(0, 1, 101)
        instance = np.tile(np.array([[[1, 1, 2]]]), (101, 1, 1))
        r = score_exposure(t, [2], np.zeros((101, 1, 3)), np.ones((101, 1, 3)), instance, self.policy)
        per_second = float(distance_intensity(np.array([0.0]))[0])
        np.testing.assert_allclose(r["instantaneous_exposure"], 2 * per_second)
        self.assertAlmostEqual(r["exposure_integral"], 2 * per_second)
        self.assertAlmostEqual(r["persistence_cost"], 2 * per_second * 1.5)


class PlannerEndpointTests(unittest.TestCase):
    def test_swapped_rrt_tree_returns_start_to_goal(self):
        class Checker:
            q_lo = np.full(5, -1.); q_hi = np.full(5, 1.)
            def __init__(self): self.calls = 0
            def edge(self, a, b, step):
                self.calls += 1
                # Force one failed first iteration, so the trees are swapped.
                return self.calls > 1, 0.
        a = np.full(5, -.2); b = np.full(5, .2)
        path, _ = rrt_connect(Checker(), a, b, iterations=50)
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[0], a)
        np.testing.assert_allclose(path[-1], b)





class WeightedPlannerTests(unittest.TestCase):
    class Checker:
        q_lo = np.full(5, -1.); q_hi = np.full(5, 1.)
        def edge_cost(self, a, b, max_step_rad=.025, risk_weight=1.):
            q = np.linspace(a,b,max(3,int(np.ceil(np.linalg.norm(b-a)/.02))))
            # Hard forbidden lower boundary, soft expensive strip through middle.
            if np.any(q[:,1]<-.5):return False,np.inf,np.inf,float(np.linalg.norm(b-a))
            risk = np.where((abs(q[:,0])<.2)&(abs(q[:,1])<.25),30.,0.)
            length=float(np.linalg.norm(b-a));integral=float(np.trapezoid(risk,dx=1/(len(q)-1))*length)
            return True,length+risk_weight*integral,integral,length

    def test_rrtstar_improves_high_weight_direct_route(self):
        c=self.Checker();a=np.array([-.7,0,0,0,0.]);b=np.array([.7,0,0,0,0.])
        path,stats,_=rrt_star(c,a,b,iterations=180,max_seconds=30,seed=12,initial_paths=[np.vstack([a,b])])
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[0],a);np.testing.assert_allclose(path[-1],b)
        self.assertLess(path_cost(c,path),path_cost(c,np.vstack([a,b])))
        self.assertTrue(all(c.edge_cost(x,y)[0] for x,y in zip(path[:-1],path[1:])))

    def test_shortcut_preserves_weighted_detour(self):
        c=self.Checker();p=np.array([[-.7,0,0,0,0],[-.5,.5,0,0,0],[.5,.5,0,0,0],[.7,0,0,0,0]])
        result=shortcut(c,p,attempts=100,seed=8)
        self.assertLessEqual(path_cost(c,result),path_cost(c,p)+1e-8)
        self.assertGreater(len(result),2)

if __name__ == "__main__":
    unittest.main()
