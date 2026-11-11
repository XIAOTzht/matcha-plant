import unittest

import mujoco
import numpy as np

from robot_volume_query import AdaptiveVolumeQuery


class VolumeDistanceTests(unittest.TestCase):
    def cylinder(self, x, z):
        model = mujoco.MjModel.from_xml_string(
            f'<mujoco><worldbody><geom name="nozzle" type="cylinder" '
            f'pos="{x} 0 {z}" size="0.004 0.03"/></worldbody></mujoco>')
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        return model, data, AdaptiveVolumeQuery(model, [0], dict(cell_tolerance_mm=.1))

    def test_near_soil_and_wall_rejects_large_sphere_but_accepts_cylinder(self):
        model, data, query = self.cylinder(.012, .032)
        distance = lambda points: np.minimum(abs(points[:, 0]), abs(points[:, 2]))
        self.assertLess(distance(data.geom_xpos)[0] - model.geom_rbound[0], 0)
        bound = query.clearance(data, 0, distance, .0005)
        self.assertGreater(bound, .0005)
        # Real cylinder clearance is min(12-4, 32-30) = 2 mm.
        self.assertLessEqual(bound, .002 + 1e-12)

    def test_nozzle_crossing_soil_is_never_accepted(self):
        _, data, query = self.cylinder(.012, .029)
        bound = query.clearance(data, 0, lambda p: abs(p[:, 2]), .0005)
        self.assertLessEqual(bound, .0005)

    def test_query_budget_cannot_convert_ambiguity_to_success(self):
        model, data, _ = self.cylinder(.012, .032)
        query = AdaptiveVolumeQuery(model, [0], dict(max_cells=2, cell_tolerance_mm=.1))
        bound = query.clearance(data, 0, lambda p: np.minimum(abs(p[:, 0]), abs(p[:, 2])), .0005)
        self.assertLessEqual(bound, .0005)


if __name__ == '__main__':
    unittest.main()
