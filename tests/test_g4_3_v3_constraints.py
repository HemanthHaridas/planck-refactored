#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('constraints', ROOT / 'g4_3_v3_constraints.py')
constraints_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(constraints_module)
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
CONSTRAINTS = constraints_module.load(
    ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_v3_constraints.json', REQUEST)


class G43V3ConstraintTests(unittest.TestCase):
    def test_hydrogen_requires_boundary_move_and_interval_floor(self):
        retired_boundary = {'boundaries': [.4, 1., 2.5, 3.5], 'lebedev_points': [26, 146, 1454, 770, 590]}
        weak_interval = {'boundaries': [.4, 1., 3.5, 4.5], 'lebedev_points': [26, 110, 974, 434, 302]}
        accepted = {'boundaries': [.4, 1., 3.5, 4.5], 'lebedev_points': [26, 146, 974, 434, 302]}
        self.assertFalse(constraints_module.accepts(retired_boundary, 'hydrogen-helium', REQUEST, CONSTRAINTS))
        self.assertFalse(constraints_module.accepts(weak_interval, 'hydrogen-helium', REQUEST, CONSTRAINTS))
        self.assertTrue(constraints_module.accepts(accepted, 'hydrogen-helium', REQUEST, CONSTRAINTS))

    def test_second_row_core_floor_tracks_physical_interval(self):
        weak = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [86, 770, 1730, 590, 170]}
        accepted = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [110, 770, 1730, 590, 170]}
        self.assertFalse(constraints_module.accepts(weak, 'second-row', REQUEST, CONSTRAINTS))
        self.assertTrue(constraints_module.accepts(accepted, 'second-row', REQUEST, CONSTRAINTS))


if __name__ == '__main__':
    unittest.main()
