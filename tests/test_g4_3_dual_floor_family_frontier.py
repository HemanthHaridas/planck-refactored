#!/usr/bin/env python3
"""Composition checks for the test-only G4.3.9 dual-floor frontier runner."""
import importlib.util
import json
import unittest
from pathlib import Path

import g4_3_dual_energy_floor as energy
import g4_3_second_row_floor as response


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('dual_frontier', ROOT / 'g4_3_dual_floor_family_frontier.py')
frontier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier)
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
ENERGY = energy.load(ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_dual_energy_floor.json', REQUEST)
RESPONSE = response.load(ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_second_row_band1_floor.json', REQUEST)


class DualFloorFamilyFrontierTests(unittest.TestCase):
    def test_hydrogen_filter_needs_only_energy_floor(self):
        rejected = {'boundaries': [.4, 1., 2.5, 3.5], 'lebedev_points': [26, 110, 1454, 770, 434]}
        accepted = {'boundaries': [.4, 1., 2.5, 3.5], 'lebedev_points': [26, 110, 1454, 770, 590]}
        self.assertEqual(list(frontier.constrained_rules(iter([rejected, accepted]), 'hydrogen-helium', REQUEST, ENERGY)),
                         [accepted])

    def test_second_row_composes_both_constraints(self):
        missing_response = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [86, 590, 1730, 590, 170]}
        accepted = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [86, 770, 1730, 590, 170]}
        self.assertEqual(list(frontier.constrained_rules(iter([missing_response, accepted]), 'second-row', REQUEST,
                                                        ENERGY, RESPONSE)), [accepted])


if __name__ == '__main__':
    unittest.main()
