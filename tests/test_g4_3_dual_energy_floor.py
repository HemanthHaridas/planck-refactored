#!/usr/bin/env python3
"""Offline invariants for the G4.3.9 dual energy-floor diagnostic contract."""
import json
import unittest
from pathlib import Path

import g4_3_dual_energy_floor as energy
import g4_3_second_row_floor as response


ROOT = Path(__file__).resolve().parent
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
ENERGY = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_dual_energy_floor.json'
RESPONSE = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_second_row_band1_floor.json'


class DualEnergyFloorTests(unittest.TestCase):
    def test_rejects_current_revised_hydrogen_outer_band(self):
        floor = energy.load(ENERGY, REQUEST)
        rule = {'boundaries': [.4, 1., 2.5, 3.5], 'lebedev_points': [26, 110, 1454, 770, 434]}
        self.assertFalse(energy.accepts(rule, 'hydrogen-helium', REQUEST, floor))

    def test_accepts_hydrogen_diagnostic_minimum(self):
        floor = energy.load(ENERGY, REQUEST)
        rule = {'boundaries': [.4, 1., 2.5, 3.5], 'lebedev_points': [26, 110, 1454, 770, 590]}
        self.assertTrue(energy.accepts(rule, 'hydrogen-helium', REQUEST, floor))

    def test_composes_second_row_response_and_energy_floors(self):
        energy_floor = energy.load(ENERGY, REQUEST)
        response_floor = response.load(RESPONSE, REQUEST)
        missing_response = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [86, 590, 1730, 590, 170]}
        diagnostic_minimum = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [86, 770, 1730, 590, 170]}
        self.assertTrue(energy.accepts(missing_response, 'second-row', REQUEST, energy_floor))
        self.assertFalse(response.accepts(missing_response, REQUEST, response_floor))
        self.assertTrue(energy.accepts(diagnostic_minimum, 'second-row', REQUEST, energy_floor))
        self.assertTrue(response.accepts(diagnostic_minimum, REQUEST, response_floor))


if __name__ == '__main__':
    unittest.main()
