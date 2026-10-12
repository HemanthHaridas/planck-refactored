#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
FLOOR = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_second_row_band1_floor.json'
SPEC = importlib.util.spec_from_file_location('floor', ROOT / 'g4_3_second_row_floor.py')
floor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(floor)


class G43SecondRowFloorTests(unittest.TestCase):
    def test_rejects_the_failed_434_point_band(self):
        value = floor.load(FLOOR, REQUEST)
        rule = {'boundaries': [.25, 1., 2.5, 4.5], 'lebedev_points': [50, 434, 1454, 590, 170]}
        self.assertFalse(floor.accepts(rule, REQUEST, value))

    def test_accepts_the_minimal_diagnostic_repair(self):
        value = floor.load(FLOOR, REQUEST)
        rule = {'boundaries': [.25, 1., 2.5, 4.5], 'lebedev_points': [50, 770, 1454, 590, 170]}
        self.assertTrue(floor.accepts(rule, REQUEST, value))


if __name__ == '__main__':
    unittest.main()
