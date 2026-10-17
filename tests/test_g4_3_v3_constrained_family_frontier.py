#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path

import g4_3_v3_constraints as constraints_module


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('runner', ROOT / 'g4_3_v3_constrained_family_frontier.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
CONSTRAINTS = constraints_module.load(ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_v3_constraints.json', REQUEST)


class G43V3ConstrainedFamilyFrontierTests(unittest.TestCase):
    def test_runner_uses_the_v3_constraint_filter(self):
        h = {'boundaries': [.4, 1., 3.5, 4.5], 'lebedev_points': [26, 146, 974, 434, 302]}
        second = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [110, 770, 1730, 590, 170]}
        self.assertTrue(next(runner.filter_rules(iter([h]), 'hydrogen-helium', REQUEST, CONSTRAINTS)) is h)
        self.assertTrue(next(runner.filter_rules(iter([second]), 'second-row', REQUEST, CONSTRAINTS)) is second)


if __name__ == '__main__':
    unittest.main()
