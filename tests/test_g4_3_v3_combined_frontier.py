#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path

import g4_3_v3_constraints as constraints_module


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('runner', ROOT / 'g4_3_v3_combined_frontier.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
CONSTRAINTS = constraints_module.load(ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_v3_constraints.json', REQUEST)


class G43V3CombinedFrontierTests(unittest.TestCase):
    def test_rejects_a_frontier_with_a_different_constraint_contract(self):
        path = Path('/private/tmp/g4-3-v3-wrong-constraints.json')
        value = {'contract': runner.FAMILY_CONTRACT, 'status': 'selection-atlas-only',
                 'family': 'hydrogen-helium', 'constraints': {},
                 'frontier': [{'max_points_per_atom': 1, 'total_ledger_points': 1,
                               'rule': {'boundaries': [.4, 1., 3.5, 4.5],
                                        'lebedev_points': [26, 146, 1454, 434, 434]}}]}
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'constraint'):
            runner._family_rule(path, 'hydrogen-helium', REQUEST, CONSTRAINTS)


if __name__ == '__main__':
    unittest.main()
