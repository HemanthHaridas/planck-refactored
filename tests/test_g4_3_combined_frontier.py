#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
SPEC = importlib.util.spec_from_file_location('combined_frontier', ROOT / 'g4_3_combined_frontier.py')
combined_frontier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(combined_frontier)


def table():
    records = []
    for ordinal, family in enumerate(REQUEST['families']):
        records.append(dict(atom=ordinal, atomic_number=family['atomic_numbers'][0], radial_index=0,
                            family=family['name'], u=.2, lebedev_points=2030))
    return dict(contract='g4-3-shell-residuals-v1', fixture=REQUEST['fixtures'][0], records=records)


def all_tables():
    return [dict(table(), fixture=fixture) for fixture in REQUEST['fixtures']]


class G43CombinedFrontierTests(unittest.TestCase):
    def test_combination_reconciles_direct_and_separable_costs(self):
        rules = {
            'hydrogen-helium': {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [26, 50, 86, 50, 26]},
            'second-row': {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [50, 86, 110, 86, 50]},
            'third-row': {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [86, 110, 146, 110, 86]},
        }
        fronts = {name: [dict(max_points_per_atom=rule['lebedev_points'][1],
                              total_ledger_points=rule['lebedev_points'][1] * 14,
                              rule=rule)] for name, rule in rules.items()}
        result = combined_frontier.combine(fronts, all_tables(), REQUEST)
        self.assertEqual(result['contract'], combined_frontier.CONTRACT)
        self.assertEqual(len(result['frontier']), 1)
        cost = result['frontier'][0]['cost']
        self.assertEqual(cost['max_points_per_atom'], 110)
        self.assertEqual(cost['total_ledger_points'], (50 + 86 + 110) * 14)

    def test_rejects_stale_family_cost(self):
        rule = {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [26, 50, 86, 50, 26]}
        fronts = {name: [dict(max_points_per_atom=1, total_ledger_points=1, rule=rule)]
                  for name in [item['name'] for item in REQUEST['families']]}
        with self.assertRaisesRegex(ValueError, 'does not match'):
            combined_frontier.combine(fronts, all_tables(), REQUEST)


if __name__ == '__main__':
    unittest.main()
