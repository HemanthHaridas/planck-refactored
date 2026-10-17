#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
SPEC = importlib.util.spec_from_file_location('cost', ROOT / 'g4_3_rule_cost.py')
cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cost)


def table(fixture):
    records = []
    for atom, family, u in ((0, 'hydrogen-helium', .09), (1, 'second-row', .3), (2, 'third-row', 2.)):
        records.append(dict(atom=atom, family=family, u=u, lebedev_points=2030))
    return dict(contract='g4-3-shell-residuals-v1', fixture=fixture, records=records)


class G43RuleCostTests(unittest.TestCase):
    def test_all_2030_cost_uses_each_reference_shell(self):
        rules = {family['name']: {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [2030] * 5}
                 for family in REQUEST['families']}
        tables = [table(fixture) for fixture in REQUEST['fixtures']]
        value = cost.rule_cost(rules, tables, REQUEST)
        self.assertEqual(value['max_points_per_atom'], 2030)
        self.assertEqual(value['total_ledger_points'], 14 * 3 * 2030)

    def test_rejects_incomplete_fixture_coverage(self):
        with self.assertRaisesRegex(ValueError, 'cover every'):
            cost.load_reports([], REQUEST)

    def test_frontier_removes_dominated_rows(self):
        rows = [dict(name='low', cost=dict(max_points_per_atom=1, total_ledger_points=2)),
                dict(name='high', cost=dict(max_points_per_atom=2, total_ledger_points=3)),
                dict(name='tradeoff', cost=dict(max_points_per_atom=0, total_ledger_points=4))]
        self.assertEqual([row['name'] for row in cost.pareto_frontier(rows)], ['low', 'tradeoff'])


if __name__ == '__main__':
    unittest.main()
