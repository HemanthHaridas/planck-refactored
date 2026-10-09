#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
SPEC = importlib.util.spec_from_file_location('frontier', ROOT / 'g4_3_rule_frontier.py')
frontier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier)


def table(fixture):
    return dict(contract='g4-3-shell-residuals-v1', fixture=fixture,
                records=[dict(atom=0, family='hydrogen-helium', u=.2, lebedev_points=2030)])


class G43RuleFrontierTests(unittest.TestCase):
    def test_frontier_has_anchor_and_ineligible_pyscf_specimen(self):
        value = frontier.build_frontier([table(path) for path in REQUEST['fixtures']], REQUEST)
        self.assertEqual(value['contract'], frontier.CONTRACT)
        self.assertEqual(value['anchors'][0]['name'], 'all-2030')
        self.assertFalse(value['anchors'][1]['eligible'])
        self.assertEqual(value['cost_frontier'][0]['name'], 'canonical-all-minimum')


if __name__ == '__main__':
    unittest.main()
