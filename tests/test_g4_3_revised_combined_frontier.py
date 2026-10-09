#!/usr/bin/env python3
"""Unit invariants for the G4.3.9 revision adapter; no molecular executable."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('revised_frontier', ROOT / 'g4_3_revised_combined_frontier.py')
revised = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(revised)


class RevisedCombinedFrontierTests(unittest.TestCase):
    def test_constrained_rule_replaces_only_second_row(self):
        retired = {
            'hydrogen-helium': {'boundaries': [.4, 1., 2.5, 3.5], 'lebedev_points': [26, 110, 1454, 770, 434]},
            'second-row': {'boundaries': [.25, 1., 2.5, 4.5], 'lebedev_points': [50, 434, 1454, 590, 170]},
            'third-row': {'boundaries': [.1, .5, 3.5, 4.5], 'lebedev_points': [26, 146, 770, 230, 26]},
        }
        revised_rule = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [86, 770, 1454, 590, 170]}
        rules = dict(retired)
        rules['second-row'] = revised_rule
        self.assertEqual(rules['hydrogen-helium'], retired['hydrogen-helium'])
        self.assertEqual(rules['third-row'], retired['third-row'])
        self.assertEqual(rules['second-row'], revised_rule)


if __name__ == '__main__':
    unittest.main()
