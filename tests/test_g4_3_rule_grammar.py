#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('grammar', ROOT / 'g4_3_rule_grammar.py')
grammar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grammar)
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())


class G43RuleGrammarTests(unittest.TestCase):
    def test_accepts_the_all_2030_anchor(self):
        rule = grammar.all_2030_rule(REQUEST)
        self.assertEqual(rule['lebedev_points'], [2030] * 5)

    def test_rejects_invalid_valence_shape(self):
        rule = {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [50, 86, 110, 146, 170]}
        with self.assertRaisesRegex(ValueError, 'valence-band'):
            grammar.validate_rule(rule, REQUEST)

    def test_band_assignment_is_boundary_deterministic(self):
        cuts = [.1, .25, .5, 1.]
        self.assertEqual([grammar.select_band(u, cuts) for u in (.09, .1, .25, .5, 1., 2.)],
                         [0, 1, 2, 3, 4, 4])
        self.assertEqual(grammar.select_band(.5, tuple(cuts)), 3)

    def test_enumerator_begins_in_deterministic_order(self):
        iterator = grammar.enumerate_rules(REQUEST)
        first = next(iterator)
        self.assertEqual(first['boundaries'], REQUEST['allowed_band_boundaries'][:4])
        self.assertEqual(first['lebedev_points'], [26] * 5)

    def test_literal_pyscf_specimen_is_representable(self):
        value = grammar.pyscf_nwchem_213_specimen(REQUEST)
        self.assertEqual(value['hydrogen-helium']['lebedev_points'], [50, 86, 770, 974, 770])
        self.assertEqual(value['third-row']['boundaries'], [.1, .4, .8, 2.5])
        with self.assertRaisesRegex(ValueError, 'valence-band'):
            grammar.validate_rule(value['second-row'], REQUEST)

    def test_search_space_is_finite_without_materialization(self):
        self.assertGreater(grammar.rule_space_size(REQUEST), 0)


if __name__ == '__main__':
    unittest.main()
