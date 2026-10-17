#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUEST = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json').read_text())
SPEC = importlib.util.spec_from_file_location('certificate', ROOT / 'g4_3_rule_certificate.py')
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def table():
    key = [0, 1, 0]
    return dict(fixture=REQUEST['fixtures'][0], records=[dict(atom=0, atomic_number=1, radial_index=0,
        family='hydrogen-helium', u=.2, lebedev_points=2030)],
        residuals=[dict(key=key, lebedev_points=level, xc_energy=2e-8,
                        directions=[dict(D_V=3e-7, D_fxc_D=-4e-7),
                                    dict(D_V=-3e-7, D_fxc_D=4e-7)])
                   for level in REQUEST['allowed_lebedev_points'] if level != 2030])


class G43RuleCertificateTests(unittest.TestCase):
    def test_l1_certificate_rejects_cancelling_shell_error(self):
        rule = {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [26] * 5}
        value = certificate.certificate(rule, 'hydrogen-helium', [table()], REQUEST)
        self.assertEqual(value['fixtures'][0]['l1_xc_energy'], 2e-8)
        self.assertFalse(certificate.passes(value, REQUEST))

    def test_all_2030_has_zero_certificate(self):
        rule = {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [2030] * 5}
        value = certificate.certificate(rule, 'hydrogen-helium', [table()], REQUEST)
        self.assertTrue(certificate.passes(value, REQUEST))

    def test_filter_retains_only_qualified_rule(self):
        rules = [
            {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [26] * 5},
            {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [2030] * 5},
        ]
        accepted = certificate.qualifying_rules(rules, 'hydrogen-helium', [table()], REQUEST)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['rule']['lebedev_points'], [2030] * 5)

    def test_band_vectors_preserve_the_single_shell_l1_value(self):
        value = certificate.band_l1_vectors('hydrogen-helium', [.1, .25, .5, 1.], [table()], REQUEST)
        self.assertEqual(value[1, 26][0]['xc_energy'], 2e-8)
        self.assertEqual(value[1, 2030][0]['directions'][0]['D_fxc_D'], 0.0)

    def test_branch_search_requires_2030_for_the_populated_band(self):
        rules = list(certificate.qualifying_for_boundaries('hydrogen-helium', [.1, .25, .5, 1.], [table()], REQUEST))
        self.assertTrue(rules)
        self.assertTrue(all(rule['lebedev_points'][1] == 2030 for rule in rules))

    def test_compact_frontier_keeps_only_non_dominated_costs(self):
        model = [('fixture', 0, dict(u=.2))]
        rules = [
            {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [26, 50, 50, 50, 50]},
            {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [26, 86, 86, 86, 86]},
            {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [26, 50, 50, 50, 50]},
        ]
        frontier = certificate.compact_cost_frontier(rules, model)
        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0]['total_ledger_points'], 50)

    def test_boundary_weighted_cost_matches_direct_cost(self):
        model = [('fixture', 0, dict(u=.2)), ('fixture', 0, dict(u=.7))]
        rule = {'boundaries': [.1, .25, .5, 1.], 'lebedev_points': [26, 50, 86, 110, 146]}
        self.assertEqual(certificate.weighted_rule_cost(rule, certificate.boundary_cost_weights(rule['boundaries'], model)),
                         certificate.family_rule_cost(rule, model))

    def test_precomputed_representatives_need_no_rescoring(self):
        reps = {(10, 20): {'name': 'a'}, (11, 19): {'name': 'b'}, (12, 30): {'name': 'c'}}
        self.assertEqual([row['rule']['name'] for row in certificate.compact_frontier_from_representatives(reps)],
                         ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
