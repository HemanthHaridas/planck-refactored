#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('rescore', ROOT / 'g4_3_direct_rescore_audit.py')
rescore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rescore)


RULES = {
    'hydrogen-helium': {'boundaries': [.4, 1., 2.5, 3.5], 'lebedev_points': [26, 110, 1454, 770, 434]},
    'second-row': {'boundaries': [.25, 1., 2.5, 4.5], 'lebedev_points': [50, 434, 1454, 590, 170]},
    'third-row': {'boundaries': [.1, .5, 3.5, 4.5], 'lebedev_points': [26, 146, 770, 230, 26]},
}


def shell(atom, z, u, begin):
    count = RULES[rescore._family(z)]['lebedev_points'][rescore.select_band(u, RULES[rescore._family(z)]['boundaries'])]
    return dict(atom=atom, atomic_number=z, radial_index=0, u=u, lebedev_points=count,
                begin=begin, end=begin + count)


def fixture():
    shells = [shell(0, 1, .2, 0), shell(1, 8, .2, 26), shell(2, 16, .2, 76)]
    records = []
    residuals = []
    for item in shells:
        key = [item['atom'], item['atomic_number'], item['radial_index']]
        records.append(dict(atom=item['atom'], atomic_number=item['atomic_number'], radial_index=0,
                            u=item['u'], lebedev_points=2030, xc_energy=1., direction_records=[
                                dict(index=0, spin_channels=[dict(D_V=.2, D_fxc_D=.3), dict(D_V=.4, D_fxc_D=.5)]),
                                dict(index=1, spin_channels=[dict(D_V=.6, D_fxc_D=.7), dict(D_V=.8, D_fxc_D=.9)])]))
        residuals.append(dict(key=key, lebedev_points=item['lebedev_points'], xc_energy=.01, directions=[
            dict(D_V=.02, D_fxc_D=.03), dict(D_V=.04, D_fxc_D=.05)]))
    raw = dict(contract=rescore.RAW_CONTRACT, radial_points=135, directions=2, fingerprint=rescore.FINGERPRINT, candidate=dict(
        points=shells[-1]['end'], xc_energy=3.03, alpha_norm=0., beta_norm=0.,
        directions=[dict(D_V=1.86, D_fxc_D=2.49), dict(D_V=4.32, D_fxc_D=4.95)], shells=shells))
    return raw, dict(records=records, residuals=residuals)


class G43DirectRescoreTests(unittest.TestCase):
    def test_exact_direct_value_reconciles_to_atlas(self):
        raw, table = fixture()
        result = rescore.reconcile(raw, table, RULES)
        self.assertTrue(result['passed'])
        self.assertEqual(result['points'], 222)

    def test_wrong_direct_shell_rule_is_rejected(self):
        raw, table = fixture()
        raw['candidate']['shells'][1]['lebedev_points'] = 434
        with self.assertRaisesRegex(ValueError, 'selector differs'):
            rescore.reconcile(raw, table, RULES)

    def test_revised_second_row_fingerprint_and_shell_rule_reconcile(self):
        raw, table = fixture()
        revised = dict(RULES)
        revised['second-row'] = rescore.REVISED_SECOND_ROW
        table['residuals'].append(dict(key=[1, 8, 0], lebedev_points=86, xc_energy=.01,
                                       directions=[dict(D_V=.02, D_fxc_D=.03),
                                                   dict(D_V=.04, D_fxc_D=.05)]))
        shells = raw['candidate']['shells']
        shells[1]['lebedev_points'] = 86
        shells[1]['end'] = 112
        shells[2]['begin'] = 112
        shells[2]['end'] = 258
        raw['candidate']['points'] = 258
        raw['fingerprint'] = rescore.REVISED_FINGERPRINT
        result = rescore.reconcile(raw, table, revised)
        self.assertTrue(result['passed'])
        self.assertEqual(rescore.expected_fingerprint(revised), rescore.REVISED_FINGERPRINT)

    def test_dual_floor_rule_selects_only_the_dual_floor_fingerprint(self):
        dual = dict(RULES)
        dual['hydrogen-helium'] = rescore.DUAL_FLOOR_HYDROGEN_HELIUM
        dual['second-row'] = rescore.DUAL_FLOOR_SECOND_ROW
        self.assertEqual(rescore.expected_fingerprint(dual), rescore.DUAL_FLOOR_FINGERPRINT)

    def test_v3_constrained_rule_selects_only_the_v3_fingerprint(self):
        candidate = dict(RULES)
        candidate['hydrogen-helium'] = rescore.V3_HYDROGEN_HELIUM
        candidate['second-row'] = rescore.V3_SECOND_ROW
        self.assertEqual(rescore.expected_fingerprint(candidate), rescore.V3_FINGERPRINT)

    def test_v4_robust_rule_selects_only_the_v4_fingerprint(self):
        candidate = dict(RULES)
        candidate['hydrogen-helium'] = rescore.V4_HYDROGEN_HELIUM
        candidate['second-row'] = rescore.V4_SECOND_ROW
        candidate['third-row'] = rescore.V4_THIRD_ROW
        self.assertEqual(rescore.expected_fingerprint(candidate), rescore.V4_FINGERPRINT)


if __name__ == '__main__':
    unittest.main()
