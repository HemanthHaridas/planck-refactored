#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('reference_ladder', ROOT / 'g4_3_reference_ladder_audit.py')
ladder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ladder)


def row(name, energy=0., potential=0., kernel=0.):
    return dict(name=name, points=1, xc_energy=energy, alpha_norm=0., beta_norm=0.,
                directions=[dict(D_V=potential, D_fxc_D=kernel),
                            dict(D_V=potential, D_fxc_D=kernel)])


class G43ReferenceLadderContract(unittest.TestCase):
    def test_separates_radial_and_angular_deltas(self):
        ledger = dict(contract=ladder.CONTRACT, rows=[
            row('uniform-2030-a', energy=1.),
            row('uniform-2030-b', energy=1. + 0.5e-8),
            row('uniform-2702-c', energy=1. + 0.5e-8, kernel=0.5e-7),
        ])
        report = ladder.normalize(ledger, 'fixture.hfinp')
        self.assertAlmostEqual(report['radial_delta']['energy'], 0.5e-8)
        self.assertAlmostEqual(report['angular_delta']['energy'], 0.)
        self.assertTrue(report['reference_stable'])

    def test_rejects_an_unstable_axis(self):
        ledger = dict(contract=ladder.CONTRACT, rows=[
            row('uniform-2030-a'), row('uniform-2030-b', energy=2e-8), row('uniform-2702-c', energy=2e-8),
        ])
        report = ladder.normalize(ledger, 'fixture.hfinp')
        self.assertFalse(report['radial_stable'])
        self.assertFalse(report['reference_stable'])

    def test_requires_exact_reference_rows(self):
        ledger = dict(contract=ladder.CONTRACT, rows=[
            row('uniform-2030-a'), row('uniform-2030-b'), row('wrong-c'),
        ])
        with self.assertRaisesRegex(ValueError, 'rows'):
            ladder.normalize(ledger, 'fixture.hfinp')


if __name__ == '__main__':
    unittest.main()
