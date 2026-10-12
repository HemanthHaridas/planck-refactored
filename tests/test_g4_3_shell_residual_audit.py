#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUEST = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json'
SPEC = importlib.util.spec_from_file_location('residual_audit', ROOT / 'g4_3_shell_residual_audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def request(): return json.loads(REQUEST.read_text())


def layer(size, energy):
    channels = [dict(spin='alpha', D_V=1., D_fxc_D=2.), dict(spin='beta', D_V=0., D_fxc_D=0.)]
    return dict(lebedev_points=size, records=[dict(atom=0, atomic_number=8, radial_index=0,
        radius_bohr=.8, u=.5, lebedev_points=size, xc_energy=energy,
        direction_records=[dict(index=0, spin_channels=channels), dict(index=1, spin_channels=channels)])])


def selections():
    return [dict(atom=0, atomic_number=8, radial_index=0, lebedev_points=50)]


def direct():
    return dict(rows=[dict(name='legacy', xc_energy=-.1,
        directions=[dict(D_V=1., D_fxc_D=2.), dict(D_V=1., D_fxc_D=2.)])])


class G43ShellResidualAudit(unittest.TestCase):
    def test_requires_every_frozen_lebedev_layer(self):
        req = request()
        raw = dict(contract=audit.RAW_CONTRACT, layers=[layer(size, -.1) for size in req['allowed_lebedev_points']],
                   legacy_selection=selections())
        report = audit.normalize(raw, req['fixtures'][0], req, direct())
        self.assertEqual(len(report['residuals']), len(req['allowed_lebedev_points']) - 1)
        self.assertEqual(report['legacy_reconstruction']['selected_shells'], 1)

    def test_rejects_missing_layer(self):
        req = request()
        raw = dict(contract=audit.RAW_CONTRACT, layers=[layer(size, -.1) for size in req['allowed_lebedev_points'][1:]],
                   legacy_selection=selections())
        with self.assertRaisesRegex(ValueError, 'cover'):
            audit.normalize(raw, req['fixtures'][0], req)

    def test_rejects_bad_legacy_reconstruction(self):
        req = request()
        raw = dict(contract=audit.RAW_CONTRACT, layers=[layer(size, -.1) for size in req['allowed_lebedev_points']],
                   legacy_selection=selections())
        bad = direct()
        bad['rows'][0]['xc_energy'] = -.2
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            audit.normalize(raw, req['fixtures'][0], req, bad)


if __name__ == '__main__':
    unittest.main()
