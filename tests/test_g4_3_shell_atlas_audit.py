#!/usr/bin/env python3
import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUEST_PATH = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json'
SPEC = importlib.util.spec_from_file_location('atlas_audit', ROOT / 'g4_3_shell_atlas_audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def request():
    return json.loads(REQUEST_PATH.read_text())


def raw():
    channel = lambda scale: [dict(spin='alpha', D_V=scale, D_fxc_D=2. * scale),
                             dict(spin='beta', D_V=0., D_fxc_D=0.)]
    record = lambda energy, scale: dict(atom=0, atomic_number=8, radial_index=0,
                                        radius_bohr=.8, u=.5, lebedev_points=2030,
                                        xc_energy=energy, direction_records=[dict(index=0, spin_channels=channel(scale)),
                                                                              dict(index=1, spin_channels=channel(scale))])
    return dict(contract=audit.RAW_CONTRACT, layers=[dict(lebedev_points=2030, xc_energy=.3,
                directions=[dict(D_V=3., D_fxc_D=6.), dict(D_V=3., D_fxc_D=6.)],
                records=[record(.1, 1.), record(.2, 2.)])])


class G43ShellAtlasAudit(unittest.TestCase):
    def test_reconstructs_monolithic_energy_and_directions(self):
        atlas = audit.normalize(raw(), request()['fixtures'][0], request())
        self.assertEqual(atlas['contract'], 'g4-3-shell-atlas-v1')
        self.assertAlmostEqual(atlas['reconstruction']['xc_energy']['shell_sum'], .3)

    def test_rejects_nonadditive_shell_energy(self):
        value = raw()
        value['layers'][0]['records'][1]['xc_energy'] = .3
        with self.assertRaisesRegex(ValueError, 'energy does not reconstruct'):
            audit.normalize(value, request()['fixtures'][0], request())

    def test_rejects_nonadditive_direction(self):
        value = raw()
        value['layers'][0]['directions'][1]['D_fxc_D'] = 7.
        with self.assertRaisesRegex(ValueError, 'direction 1'):
            audit.normalize(value, request()['fixtures'][0], request())


if __name__ == '__main__':
    unittest.main()
