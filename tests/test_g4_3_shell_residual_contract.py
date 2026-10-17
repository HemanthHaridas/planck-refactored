#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUEST_PATH = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json'
ATLAS_PATH = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_shell_atlas_synthetic.json'
SPEC = importlib.util.spec_from_file_location('residuals', ROOT / 'g4_3_shell_residual_contract.py')
residuals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(residuals)


def request():
    return json.loads(REQUEST_PATH.read_text())


def table():
    value = json.loads(ATLAS_PATH.read_text())
    value['contract'] = residuals.CONTRACT
    lower = json.loads(json.dumps(value['records'][0]))
    lower['lebedev_points'] = 50
    lower['xc_energy'] -= .01
    lower['direction_records'][0]['spin_channels'][0]['D_fxc_D'] -= .02
    value['records'].append(lower)
    return value


class G43ShellResidualContract(unittest.TestCase):
    def test_derives_lower_order_shell_residuals(self):
        result = residuals.validate(table(), request())
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['xc_energy'], -.01)
        self.assertAlmostEqual(result[0]['directions'][0]['D_fxc_D'], -.02)

    def test_rejects_missing_reference_shell(self):
        value = table()
        value['records'] = value['records'][1:]
        with self.assertRaisesRegex(ValueError, 'missing 2030'):
            residuals.validate(value, request())

    def test_rejects_changed_shell_identity(self):
        value = table()
        value['records'][1]['u'] = .6
        with self.assertRaisesRegex(ValueError, 'physical band|identity'):
            residuals.validate(value, request())


if __name__ == '__main__':
    unittest.main()
