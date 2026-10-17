#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUEST_PATH = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json'
ATLAS_PATH = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_shell_atlas_synthetic.json'
SPEC = importlib.util.spec_from_file_location('shell_atlas', ROOT / 'g4_3_shell_atlas_contract.py')
atlas_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(atlas_contract)


def request():
    return json.loads(REQUEST_PATH.read_text())


def atlas():
    return json.loads(ATLAS_PATH.read_text())


class G43ShellAtlasContract(unittest.TestCase):
    def test_valid_synthetic_atlas(self):
        atlas_contract.validate(atlas(), request())

    def test_selector_is_geometry_independent(self):
        boundaries = [0.25, 0.5, 1.0, 4.5]
        self.assertEqual(atlas_contract.physical_band(0.5, boundaries), 1)
        self.assertEqual(atlas_contract.physical_band(0.5000001, boundaries), 2)

    def test_rejects_misassigned_family(self):
        value = atlas()
        value['records'][0]['family'] = 'third-row'
        with self.assertRaisesRegex(ValueError, 'family'):
            atlas_contract.validate(value, request())

    def test_rejects_misassigned_band(self):
        value = atlas()
        value['records'][0]['physical_band'] = 2
        with self.assertRaisesRegex(ValueError, 'physical band'):
            atlas_contract.validate(value, request())

    def test_rejects_duplicate_spin_channel(self):
        value = atlas()
        value['records'][0]['direction_records'][0]['spin_channels'][1]['spin'] = 'alpha'
        with self.assertRaisesRegex(ValueError, 'spin channel'):
            atlas_contract.validate(value, request())

    def test_external_fixture_requires_explicit_holdout_mode(self):
        value = atlas()
        value['fixture'] = 'tests/inputs/exploratory/dh_gradient/grid_calibration/holdout/example.hfinp'
        with self.assertRaisesRegex(ValueError, 'derivation request'):
            atlas_contract.validate(value, request())
        atlas_contract.validate(value, request(), allow_external_fixture=True)


if __name__ == '__main__':
    unittest.main()
