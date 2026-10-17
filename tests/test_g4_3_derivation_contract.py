#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUEST = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json'
SPEC = importlib.util.spec_from_file_location('g4_3_contract', ROOT / 'g4_3_derivation_contract.py')
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def request_copy():
    return json.loads(REQUEST.read_text())


class G43DerivationContract(unittest.TestCase):
    def test_checked_in_request_is_valid(self):
        request = contract.load(REQUEST)
        self.assertEqual(len(request['fixtures']), 14)
        self.assertEqual(request['allowed_lebedev_points'][-1], 2030)
        repository = ROOT.parent
        for fixture in request['fixtures']:
            self.assertTrue((repository / fixture).is_file(), fixture)

    def test_rejects_duplicate_fixture(self):
        request = request_copy()
        request['fixtures'][-1] = request['fixtures'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate fixture'):
            contract.validate(request)

    def test_rejects_unsupported_lebedev_size(self):
        request = request_copy()
        request['allowed_lebedev_points'].append(123)
        with self.assertRaisesRegex(ValueError, 'sorted and unique|unsupported'):
            contract.validate(request)

    def test_rejects_non_increasing_boundary_specimen(self):
        request = request_copy()
        request['allowed_band_boundaries'][3] = 0.2
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            contract.validate(request)

    def test_rejects_live_selector_dependency(self):
        request = request_copy()
        request['shell_selector']['depends_on'].append('density')
        with self.assertRaisesRegex(ValueError, 'selector dependencies'):
            contract.validate(request)

    def test_rejects_missing_direction(self):
        request = request_copy()
        request['directions_per_fixture'] = 1
        with self.assertRaisesRegex(ValueError, 'two AO directions'):
            contract.validate(request)


if __name__ == '__main__':
    unittest.main()
