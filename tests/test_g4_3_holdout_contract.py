#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUEST = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_holdout_request.json'
FRESH_REQUEST = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_dual_floor_holdout_request.json'
V3_REQUEST = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_v3_holdout_request.json'
SPEC = importlib.util.spec_from_file_location('holdout', ROOT / 'g4_3_holdout_contract.py')
holdout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(holdout)


class G43HoldoutContractTests(unittest.TestCase):
    def test_frozen_request_is_valid(self):
        value = holdout.load(REQUEST)
        self.assertEqual(value['contract'], holdout.CONTRACT)

    def test_rejects_changed_limits(self):
        value = json.loads(REQUEST.read_text())
        value['limits']['xc_energy_ha'] = 1e-6
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as stream:
            json.dump(value, stream)
            stream.flush()
            with self.assertRaisesRegex(ValueError, 'limits'):
                holdout.load(stream.name)

    def test_fresh_request_is_valid_and_distinct(self):
        value = holdout.load(FRESH_REQUEST)
        self.assertEqual(value['contract'], holdout.FRESH_CONTRACT)
        original = holdout.load(REQUEST)
        fresh_names = {case['name'] for case in value['cases']}
        original_names = {case['name'] for case in original['cases']}
        self.assertTrue(fresh_names.isdisjoint(original_names))

    def test_triplet_holdout_keeps_tight_root_and_original_budget(self):
        path = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/holdout_v2/water_triplet_uks_pbe_displaced_sto3g.hfinp'
        text = path.read_text()
        self.assertIn('level_shift 0.3', text)
        self.assertIn('max_cycles  120', text)
        self.assertIn('tol_energy  1.0e-10', text)
        self.assertIn('tol_density 1.0e-10', text)

    def test_v3_request_is_valid_and_input_states_are_new(self):
        value = holdout.load(V3_REQUEST)
        self.assertEqual(value['contract'], holdout.V3_CONTRACT)
        all_inputs = [case['input'] for case in value['cases']]
        self.assertEqual(len(all_inputs), len(set(all_inputs)))


if __name__ == '__main__':
    unittest.main()
