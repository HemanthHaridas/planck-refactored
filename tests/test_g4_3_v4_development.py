#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUEST = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_v4_development_request.json'
SPEC = importlib.util.spec_from_file_location('v4', ROOT / 'g4_3_v4_development.py')
v4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v4)


class G43V4DevelopmentTests(unittest.TestCase):
    def test_complete_consumed_bank_is_valid(self):
        self.assertEqual(len(v4.load(REQUEST)['cases']), 9)

    def test_rejects_a_missing_consumed_case(self):
        value = json.loads(REQUEST.read_text())
        value['cases'].pop()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=REQUEST.parent) as stream:
            json.dump(value, stream)
            stream.flush()
            with self.assertRaisesRegex(ValueError, 'retain every'):
                v4.load(stream.name)


if __name__ == '__main__':
    unittest.main()
