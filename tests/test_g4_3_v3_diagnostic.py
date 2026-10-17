#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUEST = ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_v3_diagnostic_request.json'
SPEC = importlib.util.spec_from_file_location('v3', ROOT / 'g4_3_v3_diagnostic.py')
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)


class G43V3DiagnosticTests(unittest.TestCase):
    def test_frozen_request_is_valid(self):
        self.assertEqual(v3.load(REQUEST)['contract'], v3.CONTRACT)

    def test_rejects_using_the_holdout_as_selection_data(self):
        value = json.loads(REQUEST.read_text())
        value['forbids'].remove('selection-atlas-augmentation')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=REQUEST.parent) as stream:
            json.dump(value, stream)
            stream.flush()
            with self.assertRaisesRegex(ValueError, 'scope'):
                v3.load(stream.name)

    def test_rejects_hydrogen_single_band_shortcut(self):
        value = json.loads(REQUEST.read_text())
        value['hydrogen_helium']['forbid_single_band_floor'] = False
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=REQUEST.parent) as stream:
            json.dump(value, stream)
            stream.flush()
            with self.assertRaisesRegex(ValueError, 'five-band'):
                v3.load(stream.name)


if __name__ == '__main__':
    unittest.main()
