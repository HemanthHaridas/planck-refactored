#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('manifest', ROOT / 'g4_calibration_manifest.py')
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


class G4CalibrationManifestContract(unittest.TestCase):
    def test_response_maximum_includes_potential_and_kernel(self):
        self.assertEqual(manifest.response_maximum({'directions': [
            {'D_V': -2., 'D_fxc_D': 1.}, {'D_V': .5, 'D_fxc_D': -3.}]}), 3.)

    def test_candidate_contract_is_explicit(self):
        self.assertEqual(manifest.CANDIDATES,
                         ('legacy-unpruned-max', 'pyscf-nwchem-2.13'))


if __name__ == '__main__':
    unittest.main()
