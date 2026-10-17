#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path

from g4_3_derivation_contract import load as load_derivation
from g4_3_rule_certificate import certificate, passes


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('frontier', ROOT / 'g4_3_v3_hydrogen_frontier.py')
frontier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier)
DERIVATION = load_derivation(ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
TABLE = json.loads((ROOT / 'inputs/exploratory/dh_gradient/grid_calibration/artifacts/g4_3_9_revision_2026_10_03/holdouts_v2/water-triplet-uks-pbe-displaced-sto3g/residual.json').read_text())


class G43V3HydrogenFrontierTests(unittest.TestCase):
    def test_one_boundary_diagnostic_frontier_is_nonempty_and_qualified(self):
        boundaries = [[0.4, 1.0, 2.5, 3.5]]
        rows = frontier.diagnostic_frontier(TABLE, DERIVATION, boundaries)
        self.assertTrue(rows)
        for row in rows:
            proof = certificate(row['rule'], 'hydrogen-helium', [TABLE], DERIVATION)
            self.assertTrue(passes(proof, DERIVATION))


if __name__ == '__main__':
    unittest.main()
