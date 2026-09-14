"""Fast runner tests; no Planck binary or compiler is invoked."""
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import dh_cartesian_fd_audit as audit


class ProductionCartesianAuditTests(unittest.TestCase):
    def test_missing_or_invalid_production_data_is_rejected(self):
        for data in (
            {"natoms": 1, "total_energy": math.nan},
            {"natoms": 1, "total_energy": 0.0},
            {"natoms": 1, "total_energy": 0.0, "gradient": [[0, 0]]},
            {"natoms": 1, "total_energy": 0.0, "gradient": [[0, math.inf, 0]]},
        ):
            with self.assertRaises(ValueError):
                audit.validate_result(data, 1, True)

    def test_complete_normal_json_workflow_without_debug_flags(self):
        with tempfile.TemporaryDirectory(prefix="dh-runner-unit-") as directory:
            root = Path(directory)
            template = root / "fixture.hfinp"
            template.write_text("calculation gradient\n%begin_coords\n2\n0 1\n"
                                "O 0.1 0.2 -0.1\nH 0.8 -0.4 0.6\n%end_coords\n")
            calls = []

            def fake_run(argv, *, env, text, capture_output):
                self.assertFalse(any(k.startswith("PLANCK_DFT_DH_") for k in env))
                self.assertEqual(argv[2], "--json")
                input_text = Path(argv[1]).read_text()
                _, coords, _, _ = audit.parse_geometry(input_text)
                bohr = [[v * audit.ANGSTROM_TO_BOHR for v in row] for row in coords]
                # Independent quadratic total energy; its gradient is R.
                energy = 0.5 * sum(v * v for row in bohr for v in row)
                data = {"natoms": 2, "total_energy": energy}
                if "calculation gradient" in input_text:
                    data["gradient"] = bohr
                Path(argv[3]).write_text(json.dumps(data))
                calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, "Converged : true\n", "")

            with patch.object(sys, "argv", ["audit", str(template), "--jobs", "2"]), \
                 patch.object(audit.tempfile, "mkdtemp", return_value=str(root)), \
                 patch.object(audit.subprocess, "run", side_effect=fake_run), \
                 patch.dict(audit.os.environ, {"PLANCK_DFT_DH_LITERAL_EQ47_RHS": "0"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(), 0)
            report = json.loads((root / "results.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(len(calls), 25)  # center + 6 coordinates * 2 steps * 2 signs
            self.assertEqual(len(report["results"]), 12)
            self.assertLess(max(row["max_abs"] for row in report["summaries"]), 1e-9)


if __name__ == "__main__":
    unittest.main()
