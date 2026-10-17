"""Runner/ledger tests only: no compiler or molecular executable is invoked."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import dh_hessian_swap_audit as audit


def ledger(status="PASS"):
    lines = ["DH_HESSIAN_SWAP_PROBE 1"]
    for name in ("H.total", "H.orbital", "H.J", "H.K", "H.XC", "gradient.total"):
        for suffix in ("dense", "candidate", "delta"):
            lines += [f"MATRIX {name}.{suffix} 1 1", "0.0"]
    for name in ("gmres_dense", "gmres_eq27", "gmres_shared"):
        lines += [f"SCALAR {name}.converged 1", f"ITERATION {name} 0 0.0"]
    lines += [f"STATUS {status}"]
    return "\n".join(lines) + "\n"


class HessianSwapAuditTests(unittest.TestCase):
    def test_ledger_requires_completed_probe_and_all_channels(self):
        self.assertEqual(audit.parse_ledger(ledger())["status"], "PASS")
        self.assertEqual(audit.parse_ledger(ledger("DIFFERENCE"))["status"], "DIFFERENCE")
        for malformed in ("", ledger().replace("STATUS PASS", "STATUS FAILURE"),
                          ledger().replace("STATUS PASS", ""),
                          ledger().replace("MATRIX H.XC.delta 1 1\n0.0\n", ""),
                          ledger().replace("MATRIX H.J.delta 1 1\n0.0", "MATRIX H.J.delta 2 1\n0.0"),
                          ledger().replace("MATRIX H.J.delta 1 1\n0.0", "MATRIX H.J.delta 1 1\nnan"),
                          ledger().replace("gmres_shared.converged 1", "gmres_shared.converged 0")):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                audit.parse_ledger(malformed)

    def test_matrix_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.matrix_error([[0, 1]], [[0]])
        self.assertEqual(audit.matrix_error([[1, 2]], [[1, 3]]), 1)

    def run_mock(self, *, status="PASS", marker=True, fd=False, swapped_offset=0.0):
        with tempfile.TemporaryDirectory(prefix="dh-swap-unit-") as temp:
            root = Path(temp)
            fixture = root / "fixture.hfinp"
            fixture.write_text("calculation gradient\n%begin_coords\n2\n0 1\n"
                               "O 0.1 0.2 -0.1\nH 0.8 -0.4 0.6\n%end_coords\n")
            calls = []

            def fake_run(argv, *, env, text, capture_output):
                source = Path(argv[1]).read_text()
                _, coords, _, _ = audit.parse_geometry(source)
                bohr = [[x * audit.ANGSTROM_TO_BOHR for x in row] for row in coords]
                data = {"natoms": 2, "total_energy": 0.5 * sum(x*x for row in bohr for x in row)}
                probe = "PLANCK_DFT_DH_HESSIAN_PROBE_LOG" in env
                self.assertNotIn("PLANCK_DFT_DH_LITERAL_EQ47_RHS", env)
                if "calculation gradient" in source:
                    data["gradient"] = [row[:] for row in bohr]
                    if probe:
                        data["gradient"][0][0] += swapped_offset
                else:
                    self.assertFalse(probe, "FD endpoints must use the unmodified energy path")
                if probe:
                    Path(env["PLANCK_DFT_DH_HESSIAN_PROBE_LOG"]).write_text(ledger(status))
                Path(argv[3]).write_text(json.dumps(data))
                calls.append((Path(argv[1]).stem, probe))
                output = "Converged : true\n"
                if probe and marker:
                    output += "DH Hessian Probe : Returning shared-action Z gradient\n"
                return subprocess.CompletedProcess(argv, 0, output, "")

            argv = ["audit", str(fixture)] + (["--fd", "--steps", "1e-4"] if fd else [])
            with patch.object(sys, "argv", argv), \
                 patch.object(audit.tempfile, "mkdtemp", return_value=str(root)), \
                 patch.object(audit.subprocess, "run", side_effect=fake_run), \
                 patch.dict(os.environ, {"PLANCK_DFT_DH_LITERAL_EQ47_RHS": "0", "PLANCK_DFT_DH_HESSIAN_PROBE_LOG": "stale"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = audit.main()
            self.assertEqual(calls[:2], [("dense", False), ("shared", True)])
            summary = json.loads((root / "summary.json").read_text())
            report_path = root / "0-fixture/report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else None
            return code, calls, summary, report

    def test_same_fd_is_compared_to_both_gradients_and_only_probe_is_swapped(self):
        code, calls, summary, report = self.run_mock(fd=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 14)
        self.assertTrue(summary[0]["passed"])
        self.assertEqual(len(report["fd"]), 6)
        self.assertLess(max(abs(row["dense_error"]) for row in report["fd"]), 1e-9)
        self.assertLess(max(abs(row["shared_error"]) for row in report["fd"]), 1e-9)

    def test_operator_disagreement_is_failure_even_if_gradient_agrees(self):
        code, _, _, report = self.run_mock(status="DIFFERENCE")
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])

    def test_gradient_disagreement_is_failure_even_if_ledger_says_pass(self):
        code, _, _, report = self.run_mock(swapped_offset=1e-5)
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])

    def test_stale_binary_cannot_silently_pass(self):
        code, _, summary, _ = self.run_mock(marker=False)
        self.assertEqual(code, 1)
        self.assertIn("not executed", summary[0]["error"])


if __name__ == "__main__":
    unittest.main()
