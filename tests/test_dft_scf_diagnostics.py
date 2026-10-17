#!/usr/bin/env python3
"""Gates for three planck-dft SCF features that HF already had.

1. MO energy printing -- planck-dft printed NO orbital energies at any
   verbosity, so an SCF stall could not be attributed without patching the
   source (diagnosing the h2o2-cation UKS stall needed a temporary probe).
2. DIIS restart -- `diis_restart_factor` was implemented in run_rhf/run_uhf
   but had zero references in the DFT driver, so the KS SCF had no escape
   from a poisoned DIIS subspace.
3. Auto-SOSCF -- SOSCF had to be switched on at a hand-picked iteration.
   The case it exists for is a slow orbital-rotation mode whose onset is not
   known in advance (h2o2 cation: DIIS crawls 1417 iterations along a
   nearly-flat direction; engaging SOSCF converges it in ~300).
"""
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DFT = ROOT / "build" / "planck-dft"
UKS = ROOT / "tests/inputs/exploratory/dh_gradient/uks_u0/water_cation_asymmetric_b2plyp_sto3g.hfinp"
RKS = ROOT / "tests/inputs/regression/dft/h2_dft_b3lyp.hfinp"
STALL = ROOT / "tests/inputs/exploratory/dh_gradient/uks_u0/h2o2_cation_c1_b2plyp_sto3g.hfinp"

ENERGY = re.compile(r"DFT Energy\s*:\s*(-?[0-9.]+)\s+Eh")
ITER = re.compile(r"^\s*(\d+)\s+-?[0-9.]", re.MULTILINE)


def run(text_or_path, **edits):
    text = text_or_path.read_text() if isinstance(text_or_path, Path) else text_or_path
    for old, new in edits.items():
        text = text.replace(old.replace("__", " "), new)
    env = dict(os.environ, BASIS_PATH=str(ROOT / "basis-sets"))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.hfinp"
        p.write_text(text)
        return subprocess.run([str(DFT), str(p)], cwd=d, env=env,
                              capture_output=True, text=True, timeout=3600).stdout


@unittest.skipUnless(DFT.exists(), "planck-dft not built")
class TestDftScfDiagnostics(unittest.TestCase):

    def test_uks_prints_both_spin_mo_tables(self):
        out = run(UKS)
        self.assertIn("Alpha MOs", out)
        self.assertIn("Beta MOs", out)
        # HOMO/LUMO must be marked in each table, as the HF driver does.
        self.assertEqual(out.count("<-- HOMO"), 2, "expected one HOMO per spin")
        self.assertEqual(out.count("<-- LUMO"), 2, "expected one LUMO per spin")

    def test_rks_prints_one_mo_table(self):
        out = run(RKS)
        self.assertIn("RKS MOs", out)
        self.assertIn("<-- HOMO", out)
        self.assertIn("<-- LUMO", out)

    def test_diis_restart_fires_and_is_configurable(self):
        # The stalling case grows its DIIS error past 2x several times.
        base = STALL.read_text().replace("max_cycles  200", "max_cycles  3000")
        base = "\n".join(l for l in base.splitlines() if "level_shift" not in l)
        self.assertIn("Subspace restarted", run(base),
                      "DIIS restart never fired on a case whose error doubles")
        # diis_restart 0 disables it, matching the HF semantics.
        off = base.replace("    use_diis    .true.",
                           "    use_diis    .true.\n    diis_restart 0.0")
        self.assertNotIn("Subspace restarted", run(off),
                         "diis_restart 0 must disable restarts")

    def test_auto_soscf_engages_itself_and_is_off_by_default(self):
        base = STALL.read_text().replace("max_cycles  200", "max_cycles  3000")
        base = "\n".join(l for l in base.splitlines() if "level_shift" not in l)

        plain = run(base)
        self.assertNotIn("Auto-engaged", plain, "auto-SOSCF must be off by default")
        n_plain = len(ITER.findall(plain))

        auto = run(base.replace(
            "    max_cycles  3000",
            "    max_cycles  3000\n    scf_soscf_auto_stall 5\n    scf_soscf_cycles 30"))
        self.assertIn("Auto-engaged", auto, "auto-SOSCF never engaged on a stalling case")
        n_auto = len(ITER.findall(auto))

        # Same solution. Compared at 1e-4 Eh, not to machine precision: the two
        # paths stop at different points inside the same basin, and a DFT energy
        # carries XC-grid noise well above 1e-6 (measured 1.3e-5 between these
        # two runs). The assertion that matters is "same basin", which a 1e-4
        # bound establishes while a 1e-6 one merely measures quadrature noise.
        e_plain, e_auto = ENERGY.search(plain), ENERGY.search(auto)
        self.assertIsNotNone(e_auto, "auto-SOSCF run did not converge")
        self.assertLess(abs(float(e_plain.group(1)) - float(e_auto.group(1))), 1e-4,
                        "auto-SOSCF converged to a different basin")
        # ... reached substantially faster. That is the whole point.
        self.assertLess(n_auto, n_plain / 2,
                        f"auto-SOSCF took {n_auto} iterations vs {n_plain} without")


if __name__ == "__main__":
    unittest.main(verbosity=2)
