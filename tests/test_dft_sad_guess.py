#!/usr/bin/env python3
"""Gate: `guess sad` is honored by planck-dft, for RKS and UKS.

Before this gate, initialize_ks_guess() built an HCore density unconditionally
and never read calculator._scf._guess, so `guess sad` parsed fine and was then
silently discarded -- hcore and sad produced bit-identical trajectories.

Two assertions per reference type, and BOTH are load-bearing:

  1. same converged energy  -- SAD's molecular projection is correct
  2. different iteration 1  -- SAD actually reached the guess (non-vacuity)

(1) alone passes against the old no-op code, since a discarded guess trivially
gives the same energy. (2) is what fails if the wiring regresses.
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DFT = ROOT / "build" / "planck-dft"
FIXTURES = ROOT / "tests/inputs/exploratory/dh_gradient/uks_u0"

ITER1 = re.compile(r"^1\s+(-?[0-9.]+)", re.MULTILINE)
ENERGY = re.compile(r"DFT Energy\s*:\s*(-?[0-9.]+)\s+Eh")


def run(text):
    env = dict(os.environ, BASIS_PATH=str(ROOT / "basis-sets"))
    with tempfile.TemporaryDirectory() as d:
        inp = Path(d) / "case.hfinp"
        inp.write_text(text)
        out = subprocess.run([str(DFT), str(inp)], cwd=d, env=env,
                             capture_output=True, text=True, timeout=1800).stdout
    e, i1 = ENERGY.search(out), ITER1.search(out)
    assert e and i1, f"could not parse run output:\n{out[-2000:]}"
    return float(e.group(1)), float(i1.group(1))


@unittest.skipUnless(DFT.exists(), "planck-dft not built")
class TestDftSadGuess(unittest.TestCase):

    def _check(self, text):
        assert "guess       hcore" in text
        e_h, i_h = run(text)
        e_s, i_s = run(text.replace("guess       hcore", "guess       sad"))
        # SAD must reach the same solution ...
        self.assertAlmostEqual(e_h, e_s, places=7,
                               msg=f"SAD converged elsewhere: {e_h} vs {e_s}")
        # ... from a genuinely different starting density.
        self.assertNotAlmostEqual(
            i_h, i_s, places=6,
            msg=f"iteration-1 energy identical ({i_h}) -- guess sad is a no-op")

    def test_uks(self):
        self._check((FIXTURES / "water_cation_asymmetric_b2plyp_sto3g.hfinp").read_text())

    def test_rks(self):
        text = (FIXTURES / "water_triplet_asymmetric_b2plyp_sto3g.hfinp").read_text()
        self._check(text.replace("scf_type    uhf", "scf_type    rhf").replace("0 3", "0 1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
