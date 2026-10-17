#!/usr/bin/env python3
"""Gate: the UKS level shift must be released early, not fully converged.

A level shift is a convergence aid. Its fixed point is NOT the answer -- when
the shift comes off the density moves a long way (dP jumps to ~1.4 here), so
every iteration spent converging the SHIFTED phase tightly is work on a result
about to be discarded.

Before the fix the shift was held until the full convergence gate passed:
water_triplet/PBE spent 70 of an 80-iteration budget shifted (56 of those below
dP 1e-5), leaving 10 for the real unshifted solve, so the run reported failure
while converging correctly. Releasing at dP < shift_release_tol hands off at
iteration 15 instead of 71 and finishes in 58 instead of 113.

Both assertions are load-bearing:
  1. converges inside the input's own budget, to the reference energy
  2. handoff happens EARLY -- assertion 1 alone would pass if someone simply
     raised max_cycles, which treats the symptom and not the waste.
"""
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DFT = ROOT / "build" / "planck-dft"
CASE = ROOT / "tests/inputs/regression/dft/water_triplet_uks_pbe_sto3g.hfinp"

REFERENCE_ENERGY = -74.8423100229
RELEASE = re.compile(r"Shift released at dP")
ENERGY = re.compile(r"DFT Energy\s*:\s*(-?[0-9.]+)\s+Eh")
ITER = re.compile(r"^\s*(\d+)\s+-?[0-9.]", re.MULTILINE)


@unittest.skipUnless(DFT.exists() and CASE.exists(), "planck-dft or fixture missing")
class TestUksLevelShiftHandoff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        env = dict(os.environ, BASIS_PATH=str(ROOT / "basis-sets"))
        cls.out = subprocess.run([str(DFT), str(CASE)], cwd=ROOT, env=env,
                                 capture_output=True, text=True, timeout=1800).stdout
        cls.iters = [int(m) for m in ITER.findall(cls.out)]

    def test_converges_within_the_inputs_own_budget(self):
        energy = ENERGY.search(self.out)
        self.assertIsNotNone(
            energy, "run did not converge inside the input's own max_cycles")
        self.assertAlmostEqual(float(energy.group(1)), REFERENCE_ENERGY, places=5)

    def test_shift_is_released_early(self):
        self.assertRegex(self.out, RELEASE, "shift was never released early")
        # The release line is printed on the handoff iteration; everything after
        # it is the real unshifted solve. Most of the budget must be left for it.
        before = self.out.split("Shift released at dP")[0]
        handoff = len(ITER.findall(before))
        total = len(self.iters)
        self.assertLess(
            handoff, total / 2,
            f"handoff at iteration {handoff} of {total} -- the shifted phase is "
            "still consuming most of the budget on a discarded fixed point")


if __name__ == "__main__":
    unittest.main(verbosity=2)
