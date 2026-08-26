"""R4.2 fixture gate: the amplitudes must be a fixed point, and t3 must be LIVE.

Two independent vacuity traps this pins:

1. **Mismatched integrals.** The integrals MUST come from the same SCF call the
   solve used. Be/STO-3G has a 6-fold degenerate virtual (2p) shell, so GHF
   picks an arbitrary rotation within it -- repeated ``_spinorbital_integrals``
   calls in ONE process return different ``mo_coeff``. Rebuilding ``v`` for the
   check compares amplitudes in one orbital basis against integrals in another
   and yields a spurious ~1e-01 doubles residual. The energy is
   rotation-invariant, so it stays stable to 12 digits and hides the mismatch.

2. **Inert manifolds.** On Be/STO-3G both t1 and t3 sit at machine zero
   (|t1| ~2e-15, |t3| ~8e-19; the triples residual is 7e-18 even at converged
   t1/t2). A rank-3 probe there compares zero against zero on two of three
   manifolds and passes regardless of whether the kernel is correct. LiH is the
   R4.2 fixture precisely because its t3 is live at ~8e-04.
"""
import unittest

import numpy as np

import ccgen.tests.test_reference_vs_pyscf as T


def _solve_with_matched_integrals(atom, basis, spin=0):
    """Solve, returning the amplitudes AND the integrals that solve actually used."""
    captured = {}
    orig = T._spinorbital_integrals

    def spy(*args, **kwargs):
        captured["r"] = orig(*args, **kwargs)
        return captured["r"]

    T._spinorbital_integrals = spy
    try:
        e_corr, amps, _mf, _no, _nv = T.ccgen_iterate_amps(
            "ccsdt", atom, basis, ["singles", "doubles", "triples"], spin=spin)
    finally:
        T._spinorbital_integrals = orig
    return e_corr, amps, captured["r"]


class IterateAmpsFixedPointTests(unittest.TestCase):
    def _residuals(self, amps, integrals):
        from ccgen.generate import generate_cc_equations
        from ccgen.tests.residual_eval import residual_einsum

        _mf, fock, v, nocc, _nmo, nvir = integrals
        eqs = generate_cc_equations("ccsdt")
        tensors = {"v": v, "f": fock, **amps}
        return {
            m: float(np.max(np.abs(
                sum(residual_einsum(t, nocc, nvir, tensors=tensors)
                    for t in eqs[m]))))
            for m in ("singles", "doubles", "triples")
        }

    def test_lih_is_a_fixed_point_with_live_triples(self):
        """The R4.2 fixture: converged AND t3 actually carries signal."""
        try:
            e_corr, amps, integrals = _solve_with_matched_integrals(
                "Li 0 0 0; H 0 0 1.6", "sto-3g")
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"pyscf not importable: {exc}")

        # matches hand-written C++ RCCSDT (-0.0204594700) to ~2e-09
        self.assertAlmostEqual(e_corr, -0.020459472, places=7)

        for manifold, r in self._residuals(amps, integrals).items():
            self.assertLess(r, 1e-10,
                            f"{manifold} residual is not ~0 at the returned amplitudes")

        # The whole point of choosing LiH over Be: a probe on an inert manifold
        # compares zero against zero and cannot fail.
        self.assertGreater(float(np.max(np.abs(amps["t3"]))), 1e-5,
                           "t3 is inert -- this fixture cannot validate a rank-3 kernel")

    def test_be_triples_are_inert(self):
        """Why Be must NOT be used for a rank-3 probe. Pins the trap itself."""
        try:
            _e, amps, _integrals = _solve_with_matched_integrals(
                "Be 0 0 0", "sto-3g")
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"pyscf not importable: {exc}")

        self.assertLess(float(np.max(np.abs(amps["t3"]))), 1e-12)
        self.assertLess(float(np.max(np.abs(amps["t1"]))), 1e-12)


if __name__ == "__main__":
    unittest.main()
