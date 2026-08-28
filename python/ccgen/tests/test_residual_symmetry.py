"""e.2.5.0/e.2.5.1: the numeric fixture must carry real integral symmetry, and the
adapted dressed equation must equal the adapted raw equation numerically.

WHY THIS FILE EXISTS. `random_tensors` used to build `v` with intra-pair antisymmetry
only, missing the bra<->ket exchange `<pq||rs> = <rs||pq>` that real antisymmetrized
integrals satisfy (verified against pyscf on H2/STO-3G: all three residuals ~1e-16).
`_ERI_PERMUTATIONS` in dressing.py folds by that symmetry, so any numeric comparison of
two writings related by the exchange reported a spurious difference -- concretely, it
made "GCC dressed-expansion vs GCC raw" look 170% off on manifolds the symbolic fold
(correctly) calls identical.

That is also the answer to V1.1e's doubles=14: the symbolic term-by-term multiset
comparison is the wrong instrument when both sides may pick among symmetry-equivalent
written forms. The numeric gate below is the right one, and it passes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import _dress_operator_equations, generate_cc_equations  # noqa: E402
from ccgen.optimization.dressed_equation import expand_dressed_term  # noqa: E402
from ccgen.spin import spin_adapt_equations  # noqa: E402
from ccgen.tests.residual_eval import random_tensors, residual_of  # noqa: E402

# (no, nv, seed) -- more than one so a single lucky fixture cannot pass the gate.
CASES = ((2, 3, 0), (3, 4, 11), (4, 5, 3))


class FixtureSymmetryTests(unittest.TestCase):
    """The fixture's `v` must satisfy every symmetry real `<pq||rs>` has."""

    def test_v_carries_all_three_symmetries(self):
        for no, nv, seed in CASES:
            v = random_tensors(no, nv, seed=seed)["v"]
            with self.subTest(no=no, nv=nv, seed=seed):
                self.assertLess(np.abs(v + v.transpose(1, 0, 2, 3)).max(), 1e-12,
                                "bra antisymmetry")
                self.assertLess(np.abs(v + v.transpose(0, 1, 3, 2)).max(), 1e-12,
                                "ket antisymmetry")
                self.assertLess(np.abs(v - v.transpose(2, 3, 0, 1)).max(), 1e-12,
                                "bra<->ket exchange -- the one that was missing")

    def test_f_is_symmetric(self):
        for no, nv, seed in CASES:
            f = random_tensors(no, nv, seed=seed)["f"]
            with self.subTest(no=no, nv=nv, seed=seed):
                self.assertLess(np.abs(f - f.T).max(), 1e-12)


class AdaptedDressedNumericTests(unittest.TestCase):
    """V1.1e's real requirement: the dressed spatial equation reproduces the raw one.

    Replaces the symbolic `{"doubles": 14}` count, which measured written form rather
    than algebra.
    """

    @classmethod
    def setUpClass(cls):
        eqs = generate_cc_equations("ccsd", engine="diagram", canonical_fock=True)
        dressed, _ = _dress_operator_equations(eqs)
        expanded = {m: [x for t in ts for x in expand_dressed_term(t)]
                    for m, ts in dressed.items()}
        cls.raw = eqs
        cls.expanded = expanded
        cls.adapted_dressed = spin_adapt_equations(expanded)
        cls.adapted_raw = spin_adapt_equations(eqs)

    def _compare(self, left, right):
        for no, nv, seed in CASES:
            tensors = random_tensors(no, nv, seed=seed)
            for manifold in ("energy", "singles", "doubles"):
                a = residual_of(left[manifold], no, nv, tensors=tensors)
                b = residual_of(right[manifold], no, nv, tensors=tensors)
                scale = max(float(np.abs(b).max()), 1.0)
                with self.subTest(no=no, nv=nv, seed=seed, manifold=manifold):
                    self.assertLess(float(np.abs(a - b).max()) / scale, 1e-12)

    def test_gcc_expansion_matches_raw(self):
        """Precondition: the dressed assembly is exact before adaptation."""
        self._compare(self.expanded, self.raw)

    def test_adapted_dressed_matches_adapted_raw(self):
        """The V1.1e gate. Measured ~1e-14 relative on all three manifolds."""
        self._compare(self.adapted_dressed, self.adapted_raw)


if __name__ == "__main__":
    unittest.main()
