"""T2: canonical-Fock generation mode.

A canonical HF reference has ``f_ov = f_vo = 0``, so terms carrying an ``f``
factor with an occupied-virtual block vanish. `canonical_fock=True` drops them;
the default keeps the general Fock and is byte-identical to before. This matches
the hand-written GCCSD reference (`src/post_hf/cc/ccsd.cpp`), which uses only
diagonal orbital energies.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations  # noqa: E402


def _has_fov(term):
    for fac in term.factors:
        if fac.name == "f" and len(fac.indices) == 2:
            p, q = fac.indices
            if {p.space, q.space} == {"occ", "vir"}:
                return True
    return False


class CanonicalFockTests(unittest.TestCase):
    def test_default_is_unchanged(self):
        # The load-bearing safety property: default off -> byte-identical.
        for method in ("ccd", "ccsd"):
            base = generate_cc_equations(method)
            default = generate_cc_equations(method, canonical_fock=False)
            for manifold in base:
                with self.subTest(method=method, manifold=manifold):
                    self.assertEqual(
                        [repr(t) for t in base[manifold]],
                        [repr(t) for t in default[manifold]],
                    )

    def test_canonical_drops_only_fov_terms(self):
        for method in ("ccd", "ccsd"):
            base = generate_cc_equations(method)
            canon = generate_cc_equations(method, canonical_fock=True)
            for manifold in base:
                expected = [t for t in base[manifold] if not _has_fov(t)]
                with self.subTest(method=method, manifold=manifold):
                    self.assertEqual(
                        [repr(t) for t in canon[manifold]],
                        [repr(t) for t in expected],
                    )

    def test_no_fov_survives_under_canonical(self):
        for method in ("ccd", "ccsd", "ccsdt"):
            canon = generate_cc_equations(method, canonical_fock=True)
            for manifold, terms in canon.items():
                with self.subTest(method=method, manifold=manifold):
                    self.assertFalse(any(_has_fov(t) for t in terms))

    def test_diagonal_fock_terms_are_kept(self):
        # f_oo / f_vv (the t2*Fae, t2*Fmi eps pieces) must SURVIVE -- the
        # reference keeps them. Only the ov/vo block is Brillouin-zero.
        canon = generate_cc_equations("ccsd", canonical_fock=True)
        f_terms = [t for t in canon["doubles"]
                   if any(fac.name == "f" for fac in t.factors)]
        self.assertTrue(f_terms)  # some f-terms remain
        for t in f_terms:
            for fac in t.factors:
                if fac.name == "f":
                    p, q = fac.indices
                    self.assertEqual(p.space, q.space)  # oo or vv, never ov

    def test_removes_the_fock_driven_bug_class(self):
        # The f*t1*t2 group (the Fock-driven half of the antisymmetry bug) is
        # exactly the f_ov terms, so canonical_fock removes it entirely.
        base = generate_cc_equations("ccsd")["doubles"]
        ft1t2 = [t for t in base
                 if frozenset(f.name for f in t.factors) == frozenset(("f", "t1", "t2"))]
        self.assertTrue(ft1t2)
        self.assertTrue(all(_has_fov(t) for t in ft1t2))
        canon = generate_cc_equations("ccsd", canonical_fock=True)["doubles"]
        self.assertFalse(any(
            frozenset(f.name for f in t.factors) == frozenset(("f", "t1", "t2"))
            for t in canon
        ))


if __name__ == "__main__":
    unittest.main()
