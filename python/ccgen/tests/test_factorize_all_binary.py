"""Every derived-dressed residual term is a binary contraction, and the
factorization does not change what the residual computes.

The derived route used to inline any child matching a seeded CCSD operator
(Wmbej, Wmnij, Fae, Fmi), because it emits no builders for those. That left
68 spin-adapted ccsdt triples terms ternary -- t2.t3.v at o^5v^5, ~74 % of
modelled FLOPs -- plus 33 doubles and 11 singles
(docs/CCGEN_RCCSDT_COST_SCOPE.md step 1). The census test fails if any comes
back. The value test compares against the UNDRESSED spin-adapted emit, which
shares no operator with the derived one, so a wrong hoisted operator cannot
agree with it by construction.
"""
from __future__ import annotations

import re
import unittest

import numpy as np

from ccgen.tests.test_merged_call_sites import _residuals


def _emit(dressing: str, method: str = "ccsdt") -> str:
    from ccgen.generate import print_cpp_planck
    return print_cpp_planck(method, dressing=dressing, spin_adapt=True,
                            force_arbitrary=True)


def _residual_bodies(src: str, method: str) -> list[str]:
    """Bodies of the residual functions and their _partN chunks (not builders)."""
    out = []
    for m in re.finditer(
            r'(?:static void|Tensor\dD|TensorND) (compute_%s_\w+_residual(?:_part\d+)?)\('
            r'.*?\n\}\n' % method, src, re.S):
        out.append(m.group(0))
    return out


def _factor_count(expr: str) -> int:
    # Tensor reads are `name(` or `name({`; coefficients are bare numbers.
    return len(re.findall(r'[A-Za-z_][A-Za-z0-9_.]*\s*\(', expr))


class FactorizeAllBinaryTests(unittest.TestCase):
    def test_no_nary_residual_terms(self):
        try:
            src = _emit("derived")
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"ccgen deps unavailable: {exc}")
        bodies = _residual_bodies(src, "ccsdt")
        self.assertTrue(bodies, "no residual functions found")
        bad = [e.strip() for b in bodies
               for e in re.findall(r'acc\s*\+=\s*([^;]*);', b)
               if _factor_count(e) >= 3]
        self.assertEqual(bad, [], f"{len(bad)} n-ary residual terms, e.g. {bad[:3]}")

    def test_derived_residual_matches_undressed(self):
        try:
            derived = _residuals(_emit("derived"))
            plain = _residuals(_emit("none"))
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"ccgen deps unavailable: {exc}")
        self.assertEqual(sorted(derived), ["doubles", "singles", "triples"])
        self.assertEqual(sorted(derived), sorted(plain))
        for tag in derived:
            a, b = derived[tag], plain[tag]
            scale = max(float(np.abs(b).max()), 1.0)
            self.assertGreater(float(np.abs(b).max()), 1e-8, f"{tag} is zero")
            d = float(np.abs(a - b).max())
            self.assertLess(d, 1e-12 * scale,
                            f"ccsdt {tag}: derived vs undressed rel={d/scale:.2e}")


if __name__ == "__main__":
    unittest.main()
