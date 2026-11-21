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
from unittest import mock

import numpy as np

from ccgen.tests import test_merged_call_sites as M
from ccgen.tests.test_emitted_builder_matches_spec import _spatial_tensors


def _emit(dressing: str, method: str = "ccsdt", fold: bool = True) -> str:
    """Spin-adapted emit. ``fold=False`` swaps the triples symmetry group for
    the identity, so the kernel computes R directly: the reference arm for the
    fold, sharing nothing with it."""
    from ccgen.generate import print_cpp_planck
    from ccgen.optimization import permutation
    group = permutation.AAB_TRIPLES_GROUP if fold else (({}, 1),)
    with mock.patch.object(permutation, "AAB_TRIPLES_GROUP", group):
        return print_cpp_planck(method, dressing=dressing, spin_adapt=True,
                                force_arbitrary=True)


def aab_tensors(no, nv):
    """The physical fixture for the spin-adapted manifold. t3 is the aabaab
    block, antisymmetric under a<->b and i<->j (an all-six-pair-permutation t3,
    as `_physical_tensors` builds, is not this block, and the residual is not
    antisymmetric under it). v carries every real-orbital ERI symmetry."""
    t = _spatial_tensors(no, nv)
    v = t["v"]
    v = v + v.transpose(2, 1, 0, 3)
    v = v + v.transpose(1, 0, 3, 2)
    v = v + v.transpose(2, 3, 0, 1)
    t["v"] = v
    sl = {"o": slice(0, no), "v": slice(no, no + nv)}
    t["_blocks"] = {k: v[tuple(sl[c] for c in k)] for k in t["_blocks"]}
    t["t2"] = t["t2"] + np.transpose(t["t2"], (1, 0, 3, 2))
    t3 = t["t3"]                                   # (a, b, c, i, j, k)
    t3 = t3 - np.transpose(t3, (1, 0, 2, 3, 4, 5))
    t3 = t3 - np.transpose(t3, (0, 1, 2, 4, 3, 5))
    t["t3"] = t3
    return t


def _residuals(src):
    with mock.patch.object(M, "_physical_tensors", aab_tensors):
        return M._residuals(src)


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
        """Derived and folded against undressed and unfolded: neither the
        operators nor the fold are shared by the two arms."""
        try:
            derived = _residuals(_emit("derived"))
            plain = _residuals(_emit("none", fold=False))
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


    def test_fold_cuts_triples_terms(self):
        """The fold is live: the folded kernel emits far fewer triples nests and
        carries the symmetrize pass; the reference arm carries neither."""
        try:
            folded, plain = _emit("derived"), _emit("derived", fold=False)
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"ccgen deps unavailable: {exc}")

        def triples_nests(src):
            return sum(b.count("acc +=") for b in _residual_bodies(src, "ccsdt")
                       if "triples" in b.split("(")[0])

        self.assertIn("// symmetrize ijkabc:", folded)
        self.assertNotIn("// symmetrize", plain)
        self.assertLess(triples_nests(folded), 0.5 * triples_nests(plain))


if __name__ == "__main__":
    unittest.main()
