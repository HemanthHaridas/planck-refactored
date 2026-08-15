"""Is dressed-operator recognition value-preserving on SPATIAL (spin-adapted) terms?

Dressing rewrites a residual by recognizing sub-expressions as CC intermediates
(`Wmnij`/`Wabef`/`Wmbej` + `tau`/`tau_c`). That rewrite must not change what the
equation computes. This gate checks that **numerically**, on spatial terms.

Why not reuse `verify_dressed_equation`:

    It compares ERI-canonical multisets via `_eri_canonical`, whose own docstring
    says it "folds v's bra<->ket exchange symmetry (which _canonical_key alone does
    not, since ccgen's v carries only intra-pair antisymmetry)". Those are properties
    of the ANTISYMMETRIZED spin-orbital integral <pq||rs>. A spatial ERI <pq|rs> has
    neither, so on spin-adapted input that canonicalization equates terms that are not
    numerically equal.

    Measured: on spin-adapted `ccsd` doubles it reports **0 mismatched keys** while the
    two manifolds evaluate to 983.79 and 1412.22 -- a 9.58e+02 discrepancy it cannot
    see. It is sound on GCC input and unsound on spatial input; this file covers the
    case it cannot.

A second tell that the symbolic check is not closing: recognition takes 113 adapted
terms to 92, and expanding those 92 back gives **116** -- the round trip does not
return to 113, yet the symbolic diff is empty.

Evaluation notes:

  * `no=3, nv=4` on purpose. With `nv == no` a wrongly-ordered read stays in bounds and
    returns a wrong number silently; asymmetric extents make such a bug raise instead.
  * Explicit loops rather than einsum, so the contraction cannot drift from what the
    term object says. (Hand-written einsum was tried during diagnosis and got a
    transpose wrong -- exactly the error class this gate exists to catch.)
  * Intermediates are evaluated from their own definitions in dependency order, then
    read at usage sites in their stored layout.
"""

from __future__ import annotations

import itertools
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from ccgen.generate import (  # noqa: E402
    generate_cc_equations,
    _dress_operator_equations,
)
from ccgen.spin import spin_adapt_equations  # noqa: E402

NO, NV = 3, 4
NORB = NO + NV
OFF = {"o": 0, "v": NO}
DIM = {"o": NO, "v": NV}
AMPLITUDE = re.compile(r"t(\d+)$")
TOL = 1e-9


def _space(index) -> str:
    return "o" if index.space == "occ" else "v"


def _fixture():
    rng = np.random.default_rng(20260815)
    eri = rng.random((NORB,) * 4)
    # The symmetries a SPATIAL <pq|rs> genuinely has, and only those.
    eri = eri + eri.transpose(2, 3, 0, 1)
    eri = eri + eri.transpose(1, 0, 3, 2)
    amps = {1: rng.random((NO, NV)), 2: rng.random((NO, NO, NV, NV))}
    eps = {"o": np.arange(1.0, NO + 1.0), "v": np.arange(10.0, 10.0 + NV)}
    return eri, amps, eps


def _evaluate(term, out_indices, eri, amps, eps, intermediates):
    free = list(out_indices)
    summed = list(term.summed_indices)
    out = np.zeros(tuple(DIM[_space(i)] for i in free))
    for combo in itertools.product(*[range(DIM[_space(i)]) for i in free + summed]):
        val = {ix.name: c for ix, c in zip(free + summed, combo)}
        prod = float(term.coeff)
        for factor in term.factors:
            ids = [val[i.name] for i in factor.indices]
            spaces = [_space(i) for i in factor.indices]
            if factor.name in intermediates:
                array, stored = intermediates[factor.name]
                remaining = list(range(len(ids)))
                order = []
                for want in stored:
                    pick = next(k for k in remaining if spaces[k] == want)
                    remaining.remove(pick)
                    order.append(ids[pick])
                prod *= array[tuple(order)]
            elif factor.name == "v":
                prod *= eri[tuple(OFF[s] + k for s, k in zip(spaces, ids))]
            elif AMPLITUDE.fullmatch(factor.name):
                rank = int(factor.name[1:])
                occ = [k for k, s in zip(ids, spaces) if s == "o"]
                vir = [k for k, s in zip(ids, spaces) if s == "v"]
                prod *= amps[rank][tuple(occ + vir)]
            elif factor.name == "f":
                left, right = factor.indices
                same = (_space(left) == _space(right)
                        and val[left.name] == val[right.name])
                prod *= eps[_space(left)][val[left.name]] if same else 0.0
            else:
                raise AssertionError(f"unexpected factor {factor.name!r}")
        out[tuple(val[i.name] for i in free)] += prod
    return out


class SpatialRecognitionEquivalenceTests(unittest.TestCase):
    """Recognition must preserve the residual's value on spin-adapted terms."""

    @classmethod
    def setUpClass(cls):
        if np is None:
            raise unittest.SkipTest("numpy not available")
        cls.eri, cls.amps, cls.eps = _fixture()
        cls.raw = generate_cc_equations("ccsd", canonical_fock=True)
        cls.adapted = spin_adapt_equations(cls.raw)
        cls.dressed, cls.specs = _dress_operator_equations(cls.adapted)
        cls.stored = {s.name: [_space(i) for i in s.indices] for s in cls.specs}

    def _intermediates(self):
        """Evaluate every spec from its own definition, in dependency order."""
        built = {}
        for spec in self.specs:                    # specs are dependency-ordered
            built[spec.name] = (
                sum(_evaluate(t, list(spec.indices), self.eri, self.amps,
                              self.eps, built)
                    for t in spec.definition_terms),
                self.stored[spec.name])
        return built

    def _value(self, terms, out_indices, intermediates):
        return sum(_evaluate(t, out_indices, self.eri, self.amps, self.eps,
                             intermediates) for t in terms)

    def test_recognition_preserves_the_doubles_residual(self):
        """The gate. Recognition replaces N terms with M; the value must not move."""
        out_ix = list(self.adapted["doubles"][0].free_indices)
        before = self._value(self.adapted["doubles"], out_ix, {})
        after = self._value(self.dressed["doubles"], out_ix, self._intermediates())
        diff = float(np.linalg.norm(after - before))
        self.assertLess(
            diff, TOL * max(1.0, float(np.linalg.norm(before))),
            f"dressed-operator recognition changed the doubles residual on spatial "
            f"terms: ‖before‖={np.linalg.norm(before):.4f} "
            f"({len(self.adapted['doubles'])} terms) -> ‖after‖={np.linalg.norm(after):.4f} "
            f"({len(self.dressed['doubles'])} terms), ‖diff‖={diff:.4e}. "
            "Recognition must be a pure rewrite.")

    def test_operator_free_terms_are_untouched(self):
        """Control: terms recognition did not rewrite must evaluate identically.

        Isolates the defect to the rewritten terms. If this fails too, the cause is
        in evaluation or adaptation rather than in recognition.
        """
        names = {s.name for s in self.specs}
        out_ix = list(self.adapted["doubles"][0].free_indices)
        untouched = [t for t in self.dressed["doubles"]
                     if not any(f.name in names for f in t.factors)]
        # every untouched dressed term should appear verbatim in the adapted input
        adapted_keys = {self._key(t) for t in self.adapted["doubles"]}
        missing = [self._key(t) for t in untouched if self._key(t) not in adapted_keys]
        self.assertFalse(
            missing[:5],
            f"{len(missing)} operator-free dressed term(s) are not present unchanged "
            "in the adapted input -- recognition altered a term it did not rewrite")

    @staticmethod
    def _key(term):
        return (term.coeff,
                tuple((f.name, tuple(i.name for i in f.indices))
                      for f in term.factors))

    def test_symbolic_verifier_is_not_a_substitute(self):
        """Pins WHY this file exists: the symbolic check passes on the same input.

        `_eri_canonical` folds `v`'s bra<->ket exchange symmetry, which the
        antisymmetrized spin-orbital integral has and a spatial one does not. If this
        test ever fails -- i.e. the symbolic check starts reporting the mismatch --
        then it has been made spatial-aware and this file's rationale should be
        revisited.
        """
        from ccgen.optimization.dressed_equation import verify_dressed_equation
        ok, diff = verify_dressed_equation(self.dressed["doubles"],
                                           self.adapted["doubles"])
        self.assertTrue(
            ok and not diff,
            "the symbolic verifier now reports a mismatch on spatial input; if it has "
            "been made spatial-aware, re-evaluate whether this numeric gate is still "
            "the right check")


if __name__ == "__main__":
    unittest.main()
