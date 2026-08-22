"""R1: a dressed intermediate must equal its primitive-expansion oracle, numerically.

The defect this gate exists for: `tau = t2 + t1*t1` is a SUM of two objects that
spin-adapt with different weights, but `spin_adapt_equations` treats it as one opaque
rank-4 amplitude and applies a single spin-summation factor to the whole thing:

    GCC term:  1/4 tau(e,f,i,j) * v(m,n,e,f)      [Wmnij definition, term 3]

    oracle (expand tau, THEN adapt)      spec (adapt with tau opaque)
      1/4 t2*v      -> 1/2 t2*v            1/4 tau*v  ->  1/2 tau*v
      1/2 t1*t1*v   -> 1/2 t1*t1*v

The doubling is correct for the `t2` component -- its spin cases sum to give that
factor -- and wrong for `t1*t1`, whose coefficient is already 1/2 in both paths. So the
`t1*t1` half of every `tau` is doubled, and every operator that references `tau` is
built wrong.

Why a NUMERIC gate rather than a symbolic one:

Four earlier fix attempts for this defect were index/layout theories. Each passed a
structural gate (`test_intermediate_layout_agreement`) and made the dressed Be CCSDTQ
energy WORSE -- -0.0247 -> -0.0145 -> -0.0119 against an exact -0.0518. Layout
self-consistency is necessary but does not constrain values. This gate constrains
values, and it is the check that identified the real cause.

Why `no != nv` matters, and why it is 3 vs 4 here:

On Be/STO-3G the production validation system has `nv == no == 4`, so an intermediate
read in the wrong slot layout stays IN BOUNDS and silently returns the wrong element.
At `no=3, nv=4` the same read raises IndexError. The asymmetric shape is what makes
this class of defect loud, and it is why the harness found in seconds what a 3-minute
Be run could only report as "the number is wrong".

Scope: this file gates the intermediates' VALUES. `test_intermediate_layout_agreement`
gates their slot layout. Both are needed -- the layout defect is real and separate (see
docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md), it is simply not what costs the 52 %.
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
except ImportError:  # pragma: no cover - numpy is a test-only dependency here
    np = None

from ccgen.generate import (  # noqa: E402
    generate_cc_equations,
    _dress_operator_equations,
)
from ccgen.optimization.dressed_equation import expand_dressed_term  # noqa: E402
from ccgen.spin import (  # noqa: E402
    adapt_intermediate_spec,
    intermediate_template,
    spin_adapt_equations,
)

# Deliberately asymmetric: nv == no is what makes the sibling layout defect silent.
NO, NV = 3, 4
NORB = NO + NV
OFF = {"o": 0, "v": NO}
DIM = {"o": NO, "v": NV}
AMPLITUDE = re.compile(r"t(\d+)$")
DRESSED_ORDER = ("tau", "tau_c", "Wmnij", "Wabef", "Wmbej")
TOL = 1e-10


def _space(index) -> str:
    return "o" if index.space == "occ" else "v"


def _fixture():
    """Deterministic random integrals/amplitudes with the real ERI index symmetries."""
    rng = np.random.default_rng(20260815)
    eri = rng.random((NORB,) * 4)
    eri = eri + eri.transpose(2, 3, 0, 1)      # <pq|rs> = <rs|pq>
    eri = eri + eri.transpose(1, 0, 3, 2)      # <pq|rs> = <qp|sr>
    amps = {1: rng.random((NO, NV)), 2: rng.random((NO, NO, NV, NV))}
    eps = {"o": np.arange(1.0, NO + 1.0), "v": np.arange(10.0, 10.0 + NV)}
    return eri, amps, eps


def _evaluate(term, out_indices, eri, amps, eps, intermediates):
    """Contract one AlgebraTerm into an array over `out_indices`, by explicit loops.

    Explicit loops rather than einsum on purpose: hand-written einsum strings were tried
    while diagnosing this and got a transpose wrong, which is exactly the class of error
    the gate is meant to detect. Looping over the term's own index objects cannot drift
    from what the term says.

    `intermediates` maps name -> (array, stored_space_signature). A usage site is
    reordered into the stored layout by space; that is not a claim about which
    permutation is physically right (see the scope doc), only what is needed to read the
    array at all when nv != no.
    """
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


class DressedIntermediateOracleTests(unittest.TestCase):
    """Each intermediate, built from its adapted spec, vs its primitive expansion."""

    @classmethod
    def setUpClass(cls):
        if np is None:
            raise unittest.SkipTest("numpy not available")
        cls.eri, cls.amps, cls.eps = _fixture()
        eqs = generate_cc_equations("ccsd", canonical_fock=True)
        cls.dressed, specs = _dress_operator_equations(eqs)
        cls.raw = eqs
        cls.gcc = {s.name: s for s in specs}
        cls.adapted = {s.name: adapt_intermediate_spec(s) for s in specs}
        cls.stored = {n: [_space(i) for i in s.indices]
                      for n, s in cls.adapted.items()}

    def _spec_value(self, name, built):
        spec = self.adapted[name]
        return sum(_evaluate(t, list(spec.indices), self.eri, self.amps,
                             self.eps, built)
                   for t in spec.definition_terms)

    def _oracle_value(self, name):
        """Expand the GCC definition to primitives FIRST, then adapt.

        This is the ground truth precisely because it never lets a pseudo-amplitude
        cross the adapter as an opaque object: after expansion only t1/t2/v/f remain,
        each of which the adapter weights correctly.
        """
        gcc = self.gcc[name]
        primitives = [p for t in gcc.definition_terms
                      for p in expand_dressed_term(t)]
        adapted = spin_adapt_equations(
            {name: primitives},
            templates={name: intermediate_template(gcc)})[name]
        target = list(self.adapted[name].indices)
        return sum(_evaluate(t, target, self.eri, self.amps, self.eps, {})
                   for t in adapted)

    def _build_all(self):
        built = {}
        for name in DRESSED_ORDER:
            built[name] = (self._spec_value(name, built), self.stored[name])
        return built

    def test_tau_and_wmbej_already_agree(self):
        """The control. These reference no pseudo-amplitude, so they must be exact
        both before and after the fix -- if one of them ever fails, the cause is
        something other than the tau-opacity defect."""
        built = self._build_all()
        for name in ("tau", "tau_c", "Wmbej"):
            with self.subTest(operator=name):
                spec, _ = built[name]
                self.assertLess(
                    np.linalg.norm(spec - self._oracle_value(name)), TOL,
                    f"{name} should match its oracle exactly")

        # XFAIL: dressing and spin adaptation do not compose -- recognition
        # subtracts what an operator absorbs against a term set that adaptation
        # then changes. Measured Be/STO-3G CCSDTQ E_corr = -0.0247182895 vs an
        # exact -0.0517746319 (52 % short). The dressed route is RETIRED
        # (vault/Status/Completion.md, docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md),
        # so this is not expected to pass; making it pass would resume an
        # abandoned route. Kept rather than deleted because the retirement note
        # keeps this file deliberately -- it is a numeric instrument that catches
        # VALUE defects structural gates cannot. An unexpected PASS here means
        # the composition was genuinely fixed and the retirement should be
        # revisited.
    @unittest.expectedFailure
    def test_every_intermediate_matches_its_oracle(self):
        """The gate. Pre-fix this FAILS on Wmnij and Wabef -- the two operators whose
        definitions reference `tau`."""
        built = self._build_all()
        bad = []
        for name in DRESSED_ORDER:
            spec, _ = built[name]
            oracle = self._oracle_value(name)
            diff = float(np.linalg.norm(spec - oracle))
            if diff > TOL:
                bad.append(f"  {name}: ‖spec‖={np.linalg.norm(spec):.4f} "
                           f"‖oracle‖={np.linalg.norm(oracle):.4f} ‖diff‖={diff:.4e}")
        self.assertFalse(
            bad,
            "dressed intermediates disagree with their primitive expansion -- a "
            "pseudo-amplitude (tau) is being spin-adapted as one opaque amplitude, so "
            "its t1*t1 component receives the t2 spin-summation factor:\n"
            + "\n".join(bad))

    def test_only_tau_referencing_operators_are_affected(self):
        """Pins the CAUSE, not just the symptom: the failing set must be exactly the
        set of operators referencing a pseudo-amplitude. If that correspondence ever
        breaks, the diagnosis in the scope doc is wrong and should be re-derived."""
        built = self._build_all()
        failing = {n for n in DRESSED_ORDER
                   if np.linalg.norm(built[n][0] - self._oracle_value(n)) > TOL}
        referencing = {
            n for n in DRESSED_ORDER
            if any(f.name in ("tau", "tau_c")
                   for t in self.adapted[n].definition_terms for f in t.factors)
        }
        self.assertEqual(
            failing, referencing,
            f"operators failing the oracle {sorted(failing)} should be exactly those "
            f"referencing a pseudo-amplitude {sorted(referencing)}")


class DressedResidualOracleTests(unittest.TestCase):
    """The dressed residual must reproduce the undressed one, numerically."""

    @classmethod
    def setUpClass(cls):
        if np is None:
            raise unittest.SkipTest("numpy not available")
        cls.eri, cls.amps, cls.eps = _fixture()
        eqs = generate_cc_equations("ccsd", canonical_fock=True)
        dressed, specs = _dress_operator_equations(eqs)
        cls.raw_adapted = spin_adapt_equations(eqs)
        cls.dressed_adapted = spin_adapt_equations(dressed)
        cls.adapted = {s.name: adapt_intermediate_spec(s) for s in specs}
        cls.stored = {n: [_space(i) for i in s.indices]
                      for n, s in cls.adapted.items()}

        # XFAIL: dressing and spin adaptation do not compose -- recognition
        # subtracts what an operator absorbs against a term set that adaptation
        # then changes. Measured Be/STO-3G CCSDTQ E_corr = -0.0247182895 vs an
        # exact -0.0517746319 (52 % short). The dressed route is RETIRED
        # (vault/Status/Completion.md, docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md),
        # so this is not expected to pass; making it pass would resume an
        # abandoned route. Kept rather than deleted because the retirement note
        # keeps this file deliberately -- it is a numeric instrument that catches
        # VALUE defects structural gates cannot. An unexpected PASS here means
        # the composition was genuinely fixed and the retirement should be
        # revisited.
    @unittest.expectedFailure
    def test_dressed_doubles_residual_equals_undressed(self):
        """End-to-end value check. Dressing is a refactorization, so the two residuals
        are the same function; any difference is a defect."""
        built = {}
        for name in DRESSED_ORDER:
            spec = self.adapted[name]
            built[name] = (sum(_evaluate(t, list(spec.indices), self.eri, self.amps,
                                         self.eps, built)
                               for t in spec.definition_terms),
                           self.stored[name])
        out_ix = list(self.raw_adapted["doubles"][0].free_indices)
        raw = sum(_evaluate(t, out_ix, self.eri, self.amps, self.eps, {})
                  for t in self.raw_adapted["doubles"])
        dressed = sum(_evaluate(t, out_ix, self.eri, self.amps, self.eps, built)
                      for t in self.dressed_adapted["doubles"])
        self.assertLess(
            float(np.linalg.norm(dressed - raw)), TOL * max(1.0, np.linalg.norm(raw)),
            f"dressed doubles residual ‖{np.linalg.norm(dressed):.4f}‖ != undressed "
            f"‖{np.linalg.norm(raw):.4f}‖; ‖diff‖={np.linalg.norm(dressed - raw):.4e}")


if __name__ == "__main__":
    unittest.main()
