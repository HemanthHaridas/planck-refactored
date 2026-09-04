"""V1.1e.3: per-operator localization of the dressed-equation gate.

The V1.1e gate (`test_residual_symmetry`) dresses the full seeded operator family at once,
so a regression reports "doubles is wrong" without saying which operator's definition
broke. This runs the same numeric comparison with **one operator dressed at a time**, so a
future failure names `Wmnij` / `Wabef` / `Wmbej` directly.

NUMERIC, NOT SYMBOLIC. The originally-scoped e.3 gate was "0 mismatches on every manifold,
per-operator and combined" -- i.e. the symbolic term-multiset count. e.2.5 showed that
count measures written form rather than algebra: it sat at 14 while the algebra was exact,
because a multiset cannot tell "different algebra" from "same algebra, different
symmetry-equivalent writing". So this gate compares residual VALUES on symmetry-correct
tensors, the same instrument that resolved V1.1e.

WHY ONLY THE THREE W OPERATORS ARE DRESSED. `seeded_operators()` returns six (Fme, Fae,
Fmi, Wmnij, Wabef, Wmbej), but under `canonical_fock=True` -- the only mode Planck feeds CC
(`cc_canonical_fock_only`) -- the F operators' `f_ov`-bearing definition terms are
Brillouin-zero and drop, so Fme collapses to its `t1*oovv` piece and Fae/Fmi lose their
`f_ov*t1` corrections (`operator_to_intermediate_spec` docstring). Measured: dressing the
full family references only `Wabef`, `Wmbej`, `Wmnij`, `tau`, `tau_c` -- no F operator
appears, with the full family or alone. `test_f_operators_are_inert_under_canonical_fock`
pins that as intended behavior rather than leaving it to look like a gap.
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
from ccgen.optimization.dressing import seeded_operators  # noqa: E402
from ccgen.spin import spin_adapt_equations  # noqa: E402
from ccgen.tests.residual_eval import random_tensors, residual_of  # noqa: E402

MANIFOLDS = ("energy", "singles", "doubles")
CASES = ((2, 3, 0), (3, 4, 11))
PRIMITIVES = frozenset({"t1", "t2", "v", "f"})

# Dressed under canonical Fock; the F operators are inert (see module docstring).
DRESSING_OPERATORS = ("Wmnij", "Wabef", "Wmbej")


def _intermediate_names(equations):
    return {f.name
            for terms in equations.values()
            for term in terms
            for f in term.factors
            if f.name not in PRIMITIVES}


class PerOperatorNumericTests(unittest.TestCase):
    """Each operator's dressed form must reproduce the raw residual on its own."""

    @classmethod
    def setUpClass(cls):
        cls.raw = generate_cc_equations("ccsd", engine="diagram", canonical_fock=True)
        cls.adapted_raw = spin_adapt_equations(cls.raw)
        cls.operators = {op.name: op for op in seeded_operators()}

    def _adapted_dressed(self, operators):
        dressed, _ = _dress_operator_equations(self.raw, operators=operators)
        expanded = {m: [x for t in ts for x in expand_dressed_term(t)]
                    for m, ts in dressed.items()}
        return spin_adapt_equations(expanded)

    def _assert_matches_raw(self, adapted, label):
        for no, nv, seed in CASES:
            tensors = random_tensors(no, nv, seed=seed)
            for manifold in MANIFOLDS:
                a = residual_of(adapted[manifold], no, nv, tensors=tensors)
                b = residual_of(self.adapted_raw[manifold], no, nv, tensors=tensors)
                scale = max(float(np.abs(b).max()), 1.0)
                with self.subTest(operator=label, manifold=manifold, no=no, seed=seed):
                    self.assertLess(float(np.abs(a - b).max()) / scale, 1e-12)

    def test_each_operator_alone(self):
        """The localization itself: one operator dressed, the rest left bare."""
        for name in DRESSING_OPERATORS:
            with self.subTest(operator=name):
                adapted = self._adapted_dressed([self.operators[name]])
                self._assert_matches_raw(adapted, name)

    def test_each_operator_alone_actually_dresses(self):
        """Guards against the gate passing vacuously: if an operator silently stopped
        being recognized, `test_each_operator_alone` would still pass (nothing dressed
        == raw), so assert the operator is genuinely referenced."""
        for name in DRESSING_OPERATORS:
            with self.subTest(operator=name):
                dressed, _ = _dress_operator_equations(
                    self.raw, operators=[self.operators[name]])
                self.assertIn(name, _intermediate_names(dressed))

    def test_full_family_combined(self):
        """The combined case, for parity with the per-operator runs."""
        self._assert_matches_raw(self._adapted_dressed(None), "full-family")

    def test_f_operators_are_inert_under_canonical_fock(self):
        """Documents measured behavior so it is not mistaken for a missing feature:
        under a canonical Fock no F operator is referenced, alone or with the family."""
        for name in ("Fme", "Fae", "Fmi"):
            with self.subTest(operator=name):
                dressed, _ = _dress_operator_equations(
                    self.raw, operators=[self.operators[name]])
                self.assertNotIn(name, _intermediate_names(dressed))

        self.assertEqual(
            _intermediate_names(_dress_operator_equations(self.raw)[0]),
            {"Wmnij", "Wabef", "Wmbej", "tau", "tau_c"},
        )


class OperatorSelectionTests(unittest.TestCase):
    """The `operators` parameter must not disturb the default path."""

    @classmethod
    def setUpClass(cls):
        cls.raw = generate_cc_equations("ccsd", engine="diagram", canonical_fock=True)

    def test_none_is_the_full_seeded_family(self):
        default, default_specs = _dress_operator_equations(self.raw)
        explicit, explicit_specs = _dress_operator_equations(
            self.raw, operators=seeded_operators())
        self.assertEqual({m: len(t) for m, t in default.items()},
                         {m: len(t) for m, t in explicit.items()})
        self.assertEqual([s.name for s in default_specs],
                         [s.name for s in explicit_specs])

    def test_empty_selection_leaves_the_equation_bare(self):
        dressed, specs = _dress_operator_equations(self.raw, operators=[])
        self.assertEqual(_intermediate_names(dressed), set())
        self.assertEqual(specs, [])


if __name__ == "__main__":
    unittest.main()
