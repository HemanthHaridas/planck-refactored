"""V1.0 gate: the slot-ordering contract at the dress/adapt boundary.

`spin_adapt_equations` assigns an external spin block by slot POSITION, and
`_line_pairs` pairs slot k with slot k+n. The default template
(`_residual_template`) reorders free indices virtuals-first, which is the
convention the C++ runtime's `rank_dims` depends on -- correct for residual
targets, WRONG for a dressed intermediate whose own slot order differs.

`Wmbej` is the case that forces this. Its `ovvo` slots are [m,b,e,j] with physical
lines m-e and b-j; virtuals-first reorders them to [b,e,m,j], so the external block
is assigned as b=a, e=b, m=a, j=b. That block is spin-valid *on the reordered
output*, but it is applied to FACTORS carrying the operator's real pairing: the bare
integral `v(m,b,e,j)` then gets tag `aabb`, whose m-e line has m=a/e=b, and is
rejected. Every spin case of every term dies the same way, so the operator adapts to
ZERO terms -- silently, since dropping a forbidden block is the normal discard path.
Same silent-wrong-answer class as the R3.1.2 bridge and B5 ERI-convention defects.

These tests pin the fix (`intermediate_template` + the `templates` override) and
the guard that makes a zero adaptation loud instead of silent.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.optimization.dressing import (  # noqa: E402
    operator_to_intermediate_spec,
    seeded_operators,
)
from ccgen.spin import (  # noqa: E402
    _residual_template,
    adapt_intermediate_spec,
    intermediate_template,
    spin_adapt_equations,
)

# The operators the assembled dressed CCSD residual actually references (V1.1
# Finding A). Fme/Fae/Fmi are recognized but fold away under canonical Fock, so
# they are out of V1.1's scope -- adapting them would be dead code.
REFERENCED = ("Wmnij", "Wabef", "Wmbej")

# Adapted definition-term counts for the referenced operators (V1.1 Finding D).
# Wmbej's 5 -> 8 growth is the adapter splitting terms across spin cases.
ADAPTED_TERM_COUNTS = {"Wmnij": 4, "Wabef": 4, "Wmbej": 8}


def _specs():
    """The six seeded CCSD operators as canonical-Fock IntermediateSpecs."""
    return [operator_to_intermediate_spec(op, canonical_fock=True)
            for op in seeded_operators()]


def _spec(name):
    return next(s for s in _specs() if s.name == name)


class DressedIntermediateAdaptationTests(unittest.TestCase):
    """Every seeded operator must adapt to a non-empty spatial term list when
    adapted on its OWN slot order."""

    def test_all_seeded_operators_adapt_nonempty(self):
        for spec in _specs():
            with self.subTest(operator=spec.name):
                got = spin_adapt_equations(
                    {spec.name: list(spec.definition_terms)},
                    templates={spec.name: intermediate_template(spec)},
                )
                self.assertIn(spec.name, got)
                self.assertGreater(
                    len(got[spec.name]), 0,
                    f"{spec.name} ({spec.index_space_sig}) adapted to zero terms")

    def test_wmbej_is_the_regression_case(self):
        """Wmbej specifically: zero on the virtuals-first default, non-zero on its
        own slot order. This is the measured defect V1.0 fixes -- if the default
        ever stops zeroing it, the override may no longer be load-bearing."""
        spec = next(s for s in _specs() if s.name == "Wmbej")
        self.assertEqual(spec.index_space_sig, "ovvo")

        # the operator's own order preserves the m-e / b-j line pairing
        own = spin_adapt_equations(
            {spec.name: list(spec.definition_terms)},
            templates={spec.name: intermediate_template(spec)},
        )
        self.assertGreater(len(own[spec.name]), 0)

        # the virtuals-first default reorders [m,b,e,j] -> [b,e,m,j] and zeroes it,
        # which the V1.0 guard now raises on rather than returning silently
        default_tpl = _residual_template(spec.name, list(spec.definition_terms))
        self.assertNotEqual(
            [i.name for i in default_tpl.indices],
            [i.name for i in spec.indices],
            "Wmbej's virtuals-first template no longer differs from its own order")
        with self.assertRaises(ValueError) as ctx:
            spin_adapt_equations({spec.name: list(spec.definition_terms)})
        self.assertIn("ZERO", str(ctx.exception))

    def test_representative_block_conserves_spin_on_own_order(self):
        """The contract behind the fix, stated as the adapter actually enforces it:
        on a spec's own slot order the representative external block must be
        spin-CONSERVING along every line (slot k with slot k+n) -- exactly what
        `block_exists` checks and what decides whether any term survives.

        Nothing is asserted about the two ends' index SPACES: a line can join an occ
        to a vir (`Wmbej` `ovvo`: m-e, b-j) or two of a kind (`Wabef` `vvvv`: a-e,
        b-f). What must hold is that the pairing the adapter uses is the operator's
        real one, which is what the own-order template guarantees and what the
        virtuals-first reorder destroys for `Wmbej`."""
        from ccgen.spin import _representative_block_for_sector, block_exists

        for spec in _specs():
            n = len(spec.indices) // 2
            tpl = intermediate_template(spec)
            block = _representative_block_for_sector(tpl, -(-n // 2))
            label = {i.name: type("L", (), {"spin": block[i.name]})()
                     for i in spec.indices}
            with self.subTest(operator=spec.name):
                self.assertTrue(
                    block_exists(tpl, label),
                    f"{spec.name}: representative block "
                    f"{''.join(block[i.name] for i in spec.indices)!r} does not "
                    f"conserve spin on its own slot order")

    def test_virtuals_first_breaks_wmbej_factor_blocks(self):
        """The mechanism behind the zero, pinned precisely.

        The virtuals-first OUTPUT block is itself spin-valid (slots [b,e,m,j] with
        b=a,e=b,m=a,j=b conserves on the pairing b-m, e-j). The failure is one level
        down, in the FACTORS: the same external assignment gives the bare integral
        `v(m,b,e,j)` the tag `aabb`, and `v`'s own lines are m-e and b-j, so m=a/e=b
        violates conservation. Every spin case of every term dies this way, so the
        operator integrates to zero -- silently, since a dropped block is the normal
        way a forbidden term is discarded."""
        from ccgen.spin import (_representative_block_for_sector, block_exists,
                                resolve_block, spin_label_cases)

        spec = next(s for s in _specs() if s.name == "Wmbej")
        terms = list(spec.definition_terms)
        tpl = _residual_template(spec.name, terms)
        block = _representative_block_for_sector(tpl, 1)

        # the output template block IS valid -- the defect is not here
        out_label = {i.name: type("L", (), {"spin": block[i.name]})()
                     for i in tpl.indices}
        self.assertTrue(block_exists(tpl, out_label))

        # ...but the bare-integral factor is rejected in every spin case
        bare = next(t for t in terms if len(t.factors) == 1)
        tags = set()
        for label in spin_label_cases(bare, block):
            for f in bare.factors:
                tag, exists = resolve_block(f, label)
                tags.add((f.name, tag, exists))
        self.assertTrue(tags, "no spin cases enumerated")
        self.assertTrue(
            all(not exists for _, _, exists in tags),
            f"expected every factor block forbidden, got {sorted(tags)}")
        self.assertIn(("v", "aabb", False), tags)

    def test_space_homogeneous_operators_agree_by_coincidence(self):
        """Documents WHY the defect hid: the four space-homogeneous operators
        (oooo/vvvv/vv/oo) have identical default and own orders, so only the two
        mixed-space ones (Fme `ov`, Wmbej `ovvo`) are reordered at all."""
        reordered = set()
        for spec in _specs():
            default_tpl = _residual_template(spec.name, list(spec.definition_terms))
            if [i.name for i in default_tpl.indices] != [i.name for i in spec.indices]:
                reordered.add(spec.name)
        self.assertEqual(reordered, {"Fme", "Wmbej"})


class ZeroAdaptationGuardTests(unittest.TestCase):
    """A non-empty GCC manifold adapting to nothing must raise, not return {}."""

    def test_guard_fires_on_zero_adaptation(self):
        spec = next(s for s in _specs() if s.name == "Wmbej")
        with self.assertRaises(ValueError) as ctx:
            spin_adapt_equations({spec.name: list(spec.definition_terms)})
        msg = str(ctx.exception)
        self.assertIn("Wmbej", msg)
        self.assertIn("templates", msg, "the error should name the fix")

    def test_guard_does_not_fire_on_a_genuinely_empty_manifold(self):
        """An empty input is not a bug -- only a non-empty one that vanishes is."""
        got = spin_adapt_equations({"singles": []})
        self.assertEqual(got.get("singles"), [])


class ResidualPathUnchangedTests(unittest.TestCase):
    """V1.0 must not move the residual layout contract (R3.1.2 half (ii),
    02364db) that the C++ runtime's `rank_dims` depends on."""

    def test_residual_adaptation_is_identical_without_templates(self):
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd")
        base = spin_adapt_equations(eqs)
        with_empty = spin_adapt_equations(eqs, templates={})
        with_none = spin_adapt_equations(eqs, templates=None)
        self.assertEqual(list(base), list(with_empty))
        self.assertEqual(list(base), list(with_none))
        for key in base:
            self.assertEqual(len(base[key]), len(with_empty[key]))
            self.assertEqual([str(t) for t in base[key]],
                             [str(t) for t in with_empty[key]])


class SpecAdaptationTests(unittest.TestCase):
    """V1.1a: a dressed intermediate's definition terms adapt to spatial form.

    Scope is the three REFERENCED operators (Finding A). V1.1a's only claim is that
    the terms adapt at the expected counts -- re-deriving indices/sig is V1.1b,
    block-keying the name V1.1c, recounting usage V1.1d, and the faithfulness gate
    V1.1e.
    """

    def test_referenced_operators_adapt_at_expected_counts(self):
        for name in REFERENCED:
            with self.subTest(operator=name):
                adapted = adapt_intermediate_spec(_spec(name))
                self.assertEqual(len(adapted.definition_terms),
                                 ADAPTED_TERM_COUNTS[name])

    def test_adaptation_never_empties_a_referenced_operator(self):
        """The V1.0 guard makes a silent vanish impossible, but assert the outcome
        directly too -- an empty builder compiles and computes zero."""
        for name in REFERENCED:
            with self.subTest(operator=name):
                self.assertGreater(
                    len(adapt_intermediate_spec(_spec(name)).definition_terms), 0)

    def test_metadata_is_carried_through_unchanged(self):
        """V1.1a changes ONLY definition_terms. If a later step (b/c/d) starts
        moving indices/sig/name/usage, it should do so deliberately, not as a side
        effect of V1.1a."""
        for name in REFERENCED:
            with self.subTest(operator=name):
                spec = _spec(name)
                adapted = adapt_intermediate_spec(spec)
                self.assertEqual(adapted.name, spec.name)
                self.assertEqual(adapted.indices, spec.indices)
                self.assertEqual(adapted.index_space_sig, spec.index_space_sig)
                self.assertEqual(adapted.usage_count, spec.usage_count)
                self.assertEqual(adapted.usage_targets, spec.usage_targets)
                self.assertNotEqual(adapted.definition_terms,
                                    spec.definition_terms)

    def test_adapted_terms_are_spatial(self):
        """The adapted terms must be plain AlgebraTerms (bridged out of SpinTerm),
        which is what the lowering/emit layers consume."""
        from ccgen.project import AlgebraTerm

        for name in REFERENCED:
            with self.subTest(operator=name):
                for term in adapt_intermediate_spec(_spec(name)).definition_terms:
                    self.assertIsInstance(term, AlgebraTerm)

    def test_all_six_adapt_even_though_three_are_unreferenced(self):
        """Fme/Fae/Fmi are out of scope but must not be BROKEN -- a later method may
        reference them, and V1.1a should apply unchanged when it does."""
        for spec in _specs():
            with self.subTest(operator=spec.name):
                self.assertGreater(
                    len(adapt_intermediate_spec(spec).definition_terms), 0)

    def test_adapter_is_injectable(self):
        """The `adapter` parameter is what makes V5 (UCC) a substitution rather
        than a second code path, so pin that it is actually used."""
        calls = []

        def fake(equations, templates=None):
            calls.append((sorted(equations), sorted(templates or {})))
            return {k: list(v) for k, v in equations.items()}

        spec = _spec("Wmnij")
        out = adapt_intermediate_spec(spec, adapter=fake)
        self.assertEqual(calls, [(["Wmnij"], ["Wmnij"])])
        self.assertEqual(out.definition_terms, spec.definition_terms)

    def test_adapter_returning_a_split_manifold_is_an_error(self):
        """A dressed intermediate has exactly one target. If an adapter splits it
        into sectors (the multi-Sz residual path), that is a bug, not a result."""
        def splitter(equations, templates=None):
            return {f"{k}_aaabaaab": list(v) for k, v in equations.items()}

        with self.assertRaises(ValueError) as ctx:
            adapt_intermediate_spec(_spec("Wmnij"), adapter=splitter)
        self.assertIn("Wmnij", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
