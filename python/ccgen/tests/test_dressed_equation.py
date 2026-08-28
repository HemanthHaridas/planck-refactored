"""Tests for the dressed-equation framework (Option B).

The rank-agnostic machinery -- expand_dressed_term / verify_dressed_equation --
is verified here.  The per-method transcription (ccsd_dressed_r2) is a WIP and
is only smoke-tested (expands without error), not asserted to diff to 0 yet.
"""

from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.indices import make_occ, make_vir  # noqa: E402
from ccgen.project import AlgebraTerm  # noqa: E402
from ccgen.tensors import Tensor, t1, t2, v  # noqa: E402
from ccgen.optimization.tau import tau  # noqa: E402
from ccgen.optimization.dressed_equation import (  # noqa: E402
    ccsd_dressed_r2,
    dressed_multiset,
    expand_dressed_term,
    expand_then_adapt,
    raw_multiset,
    verify_adapted_dressed_equation,
    verify_dressed_equation,
)


class FrameworkTests(unittest.TestCase):
    """The rank-agnostic dressed-equation machinery."""

    def test_raw_equation_verifies_against_itself(self) -> None:
        # A residual with no operators must diff to 0 against itself.
        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        raw = [
            AlgebraTerm(coeff=Fraction(1), factors=(v(a, b, i, j),),
                        free_indices=(a, b, i, j), summed_indices=(),
                        connected=True),
        ]
        ok, diff = verify_dressed_equation(raw, raw)
        self.assertTrue(ok)
        self.assertEqual(diff, {})

    def test_operator_factor_expands_to_definition(self) -> None:
        # A single Wmnij factor expands to its 4 definition terms (tau then
        # further expanded to primitives -> t2 + t1t1).
        m, n = make_occ("m"), make_occ("n")
        i, j = make_occ("i"), make_occ("j")
        term = AlgebraTerm(
            coeff=Fraction(1), factors=(Tensor("Wmnij", (m, n, i, j)),),
            free_indices=(m, n, i, j), summed_indices=(), connected=True,
        )
        prims = expand_dressed_term(term)
        # every expanded term is primitive (no operator / pseudo-amp names)
        opnames = {"Wmnij", "Wabef", "Fae", "Fmi", "Fme", "Wmbej",
                   "tau", "tau_tilde"}
        for p in prims:
            self.assertFalse(any(f.name in opnames for f in p.factors))
        # the bare-ERI defining piece survives
        self.assertTrue(any(
            [f.name for f in p.factors] == ["v"] for p in prims
        ))

    def test_dressed_pseudo_amplitude_expands(self) -> None:
        # 1/2 Wmnij * tau expands to a nonzero primitive multiset.
        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        m, n = make_occ("m"), make_occ("n")
        term = AlgebraTerm(
            coeff=Fraction(1, 2),
            factors=(Tensor("Wmnij", (m, n, i, j)), tau(a, b, m, n)),
            free_indices=(a, b, i, j), summed_indices=(m, n), connected=True,
        )
        ms = dressed_multiset([term])
        self.assertTrue(ms)
        self.assertTrue(all(isinstance(c, Fraction) for c in ms.values()))

    def test_verify_detects_a_wrong_coefficient(self) -> None:
        # The diff must be non-empty when a dressed term is scaled wrongly.
        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        raw = [AlgebraTerm(coeff=Fraction(1), factors=(v(a, b, i, j),),
                           free_indices=(a, b, i, j), summed_indices=(),
                           connected=True)]
        wrong = [AlgebraTerm(coeff=Fraction(2), factors=(v(a, b, i, j),),
                             free_indices=(a, b, i, j), summed_indices=(),
                             connected=True)]
        ok, diff = verify_dressed_equation(wrong, raw)
        self.assertFalse(ok)
        self.assertTrue(diff)


class TranscriptionWipTests(unittest.TestCase):
    """The CCSD R2 transcription -- smoke only (WIP, not yet diff==0)."""

    def test_r2_expands_without_error(self) -> None:
        # The transcribed R2 must at least expand to a primitive multiset
        # (exercises operator + pseudo-amplitude + antisymmetrizer expansion).
        ms = dressed_multiset(ccsd_dressed_r2())
        self.assertTrue(ms)

    def test_r2_not_yet_verified(self) -> None:
        # Documents current WIP state: R2 does not yet diff to 0. This test
        # should be INVERTED to assertTrue once BOTH the residual defect below
        # and the transcription are fixed.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])
        ok, _diff = verify_dressed_equation(ccsd_dressed_r2(), d)
        self.assertFalse(ok)  # WIP marker -- flip when transcription lands

    def test_r2_mismatch_decomposition_against_diagram(self) -> None:
        # V0.1/V0.2 tripwire: pins WHY ccsd_dressed_r2 fails to verify against the
        # FCI-validated diagram engine, decomposed into its two independent
        # causes so a fix to one is not masked by the other.  Flip / tighten each
        # count as the corresponding fix lands.
        #
        #   (A) STALE TRANSCRIPTION (dominant): ccsd_dressed_r2 is an incomplete
        #       hand transcription -- it carries terms the raw residual does not
        #       (dressed-only) and omits terms the raw residual has (raw-only).
        #       These have NO t1t1-pair / ratio-2 signature; they are just wrong
        #       or missing terms.  Fix = re-transcribe (or auto-generate) R2.
        #   (B) TAU WRITTEN-WEIGHT (secondary): keys with a t1t1 pair contracted
        #       into a same-space antisym `v` come out 2x (or 1/2x) because
        #       TAU_SPEC.written_t1t1_weight=2 is only correct for an EXTERNAL tau
        #       pair; a summed-and-antisym-contracted pair needs weight 1 (the
        #       Wabef diagnosis, D7.2.5.2 W1).
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", engine="diagram")
        raw = raw_multiset(eqs["doubles"])
        full = dressed_multiset(ccsd_dressed_r2())
        ok, diff = verify_dressed_equation(ccsd_dressed_r2(), eqs["doubles"])
        self.assertFalse(ok)

        tau_weight = struct = 0
        for k in diff:
            shape = [name for name, _ in k[0]]
            r = raw.get(k, Fraction(0))
            f = full.get(k, Fraction(0))
            ratio = (f / r) if r != 0 else None
            if shape.count("t1") >= 2 and ratio in (2, Fraction(1, 2)):
                tau_weight += 1
            else:
                struct += 1

        # Current pinned state (diagram engine): 7 tau-weight + 7 structural.
        # (Was 7 + 19 = 26; the D7.2.5.2 Fmi fix to _eri_canonical -- fold
        # bra<->ket AFTER dummy relabel, not before -- cut the structural class
        # 19->7 by making name-independent v orientations compare equal.)
        # These are the numbers a further fix must drive to 0; assert the split
        # so a partial fix that only closes one class is visibly reflected here.
        self.assertEqual(tau_weight + struct, len(diff))
        self.assertEqual(len(diff), 14)
        self.assertEqual(tau_weight, 7)
        self.assertEqual(struct, 7)

    def test_recognition_recon_overcounts_shared_primitives(self) -> None:
        # D7.3.0 tripwire: D7.2 recognition is SOUND per-occurrence but the 12
        # occurrences are NOT a partition -- naive summation of every occurrence's
        # expansion over-counts primitives shared between overlapping operator
        # definitions (Fae's -1/2 t1*Fme correction vs Fme itself, both tau pieces
        # of Wabef/Wmnij, Fmi's own corrections). Pins the current 24-mismatch
        # state so D7.3.0's coefficient reconciliation drives it to 0.
        from fractions import Fraction
        from ccgen.generate import generate_cc_equations
        from ccgen.optimization.dressing import (seeded_operators,
                                                  find_operator_occurrences)
        from ccgen.optimization.dressed_equation import expand_dressed_term

        eqs = generate_cc_equations("ccsd", engine="diagram")
        raw = raw_multiset(eqs["doubles"])
        recon: dict[tuple, Fraction] = {}
        for op in seeded_operators():
            for occ in find_operator_occurrences(op, eqs["doubles"]):
                for prim in expand_dressed_term(occ["term"], {op.name: op}):
                    key, coeff = self._eri_key_coeff(prim)
                    if coeff:
                        recon[key] = recon.get(key, Fraction(0)) + coeff

        mismatched = [k for k in set(recon) | set(raw)
                      if recon.get(k, Fraction(0)) != raw.get(k, Fraction(0))]
        # Sound recognition would give 0; the over-count gives 24 today. This is
        # the D7.3.0 gap, NOT a recognition bug -- flip to assertEqual(...,0) when
        # D7.3.0 coefficient reconciliation lands.
        self.assertEqual(len(mismatched), 24)

    @staticmethod
    def _eri_key_coeff(term):
        from ccgen.optimization.dressing import _eri_canonical
        return _eri_canonical(term)


class GeneratedResidualIntegrityTests(unittest.TestCase):
    """The generated residual has proper (non-degenerate) contractions.

    These were originally tripwires pinning a real defect: cluster amplitudes
    use dummy names (a, b, i, j) that collide with the projector's like-named
    externals, and ``apply_deltas`` keyed its protected-external lookup on
    ``(space, name)`` only.  Any dummy sharing a name with an external was
    silently rewritten INTO it, collapsing summations into degenerate terms
    (``f(a,a)``, ``t2(b,b,i,j)``, ``t2(a,b,i,j) v(i,j,i,j)``) and deleting the
    genuine ladder contractions.  Fixed by matching on union-find COMPONENT
    instead of name.  The tests are now inverted to assert correctness.
    """

    def _doubles(self):
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        return list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])

    def test_energy_manifold_is_clean(self) -> None:
        # Control: the energy manifold has NO degenerate contractions.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        e = list(generate_cc_equations("ccsd", parallel_workers=1)["energy"])
        for t in e:
            for fac in t.factors:
                names = [x.name for x in fac.indices]
                self.assertEqual(len(set(names)), len(names),
                                 f"energy term has repeated index: {t!r}")

    def test_t2v_ladder_contractions_are_present(self) -> None:
        # Every t2*v term must be a genuine contraction (summed indices), and
        # both CCSD ladders must appear: 1/2 sum_kl t2_abkl <ij||kl> (hole-hole)
        # and 1/2 sum_cd t2_cdij <cd||ab> (particle-particle).
        d = self._doubles()
        t2v = [t for t in d if sorted(f.name for f in t.factors) == ["t2", "v"]]
        self.assertTrue(t2v)
        self.assertTrue(
            all(t.summed_indices for t in t2v),
            "a t2*v term has no summed indices -- degenerate contraction",
        )
        # hole-hole ladder: v factor entirely occupied, two summed occ indices
        hh = [t for t in t2v
              if all(x.space == "occ"
                     for fac in t.factors if fac.name == "v"
                     for x in fac.indices)]
        # particle-particle ladder: v factor entirely virtual
        pp = [t for t in t2v
              if all(x.space == "vir"
                     for fac in t.factors if fac.name == "v"
                     for x in fac.indices)]
        self.assertTrue(hh, "hole-hole (Wmnij) ladder missing")
        self.assertTrue(pp, "particle-particle (Wabef) ladder missing")

    def test_no_term_repeats_an_index_inside_a_factor(self) -> None:
        # No factor may carry the same index twice -- that is the fingerprint of
        # a collapsed summation.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        for manifold in ("energy", "singles", "doubles"):
            for t in eqs[manifold]:
                for fac in t.factors:
                    names = [x.name for x in fac.indices]
                    self.assertEqual(
                        len(set(names)), len(names),
                        f"degenerate factor {fac!r} in {manifold}: {t!r}",
                    )

    def test_no_diagonal_fock_terms(self) -> None:
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        for manifold in ("energy", "singles", "doubles"):
            for t in eqs[manifold]:
                for fac in t.factors:
                    if fac.name == "f":
                        self.assertNotEqual(
                            fac.indices[0].name, fac.indices[1].name,
                            f"diagonal Fock in {manifold}: {t!r}",
                        )


class PartialCoverageRemainderTests(unittest.TestCase):
    """`assemble_dressed_equation` must keep the UNCOVERED part of a raw term whose
    key the operator expansions supply only partially.

    The bare/dressed partition used to be an all-or-nothing membership test on the
    expansion footprint: a key any expansion touched was dropped entirely. In CCSD
    singles the raw term `t1(b,j) t2(a,c,i,k) v(b,c,j,k)` has coefficient 1 while
    Wmbej's textbook `-1/2 t2*v` definition supplies only 1/2 of it through
    `Wmbej*t1`, so the missing 1/2 vanished -- the one mismatch in the GCC singles
    baseline. It is now emitted as a scaled remainder.
    """

    @classmethod
    def setUpClass(cls):
        from ccgen.generate import _dress_operator_equations, generate_cc_equations

        cls.raw = generate_cc_equations(
            "ccsd", engine="diagram", canonical_fock=True)
        cls.dressed, _ = _dress_operator_equations(cls.raw)

    def test_every_manifold_reexpands_exactly(self):
        """The headline: the dressed CCSD equation is exact on ALL manifolds. Singles
        was 1 mismatch before this fix; energy and doubles were already 0 and must
        stay there."""
        for manifold in self.raw:
            with self.subTest(manifold=manifold):
                ok, diff = verify_dressed_equation(
                    self.dressed[manifold], self.raw[manifold])
                self.assertTrue(ok, f"{manifold}: {len(diff)} mismatch(es): {diff}")

    def test_ccd_reexpands_exactly(self):
        """A second method, to pin that the remainder logic is not CCSD-shaped."""
        from ccgen.generate import _dress_operator_equations, generate_cc_equations

        raw = generate_cc_equations("ccd", engine="diagram", canonical_fock=True)
        dressed, _ = _dress_operator_equations(raw)
        for manifold in raw:
            with self.subTest(manifold=manifold):
                ok, diff = verify_dressed_equation(dressed[manifold], raw[manifold])
                self.assertTrue(ok, f"{manifold}: {diff}")

    def test_the_partial_term_is_actually_emitted(self):
        """Pin the mechanism, not just the outcome: the recovered remainder appears
        in the dressed singles manifold as a t1*t2*v term at +1/2. If a future
        refactor makes the totals balance some other way, this says whether the
        remainder path is still the reason."""
        from fractions import Fraction

        found = [
            t for t in self.dressed["singles"]
            if tuple(sorted(f.name for f in t.factors)) == ("t1", "t2", "v")
            and t.coeff == Fraction(1, 2)
        ]
        self.assertTrue(
            found, "the +1/2 t1*t2*v remainder is missing from dressed singles")

    def test_fully_covered_keys_are_not_duplicated(self):
        """The other side of the remainder: a key the expansions cover EXACTLY must
        contribute no bare term. Otherwise every dressed term would be double
        counted -- which would show up as a mismatch above, but assert the count
        directly so the failure is legible."""
        ok, diff = verify_dressed_equation(
            self.dressed["doubles"], self.raw["doubles"])
        self.assertTrue(ok)
        # doubles carries the tau/tau_c overlap corrections; if the remainder logic
        # ignored them it would re-add their share (measured: -1/8 and -1/4).
        self.assertEqual(len(diff), 0)


class AdaptedExpansionOrderTests(unittest.TestCase):
    """V1.1e.1: the pinned order for validating a dressed SPATIAL equation.

    Expand the dressed manifold to primitives in GCC, then spin-adapt -- the order
    Decision 5 implies (`GCC -> dress -> adapt`). The alternative (adapt the operator
    definitions and the residual separately, then verify against an adapted operator
    table) is measurably worse and is pinned here as rejected.
    """

    @classmethod
    def setUpClass(cls):
        from ccgen.generate import _dress_operator_equations, generate_cc_equations

        cls.raw = generate_cc_equations(
            "ccsd", engine="diagram", canonical_fock=True)
        cls.dressed, _ = _dress_operator_equations(cls.raw)

    def test_expansion_order_is_pinned(self):
        """V1.1e.1: expand-in-GCC-THEN-adapt is the validated order, and the
        alternative is measurably worse. Pinning both numbers is what stops the
        ordering being silently revisited later.

        Measured on dressed CCSD (mismatches vs the adapted raw residual):

            adapt-then-verify (REJECTED)   energy 0  singles 13  doubles 61
            expand-then-adapt (CHOSEN)     energy 0  singles  0  doubles 14

        The doubles=14 is a real open defect (V1.1e.2: `v` bra<->ket orientation
        sensitivity in the adapter), NOT a property of this ordering -- which is
        exactly why the order is pinned first, so that residue is one reproducible
        number.
        """
        from dataclasses import replace

        from ccgen.optimization.dressing import seeded_operators
        from ccgen.spin import spin_adapt_equations
        from ccgen.tensors import Tensor

        # (a) the chosen order
        chosen = verify_adapted_dressed_equation(self.dressed, self.raw)
        self.assertEqual({m: len(d) for m, d in chosen.items()}, {"doubles": 14})
        self.assertNotIn("energy", chosen)
        self.assertNotIn("singles", chosen)

        # (b) the rejected order, for contrast: adapt the operator definitions and
        # the residual separately, then verify against an adapted operator table
        table = {}
        for op in seeded_operators():
            adapted = spin_adapt_equations(
                {op.name: list(op.definition_terms)},
                templates={op.name: Tensor(op.name, tuple(op.block))},
            )
            table[op.name] = replace(
                op, definition_terms=tuple(adapted[op.name]))
        adapted_raw = spin_adapt_equations(self.raw)
        adapted_dressed = spin_adapt_equations(self.dressed)
        rejected = {}
        for manifold in self.raw:
            _ok, diff = verify_dressed_equation(
                adapted_dressed[manifold], adapted_raw[manifold], table)
            if diff:
                rejected[manifold] = len(diff)
        self.assertEqual(rejected, {"singles": 13, "doubles": 61})

        # the chosen order must be strictly better on every manifold
        for manifold in self.raw:
            self.assertLessEqual(
                len(chosen.get(manifold, {})), rejected.get(manifold, 0),
                f"{manifold}: chosen order is not better than the rejected one")

    def test_expand_then_adapt_is_additive(self):
        """The ordering is only meaningful if adaptation is linear over term
        partitions -- otherwise no expansion order could be correct. Measured: it is
        (0 mismatches splitting doubles in half), which is why the residue is a
        write-order sensitivity rather than a merge failure."""
        from ccgen.spin import _residual_template, spin_adapt_equations

        raw = self.raw["doubles"]
        template = _residual_template("doubles", raw)
        whole = raw_multiset(
            spin_adapt_equations({"doubles": list(raw)},
                                 templates={"doubles": template})["doubles"])
        halves: dict = {}
        mid = len(raw) // 2
        for part in (raw[:mid], raw[mid:]):
            got = raw_multiset(
                spin_adapt_equations({"doubles": list(part)},
                                     templates={"doubles": template})["doubles"])
            for key, coeff in got.items():
                halves[key] = halves.get(key, 0) + coeff
        mismatched = {
            k for k in set(whole) | set(halves)
            if whole.get(k, 0) != halves.get(k, 0)
        }
        self.assertEqual(mismatched, set())

    def test_expand_then_adapt_keeps_operators_out_of_the_adapter(self):
        """The mechanical reason for the order: expansion is what introduces the
        operator-internal dummies (__Wmnij_e, __Wabef_m). Doing it in GCC means the
        adapter only ever sees primitive factors."""
        adapted = expand_then_adapt(self.dressed, operators=None)
        operator_names = {"Wmnij", "Wabef", "Wmbej", "Fme", "Fae", "Fmi",
                          "tau", "tau_c", "tau_tilde"}
        for manifold, terms in adapted.items():
            for term in terms:
                for factor in term.factors:
                    self.assertNotIn(
                        factor.name, operator_names,
                        f"{manifold}: unexpanded {factor.name} reached the adapter")


if __name__ == "__main__":
    unittest.main()
