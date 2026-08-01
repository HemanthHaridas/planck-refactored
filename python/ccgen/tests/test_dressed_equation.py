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
    raw_multiset,
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

        # Current pinned state (diagram engine): 7 tau-weight + 19 structural.
        # These are the numbers the fix must drive to 0; assert the split so a
        # partial fix that only closes one class is visibly reflected here.
        self.assertEqual(tau_weight + struct, len(diff))
        self.assertEqual(len(diff), 26)
        self.assertEqual(tau_weight, 7)
        self.assertEqual(struct, 19)


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


if __name__ == "__main__":
    unittest.main()
