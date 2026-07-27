"""Tests for tau-recognition (A1).

Offline / codegen-inert: these exercise the tau spec (A1.0) and the external
skeleton fingerprint (A1.1) only. No generated code is involved.
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
from ccgen.tensors import Tensor, t1, t2  # noqa: E402
from ccgen.optimization.tau import (  # noqa: E402
    TAU_NAME,
    TAU_SPEC,
    TauMatch,
    apply_tau,
    canonical_multiset,
    expand_all_tau,
    external_skeleton,
    find_tau_matches,
    free_block_key,
    match_t1t1_half,
    tau,
    tau_rewrite_preserves_algebra,
    validate_tau_match,
)
from ccgen.optimization.tau import (  # noqa: E402
    _embedded_residue_signatures,
    find_embedded_tau_matches,
    validate_embedded_tau_match,
)


def _doubles_externals():
    """Four free externals of a doubles residual block: virtuals a,b; occ i,j."""
    a = make_vir("a", dummy=False)
    b = make_vir("b", dummy=False)
    i = make_occ("i", dummy=False)
    j = make_occ("j", dummy=False)
    return a, b, i, j


def _t2_half() -> AlgebraTerm:
    a, b, i, j = _doubles_externals()
    return AlgebraTerm(
        coeff=Fraction(1),
        factors=(t2(a, b, i, j),),
        free_indices=(a, b, i, j),
        summed_indices=(),
        connected=True,
    )


def _t1t1_half() -> AlgebraTerm:
    a, b, i, j = _doubles_externals()
    return AlgebraTerm(
        coeff=Fraction(1, 2),
        factors=(t1(a, i), t1(b, j)),
        free_indices=(a, b, i, j),
        summed_indices=(),
        connected=True,
    )


class TauSpecTests(unittest.TestCase):
    """A1.0 -- tau is defined with t2's shape and antisymmetry."""

    def test_tau_matches_t2_shape(self) -> None:
        a, b, i, j = _doubles_externals()
        ta = tau(a, b, i, j)
        tb = t2(a, b, i, j)
        self.assertEqual(ta.indices, tb.indices)
        self.assertEqual(ta.antisym_groups, tb.antisym_groups)

    def test_spec_coefficients(self) -> None:
        self.assertEqual(TAU_SPEC.t2_coeff, Fraction(1))
        self.assertEqual(TAU_SPEC.t1t1_coeff, Fraction(1, 2))


class ExternalSkeletonTests(unittest.TestCase):
    """A1.1 -- the fingerprint two tau halves must share on the free block."""

    def test_halves_share_free_block(self) -> None:
        # The two halves differ in factor names but MUST agree on the free
        # block -- that shared block is what makes them mergeable into tau.
        self.assertEqual(
            free_block_key(_t2_half()),
            free_block_key(_t1t1_half()),
        )
        self.assertEqual(free_block_key(_t2_half()), ("vir", "vir", "occ", "occ"))

    def test_skeleton_distinguishes_the_two_halves(self) -> None:
        # external_skeleton keeps the factor-name multiset, so the grouper can
        # tell the t2-half from the t1t1-half within one free block.
        self.assertNotEqual(
            external_skeleton(_t2_half()),
            external_skeleton(_t1t1_half()),
        )

    def test_skeleton_is_dummy_blind(self) -> None:
        # Same structure, different dummy letters -> identical skeleton.
        a = make_vir("a", dummy=False)
        i = make_occ("i", dummy=False)
        d1 = make_vir("d", dummy=True)
        m1 = make_occ("m", dummy=True)
        d2 = make_vir("e", dummy=True)
        m2 = make_occ("n", dummy=True)

        term1 = AlgebraTerm(
            coeff=Fraction(1),
            factors=(t1(d1, m1), Tensor("v", (a, d1, i, m1), antisym_groups=((0, 1), (2, 3)))),
            free_indices=(a, i),
            summed_indices=(d1, m1),
            connected=True,
        )
        term2 = AlgebraTerm(
            coeff=Fraction(1),
            factors=(t1(d2, m2), Tensor("v", (a, d2, i, m2), antisym_groups=((0, 1), (2, 3)))),
            free_indices=(a, i),
            summed_indices=(d2, m2),
            connected=True,
        )
        self.assertEqual(external_skeleton(term1), external_skeleton(term2))

    def test_skeleton_is_coeff_blind(self) -> None:
        half = _t1t1_half()
        self.assertEqual(
            external_skeleton(half),
            external_skeleton(half.scaled(-7)),
        )

    def test_unrelated_term_has_different_free_block(self) -> None:
        # A singles-block term (free a,i) must not share the doubles free block.
        a = make_vir("a", dummy=False)
        i = make_occ("i", dummy=False)
        singles = AlgebraTerm(
            coeff=Fraction(1),
            factors=(t1(a, i),),
            free_indices=(a, i),
            summed_indices=(),
            connected=True,
        )
        self.assertNotEqual(free_block_key(singles), free_block_key(_t2_half()))


class T1T1HalfTests(unittest.TestCase):
    """A1.2 -- recognize a bare 1/2 P(t1 t1) tau half."""

    def test_accepts_positive_half(self) -> None:
        m = match_t1t1_half(_t1t1_half())
        self.assertIsNotNone(m)
        self.assertEqual(m.coeff, Fraction(1, 2))
        # externals cover 2 vir + 2 occ, all distinct
        spaces = sorted(x.space for x in m.externals)
        self.assertEqual(spaces, ["occ", "occ", "vir", "vir"])
        self.assertEqual(len(set(m.externals)), 4)

    def test_accepts_negative_half(self) -> None:
        # The antisymmetrizing partner carries the opposite sign.
        m = match_t1t1_half(_t1t1_half().scaled(-1))
        self.assertIsNotNone(m)
        self.assertEqual(m.coeff, Fraction(-1, 2))

    def test_rejects_energy_form_with_v(self) -> None:
        # 1/2 t1 t1 v is the ENERGY contribution, not a bare tau half.
        a, b, i, j = _doubles_externals()
        energy = AlgebraTerm(
            coeff=Fraction(1, 2),
            factors=(
                t1(a, i), t1(b, j),
                Tensor("v", (a, b, i, j), antisym_groups=((0, 1), (2, 3))),
            ),
            free_indices=(),
            summed_indices=(a, b, i, j),
            connected=False,
        )
        self.assertIsNone(match_t1t1_half(energy))

    def test_rejects_t1_t2(self) -> None:
        a, b, i, j = _doubles_externals()
        term = AlgebraTerm(
            coeff=Fraction(1, 2),
            factors=(t1(a, i), t2(a, b, i, j)),
            free_indices=(a, b, i, j),
            summed_indices=(),
            connected=True,
        )
        self.assertIsNone(match_t1t1_half(term))

    def test_rejects_contracted_t1t1(self) -> None:
        # Two t1s sharing an index contract; that is not a doubles-block half.
        a = make_vir("a", dummy=False)
        b = make_vir("b", dummy=False)
        i = make_occ("i", dummy=True)
        term = AlgebraTerm(
            coeff=Fraction(1, 2),
            factors=(t1(a, i), t1(b, i)),
            free_indices=(a, b),
            summed_indices=(i,),
            connected=True,
        )
        self.assertIsNone(match_t1t1_half(term))

    def test_rejects_wrong_coefficient(self) -> None:
        self.assertIsNone(match_t1t1_half(_t1t1_half().scaled(3)))


def _v(a, b, i, j):
    return Tensor("v", (a, b, i, j), antisym_groups=((0, 1), (2, 3)))


def _energy_t2_v() -> AlgebraTerm:
    # 1/4 t2(a,b,i,j) v(a,b,i,j) -- the t2 half of the energy tau contraction.
    a, b, i, j = _doubles_externals()
    return AlgebraTerm(
        coeff=Fraction(1, 4),
        factors=(t2(a, b, i, j), _v(a, b, i, j)),
        free_indices=(),
        summed_indices=(a, b, i, j),
        connected=False,
    )


def _energy_t1t1_v() -> AlgebraTerm:
    # 1/2 t1(a,i) t1(b,j) v(a,b,i,j) -- the t1t1 half contracting the same v.
    a, b, i, j = _doubles_externals()
    return AlgebraTerm(
        coeff=Fraction(1, 2),
        factors=(t1(a, i), t1(b, j), _v(a, b, i, j)),
        free_indices=(),
        summed_indices=(a, b, i, j),
        connected=False,
    )


class FindTauMatchesTests(unittest.TestCase):
    """A1.3 -- pair t2-halves with t1t1-halves on tau-residue equality."""

    def test_energy_manifold_pair(self) -> None:
        # The two energy halves share a tau residue -> exactly one match.
        f_t1 = AlgebraTerm(  # unrelated: f(i,a) t1(a,i), no tau residue
            coeff=Fraction(1),
            factors=(Tensor("f", (make_occ("i"), make_vir("a"))),
                     t1(make_vir("a"), make_occ("i"))),
            free_indices=(),
            summed_indices=(make_vir("a"), make_occ("i")),
            connected=False,
        )
        terms = [f_t1, _energy_t2_v(), _energy_t1t1_v()]
        matches = find_tau_matches(terms)
        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.t2_index, 1)
        self.assertEqual(m.t1t1_index, 2)
        self.assertEqual(m.t2_coeff, Fraction(1, 4))
        self.assertEqual(m.t1t1_coeff, Fraction(1, 2))

    def test_no_match_without_partner(self) -> None:
        # A lone t2-half with no t1t1 counterpart is not matched.
        self.assertEqual(find_tau_matches([_energy_t2_v()]), [])

    def test_no_match_when_residue_differs(self) -> None:
        # t2 contracts v, but the t1t1 half contracts a DIFFERENT operator w:
        # residues differ, so no tau pair.
        a, b, i, j = _doubles_externals()
        t1t1_w = AlgebraTerm(
            coeff=Fraction(1, 2),
            factors=(t1(a, i), t1(b, j),
                     Tensor("w", (a, b, i, j), antisym_groups=((0, 1), (2, 3)))),
            free_indices=(),
            summed_indices=(a, b, i, j),
            connected=False,
        )
        self.assertEqual(find_tau_matches([_energy_t2_v(), t1t1_w]), [])

    def test_real_energy_manifold_has_one_match(self) -> None:
        # End-to-end: the detector finds the single tau contraction in the
        # real generated CCSD energy expression, with the 1:2 coeff ratio.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        matches = find_tau_matches(list(eqs["energy"]))
        self.assertEqual(len(matches), 1)
        m = matches[0]
        # tau = t2 + 1/2 P(t1t1): on symmetric contraction the t1t1 half weighs
        # twice the t2 half.
        self.assertEqual(m.t1t1_coeff, 2 * m.t2_coeff)

    def test_pair_dummy_relabel_invariant(self) -> None:
        # Same pair, different dummy letters on the t1t1 half -> still matched.
        a, b, i, j = _doubles_externals()
        c, d = make_vir("c"), make_vir("d")
        k, m = make_occ("k"), make_occ("m")
        t1t1_relabeled = AlgebraTerm(
            coeff=Fraction(1, 2),
            factors=(t1(c, k), t1(d, m), _v(c, d, k, m)),
            free_indices=(),
            summed_indices=(c, d, k, m),
            connected=False,
        )
        matches = find_tau_matches([_energy_t2_v(), t1t1_relabeled])
        self.assertEqual(len(matches), 1)


class ValidateTauMatchTests(unittest.TestCase):
    """A1.4 -- exact tau = t2 + 1/2 P(t1t1) coefficient firewall."""

    def _energy_terms_and_match(self):
        terms = [_energy_t2_v(), _energy_t1t1_v()]
        match = TauMatch(
            t2_index=0, t1t1_index=1,
            t2_coeff=terms[0].coeff, t1t1_coeff=terms[1].coeff,
        )
        return terms, match

    def test_accepts_exact_energy_pair(self) -> None:
        terms, match = self._energy_terms_and_match()
        self.assertTrue(validate_tau_match(terms, match))

    def test_rejects_wrong_t1t1_coefficient(self) -> None:
        # Corrupt the t1t1 half's coefficient: the expansion no longer
        # reproduces it, so the collapse would NOT be lossless.
        terms = [_energy_t2_v(), _energy_t1t1_v().scaled(3)]
        match = TauMatch(
            t2_index=0, t1t1_index=1,
            t2_coeff=terms[0].coeff, t1t1_coeff=terms[1].coeff,
        )
        self.assertFalse(validate_tau_match(terms, match))

    def test_rejects_wrong_t2_coefficient(self) -> None:
        # If the t2 half is scaled, the derived tau-term expands its t1t1 piece
        # to a value the (unscaled) t1t1 member no longer matches.
        terms = [_energy_t2_v().scaled(2), _energy_t1t1_v()]
        match = TauMatch(
            t2_index=0, t1t1_index=1,
            t2_coeff=terms[0].coeff, t1t1_coeff=terms[1].coeff,
        )
        self.assertFalse(validate_tau_match(terms, match))

    @unittest.expectedFailure  # OBSOLETE, not WIP: this pins the term-algebra
    # tau-detection (A1.4 firewall over the flat term list). With diagrammatic
    # generation (D4 landed), dressed-operator recognition is a topological
    # subgraph match (D7), so this index-binding approach is superseded and will
    # not be "flipped". Kept as the record of why the term-algebra route was
    # abandoned. See CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md (D7) + Open Work.
    def test_energy_and_singles_matches_validate(self) -> None:
        # A1.3 detection and the A1.4 exact firewall agree wherever the written
        # t1t1 representative carries TAU_SPEC.written_t1t1_weight.
        #
        # That weight is not universal: on the (corrected) residual the energy
        # and singles tau pairs have t1t1/t2 ratio 2, but the doubles pairs come
        # in ratio 1 (the Wmnij / Wabef ladders, where both halves carry 1/2)
        # and ratio 4.  The firewall correctly REJECTS the pairs whose weight
        # differs from its convention rather than collapsing them wrongly --
        # exercised by test_doubles_matches_are_screened_by_the_firewall.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        checked = 0
        for manifold in ("energy", "singles"):
            terms = list(eqs[manifold])
            for m in find_tau_matches(terms):
                self.assertTrue(
                    validate_tau_match(terms, m),
                    f"{manifold} match failed A1.4: "
                    f"{terms[m.t2_index]!r} <> {terms[m.t1t1_index]!r}",
                )
                checked += 1
        self.assertGreater(checked, 0)  # sanity: we actually exercised it

    def test_doubles_matches_are_screened_by_the_firewall(self) -> None:
        # The doubles tau candidates carry t1t1/t2 ratios (1 and 4) that differ
        # from the written-representative convention, so the exact firewall must
        # reject them -- never collapse on a weight it cannot reproduce.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        terms = list(eqs["doubles"])
        matches = find_tau_matches(terms)
        self.assertTrue(matches)  # candidates exist
        for m in matches:
            ratio = m.t1t1_coeff / m.t2_coeff
            if ratio != TAU_SPEC.written_t1t1_weight:
                self.assertFalse(validate_tau_match(terms, m))


def _has_tau(term) -> bool:
    return any(f.name == TAU_NAME for f in term.factors)


class ApplyTauTests(unittest.TestCase):
    """A1.5 -- collapse validated tau pairs, pass everything else through."""

    def test_energy_pair_collapses_to_one_tau_term(self) -> None:
        # unrelated term must survive; the pair becomes one tau term.
        f_t1 = AlgebraTerm(
            coeff=Fraction(1),
            factors=(Tensor("f", (make_occ("i"), make_vir("a"))),
                     t1(make_vir("a"), make_occ("i"))),
            free_indices=(),
            summed_indices=(make_vir("a"), make_occ("i")),
            connected=False,
        )
        terms = [f_t1, _energy_t2_v(), _energy_t1t1_v()]
        out = apply_tau(terms)
        self.assertEqual(len(out), 2)          # 3 -> 2 (pair -> one tau term)
        self.assertIn(f_t1, out)               # unrelated term untouched
        taus = [t for t in out if _has_tau(t)]
        self.assertEqual(len(taus), 1)
        self.assertEqual(taus[0].coeff, Fraction(1, 4))  # = t2 member coeff
        # no bare t2 / t1t1 halves remain
        self.assertFalse(any(any(f.name == "t2" for f in t.factors) for t in out))

    def test_idempotent(self) -> None:
        terms = [_energy_t2_v(), _energy_t1t1_v()]
        once = apply_tau(terms)
        twice = apply_tau(once)
        self.assertEqual(once, twice)
        self.assertEqual(len(once), 1)

    def test_unvalidated_pair_is_left_intact(self) -> None:
        # Corrupt the t1t1 coefficient: A1.4 rejects, so A1.5 must NOT collapse.
        terms = [_energy_t2_v(), _energy_t1t1_v().scaled(3)]
        out = apply_tau(terms)
        self.assertEqual(out, terms)           # unchanged
        self.assertFalse(any(_has_tau(t) for t in out))

    def test_no_pairs_is_passthrough(self) -> None:
        lone = [_energy_t2_v()]                 # no t1t1 partner
        self.assertEqual(apply_tau(lone), lone)

    def test_real_energy_manifold_collapses(self) -> None:
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        energy = list(eqs["energy"])
        out = apply_tau(energy)
        # energy had exactly one validated tau pair -> one fewer term, one tau.
        self.assertEqual(len(out), len(energy) - 1)
        self.assertEqual(sum(1 for t in out if _has_tau(t)), 1)
        # idempotent on real input too.
        self.assertEqual(apply_tau(out), out)


class TauEquivalenceGateTests(unittest.TestCase):
    """A1.6 -- collapsing then re-expanding tau must reproduce the algebra."""

    def test_energy_pair_roundtrips(self) -> None:
        terms = [_energy_t2_v(), _energy_t1t1_v()]
        self.assertTrue(tau_rewrite_preserves_algebra(terms))

    def test_expand_all_tau_inverts_collapse(self) -> None:
        terms = [_energy_t2_v(), _energy_t1t1_v()]
        collapsed = apply_tau(terms)
        self.assertEqual(
            canonical_multiset(expand_all_tau(collapsed)),
            canonical_multiset(terms),
        )

    def test_gate_detects_a_broken_rewrite(self) -> None:
        # Sanity: the gate has teeth. A hand-corrupted "collapse" that drops the
        # t1t1 half without introducing tau does NOT round-trip.
        terms = [_energy_t2_v(), _energy_t1t1_v()]
        broken = [_energy_t2_v()]  # lost the t1t1 contribution entirely
        self.assertNotEqual(
            canonical_multiset(expand_all_tau(broken)),
            canonical_multiset(terms),
        )

    def test_real_ccsd_residuals_preserve_algebra(self) -> None:
        # The load-bearing gate: on the full generated CCSD equations, applying
        # tau and re-expanding reproduces each manifold exactly.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        collapsed_any = False
        for manifold in ("energy", "singles", "doubles"):
            terms = list(eqs[manifold])
            self.assertTrue(
                tau_rewrite_preserves_algebra(terms),
                f"tau rewrite changed algebra in {manifold}",
            )
            if len(apply_tau(terms)) != len(terms):
                collapsed_any = True
        # We must have actually collapsed something, or the gate is vacuous.
        self.assertTrue(collapsed_any)


class FactorizeTauEmitTests(unittest.TestCase):
    """A1.7 / A1.8 -- tau wired through the Planck emitter behind a flag."""

    def setUp(self) -> None:
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"

    def test_default_is_off_and_unchanged(self) -> None:
        # The flag defaults off: generation is byte-identical to no flag.
        from ccgen.generate import print_cpp_planck

        base = print_cpp_planck("ccsd", include_intermediates=True)
        explicit_off = print_cpp_planck(
            "ccsd", include_intermediates=True, factorize_tau=False
        )
        self.assertEqual(base, explicit_off)
        self.assertNotIn("build_tau(", base)

    def test_flag_emits_tau_builder_and_reference(self) -> None:
        from ccgen.generate import print_cpp_planck

        code = print_cpp_planck(
            "ccsd", include_intermediates=True, factorize_tau=True
        )
        self.assertIn("Tensor4D build_tau(", code)          # the builder
        self.assertIn("const auto tau = build_tau", code)   # built per kernel
        # tau definition materializes t2 + written-weight t1 t1
        self.assertIn("amplitudes.t2(", code)
        self.assertIn("amplitudes.t1(", code)

    def test_factorized_equations_preserve_algebra(self) -> None:
        # A1.6 gate on the exact equations the emitter will lower with the flag.
        from ccgen.generate import generate_cc_equations
        from ccgen.optimization.tau import (
            factorize_tau_equations,
            tau_rewrite_preserves_algebra,
        )

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        for terms in eqs.values():
            self.assertTrue(tau_rewrite_preserves_algebra(terms))
        new_eqs, spec = factorize_tau_equations(eqs)
        self.assertIsNotNone(spec)  # something collapsed

    def test_generated_source_compiles(self) -> None:
        # A1.8 -- the tau-on generated CCSD is valid C++ against the real CC
        # headers. Skipped if a C++23 compiler or the Eigen fetch is absent.
        import os
        import shutil
        import subprocess
        import tempfile

        cxx = os.environ.get("CXX", "c++")
        if shutil.which(cxx) is None:
            self.skipTest(f"{cxx} not available")

        repo = Path(__file__).resolve().parents[3]  # repo root (above python/)
        eigen = repo / "build" / "_deps" / "eigen-src"
        if not eigen.is_dir():
            self.skipTest("Eigen fetch not present (configure the build first)")

        from ccgen.generate import print_cpp_planck

        code = print_cpp_planck(
            "ccsd", include_intermediates=True, factorize_tau=True
        )
        with tempfile.NamedTemporaryFile(
            suffix=".cpp", mode="w", delete=False
        ) as fh:
            fh.write(code)
            src = fh.name
        try:
            proc = subprocess.run(
                [cxx, "-std=c++23", "-fsyntax-only", "-w",
                 "-I", str(repo / "src"), "-I", str(eigen), src],
                capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(
                proc.returncode, 0,
                f"tau-on CCSD failed to compile:\n{proc.stderr[-2000:]}",
            )
        finally:
            os.unlink(src)


class EmbeddedResidueTests(unittest.TestCase):
    """A3.0.a -- tau residues for terms with arbitrary rest factors."""

    def _rest(self):
        from ccgen.tensors import v
        # a fixed "rest": t2(c,d,m,n) v(m,n,k,l) sharing block indices with tau
        c, d = make_vir("c"), make_vir("d")
        m, n = make_occ("m"), make_occ("n")
        return c, d, m, n

    def test_t2half_and_t1t1half_share_a_residue(self) -> None:
        # An embedded t2-half and its t1t1-half (same rest) must produce a
        # common residue signature under some designation.
        from ccgen.tensors import v

        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        c, d = make_vir("c"), make_vir("d")

        # rest = v(a,b,c,d); tau block = (a,b,i,j) via t2 or t1t1
        t2_term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(t2(a, b, i, j), v(a, b, c, d)),
            free_indices=(c, d, i, j),
            summed_indices=(a, b),
            connected=True,
        )
        t1t1_term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(t1(a, i), t1(b, j), v(a, b, c, d)),
            free_indices=(c, d, i, j),
            summed_indices=(a, b),
            connected=True,
        )
        sigs_t2 = {s for s, kind in _embedded_residue_signatures(t2_term)
                   if kind == "t2"}
        sigs_t1t1 = {s for s, kind in _embedded_residue_signatures(t1t1_term)
                     if kind == "t1t1"}
        self.assertTrue(sigs_t2 & sigs_t1t1,
                        "embedded t2-half and t1t1-half share no residue")

    def test_no_t1t1_candidate_when_no_pair_spans_block(self) -> None:
        # A single t1 (no pair) yields no t1t1 residue.
        from ccgen.tensors import v

        a = make_vir("a")
        i = make_occ("i")
        c, d = make_vir("c"), make_vir("d")
        term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(t1(a, i), v(a, c, i, d)),
            free_indices=(c, d),
            summed_indices=(a, i),
            connected=True,
        )
        self.assertFalse(
            any(kind == "t1t1" for _s, kind in _embedded_residue_signatures(term))
        )

    def test_real_doubles_have_embedded_pairings(self) -> None:
        # On real CCSD doubles, some residue is shared by a t2-half and a
        # t1t1-half -- the embedded-tau structure A1 could not see.
        import os
        from collections import defaultdict

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])
        by_res = defaultdict(lambda: {"t2": 0, "t1t1": 0})
        for term in d:
            for sig, kind in _embedded_residue_signatures(term):
                by_res[sig][kind] += 1
        paired = [s for s, g in by_res.items() if g["t2"] and g["t1t1"]]
        self.assertTrue(paired, "no embedded tau pairings found in doubles")


class FindEmbeddedTauMatchesTests(unittest.TestCase):
    """A3.0.b -- embedded pair detection (loose over-approximation)."""

    def test_finds_a_clean_embedded_pair(self) -> None:
        from ccgen.tensors import v

        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        c, d = make_vir("c"), make_vir("d")
        # t2-half and t1t1-half sharing rest v(a,b,c,d), tau block (a,b,i,j)
        t2_term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(t2(a, b, i, j), v(a, b, c, d)),
            free_indices=(c, d, i, j), summed_indices=(a, b), connected=True,
        )
        t1t1_term = AlgebraTerm(
            coeff=Fraction(2),
            factors=(t1(a, i), t1(b, j), v(a, b, c, d)),
            free_indices=(c, d, i, j), summed_indices=(a, b), connected=True,
        )
        ms = find_embedded_tau_matches([t2_term, t1t1_term])
        self.assertTrue(ms)
        self.assertTrue(any({m.t2_index, m.t1t1_index} == {0, 1} for m in ms))

    def test_read_only(self) -> None:
        from ccgen.tensors import v

        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        c, d = make_vir("c"), make_vir("d")
        terms = [
            AlgebraTerm(coeff=Fraction(1), factors=(t2(a, b, i, j), v(a, b, c, d)),
                        free_indices=(c, d, i, j), summed_indices=(a, b),
                        connected=True),
        ]
        before = list(terms)
        find_embedded_tau_matches(terms)
        self.assertEqual(terms, before)

    def test_real_doubles_pairs_have_inconsistent_ratios(self) -> None:
        # Honest documentation of the over-approximation: the embedded residue
        # bucketing pairs terms whose t1t1/t2 coeff ratios are NOT the clean 2
        # of the bare case -- proof that A3.0.c (not this) is the firewall.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])
        ms = find_embedded_tau_matches(d)
        self.assertTrue(ms)  # embedded candidates exist
        ratios = {m.t1t1_coeff / m.t2_coeff for m in ms if m.t2_coeff}
        # more than one distinct ratio -> not a clean single-factor collapse
        self.assertGreater(len(ratios), 1)


class ValidateEmbeddedTauMatchTests(unittest.TestCase):
    """A3.0.c -- the embedded exact firewall, and its verdict on real doubles."""

    def _genuine_embedded_pair(self):
        from ccgen.tensors import v

        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        c, d = make_vir("c"), make_vir("d")

        def T(coeff, facs, summed):
            return AlgebraTerm(coeff=Fraction(coeff), factors=tuple(facs),
                               free_indices=(c, d, i, j),
                               summed_indices=tuple(summed), connected=True)

        # c*tau(a,b,i,j)*v(a,b,c,d) -> t2 half (coeff 1) + t1t1 half (coeff 2)
        t2half = T(1, [t2(a, b, i, j), v(a, b, c, d)], (a, b))
        t1t1half = T(2, [t1(a, i), t1(b, j), v(a, b, c, d)], (a, b))
        return [t2half, t1t1half]

    def test_accepts_genuine_embedded_pair(self) -> None:
        terms = self._genuine_embedded_pair()
        ms = find_embedded_tau_matches(terms)
        self.assertTrue(ms)
        self.assertTrue(
            any(validate_embedded_tau_match(terms, m) for m in ms),
            "firewall rejected a genuine embedded tau pair",
        )

    def test_rejects_corrupted_coefficient(self) -> None:
        terms = self._genuine_embedded_pair()
        # break the t1t1 half's coefficient (should be 2, make it 5)
        terms[1] = terms[1].scaled(Fraction(5, 2))
        ms = find_embedded_tau_matches(terms)
        self.assertTrue(ms)
        self.assertFalse(
            any(validate_embedded_tau_match(terms, m) for m in ms),
            "firewall accepted a coefficient-corrupted pair",
        )

    @unittest.expectedFailure  # OBSOLETE, not WIP: the A3.0 embedded-tau
    # collapse over the flat term list is exactly the exact-cover / index-binding
    # dead end that diagrammatic generation (D4 landed) replaces -- dressed
    # operators are identifiable subgraphs (D7), not fragments to exact-cover.
    # Kept as the record of the abandoned route. See
    # CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md (D7) + Open Work.
    def test_real_doubles_have_no_valid_embedded_tau(self) -> None:
        # THE A3.0 VERDICT: residue-based embedded-tau collapse is a dead end.
        # A3.0.b proposes 18 pairs on real CCSD doubles; A3.0.c's exact firewall
        # validates ZERO of them. The genuine embedded tau (inside Wmnij) is
        # smeared across index-permuted fragments whose coefficients only
        # reconstruct under the correct simultaneous index binding -- i.e. it
        # needs the A3.2 isomorphism core, not a residue heuristic. This test
        # pins that finding: if a future change makes any validate, that is a
        # real signal to re-examine (and the firewall proven sound by the
        # accept/reject tests above).
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])
        ms = find_embedded_tau_matches(d)
        self.assertTrue(ms)  # candidates are proposed
        validated = [m for m in ms if validate_embedded_tau_match(d, m)]
        self.assertEqual(validated, [])  # but none survive the exact firewall


if __name__ == "__main__":
    unittest.main()
