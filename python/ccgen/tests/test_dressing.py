"""Tests for seeded dressed-operator hypotheses (A2).

Offline / codegen-inert: A2.0 (operator data model) and A2.1 (definition
self-consistency gate) only. No generated code is involved.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.optimization.dressing import (  # noqa: E402
    DressedOperator,
    FragmentLineGraph,
    OperatorFragments,
    TAU_TILDE_NAME,
    _build_wmnij,
    _build_fae,
    _build_fmi,
    _build_wabef,
    _build_wmbej,
    _perm_parity,
    _antisym_sort_factor,
    factor_to_fragment,
    fragment_signature,
    candidate_factor_subsets,
    collect_fragment_occurrences,
    enumerate_hypotheses,
    find_operator_occurrences,
    fragments_match,
    hypothesis_is_consistent,
    hypothesize_operator_term,
    induced_subfragment,
    match_fragment,
    operator_fragments,
    residual_term_to_fragment,
    tau_expanded_operator_fragments,
    term_to_fragment,
    _eri_canonical,
    bind_definition_term,
    bind_occurrence,
    classify_operator_occurrences,
    expected_instance_fragments,
    find_operator_pieces,
    footprints_are_distinct,
    operator_definition_is_consistent,
    operator_to_intermediate_spec,
    intermediate_dependencies,
    operator_footprint,
    seeded_operators,
    tau_expanded_operator,
    tau_tilde,
    verify_occurrence,
)
from ccgen.optimization.tau import TAU_NAME  # noqa: E402

EXPECTED = {
    "Fme": "ov",
    "Fae": "vv",
    "Fmi": "oo",
    "Wmnij": "oooo",
    "Wabef": "vvvv",
    "Wmbej": "ovvo",
}


class OperatorFamilyTests(unittest.TestCase):
    """A2.0 -- the seeded family is present with the right blocks."""

    def test_all_six_operators_present(self) -> None:
        names = sorted(op.name for op in seeded_operators())
        self.assertEqual(names, sorted(EXPECTED))

    def test_operator_blocks_match_expected_spaces(self) -> None:
        for op in seeded_operators():
            self.assertEqual(op.space_sig(), EXPECTED[op.name], op.name)

    def test_definition_terms_carry_the_block(self) -> None:
        for op in seeded_operators():
            block = frozenset(op.block)
            for term in op.definition_terms:
                self.assertEqual(
                    frozenset(term.free_indices), block,
                    f"{op.name} term {term!r} free indices != block",
                )

    def test_tau_dependencies_declared(self) -> None:
        by_name = {op.name: op for op in seeded_operators()}
        # Wmnij / Wabef use tau; Fae / Fmi use tau_tilde; Fme / Wmbej neither.
        self.assertIn(TAU_NAME, by_name["Wmnij"].uses)
        self.assertIn(TAU_NAME, by_name["Wabef"].uses)
        self.assertIn(TAU_TILDE_NAME, by_name["Fae"].uses)
        self.assertIn(TAU_TILDE_NAME, by_name["Fmi"].uses)
        self.assertEqual(by_name["Fme"].uses, frozenset())
        self.assertEqual(by_name["Wmbej"].uses, frozenset())


class DefinitionConsistencyTests(unittest.TestCase):
    """A2.1 -- transcribed definitions are self-consistent."""

    def test_all_seeded_operators_consistent(self) -> None:
        for op in seeded_operators():
            self.assertTrue(
                operator_definition_is_consistent(op),
                f"{op.name} definition is not self-consistent",
            )

    def test_empty_definition_rejected(self) -> None:
        op = seeded_operators()[0]
        broken = replace(op, definition_terms=())
        self.assertFalse(operator_definition_is_consistent(broken))

    def test_wrong_block_term_rejected(self) -> None:
        # A term whose free indices are not the operator block must fail.
        from ccgen.tensors import t1
        from ccgen.indices import make_vir, make_occ
        from ccgen.project import AlgebraTerm

        a = make_vir("a")
        i = make_occ("i")
        bad = DressedOperator(
            name="Bogus",
            block=(a,),  # rank-1 block, but the term below is over (a, i)
            definition_terms=(
                AlgebraTerm(
                    coeff=Fraction(1), factors=(t1(a, i),),
                    free_indices=(a, i), summed_indices=(), connected=True,
                ),
            ),
        )
        self.assertFalse(operator_definition_is_consistent(bad))

    def test_cancelling_transcription_rejected(self) -> None:
        # Two identical defining terms with opposite coefficients cancel the
        # structure to zero -> a dropped contribution -> rejected.
        from ccgen.tensors import v
        from ccgen.indices import make_occ

        m, n, i, j = (make_occ(x) for x in "mnij")
        block = (m, n, i, j)
        from ccgen.project import AlgebraTerm

        op = DressedOperator(
            name="BadWmnij",
            block=block,
            definition_terms=(
                AlgebraTerm(coeff=Fraction(1), factors=(v(m, n, i, j),),
                            free_indices=block, summed_indices=(), connected=True),
                AlgebraTerm(coeff=Fraction(-1), factors=(v(m, n, i, j),),
                            free_indices=block, summed_indices=(), connected=True),
            ),
        )
        self.assertFalse(operator_definition_is_consistent(op))

    def test_tau_tilde_shape_matches_t2(self) -> None:
        from ccgen.tensors import t2
        from ccgen.indices import make_vir, make_occ

        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        tt = tau_tilde(a, b, i, j)
        t2f = t2(a, b, i, j)
        self.assertEqual(tt.indices, t2f.indices)
        self.assertEqual(tt.antisym_groups, t2f.antisym_groups)


class OperatorToIntermediateSpecTests(unittest.TestCase):
    """D7.3.1: bridge a DressedOperator to the emit pipeline's IntermediateSpec."""

    def test_field_mapping(self):
        # D7.3.1a: name/indices/index_space_sig/definition_terms map directly.
        for op in seeded_operators():
            spec = operator_to_intermediate_spec(op)
            self.assertEqual(spec.name, op.name)
            self.assertEqual(spec.indices, tuple(op.block))
            self.assertEqual(spec.index_space_sig, op.space_sig())
            self.assertEqual(spec.definition_terms, op.definition_terms)
            self.assertEqual(spec.rank, len(op.block))

    def test_round_trip_faithful(self):
        # D7.3.1b: the spec is a lossless re-encoding -- its definition_terms
        # expand to the same primitive multiset as the operator's.
        from fractions import Fraction
        from ccgen.optimization.dressed_equation import expand_dressed_term

        def prims(terms):
            acc: dict = {}
            for t in terms:
                for p in expand_dressed_term(t, {}):
                    k, c = _eri_canonical(p)
                    if c:
                        acc[k] = acc.get(k, Fraction(0)) + c
            return {k: v for k, v in acc.items() if v}

        for op in seeded_operators():
            spec = operator_to_intermediate_spec(op)
            self.assertEqual(prims(spec.definition_terms), prims(op.definition_terms),
                             op.name)

    def test_canonical_fock_drops_only_fov_defterms(self):
        # D7.3.1c: canonical_fock drops f_ov/f_vo definition terms (runtime-inert
        # in Planck's canonical-Fock CC) and keeps everything else. Fme collapses
        # to its t1*oovv piece; Fae/Fmi lose their f_ov*t1 correction; the
        # f-free / diagonal-f / tau-bearing operators are unchanged.
        expected_defterm_counts = {  # (general, canonical)
            "Fme": (2, 1), "Fae": (4, 3), "Fmi": (4, 3),
            "Wmnij": (4, 4), "Wabef": (4, 4), "Wmbej": (5, 5),
        }
        for op in seeded_operators():
            gen = operator_to_intermediate_spec(op)
            canon = operator_to_intermediate_spec(op, canonical_fock=True)
            exp_gen, exp_canon = expected_defterm_counts[op.name]
            self.assertEqual(len(gen.definition_terms), exp_gen, op.name)
            self.assertEqual(len(canon.definition_terms), exp_canon, op.name)
            # no surviving canonical term carries an f_ov factor
            for t in canon.definition_terms:
                for f in t.factors:
                    if f.name == "f" and len(f.indices) == 2:
                        self.assertNotEqual(
                            {i.space for i in f.indices}, {"occ", "vir"}, op.name)

    def test_usage_annotation(self):
        # D7.3.1d: usage_count = total INSTANCES (P-branch-consolidated
        # occurrences summed over manifolds), usage_targets = manifolds present.
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        res = {"singles": eqs["singles"], "doubles": eqs["doubles"]}
        expected = {  # (usage_count, sorted targets)
            "Fme": (2, ["doubles", "singles"]),
            "Fae": (1, ["doubles"]),
            "Fmi": (2, ["doubles", "singles"]),
            "Wmnij": (1, ["doubles"]),
            "Wabef": (1, ["doubles"]),
            "Wmbej": (2, ["doubles", "singles"]),
        }
        for op in seeded_operators():
            spec = operator_to_intermediate_spec(op, residuals_by_manifold=res)
            exp_count, exp_targets = expected[op.name]
            self.assertEqual(spec.usage_count, exp_count, op.name)
            self.assertEqual(sorted(spec.usage_targets), exp_targets, op.name)

    def test_usage_default_zero_without_residuals(self):
        # Pure-spec case: no residuals -> usage stays at defaults.
        for op in seeded_operators():
            spec = operator_to_intermediate_spec(op)
            self.assertEqual(spec.usage_count, 0)
            self.assertEqual(spec.usage_targets, ())

    def test_dependencies(self):
        # D7.3.1e: tau/tau_tilde deps (the emit topo-sort edges) read from the
        # spec's definition terms, matching op.uses.
        for op in seeded_operators():
            spec = operator_to_intermediate_spec(op)
            self.assertEqual(intermediate_dependencies(spec), frozenset(op.uses),
                             op.name)

    def test_dependencies_track_canonical_filtering(self):
        # deps are read from the (possibly filtered) definition terms, so a
        # canonical spec whose only tau-bearing term survived still reports it;
        # here canonical filtering removes only f_ov terms, so deps are unchanged.
        for op in seeded_operators():
            gen = intermediate_dependencies(operator_to_intermediate_spec(op))
            canon = intermediate_dependencies(
                operator_to_intermediate_spec(op, canonical_fock=True))
            self.assertEqual(gen, canon, op.name)


def _factor_names(key) -> list[str]:
    return [f[0] for f in key[0]]


class OperatorFootprintTests(unittest.TestCase):
    """A2.2 -- slotized definition footprints."""

    def test_entry_count_matches_definition(self) -> None:
        # Each operator's footprint has one entry per distinct defining term.
        expected = {
            "Fme": 2, "Fae": 4, "Fmi": 4,
            "Wmnij": 4, "Wabef": 4, "Wmbej": 5,
        }
        for op in seeded_operators():
            fp = operator_footprint(op)
            self.assertEqual(len(fp.entries), expected[op.name], op.name)

    def test_wmnij_footprint_wiring(self) -> None:
        fp = operator_footprint(_build_wmnij())
        # block (m,n,i,j) -> ($0,$1,$2,$3); the bare ERI seed must be present.
        bare = None
        for key in fp.entries:
            if _factor_names(key) == ["v"]:
                bare = key
        self.assertIsNotNone(bare)
        bare_slots = tuple(slot for _space, slot in bare[0][0][1])
        self.assertEqual(bare_slots, ("$0", "$1", "$2", "$3"))
        # a tau * v defining piece must appear (the tau ladder into Wmnij).
        self.assertTrue(
            any(sorted(_factor_names(k)) == ["tau", "v"] for k in fp.entries),
            "Wmnij footprint missing the tau*v defining piece",
        )
        # and a t1 * v piece (the P(ij) t1 ooov term).
        self.assertTrue(
            any(sorted(_factor_names(k)) == ["t1", "v"] for k in fp.entries)
        )

    def test_block_names_are_slotized(self) -> None:
        # No raw block index name should survive in any footprint key.
        for op in seeded_operators():
            block_names = {idx.name for idx in op.block}
            fp = operator_footprint(op)
            for key in fp.entries:
                for _name, indices in key[0]:
                    for _space, nm in indices:
                        self.assertNotIn(
                            nm, block_names,
                            f"{op.name}: raw block name {nm!r} leaked into footprint",
                        )

    def test_footprints_pairwise_distinct(self) -> None:
        self.assertTrue(footprints_are_distinct(seeded_operators()))

    def test_footprint_coeffs_are_nonzero(self) -> None:
        for op in seeded_operators():
            fp = operator_footprint(op)
            self.assertTrue(all(c != 0 for c in fp.entries.values()), op.name)


class FindOperatorPiecesTests(unittest.TestCase):
    """A2.3 -- read-only occurrence detection (sound over-approximation)."""

    def _make_terms(self):
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import t2, v
        from ccgen.indices import make_vir, make_occ

        a, b = make_vir("a"), make_vir("b")
        i, j, m, n = (make_occ(x) for x in "ijmn")
        # 1/2 t2(a,b,m,n) v(m,n,i,j) -- the Wmnij oooo ladder piece
        wmnij_piece = AlgebraTerm(
            coeff=Fraction(1, 2),
            factors=(t2(a, b, m, n), v(m, n, i, j)),
            free_indices=(a, b, i, j),
            summed_indices=(m, n),
            connected=True,
        )
        # an unrelated term with no ERI factor at all
        no_v = AlgebraTerm(
            coeff=Fraction(1),
            factors=(t2(a, b, i, j),),
            free_indices=(a, b, i, j),
            summed_indices=(),
            connected=True,
        )
        return [wmnij_piece, no_v]

    def test_detects_oooo_seed_piece(self) -> None:
        terms = self._make_terms()
        matches = find_operator_pieces(terms, _build_wmnij())
        # the oooo ladder term (index 0) is reported; the no-ERI term (1) is not
        matched_indices = {m.term_index for m in matches}
        self.assertIn(0, matched_indices)
        self.assertNotIn(1, matched_indices)

    def test_reports_entry_coefficient(self) -> None:
        terms = self._make_terms()
        matches = find_operator_pieces(terms, _build_wmnij())
        # every reported match carries a nonzero defining-piece coefficient
        self.assertTrue(matches)
        self.assertTrue(all(m.entry_coeff != 0 for m in matches))
        self.assertTrue(all(m.operator == "Wmnij" for m in matches))

    def test_empty_input(self) -> None:
        self.assertEqual(find_operator_pieces([], _build_wmnij()), [])

    def test_sound_on_real_doubles(self) -> None:
        # On the real generated doubles, every reported Wmnij piece must
        # actually contain a v factor (soundness of the ERI-shape match).
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        doubles = list(eqs["doubles"])
        matches = find_operator_pieces(doubles, _build_wmnij())
        self.assertTrue(matches)  # Wmnij pieces exist in the doubles residual
        for m in matches:
            self.assertTrue(
                any(f.name == "v" for f in doubles[m.term_index].factors),
                "reported Wmnij piece has no ERI factor",
            )


class ClassifyOccurrencesTests(unittest.TestCase):
    """A2.4 -- coverage classification (complete vs partial)."""

    def _complete_wmnij_occurrence(self):
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import t2, t1, v
        from ccgen.optimization.tau import tau
        from ccgen.indices import make_vir, make_occ

        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        m, n = make_occ("m"), make_occ("n")
        e, f = make_vir("e"), make_vir("f")

        def T(c, facs, summed):
            return AlgebraTerm(
                coeff=Fraction(c), factors=tuple(facs),
                free_indices=(a, b, i, j), summed_indices=tuple(summed),
                connected=True,
            )

        # c * t2(a,b,m,n) * [all four Wmnij defining pieces over (m,n,i,j)]
        return [
            T(1, [t2(a, b, m, n), v(m, n, i, j)], (m, n)),
            T(1, [t2(a, b, m, n), t1(e, j), v(m, n, i, e)], (m, n, e)),
            T(-1, [t2(a, b, m, n), t1(e, i), v(m, n, j, e)], (m, n, e)),
            T(Fraction(1, 4), [t2(a, b, m, n), tau(e, f, i, j), v(m, n, e, f)],
              (m, n, e, f)),
        ]

    def test_complete_occurrence_is_covered(self) -> None:
        occs = classify_operator_occurrences(
            self._complete_wmnij_occurrence(), _build_wmnij()
        )
        self.assertEqual(len(occs), 1)
        self.assertTrue(occs[0].covered)
        self.assertEqual(len(occs[0].matched_entries), 4)

    def test_missing_piece_is_partial(self) -> None:
        # Drop the tau piece -> coverage incomplete -> partial.
        terms = self._complete_wmnij_occurrence()[:-1]
        occs = classify_operator_occurrences(terms, _build_wmnij())
        self.assertTrue(all(not o.covered for o in occs))

    def test_real_doubles_have_no_covered_wmnij(self) -> None:
        # Honest state of the CURRENT pipeline: raw CCSD doubles contain no
        # coverage-complete Wmnij occurrence, because forming it needs the
        # embedded-tau collapse (an A3 capability) and shared-t2 binding. A2.4
        # must NOT falsely report one collapsible.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        occs = classify_operator_occurrences(list(eqs["doubles"]), _build_wmnij())
        self.assertTrue(occs)  # candidate groups exist
        self.assertFalse(any(o.covered for o in occs))  # but none collapsible


class BindDefinitionTermTests(unittest.TestCase):
    """A3.1 -- bind one operator definition term into a residual term."""

    def test_binds_wmnij_seed_to_real_oooo_ladder(self) -> None:
        # Bind the Wmnij seed v(m,n,i,j) against the GENUINE hole-hole ladder
        # term of the doubles residual: 1/2 sum_kl t2(a,b,k,l) v(i,j,k,l).
        # (Before the apply_deltas dummy/external collision fix this term was
        # the degenerate 1/2 t2(a,b,i,j) v(i,j,i,j); it is now a real
        # contraction, which is a much stronger binding target.)
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        op = _build_wmnij()
        seed = op.definition_terms[0]  # v(m,n,i,j)
        eqs = generate_cc_equations("ccsd", parallel_workers=1)

        ladder = None
        for t in eqs["doubles"]:
            names = sorted(f.name for f in t.factors)
            if names != ["t2", "v"]:
                continue
            vfac = next(f for f in t.factors if f.name == "v")
            if all(x.space == "occ" for x in vfac.indices) and t.summed_indices:
                ladder = t
                break
        self.assertIsNotNone(ladder, "hole-hole ladder term not found")

        binds = bind_definition_term(seed, ladder)
        self.assertTrue(binds, "Wmnij seed did not bind to the hole-hole ladder")
        # Every binding maps the operator block onto the ladder's occ indices,
        # and (unlike the old degenerate target) the bra pair m,n must bind to
        # the SUMMED indices, distinct from the free externals i,j.
        summed_names = {x.name for x in ladder.summed_indices}
        free_names = {x.name for x in ladder.free_indices}
        self.assertTrue(
            any(
                {b_m.name for k, b_m in b.items() if k.name in ("m", "n")}
                <= summed_names
                and {b_m.name for k, b_m in b.items() if k.name in ("i", "j")}
                <= free_names
                for b in binds
            ),
            "no binding maps Wmnij's bra pair to the contracted indices",
        )

    def test_no_binding_on_wrong_space(self) -> None:
        from ccgen.tensors import v
        from ccgen.project import AlgebraTerm
        from ccgen.indices import make_vir, make_occ

        op = _build_wmnij()
        seed = op.definition_terms[0]  # v(oooo)
        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        vvvv = AlgebraTerm(coeff=Fraction(1), factors=(v(a, b, a, b),),
                           free_indices=(i, j), summed_indices=(a, b),
                           connected=True)
        self.assertEqual(bind_definition_term(seed, vvvv), [])

    def test_ambiguous_binding_reports_all(self) -> None:
        from ccgen.tensors import v
        from ccgen.project import AlgebraTerm
        from ccgen.indices import make_occ

        op = _build_wmnij()
        seed = op.definition_terms[0]
        i, j = make_occ("i"), make_occ("j")
        two = AlgebraTerm(coeff=Fraction(1),
                          factors=(v(i, j, i, j), v(j, i, j, i)),
                          free_indices=(i, j), summed_indices=(), connected=True)
        binds = bind_definition_term(seed, two)
        # Two v factors, each with antisymmetry-equivalent orderings -> several
        # embeddings. At least the two distinct factor choices are present.
        self.assertGreaterEqual(len(binds), 2)

    def test_binding_covers_the_block(self) -> None:
        # Every returned binding maps exactly the operator's block indices.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        op = _build_wmnij()
        block_names = {idx.name for idx in op.block}
        eqs = generate_cc_equations("ccsd", parallel_workers=1)
        for t in eqs["doubles"]:
            for binds in [bind_definition_term(dt, t) for dt in op.definition_terms]:
                for b in binds:
                    self.assertEqual({k.name for k in b}, block_names)


class BindOccurrenceTests(unittest.TestCase):
    """A3.2 -- global binding across all of an operator's definition terms."""

    def _complete_wmnij(self):
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import t2, t1, v
        from ccgen.optimization.tau import tau
        from ccgen.indices import make_vir, make_occ

        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        m, n = make_occ("m"), make_occ("n")
        e, f = make_vir("e"), make_vir("f")

        def T(c, facs, summed):
            return AlgebraTerm(coeff=Fraction(c), factors=tuple(facs),
                               free_indices=(a, b, i, j), summed_indices=tuple(summed),
                               connected=True)

        return [
            T(1, [t2(a, b, m, n), v(m, n, i, j)], (m, n)),
            T(1, [t2(a, b, m, n), t1(e, j), v(m, n, i, e)], (m, n, e)),
            T(-1, [t2(a, b, m, n), t1(e, i), v(m, n, j, e)], (m, n, e)),
            T(Fraction(1, 4), [t2(a, b, m, n), tau(e, f, i, j), v(m, n, e, f)],
              (m, n, e, f)),
        ]

    def test_assembles_complete_wmnij(self) -> None:
        obs = bind_occurrence(_build_wmnij(), self._complete_wmnij())
        # Antisymmetry-aware binding finds the occurrence under every symmetry-
        # equivalent block permutation (Wmnij is antisymmetric in m<->n and
        # i<->j), so several bindings are reported for the one occurrence. Each
        # must be a valid, fully-covered global binding; the identity binding
        # must be among them.
        self.assertTrue(obs)
        self.assertTrue(all(len(o.coverage) == 4 for o in obs))
        bindings = [dict(o.binding) for o in obs]
        self.assertIn({"m": "m", "n": "n", "i": "i", "j": "j"}, bindings)

    def test_missing_piece_yields_no_global_binding(self) -> None:
        # Drop the tau piece: no global binding covers all definition terms.
        terms = self._complete_wmnij()[:-1]
        self.assertEqual(bind_occurrence(_build_wmnij(), terms), [])

    def test_raw_doubles_wmnij_assembles_with_full_eri_symmetry(self) -> None:
        # A3.2 FIXED. Two changes let the seeded Wmnij bind to the REAL raw
        # doubles residual:
        #   1. Binding uses the FULL 8-fold ERI symmetry (intra-pair antisym
        #      PLUS bra<->ket exchange). The exchange reconciles the textbook
        #      Wmnij t1-piece's <oo||ov> with the pipeline's raw <ov||oo>.
        #   2. Binding targets the tau-EXPANDED operator variant, whose tau
        #      piece is written as t2 + t1t1 -- matching the raw residual, which
        #      carries no literal tau factor (A3.0.c's finding, bypassed rather
        #      than fought).
        # Result: every one of Wmnij's definition pieces binds under a
        # consistent global block map, so real Wmnij occurrences are found.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])
        op = tau_expanded_operator(_build_wmnij())
        obs = bind_occurrence(op, d)
        self.assertTrue(obs, "seeded Wmnij should assemble from raw doubles")
        # every binding covers all expanded definition pieces...
        n_pieces = len(op.definition_terms)
        self.assertTrue(all(len(o.coverage) == n_pieces for o in obs))
        # ...and every block index (all occ for Wmnij) maps to an occ index.
        block_names = {idx.name for idx in _build_wmnij().block}
        for o in obs:
            self.assertEqual({k for k, _v in o.binding}, block_names)


class TauExpandedOperatorTests(unittest.TestCase):
    """A3.2 completion -- the tau-expanded operator variant."""

    def test_expansion_replaces_tau_with_t2_and_t1t1(self) -> None:
        base = _build_wmnij()
        exp = tau_expanded_operator(base)
        # base Wmnij has 4 terms (one is 1/4 tau*v); expanding tau splits that
        # one into t2*v + t1t1*v -> 5 terms, none carrying a tau/tau_tilde.
        self.assertEqual(len(exp.definition_terms), 5)
        names = {f.name for t in exp.definition_terms for f in t.factors}
        self.assertNotIn(TAU_NAME, names)
        self.assertNotIn(TAU_TILDE_NAME, names)

    def test_expanded_operator_still_consistent(self) -> None:
        # The expanded definition must remain self-consistent (A2.1).
        for op in seeded_operators():
            exp = tau_expanded_operator(op)
            self.assertTrue(
                operator_definition_is_consistent(exp),
                f"tau-expanded {op.name} not self-consistent",
            )

    def test_operators_without_tau_are_unchanged(self) -> None:
        # Fme / Wmbej reference no pseudo-amplitude -> expansion is a no-op
        # on the definition-term count.
        by_name = {op.name: op for op in seeded_operators()}
        for name in ("Fme", "Wmbej"):
            base = by_name[name]
            exp = tau_expanded_operator(base)
            self.assertEqual(
                len(exp.definition_terms), len(base.definition_terms), name
            )


class VerifyOccurrenceTests(unittest.TestCase):
    """A3.3 -- exact-coefficient firewall for a bound occurrence."""

    def _synthetic_complete_wmnij(self, c=Fraction(1, 2)):
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import t2, t1, v
        from ccgen.indices import make_vir, make_occ

        a, b = make_vir("a"), make_vir("b")
        i, j = make_occ("i"), make_occ("j")
        m, n = make_occ("m"), make_occ("n")
        e, f = make_vir("e"), make_vir("f")

        def T(coeff, facs, summed):
            return AlgebraTerm(coeff=Fraction(coeff), factors=tuple(facs),
                               free_indices=(a, b, i, j), summed_indices=tuple(summed),
                               connected=True)

        # c * [each tau-expanded Wmnij defn piece] * t2(a,b,m,n)
        return [
            T(c * 1, [v(m, n, i, j), t2(a, b, m, n)], (m, n)),
            T(c * 1, [t1(e, j), v(m, n, i, e), t2(a, b, m, n)], (m, n, e)),
            T(c * -1, [t1(e, i), v(m, n, j, e), t2(a, b, m, n)], (m, n, e)),
            T(c * Fraction(1, 4), [t2(e, f, i, j), v(m, n, e, f), t2(a, b, m, n)],
              (m, n, e, f)),
            T(c * Fraction(1, 2), [t1(e, i), t1(f, j), v(m, n, e, f), t2(a, b, m, n)],
              (m, n, e, f)),
        ]

    def test_verifies_synthetic_complete_occurrence(self) -> None:
        from ccgen.optimization.dressing import bind_occurrence

        terms = self._synthetic_complete_wmnij()
        op = _build_wmnij()
        obs = bind_occurrence(tau_expanded_operator(op), terms)
        self.assertTrue(obs)
        # At least one binding verifies as an exact instance. Several may verify
        # -- they are the symmetry-equivalent orientations of the antisymmetric
        # Wmnij, all genuine; A3.4 dedups them when selecting a cover.
        self.assertGreaterEqual(sum(verify_occurrence(op, terms, o) for o in obs), 1)

    def test_rejects_corrupted_coefficient(self) -> None:
        from ccgen.optimization.dressing import bind_occurrence

        terms = self._synthetic_complete_wmnij()
        terms[1] = terms[1].scaled(5)  # break one piece's coefficient
        op = _build_wmnij()
        obs = bind_occurrence(tau_expanded_operator(op), terms)
        self.assertTrue(obs)
        self.assertEqual(sum(verify_occurrence(op, terms, o) for o in obs), 0)

    def test_raw_doubles_single_fragment_slice_does_not_verify(self) -> None:
        # SCOPE finding (A3.3<->A3.4 coupling): in the raw residual each Wmnij
        # definition-term contribution is fractured across several fragments
        # that must SUM to c*defn_coeff. bind_occurrence's coverage claims one
        # fragment per term, so the exact check fails until A3.4 gathers the
        # full fracture. A3.3 correctly returns False here -- it never accepts a
        # numerically-incomplete slice as a valid collapse.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations
        from ccgen.optimization.dressing import bind_occurrence

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])
        op = _build_wmnij()
        obs = bind_occurrence(tau_expanded_operator(op), d)
        self.assertTrue(obs)  # bindings exist
        self.assertEqual(sum(verify_occurrence(op, d, o) for o in obs), 0)


class InstanceFragmentTests(unittest.TestCase):
    """A3.4 groundwork -- fracture spec + ERI-canonical matching."""

    def test_eri_canonical_folds_bra_ket_exchange(self) -> None:
        # Two v arrangements related by bra<->ket exchange must share a key.
        from ccgen.tensors import t1, v
        from ccgen.project import AlgebraTerm
        from ccgen.indices import make_vir, make_occ

        i, j, k = make_occ("i"), make_occ("j"), make_occ("k")
        a, c = make_vir("a"), make_vir("c")
        # <ij||ic> style vs <ic||ij> style, same integral under exchange
        t_a = AlgebraTerm(coeff=Fraction(1), factors=(t1(c, j), v(i, j, i, c)),
                          free_indices=(a, k), summed_indices=(i, j, c),
                          connected=True)
        t_b = AlgebraTerm(coeff=Fraction(1), factors=(t1(c, j), v(i, c, i, j)),
                          free_indices=(a, k), summed_indices=(i, j, c),
                          connected=True)
        self.assertEqual(_eri_canonical(t_a)[0], _eri_canonical(t_b)[0])

    def test_expected_fragments_all_present_in_real_residual(self) -> None:
        # The fracture spec: every expected Wmnij fragment's ERI-canonical key
        # appears in the raw doubles (the exchange-aware matching that A3.2's
        # binding fix and this ERI key together unlock).
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])
        res_keys = set()
        for t in d:
            res_keys.add(_eri_canonical(t)[0])

        op = _build_wmnij()
        obs = bind_occurrence(tau_expanded_operator(op), d)
        self.assertTrue(obs)
        exp = expected_instance_fragments(op, d, obs[0])
        self.assertEqual(len(exp), 5)
        # every expected fragment structurally occurs in the residual
        self.assertTrue(all(key in res_keys for key in exp))

    def test_global_sums_differ_shared_fragments_need_assignment(self) -> None:
        # A3.4 SCOPE: expected fragments occur, but the GLOBAL residual
        # coefficient sums do not equal one instance's expected coefficients --
        # because symmetry-equivalent bindings SHARE fragments, splitting each
        # residual coefficient across orientations (2x, 3x ratios observed). So
        # a naive global-sum check over-claims; the exact collapse needs
        # per-instance fragment ASSIGNMENT (exact cover), the remaining hard
        # A3.4 core. This pins that the machinery is right but assignment is
        # still open.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])
        res_sums = {}
        for t in d:
            key, coeff = _eri_canonical(t)
            res_sums[key] = res_sums.get(key, Fraction(0)) + coeff

        op = _build_wmnij()
        obs = bind_occurrence(tau_expanded_operator(op), d)
        exp = expected_instance_fragments(op, d, obs[0])
        # seed matches globally, but not all pieces (shared-fragment splitting)
        matches = sum(1 for key, ec in exp.items()
                      if res_sums.get(key, Fraction(0)) == ec)
        self.assertLess(matches, len(exp))


def _system_is_consistent(A, b):
    """Exact Gaussian elimination over Fractions: is A x = b consistent?"""
    rows = len(A)
    ncol = len(A[0]) if A else 0
    M = [list(A[i]) + [b[i]] for i in range(rows)]
    r0 = 0
    for c in range(ncol):
        pr = next((rr for rr in range(r0, rows) if M[rr][c] != 0), None)
        if pr is None:
            continue
        M[r0], M[pr] = M[pr], M[r0]
        piv = M[r0][c]
        M[r0] = [x / piv for x in M[r0]]
        for rr in range(rows):
            if rr != r0 and M[rr][c] != 0:
                fct = M[rr][c]
                M[rr] = [a - fct * b2 for a, b2 in zip(M[rr], M[r0])]
        r0 += 1
        if r0 == rows:
            break
    # inconsistent iff some row has all-zero coeffs but nonzero rhs
    return not any(
        all(M[rr][c] == 0 for c in range(ncol)) and M[rr][ncol] != 0
        for rr in range(rows)
    )


class ExactCoverDisprovenTests(unittest.TestCase):
    """The exact-cover model is DISPROVEN -- pinned as a tripwire.

    The doubles residual is NOT a sum of independent ``operator*rest``
    instances.  Re-derived on the CORRECTED equations (after the apply_deltas
    dummy/external collision fix): the candidate set is 6 instances (Wmnij x2,
    Wabef x2, Wmbej x2 -- the F-operators bind globally not at all) over 20
    fragment keys, and the exact linear system is INCONSISTENT.

    The decisive evidence is coverage, not coefficients: those instances touch
    only ~20 of ~70 residual keys, so most of the equation lies outside the
    instance form entirely -- exactly as the dressed R2 structure predicts
    (operators multiply *specific* amplitudes, plus bare terms).  The path is
    curated dressed-equation templates, not automatic exact cover.

    If either assertion flips, the model assumptions changed -- re-examine
    before building a cover solver.
    """

    def test_operator_instance_system_is_inconsistent(self) -> None:
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])

        instances = []
        for op in seeded_operators():
            seen = set()
            for o in bind_occurrence(tau_expanded_operator(op), d):
                e = expected_instance_fragments(op, d, o)
                if e is None:
                    continue
                key = frozenset(e.items())
                if key in seen:
                    continue
                seen.add(key)
                instances.append(e)

        keys = sorted({k for e in instances for k in e}, key=str)
        kidx = {k: i for i, k in enumerate(keys)}

        res = {}
        for t in d:
            k, c = _eri_canonical(t)
            res[k] = res.get(k, Fraction(0)) + c
        b = [res.get(k, Fraction(0)) for k in keys]

        A = [[Fraction(0)] * len(instances) for _ in keys]
        for ci, e in enumerate(instances):
            for k, c in e.items():
                A[kidx[k]][ci] = c

        self.assertFalse(
            _system_is_consistent(A, b),
            "operator-instance system became consistent -- model changed, "
            "re-examine exact cover",
        )

    def test_operator_instances_cover_only_a_minority_of_the_residual(self) -> None:
        # The decisive evidence: it is a MODEL failure, not a coefficient one.
        # The operator instances do not even reach most residual fragments, so
        # no assignment of them could reproduce the equation.
        import os

        os.environ["CCGEN_NO_ACCEL"] = "1"
        from ccgen.generate import generate_cc_equations

        d = list(generate_cc_equations("ccsd", parallel_workers=1)["doubles"])

        touched = set()
        for op in seeded_operators():
            for o in bind_occurrence(tau_expanded_operator(op), d):
                e = expected_instance_fragments(op, d, o)
                if e:
                    touched |= set(e)

        residual_keys = {_eri_canonical(t)[0] for t in d}
        untouched = residual_keys - touched
        self.assertTrue(
            untouched,
            "operator instances now reach every residual fragment -- the "
            "instance model may be viable; re-examine exact cover",
        )
        # Most of the equation is outside the instance form.
        self.assertGreater(len(untouched), len(residual_keys) / 2)


class FragmentLineGraphModelTests(unittest.TestCase):
    """D7.1.0: the fragment line-graph data model round-trips.

    Data model only -- no encoder yet (D7.1.1/1.2). Uses a hand-built fragment
    for the Wmnij ``1/4 tau_ijef v_mnef`` term: 2 internal particle lines
    (tau<->v on e,f) + 4 dangling hole lines (block ports m,n,i,j). Confirms the
    internal/dangling split, port-species read-back, and the line format is
    identical in shape to diagram.LineGraph (species, endpoint_a, endpoint_b)."""

    def _tau_v_fragment(self):
        # factor 0 = tau(e,f,i,j) [p,p,h,h]; factor 1 = v(m,n,e,f) [h,h,p,p]
        # block = (m, n, i, j) -> ports 0,1,2,3.  Internal lines: e (p), f (p)
        # between tau and v.  Dangling: tau's i,j -> ports 2,3 (h); v's m,n ->
        # ports 0,1 (h).
        F0, F1 = ("factor", 0), ("factor", 1)
        lines = (
            ("p", F0, F1),                 # e: tau <-> v
            ("p", F0, F1),                 # f: tau <-> v
            ("h", F0, ("port", 2)),        # i (tau) -> port 2
            ("h", F0, ("port", 3)),        # j (tau) -> port 3
            ("h", F1, ("port", 0)),        # m (v)  -> port 0
            ("h", F1, ("port", 1)),        # n (v)  -> port 1
        )
        return FragmentLineGraph(lines=lines, n_factors=2, n_ports=4)

    def test_internal_dangling_split(self):
        fr = self._tau_v_fragment()
        self.assertEqual(len(fr.internal_lines), 2)
        self.assertEqual(len(fr.dangling_lines), 4)
        # internal lines are particle (the summed e,f); dangling are hole
        self.assertTrue(all(sp == "p" for sp, _, _ in fr.internal_lines))
        self.assertTrue(all(sp == "h" for sp, _, _ in fr.dangling_lines))

    def test_port_species(self):
        fr = self._tau_v_fragment()
        # all four Wmnij block ports are occupied -> hole species
        self.assertEqual(fr.port_species, {0: "h", 1: "h", 2: "h", 3: "h"})

    def test_line_format_matches_linegraph(self):
        # each line is a 3-tuple (species, endpoint_a, endpoint_b) with species
        # in {"p","h"} -- identical shape to diagram.LineGraph.lines, so a
        # subgraph match runs on one representation.
        from ccgen.diagram import LineGraph
        fr = self._tau_v_fragment()
        for line in fr.lines:
            self.assertEqual(len(line), 3)
            self.assertIn(line[0], ("p", "h"))
        # LineGraph accepts the same tuple shape (constructs without error)
        LineGraph(lines=fr.lines, bra_level=0, h_rank=2)

    def test_operator_fragments_container(self):
        # the D7.1.3 container holds (coeff, FragmentLineGraph) pairs + metadata
        fr = self._tau_v_fragment()
        of = OperatorFragments(
            name="Wmnij", block=_build_wmnij().block,
            fragments=((Fraction(1, 4), fr),),
            uses=frozenset({"tau"}))
        self.assertEqual(of.name, "Wmnij")
        self.assertEqual(len(of.fragments), 1)
        self.assertEqual(of.fragments[0][0], Fraction(1, 4))
        self.assertIn("tau", of.uses)


class FactorToFragmentTests(unittest.TestCase):
    """D7.1.1: the single-factor encoder emits one line per index -- a dangling
    ("port", slot) for a block index, a ("stub", name) for a summed index, with
    occ->hole / vir->particle species. Checked against the concrete factors of
    the Wmnij definition."""

    def setUp(self):
        self.w = _build_wmnij()
        self.block = self.w.block          # (m, n, i, j), all occ

    def _term(self, k):
        return self.w.definition_terms[k]

    def test_bare_v_all_ports(self):
        # term 0: v(m,n,i,j) -- every index is a block port, all hole
        v_factor = self._term(0).factors[0]
        lines = factor_to_fragment(v_factor, ("factor", 0), self.block)
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(sp == "h" for sp, _, _ in lines))
        ends = {e for _, _, e in lines}
        self.assertEqual(ends, {("port", 0), ("port", 1), ("port", 2), ("port", 3)})

    def test_t1_one_stub_one_port(self):
        # term 1: t1(e,j) -- e summed (vir->p stub), j block (occ->h port slot 3)
        t1_factor = self._term(1).factors[0]
        self.assertEqual(t1_factor.name, "t1")
        lines = factor_to_fragment(t1_factor, ("factor", 0), self.block)
        self.assertIn(("p", ("factor", 0), ("stub", "e")), lines)
        self.assertIn(("h", ("factor", 0), ("port", 3)), lines)  # j is block slot 3

    def test_tau_two_stubs_two_ports(self):
        # term 3: tau(e,f,i,j) -- e,f summed (p stubs), i,j block (h ports 2,3)
        tau_factor = self._term(3).factors[0]
        self.assertEqual(tau_factor.name, "tau")
        lines = factor_to_fragment(tau_factor, ("factor", 0), self.block)
        stubs = {e for sp, _, e in lines if e[0] == "stub"}
        ports = {e for sp, _, e in lines if e[0] == "port"}
        self.assertEqual(stubs, {("stub", "e"), ("stub", "f")})
        self.assertEqual(ports, {("port", 2), ("port", 3)})
        # e,f are virtual -> particle stubs
        self.assertTrue(all(sp == "p" for sp, _, e in lines if e[0] == "stub"))

    def test_interaction_v_ports_and_stubs(self):
        # term 3 factor 1: v(m,n,e,f) -- m,n block (h ports 0,1), e,f summed (p stubs)
        v_factor = self._term(3).factors[1]
        lines = factor_to_fragment(v_factor, ("factor", 1), self.block)
        ports = {e for _, _, e in lines if e[0] == "port"}
        stubs = {e for _, _, e in lines if e[0] == "stub"}
        self.assertEqual(ports, {("port", 0), ("port", 1)})
        self.assertEqual(stubs, {("stub", "e"), ("stub", "f")})


class TermToFragmentTests(unittest.TestCase):
    """D7.1.2: assemble a definition term into a FragmentLineGraph, joining
    summed-index stubs into internal factor<->factor lines. Checked against the
    Wmnij definition terms, including the tau*v term (the D7.1.0 oracle)."""

    def setUp(self):
        self.w = _build_wmnij()
        self.block = self.w.block

    def test_bare_v_term_all_ports(self):
        # term 0: v(m,n,i,j) -> 4 dangling hole ports, no internal lines
        fr = term_to_fragment(self.w.definition_terms[0], self.block)
        self.assertEqual(len(fr.internal_lines), 0)
        self.assertEqual(len(fr.dangling_lines), 4)
        self.assertEqual(fr.port_species, {0: "h", 1: "h", 2: "h", 3: "h"})

    def test_t1v_term_one_internal(self):
        # term 1: t1(e,j) v(m,n,i,e) -> one internal particle line (e), plus
        # dangling ports for j (t1) and m,n,i (v)
        fr = term_to_fragment(self.w.definition_terms[1], self.block)
        self.assertEqual(len(fr.internal_lines), 1)
        self.assertEqual(fr.internal_lines[0][0], "p")   # e is virtual
        self.assertEqual(len(fr.dangling_lines), 4)      # j + m,n,i

    def test_tauv_term_matches_oracle(self):
        # term 3: 1/4 tau(e,f,i,j) v(m,n,e,f) -> the D7.1.0 hand-built fragment:
        # 2 internal particle lines (e,f) + 4 dangling hole ports.
        fr = term_to_fragment(self.w.definition_terms[3], self.block)
        self.assertEqual(len(fr.internal_lines), 2)
        self.assertTrue(all(sp == "p" for sp, _, _ in fr.internal_lines))
        self.assertEqual(fr.port_species, {0: "h", 1: "h", 2: "h", 3: "h"})
        # internal lines both connect tau (factor 0) <-> v (factor 1)
        for sp, a, b in fr.internal_lines:
            self.assertEqual({a, b}, {("factor", 0), ("factor", 1)})

    def test_uncontracted_dummy_raises(self):
        # a definition term with a dangling summed index (appears once) is
        # malformed -- the assembler must reject it rather than emit a half-line.
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import t1
        from ccgen.indices import make_occ, make_vir
        e = make_vir("e", dummy=True)
        m, n, i, j = self.block
        bad = AlgebraTerm(coeff=Fraction(1), factors=(t1(e, j),),
                          free_indices=self.block, summed_indices=(e,),
                          connected=True)
        with self.assertRaises(ValueError):
            term_to_fragment(bad, self.block)


class OperatorFragmentsTests(unittest.TestCase):
    """D7.1.3: every seeded operator encodes to line-graph fragments -- one per
    definition term, with ports matching the operator block. Exercises the whole
    family (Wmnij/Wabef/Wmbej + Fae/Fmi/Fme), so f-tensor factors, tau_tilde, and
    the P(ab) virtual-block operators all pass through the encoder."""

    def test_all_seeded_operators_encode(self):
        for op in seeded_operators():
            of = operator_fragments(op)
            self.assertEqual(of.name, op.name)
            self.assertEqual(of.block, op.block)
            self.assertEqual(of.uses, op.uses)
            # one fragment per defining term
            self.assertEqual(len(of.fragments), len(op.definition_terms))
            for coeff, fr in of.fragments:
                # ports span the whole block, correct arity
                self.assertEqual(fr.n_ports, len(op.block))
                # every block slot is wired by some dangling line
                self.assertEqual(set(fr.port_species), set(range(len(op.block))))
                # port species agree with the block's occ/vir pattern
                for slot, idx in enumerate(op.block):
                    want = "h" if idx.space == "occ" else "p"
                    self.assertEqual(fr.port_species[slot], want,
                                     f"{op.name} slot {slot} species")

    def test_wabef_virtual_block_ports(self):
        # Wabef block = (a,b,e,f) all virtual -> all ports are particle species
        (of,) = [operator_fragments(o) for o in seeded_operators()
                 if o.name == "Wabef"]
        for _, fr in of.fragments:
            self.assertTrue(all(sp == "p" for sp in fr.port_species.values()))


class FragmentFidelityTests(unittest.TestCase):
    """D7.1.4: the fragment encoding is lossless enough for matching -- distinct
    definition terms across the whole seeded family have DISTINCT signatures (no
    false collision).

    This gate DROVE a data-model fix: line topology alone collides t2*v with
    t1*t1*v (both wire identically) -- caught here -- so FragmentLineGraph carries
    factor_names and fragment_signature keys on factor species. Without it a D7.2
    match could mis-recognize one operator term as another."""

    def _all_fragments(self):
        out = []
        for op in seeded_operators():
            for k, (coeff, fr) in enumerate(operator_fragments(op).fragments):
                out.append((op.name, k, fr))
        return out

    def test_factor_names_carried(self):
        # every fragment records the tensor species of each factor node
        for op in seeded_operators():
            for k, (_, fr) in enumerate(operator_fragments(op).fragments):
                self.assertEqual(len(fr.factor_names), fr.n_factors)
                self.assertEqual(
                    fr.factor_names,
                    tuple(f.name for f in op.definition_terms[k].factors))

    def test_signatures_distinct_within_operator(self):
        # within one operator, no two defining terms share a signature (else the
        # match would double-count / mis-attribute a term)
        for op in seeded_operators():
            sigs = [fragment_signature(fr)
                    for _, fr in operator_fragments(op).fragments]
            self.assertEqual(len(sigs), len(set(sigs)),
                             f"{op.name} has colliding definition-term fragments")

    def test_t2v_vs_t1t1v_distinguished(self):
        # the D7.1.4 finding: Wmbej's t2*v and t1*t1*v terms wire identically but
        # must be distinguished by factor species. Their signatures differ.
        wmbej = next(o for o in seeded_operators() if o.name == "Wmbej")
        frags = operator_fragments(wmbej).fragments
        sigs = {frozenset(fr.factor_names): fragment_signature(fr)
                for _, fr in frags}
        self.assertIn(frozenset({"t2", "v"}), sigs)
        self.assertIn(frozenset({"t1", "v"}), sigs)   # the t1,t1,v term
        self.assertNotEqual(sigs[frozenset({"t2", "v"})],
                            sigs[frozenset({"t1", "v"})])


class ResidualTermToFragmentTests(unittest.TestCase):
    """D7.2.0: a residual AlgebraTerm encodes to a FragmentLineGraph via the same
    machinery, with its free indices as ports and summed indices as internal
    lines. The substrate a D7.2 subgraph match runs against."""

    def _find_term(self, shape, v_all_occ=False):
        from ccgen.generate import generate_cc_equations
        for t in generate_cc_equations("ccsd", engine="diagram")["doubles"]:
            if sorted(f.name for f in t.factors) != shape:
                continue
            if v_all_occ:
                vf = [f for f in t.factors if f.name == "v"][0]
                if [i.space for i in vf.indices] != ["occ"] * 4:
                    continue
            return t
        self.fail(f"no doubles term with shape {shape}")

    def test_t2v_term_encodes(self):
        # 1/2 t2(a,b,k,l) v(i,j,k,l): 4 external ports (i,j hole; a,b particle),
        # 2 internal hole lines (k,l joining t2<->v)
        term = self._find_term(["t2", "v"], v_all_occ=True)
        fr = residual_term_to_fragment(term)
        self.assertEqual(fr.n_factors, 2)
        self.assertEqual(fr.n_ports, 4)
        self.assertEqual(fr.factor_names, ("t2", "v"))
        self.assertEqual(len(fr.internal_lines), 2)
        self.assertTrue(all(sp == "h" for sp, _, _ in fr.internal_lines))
        # external ports: i,j occ -> hole; a,b vir -> particle
        self.assertEqual(sorted(fr.port_species.values()), ["h", "h", "p", "p"])

    def test_every_residual_term_encodes(self):
        # the contraction-is-one-edge property holds across the manifold, so no
        # residual term trips the 2-endpoint guard
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        for name in ("singles", "doubles"):
            for t in eqs[name]:
                fr = residual_term_to_fragment(t)   # must not raise
                self.assertEqual(fr.n_factors, len(t.factors))
                self.assertEqual(fr.n_ports, len(t.free_indices))

    def test_bare_v_operator_piece_species_match(self):
        # preview of D7.2.2: the v factor of the t2*v residual term has the same
        # port-species multiset as Wmnij's bare-v defining fragment (all hole).
        term = self._find_term(["t2", "v"], v_all_occ=True)
        # isolate the v factor as its own residual-style fragment
        vf = [f for f in term.factors if f.name == "v"][0]
        from ccgen.project import AlgebraTerm
        v_term = AlgebraTerm(coeff=term.coeff, factors=(vf,),
                             free_indices=vf.indices, summed_indices=(),
                             connected=True)
        v_frag = residual_term_to_fragment(v_term)
        wmnij_bare = operator_fragments(_build_wmnij()).fragments[0][1]  # bare v
        self.assertEqual(sorted(v_frag.port_species.values()),
                         sorted(wmnij_bare.port_species.values()))
        self.assertEqual(v_frag.factor_names, wmnij_bare.factor_names)  # ("v",)


class TauExpandedFragmentsTests(unittest.TestCase):
    """D7.2.1: the operator patterns, tau-expanded to raw tensors -- the set a
    D7.2.2 match searches for in the raw residual (which never carries tau)."""

    def test_no_pseudo_amplitude_survives(self):
        for op in seeded_operators():
            of = tau_expanded_operator_fragments(op)
            self.assertEqual(of.uses, frozenset())
            for _, fr in of.fragments:
                self.assertNotIn("tau", fr.factor_names)
                self.assertNotIn("tau_tilde", fr.factor_names)

    def test_wmnij_tau_split_grew_fragments(self):
        # raw Wmnij has 4 defining terms (one is 1/4 tau v); tau -> t2 + t1t1
        # splits it, so the expanded form has 5.
        raw = operator_fragments(_build_wmnij())
        exp = tau_expanded_operator_fragments(_build_wmnij())
        self.assertEqual(len(raw.fragments), 4)
        self.assertEqual(len(exp.fragments), 5)
        names = sorted(fr.factor_names for _, fr in exp.fragments)
        self.assertIn(("t2", "v"), names)          # from 1/4 tau v
        self.assertIn(("t1", "t1", "v"), names)    # from 1/4 tau v

    def test_expanded_t2v_matches_residual_signature(self):
        # D7.2.2 preview: the tau-expanded Wmnij t2*v fragment has the SAME
        # signature as the residual t2*v term restricted to its Wmnij piece.
        # Both are t2*v with 2 internal hole lines and all-hole ports.
        exp = tau_expanded_operator_fragments(_build_wmnij())
        t2v_op = next(fr for _, fr in exp.fragments
                      if fr.factor_names == ("t2", "v"))
        self.assertEqual(len(t2v_op.internal_lines), 2)
        self.assertTrue(all(sp == "h" for sp in t2v_op.port_species.values()))
        self.assertEqual(sorted(fragment_signature(t2v_op)[0]), ["t2", "v"])


class FindOperatorOccurrencesTests(unittest.TestCase):
    """D7.2.3d: the driver -- enumerate anchors, verify, dedup to maximal covers.
    The payoff: Wmnij is AUTOMATICALLY recognized as 1/2 Wmnij*tau, matching the
    previously hand-transcribed ccsd_dressed_r2 reference."""

    def _doubles(self):
        from ccgen.generate import generate_cc_equations
        return generate_cc_equations("ccsd", engine="diagram")["doubles"]

    def test_recognizes_single_wmnij_tau(self):
        occs = find_operator_occurrences(_build_wmnij(), self._doubles())
        self.assertEqual(len(occs), 1)                 # deduped to one occurrence
        term = occs[0]["term"]
        self.assertEqual(term.coeff, Fraction(1, 2))
        self.assertEqual(sorted(f.name for f in term.factors), ["Wmnij", "tau"])
        # cover is closed under the residual's external-pair antisymmetry
        # (D7.2.5.2 W3): the 10 directly-expanded keys plus the antisym-partner
        # keys the single written t1t1 representative omits.
        self.assertEqual(len(occs[0]["cover"]), 12)

    def test_maximal_cover_beats_partial(self):
        # the partial Wmnij*t2 / Wmnij*t1t1 covers (size 5 each) are subsets of
        # the tau cover (size 10) and must be dropped
        occs = find_operator_occurrences(_build_wmnij(), self._doubles())
        for o in occs:
            rest = [f.name for f in o["term"].factors[1:]]
            self.assertEqual(rest, ["tau"], "a partial rest survived dedup")

    def test_matches_hand_transcribed_reference(self):
        from ccgen.optimization.dressed_equation import ccsd_dressed_r2
        found = find_operator_occurrences(_build_wmnij(), self._doubles())[0]["term"]
        truth = next(t for t in ccsd_dressed_r2()
                     if any(f.name == "Wmnij" for f in t.factors))
        self.assertEqual(found.coeff, truth.coeff)
        self.assertEqual(sorted(f.name for f in found.factors),
                         sorted(f.name for f in truth.factors))


class HypothesisConsistencyTests(unittest.TestCase):
    """D7.2.3c-1: the sound containment filter. A hypothesis is consistent iff
    every expansion primitive is in the residual with matching sign and magnitude
    <= the residual's. The correct Wmnij(k,l,i,j) orientations pass; wrong
    orientations (primitive absent) fail. Exactness is deferred to the
    whole-equation verify (D7.3), so partial-but-consistent hypotheses (rest=t2)
    also pass -- that is sound, not a false accept."""

    def _setup(self):
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        terms = eqs["doubles"]
        op = _build_wmnij()
        anchor = next(o for o in collect_fragment_occurrences(op, terms)
                      if o["frag_id"] == 0)
        return op, anchor, terms[anchor["term_id"]], terms

    def test_correct_hypothesis_passes(self):
        op, anchor, term, terms = self._setup()
        good = [h for h in enumerate_hypotheses(op, anchor, term)
                if hypothesis_is_consistent(h, terms)]
        self.assertTrue(good)
        # every consistent candidate is a correct orientation: bra pair -> summed
        # (k,l), ket pair -> external (i,j).  After the D7.2.5 v-parity sign fold
        # the antisym-equivalent orientation (l,k,j,i) is ALSO sound (the even
        # double-swap of (k,l,i,j)), so assert the SETS, not the slot order;
        # find_operator_occurrences dedups them to one instance.
        for h in good:
            names = [i.name for i in h.factors[0].indices]
            self.assertEqual(set(names[:2]), {"k", "l"})
            self.assertEqual(set(names[2:]), {"i", "j"})
        # the correct tau-rest hypothesis is among them
        self.assertTrue(any(f.name == "tau" for h in good for f in h.factors[1:]))

    def test_wrong_orientation_rejected(self):
        op, anchor, term, terms = self._setup()
        for h in enumerate_hypotheses(op, anchor, term):
            names = [i.name for i in h.factors[0].indices]
            if names == ["i", "j", "k", "l"]:      # the wrong orientation
                self.assertFalse(hypothesis_is_consistent(h, terms))

    def test_filter_is_selective(self):
        # the filter must reject the vast majority (only the 4 antisym-equivalent
        # correct orientations x rests survive of 48 enumerated)
        op, anchor, term, terms = self._setup()
        hyps = list(enumerate_hypotheses(op, anchor, term))
        good = [h for h in hyps if hypothesis_is_consistent(h, terms)]
        self.assertLess(len(good), len(hyps) // 4)


class VParitySignFoldTests(unittest.TestCase):
    """D7.2.5.1: the v-antisymmetry sign fold + antisym-aware dedup. The parity
    primitives are load-bearing (a wrong sign silently rejects correct Fae/Wabef
    hypotheses), and the family sweep is the payoff the unit exists to deliver."""

    def test_perm_parity_of_generators(self):
        # identity even; single transposition odd; double-swap even
        self.assertEqual(_perm_parity((0, 1, 2, 3)), 1)
        self.assertEqual(_perm_parity((1, 0, 2, 3)), -1)   # swap bra
        self.assertEqual(_perm_parity((0, 1, 3, 2)), -1)   # swap ket
        self.assertEqual(_perm_parity((1, 0, 3, 2)), 1)    # both -> even

    def test_antisym_sort_folds_even_double_swap(self):
        # Wmnij(l,k,j,i) is the EVEN double intra-pair swap of (k,l,i,j): sorting
        # each group back must recover (k,l,i,j) with sign +1.
        from ccgen.project import Index, Tensor
        k, l = Index("k", "occ", True), Index("l", "occ", True)
        i, j = Index("i", "occ", False), Index("j", "occ", False)
        w = Tensor("Wmnij", (l, k, j, i), antisym_groups=((0, 1), (2, 3)))
        sf, sign = _antisym_sort_factor(w)
        self.assertEqual([x.name for x in sf.indices], ["k", "l", "i", "j"])
        self.assertEqual(sign, 1)
        # a single intra-pair swap is odd
        w1 = Tensor("Wmnij", (l, k, i, j), antisym_groups=((0, 1), (2, 3)))
        _, s1 = _antisym_sort_factor(w1)
        self.assertEqual(s1, -1)

    def test_family_sweep_unblocks_fae_and_wabef(self):
        # the D7.2.5.1 payoff: Fae/Wabef go from 0 occurrences to nonzero once
        # the v-parity sign is folded; Wmnij stays exactly ONE (dedup guard).
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        n_wmnij = len(find_operator_occurrences(_build_wmnij(), eqs["doubles"]))
        n_fae = len(find_operator_occurrences(_build_fae(), eqs["doubles"]))
        n_wabef = len(find_operator_occurrences(_build_wabef(), eqs["doubles"]))
        self.assertEqual(n_wmnij, 1)
        self.assertGreater(n_fae, 0)
        self.assertGreater(n_wabef, 0)

    def test_wabef_assembles_single_tau_c_occurrence(self):
        # D7.2.5.2 V0.4 payoff: with the tau_c half-weight (W2) and the
        # antisym-partner cover closure (W3), Wabef recognizes as exactly ONE
        # cover-complete occurrence 1/2 Wabef*tau_c -- matching how Wmnij assembles
        # 1/2 Wmnij*tau -- instead of the pre-V0.4 three cover-5 pieces.
        from fractions import Fraction
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        occs = find_operator_occurrences(_build_wabef(), eqs["doubles"])
        self.assertEqual(len(occs), 1)
        term = occs[0]["term"]
        self.assertEqual(term.coeff, Fraction(1, 2))
        self.assertEqual(sorted(f.name for f in term.factors), ["Wabef", "tau_c"])
        # and Wmnij is unchanged -- one occurrence, standard (weight-2) tau
        wmnij = find_operator_occurrences(_build_wmnij(), eqs["doubles"])
        self.assertEqual(len(wmnij), 1)
        self.assertIn("tau", [f.name for f in wmnij[0]["term"].factors])

    def test_fmi_recognized(self):
        # D7.2.5.2 Fmi (gap 4): unblocked by the _eri_canonical ordering fix
        # (fold bra<->ket AFTER dummy relabel, so Fmi's t1(e,n)v(m,n,i,e) piece
        # canonicalizes to the same v orientation as the residual's t1v terms).
        # Fmi recognizes as -Fmi*t1 in singles and the -P(ij) Fmi*t2 pair in
        # doubles (the same legitimate P-antisymmetrizer multiplicity as Fae).
        from fractions import Fraction
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        singles = find_operator_occurrences(_build_fmi(), eqs["singles"])
        doubles = find_operator_occurrences(_build_fmi(), eqs["doubles"])
        self.assertEqual(len(singles), 1)
        self.assertEqual(sorted(f.name for f in singles[0]["term"].factors),
                         ["Fmi", "t1"])
        self.assertEqual(len(doubles), 2)
        for o in doubles:
            self.assertEqual(sorted(f.name for f in o["term"].factors),
                             ["Fmi", "t2"])

    def test_wmbej_recognized(self):
        # D7.2.5.3 Wmbej (gap 3): unblocked by the asymmetric-block binding sign.
        # Wmbej is the only ovvo operator; its genuine block orientations carry
        # the bare-v antisymmetry sign the bare coeff omitted, so every hypothesis
        # was sign-flipped vs the residual (0 occurrences). Applying the binding
        # sign GATED on block asymmetry recovers recognition without disturbing
        # the symmetric-block operators (oooo/vvvv, where signing would over-admit
        # spurious orientations). Wmbej now recognizes as +Wmbej*t1 (singles) and
        # the P(ij)P(ab) Wmbej*t2 quartet (doubles).
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        singles = find_operator_occurrences(_build_wmbej(), eqs["singles"])
        doubles = find_operator_occurrences(_build_wmbej(), eqs["doubles"])
        self.assertEqual(len(singles), 1)
        self.assertEqual(sorted(f.name for f in singles[0]["term"].factors),
                         ["Wmbej", "t1"])
        self.assertEqual(len(doubles), 4)   # P(ij)P(ab) -> 4 branches
        for o in doubles:
            self.assertEqual(sorted(f.name for f in o["term"].factors),
                             ["Wmbej", "t2"])

    def test_full_operator_family_recognized(self):
        # D7.2.5 complete: all six seeded CCSD operators now recognize in the
        # doubles residual (Fmi/Fae also in singles). Guards the whole family
        # against a regression in any one operator's recognition path.
        from ccgen.generate import generate_cc_equations
        from ccgen.optimization.dressing import seeded_operators
        eqs = generate_cc_equations("ccsd", engine="diagram")
        counts = {op.name: len(find_operator_occurrences(op, eqs["doubles"]))
                  for op in seeded_operators()}
        self.assertEqual(counts, {"Fme": 2, "Fae": 2, "Fmi": 2,
                                  "Wmnij": 1, "Wabef": 1, "Wmbej": 4})

    def test_p_branch_consolidation(self):
        # D7.3.0b: an operator's occurrences fold into ONE antisymmetrized dressed
        # term (its P(ij)/P(ab) branches), giving a single per-operator handle for
        # D7.3.0c's cross-operator coefficient reconciliation. Consolidation is
        # lossless: it partitions the occurrences (every branch in exactly one
        # group, no loss/dup) and preserves covers.
        from ccgen.generate import generate_cc_equations
        from ccgen.optimization.dressing import (seeded_operators,
                                                 consolidate_p_branches)
        eqs = generate_cc_equations("ccsd", engine="diagram")
        expected_pairs = {
            "Fme": {("a", "b")}, "Fae": {("a", "b")}, "Fmi": {("i", "j")},
            "Wmnij": set(), "Wabef": set(),
            "Wmbej": {("i", "j"), ("a", "b")},
        }
        for op in seeded_operators():
            occs = find_operator_occurrences(op, eqs["doubles"])
            groups = consolidate_p_branches(op, occs)
            # exactly one group per operator here (all its occurrences are one
            # antisymmetrized term)
            self.assertEqual(len(groups), 1, op.name)
            g = groups[0]
            # lossless partition: branch count == occurrence count
            self.assertEqual(g["branches"], len(occs), op.name)
            pairs = {(x.name, y.name) for x, y in g["antisym_pairs"]}
            self.assertEqual(pairs, expected_pairs[op.name], op.name)
            # cover preserved
            occ_cover = frozenset().union(*(o["cover"] for o in occs))
            self.assertEqual(g["cover"], occ_cover, op.name)

    def test_nesting_scale_reconciliation(self):
        # D7.3.0c-1: the Fme nesting scale is DERIVED (not hardcoded) as the
        # complement of the -1/2 f*t1 Fme-correction that Fae/Fmi already carry.
        # Fme is over-counted standalone; its correct scale is 1/2, uniquely
        # consistent across all its keys. Applying it drops the recognition-recon
        # over-count 24 -> 20 (the 4 Fme/Fae keys close; the 4 Fmi tau~-tail keys
        # (0d) and 2 Wabef/Wmnij tau-overlap keys (0c-2) plus the 14 uncovered
        # remainder stay).
        from fractions import Fraction
        from ccgen.generate import generate_cc_equations
        from ccgen.optimization.dressing import (seeded_operators,
            reconcile_operator_scales, _operator_unit_expansion)
        from ccgen.optimization.dressed_equation import raw_multiset
        eqs = generate_cc_equations("ccsd", engine="diagram")
        terms = eqs["doubles"]
        ops = seeded_operators()
        scale = reconcile_operator_scales(ops, terms)
        self.assertEqual(scale["Fme"], Fraction(1, 2))
        for n in ("Fae", "Fmi", "Wmnij", "Wabef", "Wmbej"):
            self.assertEqual(scale[n], Fraction(1), n)
        # applying the scales reduces the over-count 24 -> 20
        raw = raw_multiset(terms)
        units = {op.name: _operator_unit_expansion(op, terms) for op in ops}
        recon: dict = {}
        for n, u in units.items():
            for k, c in u.items():
                recon[k] = recon.get(k, Fraction(0)) + c * scale[n]
        mism = sum(1 for k in set(recon) | set(raw)
                   if recon.get(k, Fraction(0)) != raw.get(k, Fraction(0)))
        self.assertEqual(mism, 20)

    def test_tau_overlap_correction(self):
        # D7.3.0c-2: the {Wabef,Wmnij} tau/tau_c overlap. On a primitive shared
        # between the external-tau operator (Wmnij, weight 2) and the contracted
        # tau_c operator (Wabef, weight 1), the t1t1-half is over-counted -- the
        # tau contribution is exactly DOUBLE the tau_c one and raw == tau, so the
        # external-tau operator owns it and the tau_c duplicate is subtracted. The
        # genuinely-additive shared keys (ratio 1) are untouched. Applying 0c-1 +
        # 0c-2 drops the over-count 24 -> 18 (leaving 14 uncovered + 4 Fmi/0d).
        from fractions import Fraction
        from ccgen.generate import generate_cc_equations
        from ccgen.optimization.dressing import (seeded_operators,
            reconcile_operator_scales, tau_overlap_corrections,
            _operator_unit_expansion)
        from ccgen.optimization.dressed_equation import raw_multiset
        eqs = generate_cc_equations("ccsd", engine="diagram")
        terms = eqs["doubles"]
        ops = seeded_operators()
        scale = reconcile_operator_scales(ops, terms)
        corr = tau_overlap_corrections(ops, terms, scale)
        # exactly 2 over keys, each subtracting the tau_c duplicate (negative)
        self.assertEqual(len(corr), 2)
        self.assertTrue(all(v < 0 for v in corr.values()))
        raw = raw_multiset(terms)
        units = {op.name: _operator_unit_expansion(op, terms) for op in ops}
        recon: dict = {}
        for n, u in units.items():
            for k, c in u.items():
                recon[k] = recon.get(k, Fraction(0)) + c * scale[n]
        for k, d in corr.items():
            recon[k] = recon.get(k, Fraction(0)) + d
        mism = sum(1 for k in set(recon) | set(raw)
                   if recon.get(k, Fraction(0)) != raw.get(k, Fraction(0)))
        self.assertEqual(mism, 18)

    def test_canonical_fock_recon_is_exact_partition(self):
        # D7.3.0 payoff: Planck feeds only a CANONICAL Fock (f_ov=0 by
        # construction), so the assembly target is the canonical-Fock diagram
        # residual. Against it, 0c-1 + 0c-2 reconstruct EXACTLY -- zero
        # non-uncovered mismatches -- with no 0d work needed: the former 4 real
        # general-Fock mismatches were all f_ov-entangled (2 direct f_ov terms,
        # 2 tau~ t1t1*t2*v keys) and dissolve in canonical mode. The only residual
        # mismatches are the legitimate uncovered remainder (stay as bare terms)
        # plus dressed-only f_ov keys the operators still expand but the canonical
        # raw drops -- both physically inert.
        from fractions import Fraction
        from ccgen.generate import generate_cc_equations
        from ccgen.optimization.dressing import (seeded_operators,
            reconcile_operator_scales, tau_overlap_corrections,
            _operator_unit_expansion)
        from ccgen.optimization.dressed_equation import raw_multiset

        def has_fov(key):
            for name, idx in key[0]:
                spaces = {s[0] for s in idx}
                if name == "f" and spaces == {"occ", "vir"}:
                    return True
            return False

        eqs = generate_cc_equations("ccsd", engine="diagram", canonical_fock=True)
        terms = eqs["doubles"]
        ops = seeded_operators()
        scale = reconcile_operator_scales(ops, terms)
        corr = tau_overlap_corrections(ops, terms, scale)
        raw = raw_multiset(terms)
        units = {op.name: _operator_unit_expansion(op, terms) for op in ops}
        recon: dict = {}
        for n, u in units.items():
            for k, c in u.items():
                recon[k] = recon.get(k, Fraction(0)) + c * scale[n]
        for k, d in corr.items():
            recon[k] = recon.get(k, Fraction(0)) + d
        # non-uncovered, non-f_ov mismatches must be ZERO: the exact partition
        real = [k for k in set(recon) | set(raw)
                if recon.get(k, Fraction(0)) != raw.get(k, Fraction(0))
                and recon.get(k, Fraction(0)) != 0
                and not has_fov(k)]
        self.assertEqual(real, [])


class EnumerateHypothesesTests(unittest.TestCase):
    """D7.2.3c-0: a single anchor underdetermines the hypothesis, so enumerate
    {block orientation} x {rest as-is, rest-as-tau}. The CORRECT Wmnij*tau
    (block (m,n,i,j)->(k,l,i,j), rest tau) must be among the candidates -- its
    expansion is fully present in the residual, while the wrong orientations /
    raw-t2 rest are not."""

    def _setup(self):
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        terms = eqs["doubles"]
        op = _build_wmnij()
        occ = collect_fragment_occurrences(op, terms)
        anchor = next(o for o in occ if o["frag_id"] == 0)  # bare-v
        return op, anchor, terms[anchor["term_id"]], terms

    def _fully_present(self, hyp, terms):
        from fractions import Fraction
        from ccgen.optimization.dressed_equation import (expand_dressed_term,
                                                         raw_multiset)
        from ccgen.optimization.dressing import _eri_canonical
        raw = raw_multiset(terms)
        hm = {}
        for e in expand_dressed_term(hyp, {"Wmnij": _build_wmnij()}):
            k, c = _eri_canonical(e)
            hm[k] = hm.get(k, Fraction(0)) + c
        return all(k in raw for k in hm), len(hm)

    def test_enumeration_includes_correct_hypothesis(self):
        op, anchor, term, terms = self._setup()
        hyps = list(enumerate_hypotheses(op, anchor, term))
        # some candidate expands fully into the residual (the correct Wmnij*tau)
        good = [h for h in hyps if self._fully_present(h, terms)[0]]
        self.assertTrue(good, "no fully-present hypothesis enumerated")
        # the correct one has a tau rest and 10 expansion keys
        best = max(good, key=lambda h: self._fully_present(h, terms)[1])
        self.assertEqual(self._fully_present(best, terms)[1], 10)
        self.assertTrue(any(f.name == "tau" for f in best.factors))

    def test_both_orientations_and_rests_enumerated(self):
        op, anchor, term, terms = self._setup()
        hyps = list(enumerate_hypotheses(op, anchor, term))
        rests = {tuple(sorted(f.name for f in h.factors[1:])) for h in hyps}
        self.assertIn(("t2",), rests)       # raw rest
        self.assertIn(("tau",), rests)      # dressed rest
        # multiple distinct block orientations
        orients = {tuple(i.name for i in h.factors[0].indices) for h in hyps}
        self.assertGreater(len(orients), 1)

    def test_wrong_orientation_not_fully_present(self):
        # the orientation match_fragment originally returned (i,j,k,l) is NOT a
        # full collapse -- confirms the enumeration was necessary
        op, anchor, term, terms = self._setup()
        for h in enumerate_hypotheses(op, anchor, term):
            names = [i.name for i in h.factors[0].indices]
            if names == ["i", "j", "k", "l"] and h.factors[1].name == "t2":
                self.assertFalse(self._fully_present(h, terms)[0])
                return
        self.fail("expected the (i,j,k,l)+t2 candidate in the enumeration")


class HypothesizeOperatorTermTests(unittest.TestCase):
    """D7.2.3b: build the dressed W*rest term implied by one anchor fragment
    match. The coefficient divides out the anchor's operator-internal coeff; the
    W factor carries the residual indices its block bound to; rest = the factors
    outside the anchor subset."""

    def _wmnij_bare_v_anchor(self):
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        terms = eqs["doubles"]
        op = _build_wmnij()
        occ = collect_fragment_occurrences(op, terms)
        # the bare-v anchor (frag_id 0) inside 1/2 t2 v
        anchor = next(o for o in occ if o["frag_id"] == 0)
        return op, anchor, terms[anchor["term_id"]]

    def test_hypothesis_is_W_times_rest(self):
        op, anchor, term = self._wmnij_bare_v_anchor()
        hyp = hypothesize_operator_term(op, anchor, term)
        names = [f.name for f in hyp.factors]
        self.assertEqual(names[0], "Wmnij")
        self.assertIn("t2", names[1:])                 # the outer t2 rest
        # coeff = term_coeff / anchor op_coeff (bare v op_coeff = 1)
        self.assertEqual(hyp.coeff, term.coeff / anchor["op_coeff"])

    def test_W_block_indices_from_binding(self):
        op, anchor, term = self._wmnij_bare_v_anchor()
        hyp = hypothesize_operator_term(op, anchor, term)
        w = hyp.factors[0]
        bound = [anchor["port_index"][s] for s in range(len(op.block))]
        self.assertEqual([i.name for i in w.indices], bound)

    def test_expands_to_operator_definition(self):
        from ccgen.optimization.dressed_equation import expand_dressed_term
        op, anchor, term = self._wmnij_bare_v_anchor()
        hyp = hypothesize_operator_term(op, anchor, term)
        expanded = expand_dressed_term(hyp, {op.name: op})
        # Wmnij (tau-expanded) has 5 raw pieces
        self.assertEqual(len(expanded), 5)
        # no operator or pseudo-amplitude factor survives the expansion
        for e in expanded:
            for f in e.factors:
                self.assertIn(f.name, ("t1", "t2", "v", "f"))


class CollectFragmentOccurrencesTests(unittest.TestCase):
    """D7.2.3a: fan the fragment matcher out over the whole residual, per operator
    fragment. Each occurrence carries the fragment/term coefficients + binding
    needed to group into whole-operator instances (D7.2.3c) and rewrite (D7.3)."""

    def test_wmnij_covers_all_fragments(self):
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        occ = collect_fragment_occurrences(_build_wmnij(), eqs["doubles"])
        n_frags = len(tau_expanded_operator_fragments(_build_wmnij()).fragments)
        covered = {o["frag_id"] for o in occ}
        # every tau-expanded Wmnij fragment has at least one occurrence
        self.assertEqual(covered, set(range(n_frags)))

    def test_records_carry_coeffs_and_binding(self):
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        occ = collect_fragment_occurrences(_build_wmnij(), eqs["doubles"])
        self.assertTrue(occ)
        for o in occ:
            self.assertIn("op_coeff", o)
            self.assertIn("term_coeff", o)
            self.assertIn("term_id", o)
            self.assertIn("subset", o)
            self.assertIn("port_index", o)
            # port_index binds every operator block slot
            self.assertEqual(set(o["port_index"]),
                             set(range(len(_build_wmnij().block))))


class MatchFragmentTests(unittest.TestCase):
    """D7.2.2d: the match driver -- every occurrence of an operator fragment in a
    residual term, with the residual index each port bound to (for D7.2.3)."""

    def _doubles_term(self, shape, v_all_occ=False):
        from ccgen.generate import generate_cc_equations
        for t in generate_cc_equations("ccsd", engine="diagram")["doubles"]:
            if sorted(f.name for f in t.factors) != shape:
                continue
            if v_all_occ:
                vf = [f for f in t.factors if f.name == "v"][0]
                if [i.space for i in vf.indices] != ["occ"] * 4:
                    continue
            return t
        self.fail(f"no doubles term {shape}")

    def test_bare_v_occurrence_binds_indices(self):
        term = self._doubles_term(["t2", "v"], v_all_occ=True)  # 1/2 t2 v(i,j,k,l)
        bare = operator_fragments(_build_wmnij()).fragments[0][1]
        occ = match_fragment(bare, term)
        self.assertEqual(len(occ), 1)
        o = occ[0]
        # subset is the v factor; ports bind to v's four occ indices
        self.assertEqual([term.factors[k].name for k in o["subset"]], ["v"])
        self.assertEqual(set(o["port_index"].values()),
                         {i.name for i in term.factors[o["subset"][0]].indices})

    def test_t2v_operator_no_false_occurrence(self):
        # the t2*v operator fragment finds NO occurrence in t2*t2*v (extra l-line)
        term = self._doubles_term(["t2", "t2", "v"])
        op = next(fr for _, fr in
                  tau_expanded_operator_fragments(_build_wmnij()).fragments
                  if fr.factor_names == ("t2", "v"))
        self.assertEqual(match_fragment(op, term), [])

    def test_no_crash_over_full_manifold(self):
        # the driver runs cleanly over every doubles term for every Wmnij fragment
        from ccgen.generate import generate_cc_equations
        eqs = generate_cc_equations("ccsd", engine="diagram")
        frags = [fr for _, fr in
                 tau_expanded_operator_fragments(_build_wmnij()).fragments]
        for t in eqs["doubles"]:
            for fr in frags:
                for o in match_fragment(fr, t):
                    # every reported occurrence has a full port binding
                    self.assertEqual(set(o["port_index"]),
                                     set(range(fr.n_ports)))


class FragmentsMatchTests(unittest.TestCase):
    """D7.2.2c: the exact induced-sub-fragment isomorphism test -- the core of
    D7.2. A match is a species-consistent node bijection carrying the operator's
    internal lines and ports onto the induced ones exactly. The load-bearing
    property: an extra induced contraction line (the l-line) blocks the match."""

    def _doubles_term(self, shape, v_all_occ=False):
        from ccgen.generate import generate_cc_equations
        for t in generate_cc_equations("ccsd", engine="diagram")["doubles"]:
            if sorted(f.name for f in t.factors) != shape:
                continue
            if v_all_occ:
                vf = [f for f in t.factors if f.name == "v"][0]
                if [i.space for i in vf.indices] != ["occ"] * 4:
                    continue
            return t
        self.fail(f"no doubles term {shape}")

    def test_bare_v_matches_with_binding(self):
        term = self._doubles_term(["t2", "v"], v_all_occ=True)
        v_pos = [k for k, f in enumerate(term.factors) if f.name == "v"][0]
        ind = induced_subfragment(term, (v_pos,))
        bare = operator_fragments(_build_wmnij()).fragments[0][1]
        b = fragments_match(bare, ind)
        self.assertIsNotNone(b)
        self.assertEqual(b["nodes"], {("factor", 0): ("factor", 0)})
        self.assertEqual(set(b["ports"]), {0, 1, 2, 3})

    def test_extra_line_blocks_match(self):
        # the t2*v operator fragment (2 internal particle lines) must NOT match
        # any t2*t2*v subset -- one shares 1 line, the other shares 3 (l-line).
        term = self._doubles_term(["t2", "t2", "v"])
        op = next(fr for _, fr in
                  tau_expanded_operator_fragments(_build_wmnij()).fragments
                  if fr.factor_names == ("t2", "v"))
        for s in candidate_factor_subsets(op, term):
            ind = induced_subfragment(term, s)
            self.assertIsNone(fragments_match(op, ind),
                              f"subset {s} wrongly matched t2*v operator")

    def test_species_mismatch_no_match(self):
        # a t1*v operator fragment cannot match a t2*v induced fragment even if
        # both had one internal line -- factor species disagree
        term = self._doubles_term(["t2", "v"], v_all_occ=True)
        ind = induced_subfragment(term, tuple(range(len(term.factors))))
        t1v = next(fr for _, fr in operator_fragments(_build_wmnij()).fragments
                   if fr.factor_names == ("t1", "v"))
        self.assertIsNone(fragments_match(t1v, ind))


class InducedSubfragmentTests(unittest.TestCase):
    """D7.2.2b: the sub-fragment a residual factor subset induces. Within-subset
    shared indices are internal lines; outward/external indices are ports. The
    load-bearing case: a subset that shares MORE lines than the operator (the
    l-line) must show the extra internal line so D7.2.2c can reject it."""

    def _doubles_term(self, shape, v_all_occ=False):
        from ccgen.generate import generate_cc_equations
        for t in generate_cc_equations("ccsd", engine="diagram")["doubles"]:
            if sorted(f.name for f in t.factors) != shape:
                continue
            if v_all_occ:
                vf = [f for f in t.factors if f.name == "v"][0]
                if [i.space for i in vf.indices] != ["occ"] * 4:
                    continue
            return t
        self.fail(f"no doubles term {shape}")

    def test_bare_v_subset_matches_operator(self):
        term = self._doubles_term(["t2", "v"], v_all_occ=True)
        v_pos = [k for k, f in enumerate(term.factors) if f.name == "v"][0]
        fr = induced_subfragment(term, (v_pos,))
        bare = operator_fragments(_build_wmnij()).fragments[0][1]
        self.assertEqual(len(fr.internal_lines), 0)
        self.assertEqual(fragment_signature(fr), fragment_signature(bare))

    def test_extra_line_subset_has_extra_internal(self):
        # t2(c,d,j,l) v(c,d,k,l) shares c,d AND l -> 3 internal lines (2 p + 1 h),
        # more than the operator t2*v's 2 -> must NOT match (rejected in D7.2.2c).
        term = self._doubles_term(["t2", "t2", "v"])
        # the t2 that shares c,d,l with v (the induced 3-internal case)
        subs = candidate_factor_subsets(
            next(fr for _, fr in
                 tau_expanded_operator_fragments(_build_wmnij()).fragments
                 if fr.factor_names == ("t2", "v")), term)
        found_extra = False
        for s in subs:
            fr = induced_subfragment(term, s)
            if len(fr.internal_lines) == 3:
                found_extra = True
                species = sorted(sp for sp, _, _ in fr.internal_lines)
                self.assertEqual(species, ["h", "p", "p"])
        self.assertTrue(found_extra, "expected an extra-line induced subset")

    def test_node_renumbering_is_local(self):
        # induced factor nodes are 0..n-1 by subset position, so the result is
        # directly comparable to an operator fragment
        term = self._doubles_term(["t2", "t2", "v"])
        fr = induced_subfragment(term, (1, 2))
        nodes = {e for _, a, b in fr.lines for e in (a, b)
                 if isinstance(e, tuple) and e[0] == "factor"}
        self.assertEqual(nodes, {("factor", 0), ("factor", 1)})


class CandidateSubsetsTests(unittest.TestCase):
    """D7.2.2a: the factor-name prefilter -- only residual factor subsets whose
    name multiset equals the operator fragment's survive."""

    def _doubles_term(self, shape):
        from ccgen.generate import generate_cc_equations
        for t in generate_cc_equations("ccsd", engine="diagram")["doubles"]:
            if sorted(f.name for f in t.factors) == shape:
                return t
        self.fail(f"no doubles term {shape}")

    def test_t2v_op_on_t2t2v_term(self):
        # residual t2*t2*v: a t2*v operator fragment can pair EITHER t2 with the
        # v -> exactly 2 candidate subsets; never the (t2,t2) pair.
        term = self._doubles_term(["t2", "t2", "v"])
        # Wmnij's raw t2*v piece only exists after tau expansion (D7.2.1)
        op = next(fr for _, fr in
                  tau_expanded_operator_fragments(_build_wmnij()).fragments
                  if fr.factor_names == ("t2", "v"))
        subs = candidate_factor_subsets(op, term)
        self.assertEqual(len(subs), 2)
        # each surviving subset is one t2 + the v
        v_pos = [k for k, f in enumerate(term.factors) if f.name == "v"][0]
        for s in subs:
            self.assertIn(v_pos, s)
            self.assertEqual(len(s), 2)

    def test_bare_v_op_matches_the_v(self):
        term = self._doubles_term(["t2", "v"])
        bare_v = operator_fragments(_build_wmnij()).fragments[0][1]  # ("v",)
        subs = candidate_factor_subsets(bare_v, term)
        self.assertEqual(len(subs), 1)
        self.assertEqual([term.factors[k].name for k in subs[0]], ["v"])

    def test_t1v_op_no_match_on_t2v_term(self):
        term = self._doubles_term(["t2", "v"])
        t1v = next(fr for _, fr in operator_fragments(_build_wmnij()).fragments
                   if fr.factor_names == ("t1", "v"))
        self.assertEqual(candidate_factor_subsets(t1v, term), [])


if __name__ == "__main__":
    unittest.main()
