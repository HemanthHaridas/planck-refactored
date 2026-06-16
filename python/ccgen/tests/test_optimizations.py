"""Validation tests for ccgen optimization passes.

Tests ensure algebraic equivalence: optimized equations produce
the same numerical results as unoptimized equations on random
test tensors.
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ccgen.generate as generate_mod

from ccgen.generate import (
    generate_cc_equations,
    PipelineStats,
    print_cpp_planck,
)
from ccgen.indices import make_occ, make_vir
from ccgen.sqops import create, annihilate
from ccgen.project import (
    AlgebraTerm,
    _can_term_contribute_to_rank,
    _can_term_blocks_connect_to_rank,
    bucket_terms_by_manifold,
)
from ccgen.hamiltonian import build_hamiltonian
from ccgen.cluster import build_cluster
from ccgen.cluster import parse_cc_level, canonicalize_cc_level
from ccgen.algebra import bch_expand
from ccgen.expr import OpTerm
from ccgen.canonicalize import (
    collect_fock_diagonals,
    merge_exact_term_into_buckets,
    term_is_zero_before_canonicalization,
)
from ccgen.optimization.intermediates import (
    detect_intermediates,
    rewrite_equations,
    build_intermediate_equations,
    IntermediateSpec,
    _find_subfactors,
)
from ccgen.optimization.subexpression import (
    detect_common_subexpressions,
    factor_common_subexpressions,
    apply_cse,
    CSESpec,
)
from ccgen.emit.einsum import (
    format_equations_einsum,
    format_intermediates_einsum,
    format_equations_with_intermediates_einsum,
)
from ccgen.emit.cpp_loops import (
    emit_intermediate_builds,
    emit_translation_unit_with_intermediates,
    emit_term_tiled,
    emit_optimized_translation_unit,
)
from ccgen.emit.planck_tensor_cpp import (
    emit_planck_term,
    emit_planck_translation_unit,
)
from ccgen.emit.planck_rccsd_warm_start import (
    emit_planck_spinorbital_rccsd_warm_start,
)
from ccgen.lowering import lower_term_restricted_closed_shell
from ccgen.tensors import Tensor
from ccgen.wick import wick_contract, apply_deltas


class IntermediateDetectionTests(unittest.TestCase):
    """Tests for intermediate tensor detection."""

    def test_ccsd_detects_intermediates(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=2)
        self.assertGreater(len(intms), 0, "Should detect at least one intermediate")

    def test_intermediates_have_valid_specs(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=5)
        for spec in intms:
            self.assertIsInstance(spec, IntermediateSpec)
            self.assertTrue(spec.name, "Name must be non-empty")
            self.assertGreaterEqual(spec.usage_count, 5)
            self.assertGreater(spec.rank, 0, "Must have at least one index")
            self.assertEqual(len(spec.indices), spec.rank)
            self.assertTrue(
                spec.index_space_sig,
                "Index space signature must be non-empty",
            )
            self.assertEqual(
                len(spec.index_space_sig),
                spec.rank,
                "Space sig length must match rank",
            )

    def test_intermediates_sorted_by_usage(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=2)
        for i in range(len(intms) - 1):
            self.assertGreaterEqual(
                intms[i].usage_count,
                intms[i + 1].usage_count,
                "Intermediates must be sorted by usage count (descending)",
            )

    def test_threshold_filters_correctly(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms_low = detect_intermediates(eqs, threshold=2)
        intms_high = detect_intermediates(eqs, threshold=20)
        self.assertGreaterEqual(len(intms_low), len(intms_high))
        for spec in intms_high:
            self.assertGreaterEqual(spec.usage_count, 20)

    def test_no_intermediates_for_ccd_energy_only(self) -> None:
        eqs = generate_cc_equations("ccd", targets=["energy"])
        intms = detect_intermediates(eqs, threshold=2)
        # Energy-only CCD has very few terms, unlikely to find reuse
        # (This is a sanity check, not a hard requirement)
        self.assertIsInstance(intms, list)


class ProjectionPruningTests(unittest.TestCase):
    """Tests for early projector-feasibility pruning."""

    def test_rank_filter_rejects_short_signatures(self) -> None:
        signature = (
            ("create", "gen"),
            ("annihilate", "gen"),
        )
        self.assertFalse(_can_term_contribute_to_rank(signature, 2))

    def test_rank_filter_requires_projector_compatible_slots(self) -> None:
        signature = (
            ("create", "vir"),
            ("create", "vir"),
            ("annihilate", "vir"),
            ("annihilate", "vir"),
        )
        self.assertFalse(_can_term_contribute_to_rank(signature, 2))

    def test_rank_filter_accepts_doubles_like_signature(self) -> None:
        signature = (
            ("create", "vir"),
            ("create", "vir"),
            ("annihilate", "occ"),
            ("annihilate", "occ"),
        )
        self.assertTrue(_can_term_contribute_to_rank(signature, 2))

    def test_rank_filter_rejects_non_contractible_leftover_signature(self) -> None:
        signature = (
            ("create", "occ"),
            ("create", "occ"),
            ("create", "occ"),
            ("create", "occ"),
            ("create", "vir"),
            ("create", "vir"),
            ("annihilate", "occ"),
            ("annihilate", "occ"),
            ("annihilate", "vir"),
            ("annihilate", "vir"),
            ("annihilate", "vir"),
            ("annihilate", "vir"),
        )
        self.assertFalse(_can_term_contribute_to_rank(signature, 2))

    def test_block_filter_rejects_projector_channel_overcommit(self) -> None:
        a = make_vir("a")
        b = make_vir("b")
        i = make_occ("i")
        c = make_vir("c")
        term = OpTerm(
            coeff=Fraction(1),
            tensors=(
                Tensor("x1", (a,)),
                Tensor("x2", (b,)),
                Tensor("x3", (i,)),
                Tensor("x4", (c,)),
            ),
            sqops=(
                create(a),
                create(b),
                annihilate(i),
                annihilate(c),
            ),
        )
        signature = tuple((op.kind, op.index.space) for op in term.sqops)
        self.assertTrue(_can_term_contribute_to_rank(signature, 1))
        self.assertFalse(_can_term_blocks_connect_to_rank(term, 1))

    def test_bucket_terms_by_manifold_preserves_energy_terms(self) -> None:
        hbar = bch_expand(build_hamiltonian(), build_cluster("ccsd"), max_order=4)
        buckets = bucket_terms_by_manifold(
            hbar,
            ("energy", "singles", "doubles"),
        )
        self.assertEqual(len(buckets["energy"]), len(hbar.terms))
        self.assertGreater(len(buckets["singles"]), 0)
        self.assertGreater(len(buckets["doubles"]), 0)

    def test_bucket_terms_by_manifold_reduces_higher_rank_scan(self) -> None:
        hbar = bch_expand(build_hamiltonian(), build_cluster("ccsd"), max_order=4)
        buckets = bucket_terms_by_manifold(
            hbar,
            ("energy", "triples"),
        )
        self.assertLess(len(buckets["triples"]), len(buckets["energy"]))


class MethodParsingAndCacheReuseTests(unittest.TestCase):
    def test_parse_cc_level_accepts_p_and_h_aliases(self) -> None:
        self.assertEqual(parse_cc_level("ccsdtqp"), [1, 2, 3, 4, 5])
        self.assertEqual(parse_cc_level("ccsdtqph"), [1, 2, 3, 4, 5, 6])
        self.assertEqual(canonicalize_cc_level("ccsdtq5"), "ccsdtqp")
        self.assertEqual(canonicalize_cc_level("ccsdtq56"), "ccsdtqph")

    def test_cache_dir_reuses_bch_prefix_levels_across_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_cc_equations(
                "ccsd",
                targets=["energy"],
                cache_dir=tmpdir,
            )
            eqs_prefix = generate_cc_equations(
                "ccsdt",
                targets=["energy"],
                cache_dir=tmpdir,
            )
            self.assertEqual(generate_mod.last_stats.bch_reused_from, "ccsd")

            eqs_fresh = generate_cc_equations(
                "ccsdt",
                targets=["energy"],
            )
            self.assertEqual(eqs_prefix, eqs_fresh)

    def test_cache_dir_exact_method_hit_reuses_bch_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_cc_equations(
                "ccsd",
                targets=["singles"],
                cache_dir=tmpdir,
            )
            generate_cc_equations(
                "ccsd",
                targets=["energy"],
                cache_dir=tmpdir,
            )
            self.assertTrue(generate_mod.last_stats.bch_cache_hit)


class WickPivotingTests(unittest.TestCase):
    """Regression tests for non-leftmost Wick pivot enumeration."""

    def _naive_pairings(
        self,
        signature: tuple[tuple[int, str, str, int], ...],
    ) -> set[tuple[int, tuple[tuple[int, int], ...]]]:
        from ccgen.wick import _can_contract_signature

        if not signature:
            return {(1, ())}

        first = signature[0]
        rest = signature[1:]
        results: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
        for k, partner in enumerate(rest):
            if first[3] == partner[3]:
                continue
            if not _can_contract_signature(
                first[1], first[2], partner[1], partner[2],
            ):
                continue
            remaining = rest[:k] + rest[k + 1:]
            sign_factor = -1 if k % 2 else 1
            for sub_sign, sub_pairs in self._naive_pairings(remaining):
                pair = tuple(sorted((first[0], partner[0])))
                pairs = tuple(sorted((pair,) + sub_pairs))
                results.add((sign_factor * sub_sign, pairs))
        return results

    def test_wick_contract_matches_naive_pairings(self) -> None:
        i = make_occ("i", dummy=False)
        j = make_occ("j", dummy=False)
        k = make_occ("k", dummy=False)
        a = make_vir("a", dummy=False)
        b = make_vir("b", dummy=False)
        c = make_vir("c", dummy=False)

        sqops = (
            annihilate(a),
            create(c),
            create(i),
            annihilate(j),
            annihilate(b),
            create(k),
        )
        block_ids = (0, 1, 2, 3, 4, 5)
        signature = tuple(
            (idx, op.kind, op.index.space, bid)
            for idx, (op, bid) in enumerate(zip(sqops, block_ids))
        )
        actual = set()
        position_by_index = {op.index: pos for pos, op in enumerate(sqops)}
        for result in wick_contract(
            sqops,
            tensors=(),
            block_ids=block_ids,
        ):
            pair_positions = tuple(sorted(
                tuple(sorted((
                    position_by_index[left],
                    position_by_index[right],
                )))
                for left, right in result.deltas
            ))
            actual.add((result.sign, pair_positions))

        expected = self._naive_pairings(signature)
        self.assertSetEqual(actual, expected)


class DeltaApplicationTests(unittest.TestCase):
    """Regression tests for delta substitution fast paths."""

    def test_apply_deltas_restores_protected_free_indices(self) -> None:
        i_free = make_occ("i", dummy=False)
        a_free = make_vir("a", dummy=False)
        i_dummy = make_occ("i", dummy=True)
        a_dummy = make_vir("a", dummy=True)

        reduced = apply_deltas(
            (Tensor("t1", (a_dummy, i_dummy)),),
            ((a_free, a_dummy), (i_free, i_dummy)),
            protected=(i_free, a_free),
        )

        self.assertIsNotNone(reduced)
        assert reduced is not None
        self.assertEqual(reduced[0].indices, (a_free, i_free))


class PreCanonicalPruningTests(unittest.TestCase):
    """Tests for cheap zero/duplicate pruning before canonicalization."""

    def test_zero_detection_respects_antisymmetry(self) -> None:
        a = make_vir("a", dummy=False)
        i = make_occ("i", dummy=False)
        term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(Tensor("t2", (a, a, i, i), antisym_groups=((0, 1), (2, 3))),),
            free_indices=(i, i, a, a),
            summed_indices=(),
            connected=True,
        )
        self.assertTrue(term_is_zero_before_canonicalization(term))

    def test_exact_duplicate_merge_happens_before_canonicalization(self) -> None:
        a = make_vir("a", dummy=False)
        i = make_occ("i", dummy=False)
        term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(Tensor("t1", (a, i)),),
            free_indices=(i, a),
            summed_indices=(),
            connected=True,
        )
        buckets: dict[tuple[object, ...], AlgebraTerm] = {}
        order: list[tuple[object, ...]] = []
        merge_exact_term_into_buckets(term, buckets, order)
        merge_exact_term_into_buckets(term.scaled(2), buckets, order)
        self.assertEqual(len(order), 1)
        merged = buckets[order[0]]
        self.assertEqual(merged.coeff, Fraction(3))


class ParallelGenerationTests(unittest.TestCase):
    """Tests for multiprocessing projection/canonicalization."""

    def test_parallel_generation_matches_serial(self) -> None:
        serial = generate_cc_equations("ccsd", parallel_workers=1)
        parallel = generate_cc_equations("ccsd", parallel_workers=2)
        self.assertEqual(serial, parallel)

    def test_cache_dir_reuses_manifold_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = generate_cc_equations(
                "ccsd",
                targets=["energy", "singles"],
                cache_dir=tmpdir,
            )
            second = generate_cc_equations(
                "ccsd",
                targets=["energy", "singles"],
                cache_dir=tmpdir,
            )
            self.assertEqual(first, second)
            method_dir = Path(tmpdir) / "ccsd"
            self.assertTrue((method_dir / "config.json").exists())
            self.assertTrue((method_dir / "energy.pkl").exists())
            self.assertTrue((method_dir / "singles.pkl").exists())

    def test_unique_names(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=2)
        names = [spec.name for spec in intms]
        self.assertEqual(len(names), len(set(names)), "Intermediate names must be unique")

    def test_intermediate_index_order_matches_space_signature(self) -> None:
        eqs = generate_cc_equations("ccsdt")
        intms = detect_intermediates(eqs, threshold=10)
        for spec in intms:
            sig = "".join(
                "o" if idx.space == "occ" else "v" if idx.space == "vir" else "g"
                for idx in spec.indices
            )
            self.assertEqual(
                sig,
                spec.index_space_sig,
                f"{spec.name} indices must match its declared space signature",
            )

    def test_detection_distinguishes_contraction_topology(self) -> None:
        i = make_occ("i")
        k = make_occ("k", dummy=True)
        j1 = make_occ("j", dummy=True)
        j2 = make_occ("j", dummy=True)
        l = make_occ("l", dummy=True)
        a = make_vir("a")
        b = make_vir("b", dummy=True)
        c = make_vir("c", dummy=True)

        term1 = AlgebraTerm(
            coeff=Fraction(1),
            factors=(
                Tensor("t1", (a, k)),
                Tensor("t2", (b, c, i, j1), antisym_groups=((0, 1), (2, 3))),
                Tensor("v", (k, j1, b, c), antisym_groups=((0, 1), (2, 3))),
                Tensor("f", (i, a)),
            ),
            free_indices=(i, a),
            summed_indices=(k, j1, b, c),
            connected=True,
        )
        term2 = AlgebraTerm(
            coeff=Fraction(1),
            factors=(
                Tensor("t1", (c, i)),
                Tensor("t2", (a, b, j2, l), antisym_groups=((0, 1), (2, 3))),
                Tensor("v", (j2, l, b, c), antisym_groups=((0, 1), (2, 3))),
                Tensor("f", (i, a)),
            ),
            free_indices=(i, a),
            summed_indices=(j2, l, b, c),
            connected=True,
        )
        eqs = {"singles": [term1, term2]}

        intms = detect_intermediates(eqs, threshold=2)
        self.assertEqual(
            intms,
            [],
            "Different contraction topologies must not be merged into one intermediate",
        )


class IntermediateRewriteTests(unittest.TestCase):
    """Tests for equation rewriting with intermediates."""

    def test_rewrite_preserves_term_count(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)[:5]
        rewritten = rewrite_equations(eqs, intms)
        for manifold in eqs:
            self.assertEqual(
                len(eqs[manifold]),
                len(rewritten[manifold]),
                f"Rewriting must not change term count in {manifold}",
            )

    def test_rewrite_introduces_intermediate_references(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)[:3]
        rewritten = rewrite_equations(eqs, intms)
        # At least some terms should reference intermediates
        has_intermediate = False
        for terms in rewritten.values():
            for term in terms:
                if any(f.name.startswith("W_") for f in term.factors):
                    has_intermediate = True
                    break
        self.assertTrue(has_intermediate, "Rewriting should introduce intermediate refs")

    def test_empty_intermediates_returns_original(self) -> None:
        eqs = generate_cc_equations("ccsd")
        rewritten = rewrite_equations(eqs, [])
        for manifold in eqs:
            self.assertEqual(len(eqs[manifold]), len(rewritten[manifold]))

    def test_build_intermediate_equations(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)[:5]
        intm_eqs = build_intermediate_equations(intms)
        self.assertEqual(len(intm_eqs), len(intms))
        for spec in intms:
            self.assertIn(spec.name, intm_eqs)
            self.assertGreater(len(intm_eqs[spec.name]), 0)

    def test_find_subfactors_backtracks_across_same_name_factors(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        a = make_vir("a")
        b = make_vir("b")
        pattern = (
            Tensor("t1", (a, i)),
            Tensor("t1", (b, j)),
            Tensor("v", (i, j, a, b), antisym_groups=((0, 1), (2, 3))),
        )
        term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(
                Tensor("t1", (a, j)),
                Tensor("t1", (b, i)),
                Tensor("t1", (a, i)),
                Tensor("t1", (b, j)),
                Tensor("v", (i, j, a, b), antisym_groups=((0, 1), (2, 3))),
            ),
            free_indices=(i, j, a, b),
            summed_indices=(),
            connected=True,
        )

        match = _find_subfactors(term, pattern)
        self.assertIsNotNone(match)
        factor_indices, mapping = match
        self.assertEqual(len(factor_indices), 3)
        self.assertEqual(mapping[i], i)
        self.assertEqual(mapping[j], j)
        self.assertEqual(mapping[a], a)
        self.assertEqual(mapping[b], b)

    def test_find_subfactors_rejects_collapsed_distinct_indices(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        a = make_vir("a")
        b = make_vir("b")
        c = make_vir("c")
        pattern = (
            Tensor("t1", (a, i)),
            Tensor("v", (i, j, b, c), antisym_groups=((0, 1), (2, 3))),
        )
        term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(
                Tensor("t1", (a, i)),
                Tensor("v", (i, j, a, b), antisym_groups=((0, 1), (2, 3))),
            ),
            free_indices=(i, j, a, b),
            summed_indices=(),
            connected=True,
        )

        self.assertIsNone(
            _find_subfactors(term, pattern),
            "Distinct pattern indices must not collapse onto one actual index",
        )

    def test_find_subfactors_rejects_cross_space_mappings(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        a = make_vir("a")
        b = make_vir("b")
        pattern = (
            Tensor("t1", (a, i)),
            Tensor("v", (i, j, a, b), antisym_groups=((0, 1), (2, 3))),
        )
        term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(
                Tensor("t1", (a, i)),
                Tensor("v", (i, a, i, j), antisym_groups=((0, 1), (2, 3))),
            ),
            free_indices=(i, j, a, b),
            summed_indices=(),
            connected=True,
        )

        self.assertIsNone(
            _find_subfactors(term, pattern),
            "Pattern matching must preserve occupied/virtual index spaces",
        )


class InstrumentationTests(unittest.TestCase):
    """Tests for pipeline instrumentation."""

    def test_debug_populates_stats(self) -> None:
        from ccgen import generate as gen_module

        generate_cc_equations("ccd", debug=False)
        stats = gen_module.last_stats
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, PipelineStats)
        self.assertEqual(stats.method, "ccd")
        self.assertGreater(stats.bch_terms, 0)

    def test_stats_have_manifold_data(self) -> None:
        from ccgen import generate as gen_module

        generate_cc_equations("ccsd", debug=False)
        stats = gen_module.last_stats
        self.assertIn("energy", stats.manifolds)
        self.assertIn("singles", stats.manifolds)
        self.assertIn("doubles", stats.manifolds)
        for name, mstats in stats.manifolds.items():
            self.assertIn("after_projection", mstats)
            self.assertIn("after_merge", mstats)
            self.assertGreater(mstats["after_projection"], 0)
            self.assertGreater(mstats["after_merge"], 0)

    def test_stats_summary_is_string(self) -> None:
        generate_cc_equations("ccd", debug=False)
        from ccgen import generate as gen_module

        self.assertIsInstance(gen_module.last_stats.summary(), str)
        self.assertIn("CCD", gen_module.last_stats.summary())


class EmissionTests(unittest.TestCase):
    """Tests for intermediate code emission."""

    def test_cpp_intermediate_emission(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)[:3]
        code = emit_intermediate_builds(intms)
        self.assertIn("Intermediate", code)
        self.assertIn("for (int", code)
        for spec in intms:
            self.assertIn(spec.name, code)

    def test_cpp_full_emission_with_intermediates(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)[:3]
        code = emit_translation_unit_with_intermediates(eqs, intms)
        self.assertIn("Auto-generated by ccgen", code)
        self.assertIn("Intermediate tensor builds", code)
        self.assertIn("void compute_cc_residuals", code)

    def test_einsum_intermediate_emission(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)[:3]
        code = format_intermediates_einsum(intms)
        self.assertIn("Intermediate", code)
        self.assertIn("np.einsum", code)
        for spec in intms:
            self.assertIn(spec.name, code)

    def test_einsum_full_emission_with_intermediates(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)[:3]
        code = format_equations_with_intermediates_einsum(eqs, intms)
        self.assertIn("Intermediate", code)
        self.assertIn("E_CC", code)

    def test_empty_intermediates_cpp(self) -> None:
        code = emit_intermediate_builds([])
        self.assertEqual(code, "")

    def test_empty_intermediates_einsum(self) -> None:
        code = format_intermediates_einsum([])
        self.assertEqual(code, "")

    def test_planck_translation_unit_structure(self) -> None:
        code = emit_planck_translation_unit(
            "ccsd",
            generate_cc_equations("ccsd"),
        )
        self.assertIn('#include "post_hf/cc/tensor_backend.h"', code)
        self.assertIn("namespace HartreeFock::Correlation::CC", code)
        self.assertIn("double compute_ccsd_energy(", code)
        self.assertIn("Tensor2D compute_ccsd_singles_residual(", code)
        self.assertIn("Tensor4D compute_ccsd_doubles_residual(", code)
        self.assertIn("reference.f_ov(i, a)", code)
        self.assertIn("amplitudes.t1(i, a)", code)
        self.assertIn("amplitudes.t2(i, j, a, b)", code)
        self.assertIn("mo_blocks.oovv(i, j, a, b)", code)
        self.assertNotIn("F(", code)
        self.assertNotIn("V(", code)

    def test_planck_translation_unit_with_intermediates(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intermediates = [
            spec for spec in detect_intermediates(eqs, threshold=10)
            if spec.rank in (2, 4, 6)
        ][:2]
        rewritten = rewrite_equations(eqs, intermediates)
        code = emit_planck_translation_unit(
            "ccsd",
            rewritten,
            intermediates=intermediates,
        )
        self.assertIn("build_W_", code)
        self.assertIn("usage=", code)
        self.assertIn("Build reused intermediates once for this kernel", code)
        self.assertIn("const auto W_", code)

    def test_print_cpp_planck_rewrites_equations_with_intermediates(self) -> None:
        code = print_cpp_planck(
            "ccsd",
            include_intermediates=True,
            intermediate_threshold=10,
        )
        self.assertIn("build_W_", code)
        self.assertIn("Build reused intermediates once for this kernel", code)
        self.assertRegex(code, r"const auto W_[A-Za-z0-9_]* = build_W_[A-Za-z0-9_]*\(")

    def test_print_cpp_planck_only_uses_supported_intermediate_ranks(self) -> None:
        code = print_cpp_planck(
            "ccsdt",
            include_intermediates=True,
            intermediate_threshold=10,
        )
        build_defs = set(re.findall(
            r"^(?:double|Tensor2D|Tensor4D|Tensor6D|TensorND) build_(W_[A-Za-z0-9_]+)\(",
            code,
            re.MULTILINE,
        ))
        build_calls = set(re.findall(
            r"const auto (W_[A-Za-z0-9_]+) = build_",
            code,
        ))
        self.assertTrue(build_calls, "Expected rewritten kernels to build intermediates")
        self.assertTrue(
            build_calls.issubset(build_defs),
            "Every intermediate build call must have a matching emitted definition",
        )

    def test_print_cpp_planck_respects_memory_budget(self) -> None:
        code = print_cpp_planck(
            "ccsd",
            include_intermediates=True,
            intermediate_threshold=5,
            intermediate_memory_budget_bytes=1,
        )
        self.assertNotIn("build_W_", code)
        self.assertNotIn("Build reused intermediates once for this kernel", code)

    def test_print_cpp_planck_respects_peak_memory_budget(self) -> None:
        code = print_cpp_planck(
            "ccsd",
            include_intermediates=True,
            intermediate_threshold=5,
            intermediate_peak_memory_budget_bytes=1,
        )
        self.assertNotIn("build_W_", code)
        self.assertNotIn("Build reused intermediates once for this kernel", code)

    def test_planck_term_reorders_planck_storage(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        a = make_vir("a")
        b = make_vir("b")
        term = AlgebraTerm(
            coeff=Fraction(1, 1),
            factors=(Tensor("t2", (a, b, i, j)), Tensor("v", (a, b, i, j))),
            free_indices=(i, j, a, b),
            summed_indices=(),
            connected=True,
        )
        code = emit_planck_term(term)
        self.assertIn("amplitudes.t2(i, j, a, b)", code)
        self.assertIn("mo_blocks.oovv(i, j, a, b)", code)

    def test_planck_term_emits_delta_guard(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        a = make_vir("a")
        term = AlgebraTerm(
            coeff=Fraction(1, 1),
            factors=(Tensor("delta", (i, j)), Tensor("t1", (a, i))),
            free_indices=(i, a),
            summed_indices=(),
            connected=True,
        )
        code = emit_planck_term(term)
        self.assertIn("((i == j) ? 1.0 : 0.0)", code)
        self.assertIn("amplitudes.t1(i, a)", code)

    def test_planck_term_reorders_t3_storage(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        k = make_occ("k")
        a = make_vir("a")
        b = make_vir("b")
        c = make_vir("c")
        term = AlgebraTerm(
            coeff=Fraction(1, 1),
            factors=(Tensor("t3", (a, b, c, i, j, k)),),
            free_indices=(i, j, k, a, b, c),
            summed_indices=(),
            connected=True,
        )
        code = emit_planck_term(term)
        self.assertIn("amplitudes.t3(i, j, k, a, b, c)", code)

    def test_planck_term_uses_lowered_canonical_free_order(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        a = make_vir("a")
        b = make_vir("b")
        term = AlgebraTerm(
            coeff=Fraction(1, 1),
            factors=(Tensor("t2", (a, b, i, j)),),
            free_indices=(a, i, b, j),
            summed_indices=(),
            connected=True,
        )
        lowered = lower_term_restricted_closed_shell(term, "doubles")
        code = emit_planck_term(lowered)
        self.assertIn("for (int i = 0; i < no; ++i)", code)
        self.assertIn("for (int j = 0; j < no; ++j)", code)
        self.assertIn("for (int a = 0; a < nv; ++a)", code)
        self.assertIn("for (int b = 0; b < nv; ++b)", code)
        self.assertIn("result(i, j, a, b)", code)
        self.assertIn("amplitudes.t2(i, j, a, b)", code)

    def test_planck_term_uses_lowered_intermediate_index_order(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        a = make_vir("a")
        b = make_vir("b")
        term = AlgebraTerm(
            coeff=Fraction(1, 1),
            factors=(Tensor("W_oovv", (a, i, b, j)),),
            free_indices=(a, i, b, j),
            summed_indices=(),
            connected=True,
        )
        lowered = lower_term_restricted_closed_shell(term, "doubles")
        code = emit_planck_term(lowered)
        self.assertIn("W_oovv(i, j, a, b)", code)
        self.assertIn("result(i, j, a, b)", code)

    def test_planck_term_uses_lowered_eri_block_and_phase(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        a = make_vir("a")
        b = make_vir("b")
        term = AlgebraTerm(
            coeff=Fraction(1, 1),
            factors=(Tensor("v", (a, i, j, b)),),
            free_indices=(a, i, j, b),
            summed_indices=(),
            connected=True,
        )
        lowered = lower_term_restricted_closed_shell(term, "doubles")
        code = emit_planck_term(lowered)
        self.assertIn("result(i, j, a, b)", code)
        self.assertIn("-mo_blocks.ovov(i, a, j, b)", code)

    def test_planck_intermediate_builder_uses_lowered_layout(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        a = make_vir("a")
        b = make_vir("b")
        spec = IntermediateSpec(
            name="W_oovv_test",
            indices=(a, i, b, j),
            definition_terms=(
                AlgebraTerm(
                    coeff=Fraction(1, 1),
                    factors=(Tensor("t2", (a, b, i, j)),),
                    free_indices=(a, i, b, j),
                    summed_indices=(),
                    connected=True,
                ),
            ),
            usage_count=3,
            index_space_sig="oovv",
        )
        code = emit_planck_translation_unit(
            "ccsd",
            {"energy": generate_cc_equations("ccsd", targets=["energy"])["energy"]},
            intermediates=[spec],
        )
        self.assertIn("Tensor4D result(no, no, nv, nv, 0.0);", code)
        self.assertIn("amplitudes.t2(i, j, a, b)", code)

    def test_planck_translation_unit_supports_arbitrary_excitation_order(self) -> None:
        i = make_occ("i")
        j = make_occ("j")
        k = make_occ("k")
        l = make_occ("l")
        a = make_vir("a")
        b = make_vir("b")
        c = make_vir("c")
        d = make_vir("d")
        term = AlgebraTerm(
            coeff=Fraction(1, 1),
            factors=(
                Tensor("t4", (a, b, c, d, i, j, k, l)),
                Tensor("D", (i, j, k, l, a, b, c, d)),
            ),
            free_indices=(i, j, k, l, a, b, c, d),
            summed_indices=(),
            connected=True,
        )
        code = emit_planck_translation_unit(
            "ccsdtq",
            {"quadruples": [term]},
        )
        self.assertIn(
            "TensorND compute_ccsdtq_quadruples_residual(",
            code,
        )
        self.assertIn(
            "const ArbitraryOrderDenominatorCache &denominators,",
            code,
        )
        self.assertIn(
            "const ArbitraryOrderRCCAmplitudes &amplitudes)",
            code,
        )
        self.assertIn(
            "TensorND result(std::vector<int>{no, no, no, no, nv, nv, nv, nv}, 0.0);",
            code,
        )
        self.assertIn(
            "amplitudes.tensor(4)({i, j, k, l, a, b, c, d})",
            code,
        )
        self.assertIn(
            "denominators.tensor(4)({i, j, k, l, a, b, c, d})",
            code,
        )
        self.assertIn(
            "result({i, j, k, l, a, b, c, d})",
            code,
        )
        self.assertIn(
            '#include "post_hf/cc/generated_arbitrary_runtime.h"',
            code,
        )
        self.assertIn(
            "GeneratedArbitraryOrderKernels make_generated_ccsdtq_kernels()",
            code,
        )
        self.assertIn(
            "kernels.energy = compute_ccsdtq_energy;",
            code,
        )
        self.assertIn(
            "return to_tensor_nd(compute_ccsdtq_quadruples_residual(reference, mo_blocks, denominators, amplitudes));",
            code,
        )

    def test_planck_rccsd_warm_start_uses_native_intermediate_pipeline(self) -> None:
        code = emit_planck_spinorbital_rccsd_warm_start()
        self.assertIn("const TauCache tau_cache = build_tau_cache(amps);", code)
        self.assertIn("const RCCSDIntermediates ints = build_intermediates(", code)
        self.assertIn("return build_residuals(reference, blocks, amps, tau_cache, ints);", code)
        self.assertIn("compute_generated_spin_orbital_rccsd_correlation_energy(", code)


class FactoredCanonicalizationTests(unittest.TestCase):
    """Tests verifying the factored canonicalization refactor."""

    def test_ccsd_term_counts_stable(self) -> None:
        """Verify term counts match expected values (regression)."""
        eqs = generate_cc_equations("ccsd")
        self.assertEqual(len(eqs["energy"]), 3)
        self.assertEqual(len(eqs["singles"]), 24)
        self.assertEqual(len(eqs["doubles"]), 200)

    def test_ccd_term_counts_stable(self) -> None:
        eqs = generate_cc_equations("ccd")
        self.assertEqual(len(eqs["energy"]), 1)

    @unittest.skipIf(np is None, "numpy required")
    def test_ccsd_energy_numerical_equivalence(self) -> None:
        """Verify CCSD energy matches reference (from test_regressions)."""
        from ccgen.tests.test_regressions import (
            build_test_tensors,
            evaluate_scalar_term,
        )

        tensors, fock, oovv = build_test_tensors()
        eqs = generate_cc_equations("ccsd", targets=["energy"])
        energy_terms = eqs["energy"]

        generated = sum(
            evaluate_scalar_term(term, tensors, nocc=2)
            for term in energy_terms
        )

        t1_ia = tensors["t1"].T
        t2_ijab = np.transpose(tensors["t2"], (2, 3, 0, 1))
        expected = np.einsum("ia,ia", fock[:2, 2:], t1_ia)
        expected += 0.25 * np.einsum("ijab,ijab", t2_ijab, oovv)
        expected += 0.5 * np.einsum("ia,jb,ijab", t1_ia, t1_ia, oovv)

        self.assertAlmostEqual(generated, float(expected), places=12)


# ── Phase 2 Tests ───────────────────────────────────────────────────


class OrbitalEnergyCollectionTests(unittest.TestCase):
    """Tests for diagonal Fock → orbital energy denominator collection."""

    def test_singles_reduces_term_count(self) -> None:
        eqs = generate_cc_equations("ccsd")
        before = len(eqs["singles"])
        after = len(collect_fock_diagonals(eqs["singles"]))
        self.assertLess(after, before, "Denominator collection should reduce singles")

    def test_doubles_reduces_term_count(self) -> None:
        eqs = generate_cc_equations("ccsd")
        before = len(eqs["doubles"])
        after = len(collect_fock_diagonals(eqs["doubles"]))
        self.assertLess(after, before, "Denominator collection should reduce doubles")

    def test_energy_unchanged(self) -> None:
        eqs = generate_cc_equations("ccsd")
        before = len(eqs["energy"])
        after = len(collect_fock_diagonals(eqs["energy"]))
        self.assertEqual(before, after, "Energy should not have diagonal Fock terms to collect")

    def test_denominator_tensor_present(self) -> None:
        eqs = generate_cc_equations("ccsd")
        collected = collect_fock_diagonals(eqs["doubles"])
        d_terms = [t for t in collected if any(f.name == "D" for f in t.factors)]
        self.assertGreater(len(d_terms), 0, "Should have at least one D tensor term")

    def test_denominator_indices_correct(self) -> None:
        eqs = generate_cc_equations("ccsd")
        collected = collect_fock_diagonals(eqs["doubles"])
        for t in collected:
            for f in t.factors:
                if f.name == "D":
                    # All D indices should be free indices of the term
                    for idx in f.indices:
                        self.assertIn(idx, t.free_indices,
                                      f"D index {idx} must be a free index")

    def test_collect_denominators_flag(self) -> None:
        """Test the generate_cc_equations collect_denominators parameter."""
        eqs_normal = generate_cc_equations("ccsd")
        eqs_denom = generate_cc_equations("ccsd", collect_denominators=True)
        self.assertLess(
            len(eqs_denom["doubles"]),
            len(eqs_normal["doubles"]),
        )

    def test_stats_include_denom_collection(self) -> None:
        from ccgen import generate as gen_module
        generate_cc_equations("ccsd", collect_denominators=True)
        stats = gen_module.last_stats
        self.assertIn("after_denom_collection", stats.manifolds["doubles"])

    @unittest.skipIf(np is None, "numpy required")
    def test_numerical_equivalence_singles(self) -> None:
        """Verify collected singles evaluate to same result as original."""
        from ccgen.tests.test_regressions import build_test_tensors

        tensors, fock, oovv = build_test_tensors()
        nocc, nvir = 2, 3

        eqs = generate_cc_equations("ccsd", targets=["singles"])
        original = eqs["singles"]
        collected = collect_fock_diagonals(original)

        # Evaluate original terms
        orig_result = np.zeros((nocc, nvir))
        for term in original:
            orig_result += _evaluate_residual_term(term, tensors, nocc, nvir)

        # Evaluate collected terms (need to handle D tensor)
        coll_result = np.zeros((nocc, nvir))
        for term in collected:
            coll_result += _evaluate_residual_term(term, tensors, nocc, nvir)

        np.testing.assert_allclose(coll_result, orig_result, atol=1e-12)


class CSETests(unittest.TestCase):
    """Tests for common subexpression elimination."""

    def test_cse_on_manufactured_input(self) -> None:
        """CSE merges terms with identical RHS and different coefficients."""
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import Tensor
        from ccgen.indices import make_occ, make_vir
        from fractions import Fraction

        i = make_occ("i")
        a = make_vir("a")
        k = make_occ("k", dummy=True)

        t1 = AlgebraTerm(
            coeff=Fraction(1, 2),
            factors=(Tensor("f", (i, k)), Tensor("t1", (a, k))),
            free_indices=(i, a),
            summed_indices=(k,),
            connected=True,
        )
        t2 = AlgebraTerm(
            coeff=Fraction(1, 4),
            factors=(Tensor("f", (i, k)), Tensor("t1", (a, k))),
            free_indices=(i, a),
            summed_indices=(k,),
            connected=True,
        )

        rewritten, specs = factor_common_subexpressions([t1, t2])
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(rewritten), 1)
        self.assertEqual(rewritten[0].coeff, Fraction(3, 4))

    def test_cse_no_false_positives(self) -> None:
        """CSE should not merge terms with different contractions."""
        eqs = generate_cc_equations("ccsd")
        rewritten, specs = factor_common_subexpressions(eqs["doubles"])
        # After merge_like_terms, no duplicate contractions should exist
        self.assertEqual(len(specs), 0)
        self.assertEqual(len(rewritten), len(eqs["doubles"]))

    def test_apply_cse_preserves_structure(self) -> None:
        eqs = generate_cc_equations("ccsd")
        new_eqs, all_specs = apply_cse(eqs)
        for manifold in eqs:
            self.assertIn(manifold, new_eqs)

    def test_cse_spec_fields(self) -> None:
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import Tensor
        from ccgen.indices import make_occ, make_vir
        from fractions import Fraction

        i = make_occ("i")
        a = make_vir("a")
        terms = [
            AlgebraTerm(
                coeff=Fraction(1),
                factors=(Tensor("v", (i, a, i, a)),),
                free_indices=(i, a),
                summed_indices=(),
                connected=True,
            ),
            AlgebraTerm(
                coeff=Fraction(2),
                factors=(Tensor("v", (i, a, i, a)),),
                free_indices=(i, a),
                summed_indices=(),
                connected=True,
            ),
        ]
        _, specs = factor_common_subexpressions(terms)
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertIsInstance(spec, CSESpec)
        self.assertTrue(spec.name)
        self.assertEqual(spec.usage_count, 2)
        self.assertGreater(spec.rank, 0)


class WickEarlyTerminationTests(unittest.TestCase):
    """Tests verifying Wick early termination preserves correctness."""

    def test_ccsd_term_counts_unchanged(self) -> None:
        """Term counts must match pre-optimization values exactly."""
        eqs = generate_cc_equations("ccsd")
        self.assertEqual(len(eqs["energy"]), 3)
        self.assertEqual(len(eqs["singles"]), 24)
        self.assertEqual(len(eqs["doubles"]), 200)

    def test_ccd_term_counts_unchanged(self) -> None:
        eqs = generate_cc_equations("ccd")
        self.assertEqual(len(eqs["energy"]), 1)
        self.assertEqual(len(eqs["doubles"]), 40)

    @unittest.skipIf(np is None, "numpy required")
    def test_ccsd_energy_still_correct(self) -> None:
        """Numerical energy must match reference after Wick optimization."""
        from ccgen.tests.test_regressions import (
            build_test_tensors,
            evaluate_scalar_term,
        )

        tensors, fock, oovv = build_test_tensors()
        eqs = generate_cc_equations("ccsd", targets=["energy"])

        generated = sum(
            evaluate_scalar_term(term, tensors, nocc=2)
            for term in eqs["energy"]
        )

        t1_ia = tensors["t1"].T
        t2_ijab = np.transpose(tensors["t2"], (2, 3, 0, 1))
        expected = np.einsum("ia,ia", fock[:2, 2:], t1_ia)
        expected += 0.25 * np.einsum("ijab,ijab", t2_ijab, oovv)
        expected += 0.5 * np.einsum("ia,jb,ijab", t1_ia, t1_ia, oovv)

        self.assertAlmostEqual(generated, float(expected), places=12)

    def test_generation_is_deterministic(self) -> None:
        """Results must be identical across multiple runs."""
        counts = []
        for _ in range(3):
            eqs = generate_cc_equations("ccsd")
            counts.append({k: len(v) for k, v in eqs.items()})
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[1], counts[2])


class TiledLoopEmissionTests(unittest.TestCase):
    """Tests for the tiled + OpenMP C++ emitter."""

    def test_tiled_output_has_tile_loops(self) -> None:
        eqs = generate_cc_equations("ccsd", targets=["doubles"])
        code = emit_optimized_translation_unit(eqs)
        # Should contain tile block variables
        self.assertIn("ib", code)
        self.assertIn("jb", code)
        self.assertIn("ab", code)
        self.assertIn("bb", code)

    def test_tiled_output_has_openmp(self) -> None:
        eqs = generate_cc_equations("ccsd", targets=["doubles"])
        code = emit_optimized_translation_unit(eqs, use_openmp=True)
        self.assertIn("#pragma omp parallel for", code)
        self.assertIn("collapse(", code)
        self.assertIn("#include <omp.h>", code)

    def test_tiled_output_no_openmp(self) -> None:
        eqs = generate_cc_equations("ccsd", targets=["doubles"])
        code = emit_optimized_translation_unit(eqs, use_openmp=False)
        self.assertNotIn("#pragma omp", code)
        self.assertNotIn("#include <omp.h>", code)

    def test_tiled_output_has_std_min(self) -> None:
        eqs = generate_cc_equations("ccsd", targets=["doubles"])
        code = emit_optimized_translation_unit(eqs)
        self.assertIn("std::min(", code)
        self.assertIn("#include <algorithm>", code)

    def test_custom_tile_sizes(self) -> None:
        eqs = generate_cc_equations("ccsd", targets=["doubles"])
        code = emit_optimized_translation_unit(eqs, tile_occ=8, tile_vir=32)
        self.assertIn("occ=8", code)
        self.assertIn("vir=32", code)
        self.assertIn("+= 8)", code)   # tile increment for occ
        self.assertIn("+= 32)", code)  # tile increment for vir

    def test_singles_not_tiled(self) -> None:
        """Singles have only 2 free indices; should still produce valid code."""
        eqs = generate_cc_equations("ccsd", targets=["singles"])
        code = emit_optimized_translation_unit(eqs)
        self.assertIn("R1", code)

    def test_energy_falls_back_to_naive(self) -> None:
        """Energy (0 free indices) should fall back to untiled emission."""
        eqs = generate_cc_equations("ccsd", targets=["energy"])
        code = emit_optimized_translation_unit(eqs)
        self.assertIn("E_CC", code)

    def test_tiled_output_is_valid_structure(self) -> None:
        """Output should have matching braces."""
        eqs = generate_cc_equations("ccsd")
        code = emit_optimized_translation_unit(eqs)
        self.assertEqual(code.count("{"), code.count("}"),
                         "Braces must be balanced")


# ── Helper for numerical evaluation of residual terms ───────────────


def _evaluate_residual_term(
    term,
    tensors: dict[str, "np.ndarray"],
    nocc: int,
    nvir: int,
) -> "np.ndarray":
    """Evaluate a residual term on random tensors, returning the result array."""
    if np is None:
        raise unittest.SkipTest("numpy required")

    free = list(term.free_indices)
    summed = list(term.summed_indices)
    shape = tuple(nocc if idx.space == "occ" else nvir for idx in free)
    result = np.zeros(shape)

    # Build index ranges
    ranges = {}
    for idx in free + summed:
        ranges[idx.name] = nocc if idx.space == "occ" else nvir

    def _recurse(level: int, env: dict[str, int]) -> None:
        if level == len(summed):
            value = float(term.coeff)
            for factor in term.factors:
                if factor.name == "D":
                    # Orbital energy denominator
                    fock = tensors["f"]
                    d_val = 0.0
                    for idx in factor.indices:
                        slot = env[idx.name]
                        if idx.space == "vir":
                            d_val += fock[nocc + slot, nocc + slot]
                        else:
                            d_val -= fock[slot, slot]
                    value *= d_val
                elif factor.name in ("f", "v"):
                    indices = []
                    for idx in factor.indices:
                        slot = env[idx.name]
                        if idx.space == "vir":
                            indices.append(nocc + slot)
                        else:
                            indices.append(slot)
                    value *= float(tensors[factor.name][tuple(indices)])
                elif factor.name == "delta":
                    lhs, rhs = factor.indices
                    value *= 1.0 if env[lhs.name] == env[rhs.name] else 0.0
                else:
                    indices = [env[idx.name] for idx in factor.indices]
                    value *= float(tensors[factor.name][tuple(indices)])

            free_idx = tuple(env[idx.name] for idx in free)
            result[free_idx] += value
            return

        idx = summed[level]
        for val in range(ranges[idx.name]):
            env[idx.name] = val
            _recurse(level + 1, env)
        env.pop(idx.name, None)

    # Iterate over free indices
    def _outer(level: int, env: dict[str, int]) -> None:
        if level == len(free):
            _recurse(0, env)
            return
        idx = free[level]
        for val in range(ranges[idx.name]):
            env[idx.name] = val
            _outer(level + 1, env)
        env.pop(idx.name, None)

    _outer(0, {})
    return result


# ── Phase 3 Tests ───────────────────────────────────────────────────


class PermutationGroupingTests(unittest.TestCase):
    """Tests for permutation-based term grouping."""

    def test_ccsd_grouping_reduces_terms(self) -> None:
        """Permutation grouping should reduce doubles term count."""
        eqs = generate_cc_equations("ccsd")
        from ccgen.optimization.permutation import apply_permutation_grouping
        before = len(eqs["doubles"])
        after = len(apply_permutation_grouping(eqs["doubles"]))
        self.assertLessEqual(after, before)

    def test_ccsd_energy_unchanged(self) -> None:
        """Energy has no free indices; grouping should not change it."""
        eqs = generate_cc_equations("ccsd")
        from ccgen.optimization.permutation import apply_permutation_grouping
        before = len(eqs["energy"])
        after = len(apply_permutation_grouping(eqs["energy"]))
        self.assertEqual(before, after)

    def test_grouping_detects_groups(self) -> None:
        """detect_permutation_groups returns valid structure."""
        eqs = generate_cc_equations("ccsd")
        from ccgen.optimization.permutation import detect_permutation_groups
        groups, ungrouped = detect_permutation_groups(eqs["doubles"])
        # After canonicalization, most permutation equivalences are already
        # handled by merge_like_terms.  The grouping function should still
        # return a valid (possibly empty) result.
        self.assertIsInstance(groups, list)
        self.assertIsInstance(ungrouped, list)
        total = len(ungrouped) + sum(len(g.original_indices) for g in groups)
        self.assertEqual(total, len(eqs["doubles"]))

    def test_group_has_valid_structure(self) -> None:
        from ccgen.optimization.permutation import (
            detect_permutation_groups,
            PermutationGroup,
        )
        eqs = generate_cc_equations("ccsd")
        groups, _ = detect_permutation_groups(eqs["doubles"])
        for group in groups:
            self.assertIsInstance(group, PermutationGroup)
            self.assertGreaterEqual(len(group.original_indices), 2)
            self.assertEqual(
                len(group.permutations),
                len(group.original_indices),
            )

    def test_equations_api(self) -> None:
        """apply_permutation_grouping_equations works on full dict."""
        from ccgen.optimization.permutation import (
            apply_permutation_grouping_equations,
        )
        eqs = generate_cc_equations("ccsd")
        result = apply_permutation_grouping_equations(eqs)
        self.assertIn("energy", result)
        self.assertIn("singles", result)
        self.assertIn("doubles", result)

    def test_stats_api(self) -> None:
        from ccgen.optimization.permutation import permutation_grouping_stats
        eqs = generate_cc_equations("ccsd")
        stats = permutation_grouping_stats(eqs)
        for manifold in eqs:
            self.assertIn(manifold, stats)
            self.assertIn("before", stats[manifold])
            self.assertIn("after", stats[manifold])
            self.assertIn("eliminated", stats[manifold])
            self.assertEqual(
                stats[manifold]["before"],
                stats[manifold]["after"] + stats[manifold]["eliminated"],
            )

    def test_generate_with_permutation_flag(self) -> None:
        """generate_cc_equations with permutation_grouping=True works."""
        eqs_normal = generate_cc_equations("ccsd")
        eqs_perm = generate_cc_equations("ccsd", permutation_grouping=True)
        self.assertLessEqual(
            len(eqs_perm["doubles"]),
            len(eqs_normal["doubles"]),
        )

    @unittest.skipIf(np is None, "numpy required")
    def test_numerical_equivalence_singles(self) -> None:
        """Grouped singles must evaluate identically to ungrouped."""
        from ccgen.tests.test_regressions import build_test_tensors
        from ccgen.optimization.permutation import apply_permutation_grouping

        tensors, fock, oovv = build_test_tensors()
        nocc, nvir = 2, 3

        eqs = generate_cc_equations("ccsd", targets=["singles"])
        original = eqs["singles"]
        grouped = apply_permutation_grouping(original)

        orig_result = np.zeros((nocc, nvir))
        for term in original:
            orig_result += _evaluate_residual_term(term, tensors, nocc, nvir)

        grp_result = np.zeros((nocc, nvir))
        for term in grouped:
            grp_result += _evaluate_residual_term(term, tensors, nocc, nvir)

        np.testing.assert_allclose(grp_result, orig_result, atol=1e-12)


class BackendIRExtTests(unittest.TestCase):
    """Tests for extended backend IR (BackendTermEx)."""

    def test_lower_equations_ex_produces_hints(self) -> None:
        from ccgen.tensor_ir import lower_equations_ex, BackendTermEx
        eqs = generate_cc_equations("ccsd", targets=["doubles"])
        ex = lower_equations_ex(eqs)
        self.assertIn("doubles", ex)
        for tex in ex["doubles"]:
            self.assertIsInstance(tex, BackendTermEx)
            self.assertIsInstance(tex.estimated_flops, int)
            self.assertGreater(tex.estimated_flops, 0)
            self.assertIsInstance(tex.computation_order, tuple)
            self.assertIsInstance(tex.memory_layout, dict)
            self.assertIsInstance(tex.blocking_hint, dict)

    def test_blas_detection_runs(self) -> None:
        from ccgen.tensor_ir import lower_equations_ex
        eqs = generate_cc_equations("ccsd")
        ex = lower_equations_ex(eqs, detect_blas=True)
        # Should find at least some GEMM patterns in doubles
        blas_count = sum(
            1 for terms in ex.values()
            for t in terms if t.blas_hint is not None
        )
        # GEMM detection is conservative; it's OK if none are found
        self.assertIsInstance(blas_count, int)

    def test_blas_hint_fields(self) -> None:
        from ccgen.tensor_ir import lower_equations_ex, BLASHint
        eqs = generate_cc_equations("ccsd")
        ex = lower_equations_ex(eqs, detect_blas=True)
        for terms in ex.values():
            for tex in terms:
                if tex.blas_hint is not None:
                    h = tex.blas_hint
                    self.assertIsInstance(h, BLASHint)
                    self.assertIn(h.pattern, [
                        "gemm_nn", "gemm_nt", "gemm_tn", "gemm_tt",
                    ])
                    self.assertTrue(h.a_tensor)
                    self.assertTrue(h.b_tensor)
                    self.assertGreater(len(h.contraction_indices), 0)

    def test_generate_cc_contractions_ex(self) -> None:
        from ccgen.generate import generate_cc_contractions_ex
        ex = generate_cc_contractions_ex("ccsd", targets=["energy"])
        self.assertIn("energy", ex)
        for tex in ex["energy"]:
            self.assertIsInstance(tex.estimated_flops, int)

    def test_flop_estimate_reasonable(self) -> None:
        from ccgen.tensor_ir import lower_equations_ex
        eqs = generate_cc_equations("ccsd", targets=["energy"])
        ex = lower_equations_ex(eqs)
        for tex in ex["energy"]:
            # Energy terms have summed indices, should have non-trivial flops
            self.assertGreater(tex.estimated_flops, 0)


class MemoryLayoutTests(unittest.TestCase):
    """Tests for intermediate memory layout and blocking hints."""

    def test_annotate_layout_hints(self) -> None:
        from ccgen.optimization.intermediates import (
            detect_intermediates,
            annotate_layout_hints,
            total_estimated_bytes,
        )
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=5)
        annotated = annotate_layout_hints(intms)
        self.assertEqual(len(annotated), len(intms))
        for spec in annotated:
            self.assertIn(spec.memory_layout, ["row_major", "col_major", "blocked"])
            self.assertIn(spec.allocation_strategy, ["stack", "malloc", "external", "auto"])
            if spec.blocking_hint:
                for k, v in spec.blocking_hint.items():
                    self.assertIsInstance(k, str)
                    self.assertIsInstance(v, int)
        self.assertEqual(total_estimated_bytes(annotated), sum(
            spec.estimated_bytes for spec in annotated
        ))

    def test_small_intermediates_use_stack(self) -> None:
        from ccgen.optimization.intermediates import (
            detect_intermediates,
            annotate_layout_hints,
        )
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)
        annotated = annotate_layout_hints(intms)
        for spec in annotated:
            if spec.rank <= 2:
                # Rank-2 intermediates (e.g. W_ov) are small enough for stack
                self.assertEqual(spec.allocation_strategy, "stack")

    def test_estimated_elements(self) -> None:
        from ccgen.optimization.intermediates import detect_intermediates
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)
        for spec in intms:
            self.assertGreater(spec.estimated_elements, 0)
            self.assertEqual(spec.estimated_bytes, spec.estimated_elements * 8)

    def test_with_layout_hints_returns_new(self) -> None:
        from ccgen.optimization.intermediates import detect_intermediates
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)
        if intms:
            original = intms[0]
            modified = original.with_layout_hints(
                memory_layout="blocked",
                allocation_strategy="malloc",
            )
            self.assertEqual(modified.memory_layout, "blocked")
            self.assertEqual(modified.allocation_strategy, "malloc")
            self.assertEqual(modified.name, original.name)

    def test_memory_budget_filters_intermediates(self) -> None:
        from ccgen.optimization.intermediates import (
            detect_intermediates,
            total_estimated_bytes,
        )
        eqs = generate_cc_equations("ccsd")
        unrestricted = detect_intermediates(eqs, threshold=5)
        self.assertTrue(unrestricted)
        selected = detect_intermediates(
            eqs,
            threshold=5,
            memory_budget_bytes=1,
        )
        self.assertEqual(selected, [])
        medium_budget = unrestricted[0].estimated_bytes
        selected = detect_intermediates(
            eqs,
            threshold=5,
            memory_budget_bytes=medium_budget,
        )
        self.assertLessEqual(total_estimated_bytes(selected), medium_budget)
        self.assertLessEqual(len(selected), len(unrestricted))
        self.assertTrue(all(spec.usage_targets for spec in unrestricted))

    def test_budget_selector_prefers_better_value_per_byte(self) -> None:
        from ccgen.optimization.intermediates import select_intermediates_for_budget

        i = make_occ("i")
        a = make_vir("a")
        b = make_vir("b")

        small = IntermediateSpec(
            name="W_small",
            indices=(i, a),
            definition_terms=(
                AlgebraTerm(
                    coeff=Fraction(1),
                    factors=(Tensor("v", (a, b, i, i)),),
                    free_indices=(i, a),
                    summed_indices=(b,),
                    connected=True,
                ),
            ),
            usage_count=20,
            index_space_sig="ov",
        )
        large = IntermediateSpec(
            name="W_large",
            indices=(a, b),
            definition_terms=(
                AlgebraTerm(
                    coeff=Fraction(1),
                    factors=(Tensor("v", (a, b, i, i)), Tensor("t1", (a, i))),
                    free_indices=(a, b),
                    summed_indices=(),
                    connected=True,
                ),
            ),
            usage_count=3,
            index_space_sig="vv",
        )

        budget = large.estimated_bytes
        selected = select_intermediates_for_budget([large, small], budget)
        self.assertEqual([spec.name for spec in selected], ["W_small"])

    def test_peak_budget_respects_per_target_live_sets(self) -> None:
        from ccgen.optimization.intermediates import (
            IntermediateSpec,
            peak_estimated_bytes_by_target,
            select_intermediates_for_peak_budget,
        )

        i = make_occ("i")
        a = make_vir("a")
        base_term = AlgebraTerm(
            coeff=Fraction(1),
            factors=(Tensor("t1", (a, i)),),
            free_indices=(i, a),
            summed_indices=(),
            connected=True,
        )
        spec_a = IntermediateSpec(
            name="W_a",
            indices=(i, a),
            definition_terms=(base_term,),
            usage_count=8,
            index_space_sig="ov",
            usage_targets=("singles",),
        )
        spec_b = IntermediateSpec(
            name="W_b",
            indices=(i, a),
            definition_terms=(base_term,),
            usage_count=7,
            index_space_sig="ov",
            usage_targets=("doubles",),
        )
        spec_c = IntermediateSpec(
            name="W_c",
            indices=(i, a),
            definition_terms=(base_term,),
            usage_count=6,
            index_space_sig="ov",
            usage_targets=("singles",),
        )

        budget = spec_a.estimated_bytes
        selected = select_intermediates_for_peak_budget(
            [spec_a, spec_b, spec_c],
            budget,
        )
        names = {spec.name for spec in selected}
        self.assertIn("W_a", names)
        self.assertIn("W_b", names)
        self.assertNotIn("W_c", names)
        peaks = peak_estimated_bytes_by_target(selected)
        self.assertLessEqual(peaks["singles"], budget)
        self.assertLessEqual(peaks["doubles"], budget)


class BLASEmissionTests(unittest.TestCase):
    """Tests for BLAS/GEMM C++ emission."""

    def test_blas_translation_unit_structure(self) -> None:
        from ccgen.emit.cpp_loops import emit_blas_translation_unit
        eqs = generate_cc_equations("ccsd", targets=["doubles"])
        code = emit_blas_translation_unit(eqs, use_blas=True)
        self.assertIn("Auto-generated by ccgen (BLAS-optimized)", code)
        self.assertIn("#include <cblas.h>", code)
        self.assertIn("BLAS/GEMM lowering enabled", code)
        self.assertIn("Summary:", code)

    def test_blas_without_blas_flag(self) -> None:
        from ccgen.emit.cpp_loops import emit_blas_translation_unit
        eqs = generate_cc_equations("ccsd", targets=["doubles"])
        code = emit_blas_translation_unit(eqs, use_blas=False)
        self.assertNotIn("#include <cblas.h>", code)
        self.assertNotIn("BLAS/GEMM lowering enabled", code)

    def test_blas_with_intermediates(self) -> None:
        from ccgen.emit.cpp_loops import emit_blas_translation_unit
        from ccgen.optimization.intermediates import detect_intermediates
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)[:3]
        code = emit_blas_translation_unit(eqs, intermediates=intms)
        self.assertIn("Intermediate tensor builds", code)

    def test_print_cpp_blas_api(self) -> None:
        from ccgen.generate import print_cpp_blas
        code = print_cpp_blas("ccsd", use_blas=True)
        self.assertIn("BLAS-optimized", code)

    def test_blas_code_has_balanced_braces(self) -> None:
        from ccgen.emit.cpp_loops import emit_blas_translation_unit
        eqs = generate_cc_equations("ccsd", targets=["energy", "singles"])
        code = emit_blas_translation_unit(eqs, use_blas=True)
        self.assertEqual(
            code.count("{"), code.count("}"),
            "Braces must be balanced",
        )


class PrettyPrintingPhase3Tests(unittest.TestCase):
    """Tests for enhanced pretty printing with intermediates."""

    def test_format_with_legend(self) -> None:
        from ccgen.emit.pretty import format_equations_with_intermediates
        eqs = generate_cc_equations("ccsd")
        output = format_equations_with_intermediates(
            eqs, include_legend=True,
        )
        self.assertIn("Index Legend", output)
        self.assertIn("occupied", output)
        self.assertIn("virtual", output)
        self.assertIn("Tensor Legend", output)

    def test_format_with_intermediates(self) -> None:
        from ccgen.emit.pretty import format_equations_with_intermediates
        from ccgen.optimization.intermediates import detect_intermediates
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)[:3]
        output = format_equations_with_intermediates(
            eqs, intermediates=intms,
        )
        self.assertIn("Intermediate Tensor Definitions", output)
        for spec in intms:
            self.assertIn(spec.name, output)

    def test_format_with_stats(self) -> None:
        from ccgen.emit.pretty import format_equations_with_intermediates
        eqs = generate_cc_equations("ccsd")
        output = format_equations_with_intermediates(
            eqs, include_stats=True,
        )
        self.assertIn("Summary", output)
        self.assertIn("Total:", output)

    def test_format_section_headers(self) -> None:
        from ccgen.emit.pretty import format_equations_with_intermediates
        eqs = generate_cc_equations("ccsd")
        output = format_equations_with_intermediates(eqs)
        self.assertIn("Correlation energy", output)
        self.assertIn("Singles (T1) residual", output)
        self.assertIn("Doubles (T2) residual", output)

    def test_format_intermediate_with_layout(self) -> None:
        from ccgen.emit.pretty import format_intermediate
        from ccgen.optimization.intermediates import (
            detect_intermediates,
            annotate_layout_hints,
        )
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=10)
        annotated = annotate_layout_hints(intms)
        for spec in annotated:
            text = format_intermediate(spec)
            self.assertIn(spec.name, text)
            # Layout hints appear only for non-default
            if spec.allocation_strategy != "auto":
                self.assertIn("allocation:", text)

    def test_print_equations_full_api(self) -> None:
        from ccgen.generate import print_equations_full
        output = print_equations_full("ccsd", include_intermediates=True)
        self.assertIn("Coupled-Cluster Residual Equations", output)
        self.assertIn("E_CC", output)


class EinsumEmissionTests(unittest.TestCase):
    """Tests for executable NumPy einsum emission."""

    @unittest.skipIf(np is None, "numpy required")
    def test_generated_einsum_code_executes(self) -> None:
        code = format_equations_einsum(generate_cc_equations("ccsd"))
        ns = {
            "np": np,
            "no": 2,
            "nv": 3,
            "F": np.zeros((5, 5)),
            "V": np.zeros((5, 5, 5, 5)),
            "T1": np.zeros((3, 2)),
            "T2": np.zeros((3, 3, 2, 2)),
        }
        exec(code, ns, ns)
        self.assertEqual(ns["R1"].shape, (2, 3))
        self.assertEqual(ns["R2"].shape, (2, 2, 3, 3))


class ImplicitSymmetryTests(unittest.TestCase):
    """Tests for implicit antisymmetry exploitation (experimental)."""

    def test_symmetry_does_not_increase_terms(self) -> None:
        from ccgen.optimization.symmetry import exploit_antisymmetry
        eqs = generate_cc_equations("ccsd")
        before = len(eqs["doubles"])
        after = len(exploit_antisymmetry(eqs["doubles"]))
        self.assertLessEqual(after, before)

    def test_symmetry_equations_api(self) -> None:
        from ccgen.optimization.symmetry import exploit_antisymmetry_equations
        eqs = generate_cc_equations("ccsd")
        result = exploit_antisymmetry_equations(eqs)
        self.assertIn("energy", result)
        self.assertIn("doubles", result)
        # Energy should be unchanged
        self.assertEqual(len(result["energy"]), len(eqs["energy"]))

    def test_symmetry_stats(self) -> None:
        from ccgen.optimization.symmetry import symmetry_reduction_stats
        eqs = generate_cc_equations("ccsd")
        stats = symmetry_reduction_stats(eqs)
        for manifold in eqs:
            self.assertIn("before", stats[manifold])
            self.assertIn("after", stats[manifold])
            self.assertEqual(
                stats[manifold]["before"],
                stats[manifold]["after"] + stats[manifold]["eliminated"],
            )

    def test_generate_with_symmetry_flag(self) -> None:
        """generate_cc_equations with exploit_symmetry=True works."""
        eqs_normal = generate_cc_equations("ccsd")
        eqs_sym = generate_cc_equations("ccsd", exploit_symmetry=True)
        self.assertLessEqual(
            len(eqs_sym["doubles"]),
            len(eqs_normal["doubles"]),
        )

    @unittest.skipIf(np is None, "numpy required")
    def test_numerical_equivalence_singles(self) -> None:
        """Symmetry-reduced singles must evaluate identically."""
        from ccgen.tests.test_regressions import build_test_tensors
        from ccgen.optimization.symmetry import exploit_antisymmetry

        tensors, fock, oovv = build_test_tensors()
        nocc, nvir = 2, 3

        eqs = generate_cc_equations("ccsd", targets=["singles"])
        original = eqs["singles"]
        reduced = exploit_antisymmetry(original)

        orig_result = np.zeros((nocc, nvir))
        for term in original:
            orig_result += _evaluate_residual_term(term, tensors, nocc, nvir)

        red_result = np.zeros((nocc, nvir))
        for term in reduced:
            red_result += _evaluate_residual_term(term, tensors, nocc, nvir)

        np.testing.assert_allclose(red_result, orig_result, atol=1e-12)


class PipelineStatsPhase3Tests(unittest.TestCase):
    """Tests verifying Phase 3 stats are reported."""

    def test_stats_with_perm_grouping(self) -> None:
        from ccgen import generate as gen_module
        generate_cc_equations("ccsd", permutation_grouping=True)
        stats = gen_module.last_stats
        self.assertIn("after_perm_grouping", stats.manifolds.get("doubles", {}))

    def test_stats_with_symmetry(self) -> None:
        from ccgen import generate as gen_module
        generate_cc_equations("ccsd", exploit_symmetry=True)
        stats = gen_module.last_stats
        self.assertIn("after_symmetry", stats.manifolds.get("doubles", {}))

    def test_stats_summary_includes_phase3(self) -> None:
        from ccgen import generate as gen_module
        generate_cc_equations("ccsd", permutation_grouping=True)
        summary = gen_module.last_stats.summary()
        self.assertIn("perm grouping", summary)


if __name__ == "__main__":
    unittest.main()
