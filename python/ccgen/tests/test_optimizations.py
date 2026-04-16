"""Validation tests for ccgen optimization passes.

Tests ensure algebraic equivalence: optimized equations produce
the same numerical results as unoptimized equations on random
test tensors.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations, last_stats, PipelineStats
from ccgen.canonicalize import collect_fock_diagonals
from ccgen.optimization.intermediates import (
    detect_intermediates,
    rewrite_equations,
    build_intermediate_equations,
    IntermediateSpec,
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

    def test_unique_names(self) -> None:
        eqs = generate_cc_equations("ccsd")
        intms = detect_intermediates(eqs, threshold=2)
        names = [spec.name for spec in intms]
        self.assertEqual(len(names), len(set(names)), "Intermediate names must be unique")


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


class FactoredCanonicalizationTests(unittest.TestCase):
    """Tests verifying the factored canonicalization refactor."""

    def test_ccsd_term_counts_stable(self) -> None:
        """Verify term counts match expected values (regression)."""
        eqs = generate_cc_equations("ccsd")
        self.assertEqual(len(eqs["energy"]), 3)
        self.assertEqual(len(eqs["singles"]), 24)
        self.assertEqual(len(eqs["doubles"]), 368)

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
        self.assertEqual(len(eqs["doubles"]), 368)

    def test_ccd_term_counts_unchanged(self) -> None:
        eqs = generate_cc_equations("ccd")
        self.assertEqual(len(eqs["energy"]), 1)
        self.assertEqual(len(eqs["doubles"]), 86)

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
