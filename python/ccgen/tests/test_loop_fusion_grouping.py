"""F2: the loop-signature grouping that fusion will key on.

Emits nothing and changes nothing. It pins two properties the fusion work
(F3-F5, docs/CCGEN_WHY_GENERATED_IS_SLOW.md) depends on:

  1. the grouping reproduces the census measured on the emitted C++
     (414 nests -> 13 signatures, largest 81), and
  2. `term_loop_signature` agrees with what `emit_planck_term` actually emits,
     which is the invariant that makes the grouping meaningful at all.

(2) is the load-bearing one. The grouping is only correct if it describes the
emitted loop headers; a helper that drifts from `emit_planck_term` would group
terms that emit DIFFERENT nests, and fusing those is a wrong answer, not a slow
one.
"""

from __future__ import annotations

import re
import unittest

from ccgen.generate import generate_cc_equations
from ccgen.emit.planck_tensor_cpp import (
    emit_planck_term,
    group_terms_by_loop_signature,
    term_loop_signature,
)
from ccgen.lowering.restricted_closed_shell import lower_equations_restricted_closed_shell


def _spatial_triples():
    eqs = generate_cc_equations("ccsdt", engine="diagram", canonical_fock=True)
    return lower_equations_restricted_closed_shell(eqs)["triples"]


class LoopSignatureMatchesEmittedCode(unittest.TestCase):
    def test_signature_equals_the_loops_emit_planck_term_writes(self):
        """The helper must describe the emitted header, not merely resemble it."""
        terms = _spatial_triples()
        self.assertGreater(len(terms), 100, "fixture is too small to be meaningful")

        for term in terms:
            free, summed = term_loop_signature(term)
            text = emit_planck_term(term, lhs="result", indent=4)

            # Everything before `double acc` is the free-index nest; everything
            # between that and the accumulation is the summed nest.
            head, _, tail = text.partition("double acc = 0.0;")
            self.assertTrue(_, "emitted term has no accumulator")
            emitted_free = tuple(re.findall(r"for \(int (\w+) = 0", head))
            emitted_summed = tuple(re.findall(r"for \(int (\w+) = 0", tail))

            self.assertEqual(free, emitted_free)
            self.assertEqual(summed, emitted_summed)


class GroupingReproducesTheCensus(unittest.TestCase):
    def test_undressed_rank3_spatial_triples_collapse_to_eight_signatures(self):
        """UNDRESSED manifold: 399 terms -> 8 signatures, largest 153.

        NOTE the fixture. The 414-nests/13-signatures/81-largest census in
        docs/CCGEN_WHY_GENERATED_IS_SLOW.md was measured on the FACTORIZED
        (`--dressing derived`) TU, whose terms differ -- factorization splits
        contractions and introduces `W_*` operands, changing both the term count
        and the summed-index sets. This fixture is the undressed manifold, and it
        groups even more tightly (8 vs 13). Asserting the factorized numbers here
        would be asserting one fixture's property against another's.
        """
        terms = _spatial_triples()
        groups = group_terms_by_loop_signature(terms)
        sizes = sorted((len(v) for v in groups.values()), reverse=True)

        # Every term is placed exactly once.
        self.assertEqual(sum(sizes), len(terms))
        self.assertEqual(len(terms), 399, f"fixture changed: {len(terms)} terms")

        self.assertEqual(len(groups), 8, f"expected 8 signatures, got {len(groups)}")
        self.assertEqual(sizes[0], 153, f"expected largest group 153, got {sizes[0]}")

    def test_fusion_ratio_is_worth_the_work(self):
        """The ratio is the whole point: it is what fusion buys in traversals."""
        terms = _spatial_triples()
        groups = group_terms_by_loop_signature(terms)
        ratio = len(terms) / len(groups)
        # 399/8 ~ 50x on the undressed manifold; the doc's 32x is the factorized
        # figure. Both are large; this pins that the lever has not evaporated.
        self.assertGreater(ratio, 20.0, f"fusion ratio collapsed to {ratio:.1f}x")

    def test_every_group_is_a_real_fusion_candidate(self):
        """No singletons: a group of one saves nothing and would flatter the ratio."""
        groups = group_terms_by_loop_signature(_spatial_triples())
        singletons = [k for k, v in groups.items() if len(v) == 1]
        self.assertEqual(singletons, [], f"unexpected singleton groups: {singletons}")

    def test_grouping_preserves_emission_order(self):
        """Positions within a group ascend, so a fused emit stays deterministic."""
        groups = group_terms_by_loop_signature(_spatial_triples())
        for key, positions in groups.items():
            self.assertEqual(positions, sorted(positions), f"group {key} out of order")


if __name__ == "__main__":
    unittest.main()
