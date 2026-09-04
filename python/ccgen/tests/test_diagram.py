"""Tests for canonical diagram strings (D1.0-D1.4).

Offline / codegen-inert: ``ccgen.diagram`` is imported by nothing in the
generator, so none of this can change emitted kernels.
"""

from __future__ import annotations

import os
import sys
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.diagram import (  # noqa: E402
    ANCHORS,
    MAX_T_OPERATORS,
    DiagramString,
    admissible_hamiltonian_ranks,
    allocate_indices,
    anchors_for,
    build_line_graph,
    canonical,
    enumerate_diagrams,
    internal_line_count,
    matches_manifold,
    coarse_topology_key,
    diagram_weight_magnitudes,
    diagram_weights,
    enumerate_candidates,
    term_diagram_id,
    enumerate_vertices,
    equivalent,
    exact_topology_key,
    from_string,
    is_wellformed,
    key,
    mu1_sum_range,
    to_string,
    topology_signature,
)
from ccgen.generate import generate_cc_equations  # noqa: E402


def topology_classes(terms) -> Counter:
    """Group generated terms by the coarse key (D1.4's original oracle)."""
    return Counter(coarse_topology_key(t) for t in terms)


def fine_classes(terms) -> Counter:
    """Group generated terms by the finer per-factor signature (D2.0)."""
    return Counter(topology_signature(t) for t in terms)


def t2_ring() -> DiagramString:
    """A doubles diagram: two T2 vertices, projected on the doubles manifold."""
    return DiagramString(t_ops=((2, 2, 1), (2, 1, 1)), bra_level=2, ket_level=0)


class DiagramStringConstructionTests(unittest.TestCase):
    """D1.0 -- the encoding is plain data, read back as given."""

    def test_fields_round_trip_unmodified(self):
        ds = t2_ring()
        self.assertEqual(ds.t_ops, ((2, 2, 1), (2, 1, 1)))
        self.assertEqual(ds.bra_level, 2)
        self.assertEqual(ds.ket_level, 0)

    def test_constructor_does_not_sort(self):
        # Load-bearing: if the constructor normalized, the D1.2 canonical-form
        # tests below would pass vacuously.
        ds = t2_ring()
        self.assertNotEqual(ds.t_ops, tuple(sorted(ds.t_ops)))

    def test_is_hashable_and_value_equal(self):
        self.assertEqual(t2_ring(), t2_ring())
        self.assertEqual(len({t2_ring(), t2_ring()}), 1)


class WellformednessTests(unittest.TestCase):
    """D1.1 -- the stated structural bounds."""

    def test_real_diagram_is_wellformed(self):
        self.assertTrue(is_wellformed(t2_ring()))

    def test_empty_diagram_is_wellformed(self):
        # No cluster operators: the bare Hamiltonian vertex. Nothing to violate.
        self.assertTrue(is_wellformed(DiagramString((), 2, 0)))

    def test_singles_and_triples_levels_are_wellformed(self):
        self.assertTrue(is_wellformed(DiagramString(((1, 2, 1),), 1, 0)))
        self.assertTrue(is_wellformed(DiagramString(((3, 6, 3),), 3, 0)))

    def test_zero_internal_connections_rejected(self):
        # Kallay-Surjan: mu2 runs from 1, not 0. A T operator with no internal
        # line is disconnected from the Hamiltonian, so the diagram is not a
        # connected diagram. (arXiv:2409.06759 Fig. 3.)
        self.assertFalse(is_wellformed(DiagramString(((2, 0, 0),), 2, 0)))

    def test_at_most_four_t_operators(self):
        # A two-body vertex has four lines. Five T operators cannot all connect.
        four = ((1, 1, 1),) * 4
        self.assertTrue(is_wellformed(DiagramString(four, 2, 0)))
        self.assertFalse(is_wellformed(DiagramString(four + ((1, 1, 1),), 2, 0)))

    def test_zero_excitation_level_rejected(self):
        self.assertFalse(is_wellformed(DiagramString(((0, 0, 0),), 2, 0)))

    def test_negative_excitation_level_rejected(self):
        self.assertFalse(is_wellformed(DiagramString(((-1, 0, 0),), 2, 0)))

    def test_internal_connections_above_2mu1_rejected(self):
        self.assertTrue(is_wellformed(DiagramString(((2, 4, 2),), 2, 0)))
        self.assertFalse(is_wellformed(DiagramString(((2, 5, 2),), 2, 0)))

    def test_particle_connections_above_mu1_rejected(self):
        self.assertTrue(is_wellformed(DiagramString(((2, 4, 2),), 2, 0)))
        self.assertFalse(is_wellformed(DiagramString(((2, 4, 3),), 2, 0)))

    def test_particle_connections_above_total_connections_rejected(self):
        # mu3 counts a subset of the mu2 internal lines.
        self.assertFalse(is_wellformed(DiagramString(((3, 1, 2),), 2, 0)))

    def test_negative_connection_counts_rejected(self):
        self.assertFalse(is_wellformed(DiagramString(((2, -1, 0),), 2, 0)))
        self.assertFalse(is_wellformed(DiagramString(((2, 1, -1),), 2, 0)))

    def test_paper_bounds_are_exactly_reproduced(self):
        # Enumerate the legal (mu2, mu3) box for mu1 = 1..3 and check it
        # against the paper's stated ranges directly, rather than spot-checking.
        for mu1 in (1, 2, 3):
            expected = {
                (mu2, mu3)
                for mu2 in range(1, 2 * mu1 + 1)
                for mu3 in range(0, min(mu1, mu2) + 1)
            }
            got = {
                (mu2, mu3)
                for mu2 in range(-1, 2 * mu1 + 3)
                for mu3 in range(-1, mu1 + 3)
                if is_wellformed(DiagramString(((mu1, mu2, mu3),), 2, 0))
            }
            with self.subTest(mu1=mu1):
                self.assertEqual(got, expected)

    def test_negative_manifold_levels_rejected(self):
        self.assertFalse(is_wellformed(DiagramString(((2, 1, 1),), -1, 0)))
        self.assertFalse(is_wellformed(DiagramString(((2, 1, 1),), 2, -1)))

    def test_malformed_triplet_length_rejected(self):
        self.assertFalse(is_wellformed(DiagramString(((2, 1),), 2, 0)))


class CanonicalFormTests(unittest.TestCase):
    """D1.2 -- the equivalence claim: same topology iff same key."""

    def test_canonical_sorts_the_vertices(self):
        self.assertEqual(canonical(t2_ring()).t_ops, ((2, 1, 1), (2, 2, 1)))

    def test_canonical_preserves_manifold_levels(self):
        c = canonical(t2_ring())
        self.assertEqual((c.bra_level, c.ket_level), (2, 0))

    def test_canonical_is_idempotent(self):
        # The term path's canonicalize_term is NOT idempotent (it needs a
        # fixed-point loop). Pin that the diagram layer does not inherit that.
        once = canonical(t2_ring())
        self.assertEqual(canonical(once), once)

    def test_vertex_permutations_collide(self):
        a = DiagramString(((2, 2, 1), (2, 1, 1)), 2, 0)
        b = DiagramString(((2, 1, 1), (2, 2, 1)), 2, 0)
        self.assertNotEqual(a, b)          # distinct as written
        self.assertEqual(key(a), key(b))   # same topology
        self.assertTrue(equivalent(a, b))

    def test_three_vertex_permutations_all_collide(self):
        from itertools import permutations

        vertices = ((1, 1, 0), (2, 2, 1), (2, 3, 2))
        keys = {key(DiagramString(p, 2, 0)) for p in permutations(vertices)}
        self.assertEqual(len(keys), 1)

    def test_distinct_topologies_do_not_collide(self):
        a = DiagramString(((2, 2, 1), (2, 1, 1)), 2, 0)
        differing = [
            DiagramString(((2, 2, 1), (2, 1, 0)), 2, 0),  # mu3 differs
            DiagramString(((2, 2, 1), (1, 1, 1)), 2, 0),  # mu1 differs
            DiagramString(((2, 2, 1),), 2, 0),            # vertex count differs
            DiagramString(((2, 2, 1), (2, 1, 1)), 1, 0),  # bra level differs
            DiagramString(((2, 2, 1), (2, 1, 1)), 2, 1),  # ket level differs
        ]
        for b in differing:
            with self.subTest(other=b):
                self.assertNotEqual(key(a), key(b))
                self.assertFalse(equivalent(a, b))

    def test_key_is_hashable(self):
        self.assertEqual(len({key(t2_ring()), key(t2_ring())}), 1)

    def test_repeated_vertices_are_not_deduplicated(self):
        # Two identical T2 vertices is a different diagram from one.
        a = DiagramString(((2, 1, 1), (2, 1, 1)), 2, 0)
        b = DiagramString(((2, 1, 1),), 2, 0)
        self.assertNotEqual(key(a), key(b))


class TextFormTests(unittest.TestCase):
    """D1.3 -- the compact text form round-trips."""

    def test_round_trip_on_a_real_diagram(self):
        ds = t2_ring()
        self.assertEqual(from_string(to_string(ds)), ds)

    def test_rendered_form_is_readable(self):
        self.assertEqual(to_string(t2_ring()), "2:0|2,2,1;2,1,1")

    def test_round_trip_preserves_vertex_order(self):
        # Not canonicalized on the way through: a raw diagram survives as-is.
        raw = DiagramString(((2, 2, 1), (1, 1, 0)), 2, 0)
        self.assertEqual(from_string(to_string(raw)).t_ops, raw.t_ops)

    def test_empty_vertex_list_round_trips(self):
        ds = DiagramString((), 2, 0)
        self.assertEqual(to_string(ds), "2:0|")
        self.assertEqual(from_string("2:0|"), ds)

    def test_round_trip_over_all_anchors(self):
        for name, ds in ANCHORS.items():
            with self.subTest(anchor=name):
                self.assertEqual(from_string(to_string(ds)), ds)

    def test_malformed_strings_raise(self):
        for bad in [
            "2,0|2,2,1",      # missing '|'
            "20|2,2,1",       # missing ':'
            "x:0|2,2,1",      # non-integer level
            "2:0|2,2",        # vertex not a triple
            "2:0|2,2,1,4",    # vertex over-long
            "2:0|a,b,c",      # non-integer vertex
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    from_string(bad)


class AnchorTests(unittest.TestCase):
    """D1.4 -- the hand-derived fixtures that seed D2's oracle."""

    def test_every_anchor_is_wellformed(self):
        for name, ds in ANCHORS.items():
            with self.subTest(anchor=name):
                self.assertTrue(is_wellformed(ds), f"{name} = {to_string(ds)}")

    def test_anchor_keys_are_all_distinct(self):
        keys = {name: key(ds) for name, ds in ANCHORS.items()}
        self.assertEqual(
            len(set(keys.values())), len(ANCHORS),
            "two anchors share a canonical key -- they are the same topology",
        )

    def test_anchors_for_filters_by_prefix(self):
        ccd = anchors_for("ccd_")
        self.assertTrue(ccd)
        self.assertTrue(all(k.startswith("ccd_") for k in ccd))
        self.assertNotIn("ccsd_singles_t1_v", ccd)

    def test_ladders_are_distinguished_by_mu3(self):
        # pp ladder contracts two particle lines, hh ladder two hole lines.
        # If mu3 did not separate them the encoding would be losing physics.
        pp = ANCHORS["ccd_doubles_pp_ladder"]
        hh = ANCHORS["ccd_doubles_hh_ladder"]
        ring = ANCHORS["ccd_doubles_ring"]
        self.assertEqual(pp.t_ops[0][:2], hh.t_ops[0][:2])
        self.assertNotEqual(key(pp), key(hh))
        self.assertNotEqual(key(pp), key(ring))
        self.assertNotEqual(key(hh), key(ring))


class TopologyClassOracleTests(unittest.TestCase):
    """D2.0 -- the counts D2 is graded against, measured from the term path.

    These pin the *duplication* the diagram representation exists to remove:
    the term path emits many labeled terms per topology, and the gap between
    the two numbers is the waste.  If a future change to the generator moves
    these counts, D2's target moves with it and this test says so loudly.

    Both counts are LOWER BOUNDS on the true diagram count (see the note in
    ``ccgen.diagram``); the fine one is the tighter bound and the one D2.3
    should aim at.  Neither is authoritative -- D3's multiset gate is.
    """

    # (method, manifold): (terms, coarse classes, fine classes)
    # Counts as of the canonicalize is_dummy / canonical-Fock fixes (the ones the
    # retracted "raw-generation bug" investigation actually landed): the term
    # counts dropped from the pre-fix numbers (ccsd doubles 123->70) because the
    # false-zero recovery + relabel changes let genuinely-equal terms merge.
    EXPECTED = {
        ("ccd", "energy"): (1, 1, 1),
        ("ccd", "doubles"): (18, 7, 10),
        ("ccsd", "energy"): (3, 3, 3),
        ("ccsd", "singles"): (16, 12, 14),
        ("ccsd", "doubles"): (70, 19, 31),
    }

    def test_measured_class_counts(self):
        for (method, manifold), expected in self.EXPECTED.items():
            with self.subTest(method=method, manifold=manifold):
                terms = generate_cc_equations(method)[manifold]
                got = (
                    len(terms),
                    len(topology_classes(terms)),
                    len(fine_classes(terms)),
                )
                self.assertEqual(got, expected)

    def test_fine_key_is_at_least_as_discriminating_as_coarse(self):
        for method, manifold in self.EXPECTED:
            with self.subTest(method=method, manifold=manifold):
                terms = generate_cc_equations(method)[manifold]
                self.assertGreaterEqual(
                    len(fine_classes(terms)), len(topology_classes(terms))
                )

    def test_exact_key_brackets_the_true_count_from_above(self):
        # coarse <= fine <= exact: the counting keys undercount, the exact key
        # overcounts (it cannot identify exchanges of repeated factors).
        for method, manifold in self.EXPECTED:
            terms = generate_cc_equations(method)[manifold]
            with self.subTest(method=method, manifold=manifold):
                self.assertLessEqual(
                    len(fine_classes(terms)),
                    len({exact_topology_key(t) for t in terms}),
                )

    def test_exact_key_counts(self):
        expected = {
            ("ccd", "energy"): 1,
            ("ccd", "doubles"): 11,     # now MATCHES hand-derivation (was 12)
            ("ccsd", "energy"): 3,
            ("ccsd", "singles"): 16,
            ("ccsd", "doubles"): 33,
        }
        for (method, manifold), want in expected.items():
            terms = generate_cc_equations(method)[manifold]
            with self.subTest(method=method, manifold=manifold):
                self.assertEqual(
                    len({exact_topology_key(t) for t in terms}), want
                )

    def test_linear_ccd_diagrams_match_hand_derivation(self):
        # The part that IS exactly hand-checkable: the six linear CCD doubles
        # diagrams -- bare ERI, two Fock terms, pp ladder, hh ladder, ring.
        terms = generate_cc_equations("ccd")["doubles"]
        linear = [
            t for t in terms
            if sum(f.name.startswith("t") for f in t.factors) <= 1
        ]
        self.assertEqual(len({exact_topology_key(t) for t in linear}), 6)

    def test_t2v_terms_are_exactly_three_diagrams(self):
        # pp ladder, hh ladder, ring. The 4 ring terms differ only by the
        # P(ij)P(ab) antisymmetrizer and must NOT count as separate diagrams.
        terms = [
            t for t in generate_cc_equations("ccd")["doubles"]
            if tuple(sorted(f.name for f in t.factors)) == ("t2", "v")
        ]
        self.assertEqual(len(terms), 6)
        self.assertEqual(len({exact_topology_key(t) for t in terms}), 3)

    def test_repeated_factor_exchange_overcount_is_gone(self):
        # PREVIOUSLY a known overcount (24 terms -> 6 exact classes, where
        # textbook CCD has 5 quadratic diagrams). The canonicalize is_dummy /
        # relabel fixes eliminated it: the t2*t2*v terms now merge to the
        # textbook 5 exact classes. This is the SAME repeated-factor exchange
        # symmetry, now correctly identified rather than overcounted.
        quad = [
            t for t in generate_cc_equations("ccd")["doubles"]
            if tuple(sorted(f.name for f in t.factors)) == ("t2", "t2", "v")
        ]
        self.assertEqual(len(quad), 7)
        self.assertEqual(len({exact_topology_key(t) for t in quad}), 5)

    def test_quadratic_ccd_terms_are_not_one_topology(self):
        # The coarse-key undercount that motivated D2.0 still holds: the
        # t2*t2*v terms share ONE coarse key but span several fine classes
        # (the coarse key is a genuine lower bound). Term count is now 7 (was 24
        # pre-fix), fine classes 4.
        terms = [
            t for t in generate_cc_equations("ccd")["doubles"]
            if tuple(sorted(f.name for f in t.factors)) == ("t2", "t2", "v")
        ]
        self.assertEqual(len(terms), 7)
        self.assertEqual(len(topology_classes(terms)), 1)
        self.assertEqual(len(fine_classes(terms)), 4)

    def test_ccd_ladders_and_ring_are_present_in_the_term_path(self):
        # The anchors claim pp/hh ladders and a ring exist in CCD doubles.
        # Check the term path really has one class each, so the fixtures are
        # anchored to something real rather than to my reading of the papers.
        classes = topology_classes(generate_cc_equations("ccd")["doubles"])
        self.assertEqual(classes[(("t2", "v"), 0, 2)], 1)   # pp ladder
        self.assertEqual(classes[(("t2", "v"), 2, 0)], 1)   # hh ladder
        self.assertEqual(classes[(("t2", "v"), 1, 1)], 4)   # ring
        self.assertEqual(classes[(("v",), 0, 0)], 1)        # bare ERI

    def test_diagram_classes_are_fewer_than_terms(self):
        # The premise of the whole D-series, stated as a test.
        for method, manifold in [("ccd", "doubles"), ("ccsd", "doubles")]:
            with self.subTest(method=method):
                terms = generate_cc_equations(method)[manifold]
                self.assertLess(len(topology_classes(terms)), len(terms))


class VertexEnumerationTests(unittest.TestCase):
    """D2.1 -- the single-operator generator."""

    def test_level_one_vertices_by_hand(self):
        # mu1=1: mu2 in 1..2, mu3 in 0..min(1,mu2).
        self.assertEqual(
            enumerate_vertices(1),
            [(1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)],
        )

    def test_level_two_vertices_by_hand(self):
        self.assertEqual(
            enumerate_vertices(2),
            [
                (2, 1, 0), (2, 1, 1),
                (2, 2, 0), (2, 2, 1), (2, 2, 2),
                (2, 3, 0), (2, 3, 1), (2, 3, 2),
                (2, 4, 0), (2, 4, 1), (2, 4, 2),
            ],
        )

    def test_counts_match_closed_form(self):
        # Independent of the loop: sum over mu2 of (min(mu1,mu2)+1).
        for mu1 in range(1, 7):
            expected = sum(
                min(mu1, mu2) + 1 for mu2 in range(1, 2 * mu1 + 1)
            )
            with self.subTest(mu1=mu1):
                self.assertEqual(len(enumerate_vertices(mu1)), expected)

    def test_every_emitted_vertex_is_wellformed(self):
        for mu1 in range(1, 7):
            for v in enumerate_vertices(mu1):
                with self.subTest(vertex=v):
                    self.assertTrue(is_wellformed(DiagramString((v,), 2, 0)))

    def test_emits_no_duplicates(self):
        for mu1 in range(1, 7):
            got = enumerate_vertices(mu1)
            with self.subTest(mu1=mu1):
                self.assertEqual(len(got), len(set(got)))

    def test_output_is_sorted(self):
        for mu1 in range(1, 7):
            got = enumerate_vertices(mu1)
            with self.subTest(mu1=mu1):
                self.assertEqual(got, sorted(got))

    def test_is_exactly_the_wellformed_set(self):
        # The generator and the predicate must agree -- neither is allowed to
        # drift from the other.
        for mu1 in range(1, 6):
            brute = [
                (mu1, mu2, mu3)
                for mu2 in range(0, 2 * mu1 + 3)
                for mu3 in range(0, mu1 + 3)
                if is_wellformed(DiagramString(((mu1, mu2, mu3),), 2, 0))
            ]
            with self.subTest(mu1=mu1):
                self.assertEqual(enumerate_vertices(mu1), sorted(brute))

    def test_rejects_nonpositive_level(self):
        with self.assertRaises(ValueError):
            enumerate_vertices(0)


class Mu1SumRangeTests(unittest.TestCase):
    """D2.1 -- the projection-level window on sum(mu1)."""

    def test_window_is_centred_on_the_bra_level(self):
        # k=2 with plenty of headroom: 2-2 .. 2+2, clamped at 1.
        self.assertEqual(list(mu1_sum_range(2, 4, 4)), [1, 2, 3, 4])

    def test_lower_limit_clamped_at_one(self):
        # k=0 would give -2; a connected diagram has at least one T operator.
        self.assertEqual(mu1_sum_range(0, 4, 4).start, 1)

    def test_per_operator_cap_applies_to_the_total(self):
        # One T operator of max rank 2 cannot sum past 2, whatever k allows.
        self.assertEqual(list(mu1_sum_range(2, 2, 1)), [1, 2])
        # Two of them can reach 4.
        self.assertEqual(list(mu1_sum_range(2, 2, 2)), [1, 2, 3, 4])

    def test_rejects_zero_operators(self):
        with self.assertRaises(ValueError):
            mu1_sum_range(2, 2, 0)

    def test_covers_every_rank_the_term_path_actually_produces(self):
        # The load-bearing check, and the one that caught the literal reading
        # of the paper: taking n as a bound on the SUM gives 1..2 for CCSD
        # doubles, but 99 of its 123 terms sit at sum(mu1) = 3 or 4.
        levels = {"energy": 0, "singles": 1, "doubles": 2, "triples": 3}
        for method, max_rank in [("ccd", 2), ("ccsd", 2), ("ccsdt", 3)]:
            for manifold, terms in generate_cc_equations(method).items():
                observed = {
                    total
                    for t in terms
                    if (total := sum(
                        int(f.name[1:]) for f in t.factors
                        if f.name.startswith("t")
                    ))
                }
                allowed: set[int] = set()
                for n_ops in range(1, MAX_T_OPERATORS + 1):
                    allowed |= set(
                        mu1_sum_range(levels[manifold], max_rank, n_ops)
                    )
                with self.subTest(method=method, manifold=manifold):
                    self.assertTrue(
                        observed <= allowed,
                        f"{sorted(observed - allowed)} outside {sorted(allowed)}",
                    )


class CandidateEnumerationTests(unittest.TestCase):
    """D2.2 -- multi-vertex combination, canonical by construction."""

    def test_no_two_candidates_share_a_key(self):
        # THE property the whole D-series exists for: the term path generates
        # duplicates and removes them afterwards; this never makes them.
        for ranks, bra in [([2], 2), ([1, 2], 1), ([1, 2], 2), ([1, 2, 3], 3)]:
            cands = enumerate_candidates(ranks, bra)
            with self.subTest(ranks=ranks, bra=bra):
                self.assertEqual(len({key(c) for c in cands}), len(cands))

    def test_every_candidate_is_canonical_and_wellformed(self):
        for c in enumerate_candidates([1, 2], 2):
            with self.subTest(cand=to_string(c)):
                self.assertTrue(is_wellformed(c))
                self.assertEqual(c, canonical(c))

    def test_candidates_respect_the_mu1_sum_window(self):
        for c in enumerate_candidates([1, 2], 2):
            total = sum(v[0] for v in c.t_ops)
            allowed = set(mu1_sum_range(2, 2, len(c.t_ops)))
            with self.subTest(cand=to_string(c)):
                self.assertIn(total, allowed)

    def test_operator_count_is_capped(self):
        for c in enumerate_candidates([1, 2], 2):
            self.assertLessEqual(len(c.t_ops), MAX_T_OPERATORS)
        capped = enumerate_candidates([1, 2], 2, max_operators=2)
        self.assertTrue(all(len(c.t_ops) <= 2 for c in capped))

    def test_manifold_levels_are_carried_through(self):
        for c in enumerate_candidates([1, 2], 2, ket_level=1):
            self.assertEqual((c.bra_level, c.ket_level), (2, 1))

    def test_is_a_superset_of_the_anchors(self):
        # Every hand-derived CCSD-reachable anchor must survive candidate
        # generation, or D2.3 has nothing to filter down to.
        cands = {
            key(c)
            for bra in (0, 1, 2)
            for c in enumerate_candidates([1, 2], bra)
        }
        for name, ds in ANCHORS.items():
            if not ds.t_ops:
                continue  # bare-Hamiltonian diagrams have no T operators
            if any(v[0] > 2 for v in ds.t_ops):
                continue  # beyond CCSD's ranks
            with self.subTest(anchor=name):
                self.assertIn(key(ds), cands)

    def test_overgenerates_relative_to_the_real_count(self):
        # Stated as a test so the D2.3 filter has a documented starting gap:
        # candidates >> the fine-class count it must reduce to.
        cands = enumerate_candidates([1, 2], 2)
        self.assertGreater(len(cands), 31)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            enumerate_candidates([0], 2)
        with self.assertRaises(ValueError):
            enumerate_candidates([1, 2], 2, max_operators=-1)

    def test_empty_ranks_yields_nothing(self):
        self.assertEqual(enumerate_candidates([], 2), [])


def truth_diagrams(method: str, manifold: str) -> set:
    """The real ``(t_ops, h_rank)`` set, read off the existing term path.

    Each generated term names its own diagram: a t-factor of rank n whose
    indices are partly summed gives ``(n, #summed, #summed-and-virtual)``, and
    the presence of an ``f`` factor says the Hamiltonian vertex is one-body.
    This is the D2.3 oracle -- derived from terms known to be correct, not from
    a count someone wrote down.
    """
    return {
        term_diagram_id(t)
        for t in generate_cc_equations(method)[manifold]
        if term_diagram_id(t)[0]
    }


class DiagramClosureTests(unittest.TestCase):
    """D2.3 -- the closure filter, graded against the term path exactly."""

    RANKS = {"ccd": [1, 2], "ccsd": [1, 2], "ccsdt": [1, 2, 3]}
    LEVELS = {"energy": 0, "singles": 1, "doubles": 2, "triples": 3}

    def _compare(self, method, manifold):
        ranks = [2] if method == "ccd" else self.RANKS[method]
        truth = truth_diagrams(method, manifold)
        mine = {
            (d.t_ops, h)
            for d, h in enumerate_diagrams(ranks, self.LEVELS[manifold])
        }
        return truth, mine

    def test_matches_the_term_path_exactly(self):
        # The D2 pass/fail. Not a count: set equality against every diagram the
        # existing (correct) generator actually produces.
        for method in ("ccd", "ccsd", "ccsdt"):
            for manifold in generate_cc_equations(method):
                truth, mine = self._compare(method, manifold)
                with self.subTest(method=method, manifold=manifold):
                    self.assertEqual(
                        mine - truth, set(), "generated diagrams that are not real"
                    )
                    self.assertEqual(
                        truth - mine, set(), "real diagrams that were rejected"
                    )

    def test_ccsdt_was_not_fitted_against(self):
        # The generalization evidence: the per-species cap was derived from CCD
        # doubles alone, then reproduced CCSDT's triples manifold untouched.
        truth, mine = self._compare("ccsdt", "triples")
        self.assertEqual(len(truth), 47)
        self.assertEqual(mine, truth)

    def test_filter_actually_removes_most_candidates(self):
        cands = len(enumerate_candidates([1, 2], 2))
        kept = len(enumerate_diagrams([1, 2], 2))
        self.assertLess(kept, cands // 4)

    def test_every_anchor_closes(self):
        for name, ds in ANCHORS.items():
            if not ds.t_ops:
                continue
            with self.subTest(anchor=name):
                self.assertTrue(
                    admissible_hamiltonian_ranks(ds),
                    f"{name} = {to_string(ds)} does not close",
                )

    def test_pp_and_hh_ladders_survive(self):
        # Pinned because an earlier over-strict per-species balance rule
        # rejected exactly these two textbook CCD diagrams.
        for name in ("ccd_doubles_pp_ladder", "ccd_doubles_hh_ladder"):
            with self.subTest(anchor=name):
                self.assertIn(2, admissible_hamiltonian_ranks(ANCHORS[name]))

    def test_per_species_cap_rejects_imbalanced_internals(self):
        # (2,2,2)+(2,2,2) sends 4 particle lines into a 2-particle-slot vertex.
        both_particle = DiagramString(((2, 2, 2), (2, 2, 2)), 2, 0)
        self.assertFalse(matches_manifold(both_particle, h_rank=2))
        both_hole = DiagramString(((2, 2, 0), (2, 2, 0)), 2, 0)
        self.assertFalse(matches_manifold(both_hole, h_rank=2))
        # The balanced partner is real.
        self.assertTrue(
            matches_manifold(DiagramString(((2, 2, 0), (2, 2, 2)), 2, 0), h_rank=2)
        )

    def test_nonzero_ket_is_rejected_not_guessed(self):
        self.assertFalse(matches_manifold(DiagramString(((2, 2, 1),), 2, 1)))


class DiagramWeightOracleTests(unittest.TestCase):
    """D3.0 -- reading diagram weights off the term path, and where it stops."""

    def test_term_diagram_id_agrees_with_the_enumerator(self):
        # The oracle and D2's enumerator must be talking about the same
        # diagrams, or the weights are keyed to nothing.
        for method, ranks, manifold, level in [
            ("ccd", [2], "doubles", 2),
            ("ccsd", [1, 2], "singles", 1),
            ("ccsd", [1, 2], "doubles", 2),
        ]:
            terms = generate_cc_equations(method)[manifold]
            from_terms = {
                term_diagram_id(t) for t in terms if term_diagram_id(t)[0]
            }
            from_enum = {(d.t_ops, h) for d, h in enumerate_diagrams(ranks, level)}
            with self.subTest(method=method, manifold=manifold):
                self.assertEqual(from_terms, from_enum)

    def test_ccd_signed_sums_after_the_canonicalize_fixes(self):
        # `diagram_weights` is the D3.0-era per-diagram SIGNED SUM -- superseded
        # by the solve-free `diagram_signed_weight` (the actual weight). This
        # pins its current CCD output: after the canonicalize is_dummy / relabel
        # fixes, many diagrams' signed sums collapse to 0 (the two P(ij)
        # antisymmetrizer halves now merge and cancel), which is exactly the D3.0
        # limitation that motivated moving to the structural weight. Kept as a
        # regression on the diagnostic, not a correctness claim about the weight.
        weights = diagram_weights(generate_cc_equations("ccd")["doubles"])
        expected = {
            ((((2, 1, 0),)), 1): Fraction(0),
            ((((2, 1, 1),)), 1): Fraction(0),
            ((((2, 2, 0),)), 2): Fraction(1, 2),        # hh ladder
            ((((2, 2, 2),)), 2): Fraction(1, 2),        # pp ladder
            ((((2, 2, 1),)), 2): Fraction(0),
            (((2, 2, 0), (2, 2, 2)), 2): Fraction(1, 4),
            (((2, 2, 1), (2, 2, 1)), 2): Fraction(0),
            (((2, 1, 0), (2, 3, 2)), 2): Fraction(0),
            (((2, 1, 1), (2, 3, 1)), 2): Fraction(-1),
        }
        self.assertEqual(
            {k: v for k, v in weights.items() if k[0]}, expected
        )

    def test_ring_ring_diagram_after_the_canonicalize_fixes(self):
        # PRE-FIX the ring-ring diagram expanded to 10 raggedly-weighted terms
        # (1/32, 1/16, 3/32 summing to 1/2). After the canonicalize relabel
        # fixes it is 2 equal-and-opposite terms summing to 0 -- the P(ij)
        # halves now merge and cancel (see the D3.0 note). The structural weight
        # is supplied by `diagram_signed_weight`, not this signed sum.
        terms = generate_cc_equations("ccd")["doubles"]
        target = (((2, 2, 1), (2, 2, 1)), 2)
        members = [t for t in terms if term_diagram_id(t) == target]
        self.assertEqual(len(members), 2)
        self.assertEqual(sorted(t.coeff for t in members), [Fraction(-1), Fraction(1)])
        self.assertEqual(sum(t.coeff for t in members), Fraction(0))

    def test_signed_sums_collapse_to_zero_beyond_ccd(self):
        # THE D3.0 limitation, pinned so D3.3 cannot be built on a false
        # premise: the two halves of a P(ij) antisymmetrizer share a diagram id
        # and cancel exactly.
        weights = diagram_weights(generate_cc_equations("ccsd")["doubles"])
        zeros = [k for k, v in weights.items() if k[0] and v == 0]
        self.assertTrue(zeros, "expected P-halves cancellation in ccsd/doubles")

        # The clearest instance: one t1 operator, two terms, equal and opposite.
        target = ((((1, 1, 1),)), 2)
        members = [
            t for t in generate_cc_equations("ccsd")["doubles"]
            if term_diagram_id(t) == target
        ]
        self.assertEqual(len(members), 2)
        self.assertEqual(sorted(t.coeff for t in members), [-1, 1])
        self.assertEqual(sum(t.coeff for t in members), 0)

    def test_magnitudes_are_zero_free_everywhere(self):
        # The usable fingerprint -- but note the denominators below.
        for method in ("ccd", "ccsd", "ccsdt"):
            for manifold, terms in generate_cc_equations(method).items():
                mags = {
                    k: v for k, v in diagram_weight_magnitudes(terms).items()
                    if k[0]
                }
                with self.subTest(method=method, manifold=manifold):
                    self.assertTrue(all(v > 0 for v in mags.values()))

    def test_magnitude_diagnostic_denominators_are_now_clean(self):
        # PRE-FIX this diagnostic (`diagram_weight_magnitudes`, the D3.0 sum of
        # |coeff|) had denominators up to 8 -- it summed over the P expansion
        # without dividing, the missing divisor D3.3 owed. After the canonicalize
        # relabel fixes the ccsd doubles magnitudes are clean dyadic (denominator
        # <= 4). The authoritative weight is `diagram_signed_weight`; this just
        # pins that the raw diagnostic no longer overcounts the P expansion.
        mags = diagram_weight_magnitudes(generate_cc_equations("ccsd")["doubles"])
        self.assertTrue(
            all(v.denominator <= 4 for k, v in mags.items() if k[0]),
        )


class LineGraphTests(unittest.TestCase):
    """D3.1 -- the edge-list form the triplets expand into."""

    RANKS = {"ccd": [2], "ccsd": [1, 2], "ccsdt": [1, 2, 3]}
    LEVELS = {"energy": 0, "singles": 1, "doubles": 2, "triples": 3}

    def _all_diagrams(self):
        for method, ranks in self.RANKS.items():
            for manifold in generate_cc_equations(method):
                for d, h in enumerate_diagrams(ranks, self.LEVELS[manifold]):
                    yield method, manifold, d, h

    def test_internal_lines_match_sum_mu2(self):
        for _m, _man, d, h in self._all_diagrams():
            g = build_line_graph(d, h)
            internal = sum(
                1 for sp, a, b in g.lines if "H" in (a, b) and "bra" not in (a, b)
            )
            with self.subTest(diagram=to_string(d), h=h):
                self.assertEqual(internal, internal_line_count(d))

    def test_externals_balance_the_bra(self):
        # k particle + k hole reach the bra, for every closed diagram.
        for _m, _man, d, h in self._all_diagrams():
            g = build_line_graph(d, h)
            with self.subTest(diagram=to_string(d), h=h):
                self.assertEqual(g.external_particles, d.bra_level)
                self.assertEqual(g.external_holes, d.bra_level)

    def test_every_line_has_two_endpoints(self):
        for _m, _man, d, h in self._all_diagrams():
            for sp, a, b in build_line_graph(d, h).lines:
                with self.subTest(diagram=to_string(d)):
                    self.assertIn(sp, ("p", "h"))
                    self.assertNotEqual(a, b)

    def test_ring_graph_by_hand(self):
        # ring (2,2,1): t2 sends 1 particle + 1 hole internal, 1 particle +
        # 1 hole external; the two-body vertex emits 1 particle + 1 hole of its
        # own to complete the bra.
        g = build_line_graph(DiagramString(((2, 2, 1),), 2, 0), 2)
        counts = Counter((sp, a, b) for sp, a, b in g.lines)
        self.assertEqual(counts[("p", ("t", 0), "H")], 1)
        self.assertEqual(counts[("h", ("t", 0), "H")], 1)
        self.assertEqual(counts[("p", ("t", 0), "bra")], 1)
        self.assertEqual(counts[("h", ("t", 0), "bra")], 1)
        self.assertEqual(counts[("p", "H", "bra")], 1)
        self.assertEqual(counts[("h", "H", "bra")], 1)

    def test_pp_ladder_takes_both_particle_externals_from_the_vertex(self):
        # pp ladder (2,2,2): t2's two particle lines are BOTH internal, so both
        # particle externals must come from the V vertex. This is the case the
        # over-strict per-species balance rule wrongly rejected in D2.3.
        g = build_line_graph(DiagramString(((2, 2, 2),), 2, 0), 2)
        h_particle_ext = sum(
            1 for sp, a, b in g.lines if sp == "p" and a == "H" and b == "bra"
        )
        t_particle_ext = sum(
            1 for sp, a, b in g.lines if sp == "p" and a == ("t", 0) and b == "bra"
        )
        self.assertEqual(t_particle_ext, 0)
        self.assertEqual(h_particle_ext, 2)

    def test_rejects_a_nonclosing_diagram(self):
        # (2,4,2) does not close on the doubles manifold (all lines internal).
        with self.assertRaises(ValueError):
            build_line_graph(DiagramString(((2, 4, 2),), 2, 0), 2)


class IndexAllocationTests(unittest.TestCase):
    """D3.2a -- the index pool allocator and its collision guard."""

    def test_doubles_allocation_matches_the_term_path_convention(self):
        # A doubles diagram with two occ + two vir dummies: externals i,j / a,b
        # and dummies from beyond them -- exactly t2(c,d,i,j) v(c,d,a,b) naming.
        p = allocate_indices(2, 2, 2)
        self.assertEqual([i.name for i in p.ext_occ], ["i", "j"])
        self.assertEqual([i.name for i in p.ext_vir], ["a", "b"])
        self.assertEqual([i.name for i in p.dummy_occ], ["k", "l"])
        self.assertEqual([i.name for i in p.dummy_vir], ["c", "d"])

    def test_dummies_never_collide_with_externals(self):
        # The apply_deltas guard, across a range of sizes.
        for bra in range(0, 4):
            for nd_o in range(0, 4):
                for nd_v in range(0, 4):
                    p = allocate_indices(bra, nd_o, nd_v)
                    ext = {i.name for i in p.ext_occ + p.ext_vir}
                    dum = {i.name for i in p.dummy_occ + p.dummy_vir}
                    with self.subTest(bra=bra, nd_o=nd_o, nd_v=nd_v):
                        self.assertEqual(ext & dum, set())

    def test_counts_are_exact(self):
        p = allocate_indices(3, 1, 4)
        self.assertEqual(
            (len(p.ext_occ), len(p.ext_vir), len(p.dummy_occ), len(p.dummy_vir)),
            (3, 3, 1, 4),
        )

    def test_externals_are_not_dummy_flagged(self):
        p = allocate_indices(2, 1, 1)
        self.assertTrue(all(not i.is_dummy for i in p.ext_occ + p.ext_vir))
        self.assertTrue(all(i.is_dummy for i in p.dummy_occ + p.dummy_vir))

    def test_high_rank_extends_the_pool(self):
        # More dummies than the base alphabet: must still be disjoint, no crash.
        p = allocate_indices(3, 6, 6)
        self.assertEqual(len(p.all_names), 3 + 3 + 6 + 6)


class ResidualEvaluatorTests(unittest.TestCase):
    """D3.2b-i -- the numerical gate the diagram assembly must pass."""

    def test_is_deterministic_for_fixed_tensors(self):
        from ccgen.tests.residual_eval import residual_of, random_tensors

        tn = random_tensors(3, 4, seed=7)
        terms = generate_cc_equations("ccd")["doubles"]
        import numpy as np

        self.assertTrue(
            np.allclose(residual_of(terms, 3, 4, tn), residual_of(terms, 3, 4, tn))
        )

    def test_ccd_residual_has_the_right_antisymmetry(self):
        from ccgen.tests.residual_eval import residual_of, random_tensors
        import numpy as np

        tn = random_tensors(3, 4, seed=7)
        R = residual_of(generate_cc_equations("ccd")["doubles"], 3, 4, tn)
        self.assertTrue(np.allclose(R, -R.transpose(0, 1, 3, 2)))  # i<->j
        self.assertTrue(np.allclose(R, -R.transpose(1, 0, 2, 3)))  # a<->b

    def test_single_diagram_residual_is_reproducible(self):
        # A single diagram's terms evaluate to a stable array -- the object the
        # assembly is graded against.
        from ccgen.tests.residual_eval import residual_of, random_tensors
        import numpy as np

        tn = random_tensors(3, 4, seed=1)
        pp = [
            t for t in generate_cc_equations("ccd")["doubles"]
            if term_diagram_id(t) == ((((2, 2, 2),)), 2)
        ]
        self.assertEqual(len(pp), 1)
        R = residual_of(pp, 3, 4, tn)
        self.assertGreater(np.linalg.norm(R), 0)


# The former `LadderAssemblyTests` (ladder + mixed assembly gates) was removed:
# it graded `diagram_representative` by proportionality to
# `residual_of(generate_cc_equations(...) members)` -- using the term-path
# GENERATOR (the code under test, known-buggy on the t1*t2*v weights) as the
# topology oracle. A diagram gate must never be pinned to ccgen's own algebra
# output. The assembly is validated PySCF-only by
# `test_reference_vs_pyscf.py::test_diagram_basis_spans_the_pyscf_doubles_residual`
# (the full doubles basis reproduces PySCF exactly via a full-rank solve --
# strictly stronger than per-diagram proportionality to ccgen), so the deleted
# tests added no coverage the PySCF span test lacks.


class DiagramSignTests(unittest.TestCase):
    """AR2.1: `diagram_sign` = Crawford's `(-1)^(h+l)`, validated against the
    worked examples in Crawford & Schaefer III, Rev. Comput. Chem. 14 (2000).

    Fast (no pyscf): consumes `diagram_representative`s directly. This pins the
    directed-loop sign as a source-validated diagram invariant. It does NOT
    assert agreement with the PySCF weight-table sign -- that is the AR2.3
    convention delta (rep external-labeling / P-orbit), out of AR2.1's scope."""

    def test_crawford_worked_examples(self):
        from ccgen.diagram import (
            diagram_representative, directed_loops, diagram_hole_lines,
            diagram_sign,
        )

        # (t_ops, h_rank, bra_level, expected_l, expected_h_or_None, expected_sign_or_None)
        cases = [
            (((2, 4, 2),), 2, 0, 2, None, None),          # p.84 f2 energy
            (((1, 2, 1), (1, 2, 1)), 2, 0, 2, None, None),  # p.87 (t1 t1) energy
            (((1, 1, 1),), 2, 2, 2, 2, +1),                 # p.91 Eq.[180] LEFT
            (((1, 1, 0),), 2, 2, 2, 3, -1),                 # p.91 Eq.[180] RIGHT
        ]
        for tops, hr, bl, l_exp, h_exp, s_exp in cases:
            rep = diagram_representative(DiagramString(tops, bl, 0), hr)
            self.assertEqual(directed_loops(rep), l_exp, f"{tops} loops")
            if h_exp is not None:
                self.assertEqual(diagram_hole_lines(rep), h_exp, f"{tops} holes")
            if s_exp is not None:
                self.assertEqual(diagram_sign(rep), s_exp, f"{tops} sign")

    def test_sign_is_a_topology_invariant(self):
        # diagram_sign depends only on (t_ops, h_rank); no seed, no molecule.
        from ccgen.diagram import diagram_representative, diagram_sign

        for tops, hr in [(((2, 2, 1),), 2), (((1, 1, 0), (2, 2, 1)), 2)]:
            s1 = diagram_sign(diagram_representative(DiagramString(tops, 2, 0), hr))
            s2 = diagram_sign(diagram_representative(DiagramString(tops, 2, 0), hr))
            self.assertEqual(s1, s2)
            self.assertIn(s1, (+1, -1))


class EquivalentLinePairsTests(unittest.TestCase):
    """AR2.2a: `equivalent_line_pairs` (Crawford p.85). Each pair -> a factor of
    1/2 in |weight|. Validated against known pair counts. This is the
    equivalent-LINE part only; equivalent-vertex (AR2.2b) and amplitude
    normalization (AR2.2c) are the rest of the magnitude formula."""

    def test_known_pair_counts(self):
        from ccgen.diagram import diagram_representative, equivalent_line_pairs

        # (t_ops, h_rank, bra_level, expected_pairs)
        cases = [
            (((2, 2, 0),), 2, 2, 1),   # hh-ladder: k,l both t2<->v, occ
            (((2, 2, 2),), 2, 2, 1),   # pp-ladder: c,d both t2<->v, vir
            (((2, 2, 1),), 2, 2, 0),   # ring: 1 vir + 1 occ, mixed species
            (((1, 1, 0), (1, 1, 0)), 2, 2, 0),   # distinct t1 starts
            (((2, 4, 2),), 2, 0, 2),   # f2 energy: 2 vir + 2 occ t2<->v
            (((2, 2, 0), (2, 2, 2)), 2, 2, 2),   # quadratic t2 t2 v
        ]
        for tops, hr, bl, exp in cases:
            rep = diagram_representative(DiagramString(tops, bl, 0), hr)
            self.assertEqual(equivalent_line_pairs(rep), exp, str(tops))


class EquivalentVertexFactorTests(unittest.TestCase):
    """AR2.2b: `equivalent_vertex_factor` (Crawford p.87), 1/n! per group of
    identical operators connected the SAME manner. Fixes the naive-1/n! over-fire
    (which counted all identical-rank ops). Validated two ways: worked value on
    the (t1 t1) energy, and that combining it with `equivalent_line_pairs` leaves
    a CLEAN DYADIC residual vs the weight table -- the remaining factor is the
    amplitude normalization (AR2.2c), not a vertex-factor error."""

    def test_t1t1_energy_is_one_half(self):
        from fractions import Fraction
        from ccgen.diagram import diagram_representative, equivalent_vertex_factor

        # (t1 t1) energy: both T1 connect to v the same manner -> 1/2.
        rep = diagram_representative(DiagramString(((1, 2, 1), (1, 2, 1)), 0, 0), 2)
        self.assertEqual(equivalent_vertex_factor(rep), Fraction(1, 2))

    def test_distinct_role_identical_ops_do_not_fire(self):
        from fractions import Fraction
        from ccgen.diagram import diagram_representative, equivalent_vertex_factor

        # Two T1 in DIFFERENT roles (one sends vir-internal, one occ-internal) are
        # NOT equivalent -> factor 1 (the naive 1/2! over-fire this fixes).
        rep = diagram_representative(DiagramString(((1, 1, 0), (1, 1, 1)), 2, 0), 2)
        self.assertEqual(equivalent_vertex_factor(rep), Fraction(1))
        # Three T1 where only two share a manner -> 1/2!, not 1/3!.
        rep3 = diagram_representative(
            DiagramString(((1, 1, 0), (1, 1, 0), (1, 1, 1)), 2, 0), 2
        )
        self.assertEqual(equivalent_vertex_factor(rep3), Fraction(1, 2))

    def test_line_plus_vertex_leaves_clean_dyadic_residual(self):
        # (1/2)^elp * equiv_vertex_factor vs the table magnitude: residual must be
        # a power of two everywhere (the AR2.2c amplitude-normalization factor).
        # This pins that AR2.2b removed the messy 3/6 over-fire residuals.
        import ast
        import json
        from fractions import Fraction
        from pathlib import Path
        from ccgen.diagram import (
            diagram_representative, equivalent_line_pairs, equivalent_vertex_factor,
        )

        table = json.load(
            open(Path(__file__).with_name("ccsd_diagram_weights.json"))
        )
        for key, (num, den) in table.items():
            if key == "bare":
                continue
            tops, hr = ast.literal_eval(key)
            rep = diagram_representative(DiagramString(tops, 2, 0), hr)
            pred = Fraction(1, 2 ** equivalent_line_pairs(rep)) * \
                equivalent_vertex_factor(rep)
            resid = pred / abs(Fraction(num, den))
            # residual is a power of two (num or den a power of 2, the other 1)
            self.assertEqual(resid.numerator & (resid.numerator - 1), 0, key)
            self.assertEqual(resid.denominator & (resid.denominator - 1), 0, key)


class ExternalPairFactorTests(unittest.TestCase):
    """AR2.2c: `external_pair_factor` -- 2^(same-species external-line pairs,
    per amplitude + on the vertex), the bare-antisymmetric-storage normalization.
    Together with AR2.2a+2.2b this CLOSES the CCSD-doubles magnitude formula
    (30/30 vs the weight table)."""

    def test_worked_values(self):
        from ccgen.diagram import DiagramString, external_pair_factor

        # single T1, doubles vertex: the external pair sits on the VERTEX
        # (v carries both a,b or both i,j) -> 2. amplitude-only counting missed
        # exactly these rows.
        self.assertEqual(external_pair_factor(DiagramString(((1, 1, 0),), 2, 0)), 2)
        self.assertEqual(external_pair_factor(DiagramString(((1, 1, 1),), 2, 0)), 2)
        # single T2 with both externals same species: amplitude pair + vertex
        # pair (the other species is fully internal) -> 4.
        self.assertEqual(external_pair_factor(DiagramString(((2, 2, 0),), 2, 0)), 4)
        self.assertEqual(external_pair_factor(DiagramString(((2, 2, 2),), 2, 0)), 4)
        # two T1s, one external each -> the pair is split across amplitudes, so
        # neither amplitude has a pair; the vertex has none either -> 2 comes
        # from... the two like-species externals landing on the vertex+amp mix.
        self.assertEqual(
            external_pair_factor(DiagramString(((1, 1, 0), (1, 1, 0)), 2, 0)), 2
        )
        # ring (2,2,1): one particle-ext + one hole-ext on the amplitude, one of
        # each on the vertex -> no same-species pair anywhere -> 1.
        self.assertEqual(external_pair_factor(DiagramString(((2, 2, 1),), 2, 0)), 1)

    def test_magnitude_reproduces_the_full_weight_table(self):
        import ast
        import json
        from fractions import Fraction
        from pathlib import Path
        from ccgen.diagram import DiagramString, diagram_magnitude

        table = json.load(
            open(Path(__file__).with_name("ccsd_diagram_weights.json"))
        )
        for key, (num, den) in table.items():
            if key == "bare":
                continue
            tops, hr = ast.literal_eval(key)
            mag = diagram_magnitude(DiagramString(tops, 2, 0), hr)
            self.assertEqual(mag, abs(Fraction(num, den)), key)

    def test_amplitude_only_count_would_miss(self):
        # Guards the load-bearing insight: dropping the VERTEX pairs regresses to
        # the amplitude-only count, which misses the 6 single-vertex-species rows.
        # (Mutation test -- fails if someone removes the vertex term.)
        from ccgen.diagram import DiagramString, external_pair_factor

        # single T1: without the vertex pair this would be 1, not 2.
        self.assertNotEqual(
            external_pair_factor(DiagramString(((1, 1, 0),), 2, 0)), 1
        )

    def test_amplitude_norm_equals_pair_count_on_doubles(self):
        # M1.0/M1.1 invariance: the new prod(1/n_ext!) amplitude factor is
        # IDENTICAL to the old (1/2)^(amp_pairs) on every doubles diagram (for
        # T2, k_ext in {0,1,2} and (1/2)^(k//2) == 1/k!). This is why swapping it
        # in keeps diagram_magnitude at 30/30 (asserted separately). It DIVERGES
        # at T3, which is the point.
        from fractions import Fraction
        from ccgen.diagram import (
            DiagramString, _amplitude_norm_factor, pyscf_signed_weights,
        )

        for did in pyscf_signed_weights():
            if did == "bare":
                continue
            tops, hr = did
            ds = DiagramString(tops, 2, 0)
            amp_pairs = sum(
                (m1 - m3) // 2 + (m1 - (m2 - m3)) // 2 for m1, m2, m3 in tops
            )
            self.assertEqual(
                _amplitude_norm_factor(ds), Fraction(1, 2 ** amp_pairs), str(did)
            )

    def test_t3_amplitude_factor_is_non_dyadic(self):
        # M1.1: the T3 amplitude normalization (1/3! = 1/6) appears -- triples
        # magnitudes go non-dyadic where a T3 sits, which the old pair-count
        # (saturating at (1/2)^(3//2)=1/2) could never produce.
        from ccgen.diagram import DiagramString, diagram_magnitude

        # single T3, doubles ERI vertex: carries a 1/3! -> denominator has a 3.
        m = diagram_magnitude(DiagramString(((3, 2, 0),), 3, 0), 2)
        self.assertEqual(m.denominator % 3, 0, "T3 amplitude factor missing")


class PyscfSignedWeightOracleTests(unittest.TestCase):
    """AR2.3(i).0: `pyscf_signed_weights` is the PySCF-derived ground-truth
    SIGNED weight per CCSD-doubles diagram -- the oracle AR2.3(i) reconciles the
    diagram sign against. These checks are OFFLINE (read the committed table, no
    PySCF); the table's PySCF freshness is pinned in test_reference_vs_pyscf.py."""

    def test_magnitude_matches_the_ar22c_formula(self):
        # |signed weight| == diagram_magnitude for all 30, so AR2.3(i) has ONLY
        # the sign left to reconcile -- the magnitude (AR2.2) is already closed.
        from ccgen.diagram import (
            DiagramString, diagram_magnitude, pyscf_signed_weights,
        )

        w = pyscf_signed_weights()
        n = 0
        for did, wi in w.items():
            if did == "bare":
                continue
            tops, hr = did
            self.assertEqual(
                abs(wi), diagram_magnitude(DiagramString(tops, 2, 0), hr), str(did)
            )
            n += 1
        self.assertEqual(n, 30)

    def test_all_signs_are_plus_or_minus_one(self):
        from ccgen.diagram import pyscf_signed_weights

        for did, wi in pyscf_signed_weights().items():
            self.assertIn(
                1 if wi > 0 else -1, (1, -1), str(did)
            )  # sign is well-defined (nonzero)
            self.assertNotEqual(wi, 0, str(did))

    def test_diagram_sign_baseline_agreement_is_19_of_30(self):
        # Baseline BEFORE the AR2.3(i) canonical relabel: Crawford's raw
        # `diagram_sign` matches the PySCF signed-weight sign on 19/30. Pins the
        # starting point so AR2.3(i).1 can show it reaching 30/30. (Also a
        # tripwire: if this number moves, the sign convention shifted upstream.)
        from ccgen.diagram import (
            DiagramString, diagram_representative, diagram_sign,
            pyscf_signed_weights,
        )

        w = pyscf_signed_weights()
        agree = 0
        for did, wi in w.items():
            if did == "bare":
                continue
            tops, hr = did
            rep = diagram_representative(DiagramString(tops, 2, 0), hr)
            if diagram_sign(rep) == (1 if wi > 0 else -1):
                agree += 1
        self.assertEqual(agree, 19)

    def test_signed_weight_reproduces_the_pyscf_table(self):
        # AR2.3(i).1a: diagram_signed_weight == PySCF signed weight, all 30. This
        # is the full AR2 deliverable (sign x magnitude) for CCSD doubles, the
        # 19/30 baseline above lifted to 30/30 by the stored sign correction.
        from ccgen.diagram import (
            DiagramString, diagram_signed_weight, pyscf_signed_weights,
        )

        w = pyscf_signed_weights()
        n = 0
        for did, wi in w.items():
            if did == "bare":
                continue
            tops, hr = did
            self.assertEqual(
                diagram_signed_weight(DiagramString(tops, 2, 0), hr), wi, str(did)
            )
            n += 1
        self.assertEqual(n, 30)


class CrossingParityTests(unittest.TestCase):
    """AR2.3(i).1b.0/.1b.1: the open-loop external pairing and its crossing
    parity -- the structural lead for a solve-free sign correction. All checks
    are pure topology or vs the .1a stored correction (PySCF oracle); no
    generator. The residual mismatches are AR2.3(i).1b.2 (open)."""

    def test_directed_loops_unchanged_by_the_refactor(self):
        # .1b.0 factored the tracer into _trace_directed_loops; the l count that
        # AR2.1's diagram_sign depends on must be byte-identical for all 30.
        from ccgen.diagram import (
            DiagramString, diagram_representative, directed_loops,
            pyscf_signed_weights,
        )

        # hand-traced l values from the AR2.0 module note / earlier probe
        expected = {
            (((2, 2, 2),), 2): 2,
            (((2, 2, 1),), 2): 3,
            (((1, 1, 0),), 2): 2,
        }
        for did, l in expected.items():
            tops, hr = did
            rep = diagram_representative(DiagramString(tops, 2, 0), hr)
            self.assertEqual(directed_loops(rep), l, str(did))
        # and every table diagram still traces without error
        for did in pyscf_signed_weights():
            if did == "bare":
                continue
            tops, hr = did
            directed_loops(diagram_representative(DiagramString(tops, 2, 0), hr))

    def test_open_pairing_is_one_of_the_two_doubles_pairings(self):
        # .1b.0: every doubles diagram pairs its externals (i,j)x(a,b) either
        # identity or crossed -- never a malformed pairing.
        from ccgen.diagram import (
            DiagramString, diagram_representative, open_loop_external_pairing,
            pyscf_signed_weights,
        )

        for did in pyscf_signed_weights():
            if did == "bare":
                continue
            tops, hr = did
            rep = diagram_representative(DiagramString(tops, 2, 0), hr)
            d = dict(open_loop_external_pairing(rep))
            self.assertEqual(set(d.keys()), {"i", "j"}, str(did))
            self.assertEqual(set(d.values()), {"a", "b"}, str(did))

    def test_crossing_parity_reproduces_correction_23_of_30(self):
        # .1b.1: crossing parity alone matches the .1a stored sign correction on
        # 23/30 -- the best structural predictor found (all count-formulas
        # <=21/30). Pins the baseline for .1b.2 (lifting 23 -> 30 with the
        # residual factor), the same role 19/30 played for .1a.
        from ccgen.diagram import (
            DiagramString, diagram_representative, diagram_sign, crossing_parity,
            pyscf_signed_weights,
        )

        w = pyscf_signed_weights()
        agree = 0
        for did, wi in w.items():
            if did == "bare":
                continue
            tops, hr = did
            rep = diagram_representative(DiagramString(tops, 2, 0), hr)
            correction = (1 if wi > 0 else -1) * diagram_sign(rep)
            if crossing_parity(rep) == correction:
                agree += 1
        self.assertEqual(agree, 23)

    def test_structural_sign_solves_the_whole_eri_manifold(self):
        # .1b.2a: crossing_parity * (-1)^l reproduces the PySCF-solved sign for
        # EVERY ERI-vertex (h_rank==2) diagram -- the entire ERI manifold sign,
        # solve-free. Fock-vertex (h_rank==1) is .1b.2b and stays out (guarded
        # below).
        from ccgen.diagram import (
            DiagramString, diagram_representative, structural_sign,
            pyscf_signed_weights,
        )

        w = pyscf_signed_weights()
        eri = eri_ok = 0
        for did, wi in w.items():
            if did == "bare":
                continue
            tops, hr = did
            if hr != 2:
                continue
            eri += 1
            rep = diagram_representative(DiagramString(tops, 2, 0), hr)
            if structural_sign(rep, hr) == (1 if wi > 0 else -1):
                eri_ok += 1
        self.assertEqual((eri_ok, eri), (26, 26))

    def test_structural_sign_reproduces_every_pyscf_sign(self):
        # .1b.2b complete: crossing_parity * (-1)^l * (-1 if Fock line is a hole)
        # reproduces the PySCF-solved sign for ALL 30 CCSD-doubles diagrams --
        # a fully solve-free sign, no table, no generator.
        from ccgen.diagram import (
            DiagramString, diagram_representative, structural_sign,
            pyscf_signed_weights,
        )

        w = pyscf_signed_weights()
        n = 0
        for did, wi in w.items():
            if did == "bare":
                continue
            tops, hr = did
            rep = diagram_representative(DiagramString(tops, 2, 0), hr)
            self.assertEqual(
                structural_sign(rep, hr), 1 if wi > 0 else -1, str(did)
            )
            n += 1
        self.assertEqual(n, 30)

    def test_fock_correction_is_species_dependent_not_a_flat_flip(self):
        # Pins .1b.2b's key finding: the Fock factor depends on the contracted
        # line's SPECIES. (2,1,0) [fock-occ] and (2,1,1) [fock-vir] are both
        # single-T2 Fock, crossed pairing, yet opposite sign -- a flat "-1 if
        # fock" would get one wrong.
        from ccgen.diagram import (
            DiagramString, diagram_representative, structural_sign,
            pyscf_signed_weights,
        )

        w = pyscf_signed_weights()
        for tops in (((2, 1, 0),), ((2, 1, 1),)):
            rep = diagram_representative(DiagramString(tops, 2, 0), 1)
            self.assertEqual(
                structural_sign(rep, 1), 1 if w[(tops, 1)] > 0 else -1, str(tops)
            )

    def test_structural_sign_agrees_with_the_stored_correction(self):
        # .1b.3 regression: the solve-free structural_sign must equal the .1a
        # stored (sign_correction * diagram_sign) everywhere. Catches a
        # convention drift in either the structural rule or the table.
        from ccgen.diagram import (
            DiagramString, diagram_representative, diagram_sign, sign_correction,
            structural_sign, pyscf_signed_weights,
        )

        for did in pyscf_signed_weights():
            if did == "bare":
                continue
            tops, hr = did
            ds = DiagramString(tops, 2, 0)
            rep = diagram_representative(ds, hr)
            self.assertEqual(
                structural_sign(rep, hr),
                diagram_sign(rep) * sign_correction(ds, hr),
                str(did),
            )

    def test_permutation_parity_unit(self):
        # B1.0: _permutation_parity counts inversions -> sign.
        from ccgen.diagram import _permutation_parity

        self.assertEqual(_permutation_parity([]), 1)
        self.assertEqual(_permutation_parity([0]), 1)
        self.assertEqual(_permutation_parity(["a", "b"]), 1)   # sorted
        self.assertEqual(_permutation_parity(["b", "a"]), -1)  # one swap
        self.assertEqual(_permutation_parity(["a", "b", "c"]), 1)
        self.assertEqual(_permutation_parity(["a", "c", "b"]), -1)
        self.assertEqual(_permutation_parity(["c", "a", "b"]), 1)   # two swaps
        self.assertEqual(_permutation_parity(["c", "b", "a"]), -1)  # reverse of 3

    def test_external_pairing_parity_reduces_to_crossing_parity_on_doubles(self):
        # B1.1: the rank-general external_pairing_parity is IDENTICAL to the
        # doubles-hardcoded crossing_parity on all 30 CCSD-doubles diagrams --
        # a pure refactor-safety check (no oracle needed). This is what lets
        # structural_sign switch to the general form without moving the 30/30.
        from ccgen.diagram import (
            DiagramString, diagram_representative, crossing_parity,
            external_pairing_parity, pyscf_signed_weights,
        )

        for did in pyscf_signed_weights():
            if did == "bare":
                continue
            tops, hr = did
            rep = diagram_representative(DiagramString(tops, 2, 0), hr)
            self.assertEqual(
                external_pairing_parity(rep), crossing_parity(rep), str(did)
            )


class DiagramOrbitTermsTests(unittest.TestCase):
    """D4.0: `diagram_orbit_terms` emits the signed AlgebraTerm orbit whose
    summed evaluation reproduces the array-level `weight * orbit(rep)` (the
    M1.3-validated target). This lifts the validated array weight to symbolic
    terms for D4 generation. Array-level (random tensors), no PySCF."""

    def test_emitted_orbit_matches_array_orbit(self):
        import numpy as np
        from ccgen.diagram import (
            enumerate_diagrams, diagram_representative, diagram_signed_weight,
            diagram_orbit_terms,
        )
        from ccgen.tests.residual_eval import (
            residual_einsum, _antisymmetrize_block, random_tensors,
        )

        no, nv = 4, 5
        tn = random_tensors(no, nv, seed=2)

        def orbit(base, k):
            r = _antisymmetrize_block(base, tuple(range(k)))
            return _antisymmetrize_block(r, tuple(range(k, 2 * k)))

        checked = 0
        for bra in (1, 2, 3):
            for ds, hr in enumerate_diagrams([1, 2, 3], bra):
                rep = diagram_representative(ds, hr)
                w = float(diagram_signed_weight(ds, hr))
                target = w * orbit(residual_einsum(rep, no, nv, tensors=tn), bra)
                emitted = sum(
                    residual_einsum(t, no, nv, tensors=tn)
                    for t in diagram_orbit_terms(ds, hr)
                )
                self.assertTrue(
                    np.allclose(target, emitted, atol=1e-12),
                    f"{ds.t_ops} h={hr}: {np.max(np.abs(target - emitted)):.2e}",
                )
                checked += 1
        self.assertGreater(checked, 90)

    def test_orbit_size_and_coeffs(self):
        # A doubles diagram emits (2!)*(2!) = 4 terms; a triples one (3!)*(3!)=36.
        # Coefficients are +/- the signed weight (no 1/k! normalization).
        from ccgen.diagram import DiagramString, diagram_signed_weight, diagram_orbit_terms

        w2 = diagram_signed_weight(DiagramString(((2, 2, 2),), 2, 0), 2)
        terms2 = diagram_orbit_terms(DiagramString(((2, 2, 2),), 2, 0), 2)
        self.assertEqual(len(terms2), 4)
        self.assertTrue(all(abs(t.coeff) == abs(w2) for t in terms2))

        terms3 = diagram_orbit_terms(DiagramString(((3, 2, 1),), 3, 0), 2)
        self.assertEqual(len(terms3), 36)

    def test_manifold_terms_match_full_ccgen_residual(self):
        # D4.1: diagram_manifold_terms (all orbits + the bare Hamiltonian term)
        # summed+evaluated == the FULL ccgen manifold residual, across
        # singles/doubles/triples. This is the whole-manifold form the diagram
        # front-end emits, built with no BCH/Wick. Array-level (random tensors).
        import numpy as np
        from ccgen.diagram import diagram_manifold_terms
        from ccgen.generate import generate_cc_equations
        from ccgen.tests.residual_eval import residual_einsum, random_tensors

        no, nv = 4, 5
        tn = random_tensors(no, nv, seed=5)
        for manifold, bra in [("singles", 1), ("doubles", 2), ("triples", 3)]:
            rgen = sum(
                residual_einsum(t, no, nv, tensors=tn)
                for t in generate_cc_equations("ccsdt")[manifold]
            )
            rdia = sum(
                residual_einsum(t, no, nv, tensors=tn)
                for t in diagram_manifold_terms([1, 2, 3], bra)
            )
            self.assertTrue(
                np.allclose(rdia, rgen, atol=1e-11),
                f"{manifold}: {np.max(np.abs(rdia - rgen)):.2e}",
            )

    def test_bare_terms(self):
        # D4.1: singles -> f(a,i), doubles -> v(i,j,a,b), triples -> none.
        from ccgen.diagram import _bare_manifold_term

        self.assertEqual([f.name for f in _bare_manifold_term(1).factors], ["f"])
        self.assertEqual([f.name for f in _bare_manifold_term(2).factors], ["v"])
        self.assertIsNone(_bare_manifold_term(3))


@unittest.skipUnless(
    os.environ.get("CCGEN_SLOW_TESTS"),
    "AR1 ccsdtq gate is slow (~30s); set CCGEN_SLOW_TESTS=1 to run",
)
class CcsdtqAssemblyTests(unittest.TestCase):
    """AR1: the diagram assembler produces a distinct, correctly-antisymmetric
    contraction for every CCSDTQ (quadruples) diagram.

    No PySCF GCCSDTQ oracle exists, so this validates two PySCF-free necessary
    conditions that together prove the assembly is structurally sound:
      1. every quadruples diagram assembles (no crash / NotImplementedError);
      2. each diagram's P(ijkl)P(abcd) orbit residual is antisymmetric;
      3. the 74 orbit residuals are LINEARLY INDEPENDENT (full rank) -- no two
         diagrams collapse to the same contraction, i.e. enumeration produced no
         duplicates AND the assembler distinguishes every diagram.

    Requires no, nv >= 6: a P(ijkl)P(abcd) antisymmetric tensor has only
    C(n,4)^2 independent components, which is 1 at n=4 (rank collapses
    spuriously) and 225 at n=6 (enough to separate 74 diagrams). This is the
    rank-4 form of the small-dimensions rule that bit the earlier D3.2 probes.

    Topology CORRECTNESS (not just well-formedness) at rank 4 is deferred to AR4
    (diagram path == term path), since the only rank-4 oracle is the slow
    term-path ccsdtq residual.
    """

    def test_all_ccsdtq_diagrams_assemble_antisymmetric_and_independent(self):
        import numpy as np
        from ccgen.diagram import enumerate_diagrams, diagram_representative
        from ccgen.tests.residual_eval import (
            random_tensors, residual_einsum, _antisymmetrize_block,
        )

        no = nv = 6
        tensors = random_tensors(no, nv, seed=0)

        def orbit(r):  # P(abcd) P(ijkl) on [a,b,c,d,i,j,k,l]
            r = _antisymmetrize_block(r, (0, 1, 2, 3))
            return _antisymmetrize_block(r, (4, 5, 6, 7))

        diagrams = enumerate_diagrams([1, 2, 3, 4], 4)
        self.assertEqual(len(diagrams), 74)

        vecs = []
        for ds, hr in diagrams:
            rep = diagram_representative(ds, hr)  # (1) assembles or raises
            r = orbit(residual_einsum(rep, no, nv, tensors=tensors))
            # (2) antisymmetric under a<->b and i<->j (a representative pair of
            # the full P(abcd)P(ijkl), which the orbit already enforced)
            self.assertLess(
                np.max(np.abs(r + np.swapaxes(r, 0, 1))), 1e-9,
                f"{ds.t_ops}: not antisymmetric a<->b",
            )
            self.assertLess(
                np.max(np.abs(r + np.swapaxes(r, 4, 5))), 1e-9,
                f"{ds.t_ops}: not antisymmetric i<->j",
            )
            vecs.append(r.ravel())

        # (3) all 74 diagrams linearly independent
        rank = np.linalg.matrix_rank(np.array(vecs), tol=1e-9)
        self.assertEqual(rank, len(diagrams), "diagrams not linearly independent")


if __name__ == "__main__":
    unittest.main()
