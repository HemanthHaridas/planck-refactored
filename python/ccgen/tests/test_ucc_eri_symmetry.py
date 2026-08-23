"""U3.0 -- which ERI symmetries survive spin resolution, and what that costs.

`_ERI_SYMMETRY_PERMUTATIONS` lists four +1 symmetries of the spatial physicist
integral `<pq|rs>`. Applying them is sound while there is ONE such integral, which
is the RCC case. Under UCC there are three (`v_aaaa`, `v_abab`, `v_bbbb`), and a
permutation that reorders indices also reorders their spins -- so it is usable on a
block only when it maps that block's spin string to itself.

WHY THIS IS A GATE AND NOT AN OBVIOUS ONE-LINER. The emitter applies all four
spin-blindly today, and measured on the CCSD UCC manifold, 37 of 142 `v_abab` reads
go through a permutation that holds only for `baba`. That is a second, independent
defect from the array-name collapse `test_ucc_emit_distinct_blocks` pins -- and the
more dangerous of the two to "fix", because routing the names correctly WITHOUT
this predicate sends those 37 reads to the right array with permuted indices. Wrong,
and quieter than what is there now.

The same-spin blocks are why it never surfaced: every permutation of `aaaa` is
`aaaa`, so on the RCC path all four are valid and the spin-blindness is invisible.

THE COVERAGE CONSEQUENCE, which is the load-bearing result here. The table's own
comment claims the four permutations "cover all 16 four-index o/v patterns
(verified), so no coverage is lost". That claim is spin-free and does not survive
spin resolution: with only the two valid permutations, `abab` reaches 11 of 16
patterns from the 7 canonical blocks, and four of the five it cannot reach
(`oovo`, `vooo`, `vovo`, `vovv`) are patterns the manifold actually requires. So
the mixed block needs 10 stored arrays rather than 7, three of which (`oovo`,
`vovo`, `vovv`) have no counterpart in the RCC cache at all. U3.1 must size the
spin-blocked cache from this, not from "three copies of the RCC blocks".
"""

from __future__ import annotations

import itertools
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.emit.planck_tensor_cpp import (  # noqa: E402
    _CANONICAL_ERI_BLOCKS,
    _ERI_SYMMETRY_PERMUTATIONS,
    _canonical_eri_blocks_for,
    emit_planck_translation_unit,
    eri_permutation_preserves_block,
    eri_permutations_for_block,
)
from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.spin import ucc_adapt_equations  # noqa: E402

PARTICLE = (1, 0, 3, 2)     # <qp|sr>
BRA_KET = (2, 3, 0, 1)      # <rs|pq>
PRODUCT = (3, 2, 1, 0)      # <sr|qp>
IDENTITY = (0, 1, 2, 3)


class PredicateTests(unittest.TestCase):
    def test_same_spin_blocks_keep_every_permutation(self):
        """Every permutation of an all-one-spin tag is that tag -- so RCC is safe."""
        for tag in ("aaaa", "bbbb", "aaaaaa", "bbbbbb"):
            with self.subTest(tag=tag):
                perms = [p for p, _ in _ERI_SYMMETRY_PERMUTATIONS
                         if len(p) == len(tag)]
                for perm in perms:
                    self.assertTrue(eri_permutation_preserves_block(tag, perm))

    def test_mixed_block_keeps_only_identity_and_bra_ket(self):
        self.assertEqual(
            [p for p, _ in eri_permutations_for_block("abab")],
            [IDENTITY, BRA_KET])

    def test_the_two_rejected_permutations_map_abab_to_baba(self):
        """Not just 'invalid' -- they name a specific other block, which is why
        `baba` need not be stored: it is reachable, knowingly, from `abab`."""
        for perm in (PARTICLE, PRODUCT):
            with self.subTest(perm=perm):
                self.assertFalse(eri_permutation_preserves_block("abab", perm))
                self.assertEqual("".join("abab"[i] for i in perm), "baba")

    def test_rcc_path_keeps_all_four(self):
        """A bare `v` (no spin resolution) must be unaffected -- one spatial
        tensor, nothing to leave. This is what keeps the RCC emit byte-identical."""
        self.assertEqual(eri_permutations_for_block(None), _ERI_SYMMETRY_PERMUTATIONS)

    def test_length_mismatch_is_an_error_not_a_silent_false(self):
        with self.assertRaises(ValueError):
            eri_permutation_preserves_block("abab", (0, 1))


class NumericGroundingTests(unittest.TestCase):
    """The predicate is tag algebra; this checks the algebra against real integrals.

    Tag reasoning is exactly the kind of argument that can be self-consistently
    wrong, so the three claims it rests on are verified on random real orbitals.
    """

    @staticmethod
    def _physicist(tag, g, C):
        import numpy as np
        p, q, r, s = (C[c] for c in tag)
        # chemists (pr|qs) -> physicist <pq|rs>
        chem = np.einsum("pqrs,pi,qj,rk,sl->ijkl", g, p, r, q, s, optimize=True)
        return chem.transpose(0, 2, 1, 3)

    def setUp(self):
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            self.skipTest("numpy unavailable")
        rng = np.random.default_rng(0)
        n = 6
        a = rng.normal(size=(n, n, n, n))
        # impose the full 8-fold real-AO chemists symmetry
        g = a + a.transpose(1, 0, 2, 3)
        g = g + g.transpose(0, 1, 3, 2)
        self.g = g + g.transpose(2, 3, 0, 1)
        self.C = {"a": rng.normal(size=(n, n)), "b": rng.normal(size=(n, n))}

    def test_bra_ket_really_is_a_symmetry_of_abab(self):
        import numpy as np
        abab = self._physicist("abab", self.g, self.C)
        self.assertTrue(np.allclose(abab, abab.transpose(*BRA_KET), atol=1e-10))

    def test_particle_swap_really_is_not(self):
        """The claim the predicate exists to enforce. If this ever passes, the
        whole U3.0 restriction is unnecessary -- so it is the falsifier."""
        import numpy as np
        abab = self._physicist("abab", self.g, self.C)
        self.assertFalse(np.allclose(abab, abab.transpose(*PARTICLE), atol=1e-10))

    def test_baba_is_abab_under_the_particle_swap(self):
        """Justifies not storing `baba` -- and justifies emitting the swap
        explicitly rather than pretending the permutation was valid."""
        import numpy as np
        abab = self._physicist("abab", self.g, self.C)
        baba = self._physicist("baba", self.g, self.C)
        self.assertTrue(np.allclose(baba, abab.transpose(*PARTICLE), atol=1e-10))


class CoverageConsequenceTests(unittest.TestCase):
    """What restricting the group costs, which is what U3.1 must be sized from."""

    @staticmethod
    def _reachable(tag):
        perms = [p for p, _ in eri_permutations_for_block(tag)]
        return {"".join(spaces[i] for i in perm)
                for spaces in _CANONICAL_ERI_BLOCKS.values()
                for perm in perms}

    def test_same_spin_still_covers_all_sixteen(self):
        """The table's existing 'covers all 16 patterns' claim, still true per
        same-spin block -- so nothing about the RCC path changes."""
        every = {"".join(p) for p in itertools.product("ov", repeat=4)}
        for tag in ("aaaa", "bbbb"):
            with self.subTest(tag=tag):
                self.assertEqual(self._reachable(tag), every)

    def test_mixed_block_loses_five_patterns(self):
        """And the claim does NOT survive spin resolution."""
        every = {"".join(p) for p in itertools.product("ov", repeat=4)}
        missing = every - self._reachable("abab")
        self.assertEqual(
            missing, {"oovo", "vooo", "vovo", "vovv", "vvvo"},
            "the set of patterns unreachable for the mixed block changed; U3.1's "
            "stored-block list is derived from exactly this set")

    def test_the_manifold_actually_needs_four_of_the_missing_patterns(self):
        """The measurement that turns a curiosity into required work: these are
        not hypothetical patterns, the CCSD UCC residuals read them."""
        eqs = ucc_adapt_equations(generate_cc_equations("ccsd"))
        needed = {
            "".join("o" if i.space == "occ" else "v" for i in f.indices)
            for terms in eqs.values() for t in terms for f in t.factors
            if f.name == "v_abab"
        }
        self.assertEqual(len(needed), 13, "mixed-block space patterns changed")

        unreachable = needed - self._reachable("abab")
        self.assertEqual(
            unreachable, {"oovo", "vooo", "vovo", "vovv"},
            "U3.1 must store canonical blocks covering exactly these; three of "
            "them (oovo, vovo, vovv) have no RCC counterpart at all")

    def test_mixed_block_needs_ten_stored_arrays_not_seven(self):
        """The number U3.1 is sized from. Orbits of the needed patterns under the
        RESTRICTED group -- one stored array per orbit."""
        eqs = ucc_adapt_equations(generate_cc_equations("ccsd"))
        needed = {
            "".join("o" if i.space == "occ" else "v" for i in f.indices)
            for terms in eqs.values() for t in terms for f in t.factors
            if f.name == "v_abab"
        }
        perms = [p for p, _ in eri_permutations_for_block("abab")]
        orbits, seen = [], set()
        for pattern in sorted(needed):
            if pattern in seen:
                continue
            orbit = frozenset("".join(pattern[i] for i in perm) for perm in perms)
            orbits.append(orbit)
            seen |= orbit
        self.assertEqual(
            len(orbits), 10,
            f"expected 10 stored mixed-spin arrays, got {len(orbits)}: "
            f"{sorted(sorted(o) for o in orbits)}")
        self.assertGreater(len(orbits), len(_CANONICAL_ERI_BLOCKS),
                           "the mixed cache must be LARGER than the RCC one")


class EmitterConsumesThePredicateTests(unittest.TestCase):
    """The U3.2 acceptance criterion: no emitted read uses an invalid permutation.

    THIS ASSERTS ON THE EMITTED TEXT, not on a re-derivation. The first version of
    this gate re-implemented the routing inline against `_CANONICAL_ERI_BLOCKS` and
    `_ERI_SYMMETRY_PERMUTATIONS`, which measured a SIMULATION of the old emitter --
    so it stayed red at exactly 37 even after `_map_eri_tensor` was fixed, because
    it never called the code it was supposedly gating. A gate that cannot observe
    the fix cannot certify it.

    Reading the emitted TU instead makes the assertion structural: every ERI read a
    UCC kernel performs must name a per-block view, and every such view must be one
    the block's own symmetry group can legitimately reach.
    """

    @staticmethod
    def _ucc_tu(method: str = "ccsd") -> str:
        eqs = ucc_adapt_equations(generate_cc_equations(method))
        return emit_planck_translation_unit(
            method, eqs, force_arbitrary=True, spin_adapted=True)

    def test_every_eri_read_names_a_spin_blocked_view(self):
        """No bare `mo_blocks.<space>(` read survives in a UCC kernel -- that form
        is the array-name collapse, and it is what the whole step removes."""
        tu = self._ucc_tu()
        leftover = re.findall(r"mo_blocks\.[a-z]{4}\(", tu)
        self.assertEqual(
            leftover, [],
            f"{len(leftover)} untagged mo_blocks reads remain in the UCC TU; each "
            f"collapses three different UHF integrals onto one array")

    def test_bound_views_are_reachable_within_each_block(self):
        """Every bound view is a canonical block of ITS OWN tag's group.

        This is the half that catches an invalid permutation: a read routed
        through `particle`/`product` on a mixed block would have to name a block
        outside `_canonical_eri_blocks_for("abab")` to be expressible at all.
        """
        tu = self._ucc_tu()
        bound = set(re.findall(r'spin_block\("(\w+)", "(\w+)"\)', tu))
        self.assertTrue(bound, "the UCC TU binds no spin-blocked ERI view at all")
        for space, tag in bound:
            with self.subTest(space=space, tag=tag):
                self.assertIn(
                    space, _canonical_eri_blocks_for(tag),
                    f"{tag} binds '{space}', which is not a canonical block of its "
                    f"own symmetry group -- it was reached by a permutation that "
                    f"is not a symmetry of {tag}")

    def test_every_read_view_was_actually_bound(self):
        """A read of an unbound view would not compile; catch it here rather than
        in a C++ build, since `_eri_read` and `_eri_view_bindings` are two halves
        of one naming convention and can drift apart."""
        tu = self._ucc_tu()
        bound = {f"v_{tag}_{space}" for space, tag
                 in re.findall(r'spin_block\("(\w+)", "(\w+)"\)', tu)}
        read = set(re.findall(r"\b(v_[ab]+_[a-z]{4})\(", tu))
        self.assertTrue(read, "the UCC TU reads no spin-blocked ERI at all")
        self.assertEqual(
            read - bound, set(),
            "these ERI views are read but never bound (the emit would not compile)")

    def test_every_read_is_the_one_the_valid_group_selects(self):
        """The assertion that actually catches an invalid permutation.

        MUTATION-DRIVEN. The other tests here check which arrays are BOUND, and a
        mutation that restored the spin-blind permutation list survived all of
        them: it still binds only legitimate blocks, because the per-tag block set
        is wide enough to express its answer. What it changes is which array each
        READ names and in what INDEX ORDER -- measured, 37 of 142 mixed reads move,
        e.g.

            valid-only : v_abab_vovv(a, j, b, c)
            spin-blind : v_abab_ovvv(j, a, c, b)

        Right-looking, wrong integral. Nothing that inspects the bound set can see
        it, so this compares every emitted read against an independent replay of
        the routing under the block's OWN symmetry group.
        """
        import ccgen.emit.planck_tensor_cpp as emitter

        eqs = ucc_adapt_equations(generate_cc_equations("ccsd"))
        expected: list[str] = []
        for terms in eqs.values():
            for term in terms:
                for factor in term.factors:
                    obj = emitter._source_tensor(factor)
                    m = re.fullmatch(r"v_([ab]+)", obj.name)
                    if not m:
                        continue
                    tag = m.group(1)
                    spaces = tuple(emitter._space_char(i) for i in obj.indices)
                    names = [i.name for i in obj.indices]
                    # Replay the search using ONLY this block's symmetries.
                    chosen = None
                    for block, block_spaces in _canonical_eri_blocks_for(tag).items():
                        for perm, _sign in eri_permutations_for_block(tag):
                            if tuple(block_spaces[i] for i in perm) == spaces:
                                inverse = emitter._inverse_permutation(perm)
                                chosen = (f"v_{tag}_{block}("
                                          + ", ".join(names[i] for i in inverse) + ")")
                                break
                        if chosen:
                            break
                    self.assertIsNotNone(
                        chosen, f"no valid routing for {''.join(spaces)} in {tag}")
                    expected.append(chosen)

        self.assertTrue(expected, "no v_<tag> factors found; fixture drift")

        tu = self._ucc_tu()
        missing = sorted({e for e in expected if e not in tu})
        self.assertEqual(
            missing, [],
            f"{len(missing)} ERI read(s) are not emitted in the form the block's "
            f"own symmetry group selects, e.g. {missing[:3]}. A permutation that "
            f"is not a symmetry of the block was used, so the read names the "
            f"wrong array and/or a permuted index order.")

    def test_the_mixed_block_needs_more_arrays_than_the_same_spin_ones(self):
        """The measured consequence, asserted end-to-end rather than in the
        abstract: 10 arrays for `abab`, 6 for each same-spin block."""
        tu = self._ucc_tu()
        bound = re.findall(r'spin_block\("(\w+)", "(\w+)"\)', tu)
        per_tag: dict[str, set[str]] = {}
        for space, tag in bound:
            per_tag.setdefault(tag, set()).add(space)
        self.assertEqual(len(per_tag["abab"]), 10)
        self.assertEqual(len(per_tag["aaaa"]), 6)
        self.assertEqual(len(per_tag["bbbb"]), 6)


if __name__ == "__main__":
    unittest.main()
