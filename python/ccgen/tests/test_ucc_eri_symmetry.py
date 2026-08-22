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
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.emit.planck_tensor_cpp import (  # noqa: E402
    _CANONICAL_ERI_BLOCKS,
    _ERI_SYMMETRY_PERMUTATIONS,
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


class CurrentEmitterIsSpinBlindTests(unittest.TestCase):
    """Pins the defect this predicate exists to fix. RED until U3.2 consumes it."""

    def test_thirty_seven_abab_reads_use_an_invalid_permutation(self):
        import ccgen.emit.planck_tensor_cpp as emitter

        eqs = ucc_adapt_equations(generate_cc_equations("ccsd"))
        invalid = total = 0
        for terms in eqs.values():
            for term in terms:
                for factor in term.factors:
                    if factor.name != "v_abab":
                        continue
                    obj = emitter._source_tensor(factor)
                    spaces = tuple(emitter._space_char(i) for i in obj.indices)
                    for block_spaces in _CANONICAL_ERI_BLOCKS.values():
                        chosen = next(
                            (perm for perm, _ in _ERI_SYMMETRY_PERMUTATIONS
                             if tuple(block_spaces[i] for i in perm) == spaces),
                            None)
                        if chosen is None:
                            continue
                        total += 1
                        if not eri_permutation_preserves_block("abab", chosen):
                            invalid += 1
                        break

        self.assertEqual(total, 142, "mixed-block read count changed")
        self.assertEqual(
            invalid, 0,
            f"{invalid} of {total} v_abab reads are routed through a permutation "
            f"that is not a symmetry of `abab` (it maps to `baba`), so they read "
            f"the right array with permuted indices. Expected 37 before U3.2 "
            f"lands; this assertion is the U3.2 acceptance criterion.")


if __name__ == "__main__":
    unittest.main()
