"""U0: the UCC block vocabulary.

`ucc_independent_blocks(rank)` names the spin blocks of a rank-2n amplitude that must be
STORED in UCC. Unlike RCC there is no global a<->b flip available (α and β are different
orbitals), so all n+1 α-count sectors are independent and the all-α / all-β blocks do not
fold away.

WHY THIS IS NOT GATED AGAINST `external_blocks`. That function folds a<->b by default
(`key = min(combo, flip)`), returning doubles `['aaaa','abab']` with no `bbbb`. The
original U0 gate said "every block returned by external_blocks folds into exactly one
returned tag" -- which passes VACUOUSLY, because the β-majority blocks it should be
checking were already folded away upstream. The load-bearing assertion here is instead
against PySCF UCCSD's block names directly, plus the opposite-direction relationship
(unfolded is strictly larger, and folding it reproduces the folded set).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.spin import (  # noqa: E402
    _residual_template,
    _ucc_block_tag,
    external_blocks,
    ucc_independent_blocks,
)


def _tags(template, **kwargs):
    return ["".join(b[i.name] for i in template.indices)
            for b in external_blocks(template, **kwargs)]


class VocabularyTests(unittest.TestCase):
    def test_rank2_and_rank4(self):
        self.assertEqual(ucc_independent_blocks(2), ["aa", "bb"])
        self.assertEqual(ucc_independent_blocks(4), ["aaaa", "abab", "bbbb"])

    def test_matches_pyscf_uccsd_block_names(self):
        """The load-bearing assertion: PySCF UCCSD stores t1a/t1b and t2aa/t2ab/t2bb."""
        self.assertEqual(
            ["t1" + b[:1] for b in ucc_independent_blocks(2)],
            ["t1a", "t1b"],
        )
        self.assertEqual(
            ["t2" + b[:2] for b in ucc_independent_blocks(4)],
            ["t2aa", "t2ab", "t2bb"],
        )

    def test_n_plus_one_sectors_per_rank(self):
        for rank in (2, 4, 6, 8, 10):
            with self.subTest(rank=rank):
                self.assertEqual(len(ucc_independent_blocks(rank)), rank // 2 + 1)

    def test_higher_ranks_are_explicit(self):
        self.assertEqual(ucc_independent_blocks(6),
                         ["aaaaaa", "aabaab", "abbabb", "bbbbbb"])
        self.assertEqual(ucc_independent_blocks(8),
                         ["aaaaaaaa", "aaabaaab", "aabbaabb", "abbbabbb", "bbbbbbbb"])

    def test_blocks_are_spin_conserving_and_canonical(self):
        """Each half has the same α-count (spin conserved per line), and each half is
        α-before-β (so the tag is already its own canonical form)."""
        for rank in (2, 4, 6, 8):
            n = rank // 2
            for block in ucc_independent_blocks(rank):
                with self.subTest(rank=rank, block=block):
                    bra, ket = block[:n], block[n:]
                    self.assertEqual(bra.count("a"), ket.count("a"))
                    self.assertEqual(_ucc_block_tag(block), block)

    def test_no_duplicates(self):
        for rank in (2, 4, 6, 8, 10):
            blocks = ucc_independent_blocks(rank)
            with self.subTest(rank=rank):
                self.assertEqual(len(blocks), len(set(blocks)))

    def test_rejects_odd_and_degenerate_rank(self):
        for bad in (0, 1, 3, -2):
            with self.subTest(rank=bad), self.assertRaises(ValueError):
                ucc_independent_blocks(bad)


class BlockTagTests(unittest.TestCase):
    def test_folds_within_half_antisymmetry_only(self):
        self.assertEqual(_ucc_block_tag("baba"), "abab")
        self.assertEqual(_ucc_block_tag("abab"), "abab")

    def test_does_not_fold_the_spin_flip(self):
        """The closed-shell-only step that must NOT happen: bbbb is its own amplitude."""
        self.assertEqual(_ucc_block_tag("bbbb"), "bbbb")
        self.assertNotEqual(_ucc_block_tag("bbbb"), _ucc_block_tag("aaaa"))
        self.assertEqual(_ucc_block_tag("abbabb"), "abbabb")
        self.assertNotEqual(_ucc_block_tag("abbabb"), _ucc_block_tag("aabaab"))


class ExternalBlocksFoldFlagTests(unittest.TestCase):
    """`fold_spin_flip` and its relationship to the folded default."""

    @classmethod
    def setUpClass(cls):
        eqs = generate_cc_equations("ccsd", engine="diagram", canonical_fock=True)
        cls.templates = {m: _residual_template(m, eqs[m])
                         for m in ("singles", "doubles")}

    def test_default_is_unchanged_for_rcc_callers(self):
        self.assertEqual(_tags(self.templates["singles"]), ["aa"])
        self.assertEqual(_tags(self.templates["doubles"]), ["aaaa", "abab"])

    def test_unfolded_exposes_the_beta_majority_blocks(self):
        self.assertEqual(_tags(self.templates["singles"], fold_spin_flip=False),
                         ["aa", "bb"])
        self.assertEqual(_tags(self.templates["doubles"], fold_spin_flip=False),
                         ["aaaa", "abab", "baba", "bbbb"])

    def test_unfolded_is_strictly_larger(self):
        """Opposite direction to the original vacuous gate."""
        for manifold, template in self.templates.items():
            with self.subTest(manifold=manifold):
                folded = set(_tags(template))
                unfolded = set(_tags(template, fold_spin_flip=False))
                self.assertTrue(folded < unfolded)

    def test_folding_the_unfolded_set_reproduces_the_folded_set(self):
        """The relationship that IS meaningful: the two differ exactly by the a<->b
        identification, nothing else."""
        for manifold, template in self.templates.items():
            with self.subTest(manifold=manifold):
                flip = str.maketrans("ab", "ba")
                collapsed = set()
                for tag in _tags(template, fold_spin_flip=False):
                    collapsed.add(min(tag, tag.translate(flip)))
                self.assertEqual(collapsed, set(_tags(template)))

    def test_residual_blocks_fold_into_the_stored_vocabulary(self):
        """Every unfolded residual block maps into `ucc_independent_blocks` for its rank
        -- the non-vacuous form of the original gate."""
        for manifold, template in self.templates.items():
            rank = len(template.indices)
            stored = set(ucc_independent_blocks(rank))
            for tag in _tags(template, fold_spin_flip=False):
                with self.subTest(manifold=manifold, tag=tag):
                    self.assertIn(_ucc_block_tag(tag), stored)


if __name__ == "__main__":
    unittest.main()
