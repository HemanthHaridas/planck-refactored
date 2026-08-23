"""U2.0 -- the emitted UCC translation unit must read a DISTINCT array per spin block.

This is a structural pre-gate, and it fails on the tree as of 2026-08-22. It exists
because the emitter already produces a complete, plausible-looking UCC TU today:
`ucc_adapt_equations('ccsd')` -> `emit_planck_translation_unit(...)` emits six kernels
and a correct registry (`sector_tags` {1,"aa"} {1,"bb"} {2,"aaaa"} {2,"abab"}
{2,"bbbb"}) with no error. The amplitudes are genuinely block-resolved -- U1.1 did its
job, and `t2_aaaa` / `t2_abab` / `t2_bbbb` each bind their own `sector_tensor` view.

The ERIs and the Fock matrix are NOT. Measured on the emitted CCSD TU: `v_aaaa` and
`v_abab` BOTH emit as `mo_blocks.oovv`, and `f_aa` and `f_bb` BOTH emit as
`reference.f_ov`. `_map_eri_tensor` never receives the spin tag at all --
`planck_tensor_cpp.py`'s integral branch strips the suffix with
`re.fullmatch(r"([vf])(?:_([ab]+))?", ...)` and then uses only group(1).

The stripping comment there says the tag "routes STORAGE ... and does not change which
space block it is, so the mapping below is unchanged once the suffix is stripped". The
first clause is right and the conclusion does not follow: the tag does not change which
SPACE block it is (`oovv` stays `oovv`), but it absolutely changes which ARRAY -- under
UHF, <aa|aa>, <ab|ab> and <bb|bb> are three different integrals. Stripping the suffix
discards precisely the routing the first clause says it performs.

WHY A GATE AND NOT JUST A FIX. The defect is silent in the worst way: the TU compiles,
links, runs, and returns a plausible correlation energy. It is the B5 physicist-ERI
failure mode exactly (found only by injecting an FCI-correct oracle into live C++
state, after days). This gate costs seconds, needs no C++, no solve and no PySCF, and
it fires TODAY -- so the defect cannot be shipped by someone who sees a working emit
and reasonably concludes U4 is done.

Deliberately structural, not numeric. The scope doc is emphatic that UCC *equality*
must be gated numerically (a term-multiset comparison cannot tell "different algebra"
from "same algebra, different symmetry-equivalent writing" -- that cost V1.1e five
sub-steps). That warning is about comparing two residual manifolds. This is a
different question: does the emitted text read two distinct arrays where the algebra
says two distinct arrays? A distinctness assertion has no symmetry-equivalent-writing
freedom to be confused by, and every numeric UCC gate is downstream of it.

Scope: this checks storage ROUTING only -- that distinct blocks reach distinct arrays.
It does NOT check that each block reaches the CORRECT array; that is U3's
RHF-degenerate bytewise gate.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.emit.planck_tensor_cpp import emit_planck_translation_unit  # noqa: E402
from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.spin import ucc_adapt_equations  # noqa: E402


def _ucc_tu(method: str = "ccsd") -> str:
    """The emitted UCC translation unit for `method`.

    `force_arbitrary=True` because UCC rides the arbitrary-order runtime at every
    rank (the block-tagged amplitudes are `sector_tensor` reads, which only the
    arbitrary-order path exposes). `spin_adapted=True` because the equations are
    already resolved AlgebraTerms -- the closed-shell lowering must not run on them.
    """
    eqs = ucc_adapt_equations(generate_cc_equations(method))
    return emit_planck_translation_unit(
        method, eqs, force_arbitrary=True, spin_adapted=True)


class EmitReachesUccAtAllTests(unittest.TestCase):
    """Pin what already works, so a regression here is not misread as the defect below."""

    def test_emit_succeeds_and_covers_every_block(self):
        tu = _ucc_tu()
        for tag in ("singles_aa", "singles_bb",
                    "doubles_aaaa", "doubles_abab", "doubles_bbbb"):
            self.assertIn(f"compute_ccsd_{tag}_residual", tu,
                          f"no kernel emitted for UCC block {tag}")

    def test_registry_declares_every_block_as_a_sector(self):
        tu = _ucc_tu()
        declared = set(re.findall(r'sector_tags\.push_back\(\{(\d+), "([ab]+)"\}\)', tu))
        self.assertEqual(
            declared,
            {("1", "aa"), ("1", "bb"), ("2", "aaaa"), ("2", "abab"), ("2", "bbbb")})

    def test_amplitudes_already_read_distinct_arrays(self):
        """U1.1 landed this; asserted so the failures below are unambiguously v/f."""
        tu = _ucc_tu()
        for tag in ("aaaa", "abab", "bbbb"):
            self.assertIn(f'amplitudes.sector_tensor(2, "{tag}")', tu,
                          f"t2_{tag} does not bind its own sector view")


class DistinctArrayPerBlockTests(unittest.TestCase):
    """The pre-gate. Both of these fail on the current tree -- that is the point."""

    # WHY THIS IS COUNTED, NOT COMPARED PER KERNEL. The first draft of this gate
    # asserted "the aa singles kernel's Fock reads differ from the bb kernel's". That
    # is WRONG and a falsifiability probe caught it: measured, `singles_aa` and
    # `singles_bb` BOTH reference `f_aa` and `f_bb` (each block's residual couples to
    # the other spin through t1), so a CORRECTLY emitted TU would give the two kernels
    # identical accessor SETS. The draft assertion would have stayed red forever and
    # been "fixed" by deleting it.
    #
    # The invariant that actually holds is a counting one, per factor rather than per
    # kernel: the algebra names N distinct tagged tensors, so the emitted text must
    # name N distinct arrays. Collapsing any two of them is the defect.

    def _distinct_factor_names(self, root: str) -> set[str]:
        """The distinct `v_*` / `f_*` factor names the UCC algebra actually uses."""
        eqs = ucc_adapt_equations(generate_cc_equations("ccsd"))
        return {f.name for terms in eqs.values() for t in terms for f in t.factors
                if re.fullmatch(rf"{root}_[ab]+", f.name)}

    def test_eri_blocks_reach_distinct_arrays(self):
        """`v_aaaa`, `v_abab`, `v_bbbb` must each reach their own arrays.

        Post-U3.2 the emitted form is a per-block view (`v_abab_oovv`) bound from
        `mo_blocks.spin_block("oovv", "abab")`, not a bare `mo_blocks.oovv`. The
        earlier version of this assertion grepped for `mo_blocks\.<name>(` and,
        once the emit changed, matched `spin_block(` itself -- grouping every read
        under a phantom space block called "spin" and staying red for a reason
        that had nothing to do with the defect. A gate keyed to the shape of the
        code it is watching has to move when that shape does.
        """
        tu = _ucc_tu()
        expected = self._distinct_factor_names("v")
        self.assertEqual(len(expected), 3,
                         f"fixture drift: expected 3 v blocks, got {sorted(expected)}")

        # No untagged read may survive: that form is the collapse itself.
        self.assertEqual(
            re.findall(r"mo_blocks\.[a-z]{4}\(", tu), [],
            "untagged mo_blocks reads remain, collapsing three UHF integrals")

        reads = set(re.findall(r"\b(v_([ab]+)_([a-z]{4}))\(", tu))
        self.assertTrue(reads, "the TU reads no spin-blocked ERI at all")

        # Group by SPACE block: each must be served by more than one array, since
        # the algebra names three distinct spin blocks.
        by_space: dict[str, set[str]] = {}
        for full, _tag, space in reads:
            by_space.setdefault(space, set()).add(full)

        # A space block served by ONE array is only a collapse if more than one
        # spin block actually reads that space. Some spaces are legitimately
        # single-tag: `oovo`/`vovo`/`vovv` exist only for the mixed block (no
        # same-spin group needs them), and `ovvo` is reached only by the mixed
        # block too, because for same-spin it folds into `ovov` under the particle
        # swap -- a symmetry there and not for `abab`.
        #
        # Derived from the reads rather than hardcoded: a hardcoded exemption list
        # would have to be re-guessed every time the manifold changes, and would
        # quietly excuse a real collapse that happened to land on a listed name.
        tags_per_space: dict[str, set[str]] = {}
        for _full, tag, space in reads:
            tags_per_space.setdefault(space, set()).add(tag)

        collapsed = sorted(
            space for space, names in by_space.items()
            if len(names) == 1 and len(tags_per_space[space]) > 1)
        self.assertEqual(
            collapsed, [],
            f"these ERI space blocks are read by MULTIPLE spin blocks yet emit as "
            f"a SINGLE array: {collapsed}. The algebra names {sorted(expected)} -- "
            f"three different integrals under UHF.")

        # And the converse, so the exemption above cannot hide a total collapse:
        # at least one space must genuinely be served by three arrays.
        self.assertTrue(
            any(len(names) == 3 for names in by_space.values()),
            "no ERI space block is served by three arrays; the spin tag is not "
            "routing storage at all")

    def test_fock_blocks_reach_distinct_arrays(self):
        """`f_aa` and `f_bb` must reach their own matrices, not one `reference.f_ov`.

        Post-U3.3 the emitted form is a per-spin view (`f_aa_ov`) bound from
        `reference.spin_block("ov", "aa")`. Same shape update as the ERI
        assertion above, for the same reason.

        Still counted per factor, never compared per kernel: measured, both
        singles kernels legitimately reference `f_aa` AND `f_bb`, so a correct
        emit gives them identical accessor sets and a per-kernel difference
        assertion would be permanently red. That was caught by a falsifiability
        probe before this gate first landed and is the reason it is written this
        way.
        """
        tu = _ucc_tu()
        expected = self._distinct_factor_names("f")
        self.assertEqual(len(expected), 2,
                         f"fixture drift: expected 2 f blocks, got {sorted(expected)}")

        self.assertEqual(
            re.findall(r"reference\.f_[ov]{2}\(", tu), [],
            "untagged reference.f_* reads remain, collapsing f^alpha and f^beta")

        reads = set(re.findall(r"\b(f_([ab]+)_([ov]{2}))\(", tu))
        self.assertTrue(reads, "the TU reads no spin-resolved Fock element at all")

        by_space: dict[str, set[str]] = {}
        for full, _tag, space in reads:
            by_space.setdefault(space, set()).add(full)

        collapsed = sorted(sp for sp, names in by_space.items() if len(names) == 1)
        self.assertEqual(
            collapsed, [],
            f"these Fock space blocks emit as a SINGLE accessor: {collapsed}. "
            f"The algebra names {sorted(expected)} -- f^alpha and f^beta are "
            f"different matrices under UHF.")


if __name__ == "__main__":
    unittest.main()
