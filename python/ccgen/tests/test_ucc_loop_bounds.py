"""U3b.2a -- the emitted UCC kernels must take their loop bounds from the index's SPIN.

THE DEFECT THIS CLOSES. Every emitted kernel opened with

    const int no = reference.orbital_partition.n_occ;
    const int nv = reference.orbital_partition.n_virt;

and every loop ran `< no` / `< nv`. `build_ucc_fock_blocks` never sets
`orbital_partition`, so on a UCC state both counts are 0 -- which is why
`correlation ucc2` died at `sector residual shape mismatch at (rank 1, tag aa)`
rather than returning a wrong number. Measured pre-fix on the CCSD UCC TU: 1182
`< no`, 1192 `< nv`, all spin-blind.

WHERE THE SPIN COMES FROM. An `Index` carries only a space (occ/vir); the U1 bridge
drops spin by design, because RCC stores one spatial tensor per rank and has nothing
to route. `ucc_term_index_spins` (U3b.0) recovers it from the FACTORS instead: slot
*k* of `t2_abab` / `v_aaaa` / `f_bb` carries `tag[k]`. It returns {} for RCC terms,
which is what keeps the RCC emit byte-identical -- `_loop_bound` then sees spin=None
everywhere and emits the same `no`/`nv` it always did.

WHY THE ASSERTIONS ARE SHAPED THIS WAY. The obvious check -- "the aaaa kernel must
use only alpha counts" -- is WRONG, and believing it cost a wrong turn here. The
`aaaa` residual legitimately carries beta bounds: terms like `t2_abab * v_abab`
contribute to it through SUMMED beta indices even though all four FREE indices are
alpha (measured: 36 beta bounds in `doubles_aaaa`). What actually holds is the
MIRROR: `singles_aa` and `singles_bb` must be exact a<->b reflections of each other,
as must `doubles_aaaa` and `doubles_bbbb`, and `abab` must be balanced.

That mirror is the assertion with teeth. Under the space-grouped misreading of the
spin tag -- the trap this scope has hit four times, where `aaaa`/`bbbb` agree and
only `abab` differs -- `ucc_term_index_spins` raises on `v_abab` before emission,
so a same-spin-only check would never have been reached anyway. Mutation-tested in
both directions: ignoring the spin map restores all 1182/1192 spin-blind bounds, and
the space-grouped variant raises a slot-mapping conflict.
"""

from __future__ import annotations

import re
import unittest

from ccgen.generate import print_cpp_planck


def _kernel_bound_counts(src: str) -> dict[str, dict[str, int]]:
    """Per-kernel counts of each spin-resolved loop bound, scoped by line.

    Line-scoped rather than regex-spanned on purpose: a `.*?\\n\\}\\n` span match
    silently captures the wrong region (it stops at the first closing brace) and
    reports counts for a kernel it is not looking at.
    """
    counts: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in src.split("\n"):
        header = re.match(r"^\w[\w:<>,& ]*\s(compute_ucc_\w+)\(", line)
        if header:
            current = header.group(1)
            counts.setdefault(current, {k: 0 for k in ("noa", "nob", "nva", "nvb")})
        if current is not None:
            for name in counts[current]:
                counts[current][name] += len(re.findall(r"< " + name + r"\b", line))
    return counts


class UccLoopBoundsAreSpinResolvedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.src = print_cpp_planck("ccsd", ucc=True)
        cls.counts = _kernel_bound_counts(cls.src)

    def test_no_spin_blind_bounds_remain(self):
        """Not one `< no` / `< nv` survives in the UCC TU (pre-fix: 1182 / 1192)."""
        self.assertEqual(len(re.findall(r"< no(?![ab])", self.src)), 0)
        self.assertEqual(len(re.findall(r"< nv(?![ab])", self.src)), 0)

    def test_every_bound_is_accounted_for(self):
        """The spin-resolved bounds sum to the pre-fix spin-blind totals.

        Catches a fix that DROPS loops rather than routing them -- zero spin-blind
        bounds is also achievable by emitting nothing.
        """
        total_o = sum(len(re.findall(r"< " + s + r"\b", self.src)) for s in ("noa", "nob"))
        total_v = sum(len(re.findall(r"< " + s + r"\b", self.src)) for s in ("nva", "nvb"))
        self.assertEqual(total_o, 1182)
        self.assertEqual(total_v, 1192)

    def test_same_spin_blocks_are_exact_mirrors(self):
        """`aa`<->`bb` and `aaaa`<->`bbbb` must reflect under a<->b.

        The discriminating assertion: a spin routing that is wrong in any
        asymmetric way breaks the reflection.
        """
        def flip(d):
            return {"noa": d["nob"], "nob": d["noa"], "nva": d["nvb"], "nvb": d["nva"]}

        for a, b in (("singles_aa", "singles_bb"), ("doubles_aaaa", "doubles_bbbb")):
            ka = f"compute_ucc_ccsd_{a}_residual"
            kb = f"compute_ucc_ccsd_{b}_residual"
            self.assertEqual(flip(self.counts[ka]), self.counts[kb], f"{a} <-> {b}")

    def test_mixed_block_is_balanced(self):
        """`abab` touches alpha and beta equally -- the block where a wrong
        hypothesis changes the SHAPE, not merely the values."""
        c = self.counts["compute_ucc_ccsd_doubles_abab_residual"]
        self.assertEqual(c["noa"], c["nob"])
        self.assertEqual(c["nva"], c["nvb"])

    def test_beta_bounds_appear_in_the_aaaa_kernel(self):
        """Pins the fact that made the naive assertion wrong.

        `doubles_aaaa` has four ALPHA free indices but genuinely sums over beta
        (e.g. `t2_abab * v_abab`). If this ever reaches 0, either the manifold
        changed or someone "fixed" the bounds by forcing the target's tag onto
        every index -- which is the spin-blind defect wearing a spin suffix.
        """
        c = self.counts["compute_ucc_ccsd_doubles_aaaa_residual"]
        self.assertGreater(c["nob"], 0)
        self.assertGreater(c["nvb"], 0)


class RccEmitIsUnchangedTests(unittest.TestCase):
    """The RCC path must not observe U3b.2a at all."""

    def test_rcc_bounds_stay_spin_blind(self):
        for method in ("ccsd", "ccsdt"):
            src = print_cpp_planck(method)
            with self.subTest(method=method):
                self.assertEqual(len(re.findall(r"< n[ov][ab]\b", src)), 0)
                self.assertGreater(len(re.findall(r"< no\b", src)), 0)


if __name__ == "__main__":
    unittest.main()
