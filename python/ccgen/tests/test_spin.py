"""S0 gate: the spin-adaptation index model + single-term spin labeling.

Structural checks only (S0 is labeling, not coefficient integration): a term's
indices get spins consistent along shared lines, free indices take the requested
external block, and summed indices enumerate 2^(#distinct summed names) cases.
Coefficient algebra (UCC blocks = S1, RCC alpha=beta collapse = S2) is later.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.indices import make_occ, make_vir  # noqa: E402
from ccgen.spin import SpinIndex, spin_label_cases, SPINS  # noqa: E402


class SpinIndexTests(unittest.TestCase):
    def test_wraps_base_and_validates_spin(self):
        a = make_vir("a")
        sa = SpinIndex(a, "a")
        self.assertEqual(sa.name, "a")
        self.assertEqual(sa.space, "vir")
        self.assertEqual(sa.spin, "a")
        self.assertEqual(repr(sa), "aa")
        with self.assertRaises(ValueError):
            SpinIndex(a, "x")

    def test_identity_is_base_plus_spin(self):
        i = make_occ("i")
        self.assertEqual(SpinIndex(i, "a"), SpinIndex(make_occ("i"), "a"))
        self.assertNotEqual(SpinIndex(i, "a"), SpinIndex(i, "b"))


class SpinLabelCasesTests(unittest.TestCase):
    """Gate on the pp-ladder doubles term 1/2 t2(c,d,i,j) v(c,d,a,b):
    free i,j,a,b ; summed c,d."""

    def _pp_ladder(self):
        terms = [
            t for t in generate_cc_equations("ccd")["doubles"]
            if tuple(sorted(f.name for f in t.factors)) == ("t2", "v")
        ]
        pp = [
            t for t in terms
            if [i.name for i in t.factors[0].indices] == ["c", "d", "i", "j"]
        ]
        self.assertEqual(len(pp), 1)
        return pp[0]

    def test_case_count_is_two_to_the_summed_names(self):
        pp = self._pp_ladder()
        cases = spin_label_cases(pp, {"i": "a", "j": "b", "a": "a", "b": "b"})
        # two distinct summed names (c, d) -> 2^2 = 4 spin cases
        self.assertEqual(len(cases), 4)

    def test_free_indices_take_the_requested_external_block(self):
        pp = self._pp_ladder()
        ext = {"i": "a", "j": "b", "a": "a", "b": "b"}
        for label in spin_label_cases(pp, ext):
            for name, spin in ext.items():
                self.assertEqual(label[name].spin, spin, name)

    def test_shared_summed_name_has_one_consistent_spin(self):
        # c appears in BOTH t2 and v; it is ONE line and must carry ONE spin in
        # each case (the contracted line preserves spin). spin_label_cases keys
        # by name, so this holds by construction -- assert it explicitly.
        pp = self._pp_ladder()
        for label in spin_label_cases(pp, {"i": "a", "j": "b", "a": "a", "b": "b"}):
            # every index NAME maps to exactly one SpinIndex
            self.assertIn("c", label)
            self.assertIn("d", label)
            self.assertIn(label["c"].spin, SPINS)

    def test_summed_spins_are_enumerated_exhaustively(self):
        pp = self._pp_ladder()
        cases = spin_label_cases(pp, {"i": "a", "j": "b", "a": "a", "b": "b"})
        cd = {(label["c"].spin, label["d"].spin) for label in cases}
        self.assertEqual(cd, {("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")})

    def test_missing_external_spin_raises(self):
        pp = self._pp_ladder()
        with self.assertRaises(ValueError):
            spin_label_cases(pp, {"i": "a", "j": "b", "a": "a"})  # missing b


class BlockModelTests(unittest.TestCase):
    """S1.0/S1.1: spin conservation per line -> valid UCC blocks, and
    resolve_block on labeled factors."""

    @staticmethod
    def _factor(name, *spaces):
        # a bare Tensor with the given index spaces; spins assigned in the test
        from ccgen.tensors import Tensor
        idx = []
        for k, sp in enumerate(spaces):
            nm = f"x{k}"
            idx.append(make_occ(nm) if sp == "o" else make_vir(nm))
        return Tensor(name, tuple(idx))

    def _label(self, factor, spins):
        return {i.name: SpinIndex(i, s) for i, s in zip(factor.indices, spins)}

    def test_t1_blocks_are_aa_and_bb_only(self):
        from ccgen.spin import block_exists
        f = self._factor("t1", "v", "o")            # [vir, occ], line 0-1
        self.assertTrue(block_exists(f, self._label(f, "aa")))
        self.assertTrue(block_exists(f, self._label(f, "bb")))
        self.assertFalse(block_exists(f, self._label(f, "ab")))   # spin not conserved
        self.assertFalse(block_exists(f, self._label(f, "ba")))

    def test_t2_valid_blocks(self):
        from ccgen.spin import block_exists
        f = self._factor("t2", "v", "v", "o", "o")  # lines 0-2, 1-3
        # aaaa, bbbb: both lines same spin
        self.assertTrue(block_exists(f, self._label(f, "aaaa")))
        self.assertTrue(block_exists(f, self._label(f, "bbbb")))
        # abab: line 0-2 = a,a ; line 1-3 = b,b -> conserved
        self.assertTrue(block_exists(f, self._label(f, "abab")))
        self.assertTrue(block_exists(f, self._label(f, "baba")))
        # aabb: line 0-2 = a,b -> NOT conserved
        self.assertFalse(block_exists(f, self._label(f, "aabb")))
        self.assertFalse(block_exists(f, self._label(f, "aaab")))

    def test_v_physicist_lines_conserve_p_r_and_q_s(self):
        from ccgen.spin import block_exists
        # v = <pq||rs>, ccgen order [v,v,o,o]; physicist lines pair slot0-2, slot1-3
        f = self._factor("v", "v", "v", "o", "o")
        self.assertTrue(block_exists(f, self._label(f, "abab")))   # p=r=a, q=s=b
        self.assertTrue(block_exists(f, self._label(f, "aaaa")))
        self.assertFalse(block_exists(f, self._label(f, "aabb")))  # p=a,r=b broken

    def test_odd_rank_raises(self):
        from ccgen.spin import block_exists
        f = self._factor("odd", "v", "o", "o")
        with self.assertRaises(ValueError):
            block_exists(f, self._label(f, "aaa"))

    def test_resolve_block_tags_and_flags(self):
        from ccgen.spin import resolve_block
        f = self._factor("t2", "v", "v", "o", "o")
        tag, ok = resolve_block(f, self._label(f, "abab"))
        self.assertEqual((tag, ok), ("abab", True))
        tag, ok = resolve_block(f, self._label(f, "aabb"))
        self.assertEqual((tag, ok), ("aabb", False))

    def test_pp_ladder_factor_blocks_over_s0_cases(self):
        # Integration of S0 + S1.1 on the real pp-ladder: for the abab external
        # block, resolve each factor over the 4 summed-spin cases and see which
        # survive. t2(c,d,i,j) with i=a,j=b needs (c,d) so lines c-i, d-j conserve
        # -> c=a, d=b. Same for v(c,d,a,b) with a=a,b=b. So only the (c=a,d=b)
        # case leaves BOTH factors valid.
        from ccgen.spin import resolve_block
        terms = [
            t for t in generate_cc_equations("ccd")["doubles"]
            if tuple(sorted(f.name for f in t.factors)) == ("t2", "v")
        ]
        pp = [t for t in terms
              if [i.name for i in t.factors[0].indices] == ["c", "d", "i", "j"]][0]
        survivors = 0
        for label in spin_label_cases(pp, {"i": "a", "j": "b", "a": "a", "b": "b"}):
            if all(resolve_block(f, label)[1] for f in pp.factors):
                survivors += 1
                self.assertEqual(label["c"].spin, "a")
                self.assertEqual(label["d"].spin, "b")
        self.assertEqual(survivors, 1)


def _spin_structured_tensor(shape, factor_template, seed):
    """A random spin-orbital tensor with the physical UCC block structure: the
    array is zeroed on every FORBIDDEN block (spin not conserved along a line),
    and antisymmetrized within each vir/occ pair. Spin-orbital convention: index
    k has spin 'a' if k is even, 'b' if odd. ``factor_template`` is a bare Tensor
    of the same index-space pattern, used for the line-pairing rule."""
    import numpy as np
    from ccgen.spin import _line_pairs

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    n = len(shape) // 2
    # antisymmetrize within the first-n (vir) and last-n (occ) index blocks.
    # These tests use n==2 (rank-4 tensors: [v,v,o,o] and [v,v,v,v]).
    if n == 2:
        x = x - x.transpose(1, 0, 2, 3)   # antisym slots 0,1
        x = x - x.transpose(0, 1, 3, 2)   # antisym slots 2,3
    spin = lambda k: "a" if k % 2 == 0 else "b"
    mask = np.zeros_like(x)
    it = np.ndindex(*shape)
    pairs = _line_pairs(factor_template)
    for idx in it:
        if all(spin(idx[a]) == spin(idx[b]) for a, b in pairs):
            mask[idx] = 1.0
    return x * mask


def _slice_spin_block(arr, spin_factor):
    """Slice the spatial block of a spin-orbital array selected by a SpinFactor's
    per-slot spins (even indices = alpha, odd = beta)."""
    import numpy as np

    sets = [
        list(range(0, arr.shape[k], 2)) if si.spin == "a"
        else list(range(1, arr.shape[k], 2))
        for k, si in enumerate(spin_factor.indices)
    ]
    return arr[np.ix_(*sets)]


class UccIntegrateTermTests(unittest.TestCase):
    """S1.2: ucc_integrate_term reproduces the GCC residual's external block.

    The gate is the spin-orbital identity (no PySCF): on a SPIN-STRUCTURED
    spin-orbital tensor (forbidden blocks zeroed, as physical CC tensors are),
    the chosen external block of the GCC term equals the sum of its surviving
    integrated SpinTerms evaluated on the matching spatial block slices. This
    validates both the block filter and the raw GCC coefficient.
    """

    def _pp_ladder(self):
        terms = [
            t for t in generate_cc_equations("ccd")["doubles"]
            if tuple(sorted(f.name for f in t.factors)) == ("t2", "v")
        ]
        return [t for t in terms
                if [i.name for i in t.factors[0].indices] == ["c", "d", "i", "j"]][0]

    def test_pp_ladder_reproduces_gcc_abab_block(self):
        import numpy as np
        from ccgen.spin import ucc_integrate_term
        from ccgen.tensors import Tensor

        nocc_sp, nvir_sp = 2, 3
        no, nv = 2 * nocc_sp, 2 * nvir_sp
        t2tpl = Tensor("t2", (make_vir("a"), make_vir("b"), make_occ("i"), make_occ("j")))
        vtpl = Tensor("v", (make_vir("c"), make_vir("d"), make_vir("a"), make_vir("b")))
        t2 = _spin_structured_tensor((nv, nv, no, no), t2tpl, seed=1)
        v = _spin_structured_tensor((nv, nv, nv, nv), vtpl, seed=2)

        pp = self._pp_ladder()
        R = float(pp.coeff) * np.einsum("cdij,cdab->abij", t2, v)
        ea, eb = list(range(0, nv, 2)), list(range(1, nv, 2))
        oa, ob = list(range(0, no, 2)), list(range(1, no, 2))
        R_abab = R[np.ix_(ea, eb, oa, ob)]

        acc = np.zeros_like(R_abab)
        for st in ucc_integrate_term(pp, {"i": "a", "j": "b", "a": "a", "b": "b"}):
            f = {s.name: s for s in st.factors}
            acc += float(st.coeff) * np.einsum(
                "cdij,cdab->abij",
                _slice_spin_block(t2, f["t2"]),
                _slice_spin_block(v, f["v"]),
            )
        self.assertTrue(np.allclose(R_abab, acc, atol=1e-12),
                        np.max(np.abs(R_abab - acc)))

    def test_pp_ladder_reproduces_gcc_aaaa_block(self):
        # the same-spin (aaaa) external block: all four externals alpha, forcing
        # the summed c,d = a,a. A different external block from abab, exercising
        # the same identity on the fully-alpha sector. (Multi-survivor summation
        # is exercised at S1.3 on the full manifold, not by this single term.)
        import numpy as np
        from ccgen.spin import ucc_integrate_term
        from ccgen.tensors import Tensor

        nocc_sp, nvir_sp = 2, 3
        no, nv = 2 * nocc_sp, 2 * nvir_sp
        t2tpl = Tensor("t2", (make_vir("a"), make_vir("b"), make_occ("i"), make_occ("j")))
        vtpl = Tensor("v", (make_vir("c"), make_vir("d"), make_vir("a"), make_vir("b")))
        t2 = _spin_structured_tensor((nv, nv, no, no), t2tpl, seed=3)
        v = _spin_structured_tensor((nv, nv, nv, nv), vtpl, seed=4)

        pp = self._pp_ladder()
        R = float(pp.coeff) * np.einsum("cdij,cdab->abij", t2, v)
        ea, oa = list(range(0, nv, 2)), list(range(0, no, 2))
        R_aaaa = R[np.ix_(ea, ea, oa, oa)]

        acc = np.zeros_like(R_aaaa)
        for st in ucc_integrate_term(pp, {"i": "a", "j": "a", "a": "a", "b": "a"}):
            f = {s.name: s for s in st.factors}
            acc += float(st.coeff) * np.einsum(
                "cdij,cdab->abij",
                _slice_spin_block(t2, f["t2"]),
                _slice_spin_block(v, f["v"]),
            )
        self.assertTrue(np.allclose(R_aaaa, acc, atol=1e-12),
                        np.max(np.abs(R_aaaa - acc)))

    def test_forbidden_external_block_is_empty(self):
        # An external block that violates spin conservation on the residual's own
        # lines produces no surviving terms. R2(a,b,i,j) lines a-i, b-j: external
        # a=alpha,i=beta breaks the a-i line -> the v/t2 factors can't all be valid.
        from ccgen.spin import ucc_integrate_term

        pp = self._pp_ladder()
        # i=b but a=a: the residual line a<-i would be a<-b (broken) at the vertex
        sts = ucc_integrate_term(pp, {"i": "b", "j": "b", "a": "a", "b": "a"})
        # every case forces c=a (from a-c) and c=b (from i-c) simultaneously -> none
        self.assertEqual(sts, [])


if __name__ == "__main__":
    unittest.main()
