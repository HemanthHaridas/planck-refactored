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


def _sc(p, no):
    """Spin of combined-space position p (occ block [0,no), vir block [no,n)):
    even offset within its block = alpha (0), odd = beta (1)."""
    return (p if p < no else p - no) % 2


def _slice_spinterm_factor(sf, tensors, no, n):
    """Slice a SpinFactor's spatial block from the spin-orbital tensors. `v`/`f`
    are over the combined n-space (occ then vir); amplitudes are vir/occ-sized."""
    import numpy as np

    arr = tensors[sf.name]
    sets = []
    for k, si in enumerate(sf.indices):
        want = 0 if si.spin == "a" else 1
        if sf.name in ("v", "f"):
            base = range(0, no) if si.space == "occ" else range(no, n)
            sel = [p for p in base if _sc(p, no) == want]
        else:
            sel = [p for p in range(arr.shape[k]) if p % 2 == want]
        sets.append(sel)
    return arr[np.ix_(*sets)]


def _eval_spinterm(st, tensors, no, n, out_names):
    import string

    import numpy as np

    letters: dict = {}
    it = iter(string.ascii_lowercase)

    def L(nm):
        return letters.setdefault(nm, next(it))

    subs = [
        "".join(L(si.name) for si in sf.indices) for sf in st.factors
    ]
    out = "".join(L(nm) for nm in out_names)
    arrs = [_slice_spinterm_factor(sf, tensors, no, n) for sf in st.factors]
    return float(st.coeff) * np.einsum(
        f"{','.join(subs)}->{out}", *arrs
    )


class UccManifoldTests(unittest.TestCase):
    """S1.3a: full-manifold aggregation into UCC blocks.

    Structural checks on external_blocks / ucc_manifold, plus the full-manifold
    spin-orbital identity on the t2*v subset (all six terms, exercising
    multi-term aggregation + multi-survivor summation -- which the single
    pp-ladder does not). The f-containing terms use a 2D factor the compact
    evaluator here does not handle; the general evaluator + PySCF cross-check are
    S1.3b.
    """

    def _R2(self):
        from ccgen.tensors import Tensor
        return Tensor("R2", (make_vir("a"), make_vir("b"),
                             make_occ("i"), make_occ("j")))

    def test_doubles_external_blocks_are_canonical(self):
        from ccgen.spin import external_blocks
        blocks = external_blocks(self._R2())
        tags = {"".join(b[n] for n in ["a", "b", "i", "j"]) for b in blocks}
        # bbbb folds under global a<->b; the minimal UCC set is aaaa + abab
        self.assertEqual(tags, {"aaaa", "abab"})

    def test_singles_external_blocks(self):
        from ccgen.spin import external_blocks
        from ccgen.tensors import Tensor
        R1 = Tensor("R1", (make_vir("a"), make_occ("i")))
        tags = {"".join(b[n] for n in ["a", "i"])
                for b in external_blocks(R1)}
        self.assertEqual(tags, {"aa"})   # bb folds under a<->b

    def test_manifold_is_the_union_of_per_term_integrations(self):
        from ccgen.spin import ucc_manifold, ucc_integrate_term, external_blocks
        terms = generate_cc_equations("ccd")["doubles"]
        man = ucc_manifold(terms, self._R2())
        for block in external_blocks(self._R2()):
            tag = "".join(block[n] for n in ["a", "b", "i", "j"])
            expected = []
            for t in terms:
                expected.extend(ucc_integrate_term(t, block))
            self.assertEqual(man[tag], expected)

    def test_t2v_subset_reproduces_gcc_blocks(self):
        import numpy as np
        from ccgen.spin import ucc_integrate_term
        from ccgen.tests.residual_eval import residual_einsum, random_tensors

        nosp, nvsp = 2, 3
        no, nv = 2 * nosp, 2 * nvsp
        n = no + nv
        tn = random_tensors(no, nv, seed=0)
        # spin-structure t2 [v,v,o,o] and v (combined space): zero forbidden blocks
        t2 = tn["t2"]
        mask = np.zeros_like(t2)
        for idx in np.ndindex(*t2.shape):
            if (idx[0] % 2 == idx[2] % 2) and (idx[1] % 2 == idx[3] % 2):
                mask[idx] = 1
        tn["t2"] = t2 * mask
        v = tn["v"]
        mv = np.zeros_like(v)
        for idx in np.ndindex(*v.shape):
            if _sc(idx[0], no) == _sc(idx[2], no) and _sc(idx[1], no) == _sc(idx[3], no):
                mv[idx] = 1
        tn["v"] = v * mv

        t2v = [
            t for t in generate_cc_equations("ccd")["doubles"]
            if tuple(sorted(f.name for f in t.factors)) == ("t2", "v")
        ]
        Rgcc = sum(residual_einsum(t, no, nv, tensors=tn) for t in t2v)
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))
        for tag, sl in [("aaaa", (ve, ve, oe, oe)), ("abab", (ve, vo, oe, oo))]:
            Rb = Rgcc[np.ix_(*sl)]
            acc = np.zeros_like(Rb)
            for t in t2v:
                for st in ucc_integrate_term(t, dict(zip(["a", "b", "i", "j"], tag))):
                    acc += _eval_spinterm(st, tn, no, n, ["a", "b", "i", "j"])
            self.assertTrue(np.allclose(Rb, acc, atol=1e-12),
                            f"{tag}: {np.max(np.abs(Rb - acc))}")


def _spin_structure_all(tn, no, n):
    """Zero the forbidden blocks of every tensor in a residual_einsum tensor dict:
    t1/t2/t3 lines are pos%2 per axis; v/f are combined-space (occ then vir), spin
    by _sc. Returns the same dict, mutated."""
    import numpy as np

    def structure(arr, spin_of_axis):
        r = len(arr.shape) // 2
        m = np.zeros_like(arr)
        for idx in np.ndindex(*arr.shape):
            if all(spin_of_axis(idx[k]) == spin_of_axis(idx[k + r]) for k in range(r)):
                m[idx] = 1.0
        return arr * m

    for name in ("t1", "t2", "t3"):
        if name in tn:
            tn[name] = structure(tn[name], lambda p: p % 2)
    for name in ("v", "f"):
        if name in tn:
            tn[name] = structure(tn[name], lambda p: _sc(p, no))
    return tn


class UccFullManifoldTests(unittest.TestCase):
    """S1.3b: the general SpinTerm evaluator handles ALL factor kinds (t1, t2, f,
    v), so the full-manifold spin-orbital identity holds for the complete CCD and
    CCSD residuals -- not just the t2*v subset. This validates the UCC
    spin-integration MECHANISM end to end.

    (A direct PySCF uccsd.update_amps cross-check would additionally exercise the
    physicist->chemist ERI convention, which is an EMIT concern (S3); the
    integration itself is validated here against ccgen's own spin-orbital GCC
    residual, which is already PySCF-gccsd-validated to 1e-16.)
    """

    def _check(self, method, nosp, nvsp, seed):
        import numpy as np
        from ccgen.spin import ucc_manifold
        from ccgen.tensors import Tensor
        from ccgen.tests.residual_eval import residual_einsum, random_tensors

        no, nv = 2 * nosp, 2 * nvsp
        n = no + nv
        tn = _spin_structure_all(random_tensors(no, nv, seed=seed), no, n)
        eqs = generate_cc_equations(method)
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))

        if "singles" in eqs:
            R1 = Tensor("R1", (make_vir("a"), make_occ("i")))
            Rg = sum(residual_einsum(t, no, nv, tensors=tn) for t in eqs["singles"])
            acc = np.zeros_like(Rg[np.ix_(ve, oe)])
            for st in ucc_manifold(eqs["singles"], R1)["aa"]:
                acc += _eval_spinterm(st, tn, no, n, ["a", "i"])
            self.assertTrue(np.allclose(Rg[np.ix_(ve, oe)], acc, atol=1e-11),
                            f"{method} singles aa: "
                            f"{np.max(np.abs(Rg[np.ix_(ve, oe)] - acc))}")

        R2 = Tensor("R2", (make_vir("a"), make_vir("b"), make_occ("i"), make_occ("j")))
        Rg = sum(residual_einsum(t, no, nv, tensors=tn) for t in eqs["doubles"])
        man = ucc_manifold(eqs["doubles"], R2)
        for tag, sl in [("aaaa", (ve, ve, oe, oe)), ("abab", (ve, vo, oe, oo))]:
            Rb = Rg[np.ix_(*sl)]
            acc = np.zeros_like(Rb)
            for st in man[tag]:
                acc += _eval_spinterm(st, tn, no, n, ["a", "b", "i", "j"])
            self.assertTrue(np.allclose(Rb, acc, atol=1e-11),
                            f"{method} doubles {tag}: {np.max(np.abs(Rb - acc))}")

    def test_ccd_full_manifold(self):
        self._check("ccd", nosp=2, nvsp=3, seed=0)

    def test_ccsd_full_manifold(self):
        self._check("ccsd", nosp=3, nvsp=4, seed=1)


if __name__ == "__main__":
    unittest.main()
