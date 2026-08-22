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


def _rcc_doubles_residual(terms, tensors, no, n):
    """Spatial RCC doubles residual R2[a,b,i,j] (S2.2d-0): sum every single-block
    spatial SpinTerm over its factors' spatial slices. Each collapsed term is one
    fixed spin block, so evaluating it is a plain contraction (no spin sum) --
    _eval_spinterm already slices each factor by spin+space and einsums, so this
    is the RCC analog of residual_of over ONE spatial (abab) block. Output layout
    [a,b,i,j] matches the sliced GCC abab residual, so it reconnects the collapsed
    form to the already-validated S1/S2.1 identity."""
    import numpy as np

    nvs, nos = (n - no) // 2, no // 2
    acc = np.zeros((nvs, nvs, nos, nos))
    for st in terms:
        acc = acc + _eval_spinterm(st, tensors, no, n, ["a", "b", "i", "j"])
    return acc


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


class S4HigherRankTests(unittest.TestCase):
    """S4: the general-rank `_antisym_to_allowed` handles rank-6 (`t3`) / rank-8
    (`t4`) factors, where the old rank-4-only version silently dropped them.

    Structural gate (PySCF-free). The old code returned None ("genuinely zero")
    for reachable rank-6/8 cases via its rank-4-only candidate list, dropping
    valid terms; the general multiset+parity version integrates them. This gate
    pins that behavior: (1) `_antisym_to_allowed` maps a rank-6 t3 factor to an
    allowed block with a ±1 sign (not None) whenever the bra/ket spin multisets
    match, and returns None only when they genuinely differ; (2) the CCSDT
    triples manifold (diagram engine) integrates to a nonzero set of survivors.

    The rank-4 exchange mechanism is validated NUMERICALLY end-to-end
    (S1AntisymIntegrationTests, real integrals) and the general method is proven
    equivalent to the old rank-4 path there (all rank-4 gates green). A rank-6
    NUMERIC gate needs a closed-shell ANTISYMMETRIC t3 fixture (both properties at
    once -- the antisym re-expression is invalid on spin-structured tensors whose
    forbidden blocks are artificially zeroed); that fixture is future work, noted
    in the S4 scope.
    """

    def test_rank6_factor_maps_to_allowed_block(self):
        from ccgen.spin import _antisym_to_allowed, SpinIndex
        from ccgen.indices import make_occ, make_vir

        class _F:
            pass

        base = [make_vir("a"), make_vir("b"), make_vir("c"),
                make_occ("i"), make_occ("j"), make_occ("k")]
        f = _F()
        f.indices = base
        # bra spins {a,a,b}, ket spins {a,a,b}: multisets match -> allowed, ±1 sign
        spins = ["a", "a", "b", "a", "b", "a"]  # bra a,a,b ; ket a,b,a
        label = {b.name: SpinIndex(b, s) for b, s in zip(base, spins)}
        res = _antisym_to_allowed(f, label)
        self.assertIsNotNone(res, "reachable rank-6 block wrongly dropped")
        sign, idx = res
        self.assertIn(sign, (1, -1))
        # every line conserves spin after the re-expression
        rr = len(idx) // 2
        self.assertTrue(all(idx[m].spin == idx[m + rr].spin for m in range(rr)))
        # bra/ket multiset MISMATCH -> genuinely None
        spins2 = ["a", "a", "a", "a", "b", "b"]  # bra a,a,a ; ket a,b,b
        label2 = {b.name: SpinIndex(b, s) for b, s in zip(base, spins2)}
        self.assertIsNone(_antisym_to_allowed(f, label2),
                          "spin-multiset mismatch should be genuinely zero")

    def test_ccsdt_triples_integrate_without_dropping(self):
        from ccgen.spin import ucc_integrate_term_antisym
        ext = {"a": "a", "b": "b", "c": "a", "i": "a", "j": "b", "k": "a"}
        t3terms = [t for t in generate_cc_equations("ccsdt",
                                                    engine="diagram")["triples"]
                   if any(f.name == "t3" for f in t.factors)]
        self.assertTrue(t3terms, "no t3-containing triples terms")
        total = sum(len(ucc_integrate_term_antisym(t, ext)) for t in t3terms)
        # the old rank-4 bug returned None for every rank-6 factor -> zero
        # survivors across all t3 terms; the general version must integrate them.
        self.assertGreater(total, 0, "t3 terms all dropped (rank>4 bug)")


def _closed_shell_tensors(no, nv, seed):
    """A CLOSED-SHELL (alpha == beta) spin-orbital tensor dict for the general
    SpinTerm evaluator, built from a single SPATIAL seed and lifted into the
    interleaved (even=alpha, odd=beta) spin-orbital layout.

    For a closed-shell RHF reference the spin-orbital tensors are fixed by
    spatial data: pick spatial amplitudes/integrals once, then for every
    spin-allowed block copy the spatial values with the physical antisymmetry
    sign. This makes the closed-shell relations hold by construction (t1a==t1b;
    t2aa==antisym(t2ab)); S2.0 then VERIFIES `t2aa_from_t2ab` extracts that
    relation, and the strong independent S2.0 check is PySCF UCCSD (S2PyscfTests).
    Layout matches residual_eval: t1/t2 are [v..,o..] spin-orbital; v/f are over
    the combined n-space (occ then vir), each block interleaved.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    nos, nvs = no // 2, nv // 2
    n = no + nv

    # spatial seeds
    t1s = rng.standard_normal((nvs, nos))
    t2s = rng.standard_normal((nvs, nvs, nos, nos))          # abab spatial
    vs = rng.standard_normal((n // 2, n // 2, n // 2, n // 2))
    fs = rng.standard_normal((n // 2, n // 2))

    def sp_amp(p):        # spatial index of amplitude axis (even=a, odd=b)
        return p // 2

    def sp_v(p):          # spatial index of combined v/f axis
        return (p // 2) if p < no else (p - no) // 2

    t1 = np.zeros((nv, no))
    for a in range(nv):
        for i in range(no):
            if a % 2 == i % 2:                               # spin conserved
                t1[a, i] = t1s[sp_amp(a), sp_amp(i)]

    t2 = np.zeros((nv, nv, no, no))
    for a in range(nv):
        for b in range(nv):
            for i in range(no):
                for j in range(no):
                    if a % 2 == i % 2 and b % 2 == j % 2:    # spin conserved
                        # antisymmetric spatial doubles = t2ab - swaps as needed
                        t2[a, b, i, j] = (
                            t2s[sp_amp(a), sp_amp(b), sp_amp(i), sp_amp(j)]
                            - t2s[sp_amp(b), sp_amp(a), sp_amp(i), sp_amp(j)]
                            if (a % 2) == (b % 2)             # same-spin: antisym
                            else t2s[sp_amp(a), sp_amp(b), sp_amp(i), sp_amp(j)]
                        )

    # v is spin-conserving-per-line over the combined space (occ then vir),
    # alpha==beta by the sp_v spatial seed. This is NOT the fully antisymmetric
    # physicist integral -- S2.1/S2.2a/b are pure block-slicing identities that
    # hold for any v. NOTE (S2.2c finding): ccgen's v structurally CANNOT carry
    # the closed-shell relation v[aaaa] = v[abab] - P(v[abab]); that relation
    # needs the exchange term, which a per-line-spin-conserving v lacks (the
    # ket-swapped abab entry is spin-forbidden -> zero here). See the S2.2c note
    # in CCGEN_SPIN_ADAPTATION_SCOPE.md.
    v = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if _sc(p, no) == _sc(r, no) and _sc(q, no) == _sc(s, no):
                        v[p, q, r, s] = vs[sp_v(p), sp_v(q), sp_v(r), sp_v(s)]

    f = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            if _sc(p, no) == _sc(q, no):
                f[p, q] = fs[sp_v(p), sp_v(q)]

    return {"t1": t1, "t2": t2, "v": v, "f": f}


class S2ClosedShellRelationTests(unittest.TestCase):
    """S2.0: the closed-shell (alpha==beta) doubles block relation + t1 collapse.

    Pins the swap+sign convention of `t2aa = t2ab - P(t2ab)`: on a closed-shell
    spin-orbital t2, the directly-sliced same-spin (aaaa) block equals
    `t2aa_from_t2ab` of the sliced mixed (abab) block, and t1a==t1b. This settles
    the single most error-prone spot before any equation work. The STRONG,
    independent S2.0 gate is PySCF UCCSD (S2PyscfTests.test_s20_*); this class is
    the always-runnable tripwire on ccgen's own closed-shell lift.
    """

    def _tn(self, seed=7):
        return _closed_shell_tensors(no=6, nv=8, seed=seed)

    def test_t2aa_is_antisymmetrized_t2ab(self):
        import numpy as np
        from ccgen.spin import t2aa_from_t2ab
        t2 = self._tn()["t2"]
        nv, no = t2.shape[0], t2.shape[2]
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))
        t2ab = t2[np.ix_(ve, vo, oe, oo)]
        t2aa = t2[np.ix_(ve, ve, oe, oe)]
        self.assertTrue(np.allclose(t2aa, t2aa_from_t2ab(t2ab), atol=1e-13),
                        np.max(np.abs(t2aa - t2aa_from_t2ab(t2ab))))

    def test_t1a_equals_t1b(self):
        import numpy as np
        t1 = self._tn(seed=8)["t1"]
        nv, no = t1.shape
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))
        self.assertTrue(np.allclose(t1[np.ix_(ve, oe)], t1[np.ix_(vo, oo)],
                                    atol=1e-13))


class S2AbabSubstitutionTests(unittest.TestCase):
    """S2.1: the RCC single residual as the UCC abab-block residual evaluated
    under the S2.0 substitution.

    Proves the 'abab + substitution' model BEFORE any symbolic collapse: evaluate
    the UCC abab-block manifold reading the same-spin (aaaa/bbbb) t2 slices ONLY
    through `t2aa_from_t2ab(t2ab)` -- i.e. the RCC model stores a single mixed
    block and reconstructs the same-spin one -- and require it reproduces the
    directly-sliced GCC abab residual on closed-shell tensors. If this fails the
    'abab + substitution' model is wrong and no symbolic work (S2.2) saves it.
    The independent numeric anchor is PySCF rccsd.update_amps (S2PyscfTests).
    """

    def _check(self, method, seed):
        import numpy as np
        from ccgen.spin import ucc_manifold, t2aa_from_t2ab
        from ccgen.tensors import Tensor
        from ccgen.tests.residual_eval import residual_einsum

        no, nv = 6, 8
        n = no + nv
        tn = _closed_shell_tensors(no, nv, seed=seed)
        eqs = generate_cc_equations(method)
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))

        # the GCC abab residual on the closed-shell tensors (the RCC target)
        R2 = Tensor("R2", (make_vir("a"), make_vir("b"),
                           make_occ("i"), make_occ("j")))
        Rg = sum(residual_einsum(t, no, nv, tensors=tn) for t in eqs["doubles"])
        Rb = Rg[np.ix_(ve, vo, oe, oo)]

        # substituted tensor set: DROP the same-spin t2 slices and rebuild them
        # from the stored mixed block, so only t2ab is independent RCC data.
        t2 = tn["t2"].copy()
        t2aa_rec = t2aa_from_t2ab(t2[np.ix_(ve, vo, oe, oo)])
        t2[np.ix_(ve, ve, oe, oe)] = t2aa_rec
        t2[np.ix_(vo, vo, oo, oo)] = t2aa_rec       # alpha==beta closed shell
        tn_sub = dict(tn, t2=t2)

        acc = np.zeros_like(Rb)
        for st in ucc_manifold(eqs["doubles"], R2)["abab"]:
            acc += _eval_spinterm(st, tn_sub, no, n, ["a", "b", "i", "j"])
        self.assertTrue(np.allclose(Rb, acc, atol=1e-11),
                        f"{method} abab-substitution: {np.max(np.abs(Rb - acc))}")

    def test_ccd_abab_substitution(self):
        self._check("ccd", seed=0)

    def test_ccsd_abab_substitution(self):
        self._check("ccsd", seed=1)


class S22aCanonicalizeBlocksTests(unittest.TestCase):
    """S2.2a: canonicalize the abab-residual factors to the global-flip block rep.

    The first, purely-mechanical piece of the S2.2 collapse: flip a<->b on any
    factor whose spin block is not canonical (baba->abab, bbbb->aaaa), keeping
    spatial indices and coefficients. Under the closed-shell alpha==beta symmetry
    the flipped factor is the identical spatial quantity, so the rewrite must be a
    NO-OP on the residual value -- gated at maxdiff 0 by the S2.1 harness. Also
    asserts only {aaaa, abab} blocks survive (the reduction the step exists for).
    """

    def _check(self, method, seed):
        import numpy as np
        from ccgen.spin import ucc_manifold, canonicalize_spin_blocks
        from ccgen.tensors import Tensor
        from ccgen.tests.residual_eval import residual_einsum

        no, nv = 6, 8
        n = no + nv
        tn = _closed_shell_tensors(no, nv, seed=seed)
        eqs = generate_cc_equations(method)
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))

        R2 = Tensor("R2", (make_vir("a"), make_vir("b"),
                           make_occ("i"), make_occ("j")))
        Rg = sum(residual_einsum(t, no, nv, tensors=tn) for t in eqs["doubles"])
        Rb = Rg[np.ix_(ve, vo, oe, oo)]

        raw = ucc_manifold(eqs["doubles"], R2)["abab"]
        canon = [canonicalize_spin_blocks(st) for st in raw]

        # structural: every surviving block is its own global-flip canonical
        # form (doubles collapse to {aaaa, abab}; singles factors to {aa}). The
        # non-canonical reps baba/bbbb/bb must all be gone.
        from ccgen.spin import _canonical_block
        blocks = {f.block for st in canon for f in st.factors}
        noncanon = {b for b in blocks if _canonical_block(b)[1]}
        self.assertEqual(noncanon, set(),
                         f"{method}: non-canonical blocks survived: {noncanon}")
        self.assertTrue({"baba", "bbbb", "bb"}.isdisjoint(blocks))

        # consistency: each factor's block tag must match its SpinIndex spins.
        # (The numeric no-op alone cannot catch a relabel that leaves the indices
        # unflipped -- eval reads spins, not the tag -- but S2.2b/c read the tag,
        # so a tag/spin mismatch would silently corrupt the collapse.)
        for st in canon:
            for f in st.factors:
                self.assertEqual(f.block, "".join(si.spin for si in f.indices),
                                 f"{method}: {f.name} block/spin mismatch")

        # numeric no-op: canonicalized terms reproduce the abab residual exactly
        acc = np.zeros_like(Rb)
        for st in canon:
            acc += _eval_spinterm(st, tn, no, n, ["a", "b", "i", "j"])
        self.assertTrue(np.allclose(Rb, acc, atol=1e-13),
                        f"{method} canonicalize: {np.max(np.abs(Rb - acc))}")

    def test_ccd_canonicalize_is_noop(self):
        self._check("ccd", seed=0)

    def test_ccsd_canonicalize_is_noop(self):
        self._check("ccsd", seed=1)


class S22bAmplitudeCollapseTests(unittest.TestCase):
    """S2.2b: collapse the same-spin amplitude block t2[aaaa] -> t2ab - P(t2ab).

    The first step where coefficients change (one term splits into two). Applies
    `collapse_amplitudes` to the canonicalized abab residual and requires:
    (1) no t2[aaaa] factor survives -- every t2 is now the single spatial abab
    block; (2) the term count grew (the split actually fired); (3) the residual
    value is UNCHANGED (S2.1 harness, maxdiff ~1e-13) -- proving the symbolic
    split equals the numeric S2.0 relation already validated at S2.1. v[aaaa] is
    intentionally still present (S2.2c handles integrals).
    """

    def _check(self, method, seed):
        import numpy as np
        from ccgen.spin import ucc_manifold, canonicalize_spin_blocks
        from ccgen.spin import collapse_amplitudes
        from ccgen.tensors import Tensor
        from ccgen.tests.residual_eval import residual_einsum

        no, nv = 6, 8
        n = no + nv
        tn = _closed_shell_tensors(no, nv, seed=seed)
        eqs = generate_cc_equations(method)
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))

        R2 = Tensor("R2", (make_vir("a"), make_vir("b"),
                           make_occ("i"), make_occ("j")))
        Rg = sum(residual_einsum(t, no, nv, tensors=tn) for t in eqs["doubles"])
        Rb = Rg[np.ix_(ve, vo, oe, oo)]

        canon = [canonicalize_spin_blocks(st)
                 for st in ucc_manifold(eqs["doubles"], R2)["abab"]]
        collapsed = [c for st in canon for c in collapse_amplitudes(st)]

        # (1) no same-spin t2 amplitude remains
        t2blocks = {f.block for st in collapsed for f in st.factors
                    if f.name == "t2"}
        self.assertEqual(t2blocks, {"abab"}, f"{method}: t2 blocks {t2blocks}")
        # tag/spin consistency preserved through the split
        for st in collapsed:
            for f in st.factors:
                self.assertEqual(f.block, "".join(si.spin for si in f.indices))
        # (2) the split fired -- more terms out than in
        self.assertGreater(len(collapsed), len(canon))

        # (3) residual value unchanged
        acc = np.zeros_like(Rb)
        for st in collapsed:
            acc += _eval_spinterm(st, tn, no, n, ["a", "b", "i", "j"])
        self.assertTrue(np.allclose(Rb, acc, atol=1e-13),
                        f"{method} amplitude-collapse: {np.max(np.abs(Rb - acc))}")

    def test_ccd_amplitude_collapse(self):
        self._check("ccd", seed=0)

    def test_ccsd_amplitude_collapse(self):
        self._check("ccsd", seed=1)


class S22cIntegralCollapseStructureTests(unittest.TestCase):
    """S2.2c: `collapse_integrals` splits the same-spin integral block
    v[aaaa] -> v[abab] - v[abab](ket swap), the integral analog of S2.2b.

    STRUCTURAL gate only. Unlike S2.2b, the residual-value no-op CANNOT be gated
    on the synthetic fixture: ccgen's v is spin-conserving-per-line and
    structurally cannot carry the closed-shell relation v[aaaa] = v[abab] - P
    (the ket-swapped abab entry is spin-forbidden -> zero for a per-line v, so
    v[aaaa] would collapse to just v[abab], not the antisymmetrized combination).
    The exchange term the relation needs lives in separate ccgen terms, not in v.
    So the numeric no-op belongs to a later step with real (chemist 2J-K)
    integrals -- see the S2.2c note in CCGEN_SPIN_ADAPTATION_SCOPE.md and S2.2d.
    Here we pin the rewrite's STRUCTURE: after S2.2a->b->c every doubles factor is
    a single spatial block, the v split fires, and tag/spin stays consistent.
    """

    def _check(self, method, seed):
        from ccgen.spin import (ucc_manifold, canonicalize_spin_blocks,
                                 collapse_amplitudes, collapse_integrals)
        from ccgen.tensors import Tensor

        no, nv = 6, 8
        R2 = Tensor("R2", (make_vir("a"), make_vir("b"),
                           make_occ("i"), make_occ("j")))
        canon = [canonicalize_spin_blocks(st)
                 for st in ucc_manifold(generate_cc_equations(method)["doubles"],
                                        R2)["abab"]]
        amp = [c for st in canon for c in collapse_amplitudes(st)]
        collapsed = [c for st in amp for c in collapse_integrals(st)]

        # every v is the single spatial abab block; no same-spin v[aaaa] remains
        vblocks = {f.block for st in collapsed for f in st.factors
                   if f.name == "v"}
        self.assertEqual(vblocks, {"abab"}, f"{method}: v blocks {vblocks}")
        # after the full pipeline every doubles factor is a canonical single block
        allblocks = {(f.name, f.block) for st in collapsed for f in st.factors}
        self.assertTrue(all(b in ("abab", "aa") for _, b in allblocks),
                        f"{method}: non-single blocks {allblocks}")
        # the v split fired, and tag/spin stays consistent
        self.assertGreater(len(collapsed), len(amp))
        for st in collapsed:
            for f in st.factors:
                self.assertEqual(f.block, "".join(si.spin for si in f.indices))

    def test_ccd_integral_collapse_structure(self):
        self._check("ccd", seed=0)

    def test_ccsd_integral_collapse_structure(self):
        self._check("ccsd", seed=1)


class S22d0SpatialResidualTests(unittest.TestCase):
    """S2.2d-0: the spatial RCC doubles residual evaluator reproduces the
    already-validated S2.1 abab block, PySCF-free.

    `_rcc_doubles_residual` sums the collapsed single-block SpinTerms as plain
    spatial contractions. Gated on the AMPLITUDE-collapsed manifold (post-S2.2b):
    there every factor is single-block AND the value is preserved on the
    synthetic fixture (~1e-13 vs the sliced GCC abab residual). The FULL integral
    collapse (post-S2.2c) is intentionally NOT value-gated here -- ccgen's
    spin-conserving v cannot carry v[aaaa]=v[abab]-P, so the v-split is only
    value-correct on real chemist integrals (S2.2d-2). This asserts that gap
    explicitly, so the split between the PySCF-free baseline (amp) and the
    real-integral proof (S2.2d-2) is pinned, not forgotten.
    """

    def _manifolds(self, method):
        from ccgen.spin import (ucc_manifold, canonicalize_spin_blocks,
                                 collapse_amplitudes, collapse_integrals)
        from ccgen.tensors import Tensor
        R2 = Tensor("R2", (make_vir("a"), make_vir("b"),
                           make_occ("i"), make_occ("j")))
        canon = [canonicalize_spin_blocks(st)
                 for st in ucc_manifold(generate_cc_equations(method)["doubles"],
                                        R2)["abab"]]
        amp = [c for st in canon for c in collapse_amplitudes(st)]
        collapsed = [c for st in amp for c in collapse_integrals(st)]
        return amp, collapsed

    def _check(self, method, seed):
        import numpy as np
        from ccgen.tensors import Tensor
        from ccgen.tests.residual_eval import residual_einsum

        no, nv = 6, 8
        n = no + nv
        tn = _closed_shell_tensors(no, nv, seed=seed)
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))

        Rg = sum(residual_einsum(t, no, nv, tensors=tn)
                 for t in generate_cc_equations(method)["doubles"])
        Rb = Rg[np.ix_(ve, vo, oe, oo)]

        amp, collapsed = self._manifolds(method)

        # PySCF-free baseline: amp-collapsed spatial residual == sliced GCC abab
        R_amp = _rcc_doubles_residual(amp, tn, no, n)
        self.assertTrue(np.allclose(R_amp, Rb, atol=1e-11),
                        f"{method} spatial(amp): {np.max(np.abs(R_amp - Rb))}")

        # the FULL integral collapse is NOT value-preserving on the synthetic
        # spin-conserving v (the documented S2.2c finding) -- assert the gap so
        # the deferral to S2.2d-2 is explicit, not silently wrong.
        R_full = _rcc_doubles_residual(collapsed, tn, no, n)
        self.assertFalse(np.allclose(R_full, Rb, atol=1e-6),
                         f"{method}: integral collapse unexpectedly value-preserving "
                         "on synthetic v -- S2.2c finding may be stale, recheck")

    def test_ccd_spatial_residual(self):
        self._check("ccd", seed=0)

    def test_ccsd_spatial_residual(self):
        self._check("ccsd", seed=1)


class S22d1MergeTests(unittest.TestCase):
    """S2.2d-1: merge structurally-identical spatial terms.

    `merge_terms` groups collapsed SpinTerms by a factor-order- and
    summed-relabel-invariant signature and sums coefficients. Requires: (1) the
    residual VALUE is unchanged (merge is pure algebra -- value-preserving on the
    synthetic fixture regardless of v); (2) the term count actually dropped (real
    merges fired, not a no-op); (3) the merged coefficients carry the
    characteristic RCC 2J-K combinations (e.g. |coeff| in {2, 4} appears -- the
    exchange/Coulomb pair sums), which the un-merged collapsed list does not.
    """

    def _collapsed(self, method):
        from ccgen.spin import (ucc_manifold, canonicalize_spin_blocks,
                                 collapse_amplitudes, collapse_integrals)
        from ccgen.tensors import Tensor
        R2 = Tensor("R2", (make_vir("a"), make_vir("b"),
                           make_occ("i"), make_occ("j")))
        canon = [canonicalize_spin_blocks(st)
                 for st in ucc_manifold(generate_cc_equations(method)["doubles"],
                                        R2)["abab"]]
        amp = [c for st in canon for c in collapse_amplitudes(st)]
        return [c for st in amp for c in collapse_integrals(st)]

    def _check(self, method, seed):
        import numpy as np
        from ccgen.spin import merge_terms

        no, nv = 6, 8
        n = no + nv
        tn = _closed_shell_tensors(no, nv, seed=seed)
        externals = {"a", "b", "i", "j"}

        collapsed = self._collapsed(method)
        merged = merge_terms(collapsed, externals)

        # (2) real merges fired
        self.assertLess(len(merged), len(collapsed),
                        f"{method}: no merge ({len(merged)} == {len(collapsed)})")
        # (1) value unchanged (merge is tensor-independent algebra)
        R_before = _rcc_doubles_residual(collapsed, tn, no, n)
        R_after = _rcc_doubles_residual(merged, tn, no, n)
        self.assertTrue(np.allclose(R_before, R_after, atol=1e-12),
                        f"{method} merge: {np.max(np.abs(R_before - R_after))}")
        # (3) the RCC 2J-K coefficient combinations appear
        merged_absc = {abs(t.coeff) for t in merged}
        collapsed_absc = {abs(t.coeff) for t in collapsed}
        self.assertTrue(any(c > 1 for c in merged_absc),
                        f"{method}: no 2J-K combination in merged coeffs "
                        f"{sorted(str(c) for c in merged_absc)}")
        self.assertFalse(any(c > 1 for c in collapsed_absc),
                         "unmerged collapsed terms should have |coeff|<=1")

        # idempotent: merging again is a no-op
        self.assertEqual(len(merge_terms(merged, externals)), len(merged))

    def test_ccd_merge(self):
        self._check("ccd", seed=0)

    def test_ccsd_merge(self):
        self._check("ccsd", seed=1)


try:
    from pyscf import gto, scf  # noqa: F401
    from pyscf.cc import rccsd, uccsd  # noqa: F401
    _HAVE_PYSCF = True
except ImportError:  # pragma: no cover - pyscf lives in tests/pyscf/.venv
    _HAVE_PYSCF = False


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S2PyscfTests(unittest.TestCase):
    """S2.0 against an independent oracle: PySCF UCCSD's own closed-shell block
    relation. rccsd/uccsd are the RHF closed-shell CC residual -- exactly the RCC
    target. Run with the pyscf venv:

        tests/pyscf/.venv/bin/python -m unittest ccgen.tests.test_spin
    """

    def test_s20_t2aa_from_uccsd_blocks(self):
        # Converge UCCSD from a UHF reference on a closed-shell molecule; its
        # spin blocks satisfy the closed-shell relation. PySCF layout is
        # [i,j,a,b]; our helper is [v,v,o,o] with a virtual swap. Confirm both
        # give PySCF's own t2aa from its t2ab. Water/sto-3g stays symmetric
        # (UHF==RHF) and has 5 occ / 2 vir spatials, so t2aa is a real nonzero
        # block that exercises the swap sign; equally-spaced H4 breaks spin
        # symmetry, which would violate alpha==beta.
        import numpy as np
        from ccgen.spin import t2aa_from_t2ab
        mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
                    basis="sto-3g", spin=0, verbose=0)
        mf = scf.UHF(mol).run()
        mc = uccsd.UCCSD(mf)
        mc.kernel()
        t2aa, t2ab, _ = mc.t2                       # [i,j,a,b]
        self.assertGreater(np.max(np.abs(t2aa)), 1e-3, "t2aa must be nonzero")
        # PySCF's relation: same-spin = antisymmetrize the mixed block
        self.assertLess(np.max(np.abs(t2aa - (t2ab - t2ab.transpose(1, 0, 2, 3)))),
                        1e-7, "PySCF occ-swap relation")
        # our helper in [v,v,o,o] must reproduce it
        t2ab_vvoo = t2ab.transpose(2, 3, 0, 1)
        got = t2aa_from_t2ab(t2ab_vvoo).transpose(2, 3, 0, 1)  # back to [i,j,a,b]
        self.assertLess(np.max(np.abs(got - t2aa)), 1e-7,
                        "t2aa_from_t2ab vs PySCF UCCSD")
        # t1a == t1b on the closed-shell reference
        self.assertLess(np.max(np.abs(mc.t1[0] - mc.t1[1])), 1e-5)


def _real_antisym_tensors(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
                          basis="sto-3g"):
    """Build the REAL antisymmetric spin-orbital CC tensors from a PySCF RHF
    RCCSD reference, in the residual_eval layout. `v = <pq||rs>` (fully
    antisym, from ao2mo <pq|rs> minus exchange); `t1`/`t2` from the converged
    RCCSD amplitudes lifted to spin-orbital (closed-shell fill); `f` diagonal MO
    energies. These are the tensors ccgen's GCC equations actually consume, so
    they are the correct oracle for spin integration -- unlike the synthetic
    spin-conserving `_closed_shell_tensors`, their forbidden blocks are nonzero
    (they carry exchange). Returns (tensors, no, nv, mf, cc)."""
    import numpy as np
    mol = gto.M(atom=atom, basis=basis, spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    cc = rccsd.RCCSD(mf)
    cc.kernel()
    nocc = cc.nocc
    nmo = mf.mo_coeff.shape[1]
    nvir = nmo - nocc
    no, nv, n = 2 * nocc, 2 * nvir, 2 * nmo
    from pyscf import ao2mo
    eri = ao2mo.kernel(mol, mf.mo_coeff, aosym="s1").reshape(nmo, nmo, nmo, nmo)
    g = eri.transpose(0, 2, 1,3)   # physicist <pq|rs>

    def csp(p):
        return (p // 2) if p < no else nocc + ((p - no) // 2)

    def cspin(p):
        return (p % 2) if p < no else ((p - no) % 2)

    v = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    c = (g[csp(p), csp(q), csp(r), csp(s)]
                         if cspin(p) == cspin(r) and cspin(q) == cspin(s) else 0)
                    e = (g[csp(p), csp(q), csp(s), csp(r)]
                         if cspin(p) == cspin(s) and cspin(q) == cspin(r) else 0)
                    v[p, q, r, s] = c - e
    t2ab = cc.t2

    def so_t2(sa, sb, si, sj, A, B, I, J):
        if sa == si and sb == sj and not (sa == sj and sb == si):
            return t2ab[I, J, A, B]
        if sa == sj and sb == si and not (sa == si and sb == sj):
            return -t2ab[I, J, B, A]
        if sa == sb == si == sj:
            return t2ab[I, J, A, B] - t2ab[I, J, B, A]
        return 0.0

    t2 = np.zeros((nv, nv, no, no))
    for a in range(nv):
        for b in range(nv):
            for i in range(no):
                for j in range(no):
                    t2[a, b, i, j] = so_t2(a % 2, b % 2, i % 2, j % 2,
                                           a // 2, b // 2, i // 2, j // 2)
    t1 = np.zeros((nv, no))
    for a in range(nv):
        for i in range(no):
            if a % 2 == i % 2:
                t1[a, i] = cc.t1[i // 2, a // 2]
    f = np.zeros((n, n))
    for p in range(n):
        f[p, p] = mf.mo_energy[csp(p)]
    return {"t1": t1, "t2": t2, "v": v, "f": f}, no, nv, mf, cc


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S1AntisymIntegrationTests(unittest.TestCase):
    """S1.2' / S2.2d-2 blocker resolution: `ucc_integrate_term_antisym`
    reproduces the GCC residual on REAL antisymmetric integrals.

    The plain `ucc_integrate_term` (block filter) drops the forbidden-block
    cases, which is exact only for spin-conserving tensors; on real
    antisymmetric integrals it fails the S2.1 identity (~0.06). The antisym
    variant re-expresses each forbidden factor into its allowed block via
    bra/ket swaps with sign, and matches GCC to ~1e-16 -- singles and doubles,
    on the tensors ccgen's GCC actually consumes.
    """

    def _check(self, method):
        import numpy as np
        from ccgen.spin import ucc_integrate_term_antisym
        from ccgen.tests.residual_eval import residual_einsum

        tn, no, nv, mf, cc = _real_antisym_tensors()
        n = no + nv
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))
        eqs = generate_cc_equations(method)

        # The gate is the IDENTITY (antisym integration == GCC slice), which holds
        # for any amplitudes -- not that the residual vanishes (CCD at CCSD amps
        # does not). CCSD full-residual vanishing is a separate check below.
        Rg_d = sum(residual_einsum(t, no, nv, tensors=tn) for t in eqs["doubles"])

        # doubles abab: antisym integration == GCC slice
        Rb = Rg_d[np.ix_(ve, vo, oe, oo)]
        acc = np.zeros_like(Rb)
        for t in eqs["doubles"]:
            for st in ucc_integrate_term_antisym(t, {"a": "a", "b": "b",
                                                     "i": "a", "j": "b"}):
                acc += _eval_spinterm(st, tn, no, n, ["a", "b", "i", "j"])
        self.assertLess(np.max(np.abs(acc - Rb)), 1e-10,
                        f"{method} doubles antisym != GCC abab")

        # singles aa: same
        if "singles" in eqs:
            Rs = sum(residual_einsum(t, no, nv, tensors=tn)
                     for t in eqs["singles"])
            Rsa = Rs[np.ix_(ve, oe)]
            accs = np.zeros_like(Rsa)
            for t in eqs["singles"]:
                for st in ucc_integrate_term_antisym(t, {"a": "a", "i": "a"}):
                    accs += _eval_spinterm(st, tn, no, n, ["a", "i"])
            self.assertLess(np.max(np.abs(accs - Rsa)), 1e-10,
                            f"{method} singles antisym != GCC aa")

    def test_ccsd_antisym_matches_gcc(self):
        self._check("ccsd")

    def test_ccd_antisym_matches_gcc(self):
        self._check("ccd")

    def test_gcc_vanishes_at_converged_ccsd_amps(self):
        # validates the real-antisym tensor build: ccgen's GCC CCSD residual
        # (both blocks) is ~0 at PySCF's converged RCCSD amps, and the energy
        # matches cc.e_corr. If this fails the oracle tensors are wrong, not the
        # integration.
        import numpy as np
        from ccgen.tests.residual_eval import residual_einsum
        tn, no, nv, mf, cc = _real_antisym_tensors()
        eqs = generate_cc_equations("ccsd")
        for block in ("singles", "doubles"):
            R = sum(residual_einsum(t, no, nv, tensors=tn) for t in eqs[block])
            self.assertLess(np.max(np.abs(R)), 1e-5,
                            f"GCC {block} residual nonzero at converged amps")
        E = sum(float(residual_einsum(t, no, nv, tensors=tn))
                for t in eqs["energy"])
        self.assertLess(abs(E - cc.e_corr), 1e-6, "GCC energy != cc.e_corr")


def _rcc_pipeline(method, block):
    """The full S2.2a->d spatial RCC residual (or energy) for one manifold, built
    on the antisymmetry-correct integration: antisym-integrate the external block,
    then canonicalize -> collapse amplitudes -> collapse integrals -> merge.
    ``block`` is "energy", "singles", or "doubles"; the collapse steps are
    block-agnostic (they act on t2[aaaa]/v[aaaa] factors regardless of the
    residual's own externals). "energy" has an empty external block (a fully
    contracted scalar). Returns the merged spatial SpinTerms."""
    from ccgen.spin import (ucc_integrate_term_antisym, canonicalize_spin_blocks,
                            collapse_amplitudes, collapse_integrals, merge_terms)
    ext = {"energy": {},
           "singles": {"a": "a", "i": "a"},
           "doubles": {"a": "a", "b": "b", "i": "a", "j": "b"}}[block]
    manifold = []
    for t in generate_cc_equations(method)[block]:
        manifold.extend(ucc_integrate_term_antisym(t, ext))
    canon = [canonicalize_spin_blocks(st) for st in manifold]
    amp = [c for st in canon for c in collapse_amplitudes(st)]
    coll = [c for st in amp for c in collapse_integrals(st)]
    return merge_terms(coll, set(ext))


def _rcc_doubles_pipeline(method):
    """Back-compat alias: the doubles RCC residual pipeline."""
    return _rcc_pipeline(method, "doubles")


def _rcc_pipeline_filtered(method, block):
    """PySCF-free variant of the RCC pipeline for the SYNTHETIC spin-conserving
    fixture: uses the filtered `ucc_manifold` (valid on that fixture) instead of
    the antisym integration. Same canonicalize -> collapse -> merge tail. Used by
    tests that only need the merged spatial term STRUCTURE (S3 bridge/lowering),
    not real-integral physics."""
    from ccgen.spin import (ucc_manifold, canonicalize_spin_blocks,
                            collapse_amplitudes, collapse_integrals, merge_terms)
    from ccgen.tensors import Tensor
    if block == "singles":
        tpl = Tensor("R1", (make_vir("a"), make_occ("i")))
        ext = {"a", "i"}
    else:
        tpl = Tensor("R2", (make_vir("a"), make_vir("b"),
                            make_occ("i"), make_occ("j")))
        ext = {"a", "b", "i", "j"}
    man = ucc_manifold(generate_cc_equations(method)[block], tpl)
    tag = "".join(dict(a="a", b="b", i="a", j="b")[nm]
                  for nm in [x.name for x in tpl.indices])
    canon = [canonicalize_spin_blocks(st) for st in man[tag]]
    amp = [c for st in canon for c in collapse_amplitudes(st)]
    coll = [c for st in amp for c in collapse_integrals(st)]
    return merge_terms(coll, ext)


class S30BridgeTests(unittest.TestCase):
    """S3.0: `spinterm_to_algebraterm` bridges a spatial RCC SpinTerm to the
    AlgebraTerm the emit path consumes.

    PySCF-free (uses the filtered synthetic-fixture pipeline -- the bridge is a
    pure structural transform, independent of the tensor values). Gate: every
    converted AlgebraTerm preserves the coefficient, the factor tensor names +
    per-factor spatial index identities, and the free/summed split matches the
    SpinTerm's externals. This is the "algebra unchanged, wrapper differs"
    contract; downstream (S3.1 lowering, S3.2 emit) rides on it.
    """

    def _check(self, method, block, externals, res_indices):
        from ccgen.spin import spinterm_to_algebraterm
        merged = _rcc_pipeline_filtered(method, block)
        self.assertTrue(merged, f"{method} {block}: empty pipeline")
        for st in merged:
            at = spinterm_to_algebraterm(st, externals)
            # coefficient preserved
            self.assertEqual(at.coeff, st.coeff)
            # factor names + spatial index names, in order, preserved
            self.assertEqual([f.name for f in at.factors],
                             [f.name for f in st.factors])
            for af, sf in zip(at.factors, st.factors):
                self.assertEqual([x.name for x in af.indices],
                                 [si.name for si in sf.indices])
                self.assertEqual([x.space for x in af.indices],
                                 [si.base.space for si in sf.indices])
            # free = externals present; summed = the rest; disjoint, complete
            free = {x.name for x in at.free_indices}
            summed = {x.name for x in at.summed_indices}
            allnames = {si.name for f in st.factors for si in f.indices}
            self.assertEqual(free, allnames & set(externals))
            self.assertEqual(summed, allnames - set(externals))
            self.assertEqual(free & summed, set())
            # connected flag set (RCC residual terms are connected)
            self.assertTrue(at.connected)

    def test_ccd_doubles_bridge(self):
        self._check("ccd", "doubles", {"a", "b", "i", "j"}, ["a", "b", "i", "j"])

    def test_ccsd_doubles_bridge(self):
        self._check("ccsd", "doubles", {"a", "b", "i", "j"}, ["a", "b", "i", "j"])

    def test_ccsd_singles_bridge(self):
        self._check("ccsd", "singles", {"a", "i"}, ["a", "i"])


class S31LoweringTests(unittest.TestCase):
    """S3.1: the bridged RCC AlgebraTerms lower cleanly through
    `lower_term_restricted_closed_shell` to emit-ready spatial IR.

    Follows the house style of the existing lowering regressions (structural, not
    numeric — the numeric proof is the S3.2 energy gate). Confirms every bridged
    RCC term lowers without error, the canonical free indices carry the right
    occ/vir spaces for the manifold, every factor gets a valid block signature
    (only o/v glyphs), amplitude factors land in the expected block, and `v`
    factors are mapped to a canonical ERI block. PySCF-free.
    """

    _ERI_BLOCKS = {"oooo", "ooov", "oovv", "ovov", "ovvo", "ovvv", "vvvv"}

    # each amplitude tensor lands in one canonical block regardless of which
    # residual it appears in (a singles residual still contains t2 factors).
    _AMP_BLOCK = {"t1": "ov", "t2": "oovv"}

    def _check(self, method, block, free_spaces):
        from ccgen.spin import spinterm_to_algebraterm
        from ccgen.lowering import lower_term_restricted_closed_shell
        externals = {"doubles": {"a", "b", "i", "j"},
                     "singles": {"a", "i"}}[block]
        merged = _rcc_pipeline_filtered(method, block)
        self.assertTrue(merged)
        for st in merged:
            at = spinterm_to_algebraterm(st, externals)
            lt = lower_term_restricted_closed_shell(at, block)
            # coefficient carried through
            self.assertEqual(lt.coeff, st.coeff)
            # canonical free indices have the manifold's occ/vir signature
            self.assertEqual(
                tuple(i.space for i in lt.canonical_free_indices), free_spaces)
            for f in lt.factors:
                # block signature is only o/v glyphs, length = factor rank
                self.assertTrue(set(f.spatial_block) <= {"o", "v"}, f.spatial_block)
                self.assertEqual(len(f.spatial_block), len(f.source.indices))
                self.assertIn(f.phase, (1, -1))
                if f.name in self._AMP_BLOCK:
                    self.assertEqual(f.spatial_block, self._AMP_BLOCK[f.name],
                                     f"{f.name} block {f.spatial_block}")
                if f.name == "v":
                    self.assertIn(f.spatial_block, self._ERI_BLOCKS,
                                  f"v block {f.spatial_block} not canonical")

    def test_ccd_doubles_lowering(self):
        self._check("ccd", "doubles", ("occ", "occ", "vir", "vir"))

    def test_ccsd_doubles_lowering(self):
        self._check("ccsd", "doubles", ("occ", "occ", "vir", "vir"))

    def test_ccsd_singles_lowering(self):
        self._check("ccsd", "singles", ("occ", "vir"))


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S22dEndToEndTests(unittest.TestCase):
    """S2.2d-2: the whole S2.2a->d collapse, built on the antisym integration,
    reproduces the RCC residual on REAL integrals end to end.

    This is the payoff the block-filter fix unblocked. The merged spatial RCC
    doubles (and singles) residual, evaluated on the real antisymmetric
    water/STO-3G tensors at PySCF's converged RCCSD amplitudes, vanishes (== the
    GCC residual there, ~1e-7) -- so the collapsed equation IS the RCC residual.
    The merged doubles coefficients carry the RCC `2J - K` combinations
    (|coeff| in {2, 4}).
    """

    def test_ccsd_rcc_residual_vanishes_at_converged_amps(self):
        import numpy as np
        tn, no, nv, mf, cc = _real_antisym_tensors()
        n = no + nv
        merged = _rcc_doubles_pipeline("ccsd")
        acc = np.zeros((nv // 2, nv // 2, no // 2, no // 2))
        for st in merged:
            acc += _eval_spinterm(st, tn, no, n, ["a", "b", "i", "j"])
        self.assertLess(np.max(np.abs(acc)), 1e-6,
                        "merged RCC doubles residual should vanish at converged "
                        f"amps, got {np.max(np.abs(acc))}")
        # the RCC 2J-K coefficients are present
        self.assertTrue(any(abs(t.coeff) > 1 for t in merged),
                        "no 2J-K combination in merged RCC coefficients")

    def test_ccsd_rcc_matches_gcc_slice(self):
        # stronger than "vanishes": the merged RCC residual equals the GCC abab
        # slice for ANY amplitudes (not just at the solution). Perturb the amps
        # off the solution and require the identity still holds.
        import numpy as np
        from ccgen.tests.residual_eval import residual_einsum
        tn, no, nv, mf, cc = _real_antisym_tensors()
        n = no + nv
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))
        Rb = sum(residual_einsum(t, no, nv, tensors=tn)
                 for t in generate_cc_equations("ccsd")["doubles"]
                 )[np.ix_(ve, vo, oe, oo)]
        merged = _rcc_doubles_pipeline("ccsd")
        acc = np.zeros_like(Rb)
        for st in merged:
            acc += _eval_spinterm(st, tn, no, n, ["a", "b", "i", "j"])
        self.assertLess(np.max(np.abs(acc - Rb)), 1e-10,
                        f"merged RCC != GCC abab: {np.max(np.abs(acc - Rb))}")

    def test_ccsd_rcc_singles_matches_gcc_slice(self):
        # the singles spatial residual through the same pipeline: merged RCC
        # singles == GCC aa slice, and vanishes at the converged amps.
        import numpy as np
        from ccgen.tests.residual_eval import residual_einsum
        tn, no, nv, mf, cc = _real_antisym_tensors()
        n = no + nv
        ve, oe = list(range(0, nv, 2)), list(range(0, no, 2))
        Rb = sum(residual_einsum(t, no, nv, tensors=tn)
                 for t in generate_cc_equations("ccsd")["singles"]
                 )[np.ix_(ve, oe)]
        merged = _rcc_pipeline("ccsd", "singles")
        acc = np.zeros_like(Rb)
        for st in merged:
            acc += _eval_spinterm(st, tn, no, n, ["a", "i"])
        self.assertLess(np.max(np.abs(acc - Rb)), 1e-10,
                        f"merged RCC singles != GCC aa: {np.max(np.abs(acc - Rb))}")
        # vanishes at converged amps (Rb itself is the residual at the solution)
        self.assertLess(np.max(np.abs(acc)), 1e-6,
                        "merged RCC singles residual should vanish at converged amps")


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S32EnergyTests(unittest.TestCase):
    """S3.2 (numeric half): the spin-adapted RCC ENERGY expression reaches the
    PySCF RCCSD correlation energy.

    The energy manifold (`E = f_ia t1 + 1/4 t2 v + 1/2 t1 t1 v`) runs through the
    same S2.2a→d pipeline with an EMPTY external block (a fully contracted
    scalar). Evaluated on the real antisymmetric water/STO-3G tensors at PySCF's
    converged RCCSD amplitudes, the merged RCC energy equals `cc.e_corr` to
    ~1e-8. This is the convention-robust "evaluate at PySCF amps" scalar gate --
    it, together with the singles+doubles residuals vanishing (S22dEndToEndTests),
    is the numeric end-to-end proof of the whole adapted RCC equation set. The
    remaining S3.2 work is C++ EMISSION (route the adapted AlgebraTerms through
    `emit_planck_translation_unit` and compile), not this numeric check.
    """

    def test_ccsd_rcc_energy_matches_pyscf(self):
        tn, no, nv, mf, cc = _real_antisym_tensors()
        n = no + nv
        merged = _rcc_pipeline("ccsd", "energy")
        E = sum(float(_eval_spinterm(st, tn, no, n, [])) for st in merged)
        self.assertLess(abs(E - cc.e_corr), 1e-6,
                        f"adapted RCC energy {E} != PySCF e_corr {cc.e_corr}")


def _uccsdt_t3_blocks(atom="N 0 0 0; N 0 0 1.3", basis="sto-3g"):
    """S4a.0b oracle: converged UCCSDT `t3` spin blocks on a closed-shell
    RHF->UHF-converted reference. Returns (aaa, aab, bba, bbb, nocc, nvir, cc).

    The RHF->UHF convert is load-bearing: a fresh scf.UHF(N2) at 1.3 A relaxes to
    a symmetry-broken solution (aaa != bbb), so we lift the RHF orbitals instead
    (aaa == bbb to ~1e-18). Blocks (from tamps_tri2full_uhf): aaa/bbb are
    [i,j,k,a,b,c] and genuinely antisym; aab/bba are [i,j,a,b,k,c] (block layout
    from the pyscf.cc.uccsdt docstring: block[2] is bba = 2-beta-1-alpha, NOT
    abb), antisym in the occ-pair (0,1) and vir-pair (2,3) separately."""
    from pyscf.cc import uccsdt
    mol = gto.M(atom=atom, basis=basis, spin=0, verbose=0)
    mol.cart = True
    mf = scf.addons.convert_to_uhf(scf.RHF(mol).run())
    cc = uccsdt.UCCSDT(mf)
    cc.conv_tol = 1e-10
    cc.kernel()
    aaa, aab, bba, bbb = uccsdt.tamps_tri2full_uhf(cc, cc.t3)
    nocc = int(cc.nocc[0])
    nvir = aaa.shape[3]
    return aaa, aab, bba, bbb, nocc, nvir, cc


def _t3so_canonical_read(a, b, c, i, j, k, blocks):
    """map.1: the closed-form spin-orbital t3so[a,b,c,i,j,k] read for a CANONICAL
    entry -- one where each ccgen line (vir slot k / occ slot k: (a,i),(b,j),(c,k))
    is spin-conserving. Returns the UCCSDT block value at the entry's spatial
    indices. Canonical means the entry is already in the block's stored slot order
    for its spin pattern; the full line-swap antisymmetry that reorders lines is
    map.2. Returns None for a non-spin-conserving line (0 by construction) and
    "MIXED-ORDER" for a spin pattern needing a line reorder (deferred to map.2)."""
    aaa, aab, bba, bbb = blocks
    sa, sb, sc = a % 2, b % 2, c % 2
    si, sj, sk = i % 2, j % 2, k % 2
    if not (sa == si and sb == sj and sc == sk):
        return None
    A, B, C = a // 2, b // 2, c // 2
    I, J, K = i // 2, j // 2, k // 2
    sp = (sa, sb, sc)
    if sp == (0, 0, 0):
        return aaa[I, J, K, A, B, C]
    if sp == (1, 1, 1):
        return bbb[I, J, K, A, B, C]
    if sp == (0, 0, 1):          # two alpha lines then one beta line
        return aab[I, J, A, B, K, C]
    if sp == (1, 1, 0):          # two beta lines then one alpha line
        return bba[I, J, A, B, K, C]
    return "MIXED-ORDER"


def _line_parity(order):
    """Sign of the permutation `order` (order[k] = source slot placed at k)."""
    seen = [False] * len(order)
    par = 1
    for start in range(len(order)):
        if seen[start]:
            continue
        j, length = start, 0
        while not seen[j]:
            seen[j] = True
            j = order[j]
            length += 1
        if length % 2 == 0:
            par = -par
    return par


def _read_ascending(virs, occs, spins, blocks):
    """Read a SPIN-CONSERVING arrangement (line k has spin == spins[k], and spins
    is ascending) from the correct PySCF UCCSDT block. The one non-face-value case
    is (0,1,1): PySCF stores that multiset as `bba` in majority-first order
    (1,1,0), so the ascending (0,1,1) lines are reordered to (1,1,0) with the
    resulting line-permutation parity."""
    aaa, aab, bba, bbb = blocks
    A, B, C = (x // 2 for x in virs)
    I, J, K = (x // 2 for x in occs)
    if spins == (0, 0, 0):
        return aaa[I, J, K, A, B, C]
    if spins == (1, 1, 1):
        return bbb[I, J, K, A, B, C]
    if spins == (0, 0, 1):
        return aab[I, J, A, B, K, C]
    if spins == (0, 1, 1):
        order = [1, 2, 0]                       # (alpha,beta,beta) -> (beta,beta,alpha)
        v2 = [virs[o] for o in order]
        o2 = [occs[o] for o in order]
        A, B, C = (x // 2 for x in v2)
        I, J, K = (x // 2 for x in o2)
        return _line_parity(order) * bba[I, J, A, B, K, C]
    return None


def _t3so_read(a, b, c, i, j, k, blocks):
    """map.2: the general spin-orbital t3so[a,b,c,i,j,k] read, valid for ANY line
    ordering. The GCC t3 is antisymmetric INDEPENDENTLY within the bra group
    (virtuals a,b,c) and the ket group (occupieds i,j,k) -- exactly the convention
    production `spin.py::_antisym_to_allowed` consumes, and the one PySCF's raw
    `aaa` block satisfies (antisym under a lone occ-swap and a lone vir-swap;
    SYMMETRIC under a joint line swap). It is NOT antisymmetric under a physical
    line swap -- that earlier scoping was a misconception (a joint bra+ket swap is
    (-1)(-1) = +1). So: sort the bra by spin and the ket by spin independently
    (sign = product of the two parities), landing on a spin-conserving ascending
    arrangement, then read the correct block."""
    virs, occs = [a, b, c], [i, j, k]
    bs = [x % 2 for x in virs]
    ks = [x % 2 for x in occs]
    if sorted(bs) != sorted(ks):
        return 0.0                              # no line can conserve spin
    bord = sorted(range(3), key=lambda t: bs[t])
    kord = sorted(range(3), key=lambda t: ks[t])
    sign = _line_parity(bord) * _line_parity(kord)
    v2 = [virs[o] for o in bord]
    o2 = [occs[o] for o in kord]
    spins = tuple(x % 2 for x in v2)             # ascending, spin-conserving
    return sign * _read_ascending(v2, o2, spins, blocks)


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S4aMap1CanonicalReadTests(unittest.TestCase):
    """map.1: pin the canonical spin-orbital t3so read against PySCF's UCCSDT
    stored blocks. On CANONICAL entries (each line spin-conserving, in the block's
    stored slot order) t3so == block for every spin pattern. This inherits the
    single occ-pair / vir-pair antisymmetry from the blocks; the physical
    line-swap antisymmetry that reorders lines is map.2."""

    @classmethod
    def setUpClass(cls):
        cls.aaa, cls.aab, cls.bba, cls.bbb, cls.nocc, cls.nvir, cls.cc = \
            _uccsdt_t3_blocks()
        cls.blocks = (cls.aaa, cls.aab, cls.bba, cls.bbb)

    def _sweep(self, spins, block, layout):
        """spins = per-line (vir/occ) spins; layout maps (I,J,K,A,B,C)->block idx."""
        import numpy as np
        no, nv = self.nocc, self.nvir
        maxerr = 0.0
        for I in range(no):
            for J in range(no):
                for K in range(no):
                    for A in range(nv):
                        for B in range(nv):
                            for C in range(nv):
                                a = 2 * A + spins[0]
                                b = 2 * B + spins[1]
                                c = 2 * C + spins[2]
                                i = 2 * I + spins[0]
                                j = 2 * J + spins[1]
                                k = 2 * K + spins[2]
                                got = _t3so_canonical_read(a, b, c, i, j, k,
                                                           self.blocks)
                                ref = block[layout(I, J, K, A, B, C)]
                                maxerr = max(maxerr, abs(got - ref))
        return maxerr

    def test_aaa_canonical(self):
        err = self._sweep((0, 0, 0), self.aaa,
                          lambda I, J, K, A, B, C: (I, J, K, A, B, C))
        self.assertLess(err, 1e-14, f"aaa canonical read off {err}")

    def test_bbb_canonical(self):
        err = self._sweep((1, 1, 1), self.bbb,
                          lambda I, J, K, A, B, C: (I, J, K, A, B, C))
        self.assertLess(err, 1e-14, f"bbb canonical read off {err}")

    def test_aab_canonical(self):
        # aab layout [i,j,a,b,k,c]: two alpha lines (I,A),(J,B) then beta (K,C)
        err = self._sweep((0, 0, 1), self.aab,
                          lambda I, J, K, A, B, C: (I, J, A, B, K, C))
        self.assertLess(err, 1e-14, f"aab canonical read off {err}")

    def test_bba_canonical(self):
        # bba layout [i,j,a,b,k,c]: two beta lines then one alpha line
        err = self._sweep((1, 1, 0), self.bba,
                          lambda I, J, K, A, B, C: (I, J, A, B, K, C))
        self.assertLess(err, 1e-14, f"bba canonical read off {err}")

    def test_closed_shell_fixture(self):
        # the RHF->UHF convert must give a genuinely closed-shell t3 (aaa == bbb)
        import numpy as np
        self.assertLess(np.max(np.abs(self.aaa - self.bbb)), 1e-12,
                        "fixture not closed-shell (aaa != bbb)")


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S4aMap2GeneralReadTests(unittest.TestCase):
    """map.2: the general spin-orbital t3so read (any line order) is the correct
    GCC-antisymmetric t3.

    FINDING (reshapes the doc's map.2 gate): the invariant is INDEPENDENT bra/ket
    antisymmetry -- antisym under any permutation of the virtuals (a,b,c) and,
    separately, any permutation of the occupieds (i,j,k) -- exactly what
    production `_antisym_to_allowed` consumes. The doc scoped map.2 as "the three
    physical LINE-swaps all ~1e-12", but a valid t3 is SYMMETRIC under a joint
    line swap (a joint bra+ket transposition is (-1)(-1)=+1). PySCF's raw `aaa`
    block confirms this directly (test_ground_truth_block_symmetry). So the
    line-swap gate was a misconception; the real gate is bra/ket antisymmetry.
    map.2's read also reproduces map.1's canonical block reads exactly."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.aaa, cls.aab, cls.bba, cls.bbb, cls.nocc, cls.nvir, cls.cc = \
            _uccsdt_t3_blocks()
        cls.blocks = (cls.aaa, cls.aab, cls.bba, cls.bbb)
        no, nv = 2 * cls.nocc, 2 * cls.nvir
        t3 = np.zeros((nv, nv, nv, no, no, no))
        for a in range(nv):
            for b in range(nv):
                for c in range(nv):
                    for i in range(no):
                        for j in range(no):
                            for k in range(no):
                                t3[a, b, c, i, j, k] = _t3so_read(
                                    a, b, c, i, j, k, cls.blocks)
        cls.t3so = t3

    def test_ground_truth_block_symmetry(self):
        # The raw antisym block is symmetric under a JOINT line swap and antisym
        # under lone occ / lone vir swaps -- this is WHY the line-swap gate is
        # wrong and the bra/ket gate is right.
        import numpy as np
        aaa = self.aaa                                    # [i,j,k,a,b,c]
        self.assertLess(np.abs(aaa - aaa.transpose(1, 0, 2, 4, 3, 5)).max(),
                        1e-12, "aaa NOT symmetric under joint line swap")
        self.assertLess(np.abs(aaa + aaa.transpose(1, 0, 2, 3, 4, 5)).max(),
                        1e-12, "aaa NOT antisym under lone occ swap")
        self.assertLess(np.abs(aaa + aaa.transpose(0, 1, 2, 4, 3, 5)).max(),
                        1e-12, "aaa NOT antisym under lone vir swap")

    def test_bra_ket_independent_antisymmetry(self):
        import numpy as np
        t = self.t3so
        for name, tr in [("vir a<->b", (1, 0, 2, 3, 4, 5)),
                         ("vir b<->c", (0, 2, 1, 3, 4, 5)),
                         ("occ i<->j", (0, 1, 2, 4, 3, 5)),
                         ("occ j<->k", (0, 1, 2, 3, 5, 4))]:
            err = np.abs(t + t.transpose(tr)).max()
            self.assertLess(err, 1e-11, f"t3so not antisym under {name}: {err}")

    def test_symmetric_under_joint_line_swap(self):
        # the property the misconceived line-swap gate demanded be ANTISYM; it is
        # SYMMETRIC. Pinned so the finding does not silently regress.
        import numpy as np
        t = self.t3so
        err = np.abs(t - t.transpose(1, 0, 2, 4, 3, 5)).max()
        self.assertLess(err, 1e-11, f"joint line swap not symmetric: {err}")

    def test_matches_canonical_read(self):
        # map.2 (general) reproduces map.1 (canonical block reads) where map.1 is
        # defined -- the aaa and aab canonical slots.
        import numpy as np
        no, nv = self.nocc, self.nvir
        wa = wb = 0.0
        for I in range(no):
            for J in range(no):
                for K in range(no):
                    for A in range(nv):
                        for B in range(nv):
                            for C in range(nv):
                                wa = max(wa, abs(
                                    self.t3so[2*A, 2*B, 2*C, 2*I, 2*J, 2*K]
                                    - self.aaa[I, J, K, A, B, C]))
        for I in range(no):
            for J in range(no):
                for A in range(nv):
                    for B in range(nv):
                        for K in range(no):
                            for C in range(nv):
                                wb = max(wb, abs(
                                    self.t3so[2*A, 2*B, 2*C+1, 2*I, 2*J, 2*K+1]
                                    - self.aab[I, J, A, B, K, C]))
        self.assertLess(wa, 1e-14, f"aaa canonical mismatch {wa}")
        self.assertLess(wb, 1e-14, f"aab canonical mismatch {wb}")


def _uccsdt_so_tensors(atom="N 0 0 0; N 0 0 1.3", basis="sto-3g"):
    """map.3: build the spin-orbital CC tensors (t1, t2, t3, v, f) from a converged
    UCCSDT closed-shell reference, in the residual_eval layout (amplitudes
    vir-first: t1 [a,i], t2 [a,b,i,j], t3 [a,b,c,i,j,k]; v = <pq||rs>; f diagonal).

    The load-bearing layout fact (this note's whole point): UCCSDT stores t2ab as
    **[i,a,j,b]** (nocca,nvira,noccb,nvirb) -- unlike rccsd's AND pyscf.cc.uccsd's
    [i,j,a,b]. The mixed-spin so_t2 fill must index it as [i,a,j,b]; t2aa/t2bb are
    [i,j,a,b]. t3 comes from the map.2 read (`_t3so_read`).

    Returns (tensors, no, nv, cc)."""
    import numpy as np
    from pyscf import ao2mo
    from pyscf.cc import uccsdt
    mol = gto.M(atom=atom, basis=basis, spin=0, verbose=0)
    mol.cart = True
    rhf = scf.RHF(mol).run()
    mf = scf.addons.convert_to_uhf(rhf)
    cc = uccsdt.UCCSDT(mf)
    cc.conv_tol = 1e-12
    cc.max_cycle = 200
    cc.kernel()
    nocc = int(cc.nocc[0])
    nmo = rhf.mo_coeff.shape[1]
    nvir = nmo - nocc
    no, nv, n = 2 * nocc, 2 * nvir, 2 * nmo

    def csp(p):
        return (p // 2) if p < no else nocc + ((p - no) // 2)

    def cspin(p):
        return (p % 2) if p < no else ((p - no) % 2)

    eri = ao2mo.kernel(mol, rhf.mo_coeff, aosym="s1").reshape(nmo, nmo, nmo, nmo)
    g = eri.transpose(0, 2, 1, 3)                       # physicist <pq|rs>
    v = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    c = (g[csp(p), csp(q), csp(r), csp(s)]
                         if cspin(p) == cspin(r) and cspin(q) == cspin(s) else 0)
                    e = (g[csp(p), csp(q), csp(s), csp(r)]
                         if cspin(p) == cspin(s) and cspin(q) == cspin(r) else 0)
                    v[p, q, r, s] = c - e
    f = np.zeros((n, n))
    for p in range(n):
        f[p, p] = rhf.mo_energy[csp(p)]

    t1a = cc.t1[0]                                      # [i,a]
    t1 = np.zeros((nv, no))
    for a in range(nv):
        for i in range(no):
            if a % 2 == i % 2:
                t1[a, i] = t1a[i // 2, a // 2]

    t2aa, t2ab, t2bb = cc.t2                            # aa/bb [i,j,a,b]; ab [i,a,j,b]
    t2 = np.zeros((nv, nv, no, no))
    for a in range(nv):
        for b in range(nv):
            for i in range(no):
                for j in range(no):
                    sa, sb, si, sj = a % 2, b % 2, i % 2, j % 2
                    A, B, I, J = a // 2, b // 2, i // 2, j // 2
                    if sa == si and sb == sj:          # direct: line0=(a,i), line1=(b,j)
                        if sa == sb == 0:
                            t2[a, b, i, j] = t2aa[I, J, A, B]
                        elif sa == sb == 1:
                            t2[a, b, i, j] = t2bb[I, J, A, B]
                        elif sa == 0:                  # abab -- t2ab is [i,a,j,b]
                            t2[a, b, i, j] = t2ab[I, A, J, B]
                        else:                          # baba
                            t2[a, b, i, j] = t2ab[J, B, I, A]
                    elif sa == sj and sb == si:        # exchange fill (abba / baab)
                        if sa == 0:
                            t2[a, b, i, j] = -t2ab[J, A, I, B]
                        else:
                            t2[a, b, i, j] = -t2ab[I, B, J, A]

    aaa, aab, bba, bbb = uccsdt.tamps_tri2full_uhf(cc, cc.t3)
    blocks = (aaa, aab, bba, bbb)
    t3 = np.zeros((nv, nv, nv, no, no, no))
    for a in range(nv):
        for b in range(nv):
            for c in range(nv):
                for i in range(no):
                    for j in range(no):
                        for k in range(no):
                            t3[a, b, c, i, j, k] = _t3so_read(a, b, c, i, j, k,
                                                              blocks)
    return {"t1": t1, "t2": t2, "t3": t3, "v": v, "f": f}, no, nv, cc


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S4aMap3T2LayoutTests(unittest.TestCase):
    """map.3: fix the fixture's t2ab layout ([i,a,j,b], NOT [i,j,a,b]).

    Proven correct (this is what map.3 delivers): the assembled spin-orbital t2 is
    bit-identical to the validated so_t2 fill (`_real_antisym_tensors`) rebuilt
    from the transposed t2ab, and the GCC energy at these amps hits PySCF's
    e_corr. The full doubles/triples residual < 1e-7 gate is NOT here -- with every
    base tensor proven and t3 round-tripping to its blocks exactly, the residual
    isolates a remaining t3 contraction-convention question, which is S4a.0c's job
    (this note explicitly separates the layout fix from the residual gate)."""

    @classmethod
    def setUpClass(cls):
        cls.tn, cls.no, cls.nv, cls.cc = _uccsdt_so_tensors()

    def test_t2_matches_so_t2_reference(self):
        # Rebuild t2 via the proven so_t2 fill from the [i,j,a,b]-transposed t2ab
        # and require bit-identity -- pins the [i,a,j,b] indexing.
        import numpy as np
        _, t2ab, _ = self.cc.t2
        T = t2ab.transpose(0, 2, 1, 3)                  # [i,a,j,b] -> [i,j,a,b]
        no, nv = self.no, self.nv

        def so(sa, sb, si, sj, A, B, I, J):
            if sa == si and sb == sj and not (sa == sj and sb == si):
                return T[I, J, A, B]
            if sa == sj and sb == si and not (sa == si and sb == sj):
                return -T[I, J, B, A]
            if sa == sb == si == sj:
                return T[I, J, A, B] - T[I, J, B, A]
            return 0.0

        ref = np.zeros((nv, nv, no, no))
        for a in range(nv):
            for b in range(nv):
                for i in range(no):
                    for j in range(no):
                        ref[a, b, i, j] = so(a % 2, b % 2, i % 2, j % 2,
                                              a // 2, b // 2, i // 2, j // 2)
        self.assertLess(np.max(np.abs(self.tn["t2"] - ref)), 1e-12,
                        "t2 fill disagrees with so_t2 reference (t2ab layout bug)")

    def test_t2_antisymmetric(self):
        import numpy as np
        t2 = self.tn["t2"]
        self.assertLess(np.abs(t2 + t2.transpose(1, 0, 2, 3)).max(), 1e-12)
        self.assertLess(np.abs(t2 + t2.transpose(0, 1, 3, 2)).max(), 1e-12)

    def test_gcc_energy_at_amps(self):
        # Fully-contracted scalar -> convention-robust; breaks on any t1/t2/v/f bug.
        from ccgen.generate import generate_cc_equations
        from ccgen.tests.residual_eval import residual_einsum
        eqs = generate_cc_equations("ccsd")
        E = sum(float(residual_einsum(t, self.no, self.nv, tensors=self.tn))
                for t in eqs["energy"])
        self.assertLess(abs(E - self.cc.e_corr), 1e-8,
                        f"GCC energy {E} != UCCSDT e_corr {self.cc.e_corr}")


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S4a0cTriplesResidualTests(unittest.TestCase):
    """S4a.0c: the numeric gate the S4 structural gate defers. With the correct
    spin-orbital t3 (map.2 read) and the full UCCSDT fixture (map.3), ccgen's GCC
    CCSDT residual VANISHES at PySCF's converged UCCSDT amplitudes on the
    strong-correlation N2/STO-3G fixture (|t3| ~ 0.03, ~20x LiH -- large enough to
    expose any t3 error, unlike the LiH the earlier scoping showed was too weak).

    This is the oracle S4a.0a identified: the residual vanishing at the converged
    UCCSDT amps (a self-consistent full t1/t2/t3 set), NOT an RCCSDT t3full
    inversion. It validates BOTH generation engines (wick and diagram) and pins
    the whole map.1->map.3 t3 assembly end-to-end: singles/doubles/triples all
    < 1e-7 and the energy == e_corr."""

    @classmethod
    def setUpClass(cls):
        cls.tn, cls.no, cls.nv, cls.cc = _uccsdt_so_tensors()

    def _residuals(self, engine):
        import numpy as np
        from ccgen.generate import generate_cc_equations
        from ccgen.tests.residual_eval import residual_einsum
        eqs = generate_cc_equations("ccsdt", engine=engine)
        out = {}
        for name in ("singles", "doubles", "triples"):
            R = None
            for term in eqs[name]:
                r = residual_einsum(term, self.no, self.nv, tensors=self.tn)
                R = r if R is None else R + r
            out[name] = np.abs(R).max()
        out["energy"] = abs(
            sum(float(residual_einsum(t, self.no, self.nv, tensors=self.tn))
                for t in eqs["energy"]) - self.cc.e_corr)
        return out

    def test_wick_engine_residual_vanishes(self):
        r = self._residuals("wick")
        for name in ("singles", "doubles", "triples", "energy"):
            self.assertLess(r[name], 1e-7,
                            f"wick {name} residual {r[name]} not ~0 at UCCSDT amps")

    def test_diagram_engine_residual_vanishes(self):
        r = self._residuals("diagram")
        for name in ("singles", "doubles", "triples", "energy"):
            self.assertLess(r[name], 1e-7,
                            f"diagram {name} residual {r[name]} not ~0 at UCCSDT amps")


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S4a1Rank6IdentityTests(unittest.TestCase):
    """S4a.1: the rank-6 S1' identity -- the numeric proof the S4 STRUCTURAL gate
    defers. Production `spin.py::ucc_integrate_term_antisym` (which drives the
    general rank-2n `_antisym_to_allowed`) reproduces the GCC TRIPLES residual on
    the real closed-shell antisymmetric integrals, sliced to a canonical external
    block. This is the rank-6 analog of `S1AntisymIntegrationTests` (rank-4),
    exercising the production path -- not the fixture -- at rank 6.

    It is a per-term ALGEBRAIC identity (holds for any amplitudes), so the UCCSDT
    fixture's t3 is used only as a convenient real antisym tensor set; the gate is
    `sum(integrate_antisym) == GCC-slice`, not residual-vanishing (that is S4a.0c).
    """

    @classmethod
    def setUpClass(cls):
        cls.tn, cls.no, cls.nv, cls.cc = _uccsdt_so_tensors()
        # Diagram engine: ~equivalent to wick for CCSDT (proven residual-vanishing
        # both ways in S4a0cTriplesResidualTests) and much faster to generate.
        # Generate once and share across the two external-block checks.
        cls.triples = generate_cc_equations("ccsdt", engine="diagram")["triples"]

    def _check_external(self, ext, vir_pat, occ_pat):
        import numpy as np
        from ccgen.spin import ucc_integrate_term_antisym
        from ccgen.tests.residual_eval import residual_einsum
        tn, no, nv = self.tn, self.no, self.nv
        n = no + nv
        triples = self.triples
        Rg = sum(residual_einsum(t, no, nv, tensors=tn) for t in triples)
        # R layout is [a,b,c,i,j,k]; slice each axis to its spin (a=even, b=odd)
        vsl = {"a": list(range(0, nv, 2)), "b": list(range(1, nv, 2))}
        osl = {"a": list(range(0, no, 2)), "b": list(range(1, no, 2))}
        Rb = Rg[np.ix_(vsl[vir_pat[0]], vsl[vir_pat[1]], vsl[vir_pat[2]],
                       osl[occ_pat[0]], osl[occ_pat[1]], osl[occ_pat[2]])]
        acc = np.zeros_like(Rb)
        for t in triples:
            for st in ucc_integrate_term_antisym(t, ext):
                acc += _eval_spinterm(st, tn, no, n,
                                      ["a", "b", "c", "i", "j", "k"])
        self.assertLess(np.max(np.abs(acc - Rb)), 1e-10,
                        f"triples antisym != GCC slice for external {ext}")

    def test_aaa_external(self):
        self._check_external(
            {"a": "a", "b": "a", "c": "a", "i": "a", "j": "a", "k": "a"},
            "aaa", "aaa")

    def test_aab_external(self):
        # two alpha lines (a,i),(b,j); one beta line (c,k)
        self._check_external(
            {"a": "a", "b": "a", "c": "b", "i": "a", "j": "a", "k": "b"},
            "aab", "aab")


def _collapse_same_spin_block(t_mixed, n):
    """S4b.0: reconstruct the all-alpha same-spin block t_n[a..a] from the mixed
    block whose bra spins are (a,..,a,b) (the single beta bra-slot at position
    n-1). BRA-ONLY antisymmetrization: sum over placing the beta bra-slot in each
    of the n bra positions, with the transposition sign; the ket is FIXED.

    This is the rank-2n generalization of `_split_t2aaaa`'s S2.0 relation
    (t2[aaaa] = t2[abab] - t2[abab](bra swap)) -- there n=2, one beta bra-slot
    over 2 positions -> 2 terms. `t_mixed` is spatial [v*n, o*n]."""
    import numpy as np
    out = np.zeros_like(t_mixed)
    for pos in range(n):
        bra = [x for x in range(n) if x != n - 1]
        bra.insert(pos, n - 1)                      # move the beta-slot to `pos`
        sign = 1
        for a in range(n):
            for b in range(a + 1, n):
                if bra[a] > bra[b]:
                    sign = -sign
        axes = tuple(bra) + tuple(range(n, 2 * n))  # bra permuted, ket fixed
        out += sign * t_mixed.transpose(axes)
    return out


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S4bZeroCollapseRelationTests(unittest.TestCase):
    """S4b.0: pin the rank-2n same-spin amplitude-collapse relation numerically,
    BEFORE writing the general splitter (same discipline as S2.0 / map.1).

    The all-alpha same-spin block collapses to the mixed block by BRA-ONLY
    antisymmetrization of the single beta bra-slot across the n bra positions
    (ket fixed). Verified on the real closed-shell antisym UCCSDT fixture at:
      - rank-4 (t2[aaaa] from t2[abab]) -- anchors the existing `_split_t2aaaa`
        relation as the n=2 case,
      - rank-6 (t3[aaaaaa] from the aab block) -- the new rank-6 content.
    A joint (bra+ket) swap does NOT reproduce it (checked in scoping: ~0.014), so
    the bra-only rule is load-bearing."""

    @classmethod
    def setUpClass(cls):
        cls.tn, cls.no, cls.nv, cls.cc = _uccsdt_so_tensors()
        cls.ve = list(range(0, cls.nv, 2))
        cls.vo = list(range(1, cls.nv, 2))
        cls.oe = list(range(0, cls.no, 2))
        cls.oo = list(range(1, cls.no, 2))

    def test_rank4_t2_relation(self):
        import numpy as np
        t2 = self.tn["t2"]
        ve, vo, oe, oo = self.ve, self.vo, self.oe, self.oo
        t2_abab = t2[np.ix_(ve, vo, oe, oo)]        # abab: bra (a,b), ket (a,b)
        t2_aaaa = t2[np.ix_(ve, ve, oe, oe)]
        recon = _collapse_same_spin_block(t2_abab, 2)
        self.assertLess(np.abs(recon - t2_aaaa).max(), 1e-12,
                        "rank-4 same-spin collapse relation broken")

    def test_rank6_t3_relation(self):
        import numpy as np
        t3 = self.tn["t3"]
        ve, vo, oe, oo = self.ve, self.vo, self.oe, self.oo
        # mixed aab: bra (a,a,b), ket (a,a,b) -- beta at bra slot 2 and ket slot 2
        t3_aab = t3[np.ix_(ve, ve, vo, oe, oe, oo)]
        t3_aaa = t3[np.ix_(ve, ve, ve, oe, oe, oe)]
        recon = _collapse_same_spin_block(t3_aab, 3)
        self.assertLess(np.abs(recon - t3_aaa).max(), 1e-12,
                        "rank-6 same-spin collapse relation broken")

    def test_joint_swap_is_wrong(self):
        # pins the finding: a JOINT (bra+ket) antisymmetrization does NOT give the
        # all-alpha block, so the bra-only rule cannot be replaced by a line swap.
        import numpy as np
        t3 = self.tn["t3"]
        ve, vo, oe, oo = self.ve, self.vo, self.oe, self.oo
        t3_aab = t3[np.ix_(ve, ve, vo, oe, oe, oo)]
        t3_aaa = t3[np.ix_(ve, ve, ve, oe, oe, oe)]
        out = np.zeros_like(t3_aab)
        for pos, vp, sign in [(2, (0, 1, 2), 1), (1, (0, 2, 1), -1),
                              (0, (2, 1, 0), -1)]:
            out += sign * t3_aab.transpose(vp[0], vp[1], vp[2],
                                           3 + vp[0], 3 + vp[1], 3 + vp[2])
        self.assertGreater(np.abs(out - t3_aaa).max(), 1e-3,
                           "joint swap unexpectedly matched -- rule ambiguous")


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S4bSplitterTests(unittest.TestCase):
    """S4b.1 + S4b.2: the general rank-2n same-spin amplitude splitter and its
    wiring into `collapse_amplitudes`.

    S4b.1: `_split_same_spin_amplitude` emits the pinned S4b.0 relation as
    SpinFactors -- summed over the split (with signs) it reproduces the all-alpha
    block, at rank-4 (t2) and rank-6 (t3), evaluated on the real UCCSDT fixture.
    S4b.2: `_is_same_spin_amplitude` / `collapse_amplitudes` fire on t3 too, not
    just t2 (no all-alpha amplitude block survives)."""

    @classmethod
    def setUpClass(cls):
        cls.tn, cls.no, cls.nv, cls.cc = _uccsdt_so_tensors()

    def _eval_split(self, factor):
        """Sum the splitter's (sign, SpinFactor) output, sliced from the fixture,
        into the all-alpha block layout keyed by the factor's base names."""
        import numpy as np
        from ccgen.spin import SpinTerm, _split_same_spin_amplitude
        out_names = [si.base.name for si in factor.indices]
        acc = None
        for sign, sf in _split_same_spin_amplitude(factor):
            st = SpinTerm(coeff=sign, external_block="", factors=(sf,))
            arr = _eval_spinterm(st, self.tn, self.no, self.no + self.nv,
                                 out_names)
            acc = arr if acc is None else acc + arr
        return acc

    def _all_alpha_block(self, name, n):
        import numpy as np
        arr = self.tn[name]
        ve = list(range(0, self.nv, 2))
        oe = list(range(0, self.no, 2))
        return arr[np.ix_(*([ve] * n + [oe] * n))]

    def test_rank4_splitter_reproduces_block(self):
        import numpy as np
        from ccgen.spin import SpinFactor, SpinIndex
        idx = tuple(SpinIndex(b, "a") for b in
                    (make_vir("a"), make_vir("b"), make_occ("i"), make_occ("j")))
        f = SpinFactor(name="t2", block="aaaa", indices=idx)
        got = self._eval_split(f)
        ref = self._all_alpha_block("t2", 2)
        self.assertLess(np.abs(got - ref).max(), 1e-12,
                        "rank-4 splitter != t2[aaaa]")

    def test_rank6_splitter_reproduces_block(self):
        import numpy as np
        from ccgen.spin import SpinFactor, SpinIndex
        idx = tuple(SpinIndex(b, "a") for b in
                    (make_vir("a"), make_vir("b"), make_vir("c"),
                     make_occ("i"), make_occ("j"), make_occ("k")))
        f = SpinFactor(name="t3", block="aaaaaa", indices=idx)
        got = self._eval_split(f)
        ref = self._all_alpha_block("t3", 3)
        self.assertLess(np.abs(got - ref).max(), 1e-12,
                        "rank-6 splitter != t3[aaaaaa]")
        # the split has n=3 terms
        from ccgen.spin import _split_same_spin_amplitude
        self.assertEqual(len(_split_same_spin_amplitude(f)), 3)

    def test_collapse_fires_on_t3(self):
        # S4b.2: collapse_amplitudes dispatches the same-spin t3 block through the
        # general splitter -- no all-alpha t3 factor survives.
        from ccgen.spin import (SpinFactor, SpinIndex, SpinTerm,
                                collapse_amplitudes, _is_same_spin_amplitude)
        t3 = SpinFactor(name="t3", block="aaaaaa", indices=tuple(
            SpinIndex(b, "a") for b in
            (make_vir("a"), make_vir("b"), make_vir("c"),
             make_occ("i"), make_occ("j"), make_occ("k"))))
        self.assertTrue(_is_same_spin_amplitude(t3))
        st = SpinTerm(coeff=1, external_block="aaaaaa", factors=(t3,))
        collapsed = collapse_amplitudes(st)
        blocks = {f.block for c in collapsed for f in c.factors if f.name == "t3"}
        self.assertNotIn("aaaaaa", blocks, f"same-spin t3 survived: {blocks}")
        self.assertEqual(len(collapsed), 3)
        # tag/spin consistency preserved
        for c in collapsed:
            for f in c.factors:
                self.assertEqual(f.block, "".join(si.spin for si in f.indices))

    def test_t1_not_split(self):
        # t1 (rank-2, block 'aa') is already single-block -- must NOT be split.
        from ccgen.spin import SpinFactor, SpinIndex, _is_same_spin_amplitude
        t1 = SpinFactor(name="t1", block="aa", indices=(
            SpinIndex(make_vir("a"), "a"), SpinIndex(make_occ("i"), "a")))
        self.assertFalse(_is_same_spin_amplitude(t1))


class S4cIntegralRankTests(unittest.TestCase):
    """S4c.0: the integral splitter `_split_vaaaa` needs NO rank-2n generalization
    -- the two-electron integral is fundamentally rank-4, so every `v` factor is
    rank-4 across CCSDT and CCSDTQ (no rank-6/8 integral exists in CC theory).
    S4c is therefore a confirmed no-op, not deferred work. Pinned so a future
    manifold change can't silently introduce a higher-rank `v` the 4-index
    `_split_vaaaa` would mis-handle. Diagram engine (fast, equivalent to wick)."""

    def test_v_is_always_rank4(self):
        for method in ("ccsdt", "ccsdtq"):
            eqs = generate_cc_equations(method, engine="diagram")
            ranks = {len(f.indices)
                     for terms in eqs.values() for t in terms
                     for f in t.factors if f.name == "v"}
            self.assertEqual(ranks, {4},
                             f"{method}: v factors not all rank-4: {ranks}")


def _ccsdtq_fci_limit_tensors(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0",
                              basis="sto-3g"):
    """S4d fixture: a closed-shell ANTISYMMETRIC rank-8 `t4` (with t1/t2/t3, v, f)
    obtained by Jacobi-iterating ccgen's GCC CCSDTQ residual to self-consistency
    on RHF-derived spin-orbitals (even=alpha / odd=beta).

    This is the fixture the oracle wall denied: PySCF 2.13.0 has no `uccsdtq`, and
    `rccsdtq` gives a SYMMETRIC triangular `t4full` that self-cancels under
    antisymmetrization (the S4a.0a wall, one rank up). GHF-CCSDTQ reaches FCI but
    is spin-mixed (no clean alpha/beta partition the adaptation needs). Iterating
    the GCC residual on the RHF even/odd basis sidesteps both: for a 4-electron
    system CCSDTQ == FCI, so the converged amps ARE the exact closed-shell antisym
    tensors -- no lift, no oracle. Returns (tensors, no, nv, e_corr, e_fci_tot,
    e_hf)."""
    import numpy as np
    from pyscf import gto, scf, ao2mo, fci
    from ccgen.tests.residual_eval import residual_einsum

    mol = gto.M(atom=atom, basis=basis, spin=0, verbose=0)
    mol.cart = True
    mf = scf.RHF(mol).run()
    nocc_sp = mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    nvir_sp = nmo - nocc_sp
    no, nv, n = 2 * nocc_sp, 2 * nvir_sp, 2 * nmo

    def csp(p):
        return (p // 2) if p < no else nocc_sp + ((p - no) // 2)

    def cspin(p):
        return (p % 2) if p < no else ((p - no) % 2)

    eri = ao2mo.kernel(mol, mf.mo_coeff, aosym="s1").reshape(nmo, nmo, nmo, nmo)
    g = eri.transpose(0, 2, 1, 3)
    v = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    c = (g[csp(p), csp(q), csp(r), csp(s)]
                         if cspin(p) == cspin(r) and cspin(q) == cspin(s) else 0)
                    e = (g[csp(p), csp(q), csp(s), csp(r)]
                         if cspin(p) == cspin(s) and cspin(q) == cspin(r) else 0)
                    v[p, q, r, s] = c - e
    f = np.zeros((n, n))
    for p in range(n):
        f[p, p] = mf.mo_energy[csp(p)]

    e = f.diagonal()
    eo, ev = e[:no], e[no:]

    def denom(r):
        D = np.zeros((nv,) * r + (no,) * r)
        it = np.nditer(D, flags=["multi_index"], op_flags=["writeonly"])
        for _ in it:
            idx = it.multi_index
            it[0] = (sum(eo[o] for o in idx[r:])
                     - sum(ev[a] for a in idx[:r]))
        return D

    targets = ["singles", "doubles", "triples", "quadruples"]
    rk = {"singles": 1, "doubles": 2, "triples": 3, "quadruples": 4}
    tn_name = {"singles": "t1", "doubles": "t2", "triples": "t3",
               "quadruples": "t4"}
    D = {r: denom(r) for r in (1, 2, 3, 4)}
    amps = {tn_name[m]: np.zeros((nv,) * rk[m] + (no,) * rk[m]) for m in targets}
    amps["t2"] = v[:no, :no, no:, no:].transpose(2, 3, 0, 1) / D[2]

    eqs = generate_cc_equations("ccsdtq", engine="diagram")

    def tensors():
        return {"v": v, "f": f, **amps}

    for _ in range(500):
        delta, upd = 0.0, {}
        for m in targets:
            R = sum(residual_einsum(t, no, nv, tensors=tensors()) for t in eqs[m])
            new = amps[tn_name[m]] + R / D[rk[m]]
            upd[tn_name[m]] = new
            delta = max(delta, float(np.max(np.abs(new - amps[tn_name[m]]))))
        amps.update(upd)
        if delta < 1e-11:
            break

    e_corr = sum(float(residual_einsum(t, no, nv, tensors=tensors()))
                 for t in eqs["energy"])
    e_fci, _ = fci.FCI(mf).kernel()
    return ({"v": v, "f": f, **amps}, no, nv, e_corr,
            float(e_fci), float(mf.e_tot))


class R313IndependentBlocksTests(unittest.TestCase):
    """R3.1.3a: the independent spin-block enumeration -- a pure-function gate
    (no PySCF, seconds). `independent_spin_blocks(rank)` lists one representative
    per Sz sector; `_amplitude_block_tag(block)` folds any block to its sector.
    These are the precondition for storing/reading t4's second Sz sector."""

    def test_independent_blocks_low_ranks(self):
        from ccgen.spin import independent_spin_blocks
        self.assertEqual(independent_spin_blocks(2), ["aa"])         # t1
        self.assertEqual(independent_spin_blocks(4), ["abab"])       # t2
        self.assertEqual(independent_spin_blocks(6), ["aabaab"])     # t3, one
        self.assertEqual(independent_spin_blocks(8),                 # t4, TWO
                         ["aabbaabb", "aaabaaab"])
        self.assertEqual(independent_spin_blocks(10),                # t5, TWO
                         ["aaabbaaabb", "aaaabaaaab"])

    def test_block_tag_folds_flip_partners(self):
        from ccgen.spin import _amplitude_block_tag
        # t2: one component
        self.assertEqual(_amplitude_block_tag("abab"), "abab")
        # t3: aabaab reference, abbabb is its flip partner -> same component
        self.assertEqual(_amplitude_block_tag("aabaab"), "aabaab")
        self.assertEqual(_amplitude_block_tag("abbabb"), "aabaab")
        # t4: aabb reference; aaab its own sector; abbb folds to aaab
        self.assertEqual(_amplitude_block_tag("aabbaabb"), "aabbaabb")
        self.assertEqual(_amplitude_block_tag("aaabaaab"), "aaabaaab")
        self.assertEqual(_amplitude_block_tag("abbbabbb"), "aaabaaab")

    def test_census_blocks_fold_into_the_independent_set(self):
        # every amplitude block that appears in the merged rank-6/rank-8 manifold
        # must fold (via _amplitude_block_tag) into independent_spin_blocks(rank).
        # This proves the enumeration is COMPLETE for the manifolds we emit.
        from ccgen.spin import independent_spin_blocks, _amplitude_block_tag
        census = {
            4: {"abab"},                                   # t2
            6: {"aabaab", "abbabb"},                       # t3
            8: {"aabbaabb", "aaabaaab", "abbbabbb"},       # t4
        }
        for rank, blocks in census.items():
            allowed = set(independent_spin_blocks(rank))
            for blk in blocks:
                self.assertIn(_amplitude_block_tag(blk), allowed,
                              f"rank-{rank} block {blk} folds to "
                              f"{_amplitude_block_tag(blk)} not in {allowed}")

    def test_representative_block_for_sector(self):
        # R3.1.3d: the external residual block for a given Sz sector k. The
        # reference (k=ceil(n/2)) reproduces _closed_shell_representative_block;
        # the second t4 sector (k=3) is the 3α1β external the aaabaaab residual
        # integrates on.
        from ccgen.spin import (_representative_block_for_sector,
                                _closed_shell_representative_block, _residual_template)
        from ccgen import generate_cc_equations
        from ccgen.spin import spin_adapt_equations  # noqa: F401  (import guard)
        eqs = generate_cc_equations("ccsdtq", engine="diagram")
        tmpl = _residual_template("quadruples", eqs["quadruples"])
        names = [i.name for i in tmpl.indices]
        n = len(names) // 2
        # k=2 (reference) == the existing helper
        ref = _closed_shell_representative_block(tmpl)
        self.assertEqual(_representative_block_for_sector(tmpl, (n + 1) // 2), ref)
        # k=3 second sector: 3 alpha then 1 beta per half
        sec = _representative_block_for_sector(tmpl, 3)
        bra_spins = [sec[names[j]] for j in range(n)]
        self.assertEqual(bra_spins, ["a", "a", "a", "b"])
        ket_spins = [sec[names[n + j]] for j in range(n)]
        self.assertEqual(ket_spins, ["a", "a", "a", "b"])

    def test_ccsd_ccsdt_adapt_keys_unchanged(self):
        # R3.1.3d must be byte-identical for n<=3 targets: no extra Sz sector, so
        # the key set is exactly the targets (backward-compatible with the emit
        # path). Only quadruples (n=4) gains a `_aaabaaab` key.
        from ccgen.spin import spin_adapt_equations
        from ccgen import generate_cc_equations
        self.assertEqual(set(spin_adapt_equations(generate_cc_equations("ccsd"))),
                         {"energy", "singles", "doubles"})
        self.assertEqual(set(spin_adapt_equations(generate_cc_equations("ccsdt"))),
                         {"energy", "singles", "doubles", "triples"})

    def test_ccsdtq_adapt_emits_both_t4_sectors(self):
        # R3.1.3d: the ccsdtq quadruples residual is emitted as TWO blocks -- the
        # reference `quadruples` (aabbaabb) and `quadruples_aaabaaab` (the second
        # independent Sz sector) -- so both stored t4 blocks get their own residual
        # to iterate. ~5s (diagram engine cached). Both must be non-empty.
        from ccgen.spin import spin_adapt_equations
        from ccgen import generate_cc_equations
        adapted = spin_adapt_equations(generate_cc_equations("ccsdtq",
                                                             engine="diagram"))
        self.assertEqual(
            set(adapted),
            {"energy", "singles", "doubles", "triples",
             "quadruples", "quadruples_aaabaaab"})
        self.assertGreater(len(adapted["quadruples"]), 0)
        self.assertGreater(len(adapted["quadruples_aaabaaab"]), 0)


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S4dRank8IdentityTests(unittest.TestCase):
    """S4d: the rank-8 numeric gate. On a closed-shell antisymmetric `t4` obtained
    by iterating GCC CCSDTQ to the FCI limit (`_ccsdtq_fci_limit_tensors`),
    production `ucc_integrate_term_antisym` reproduces the GCC QUADRUPLES residual
    sliced to a canonical external block -- the rank-8 analog of S4a.1, exercising
    the general rank-2n `_antisym_to_allowed` at rank 8.

    The FCI-limit route replaces the missing `uccsdtq` oracle: for a 4-electron
    system CCSDTQ == FCI, so the iterated amps are exact closed-shell antisym
    tensors. The all-alpha rank-8 block is structurally impossible at 4 electrons
    (needs 4 same-spin occupieds), so the gate uses a mixed `aabb` external, and
    perturbs t4 (x0.5) so the residual is genuinely nonzero (a real identity test,
    not 0 == 0)."""

    @classmethod
    def setUpClass(cls):
        (cls.tn, cls.no, cls.nv, cls.e_corr,
         cls.e_fci, cls.e_hf) = _ccsdtq_fci_limit_tensors()

    def test_fixture_reaches_fci(self):
        # the fixture is only valid if the iterated CCSDTQ energy == FCI
        self.assertLess(abs(self.e_hf + self.e_corr - self.e_fci), 1e-6,
                        f"CCSDTQ e_corr does not reach FCI "
                        f"({self.e_hf + self.e_corr} vs {self.e_fci})")

    def test_t4_closed_shell_antisym(self):
        import numpy as np
        t4 = self.tn["t4"]
        self.assertLess(np.abs(t4 + t4.transpose(1, 0, 2, 3, 4, 5, 6, 7)).max(),
                        1e-12, "t4 not antisym in a vir pair")
        self.assertLess(np.abs(t4 + t4.transpose(0, 1, 2, 3, 5, 4, 6, 7)).max(),
                        1e-12, "t4 not antisym in an occ pair")

    def test_rank8_aabb_identity(self):
        import numpy as np
        from ccgen.spin import ucc_integrate_term_antisym
        from ccgen.tests.residual_eval import residual_einsum
        no, nv, n = self.no, self.nv, self.no + self.nv
        tn = dict(self.tn)
        tn["t4"] = 0.5 * tn["t4"]              # perturb -> nonzero residual
        eqs = generate_cc_equations("ccsdtq", engine="diagram")
        Rg = sum(residual_einsum(t, no, nv, tensors=tn)
                 for t in eqs["quadruples"])
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))
        Rb = Rg[np.ix_(ve, ve, vo, vo, oe, oe, oo, oo)]
        self.assertGreater(np.abs(Rb).max(), 1e-6,
                           "perturbed residual should be nonzero")
        ext = {"a": "a", "b": "a", "c": "b", "d": "b",
               "i": "a", "j": "a", "k": "b", "l": "b"}
        acc = np.zeros_like(Rb)
        for t in eqs["quadruples"]:
            for st in ucc_integrate_term_antisym(t, ext):
                acc += _eval_spinterm(st, tn, no, n,
                                      ["a", "b", "c", "d", "i", "j", "k", "l"])
        self.assertLess(np.abs(acc - Rb).max(), 1e-10,
                        f"rank-8 antisym != GCC aabb slice: "
                        f"{np.abs(acc - Rb).max()}")

    def test_rank8_full_collapse_pipeline(self):
        # R3.1.0 -- fast rank-8 gate for the FULL collapse+merge pipeline
        # (canonicalize -> collapse_amplitudes -> collapse_integrals -> merge),
        # the rank-8 analog of test_rcc_pipeline_generalizes_rank6. GREEN: the
        # per-block collapse+merge DOES reproduce the GCC aabb slice at rank 8.
        # So the rank-8 collapse is NOT the bug -- the Be CCSDTQ failure is
        # downstream (how spin_adapt_equations assembles the spatial residual the
        # solver iterates, not per-block correctness). Seconds to run.
        import numpy as np
        from ccgen.spin import (ucc_integrate_term_antisym,
                                canonicalize_spin_blocks, collapse_amplitudes,
                                collapse_integrals, merge_terms)
        from ccgen.tests.residual_eval import residual_einsum
        no, nv, n = self.no, self.nv, self.no + self.nv
        tn = dict(self.tn)
        tn["t4"] = 0.5 * tn["t4"]
        eqs = generate_cc_equations("ccsdtq", engine="diagram")
        Rg = sum(residual_einsum(t, no, nv, tensors=tn)
                 for t in eqs["quadruples"])
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))
        Rb = Rg[np.ix_(ve, ve, vo, vo, oe, oe, oo, oo)]
        ext = {"a": "a", "b": "a", "c": "b", "d": "b",
               "i": "a", "j": "a", "k": "b", "l": "b"}
        manifold = []
        for t in eqs["quadruples"]:
            manifold.extend(ucc_integrate_term_antisym(t, ext))
        canon = [canonicalize_spin_blocks(st) for st in manifold]
        amp = [c for st in canon for c in collapse_amplitudes(st)]
        coll = [c for st in amp for c in collapse_integrals(st)]
        merged = merge_terms(coll, set(ext))
        acc = np.zeros_like(Rb)
        for st in merged:
            acc += _eval_spinterm(st, tn, no, n,
                                  ["a", "b", "c", "d", "i", "j", "k", "l"])
        self.assertLess(np.abs(acc - Rb).max(), 1e-10,
                        f"rank-8 collapse+merge != GCC aabb slice: "
                        f"{np.abs(acc - Rb).max()}")

    def test_rank8_bridge_solve_path(self):
        # R3.1.3 gate: the rank-8 analog of test_rcc_bridge_solve_path_rank6. The
        # SOLVE path (spinterm_to_algebraterm + residual_einsum) must match the
        # aabb GCC slice, like the per-spin-block oracle does. GREEN as of R3.1.3c:
        # t4 has TWO independent Sz sectors -- aabbaabb (reference) and aaabaaab --
        # and aaab is NOT a permutation or spin-flip of aabb (proven: not even a
        # signed-perm combination from one shared spatial tau). So it cannot be
        # folded onto the reference; the bridge now NAMES the second-sector factors
        # `t4_aaabaaab` (abbbabbb folds onto it via the existing flip), and the
        # solve reads that block from its own stored tensor. Rank-8, ~30s -- iterate
        # here, not the ~15min Be CCSDTQ solve. See
        # docs/CCGEN_CCSDTQ_MULTISECTOR.md (R3.1.3).
        import numpy as np
        from ccgen.spin import (ucc_integrate_term_antisym,
                                canonicalize_spin_blocks, collapse_amplitudes,
                                collapse_integrals, merge_terms,
                                spinterm_to_algebraterm)
        from ccgen.tests.residual_eval import residual_einsum
        no, nv, n = self.no, self.nv, self.no + self.nv
        nos, nvs = no // 2, nv // 2
        tn = dict(self.tn)
        tn["t4"] = 0.5 * tn["t4"]
        tn["t3"] = 0.7 * tn["t3"]
        tn["t2"] = 0.9 * tn["t2"]

        def block_slice(so, block):
            sets = [[p for p in range(so.shape[k])
                     if p % 2 == (0 if s == "a" else 1)]
                    for k, s in enumerate(block)]
            return so[np.ix_(*sets)]

        spatial = {
            "t1": block_slice(tn["t1"], "aa"),
            "t2": block_slice(tn["t2"], "abab"),
            "t3": block_slice(tn["t3"], "aabaab"),
            "t4": block_slice(tn["t4"], "aabbaabb"),
            # R3.1.3c: the second independent t4 Sz sector, stored separately.
            "t4_aaabaaab": block_slice(tn["t4"], "aaabaaab"),
            "v": block_slice(tn["v"], "abab"),
            "f": block_slice(tn["f"], "aa"),
        }
        eqs = generate_cc_equations("ccsdtq", engine="diagram")
        Rg = sum(residual_einsum(t, no, nv, tensors=tn)
                 for t in eqs["quadruples"])
        ve, vo = list(range(0, nv, 2)), list(range(1, nv, 2))
        oe, oo = list(range(0, no, 2)), list(range(1, no, 2))
        Rb = Rg[np.ix_(ve, ve, vo, vo, oe, oe, oo, oo)]
        ext = {"a": "a", "b": "a", "c": "b", "d": "b",
               "i": "a", "j": "a", "k": "b", "l": "b"}
        manifold = []
        for t in eqs["quadruples"]:
            manifold.extend(ucc_integrate_term_antisym(t, ext))
        canon = [canonicalize_spin_blocks(st) for st in manifold]
        amp = [c for st in canon for c in collapse_amplitudes(st)]
        coll = [c for st in amp for c in collapse_integrals(st)]
        merged = merge_terms(coll, set(ext))
        alg = [spinterm_to_algebraterm(st, set(ext)) for st in merged]
        R = sum(residual_einsum(t, nos, nvs, tensors=spatial) for t in alg)
        self.assertLess(np.abs(R - Rb).max(), 1e-10,
                        f"rank-8 bridge solve-path != GCC aabb slice: "
                        f"{np.abs(R - Rb).max()}")


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable in this interpreter")
class S4a2ArbitraryOrderTests(unittest.TestCase):
    """S4a.2 (reframed): the RCC/UCC spin-adaptation PIPELINE already generalizes
    to arbitrary order -- no per-rank `closed_shell_antisym_lift` is needed.

    The originally-scoped deliverable (a unified spatial->antisym lift) turned out
    to be a dead end: the three fixture fills have different data sources (so_t2
    lifts a spatial t2ab, _t3so_read reads UCCSDT spin-blocks, t4 is FCI-iterated),
    and a spatial->antisym lift of the SYMMETRIC rccsdtq t4full is provably
    impossible (antisymmetrizing self-cancels -- the spin-summation inverse).
    S4d already showed the lift is unnecessary for correctness.

    The right question is whether the existing code generalizes. It does: every
    pipeline stage is written on rank-agnostic SpinFactor/SpinIndex operations
    (`_antisym_to_allowed` and `_split_same_spin_amplitude` infer rank from
    `len(indices)//2`; `_canonical_block`/`merge_terms` are tag-string ops; the
    only rank-4 hardcode, `_split_vaaaa`, is CORRECTLY rank-4 because integrals
    are always rank-4, S4c). This gate runs the FULL RCC collapse+merge and the
    UCC integration on the rank-6 CCSDT triples manifold with NO code changes and
    requires the numeric result to match the GCC slice (perturbed amps -> nonzero
    residual, a real identity test)."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        tn, cls.no, cls.nv, cls.cc = _uccsdt_so_tensors()
        # perturb so the triples residual is genuinely nonzero
        tn = dict(tn)
        tn["t3"] = 0.5 * tn["t3"]
        tn["t2"] = 0.7 * tn["t2"]
        cls.tn = tn
        cls.triples = generate_cc_equations("ccsdt", engine="diagram")["triples"]

    def _gcc_slice(self, ext):
        import numpy as np
        from ccgen.tests.residual_eval import residual_einsum
        no, nv = self.no, self.nv
        Rg = sum(residual_einsum(t, no, nv, tensors=self.tn) for t in self.triples)
        v_ = {"a": list(range(0, nv, 2)), "b": list(range(1, nv, 2))}
        o_ = {"a": list(range(0, no, 2)), "b": list(range(1, no, 2))}
        idx = [v_[ext[x]] for x in "abc"] + [o_[ext[x]] for x in "ijk"]
        return Rg[np.ix_(*idx)]

    def test_ucc_integration_generalizes_rank6(self):
        # UCC = antisym integration into a spin block, no collapse. Same-spin (aaa)
        # and mixed (aab) externals both reproduce the GCC slice at rank 6.
        import numpy as np
        from ccgen.spin import ucc_integrate_term_antisym
        no, nv, n = self.no, self.nv, self.no + self.nv
        for ext in ({"a": "a", "b": "a", "c": "a",
                     "i": "a", "j": "a", "k": "a"},
                    {"a": "a", "b": "a", "c": "b",
                     "i": "a", "j": "a", "k": "b"}):
            acc = None
            for t in self.triples:
                for st in ucc_integrate_term_antisym(t, ext):
                    arr = _eval_spinterm(st, self.tn, no, n,
                                         ["a", "b", "c", "i", "j", "k"])
                    acc = arr if acc is None else acc + arr
            Rb = self._gcc_slice(ext)
            self.assertGreater(np.abs(Rb).max(), 1e-6, "residual should be nonzero")
            self.assertLess(np.abs(acc - Rb).max(), 1e-10,
                            f"UCC rank-6 integration != GCC slice for {ext}")

    def test_rcc_pipeline_generalizes_rank6(self):
        # The FULL RCC collapse+merge (canonicalize -> collapse amps -> collapse
        # integrals -> merge) runs on the rank-6 triples manifold unchanged and the
        # merged spatial residual reproduces the GCC slice.
        import numpy as np
        from ccgen.spin import (ucc_integrate_term_antisym,
                                canonicalize_spin_blocks, collapse_amplitudes,
                                collapse_integrals, merge_terms)
        no, nv, n = self.no, self.nv, self.no + self.nv
        ext = {"a": "a", "b": "a", "c": "b", "i": "a", "j": "a", "k": "b"}
        manifold = []
        for t in self.triples:
            manifold.extend(ucc_integrate_term_antisym(t, ext))
        canon = [canonicalize_spin_blocks(st) for st in manifold]
        amp = [c for st in canon for c in collapse_amplitudes(st)]
        coll = [c for st in amp for c in collapse_integrals(st)]
        merged = merge_terms(coll, set(ext))
        # every merged factor is a single spatial block (t2/v abab, t1/f aa)
        for st in merged:
            for f in st.factors:
                self.assertIn(set(f.block), ({"a"}, {"a", "b"}))
        acc = None
        for st in merged:
            arr = _eval_spinterm(st, self.tn, no, n,
                                 ["a", "b", "c", "i", "j", "k"])
            acc = arr if acc is None else acc + arr
        Rb = self._gcc_slice(ext)
        self.assertLess(np.abs(acc - Rb).max(), 1e-10,
                        "merged RCC rank-6 residual != GCC aab slice")

    def test_rcc_bridge_solve_path_rank6(self):
        # R3.1.2 whole-residual gate: the SOLVE path (spinterm_to_algebraterm +
        # residual_einsum on ONE spatial tensor per amplitude) must match the GCC
        # slice, like the _eval_spinterm path does. GREEN as of R3.1.2 (was
        # ~4.8e-3): the bridge now (i) canonicalizes its output layout and (ii)
        # maps every amplitude factor onto its stored reference block via the
        # spin-flip in `_canonicalize_amplitude_factor`, so reading one spatial
        # tensor per amplitude reproduces the per-spin-block oracle. Rank-6,
        # seconds -- the R3.1.2 inner loop.
        import numpy as np
        from ccgen.spin import (ucc_integrate_term_antisym,
                                canonicalize_spin_blocks, collapse_amplitudes,
                                collapse_integrals, merge_terms,
                                spinterm_to_algebraterm)
        from ccgen.tests.residual_eval import residual_einsum
        no, nv = self.no, self.nv
        nos, nvs = no // 2, nv // 2
        ext = {"a": "a", "b": "a", "c": "b", "i": "a", "j": "a", "k": "b"}

        def block_slice(so, block):
            sets = [[p for p in range(so.shape[k]) if p % 2 == (0 if s == "a" else 1)]
                    for k, s in enumerate(block)]
            return so[np.ix_(*sets)]

        # one spatial tensor per amplitude, on its own OUTPUT block
        spatial = {
            "t1": block_slice(self.tn["t1"], "aa"),
            "t2": block_slice(self.tn["t2"], "abab"),
            "t3": block_slice(self.tn["t3"], "aabaab"),
            # v/f are over the n-space; their RCC spatial tensor is the abab / aa
            # spin block, NOT an all-α slice (v[abab] has spins a,b,a,b).
            "v": block_slice(self.tn["v"], "abab"),
            "f": block_slice(self.tn["f"], "aa"),
        }
        manifold = []
        for t in self.triples:
            manifold.extend(ucc_integrate_term_antisym(t, ext))
        canon = [canonicalize_spin_blocks(st) for st in manifold]
        amp = [c for st in canon for c in collapse_amplitudes(st)]
        coll = [c for st in amp for c in collapse_integrals(st)]
        merged = merge_terms(coll, set(ext))
        alg = [spinterm_to_algebraterm(st, set(ext)) for st in merged]
        R = sum(residual_einsum(t, nos, nvs, tensors=spatial) for t in alg)
        Rb = self._gcc_slice(ext)
        self.assertLess(np.abs(R - Rb).max(), 1e-10,
                        f"bridge solve-path rank-6 != GCC aab slice: "
                        f"{np.abs(R - Rb).max()}")

    def _rank6_bridge_spatial(self):
        """Shared rank-6 setup: the merged SpinTerms and the one-spatial-tensor-
        per-amplitude dict the solve path uses (P2 harness)."""
        import numpy as np
        from ccgen.spin import (ucc_integrate_term_antisym,
                                canonicalize_spin_blocks, collapse_amplitudes,
                                collapse_integrals, merge_terms)
        no, nv = self.no, self.nv
        ext = {"a": "a", "b": "a", "c": "b", "i": "a", "j": "a", "k": "b"}

        def block_slice(so, block):
            sets = [[p for p in range(so.shape[k]) if p % 2 == (0 if s == "a" else 1)]
                    for k, s in enumerate(block)]
            return so[np.ix_(*sets)]

        spatial = {
            "t1": block_slice(self.tn["t1"], "aa"),
            "t2": block_slice(self.tn["t2"], "abab"),
            "t3": block_slice(self.tn["t3"], "aabaab"),
            # v/f are over the n-space; their RCC spatial tensor is the abab / aa
            # spin block, NOT an all-α slice (v[abab] has spins a,b,a,b).
            "v": block_slice(self.tn["v"], "abab"),
            "f": block_slice(self.tn["f"], "aa"),
        }
        manifold = []
        for t in self.triples:
            manifold.extend(ucc_integrate_term_antisym(t, ext))
        canon = [canonicalize_spin_blocks(st) for st in manifold]
        amp = [c for st in canon for c in collapse_amplitudes(st)]
        coll = [c for st in amp for c in collapse_integrals(st)]
        merged = merge_terms(coll, set(ext))
        return merged, spatial, ext

    # Canonical single spatial block each amplitude/integral is stored in (the
    # block the spin-free bridge reads from). rank -> block; v/f are always rank 4.
    _REF_BLOCK = {"t1": "aa", "t2": "abab", "t3": "aabaab", "v": "abab", "f": "aa"}

    @classmethod
    def _mech_spin(cls, st, ext):
        """P2 mechanism 1 (SPIN). The bridge drops per-index spin and reads each
        factor from ONE canonical spatial block (`_REF_BLOCK`). True iff that read
        is wrong on any of three surfaces -- all the same root cause:
          (a) a factor whose slot spins differ from its ref block;
          (b) a summed index contracted across slots whose ref-block spins differ
              (the spatial contraction sums the wrong channel); or
          (c) a free index landing on a slot whose ref-block spin != its external
              spin.
        This is the single spin gap; `_has_mixed_spin_summed_index` was only (b)."""
        rb = cls._REF_BLOCK
        for f in st.factors:
            if "".join(si.spin for si in f.indices) != rb[f.name]:
                return True
        occ: dict = {}
        for f in st.factors:
            for k, si in enumerate(f.indices):
                if si.name not in ext:
                    occ.setdefault(si.name, set()).add(rb[f.name][k])
        if any(len(s) > 1 for s in occ.values()):
            return True
        for f in st.factors:
            for k, si in enumerate(f.indices):
                if si.name in ext and rb[f.name][k] != ext[si.name]:
                    return True
        return False

    @staticmethod
    def _bridge_output_layout(at):
        """The residual axis order `residual_einsum` emits for an AlgebraTerm:
        [ext_vir..., ext_occ...] in FIRST-APPEARANCE order."""
        fr = list(at.free_indices)
        ev = [getattr(i, "name", i) for i in fr if i.space == "vir"]
        eo = [getattr(i, "name", i) for i in fr if i.space == "occ"]
        return ev + eo

    @classmethod
    def _mech_layout(cls, at, names):
        """P2 mechanism 2 (LAYOUT). Even with consistent spins, the bridge's
        `free_indices` are in first-appearance order, so `residual_einsum` lays
        the residual out transposed from the canonical [a,b,c,i,j,k]. A pure
        output-axis permutation -- value-identical, layout-wrong. This is a bridge
        bug, NOT a spin-model gap: canonicalizing `free_indices` fixes it."""
        return cls._bridge_output_layout(at) != names

    @staticmethod
    def _has_mixed_spin_summed_index(st, ext):
        """True iff some summed (internal) index appears with DIFFERENT spins in
        its two factor occurrences -- only surface (b) of the spin mechanism.
        Retained for the historical P2.1 gate below; `_mech_spin` supersedes it."""
        spins = {}
        for f in st.factors:
            for si in f.indices:
                if si.name in ext:
                    continue
                spins.setdefault(si.name, set()).add(si.spin)
        return any(len(s) > 1 for s in spins.values())

    def test_p20_bridge_matches_eval_per_term_rank6(self):
        # P2.0: per-term gate. For every merged rank-6 term the bridge
        # (spinterm_to_algebraterm + residual_einsum) must equal _eval_spinterm
        # (the oracle, which slices each factor per spin block). GREEN as of
        # R3.1.2: the two mechanisms P2.1 partitioned the failures into are both
        # fixed -- half (ii) canonicalizes the output layout, and half (i)
        # (`_canonicalize_amplitude_factor`'s spin-flip of β-majority blocks)
        # maps every amplitude factor onto its stored reference block. Was 718 of
        # 859 failing (595 both, 116 layout-only, 7 spin-only); now 0.
        import numpy as np
        from ccgen.spin import spinterm_to_algebraterm
        from ccgen.tests.residual_eval import residual_einsum
        no, nv = self.no, self.nv
        nos, nvs = no // 2, nv // 2
        names = ["a", "b", "c", "i", "j", "k"]
        merged, spatial, ext = self._rank6_bridge_spatial()
        bad = 0
        for st in merged:
            A = _eval_spinterm(st, self.tn, no, no + nv, names)
            B = residual_einsum(spinterm_to_algebraterm(st, set(ext)),
                                nos, nvs, tensors=spatial)
            if np.abs(A - B).max() > 1e-10:
                bad += 1
        self.assertEqual(bad, 0, f"{bad}/{len(merged)} merged terms: bridge != eval")

    def test_p21_failures_partition_into_spin_and_layout_rank6(self):
        # P2.1 (RE-SCOPED). The original P2.1 tested a single hypothesis (only
        # mixed-spin-summed terms fail) and DISPROVED it as xfail. The re-scoped
        # gate asserts the COMPLETE model: every failing rank-6 bridge term is
        # explained by at least one of exactly two mechanisms --
        #   SPIN   (`_mech_spin`): the bridge drops per-index spin, reading the
        #          wrong spatial block / summing the wrong channel; and
        #   LAYOUT (`_mech_layout`): the bridge's free-index order transposes the
        #          canonical residual axes (a value-identical output permutation).
        # 0 unexplained proves the inventory is exhaustive -- the precondition for
        # the fix (encode summed/free-index spin AND canonicalize the output
        # layout). Measured now: 595 both, 116 layout-only, 7 spin-only, 0 other.
        import numpy as np
        from ccgen.spin import spinterm_to_algebraterm
        from ccgen.tests.residual_eval import residual_einsum
        no, nv = self.no, self.nv
        nos, nvs = no // 2, nv // 2
        names = ["a", "b", "c", "i", "j", "k"]
        merged, spatial, ext = self._rank6_bridge_spatial()
        unexplained = []
        for st in merged:
            at = spinterm_to_algebraterm(st, set(ext))
            A = _eval_spinterm(st, self.tn, no, no + nv, names)
            B = residual_einsum(at, nos, nvs, tensors=spatial)
            if np.abs(A - B).max() > 1e-10:
                if not (self._mech_spin(st, ext) or self._mech_layout(at, names)):
                    unexplained.append(
                        [(f.name, "".join(si.spin for si in f.indices),
                          "".join(si.name for si in f.indices))
                         for f in st.factors])
        self.assertEqual(unexplained, [],
                         f"{len(unexplained)} failing terms fit neither the spin "
                         f"nor the layout mechanism, e.g. {unexplained[:3]}")

    def test_p22_layout_mechanism_is_fixed_rank6(self):
        # P2.2 (post-fix regression gate). R3.1.2 half (ii) canonicalizes the
        # bridge's `free_indices` (name-sorted within each space; occ-first
        # between spaces to match the C++ runtime -- see the note in
        # spinterm_to_algebraterm). `residual_einsum` re-splits by space so its
        # output stays the canonical [a,b,c,i,j,k] (vir+occ) layout regardless.
        # This must eliminate the LAYOUT mechanism entirely: no failing term is
        # `_mech_layout`, AND every remaining failure is a spin error (so the two
        # mechanisms are now disjoint -- layout resolved, only spin left).
        # Pre-fix this was 718 failures (595 both, 116 layout-only); post-fix it
        # is 52, all spin-only. Guards against a regression that reintroduces a
        # per-term output ordering.
        import numpy as np
        from ccgen.spin import spinterm_to_algebraterm
        from ccgen.tests.residual_eval import residual_einsum
        no, nv = self.no, self.nv
        nos, nvs = no // 2, nv // 2
        names = ["a", "b", "c", "i", "j", "k"]
        merged, spatial, ext = self._rank6_bridge_spatial()
        for st in merged:
            at = spinterm_to_algebraterm(st, set(ext))
            # the bridge output layout is now always canonical
            self.assertFalse(self._mech_layout(at, names),
                             f"bridge still emits a non-canonical layout: "
                             f"{self._bridge_output_layout(at)}")
            A = _eval_spinterm(st, self.tn, no, no + nv, names)
            B = residual_einsum(at, nos, nvs, tensors=spatial)
            if np.abs(A - B).max() > 1e-10:
                # every surviving failure must be a spin error
                self.assertTrue(self._mech_spin(st, ext),
                                f"a non-spin term still fails after the layout fix: "
                                f"{[(f.name, f.block) for f in st.factors]}")


if __name__ == "__main__":
    unittest.main()


class U11BlockResolvedFactorNamesTests(unittest.TestCase):
    """U1.1 -- under UCC every block-stored amplitude factor must reach the emit
    layer with its OWN name and its OWN slot order.

    Two mechanisms conspire against that, and both were measured on the tree
    before this gate existed (see docs/CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md):

    1. `_factor_tensor_name`'s gate is `len(block) >= 8`, so t1/t2/t3 can never
       receive a block tag -- `t2[aaaa]` and `t2[bbbb]` both emit as bare `t2`.
    2. `_canonicalize_amplitude_factor` reorders every rank >= 2 amplitude onto
       one reference layout, because the RCC bridge drops the spin label and all
       blocks must index one stored spatial tensor. Under UCC they are separate
       arrays, so that reordering reads the wrong array's layout.

    Naming alone is not enough: fixing (1) without (2) leaves the name right and
    the slots permuted. These assertions need no solve and no C++.
    """

    @staticmethod
    def _t2(block, spins):
        from ccgen.spin import SpinFactor, SpinIndex
        bases = (make_occ("i"), make_occ("j"), make_vir("a"), make_vir("b"))
        return SpinFactor(name="t2", block=block,
                          indices=tuple(SpinIndex(b, s) for b, s in zip(bases, spins)))

    def _bridged_names(self, factor):
        from ccgen.spin import SpinTerm, ucc_spinterm_to_algebraterm
        from fractions import Fraction
        term = SpinTerm(coeff=Fraction(1), external_block=factor.block,
                        factors=(factor,))
        at = ucc_spinterm_to_algebraterm(term, {"i", "j", "a", "b"})
        return [f.name for f in at.factors], [t.indices for t in at.factors]

    def test_same_spin_blocks_get_distinct_names(self):
        """t2aa and t2bb are different arrays under UCC; a bare `t2` for both
        silently sums them into one tensor."""
        aa_names, _ = self._bridged_names(self._t2("aaaa", "aaaa"))
        bb_names, _ = self._bridged_names(self._t2("bbbb", "bbbb"))
        self.assertNotEqual(aa_names, bb_names,
                            "t2[aaaa] and t2[bbbb] emit the same tensor name")
        for names in (aa_names, bb_names):
            self.assertTrue(all("_" in n for n in names),
                            f"UCC factor emitted without a block tag: {names}")

    def test_mixed_block_slots_are_not_reordered(self):
        """`baba` must keep its own slot order. The RCC canonicalizer maps it to
        [j,i,b,a] so it can index the single stored `abab` tensor -- correct
        there, wrong when `baba`'s own array is what is being read."""
        _, idx = self._bridged_names(self._t2("baba", "baba"))
        self.assertEqual([i.name for i in idx[0]], ["i", "j", "a", "b"],
                         "UCC bridge permuted a block's slots onto another layout")

    def test_rcc_bridge_is_unchanged(self):
        """The RCC path must stay byte-identical: same names, same slot order."""
        from ccgen.spin import SpinTerm, spinterm_to_algebraterm
        from fractions import Fraction
        f = self._t2("abab", "abab")
        term = SpinTerm(coeff=Fraction(1), external_block="abab", factors=(f,))
        at = spinterm_to_algebraterm(term, {"i", "j", "a", "b"})
        self.assertEqual([t.name for t in at.factors], ["t2"],
                         "RCC bridge naming changed")


class U10UccAdaptEntryTests(unittest.TestCase):
    """U1.0 -- the no-collapse driver: one residual per UCC block, keyed
    `{target}_{tag}`.

    RCC drives `_adapt_on_block` once per *Sz sector* and runs three collapse
    steps that fold spin blocks into one spatial tensor. UCC drives the same
    integrate/merge/bridge pipeline once per *stored block* from
    `ucc_independent_blocks` and runs none of them.
    """

    def _eqs(self, method="ccsd"):
        return generate_cc_equations(method, engine="diagram", canonical_fock=True)

    def test_every_ucc_block_gets_a_nonempty_residual(self):
        """A block that integrates to nothing is a routing bug, not a physical
        result -- the same failure mode spin_adapt_equations guards with its
        'adapted to ZERO' raise."""
        from ccgen.spin import ucc_adapt_equations
        out = ucc_adapt_equations(self._eqs())
        self.assertIn("doubles_aaaa", out)
        self.assertIn("doubles_abab", out)
        self.assertIn("doubles_bbbb", out)
        for key, terms in out.items():
            self.assertTrue(terms, f"UCC target {key!r} adapted to zero terms")

    def test_same_spin_blocks_have_equal_term_counts(self):
        """alpha<->beta is a symmetry of the equations, so aaaa and bbbb must
        produce the same number of terms. Cheap, and it catches a block-routing
        bug that a single-block smoke test cannot."""
        from ccgen.spin import ucc_adapt_equations
        out = ucc_adapt_equations(self._eqs())
        self.assertEqual(len(out["doubles_aaaa"]), len(out["doubles_bbbb"]),
                         "aaaa/bbbb term counts differ -- blocks are misrouted")
        self.assertEqual(len(out["singles_aa"]), len(out["singles_bb"]),
                         "singles aa/bb term counts differ")

    def test_factor_names_are_block_resolved(self):
        """U1.0 must bridge through the UCC bridge (U1.1), not the RCC one."""
        from ccgen.spin import ucc_adapt_equations
        out = ucc_adapt_equations(self._eqs())
        for key, terms in out.items():
            if key.startswith("energy"):
                continue
            for t in terms:
                for f in t.factors:
                    if f.name.startswith("t"):
                        self.assertIn("_", f.name,
                                      f"{key}: bare amplitude name {f.name!r}")

    def test_rcc_is_untouched(self):
        """spin_adapt_equations must be byte-identical -- UCC is a sibling, not a
        modification."""
        from ccgen.spin import spin_adapt_equations
        eqs = self._eqs()
        adapted = spin_adapt_equations(eqs)
        self.assertEqual(sorted(adapted.keys()), ["doubles", "energy", "singles"])
        self.assertEqual({k: len(v) for k, v in adapted.items()},
                         {"singles": 30, "doubles": 113, "energy": 4})


class F1UccRandomTensorsTests(unittest.TestCase):
    """F1 -- the spin-resolved tensor bundle for the UCC numeric gate.

    The RCC fixture (`random_tensors`) carries ONE (no, nv) pair and one
    spin-free `v`. UCC needs per-spin spaces of DIFFERENT sizes (CH3/STO-3G:
    noa=5 nva=4, nob=4 nvb=5) and per-block ERIs, so it gets its own bundle
    rather than a parameter on the shared one -- seven consumers call the RCC
    fixture and its signature must not move.

    Layout contract, read off the bridge's own output: amplitudes are
    (vir..., occ...) like RCC, and a block tag's FIRST half indexes the virtual
    slots, its SECOND half the occupied ones.
    """

    DIMS = dict(noa=5, nva=4, nob=4, nvb=5)   # non-square in both spins

    def test_amplitude_shapes_match_the_block_tag(self):
        from ccgen.tests.residual_eval import ucc_random_tensors
        t = ucc_random_tensors(**self.DIMS, seed=0)
        noa, nva, nob, nvb = (self.DIMS[k] for k in ("noa", "nva", "nob", "nvb"))
        self.assertEqual(t["t1_aa"].shape, (nva, noa))
        self.assertEqual(t["t1_bb"].shape, (nvb, nob))
        self.assertEqual(t["t2_aaaa"].shape, (nva, nva, noa, noa))
        self.assertEqual(t["t2_abab"].shape, (nva, nvb, noa, nob))
        self.assertEqual(t["t2_bbbb"].shape, (nvb, nvb, nob, nob))

    def test_same_spin_blocks_are_antisymmetric(self):
        """aaaa/bbbb are antisymmetric within bra and within ket independently."""
        import numpy as np
        from ccgen.tests.residual_eval import ucc_random_tensors
        t = ucc_random_tensors(**self.DIMS, seed=0)
        for name in ("t2_aaaa", "t2_bbbb"):
            a = t[name]
            self.assertLess(np.abs(a + a.transpose(1, 0, 2, 3)).max(), 1e-14,
                            f"{name} not antisymmetric in its virtual pair")
            self.assertLess(np.abs(a + a.transpose(0, 1, 3, 2)).max(), 1e-14,
                            f"{name} not antisymmetric in its occupied pair")

    def test_mixed_block_is_NOT_antisymmetrized(self):
        """t2_abab's two halves are different spin spaces, so swapping them is
        not a symmetry. Antisymmetrizing it is the easiest way to build a fixture
        that silently disagrees with PySCF -- and on a non-square case the
        transpose is not even shape-legal, which is what makes this assertable."""
        import numpy as np
        from ccgen.tests.residual_eval import ucc_random_tensors
        t = ucc_random_tensors(**self.DIMS, seed=0)
        a = t["t2_abab"]
        # non-square in BOTH pairs, so a spin-swap transpose is not shape-legal --
        # which is precisely why an accidental antisymmetrization cannot hide here.
        self.assertNotEqual(a.shape[0], a.shape[1],
                            "fixture dims must be non-square to make this check bite")
        self.assertNotEqual(a.shape[2], a.shape[3])
        self.assertGreater(np.abs(a).max(), 0.0, "t2_abab is all zeros")
        # the aa/bb blocks ARE antisymmetric; abab must not have inherited it by
        # being built from the same code path.
        self.assertGreater(np.abs(a + a.transpose(0, 1, 2, 3)).max(), 0.0)

    def test_eri_blocks_have_the_right_shapes_and_symmetry(self):
        """UCC needs per-block ERIs. <pq||rs> antisymmetry holds WITHIN a spin
        (aaaa/bbbb); the mixed block only carries bra<->ket."""
        import numpy as np
        from ccgen.tests.residual_eval import ucc_random_tensors
        t = ucc_random_tensors(**self.DIMS, seed=0)
        na = self.DIMS["noa"] + self.DIMS["nva"]
        nb = self.DIMS["nob"] + self.DIMS["nvb"]
        self.assertEqual(t["v_aaaa"].shape, (na, na, na, na))
        self.assertEqual(t["v_bbbb"].shape, (nb, nb, nb, nb))
        self.assertEqual(t["v_abab"].shape, (na, nb, na, nb))
        for name in ("v_aaaa", "v_bbbb"):
            v = t[name]
            self.assertLess(np.abs(v + v.transpose(1, 0, 2, 3)).max(), 1e-12,
                            f"{name} missing bra antisymmetry")
            self.assertLess(np.abs(v - v.transpose(2, 3, 0, 1)).max(), 1e-12,
                            f"{name} missing bra<->ket symmetry")
        vab = t["v_abab"]
        self.assertLess(np.abs(vab - vab.transpose(2, 3, 0, 1)).max(), 1e-12,
                        "v_abab missing bra<->ket symmetry")

    def test_fock_is_per_spin(self):
        from ccgen.tests.residual_eval import ucc_random_tensors
        t = ucc_random_tensors(**self.DIMS, seed=0)
        na = self.DIMS["noa"] + self.DIMS["nva"]
        nb = self.DIMS["nob"] + self.DIMS["nvb"]
        self.assertEqual(t["f_aa"].shape, (na, na))
        self.assertEqual(t["f_bb"].shape, (nb, nb))

    def test_rcc_fixture_is_unchanged(self):
        """The seven existing consumers must see byte-identical output."""
        import numpy as np
        from ccgen.tests.residual_eval import random_tensors
        t = random_tensors(3, 4, seed=0)
        self.assertEqual(sorted(t.keys()), ["f", "t1", "t2", "t3", "t4", "v"])
        self.assertEqual(t["t2"].shape, (4, 4, 3, 3))
        self.assertLess(np.abs(t["v"] - t["v"].transpose(2, 3, 0, 1)).max(), 1e-12)


class F20bBlockTaggedEriTests(unittest.TestCase):
    """F2.0b -- v/f carry their spin block on the UCC path, and the emitter
    accepts the tag.

    U1.1 block-tagged the amplitudes; v/f were left bare, so the evaluator has no
    way to pick between v_aaaa / v_abab / v_bbbb (different shapes). Inference
    from term context is not available: on doubles_abab, 51 of 82 terms have a
    v/f index that appears on no amplitude factor.

    The emitter dispatches on the EXACT strings "v" and "f", while its amplitude
    branch already tolerates a `_<tag>` suffix. So tagging v/f without loosening
    those two branches would emit a name that falls through to
    NotImplementedError -- which is why the two edits are one step.
    """

    def _ucc(self, method="ccsd"):
        from ccgen.spin import ucc_adapt_equations
        return ucc_adapt_equations(
            generate_cc_equations(method, engine="diagram", canonical_fock=True))

    def test_ucc_eri_and_fock_factors_carry_a_block_tag(self):
        out = self._ucc()
        bare = set()
        tagged = set()
        for key, terms in out.items():
            for t in terms:
                for f in t.factors:
                    root = f.name.split("_", 1)[0]
                    if root in ("v", "f"):
                        (tagged if "_" in f.name else bare).add(f.name)
        self.assertFalse(bare, f"UCC emitted untagged integral factors: {sorted(bare)}")
        self.assertTrue(tagged, "no tagged v/f factors emitted at all")
        # every tag must be drawn from U0's vocabulary (a/b strings only)
        for name in tagged:
            tag = name.split("_", 1)[1]
            self.assertTrue(set(tag) <= {"a", "b"},
                            f"non-spin tag on integral factor: {name!r}")

    def test_rcc_still_emits_bare_v_and_f(self):
        """The superset claim: the RCC bridge is untouched, so its integral
        factors must still be exactly `v` and `f`."""
        from ccgen.spin import spin_adapt_equations
        out = spin_adapt_equations(
            generate_cc_equations("ccsd", engine="diagram", canonical_fock=True))
        seen = set()
        for terms in out.values():
            for t in terms:
                for f in t.factors:
                    if f.name.split("_", 1)[0] in ("v", "f"):
                        seen.add(f.name)
        self.assertEqual(seen, {"v", "f"},
                         f"RCC integral factor names changed: {sorted(seen)}")

    def test_emitter_accepts_a_tagged_eri_and_is_unchanged_on_bare(self):
        """Both halves of the emitter gate: a tagged name must map, and a bare
        name must map EXACTLY as before (byte-identical string)."""
        from ccgen.emit.planck_tensor_cpp import _map_factor
        from ccgen.tensors import Tensor, Index
        ijab = (Index("i", "occ"), Index("j", "occ"),
                Index("a", "vir"), Index("b", "vir"))
        sign_bare, expr_bare = _map_factor(Tensor("v", ijab), None, False)
        sign_tag, expr_tag = _map_factor(Tensor("v_abab", ijab), None, False)
        self.assertEqual(expr_bare, "mo_blocks.oovv(i, j, a, b)",
                         "bare v no longer emits what it emitted before")
        self.assertEqual(sign_bare, sign_tag)
        self.assertEqual(expr_bare, expr_tag,
                         "tagged v must resolve to the same block expression; the "
                         "tag routes storage, it does not change the block")
        oo = (Index("i", "occ"), Index("j", "occ"))
        self.assertEqual(_map_factor(Tensor("f", oo), None, False)[1],
                         "reference.f_oo(i, j)")
        self.assertEqual(_map_factor(Tensor("f_aa", oo), None, False)[1],
                         "reference.f_oo(i, j)")


class F21FactorResolutionTests(unittest.TestCase):
    """F2.1 -- resolve ONE factor to its array: pick the block by name, slice
    each axis by (space, spin).

    Isolated from einsum and from terms on purpose. A slice-assignment bug lives
    exactly here, and debugging it through a full residual is what the scope
    warns against.

    The tag is POSITIONAL: character k of the block tag is slot k's spin,
    independent of that slot's space. Verified on the emitted vocabulary --
    v_abab appears with (occ,occ,vir,vir), (occ,vir,vir,occ), (vir,occ,vir,vir)
    and 10 other space patterns, all sharing the one tag.
    """

    DIMS = dict(noa=5, nva=4, nob=4, nvb=5)   # non-square, and noa != nob

    def _tensors(self):
        from ccgen.tests.residual_eval import ucc_random_tensors
        return ucc_random_tensors(**self.DIMS, seed=0)

    def _resolve(self, name, spaces):
        from ccgen.tests.residual_eval import ucc_resolve_factor
        from ccgen.tensors import Tensor, Index
        idx = tuple(Index(f"x{k}", sp) for k, sp in enumerate(spaces))
        return ucc_resolve_factor(Tensor(name, idx), self._tensors(), self.DIMS)

    def test_amplitude_blocks_resolve_to_their_own_shape(self):
        noa, nva, nob, nvb = (self.DIMS[k] for k in ("noa", "nva", "nob", "nvb"))
        self.assertEqual(self._resolve("t1_aa", ("vir", "occ")).shape, (nva, noa))
        self.assertEqual(self._resolve("t1_bb", ("vir", "occ")).shape, (nvb, nob))
        self.assertEqual(
            self._resolve("t2_abab", ("vir", "vir", "occ", "occ")).shape,
            (nva, nvb, noa, nob))
        self.assertEqual(
            self._resolve("t2_bbbb", ("vir", "vir", "occ", "occ")).shape,
            (nvb, nvb, nob, nob))

    def test_eri_slice_follows_space_AND_spin_per_slot(self):
        """v_abab with (occ,vir,occ,vir) must slice [occ_a, vir_b, occ_a, vir_b]
        of the v_abab block -- alpha slots from the alpha space, beta from beta."""
        noa, nva, nob, nvb = (self.DIMS[k] for k in ("noa", "nva", "nob", "nvb"))
        got = self._resolve("v_abab", ("occ", "vir", "occ", "vir"))
        self.assertEqual(got.shape, (noa, nvb, noa, nvb))
        # and the same block under a different space pattern
        got2 = self._resolve("v_abab", ("occ", "occ", "vir", "vir"))
        self.assertEqual(got2.shape, (noa, nob, nva, nvb))

    def test_same_spin_eri_uses_its_own_space(self):
        noa, nva, nob, nvb = (self.DIMS[k] for k in ("noa", "nva", "nob", "nvb"))
        self.assertEqual(
            self._resolve("v_aaaa", ("occ", "occ", "vir", "vir")).shape,
            (noa, noa, nva, nva))
        self.assertEqual(
            self._resolve("v_bbbb", ("occ", "vir", "vir", "occ")).shape,
            (nob, nvb, nvb, nob))

    def test_fock_blocks(self):
        noa, nva, nob, nvb = (self.DIMS[k] for k in ("noa", "nva", "nob", "nvb"))
        self.assertEqual(self._resolve("f_aa", ("occ", "occ")).shape, (noa, noa))
        self.assertEqual(self._resolve("f_bb", ("vir", "vir")).shape, (nvb, nvb))
        self.assertEqual(self._resolve("f_aa", ("occ", "vir")).shape, (noa, nva))

    def test_slice_values_match_a_hand_written_index(self):
        """Shape agreement is necessary, not sufficient: a wrong offset keeps the
        shape. Pin the actual elements against an explicit slice."""
        import numpy as np
        t = self._tensors()
        noa, nvb = self.DIMS["noa"], self.DIMS["nvb"]
        got = self._resolve("v_abab", ("occ", "vir", "occ", "vir"))
        ref = t["v_abab"][0:noa, self.DIMS["nob"]:, 0:noa, self.DIMS["nob"]:]
        self.assertLess(np.abs(got - ref).max(), 1e-15)

    def test_unknown_block_raises(self):
        """A missing block must fail loudly, not fall back to a spin-free array."""
        with self.assertRaises((KeyError, ValueError)):
            self._resolve("v_aabb", ("occ", "occ", "vir", "vir"))


class F22aTermSpinMapTests(unittest.TestCase):
    """F2.2a -- the per-term spin map: {index_name: 'a'|'b'}, read positionally
    off each factor's block tag.

    Measured before writing this, on the emitted manifolds:

        ccsd     328 terms   0 spin-clashes   0 untagged
        ccsdt   2490 terms   0 spin-clashes   0 untagged
        ccsdtq 18137 terms   0 spin-clashes   0 untagged   0 tag-length mismatches

    So the map is well-defined today at every rank the generator reaches. The
    clash and tag-length branches are asserted anyway: they are the invariant the
    rest of F2.2 rests on, and rank 8 -- where RCC's beta-majority fold lost a
    sector -- is exactly where a future change would break them first.

    Note UCC cannot repeat that RCC defect by construction: ucc_independent_blocks
    enumerates every alpha-count sector separately (rank 8: five blocks, including
    aaabaaab and the beta-majority abbbabbb) rather than folding them.
    """

    def _ucc(self, method):
        from ccgen.spin import ucc_adapt_equations
        return ucc_adapt_equations(
            generate_cc_equations(method, engine="diagram", canonical_fock=True))

    def test_map_is_total_over_every_index(self):
        from ccgen.spin import ucc_term_spins
        out = self._ucc("ccsd")
        for key, terms in out.items():
            for t in terms:
                spins = ucc_term_spins(t)
                for idx in list(t.free_indices) + list(t.summed_indices):
                    self.assertIn(idx.name, spins,
                                  f"{key}: index {idx.name!r} has no spin")
                    self.assertIn(spins[idx.name], ("a", "b"))

    def test_free_index_spins_match_the_target_block(self):
        """doubles_abab's free indices must resolve to (a,b,a,b) -- the map has to
        agree with the block the residual is keyed by, or the output shape is wrong."""
        from ccgen.spin import ucc_term_spins
        out = self._ucc("ccsd")
        for key in ("doubles_aaaa", "doubles_abab", "doubles_bbbb"):
            tag = key.split("_", 1)[1]
            for t in out[key]:
                spins = ucc_term_spins(t)
                # free indices are occ-first; the target tag is bra-half then
                # ket-half, so compare as a multiset per space
                occ = [spins[i.name] for i in t.free_indices if i.space == "occ"]
                vir = [spins[i.name] for i in t.free_indices if i.space == "vir"]
                n = len(tag) // 2
                self.assertEqual(sorted(occ), sorted(tag[n:]),
                                 f"{key}: occ free-index spins {occ} != {tag[n:]}")
                self.assertEqual(sorted(vir), sorted(tag[:n]),
                                 f"{key}: vir free-index spins {vir} != {tag[:n]}")

    def test_raises_when_one_index_would_get_two_spins(self):
        """Unreachable on today's equations (0/18137 at ccsdtq). Asserted because
        it is the invariant F2.2b/c rest on -- a silent last-write-wins here would
        surface as a wrong slice much later."""
        from ccgen.spin import ucc_term_spins
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import Tensor, Index
        from fractions import Fraction
        i, a = Index("i", "occ"), Index("a", "vir")
        clash = AlgebraTerm(
            Fraction(1),
            (Tensor("t1_aa", (a, i)), Tensor("t1_bb", (a, i))),  # 'a' is both spins
            (), (a, i), True)
        with self.assertRaises(ValueError):
            ucc_term_spins(clash)

    def test_raises_on_tag_length_mismatch(self):
        """A tag whose length differs from the slot count is an off-by-one in the
        naming, and rank 8 (eight-character tags) is where it would first appear."""
        from ccgen.spin import ucc_term_spins
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import Tensor, Index
        from fractions import Fraction
        i, a = Index("i", "occ"), Index("a", "vir")
        bad = AlgebraTerm(Fraction(1), (Tensor("t1_aaa", (a, i)),), (), (a, i), True)
        with self.assertRaises(ValueError):
            ucc_term_spins(bad)

    def test_raises_on_an_untagged_factor(self):
        from ccgen.spin import ucc_term_spins
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import Tensor, Index
        from fractions import Fraction
        i, a = Index("i", "occ"), Index("a", "vir")
        bare = AlgebraTerm(Fraction(1), (Tensor("t1", (a, i)),), (), (a, i), True)
        with self.assertRaises(ValueError):
            ucc_term_spins(bare)


class F22bcUccResidualEinsumTests(unittest.TestCase):
    """F2.2b+c -- assemble the operands and lay out the result.

    Scoped as two steps, landed as one: F2.2c is a single line of free-index
    ordering INSIDE F2.2b's function body, with no state between them where a
    separate gate would localize anything.

    Dims are non-square AND asymmetric -- noa != nva, nob != nvb, and noa != nob.
    A square case hides a transposed axis (the trap recorded in
    CCGEN_RANK3_KERNEL_AND_SOLVER.md); a spin-symmetric one additionally hides a
    swapped alpha/beta slot, which is the failure this step can introduce.
    """

    DIMS = dict(noa=5, nva=4, nob=4, nvb=5)

    def _tensors(self):
        from ccgen.tests.residual_eval import ucc_random_tensors
        return ucc_random_tensors(**self.DIMS, seed=0)

    def _ucc(self, method="ccsd"):
        from ccgen.spin import ucc_adapt_equations
        return ucc_adapt_equations(
            generate_cc_equations(method, engine="diagram", canonical_fock=True))

    def _eval(self, terms):
        from ccgen.tests.residual_eval import ucc_residual_einsum
        tensors, out = self._tensors(), None
        for t in terms:
            r = ucc_residual_einsum(t, self.DIMS, tensors)
            out = r if out is None else out + r
        return out

    def test_one_code_path_yields_different_shapes_per_block(self):
        """The point of the whole step: doubles_abab and doubles_bbbb go through
        identical code and come out different shapes, each axis sized from its own
        index's spin."""
        noa, nva, nob, nvb = (self.DIMS[k] for k in ("noa", "nva", "nob", "nvb"))
        out = self._ucc()
        self.assertEqual(self._eval(out["doubles_abab"]).shape, (nva, nvb, noa, nob))
        self.assertEqual(self._eval(out["doubles_bbbb"]).shape, (nvb, nvb, nob, nob))
        self.assertEqual(self._eval(out["doubles_aaaa"]).shape, (nva, nva, noa, noa))
        self.assertEqual(self._eval(out["singles_aa"]).shape, (nva, noa))
        self.assertEqual(self._eval(out["singles_bb"]).shape, (nvb, nob))

    def test_layout_is_virtuals_first_like_the_rcc_evaluator(self):
        """F2.2c. The output layout must be R[vir_ext..., occ_ext...], the same
        convention residual_einsum uses -- F2.3's oracle compares the two arrays
        directly, so an occ-first layout would make every element disagree.

        Shape alone does NOT pin this: a doubles block is (vir,vir,occ,occ) and
        the occ-first swap gives (occ,occ,vir,vir), which at spin-symmetric dims
        can collide in shape. So this asserts on VALUES, against an independently
        constructed einsum, on a term whose four free-index axes have four
        DISTINCT lengths.
        """
        import string
        import numpy as np
        from ccgen.tests.residual_eval import ucc_residual_einsum, ucc_resolve_factor

        from ccgen.tests.residual_eval import ucc_random_tensors
        out = self._ucc()
        # the class DIMS give free axes (4,5,5,4) -- not distinct enough to pin a
        # permuted layout by shape, so use dims where all four differ
        dims = dict(noa=6, nva=3, nob=5, nvb=4)
        tensors = ucc_random_tensors(**dims, seed=1)

        for t in out["doubles_abab"][:12]:
            with self.subTest(term=str(t)[:60]):
                letters, pool = {}, iter(string.ascii_lowercase + string.ascii_uppercase)
                for idx in list(t.free_indices) + list(t.summed_indices):
                    letters[idx] = next(pool)
                subs = ["".join(letters[i] for i in f.indices)
                        for f in t.factors]
                ops = [ucc_resolve_factor(f, tensors, dims) for f in t.factors]
                ext = [i for i in t.free_indices if i.space == "vir"] + \
                      [i for i in t.free_indices if i.space == "occ"]
                want = np.einsum(
                    ",".join(subs) + "->" + "".join(letters[i] for i in ext),
                    *ops, optimize=True) * float(t.coeff)
                got = ucc_residual_einsum(t, dims, tensors)
                self.assertEqual(got.shape, want.shape)
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)

    def test_free_axes_are_sized_from_their_own_spin(self):
        """The four free axes of doubles_abab must take four DIFFERENT lengths at
        dims where noa/nva/nob/nvb are all distinct. This is what a swapped
        alpha/beta slot breaks, and what the symmetric fixture cannot see."""
        from ccgen.tests.residual_eval import ucc_random_tensors, ucc_residual_einsum
        dims = dict(noa=6, nva=3, nob=5, nvb=4)
        tensors = ucc_random_tensors(**dims, seed=1)
        acc = None
        for t in self._ucc()["doubles_abab"]:
            r = ucc_residual_einsum(t, dims, tensors)
            acc = r if acc is None else acc + r
        self.assertEqual(acc.shape, (3, 4, 6, 5))   # (nva, nvb, noa, nob)

    def test_full_manifold_evaluates_without_raising(self):
        """F2.2d -- every term of every target on F1's fixture. Explicitly NOT a
        correctness gate: a wrong slice that keeps its shape passes here. F2.3's
        closed-shell oracle is what catches that, and this split exists so a
        failure localizes to assembly rather than to physics."""
        import numpy as np
        noa, nva, nob, nvb = (self.DIMS[k] for k in ("noa", "nva", "nob", "nvb"))
        expected = {
            "energy": (),
            "singles_aa": (nva, noa), "singles_bb": (nvb, nob),
            "doubles_aaaa": (nva, nva, noa, noa),
            "doubles_abab": (nva, nvb, noa, nob),
            "doubles_bbbb": (nvb, nvb, nob, nob),
        }
        out = self._ucc()
        self.assertEqual(set(out), set(expected))
        for key, terms in out.items():
            with self.subTest(target=key):
                got = self._eval(terms)
                self.assertEqual(np.asarray(got).shape, expected[key])
                self.assertTrue(np.all(np.isfinite(got)))

    def test_rejects_a_term_whose_spins_disagree(self):
        """The evaluator calls ucc_term_spins for its consistency check even
        though it does not slice by it -- so an unevaluable term is refused here,
        not silently contracted into a plausible-looking array."""
        from fractions import Fraction
        from ccgen.project import AlgebraTerm
        from ccgen.tensors import Tensor, Index
        from ccgen.tests.residual_eval import ucc_residual_einsum
        # Shape-legal on purpose: at noa == nob and nva == nvb the two blocks
        # contract without complaint, so einsum CANNOT catch this. Only the
        # ucc_term_spins call refuses it -- which is why the check earns its
        # place in a function that does not slice by the map.
        dims = dict(noa=5, nva=4, nob=5, nvb=4)
        from ccgen.tests.residual_eval import ucc_random_tensors
        i, a = Index("i", "occ"), Index("a", "vir")
        clash = AlgebraTerm(
            Fraction(1),
            (Tensor("t1_aa", (a, i)), Tensor("t1_bb", (a, i))),
            (), (a, i), True)
        with self.assertRaises(ValueError):
            ucc_residual_einsum(clash, dims, ucc_random_tensors(**dims, seed=0))


class F23ClosedShellOracleTests(unittest.TestCase):
    """F2.3 -- the load-bearing gate: at closed shell the UCC residual must
    reproduce the EXISTING RCC residual for the same equations.

    Cheap, and it needs no PySCF, no open-shell reference and no converged
    amplitudes -- both sides compute the same physical quantity by different
    routes, so a slice-assignment or block-routing error shows up immediately.

    Two things the scope originally got wrong about this, both found by
    prototyping the comparison before writing it:

    1. It is a PER-TARGET PAIRING, not a sum over blocks. RCC adapts on the
       closed-shell representative external block (spin.py, the S3/R1.0 note),
       which for doubles is `abab`. So `doubles_abab` pairs with `doubles`, and
       `doubles_aaaa` / `doubles_bbbb` have NO RCC counterpart at all --
       collapse_amplitudes splits the all-alpha sector away rather than storing
       it. Those two are covered by F2.2d's shape smoke and by F3, not here.

    2. It is not free, and F1's fixture cannot serve it. The UCC blocks must be
       built FROM the spatial ones through the closure relations -- see
       ucc_closed_shell_tensors, which exists for exactly this.

    Tolerance is 1e-11, not 1e-12: the measured `doubles` difference is ~1.8e-12
    over a residual of norm ~1.6e3, so the tighter bound would flake on a
    contraction this deep.
    """

    PAIRS = (("energy", "energy"),
             ("singles_aa", "singles"),
             ("doubles_abab", "doubles"))

    NO, NV = 5, 4          # non-square: a square case hides a transposed axis
    ATOL = 1e-11

    def _manifolds(self):
        from ccgen.spin import ucc_adapt_equations, spin_adapt_equations
        eqs = generate_cc_equations("ccsd", engine="diagram", canonical_fock=True)
        return ucc_adapt_equations(eqs), spin_adapt_equations(eqs)

    def _compare(self, ucc_blocks, spatial):
        """Return {pair: max abs difference} for the three paired targets."""
        import numpy as np
        from ccgen.tests.residual_eval import ucc_residual_einsum, residual_einsum
        u_eqs, r_eqs = self._manifolds()
        dims = dict(noa=self.NO, nva=self.NV, nob=self.NO, nvb=self.NV)
        out = {}
        for ukey, rkey in self.PAIRS:
            U = sum(ucc_residual_einsum(t, dims, ucc_blocks) for t in u_eqs[ukey])
            R = sum(residual_einsum(t, self.NO, self.NV, tensors=spatial)
                    for t in r_eqs[rkey])
            self.assertEqual(np.asarray(U).shape, np.asarray(R).shape,
                             f"{ukey} vs {rkey}: shape disagreement")
            out[(ukey, rkey)] = float(np.abs(np.asarray(U) - np.asarray(R)).max())
        return out

    def test_ucc_reproduces_rcc_at_closed_shell(self):
        from ccgen.tests.residual_eval import ucc_closed_shell_tensors
        ucc_blocks, spatial = ucc_closed_shell_tensors(self.NO, self.NV, seed=7)
        for pair, diff in self._compare(ucc_blocks, spatial).items():
            with self.subTest(pair=pair):
                self.assertLess(diff, self.ATOL,
                                f"{pair[0]} != {pair[1]} at closed shell: {diff:.3e}")

    def test_the_oracle_is_falsifiable(self):
        """Committed WITH the gate, not instead of it. A gate that cannot fail is
        indistinguishable from one that passes, and this one compares two long
        contractions where an accidental agreement is easy to believe in.

        Transposing one axis of t2_abab must break every paired target by
        O(||R||) -- measured 1.06e2 / 1.47e3 and the energy off by 4.4, against
        residual norms ~1.1e3 / 1.6e3.
        """
        from ccgen.tests.residual_eval import ucc_closed_shell_tensors
        ucc_blocks, spatial = ucc_closed_shell_tensors(self.NO, self.NV, seed=7)
        ucc_blocks = dict(ucc_blocks)
        ucc_blocks["t2_abab"] = ucc_blocks["t2_abab"].transpose(1, 0, 2, 3)
        for pair, diff in self._compare(ucc_blocks, spatial).items():
            with self.subTest(pair=pair):
                self.assertGreater(diff, 1.0,
                                   f"{pair[0]}: a corrupted block still agreed "
                                   f"to {diff:.3e} -- the oracle is vacuous")

    def test_the_independent_fixture_does_NOT_satisfy_the_oracle(self):
        """Why ucc_closed_shell_tensors exists at all. F1's ucc_random_tensors
        draws each block independently, so it violates the closure relations and
        the comparison fails -- for a fixture reason, not an evaluator one.

        Asserted so that a future 'simplification' that reuses the F1 fixture
        here fails loudly instead of appearing to weaken the tolerance.
        """
        from ccgen.tests.residual_eval import ucc_random_tensors, random_tensors
        ucc_blocks = ucc_random_tensors(noa=self.NO, nva=self.NV,
                                        nob=self.NO, nvb=self.NV, seed=0)
        spatial = random_tensors(self.NO, self.NV, seed=0)
        diffs = self._compare(ucc_blocks, spatial)
        self.assertTrue(any(d > 1.0 for d in diffs.values()),
                        "the independent fixture agreed; the closure relations "
                        "are apparently not needed, which contradicts the scope")


class U14RankSixSpinFlipSymmetryTests(unittest.TestCase):
    """U1.4 — alpha<->beta symmetry of the UCC residual manifold.

    A CC residual manifold must be invariant under a global spin flip: feed
    alpha-equal-beta tensors and the `_aa` target must equal its `_bb` partner,
    `aaaa` must equal `bbbb`. This needs no PySCF, no converged amplitudes and no
    oracle — it is a symmetry of the equations themselves, and it is the cheapest
    check that exists on a block-resolved manifold. Both ranks pass: ~1e-13 at
    rank 4, ~7e-16 at rank 6.

    **This test was committed for one commit asserting the opposite** — rank 6
    `expectedFailure`, on the conclusion that the landed rank-6 UCC equations were
    defective at ~1e-1 relative. That conclusion was wrong, and the way it was
    wrong is worth keeping, because the false signal was strong and specific:
    singles, doubles AND triples all broke together, at a magnitude far above
    noise, reproducible with no PySCF involved.

    **The actual defect was in this test's own fixture**, in one line. `abbabb` is
    the spin flip of `aabaab`, and the flip is NOT the identity: flipping
    `aab` -> `bba` leaves slots `(b,b,a | b,b,a)`, which must then be re-expressed
    in `abbabb`'s own `(a,b,b | a,b,b)` order — a slot reversal within each half.
    Setting the two blocks equal produces an array that violates `abbabb`'s own
    antisymmetry (it is antisym in vir slots 1,2 and occ 4,5, not 0,1 and 3,4).

    What made it look like an equation defect, and the check that settled it:

    * The manifold IS structurally symmetric — the factor vocabularies of every
      spin-flip pair are exact mirrors, term counts match (`ccsdt`: 579/579
      triples, 469/469 mixed, 25/25 singles), and every emitted factor's slot
      spins equal its own tag (0 mismatches across all 2490 terms).
    * Three hypotheses were falsified before the fixture was suspected at all: a
      wrong block name, the `v_abab` orientation (forcing `v[p,q,r,s]=v[q,p,s,r]`
      left the relative error unchanged), and the evaluator (exact at rank 4, and
      gated against PySCF UCCSD at ~6e-16 in `test_ucc_vs_pyscf`).
    * **The check that would have found it first: does each fixture block satisfy
      its OWN antisymmetry?** `abbabb` is antisym in (1,2)/(4,5); the wrong block
      failed that by 1.5e-2 while the right one gives exactly 0. That is a
      property of one array, needs no equations, and is the same class of check
      F1 already applies to the rank-4 blocks.

    A corollary that also has to be right, and was not: the PySCF `bba` block does
    NOT map to `abbabb` by the same axis permutation `aab` maps to `aabaab`. An
    exhaustive 720-permutation search matched `bba` to `aab` and that match is
    real — but matching `aabaab` is not the same as BEING `abbabb`, and the search
    never checked the target block's symmetry. See the U1.4 section of
    `docs/CCGEN_U1_UCC_ADAPT_SCOPE.md`.
    """
    NO, NV = 4, 3       # non-square

    def _tensors(self):
        import numpy as np
        import itertools
        no, nv, n = self.NO, self.NV, self.NO + self.NV
        rng = np.random.default_rng(1)

        def anti(x, ax):
            out = np.zeros_like(x)
            for p in itertools.permutations(range(len(ax))):
                sg, pl = 1, list(p)
                for i in range(len(pl)):
                    for j in range(i + 1, len(pl)):
                        if pl[i] > pl[j]:
                            sg = -sg
                order = list(range(x.ndim))
                for s, a in enumerate(ax):
                    order[a] = ax[p[s]]
                out = out + sg * x.transpose(order)
            return out

        def a4(x):
            x = x - x.transpose(1, 0, 2, 3)
            return x - x.transpose(0, 1, 3, 2)

        t1 = rng.random((nv, no))
        t2 = rng.random((nv, nv, no, no))
        t2 = t2 + t2.transpose(1, 0, 3, 2)
        t3m = rng.random((nv, nv, nv, no, no, no))
        t3m = t3m - t3m.transpose(1, 0, 2, 3, 4, 5)
        t3m = t3m - t3m.transpose(0, 1, 2, 4, 3, 5)
        t3s = anti(anti(rng.random((nv, nv, nv, no, no, no)), (0, 1, 2)), (3, 4, 5))
        v = rng.random((n, n, n, n))
        v = v + v.transpose(1, 0, 3, 2)
        v = v + v.transpose(2, 3, 0, 1)
        f = rng.random((n, n))
        f = f + f.T
        f[:no, no:] = f[no:, :no] = 0.0        # canonical Fock, as every CC kernel gets
        return {
            "t1_aa": t1, "t1_bb": t1,
            "t2_abab": t2, "t2_aaaa": a4(t2), "t2_bbbb": a4(t2),
            "t3_aaaaaa": t3s, "t3_bbbbbb": t3s,
            # abbabb is the SPIN FLIP of aabaab, and a flip is not the identity
            # here: flipping aab->bba leaves slots (b,b,a|b,b,a), which must be
            # re-expressed in this block's own (a,b,b|a,b,b) order -- a slot
            # reversal within each half. Setting them equal (the first thing
            # tried) gives a block violating abbabb's OWN antisymmetry (antisym
            # in vir slots 1,2 and occ 4,5, not 0,1 / 3,4), and shows up as a
            # ~1e-1 spin-flip asymmetry in the residual.
            "t3_aabaab": t3m, "t3_abbabb": t3m.transpose(2, 1, 0, 5, 4, 3),
            "v_abab": v,
            "v_aaaa": v - v.transpose(0, 1, 3, 2),
            "v_bbbb": v - v.transpose(0, 1, 3, 2),
            "f_aa": f, "f_bb": f,
        }

    def _worst_asymmetry(self, method, pairs):
        import numpy as np
        from ccgen.spin import ucc_adapt_equations
        from ccgen.tests.residual_eval import ucc_residual_einsum
        T = self._tensors()
        dims = dict(noa=self.NO, nva=self.NV, nob=self.NO, nvb=self.NV)
        u = ucc_adapt_equations(
            generate_cc_equations(method, engine="diagram", canonical_fock=True))
        worst = {}
        for a, b in pairs:
            A = np.asarray(sum(ucc_residual_einsum(t, dims, T) for t in u[a]))
            B = np.asarray(sum(ucc_residual_einsum(t, dims, T) for t in u[b]))
            worst[(a, b)] = (float(np.abs(A - B).max()), float(np.abs(A).max()))
        return worst

    def test_rank4_is_spin_flip_symmetric(self):
        """ccsd. Passes — and pins that the fixture and evaluator are not the
        cause of the rank-6 failure below."""
        pairs = (("singles_aa", "singles_bb"), ("doubles_aaaa", "doubles_bbbb"))
        for pair, (diff, scale) in self._worst_asymmetry("ccsd", pairs).items():
            with self.subTest(pair=pair):
                self.assertGreater(scale, 1.0, "residual is ~zero; vacuous")
                self.assertLess(diff / scale, 1e-12,
                                f"{pair[0]} != {pair[1]}: {diff:.3e} vs |R| {scale:.3e}")

    def test_rank6_is_spin_flip_symmetric(self):
        """ccsdt. Passes at ~7e-16 once the fixture's `abbabb` block carries the
        correct flip relation — see `_tensors`. It was an `expectedFailure` for
        exactly one commit, on the belief that the equations were defective; they
        are not, and the record of that is in this class's docstring."""
        pairs = (("singles_aa", "singles_bb"),
                 ("doubles_aaaa", "doubles_bbbb"),
                 ("triples_aaaaaa", "triples_bbbbbb"))
        for pair, (diff, scale) in self._worst_asymmetry("ccsdt", pairs).items():
            with self.subTest(pair=pair):
                self.assertGreater(scale, 1.0, "residual is ~zero; vacuous")
                self.assertLess(diff / scale, 1e-12,
                                f"{pair[0]} != {pair[1]}: {diff:.3e} vs |R| {scale:.3e}")

    def test_every_fixture_block_satisfies_its_own_antisymmetry(self):
        """The check that would have found the `abbabb` defect immediately, and
        did not exist when it was needed.

        A spin block's antisymmetric slot pairs are determined by its tag: slots
        sharing a spin within a half are interchangeable, so the block must be
        antisymmetric in them. `aabaab` is antisym in vir (0,1) and occ (3,4);
        `abbabb` in vir (1,2) and occ (4,5) — DIFFERENT pairs, which is exactly
        what makes setting the two blocks equal wrong.

        This is a property of one array. It needs no equations, no evaluator and
        no oracle, and it localizes to the fixture rather than to the physics.
        """
        import numpy as np
        T = self._tensors()
        for name, tag in (("t3_aaaaaa", "aaaaaa"), ("t3_bbbbbb", "bbbbbb"),
                          ("t3_aabaab", "aabaab"), ("t3_abbabb", "abbabb"),
                          ("t2_aaaa", "aaaa"), ("t2_bbbb", "bbbb")):
            arr, n = T[name], len(tag) // 2
            for half in (0, n):                       # bra half, then ket half
                for p in range(half, half + n - 1):
                    for q in range(p + 1, half + n):
                        if tag[p] != tag[q]:
                            continue                  # different spins: no symmetry
                        order = list(range(len(tag)))
                        order[p], order[q] = order[q], order[p]
                        with self.subTest(block=name, swap=(p, q)):
                            self.assertLess(
                                np.abs(arr + arr.transpose(order)).max(), 1e-12,
                                f"{name}: not antisymmetric in slots {p},{q} — "
                                f"the block violates its own tag")
