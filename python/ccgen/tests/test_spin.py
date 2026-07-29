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


if __name__ == "__main__":
    unittest.main()
