"""U1.4 — the rank-6 UCC residual manifold against PySCF UCCSDT.

The rank-6 sibling of `test_ucc_vs_pyscf` (U1.2), reached after U1.4's four
candidate blockers were checked and dissolved. Same pattern: evaluate every UCC
target at PySCF's own perturbed amplitudes and compare against PySCF's residual,
recovered as ``R = (t_new - t) * D``.

Skipped if PySCF is not importable (it lives in tests/pyscf/.venv). Run with:

    tests/pyscf/.venv/bin/python -m unittest ccgen.tests.test_ucc_rank6_vs_pyscf

**Status: singles and doubles agree at MACHINE PRECISION (~5e-16 / ~1.4e-15).
The triples target disagrees by ~1.2e-2 against a reference of ~3.1 and is marked
`expectedFailure`** — an open discrepancy, not a passing gate.

What the split buys: the rank-6 singles and doubles residuals *consume* t3, and
they are exact, so the t3 blocks handed to ccgen and ccgen's reading of them are
both right. The discrepancy is confined to the T3 equation.

**U1.4c.2 cleared ccgen.** The closed-shell oracle (`U14c2RankSixClosedShellOracleTests`,
`test_spin.py`, no PySCF involved) reproduces ccgen's own RCC residual at rank 6
on PERTURBED amplitudes — triples to 1.5e-12 against ||R||~1e3. So the T3
equations are self-consistent, and this file's job is the INTERFACE.

**Four interface defects were found and fixed**, each of which silently handed
the two sides different amplitudes:

1. **The re-antisymmetrization was unnormalized.** PySCF's `t2aa`/`t3aaa` blocks
   arrive ALREADY antisymmetric, so re-applying `a - a.transpose(...)` does not
   project — it MULTIPLIES, by **4x for t2aa and 36x for t3aaa** (measured). The
   tell was that at PySCF's own converged amplitudes, where its residual is
   ~1e-10, this gate reported ||ref|| = 1.7e-1.
2. **`t2aa`/`t2bb` are not independent of `t2ab`** — PySCF's converged amplitudes
   satisfy `t2aa = t2ab - t2ab.transpose(0,1,3,2)` exactly.
3. **`t3aaa`/`t3bbb` are not independent of `t3aab`** either, via the same-spin
   closure.
4. **A block carrying `aabaab`'s antisymmetries is not thereby a valid amplitude
   block.** Antisymmetrized noise passes every permutation test the real block
   passes — verified exhaustively over all 36 signed occ x vir permutations, the
   real block has exactly the same two symmetries and no others — yet the derived
   `aaa` is then not a valid same-spin block, and PySCF's packed storage
   reprojects it by more than the block itself. Fixed by `_valid_t3_blocks`,
   which builds the perturbation as a slice of a genuine antisymmetric
   spin-orbital tensor.

**t1 and t2 are left at PySCF's converged values** and only t3 is perturbed:
nothing derives t3 from a perturbed t2, so any t2 perturbation reintroduces a
closure violation one rank up. Perturbing t3 alone still drives every target well
off convergence (||R|| ~1e-2 to 0.7 against ~1e-9 at convergence).

**Result: singles and doubles are EXACT (rel ~1e-14). Triples sit at rel ~1.9e-3,
down from 8.8e-2**, and remain `expectedFailure`.

**What that residual now means, which is the point of all four fixes.** Every
consistency relation the fixture can be held to is satisfied to ~1e-17: both
closure forms agree on the perturbed block, both reproduce `t3_aaaaaa`, the
packed round trip is exact, and the reference vanishes to ~1e-11 at convergence.
The 3-term closure assumption is NOT baked into ccgen's equations — that was
checked directly, and the disagreement is unchanged when the fixture is built so
that both closure conventions coincide. So the remaining ~0.2% is a genuine
disagreement about the t3-linear part of the T3 residual between ccgen and PySCF,
with the fixture no longer a candidate explanation.

Ruled out as causes, each measured: the denominator (my `D3` matches PySCF's
`eijkabc` construction to 2.8e-14, and `focka.diagonal()` equals `mo_energy`
exactly, `level_shift = 0`); the packed round-trip (every block survives
`full->tri->full` to <=3.5e-17, so ccgen and PySCF see the same t3); the closure
relations (`abbabb` holds exactly here, and forcing `aaaaaa` to satisfy its
closure leaves the discrepancy unchanged); and antisymmetry (both residuals are
bra- and ket-antisymmetric to ~1e-15, as is their difference).

Also ruled out: not a layout or symmetry artifact (both residuals bra-antisym to
~4e-16, and so is their difference); not a scale factor (elementwise ratio median
0.9969); not the fixture t3 blocks (all four satisfy their tag's antisymmetry,
`aaa == bbb` to 2e-18); not the packing round-trip (bitwise exact).

Three conventions this file carries from U1.2, and two more that rank 6 adds:

* `f_ov` zeroed on BOTH sides; layout is a transpose, not a rename; perturb off
  convergence (U1.2).
* **PySCF's real CCSDT residual entry is `update_amps_uccsdt_tri_`**, which
  mutates `tamps` in place adding `R/D`. `UCCSDT.update_amps` is the *inherited
  CCSD* one and silently omits t3 — it exists and runs, which is the trap.
* **t3 is stored packed**, and `aab`/`bba` are ONE stored sector (equal on the
  converged amplitudes, and the repack keeps only one). Perturbing them
  independently makes PySCF and ccgen see different t3 — measured, it moves the
  *singles* residual from 5e-14 to 8.9e-3. Perturb in full form, re-impose each
  block's antisymmetry, mirror `aab` into `bba`, then repack.
"""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pyscf import gto, scf, ao2mo
    from pyscf.cc import uccsdt
    _HAVE_PYSCF = True
except ImportError:  # pragma: no cover - depends on the pyscf venv
    _HAVE_PYSCF = False

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.spin import ucc_adapt_equations  # noqa: E402
from ccgen.tests.residual_eval import ucc_residual_einsum  # noqa: E402

# Water, not N2. N2/STO-3G at 1.3 A is degenerate enough that PySCF's converged
# `t2ab` is NOT reproducible across processes -- measured 0.125 / 0.184 / 0.193 on
# three runs while e_corr and t2aa stayed put to ~13 digits, i.e. the alpha-beta
# amplitudes carry a gauge freedom the energy does not see. That made the gate's
# own reported difference wander (1.56e-2 .. 1.72e-2) and would make any
# term-level bisect meaningless. Water reproduces to ~12 digits with every block
# non-trivial.
#
# 6-31g, not STO-3G: water/STO-3G has nv=2, so C(2,3)=0 distinct same-spin
# triples and `triples_aaaaaa` is IDENTICALLY ZERO -- the same vacuous-pass trap
# the fixture scope recorded for OH/STO-3G, in a new place. 6-31g gives nv=8, and
# the whole gate still runs in ~1 s.
WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"
BASIS = "6-31g"


def _parity(p):
    s, pl = 1, list(p)
    for i in range(len(pl)):
        for j in range(i + 1, len(pl)):
            if pl[i] > pl[j]:
                s = -s
    return s


def _same_spin_from_mixed(M):
    """The same-spin t3 block from the mixed one: the normalized double
    antisymmetrizer over all 36 signed bra x ket permutations.

    **Two forms of this closure exist and they are equivalent ON VALID BLOCKS.**
    The other is the 3-term form `_split_same_spin_amplitude` implies. On a
    genuine closed-shell spin-orbital t3 -- built from a spatial kernel with
    spin-conserving lines, fully antisymmetrized, then sliced -- both reproduce
    the `aaaaaa` block exactly (1.3e-15 and 2.2e-15), and they agree with each
    other to 1.8e-15.

    They diverge only on inputs that are NOT valid t3 blocks. Carrying
    `aabaab`'s two antisymmetries is not sufficient: on antisymmetrized random
    noise the two forms differ by 0.42 against a norm of 0.37. So
    `three(M) == full36(M)` is the operational test for whether a candidate block
    is a real amplitude block, and it is why `_valid_t3_blocks` builds the
    perturbation from a spin-orbital tensor rather than symmetrizing noise.
    """
    out = np.zeros_like(M)
    for bp in itertools.permutations(range(3)):
        for kp in itertools.permutations(range(3)):
            out += (_parity(bp) * _parity(kp)
                    * M.transpose(tuple(bp) + tuple(3 + x for x in kp)))
    return out / 12.0


def _valid_t3_blocks(noS, nvS, seed, scale):
    """A VALID t3 perturbation: the `aabaab` and `aaaaaa` blocks of a genuine
    fully-antisymmetric closed-shell spin-orbital t3.

    Building the perturbation this way rather than antisymmetrizing random noise
    is what makes the closure relations hold. A block that merely carries
    `aabaab`'s two antisymmetries is NOT a valid amplitude block: the real one
    additionally satisfies a linear relation tying its alpha and beta content,
    and the operational test for it is that the two same-spin closures agree.
    On a real block they agree to ~2e-15; on antisymmetrized noise they differ by
    0.42 against a norm of 0.37, and PySCF's packed storage then reprojects the
    derived `aaa` by more than the block itself.

    Returns `(aabaab, aaaaaa)` in ccgen's `[a,b,c,i,j,k]` layout, scaled.
    """
    rng = np.random.default_rng(seed)
    no, nv = 2 * noS, 2 * nvS
    K = rng.random((nvS, nvS, nvS, noS, noS, noS))
    T = np.zeros((nv, nv, nv, no, no, no))
    idx = np.indices(T.shape)
    spin_ok = ((idx[0] % 2 == idx[3] % 2) & (idx[1] % 2 == idx[4] % 2)
               & (idx[2] % 2 == idx[5] % 2))
    T[spin_ok] = K[idx[0][spin_ok] // 2, idx[1][spin_ok] // 2, idx[2][spin_ok] // 2,
                   idx[3][spin_ok] // 2, idx[4][spin_ok] // 2, idx[5][spin_ok] // 2]
    for axes in ((0, 1, 2), (3, 4, 5)):
        out = np.zeros_like(T)
        for p in itertools.permutations(range(3)):
            order = list(range(6))
            for s, a in enumerate(axes):
                order[a] = axes[p[s]]
            out = out + _parity(p) * T.transpose(order)
        T = out
    ea, eb = list(range(0, nv, 2)), list(range(1, nv, 2))
    oa, ob = list(range(0, no, 2)), list(range(1, no, 2))
    mixed = T[np.ix_(ea, ea, eb, oa, oa, ob)]
    same = T[np.ix_(ea, ea, ea, oa, oa, oa)]
    n = np.abs(mixed).max()
    return mixed * (scale / n), same * (scale / n)


def _anti(x, axes):
    out = np.zeros_like(x)
    for p in itertools.permutations(range(len(axes))):
        sg, pl = 1, list(p)
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                if pl[i] > pl[j]:
                    sg = -sg
        order = list(range(x.ndim))
        for s, a in enumerate(axes):
            order[a] = axes[p[s]]
        out = out + sg * x.transpose(order)
    return out


def _build(seed: int = 0, return_inputs: bool = False):
    """Return ({target: ccgen residual}, {target: pyscf residual}), both in
    ccgen's ``[vir..., occ...]`` layout.

    ``return_inputs`` additionally returns ``(tensors, dims, eqs)`` — the block
    bundle the ccgen side was evaluated on. Used by the term-level bisect of the
    open triples discrepancy; it changes nothing about what the gate computes.
    """
    mol = gto.M(atom=WATER, basis=BASIS, spin=0, verbose=0)
    mol.cart = True
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-13
    rhf.run()
    mf = scf.addons.convert_to_uhf(rhf)
    cc = uccsdt.UCCSDT(mf)
    cc.conv_tol = 1e-11
    cc.max_cycle = 200
    cc.kernel()
    eris = cc.ao2mo()

    noa = nob = int(cc.nocc[0])
    nva = nvb = cc.t1[0].shape[1]
    rng = np.random.default_rng(seed)

    t1 = [x.copy() for x in cc.t1]
    t2 = [x.copy() for x in cc.t2]
    t3 = [x.copy() for x in cc.t3]
    # Perturb by adding noise that carries each block's OWN symmetry, rather than
    # raw noise plus a re-symmetrization of the sum. PySCF's converged amplitudes
    # already satisfy those symmetries, so applying an unnormalized projector to
    # (block + noise) rescales the block itself -- by 4x for t2aa and 36x for
    # t3aaa, measured -- which silently moves the reference off convergence.
    # t1/t2 are left at PySCF's converged values. Perturbing them is what the
    # t3 blocks cannot be made consistent with: t3's same-spin blocks are derived
    # from its own mixed block (below), but nothing derives t3 from a perturbed
    # t2, so any t2 perturbation reintroduces a closure violation one rank up.
    # Perturbing t3 alone is enough -- it drives every target off convergence
    # (measured ||R|| ~0.3-0.45, against ~1e-9 at convergence).
    #
    # t2aa/t2bb are NOT independent of t2ab at closed shell -- PySCF's converged
    # amplitudes satisfy t2aa = t2ab - t2ab.transpose(0,1,3,2) exactly (the same
    # closure F2.3 uses at rank 4). Perturbing them separately breaks a relation
    # the equations require. Perturb the mixed block and DERIVE the same-spin ones.
    # (the relation, kept for the record: t2aa = t2ab - t2ab.transpose(0,1,3,2)
    #  on [i,j,a,b], verified exact on PySCF's converged amplitudes)

    # t3: perturb in FULL form (the packed array has no valid elementwise
    # perturbation), re-impose each block's antisymmetry, then repack.
    full = list(uccsdt.tamps_tri2full_uhf(cc, [x.copy() for x in t3]))
    # aaa/bbb are DERIVED from the mixed block below, not perturbed separately.
    dM, dA = _valid_t3_blocks(noa, nva, seed + 1, 0.02)   # ccgen [a,b,c,i,j,k]
    y = full[1] + dM.transpose(3, 4, 0, 1, 5, 2)          # -> [i,j,a,b,k,c]
    full[1], full[2] = y, y.copy()        # aab and bba are ONE stored sector
    # aaa/bbb are determined by the mixed block through the U1.4c.1 closure --
    # perturbing them independently violates a relation the equations require.
    full[0] = full[0] + dA.transpose(3, 4, 5, 0, 1, 2)
    full[3] = full[0].copy()
    t3[:] = list(uccsdt.tamps_full2tri_uhf(cc, full))

    # f_ov zeroed on the PYSCF side too, before its residual is formed. Planck CC
    # kernels are canonical-Fock by construction so ccgen's f_ov terms are
    # runtime-zero; PySCF's f_ov is SCF convergence noise that update_amps USES.
    # Zeroing only one side is worse than zeroing neither (U1.2's measurement),
    # and here it set a ~7e-10 floor under singles that no amount of tightening
    # conv_tol removed -- tightening made it worse, which is the tell.
    eris.focka = eris.focka.copy()
    eris.fockb = eris.fockb.copy()
    eris.focka[:noa, noa:] = eris.focka[noa:, :noa] = 0.0
    eris.fockb[:nob, nob:] = eris.fockb[nob:, :nob] = 0.0

    before3 = list(uccsdt.tamps_tri2full_uhf(cc, [x.copy() for x in t3]))
    b1 = [x.copy() for x in t1]
    b2 = [x.copy() for x in t2]
    tamps = [t1, t2, t3]
    uccsdt.update_amps_uccsdt_tri_(cc, tamps, eris)
    after3 = list(uccsdt.tamps_tri2full_uhf(cc, tamps[2]))

    ea, eb = eris.focka.diagonal().real, eris.fockb.diagonal().real
    Dia = ea[:noa, None] - ea[None, noa:]
    DIA = eb[:nob, None] - eb[None, nob:]
    oa, va = ea[:noa], ea[noa:]
    D3 = (oa[:, None, None, None, None, None] + oa[None, :, None, None, None, None]
          + oa[None, None, :, None, None, None] - va[None, None, None, :, None, None]
          - va[None, None, None, None, :, None] - va[None, None, None, None, None, :])
    ref = {
        "singles_aa": (tamps[0][0] - b1[0]) * Dia,
        "singles_bb": (tamps[0][1] - b1[1]) * DIA,
        "doubles_aaaa": (tamps[1][0] - b2[0]) * (Dia[:, None, :, None] + Dia[None, :, None, :]),
        "doubles_bbbb": (tamps[1][2] - b2[2]) * (DIA[:, None, :, None] + DIA[None, :, None, :]),
        "triples_aaaaaa": (after3[0] - before3[0]) * D3,
    }

    Ca, Cb = mf.mo_coeff

    def eri_phys(C1, C2):
        g = ao2mo.general(mol, (C1, C1, C2, C2), compact=False).reshape(
            C1.shape[1], C1.shape[1], C2.shape[1], C2.shape[1])
        return g.transpose(0, 2, 1, 3)

    vaa, vbb = eri_phys(Ca, Ca), eri_phys(Cb, Cb)
    fa = (Ca.T @ mf.get_fock()[0] @ Ca).copy()
    fb = (Cb.T @ mf.get_fock()[1] @ Cb).copy()
    fa[:noa, noa:] = fa[noa:, :noa] = 0.0      # canonical Fock, both sides
    fb[:nob, nob:] = fb[nob:, :nob] = 0.0
    aaa, aab, bba, bbb = before3
    M = aab.transpose(2, 3, 5, 0, 1, 4)        # t3_aabaab
    tensors = {
        "t1_aa": b1[0].T, "t1_bb": b1[1].T,
        "t2_aaaa": b2[0].transpose(2, 3, 0, 1),
        "t2_abab": b2[1].transpose(1, 3, 0, 2),    # [i,a,j,b] -> [a,b,i,j]
        "t2_bbbb": b2[2].transpose(2, 3, 0, 1),
        "t3_aaaaaa": aaa.transpose(3, 4, 5, 0, 1, 2),
        "t3_bbbbbb": bbb.transpose(3, 4, 5, 0, 1, 2),
        "t3_aabaab": M,
        # abbabb comes from pyscf's `bba` block, which is 2-BETA-1-alpha in
        # layout [i,j,a,b,k,c] -- so its ALPHA line is (c,k) at axes 5/4, and
        # ccgen's alpha-first slot order is [c,a,b, k,i,j]. Verified by the only
        # check that distinguishes it: this carries abbabb's OWN antisymmetry
        # (beta pairs, vir (1,2) and occ (4,5)), where every earlier candidate
        # carried aabaab's (0,1)/(3,4) instead.
        "t3_abbabb": bba.transpose(5, 2, 3, 4, 0, 1),
        "v_aaaa": vaa - vaa.transpose(0, 1, 3, 2),
        "v_bbbb": vbb - vbb.transpose(0, 1, 3, 2),
        "v_abab": eri_phys(Ca, Cb),
        "f_aa": fa, "f_bb": fb,
    }

    dims = dict(noa=noa, nva=nva, nob=nob, nvb=nvb)
    eqs = ucc_adapt_equations(
        generate_cc_equations("ccsdt", engine="diagram", canonical_fock=True))
    got = {k: np.asarray(sum(ucc_residual_einsum(t, dims, tensors) for t in eqs[k]))
           for k in ref}
    ref = {k: np.asarray(v).transpose(
               tuple(range(v.ndim // 2, v.ndim)) + tuple(range(v.ndim // 2)))
           for k, v in ref.items()}
    if return_inputs:
        return got, ref, (tensors, dims, eqs)
    return got, ref


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable (lives in tests/pyscf/.venv)")
class U14RankSixVsPyscfTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.got, cls.ref = _build()

    # Vacuity floor. With only t3 perturbed the singles reference is ~5e-3 --
    # small in absolute terms but ~5e6 x the ~1e-9 it has at convergence, so the
    # comparison is far from vacuous. The guard exists to catch a reference that
    # has collapsed to the convergence floor, which is orders below this.
    MIN_REF = 1e-3

    def _check(self, key, atol):
        g, r = self.got[key], self.ref[key]
        self.assertEqual(g.shape, r.shape)
        self.assertGreater(np.abs(r).max(), self.MIN_REF,
                           f"{key}: reference residual is ~zero — vacuous")
        self.assertLess(np.abs(g - r).max(), atol, f"{key}: ccgen != pyscf")

    def test_singles_and_doubles_reproduce_pyscf(self):
        """Exact at rank 6. Load-bearing beyond its own scope: these residuals
        CONSUME t3, so their agreement says the t3 blocks and ccgen's reading of
        them are both correct — which is what confines the triples discrepancy
        below to the T3 equation."""
        for key in ("singles_aa", "singles_bb", "doubles_aaaa", "doubles_bbbb"):
            with self.subTest(target=key):
                self._check(key, 1e-13)

    @unittest.expectedFailure
    def test_triples_reproduce_pyscf(self):
        """OPEN: rel ~1.3e-3, down from 8.8e-2 once the three interface defects
        were fixed. A fixed multiplicative deficit in the t3-linear part -- see
        the module docstring. An unexpected PASS means it has been resolved."""
        self._check("triples_aaaaaa", 1e-12)


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable (lives in tests/pyscf/.venv)")
class T3ClosureRelationTests(unittest.TestCase):
    """U1.4c.1 — the rank-6 closed-shell closure, DERIVED (not fitted).

    RCC stores one spatial `t3` and derives the rest, so the four UCC blocks are
    not independent at closed shell. A non-vacuous third-source comparison needs
    these relations, and a wrong one is indistinguishable from an equation defect.

        t3_aaaaaa = sum_q sign(q) * t3_aabaab[bra axes permuted by inverse(order_q)]

    with `order_q = [x for x in range(3) if x != q] + [q]` — i.e. exactly what
    `_split_same_spin_amplitude` emits, read as arrays.

    **The load-bearing subtlety, and what four earlier hand-derivations all got
    wrong: the splitter permutes BASE INDICES, not array axes.** Its output feeds
    an einsum whose subscripts come from index NAMES against a fixed output order,
    so reordering the bases applies the INVERSE permutation to the array. Read
    forward it fails by the block's full magnitude (1.2e-2 against |A| 1.1e-3);
    read as the inverse it is exact to 4.8e-18.

    **A fitted relation was committed here first and was wrong.** Least squares
    over the 36 signed bra x ket permutation images gave a uniform-1/12 double
    antisymmetrizer that reproduced both real block pairs exactly. It is not the
    same relation: on a GENERIC block the two differ by ~80% of the magnitude
    (2.87 against |S| 3.58). They coincide only on this fixture, whose 36 images
    span rank 5 where a generic block gives rank 9 — so the fit was
    underdetermined and its agreement meant nothing. The `bbb`<-`bba` cross-check
    did not catch it: both pairs share a null space, so a deliberately different
    exact fit passes that too.

    The moral, since it cost several rounds: **a relation that reproduces every
    case in a degenerate fixture is not thereby derived.** Check it against a
    generic instance, or derive it from the code that defines it.

    The second relation, for `abbabb`, is settled by symmetry rather than by
    fitting — see `_build`: PySCF's `bba` is 2-beta-1-alpha in layout
    `[i,j,a,b,k,c]`, so its alpha line is `(c,k)` and ccgen's alpha-first order is
    `[c,a,b,k,i,j]`, i.e. `bba.transpose(5,2,3,4,0,1)`. The check that picks it
    out is that it carries `abbabb`'s OWN antisymmetry (the BETA pairs, vir (1,2)
    and occ (4,5)); every earlier candidate carried `aabaab`'s (0,1)/(3,4)
    instead, including the identity map — which matches `aab` exactly on this
    fixture and is still wrong.
    """
    @classmethod
    def setUpClass(cls):
        from ccgen.tests.test_spin import _uccsdt_t3_blocks
        cls.aaa, cls.aab, cls.bba, cls.bbb, cls.nocc, cls.nvir, _cc = _uccsdt_t3_blocks()

    @staticmethod
    def _parity(p):
        s, pl = 1, list(p)
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                if pl[i] > pl[j]:
                    s = -s
        return s

    @classmethod
    def _same_spin_from_mixed(cls, M, n=3):
        """The DERIVED closure: `_split_same_spin_amplitude`'s relation, with the
        base reordering read as the INVERSE permutation on the array axes."""
        out = None
        for q in range(n - 1, -1, -1):
            order = [x for x in range(n) if x != q] + [q]
            inv = [order.index(x) for x in range(n)]
            term = cls._parity(order) * M.transpose(tuple(inv) + tuple(range(n, 2 * n)))
            out = term if out is None else out + term
        return out

    @classmethod
    def _fitted_antisymmetrizer(cls, M):
        """The REJECTED fitted form, kept so the test below can show it differs on
        a generic block. Do not use it as a closure."""
        out = np.zeros_like(M)
        for bp in itertools.permutations(range(3)):
            for kp in itertools.permutations(range(3)):
                out += (cls._parity(bp) * cls._parity(kp)
                        * M.transpose(tuple(bp) + tuple(3 + x for x in kp)))
        return out / 12.0

    def test_aaaaaa_from_aabaab(self):
        A = self.aaa.transpose(3, 4, 5, 0, 1, 2)
        M = self.aab.transpose(2, 3, 5, 0, 1, 4)
        self.assertGreater(np.abs(A).max(), 1e-6, "block is ~zero; vacuous")
        self.assertLess(np.abs(A - self._same_spin_from_mixed(M)).max(), 1e-14)

    def test_bbbbbb_from_abbabb_is_the_SAME_relation(self):
        """A necessary check, not a sufficient one. It rules out a fit to noise,
        but NOT a wrong choice of coefficients: the two pairs share a null space,
        so `c0 + 3*n` passes this too (measured). See the class docstring."""
        B = self.bbb.transpose(3, 4, 5, 0, 1, 2)
        X = self.bba.transpose(2, 3, 5, 0, 1, 4)
        self.assertGreater(np.abs(B).max(), 1e-6, "block is ~zero; vacuous")
        self.assertLess(np.abs(B - self._same_spin_from_mixed(X)).max(), 1e-14)

    def test_the_permutation_basis_is_rank_deficient(self):
        """Pins the caveat itself: the real block's 36 images span rank 5, where a
        GENERIC block of the same symmetry gives rank 9. That gap is why no fit on
        this fixture can pin the relation uniquely."""
        M = self.aab.transpose(2, 3, 5, 0, 1, 4)
        imgs = [M.transpose(tuple(bp) + tuple(3 + x for x in kp))
                for bp in itertools.permutations(range(3))
                for kp in itertools.permutations(range(3))]
        basis = np.stack([i.ravel() for i in imgs], axis=1)
        self.assertEqual(np.linalg.matrix_rank(basis), 5,
                         "rank changed; the underdetermination caveat in the "
                         "class docstring may be stale")

    def test_derived_and_fitted_forms_DIFFER_on_a_generic_block(self):
        """Why the fitted form had to be rejected even though it reproduced every
        real block pair exactly. On a generic block — no rank-5 degeneracy — the
        two relations disagree by ~80% of the magnitude. Only one of them can be
        the closure, and the derived one is the one traceable to the code that
        performs the reduction."""
        rng = np.random.default_rng(4)
        nv, no = 5, 5
        M = rng.random((nv, nv, nv, no, no, no))
        M = M - M.transpose(1, 0, 2, 3, 4, 5)
        M = M - M.transpose(0, 1, 2, 4, 3, 5)
        derived = self._same_spin_from_mixed(M)
        fitted = self._fitted_antisymmetrizer(M)
        self.assertGreater(np.abs(derived).max(), 1e-6)
        self.assertGreater(np.abs(derived - fitted).max(),
                           0.1 * np.abs(derived).max(),
                           "the two forms agree on a generic block; the "
                           "distinction this test records may be stale")
