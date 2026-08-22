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

Localized further by zeroing amplitude classes on both sides:

* **`t1 = 0`** — triples discrepancy UNCHANGED (1.29e-2). So it is not a
  T1-dressing convention difference, which was the leading hypothesis given PySCF
  builds `r3` from T1-dressed intermediates.
* **`t1 = t3 = 0`** — ccgen's `triples_aaaaaa` is **1.27e-2 while PySCF's is
  3.3e-3**, i.e. ccgen's t2-only contribution to R3 is ~4x the entire reference.
  Every one of ccgen's 579 triples terms outside the pure-t2 class evaluates to
  exactly 0 here, so the disagreement is entirely in the 108 pure-t2 terms.
* Within those 108, split by shape: `t2·v` (18 terms) contributes **1.6e-2**,
  `t2·t2·v_aaaa` (54 terms) 2.6e-3, `t2·t2·v_abab` (36 terms) 3.0e-3. Dropping the
  18 `t2·v` terms leaves 4.9e-3 against PySCF's 3.3e-3.

So the suspect set is **18 terms of shape `t2_aaaa · v_aaaa` in `triples_aaaaaa`**
— the term that generates T3 from T2. Whether ccgen over-counts them or PySCF
folds them elsewhere is not yet established, and picking a side by inspection is
what this session has repeatedly got wrong; the next step is comparing against a
third source, not re-reading either.

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
    for x in t1 + t2:
        x += 0.02 * rng.random(x.shape)
    for k in (0, 2):
        a = t2[k]
        a = a - a.transpose(1, 0, 2, 3)
        t2[k][...] = a - a.transpose(0, 1, 3, 2)

    # t3: perturb in FULL form (the packed array has no valid elementwise
    # perturbation), re-impose each block's antisymmetry, then repack.
    full = list(uccsdt.tamps_tri2full_uhf(cc, [x.copy() for x in t3]))
    for k in (0, 3):                      # aaa / bbb: [i,j,k,a,b,c]
        full[k] = _anti(_anti(full[k] + 0.02 * rng.random(full[k].shape),
                              (0, 1, 2)), (3, 4, 5))
    y = full[1] + 0.02 * rng.random(full[1].shape)   # aab: [i,j,a,b,k,c]
    y = y - y.transpose(1, 0, 2, 3, 4, 5)
    y = y - y.transpose(0, 1, 3, 2, 4, 5)
    full[1], full[2] = y, y.copy()        # aab and bba are ONE stored sector
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
    aaa, aab, _bba, bbb = before3
    M = aab.transpose(2, 3, 5, 0, 1, 4)        # t3_aabaab
    tensors = {
        "t1_aa": b1[0].T, "t1_bb": b1[1].T,
        "t2_aaaa": b2[0].transpose(2, 3, 0, 1),
        "t2_abab": b2[1].transpose(1, 3, 0, 2),    # [i,a,j,b] -> [a,b,i,j]
        "t2_bbbb": b2[2].transpose(2, 3, 0, 1),
        "t3_aaaaaa": aaa.transpose(3, 4, 5, 0, 1, 2),
        "t3_bbbbbb": bbb.transpose(3, 4, 5, 0, 1, 2),
        "t3_aabaab": M,
        # the spin flip of aabaab is a slot reversal per half, NOT the identity
        "t3_abbabb": M.transpose(2, 1, 0, 5, 4, 3),
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

    def _check(self, key, atol):
        g, r = self.got[key], self.ref[key]
        self.assertEqual(g.shape, r.shape)
        self.assertGreater(np.abs(r).max(), 1e-2,
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
        """OPEN: ~1.4e-2 relative. See the module docstring for what is ruled
        out. An unexpected PASS means it has been resolved."""
        self._check("triples_aaaaaa", 1e-12)
