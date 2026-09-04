"""F3/U1.2 -- validate the UCC residual manifold against PySCF UCCSD.

This is the gate the whole F1/F2 ladder existed to reach. U1.0/U1.1 were gated
STRUCTURALLY only (names distinct, blocks non-empty, counts symmetric), and F2.3
added a closed-shell oracle against ccgen's own RCC path -- but nothing had
compared a UCC residual to an independent implementation until this.

It evaluates every UCC target at PySCF's own (perturbed) amplitudes and compares
against `pyscf.cc.uccsd.update_amps`, converting `t_new` back to a residual via
``R = (t_new - t) * D``.

Skipped if PySCF is not importable (it lives in tests/pyscf/.venv, not the
default env). Run with that interpreter:

    tests/pyscf/.venv/bin/python -m unittest ccgen.tests.test_ucc_vs_pyscf

Three conventions had to be resolved to make the comparison meaningful, and two
of them the scope had wrong:

* **Layout is a TRANSPOSE, not a rename.** The scope recorded the PySCF mapping
  as `t2ab -> t2_abab`, a pure rename. It is not: PySCF stores amplitudes
  ``(occ..., vir...)`` while ccgen emits ``(vir..., occ...)``. The names do
  correspond one-for-one, but every array needs its halves swapped.
* **The dims are not the scoped ones.** CH3/STO-3G is `noa=5 nva=3, nob=4
  nvb=4`, not the scoped `noa=5 nva=4, nob=4 nvb=5`. Still non-trivial in every
  block and still non-square in alpha, which is what the case was chosen for.
* **`f_ov` must be zeroed on BOTH sides.** See the dedicated test below -- this
  is the difference between agreeing to 8e-9 and agreeing to 6e-16.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pyscf import gto, scf, cc, ao2mo
    _HAVE_PYSCF = True
except ImportError:  # pragma: no cover - depends on the pyscf venv
    _HAVE_PYSCF = False

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.spin import ucc_adapt_equations  # noqa: E402
from ccgen.tests.residual_eval import ucc_residual_einsum  # noqa: E402


# CH3/STO-3G, deliberately. OH/STO-3G gives noa=5 nva=1, so C(1,2)=0 distinct
# alpha-alpha doubles and the `aaaa` block is trivially zero regardless of
# correctness -- a vacuous pass. CH3 makes all five targets non-trivial
# (reference |R| between 0.34 and 1.04) and is non-square in alpha.
CH3 = "C 0 0 0; H 0 1.079 0; H 0.9345 -0.5395 0; H -0.9345 -0.5395 0"


def _build(zero_fov_pyscf: bool, zero_fov_ccgen: bool, seed: int = 0):
    """Return ({target: ccgen residual}, {target: pyscf residual}).

    Amplitudes are PERTURBED off convergence on purpose: at PySCF's converged
    amplitudes the reference residual is ~1e-8 by construction, so a kernel
    returning zero would pass any gate here. Same-spin blocks are re-antisymmetrized
    after perturbing, or PySCF's own residual is meaningless.
    """
    mol = gto.M(atom=CH3, basis="sto-3g", spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.run()
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.run()
    eris = mycc.ao2mo()

    Ca, Cb = mf.mo_coeff
    noa, nob = mol.nelec
    na, nb = Ca.shape[1], Cb.shape[1]
    nva, nvb = na - noa, nb - nob

    rng = np.random.default_rng(seed)

    def pert(x, s=0.05):
        return x + s * rng.random(x.shape)

    def anti(x):
        x = x - x.transpose(1, 0, 2, 3)
        return x - x.transpose(0, 1, 3, 2)

    t1 = (pert(mycc.t1[0]), pert(mycc.t1[1]))
    t2 = (anti(pert(mycc.t2[0])), pert(mycc.t2[1]), anti(pert(mycc.t2[2])))

    if zero_fov_pyscf:
        eris.focka = eris.focka.copy()
        eris.fockb = eris.fockb.copy()
        eris.focka[:noa, noa:] = eris.focka[noa:, :noa] = 0.0
        eris.fockb[:nob, nob:] = eris.fockb[nob:, :nob] = 0.0

    t1n, t2n = mycc.update_amps(t1, t2, eris)

    ea, eb = eris.focka.diagonal().real, eris.fockb.diagonal().real
    Dia = ea[:noa, None] - ea[None, noa:]
    DIA = eb[:nob, None] - eb[None, nob:]
    ref = {
        "singles_aa": (t1n[0] - t1[0]) * Dia,
        "singles_bb": (t1n[1] - t1[1]) * DIA,
        "doubles_aaaa": (t2n[0] - t2[0]) * (Dia[:, None, :, None] + Dia[None, :, None, :]),
        "doubles_abab": (t2n[1] - t2[1]) * (Dia[:, None, :, None] + DIA[None, :, None, :]),
        "doubles_bbbb": (t2n[2] - t2[2]) * (DIA[:, None, :, None] + DIA[None, :, None, :]),
    }

    def eri_phys(C1, C2):
        """<pq|rs> over the (C1,C2) spin pattern. ao2mo yields chemist (pr|qs);
        the mid-axis swap is the same rebind_physicist convention the C++ runtime
        needed (see the B5 fix)."""
        g = ao2mo.general(mol, (C1, C1, C2, C2), compact=False).reshape(
            C1.shape[1], C1.shape[1], C2.shape[1], C2.shape[1])
        return g.transpose(0, 2, 1, 3)

    vaa, vbb = eri_phys(Ca, Ca), eri_phys(Cb, Cb)
    fa = Ca.T @ mf.get_fock()[0] @ Ca
    fb = Cb.T @ mf.get_fock()[1] @ Cb
    if zero_fov_ccgen:
        fa, fb = fa.copy(), fb.copy()
        fa[:noa, noa:] = fa[noa:, :noa] = 0.0
        fb[:nob, nob:] = fb[nob:, :nob] = 0.0

    tensors = {
        # (occ...,vir...) -> (vir...,occ...): the transpose the scope missed
        "t1_aa": t1[0].T, "t1_bb": t1[1].T,
        "t2_aaaa": t2[0].transpose(2, 3, 0, 1),
        "t2_abab": t2[1].transpose(2, 3, 0, 1),
        "t2_bbbb": t2[2].transpose(2, 3, 0, 1),
        "v_aaaa": vaa - vaa.transpose(0, 1, 3, 2),
        "v_bbbb": vbb - vbb.transpose(0, 1, 3, 2),
        "v_abab": eri_phys(Ca, Cb),     # mixed spin: no exchange within a half
        "f_aa": fa, "f_bb": fb,
    }

    dims = dict(noa=noa, nva=nva, nob=nob, nvb=nvb)
    eqs = ucc_adapt_equations(
        generate_cc_equations("ccsd", engine="diagram", canonical_fock=True))
    got = {k: np.asarray(sum(ucc_residual_einsum(t, dims, tensors) for t in eqs[k]))
           for k in ref}
    # PySCF residuals are (occ...,vir...); ccgen's are (vir...,occ...)
    ref = {k: np.asarray(v).transpose(
               tuple(range(v.ndim // 2, v.ndim)) + tuple(range(v.ndim // 2)))
           for k, v in ref.items()}
    return got, ref


@unittest.skipUnless(_HAVE_PYSCF, "pyscf not importable (lives in tests/pyscf/.venv)")
class F3UccVsPyscfTests(unittest.TestCase):

    def test_every_block_reproduces_pyscf(self):
        """The gate. Scoped at 1e-10; holds at MACHINE PRECISION once f_ov is
        treated consistently, so it is asserted there -- a 1e-10 bound would let
        a real defect of size 1e-11 through unnoticed."""
        got, ref = _build(zero_fov_pyscf=True, zero_fov_ccgen=True)
        for key in sorted(ref):
            with self.subTest(target=key):
                self.assertEqual(got[key].shape, ref[key].shape)
                # not a vacuous pass: every reference block is O(0.1-1)
                self.assertGreater(np.abs(ref[key]).max(), 1e-2,
                                   f"{key}: reference residual is ~zero")
                self.assertLess(np.abs(got[key] - ref[key]).max(), 1e-13,
                                f"{key}: ccgen != pyscf")

    def test_fov_must_be_zeroed_on_BOTH_sides(self):
        """Why the gate zeroes f_ov, recorded as an assertion rather than a
        comment.

        Every Planck CC kernel gets a canonical Fock by construction (f_ov = 0
        identically), so the f_ov terms ccgen carries are runtime-zero. PySCF's
        f_ov is not exactly zero -- it is SCF convergence noise, ~8e-9 even at
        conv_tol=1e-14 -- and PySCF's update_amps USES it.

        So zeroing on only one side is worse than zeroing on neither: the two
        routes then disagree about whether the f_ov terms are present at all.
        Measured here, on singles_aa:

            neither side zeroed   ~8e-9   (both carry the same noise)
            ccgen side only       ~9e-9   (slightly WORSE, not better)
            both sides zeroed     ~6e-17

        The doubles blocks sit at ~7e-11 in the first two cases and ~7e-16 in the
        third, so this is not a singles-only effect.
        """
        one_sided, ref1 = _build(zero_fov_pyscf=False, zero_fov_ccgen=True)
        both, ref2 = _build(zero_fov_pyscf=True, zero_fov_ccgen=True)
        d_one = np.abs(one_sided["singles_aa"] - ref1["singles_aa"]).max()
        d_both = np.abs(both["singles_aa"] - ref2["singles_aa"]).max()
        self.assertGreater(d_one, 1e-10,
                           "one-sided f_ov zeroing agreed too well; the "
                           "convention note above is no longer describing reality")
        self.assertLess(d_both, 1e-13)
        self.assertLess(d_both * 1e4, d_one,
                        "zeroing both sides must be orders better, not marginally")

    def test_layout_is_a_transpose_not_a_rename(self):
        """Pins the correction to the scope: PySCF is (occ...,vir...), ccgen is
        (vir...,occ...). If this ever became a pure rename the shapes would stop
        matching, and on a block like t2_abab (3,4,5,4) vs (5,4,3,4) that is
        detectable -- but t2_bbbb is (4,4,4,4) either way, which is exactly how a
        layout error hides. Hence the assertion on the ASYMMETRIC blocks."""
        got, ref = _build(zero_fov_pyscf=True, zero_fov_ccgen=True)
        self.assertEqual(got["doubles_abab"].shape, (3, 4, 5, 4))   # (nva,nvb,noa,nob)
        self.assertEqual(got["singles_aa"].shape, (3, 5))           # (nva,noa)
        self.assertEqual(got["doubles_aaaa"].shape, (3, 3, 5, 5))


if __name__ == "__main__":
    unittest.main()
