"""Unrestricted Λ equations: differentiate the UCC Lagrangian
(docs/CC_LAMBDA_SCOPE.md, step 2).

With the per-block weight 1/(n_α! n_β!)² (`lagrangian._weight`) the UCC
Lagrangian is the spin-orbital one restricted to stored blocks, so UCC λ is
PySCF's UCCSD λ with no rescaling:

* Complex step on L, per amplitude block.
* At PySCF UCCSD's converged t and λ, every block's Λ residual (antisymmetrized
  for same-spin blocks) is zero, and nonzero at a perturbed λ.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.lagrangian import differentiate, lagrangian_terms  # noqa: E402
from ccgen.spin import ucc_adapt_equations  # noqa: E402
from ccgen.tests.residual_eval import (  # noqa: E402
    _antisymmetrize_block, ucc_random_tensors, ucc_residual_einsum)

BLOCKS = {"t1_aa": "singles_aa", "t1_bb": "singles_bb", "t2_aaaa": "doubles_aaaa",
          "t2_abab": "doubles_abab", "t2_bbbb": "doubles_bbbb"}


def _anti_if_same_spin(x, name):
    if name.endswith(("aaaa", "bbbb")):
        return _antisymmetrize_block(_antisymmetrize_block(x, (0, 1)), (2, 3))
    return x


def _ucc_lagrangian(canonical_fock):
    eqs = ucc_adapt_equations(
        generate_cc_equations("ccsd", engine="diagram", canonical_fock=canonical_fock))
    lag = lagrangian_terms(eqs, "ucc")
    return lag, {t: differentiate(lag, t) for t in BLOCKS}


def _value(terms, dims, tensors):
    return sum(ucc_residual_einsum(t, dims, tensors) for t in terms)


class UCCComplexStepTests(unittest.TestCase):
    def test_ccsd(self):
        lag, G = _ucc_lagrangian(canonical_fock=False)
        dims = dict(noa=3, nva=4, nob=2, nvb=5)
        tensors = ucc_random_tensors(**dims, seed=2)
        rng = np.random.default_rng(5)
        spins = {"a": (dims["noa"], dims["nva"]), "b": (dims["nob"], dims["nvb"])}
        for t in BLOCKS:
            tag = t.split("_")[1]
            n = len(tag) // 2
            occ = tuple(spins[s][0] for s in tag[:n])
            vir = tuple(spins[s][1] for s in tag[n:])
            tensors["l" + t[1:]] = _anti_if_same_spin(rng.random(occ + vir) - 0.5, t)
        for t in BLOCKS:
            D = _anti_if_same_spin(rng.random(tensors[t].shape) - 0.5, t)
            analytic = np.sum(_value(G[t], dims, tensors) * D)
            h = 1e-20
            tc = {k: v.astype(complex) for k, v in tensors.items()}
            tc[t] = tc[t] + 1j * h * D
            step = _value(lag, dims, tc).imag / h
            self.assertLess(abs(analytic - step) / abs(step), 1e-12, t)


try:
    import pyscf  # noqa: F401
    HAVE_PYSCF = True
except ImportError:
    HAVE_PYSCF = False

CH3 = "C 0 0 0; H 0 1.079 0; H 0.9345 -0.5395 0; H -0.9345 -0.5395 0"
WATER = "O 0 0 0.117176; H 0 0.756950 -0.468703; H 0 -0.756950 -0.468703"


def _pyscf_uccsd(atom, basis, charge, spin):
    from pyscf import ao2mo, cc, gto, scf

    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 300
    mycc.kernel()
    (l1a, l1b), (l2aa, l2ab, l2bb) = mycc.solve_lambda()

    Ca, Cb = mf.mo_coeff
    noa, nob = mol.nelec

    def eri_phys(C1, C2):
        g = ao2mo.general(mol, (C1, C1, C2, C2), compact=False).reshape(
            C1.shape[1], C1.shape[1], C2.shape[1], C2.shape[1])
        return g.transpose(0, 2, 1, 3)

    vaa, vbb = eri_phys(Ca, Ca), eri_phys(Cb, Cb)
    fa, fb = (C.T @ F @ C for C, F in zip((Ca, Cb), mf.get_fock()))
    (t1a, t1b), (t2aa, t2ab, t2bb) = mycc.t1, mycc.t2
    tensors = {
        "t1_aa": t1a.T, "t1_bb": t1b.T,
        "t2_aaaa": t2aa.transpose(2, 3, 0, 1), "t2_abab": t2ab.transpose(2, 3, 0, 1),
        "t2_bbbb": t2bb.transpose(2, 3, 0, 1),
        "v_aaaa": vaa - vaa.transpose(0, 1, 3, 2), "v_bbbb": vbb - vbb.transpose(0, 1, 3, 2),
        "v_abab": eri_phys(Ca, Cb), "f_aa": fa, "f_bb": fb,
        # λ in (occ..., vir...), PySCF's own layout
        "l1_aa": l1a, "l1_bb": l1b, "l2_aaaa": l2aa, "l2_abab": l2ab, "l2_bbbb": l2bb,
    }
    dims = dict(noa=noa, nva=Ca.shape[1] - noa, nob=nob, nvb=Cb.shape[1] - nob)
    return tensors, dims


@unittest.skipUnless(HAVE_PYSCF, "needs PySCF")
class PySCFUCCLambdaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # General Fock: PySCF's SCF leaves a small f_ov (see test_lambda_equations).
        cls.lag, cls.G = _ucc_lagrangian(canonical_fock=False)

    def _worst(self, tensors, dims):
        return {t: np.abs(_anti_if_same_spin(_value(self.G[t], dims, tensors), t)).max()
                for t in BLOCKS}

    def _check(self, *system):
        tensors, dims = _pyscf_uccsd(*system)
        for t, r in self._worst(tensors, dims).items():
            self.assertLess(r, 1e-9, f"{t} residual at PySCF λ")
        tensors["l2_abab"] = tensors["l2_abab"] * 1.01
        self.assertGreater(max(self._worst(tensors, dims).values()), 1e-5)

    def test_ch3_sto3g(self):
        self._check(CH3, "sto-3g", 0, 1)

    def test_water_cation_631g(self):
        self._check(WATER, "6-31g", 1, 1)


if __name__ == "__main__":
    unittest.main()
