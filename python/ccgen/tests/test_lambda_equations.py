"""Λ equations by differentiating the CC Lagrangian (docs/CC_LAMBDA_SCOPE.md, step 1).

Two gates:

* Complex step: for an antisymmetric direction D, d/dε L(T + εD) equals
  Σ G·D, with G = ∂L/∂t from :func:`differentiate`. Needs only the T
  equations, so it holds at every rank. Dropping one G term fails it.
* PySCF: at PySCF GCCSD's converged t and λ, the antisymmetrized G is zero,
  and nonzero at a perturbed λ. This fixes the λ normalization.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from residual_eval import _antisymmetrize_block, random_tensors, residual_einsum  # noqa: E402

from ccgen.generate import generate_cc_equations, parse_cc_level  # noqa: E402
from ccgen.lagrangian import differentiate, lagrangian_terms  # noqa: E402


def _antisym(arr, n, first_occ):
    """Antisymmetrize an (n, n)-slot excitation tensor within each group."""
    if n < 2:
        return arr
    return _antisymmetrize_block(_antisymmetrize_block(arr, range(n)), range(n, 2 * n))


def _residual(terms, no, nv, tensors):
    return sum(residual_einsum(t, no, nv, tensors) for t in terms)


class ComplexStepTests(unittest.TestCase):
    no, nv = 3, 4

    def _check(self, method, drop=None):
        eqs = generate_cc_equations(method, engine="diagram", canonical_fock=True)
        lag = lagrangian_terms(eqs)
        rng = np.random.default_rng(7)
        tensors = random_tensors(self.no, self.nv, seed=1)
        ranks = parse_cc_level(method)
        for n in ranks:
            tensors[f"l{n}"] = _antisym(rng.random((self.no,) * n + (self.nv,) * n), n, True)
        worst = 0.0
        for n in ranks:
            G = differentiate(lag, n)
            if drop == n:
                G = G[1:]
            D = _antisym(rng.random((self.nv,) * n + (self.no,) * n), n, False)
            analytic = np.sum(_residual(G, self.no, self.nv, tensors) * D)
            h = 1e-20
            tc = {k: v.astype(complex) for k, v in tensors.items()}
            tc[f"t{n}"] = tc[f"t{n}"] + 1j * h * D
            step = sum(residual_einsum(t, self.no, self.nv, tc) for t in lag).imag / h
            worst = max(worst, abs(analytic - step) / abs(step))
        return worst

    def test_ccsd(self):
        self.assertLess(self._check("ccsd"), 1e-12)

    def test_ccsdt(self):
        self.assertLess(self._check("ccsdt"), 1e-12)

    def test_dropped_term_fails(self):
        self.assertGreater(self._check("ccsd", drop=2), 1e-6)


try:
    import pyscf  # noqa: F401
    HAVE_PYSCF = True
except ImportError:
    HAVE_PYSCF = False

WATER = "O 0 0 0.117176; H 0 0.756950 -0.468703; H 0 -0.756950 -0.468703"
H2O2 = ("O 0 0.717 -0.041; O 0.021 -0.717 0.032; "
        "H 0.851 0.887 0.463; H -0.802 -0.911 0.395")


def _pyscf_gccsd(atom, basis, tol=1e-10):
    """Converged GCCSD t and λ in ccgen's layouts, and the full <pq||rs>."""
    from pyscf import ao2mo, cc, gto, scf

    mol = gto.M(atom=atom, basis=basis, verbose=0)
    gmf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12))
    mycc = cc.GCCSD(gmf)
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = tol
    mycc.max_cycle = 200
    mycc.kernel()
    l1, l2 = mycc.solve_lambda()  # tolerance: conv_tol_normt

    mo = gmf.mo_coeff
    nao = mol.nao
    a, b = mo[:nao], mo[nao:]
    n = mo.shape[1]
    eri = sum(ao2mo.general(mol, c, compact=False)
              for c in ((a, a, a, a), (b, b, b, b), (a, a, b, b), (b, b, a, a)))
    phys = eri.reshape(n, n, n, n).transpose(0, 2, 1, 3)
    tensors = {
        "v": phys - phys.transpose(0, 1, 3, 2),
        "f": mo.T @ gmf.get_fock() @ mo,
        "t1": mycc.t1.T,
        "t2": mycc.t2.transpose(2, 3, 0, 1),
        "l1": l1,
        "l2": l2,
    }
    return tensors, mycc.nocc, n - mycc.nocc


@unittest.skipUnless(HAVE_PYSCF, "needs PySCF")
class PySCFLambdaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # General Fock: PySCF's SCF leaves |f_ov| ~1e-8 (H2O2 6-31G 1.1e-8),
        # and with the f_ov terms dropped the rank-1 residual sits exactly there.
        eqs = generate_cc_equations("ccsd", engine="diagram", canonical_fock=False)
        lag = lagrangian_terms(eqs)
        cls.G = {n: differentiate(lag, n) for n in (1, 2)}

    def _residuals(self, tensors, no, nv):
        return {n: _antisym(_residual(self.G[n], no, nv, tensors), n, False)
                for n in (1, 2)}

    def _check(self, atom, basis):
        tensors, no, nv = _pyscf_gccsd(atom, basis)
        res = self._residuals(tensors, no, nv)
        for n, r in res.items():
            self.assertLess(np.abs(r).max(), 1e-9, f"rank {n} residual at PySCF λ")
        tensors["l2"] = tensors["l2"] * 1.01
        res = self._residuals(tensors, no, nv)
        self.assertGreater(max(np.abs(r).max() for r in res.values()), 1e-5)

    def test_water_sto3g(self):
        self._check(WATER, "sto-3g")

    def test_h2o2_c1_631g(self):
        self._check(H2O2, "6-31g")


if __name__ == "__main__":
    unittest.main()
