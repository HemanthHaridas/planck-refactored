"""Restricted (spatial) Λ equations: differentiate the spin-adapted Lagrangian
(docs/CC_LAMBDA_SCOPE.md, step 2).

The spatial λ is the multiplier of the spatial parametrization, not PySCF's
RCCSD λ, so it is gated by what does not depend on that choice:

* Complex step on L, as for the spin-orbital equations.
* The unrelaxed energy derivative. With the spatial Λ solved at PySCF's
  converged t, dL/dε for a Fock perturbation f -> f + εX equals the finite
  difference of PySCF's own E_corr with ``eris.fock += εX`` (orbitals fixed).
  A wrong λ, or a wrong weight between λ1 and λ2, fails it.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from residual_eval import residual_einsum  # noqa: E402

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.lagrangian import differentiate, lagrangian_terms  # noqa: E402
from ccgen.spin import spin_adapt_equations  # noqa: E402


def _pair_sym(x):
    """Spatial t2/λ2 symmetry: x(a,b,i,j) = x(b,a,j,i) (either layout)."""
    return 0.5 * (x + x.transpose(1, 0, 3, 2))


def _spatial_lagrangian(method="ccsd", canonical_fock=False):
    eqs = spin_adapt_equations(
        generate_cc_equations(method, engine="diagram", canonical_fock=canonical_fock))
    lag = lagrangian_terms(eqs, "rcc")
    return eqs, lag, {1: differentiate(lag, 1), 2: differentiate(lag, 2)}


def _value(terms, no, nv, tensors):
    return sum(residual_einsum(t, no, nv, tensors) for t in terms)


class SpatialComplexStepTests(unittest.TestCase):
    no, nv = 3, 4

    def _tensors(self, rng):
        no, nv = self.no, self.nv
        n = no + nv
        v = rng.standard_normal((n, n, n, n)) * 0.1
        v = v + v.transpose(1, 0, 3, 2)      # <pq|rs> = <qp|sr>
        v = v + v.transpose(2, 3, 0, 1)      # = <rs|pq>
        v = v + v.transpose(2, 1, 0, 3)      # real orbitals: = <rq|ps>
        f = rng.standard_normal((n, n))
        return {
            "v": v, "f": f + f.T,
            "t1": rng.standard_normal((nv, no)) * 0.1,
            "t2": _pair_sym(rng.standard_normal((nv, nv, no, no)) * 0.1),
            "l1": rng.standard_normal((no, nv)),
            "l2": _pair_sym(rng.standard_normal((no, no, nv, nv))),
        }

    def test_ccsd(self):
        _eqs, lag, G = _spatial_lagrangian()
        rng = np.random.default_rng(3)
        tensors = self._tensors(rng)
        for n, shape in ((1, (self.nv, self.no)), (2, (self.nv, self.nv, self.no, self.no))):
            D = rng.standard_normal(shape)
            if n == 2:
                D = _pair_sym(D)
            analytic = np.sum(_value(G[n], self.no, self.nv, tensors) * D)
            h = 1e-20
            tc = {k: v.astype(complex) for k, v in tensors.items()}
            tc[f"t{n}"] = tc[f"t{n}"] + 1j * h * D
            step = _value(lag, self.no, self.nv, tc).imag / h
            self.assertLess(abs(analytic - step) / abs(step), 1e-12, f"rank {n}")


try:
    import pyscf  # noqa: F401
    import scipy  # noqa: F401
    HAVE_PYSCF = True
except ImportError:
    HAVE_PYSCF = False

WATER = "O 0 0 0.117176; H 0 0.756950 -0.468703; H 0 -0.756950 -0.468703"
H2O2 = ("O 0 0.717 -0.041; O 0.021 -0.717 0.032; "
        "H 0.851 0.887 0.463; H -0.802 -0.911 0.395")


def _solve_lambda(G, no, nv, tensors):
    """The spatial Λ equations are affine in λ: solve G(λ) = 0 by GMRES on
    the pair-symmetric subspace."""
    from scipy.sparse.linalg import LinearOperator, gmres

    n1, n2 = no * nv, (no * nv) ** 2

    def unpack(x):
        return x[:n1].reshape(no, nv), _pair_sym(x[n1:].reshape(no, no, nv, nv))

    def residual(x):
        tensors["l1"], tensors["l2"] = unpack(x)
        g1 = _value(G[1], no, nv, tensors)                      # (nv, no)
        g2 = _pair_sym(_value(G[2], no, nv, tensors))           # (nv, nv, no, no)
        return np.concatenate([g1.T.ravel(), g2.transpose(2, 3, 0, 1).ravel()])

    b = residual(np.zeros(n1 + n2))
    op = LinearOperator((n1 + n2,) * 2, matvec=lambda x: residual(x) - b)
    x, info = gmres(op, -b, rtol=1e-12, atol=0.0, restart=200, maxiter=50)
    assert info == 0, f"GMRES did not converge ({info})"
    tensors["l1"], tensors["l2"] = unpack(x)
    return np.abs(residual(x)).max()


def _pyscf_rccsd(atom, basis):
    from pyscf import ao2mo, cc, gto, scf

    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 300
    mycc.kernel()
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape((n,) * 4)
    eris = mycc.ao2mo()
    tensors = {
        "v": eri.transpose(0, 2, 1, 3),          # <pq|rs> = (pr|qs)
        "f": np.array(eris.fock),
        "t1": mycc.t1.T,
        "t2": mycc.t2.transpose(2, 3, 0, 1),
    }
    return mycc, eris, tensors, mycc.nocc, n - mycc.nocc


def _ecorr(mycc, eris, fock):
    eris.fock = fock
    mycc.kernel(t1=mycc.t1, t2=mycc.t2, eris=eris)
    assert mycc.converged
    return mycc.e_corr


@unittest.skipUnless(HAVE_PYSCF, "needs PySCF and SciPy")
class SpatialEnergyDerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eqs, cls.lag, cls.G = _spatial_lagrangian()

    def _check(self, atom, basis):
        mycc, eris, tensors, no, nv = _pyscf_rccsd(atom, basis)

        # Conventions first: the spatial equations reproduce PySCF at its t.
        e = _value(self.eqs["energy"], no, nv, tensors)
        self.assertAlmostEqual(float(e), mycc.e_corr, places=10)
        for key in ("singles", "doubles"):
            self.assertLess(np.abs(_value(self.eqs[key], no, nv, tensors)).max(), 1e-8, key)

        self.assertLess(_solve_lambda(self.G, no, nv, tensors), 1e-9)

        rng = np.random.default_rng(11)
        X = rng.standard_normal(tensors["f"].shape)
        X = 0.5 * (X + X.T)
        h = 1e-20
        tc = {k: v.astype(complex) for k, v in tensors.items()}
        tc["f"] = tc["f"] + 1j * h * X
        analytic = _value(self.lag, no, nv, tc).imag / h

        f0, s = np.array(eris.fock), 1e-3
        e = {k: _ecorr(mycc, eris, f0 + k * s * X) for k in (-2, -1, 1, 2)}
        fd = (e[-2] - 8 * e[-1] + 8 * e[1] - e[2]) / (12 * s)
        # Measured 4.8e-10 (water) and 5.6e-9 (H2O2), the FD's noise; dropping
        # one λ2-equation term misses by 9e-2.
        self.assertLess(abs(analytic - fd), 5e-8 * max(1.0, abs(fd)),
                        f"dL/dε {analytic:.12e} vs FD of E_corr {fd:.12e}")
        return analytic, fd

    def test_water_sto3g(self):
        self._check(WATER, "sto-3g")

    def test_h2o2_c1_631g(self):
        self._check(H2O2, "6-31g")


if __name__ == "__main__":
    unittest.main()
