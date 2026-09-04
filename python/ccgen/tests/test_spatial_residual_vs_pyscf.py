"""R4.2c/d gate: ccgen's generated CCSD residual vs PySCF's spatial RCCSD.

This is the reference side of the R4.2 comparison, and it is deliberately
independent of the C++ kernel: it pins that ccgen's equations and PySCF agree,
so when the C++ probe disagrees with BOTH (measured: 0.0989 vs 0.0511 Frobenius
on LiH/STO-3G) the defect is unambiguously in the C++ layer.

Two conventions this encodes, both of which cost real time to find:

* **MO phase freedom.** A converged SCF fixes each MO only up to a sign, chosen
  independently by PySCF and Planck. Nothing is comparable ELEMENTWISE across
  the two. The Frobenius norm is phase-invariant, so it is what the C++
  comparison uses.
* **Spin-orbital vs spatial.** ccgen's residual is spin-orbital; PySCF's
  ``update_amps`` is spatial. The energy-scale comparison below goes through the
  quantity both agree on.
"""
import unittest

import numpy as np


def _lih():
    from pyscf import ao2mo, cc, gto, scf

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", spin=0)
    mol.cart = True
    mf = scf.RHF(mol).run(verbose=0)
    mycc = cc.CCSD(mf)
    nocc = mycc.nocc
    nvir = mycc.nmo - nocc
    e = mf.mo_energy
    nmo = mf.mo_coeff.shape[1]
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape((nmo,) * 4)
    oovv = eri.transpose(0, 2, 1, 3)[:nocc, :nocc, nocc:, nocc:]
    denom = (e[:nocc, None, None, None] + e[None, :nocc, None, None]
             - e[None, None, nocc:, None] - e[None, None, None, nocc:])
    return mf, mycc, oovv, denom, nocc, nvir


class SpatialResidualVsPyscfTests(unittest.TestCase):
    def test_pyscf_residual_vanishes_at_its_own_solution(self):
        """The control. Without this, a non-zero C++ residual proves nothing."""
        try:
            _mf, mycc, _oovv, denom, _no, _nv = _lih()
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"pyscf not importable: {exc}")

        mycc.conv_tol = 1e-12
        mycc.kernel()
        eris = mycc.ao2mo()
        _t1n, t2n = mycc.update_amps(mycc.t1, mycc.t2, eris)
        self.assertLess(float(np.linalg.norm((t2n - mycc.t2) * denom)), 1e-7)

    def test_pyscf_and_ccgen_agree_at_mp2_amplitudes(self):
        """Two independent references must agree, or neither can arbitrate."""
        try:
            _mf, mycc, oovv, denom, nocc, nvir = _lih()
            import ccgen.tests.test_reference_vs_pyscf as T
            from ccgen.generate import generate_cc_equations
            from ccgen.tests.residual_eval import residual_einsum
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"deps unavailable: {exc}")

        eris = mycc.ao2mo()
        t2 = oovv / denom
        _t1n, t2n = mycc.update_amps(np.zeros((nocc, nvir)), t2, eris)
        pyscf_max = float(np.max(np.abs((t2n - t2) * denom)))

        # ccgen, spin-orbital, at the spin-orbital MP2 amplitudes.
        _mf2, fock, v, so_nocc, _nmo, so_nvir = T._spinorbital_integrals(
            "Li 0 0 0; H 0 0 1.6", "sto-3g", 0, 0)
        d = T._amp_denominators(fock, so_nocc, [1, 2, 3])
        amps = {
            "t1": np.zeros((so_nvir, so_nocc)),
            "t2": v[:so_nocc, :so_nocc, so_nocc:, so_nocc:].transpose(2, 3, 0, 1) / d[2],
            "t3": np.zeros((so_nvir,) * 3 + (so_nocc,) * 3),
        }
        eqs = generate_cc_equations("ccsdt")
        tensors = {"v": v, "f": fock, **amps}
        from ccgen.tests.dump_cc_fixture import spatial_block, to_cpp_layout
        r = sum(residual_einsum(t, so_nocc, so_nvir, tensors=tensors)
                for t in eqs["doubles"])
        ccgen_max = float(np.max(np.abs(to_cpp_layout(spatial_block(r, 2), 2))))

        # Measured 2026-08-26: 4.551241e-02 (ccgen) vs 4.551242e-02 (PySCF).
        self.assertAlmostEqual(ccgen_max, pyscf_max, places=6)


if __name__ == "__main__":
    unittest.main()
