"""Λ references for the `cc_lambda` gates (docs/CC_LAMBDA_SCOPE.md, 3c).

RCC: Planck's spatial λ is not PySCF's RCCSD λ, so it is solved here from
ccgen's spatial Λ equations at PySCF's converged t (GMRES, as in
python/ccgen/tests/test_lambda_spatial.py). UCC: Planck's λ is PySCF's
UCCSD λ unscaled, so PySCF's `solve_lambda` is the reference. Both are
printed as the rms of each λ block over its full stored array, as Planck logs
them. Cartesian 6-31G from Planck's own GBS file (parse_gaussian), the
regression inputs' geometries.

Run with tests/pyscf/.venv/bin/python from the repository root.
"""
import os
import sys

import numpy as np
from pyscf import cc, gto, scf
from pyscf.gto.basis import parse_gaussian

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "ccgen", "tests"))

from test_lambda_spatial import _pyscf_rccsd, _solve_lambda, _spatial_lagrangian  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BASIS = {el: parse_gaussian.load(os.path.join(ROOT, "basis-sets", "6-31g"), el)
         for el in ("O", "H")}
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))  # noqa: E731

WATER = "O 0 0 0.117176; H 0 0.756950 -0.468703; H 0 -0.756950 -0.468703"


def rcc():
    from pyscf import gto as _gto
    real = _gto.M
    _gto.M = lambda **kw: real(cart=True, **kw)   # Planck's inputs are Cartesian
    try:
        mycc, eris, tensors, no, nv = _pyscf_rccsd(WATER, BASIS)
    finally:
        _gto.M = real
    _, _, G = _spatial_lagrangian()
    worst = _solve_lambda(G, no, nv, tensors)
    print(f"water RCCSD/6-31G  E_corr={mycc.e_corr:.10f}  max|R_Λ|={worst:.1e}")
    print(f"  l1 rms {rms(tensors['l1']):.10e}")
    print(f"  l2 rms {rms(tensors['l2']):.10e}")


def ucc():
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis=BASIS, spin=1, cart=True, verbose=0)
    # conv_tol_grad matters: converged on the energy alone, PySCF's λ1 sits
    # 7e-9 from its tightly converged value.
    mf = scf.UHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-10)
    mycc = cc.UCCSD(mf)
    mycc.conv_tol, mycc.conv_tol_normt = 1e-13, 1e-12
    mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    print(f"OH UCCSD/6-31G  E_corr={mycc.e_corr:.10f}")
    for name, x in zip(("l1_aa", "l1_bb", "l2_aaaa", "l2_abab", "l2_bbbb"), (*l1, *l2)):
        print(f"  {name} rms {rms(x):.10e}")


if __name__ == "__main__":
    rcc()
    ucc()
