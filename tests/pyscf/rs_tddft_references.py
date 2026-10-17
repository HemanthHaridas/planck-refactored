#!/usr/bin/env python3
"""PySCF references for range-separated TDDFT (docs/RS_TDDFT.md).

Water/STO-3G, Cartesian, grid level 6 -- the same fixture as the B3LYP
R0 gates.  Prints rsh_coeff first so a coefficient-convention mismatch
shows before any excitation energy is compared.
"""

from pyscf import dft, gto, tddft

GEOM = """
O 0.000000 0.000000 0.117176
H 0.000000 0.756950 -0.468703
H 0.000000 -0.756950 -0.468703
"""
FUNCTIONALS = {"hse06": "hse06", "cam_b3lyp": "camb3lyp", "wb97x": "wb97x"}
HARTREE_TO_EV = 27.211386245988


def mol(spin):
    return gto.M(atom=GEOM, basis="sto-3g", cart=True, spin=spin, verbose=0)


def scf(m, xc, unrestricted):
    mf = (dft.UKS if unrestricted else dft.RKS)(m)
    mf.xc = xc
    mf.grids.level = 6
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def roots(mf, method, nstates, singlet=True):
    td = (tddft.TDA if method == "tda" else tddft.TDDFT)(mf)
    td.nstates = nstates
    if not singlet:
        td.singlet = False
    td.conv_tol = 1e-10
    td.kernel()
    return " ".join(f"{e * HARTREE_TO_EV:.7f}" for e in td.e)


for name, xc in FUNCTIONALS.items():
    rks = scf(mol(0), xc, False)
    omega, alpha, hyb = rks._numint.rsh_and_hybrid_coeff(xc)
    print(f"{name}: omega={omega} alpha(LR)={alpha} hyb(SR)={hyb} E_RKS={rks.e_tot:.10f}")
    for method in ("casida", "tda"):
        print(f"  RKS singlet {method}: {roots(rks, method, 4)}")
        print(f"  RKS triplet {method}: {roots(rks, method, 4, singlet=False)}")
    uks = scf(mol(2), xc, True)
    print(f"  E_UKS(triplet)={uks.e_tot:.10f}")
    for method in ("casida", "tda"):
        print(f"  UKS spin-conserving {method}: {roots(uks, method, 3)}")
