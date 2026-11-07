"""Reach and size of PySCF's ECP grid-response gradient defect
(docs/PYSCF_ECP_GRID_RESPONSE_SCOPE.md, step R2).

For each case, prints max |stock - fixed| over all gradient components, where
"fixed" is the analytic gradient with the one-line fix (proton numbers in
grids_response_becke). The fixed gradient is the correct one: R1 shows it
matches the energy finite difference. One diatomic (AuH) is also checked
against a central difference of the energy, because R1 had wrongly concluded
that diatomics are immune (HI is immune only because Bragg radius(I) ==
radius(Mn), Z - n_core = 25).

Run: tests/pyscf/.venv/bin/python -I tests/pyscf/pyscf_ecp_grid_response_survey.py
"""
import os
import sys

import numpy as np
import pyscf
import pyscf.grad.rks as grks
from pyscf import dft, gto
from pyscf.data import radii
from pyscf.dft import gen_grid, radi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyscf_ecp_grid_response_repro import fixed_grids_response_becke  # noqa: E402

STOCK = grks.grids_response_becke
FIXED = fixed_grids_response_becke()

AUH_H2O = [("Au", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.524)), ("O", (2.3, 0.4, -0.9)),
           ("H", (2.9, 1.1, -0.6)), ("H", (2.7, -0.4, -0.5))]


def diatomic(sym, r):
    return [(sym, (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r))]


def make_mol(atom, charge=0, spin=0, unit="Angstrom"):
    ecp = {atom[0][0]: "def2-svp"}  # the heavy atom is always first
    return gto.M(atom=atom, unit=unit, basis="def2-svp", ecp=ecp,
                 charge=charge, spin=spin, verbose=0)


def scf(mol, xc, level, radii_adjust, scheme):
    mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=xc)
    mf.grids.level = level
    mf.grids.radii_adjust = radii_adjust
    mf.grids.becke_scheme = scheme
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    assert mf.converged
    return mf


def gradient(mf, response_fn):
    grks.grids_response_becke = response_fn
    try:
        g = mf.nuc_grad_method()
        g.grid_response = True
        return g.kernel()
    finally:
        grks.grids_response_becke = STOCK


def run(label, atom, xc="pbe", level=3, charge=0, spin=0,
        radii_adjust=radi.treutler_atomic_radii_adjust, scheme=gen_grid.original_becke):
    mf = scf(make_mol(atom, charge, spin), xc, level, radii_adjust, scheme)
    stock, fixed = gradient(mf, STOCK), gradient(mf, FIXED)
    print(f"{label:<44} {np.abs(stock - fixed).max():9.2e}", flush=True)
    return mf, stock, fixed


def auh_fd_check():
    """Diatomic AuH, bond component: stock and fixed against a fourth-order
    energy FD (a second-order one at 1e-3 Bohr has ~1e-7 truncation, the same
    size as the defect here)."""
    delta, r = 2e-3, 1.524
    mf, stock, fixed = run("AuH  PBE (FD check below)", diatomic("Au", r))
    mol = mf.mol
    coords = mol.atom_coords(unit="Bohr")

    def e(dz):
        c = coords.copy()
        c[1, 2] += dz
        m = make_mol([(mol.atom_symbol(i), c[i]) for i in range(2)], unit="Bohr")
        return scf(m, "pbe", 3, radi.treutler_atomic_radii_adjust, gen_grid.original_becke).e_tot

    fd = (8 * (e(delta) - e(-delta)) - (e(2 * delta) - e(-2 * delta))) / (12 * delta)
    print(f"    H z: stock {stock[1, 2]:+.10f}  fixed {fixed[1, 2]:+.10f}  energy FD {fd:+.10f}")


if __name__ == "__main__":
    print(f"PySCF {pyscf.__version__}, def2-SVP + def2 ECP; max |stock - fixed| in Ha/Bohr")
    print("Bragg radius (Bohr) at Z vs Z - n_core:")
    for sym in ["Au", "Ag", "Cs", "I"]:
        m = make_mol(diatomic(sym, 1.6))
        z, q = radi.elements_proton(sym), int(m.atom_charges()[0])
        print(f"    {sym:2s} Z={z:2d} {radii.BRAGG[z]:.4f}   Z-core={q:2d} {radii.BRAGG[q]:.4f}")

    print("\n-- element and core size (diatomic MH, PBE, level 3) --")
    auh_fd_check()
    run("AgH  PBE", diatomic("Ag", 1.62))
    run("CsH  PBE (46-electron core)", diatomic("Cs", 2.49))
    run("HI   PBE (control: radii coincide)", diatomic("I", 1.61))

    print("\n-- functional and spin (AuH.H2O, level 3) --")
    for xc in ["lda,vwn", "pbe", "b3lyp", "wb97x"]:
        run(f"AuH.H2O RKS {xc}", AUH_H2O, xc=xc)
    run("AuH.H2O+ UKS pbe (doublet)", AUH_H2O, charge=1, spin=1)

    print("\n-- grid level (AuH.H2O PBE) --")
    for level in [2, 3, 5, 7]:
        run(f"AuH.H2O level {level}", AUH_H2O, level=level)

    print("\n-- partition scheme and radius adjustment (AuH.H2O PBE, level 3) --")
    run("becke radii adjust", AUH_H2O, radii_adjust=radi.becke_atomic_radii_adjust)
    run("no radii adjust (control)", AUH_H2O, radii_adjust=None)
    run("stratmann scheme", AUH_H2O, scheme=gen_grid.stratmann)
    run("becke_lko scheme (control: other code path)", AUH_H2O, scheme=gen_grid.becke_lko)
