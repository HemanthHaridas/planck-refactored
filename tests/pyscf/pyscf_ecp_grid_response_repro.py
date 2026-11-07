"""Minimal pure-PySCF reproducer: DFT gradients with grid response are wrong on
ECP atoms (docs/PYSCF_ECP_GRID_RESPONSE_SCOPE.md, step R1).

pyscf/grad/rks.py, grids_response_becke._radii_adjust, looks Bragg radii up by
mol.atom_charges(), which is Z - n_core on an ECP atom. The energy's Becke
partition (pyscf/dft/radi.py, treutler_atomic_radii_adjust) uses the proton
number. The analytic gradient therefore differentiates a different partition
from the one the energy is integrated on.

Prints, per component, the stock analytic gradient, the analytic gradient with
the one-line fix (proton numbers in grids_response_becke), and a central
difference of the energy. Only PySCF built-ins are used (def2-SVP basis + ECP).
Run: tests/pyscf/.venv/bin/python tests/pyscf/pyscf_ecp_grid_response_repro.py
"""
import inspect
import textwrap

import numpy as np
import pyscf
import pyscf.grad.rks as grks
from pyscf import dft, gto
from pyscf.dft import radi

XC, GRID_LEVEL, DELTA = "pbe", 5, 1e-3   # DELTA in Bohr
# C1 AuH.H2O (Angstrom). Any molecule with an ECP atom shows the defect,
# diatomics included (AuH, AgH, CsH), unless the Bragg radius at Z - n_core
# happens to equal the one at Z: I (-> Mn), Te (-> Cr) and Y (-> Na) are immune,
# which is why HI showed nothing (pyscf_ecp_grid_response_survey.py).
ATOM = [("Au", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.524)), ("O", (2.3, 0.4, -0.9)),
        ("H", (2.9, 1.1, -0.6)), ("H", (2.7, -0.4, -0.5))]


def make_mol(atom, unit="Angstrom"):
    return gto.M(atom=atom, unit=unit, basis="def2-svp", ecp={"Au": "def2-svp"}, verbose=0)


def energy(atom, unit="Angstrom"):
    mf = dft.RKS(make_mol(atom, unit), xc=XC)
    mf.grids.level = GRID_LEVEL
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    return mf.kernel()


def analytic_gradient():
    mf = dft.RKS(make_mol(ATOM), xc=XC)
    mf.grids.level = GRID_LEVEL
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    g = mf.nuc_grad_method()
    g.grid_response = True
    return g.kernel()


def fixed_grids_response_becke():
    """grids_response_becke with the radius lookup the energy uses."""
    src = textwrap.dedent(inspect.getsource(grks.grids_response_becke))
    old = "charges = mol.atom_charges()"
    assert old in src, "line not found: already fixed upstream?"
    ns = dict(vars(grks))
    ns["elements_proton"] = radi.elements_proton
    exec(src.replace(old, "charges = [elements_proton(x) for x in mol.elements]"), ns)
    return ns["grids_response_becke"]


def finite_difference():
    mol = make_mol(ATOM)
    coords = mol.atom_coords(unit="Bohr")
    fd = np.zeros_like(coords)
    for ia in range(mol.natm):
        for k in range(3):
            def shifted(s):
                c = coords.copy()
                c[ia, k] += s * DELTA
                return [(mol.atom_symbol(i), c[i]) for i in range(mol.natm)]
            ep = energy(shifted(+1), unit="Bohr")
            em = energy(shifted(-1), unit="Bohr")
            fd[ia, k] = (ep - em) / (2 * DELTA)
    return fd


if __name__ == "__main__":
    mol = make_mol(ATOM)
    print(f"PySCF {pyscf.__version__}; {XC.upper()}/def2-SVP, grid level {GRID_LEVEL}, def2 ECP on Au")
    print(f"atom_charges() = {mol.atom_charges().tolist()}, proton numbers = "
          f"{[radi.elements_proton(x) for x in mol.elements]}")
    stock = analytic_gradient()
    grks.grids_response_becke = fixed_grids_response_becke()
    fixed = analytic_gradient()
    fd = finite_difference()
    print(f"{'atom':>4} {'axis':>4} {'stock analytic':>16} {'fixed analytic':>16} {'energy FD':>16}")
    for ia in range(mol.natm):
        for k in range(3):
            print(f"{mol.atom_symbol(ia):>4} {'xyz'[k]:>4} {stock[ia, k]:+16.10f} "
                  f"{fixed[ia, k]:+16.10f} {fd[ia, k]:+16.10f}")
    print(f"max |stock - FD| = {np.abs(stock - fd).max():.2e} Ha/Bohr")
    print(f"max |fixed - FD| = {np.abs(fixed - fd).max():.2e} Ha/Bohr "
          f"(central-difference truncation at delta {DELTA} Bohr)")
