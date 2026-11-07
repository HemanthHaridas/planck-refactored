"""PySCF >= 2.14 DFT gradients with grid response are wrong with ghost atoms
(docs/PYSCF_ECP_GRID_RESPONSE_SCOPE.md, R4).

PySCF #3302 (v2.14.0) made the Becke partition look ghost atoms' radii up by
their element (radi.py, _std_symbol_without_ghost), but grids_response_becke
still uses mol.atom_charges(), which is 0 on a ghost atom. Same mechanism as
the ECP defect. PySCF 2.13.0 is not affected (both lookups give 0 there).

Water dimer with one ghost monomer, 6-31G, PBE, grid level 3. Prints the O y
gradient component against a fourth-order energy finite difference. Measured
on master 830bf89: stock 1.41e-5 Ha/Bohr off, with the fix
(upstream/0001-*.patch) 2.8e-10.
Run with a PySCF >= 2.14 on PYTHONPATH.
"""
import pyscf
from pyscf import dft, gto

ATOM = [("O", (0, 0, 0)), ("H", (0.757, 0.586, 0)), ("H", (-0.757, 0.586, 0)),
        ("ghost-O", (0.3, -2.9, 0.2)), ("ghost-H", (1.0, -3.4, 0.5)),
        ("ghost-H", (-0.2, -3.6, -0.4))]


def make_mol(atom, unit="Angstrom"):
    return gto.M(atom=atom, unit=unit, basis="6-31g", verbose=0)


def scf(mol):
    mf = dft.RKS(mol, xc="pbe")
    mf.grids.level = 3
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    return mf


if __name__ == "__main__":
    mol = make_mol(ATOM)
    g = scf(mol).nuc_grad_method()
    g.grid_response = True
    analytic = g.kernel()[0, 1]
    coords, delta = mol.atom_coords(unit="Bohr"), 2e-3

    def energy(s):
        c = coords.copy()
        c[0, 1] += s
        return scf(make_mol([(mol.atom_symbol(i), c[i]) for i in range(mol.natm)], "Bohr")).e_tot

    fd = (8 * (energy(delta) - energy(-delta)) - (energy(2 * delta) - energy(-2 * delta))) / (12 * delta)
    print(f"PySCF {pyscf.__version__}: O y analytic {analytic:+.10f}, energy FD {fd:+.10f}, "
          f"diff {analytic - fd:+.2e} Ha/Bohr")
