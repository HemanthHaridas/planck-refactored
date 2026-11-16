"""References for the Kohn-Sham RI-JK gates (docs/RI_JK_ROUTINE_SCOPE.md steps 12-13).

PySCF dft.RKS/UKS(...).density_fit() on each input, the orbital and auxiliary
bases read from Planck's GBS files, Planck's CODATA 2018 Bohr, and Planck's
unpruned grid (grid_radial_points x grid_angular_points, prune None). Also
prints the exact (non-fitted) energy, so the fitting error is on record. For
an input with stability_check, also the lowest eigenvalues of PySCF's orbital
Hessian on the density-fitted SCF, in Planck's conventions (as
spherical_ks_stability_references.py): RKS singlet gen_g_hop_rhf/2, triplet
rhf_external, UKS gen_g_hop_uhf.
  tests/pyscf/.venv/bin/python tests/pyscf/ri_jk_dft_references.py INPUT ...
"""
import pathlib
import re
import sys

import numpy as np

from pyscf import dft, gto, lib
from pyscf.scf import stability
from pyscf.soscf import newton_ah
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
XC = {("pbe", "pbe"): "pbe,pbe", ("b3lyp", "lyp"): "b3lyp", ("pbe0", "pbe"): "pbe0"}


def field(text, key):
    m = re.search(rf"^\s*{key}\s+(\S+)", text, re.M)
    return m.group(1) if m else None


def lowest(hop, hdiag, nroots=4):
    x0 = np.zeros((nroots, hdiag.size))
    for k, i in enumerate(np.argsort(hdiag)[:nroots]):
        x0[k, i] = 1.0
    e, _ = lib.davidson(hop, x0, lambda dx, e, x0: dx / (hdiag - e + 1e-8),
                        nroots=nroots, tol=1e-12, max_cycle=200)
    return min(e)


def print_stability(mf, mult):
    if mult == 1:
        _, hop, hdiag = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ)
        print(f"  RKS -> RKS  (singlet) {lowest(hop, hdiag) / 2:+.7f}")
        _, _, hop, hdiag = stability._gen_hop_rhf_external(mf)
        print(f"  RKS -> UKS  (triplet) {lowest(hop, hdiag):+.7f}")
    else:
        _, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ)
        print(f"  UKS -> UKS            {lowest(hop, hdiag):+.7f}")


def run(path):
    text = path.read_text()
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    elements = {line.split()[0] for line in coords[2:]}
    gbs = str(ROOT / "basis-sets" / field(text, "basis"))
    aux = str(ROOT / "basis-sets" / field(text, "mp2_ri_basis"))
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, verbose=0,
                cart=field(text, "basis_type") != "spherical", charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in elements})
    energies = []
    for fitted in (True, False):
        mf = (dft.RKS if mult == 1 else dft.UKS)(mol)
        mf.xc = XC[(field(text, "exchange"), field(text, "correlation"))]
        mf.grids.atom_grid = (int(field(text, "grid_radial_points")),
                              int(field(text, "grid_angular_points")))
        mf.grids.prune = None
        if fitted:
            mf = mf.density_fit(auxbasis={e: parse_gaussian.load(aux, e) for e in elements})
        mf.init_guess = "1e"  # Planck guess hcore
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged, path.stem
        energies.append(mf.e_tot)
        if fitted and field(text, "stability_check") == ".true.":
            print_stability(mf, mult)
    print(f"{path.stem:40s} E_RI = {energies[0]:.10f}  E_exact = {energies[1]:.10f}  "
          f"fit error = {energies[0] - energies[1]:.3e}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(pathlib.Path(p).resolve())
