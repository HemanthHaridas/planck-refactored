"""References for the Kohn-Sham RI-JK gates (docs/RI_JK_ROUTINE_SCOPE.md steps 12-13).

PySCF dft.RKS/UKS(...).density_fit() on each input, the orbital and auxiliary
bases read from Planck's GBS files, Planck's CODATA 2018 Bohr, and Planck's
unpruned grid (grid_radial_points x grid_angular_points, prune None). Also
prints the exact (non-fitted) energy, so the fitting error is on record.
  tests/pyscf/.venv/bin/python tests/pyscf/ri_jk_dft_references.py INPUT ...
"""
import pathlib
import re
import sys

from pyscf import dft, gto
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
XC = {("pbe", "pbe"): "pbe,pbe", ("b3lyp", "lyp"): "b3lyp", ("pbe0", "pbe"): "pbe0"}


def field(text, key):
    m = re.search(rf"^\s*{key}\s+(\S+)", text, re.M)
    return m.group(1) if m else None


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
    print(f"{path.stem:40s} E_RI = {energies[0]:.10f}  E_exact = {energies[1]:.10f}  "
          f"fit error = {energies[0] - energies[1]:.3e}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(pathlib.Path(p).resolve())
