"""PySCF references for the def2 all-electron spherical RHF gates.

Builds each basis from Planck's own GBS file (pyscf.gto.basis.parse_gaussian),
so the comparison tests the integrals and transform, not basis coefficients.
docs/DEF2_ECP.md step 1.
"""
import pathlib
import re

from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/spherical"
# Planck uses CODATA 2018 (src/base/tables.h); PySCF defaults to 2010.
BOHR_ANGSTROM = 0.529177210903


def read_case(path):
    text = path.read_text()
    basis = re.search(r"^\s*basis\s+(\S+)", text, re.M).group(1)
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].split("\n")
    atoms = [line for line in coords[3:] if line.strip()]
    return basis, "\n".join(atoms)


for case in sorted(CASES.glob("*def2*.hfinp")):
    basis, atoms = read_case(case)
    gbs = str(ROOT / "basis-sets" / basis)
    elements = {line.split()[0] for line in atoms.splitlines()}
    mol = gto.M(atom=atoms, unit=BOHR_ANGSTROM, cart=False, verbose=0,
                basis={e: parse_gaussian.load(gbs, e) for e in elements})
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    e = mf.kernel()
    assert mf.converged, case.stem
    print(f"{case.stem:40s} nao={mol.nao:4d}  E = {e:.10f}")
