"""PySCF references for spherical TDDFT (docs/SPHERICAL_DFT.md step 3).

Water / 6-31G*, cart=False, Planck's own GBS, its CODATA 2018 Bohr and its
unpruned 150 x 590 grid. Prints the four lowest excitation energies (eV) and
oscillator strengths for each regression input.
Run: tests/pyscf/.venv/bin/python tests/pyscf/spherical_tddft_references.py
"""
import pathlib
import re

from pyscf import dft, gto
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/spherical"
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
HARTREE_EV = 27.211386245988    # CODATA 2018
XC = {"b3lyp": "b3lyp", "hse06": "hse06"}


def run(path):
    text = path.read_text()
    key = lambda k: re.search(rf"^\s*{k}\s+(\S+)", text, re.M).group(1)
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    gbs = str(ROOT / "basis-sets" / key("basis"))
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=False, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in ("O", "H")})
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol)
    mf.xc = XC[key("exchange")]
    mf.grids.atom_grid, mf.grids.prune = (150, 590), None
    mf.init_guess, mf.conv_tol, mf.conv_tol_grad = "1e", 1e-12, 1e-9
    e = mf.kernel()
    assert mf.converged, path.stem
    td = mf.TDA() if key("lr_method") == "tda" else mf.TDDFT()
    if key("lr_spin") == "triplet":
        td.singlet = False
    td.nstates, td.conv_tol = 4, 1e-10
    td.kernel()
    print(f"{path.stem:46s} E = {e:.10f}")
    for k, (w, f) in enumerate(zip(td.e, td.oscillator_strength()), 1):
        print(f"  root {k}: {w * HARTREE_EV:.7f} eV  f = {f:.6f}")


for name in ("water_tddft_b3lyp_casida_spherical_631gd", "water_tddft_b3lyp_triplet_tda_spherical_631gd",
             "water_tddft_hse06_casida_spherical_631gd", "water_cation_uks_tddft_b3lyp_spherical_631gd"):
    run(CASES / f"{name}.hfinp")
