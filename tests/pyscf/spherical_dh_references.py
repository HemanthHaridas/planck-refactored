"""PySCF references for spherical double-hybrid single points
(docs/SPHERICAL_DFT.md step 5).

B2PLYP on water / 6-31G*, cart=False, Planck's own GBS, its CODATA 2018 Bohr
and its unpruned 150 x 590 grid. PySCF has no named B2PLYP, so it is assembled
as in tests/pyscf/dft_case_utils.py: the hybrid SCF
0.53*HF + 0.47*B88, 0.73*LYP plus 0.27 x all-electron MP2 on the KS orbitals.
Run: tests/pyscf/.venv/bin/python tests/pyscf/spherical_dh_references.py
"""
import pathlib

from pyscf import dft, gto, mp
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/spherical"
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)


def run(path):
    coords = path.read_text().split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    gbs = str(ROOT / "basis-sets" / "6-31g*")
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=False, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in ("O", "H")})
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol)
    mf.xc = "0.53*HF + 0.47*B88, 0.73*LYP"
    mf.grids.atom_grid, mf.grids.prune = (150, 590), None
    mf.init_guess, mf.conv_tol, mf.conv_tol_grad = "1e", 1e-12, 1e-9
    e_scf = mf.kernel()
    assert mf.converged, path.stem
    e_pt2 = mp.MP2(mf).kernel()[0]
    print(f"{path.stem:42s} E_scf = {e_scf:.10f}  bare PT2 = {e_pt2:.10f}  "
          f"E = {e_scf + 0.27 * e_pt2:.10f}")


for name in ("water_rks_b2plyp_spherical_631gd", "water_cation_uks_b2plyp_spherical_631gd"):
    run(CASES / f"{name}.hfinp")
