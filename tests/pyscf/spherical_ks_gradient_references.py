"""PySCF references for spherical KS gradients (docs/SPHERICAL_DFT_SCOPE.md step 2).

Analytic RKS/UKS gradients, cart=False, Planck's own GBS, its CODATA 2018 Bohr
and its unpruned 150 x 590 grid. grid_response=True because Planck's gradient
includes the moving-grid (Becke partition + point translation) term.
conv_tol_grad must be set: PySCF's default sqrt(conv_tol) = 1e-6 left a 3.2e-8
Ha/Bohr error in the UKS reference (gone at 1e-9, Planck agreement 3.4e-10).
Run: tests/pyscf/.venv/bin/python tests/pyscf/spherical_ks_gradient_references.py
"""
import pathlib
import re

from pyscf import dft, gto
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/spherical"
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
XC = {"b3lyp": "b3lyp", "pbe": "pbe,pbe", "hse06": "hse06"}


def run(path):
    text = path.read_text()
    basis = re.search(r"^\s*basis\s+(\S+)", text, re.M).group(1)
    exchange = re.search(r"^\s*exchange\s+(\S+)", text, re.M).group(1)
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    gbs = str(ROOT / "basis-sets" / basis)
    elements = {line.split()[0] for line in coords[2:]}
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=False, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in elements})
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol)
    mf.xc = XC[exchange]
    mf.grids.atom_grid, mf.grids.prune = (150, 590), None
    mf.init_guess, mf.conv_tol, mf.conv_tol_grad = "1e", 1e-12, 1e-9
    e = mf.kernel()
    assert mf.converged, path.stem
    g = mf.nuc_grad_method()
    g.grid_response = True
    grad = g.kernel()
    print(f"{path.stem:46s} E = {e:.10f}")
    for i, row in enumerate(grad, 1):
        print(f"  Atom {i}: " + "  ".join(f"{v:+.12f}" for v in row))


for name in ("h2o2_rks_b3lyp_gradient_spherical_631gd", "h2o2_rks_hse06_gradient_spherical_631gd",
             "h2o2_cation_uks_pbe_gradient_spherical_631gd"):
    run(CASES / f"{name}.hfinp")


def geomopt_freq(path):
    """geometric_solver optimum, then the analytic RKS Hessian's harmonic frequencies."""
    from pyscf.geomopt.geometric_solver import optimize
    from pyscf.hessian import thermo
    text = path.read_text()
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    gbs = str(ROOT / "basis-sets" / "6-31g*")
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=False, verbose=0,
                basis={e: parse_gaussian.load(gbs, e) for e in ("O", "H")})

    def make(m):
        mf = dft.RKS(m, xc="b3lyp")
        mf.grids.atom_grid, mf.grids.prune = (150, 590), None
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        return mf

    mol_eq = optimize(make(mol), convergence_set="gau_tight", maxsteps=100)
    mf_eq = make(mol_eq)
    e = mf_eq.kernel()
    freqs = thermo.harmonic_analysis(mol_eq, mf_eq.Hessian().kernel())["freq_wavenumber"]
    print(f"{path.stem:46s} E_opt = {e:.10f}")
    print("  freqs (cm^-1): " + "  ".join(f"{f:.2f}" for f in sorted(freqs.real)))


geomopt_freq(CASES / "water_rks_b3lyp_geomoptfreq_spherical_631gd.hfinp")
