"""References for the spherical MP2 gradient gates (docs/SPHERICAL_MP2_GRADIENTS.md).

For tests/inputs/regression/spherical/*mp2_spherical_gradient_631gd.hfinp:
PySCF RMP2/UMP2 with cart following the input's basis_type, Planck's GBS and
its CODATA 2018 Bohr. UHF references are stability-checked first.

The reference is a fourth-order central FD of PySCF's MP2 ENERGY (h = 0.005
Angstrom, every component), not PySCF's analytic mp2.Gradients. On the C1
fixtures PySCF's analytic MP2 gradient misses its own energy FD by up to
4.1e-7 Ha/Bohr (CH3 radical UMP2, C x: analytic 0.0277106087, FD
0.027710197), Cartesian and spherical alike, and tightening its CPHF solve
does not move it; Planck's analytic gradient matches the FD to ~1e-8. Water
(C2v) shows little gap. Both are printed. Fixed in PySCF 2.15.0 by its
Krylov/CPHF early-termination fix (pyscf/pyscf#3334); 2.14.0 still has it.
  tests/pyscf/.venv/bin/python tests/pyscf/spherical_mp2_gradient_references.py
"""
import pathlib
import re

import numpy as np
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/spherical"
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
STEP = 0.005  # Angstrom


def field(text, key):
    return re.search(rf"^\s*{key}\s+(\S+)", text, re.M).group(1)


def setup(path):
    text = path.read_text()
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    symbols = [line.split()[0] for line in coords[2:]]
    xyz = np.array([[float(v) for v in line.split()[1:4]] for line in coords[2:]])
    gbs = str(ROOT / "basis-sets" / field(text, "basis"))
    basis = {e: parse_gaussian.load(gbs, e) for e in set(symbols)}
    opts = dict(cart=field(text, "basis_type") != "spherical", charge=charge, spin=mult - 1,
                guess="minao" if field(text, "guess") == "sad" else "1e")
    return symbols, xyz, basis, opts


def mp2(symbols, xyz, basis, opts, check_stability=False):
    mol = gto.M(atom=list(zip(symbols, xyz.tolist())), unit=BOHR_ANGSTROM, verbose=0,
                cart=opts["cart"], charge=opts["charge"], spin=opts["spin"], basis=basis)
    mf = (scf.RHF if opts["spin"] == 0 else scf.UHF)(mol)
    mf.init_guess = opts["guess"]
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    if check_stability and opts["spin"] != 0:
        _, _, stable, _ = mf.stability(return_status=True)
        assert stable, "UHF reference is not a minimum"
    return mf.MP2().run()


def run(path):
    symbols, xyz, basis, opts = setup(path)
    pt = mp2(symbols, xyz, basis, opts, check_stability=True)
    analytic = pt.nuc_grad_method().kernel()
    fd = np.zeros_like(xyz)
    h = STEP / BOHR_ANGSTROM
    for a in range(len(symbols)):
        for k in range(3):
            e = {}
            for m in (-2, -1, 1, 2):
                x = xyz.copy()
                x[a, k] += m * STEP
                e[m] = mp2(symbols, x, basis, opts).e_tot
            fd[a, k] = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * h)
    print(f"{path.stem}: E(MP2) = {pt.e_tot:.10f}")
    print("  energy FD (reference):")
    for row in fd:
        print("   " + "  ".join(f"{v:15.10f}" for v in row))
    print(f"  PySCF analytic, max |analytic - FD| = {np.abs(analytic - fd).max():.2e}")
    return fd


def optimize(path):
    """E_opt at PySCF's geometric (GAU_TIGHT) minimum, for the geomopt gate.
    E_opt is second-order in a gradient error, so the analytic-gradient gap
    above (1.4e-8 on water) does not reach it."""
    from pyscf.geomopt.geometric_solver import optimize as geometric_optimize
    symbols, xyz, basis, opts = setup(path)
    pt = mp2(symbols, xyz, basis, opts)
    mol_eq = geometric_optimize(pt, convergence_set="GAU_TIGHT")
    e = mp2(symbols, mol_eq.atom_coords(unit="Angstrom"), basis, opts).e_tot
    print(f"{path.stem}: E_opt(MP2) = {e:.10f}")


if __name__ == "__main__":
    for p in sorted(CASES.glob("*mp2_spherical_gradient_631gd.hfinp")):
        run(p)
    optimize(CASES / "water_rmp2_spherical_geomopt_631gd.hfinp")
