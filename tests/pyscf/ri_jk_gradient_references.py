"""References for the RI-JK gradient gates (docs/RI_JK_ROUTINE_SCOPE.md step 9).

PySCF density-fitted RHF gradients (mf.density_fit().nuc_grad_method()), with
the orbital and auxiliary bases read from Planck's GBS files and Planck's
CODATA 2018 Bohr, next to a fourth-order central FD of PySCF's own RI energy.
For a *geomopt* input, E_opt at PySCF's geometric GAU_TIGHT minimum; for a
*freq* input, frequencies from central differences of the analytic gradient.
  tests/pyscf/.venv/bin/python tests/pyscf/ri_jk_gradient_references.py [INPUT ...]
"""
import pathlib
import re
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
STEP = 1e-3  # Bohr


def field(text, key):
    m = re.search(rf"^\s*{key}\s+(\S+)", text, re.M)
    return m.group(1) if m else None


def setup(path):
    text = path.read_text()
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    symbols = [line.split()[0] for line in coords[2:]]
    xyz = np.array([[float(v) for v in line.split()[1:4]] for line in coords[2:]])
    gbs = str(ROOT / "basis-sets" / field(text, "basis"))
    aux = str(ROOT / "basis-sets" / field(text, "mp2_ri_basis"))
    opts = dict(
        cart=field(text, "basis_type") != "spherical", charge=charge, spin=mult - 1,
        basis={e: parse_gaussian.load(gbs, e) for e in set(symbols)},
        aux={e: parse_gaussian.load(aux, e) for e in set(symbols)},
        ref=field(text, "scf_type"))
    return symbols, xyz, opts


def rijk(symbols, xyz, opts):
    mol = gto.M(atom=list(zip(symbols, xyz.tolist())), unit=BOHR_ANGSTROM, verbose=0,
                cart=opts["cart"], charge=opts["charge"], spin=opts["spin"], basis=opts["basis"])
    mf = {"rhf": scf.RHF, "uhf": scf.UHF, "rohf": scf.ROHF}[opts["ref"]](mol)
    mf = mf.density_fit(auxbasis=opts["aux"])
    mf.init_guess = "1e"
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    return mf


def run(path):
    symbols, xyz, opts = setup(path)
    mf = rijk(symbols, xyz, opts)
    analytic = mf.nuc_grad_method().kernel()
    fd = np.zeros_like(xyz)
    for a in range(len(symbols)):
        for k in range(3):
            e = {}
            for m in (-2, -1, 1, 2):
                x = xyz.copy()
                x[a, k] += m * STEP * BOHR_ANGSTROM
                e[m] = rijk(symbols, x, opts).e_tot
            fd[a, k] = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * STEP)
    print(f"{path.stem}: E(RI-JK) = {mf.e_tot:.10f}")
    print("  PySCF analytic:")
    for row in analytic:
        print("   " + "  ".join(f"{v:15.10f}" for v in row))
    print(f"  max |analytic - energy FD| = {np.abs(analytic - fd).max():.2e}")


def optimize(path):
    """E_opt at PySCF's geometric (GAU_TIGHT) minimum, for the geomopt gate."""
    from pyscf.geomopt.geometric_solver import optimize as geometric_optimize
    symbols, xyz, opts = setup(path)
    mol_eq = geometric_optimize(rijk(symbols, xyz, opts), convergence_set="GAU_TIGHT")
    e = rijk(symbols, mol_eq.atom_coords(unit="Angstrom"), opts).e_tot
    print(f"{path.stem}: E_opt(RI-JK) = {e:.10f}")


def frequencies(path, h=5e-3):
    """Central differences of PySCF's analytic DF gradient at Planck's Hessian
    step (h Bohr), then thermo.harmonic_analysis: Planck's own scheme."""
    from pyscf.hessian import thermo
    symbols, xyz, opts = setup(path)
    mf = rijk(symbols, xyz, opts)
    n3 = 3 * len(symbols)
    hess = np.zeros((n3, n3))
    for j in range(n3):
        a, k = divmod(j, 3)
        g = {}
        for sgn in (1, -1):
            x = xyz.copy()
            x[a, k] += sgn * h * BOHR_ANGSTROM
            g[sgn] = rijk(symbols, x, opts).nuc_grad_method().kernel().ravel()
        hess[:, j] = (g[1] - g[-1]) / (2 * h)
    hess = 0.5 * (hess + hess.T)
    n = len(symbols)
    freqs = thermo.harmonic_analysis(mf.mol, hess.reshape(n, 3, n, 3).transpose(0, 2, 1, 3))
    print(f"{path.stem}: E(RI-JK) = {mf.e_tot:.10f}")
    print("  frequencies (cm-1): " + "  ".join(f"{f:.2f}" for f in freqs["freq_wavenumber"]))


if __name__ == "__main__":
    for p in map(lambda q: pathlib.Path(q).resolve(), sys.argv[1:]):
        if "geomopt" in p.stem:
            optimize(p)
        elif "freq" in p.stem:
            frequencies(p)
        else:
            run(p)
