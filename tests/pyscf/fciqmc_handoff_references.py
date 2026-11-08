"""Regenerate the independent exact energies for the FCIQMC handoff fixtures.

Run with tests/pyscf/.venv/bin/python. Optional input paths select additional
molecular controls. PySCF builds the integrals and solves the FCI using the
same GBS data and unit conversion as Planck.
"""
import argparse
import re
from pathlib import Path

from pyscf import fci, lib, scf

from def2_ecp_references import build

INPUTS = Path(__file__).resolve().parents[1] / "inputs/regression/post_hf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", default=[
        INPUTS / "be_fciqmc_spherical_631gd.hfinp",
        INPUTS / "rbh_8plus_fciqmc_def2svp_ecp.hfinp",
        INPUTS / "o2_fciqmc_rohf_sto3g.hfinp",
    ])
    args = parser.parse_args()
    lib.num_threads(1)
    for path in args.inputs:
        basis_type = re.search(r"(?im)^\s*basis_type\s+(cartesian|spherical)\b", path.read_text())
        if not basis_type:
            parser.error(f"{path}: expected an explicit cartesian or spherical basis_type")
        mol = build(path, cart=basis_type[1].lower() == "cartesian")
        mf = (scf.ROHF if mol.spin else scf.RHF)(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        solver = fci.FCI(mol, mf.mo_coeff)
        solver.conv_tol = 1e-12
        energy, _ = solver.kernel()
        if not mf.converged or not solver.converged:
            raise RuntimeError(f"{path.stem}: reference did not converge")
        print(f"{path.stem}: orbitals={mol.nao} nelec={mol.nelec} "
              f"nuclear={mol.energy_nuc():.12f} FCI={energy:.12f}")


if __name__ == "__main__":
    main()
