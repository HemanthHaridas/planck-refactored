"""Regenerate the independent exact energies for the FCIQMC handoff fixtures.

Run with tests/pyscf/.venv/bin/python. Uses the same GBS data and unit
conversion as Planck, but PySCF builds the integrals and solves the FCI.
"""
from pathlib import Path

from pyscf import fci, lib, scf

from def2_ecp_references import build

INPUTS = Path(__file__).resolve().parents[1] / "inputs/regression/post_hf"


def main():
    lib.num_threads(1)
    for name, cart in (("be_fciqmc_spherical_631gd", False),
                       ("rbh_8plus_fciqmc_def2svp_ecp", False),
                       ("o2_fciqmc_rohf_sto3g", True)):
        mol = build(INPUTS / (name + ".hfinp"), cart)
        mf = (scf.ROHF if mol.spin else scf.RHF)(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        solver = fci.FCI(mol, mf.mo_coeff)
        solver.conv_tol = 1e-12
        energy, _ = solver.kernel()
        if not mf.converged or not solver.converged:
            raise RuntimeError(f"{name}: reference did not converge")
        print(f"{name}: orbitals={mol.nao} nelec={mol.nelec} "
              f"nuclear={mol.energy_nuc():.12f} FCI={energy:.12f}")


if __name__ == "__main__":
    main()
