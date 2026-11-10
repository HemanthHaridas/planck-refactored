"""Bound the Cr2 CAS(12,18) energy and estimate the remaining correlation with SC-NEVPT2.

The input is the all-MO FCIDUMP from the fixed-orbital 1.68 Angstrom pilot.
Only the parent active orbitals are available to both calculations. Use the
12-orbital parent as a small control with a direct CI comparison.
"""
import argparse
import json

import numpy as np
from pyscf import ao2mo, fci, gto, lib, mcscf, mrpt, scf
from pyscf.tools import fcidump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fcidump")
    parser.add_argument("--parent-orbitals", type=int, choices=(12, 18), default=18)
    parser.add_argument("--trial-orbitals", type=int, choices=(10, 12), default=12)
    args = parser.parse_args()
    if args.trial_orbitals >= args.parent_orbitals:
        parser.error("choose a trial space smaller than the parent for the correlation comparison")
    lib.num_threads(4)
    mf = fcidump.to_scf(args.fcidump)
    if (mf.mol.nao, mf.mol.nelectron, mf.mol.spin) != (38, 48, 0):
        parser.error("expected the 38-orbital Cr2 singlet FCIDUMP")

    nelec = (6, 6)
    norb = args.parent_orbitals
    parent = mcscf.CASCI(mf, norb, nelec)
    h, offset = parent.get_h1eff(np.eye(38))
    g = parent.get_h2eff(np.eye(38))
    # Use the active Hamiltonian directly in an orthonormal basis.
    mol = gto.M(verbose=0)
    mol.nao = norb
    mol.nelectron = 12
    mol.incore_anyway = True
    mol.energy_nuc = lambda *args: offset

    def set_integrals(solver):
        solver.chkfile = None
        solver.get_hcore = lambda *args: h
        solver.get_ovlp = lambda *args: np.eye(norb)
        solver._eri = ao2mo.restore(8, g, norb)
        return solver

    # All twelve high-spin occupied orbitals provide a starting valence space.
    # The subsequent singlet optimization can mix them with all other parent MOs.
    mol.spin = 12
    hs = set_integrals(scf.ROHF(mol))
    hs.conv_tol = 1e-10
    hs.max_cycle = 200
    occupied = np.diag([1.] * 12 + [0.] * (norb - 12))
    hs.kernel(dm0=(occupied, np.zeros_like(occupied)))
    if not hs.converged:
        raise RuntimeError("high-spin orbital initialization failed to converge")

    mol.spin = 0
    rhf = set_integrals(scf.RHF(mol))
    rhf.mo_coeff = hs.mo_coeff
    rhf.mo_occ = np.array([2.] * 6 + [0.] * (norb - 6))
    mc = mcscf.CASSCF(rhf, args.trial_orbitals, nelec)
    mc.fcisolver = fci.direct_spin0.FCI(mol)
    mc.fcisolver.max_space = 12
    mc.fcisolver.conv_tol = 1e-10
    mc.conv_tol = 1e-8
    mc.max_memory = 450
    mc.max_cycle_macro = 80
    mc.chkfile = None
    mc.kernel(hs.mo_coeff)
    if not mc.converged:
        raise RuntimeError("nested CASSCF failed to converge")

    # Contract independent one- and two-body density matrices with the parent
    # integrals in the optimized subspace. This is the variational upper bound.
    active = mc.mo_coeff[:, :mc.ncas]
    dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, nelec)
    rayleigh = (offset + np.einsum('pq,qp', active.T @ h @ active, dm1)
                + 0.5 * np.einsum('pqrs,pqrs', ao2mo.restore(
                    1, ao2mo.incore.full(g, active), mc.ncas), dm2))
    spin_sq = fci.spin_op.spin_square0(mc.ci, mc.ncas, nelec)[0]
    if (not np.allclose(active.T @ active, np.eye(mc.ncas), atol=1e-10, rtol=0)
            or abs(np.linalg.norm(mc.ci) - 1) > 1e-8
            or abs(spin_sq) > 1e-5 or abs(rayleigh - mc.e_tot) > 1e-8):
        raise RuntimeError("nested variational reference failed verification")
    print(f"Singlet variational energy {rayleigh:.12f}", flush=True)
    lib.num_threads(1)  # Limit per-thread density-matrix scratch.
    correction = mrpt.NEVPT(mc).kernel()
    result = dict(parent_orbitals=norb, trial_orbitals=mc.ncas, spin_squared=spin_sq,
                  core_plus_nuclear_Eh=offset, rayleigh_Eh=rayleigh,
                  sc_nevpt2_correction_Eh=correction,
                  estimated_parent_total_Eh=rayleigh + correction)
    if norb == 12:
        lib.num_threads(4)
        solver = fci.direct_spin0.FCI(mol)
        solver.conv_tol = 1e-11
        solver.max_cycle = 180
        energy, ci = solver.kernel(h, g, norb, nelec, ecore=offset)
        check = offset + fci.direct_spin1.energy(h, g, ci, norb, nelec)
        if not solver.converged or abs(energy - check) > 1e-8:
            raise RuntimeError("direct CI control failed verification")
        result["direct_ci_total_Eh"] = check
    print(json.dumps(result))


if __name__ == "__main__":
    main()
