"""Compute finite-time projected and population-growth energies for the Cr2 CAS(12,10) pilot.

Pass the all-MO Planck FCIDUMP from the 1.68 Angstrom Cartesian STO-3G
fixture. The identity MO coefficients preserve the frozen and active blocks.
"""
import argparse
import math

import numpy as np
from pyscf import fci, lib, mcscf
from pyscf.tools import fcidump
from scipy.sparse.linalg import LinearOperator, expm_multiply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fcidump")
    args = parser.parse_args()
    lib.num_threads(4)
    mf = fcidump.to_scf(args.fcidump)
    if (mf.mol.nao, mf.mol.nelectron, mf.mol.spin) != (38, 48, 0):
        parser.error("expected the 38-orbital, 48-electron Cr2 singlet input")
    orbitals = np.eye(38)
    norb, electrons = 10, (6, 6)
    mc = mcscf.CASCI(mf, norb, electrons)
    h, offset = mc.get_h1eff(orbitals)
    eri = mc.get_h2eff(orbitals)
    solver = fci.direct_spin1
    diagonal = solver.make_hdiag(h, eri, norb, electrons)
    integrals = solver.absorb_h1e(h, eri, norb, electrons, 0.5)
    links = fci.cistring.gen_linkstr_index_trilidx(range(norb), electrons[0])

    # Subtract the seeded diagonal so that the exponential stays well scaled.
    # The first fixed-popcount string occupies orbitals 0..5 for both spins.
    dimension = math.comb(norb, electrons[0]) ** 2
    initial = np.zeros(dimension)
    initial[0] = 1

    def generator(vector):
        v = vector.reshape(-1)
        hv = solver.contract_2e(integrals, v, norb, electrons, (links, links))
        return diagonal[0] * v - np.asarray(hv).reshape(-1)

    operator = LinearOperator((dimension, dimension), matvec=generator,
                              rmatvec=generator, dtype=np.float64)
    windows = ((5, 6), (55, 56), (55, 60), (55, 80), (60, 80))
    max_tau = max(end for _, end in windows)
    states = expm_multiply(operator, initial, start=0, stop=max_tau, num=2 * max_tau + 1,
                           traceA=np.sum(diagonal[0] - diagonal))
    # Hermiticity gives the same projection from the Rayleigh quotient of
    # the state at half the time. Check this independently of the ratio.
    energy_origin = offset + diagonal[0]
    h_reference = -generator(initial)
    print("tau projected_Eh")
    for tau in range(max_tau + 1):
        state = states[2 * tau]
        projected = energy_origin + np.dot(h_reference, state) / state[0]
        half = states[tau]
        rayleigh = energy_origin - np.dot(half, generator(half)) / np.dot(half, half)
        if abs(rayleigh - projected) > 1e-9:
            raise RuntimeError(f"inconsistent projected energy at tau={tau}")
        print(f"{tau} {projected:.12f}")

    # The negative log change in the signed population's L1 norm is the
    # growth energy integrated over a window. Restore the energy origin
    # removed from the generator and divide by each window's duration.
    print("tau_start tau_end mean_L1_growth_Eh")
    for start, end in windows:
        norm_start = np.linalg.norm(states[2 * start], ord=1)
        norm_end = np.linalg.norm(states[2 * end], ord=1)
        growth = energy_origin - np.log(norm_end / norm_start) / (end - start)
        print(f"{start} {end} {growth:.12f}")


if __name__ == "__main__":
    main()
