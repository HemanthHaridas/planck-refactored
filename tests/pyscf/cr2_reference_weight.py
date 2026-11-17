"""How much of the exact Cr2 CASCI wavefunction does the RHF determinant hold?

Pass the all-MO Planck FCIDUMP from the 1.68 Angstrom Cartesian STO-3G
fixture. For each active size, the CASCI vector is computed in the same fixed
RHF orbitals FCIQMC samples (18 frozen, identity coefficients), and the
seeded FCIQMC reference -- orbitals 0..5 of the active block for both spins,
CI index (0, 0) -- is compared with the rest of the vector.

The L1 share |c_0| / sum |c_i| is the fraction of walkers an exactly sampled,
sign-coherent population would place on the reference, so it can be compared
directly with the driver's mean reference weight / target walkers.
"""
import argparse

import numpy as np
from pyscf import fci, lib, mcscf
from pyscf.tools import fcidump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fcidump")
    parser.add_argument("--orbitals", type=int, nargs="+", default=[10, 12])
    parser.add_argument("--walkers", type=float, default=25000.0)
    args = parser.parse_args()
    lib.num_threads(4)
    mf = fcidump.to_scf(args.fcidump)
    if (mf.mol.nao, mf.mol.nelectron, mf.mol.spin) != (38, 48, 0):
        parser.error("expected the 38-orbital, 48-electron Cr2 singlet input")
    for norb in args.orbitals:
        mc = mcscf.CASCI(mf, norb, (6, 6))
        mc.fcisolver.conv_tol = 1e-10
        energy = mc.kernel(np.eye(38))[0]
        ci = np.asarray(mc.ci)
        ss, _ = fci.spin_op.spin_square(ci, norb, (6, 6))
        flat = ci.reshape(-1)
        weights = flat ** 2
        order = np.argsort(weights)[::-1]
        cumulative = np.cumsum(weights[order])
        c0 = flat[0]
        top = np.unravel_index(order[0], ci.shape)
        alpha = fci.cistring.addrs2str(norb, 6, [top[0]])[0]
        beta = fci.cistring.addrs2str(norb, 6, [top[1]])[0]
        l1_share = abs(c0) / np.abs(flat).sum()
        print(f"CAS(12,{norb}) ndet={flat.size} E={energy:.10f} <S^2>={ss:.6f}")
        print(f"  reference c0^2={c0 ** 2:.6f}  L1 share={l1_share:.6e}  "
              f"walkers on it at {args.walkers:.0f}: {l1_share * args.walkers:.1f}")
        print(f"  largest c^2={weights[order[0]]:.6f} at alpha {alpha:#x} beta {beta:#x}; "
              f"reference rank {int(np.where(order == 0)[0][0]) + 1}")
        for fraction in (0.5, 0.9, 0.99):
            count = int(np.searchsorted(cumulative, fraction)) + 1
            print(f"  determinants for {fraction:.0%} of the norm: {count}")


if __name__ == "__main__":
    main()
