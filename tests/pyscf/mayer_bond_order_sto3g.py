"""
PySCF reference: Mayer bond orders, closed- and open-shell.

Matches Planck inputs:
  tests/inputs/regression/scf/h2_rhf_mayer_sto3g.hfinp        (RHF, closed shell)
  tests/inputs/regression/scf/h2o_cation_uhf_mayer_sto3g.hfinp (UHF, open shell)

PySCF has no built-in Mayer routine, so the bond order is evaluated here from
the converged density and overlap. Two conventions, which agree where they
overlap (closed shell):

  closed shell:  B_AB = Σ_{μ∈A,ν∈B} (P_total S)_μν (P_total S)_νμ        (no prefactor)
  open shell:    B_AB = 2·Σ_{μ∈A,ν∈B} [ (P^α S)(P^α S) + (P^β S)(P^β S) ]

These reproduce the textbook anchors B(H–H)=1 for H2 and B(H–H)=0.5 for H2+,
and the factor-of-2 in the open-shell form is what makes the spin-resolved
expression reduce to the closed-shell one when α=β=P_total/2.

Reference values pinned in tests/regression_cases.json:
  h2_rhf_mayer_bond_order_sto3g            B(H–H) = 1.00000000
  h2o_cation_uhf_mayer_bond_order_sto3g    B(O–H) = 0.76017799
"""

import numpy as np
from pyscf import gto, scf


def mayer(mol, dm, S):
    asl = mol.aoslice_by_atom()
    n = mol.natm
    B = np.zeros((n, n))
    spin_resolved = dm.ndim == 3
    if spin_resolved:
        PSa, PSb = dm[0] @ S, dm[1] @ S
    else:
        PS = dm @ S
    for A in range(n):
        a0, a1 = asl[A][2], asl[A][3]
        for Bb in range(A + 1, n):
            b0, b1 = asl[Bb][2], asl[Bb][3]
            v = 0.0
            for mu in range(a0, a1):
                for nu in range(b0, b1):
                    if spin_resolved:
                        v += 2.0 * (PSa[mu, nu] * PSa[nu, mu] + PSb[mu, nu] * PSb[nu, mu])
                    else:
                        v += PS[mu, nu] * PS[nu, mu]
            B[A, Bb] = B[Bb, A] = v
    return B


def main():
    mol = gto.M(atom="H 0 0 -0.3705; H 0 0 0.3705", basis="sto-3g",
                unit="Angstrom", cart=True)
    mf = scf.RHF(mol).run(verbose=0)
    b_h2 = mayer(mol, mf.make_rdm1(), mol.intor("int1e_ovlp"))[0, 1]
    print(f"h2_rhf_mayer_bond_order_sto3g           B(H-H) = {b_h2:.8f}")

    molc = gto.M(atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
                 basis="sto-3g", unit="Angstrom", charge=1, spin=1, cart=True)
    mfc = scf.UHF(molc).run(verbose=0)
    b_oh = mayer(molc, mfc.make_rdm1(), molc.intor("int1e_ovlp"))[0, 1]
    print(f"h2o_cation_uhf_mayer_bond_order_sto3g   B(O-H) = {b_oh:.8f}")


if __name__ == "__main__":
    main()
