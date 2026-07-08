"""
PySCF reference: H2 RI-FCI / cc-pVDZ (density-fitted integrals, exact FCI)
Matches Planck input: tests/inputs/regression/ri/h2_fci_rimp2_ccpvdz.hfinp

This is the clean isolation of Planck's density fitting from the CASSCF
orbital optimizer. FCI has no orbital optimization — it diagonalizes the
Hamiltonian in a fixed MO basis — so agreement here proves Planck's fitted
two-electron integrals match PySCF's to CI precision, independent of any
optimizer formulation.

We build the DF-fitted MO integrals exactly the way Planck's RI path does:
  1. 3-center (pq|Q) and 2-center (P|Q) in the AO basis (cc-pvdz-ri aux),
  2. fitted factors  B_{pq,Q} = (pq|R) [L^{-1}]_{RQ}   (Cholesky of the metric),
  3. (pq|rs) = Σ_Q B_{pq,Q} B_{rs,Q},
  4. transform to MO, add the exact one-electron core, diagonalize FCI.

Planck RI-FCI energy: -1.1634586999 Eh

The conventional gate story: this same molecule with EXACT integrals gives
FCI = -1.1634029596 (both codes); the density-fitting error is ~5.6e-5 and
is reproduced identically by Planck and PySCF, which is why RI-FCI agrees to
~1e-9 even though it is 5.6e-5 away from exact FCI. Tolerance 1e-6 — this is
a fitting-fidelity check, so it is tight like the exact-integral gates, not
loosened like the RI-CASSCF gate (whose spread is an optimizer-formulation
difference, not a fitting one).
"""

import numpy as np
from pyscf import df, fci, gto, scf

CASE = "h2_fci_ri_ccpvdz"
PLANCK_ENERGY = -1.1634586999
TOLERANCE = 1e-6
AUXBASIS = "cc-pvdz-ri"

mol = gto.M(
    atom="H 0 0 -0.3705; H 0 0 0.3705",
    basis="cc-pvdz",
    cart=True,        # match Planck 'basis_type cartesian'
    symmetry=False,   # match Planck use_symm .false.
    verbose=0,
)

mf = scf.RHF(mol)     # exact SCF — Planck does not density-fit the SCF
mf.conv_tol = 1e-12
mf.kernel()

# Density-fitted AO integrals, assembled as Planck's RI path does.
auxmol = df.addons.make_auxmol(mol, AUXBASIS)
j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")  # (nao,nao,naux)
j2c = auxmol.intor("int2c2e")
nao = mol.nao
naux = auxmol.nao

L = np.linalg.cholesky(j2c)                       # metric = L Lᵀ
j3c = j3c.reshape(nao * nao, naux)
B = np.linalg.solve(L, j3c.T).T.reshape(nao, nao, naux)  # fitted factors

C = mf.mo_coeff
Bmo = np.einsum("pi,pqQ,qj->ijQ", C, B, C)
eri_df = np.einsum("ijQ,klQ->ijkl", Bmo, Bmo)     # DF (ij|kl) in MO basis
h1 = C.T @ mf.get_hcore() @ C

e_fci, _ = fci.direct_spin0.FCI().kernel(
    h1, eri_df, nao, mol.nelectron, ecore=mol.energy_nuc()
)

delta = abs(e_fci - PLANCK_ENERGY)
status = "PASS" if delta < TOLERANCE else "FAIL"

print(f"CASE: {CASE}")
print(f"HF_ENERGY:     {mf.e_tot:.10f} Eh")
print(f"RI_FCI_ENERGY: {e_fci:.10f} Eh")
print(f"PLANCK_ENERGY: {PLANCK_ENERGY:.10f} Eh")
print(f"DELTA:         {delta:.2e} Eh")
print(f"STATUS:        {status}")
