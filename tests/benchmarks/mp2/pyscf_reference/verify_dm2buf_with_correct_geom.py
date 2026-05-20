#!/usr/bin/env python3
"""
Verify dm2buf_full formula using the CORRECT geometry from the test file.

The original compare_dm2buf_full.py uses O 0 0 0; H 0 1 1; H 1 0 0 (Bohr),
but the test file uses a different geometry in Angstrom.

This script uses the exact geometry from water_rmp2_gradient_sto3g.hfinp
"""

import numpy as np
from pyscf import gto, scf, mp

# Exact geometry from test file (in Angstrom)
mol = gto.M(
    atom=[['O', 0.0, 0.0, 0.0],
           ['H', 0.758602, 0.0, 0.504284],
           ['H', -0.758602, 0.0, 0.504284]],
    basis='sto-3g',
    cart=True,
    unit='angstrom'
)

print("=== Water RMP2 Gradient - dm2buf_full Verification ===")
print(f"Geometry: {mol.atom}")
print()

# Run RHF and MP2
mf = scf.RHF(mol).run()
mp2 = mp.MP2(mf).run()

nao, nocc, nvirt = mol.nao_nr(), mf.mol.nelectron//2, mol.nao_nr() - mf.mol.nelectron//2
C_occ = mf.mo_coeff[:, mf.mo_occ > 0]
C_virt = mf.mo_coeff[:, mf.mo_occ == 0]
t2 = mp2.t2

# Construct part_dm2 as Planck does
part_dm2 = np.zeros((nocc, nao, nao, nocc))  # [i, p, q, j]
for i in range(nocc):
    for j in range(nocc):
        for p in range(nao):
            for q in range(nao):
                val = 0.0
                for a in range(nvirt):
                    for b in range(nvirt):
                        tab = t2[i, j, a, b]
                        tba = t2[i, j, b, a]
                        val += C_virt[p, a] * C_virt[q, b] * (4.0 * tab - 2.0 * tba)
                part_dm2[i, p, q, j] = val

print(f"part_dm2 stats:")
print(f"  shape: {part_dm2.shape}")
print(f"  max abs: {np.max(np.abs(part_dm2))}")
print(f"  sample [0,3,4,0]: {part_dm2[0,3,4,0]}")
print()

# Try BOTH formulas
print("=== Formula Comparison ===")

# 1-term formula (from investigation document)
dm2buf_1term = np.zeros((nao, nao, nao, nao))
for p in range(nao):
    for q in range(nao):
        for r in range(nao):
            for s in range(nao):
                val = 0.0
                for i in range(nocc):
                    for j in range(nocc):
                        val += C_occ[p, i] * part_dm2[i, q, r, j] * C_occ[s, j]
                dm2buf_1term[p, q, r, s] = val

# 4-term formula (used by UMP2 and comparison script)
dm2buf_4term = np.zeros((nao, nao, nao, nao))
for p in range(nao):
    for q in range(nao):
        for r in range(nao):
            for s in range(nao):
                val = 0.0
                for i in range(nocc):
                    for j in range(nocc):
                        val += C_occ[p, i] * part_dm2[i, q, r, j] * C_occ[s, j]
                        val += C_occ[q, i] * part_dm2[i, p, r, j] * C_occ[s, j]
                        val += C_occ[p, i] * part_dm2[i, q, s, j] * C_occ[r, j]
                        val += C_occ[q, i] * part_dm2[i, p, s, j] * C_occ[r, j]
                dm2buf_4term[p, q, r, s] = val

print(f"1-term formula:")
print(f"  max abs: {np.max(np.abs(dm2buf_1term))}")
print(f"  sample [3,4,5,5]: {dm2buf_1term[3, 4, 5, 5]}")
print(f"  sample [5,5,3,4]: {dm2buf_1term[5, 5, 3, 4]}")

print(f"\n4-term formula:")
print(f"  max abs: {np.max(np.abs(dm2buf_4term))}")
print(f"  sample [3,4,5,5]: {dm2buf_4term[3, 4, 5, 5]}")
print(f"  sample [5,5,3,4]: {dm2buf_4term[5, 5, 3, 4]}")

# Compare the two
diff = dm2buf_4term - dm2buf_1term
print(f"\nDifference (4-term - 1-term):")
print(f"  max abs: {np.max(np.abs(diff))}")
print(f"  rms: {np.sqrt(np.mean(diff**2))}")

# Find which formula is likely correct by checking symmetries
print("\n=== Symmetry Analysis ===")

# Check if either formula respects expected symmetries
# The pair density should be symmetric in certain ways
print("1-term formula symmetry check:")
sym_err_1 = 0.0
for p in range(nao):
    for q in range(nao):
        for r in range(nao):
            for s in range(nao):
                # Check (pq|rs) == (rs|pq)
                v1 = dm2buf_1term[p, q, r, s]
                v2 = dm2buf_1term[r, s, p, q]
                sym_err_1 += abs(v1 - v2)
print(f"  Total symmetry error (pq,rs) vs (rs,pq): {sym_err_1}")

print("\n4-term formula symmetry check:")
sym_err_4 = 0.0
for p in range(nao):
    for q in range(nao):
        for r in range(nao):
            for s in range(nao):
                # Check (pq|rs) == (rs|pq)
                v1 = dm2buf_4term[p, q, r, s]
                v2 = dm2buf_4term[r, s, p, q]
                sym_err_4 += abs(v1 - v2)
print(f"  Total symmetry error (pq,rs) vs (rs,pq): {sym_err_4}")

# Conclusion
print("\n=== Conclusion ===")
if np.max(np.abs(dm2buf_1term)) > np.max(np.abs(dm2buf_4term)):
    print("1-term formula gives larger magnitude")
else:
    print("4-term formula gives larger magnitude")

print(f"\nFor gradient calculation, we need to integrate these with ERI derivatives.")
print(f"The correct formula should be determined by which produces the correct gradient.")
