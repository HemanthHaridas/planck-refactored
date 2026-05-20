#!/usr/bin/env python3
"""Detailed comparison of imat blocks."""
import numpy as np

# PySCF values (from trace output)
pyscf_imat_top_right = np.array([
    [-2.53556490e-04, -1.67804559e-17],
    [1.49879619e-02, 7.65851869e-17],
    [-6.10661528e-17, 2.05955981e-02],
    [1.57315613e-02, 2.08496828e-17],
    [-6.79485700e-17, -4.70604180e-17]
])

pyscf_imat_bottom_left = np.array([
    [-3.19217190e-05, 9.85111715e-03, -3.82785540e-17, 4.77831391e-03, -1.54131363e-17],
    [-1.42884984e-17, 3.31420432e-17, 1.32402418e-02, -5.56110068e-18, -4.70203139e-17]
])

# Planck values (from extract_planck_imat.py)
planck_imat_top_right = np.array([
    [2.53557073e-04, 0.00000000e+00],
    [1.49879622e-02, -1.00000000e-16],
    [-2.00000000e-16, -2.05955966e-02],
    [-1.57315582e-02, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00]
])

planck_imat_bottom_left = np.array([
    [3.19217642e-05, 9.85111652e-03, -1.00000000e-16, -4.77831292e-03, 0.00000000e+00],
    [0.00000000e+00, -1.00000000e-16, -1.32402410e-02, 0.00000000e+00, 0.00000000e+00]
])

print("=" * 80)
print("IMAT_MO[:nocc, nocc:] Comparison")
print("=" * 80)
print(f"\nPySCF:\n{pyscf_imat_top_right}")
print(f"\nPlanck:\n{planck_imat_top_right}")
print(f"\nDifference:\n{planck_imat_top_right - pyscf_imat_top_right}")

# Analyze which elements flip signs
print("\n" + "=" * 80)
print("Element-wise sign analysis (top right):")
print("=" * 80)
for i in range(pyscf_imat_top_right.shape[0]):
    for j in range(pyscf_imat_top_right.shape[1]):
        p_val = pyscf_imat_top_right[i, j]
        pl_val = planck_imat_top_right[i, j]
        # Skip near-zero noise
        if abs(p_val) < 1e-15 and abs(pl_val) < 1e-15:
            print(f"  [{i},{j}]: Both ~0 (noise)")
        elif abs(p_val) > 1e-15 and abs(pl_val) > 1e-15:
            ratio = pl_val / p_val
            if ratio < 0:
                print(f"  [{i},{j}]: SIGN FLIP (ratio={ratio:.6f})")
            else:
                print(f"  [{i},{j}]: Sign same (ratio={ratio:.6f})")
        else:
            print(f"  [{i},{j}]: PySCF={p_val:.3e}, Planck={pl_val:.3e} (one is noise)")

print("\n" + "=" * 80)
print("IMAT_MO[nocc:, :nocc] Comparison")
print("=" * 80)
print(f"\nPySCF:\n{pyscf_imat_bottom_left}")
print(f"\nPlanck:\n{planck_imat_bottom_left}")
print(f"\nDifference:\n{planck_imat_bottom_left - pyscf_imat_bottom_left}")

print("\n" + "=" * 80)
print("Element-wise sign analysis (bottom left):")
print("=" * 80)
for i in range(pyscf_imat_bottom_left.shape[0]):
    for j in range(pyscf_imat_bottom_left.shape[1]):
        p_val = pyscf_imat_bottom_left[i, j]
        pl_val = planck_imat_bottom_left[i, j]
        # Skip near-zero noise
        if abs(p_val) < 1e-15 and abs(pl_val) < 1e-15:
            print(f"  [{i},{j}]: Both ~0 (noise)")
        elif abs(p_val) > 1e-15 and abs(pl_val) > 1e-15:
            ratio = pl_val / p_val
            if ratio < 0:
                print(f"  [{i},{j}]: SIGN FLIP (ratio={ratio:.6f})")
            else:
                print(f"  [{i},{j}]: Sign same (ratio={ratio:.6f})")
        else:
            print(f"  [{i},{j}]: PySCF={p_val:.3e}, Planck={pl_val:.3e} (one is noise)")

# Now compute the RHS difference
print("\n" + "=" * 80)
print("RHS Construction Impact")
print("=" * 80)

pyscf_imat_term = pyscf_imat_top_right.T - pyscf_imat_bottom_left
planck_imat_term = planck_imat_top_right.T - planck_imat_bottom_left

print(f"\nPySCF imat_term = [:nocc, nocc:].T - [nocc:, :nocc]:\n{pyscf_imat_term}")
print(f"\nPlanck imat_term:\n{planck_imat_term}")
print(f"\nDifference:\n{planck_imat_term - pyscf_imat_term}")
