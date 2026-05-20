#!/usr/bin/env python3
"""
Compare Planck RMP2 gradient with PySCF reference after dm2buf_full fix.
"""

import sys
import re
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyscf"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyscf" / ".venv" / "lib" / "python3.14" / "site-packages"))

try:
    from pyscf import gto, scf, mp
except ImportError as e:
    print(f"Error: Could not import PySCF. {e}")
    sys.exit(1)


def extract_planck_gradient(output_file="/tmp/planck_fixed.out"):
    """Extract RMP2 gradient from Planck output."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {output_file} not found")
        return None

    # Parse gradient lines
    pattern = r"Atom\s+(\d+):\s+([-+0-9.e]+)\s+([-+0-9.e]+)\s+([-+0-9.e]+)"
    matches = re.findall(pattern, content)

    if not matches:
        print("Error: Could not find gradient values in Planck output")
        return None

    grad = []
    for atom_num, gx, gy, gz in matches:
        grad.append([float(gx), float(gy), float(gz)])

    return np.array(grad)


def compute_pyscf_rmp2_gradient(mol, mf):
    """Compute RMP2 gradient using PySCF."""
    from pyscf import grad as pyscf_grad

    # Run MP2
    mp2 = mp.MP2(mf)
    mp2.kernel()

    # Get gradient
    grad_obj = mp2.nuc_grad_method()
    g = grad_obj.kernel()

    return g


def main():
    # Water molecule
    mol = gto.M(
        atom='O 0 0 0; H 0 1 1; H 1 0 0',
        basis='sto-3g',
        cart=True
    )

    # RHF
    mf = scf.RHF(mol)
    mf.kernel()

    print("=" * 80)
    print("RMP2 Gradient Comparison: Planck vs PySCF")
    print("=" * 80)
    print()

    # Extract Planck gradient
    planck_grad = extract_planck_gradient()
    if planck_grad is None:
        sys.exit(1)

    # Compute PySCF gradient
    pyscf_grad = compute_pyscf_rmp2_gradient(mol, mf)

    # Display
    print("Planck Gradient (Ha/Bohr):")
    for i, g in enumerate(planck_grad):
        print(f"  Atom {i+1}: {g[0]:15.10f} {g[1]:15.10f} {g[2]:15.10f}")

    print()
    print("PySCF Gradient (Ha/Bohr):")
    for i, g in enumerate(pyscf_grad):
        print(f"  Atom {i+1}: {g[0]:15.10f} {g[1]:15.10f} {g[2]:15.10f}")

    print()
    print("Difference (Planck - PySCF):")
    diff = planck_grad - pyscf_grad
    for i, d in enumerate(diff):
        print(f"  Atom {i+1}: {d[0]:15.10e} {d[1]:15.10e} {d[2]:15.10e}")

    # Statistics
    max_abs_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff ** 2))

    print()
    print("=" * 80)
    print(f"Max |difference|: {max_abs_diff:.10e} Ha/Bohr")
    print(f"RMS difference:   {rms_diff:.10e} Ha/Bohr")
    print()

    if max_abs_diff < 1e-5:
        print("✓ EXCELLENT: Gradient matches PySCF to <1e-5")
    elif max_abs_diff < 1e-3:
        print("✓ GOOD: Gradient matches PySCF to <1e-3")
    elif max_abs_diff < 0.01:
        print("✗ POOR: Gradient differs by ~0.01 Ha/Bohr")
    else:
        print(f"✗ FAILED: Gradient differs by ~{max_abs_diff:.2e} Ha/Bohr")


if __name__ == "__main__":
    main()
