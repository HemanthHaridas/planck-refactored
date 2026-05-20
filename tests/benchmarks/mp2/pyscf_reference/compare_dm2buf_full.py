#!/usr/bin/env python3
"""
Compare Planck's dm2buf_full tensor with PySCF equivalent.

Extracts dm2buf_full from Planck debug output and reconstructs the equivalent
from PySCF using its MP2 gradient infrastructure.

Usage:
  PLANCK_DEBUG_DM2BUF=1 ./build/hartree-fock tests/inputs/regression/post_hf/water_rmp2_gradient_sto3g.hfinp > /tmp/planck.out
  python3 compare_dm2buf_full.py
"""

import sys
import re
import numpy as np
from pathlib import Path

# Add paths for PySCF
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyscf"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyscf" / ".venv" / "lib" / "python3.14" / "site-packages"))

try:
    from pyscf import gto, scf, mp
    from pyscf.ao2mo import _ao2mo
except ImportError as e:
    print(f"Error: Could not import PySCF. {e}")
    sys.exit(1)


def extract_planck_dm2buf(output_file="/tmp/planck.out"):
    """Extract dm2buf_full from Planck debug output."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {output_file} not found")
        print("Run: PLANCK_DEBUG_DM2BUF=1 ./build/hartree-fock ... > /tmp/planck.out")
        return None

    # Look for debug output lines starting with "PLANCK_DM2BUF_FULL"
    matches = re.findall(r'PLANCK_DM2BUF_FULL\s+(.*)', content)
    if not matches:
        print("Error: Could not find PLANCK_DM2BUF_FULL debug output")
        print("Make sure to run with PLANCK_DEBUG_DM2BUF=1")
        return None

    # Parse the last match
    values_str = matches[-1]
    try:
        values = [float(x) for x in values_str.split()]
        return np.array(values)
    except ValueError as e:
        print(f"Error parsing dm2buf_full values: {e}")
        return None


def compute_pyscf_dm2buf(mf_scf):
    """
    Reconstruct dm2buf_full from PySCF's MP2 calculation.

    This mirrors how PySCF computes the pair density tensor that
    Planck calls dm2buf_full.
    """
    mol = mf_scf.mol
    nao = mol.nao_nr()

    # Run MP2 to get orbitals and amplitudes
    mp2 = mp.MP2(mf_scf)
    mp2.kernel()

    # Extract orbital information
    mo_coeff = mf_scf.mo_coeff
    mo_energy = mf_scf.mo_energy
    mo_occ = mf_scf.mo_occ

    nocc = np.count_nonzero(mo_occ)
    nvirt = nao - nocc

    # Occupied and virtual coefficients
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ == 0]

    # Get MP2 t2 amplitudes (nocc, nocc, nvirt, nvirt)
    _, t2 = mp2.kernel()

    # Transform t2 amplitudes to AO basis for construction of pair density
    # This uses PySCF's nr_e2 which is the same as used in the gradient code
    part_dm2 = _ao2mo.nr_e2(
        t2.reshape(nocc ** 2, nvirt ** 2),
        np.asarray(orbv.T, order="F"),
        (0, nao, 0, nao),
        "s1",
        "s1",
    ).reshape(nocc, nocc, nao, nao)

    # Apply RMP2 formula: 4*(ij|ab) - 2*(ia|jb)
    # This is the part_dm2_mo equivalent: the pair density in full AO space
    part_dm2 = part_dm2.transpose(0, 2, 3, 1) * 4.0 - part_dm2.transpose(0, 3, 2, 1) * 2.0

    # Now expand to full dm2buf_full by contracting with occupied coefficients
    # part_dm2 at this point is [nocc, nao, nao, nocc] as per nr_e2 convention
    # We need to contract it as Planck does

    # Reindex to match Planck's convention: part_dm2[i, p, q, j]
    # PySCF gives us: part_dm2[i, p, q, j] after the transpose
    C_occ = orbo

    dm2buf_pyscf = np.zeros((nao, nao, nao, nao))

    for p in range(nao):
        for q in range(nao):
            for r in range(nao):
                for s in range(nao):
                    val = 0.0

                    # Base contribution (matching Planck lines 322-327)
                    for i in range(nocc):
                        for j in range(nocc):
                            val += C_occ[p, i] * part_dm2[i, q, r, j] * C_occ[s, j]
                            val += C_occ[q, i] * part_dm2[i, p, r, j] * C_occ[s, j]

                    # Swap contribution (matching Planck lines 329-334)
                    for i in range(nocc):
                        for j in range(nocc):
                            val += C_occ[p, i] * part_dm2[i, q, s, j] * C_occ[r, j]
                            val += C_occ[q, i] * part_dm2[i, p, s, j] * C_occ[r, j]

                    dm2buf_pyscf[p, q, r, s] = val

    return dm2buf_pyscf.flatten()


def main():
    # Set up water molecule
    mol = gto.M(
        atom='O 0 0 0; H 0 1 1; H 1 0 0',
        basis='sto-3g',
        cart=True
    )

    # Run RHF
    mf = scf.RHF(mol)
    mf.kernel()

    # Extract Planck dm2buf_full
    planck_dm2buf = extract_planck_dm2buf()
    if planck_dm2buf is None:
        sys.exit(1)

    # Compute PySCF dm2buf_full
    pyscf_dm2buf = compute_pyscf_dm2buf(mf)

    # Compare
    nao = mol.nao_nr()
    print(f"Molecule: water, basis: STO-3G")
    print(f"Basis functions: {nao}")
    print(f"Tensor shape: ({nao}, {nao}, {nao}, {nao}) = {nao ** 4} elements")
    print()

    diff = planck_dm2buf - pyscf_dm2buf
    max_abs_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff ** 2))

    print("Comparison Results:")
    print(f"  max_abs_diff = {max_abs_diff:.15e}")
    print(f"  rms_diff     = {rms_diff:.15e}")
    print()

    # Find largest differences
    abs_diff = np.abs(diff)
    worst_indices = np.argsort(abs_diff)[-10:][::-1]

    print("Top 10 differences (by absolute value):")
    print(f"{'Index':<20} {'Planck':<20} {'PySCF':<20} {'Diff':<20}")
    print("-" * 80)

    for idx in worst_indices:
        # Convert flat index to (p,q,r,s)
        s = idx % nao
        idx //= nao
        r = idx % nao
        idx //= nao
        q = idx % nao
        p = idx // nao

        planck_val = planck_dm2buf[p * nao ** 3 + q * nao ** 2 + r * nao + s]
        pyscf_val = pyscf_dm2buf[p * nao ** 3 + q * nao ** 2 + r * nao + s]
        diff_val = planck_val - pyscf_val

        print(
            f"({p},{q},{r},{s})           "
            f"{planck_val:<20.12e} {pyscf_val:<20.12e} {diff_val:<20.12e}"
        )

    # Pattern analysis
    print("\n" + "=" * 80)
    print("Pattern Analysis:")
    print("=" * 80)

    # Check if errors follow a pattern
    threshold = 1e-10
    sign_errors = 0
    mag_errors = 0

    for idx in range(len(diff)):
        if abs_diff[idx] > threshold:
            planck_val = planck_dm2buf[idx]
            pyscf_val = pyscf_dm2buf[idx]

            # Check if it's a pure sign flip
            if abs(abs(planck_val) - abs(pyscf_val)) < 1e-14:
                sign_errors += 1
            else:
                mag_errors += 1

    print(f"Elements with sign-flip errors: {sign_errors}")
    print(f"Elements with magnitude errors: {mag_errors}")
    print(f"Total errors above threshold: {sign_errors + mag_errors}")
    print(f"Threshold: {threshold:.0e}")


if __name__ == "__main__":
    main()
