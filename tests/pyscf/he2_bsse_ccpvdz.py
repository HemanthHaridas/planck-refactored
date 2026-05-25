#!/usr/bin/env python3
"""
PySCF vs Planck cross-check: He2 BSSE / counterpoise correction, RHF/cc-pVDZ.

Matches Planck input: tests/inputs/regression/bsse/he2_bsse_ccpvdz.hfinp

Geometry: two He atoms 3.0 Angstrom apart on the z axis.
Reference: closed-shell RHF, Cartesian cc-pVDZ basis, symmetry off.

The Boys-Bernardi counterpoise procedure produces five SCF energies:

    E(AB)   dimer in the dimer basis
    E(A)    monomer A in its own (monomer) basis
    E(B)    monomer B in its own (monomer) basis
    E(A)*   monomer A in the dimer basis (B as ghost)
    E(B)*   monomer B in the dimer basis (A as ghost)

and the derived quantities BSSE, raw interaction, and CP-corrected interaction.
This script computes all of these with PySCF (ghost atoms via the "ghost-He"
label) and parses the matching values from Planck's counterpoise report, then
checks every one of the eight numbers agrees to TOLERANCE.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from pyscf import gto, scf

CASE = "he2_bsse_ccpvdz"
TOLERANCE = 1e-8

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "tests/inputs/regression/bsse/he2_bsse_ccpvdz.hfinp"
PLANCK_EXE = REPO_ROOT / "build/hartree-fock"

BASIS = "cc-pVDZ"
A_XYZ = (0.0, 0.0, 0.0)
B_XYZ = (0.0, 0.0, 3.0)

# Planck counterpoise report lines, e.g.
#   E(AB)  dimer basis        :      -5.7103200891 Eh
PLANCK_PATTERNS = {
    "e_dimer": re.compile(r"E\(AB\)\s+dimer basis\s*:\s*([-+0-9Ee.]+)\s+Eh"),
    "e_mono_a": re.compile(r"E\(A\)\s+monomer basis\s*:\s*([-+0-9Ee.]+)\s+Eh"),
    "e_mono_b": re.compile(r"E\(B\)\s+monomer basis\s*:\s*([-+0-9Ee.]+)\s+Eh"),
    "e_mono_a_cp": re.compile(r"E\(A\)\*\s+dimer basis \(CP\)\s*:\s*([-+0-9Ee.]+)\s+Eh"),
    "e_mono_b_cp": re.compile(r"E\(B\)\*\s+dimer basis \(CP\)\s*:\s*([-+0-9Ee.]+)\s+Eh"),
    "bsse": re.compile(r"BSSE\s*:\s*([-+0-9Ee.]+)\s+Eh"),
    "interaction_raw": re.compile(r"Interaction \(uncorrected\)\s*:\s*([-+0-9Ee.]+)\s+Eh"),
    "interaction_cp": re.compile(r"Interaction \(CP-corrected\)\s*:\s*([-+0-9Ee.]+)\s+Eh"),
}


def rhf_energy(atoms: list[tuple[str, tuple[float, float, float]]], basis_map) -> float:
    mol = gto.Mole()
    mol.atom = [(label, xyz) for (label, xyz) in atoms]
    mol.basis = basis_map
    mol.charge = 0
    mol.spin = 0
    mol.cart = True
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return float(mf.e_tot)


def run_pyscf() -> dict[str, float]:
    # Monomer basis: just the real atom. Dimer basis: real atom + ghost partner.
    e_dimer = rhf_energy([("He", A_XYZ), ("He", B_XYZ)], BASIS)
    e_mono_a = rhf_energy([("He", A_XYZ)], BASIS)
    e_mono_b = rhf_energy([("He", B_XYZ)], BASIS)
    e_mono_a_cp = rhf_energy(
        [("He", A_XYZ), ("ghost-He", B_XYZ)], {"He": BASIS, "ghost-He": BASIS}
    )
    e_mono_b_cp = rhf_energy(
        [("ghost-He", A_XYZ), ("He", B_XYZ)], {"He": BASIS, "ghost-He": BASIS}
    )

    bsse = (e_mono_a_cp - e_mono_a) + (e_mono_b_cp - e_mono_b)
    interaction_raw = e_dimer - e_mono_a - e_mono_b
    interaction_cp = e_dimer - e_mono_a_cp - e_mono_b_cp

    return {
        "e_dimer": e_dimer,
        "e_mono_a": e_mono_a,
        "e_mono_b": e_mono_b,
        "e_mono_a_cp": e_mono_a_cp,
        "e_mono_b_cp": e_mono_b_cp,
        "bsse": bsse,
        "interaction_raw": interaction_raw,
        "interaction_cp": interaction_cp,
    }


def run_planck() -> dict[str, float]:
    if not PLANCK_EXE.exists():
        raise RuntimeError(f"Planck executable not found: {PLANCK_EXE}")

    proc = subprocess.run(
        [str(PLANCK_EXE), str(INPUT_PATH)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            "Planck counterpoise run failed\n"
            f"exit code: {proc.returncode}\n"
            "---- output ----\n"
            f"{output}"
        )

    values: dict[str, float] = {}
    for key, pattern in PLANCK_PATTERNS.items():
        match = pattern.search(output)
        if not match:
            raise RuntimeError(
                f"Could not parse '{key}' from Planck counterpoise report\n"
                "---- output ----\n"
                f"{output}"
            )
        values[key] = float(match.group(1))
    return values


def main() -> int:
    pyscf_vals = run_pyscf()
    planck_vals = run_planck()

    print(f"CASE:          {CASE}")
    max_delta = 0.0
    for key in PLANCK_PATTERNS:
        p = pyscf_vals[key]
        q = planck_vals[key]
        delta = abs(p - q)
        max_delta = max(max_delta, delta)
        print(f"{key:>16} : PYSCF {p:18.10f}  PLANCK {q:18.10f}  DELTA {delta:.2e} Eh")

    status = "PASS" if max_delta < TOLERANCE else "FAIL"
    print(f"DELTA_MAX:     {max_delta:.2e} Eh")
    print(f"STATUS:        {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
