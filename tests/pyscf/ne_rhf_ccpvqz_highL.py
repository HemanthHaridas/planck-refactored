#!/usr/bin/env python3
"""PySCF reference for ne_rhf_ccpvqz_highL — the high-L Rys regression anchor.

This is the only regression case whose Cartesian g-shell quartets reach the
L_AB+L_CD = (7,8)/(8,8) angular-momentum buckets that the Auto engine routes to
Rys (see docs/SHELLPAIR_GRANULARITY_HANDOFF.md Phase B). Every other suite case
tops out at d functions, so without this case the high-L Rys path is exercised
nowhere — which matters because the planned Rys shell-quartet hoist (Phase B)
would touch exactly that path.

Reproduces the committed reference in tests/regression_cases.json
(ne_rhf_ccpvqz_highL_pyscf -> rhf_total_energy). Run:

    tests/pyscf/.venv/bin/python tests/pyscf/ne_rhf_ccpvqz_highL.py

and confirm it prints the value the manifest pins. `cart=True` matches Planck's
Cartesian working basis; symmetry is off to match the input (.hfinp use_symm
.false.) and to keep the full set of high-L two-electron quartets in play.
"""

from __future__ import annotations

from pyscf import gto, scf

CASE = "ne_rhf_ccpvqz_highL_pyscf"
EXPECTED = -128.5435344972  # Eh; pinned in tests/regression_cases.json
TOLERANCE = 1e-7


def main() -> int:
    mol = gto.M(
        atom="Ne 0 0 0",
        basis="cc-pVQZ",
        cart=True,
        unit="Angstrom",
        verbose=0,
    )
    mol.symmetry = False
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    energy = mf.kernel()

    delta = abs(energy - EXPECTED)
    status = "PASS" if delta <= TOLERANCE else "FAIL"
    print(f"{CASE}: PySCF RHF/cc-pVQZ (cart) Ne = {energy:.10f} Eh")
    print(f"  manifest reference = {EXPECTED:.10f} Eh, |delta| = {delta:.3e} ({status})")
    print(f"  nbasis (cart) = {mol.nao_nr()}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
