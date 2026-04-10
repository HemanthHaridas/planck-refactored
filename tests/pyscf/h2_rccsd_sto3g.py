#!/usr/bin/env python3
"""
PySCF vs Planck cross-check: H2 RCCSD/STO-3G
Matches Planck input: tests/inputs/regression/post_hf/h2_rccsd_sto3g.hfinp

Geometry: H-H at 0.741 Angstrom
Reference: closed-shell RHF, Cartesian STO-3G basis
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from pyscf import cc, gto, scf

CASE = "h2_rccsd_sto3g"
TOLERANCE = 1e-8

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "tests/inputs/regression/post_hf/h2_rccsd_sto3g.hfinp"
PLANCK_EXE = REPO_ROOT / "build/hartree-fock"

CORR_PATTERN = re.compile(r"^\s*Correlation Energy\s+([-+0-9Ee\.]+)", re.MULTILINE)
TOTAL_PATTERN = re.compile(r"^\s*Total RCCSD Energy\s+([-+0-9Ee\.]+)", re.MULTILINE)


def parse_last_float(pattern: re.Pattern[str], text: str, label: str) -> float:
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"Could not parse {label} from Planck output")
    return float(matches[-1])


def run_planck() -> tuple[float, float]:
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
            "Planck RCCSD run failed\n"
            f"exit code: {proc.returncode}\n"
            "---- output ----\n"
            f"{output}"
        )

    return (
        parse_last_float(CORR_PATTERN, output, "RCCSD correlation energy"),
        parse_last_float(TOTAL_PATTERN, output, "RCCSD total energy"),
    )


def run_pyscf() -> tuple[float, float, float]:
    mol = gto.Mole()
    mol.atom = """
H  0.000000  0.000000  -0.370500
H  0.000000  0.000000   0.370500
"""
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.cart = True
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    e_corr, _t1, _t2 = mycc.kernel()
    e_total = mf.e_tot + e_corr
    return mf.e_tot, e_corr, e_total


def main() -> int:
    hf_energy, pyscf_corr, pyscf_total = run_pyscf()
    planck_corr, planck_total = run_planck()

    delta_corr = abs(pyscf_corr - planck_corr)
    delta_total = abs(pyscf_total - planck_total)
    status = "PASS" if max(delta_corr, delta_total) < TOLERANCE else "FAIL"

    print(f"CASE:          {CASE}")
    print(f"HF_ENERGY:     {hf_energy:.10f} Eh")
    print(f"PYSCF_CORR:    {pyscf_corr:.10f} Eh")
    print(f"PLANCK_CORR:   {planck_corr:.10f} Eh")
    print(f"DELTA_CORR:    {delta_corr:.2e} Eh")
    print(f"PYSCF_TOTAL:   {pyscf_total:.10f} Eh")
    print(f"PLANCK_TOTAL:  {planck_total:.10f} Eh")
    print(f"DELTA_TOTAL:   {delta_total:.2e} Eh")
    print(f"STATUS:        {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
