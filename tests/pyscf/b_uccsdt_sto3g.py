#!/usr/bin/env python3
"""
PySCF vs Planck cross-check: B UCCSD/UCCSDT STO-3G

Neutral B/STO-3G is the smallest unrestricted test case used here with a clear
nonzero triples contribution while still fitting inside the determinant-space
prototype limits:
- 5 electrons -> open-shell doublet reference
- 5 spatial MOs -> 10 spin orbitals total
- determinant count C(10,5) = 252
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from pyscf import cc, gto, scf

CASE = "b_uccsdt_sto3g"
TOLERANCE = 1e-7

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANCK_EXE = REPO_ROOT / "build/hartree-fock"
UCCSD_INPUT = REPO_ROOT / "tests/inputs/regression/post_hf/b_uccsd_sto3g.hfinp"
UCCSDT_INPUT = REPO_ROOT / "tests/inputs/regression/post_hf/b_uccsdt_sto3g.hfinp"

CORR_PATTERN = re.compile(
    r"^\s*(?:Correlation Energy|\[INF\]\s+UCCSDT? Correlation)\s+([-+0-9Ee\.]+)",
    re.MULTILINE,
)
UCCSD_TOTAL_PATTERN = re.compile(
    r"^\s*(?:Total UCCSD Energy|\[INF\]\s+UCCSD Energy)\s+([-+0-9Ee\.]+)",
    re.MULTILINE,
)
UCCSDT_TOTAL_PATTERN = re.compile(
    r"^\s*(?:Total UCCSDT Energy|\[INF\]\s+UCCSDT Energy)\s+([-+0-9Ee\.]+)",
    re.MULTILINE,
)


def parse_last_float(pattern: re.Pattern[str], text: str, label: str) -> float:
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"Could not parse {label} from Planck output")
    return float(matches[-1])


def run_planck(input_path: Path, total_pattern: re.Pattern[str], label: str) -> tuple[float, float]:
    if not PLANCK_EXE.exists():
        raise RuntimeError(f"Planck executable not found: {PLANCK_EXE}")

    proc = subprocess.run(
        [str(PLANCK_EXE), str(input_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"Planck {label} run failed\n"
            f"exit code: {proc.returncode}\n"
            "---- output ----\n"
            f"{output}"
        )

    return (
        parse_last_float(CORR_PATTERN, output, f"{label} correlation energy"),
        parse_last_float(total_pattern, output, f"{label} total energy"),
    )


def run_pyscf() -> tuple[float, float, float, float, float]:
    mol = gto.Mole()
    mol.atom = "B 0.000000 0.000000 0.000000"
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 1
    mol.cart = True
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    uccsd = cc.UCCSD(mf)
    uccsd.conv_tol = 1e-10
    uccsd.conv_tol_normt = 1e-8
    uccsd_corr, *_ = uccsd.kernel()

    uccsdt = cc.UCCSDT(mf)
    uccsdt.conv_tol = 1e-10
    uccsdt.conv_tol_normt = 1e-8
    uccsdt_corr, *_ = uccsdt.kernel()

    return mf.e_tot, uccsd_corr, mf.e_tot + uccsd_corr, uccsdt_corr, mf.e_tot + uccsdt_corr


def main() -> int:
    hf_energy, pyscf_uccsd_corr, pyscf_uccsd_total, pyscf_uccsdt_corr, pyscf_uccsdt_total = run_pyscf()
    planck_uccsd_corr, planck_uccsd_total = run_planck(UCCSD_INPUT, UCCSD_TOTAL_PATTERN, "UCCSD")
    planck_uccsdt_corr, planck_uccsdt_total = run_planck(UCCSDT_INPUT, UCCSDT_TOTAL_PATTERN, "UCCSDT")

    delta_uccsd_corr = abs(pyscf_uccsd_corr - planck_uccsd_corr)
    delta_uccsd_total = abs(pyscf_uccsd_total - planck_uccsd_total)
    delta_uccsdt_corr = abs(pyscf_uccsdt_corr - planck_uccsdt_corr)
    delta_uccsdt_total = abs(pyscf_uccsdt_total - planck_uccsdt_total)
    status = "PASS" if max(
        delta_uccsd_corr,
        delta_uccsd_total,
        delta_uccsdt_corr,
        delta_uccsdt_total,
    ) < TOLERANCE else "FAIL"

    print(f"CASE:             {CASE}")
    print(f"HF_ENERGY:        {hf_energy:.10f} Eh")
    print(f"PYSCF_UCCSD:      {pyscf_uccsd_corr:.10f} Eh")
    print(f"PLANCK_UCCSD:     {planck_uccsd_corr:.10f} Eh")
    print(f"DELTA_UCCSD:      {delta_uccsd_corr:.2e} Eh")
    print(f"PYSCF_UCCSD_TOT:  {pyscf_uccsd_total:.10f} Eh")
    print(f"PLANCK_UCCSD_TOT: {planck_uccsd_total:.10f} Eh")
    print(f"DELTA_UCCSD_TOT:  {delta_uccsd_total:.2e} Eh")
    print(f"PYSCF_UCCSDT:     {pyscf_uccsdt_corr:.10f} Eh")
    print(f"PLANCK_UCCSDT:    {planck_uccsdt_corr:.10f} Eh")
    print(f"DELTA_UCCSDT:     {delta_uccsdt_corr:.2e} Eh")
    print(f"TRIPLES_DELTA:    {pyscf_uccsdt_corr - pyscf_uccsd_corr:.2e} Eh")
    print(f"PYSCF_UCCSDT_TOT: {pyscf_uccsdt_total:.10f} Eh")
    print(f"PLANCK_UCCSDT_TOT:{planck_uccsdt_total:.10f} Eh")
    print(f"DELTA_UCCSDT_TOT: {delta_uccsdt_total:.2e} Eh")
    print(f"STATUS:           {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
