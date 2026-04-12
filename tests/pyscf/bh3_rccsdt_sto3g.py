#!/usr/bin/env python3
"""
PySCF vs Planck diagnostic: BH3 RCCSDT/STO-3G on the standalone tensor path.

This is the smallest closed-shell RHF/STO-3G case in the current codebase that
forces the no-fallback tensor RCCSDT branch at the present switchover limits:

- NH3/STO-3G: 8 spatial MOs -> 16 spin orbitals, 8008 determinants
- BH3/STO-3G: 8 spatial MOs -> 16 spin orbitals, 12870 determinants

The tensor backend currently keeps the determinant backstop only when both
limits are satisfied:

- n_spin_orb <= 16
- n_determinants <= 10000

BH3 is therefore the first "smallest possible" closed-shell case that exceeds
the determinant threshold while staying at the same orbital count as NH3.

This script is a diagnostic comparison rather than a regression. It reports the
PySCF RCCSDT reference energy and, if Planck does not converge on the no-
fallback path, the best staged tensor estimate that Planck reached.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

from pyscf import cc, gto, scf

CASE = "bh3_rccsdt_sto3g"
TOLERANCE = 1e-7

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "tests/inputs/regression/post_hf/bh3_rccsdt_sto3g.hfinp"
PLANCK_EXE = REPO_ROOT / "build/hartree-fock"

TENSOR_ITER_PATTERN = re.compile(
    r"^\[INF\] RCCSDT\[TENSOR-T3\]\s*:\s*(\d+)\s+E_est=([-+0-9Ee\.]+)",
    re.MULTILINE,
)
BEST_ITER_PATTERN = re.compile(
    r"best iterate from step (\d+)", re.MULTILINE
)
FAILURE_PATTERN = re.compile(
    r"no determinant backstop is available for this larger system",
    re.MULTILINE,
)
TOTAL_PATTERN = re.compile(
    r"^\s*Total RCCSDT Energy\s+([-+0-9Ee\.]+)", re.MULTILINE
)
CORR_PATTERN = re.compile(
    r"^\s*Correlation Energy\s+([-+0-9Ee\.]+)", re.MULTILINE
)


def parse_last_float(pattern: re.Pattern[str], text: str, label: str) -> float:
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"Could not parse {label} from Planck output")
    return float(matches[-1])


def run_planck() -> dict[str, float | int | bool | None]:
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

    result: dict[str, float | int | bool | None] = {
        "returncode": proc.returncode,
        "used_no_fallback_path": bool(FAILURE_PATTERN.search(output)),
        "best_tensor_step": None,
        "best_tensor_corr": None,
        "last_tensor_step": None,
        "last_tensor_corr": None,
        "planck_corr": None,
        "planck_total": None,
    }

    t3_iters = [(int(step), float(e_corr)) for step, e_corr in TENSOR_ITER_PATTERN.findall(output)]
    if t3_iters:
        last_step, last_corr = t3_iters[-1]
        result["last_tensor_step"] = last_step
        result["last_tensor_corr"] = last_corr

    best_match = BEST_ITER_PATTERN.search(output)
    if best_match and t3_iters:
        best_step = int(best_match.group(1))
        best_map = dict(t3_iters)
        result["best_tensor_step"] = best_step
        result["best_tensor_corr"] = best_map.get(best_step)

    if proc.returncode == 0:
        result["planck_corr"] = parse_last_float(CORR_PATTERN, output, "RCCSDT correlation energy")
        result["planck_total"] = parse_last_float(TOTAL_PATTERN, output, "RCCSDT total energy")

    return result


def run_pyscf() -> tuple[int, int, int, float, float, float, float]:
    mol = gto.Mole()
    mol.atom = """
B   0.000000  0.000000  0.000000
H   1.190000  0.000000  0.000000
H  -0.595000  1.030570  0.000000
H  -0.595000 -1.030570  0.000000
"""
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.cart = True
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    nmo = mol.nao_nr()
    nspin = 2 * nmo
    nelec = mol.nelectron
    ndet = math.comb(nspin, nelec)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    ccsd = cc.CCSD(mf)
    ccsd.conv_tol = 1e-10
    ccsd.conv_tol_normt = 1e-8
    ccsd_corr, *_ = ccsd.kernel()

    ccsdt = cc.RCCSDT(mf)
    ccsdt.conv_tol = 1e-10
    ccsdt.conv_tol_normt = 1e-8
    ccsdt_corr, *_ = ccsdt.kernel()

    return nmo, nspin, ndet, mf.e_tot, ccsd_corr, ccsdt_corr, mf.e_tot + ccsdt_corr


def main() -> int:
    nmo, nspin, ndet, hf_energy, pyscf_ccsd_corr, pyscf_corr, pyscf_total = run_pyscf()
    planck = run_planck()

    print(f"CASE:                 {CASE}")
    print(f"NMO:                  {nmo}")
    print(f"NSPIN:                {nspin}")
    print(f"NDET:                 {ndet}")
    print(f"HF_ENERGY:            {hf_energy:.10f} Eh")
    print(f"PYSCF_CCSD:           {pyscf_ccsd_corr:.10f} Eh")
    print(f"PYSCF_RCCSDT:         {pyscf_corr:.10f} Eh")
    print(f"TRIPLES_DELTA:        {pyscf_corr - pyscf_ccsd_corr:.2e} Eh")
    print(f"PYSCF_TOTAL:          {pyscf_total:.10f} Eh")

    if planck["planck_corr"] is not None and planck["planck_total"] is not None:
        planck_corr = float(planck["planck_corr"])
        planck_total = float(planck["planck_total"])
        delta_corr = abs(planck_corr - pyscf_corr)
        delta_total = abs(planck_total - pyscf_total)
        status = "PASS" if max(delta_corr, delta_total) < TOLERANCE else "FAIL"
        print(f"PLANCK_RCCSDT:        {planck_corr:.10f} Eh")
        print(f"PLANCK_TOTAL:         {planck_total:.10f} Eh")
        print(f"DELTA_CORR:           {delta_corr:.2e} Eh")
        print(f"DELTA_TOTAL:          {delta_total:.2e} Eh")
        print(f"STATUS:               {status}")
        return 0 if status == "PASS" else 1

    print(f"PLANCK_RETURN_CODE:   {planck['returncode']}")
    print(f"USED_NO_FALLBACK:     {planck['used_no_fallback_path']}")

    best_step = planck["best_tensor_step"]
    best_corr = planck["best_tensor_corr"]
    if best_step is not None and best_corr is not None:
        print(f"PLANCK_BEST_STAGE:    {int(best_step)}")
        print(f"PLANCK_BEST_E_EST:    {float(best_corr):.10f} Eh")
        print(f"DELTA_BEST_E_EST:     {abs(float(best_corr) - pyscf_corr):.2e} Eh")

    last_step = planck["last_tensor_step"]
    last_corr = planck["last_tensor_corr"]
    if last_step is not None and last_corr is not None:
        print(f"PLANCK_LAST_STAGE:    {int(last_step)}")
        print(f"PLANCK_LAST_E_EST:    {float(last_corr):.10f} Eh")
        print(f"DELTA_LAST_E_EST:     {abs(float(last_corr) - pyscf_corr):.2e} Eh")

    print("STATUS:               PLANCK_NO_FALLBACK_DID_NOT_CONVERGE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
