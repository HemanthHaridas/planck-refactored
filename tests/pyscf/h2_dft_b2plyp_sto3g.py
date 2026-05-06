#!/usr/bin/env python3
"""
PySCF vs Planck cross-check: H2 B2PLYP/STO-3G
Matches Planck input: tests/inputs/regression/dft/h2_dft_b2plyp.hfinp

PySCF 2.12.1 does not expose libxc's named B2PLYP functional directly, so this
reference is assembled as the libxc-equivalent hybrid SCF expression plus the
scaled MP2 correction.
"""

from __future__ import annotations

import sys

from dft_case_utils import REPO_ROOT, run_planck_dft_with_grid, run_pyscf_reference

CASE = "h2_dft_b2plyp_sto3g"
GRID = "fine"
TOLERANCE = 1e-6
INPUT_PATH = REPO_ROOT / "tests/inputs/regression/dft/h2_dft_b2plyp.hfinp"


def main() -> int:
    pyscf_result = run_pyscf_reference(INPUT_PATH, grid_name=GRID)
    planck_total, planck_time = run_planck_dft_with_grid(INPUT_PATH, GRID)

    delta_total = abs(pyscf_result["total_energy"] - planck_total)
    status = "PASS" if delta_total < TOLERANCE else "FAIL"

    print(f"CASE:           {CASE}")
    print(f"GRID:           {GRID}")
    print(f"PYSCF_XC:       {pyscf_result['xc_string']}")
    print(f"PYSCF_SCFREF:   {pyscf_result['scf_energy']:.10f} Eh")
    print(f"PYSCF_PT2:      {pyscf_result['pt2_energy']:.10f} Eh")
    print(f"PYSCF_PT2_W:    {pyscf_result['pt2_scale']:.2f}")
    print(f"PYSCF_TOTAL:    {pyscf_result['total_energy']:.10f} Eh")
    print(f"PLANCK_TOTAL:   {planck_total:.10f} Eh")
    print(f"DELTA_TOTAL:    {delta_total:.2e} Eh")
    print(f"PYSCF_TIME:     {pyscf_result['elapsed_s']:.3f} s")
    print(f"PLANCK_TIME:    {planck_time:.3f} s")
    print(f"STATUS:         {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
