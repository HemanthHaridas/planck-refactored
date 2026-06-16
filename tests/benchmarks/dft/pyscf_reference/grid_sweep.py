#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PYSCF_DIR = REPO_ROOT / "tests" / "pyscf"
if str(PYSCF_DIR) not in sys.path:
    sys.path.insert(0, str(PYSCF_DIR))

from dft_case_utils import (  # noqa: E402
    run_planck_dft_with_grid,
    run_pyscf_reference,
)

CASE_INPUTS = {
    "h2_dft_b3lyp_sto3g": REPO_ROOT / "tests/inputs/regression/dft/h2_dft_b3lyp.hfinp",
    "h2_dft_hse06_sto3g": REPO_ROOT / "tests/inputs/regression/dft/h2_dft_hse06.hfinp",
    "h2_dft_b2plyp_sto3g": REPO_ROOT / "tests/inputs/regression/dft/h2_dft_b2plyp.hfinp",
}

GRID_ORDER = ("coarse", "normal", "fine", "ultrafine")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Planck and PySCF DFT grids side by side")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    selected = args.case or list(CASE_INPUTS)
    unknown = sorted(set(selected) - set(CASE_INPUTS))
    if unknown:
        print(f"Unknown case(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    header = (
        f"{'Case':<24} {'Grid':<10} {'PySCF / Eh':>16} {'Planck / Eh':>16} "
        f"{'Delta / Eh':>12} {'PySCF s':>10} {'Planck s':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for case_id in selected:
        input_path = CASE_INPUTS[case_id]
        previous_delta: float | None = None
        for grid_name in GRID_ORDER:
            pyscf_result = run_pyscf_reference(input_path, grid_name=grid_name)
            planck_total, planck_time = run_planck_dft_with_grid(input_path, grid_name)
            delta_total = abs(pyscf_result["total_energy"] - planck_total)
            print(
                f"{case_id:<24} {grid_name:<10} {pyscf_result['total_energy']:>16.10f} "
                f"{planck_total:>16.10f} {delta_total:>12.2e} "
                f"{pyscf_result['elapsed_s']:>10.3f} {planck_time:>10.3f}"
            )
            previous_delta = delta_total
        print(sep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
