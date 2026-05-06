#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PYSCF_DIR = REPO_ROOT / "tests" / "pyscf"
if str(PYSCF_DIR) not in sys.path:
    sys.path.insert(0, str(PYSCF_DIR))

from dft_case_utils import run_planck_dft, run_pyscf_reference  # noqa: E402

CASE_INPUTS = {
    "h2_dft_hse06_sto3g": REPO_ROOT / "tests/inputs/regression/dft/h2_dft_hse06.hfinp",
    "h2_dft_b2plyp_sto3g": REPO_ROOT / "tests/inputs/regression/dft/h2_dft_b2plyp.hfinp",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Planck DFT against PySCF references")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    selected = args.case or list(CASE_INPUTS)
    unknown = sorted(set(selected) - set(CASE_INPUTS))
    if unknown:
        print(f"Unknown case(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    header = (
        f"{'Case':<24} {'PySCF / Eh':>16} {'Planck / Eh':>16} "
        f"{'Delta / Eh':>12} {'PySCF s':>10} {'Planck s':>10} {'Planck/PySCF':>14}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for case_id in selected:
        input_path = CASE_INPUTS[case_id]
        pyscf_result = run_pyscf_reference(input_path)
        planck_total, planck_time = run_planck_dft(input_path)
        delta_total = abs(pyscf_result["total_energy"] - planck_total)
        ratio = planck_time / pyscf_result["elapsed_s"] if pyscf_result["elapsed_s"] > 0.0 else float("nan")
        print(
            f"{case_id:<24} {pyscf_result['total_energy']:>16.10f} {planck_total:>16.10f} "
            f"{delta_total:>12.2e} {pyscf_result['elapsed_s']:>10.3f} {planck_time:>10.3f} {ratio:>14.3f}"
        )

    print(sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
