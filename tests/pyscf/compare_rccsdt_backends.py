#!/usr/bin/env python3
"""
Planck RCCSDT backend-to-backend convergence comparison.

This script exercises the new explicit `tensor_optimized` entry point against
the current implementation on the same Planck input deck and reports whether
both backends converge to the same final energy.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANCK_EXE = REPO_ROOT / "build/hartree-fock"
TOLERANCE = 1e-9

CORR_PATTERN = re.compile(
    r"^\s*(?:Correlation Energy|\[INF\]\s+CCSDT Correlation)\s+([-+0-9Ee\.]+)",
    re.MULTILINE,
)
TOTAL_PATTERN = re.compile(
    r"^\s*(?:Total RCCSDT Energy|\[INF\]\s+CCSDT Energy)\s+([-+0-9Ee\.]+)",
    re.MULTILINE,
)


def parse_last_float(pattern: re.Pattern[str], text: str, label: str) -> float:
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"Could not parse {label} from Planck output")
    return float(matches[-1])


def run_backend(input_path: Path, backend: str | None) -> tuple[float, float, str]:
    env = os.environ.copy()
    if backend is None:
        env.pop("PLANCK_RCCSDT_BACKEND", None)
    else:
        env["PLANCK_RCCSDT_BACKEND"] = backend

    proc = subprocess.run(
        [str(PLANCK_EXE), str(input_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"Backend run failed for {backend or 'auto'}\n"
            f"exit code: {proc.returncode}\n"
            f"---- output ----\n{output}"
        )
    return (
        parse_last_float(CORR_PATTERN, output, "RCCSDT correlation energy"),
        parse_last_float(TOTAL_PATTERN, output, "RCCSDT total energy"),
        output,
    )


def compare_case(label: str, input_relpath: str, current_backend: str | None, new_backend: str) -> int:
    input_path = REPO_ROOT / input_relpath
    current_corr, current_total, current_output = run_backend(input_path, current_backend)
    new_corr, new_total, new_output = run_backend(input_path, new_backend)

    delta_corr = abs(current_corr - new_corr)
    delta_total = abs(current_total - new_total)

    print(f"CASE:              {label}")
    print(f"INPUT:             {input_relpath}")
    print(f"CURRENT_BACKEND:   {current_backend or 'auto'}")
    print(f"NEW_BACKEND:       {new_backend}")
    print(f"CURRENT_CORR:      {current_corr:.10f} Eh")
    print(f"NEW_CORR:          {new_corr:.10f} Eh")
    print(f"DELTA_CORR:        {delta_corr:.2e} Eh")
    print(f"CURRENT_TOTAL:     {current_total:.10f} Eh")
    print(f"NEW_TOTAL:         {new_total:.10f} Eh")
    print(f"DELTA_TOTAL:       {delta_total:.2e} Eh")

    # Surface backend markers so it is obvious that the requested path ran.
    print(f"CURRENT_MARKER:    {'RCCSDT[OPT]' in current_output}")
    print(f"NEW_MARKER:        {'RCCSDT[OPT]' in new_output}")
    print("")

    return 0 if max(delta_corr, delta_total) <= TOLERANCE else 1


def main() -> int:
    if not PLANCK_EXE.exists():
        raise RuntimeError(f"Planck executable not found: {PLANCK_EXE}")

    status = 0
    status |= compare_case(
        label="LiH determinant vs optimized",
        input_relpath="tests/inputs/regression/post_hf/lih_rccsdt_sto3g.hfinp",
        current_backend="determinant_prototype",
        new_backend="tensor_optimized",
    )
    status |= compare_case(
        label="Water tensor vs optimized",
        input_relpath="tests/inputs/regression/post_hf/water_rccsdt_sto3g.hfinp",
        current_backend="tensor_production",
        new_backend="tensor_optimized",
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
