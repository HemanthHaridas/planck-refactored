#!/usr/bin/env python3
"""
Run all PySCF reference calculations and print a comparison table.

Usage:
    python3 tests/pyscf/run_all.py
    python3 tests/pyscf/run_all.py --case h2_cas22_sto3g water_cas44_sto3g
    python3 tests/pyscf/run_all.py --tolerance 1e-4

Each case is run as a subprocess so a failure in one does not abort the rest.
Output is collected and parsed for CASSCF_ENERGY / SA_ENERGY / STATUS lines.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

ALL_CASES = [
    "h2_cas22_sto3g",
    "lih_cas22_sto3g",
    "water_cas44_sto3g",
    "water_cas44_631g",
    "water_cas44_ccpvdz",
    "ethylene_casscf_321g",
    "ethylene_casscf_ccpvdz",
    "water_cas44_sto3g_sa2",
    "ethylene_cas44_sto3g_sa2",
]

PLANCK_ENERGIES: dict[str, float] = {
    "h2_cas22_sto3g":           -1.1372838351,
    "lih_cas22_sto3g":          -7.8811184797,
    "water_cas44_sto3g":       -74.4700757755,
    "water_cas44_631g":        -75.5497490402,
    "water_cas44_ccpvdz":      -75.6045806122,
    "ethylene_casscf_321g":    -77.5145223872,
    "ethylene_casscf_ccpvdz":  -77.9524855976,
    "water_cas44_sto3g_sa2":   -74.7751279351,
    "ethylene_cas44_sto3g_sa2":-77.0034974301,
}

SA_CASES = {"water_cas44_sto3g_sa2", "ethylene_cas44_sto3g_sa2"}

ENERGY_PATTERN = re.compile(
    r"^(?:CASSCF_ENERGY|SA_ENERGY)\s*:\s*([-+0-9Ee.]+)", re.MULTILINE
)
STATUS_PATTERN = re.compile(r"^STATUS\s*:\s*(\w+)", re.MULTILINE)


def run_case(script: Path, timeout: int = 300) -> tuple[str | None, str]:
    """Run one script, return (pyscf_energy_str_or_None, status_str)."""
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as exc:
        return None, f"ERROR({exc})"

    energy_match = ENERGY_PATTERN.search(output)
    status_match = STATUS_PATTERN.search(output)
    energy = energy_match.group(1) if energy_match else None
    status = status_match.group(1) if status_match else "ERROR"
    return energy, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PySCF reference calculations")
    parser.add_argument(
        "--case", nargs="*", default=None,
        help="Specific case(s) to run (default: all)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-5,
        help="Energy match tolerance in Eh (default: 1e-5)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-case timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    cases = args.case if args.case else ALL_CASES
    unknown = [c for c in cases if c not in PLANCK_ENERGIES]
    if unknown:
        print(f"Unknown case(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(ALL_CASES)}", file=sys.stderr)
        return 1

    header = (
        f"{'Case':<35} {'PySCF / Eh':>14} {'Planck / Eh':>14} "
        f"{'Delta / Eh':>12} {'Status':<8} {'Time(s)':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    passed = failed = errors = 0

    for case in cases:
        script = HERE / f"{case}.py"
        if not script.exists():
            print(f"  {case:<33} {'<script missing>':<14}")
            errors += 1
            continue

        t0 = time.monotonic()
        energy_str, status = run_case(script, timeout=args.timeout)
        elapsed = time.monotonic() - t0

        planck = PLANCK_ENERGIES[case]

        if energy_str is not None:
            try:
                pyscf_e = float(energy_str)
                delta = abs(pyscf_e - planck)
                # Re-evaluate status against caller's tolerance
                if status not in ("TIMEOUT", "ERROR") and not status.startswith("ERROR"):
                    status = "PASS" if delta < args.tolerance else "FAIL"
                row = (
                    f"  {case:<33} {pyscf_e:>14.10f} {planck:>14.10f} "
                    f"{delta:>12.2e} {status:<8} {elapsed:>7.1f}"
                )
            except ValueError:
                row = f"  {case:<33} {'<parse error>':<14}"
                status = "ERROR"
        else:
            row = (
                f"  {case:<33} {'<no energy>':<14} {planck:>14.10f} "
                f"{'':>12} {status:<8} {elapsed:>7.1f}"
            )

        print(row)

        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            errors += 1

    total = passed + failed + errors
    print(sep)
    print(f"  {passed}/{total} passed, {failed} failed, {errors} errors")
    print(sep)

    return 0 if failed == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
