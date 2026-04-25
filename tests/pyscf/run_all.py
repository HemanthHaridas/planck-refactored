#!/usr/bin/env python3
"""
Run the full PySCF equivalence corpus against the local Planck build.

Each PySCF case script performs its own PySCF vs Planck comparison and emits a
machine-readable STATUS line. This runner centralizes discovery, uses the
checked-in PySCF virtualenv by default, and makes the suite easy to run from
CTest.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_PYTHON = HERE / ".venv" / "bin" / "python"
DEFAULT_MANIFEST = HERE / "cases.json"

STATUS_PATTERN = re.compile(r"^STATUS\s*:\s*([A-Z0-9_]+)", re.MULTILINE)
DELTA_PATTERN = re.compile(r"^(DELTA_[A-Z0-9_]+)\s*:\s*([-+0-9Ee.]+)\s+Eh", re.MULTILINE)


@dataclass
class CaseSpec:
    case_id: str
    script: Path
    kind: str
    allowed_statuses: tuple[str, ...]


@dataclass
class CaseResult:
    case_id: str
    kind: str
    outcome: str
    raw_status: str
    max_delta: float | None
    elapsed_s: float
    detail: str | None


def load_cases(manifest_path: Path) -> list[CaseSpec]:
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases: list[CaseSpec] = []
    for case in payload["cases"]:
        cases.append(
            CaseSpec(
                case_id=case["id"],
                script=(HERE.parent.parent / case["script"]).resolve(),
                kind=case["kind"],
                allowed_statuses=tuple(case.get("allowed_statuses", ["PASS"])),
            )
        )
    return cases


def select_cases(
    cases: list[CaseSpec],
    requested_ids: set[str],
    requested_kind: str,
) -> list[CaseSpec]:
    selected = cases
    if requested_ids:
        selected = [case for case in selected if case.case_id in requested_ids]
    if requested_kind != "all":
        selected = [case for case in selected if case.kind == requested_kind]
    return selected


def parse_status(output: str) -> str:
    match = STATUS_PATTERN.search(output)
    return match.group(1) if match else "MISSING_STATUS"


def parse_max_delta(output: str) -> float | None:
    values = [abs(float(value)) for _label, value in DELTA_PATTERN.findall(output)]
    return max(values) if values else None


def run_case(case: CaseSpec, python_executable: Path, timeout_s: int) -> CaseResult:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [str(python_executable), str(case.script)],
            cwd=HERE.parent.parent,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        output = proc.stdout + proc.stderr
        raw_status = parse_status(output)
        max_delta = parse_max_delta(output)
        if proc.returncode == 0 and raw_status in case.allowed_statuses:
            outcome = "PASS"
            detail = None
        elif proc.returncode == 0:
            outcome = "FAIL"
            detail = f"unexpected STATUS {raw_status!r}; expected one of {case.allowed_statuses}"
        else:
            outcome = "ERROR"
            detail = f"exit code {proc.returncode}"
        if outcome != "PASS":
            trailer = "\n".join(output.strip().splitlines()[-20:])
            detail = f"{detail}\n{trailer}".strip()
    except subprocess.TimeoutExpired:
        raw_status = "TIMEOUT"
        max_delta = None
        outcome = "ERROR"
        detail = f"timed out after {timeout_s}s"

    elapsed_s = time.monotonic() - start
    return CaseResult(
        case_id=case.case_id,
        kind=case.kind,
        outcome=outcome,
        raw_status=raw_status,
        max_delta=max_delta,
        elapsed_s=elapsed_s,
        detail=detail,
    )


def format_delta(value: float | None) -> str:
    return f"{value:.2e}" if value is not None else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PySCF equivalence cases")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--kind", choices=["all", "casscf", "cc"], default="all")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    python_executable = Path(args.python).expanduser()
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    if not python_executable.exists():
        print(f"PySCF virtualenv Python not found: {python_executable}", file=sys.stderr)
        return 1

    cases = load_cases(manifest_path)
    known_case_ids = {case.case_id for case in cases}
    requested_ids = set(args.case)
    unknown = sorted(requested_ids - known_case_ids)
    if unknown:
        print(f"Unknown case(s): {', '.join(unknown)}", file=sys.stderr)
        return 1
    selected = select_cases(cases, requested_ids, args.kind)
    if args.list:
        for case in selected:
            print(f"{case.case_id}\t{case.kind}\t{case.script.relative_to(HERE.parent.parent)}")
        return 0

    header = (
        f"{'Case':<35} {'Kind':<8} {'Outcome':<8} {'Status':<34} "
        f"{'Max Delta / Eh':>14} {'Time(s)':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    passed = 0
    failed = 0

    for case in selected:
        result = run_case(case, python_executable, args.timeout)
        print(
            f"{result.case_id:<35} {result.kind:<8} {result.outcome:<8} "
            f"{result.raw_status:<34} {format_delta(result.max_delta):>14} {result.elapsed_s:>7.1f}"
        )
        if result.outcome == "PASS":
            passed += 1
            continue
        failed += 1
        if result.detail:
            print("---- detail ----")
            print(result.detail)
            print(sep)

    total = passed + failed
    print(sep)
    print(f"{passed}/{total} passed, {failed} failed")
    print(sep)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
