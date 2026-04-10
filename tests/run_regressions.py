#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "rhf_total_energy": re.compile(r"^\s*Total Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "mp2_corr_energy": re.compile(r"^\s*Correlation Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "mp2_total_energy": re.compile(r"^\s*Total MP2 Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "casscf_corr_energy": re.compile(r"^\s*CASSCF Correlation Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "casscf_total_energy": re.compile(r"^\s*CASSCF Total Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "dft_total_energy": re.compile(r"^\s*(?:\[INF\]\s+)?DFT Energy\s*:\s*([-+0-9Ee\.]+)\s+Eh", re.MULTILINE),
    "casscf_sa_gnorm": re.compile(r"sa_g=([-+0-9Ee\.]+)"),
    "casscf_root_screen_gnorm": re.compile(r"root_screen_g=([-+0-9Ee\.]+)"),
    "casscf_max_root_gnorm": re.compile(r"max_root_g=([-+0-9Ee\.]+)"),
    "gradient_max": re.compile(r"Gradient max\|g\|\s*:\s*([-+0-9Ee\.]+)\s+Ha/Bohr"),
    "gradient_rms": re.compile(r"Gradient rms\|g\|\s*:\s*([-+0-9Ee\.]+)\s+Ha/Bohr"),
    "point_group": re.compile(r"(?:Point Group\s*:\s*|Detected point group\s+)([A-Za-z0-9_+\-]+)"),
}

COUNT_PATTERNS: dict[str, re.Pattern[str]] = {
    "gradient_atom_lines": re.compile(r"Atom\s+\d+\s*:\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+"),
}

ITER_PATTERNS: dict[str, re.Pattern[str]] = {
    "scf_converged_iterations": re.compile(r"SCF Converged after\s+(\d+)\s+iterations"),
}

HOMO_PATTERN = re.compile(
    r"^\s*\d+\s+(?:[A-Za-z0-9_+\-]+\s+)?([-+0-9Ee\.]+)\s+<-- HOMO\b",
    re.MULTILINE,
)

LUMO_PATTERN = re.compile(
    r"^\s*\d+\s+(?:[A-Za-z0-9_+\-]+\s+)?([-+0-9Ee\.]+)\s+<-- LUMO\b",
    re.MULTILINE,
)


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    duration_s: float
    details: list[str]


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_metrics(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(output)
        if not matches:
            continue
        value = matches[-1]
        if key == "point_group":
            metrics[key] = value.strip()
        else:
            metrics[key] = float(value)

    for key, pattern in COUNT_PATTERNS.items():
        metrics[key] = len(pattern.findall(output))

    for key, pattern in ITER_PATTERNS.items():
        matches = pattern.findall(output)
        if matches:
            metrics[key] = int(matches[-1])

    homo_matches = HOMO_PATTERN.findall(output)
    lumo_matches = LUMO_PATTERN.findall(output)
    if homo_matches:
        metrics["homo_energy"] = float(homo_matches[-1])
    if lumo_matches:
        metrics["lumo_energy"] = float(lumo_matches[-1])
    if homo_matches and lumo_matches:
        metrics["homo_lumo_gap"] = float(lumo_matches[-1]) - float(homo_matches[-1])

    return metrics


def approx_equal(a: float, b: float, atol: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol


def resolve_executable(case: dict[str, Any], repo_root: Path, build_dir: str, default_executable: Path) -> Path:
    executable_value = case.get("executable")
    if executable_value is None:
        return default_executable

    executable_path = Path(str(executable_value))
    if executable_path.is_absolute():
        return executable_path

    if executable_path.parent == Path("."):
        return repo_root / build_dir / executable_path.name

    return repo_root / executable_path


def run_case(case: dict[str, Any], repo_root: Path, build_dir: str, default_executable: Path) -> CaseResult:
    case_id = case["id"]
    input_path = repo_root / case["input"]
    timeout_s = int(case.get("timeout_s", 120))
    executable = resolve_executable(case, repo_root, build_dir, default_executable)

    start = time.perf_counter()
    proc = subprocess.run(
        [str(executable), str(input_path)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    duration_s = time.perf_counter() - start

    output = proc.stdout + proc.stderr
    metrics = extract_metrics(output)
    details: list[str] = []
    passed = True

    expected_exit = int(case.get("expected_exit_code", 0))
    if proc.returncode != expected_exit:
        passed = False
        details.append(
            f"exit code mismatch: expected {expected_exit}, got {proc.returncode}"
        )

    for needle in case.get("contains", []):
        if needle not in output:
            passed = False
            details.append(f"missing required text: {needle!r}")

    for needle in case.get("not_contains", []):
        if needle in output:
            passed = False
            details.append(f"found forbidden text: {needle!r}")

    for check in case.get("checks", []):
        ctype = check["type"]

        if ctype == "metric_present":
            metric = check["metric"]
            if metric not in metrics:
                passed = False
                details.append(f"missing metric: {metric}")

        elif ctype == "metric_close":
            metric = check["metric"]
            expected = float(check["expected"])
            atol = float(check.get("atol", 1e-9))
            actual = metrics.get(metric)
            if actual is None or not approx_equal(float(actual), expected, atol):
                passed = False
                details.append(
                    f"{metric} mismatch: expected {expected:.10f} +/- {atol:.2e}, got {actual}"
                )

        elif ctype == "metric_le":
            metric = check["metric"]
            threshold = float(check["value"])
            actual = metrics.get(metric)
            if actual is None or not float(actual) <= threshold:
                passed = False
                details.append(f"{metric} expected <= {threshold}, got {actual}")

        elif ctype == "metric_ge":
            metric = check["metric"]
            threshold = float(check["value"])
            actual = metrics.get(metric)
            if actual is None or not float(actual) >= threshold:
                passed = False
                details.append(f"{metric} expected >= {threshold}, got {actual}")

        elif ctype == "metric_lt_metric":
            left = check["left"]
            right = check["right"]
            lv = metrics.get(left)
            rv = metrics.get(right)
            if lv is None or rv is None or not float(lv) < float(rv):
                passed = False
                details.append(f"expected {left} < {right}, got {lv} vs {rv}")

        elif ctype == "metric_le_metric":
            left = check["left"]
            right = check["right"]
            lv = metrics.get(left)
            rv = metrics.get(right)
            if lv is None or rv is None or not float(lv) <= float(rv):
                passed = False
                details.append(f"expected {left} <= {right}, got {lv} vs {rv}")

        elif ctype == "metric_eq":
            metric = check["metric"]
            expected = check["expected"]
            actual = metrics.get(metric)
            if actual != expected:
                passed = False
                details.append(f"{metric} mismatch: expected {expected}, got {actual}")

        else:
            passed = False
            details.append(f"unknown check type: {ctype}")

    if not passed:
        details.append("---- captured output ----")
        details.extend(output.strip().splitlines()[-40:])

    return CaseResult(case_id=case_id, passed=passed, duration_s=duration_s, details=details)


def should_run(case: dict[str, Any], suite: str, selected_cases: set[str]) -> bool:
    if selected_cases and case["id"] not in selected_cases:
        return False
    if suite == "all":
        return True
    return suite in set(case.get("tags", []))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Planck regression tests")
    parser.add_argument("--manifest", default="tests/regression_cases.json")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--executable", default=None)
    parser.add_argument("--suite", default="core", choices=["smoke", "core", "extended", "all"])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / args.manifest
    manifest = load_manifest(manifest_path)
    cases = manifest["cases"]

    if args.list:
        for case in cases:
            tags = ",".join(case.get("tags", []))
            print(f"{case['id']}: {case['input']} [{tags}]")
        return 0

    executable = Path(args.executable) if args.executable else repo_root / args.build_dir / "hartree-fock"
    if not executable.exists():
        print(f"executable not found: {executable}", file=sys.stderr)
        return 2

    selected_cases = set(args.case)
    chosen = [case for case in cases if should_run(case, args.suite, selected_cases)]
    if not chosen:
        print("no cases selected", file=sys.stderr)
        return 2

    print(f"Running {len(chosen)} regression case(s) from {manifest_path}")
    failures = 0
    total_start = time.perf_counter()

    for case in chosen:
        result = run_case(case, repo_root, args.build_dir, executable)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id} ({result.duration_s:.2f}s)")
        for line in result.details:
            print(f"    {line}")
        if not result.passed:
            failures += 1

    total_duration = time.perf_counter() - total_start
    print(
        f"Completed {len(chosen)} case(s) in {total_duration:.2f}s: "
        f"{len(chosen) - failures} passed, {failures} failed"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
