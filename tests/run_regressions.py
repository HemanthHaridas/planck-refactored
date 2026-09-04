#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
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
    "rccsd_total_energy": re.compile(r"^\s*Total RCCSD Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "uccsd_total_energy": re.compile(r"^\s*Total UCCSD Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "rccsdt_total_energy": re.compile(
        r"^\s*(?:Total RCCSDT Energy|\[INF\]\s+CCSDT Energy)\s+([-+0-9Ee\.]+)",
        re.MULTILINE,
    ),
    "uccsdt_total_energy": re.compile(
        r"^\s*(?:Total UCCSDT Energy|\[INF\]\s+UCCSDT Energy)\s+([-+0-9Ee\.]+)",
        re.MULTILINE,
    ),
    "rccsdtq_total_energy": re.compile(
        r"^\s*Total RCCSDTQ Energy\s+([-+0-9Ee\.]+)",
        re.MULTILINE,
    ),
    # The generated UCC path (correlation ucc2/ucc3/...) reports through the
    # generic label: PostHF::UCCGEN is not in hf_driver.cpp's method_label chain,
    # so it falls through to "Correlated". Kept generic on purpose -- naming it
    # UCC-specifically is a driver change, not a runner one.
    "correlated_total_energy": re.compile(
        r"^\s*Total Correlated Energy\s+([-+0-9Ee\.]+)",
        re.MULTILINE,
    ),
    "casscf_corr_energy": re.compile(r"^\s*CASSCF Correlation Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "casscf_total_energy": re.compile(r"^\s*CASSCF Total Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "fci_total_energy": re.compile(r"^\s*Total FCI Energy\s+([-+0-9Ee\.]+)", re.MULTILINE),
    # FCIQMC reports a MEAN and its blocked error bar. Both are extracted so a
    # case can assert the energy within its own uncertainty (metric_within_sigma)
    # rather than against a hand-picked tolerance -- a stochastic result compared
    # with metric_close would be asserting noise.
    "fciqmc_shift_energy": re.compile(
        r"Shift energy\s+([-+0-9Ee\.]+)\s+\+/-", re.MULTILINE),
    "fciqmc_shift_error": re.compile(
        r"Shift energy\s+[-+0-9Ee\.]+\s+\+/-\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "fciqmc_projected_energy": re.compile(
        r"Projected energy\s+([-+0-9Ee\.]+)\s+\+/-", re.MULTILINE),
    "fciqmc_projected_error": re.compile(
        r"Projected energy\s+[-+0-9Ee\.]+\s+\+/-\s+([-+0-9Ee\.]+)", re.MULTILINE),
    "dft_total_energy": re.compile(r"^\s*(?:\[INF\]\s+)?DFT Energy\s*:\s*([-+0-9Ee\.]+)\s+Eh", re.MULTILINE),
    "lr_root1_energy_ev": re.compile(r"^\s*1\s+[-+0-9Ee\.]+\s+([-+0-9Ee\.]+)\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+", re.MULTILINE),
    "lr_root2_energy_ev": re.compile(r"^\s*2\s+[-+0-9Ee\.]+\s+([-+0-9Ee\.]+)\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+", re.MULTILINE),
    "lr_root3_energy_ev": re.compile(r"^\s*3\s+[-+0-9Ee\.]+\s+([-+0-9Ee\.]+)\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+", re.MULTILINE),
    "casscf_converged_via_plateau": re.compile(r"casscf_converged_via_plateau=(true|false)"),
    "casscf_sa_gnorm": re.compile(r"sa_g=([-+0-9Ee\.]+)"),
    "casscf_root_screen_gnorm": re.compile(r"root_screen_g=([-+0-9Ee\.]+)"),
    "casscf_max_root_gnorm": re.compile(r"max_root_g=([-+0-9Ee\.]+)"),
    "gradient_max": re.compile(r"Gradient max\|g\|\s*:\s*([-+0-9Ee\.]+)\s+Ha/Bohr"),
    "gradient_rms": re.compile(r"Gradient rms\|g\|\s*:\s*([-+0-9Ee\.]+)\s+Ha/Bohr"),
    # Post-geomopt converged force from the "Final max|g|" summary line. Distinct
    # from gradient_max (which catches the per-step "Gradient max|g|" prints; for
    # a geomopt run, last-match is still the step-0 value because subsequent
    # steps print only "Opt Step N : ... max|g| = ..." without the full prefix).
    "geomopt_final_gradient_max": re.compile(r"Final max\|g\|\s*:\s*([-+0-9Ee\.]+)\s+Ha/Bohr"),
    # Converged geomopt energy from the "Final Energy" summary line. For a
    # correlated geomopt (correlation rmp2/ump2) this is the CORRELATED optimized
    # energy — the quantity a PySCF MP2-geomopt reference pins. The post-opt
    # "Final Symmetry SCF" prints only the SCF energy, so rhf_total_energy /
    # uhf_total_energy cannot gate a correlated optimization.
    "geomopt_final_energy": re.compile(r"Final Energy\s*:\s*([-+0-9Ee\.]+)\s+Eh"),
    # Vibrational frequencies from the freq table. The freq output is emitted
    # through the logger so each row carries an "[INF]" prefix; the MO energy
    # table (also indexed N) does NOT carry that prefix, so requiring it here
    # disambiguates the two. The freq table has two formats: 2-col
    # "  N         freq" (no symmetry) and 3-col "  N  Irrep  freq" (with
    # symmetry, e.g. inside a geomopt+freq run); the optional irrep group
    # tolerates both.
    "vib_freq_1": re.compile(
        r"^\[INF\]\s+1\s+(?:[A-Za-z][A-Za-z0-9_'\"]*\s+)?([-+0-9Ee\.]+)\s*$",
        re.MULTILINE,
    ),
    "vib_freq_2": re.compile(
        r"^\[INF\]\s+2\s+(?:[A-Za-z][A-Za-z0-9_'\"]*\s+)?([-+0-9Ee\.]+)\s*$",
        re.MULTILINE,
    ),
    "vib_freq_3": re.compile(
        r"^\[INF\]\s+3\s+(?:[A-Za-z][A-Za-z0-9_'\"]*\s+)?([-+0-9Ee\.]+)\s*$",
        re.MULTILINE,
    ),
    "zero_point_energy_eh": re.compile(r"Zero-point energy\s*:\s*([-+0-9Ee\.]+)\s+Eh"),
    "point_group": re.compile(r"(?:Point Group\s*:\s*|Detected point group\s+)([A-Za-z0-9_+\-]+)"),
    "stability_real_internal": re.compile(
        r"RHF -> RHF \(real, internal\)\s+λ_min\s*=\s*([-+0-9Ee\.]+)"
    ),
    "stability_complex_internal": re.compile(
        r"RHF -> complex RHF \(internal\)\s+λ_min\s*=\s*([-+0-9Ee\.]+)"
    ),
    "stability_triplet_external": re.compile(
        r"RHF -> UHF \(triplet, external\)\s+λ_min\s*=\s*([-+0-9Ee\.]+)"
    ),
    "stability_uhf_internal": re.compile(
        r"UHF -> UHF \(spin-conserving, internal\)\s+λ_min\s*=\s*([-+0-9Ee\.]+)"
    ),
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
    metrics: dict[str, Any]


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pyscf_references(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("cases", {})


def extract_metrics(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(output)
        if not matches:
            continue
        value = matches[-1]
        if key in ("point_group", "casscf_converged_via_plateau"):
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


def decode_stream(value: Any) -> str:
    """Return value as str; tolerates None and bytes (from TimeoutExpired)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


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


def build_command(executable: Path, input_path: Path) -> list[str]:
    if executable.suffix == ".py":
        return [sys.executable, str(executable), str(input_path)]
    return [str(executable), str(input_path)]


def resolve_metric_expectation(
    case_id: str,
    metric: str,
    check: dict[str, Any],
    pyscf_references: dict[str, Any],
) -> tuple[float, float]:
    override = pyscf_references.get(case_id, {}).get(metric)
    if override is None:
        return float(check["expected"]), float(check.get("atol", 1e-9))
    return float(override["expected"]), float(override.get("atol", check.get("atol", 1e-9)))


def within_sigma_failure(
    value: float | None,
    sigma: float | None,
    expected: float,
    n_sigma: float,
    metric: str,
    sigma_metric: str,
) -> str | None:
    """Return a failure detail string, or None if the value is within n_sigma.

    Split out from the check dispatch so it can be exercised directly: this is
    the only assertion in the runner with arithmetic worth testing, and a
    statistical gate that silently always passes is worse than no gate.
    """
    if value is None:
        return f"missing metric: {metric}"
    if sigma is None:
        return f"missing uncertainty metric: {sigma_metric}"
    if not math.isfinite(float(sigma)) or float(sigma) < 0.0:
        return f"{sigma_metric} is not a usable uncertainty: {sigma}"
    if not math.isfinite(float(value)):
        return f"{metric} is not finite: {value}"
    deviation = abs(float(value) - expected)
    allowed = n_sigma * float(sigma)
    if deviation <= allowed:
        return None
    return (
        f"{metric} outside {n_sigma:g} sigma: got {float(value):.10f}, "
        f"expected {expected:.10f}, deviation {deviation:.3e} "
        f"> {allowed:.3e} ({sigma_metric}={float(sigma):.3e})"
    )


def checkpoint_path_for(input_path: Path) -> Path:
    # Mirror the driver's rule (src/driver.cpp): parent_dir / stem + ".hfchk".
    return input_path.with_suffix(".hfchk")


def run_setup(
    case: dict[str, Any],
    repo_root: Path,
    executable: Path,
    timeout_s: int,
) -> str | None:
    """Run a case's optional 'setup' step to seed a checkpoint fixture.

    A restart case uses `guess full`, which requires its checkpoint to already
    exist on disk. Because the driver derives the checkpoint path from the input
    stem, a separate producer case cannot write to the consumer's path. The setup
    step bridges that gap on a clean checkout: it runs a `seed_input` (which writes
    its own checkpoint via the default save-on-converge path) and then copies that
    checkpoint to the consumer input's expected path. Returns an error string on
    failure, or None on success.

    Setup schema:
      "setup": { "seed_input": "<path to a guess-sad/hcore input>" }
    The seed input must describe the same system as the case input so the copied
    checkpoint is valid for the `guess full` restart.
    """
    setup = case.get("setup")
    if not setup:
        return None

    seed_input = repo_root / setup["seed_input"]
    if not seed_input.exists():
        return f"setup seed_input not found: {seed_input}"

    proc = subprocess.run(
        build_command(executable, seed_input),
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-10:]
        return "setup seed run failed:\n" + "\n".join(tail)

    seed_chk = checkpoint_path_for(seed_input)
    if not seed_chk.exists():
        return f"setup seed did not produce a checkpoint at {seed_chk}"

    target_chk = checkpoint_path_for(repo_root / case["input"])
    if seed_chk != target_chk:
        shutil.copyfile(seed_chk, target_chk)
    return None


def run_case(
    case: dict[str, Any],
    repo_root: Path,
    build_dir: str,
    default_executable: Path,
    pyscf_references: dict[str, Any],
) -> CaseResult:
    case_id = case["id"]
    input_path = repo_root / case["input"]
    timeout_s = int(case.get("timeout_s", 120))
    executable = resolve_executable(case, repo_root, build_dir, default_executable)

    start = time.perf_counter()

    setup_error = run_setup(case, repo_root, executable, timeout_s)
    if setup_error is not None:
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_s=time.perf_counter() - start,
            details=[setup_error],
            metrics={},
        )

    try:
        # A case may declare `env`: extra environment variables for this run only.
        # Needed where a code path is reachable only by env override and has no
        # input keyword -- PLANCK_RCCSDT_BACKEND being the case in point. Layered
        # over the inherited environment so the basis path and toolchain survive.
        case_env = case.get("env")
        run_env = {**os.environ, **case_env} if case_env else None
        proc = subprocess.run(
            build_command(executable, input_path),
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=run_env,
        )
    except subprocess.TimeoutExpired as exc:
        duration_s = time.perf_counter() - start
        detail_lines = [f"timed out after {timeout_s}s"]
        # TimeoutExpired.stdout/stderr come back as bytes even when subprocess.run
        # was called with text=True; decode_stream() handles that uniformly.
        partial_output = (decode_stream(exc.stdout) + decode_stream(exc.stderr)).strip()
        if partial_output:
            detail_lines.append("---- captured output ----")
            detail_lines.extend(partial_output.splitlines()[-40:])
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_s=duration_s,
            details=detail_lines,
            metrics={},
        )
    duration_s = time.perf_counter() - start

    output = proc.stdout + proc.stderr
    metrics = extract_metrics(output)
    details: list[str] = []
    passed = True

    # A case may declare `skip_if_contains`: if the run's output contains this
    # text, the case is not applicable to this build and is reported as a PASS
    # with a skip note (rather than a failure). Used by the generated-CCSDTQ case,
    # which prints a "reconfigure with -DPLANCK_CC_MAXORDER=4" message when the
    # binary was built at a lower rank — that build simply cannot run it.
    skip_marker = case.get("skip_if_contains")
    if skip_marker and skip_marker in output:
        return CaseResult(
            case_id=case_id,
            passed=True,
            duration_s=duration_s,
            details=[f"skipped: output contained {skip_marker!r}"],
            metrics=metrics,
        )

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
            expected, atol = resolve_metric_expectation(case_id, metric, check, pyscf_references)
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

        elif ctype == "metric_within_sigma":
            # Statistical gate: |value - expected| <= n_sigma * uncertainty.
            #
            # Every other check here is an exact-value comparison, which a
            # stochastic estimator cannot satisfy -- its answer is a mean with an
            # error bar. This is the one assertion that can express "consistent
            # with the reference, given the reported uncertainty".
            #
            # The uncertainty MUST come from the run itself (a blocked standard
            # error, not a naive one -- see docs/FCIQMC_RESEARCH_SCOPE.md G2), so
            # a run that under-reports its error bar fails this check rather than
            # sliding under a hand-picked tolerance.
            metric = check["metric"]
            sigma_metric = check["sigma_metric"]
            n_sigma = float(check.get("n_sigma", 3.0))
            expected, _ = resolve_metric_expectation(case_id, metric, check, pyscf_references)
            failure = within_sigma_failure(
                metrics.get(metric), metrics.get(sigma_metric),
                expected, n_sigma, metric, sigma_metric,
            )
            if failure is not None:
                passed = False
                details.append(failure)

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

        elif ctype == "metric_close_case":
            # Resolved after all selected cases have run so we can compare
            # against the referenced case's extracted metrics.
            continue

        else:
            passed = False
            details.append(f"unknown check type: {ctype}")

    if not passed:
        details.append("---- captured output ----")
        details.extend(output.strip().splitlines()[-40:])

    return CaseResult(case_id=case_id, passed=passed, duration_s=duration_s, details=details, metrics=metrics)


def apply_cross_case_checks(
    results: list[CaseResult],
    chosen_cases: list[dict[str, Any]],
) -> None:
    results_by_id = {result.case_id: result for result in results}

    for case in chosen_cases:
        result = results_by_id[case["id"]]
        for check in case.get("checks", []):
            if check["type"] != "metric_close_case":
                continue

            metric = check["metric"]
            other_case_id = check["case"]
            other_metric = check.get("other_metric", metric)
            atol = float(check.get("atol", 1e-9))

            actual = result.metrics.get(metric)
            other_result = results_by_id.get(other_case_id)
            other_value = None if other_result is None else other_result.metrics.get(other_metric)

            if actual is None or other_value is None or not approx_equal(float(actual), float(other_value), atol):
                result.passed = False
                result.details.append(
                    f"{metric} mismatch vs {other_case_id}.{other_metric}: "
                    f"expected {other_value} +/- {atol:.2e}, got {actual}"
                )


def cmake_build_options(repo_root: Path, build_dir: str) -> dict[str, str] | None:
    """The BOOL options a build tree was configured with, from its CMakeCache.txt.

    Returns None when there is no cache to read (an --executable pointing outside
    a build tree, say), which callers treat as "cannot tell" rather than "off".

    Used by `requires_build_option` so a case that needs an opt-in feature is
    SKIPPED in a build without it rather than failing. Reading the cache is the
    only honest source: the option is a compile-time define, so the binary's own
    behaviour is what a case would otherwise have to infer from an error string.
    """
    cache = repo_root / build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    options: dict[str, str] = {}
    for line in cache.read_text(errors="replace").splitlines():
        name, sep, value = line.partition(":BOOL=")
        if sep:
            options[name.strip()] = value.strip().upper()
    return options


def missing_build_option(
    case: dict[str, Any],
    build_options: dict[str, str] | None,
) -> str | None:
    """The build option this case needs but the tree does not have, if any.

    Accepts a string or a list: some cases need MORE than one option ON, and
    depend on every one of them. `PLANCK_CC_SPIN_ADAPT` is the motivating case
    -- the generated CC kernels are only correct with it ON (CMakeLists.txt
    documents OFF as the ~4x-wrong historical emit), so a generated-kernel case
    run without it silently measures the defective emit rather than the kernel
    it means to gate. That cost a full investigation before it was noticed.
    """
    required = case.get("requires_build_option")
    if not required:
        return None
    if build_options is None:
        return None
    names = [required] if isinstance(required, str) else list(required)
    for name in names:
        if build_options.get(name) != "ON":
            return name
    return None


def should_run(case: dict[str, Any], suite: str, selected_cases: set[str]) -> bool:
    if selected_cases and case["id"] not in selected_cases:
        return False
    if suite == "all":
        return True
    return suite in set(case.get("tags", []))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Planck regression tests")
    parser.add_argument("--manifest", default="tests/regression_cases.json")
    parser.add_argument("--pyscf-refs", default="tests/pyscf/regression_references.json")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--executable", default=None)
    parser.add_argument("--suite", default="core", choices=["smoke", "core", "extended", "all"])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / args.manifest
    manifest = load_manifest(manifest_path)
    pyscf_references = load_pyscf_references(repo_root / args.pyscf_refs if args.pyscf_refs else None)
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

    # A case may declare `requires_build_option`: a CMake BOOL that must be ON in
    # the tree under test. Opt-in features (PLANCK_CC_UCC) are OFF by default, so
    # such a case would otherwise FAIL in a default build for a configuration
    # reason rather than a correctness one. Skipped, and reported separately so a
    # skip can never be mistaken for a pass.
    build_options = cmake_build_options(repo_root, args.build_dir)
    skipped = [(case, opt) for case in chosen
               if (opt := missing_build_option(case, build_options)) is not None]
    skipped_ids = {case["id"] for case, _ in skipped}
    chosen = [case for case in chosen if case["id"] not in skipped_ids]

    if not chosen and not skipped:
        print("no cases selected", file=sys.stderr)
        return 2

    print(f"Running {len(chosen)} regression case(s) from {manifest_path}")
    for case, option in skipped:
        print(f"[SKIP] {case['id']} (needs -D{option}=ON)")
    failures = 0
    total_start = time.perf_counter()
    results: list[CaseResult] = []

    for case in chosen:
        result = run_case(case, repo_root, args.build_dir, executable, pyscf_references)
        results.append(result)

    apply_cross_case_checks(results, chosen)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id} ({result.duration_s:.2f}s)")
        for line in result.details:
            print(f"    {line}")
        if not result.passed:
            failures += 1

    total_duration = time.perf_counter() - total_start
    skip_note = f", {len(skipped)} skipped" if skipped else ""
    print(
        f"Completed {len(chosen)} case(s) in {total_duration:.2f}s: "
        f"{len(chosen) - failures} passed, {failures} failed{skip_note}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
