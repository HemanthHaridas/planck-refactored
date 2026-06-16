#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PYSCF_DIR = REPO_ROOT / "tests" / "pyscf"
LOCAL_PYSCF_PYTHON = REPO_ROOT / "tests" / "pyscf" / ".venv" / "bin" / "python"

if str(PYSCF_DIR) not in sys.path:
    sys.path.insert(0, str(PYSCF_DIR))


CASE_INPUTS: dict[str, Path] = {
    "h2_rmp2_gradient": REPO_ROOT / "tests" / "inputs" / "regression" / "post_hf" / "h2_rmp2_gradient.hfinp",
    "water_rmp2_gradient_sto3g": REPO_ROOT
    / "tests"
    / "inputs"
    / "regression"
    / "post_hf"
    / "water_rmp2_gradient_sto3g.hfinp",
    "water_radical_cation_uhf_ump2_gradient_sto3g": REPO_ROOT
    / "tests"
    / "inputs"
    / "regression"
    / "open_shell"
    / "water_radical_cation_uhf_ump2_gradient_sto3g.hfinp",
    "water_triplet_uhf_ump2_gradient_sto3g": REPO_ROOT
    / "tests"
    / "inputs"
    / "regression"
    / "open_shell"
    / "water_triplet_uhf_ump2_gradient_sto3g.hfinp",
}


ATOM_LINE_RE = re.compile(
    r"Atom\s+(\d+)\s*:\s*([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)"
)
MP2_ENERGY_RE = re.compile(r"^\s*Total MP2 Energy\s+([-+0-9Ee\.]+)", re.MULTILINE)
COORD_LINE_RE = re.compile(
    r"^\[INF\]\s+\d+\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s*$"
)


def ensure_pyscf_runtime() -> None:
    try:
        import pyscf  # noqa: F401
    except ModuleNotFoundError:
        current = Path(sys.executable).resolve()
        if LOCAL_PYSCF_PYTHON.exists() and current != LOCAL_PYSCF_PYTHON.resolve():
            proc = subprocess.run(
                [str(LOCAL_PYSCF_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
                check=False,
            )
            raise SystemExit(proc.returncode)
        raise


ensure_pyscf_runtime()

from pyscf import mp, scf  # noqa: E402
from input_utils import build_molecule, parse_bool, parse_input_file  # noqa: E402


@dataclass
class BenchmarkResult:
    case_id: str
    method: str
    input_path: str
    natom: int
    planck_mp2_total_energy: float
    pyscf_mp2_total_energy: float
    abs_energy_diff: float
    max_abs_gradient_diff: float
    rms_gradient_diff: float
    planck_time_s: float
    pyscf_time_s: float
    planck_over_pyscf: float
    planck_gradient: list[list[float]]
    pyscf_gradient: list[list[float]]
    gradient_diff: list[list[float]]


def parse_mp2_frozen(value: str | None) -> int | list[int] | None:
    if value is None or not value.strip():
        return None
    entries = [int(token) for token in value.split()]
    if len(entries) == 1 and entries[0] >= 0:
        return entries[0]
    return entries


def parse_planck_gradient(output: str) -> list[list[float]]:
    rows: dict[int, list[float]] = {}
    for line in output.splitlines():
        match = ATOM_LINE_RE.search(line)
        if not match:
            continue
        atom = int(match.group(1))
        rows[atom] = [
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        ]
    if not rows:
        raise RuntimeError("No gradient rows were found in Planck output.")
    return [rows[index] for index in range(1, len(rows) + 1)]


def parse_planck_mp2_total_energy(output: str) -> float:
    matches = MP2_ENERGY_RE.findall(output)
    if not matches:
        raise RuntimeError("No 'Total MP2 Energy' line was found in Planck output.")
    return float(matches[-1])


def parse_planck_standard_coords(output: str, natom: int) -> list[list[float]] | None:
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if "Standard Coordinates" not in line:
            continue
        coords: list[list[float]] = []
        cursor = index + 1
        while cursor < len(lines) and len(coords) < natom:
            match = COORD_LINE_RE.match(lines[cursor])
            if match:
                coords.append(
                    [
                        float(match.group(1)),
                        float(match.group(2)),
                        float(match.group(3)),
                    ]
                )
            cursor += 1
        if len(coords) == natom:
            return coords
    return None


def run_planck_case(
    executable: Path,
    input_path: Path,
    natom: int,
) -> tuple[float, list[list[float]], float, list[list[float]] | None]:
    start = time.perf_counter()
    proc = subprocess.run(
        [str(executable), str(input_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"Planck run failed for {input_path} with exit {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return (
        parse_planck_mp2_total_energy(proc.stdout),
        parse_planck_gradient(proc.stdout),
        elapsed,
        parse_planck_standard_coords(proc.stdout, natom),
    )


def build_mean_field(spec: dict[str, Any]):
    scf_spec = spec["scf"]
    scf_type = scf_spec["scf_type"].strip().lower()
    mol = build_molecule(spec)

    if scf_type == "rhf":
        mf = scf.RHF(mol)
    elif scf_type == "uhf":
        mf = scf.UHF(mol)
    else:
        raise RuntimeError(f"Unsupported SCF type for MP2 benchmark: {scf_type}")

    mf.verbose = 0
    mf.max_cycle = int(scf_spec.get("max_cycles", "100"))
    mf.level_shift = float(scf_spec.get("level_shift", "0.0"))
    mf.conv_tol = float(scf_spec.get("conv_tol", "1e-10"))
    mf.init_guess = scf_spec.get("guess", "hcore").strip().lower()
    if not parse_bool(scf_spec.get("use_diis", ".true.")):
        mf.diis = None
    return mf


def run_pyscf_case(input_path: Path) -> tuple[str, float, list[list[float]], float]:
    spec = parse_input_file(input_path)
    scf_spec = spec["scf"]
    corr = scf_spec["correlation"].strip().lower()

    mf = build_mean_field(spec)
    frozen = parse_mp2_frozen(scf_spec.get("mp2_frozen"))

    start = time.perf_counter()
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for {input_path}")

    if corr == "rmp2":
        post = mp.MP2(mf, frozen=frozen)
        method = "RMP2"
    elif corr == "ump2":
        post = mp.UMP2(mf, frozen=frozen)
        method = "UMP2"
    else:
        raise RuntimeError(f"Unsupported correlation model for MP2 benchmark: {corr}")

    post.verbose = 0
    post.level_shift = float(scf_spec.get("mp2_level_shift", "0.0"))
    post.conv_tol = float(scf_spec.get("mp2_conv_tol", "1e-7"))
    post.conv_tol_normt = float(scf_spec.get("mp2_conv_tol_normt", "1e-5"))
    post.max_cycle = int(scf_spec.get("mp2_max_cycle", "50"))
    post.diis_space = int(scf_spec.get("mp2_diis_space", "6"))
    post.with_t2 = parse_bool(scf_spec.get("mp2_with_t2", ".true."))
    post.kernel()
    gradient = post.nuc_grad_method().kernel()
    elapsed = time.perf_counter() - start

    gradient_rows = [[float(value) for value in row] for row in gradient.tolist()]
    return method, float(post.e_tot), gradient_rows, elapsed


def extract_input_coords(spec: dict[str, Any]) -> list[list[float]]:
    return [[float(x), float(y), float(z)] for _, (x, y, z) in spec["coords"]["atoms"]]


def build_rotation(
    source_coords: list[list[float]],
    target_coords: list[list[float]],
) -> np.ndarray:
    source = np.asarray(source_coords, dtype=float)
    target = np.asarray(target_coords, dtype=float)

    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    covariance = source_centered.T @ target_centered
    u_mat, _, v_t = np.linalg.svd(covariance)
    rotation = v_t.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        v_t[-1, :] *= -1.0
        rotation = v_t.T @ u_mat.T
    return rotation


def rotate_gradient(rows: list[list[float]], rotation: np.ndarray) -> list[list[float]]:
    rotated: list[list[float]] = []
    for row in rows:
        rotated_row = rotation @ np.asarray(row, dtype=float)
        rotated.append([float(value) for value in rotated_row.tolist()])
    return rotated


def subtract_gradients(
    left: list[list[float]],
    right: list[list[float]],
) -> list[list[float]]:
    return [
        [lhs - rhs for lhs, rhs in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def max_abs_component(rows: list[list[float]]) -> float:
    return max(abs(value) for row in rows for value in row)


def rms_component(rows: list[list[float]]) -> float:
    values = [value for row in rows for value in row]
    return math.sqrt(sum(value * value for value in values) / len(values))


def benchmark_case(executable: Path, case_id: str, input_path: Path) -> BenchmarkResult:
    spec = parse_input_file(input_path)
    natom = int(spec["coords"]["natoms"])
    planck_energy, planck_gradient, planck_time, standard_coords = run_planck_case(
        executable,
        input_path,
        natom,
    )
    method, pyscf_energy, pyscf_gradient, pyscf_time = run_pyscf_case(input_path)
    if standard_coords is not None:
        rotation = build_rotation(extract_input_coords(spec), standard_coords)
        pyscf_gradient = rotate_gradient(pyscf_gradient, rotation)
    gradient_diff = subtract_gradients(planck_gradient, pyscf_gradient)

    return BenchmarkResult(
        case_id=case_id,
        method=method,
        input_path=str(input_path),
        natom=natom,
        planck_mp2_total_energy=planck_energy,
        pyscf_mp2_total_energy=pyscf_energy,
        abs_energy_diff=abs(planck_energy - pyscf_energy),
        max_abs_gradient_diff=max_abs_component(gradient_diff),
        rms_gradient_diff=rms_component(gradient_diff),
        planck_time_s=planck_time,
        pyscf_time_s=pyscf_time,
        planck_over_pyscf=planck_time / pyscf_time if pyscf_time > 0.0 else float("nan"),
        planck_gradient=planck_gradient,
        pyscf_gradient=pyscf_gradient,
        gradient_diff=gradient_diff,
    )


def print_summary(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'Case':<44} {'Method':<6} {'|dE| / Eh':>12} {'max|dg|':>12} "
        f"{'rms|dg|':>12} {'PySCF s':>10} {'Planck s':>10} {'P/PySCF':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for result in results:
        print(
            f"{result.case_id:<44} {result.method:<6} "
            f"{result.abs_energy_diff:>12.3e} {result.max_abs_gradient_diff:>12.3e} "
            f"{result.rms_gradient_diff:>12.3e} {result.pyscf_time_s:>10.3f} "
            f"{result.planck_time_s:>10.3f} {result.planck_over_pyscf:>10.3f}"
        )
    print(sep)


def print_details(results: list[BenchmarkResult]) -> None:
    for result in results:
        print()
        print(f"[{result.case_id}] {result.method}")
        for atom, (planck_row, pyscf_row, diff_row) in enumerate(
            zip(result.planck_gradient, result.pyscf_gradient, result.gradient_diff),
            start=1,
        ):
            print(
                f"  Atom {atom:>2}  "
                f"Planck [{planck_row[0]: .8f} {planck_row[1]: .8f} {planck_row[2]: .8f}]  "
                f"PySCF [{pyscf_row[0]: .8f} {pyscf_row[1]: .8f} {pyscf_row[2]: .8f}]  "
                f"Diff [{diff_row[0]: .3e} {diff_row[1]: .3e} {diff_row[2]: .3e}]"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Planck RMP2/UMP2 gradients against local PySCF references."
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="named case to run; may be repeated",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="additional .hfinp input path to benchmark",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=REPO_ROOT / "build",
        help="directory containing hartree-fock",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="print per-atom gradient rows for each benchmark case",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="optional path to write machine-readable benchmark results",
    )
    parser.add_argument(
        "--max-abs-grad-tol",
        type=float,
        help="optional failure threshold on max absolute gradient component error",
    )
    args = parser.parse_args()

    executable = args.build_dir / "hartree-fock"
    if not executable.exists():
        raise SystemExit(f"hartree-fock binary not found at {executable}")

    selected: list[tuple[str, Path]] = []
    chosen_cases = args.case or list(CASE_INPUTS)
    unknown = sorted(set(chosen_cases) - set(CASE_INPUTS))
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")

    for case_id in chosen_cases:
        selected.append((case_id, CASE_INPUTS[case_id]))
    for raw_input in args.input:
        path = Path(raw_input).resolve()
        selected.append((path.stem, path))

    results = [benchmark_case(executable, case_id, input_path) for case_id, input_path in selected]
    print_summary(results)
    if args.details:
        print_details(results)

    if args.json:
        payload = {"results": [asdict(result) for result in results]}
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.max_abs_grad_tol is not None:
        worst = max(result.max_abs_gradient_diff for result in results)
        if worst > args.max_abs_grad_tol:
            print(
                f"FAIL: worst max|dg| {worst:.3e} exceeds tolerance {args.max_abs_grad_tol:.3e}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
