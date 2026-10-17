#!/usr/bin/env python3
"""Compare normal dense and opt-in Hessian-free DH gradients; never starts a build.

Defaults to water and nonsymmetric C1 H2O2. --fd additionally compares BOTH
gradients against the SAME central total-energy differences on every coordinate.
The probe itself evaluates dense and swapped contracts at one identical SCF state.
All logs, inputs and full-precision JSON results are retained in a unique directory.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from dh_cartesian_fd_audit import validate_result
from dh_channel_fd_audit import ANGSTROM_TO_BOHR, parse_geometry, render_gradient_input


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/inputs/exploratory/dh_gradient"


def parse_ledger(text):
    lines = iter(text.splitlines())
    if next(lines, "") != "DH_HESSIAN_SWAP_PROBE 1":
        raise ValueError("Missing Hessian swap probe marker (binary may not be rebuilt)")
    matrices, scalars, iterations = {}, {}, {}
    status = None
    for line in lines:
        if line.startswith("MATRIX "):
            _, name, rows, cols = line.split()
            rows, cols = int(rows), int(cols)
            if name in matrices or rows <= 0 or cols <= 0:
                raise ValueError("Invalid or duplicate matrix: " + name)
            values = []
            for _ in range(rows):
                row = [float(v) for v in next(lines, "").split()]
                if len(row) != cols or not all(math.isfinite(v) for v in row):
                    raise ValueError("Invalid/truncated matrix: " + name)
                values.append(row)
            matrices[name] = values
        elif line.startswith("SCALAR "):
            _, name, value = line.split()
            if name in scalars or not math.isfinite(float(value)):
                raise ValueError("Invalid or duplicate scalar: " + name)
            scalars[name] = float(value)
        elif line.startswith("ITERATION "):
            _, name, iteration, residual = line.split()
            if not math.isfinite(float(residual)):
                raise ValueError("Nonfinite iteration residual")
            iterations.setdefault(name, []).append([int(iteration), float(residual)])
        elif line.startswith("STATUS "):
            if status is not None:
                raise ValueError("Duplicate probe status")
            status = line.split()[1]
    if status not in ("PASS", "DIFFERENCE"):
        raise ValueError("Incomplete or failed probe; inspect raw ledger")
    for name in ("H.total", "H.orbital", "H.J", "H.K", "H.XC", "gradient.total"):
        for suffix in ("dense", "candidate", "delta"):
            if name + "." + suffix not in matrices:
                raise ValueError("Missing comparison: " + name + "." + suffix)
    for name in ("gmres_dense", "gmres_eq27", "gmres_shared"):
        if scalars.get(name + ".converged") != 1 or name not in iterations:
            raise ValueError("Missing converged solver: " + name)
    return dict(status=status, matrices=matrices, scalars=scalars, iterations=iterations)


def matrix_error(left, right):
    if len(left) != len(right) or not left or any(len(a) != len(b) or not a for a, b in zip(left, right)):
        raise ValueError("Matrix shape mismatch")
    return max(abs(a - b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", default=[
        FIXTURES / "water_b2plyp_gradient_fd.hfinp", FIXTURES / "h2o2_c1_b2plyp_gradient_fd.hfinp"])
    parser.add_argument("--executable", type=Path, default=ROOT / "build/planck-dft")
    parser.add_argument("--fd", action="store_true", help="Run all-coordinate total-energy FD at both steps")
    parser.add_argument("--steps", type=float, nargs="+", default=[1e-4, 2e-4])
    parser.add_argument("--fd-tolerance", type=float, default=5e-8)
    args = parser.parse_args()
    if not args.steps or len(set(args.steps)) != len(args.steps) or any(not math.isfinite(h) or h <= 0 for h in args.steps):
        parser.error("steps must be distinct positive finite values")
    if not math.isfinite(args.fd_tolerance) or args.fd_tolerance <= 0:
        parser.error("FD tolerance must be positive and finite")
    artifacts = Path(tempfile.mkdtemp(prefix="dh-hessian-swap-"))
    print("Artifacts: " + str(artifacts), flush=True)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("PLANCK_DFT_DH_")}
    reports = []
    for number, fixture in enumerate(args.inputs):
        directory = artifacts / f"{number}-{fixture.stem}"
        directory.mkdir()
        template = fixture.read_text()
        symbols, coords, charge, mult = parse_geometry(template)

        def run(tag, positions, *, gradient=False, probe=False):
            text = render_gradient_input(template, symbols, positions, charge, mult)
            if not gradient:
                text = re.sub(r"(calculation\s+)gradient", r"\g<1>energy", text, count=1)
            inp = directory / (tag + ".hfinp")
            output = directory / (tag + ".json")
            inp.write_text(text)
            env = dict(base_env)
            if probe:
                env["PLANCK_DFT_DH_HESSIAN_PROBE_LOG"] = str(directory / "probe.ledger")
            proc = subprocess.run([str(args.executable.resolve()), str(inp), "--json", str(output)],
                                  env=env, text=True, capture_output=True)
            log = proc.stdout + proc.stderr
            (directory / (tag + ".log")).write_text(log)
            if proc.returncode or not re.search(r"Converged\s*:\s*true", log):
                raise RuntimeError(f"Calculation failed: {directory / (tag + '.log')}")
            if probe and "Returning shared-action Z gradient" not in log:
                raise RuntimeError("Probe was not executed; rebuild planck-dft")
            return validate_result(json.loads(output.read_text()), len(symbols), gradient), parse_geometry(text)[1]

        try:
            dense, center = run("dense", coords, gradient=True)
            candidate, candidate_center = run("shared", center, gradient=True, probe=True)
            if matrix_error(center, candidate_center) != 0:
                raise ValueError("Center geometry changed between runs")
            ledger = parse_ledger((directory / "probe.ledger").read_text())
            # The fixture runner uses the existing no-symmetry Cartesian fixtures;
            # the ledger is in the standard frame. Do not mix those frames here.
            gradient_difference = matrix_error(dense["gradient"], candidate["gradient"])
            energy_difference = abs(dense["total_energy"] - candidate["total_energy"])
            rows = []
            if args.fd:
                for step in args.steps:
                    for atom in range(len(symbols)):
                        for axis in range(3):
                            energies, actual = [], []
                            for label, sign in (("plus", 1), ("minus", -1)):
                                displaced = [row[:] for row in center]
                                displaced[atom][axis] += sign * step / ANGSTROM_TO_BOHR
                                data, rendered = run(f"h{step:.8g}-a{atom + 1}-q{axis}-{label}", displaced)
                                energies.append(data["total_energy"])
                                actual.append(rendered[atom][axis])
                            separation = (actual[0] - actual[1]) * ANGSTROM_TO_BOHR
                            if separation <= 0:
                                raise ValueError("Rendered displacement vanished")
                            fd = (energies[0] - energies[1]) / separation
                            row = dict(atom=atom + 1, axis=axis, step=step, fd=fd,
                                       dense_error=dense["gradient"][atom][axis] - fd,
                                       shared_error=candidate["gradient"][atom][axis] - fd)
                            rows.append(row)
                            print("FD " + json.dumps(row), flush=True)
            passed = ledger["status"] == "PASS" and gradient_difference <= 1e-9 and energy_difference <= 1e-12
            passed = passed and all(abs(row[k]) <= args.fd_tolerance for row in rows for k in ("dense_error", "shared_error"))
            report = dict(fixture=str(fixture), passed=passed, gradient_difference=gradient_difference,
                          energy_difference=energy_difference, ledger=ledger, fd=rows,
                          fd_requested=args.fd, fd_tolerance=args.fd_tolerance)
            (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"{'PASS' if passed else 'FAIL'} {fixture.name}: gradient delta={gradient_difference:.6e}, energy delta={energy_difference:.6e}", flush=True)
            for key, value in ledger["scalars"].items():
                if key.startswith("audit.") or key.startswith("H."):
                    print(f"  {key}: {value:.6e}", flush=True)
            reports.append(dict(fixture=str(fixture), passed=passed, report=str(directory / "report.json")))
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            print(f"FAIL {fixture.name}: {error}", flush=True)
            reports.append(dict(fixture=str(fixture), passed=False, error=str(error)))
    (artifacts / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    return 0 if all(item["passed"] for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
