#!/usr/bin/env python3
"""All-coordinate production DH total-gradient check through the normal JSON API.

No debug flags, internal scalar ledgers, or expected-failure guards are used.
Each endpoint is a fresh converged total-energy calculation. All inputs,
stdout/stderr logs, and JSON results are kept in a unique artifact directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from dh_channel_fd_audit import ANGSTROM_TO_BOHR, parse_geometry, render_gradient_input


def validate_result(data, natom, require_gradient):
    if data.get("natoms") != natom or not math.isfinite(data["total_energy"]):
        raise ValueError("Invalid production energy or atom count")
    if require_gradient:
        gradient = data.get("gradient")
        if not isinstance(gradient, list) or len(gradient) != natom or any(
            not isinstance(row, list) or len(row) != 3 or
            any(not math.isfinite(value) for value in row) for row in gradient
        ):
            raise ValueError("Missing or invalid production gradient")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("build/planck-dft"))
    parser.add_argument("--steps", nargs="+", type=float, default=[1e-4, 2e-4])
    parser.add_argument("--tolerance", type=float, default=5e-8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--full", action="store_true",
                        help="Compatibility alias; the production total gradient is always checked")
    args = parser.parse_args()
    if args.jobs < 1 or len(set(args.steps)) != len(args.steps) or any(
        not math.isfinite(h) or h <= 0 for h in args.steps
    ) or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("jobs, distinct finite steps, and tolerance must be positive")

    template = args.input.read_text()
    symbols, coordinates, charge, mult = parse_geometry(template)
    root = Path(tempfile.mkdtemp(prefix="dh-production-cartesian-fd-"))
    print(f"Artifacts: {root}", flush=True)
    print("Scope: production total energy versus production analytic gradient; no DH debug flags", flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PLANCK_DFT_DH_")}

    def run(tag, positions, gradient=False):
        rendered = render_gradient_input(template, symbols, positions, charge, mult)
        if not gradient:
            rendered = re.sub(r"(calculation\s+)gradient", r"\g<1>energy", rendered, count=1)
        inp = root / (tag + ".hfinp")
        output_json = root / (tag + ".json")
        inp.write_text(rendered)
        proc = subprocess.run([str(args.executable.resolve()), str(inp), "--json", str(output_json)],
                              env=env, text=True, capture_output=True)
        output = proc.stdout + proc.stderr
        (root / (tag + ".log")).write_text(output)
        if proc.returncode != 0 or not re.search(r"Converged\s*:\s*true", output):
            raise RuntimeError(f"Production calculation failed or did not converge: {root / (tag + '.log')}")
        data = validate_result(json.loads(output_json.read_text()), len(symbols), gradient)
        return data, parse_geometry(rendered)[1]

    center, center_coordinates = run("center", coordinates, gradient=True)
    analytic = center["gradient"]
    print("Analytic total gradient (Ha/Bohr): " + json.dumps(analytic), flush=True)
    print("atom axis step analytic FD FD-minus-analytic", flush=True)

    def check(job):
        atom, axis, step = job
        energies, positions = [], []
        for label, sign in (("plus", 1), ("minus", -1)):
            displaced = [row[:] for row in center_coordinates]
            displaced[atom][axis] += sign * step / ANGSTROM_TO_BOHR
            tag = f"a{atom + 1}-{'xyz'[axis]}-h{step:.8g}-{label}"
            data, actual = run(tag, displaced)
            energies.append(data["total_energy"])
            positions.append(actual[atom][axis])
        separation = (positions[0] - positions[1]) * ANGSTROM_TO_BOHR
        if separation <= 0:
            raise ValueError("Rendered FD displacement vanished")
        fd = (energies[0] - energies[1]) / separation
        row = dict(atom=atom + 1, symbol=symbols[atom], axis="xyz"[axis], step=step,
                   actual_half_step=separation / 2, analytic=analytic[atom][axis],
                   fd=fd, fd_minus_analytic=fd - analytic[atom][axis])
        print(f"{symbols[atom]}{atom + 1} {row['axis']} {step:.1e} "
              f"{row['analytic']:+.12e} {fd:+.12e} {row['fd_minus_analytic']:+.6e}", flush=True)
        return row

    jobs = [(atom, axis, step) for step in args.steps
            for atom in range(len(symbols)) for axis in range(3)]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(check, jobs))
    summaries = []
    for step in args.steps:
        selected = [row for row in rows if row["step"] == step]
        errors = [row["fd_minus_analytic"] for row in selected]
        summary = dict(step=step, max_abs=max(map(abs, errors)),
                       rms=math.sqrt(sum(e * e for e in errors) / len(errors)),
                       fd_translation_sum=[sum(row["fd"] for row in selected if row["axis"] == q)
                                           for q in "xyz"])
        summaries.append(summary)
        print("SUMMARY " + json.dumps(summary), flush=True)
    translation = [sum(row[q] for row in analytic) for q in range(3)]
    passed = all(math.isfinite(row["fd_minus_analytic"]) and
                 abs(row["fd_minus_analytic"]) < args.tolerance for row in rows)
    report = dict(scope="production total gradient, no debug flags", input=str(args.input.resolve()),
                  center_energy=center["total_energy"], analytic=analytic,
                  translation_sum=translation, tolerance=args.tolerance,
                  results=rows, summaries=summaries, passed=passed)
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Analytic translation sum: " + json.dumps(translation), flush=True)
    print(("PASS" if passed else "FAIL") + f" at {args.tolerance:.1e} Ha/Bohr; "
          + str(root / "results.json"), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
