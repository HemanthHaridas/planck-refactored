#!/usr/bin/env python3
"""Validate RKS DH opt/freq callbacks against fresh standalone DH calculations.

Never builds. Uses normal JSON results, no DH diagnostic flags or external QC
reference. Retains every input, output and the full independent Hessian matrix.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from dh_channel_fd_audit import ANGSTROM_TO_BOHR, parse_geometry
from dh_cartesian_fd_audit import validate_result

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "tests/inputs/exploratory/dh_gradient/water_b2plyp_gradient_fd.hfinp"


def render(template, symbols, xyz_bohr, charge, mult, workflow, opt_coords):
    text = re.sub(r"(calculation\s+)\S+", rf"\g<1>{workflow}", template, count=1)
    text = re.sub(r"(coord_units\s+)\S+", r"\g<1>bohr", text, count=1)
    text = re.sub(r"(use_symm\s+)\S+", r"\g<1>.false.", text, count=1)
    text = re.sub(r"^\s*opt_coords\s+\S+\s*$", "", text, flags=re.M)
    text = text.replace("%end_geom", f"    opt_coords {opt_coords}\n%end_geom")
    rows = [str(len(symbols)), f"{charge} {mult}"]
    rows += [symbol + " " + " ".join(f"{v:.17g}" for v in row)
             for symbol, row in zip(symbols, xyz_bohr)]
    return re.sub(r"%begin_coords\s*\n.*?\n%end_coords",
                  "%begin_coords\n" + "\n".join(rows) + "\n%end_coords", text, flags=re.S)


def matrix_difference(a, b):
    if not a or len(a) != len(b) or any(len(x) != len(y) or not x for x, y in zip(a, b)):
        raise ValueError("Matrix shape mismatch")
    values = [abs(x-y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite matrix comparison")
    return max(values)


def symmetric_hessian(plus, minus, step):
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Invalid Hessian step")
    n = len(plus)
    if n == 0 or len(minus) != n or any(len(col) != n for col in plus + minus):
        raise ValueError("Invalid gradient columns")
    raw = [[(plus[j][i] - minus[j][i]) / (2*step) for j in range(n)] for i in range(n)]
    return [[0.5*(raw[i][j]+raw[j][i]) for j in range(n)] for i in range(n)]


def require_close(error, tolerance, label):
    if not math.isfinite(error) or error > tolerance:
        raise ValueError(f"{label}: {error:.6e} > {tolerance:.6e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "build/planck-dft")
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--workflows", nargs="+", choices=("freq", "geomopt", "geomoptfreq"),
                        default=["freq", "geomopt", "geomoptfreq"])
    parser.add_argument("--opt-coords", choices=("cartesian", "internal"), default="cartesian")
    args = parser.parse_args()
    template = args.input.read_text()
    if not re.search(r"coord_type\s+cartesian", template):
        parser.error("The oracle requires Cartesian input coordinates")
    symbols, xyz, charge, mult = parse_geometry(template)
    if re.search(r"coord_units\s+angstrom", template):
        xyz = [[v*ANGSTROM_TO_BOHR for v in row] for row in xyz]
    elif not re.search(r"coord_units\s+bohr", template):
        parser.error("Explicit angstrom or bohr units required")
    artifacts = Path(tempfile.mkdtemp(prefix="dh-workflows-"))
    print(f"Artifacts: {artifacts}", flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PLANCK_DFT_DH_")}
    report = dict(passed=False, workflows=[])

    def run(tag, workflow, positions):
        inp, log, js = [artifacts / (tag + suffix) for suffix in (".hfinp", ".log", ".json")]
        inp.write_text(render(template, symbols, positions, charge, mult, workflow, args.opt_coords))
        proc = subprocess.run([str(args.executable.resolve()), str(inp), "--json", str(js)],
                              env=env, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        log.write_text(output)
        if proc.returncode != 0 or not re.search(r"Converged\s*:\s*true", output):
            raise ValueError(f"Calculation failed or not converged: {log}")
        if workflow in ("geomopt", "geomoptfreq") and not re.search(
                r"Geometry Optimization\s*:\s*Converged in", output):
            raise ValueError(f"Optimization did not converge: {log}")
        if workflow in ("freq", "geomoptfreq") and "via analytic double-hybrid gradients" not in output:
            raise ValueError(f"Missing full-DH frequency callback marker: {log}")
        data = validate_result(json.loads(js.read_text()), len(symbols), workflow != "energy")
        if not data.get("has_correlation"):
            raise ValueError(f"Missing DH energy correction: {log}")
        return data

    try:
        for workflow in args.workflows:
            data = run(workflow, workflow, xyz)
            ref_xyz = data["coordinates_bohr"]
            if workflow == "freq":
                require_close(matrix_difference(ref_xyz, xyz), 1e-12, "Frequency geometry restoration")
            fresh = run(workflow + "-reference", "gradient", ref_xyz)
            energy_error = abs(data["total_energy"] - fresh["total_energy"])
            gradient_error = matrix_difference(data["gradient"], fresh["gradient"])
            require_close(energy_error, 1e-8, "Reference DH energy restoration")
            require_close(gradient_error, 1e-7, "Reference DH gradient restoration")
            row = dict(workflow=workflow, energy_error=energy_error, gradient_error=gradient_error)
            if workflow != "freq":
                require_close(max(abs(v) for r in fresh["gradient"] for v in r),
                              3.01e-4, "Optimized full-DH gradient")
                # Fresh total-energy FDs guard against optimizing KS energy or
                # pairing the correlated energy with an uncorrected KS gradient.
                errors = []
                for j in range(3*len(symbols)):
                    ends = []
                    for sign in (1, -1):
                        pos = [r[:] for r in ref_xyz]
                        pos[j//3][j%3] += sign*1e-4
                        ends.append(run(f"{workflow}-efd-{j}-{sign}", "energy", pos)["total_energy"])
                    errors.append(abs((ends[0]-ends[1])/2e-4 - fresh["gradient"][j//3][j%3]))
                row["energy_fd_gradient_max_error"] = max(errors)
                require_close(max(errors), 5e-8, "Optimized energy FD versus gradient")
            if workflow in ("freq", "geomoptfreq"):
                h = data["hessian_step_bohr"]
                plus, minus = [], []
                for j in range(3*len(symbols)):
                    for sign, columns in ((1, plus), (-1, minus)):
                        pos = [r[:] for r in ref_xyz]
                        pos[j//3][j%3] += sign*h
                        endpoint = run(f"{workflow}-hfd-{j}-{sign}", "gradient", pos)
                        columns.append([v for r in endpoint["gradient"] for v in r])
                oracle = symmetric_hessian(plus, minus, h)
                (artifacts / f"{workflow}-hessian-oracle.json").write_text(json.dumps(oracle, indent=2))
                row["hessian_max_error"] = matrix_difference(data["hessian"], oracle)
                require_close(row["hessian_max_error"], 2e-6, "Full DH Hessian callback versus independent gradients")
                frequencies = data["frequencies_cm1"]
                if not frequencies or not all(math.isfinite(v) for v in frequencies + [data["zpe_hartree"]]):
                    raise ValueError("Missing/nonfinite vibrational results")
                row["frequencies_cm1"] = frequencies
            report["workflows"].append(row)
            print(json.dumps(row), flush=True)
        # U0 scope must not accidentally expand to UKS derivatives.
        uks = (ROOT / "tests/inputs/exploratory/dh_gradient/uks_u0/water_cation_asymmetric_b2plyp_sto3g.hfinp").read_text()
        usym, upos, uchg, umult = parse_geometry(uks)
        upos = [[v*ANGSTROM_TO_BOHR for v in r] for r in upos]
        for workflow in ("geomopt", "freq", "geomoptfreq"):
            inp = artifacts / f"uks-{workflow}.hfinp"
            inp.write_text(render(uks, usym, upos, uchg, umult, workflow, args.opt_coords))
            proc = subprocess.run([str(args.executable.resolve()), str(inp)], env=env,
                                  capture_output=True, text=True)
            output = proc.stdout + proc.stderr
            (artifacts / f"uks-{workflow}.log").write_text(output)
            if proc.returncode == 0 or "supports only single-point energies for range-separated and double-hybrid functionals" not in output:
                raise ValueError(f"UKS {workflow} scope rejection failed")
        report["uks_scope_rejected"] = True
        report["passed"] = True
    except (ValueError, OSError, KeyError, TypeError) as error:
        report["error"] = str(error)
    (artifacts / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    print("PASS" if report["passed"] else "FAIL", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
