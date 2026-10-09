#!/usr/bin/env python3
"""
Finite-difference verification of the DFT analytic nuclear gradient.

Runs `planck-dft` on a base input to get the analytic gradient, then runs
single-point energy evaluations at +/- delta displacements per Cartesian
coordinate to build a central-difference reference gradient. Compares the
two and exits non-zero if the largest component-wise deviation exceeds
the tolerance.

This regression test covers the full KS-DFT gradient path in
`src/dft/dft_gradient.cpp`, including the XC derivative matrix and the
moving-grid (Becke partition + point-translation) response.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

ANGSTROM_TO_BOHR = 1.8897261254535


ATOM_LINE_RE = re.compile(
    r"Atom\s+(\d+)\s*:\s*([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)"
)
DFT_ENERGY_RE = re.compile(
    r"DFT Energy\s*:\s*([-+0-9Ee\.]+)\s+Eh"
)


def parse_coords_block(text: str) -> tuple[list[str], list[list[float]], int, int, str]:
    """Return (symbols, coords_angstrom, charge, multiplicity, raw_block)."""
    m = re.search(r"%begin_coords\s*\n(.*?)\n%end_coords", text, re.DOTALL)
    if not m:
        raise SystemExit("could not find %begin_coords block in input")
    block = m.group(1)
    lines = [ln for ln in block.splitlines() if ln.strip()]
    natom = int(lines[0].split()[0])
    charge_str, mult_str = lines[1].split()[:2]
    charge = int(charge_str)
    multiplicity = int(mult_str)
    symbols: list[str] = []
    coords: list[list[float]] = []
    for ln in lines[2 : 2 + natom]:
        parts = ln.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, coords, charge, multiplicity, block


def render_input(template: str, symbols: list[str], coords: list[list[float]],
                 charge: int, multiplicity: int, calculation: str) -> str:
    """Replace calculation type + coords in the template."""
    text = re.sub(
        r"(calculation\s+)\S+",
        rf"\g<1>{calculation}",
        template,
        count=1,
    )
    coord_lines = [f"{len(symbols)}", f"{charge}   {multiplicity}"]
    for sym, (x, y, z) in zip(symbols, coords):
        coord_lines.append(f"{sym:<5s}{x:14.10f}{y:14.10f}{z:14.10f}")
    new_block = "\n".join(coord_lines)
    text = re.sub(
        r"%begin_coords\s*\n.*?\n%end_coords",
        f"%begin_coords\n{new_block}\n%end_coords",
        text,
        count=1,
        flags=re.DOTALL,
    )
    return text


def run_planck_dft(executable: Path, input_path: Path) -> str:
    proc = subprocess.run(
        [str(executable), str(input_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"planck-dft failed (exit {proc.returncode}) on {input_path}"
        )
    return proc.stdout


def parse_dft_energy(output: str) -> float:
    matches = DFT_ENERGY_RE.findall(output)
    if not matches:
        raise SystemExit("no 'DFT Energy :' line found in planck-dft output")
    return float(matches[-1])


def parse_analytic_gradient(output: str, natom: int) -> list[list[float]]:
    grad: dict[int, list[float]] = {}
    for line in output.splitlines():
        m = ATOM_LINE_RE.search(line)
        if m:
            idx = int(m.group(1))
            grad[idx] = [float(m.group(2)), float(m.group(3)), float(m.group(4))]
    if len(grad) < natom:
        raise SystemExit(
            f"only parsed {len(grad)}/{natom} gradient lines from planck-dft"
        )
    return [grad[i + 1] for i in range(natom)]


def central_difference_gradient(
    executable: Path,
    template: str,
    symbols: list[str],
    coords: list[list[float]],
    charge: int,
    multiplicity: int,
    delta_bohr: float,
    workdir: Path,
    tag: str = "fd",
) -> list[list[float]]:
    """
    Build a natom x 3 numerical gradient (Ha/Bohr) by central differences
    in Cartesian coordinates.
    """
    # Convert displacement from Bohr (gradient units) to Angstrom (input units).
    delta_ang = delta_bohr / ANGSTROM_TO_BOHR

    natom = len(symbols)
    grad = [[0.0, 0.0, 0.0] for _ in range(natom)]
    for ia in range(natom):
        for q in range(3):
            results: list[float] = []
            for sign in (+1.0, -1.0):
                shifted = [row[:] for row in coords]
                shifted[ia][q] += sign * delta_ang
                inp = render_input(
                    template, symbols, shifted, charge, multiplicity,
                    calculation="energy",
                )
                inp_path = workdir / f"{tag}_{ia}_{q}_{'+' if sign > 0 else '-'}.hfinp"
                inp_path.write_text(inp)
                out = run_planck_dft(executable, inp_path)
                results.append(parse_dft_energy(out))
            e_plus, e_minus = results
            grad[ia][q] = (e_plus - e_minus) / (2.0 * delta_bohr)
    return grad


def max_abs_diff(a: list[list[float]], b: list[list[float]]) -> tuple[float, tuple[int, int]]:
    worst = 0.0
    where = (0, 0)
    for i, (ra, rb) in enumerate(zip(a, b)):
        for j, (xa, xb) in enumerate(zip(ra, rb)):
            d = abs(xa - xb)
            if d > worst:
                worst = d
                where = (i, j)
    return worst, where


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "build",
        help="directory containing planck-dft binary (default: ./build)",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent
        / "inputs" / "regression" / "dft" / "h2_dft_pbe_gradient.hfinp",
        help="base .hfinp; calculation field will be replaced as needed",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0e-3,
        help="finite-difference step in Bohr (default 1e-3)",
    )
    parser.add_argument(
        "--outer-delta",
        type=float,
        help="second central-difference radius for fourth-order extrapolation",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2.0e-4,
        help="component-wise tolerance (Ha/Bohr) on |g_analytic - g_fd|",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="keep the working directory of FD inputs/outputs for inspection",
    )
    parser.add_argument("--grid-policy", help="override the immutable DFT grid policy")
    parser.add_argument("--radial-points", type=int, help="override the radial shell count")
    parser.add_argument(
        "--scf-tolerance",
        type=float,
        help="override both SCF energy and density tolerances for the FD stencil",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    executable = args.build_dir / "planck-dft"
    if not executable.exists():
        raise SystemExit(f"planck-dft binary not found at {executable}")
    if not args.input.exists():
        raise SystemExit(f"input file not found: {args.input}")

    template = args.input.read_text()
    if args.radial_points is not None:
        if args.radial_points <= 0:
            parser.error("--radial-points must be positive")
        template = re.sub(r"^\s*grid_radial_points\s+[^\n]*\n", "", template, flags=re.MULTILINE)
        template = template.replace("%end_dft", f"    grid_radial_points {args.radial_points}\n%end_dft", 1)
    if args.grid_policy is not None:
        template = re.sub(r"^\s*grid_policy\s+[^\n]*\n", "", template, flags=re.MULTILINE)
        template = template.replace("%end_dft", f"    grid_policy {args.grid_policy}\n%end_dft", 1)
    if args.scf_tolerance is not None:
        if args.scf_tolerance <= 0.0:
            parser.error("--scf-tolerance must be positive")
        value = f"{args.scf_tolerance:.17g}"
        for key in ("tol_energy", "tol_density"):
            pattern = rf"^(\s*{key})\s+[^\n]*$"
            template, count = re.subn(pattern, rf"\1 {value}", template,
                                      count=1, flags=re.MULTILINE)
            if count != 1:
                parser.error(f"input has no {key} entry to override")
    if args.outer_delta is not None and (args.outer_delta <= args.delta or not args.outer_delta > 0):
        parser.error("--outer-delta must be positive and larger than --delta")
    symbols, coords, charge, multiplicity, _ = parse_coords_block(template)
    natom = len(symbols)

    workdir_obj = tempfile.TemporaryDirectory(prefix="planck-dft-fd-")
    workdir = Path(workdir_obj.name)
    if args.keep_tmp:
        # Detach so it sticks around.
        workdir_obj._finalizer.detach()  # type: ignore[attr-defined]

    # Analytic gradient.
    grad_inp = render_input(
        template, symbols, coords, charge, multiplicity, calculation="gradient"
    )
    grad_path = workdir / "analytic.hfinp"
    grad_path.write_text(grad_inp)
    analytic_out = run_planck_dft(executable, grad_path)
    g_analytic = parse_analytic_gradient(analytic_out, natom)

    # Central difference, optionally extrapolated from two radii to cancel
    # the O(h^2) truncation term without changing the analytic calculation.
    g_inner = central_difference_gradient(
        executable, template, symbols, coords, charge, multiplicity,
        args.delta, workdir, "inner",
    )
    g_fd = g_inner
    if args.outer_delta is not None:
        g_outer = central_difference_gradient(
            executable, template, symbols, coords, charge, multiplicity,
            args.outer_delta, workdir, "outer")
        ratio2 = (args.outer_delta / args.delta) ** 2
        g_fd = [[(ratio2 * inner - outer) / (ratio2 - 1.0)
                 for inner, outer in zip(row_inner, row_outer)]
                for row_inner, row_outer in zip(g_inner, g_outer)]

    # Report.
    print("Analytic gradient (Ha/Bohr):")
    for i, row in enumerate(g_analytic):
        print(f"  Atom {i + 1}: {row[0]:14.8f}  {row[1]:14.8f}  {row[2]:14.8f}")
    print(("Fourth-order extrapolated" if args.outer_delta is not None else "Finite-difference") + " gradient (Ha/Bohr):")
    for i, row in enumerate(g_fd):
        print(f"  Atom {i + 1}: {row[0]:14.8f}  {row[1]:14.8f}  {row[2]:14.8f}")

    worst, (ia, q) = max_abs_diff(g_analytic, g_fd)
    report = dict(input=str(args.input), grid_policy=args.grid_policy, radial_points=args.radial_points,
                  scf_tolerance=args.scf_tolerance,
                  inner_delta_bohr=args.delta, outer_delta_bohr=args.outer_delta,
                  stencil="fourth-order" if args.outer_delta is not None else "second-order",
                  analytic_gradient=g_analytic, finite_difference_gradient=g_fd,
                  max_error=worst, atom=ia + 1, axis="xyz"[q], tolerance=args.atol,
                  passed=worst <= args.atol)
    if args.keep_tmp:
        (workdir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"max |g_analytic - g_fd| = {worst:.3e} Ha/Bohr "
        f"(atom {ia + 1}, axis {'xyz'[q]})"
    )
    print(f"tolerance               = {args.atol:.3e} Ha/Bohr")

    if args.keep_tmp:
        print(f"work directory kept at: {workdir}")

    if not report["passed"]:
        print("FAIL: analytic DFT gradient deviates from finite difference")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
