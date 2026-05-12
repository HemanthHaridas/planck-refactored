#!/usr/bin/env python3
"""
Finite-difference verification of the analytic RMP2 nuclear gradient.

Runs `hartree-fock` on a base RMP2 gradient input to get the analytic
gradient, then evaluates central-difference MP2 energies at +/- displaced
Cartesian geometries. The test fails if the largest component-wise deviation
exceeds the requested tolerance.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

ANGSTROM_TO_BOHR = 1.8897261254535

ATOM_LINE_RE = re.compile(
    r"Atom\s+(\d+)\s*:\s*([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)"
)
MP2_ENERGY_RE = re.compile(
    r"^\s*Total MP2 Energy\s+([-+0-9Ee\.]+)",
    re.MULTILINE,
)


def parse_coords_block(text: str) -> tuple[list[str], list[list[float]], int, int]:
    match = re.search(r"%begin_coords\s*\n(.*?)\n%end_coords", text, re.DOTALL)
    if not match:
        raise SystemExit("could not find %begin_coords block in input")

    lines = [line for line in match.group(1).splitlines() if line.strip()]
    natom = int(lines[0].split()[0])
    charge, multiplicity = (int(value) for value in lines[1].split()[:2])

    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + natom]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return symbols, coords, charge, multiplicity


def render_input(
    template: str,
    symbols: list[str],
    coords: list[list[float]],
    charge: int,
    multiplicity: int,
    calculation: str,
) -> str:
    text = re.sub(
        r"(calculation\s+)\S+",
        rf"\g<1>{calculation}",
        template,
        count=1,
    )

    coord_lines = [f"{len(symbols)}", f"{charge}   {multiplicity}"]
    for sym, (x, y, z) in zip(symbols, coords):
        coord_lines.append(f"{sym:<5s}{x:14.10f}{y:14.10f}{z:14.10f}")

    text = re.sub(
        r"%begin_coords\s*\n.*?\n%end_coords",
        "%begin_coords\n" + "\n".join(coord_lines) + "\n%end_coords",
        text,
        count=1,
        flags=re.DOTALL,
    )
    return text


def run_hartree_fock(executable: Path, input_path: Path) -> str:
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
            f"hartree-fock failed (exit {proc.returncode}) on {input_path}"
        )
    return proc.stdout


def parse_mp2_energy(output: str) -> float:
    matches = MP2_ENERGY_RE.findall(output)
    if not matches:
        raise SystemExit("no 'Total MP2 Energy' line found in hartree-fock output")
    return float(matches[-1])


def parse_analytic_gradient(output: str, natom: int) -> list[list[float]]:
    grad: dict[int, list[float]] = {}
    for line in output.splitlines():
        match = ATOM_LINE_RE.search(line)
        if match:
            idx = int(match.group(1))
            grad[idx] = [
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)),
            ]

    if len(grad) < natom:
        raise SystemExit(
            f"only parsed {len(grad)}/{natom} gradient lines from hartree-fock"
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
) -> list[list[float]]:
    delta_ang = delta_bohr / ANGSTROM_TO_BOHR
    natom = len(symbols)
    gradient = [[0.0, 0.0, 0.0] for _ in range(natom)]

    for atom in range(natom):
        for axis in range(3):
            energies: list[float] = []
            for sign in (+1.0, -1.0):
                shifted = [row[:] for row in coords]
                shifted[atom][axis] += sign * delta_ang
                input_text = render_input(
                    template,
                    symbols,
                    shifted,
                    charge,
                    multiplicity,
                    calculation="energy",
                )
                input_path = workdir / f"fd_{atom}_{axis}_{'p' if sign > 0 else 'm'}.hfinp"
                input_path.write_text(input_text, encoding="utf-8")
                output = run_hartree_fock(executable, input_path)
                energies.append(parse_mp2_energy(output))

            e_plus, e_minus = energies
            gradient[atom][axis] = (e_plus - e_minus) / (2.0 * delta_bohr)

    return gradient


def max_abs_diff(
    left: list[list[float]],
    right: list[list[float]],
) -> tuple[float, tuple[int, int]]:
    worst = 0.0
    where = (0, 0)
    for atom, (left_row, right_row) in enumerate(zip(left, right)):
        for axis, (lhs, rhs) in enumerate(zip(left_row, right_row)):
            diff = abs(lhs - rhs)
            if diff > worst:
                worst = diff
                where = (atom, axis)
    return worst, where


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent
        / "inputs"
        / "regression"
        / "post_hf"
        / "h2_rmp2_gradient.hfinp",
        help="base .hfinp; calculation field will be replaced as needed",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "build",
        help="directory containing hartree-fock binary (default: ./build)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0e-3,
        help="finite-difference step in Bohr (default 1e-3)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=3.0e-4,
        help="component-wise tolerance (Ha/Bohr) on |g_analytic - g_fd|",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="keep the temporary finite-difference working directory",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    executable = args.build_dir / "hartree-fock"
    if not executable.exists():
        raise SystemExit(f"hartree-fock binary not found at {executable}")
    if not args.input.exists():
        raise SystemExit(f"input file not found: {args.input}")

    template = args.input.read_text(encoding="utf-8")
    symbols, coords, charge, multiplicity = parse_coords_block(template)
    natom = len(symbols)

    tmp = tempfile.TemporaryDirectory(prefix="hartree-fock-rmp2-fd-")
    workdir = Path(tmp.name)
    if args.keep_tmp:
        tmp._finalizer.detach()  # type: ignore[attr-defined]

    analytic_input = render_input(
        template, symbols, coords, charge, multiplicity, calculation="gradient"
    )
    analytic_path = workdir / "analytic.hfinp"
    analytic_path.write_text(analytic_input, encoding="utf-8")
    analytic_output = run_hartree_fock(executable, analytic_path)
    analytic_gradient = parse_analytic_gradient(analytic_output, natom)

    fd_gradient = central_difference_gradient(
        executable,
        template,
        symbols,
        coords,
        charge,
        multiplicity,
        args.delta,
        workdir,
    )

    print("Analytic gradient (Ha/Bohr):")
    for atom, row in enumerate(analytic_gradient, start=1):
        print(f"  Atom {atom}: {row[0]:14.8f}  {row[1]:14.8f}  {row[2]:14.8f}")
    print("Finite-difference gradient (Ha/Bohr):")
    for atom, row in enumerate(fd_gradient, start=1):
        print(f"  Atom {atom}: {row[0]:14.8f}  {row[1]:14.8f}  {row[2]:14.8f}")

    worst, (atom, axis) = max_abs_diff(analytic_gradient, fd_gradient)
    print(
        f"max |g_analytic - g_fd| = {worst:.3e} Ha/Bohr "
        f"(atom {atom + 1}, axis {'xyz'[axis]})"
    )
    print(f"tolerance               = {args.atol:.3e} Ha/Bohr")

    if args.keep_tmp:
        print(f"work directory kept at: {workdir}")

    if worst > args.atol:
        print("FAIL: analytic RMP2 gradient deviates from finite difference")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
