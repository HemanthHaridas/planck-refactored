#!/usr/bin/env python3
"""One-command, termwise FD audit for the disabled RKS double-hybrid gradient.

The script runs the paper-object gradient staging at R, R+h and R-h.  Each
run is expected to finish staging and then return the intentional disabled-DH
gradient error.  It compares (i) central differences of the stationary Eq. 11
energy partition to the matching Eq. 33/common scalar ledgers, and (ii) the
live MO matrices in a deterministic canonical-orbital sign gauge.

It is a localization tool, not a regression test: individual Eq. 33 terms are
not independent energy derivatives.  Its output marks those rows explicitly
and reports only the two legitimate total comparisons as residuals.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ANGSTROM_TO_BOHR = 1.8897261254535
DISABLED = "DFT double-hybrid analytic gradient remains disabled"


def parse_geometry(text: str) -> tuple[list[str], list[list[float]], int, int]:
    match = re.search(r"%begin_coords\s*\n(.*?)\n%end_coords", text, re.S)
    if not match:
        raise ValueError("input has no coordinate block")
    lines = [line for line in match.group(1).splitlines() if line.strip()]
    natom = int(lines[0].split()[0])
    charge, multiplicity = (int(value) for value in lines[1].split()[:2])
    symbols = [line.split()[0] for line in lines[2 : 2 + natom]]
    coordinates = [[float(value) for value in line.split()[1:4]]
                   for line in lines[2 : 2 + natom]]
    return symbols, coordinates, charge, multiplicity


def render_gradient_input(template: str, symbols: list[str], coordinates: list[list[float]],
                          charge: int, multiplicity: int) -> str:
    coord_lines = [str(len(symbols)), f"{charge}   {multiplicity}"]
    coord_lines.extend(f"{symbol:<5s}{xyz[0]:14.10f}{xyz[1]:14.10f}{xyz[2]:14.10f}"
                       for symbol, xyz in zip(symbols, coordinates))
    text = re.sub(r"(calculation\s+)\S+", r"\g<1>gradient", template, count=1)
    return re.sub(r"%begin_coords\s*\n.*?\n%end_coords",
                  "%begin_coords\n" + "\n".join(coord_lines) + "\n%end_coords",
                  text, count=1, flags=re.S)


def run(executable: Path, input_path: Path, dump_path: Path) -> str:
    env = os.environ | {
        "PLANCK_DFT_DH_LIVE_OBJECT_AUDIT": "1",
        "PLANCK_DFT_DH_LITERAL_EQ47_OVERLAP": "1",
        "PLANCK_DFT_DH_FOCK_DERIVATIVE_ORACLE": "1",
        "PLANCK_DFT_DH_CHANNEL_MATRIX_DUMP": str(dump_path),
    }
    process = subprocess.run([str(executable), str(input_path)], text=True,
                             capture_output=True, env=env, check=False)
    output = process.stdout + process.stderr
    if DISABLED not in output:
        raise RuntimeError(f"staging failed before the expected disabled guard:\n{output}")
    if not dump_path.exists():
        raise RuntimeError("DH channel matrix dump was not written")
    return output


def labeled_fields(output: str, label: str) -> dict[str, float]:
    line = next((line for line in output.splitlines() if label in line), None)
    if line is None:
        raise ValueError(f"missing diagnostic label: {label}")
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    return {name: float(value) for name, value in re.findall(
        rf"([A-Za-z_][A-Za-z0-9_']*)=({number})", line)}


def read_matrices(path: Path) -> dict[str, list[list[float]]]:
    lines = iter(path.read_text().splitlines())
    if next(lines, None) != "PLANCK_DH_CHANNEL_MATRIX_V1":
        raise ValueError(f"unrecognized matrix dump: {path}")
    matrices: dict[str, list[list[float]]] = {}
    for header in lines:
        name, rows, columns = header.split()
        data = [[float(value) for value in next(lines).split()] for _ in range(int(rows))]
        if len(data) != int(rows) or any(len(row) != int(columns) for row in data):
            raise ValueError(f"bad {name} dimensions in {path}")
        matrices[name] = data
    return matrices


def fd(plus: float, minus: float, step: float) -> float:
    return (plus - minus) / (2.0 * step)


def field_fd(outputs: dict[str, str], label: str, field: str, step: float) -> float:
    return fd(labeled_fields(outputs["plus"], label)[field],
              labeled_fields(outputs["minus"], label)[field], step)


def matrix_linear_combination(a: list[list[float]], b: list[list[float]], c: list[list[float]],
                              scale: float) -> list[list[float]]:
    if len(a) != len(b) or len(a) != len(c) or any(len(x) != len(y) or len(x) != len(z)
                                                       for x, y, z in zip(a, b, c)):
        raise ValueError("incompatible matrix dimensions across FD geometries")
    return [[scale * (x + y - 2.0 * z) for x, y, z in zip(row_a, row_b, row_c)]
            for row_a, row_b, row_c in zip(a, b, c)]


def matrix_difference_over_step(plus: list[list[float]], minus: list[list[float]], step: float) -> list[list[float]]:
    if len(plus) != len(minus) or any(len(x) != len(y) for x, y in zip(plus, minus)):
        raise ValueError("incompatible matrix dimensions across FD geometries")
    return [[(x - y) / (2.0 * step) for x, y in zip(row_plus, row_minus)]
            for row_plus, row_minus in zip(plus, minus)]


def frobenius(matrix: list[list[float]]) -> float:
    return sum(value * value for row in matrix for value in row) ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        default=Path("tests/inputs/exploratory/dh_gradient/water_b2plyp_gradient_fd.hfinp"),
                        nargs="?")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--atom", type=int, default=2,
                        help="one-based displaced atom (default: H1 / atom 2)")
    parser.add_argument("--axis", choices="xyz", default="x")
    parser.add_argument("--step", type=float, default=1.0e-4, help="Bohr")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    executable = args.build_dir / "planck-dft"
    if not executable.exists():
        raise SystemExit(f"missing executable: {executable}")
    template = args.input.read_text()
    symbols, coordinates, charge, multiplicity = parse_geometry(template)
    if not 1 <= args.atom <= len(symbols):
        raise SystemExit("--atom outside coordinate range")
    axis = "xyz".index(args.axis)
    workspace = tempfile.TemporaryDirectory(prefix="planck-dh-channel-fd-")
    root = Path(workspace.name)
    outputs: dict[str, str] = {}
    matrices: dict[str, dict[str, list[list[float]]]] = {}
    for tag, sign in (("center", 0.0), ("plus", 1.0), ("minus", -1.0)):
        displaced = [row[:] for row in coordinates]
        displaced[args.atom - 1][axis] += sign * args.step / ANGSTROM_TO_BOHR
        input_path = root / f"{tag}.hfinp"
        dump_path = root / f"{tag}.matrices"
        input_path.write_text(render_gradient_input(template, symbols, displaced, charge, multiplicity))
        print(f"running {tag} ...", flush=True)
        outputs[tag] = run(executable, input_path, dump_path)
        matrices[tag] = read_matrices(dump_path)

    # The first block is the only directly meaningful termwise FD partition:
    # pair + Tr[D'F] is exactly the Eq. 11 stationary energy.  The individual
    # rows are deliberately retained because their cancellation localizes a
    # defect far more sharply than the final total alone.
    pair_fd = field_fd(outputs, "DH Eq. 11 Energy :", "pair", args.step)
    fock_fd = field_fd(outputs, "DH Eq. 11 Energy :", "Dprime_F", args.step)
    total_fd = field_fd(outputs, "DH Eq. 11 Energy :", "total", args.step)
    print("\nEq. 11 live-energy central difference (Ha/Bohr)")
    print(f"  pair Eq. 47                 {pair_fd:+.12e}")
    print(f"  Tr[D'F_KS]                  {fock_fd:+.12e}")
    print(f"  sum                         {pair_fd + fock_fd:+.12e}")
    print(f"  production correction       {total_fd:+.12e}")
    print(f"  stationarity closure        {pair_fd + fock_fd - total_fd:+.3e}")
    print("\nEq. 11 D'F subchannels: live FD only (not separate Eq. 33 derivatives)")
    for name in ("h", "J", "K", "XC"):
        print(f"  D':{name:<2s}                    "
              f"{field_fd(outputs, 'DH Eq. 11 Dprime F :', name, args.step):+.12e}")

    # Common functional and Eq. 33 are the two completed analytic forms.  The
    # first comparison is the correct aggregate analytic-vs-live-FD residual;
    # the second makes a wrong stationary rearrangement immediately visible.
    common = labeled_fields(outputs["center"], "DH Common Eq11/Eq33 Ledger :")
    converted_common = common["common_total"] + common["Eq46_rearrangement"] + common["Z_rearrangement"]
    print("\nAnalytic functional forms versus the same live-energy FD (Ha/Bohr)")
    for name, analytic in (("common (unconverted)", common["common_total"]),
                           ("common + Delta_46 + Delta_Z", converted_common),
                           ("Eq. 33 stationary", common["Eq33_total"])):
        print(f"  {name:<24s} analytic={analytic:+.12e}  FD={total_fd:+.12e}  "
              f"FD-analytic={total_fd - analytic:+.12e}")
    print("\nEq. 33 scalar channels at R (not individually FD observables)")
    zero = labeled_fields(outputs["center"], "DH Dprime Eq. 33 :")
    z = labeled_fields(outputs["center"], "DH Z Eq. 33 :")
    for name, value in (("D' one", zero["one"]), ("D' overlap", zero["overlap"]),
                        ("D' separable", zero["separable"]), ("D' XC-II", zero["XC_II"]),
                        ("Eq. 47 pair", zero["Eq47_pair"]), ("Z total", z["total"])):
        print(f"  {name:<18s} {value:+.12e}")

    print("\nLive matrix central differences in canonical MO sign gauge")
    print("  matrix                 ||(A+ - A-)/(2h)||_F    ||A+ + A- - 2A0||_F")
    for name in ("DPRIME", "D", "W", "L_RESPONSE", "L_EXTERNAL", "L_INTERNAL", "L", "Z", "T_TILDE"):
        slope = matrix_difference_over_step(matrices["plus"][name], matrices["minus"][name], args.step)
        curvature = matrix_linear_combination(matrices["plus"][name], matrices["minus"][name],
                                               matrices["center"][name], 1.0)
        print(f"  {name:<20s} {frobenius(slope):+.12e}       {frobenius(curvature):+.12e}")

    if args.keep:
        workspace._finalizer.detach()  # type: ignore[attr-defined]
        print(f"\nAudit artifacts retained: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
