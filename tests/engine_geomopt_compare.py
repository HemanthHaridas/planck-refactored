#!/usr/bin/env python3
"""Cross-engine geometry-optimization validation.

Runs the supplied .hfinp twice (engine os, engine hgp), parses the
geomopt's `Final Energy : ...` line and the optimized-geometry block,
and fails if the final energies disagree by more than --atol-energy or
the optimized geometries RMSD by more than --atol-rmsd.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


FINAL_ENERGY_RE = re.compile(r"Final Energy\s*:\s*([-+0-9Ee\.]+)")
OPT_GEOM_HEADER_RE = re.compile(r"Optimized Geometry \(Angstrom\)\s*:")
OPT_GEOM_LINE_RE = re.compile(
    r"Atom\s+(\d+)\s*:\s*\d+\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)"
)


def render_with_engine(template: str, engine: str) -> str:
    text, count = re.subn(
        r"(engine\s+)\S+",
        rf"\g<1>{engine}",
        template,
        count=1,
    )
    if count == 0:
        raise SystemExit("could not find `engine` line in input to rewrite")
    return text


def run_binary(executable: Path, input_path: Path) -> str:
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
            f"{executable.name} failed (exit {proc.returncode}) on {input_path}"
        )
    return proc.stdout


def parse_final_energy(output: str) -> float:
    matches = FINAL_ENERGY_RE.findall(output)
    if not matches:
        raise SystemExit("no `Final Energy :` line found in output")
    return float(matches[-1])


def parse_optimized_geometry(output: str) -> list[list[float]]:
    # Pick the block immediately after the last `Optimized Geometry (Angstrom):`
    # header — there is exactly one per geomopt, but the post-opt symmetry SCF
    # also emits per-atom lines so we anchor on the header to avoid bleed.
    header_iter = list(OPT_GEOM_HEADER_RE.finditer(output))
    if not header_iter:
        raise SystemExit("no `Optimized Geometry (Angstrom)` header in output")
    tail = output[header_iter[-1].end():]
    geom: dict[int, list[float]] = {}
    for line in tail.splitlines():
        match = OPT_GEOM_LINE_RE.search(line)
        if match:
            idx = int(match.group(1))
            geom[idx] = [
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)),
            ]
        elif geom and not line.strip():
            # Blank line after the geometry block — stop before the post-opt SCF
            # report continues with its own per-atom output.
            break
    if not geom:
        raise SystemExit("optimized-geometry block parsed empty")
    return [geom[i] for i in sorted(geom)]


def coord_rmsd(left: list[list[float]], right: list[list[float]]) -> float:
    if len(left) != len(right):
        raise SystemExit(
            f"geometry atom-count mismatch: os={len(left)} hgp={len(right)}"
        )
    acc = 0.0
    n = 0
    for left_row, right_row in zip(left, right):
        for lhs, rhs in zip(left_row, right_row):
            acc += (lhs - rhs) ** 2
            n += 1
    return math.sqrt(acc / n) if n else 0.0


def detect_binary_for(template: str) -> str:
    if re.search(r"%begin_dft\b", template):
        return "planck-dft"
    return "hartree-fock"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="base .hfinp (engine line rewritten in scratch)")
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help="binary in --build-dir to invoke; auto-selected from input (hartree-fock vs planck-dft) when omitted",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "build",
        help="directory containing the binary (default: ./build)",
    )
    parser.add_argument(
        "--atol-energy",
        type=float,
        default=1.0e-8,
        help="absolute tolerance on |E_os_final - E_hgp_final| (Eh)",
    )
    parser.add_argument(
        "--atol-rmsd",
        type=float,
        default=1.0e-5,
        help="tolerance on optimized-geometry RMSD across engines (Angstrom)",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="keep the scratch directory with both inputs and outputs",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    template = args.input.read_text(encoding="utf-8")
    binary = args.binary if args.binary is not None else detect_binary_for(template)
    executable = args.build_dir / binary
    if not executable.exists():
        raise SystemExit(f"binary not found at {executable}")

    tmp = tempfile.TemporaryDirectory(prefix="engine-geomopt-compare-")
    workdir = Path(tmp.name)
    if args.keep_tmp:
        tmp._finalizer.detach()  # type: ignore[attr-defined]

    energies: dict[str, float] = {}
    geometries: dict[str, list[list[float]]] = {}
    for engine in ("os", "hgp"):
        rendered = render_with_engine(template, engine)
        path = workdir / f"{engine}.hfinp"
        path.write_text(rendered, encoding="utf-8")
        output = run_binary(executable, path)
        (workdir / f"{engine}.log").write_text(output, encoding="utf-8")
        energies[engine] = parse_final_energy(output)
        geometries[engine] = parse_optimized_geometry(output)

    print(f"Final Energy (OS) : {energies['os']:.10f} Eh")
    print(f"Final Energy (HGP): {energies['hgp']:.10f} Eh")
    delta_e = abs(energies["os"] - energies["hgp"])
    rmsd = coord_rmsd(geometries["os"], geometries["hgp"])
    print(f"|delta E|         : {delta_e:.3e} Eh (tol {args.atol_energy:.3e})")
    print(f"geometry RMSD     : {rmsd:.3e} Angstrom (tol {args.atol_rmsd:.3e})")

    if args.keep_tmp:
        print(f"work directory kept at: {workdir}")

    if delta_e > args.atol_energy:
        print("FAIL: HGP geomopt final energy diverges from OS reference")
        return 1
    if rmsd > args.atol_rmsd:
        print("FAIL: HGP optimized geometry diverges from OS reference")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
