#!/usr/bin/env python3
"""Cross-engine analytic-gradient validation.

Runs the supplied .hfinp twice, once with `engine os` and once with
`engine hgp`, parses the analytic nuclear gradient from each output, and
fails if any per-atom Cartesian component disagrees by more than --atol.

Covers HF and DFT gradient inputs by selecting the right binary via
--binary (default `hartree-fock`). Reuses the existing OS-engine inputs
from tests/inputs/regression/* unchanged; the driver edits only the
`engine` line in a scratch copy.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


ATOM_LINE_RE = re.compile(
    r"Atom\s+(\d+)\s*:\s*([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)"
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


def parse_gradient(output: str) -> list[list[float]]:
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
    if not grad:
        raise SystemExit("no `Atom i: gx gy gz` lines found in output")
    return [grad[i] for i in sorted(grad)]


def max_abs_diff(
    left: list[list[float]],
    right: list[list[float]],
) -> tuple[float, tuple[int, int]]:
    worst = 0.0
    where = (0, 0)
    if len(left) != len(right):
        raise SystemExit(
            f"gradient atom-count mismatch: os={len(left)} hgp={len(right)}"
        )
    for atom, (left_row, right_row) in enumerate(zip(left, right)):
        if len(left_row) != len(right_row):
            raise SystemExit("gradient row width mismatch")
        for axis, (lhs, rhs) in enumerate(zip(left_row, right_row)):
            diff = abs(lhs - rhs)
            if diff > worst:
                worst = diff
                where = (atom, axis)
    return worst, where


def detect_binary_for(template: str) -> str:
    # `%begin_dft` is present only for KS calculations; the HF binary doesn't
    # know how to consume it, so route DFT inputs to planck-dft automatically.
    if re.search(r"%begin_dft\b", template):
        return "planck-dft"
    return "hartree-fock"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="base .hfinp (engine line will be replaced)")
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
        "--atol",
        type=float,
        default=1.0e-7,
        help=(
            "component-wise tolerance on |g_os - g_hgp| (Ha/Bohr). "
            "Driver prints with 8-decimal precision, so the realistic floor "
            "from text parsing is ~5e-9 per component; the default leaves "
            "~20 ulps of headroom for cumulative SCF/dispatch roundoff."
        ),
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

    tmp = tempfile.TemporaryDirectory(prefix="engine-gradient-compare-")
    workdir = Path(tmp.name)
    if args.keep_tmp:
        tmp._finalizer.detach()  # type: ignore[attr-defined]

    results: dict[str, list[list[float]]] = {}
    for engine in ("os", "hgp"):
        rendered = render_with_engine(template, engine)
        path = workdir / f"{engine}.hfinp"
        path.write_text(rendered, encoding="utf-8")
        output = run_binary(executable, path)
        (workdir / f"{engine}.log").write_text(output, encoding="utf-8")
        results[engine] = parse_gradient(output)

    print("OS gradient (Ha/Bohr):")
    for atom, row in enumerate(results["os"], start=1):
        print(f"  Atom {atom}: {row[0]:14.10f}  {row[1]:14.10f}  {row[2]:14.10f}")
    print("HGP gradient (Ha/Bohr):")
    for atom, row in enumerate(results["hgp"], start=1):
        print(f"  Atom {atom}: {row[0]:14.10f}  {row[1]:14.10f}  {row[2]:14.10f}")

    worst, (atom, axis) = max_abs_diff(results["os"], results["hgp"])
    print(
        f"max |g_os - g_hgp| = {worst:.3e} Ha/Bohr "
        f"(atom {atom + 1}, axis {'xyz'[axis]})"
    )
    print(f"tolerance          = {args.atol:.3e} Ha/Bohr")

    if args.keep_tmp:
        print(f"work directory kept at: {workdir}")

    if worst > args.atol:
        print("FAIL: HGP analytic gradient diverges from OS reference")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
