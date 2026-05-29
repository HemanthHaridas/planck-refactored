#!/usr/bin/env python3
"""Cross-engine SCF total-energy validation.

Runs the supplied .hfinp four times (engine os, hgp, rys, auto), parses
the final `Total Energy` line from each output, and fails if any of
hgp / rys / auto disagree with the OS reference by more than --atol.

Binary (`hartree-fock` vs `planck-dft`) is auto-selected from the
presence of `%begin_dft` in the input. Default tolerance 5e-9 Eh leaves
~50 ulps of headroom over the 10-decimal canonical print precision.

Mirrors tests/engine_gradient_compare.py and engine_geomopt_compare.py so
the regression runner wires all three the same way.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


# `hartree-fock` prints "Total Energy : ..."; `planck-dft` prints
# "DFT Energy : ...". Same canonical 10-decimal precision; the regex accepts
# either label and the comparator picks the last match.
ENERGY_RE = re.compile(r"(?:Total Energy|DFT Energy)\s*:?\s+([-+0-9Ee\.]+)")


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


def parse_total_energy(output: str) -> float:
    matches = ENERGY_RE.findall(output)
    if not matches:
        raise SystemExit("no `Total Energy` or `DFT Energy` line found in output")
    return float(matches[-1])


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
        help="binary in --build-dir to invoke; auto-selected from input when omitted",
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
        default=5.0e-9,
        help=(
            "absolute tolerance on |E_os - E_hgp| (Eh). Default 5e-9 leaves ~50 "
            "ulps of headroom over the 10-decimal `Total Energy` print precision."
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

    tmp = tempfile.TemporaryDirectory(prefix="engine-scf-energy-compare-")
    workdir = Path(tmp.name)
    if args.keep_tmp:
        tmp._finalizer.detach()  # type: ignore[attr-defined]

    engines = ("os", "hgp", "rys", "auto")
    energies: dict[str, float] = {}
    for engine in engines:
        rendered = render_with_engine(template, engine)
        path = workdir / f"{engine}.hfinp"
        path.write_text(rendered, encoding="utf-8")
        output = run_binary(executable, path)
        (workdir / f"{engine}.log").write_text(output, encoding="utf-8")
        energies[engine] = parse_total_energy(output)

    # Label format matches the 2-engine comparator the regression manifest
    # still pins on: `Total Energy (OS)`, `Total Energy (HGP)`. Newer engines
    # use the same uppercase convention.
    for engine in engines:
        print(f"Total Energy ({engine.upper()}) : {energies[engine]:.10f} Eh")
    e_ref = energies["os"]
    deltas = {e: abs(energies[e] - e_ref) for e in engines if e != "os"}
    for e, d in deltas.items():
        print(f"|delta E ({e.upper()} - OS)| : {d:.3e} Eh (tol {args.atol:.3e})")

    if args.keep_tmp:
        print(f"work directory kept at: {workdir}")

    failures = [e for e, d in deltas.items() if d > args.atol]
    if failures:
        print(
            f"FAIL: {', '.join(failures)} SCF total energy diverges from OS reference"
        )
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
