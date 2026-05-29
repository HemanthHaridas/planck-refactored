#!/usr/bin/env python3
"""Cross-engine SCF total-energy validation.

Runs the supplied .hfinp twice (engine os, engine hgp), parses the final
`Total Energy` line from each output, and fails if the two energies
disagree by more than --atol.

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

    energies: dict[str, float] = {}
    for engine in ("os", "hgp"):
        rendered = render_with_engine(template, engine)
        path = workdir / f"{engine}.hfinp"
        path.write_text(rendered, encoding="utf-8")
        output = run_binary(executable, path)
        (workdir / f"{engine}.log").write_text(output, encoding="utf-8")
        energies[engine] = parse_total_energy(output)

    print(f"Total Energy (OS)  : {energies['os']:.10f} Eh")
    print(f"Total Energy (HGP) : {energies['hgp']:.10f} Eh")
    delta = abs(energies["os"] - energies["hgp"])
    print(f"|delta E|          : {delta:.3e} Eh (tol {args.atol:.3e})")

    if args.keep_tmp:
        print(f"work directory kept at: {workdir}")

    if delta > args.atol:
        print("FAIL: HGP SCF total energy diverges from OS reference")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
