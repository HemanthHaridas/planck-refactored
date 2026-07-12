#!/usr/bin/env python3
"""planck-mpi Tier-3 smoke test: distributed dispatch is bitwise-correct.

Runs the supplied input twice — once through the serial binary
(`hartree-fock` / `planck-dft`, auto-selected from the input) and once
through `mpirun -n <ranks> planck-mpi` — and fails if the final energy
disagrees by more than --atol.

At Tier 3 the two must match to the bit: planck-mpi only adds MPI_Init,
rank-0 I/O gating, and the is_dft_run() dispatch on top of the identical
Driver::run — nothing partitions work yet, so every rank computes the
same replicated result. This is the gate Tier 1 (distributed Fock build)
must not break.

planck-mpi is opt-in (`cmake -DBUILD_MPI=ON`). When the binary or `mpirun`
is absent the test SKIPs green rather than failing, so the default
BUILD_MPI=OFF CI run stays clean. Mirrors tests/engine_scf_energy_compare.py.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# `hartree-fock`/`planck-mpi` print "Total Energy : ..."; the DFT path prints
# "DFT Energy : ...". Same 10-decimal canonical precision; take the last match.
ENERGY_RE = re.compile(r"(?:Total Energy|DFT Energy)\s*:?\s+([-+0-9Ee\.]+)")


def detect_serial_binary(template: str) -> str:
    return "planck-dft" if re.search(r"%begin_dft\b", template) else "hartree-fock"


def parse_energy(output: str) -> float:
    matches = ENERGY_RE.findall(output)
    if not matches:
        raise SystemExit("no `Total Energy` or `DFT Energy` line in output")
    return float(matches[-1])


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{' '.join(cmd)} failed (exit {proc.returncode})")
    return proc.stdout


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".hfinp to run serial vs mpirun")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "build",
    )
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument(
        "--atol",
        type=float,
        default=5.0e-9,
        help="tolerance on |E_serial - E_mpi| (Eh); Tier 3 is bitwise so 5e-9 is slack",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    mpi_bin = args.build_dir / "planck-mpi"
    mpirun = shutil.which("mpirun")
    if not mpi_bin.exists() or mpirun is None:
        why = "planck-mpi not built (cmake -DBUILD_MPI=ON)" if not mpi_bin.exists() else "mpirun not found"
        print(f"SKIP: {why}")
        return 0

    template = args.input.read_text(encoding="utf-8")
    serial_bin = args.build_dir / detect_serial_binary(template)
    if not serial_bin.exists():
        raise SystemExit(f"serial binary not found at {serial_bin}")

    e_serial = parse_energy(run([str(serial_bin), str(args.input)]))
    # --oversubscribe so the test runs on a machine with fewer cores than ranks.
    e_mpi = parse_energy(
        run([mpirun, "--oversubscribe", "-n", str(args.ranks), str(mpi_bin), str(args.input)])
    )

    delta = abs(e_serial - e_mpi)
    print(f"Energy (serial)         : {e_serial:.10f} Eh")
    print(f"Energy (mpi -n {args.ranks})       : {e_mpi:.10f} Eh")
    print(f"|delta E (mpi - serial)| : {delta:.3e} Eh (tol {args.atol:.3e})")

    if delta > args.atol:
        print("FAIL: planck-mpi energy diverges from serial reference")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
