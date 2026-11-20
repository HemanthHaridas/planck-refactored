#!/usr/bin/env python3
"""planck-mpi Tier-3 smoke test: distributed dispatch is bitwise-correct.

Runs the supplied input twice — once through the serial binary
(`hartree-fock` / `planck-dft`, auto-selected from the input) and once
through `mpirun -n <ranks> planck-mpi` — and fails if the final energy
disagrees by more than --atol. Any printed MP2 total energies and gradient
rows are compared too, at the same tolerance.

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
# Also compared when present, every occurrence in order: the MP2 total energy
# and each printed gradient row (RI under MPI, docs/RI_MPI_SCOPE.md).
MP2_RE = re.compile(r"Total MP2 Energy\s*:?\s+([-+0-9Ee\.]+)")
GRAD_RE = re.compile(r"Atom\s+\d+:\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)")


def detect_serial_binary(template: str) -> str:
    return "planck-dft" if re.search(r"%begin_dft\b", template) else "hartree-fock"


def parse_energy(output: str) -> float:
    matches = ENERGY_RE.findall(output)
    if not matches:
        raise SystemExit("no `Total Energy` or `DFT Energy` line in output")
    return float(matches[-1])


def parse_extra(output: str) -> list[float]:
    values = [float(v) for v in MP2_RE.findall(output)]
    for row in GRAD_RE.findall(output):
        values.extend(float(v) for v in row)
    return values


def run(cmd: list[str], timeout: float) -> str:
    # A missing reduce leaves ranks with different data; their SCFs then
    # finish at different iterations and the collectives stop pairing up, so
    # the defect shows as a hang, not a wrong number. Time out, and SIGTERM
    # first so mpirun can take its ranks down with it.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        raise SystemExit(f"{' '.join(cmd)} timed out after {timeout:g}s "
                         "(ranks may have diverged and deadlocked in a collective)")
    if proc.returncode != 0:
        sys.stderr.write(stdout)
        sys.stderr.write(stderr)
        raise SystemExit(f"{' '.join(cmd)} failed (exit {proc.returncode})")
    return stdout


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".hfinp to run serial vs mpirun")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "build",
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[2],
                        help="one or more rank counts, each compared with serial")
    parser.add_argument(
        "--atol",
        type=float,
        default=5.0e-9,
        help="tolerance on |E_serial - E_mpi| (Eh); 0 demands every printed digit agree",
    )
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-run limit in seconds; a desynchronized run hangs instead of failing")
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

    out_serial = run([str(serial_bin), str(args.input)], args.timeout)
    e_serial = parse_energy(out_serial)
    x_serial = parse_extra(out_serial)
    print(f"Energy (serial)          : {e_serial:.10f} Eh   (+{len(x_serial)} MP2/gradient values)")
    failed = False
    for ranks in args.ranks:
        # --oversubscribe so the test runs on a machine with fewer cores than ranks.
        out_mpi = run([mpirun, "--oversubscribe", "-n", str(ranks), str(mpi_bin), str(args.input)],
                      args.timeout)
        e_mpi = parse_energy(out_mpi)
        x_mpi = parse_extra(out_mpi)
        delta = abs(e_serial - e_mpi)
        print(f"Energy (mpi -n {ranks:<2})       : {e_mpi:.10f} Eh   |delta| {delta:.3e} (tol {args.atol:.3e})")
        failed |= delta > args.atol
        if len(x_mpi) != len(x_serial):
            print(f"  MP2/gradient value count {len(x_mpi)} != serial {len(x_serial)}")
            failed = True
        elif x_serial:
            x_delta = max(abs(a - b) for a, b in zip(x_serial, x_mpi))
            print(f"  max |delta| over {len(x_serial)} MP2/gradient values: {x_delta:.3e}")
            failed |= x_delta > args.atol

    if failed:
        print("FAIL: planck-mpi energy diverges from serial reference")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
