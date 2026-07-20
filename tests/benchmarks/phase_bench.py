#!/usr/bin/env python3
"""Where does the missing strong-scaling efficiency actually go?

HF measures ~9.7-10x on 16 Notchpeak ranks (~61% efficiency) with no
load-imbalance signal, and efficiency RISES with system size. That combination
is the signature of a fixed serial cost being amortized -- Amdahl, not
imbalance. 9.7x on 16 ranks implies a serial fraction f ~= 4.3%:

    1/S = f + (1-f)/p   ->   f = (p/S - 1)/(p - 1)

This script answers the only question that matters before anyone writes a
distributed eigensolver: WHICH phase is that 4%? It splits the SCF iteration
into four buckets (Fock build / DIIS / diagonalization / rest) via the
PLANCK_PHASE_TIMING instrumentation in src/scf/scf.cpp, runs the same case at
-n 1 and -n k, and compares the measured non-Fock fraction against the f that
the observed speedup implies.

The verdict it produces:

  * diag dominates the serial residual  -> ScaLAPACK/Elemental is justified
  * diag is a rounding error            -> ScaLAPACK cannot buy the missing
                                           efficiency; the cost is scattered
                                           across setup/DIIS/density and 61%
                                           at 16 ranks is where this stops

Usage:
    python3 tests/benchmarks/phase_bench.py --build build --waters 16 --ranks 16
    MPIRUN=srun python3 tests/benchmarks/phase_bench.py --build build --ranks 16

Notes:
  * RHF only. The instrumentation is in the restricted loop; that is the path
    the 9.7x number was measured on, so it is the path to explain.
  * Iteration 1 is dropped by default (--skip-iters): first-iteration cost
    includes one-time setup that is not part of the steady-state loop and would
    inflate "rest".
  * Per-rank numbers are kept. The MAX over ranks is what sets wall time (every
    rank waits for the slowest at the Allreduce), so the max-rank Fock time is
    the honest one; the spread across ranks is reported as an imbalance check.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_bench import (  # noqa: E402  -- reuse, don't reimplement
    case_dir,
    have_launcher,
    save_output,
    water_chain,
    write_case,
)

PHASE_RE = re.compile(
    r"PLANCK_PHASE\s+rank=(\d+)\s+iter=(\d+)\s+"
    r"fock_s=([0-9.eE+-]+)\s+diis_s=([0-9.eE+-]+)\s+"
    r"diag_s=([0-9.eE+-]+)\s+rest_s=([0-9.eE+-]+)\s+iter_s=([0-9.eE+-]+)"
)
PHASES = ("fock", "diis", "diag", "rest")


def parse_phases(text: str, skip_iters: int):
    """-> {(rank, iter): {phase: seconds}}, dropping the first skip_iters."""
    out = {}
    for m in PHASE_RE.finditer(text):
        rank, it = int(m.group(1)), int(m.group(2))
        if it <= skip_iters:
            continue
        out[(rank, it)] = {
            "fock": float(m.group(3)),
            "diis": float(m.group(4)),
            "diag": float(m.group(5)),
            "rest": float(m.group(6)),
            "iter": float(m.group(7)),
        }
    return out


def per_iter_max_over_ranks(samples):
    """Collapse ranks -> one row per iteration, taking the MAX per phase.

    Max, not mean: every rank blocks at the Allreduce until the slowest
    finishes, so the slowest rank's Fock time is what the wall clock actually
    pays. Averaging would hide exactly the imbalance we want to rule out.
    """
    by_iter = defaultdict(dict)
    for (rank, it), row in samples.items():
        for ph in (*PHASES, "iter"):
            by_iter[it][ph] = max(by_iter[it].get(ph, 0.0), row[ph])
    return by_iter


def median_phases(by_iter):
    if not by_iter:
        return None
    return {
        ph: statistics.median([row[ph] for row in by_iter.values()])
        for ph in (*PHASES, "iter")
    }


def rank_spread(samples):
    """Max/min Fock time across ranks, medianed over iterations.

    ~1.0 means the Fock build is evenly distributed. This is the
    load-imbalance control: if it is large, the Amdahl story is wrong and the
    problem is distribution, not a serial section.
    """
    by_iter = defaultdict(list)
    for (rank, it), row in samples.items():
        by_iter[it].append(row["fock"])
    ratios = [max(v) / min(v) for v in by_iter.values() if v and min(v) > 0]
    return statistics.median(ratios) if ratios else float("nan")


def run(exe: Path, atoms, basis, engine, ranks, threads, max_cycles, skip_iters):
    env = dict(
        os.environ, OMP_NUM_THREADS=str(threads), PLANCK_PHASE_TIMING="1"
    )
    with case_dir(f"phase_n{len(atoms)}_r{ranks}") as (td, tag):
        inp, probe = write_case(td, tag, atoms, basis, engine, "hf", max_cycles)
        if ranks == 1:
            cmd = [str(probe), str(exe), str(inp)]
        else:
            launcher = os.environ.get("MPIRUN", "mpirun").split()
            extra = os.environ.get("MPIRUN_EXTRA", "").split()
            cmd = [*launcher, "-n", str(ranks), *extra, str(probe), str(exe), str(inp)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t0
        save_output(td, tag, p)

    if p.returncode != 0:
        return None, f"exit {p.returncode}: {(p.stderr or p.stdout)[-400:]}"
    samples = parse_phases(p.stdout, skip_iters)
    if not samples:
        return None, (
            "no PLANCK_PHASE lines -- is the binary built with the phase-timing "
            "patch in src/scf/scf.cpp, and is this an RHF case?"
        )
    return {
        "samples": samples,
        "median": median_phases(per_iter_max_over_ranks(samples)),
        "spread": rank_spread(samples) if ranks > 1 else 1.0,
        "wall": wall,
        "ranks": ranks,
        "n_iter": len({it for _, it in samples}),
    }, None


def amdahl_f(speedup: float, p: int) -> float:
    """Serial fraction implied by an observed speedup. f = (p/S - 1)/(p - 1)."""
    if p <= 1 or speedup <= 0:
        return float("nan")
    return (p / speedup - 1.0) / (p - 1.0)


def report(serial, par):
    p = par["ranks"]
    s_med, p_med = serial["median"], par["median"]

    # Speedup on the SCF iteration itself, not total wall: startup, basis setup
    # and I/O are outside the loop and would otherwise contaminate f.
    speedup = s_med["iter"] / p_med["iter"] if p_med["iter"] > 0 else float("nan")
    f_obs = amdahl_f(speedup, p)

    # Measured serial residual: everything that is NOT the distributed Fock
    # build, as a fraction of the parallel iteration.
    nonfock = sum(p_med[ph] for ph in ("diis", "diag", "rest"))
    f_measured = nonfock / p_med["iter"] if p_med["iter"] > 0 else float("nan")

    print(f"\n{'':<8}{'serial (s)':>14}{f'{p}-rank (s)':>14}{'speedup':>10}"
          f"{f'% of {p}-rank iter':>18}")
    print("-" * 64)
    for ph in PHASES:
        sp = s_med[ph] / p_med[ph] if p_med[ph] > 1e-12 else float("nan")
        pct = 100.0 * p_med[ph] / p_med["iter"] if p_med["iter"] > 0 else float("nan")
        print(f"{ph:<8}{s_med[ph]:>14.4f}{p_med[ph]:>14.4f}{sp:>10.2f}{pct:>17.1f}%")
    print("-" * 64)
    print(f"{'iter':<8}{s_med['iter']:>14.4f}{p_med['iter']:>14.4f}"
          f"{speedup:>10.2f}{100.0:>17.1f}%")

    print(f"\nranks                    {p}")
    print(f"iterations sampled       {par['n_iter']} (serial {serial['n_iter']})")
    print(f"Fock max/min over ranks  {par['spread']:.2f}"
          "   (~1.0 = balanced; large = imbalance, not Amdahl)")
    print(f"\nefficiency               {100.0 * speedup / p:.1f}%")
    print(f"f implied by speedup     {100.0 * f_obs:.2f}%")
    print(f"f measured (non-Fock)    {100.0 * f_measured:.2f}%")

    diag_share = p_med["diag"] / nonfock if nonfock > 1e-12 else float("nan")
    print(f"diag share of non-Fock   {100.0 * diag_share:.1f}%")

    print("\nverdict")
    # Does the bucket accounting explain the observed loss? If measured serial
    # residual is far below what the speedup implies, the loss is somewhere the
    # buckets do not see (comm, startup, OMP oversubscription) and chasing the
    # eigensolver would be aiming at the wrong target.
    if f_obs == f_obs and f_measured < 0.4 * f_obs:
        print("  Buckets do NOT explain the loss: measured serial residual is well")
        print("  below the Amdahl f. Suspect communication, launch overhead, or")
        print("  thread oversubscription -- not the eigensolver. Check the Fock")
        print("  speedup row above and the rank spread first.")
    elif diag_share == diag_share and diag_share > 0.5:
        print("  Diagonalization dominates the serial residual. A distributed")
        print("  eigensolver (ScaLAPACK/Elemental) targets the real bottleneck.")
        print(f"  Ceiling if diag went to zero: {1.0 / max(f_obs - f_measured * diag_share, 1e-9):.1f}x")
    else:
        print("  Diagonalization is NOT the dominant serial cost. ScaLAPACK would")
        print("  buy little here; the residual is spread across DIIS/density/setup.")
        print("  Revisit only at larger nb, where diag's nb^3 catches the Fock build.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=Path("build"))
    ap.add_argument("--waters", type=int, default=16,
                    help="water-chain length (16 ~ nb=208 at 6-31g)")
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--engine", default="os",
                    help="os is the only MPI-distributed engine; hgp/rys replicate")
    ap.add_argument("--ranks", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1,
                    help="OMP threads per rank; keep at 1 so ranks are the only axis")
    ap.add_argument("--max-cycles", type=int, default=12,
                    help="phase medians converge fast; no need to run to convergence")
    ap.add_argument("--skip-iters", type=int, default=1,
                    help="drop leading iterations (one-time setup inflates 'rest')")
    args = ap.parse_args()

    serial_exe = args.build / "hartree-fock"
    mpi_exe = args.build / "planck-mpi"
    if not serial_exe.exists():
        print(f"missing {serial_exe} -- build first", file=sys.stderr)
        return 1
    if args.ranks > 1:
        if not mpi_exe.exists():
            print(f"missing {mpi_exe} -- cmake -DBUILD_MPI=ON", file=sys.stderr)
            return 1
        if not have_launcher():
            print("no MPI launcher found ($MPIRUN)", file=sys.stderr)
            return 1

    atoms = water_chain(args.waters)
    print(f"{args.waters} waters ({len(atoms)} atoms), {args.basis}, engine {args.engine}, "
          f"{args.threads} thread(s)/rank")

    print(f"  serial (1 rank) ...", flush=True)
    serial, err = run(serial_exe, atoms, args.basis, args.engine, 1,
                      args.threads, args.max_cycles, args.skip_iters)
    if err:
        print(f"  FAILED: {err}", file=sys.stderr)
        return 1

    if args.ranks == 1:
        print("\n--ranks 1: serial phase breakdown only (no scaling verdict)")
        for ph in (*PHASES, "iter"):
            print(f"  {ph:<6}{serial['median'][ph]:>10.4f} s")
        return 0

    print(f"  {args.ranks} ranks ...", flush=True)
    par, err = run(mpi_exe, atoms, args.basis, args.engine, args.ranks,
                   args.threads, args.max_cycles, args.skip_iters)
    if err:
        print(f"  FAILED: {err}", file=sys.stderr)
        return 1

    report(serial, par)
    return 0


if __name__ == "__main__":
    sys.exit(main())
