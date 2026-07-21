#!/usr/bin/env python3
"""Scale-proving fixture: strong scaling, memory, and the DFT gap.

This is the fixture HPC Tier 1 was supposed to deliver and did not. Everything
in the regression suite is <=6 atoms (ethylene), so "it scales" and "the
memory-direct build saves memory" are currently CLAIMS, not measurements. This
script produces the numbers that turn them into either.

It answers exactly four questions, and each maps to an open scoping decision:

  Q1  Does the MPI direct-SCF Fock build strong-scale?
      -> ranks vs per-iteration time, on a system big enough to mean something.
      Decides: is Tier 1 done, or is there a load-imbalance/comm problem?

  Q2  Is the memory-direct claim real at scale, for BOTH methods?
      -> peak RSS, HF vs DFT. Both are memory-direct as of #151, so this now
      CHECKS the gap stayed closed rather than measuring an expected gap.
      Decides: did the fused J/K hold, and what nb does it unblock?

  Q3  Where does replicated serial diagonalization become the bottleneck?
      -> diag time as a fraction of per-iteration time vs nb. The HPC scope
      deferred ScaLAPACK "until nb forces it" without knowing where that is.
      Decides: is distributed diag the next tier, or still years away?

  Q4  What is the largest system that actually runs, per method?
      -> the ceiling, and which resource binds (memory vs time).
      Decides: the size of the scale-proving regression fixture to commit.

Every measurement is per-rank where that matters. `ru_maxrss` on the mpirun
child measures the whole process TREE, which would hide the very thing Q2 asks
(what does ONE rank hold?), so each rank self-reports its own high-water mark by
running under `time` -- GNU `time -v` on Linux, BSD `time -l` on macOS. Both are
supported, so Q2 is answerable on the dev box, not only on a cluster.

RESULTS ARE WRITTEN AS JSON (--out). Bring that file back and the remaining
work scopes itself off it.

See tests/benchmarks/CLUSTER_RUNBOOK.md for the full cluster procedure and what
to report. The short version:

    # correctness first -- cheap, catches a broken build before you burn an
    # allocation. Asserts every rank count gives the SAME energy.
    tests/benchmarks/scale_bench.py --verify-only --build-dir build

    # the scaling sweep (Q1/Q3/Q4)
    tests/benchmarks/scale_bench.py --build-dir build \
        --sizes 8,12,16,24,32 --ranks 1,2,4,8,16,32 \
        --basis 6-31g --methods hf,dft --threads 1 --out scale.json

    # the memory question (Q2) on its own -- 1 rank, max_cycles=1
    tests/benchmarks/scale_bench.py --memory-only --basis 6-31g --out memory.json

Needs: `build/hartree-fock`, `build/planck-dft`, and for the MPI arm
`build/planck-mpi` (cmake -DBUILD_MPI=ON) plus a launcher. Arms whose binary is
missing are SKIPPED, not failed, so a serial-only box still produces the Q2
memory curve.

Launcher: `mpirun -n k` by default. Override with $MPIRUN (e.g. "srun") and
$MPIRUN_EXTRA (e.g. "--oversubscribe", which is OPEN MPI ONLY -- Intel MPI /
MPICH / srun reject it and every rank dies, so it is not passed by default).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import contextlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Set from --workdir. None => inputs go to a temp dir and are deleted on exit
# (the default). Set => inputs, probes, and per-run stdout/stderr persist there.
WORKDIR = None

# ---------------------------------------------------------------------------
# Log scrapers. These mirror tests/benchmarks/pyscf_bench.py -- same binaries,
# same log format. ITER_RE anchors on the full 8-field SCF row rather than a
# lazy .*?, which in pyscf_bench had silently matched the wrong column.
# ---------------------------------------------------------------------------
ENERGY_RE = re.compile(r"(?:Total Energy|DFT Energy)\s*:?\s+([-+0-9Ee\.]+)")
# The two binaries announce the basis size differently: hartree-fock prints
# "Generated N Shells and M contracted functions"; planck-dft prints it on the
# grid line as "... M basis functions ...". Match either, or DFT rows come back
# with nb=None and the whole Q2 nb^4 column is unusable.
NBASIS_RE = re.compile(
    r"Generated\s+\d+\s+Shells and\s+(\d+)\s+contracted functions"
    r"|(\d+)\s+basis functions"
)
ITER_RE = re.compile(
    r"^\s*\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+([0-9.]+)\s*$", re.M
)
# Emitted by the patch in `--emit-rank-rss` mode; see rank_rss_probe().
RANK_RSS_RE = re.compile(r"PLANCK_RANK_RSS\s+rank=(\d+)\s+peak_kb=(\d+)")


def water_chain(n: int, spacing: float = 3.0):
    """n waters along z. Same generator as pyscf_bench, so the two are comparable.

    Deliberately NOT a compact cluster: a chain keeps the Schwarz sparsity
    realistic (distant pairs screen out) instead of manufacturing a dense
    worst case that flatters the fused loop.
    """
    atoms = []
    for i in range(n):
        z = i * spacing
        atoms.append(("O", 0.000, 0.000, z))
        atoms.append(("H", 0.757, 0.586, z))
        atoms.append(("H", -0.757, 0.586, z))
    return atoms


def planck_input(atoms, basis: str, engine: str, method: str,
                 max_cycles: int = 100) -> str:
    coords = "\n".join(f"{s}  {x:.6f}  {y:.6f}  {z:.6f}" for s, x, y, z in atoms)
    # DFT: B3LYP is the hybrid case -- it exercises BOTH the Coulomb and the
    # exact-exchange build, which is exactly the nb^4 path DFT_FUSED_JK_SCOPE
    # wants to kill. A pure GGA would only exercise J and would understate the
    # gap.
    dft_block = (
        """
%begin_dft
    exchange     b3lyp
    grid         normal
%end_dft
"""
        if method == "dft"
        else ""
    )
    return f"""%begin_control
    basis       {basis}
    calculation energy
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    scf_mode    direct
    engine      {engine}
    guess       hcore
    use_diis    .true.
    diis_dim    8
    max_cycles  {max_cycles}
    tol_energy  1.0e-9
    tol_density 1.0e-7
%end_scf

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
%end_geom
{dft_block}
%begin_coords
{len(atoms)}
0   1
{coords}
%end_coords
"""


def rank_rss_probe() -> str:
    """A shell wrapper that runs the binary and reports THIS rank's peak RSS.

    Why not resource.getrusage(RUSAGE_CHILDREN): under mpirun that aggregates
    the whole process tree into one number. The Q2 claim is about what ONE rank
    holds, so the aggregate is precisely the number that cannot answer it.

    Each rank runs its binary under `time`, whose max-RSS report is that
    process's own high-water mark. Both flavours are supported:

      GNU  (Linux/clusters):  time -v  -> "Maximum resident set size (kbytes)"
      BSD  (macOS):           time -l  -> "maximum resident set size" in BYTES

    The UNIT DIFFERS (KB vs bytes) -- getting that wrong is a silent 1024x
    error in the headline Q2 number, so the two branches normalise separately.
    Supporting BSD is what makes Q2 answerable on the dev box instead of only
    on the cluster.

    Rank id comes from the MPI runtime's env: OMPI_COMM_WORLD_RANK (Open MPI),
    PMI_RANK (MPICH/Intel MPI), SLURM_PROCID (srun). Defaults to 0 for a serial
    run, which is how run_serial reuses this same probe.
    """
    # Probe by CAPABILITY, not existence. macOS ships a BSD /usr/bin/time that
    # has no -v; `[ -x /usr/bin/time ]` passes there, `time -v` then fails, and
    # THE RANK DIES WITH EXIT 1 -- taking the whole mpirun job with it. (Found
    # the hard way; the fixture's own gate caught it.) `time -v true` is the
    # cheap capability test.
    #
    # No bash process substitution (`2> >(...)`): mpirun may launch this under a
    # plain sh. Redirect stderr to a temp file and post-process.
    # Separate the child's stderr from `time`'s report by FILE DESCRIPTOR, not by
    # text matching. The previous version sent both to one file and tried to
    # grep time's block back out with a pattern that included the bare words
    # Command|User|Exit|Page|... and `^\s*[0-9]+\s+[a-z]`. Those are ordinary
    # English/number shapes: any child stderr line starting with them was
    # silently eaten -- e.g. "Exit code weirdness explained here" and
    # "  12 some numeric-ish line" both vanished, taking the explanation of a
    # failure with them (verified). Here fd 3 carries the child's stderr
    # straight through untouched and time writes to its own file, so no filter
    # is needed and nothing can be misclassified.
    return (
        '#!/bin/bash\n'
        'RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}"\n'
        'TREP="$(mktemp)"\n'
        # `time` writes its report to ITS OWN stderr. Redirecting time's stderr
        # to $TREP would also capture the child's (the child inherits it), so
        # instead: give the child fd 3 (the real stderr) via `2>&3`, and point
        # time's stderr at $TREP. Order matters -- 3>&2 must be set up OUTSIDE
        # the group so fd 3 still refers to the original stderr.
        'exec 3>&2\n'
        'if /usr/bin/time -v true >/dev/null 2>&1; then\n'
        '  /usr/bin/time -v bash -c \'"$@" 2>&3\' _ "$@" 2>"$TREP"; RC=$?\n'
        '  KB="$(grep -i "Maximum resident" "$TREP" | grep -oE "[0-9]+" | tail -1)"\n'
        'elif /usr/bin/time -l true >/dev/null 2>&1; then\n'
        '  /usr/bin/time -l bash -c \'"$@" 2>&3\' _ "$@" 2>"$TREP"; RC=$?\n'
        '  BYTES="$(grep -i "maximum resident" "$TREP" | grep -oE "[0-9]+" | head -1)"\n'
        '  KB=$([ -n "$BYTES" ] && echo $((BYTES / 1024)))\n'  # BSD reports BYTES
        'else\n'
        '  rm -f "$TREP"; exec "$@"\n'
        'fi\n'
        '[ -n "$KB" ] && echo "PLANCK_RANK_RSS rank=$RANK peak_kb=$KB" >&2\n'
        'rm -f "$TREP"\n'
        'exit $RC\n'
    )


def parse_run(out: str, err: str, wall: float):
    """Pull energy, nbasis, per-iteration time, and per-rank RSS out of one run."""
    e = ENERGY_RE.findall(out)
    # NBASIS_RE alternates two patterns, so each match is a 2-tuple with one
    # empty side. Take whichever group fired, first match wins.
    nb = None
    m = NBASIS_RE.search(out)
    if m:
        nb = int(m.group(1) or m.group(2))
    iters = [float(m) for m in ITER_RE.findall(out)]

    # MEDIAN, not mean: the first iteration carries one-off setup (basis, 1e
    # integrals, grid) and the last can be short. The median isolates the
    # steady-state Fock build, which is the thing that is supposed to scale.
    #
    # BUT this is scraped from the SCF table, which under mpirun is printed by
    # ONE rank about ITS OWN iteration timer -- it does not reflect how the work
    # was distributed. Measured on a 16-water HF sweep it was bit-identical
    # (1.371943 s) at 1, 2, 4, 8 and 16 ranks while the true per-iteration time
    # went 15.75 -> 1.62 s (9.7x). Reporting it as "per_iter_s" hid the entire
    # strong-scaling result and made a working MPI build look like it did not
    # scale at all.
    #
    # So per_iter_s is wall_s / n_iters: wall is measured by THIS process around
    # the whole mpirun, so it cannot be fooled by which rank printed what. The
    # scraped value is kept as per_iter_rank_s for reference (it is still the
    # right number for a serial run, and its gap to the wall-derived value is a
    # useful load-imbalance hint).
    per_iter_rank = statistics.median(iters) if iters else None
    n_iters = len(iters)
    per_iter = (wall / n_iters) if n_iters else None

    rank_rss = {}
    for m in RANK_RSS_RE.finditer(err):
        rank_rss[int(m.group(1))] = int(m.group(2)) / 1024.0  # KB -> MB

    return {
        "energy": float(e[-1]) if e else None,
        "nbasis": nb,
        "wall_s": wall,
        "per_iter_s": per_iter,
        # The SCF table's own number. Under mpirun this is one rank's timer and
        # does NOT track rank count -- kept for reference and as a load-imbalance
        # hint (its gap to per_iter_s), never as the scaling metric.
        "per_iter_rank_s": per_iter_rank,
        "n_iters": n_iters,
        # The per-rank peak. max() over ranks is the number that decides whether
        # a system FITS -- one rank blowing up is a failed job, however small the
        # others are.
        "peak_rss_mb_per_rank": (max(rank_rss.values()) if rank_rss else None),
        "rank_rss_mb": rank_rss or None,
    }


def extract_error(stdout: str, stderr: str) -> str:
    """Best explanation of why a run produced no energy.

    Planck writes its diagnostics to STDOUT ("[ERR] DFT Driver Failed : ..."),
    while the RSS probe writes GNU-time output to STDERR. The old
    `(stderr or stdout)` picked stderr whenever the probe said anything at all,
    so every failure reported the probe's timing line and threw the actual
    error away -- which is why a failing cluster DFT run looked silent.
    Prefer real [ERR] lines from either stream; fall back to whatever exists.
    """
    errs = [ln for ln in (stdout + "\n" + stderr).splitlines() if "[ERR]" in ln]
    if errs:
        return "\n".join(errs[-3:])
    return (stdout.strip() or stderr.strip())[-400:]


@contextlib.contextmanager
def case_dir(tag: str):
    """Directory for one run's input + probe.

    Default: a temp dir, deleted on exit (unchanged behaviour). With --workdir
    set (WORKDIR global), files persist under a per-case name so a failed
    cluster run leaves its exact .hfinp behind to rerun by hand.
    """
    if WORKDIR is None:
        with tempfile.TemporaryDirectory() as td:
            yield Path(td), None
    else:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        yield WORKDIR, tag


def write_case(td: Path, tag, atoms, basis, engine, method, max_cycles):
    """Write the .hfinp + probe for one run; return (input_path, probe_path)."""
    stem = tag or "scale"
    inp = td / f"{stem}.hfinp"
    inp.write_text(planck_input(atoms, basis, engine, method, max_cycles))
    probe = td / f"{stem}.probe.sh"
    probe.write_text(rank_rss_probe())
    probe.chmod(0o755)
    return inp, probe


def save_output(td: Path, tag, p):
    """With --workdir, keep the run's stdout+stderr next to its input. This is
    the other half of debugging a cluster failure: the input that failed AND
    what the binary actually said about it."""
    if tag is None:
        return
    (td / f"{tag}.out").write_text(
        f"$ returncode = {p.returncode}\n\n--- stdout ---\n{p.stdout}\n"
        f"--- stderr ---\n{p.stderr}\n")


def run_serial(exe: Path, atoms, basis, engine, method, threads, max_cycles=100):
    """One serial process. Runs under the SAME RSS probe as the MPI arm.

    The probe is not MPI-specific -- it just wraps a process in GNU time. Using
    it here too means the 1-rank peak-RSS number (the Q2 baseline, and the whole
    point of --memory-only) is measured the same way as the MPI numbers, rather
    than being None on the serial path.
    """
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    with case_dir(f"{method}_n{len(atoms)}_r1") as (td, tag):
        inp, probe = write_case(td, tag, atoms, basis, engine, method, max_cycles)
        t0 = time.perf_counter()
        p = subprocess.run(
            [str(probe), str(exe), str(inp)],
            capture_output=True, text=True, env=env
        )
        wall = time.perf_counter() - t0
        save_output(td, tag, p)
    r = parse_run(p.stdout, p.stderr, wall)
    r["returncode"] = p.returncode
    r["ranks"] = 1
    if r["energy"] is None:
        r["error"] = extract_error(p.stdout, p.stderr)
    return r


def have_launcher() -> bool:
    """Is an MPI launcher available? Honours $MPIRUN (e.g. "srun") -- hardcoding
    `which("mpirun")` would skip the whole MPI arm on an srun-only cluster."""
    return shutil.which(os.environ.get("MPIRUN", "mpirun").split()[0]) is not None


def run_mpi(exe: Path, atoms, basis, engine, method, ranks, threads, max_cycles=100):
    """mpirun -n <ranks> planck-mpi, with each rank self-reporting peak RSS."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    with case_dir(f"{method}_n{len(atoms)}_r{ranks}") as (td, tag):
        inp, probe = write_case(td, tag, atoms, basis, engine, method, max_cycles)
        # --oversubscribe is OPEN MPI ONLY. Intel MPI / MPICH / srun reject it
        # as an unknown option and every rank dies -- which on a cluster is the
        # first thing that would break. It only exists so a dev box can request
        # more ranks than it has cores, so make it opt-in via $MPIRUN_EXTRA and
        # let the launcher itself be overridden ($MPIRUN, e.g. "srun").
        launcher = os.environ.get("MPIRUN", "mpirun").split()
        extra = os.environ.get("MPIRUN_EXTRA", "").split()
        cmd = [*launcher, "-n", str(ranks), *extra,
               str(probe), str(exe), str(inp)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t0
        save_output(td, tag, p)
    r = parse_run(p.stdout, p.stderr, wall)
    r["returncode"] = p.returncode
    r["ranks"] = ranks
    if r["energy"] is None:
        r["error"] = extract_error(p.stdout, p.stderr)
    return r


# ---------------------------------------------------------------------------
# Q1 correctness gate. Run this BEFORE burning a cluster allocation.
#
# MPI bugs are logic bugs, not scale bugs: a wrong reduction gives
# plausible-but-wrong numbers, not a crash. If energy(-n k) != energy(-n 1),
# every timing number below is measuring a broken code and is worthless.
# ---------------------------------------------------------------------------
def verify_rank_invariance(build: Path, atoms, basis, engine, methods, ranks, threads, atol):
    print("== Q0: rank-invariance gate (energies must agree bitwise-ish) ==")
    ok = True
    for method in methods:
        serial_exe = build / ("planck-dft" if method == "dft" else "hartree-fock")
        mpi_exe = build / "planck-mpi"
        if not serial_exe.exists():
            print(f"  SKIP {method}: {serial_exe.name} not built")
            continue
        ref = run_serial(serial_exe, atoms, basis, engine, method, threads)
        if ref["energy"] is None:
            print(f"  FAIL {method}: serial run produced no energy\n{ref.get('error', '')}")
            ok = False
            continue
        print(f"  {method:3s} serial          E = {ref['energy']:.10f}  (nb={ref['nbasis']})")
        if not mpi_exe.exists() or not have_launcher():
            print(f"  SKIP {method}: planck-mpi or mpirun absent (serial arms still run)")
            continue
        for n in ranks:
            r = run_mpi(mpi_exe, atoms, basis, engine, method, n, threads)
            if r["energy"] is None:
                print(f"  FAIL {method} -n {n}: no energy\n{r.get('error', '')}")
                ok = False
                continue
            d = abs(r["energy"] - ref["energy"])
            flag = "OK " if d <= atol else "FAIL"
            if d > atol:
                ok = False
            print(f"  {flag} {method:3s} -n {n:<3d}      E = {r['energy']:.10f}  dE = {d:.2e}")
    print("== gate:", "PASS ==" if ok else "FAIL -- do not trust any timing below ==")
    return ok


def run_memory_only(build: Path, sizes, args, methods) -> int:
    """Q2 in isolation: is the DFT memory gap actually closed?

    History: this mode was written BEFORE #151, to prove DFT paid an nb^4
    penalty that HF did not. #151 landed the memory-direct J/K in the DFT
    driver (_compute_2e_j_direct / _compute_2e_k_uhf_direct, including the
    range-separated short-range K), so src/dft/ no longer allocates nb^4 at
    all. The nb^4_MB column is now a COUNTERFACTUAL -- what DFT would have
    cost before #151 -- not a prediction.

    So the expected result inverted: DFT_MB should now stay flat like HF_MB
    and should NOT track nb^4_MB. If DFT still tracks nb^4, that is a
    regression in the fused path, not a confirmation of the old scope.

    A crash is still data: it would mean the memory-direct path is not being
    taken on this input.
    """
    rows = []
    print(f"{'nwat':>5} {'natoms':>7} {'nb':>5} {'nb^4_MB':>10} "
          f"{'HF_MB':>8} {'DFT_MB':>9} {'ratio':>7}")
    print("-" * 60)

    for n in sizes:
        atoms = water_chain(n)
        row = {"nwater": n, "natoms": len(atoms)}
        for m in methods:
            exe = build / ("planck-dft" if m == "dft" else "hartree-fock")
            if not exe.exists():
                continue
            r = run_serial(exe, atoms, args.basis, args.engine, m,
                           args.threads, args.max_cycles)
            row[m] = r
            if row.get("nbasis") is None and r.get("nbasis"):
                row["nbasis"] = r["nbasis"]

        nb = row.get("nbasis")
        # The dense tensor DFT still allocates and HF (fused) does not.
        nb4_mb = (nb**4) * 8 / 1e6 if nb else None
        hf = row.get("hf", {}).get("peak_rss_mb_per_rank")
        dft = row.get("dft", {}).get("peak_rss_mb_per_rank")

        # CAREFUL: a non-zero exit does NOT mean it ran out of memory. In this
        # mode max_cycles=1, so the SCF deliberately does not converge and the
        # binaries exit 1 EVERY time. Using returncode as the ceiling signal
        # would report a false ceiling on every single row (it did).
        #
        # A real allocation failure means the process died before `time` could
        # report a high-water mark -- so the true signal is "no RSS came back".
        def cell(run, mb):
            if mb:
                return f"{mb:.0f}"
            return "DIED" if run else "--"

        ratio = f"{dft / hf:.0f}x" if (hf and dft) else "--"
        print(f"{n:>5} {row['natoms']:>7} {(nb or 0):>5} "
              f"{(f'{nb4_mb:.0f}' if nb4_mb else '--'):>10} "
              f"{cell(row.get('hf'), hf):>8} {cell(row.get('dft'), dft):>9} "
              f"{ratio:>7}")
        rows.append(row)

    print()
    print("=" * 60)
    print("Q2 READ-OUT")
    print("=" * 60)
    print("Both HF and DFT are memory-direct as of #151: neither should allocate")
    print("nb^4, so BOTH columns should stay ~flat and the ratio should stay O(1).")
    print("nb^4_MB is the pre-#151 counterfactual, not a target to track.")
    print()
    measured = [r for r in rows
                if r.get("hf", {}).get("peak_rss_mb_per_rank")
                and r.get("dft", {}).get("peak_rss_mb_per_rank")]
    if measured:
        worst = max(measured,
                    key=lambda r: r["dft"]["peak_rss_mb_per_rank"]
                    / r["hf"]["peak_rss_mb_per_rank"])
        h = worst["hf"]["peak_rss_mb_per_rank"]
        d = worst["dft"]["peak_rss_mb_per_rank"]
        ratio = d / h
        print(f"Worst DFT/HF ratio: nb={worst['nbasis']} -- "
              f"HF {h:.0f} MB vs DFT {d:.0f} MB ({ratio:.1f}x).")
        # The pre-#151 gap was ~2053x. Anything near that means the fused path
        # is not being taken; a small ratio is grid + AO-on-grid, which is
        # real DFT work and is NOT what the fused J/K work removes.
        if ratio > 50:
            print("REGRESSION: that is nb^4-scale. The DFT driver should be calling")
            print("_compute_2e_j_direct / _compute_2e_k_uhf_direct -- check it still is.")
        else:
            print("Memory gap is closed (was ~2053x pre-#151). The residual is the")
            print("replicated grid + AO-on-grid buffers, which is Gap 2, not J/K.")

    # The ceiling: DFT reported no high-water mark (it died before `time` could)
    # while HF at the same size did. NOT returncode -- see the note in cell():
    # max_cycles=1 makes every run exit non-zero, so returncode would flag a
    # false ceiling on every row.
    died = [r for r in rows
            if r.get("dft") and not r["dft"].get("peak_rss_mb_per_rank")
            and r.get("hf", {}).get("peak_rss_mb_per_rank")]
    if died:
        d0 = died[0]
        print(f"\nDFT CEILING: DFT produced no RSS at nwater={d0['nwater']} "
              f"(nb={d0.get('nbasis')}) -- it died before it could be measured,")
        print(f"while HF survived on {d0['hf']['peak_rss_mb_per_rank']:.0f} MB.")
        print("Post-#151 this is a BUG, not the expected result: the DFT J/K is")
        print("memory-direct, so DFT should no longer die where HF survives.")
    if args.out:
        args.out.write_text(json.dumps({"mode": "memory_only", "rows": rows}, indent=2))
        print(f"\nWrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scale-proving fixture: strong scaling, memory, diag ceiling."
    )
    ap.add_argument("--build-dir", default="build", type=Path)
    ap.add_argument("--sizes", default="4,8,16",
                    help="water-chain lengths (molecules, 3 atoms each)")
    ap.add_argument("--ranks", default="1,2,4",
                    help="MPI rank counts for the strong-scaling sweep")
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--engine", default="os")
    ap.add_argument("--methods", default="hf,dft",
                    help="hf and/or dft. dft=B3LYP, the hybrid that exercises J AND K.")
    ap.add_argument("--threads", type=int, default=1,
                    help="OMP threads PER RANK. Keep at 1 for a clean strong-scaling "
                         "curve -- mixing OpenMP and MPI scaling confounds both.")
    ap.add_argument("--atol", type=float, default=1e-8,
                    help="max |dE| between rank counts before the gate fails")
    ap.add_argument("--out", type=Path, help="write results JSON here")
    ap.add_argument("--workdir", type=Path,
                    help="keep the generated .hfinp / probe.sh and each run's "
                         "stdout+stderr here instead of a temp dir that is "
                         "deleted on exit. Required to debug a failing cluster "
                         "run -- otherwise the input that failed is gone before "
                         "you can look at it. Files are named per case, e.g. "
                         "dft_n16_r4.hfinp / dft_n16_r4.out.")
    ap.add_argument("--verify-only", action="store_true",
                    help="run only the rank-invariance gate and exit")
    ap.add_argument("--memory-only", action="store_true",
                    help="Q2 ONLY: 1 rank, a size ladder chosen to make the nb^4 "
                         "tensor hurt, and max_cycles=1. Peak RSS is set on the "
                         "first iteration -- converging the SCF just to measure "
                         "memory wastes hours. Use --sizes to override the ladder.")
    ap.add_argument("--max-cycles", type=int, default=100,
                    help="SCF iteration cap. --memory-only forces 1: the nb^4 "
                         "allocation happens up front, so one iteration is enough "
                         "to catch the high-water mark (and DFT at nb=200+ would "
                         "otherwise take hours to converge).")
    args = ap.parse_args()

    global WORKDIR
    if args.workdir is not None:
        WORKDIR = args.workdir.resolve()
        WORKDIR.mkdir(parents=True, exist_ok=True)
        print(f"[workdir] inputs and per-run logs kept in {WORKDIR}")

    build = args.build_dir.resolve()
    sizes = [int(s) for s in args.sizes.split(",") if s]
    ranks = [int(r) for r in args.ranks.split(",") if r]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # ---------------------------------------------------------------------
    # --memory-only: answer Q2 and nothing else, as cheaply as possible.
    #
    # Q2 does not need MPI (it is a 1-rank question: what does ONE process
    # hold?), and it does not need a converged SCF -- the nb^4 tensor is
    # allocated up front, so the high-water mark is already set by the first
    # iteration. Converging a DFT run at nb=200 to measure a memory peak that
    # was reached in second one would waste hours.
    #
    # The default ladder is picked so the nb^4 tensor actually HURTS. With
    # 6-31g (cartesian) a water chain gives nb = 13*nwater, so:
    #     nwater= 8 -> nb=104 -> nb^4 =  0.9 GB
    #     nwater=12 -> nb=156 -> nb^4 =  4.7 GB
    #     nwater=16 -> nb=208 -> nb^4 = 15.0 GB
    # HF (fused) should stay flat and tiny across all three. DFT should track
    # the nb^4 column and eventually die. Both outcomes are the result.
    # ---------------------------------------------------------------------
    if args.memory_only:
        if args.sizes == ap.get_default("sizes"):
            sizes = [8, 12, 16]
        ranks = [1]
        if args.max_cycles == ap.get_default("max_cycles"):
            args.max_cycles = 1

    print(f"host    : {platform.node()} ({platform.system()})")
    print(f"build   : {build}")
    print(f"basis   : {args.basis}   engine: {args.engine}   threads/rank: {args.threads}")
    if args.memory_only:
        print(f"mode    : MEMORY-ONLY (Q2) -- 1 rank, max_cycles={args.max_cycles}, "
              f"sizes={sizes}")
        print("          Peak RSS is set by the first iteration; the SCF is NOT")
        print("          converged and no energy is expected. That is intended.")
    print()

    if args.memory_only:
        return run_memory_only(build, sizes, args, methods)

    # Gate on the SMALLEST system: correctness is size-independent, so pay the
    # cheapest possible price to catch a broken build.
    gate_atoms = water_chain(min(sizes))
    gate_ok = verify_rank_invariance(
        build, gate_atoms, args.basis, args.engine, methods,
        [r for r in ranks if r > 1] or [2], args.threads, args.atol,
    )
    print()
    if args.verify_only:
        return 0 if gate_ok else 1
    if not gate_ok:
        print("Rank-invariance FAILED. Refusing to report timings for a code that "
              "computes the wrong answer. Fix, then re-run.")
        return 1

    results = {
        "meta": {
            "host": platform.node(),
            "system": platform.system(),
            "basis": args.basis,
            "engine": args.engine,
            "threads_per_rank": args.threads,
            "sizes_nwater": sizes,
            "ranks": ranks,
            "methods": methods,
        },
        "runs": [],
    }

    for method in methods:
        serial_exe = build / ("planck-dft" if method == "dft" else "hartree-fock")
        mpi_exe = build / "planck-mpi"
        have_mpi = mpi_exe.exists() and have_launcher()
        if not serial_exe.exists():
            print(f"SKIP {method}: {serial_exe.name} not built\n")
            continue

        print(f"== {method.upper()} ==")
        hdr = f"{'nwat':>5} {'nb':>5} {'ranks':>6} {'per_iter_s':>11} {'speedup':>8} {'eff':>6} {'peakRSS_MB':>11}"
        print(hdr)
        print("-" * len(hdr))

        for n in sizes:
            atoms = water_chain(n)
            base_t = None
            for r in ranks:
                if r == 1 or not have_mpi:
                    run = run_serial(serial_exe, atoms, args.basis, args.engine,
                                     method, args.threads)
                    run["ranks"] = 1
                    if r != 1 and not have_mpi:
                        continue  # no MPI: only the serial point is meaningful
                else:
                    run = run_mpi(mpi_exe, atoms, args.basis, args.engine, method,
                                  r, args.threads)

                run["method"] = method
                run["nwater"] = n
                run["natoms"] = len(atoms)
                results["runs"].append(run)

                if run["energy"] is None:
                    # A failure IS data: it is the ceiling (Q4). Record and move on.
                    print(f"{n:>5} {'--':>5} {run['ranks']:>6} {'FAILED':>11} "
                          f"{'':>8} {'':>6} {'':>11}   <- ceiling?")
                    # Show WHY. The error text was captured into run["error"] and
                    # then only ever written to the JSON, so a failing cluster run
                    # printed "FAILED" and nothing else -- the one line you need
                    # (returncode, missing basis, libxc abort) was invisible.
                    err = (run.get("error") or "").strip()
                    if err:
                        print(f"      rc={run.get('returncode')} {err.splitlines()[-1][:160]}")
                    run["ceiling"] = True
                    break

                t = run["per_iter_s"]
                if run["ranks"] == 1:
                    base_t = t
                sp = (base_t / t) if (base_t and t) else None
                eff = (sp / run["ranks"]) if sp else None
                rss = run["peak_rss_mb_per_rank"]
                print(f"{n:>5} {run['nbasis']:>5} {run['ranks']:>6} "
                      f"{(f'{t:.3f}' if t else '--'):>11} "
                      f"{(f'{sp:.2f}x' if sp else '--'):>8} "
                      f"{(f'{eff:.0%}' if eff else '--'):>6} "
                      f"{(f'{rss:.0f}' if rss else '--'):>11}")
        print()

    # -----------------------------------------------------------------------
    # The scoping read-out. This is the point of the whole script: turn the
    # table above into the decisions listed in the module docstring.
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("SCOPING READ-OUT")
    print("=" * 72)

    def runs(method=None, ranks=None):
        return [r for r in results["runs"]
                if r["energy"] is not None
                and (method is None or r["method"] == method)
                and (ranks is None or r["ranks"] == ranks)]

    # Q1 -- strong scaling.
    print("\nQ1  Does the MPI Fock build strong-scale?")
    scaled = [r for r in runs() if r["ranks"] == max(ranks) and r["ranks"] > 1]
    if not scaled:
        print("    NO DATA (no MPI arm ran). Build with -DBUILD_MPI=ON on the cluster.")
    else:
        # Below this, per-iteration time is dominated by process startup and the
        # 1e setup, and the efficiency number is noise. Reporting "WEAK" for a
        # 7-function water would be crying wolf.
        MIN_MEANINGFUL_S = 0.05
        for r in scaled:
            base = [b for b in runs(r["method"], 1) if b["nwater"] == r["nwater"]]
            if base and base[0]["per_iter_s"] and r["per_iter_s"]:
                t1 = base[0]["per_iter_s"]
                eff = (t1 / r["per_iter_s"]) / r["ranks"]
                if t1 < MIN_MEANINGFUL_S:
                    verdict = f"too small to judge (serial iter = {t1 * 1000:.0f} ms)"
                elif eff > 0.7:
                    verdict = "GOOD -- Tier 1 holds"
                else:
                    verdict = "WEAK -- investigate load imbalance / Allreduce cost"
                print(f"    {r['method']:3s} nb={r['nbasis']:<5d} @ {r['ranks']} ranks: "
                      f"{eff:.0%} efficiency -> {verdict}")
        print("    For HF, <70% at a MEANINGFUL size => the bra-stripe is imbalanced:")
        print("    the triangular loop hands rank 0 the long rows. Fix = flatten the")
        print("    triangle and stripe the LINEAR index, as _compute_2e already does.")
        print("    For DFT, J/K IS fused and distributed as of #151, so it is judged on")
        print("    the same threshold as HF. A weak DFT number now points at the")
        print("    replicated grid (Gap 2: src/dft/ has no Mpi:: call), not at J/K.")

    # Q2 -- the DFT memory gap. The headline number for DFT_FUSED_JK_SCOPE.md.
    print("\nQ2  Is DFT paying for the missing fused J/K? (peak RSS, 1 rank)")
    hf1 = {r["nwater"]: r for r in runs("hf", 1)}
    dft1 = {r["nwater"]: r for r in runs("dft", 1)}
    common = sorted(set(hf1) & set(dft1))
    if not common or not any(hf1[n]["peak_rss_mb_per_rank"] for n in common):
        print("    NO DATA (need Linux + /usr/bin/time for per-rank RSS).")
    else:
        print(f"    {'nb':>5} {'HF_MB':>8} {'DFT_MB':>8} {'ratio':>7} {'nb^4 (MB)':>11}")
        for n in common:
            h, d = hf1[n], dft1[n]
            hm, dm = h["peak_rss_mb_per_rank"], d["peak_rss_mb_per_rank"]
            if not (hm and dm):
                continue
            nb = h["nbasis"] or 0
            nb4 = (nb**4) * 8 / 1e6  # the dense tensor DFT still allocates
            ratio = dm / hm
            print(f"    {nb:>5} {hm:>8.0f} {dm:>8.0f} {ratio:>6.1f}x {nb4:>11.0f}")
        print("    HF never allocates nb^4 (fused). DFT still does -- TWICE for")
        print("    range-separated. If DFT_MB tracks the nb^4 column, the scope in")
        print("    docs/DFT_FUSED_JK_SCOPE.md is confirmed and its priority is set by")
        print("    the ratio column.")

    # Q3 -- the non-scaling residue, and WHAT it is.
    #
    # CAREFUL: Amdahl gives one number for "everything that did not scale". It
    # cannot, by itself, tell diagonalization apart from an un-distributed Fock
    # build -- and for DFT today those are wildly different diagnoses. So the
    # interpretation is split by method:
    #
    #   HF  -- the Fock build IS distributed (fused loop + bra-stripe), so the
    #          residue really is diag + 1e + Allreduce. Amdahl means what it
    #          says. This is the number that decides ScaLAPACK.
    #
    #   DFT -- the J/K build IS distributed as of #151 (same fused loop), but
    #          the GRID is not: src/dft/ contains no Mpi:: call, so every rank
    #          builds the full grid and evaluates the full XC. That replicated
    #          grid now dominates DFT's residue, so reading it as "diag is
    #          expensive" would still send you to ScaLAPACK for a problem that
    #          is actually Gap 2. DFT's residue is not a diag verdict until the
    #          grid is rank-split.
    #
    # Either way this Amdahl split only bounds the residue; it cannot say which
    # phase owns it. tests/benchmarks/phase_bench.py measures that directly.
    print("\nQ3  What does NOT scale, and is it really diagonalization?")
    print("    Diag is O(nb^3) and is NOT distributed (every rank does it all).")
    print("    But Amdahl lumps ALL non-scaling work together -- see the per-method")
    print("    reading below. For DFT the residue is mostly the replicated grid.")
    print("    To attribute the residue by phase, run phase_bench.py.")
    mx = max(ranks)
    for m in methods:
        rs = sorted(runs(m, mx), key=lambda r: r["nbasis"] or 0)
        if len(rs) >= 2 and all(r["per_iter_s"] for r in rs):
            lo, hi = rs[0], rs[-1]
            # per-iter time that does NOT fall with ranks is the serial residue.
            b_lo = [b for b in runs(m, 1) if b["nwater"] == lo["nwater"]]
            b_hi = [b for b in runs(m, 1) if b["nwater"] == hi["nwater"]]
            if b_lo and b_hi and b_lo[0]["per_iter_s"] and b_hi[0]["per_iter_s"]:
                # Amdahl: serial_frac ~ (T_p - T_1/p) / (T_1 - T_1/p)
                def serial_frac(t1, tp, p):
                    ideal = t1 / p
                    return max(0.0, (tp - ideal) / (t1 - ideal)) if t1 > ideal else None
                f_hi = serial_frac(b_hi[0]["per_iter_s"], hi["per_iter_s"], mx)
                if f_hi is not None:
                    print(f"    {m:3s} nb={hi['nbasis']}: ~{f_hi:.0%} of per-iter time "
                          f"does not scale")
                    if m == "dft":
                        # Do NOT read this as a diag verdict. DFT's J/K is not
                        # distributed at all, so it IS the residue.
                        print("        -> DFT's J/K build is not distributed (dense nb^4),")
                        print("           so this is dominated by the MISSING FUSED J/K,")
                        print("           not by diagonalization. Land Step 3 of")
                        print("           docs/DFT_FUSED_JK_SCOPE.md, then re-measure --")
                        print("           only then does this number mean 'diag'.")
                        print("           (A residue near 100% = DFT is not distributing")
                        print("            at all, which is exactly the predicted symptom.)")
                    elif f_hi > 0.3:
                        print("        -> HF's Fock build IS distributed, so this residue")
                        print("           really is diag + 1e + Allreduce. >30% means")
                        print("           distributed diag (ScaLAPACK/ELPA) is the NEXT")
                        print("           tier, not a someday item.")
                    else:
                        print("        -> HF's Fock build IS distributed, so this residue is")
                        print("           genuinely diag + 1e + reduce, and it is still small.")
                        print("           Replicated diag holds; ScaLAPACK stays deferred.")
        else:
            print(f"    {m:3s}: need >=2 sizes at {mx} ranks to estimate.")

    # Q4 -- the ceiling, and therefore the fixture to commit.
    print("\nQ4  Largest system that ran (=> the regression fixture to commit)")
    for m in methods:
        ok = runs(m)
        if ok:
            big = max(ok, key=lambda r: r["nbasis"] or 0)
            print(f"    {m:3s}: {big['natoms']} atoms, nb={big['nbasis']}, "
                  f"{big['per_iter_s']:.2f} s/iter @ {big['ranks']} rank(s)")
        ceil = [r for r in results["runs"] if r["method"] == m and r.get("ceiling")]
        if ceil:
            print(f"         CEILING hit at {ceil[0]['natoms']} atoms -- "
                  f"that is the binding constraint. See the run's `error`.")
    print("\n    The HPC scope asked for a 30-50 atom fixture and never got one.")
    print("    Pick the largest size above that runs in <~60 s/iter serially and")
    print("    commit it to tests/regression_cases.json as the scale gate.")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}  <- bring this back; the remaining work scopes off it.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
