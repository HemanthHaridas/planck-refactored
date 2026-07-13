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

  Q2  Is the memory-direct claim real at scale, and is DFT paying for its
      absence?
      -> peak RSS, HF (fused, no nb^4) vs DFT (still materializes nb^4, twice
      for range-separated). Decides: how urgent is DFT_FUSED_JK_SCOPE.md, and
      what nb does it unblock?

  Q3  Where does replicated serial diagonalization become the bottleneck?
      -> diag time as a fraction of per-iteration time vs nb. The HPC scope
      deferred ScaLAPACK "until nb forces it" without knowing where that is.
      Decides: is distributed diag the next tier, or still years away?

  Q4  What is the largest system that actually runs, per method?
      -> the ceiling, and which resource binds (memory vs time).
      Decides: the size of the scale-proving regression fixture to commit.

Every measurement is per-rank where that matters. `ru_maxrss` on the mpirun
child measures the whole process tree, which would hide the very thing Q2 is
asking about, so ranks self-report RSS from /proc/self/status (Linux; the
cluster). On macOS that path is absent and per-rank RSS is reported as None --
the dev box can check correctness, not the memory claim.

RESULTS ARE WRITTEN AS JSON (--out). Bring that file back and the remaining
work scopes itself off it.

Usage on a cluster (typical):

    # correctness first -- cheap, catches a broken build before you burn an
    # allocation. Asserts every rank count gives the SAME energy.
    tests/benchmarks/scale_bench.py --verify-only --build-dir build

    # the real run
    tests/benchmarks/scale_bench.py \
        --build-dir build \
        --sizes 4,8,16,24,32 \
        --ranks 1,2,4,8,16 \
        --basis 6-31g \
        --methods hf,dft \
        --out scale.json

Needs: `build/hartree-fock`, `build/planck-dft`, and for the MPI arm
`build/planck-mpi` (cmake -DBUILD_MPI=ON) plus `mpirun`. Arms whose binary is
missing are SKIPPED, not failed, so a serial-only box still produces the Q2/Q3
memory and diagonalization curves.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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


def planck_input(atoms, basis: str, engine: str, method: str) -> str:
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
    max_cycles  100
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

    So each rank runs its binary under `/usr/bin/time -v`, whose "Maximum
    resident set size" is that process's own high-water mark, reported after
    exit. GNU time is universally present on Linux clusters. The rank id comes
    from the MPI runtime's env (OMPI_COMM_WORLD_RANK for Open MPI; PMI_RANK for
    MPICH/Intel MPI -- both are checked). If /usr/bin/time is absent we run the
    binary bare and simply report no per-rank RSS, rather than failing.
    """
    # NOTE: probe for GNU time by CAPABILITY, not by existence. macOS ships a
    # BSD /usr/bin/time that has no -v; testing `[ -x /usr/bin/time ]` passes
    # there, `time -v` then fails, and the RANK DIES WITH EXIT 1 -- taking the
    # whole mpirun job with it. (Found the hard way; the fixture's own gate
    # caught it.) `time -v true` is the cheap capability test.
    #
    # Also: no bash process substitution (`2> >(...)`). mpirun may launch this
    # under a plain sh. Redirect stderr to a temp file and post-process it.
    return (
        '#!/bin/bash\n'
        'RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}"\n'
        'if /usr/bin/time -v true >/dev/null 2>&1; then\n'
        '  ERR="$(mktemp)"\n'
        '  /usr/bin/time -v "$@" 2>"$ERR"\n'
        '  RC=$?\n'
        '  KB="$(grep -i "Maximum resident" "$ERR" | grep -o "[0-9]*$")"\n'
        '  grep -vi "^\\s*[A-Z].*:" "$ERR" >&2 || true\n'
        '  [ -n "$KB" ] && echo "PLANCK_RANK_RSS rank=$RANK peak_kb=$KB" >&2\n'
        '  rm -f "$ERR"\n'
        '  exit $RC\n'
        'else\n'
        '  exec "$@"\n'
        'fi\n'
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
    per_iter = statistics.median(iters) if iters else None

    rank_rss = {}
    for m in RANK_RSS_RE.finditer(err):
        rank_rss[int(m.group(1))] = int(m.group(2)) / 1024.0  # KB -> MB

    return {
        "energy": float(e[-1]) if e else None,
        "nbasis": nb,
        "wall_s": wall,
        "per_iter_s": per_iter,
        "n_iters": len(iters),
        # The per-rank peak. max() over ranks is the number that decides whether
        # a system FITS -- one rank blowing up is a failed job, however small the
        # others are.
        "peak_rss_mb_per_rank": (max(rank_rss.values()) if rank_rss else None),
        "rank_rss_mb": rank_rss or None,
    }


def run_serial(exe: Path, atoms, basis, engine, method, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "scale.hfinp"
        inp.write_text(planck_input(atoms, basis, engine, method))
        t0 = time.perf_counter()
        p = subprocess.run(
            [str(exe), str(inp)], capture_output=True, text=True, env=env
        )
        wall = time.perf_counter() - t0
    r = parse_run(p.stdout, p.stderr, wall)
    r["returncode"] = p.returncode
    if r["energy"] is None:
        r["error"] = (p.stderr or p.stdout)[-400:]
    return r


def run_mpi(exe: Path, atoms, basis, engine, method, ranks, threads):
    """mpirun -n <ranks> planck-mpi, with each rank self-reporting peak RSS."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "scale.hfinp"
        inp.write_text(planck_input(atoms, basis, engine, method))
        probe = Path(td) / "probe.sh"
        probe.write_text(rank_rss_probe())
        probe.chmod(0o755)
        cmd = [
            "mpirun",
            "-n",
            str(ranks),
            "--oversubscribe",  # let rank count exceed cores on a dev box
            str(probe),
            str(exe),
            str(inp),
        ]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t0
    r = parse_run(p.stdout, p.stderr, wall)
    r["returncode"] = p.returncode
    r["ranks"] = ranks
    if r["energy"] is None:
        r["error"] = (p.stderr or p.stdout)[-400:]
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
        if not mpi_exe.exists() or not shutil.which("mpirun"):
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
    ap.add_argument("--verify-only", action="store_true",
                    help="run only the rank-invariance gate and exit")
    args = ap.parse_args()

    build = args.build_dir.resolve()
    sizes = [int(s) for s in args.sizes.split(",") if s]
    ranks = [int(r) for r in args.ranks.split(",") if r]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"host    : {platform.node()} ({platform.system()})")
    print(f"build   : {build}")
    print(f"basis   : {args.basis}   engine: {args.engine}   threads/rank: {args.threads}")
    if platform.system() != "Linux":
        print("NOTE    : not Linux -- per-rank RSS unavailable, Q2 memory numbers "
              "will be None. Correctness (Q0) still valid.")
    print()

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
        have_mpi = mpi_exe.exists() and shutil.which("mpirun") is not None
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
                elif r["method"] == "dft":
                    # Expected, not a surprise: DFT's J/K never enters the fused
                    # loop, so there is nothing to distribute.
                    verdict = ("EXPECTED -- DFT J/K is not distributed at all "
                               "(see DFT_FUSED_JK_SCOPE.md)")
                elif eff > 0.7:
                    verdict = "GOOD -- Tier 1 holds"
                else:
                    verdict = "WEAK -- investigate load imbalance / Allreduce cost"
                print(f"    {r['method']:3s} nb={r['nbasis']:<5d} @ {r['ranks']} ranks: "
                      f"{eff:.0%} efficiency -> {verdict}")
        print("    For HF, <70% at a MEANINGFUL size => the bra-stripe is imbalanced:")
        print("    the triangular loop hands rank 0 the long rows. Fix = flatten the")
        print("    triangle and stripe the LINEAR index, as _compute_2e already does.")
        print("    For DFT, ~50% at 2 ranks (1.00x speedup) is the predicted symptom of")
        print("    the un-fused J/K -- it confirms the scope, it is not a new bug.")

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
    #   DFT -- the J/K build is NOT distributed at all (still the dense nb^4
    #          contraction; see DFT_FUSED_JK_SCOPE.md). It therefore dominates
    #          the residue, and reading it as "diag is expensive" would send you
    #          to build ScaLAPACK to fix a problem that is actually the missing
    #          fused J/K. DFT's residue is NOT a diag verdict until Step 3 of
    #          that scope lands.
    print("\nQ3  What does NOT scale, and is it really diagonalization?")
    print("    Diag is O(nb^3) and is NOT distributed (every rank does it all).")
    print("    But Amdahl lumps ALL non-scaling work together -- see the per-method")
    print("    reading below, because for DFT the residue is mostly the un-fused J/K.")
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
