#!/usr/bin/env python3
"""Planck vs PySCF: RHF direct-SCF accuracy and speed across a size ladder.

Runs the SAME geometry/basis through both codes and reports the energy delta and
the wall time. Both are put in directly-comparable configurations:

  * Cartesian basis on both (PySCF mol.cart = True == Planck basis_type cartesian).
    Without this the two use different numbers of basis functions and neither the
    energy nor the timing means anything.
  * Direct SCF on both (PySCF's default; Planck scf_mode direct).
  * Symmetry OFF on both, so the comparison is of the plain direct-SCF Fock
    build. Planck's use_symm routes to a different (full-symmetry) code path.
  * Same convergence tolerance and the same thread count.

Speed caveat, stated up front: PySCF's integral engine is libcint, a mature
hand-tuned C library. Planck's is a from-scratch C++ Obara-Saika/HGP/Rys. A
per-iteration time gap is expected and is a kernel-quality statement, not a
statement about the memory-direct Fock work -- that work changed the MEMORY
profile (nb^4 -> nb^2), which this script also reports.

Usage:
  tests/benchmarks/pyscf_bench.py [--build-dir build] [--basis 6-31g*]
                                  [--threads 4] [--engine os]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ENERGY_RE = re.compile(r"Total Energy\s*:?\s+([-+0-9Ee\.]+)")
# SCF table rows: "<iter> <E> <dE> <rms> <max> <diis> <damp> <time>" — exactly 8
# whitespace-separated fields, the last being the per-iteration wall time. Anchor
# on the full field count rather than a lazy .*? (which silently matched the wrong
# column and reported an 8x-inflated per-iteration time).
ITER_RE = re.compile(
    r"^\s*\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+([0-9.]+)\s*$", re.M)
NBASIS_RE = re.compile(r"Generated\s+\d+\s+Shells and\s+(\d+)\s+contracted functions")


def water_chain(n: int, spacing: float = 3.0) -> list[tuple[str, float, float, float]]:
    """n water molecules along z. Compact enough to be a real SCF, big enough to grow."""
    atoms = []
    for i in range(n):
        z = i * spacing
        atoms.append(("O", 0.000, 0.000, z))
        atoms.append(("H", 0.757, 0.586, z))
        atoms.append(("H", -0.757, 0.586, z))
    return atoms


def planck_input(atoms, basis: str, engine: str) -> str:
    coords = "\n".join(f"{s}  {x:.6f}  {y:.6f}  {z:.6f}" for s, x, y, z in atoms)
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

%begin_coords
{len(atoms)}
0   1
{coords}
%end_coords
"""


def run_planck(exe: Path, atoms, basis: str, engine: str):
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "bench.hfinp"
        inp.write_text(planck_input(atoms, basis, engine))
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        t0 = time.perf_counter()
        proc = subprocess.run([str(exe), str(inp)], capture_output=True, text=True)
        wall = time.perf_counter() - t0
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    out = proc.stdout
    e = ENERGY_RE.findall(out)
    nb = NBASIS_RE.findall(out)
    iters = [float(m) for m in ITER_RE.findall(out)]
    # Median per-iteration time isolates the Fock build from process startup and
    # the one-off basis/1e-integral setup, which PySCF (already in-process) does
    # not pay. Comparing raw wall clock at small nb would just measure exec().
    per_iter = sorted(iters)[len(iters) // 2] if iters else None
    if not e:
        return None, None, wall, None, None
    # ru_maxrss is bytes on macOS, KB on Linux.
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    peak_mb = max(after - before, after) * scale / 1e6
    return float(e[-1]), (int(nb[0]) if nb else None), wall, peak_mb, per_iter


def run_pyscf(atoms, basis: str):
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = [(s, (x, y, z)) for s, x, y, z in atoms]
    mol.basis = basis
    mol.cart = True          # match Planck basis_type cartesian
    mol.symmetry = False     # match Planck use_symm .false.
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.direct_scf = True     # match Planck scf_mode direct
    mf.conv_tol = 1e-9
    mf.init_guess = "hcore"  # match Planck guess hcore
    mf.max_cycle = 100

    t0 = time.perf_counter()
    e = mf.kernel()
    wall = time.perf_counter() - t0
    # PySCF does not expose a per-iteration timer; divide the kernel time by the
    # cycle count. Same quantity Planck's SCF table prints per row.
    n_iter = max(getattr(mf, "cycles", 0) or 1, 1)
    return float(e), int(mol.nao_cart()), wall, wall / n_iter, n_iter, mf.converged


def pyscf_warmup():
    """PySCF lazily imports/compiles on first use; without a warm-up the first
    timed case absorbs that cost and later (larger) cases look faster than
    smaller ones — which is how you know you are timing the import, not the SCF."""
    from pyscf import gto, scf

    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    scf.RHF(mol).kernel()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", type=Path, default=Path("build"))
    ap.add_argument("--basis", default="6-31g*")
    ap.add_argument("--engine", default="os")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    exe = args.build_dir / "hartree-fock"
    if not exe.exists():
        raise SystemExit(f"planck binary not found: {exe}")

    print(f"Planck vs PySCF — RHF direct SCF, basis {args.basis}, engine {args.engine}, "
          f"{args.threads} threads, Cartesian, no symmetry\n")
    pyscf_warmup()  # so the first timed case is not measuring the import

    hdr = (f"{'system':>9} {'nb':>5} {'E_planck':>17} {'E_pyscf':>17} "
           f"{'|dE| (Eh)':>10} {'t/iter_pk':>10} {'t/iter_py':>10} {'slowdown':>9} "
           f"{'RSS_planck':>11}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for n in args.sizes:
        atoms = water_chain(n)
        label = f"{n}xH2O"
        ep, nb_p, tp_wall, rss, pk_iter = run_planck(exe, atoms, args.basis, args.engine)
        try:
            es, nb_s, ts_wall, py_iter, n_it, conv = run_pyscf(atoms, args.basis)
        except Exception as exc:  # pragma: no cover
            print(f"{label:>9}  pyscf failed: {exc}")
            continue

        if ep is None:
            print(f"{label:>10}  planck failed (no Total Energy)")
            continue
        if nb_p is not None and nb_s is not None and nb_p != nb_s:
            print(f"{label:>10}  BASIS MISMATCH planck nb={nb_p} pyscf nb={nb_s} "
                  f"— comparison invalid")
            continue

        de = abs(ep - es)
        slow = (pk_iter / py_iter) if (pk_iter and py_iter) else float("nan")
        print(f"{label:>9} {nb_s:>5} {ep:>17.10f} {es:>17.10f} {de:>10.2e} "
              f"{pk_iter:>9.3f}s {py_iter:>9.3f}s {slow:>8.1f}x {rss:>10.1f}M")
        rows.append(dict(system=label, nbasis=nb_s, e_planck=ep, e_pyscf=es,
                         delta=de, t_iter_planck=pk_iter, t_iter_pyscf=py_iter,
                         slowdown=slow, rss_mb=rss))

    if rows:
        worst = max(r["delta"] for r in rows)
        print(f"\nworst |dE| = {worst:.2e} Eh")
        print("PASS: energies agree" if worst < 1e-7 else "FAIL: energy disagreement")
        Path("/tmp/pyscf_bench.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
