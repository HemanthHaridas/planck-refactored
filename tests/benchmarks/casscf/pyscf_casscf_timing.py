"""PySCF side of the CASSCF timing table in docs/CASSCF_PERFORMANCE.md.

Water CAS(4,4), the geometry of the water_cas44_* PySCF gates, Cartesian basis.
Run with the repo's venv:

    OMP_NUM_THREADS=4 tests/pyscf/.venv/bin/python \
        tests/benchmarks/casscf/pyscf_casscf_timing.py [--c1] [basis ...]
"""

import argparse
import os
import re
import tempfile
import time

from pyscf import gto, mcscf, scf

WATER = "O 0 0 0.117176; H 0 0.757005 -0.468704; H 0 -0.757005 -0.468704"


def run(basis, symmetry):
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "pyscf.log")
        mol = gto.M(atom=WATER, basis=basis, cart=True, symmetry=symmetry,
                    verbose=4, output=log)
        mf = scf.RHF(mol).run()
        t0 = time.perf_counter()
        e = mcscf.CASSCF(mf, 4, 4).kernel()[0]
        dt = time.perf_counter() - t0
        m = re.search(r"converged in\s+(\d+) macro \((\d+) JK", open(log).read())
    macros, jk = (m.group(1), m.group(2)) if m else ("?", "?")
    return mol.nao, e, dt, macros, jk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("basis", nargs="*", default=["6-31g", "cc-pvdz", "cc-pvtz"])
    ap.add_argument("--c1", action="store_true", help="symmetry off")
    args = ap.parse_args()
    print(f"{'basis':10} {'AOs':>4} {'E_CASSCF':>16} {'time/s':>8} {'macros':>6} {'JK':>5}")
    for b in args.basis:
        nao, e, dt, macros, jk = run(b, not args.c1)
        print(f"{b:10} {nao:4d} {e:16.10f} {dt:8.2f} {macros:>6} {jk:>5}")


if __name__ == "__main__":
    main()
