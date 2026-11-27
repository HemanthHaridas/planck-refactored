"""PySCF RCCSD(T) reference and timing for a ladder rung (docs/CCSD_T_SIZE_SCOPE.md, step 7).

Usage: OMP_NUM_THREADS=4 tests/pyscf/.venv/bin/python tests/pyscf/ccsd_t_ladder_reference.py <input.hfinp> <basis-sets/GBS file>

Reads the geometry from the Planck input; uses Planck's own GBS file through
parse_gaussian (O and H only), spherical (cart=False), tight SCF and CCSD
tolerances.
"""
import sys
import time

from pyscf import cc, gto, scf
from pyscf.cc import ccsd_t
from pyscf.gto.basis import parse_gaussian

inp, gbs = sys.argv[1], sys.argv[2]
lines = open(inp).read().split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
atom = "\n".join(lines[2:])
basis = {el: parse_gaussian.load(gbs, el) for el in ("O", "H")}
mol = gto.M(
    atom=atom,
    unit="Angstrom",
    basis=basis,
    cart=False,
    symmetry=False,
    max_memory=5000,
)
t0 = time.time()
mf = scf.RHF(mol)
mf.conv_tol = 1e-11
mf.conv_tol_grad = 1e-8
mf.kernel()
t1 = time.time()
mycc = cc.RCCSD(mf)
mycc.conv_tol = 1e-10
mycc.conv_tol_normt = 1e-8
mycc.kernel()
t2 = time.time()
e_t = ccsd_t.kernel(mycc, mycc.ao2mo())
t3 = time.time()
print(f"nao {mol.nao}  E_SCF {mf.e_tot:.10f}")
print(f"E_corr(CCSD) {mycc.e_corr:.10f}  E(T) {e_t:.10f}")
print(f"Total CCSD(T) {mf.e_tot + mycc.e_corr + e_t:.10f}")
print(f"time SCF {t1 - t0:.1f} s  CCSD {t2 - t1:.1f} s  (T) {t3 - t2:.1f} s")
