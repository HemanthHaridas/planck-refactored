"""Multiply-adds and time in one PySCF ccsd.CCSD update_amps (CCSD residual size scope, R0).

Wraps the Python-visible GEMM entry points and records m*n*k per call.
The vvvv ladder runs in libcc (C), invisible here; it is added from its
formula o(o+1)/2 v^4 (PySCF contracts tau over i >= j).
Usage: OMP_NUM_THREADS=4 tests/pyscf/.venv/bin/python tests/pyscf/ccsd_update_amps_count.py <input.hfinp> <gbs>
"""
import sys
import time
from collections import defaultdict

import numpy
from pyscf import cc, gto, lib, scf
from pyscf.gto.basis import parse_gaussian

inp, gbs = sys.argv[1], sys.argv[2]
lines = open(inp).read().split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
mol = gto.M(atom="\n".join(lines[2:]), unit="Angstrom",
            basis={el: parse_gaussian.load(gbs, el) for el in ("O", "H")},
            cart=False, symmetry=False, max_memory=5000, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = 1e-10
mf.kernel()
mycc = cc.CCSD(mf)
eris = mycc.ao2mo()
_, t1, t2 = mycc.init_amps(eris)
mycc.update_amps(t1, t2, eris)  # warm up

stats = defaultdict(lambda: [0, 0.0])
depth = [0]


def wrap(mod, name):
    orig = getattr(mod, name)

    def counted(*args, **kw):
        outer = depth[0] == 0
        depth[0] += 1
        try:
            out = orig(*args, **kw)
        finally:
            depth[0] -= 1
        if outer:
            try:
                if name == "einsum":
                    subs = args[0].replace(" ", "").split("->")[0].split(",")
                    dims = {}
                    for s, a in zip(subs, args[1:]):
                        for ch, n in zip(s, numpy.shape(a)):
                            dims[ch] = n
                    n = 1
                    for v in dims.values():
                        n *= v
                else:
                    a, b = numpy.asarray(args[0]), numpy.asarray(args[1])
                    if a.ndim == 2 and b.ndim == 2:
                        n = a.shape[0] * a.shape[1] * b.shape[1]
                    else:
                        n = a.size * (b.shape[-1] if b.ndim == 2 else 1)
                stats[f"{mod.__name__}.{name}"][0] += 1
                stats[f"{mod.__name__}.{name}"][1] += n
            except Exception:
                pass
        return out

    setattr(mod, name, counted)


for mod, name in ((lib, "ddot"), (lib, "dot"), (lib, "einsum"), (numpy, "dot"), (numpy, "einsum")):
    wrap(mod, name)

t0 = time.time()
mycc.update_amps(t1, t2, eris)
dt = time.time() - t0
no, nv = t1.shape
gemm = sum(v[1] for v in stats.values())
ladder = no * (no + 1) // 2 * nv ** 4
print(f"o={no} v={nv}: update_amps {dt:.3f} s")
for k, (c, n) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k}: {c} calls, {n:.3e}")
print(f"  python-visible GEMM {gemm:.3e} + vvvv ladder (formula) {ladder:.3e} = {gemm + ladder:.3e}")
