"""References for the RI-MP2 frequency gates (docs/RI_JK_ROUTINE.md step 8).

PySCF has no density-fitted MP2 gradient (2.15.0 raises NotImplementedError),
so the reference gradient is a fourth-order FD of the RI-MP2 ENERGY: dense
RHF/UHF plus density-fitted MP2 correlation, which is what Planck's
mp2_use_ri computes. The auxiliary basis is read from Planck's GBS file. It
shares no code with Planck's analytic RI-MP2 gradient, its per-geometry RI
cache invalidation or its Hessian loop. The Hessian is central differences
of that gradient at Planck's step, then thermo.harmonic_analysis.
  tests/pyscf/.venv/bin/python tests/pyscf/ri_mp2_frequency_references.py
"""
import pathlib
import re
import sys

import numpy as np
from pyscf import mp
from pyscf.gto.basis import parse_gaussian
from pyscf.hessian import thermo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from spherical_mp2_gradient_references import BOHR_ANGSTROM, ROOT, setup  # noqa: E402
from spherical_mp2_gradient_references import mp2 as dense_mp2  # noqa: E402

STEP = 5e-3  # Bohr, Planck's default Hessian step
GSTEP = 1e-3  # Bohr, energy FD step for each gradient
CASES = [
    ROOT / "tests/inputs/regression/post_hf/water_ri_rmp2_freq_sto3g.hfinp",
    ROOT / "tests/inputs/regression/open_shell/water_radical_cation_ri_ump2_freq_sto3g.hfinp",
]


def ri_mp2(symbols, xyz, basis, opts, aux, check_stability=False):
    # Dense SCF (Planck's RI-MP2 fits only the correlation), then DF-MP2.
    mf = dense_mp2(symbols, xyz, basis, opts, check_stability)._scf
    return mp.MP2(mf).density_fit(auxbasis=aux).run()


def run(path):
    symbols, xyz, basis, opts = setup(path)
    aux_gbs = str(ROOT / "basis-sets" /
                  re.search(r"^\s*mp2_ri_basis\s+(\S+)", path.read_text(), re.M).group(1))
    aux = {e: parse_gaussian.load(aux_gbs, e) for e in set(symbols)}
    pt = ri_mp2(symbols, xyz, basis, opts, aux, check_stability=True)
    n3 = 3 * len(symbols)

    def energy(disp):
        x = xyz + disp.reshape(-1, 3) * STEP * BOHR_ANGSTROM
        return ri_mp2(symbols, x, basis, opts, aux).e_tot

    # Same scheme as Planck: central differences, at STEP, of the gradient.
    # The gradient is a fourth-order FD of the energy at GSTEP (error ~1e-9
    # Ha/Bohr), standing in for the analytic RI-MP2 gradient PySCF lacks. A
    # plain second difference of the energy has a different O(h^2) error and
    # sits 0.08 cm-1 off at the same STEP (1480.52 vs 1480.60 on water).
    unit = np.eye(n3)

    def gradient(disp):
        g = np.zeros(n3)
        for k in range(n3):
            e = {m: energy(disp + m * (GSTEP / STEP) * unit[k]) for m in (-2, -1, 1, 2)}
            g[k] = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * GSTEP)
        return g

    e0 = pt.e_tot
    hess = np.zeros((n3, n3))
    for j in range(n3):
        hess[:, j] = (gradient(unit[j]) - gradient(-unit[j])) / (2 * STEP)
    hess = 0.5 * (hess + hess.T)
    n = len(symbols)
    freqs = thermo.harmonic_analysis(pt.mol, hess.reshape(n, 3, n, 3).transpose(0, 2, 1, 3))
    print(f"{path.stem}: E(RI-MP2) = {e0:.10f}")
    print("  frequencies (cm-1): " + "  ".join(f"{f:.2f}" for f in freqs["freq_wavenumber"]))


if __name__ == "__main__":
    for p in CASES:
        run(p)
