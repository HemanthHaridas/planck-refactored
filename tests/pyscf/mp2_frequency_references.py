"""References for the MP2 frequency gates (docs/SPHERICAL_MP2_GRADIENTS.md).

Same method as Planck: central differences of the analytic MP2 gradient at
h = 5e-3 Bohr (Planck's default hessian step), symmetrized, then
thermo.harmonic_analysis. Needs PySCF >= 2.15: older analytic mp2.Gradients
miss their own energy FD (pyscf/pyscf#3334).
  tests/pyscf/.venv/bin/python tests/pyscf/mp2_frequency_references.py
"""
import pathlib

import numpy as np
from pyscf.hessian import thermo

from spherical_mp2_gradient_references import BOHR_ANGSTROM, ROOT, mp2, setup

STEP = 5e-3  # Bohr
CASES = [
    ROOT / "tests/inputs/regression/open_shell/water_radical_cation_ump2_freq_sto3g.hfinp",
    ROOT / "tests/inputs/regression/spherical/water_rmp2_spherical_freq_631gd.hfinp",
]


def gradient(symbols, xyz, basis, opts):
    return mp2(symbols, xyz, basis, opts).nuc_grad_method().kernel()


def run(path):
    symbols, xyz, basis, opts = setup(path)
    pt = mp2(symbols, xyz, basis, opts, check_stability=True)
    n3 = 3 * len(symbols)
    hess = np.zeros((n3, n3))
    for j in range(n3):
        a, k = divmod(j, 3)
        g = {}
        for s in (1, -1):
            x = xyz.copy()
            x[a, k] += s * STEP * BOHR_ANGSTROM
            g[s] = gradient(symbols, x, basis, opts).ravel()
        hess[:, j] = (g[1] - g[-1]) / (2 * STEP)
    hess = 0.5 * (hess + hess.T)
    n = len(symbols)
    freqs = thermo.harmonic_analysis(pt.mol, hess.reshape(n, 3, n, 3).transpose(0, 2, 1, 3))
    print(f"{pathlib.Path(path).stem}: E(MP2) = {pt.e_tot:.10f}")
    print("  frequencies (cm-1): " + "  ".join(f"{f:.2f}" for f in freqs["freq_wavenumber"]))


if __name__ == "__main__":
    for p in CASES:
        run(p)
