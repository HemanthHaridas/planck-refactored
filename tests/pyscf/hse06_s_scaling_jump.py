"""Why HSE06 energy finite differences miss the analytic gradient.

libxc's HSE06 short-range PBE exchange rescales the reduced gradient with
s_scaling_2 (maple/scaling.mpl): s for s < 1, s - ln(1 + exp(s - smax)) for
s >= 1, so the scaled s jumps by ~5.15e-4 at s = 1 and the energy density jumps
with it (~1e-4 relative). A grid point whose s crosses 1 between two displaced
geometries adds its weighted jump to E(+) - E(-), which no analytic gradient
sees. This script displaces atom 1 of C1 H2O2 by +-5e-4 Bohr in y, finds the
points that cross (unsorted grids keep point i the same point), and removes
their jump energy from the central difference.

Measured 2026-11-06 (PySCF 2.13.0, libxc 7.0.0; Planck's libxc 4bd0e1e gives
the same energies to 1e-12): 7 of 354000 points cross, carrying 2.796e-8 Eh;
FD -0.0057576787 -> -0.0057856406 with the jumps removed, analytic
-0.0057855963 (Planck and PySCF agree to 1.2e-10).
Run: tests/pyscf/.venv/bin/python tests/pyscf/hse06_s_scaling_jump.py
"""
import numpy as np
from pyscf import dft, gto
from pyscf.dft import libxc
from pyscf.gto.basis import parse_gaussian
B = 0.529177210903
ROOT = "/Users/hemanthharidas/Desktop/codes/planck-refactored"
atoms = [["O", 0.000000, 0.710000, -0.050000], ["O", 0.020000, -0.720000, 0.040000],
         ["H", 0.830000, 0.930000, 0.420000], ["H", -0.910000, -0.880000, 0.380000]]
gbs = ROOT + "/basis-sets/6-31g*"
delta = 5e-4  # Bohr, atom 0 axis y

def state(sign):
    a = [list(x) for x in atoms]; a[0][2] += sign * delta * B
    mol = gto.M(atom=[[x[0], x[1:]] for x in a], unit=B, cart=False, verbose=0,
                basis={e: parse_gaussian.load(gbs, e) for e in ("O", "H")})
    mf = dft.RKS(mol, xc="hse06"); mf.grids.atom_grid, mf.grids.prune = (150, 590), None
    mf.grids.build(sort_grids=False)
    mf.init_guess, mf.conv_tol, mf.conv_tol_grad = "1e", 1e-13, 1e-9
    e = mf.kernel()
    ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
    rho = dft.numint.eval_rho(mol, ao, mf.make_rdm1(), xctype="GGA")
    g = np.linalg.norm(rho[1:4], axis=0)
    s = g / (2 * (3 * np.pi**2 * rho[0]) ** (1 / 3) * np.maximum(rho[0], 1e-300))
    return e, rho, s, mf.grids.weights

ep, rp, sp, wp = state(+1)
em, rm, sm, wm = state(-1)
assert sp.size == sm.size
cross = np.where((sp >= 1) != (sm >= 1))[0]

def jump(r0):  # e(s=1+) - e(s=1-) at density r0
    kf = (3 * np.pi**2 * r0) ** (1 / 3)
    r = np.zeros((4, 2)); r[0] = r0; r[1] = 2 * kf * r0 * np.array([1 - 1e-9, 1 + 1e-9])
    exc = libxc.eval_xc("hse06", r, deriv=0)[0]
    return r0 * (exc[1] - exc[0])

# Going from - to +: a point moving from s<1 to s>=1 picks up +jump, the reverse -jump.
dE = sum(0.5 * (wp[i] + wm[i]) * jump(0.5 * (rp[0, i] + rm[0, i])) * (1 if sp[i] >= 1 else -1) for i in cross)
fd = (ep - em) / (2 * delta)
print(f"points crossing s=1: {cross.size} of {sp.size}")
print(f"E+ - E-                     = {ep - em:+.4e}")
print(f"jump energy from crossings  = {dE:+.4e}")
print(f"FD                          = {fd:+.10f}")
print(f"FD with the jumps removed   = {(ep - em - dE) / (2 * delta):+.10f}   (analytic -0.0057855963)")
