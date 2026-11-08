"""Validate gga_x_wpbeh09 / hyb_gga_xc_hse06_09 (Henderson et al. 2009 eq. 9
HSE06) in a libxc 7.1.2 build patched with libxc-7.1.2-hse06_09.patch.

docs/WPBEH_SMOOTH_RESEARCH_SCOPE.md, "R5 prototype". To reproduce:
  1. git clone --branch 7.1.2 https://gitlab.com/libxc/libxc.git && cd libxc
     git apply <this dir>/libxc-7.1.2-hse06_09.patch
  2. python scripts/sympy2c/compile.py gga_x_wpbeh09 --maxorder 2  (~3 min, SymPy)
  3. cmake -B build -DBUILD_SHARED_LIBS=ON -DENABLE_PYTHON=ON -DMAXORDER=3 \
       -DCMAKE_INSTALL_PREFIX=<prefix> && cmake --build build -j4 && cmake --install build
  4. PYTHONPATH=<prefix>/lib DYLD_LIBRARY_PATH=<prefix>/lib \
       python wpbeh09_libxc_validate.py [points|molecule]
     with a Python that has PySCF >= 2.14 (libxc 7.1.2), numpy and mpmath.
"""
import os
import sys

import numpy as np
import pylibxc

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import wpbeh_seams as ws  # noqa: E402  (mpmath reference, preimage helper)

SMAX = float(ws.SMAX)
CX = 0.75 * (3 / np.pi) ** (1 / 3)


def s3(s):
    """Eq. 9, overflow-safe form (same algebra as the generated code)."""
    s = np.asarray(s, float)
    hi, lo = np.maximum(s, SMAX), np.minimum(s, SMAX)
    big = SMAX - np.log1p(np.exp(SMAX - hi)) + np.exp(-hi) * (hi - SMAX + np.log1p(np.exp(SMAX - hi)))
    small = lo + np.expm1(-lo) * np.log1p(np.exp(lo - SMAX))
    return np.where(s > SMAX, big, small)


def func(name, omega=None):
    f = pylibxc.LibXCFunctional(name, "unpolarized")
    if omega is not None:
        f.set_ext_params([omega])
    return f


def ev(f, rho, sigma, order=2):
    out = f.compute({"rho": rho, "sigma": sigma}, do_exc=True, do_vxc=order >= 1, do_fxc=order >= 2)
    return {k: np.asarray(v).ravel() for k, v in out.items()}


def grid(rho, s):
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    return (2 * kf * rho * s) ** 2


def points():
    import mpmath as mp
    print("== 1. omega = 0: wpbeh09 vs mpmath hole reference at s_bar = eq.9(s) ==")
    f0 = func("gga_x_wpbeh09", 0.0)
    worst = 0.0
    for s in (0.05, 0.3, 0.9, 1.0, 1.1, 3.0, 8.0, 8.57, 9.0, 14.0, 16.0, 30.0):
        F = ev(f0, np.array([1.0]), grid(1.0, np.array([s])), 0)["zk"][0] / (-CX)
        Fr = float(ws.F_ref(mp.mpf(float(s3(s))), 0))
        worst = max(worst, abs(F - Fr) / abs(Fr))
        print(f"  s={s:6.2f} s_bar={float(s3(s)):.12f} libxc={F:.15f} ref={Fr:.15f} rel={(F - Fr) / Fr:+.1e}")
    print(f"  worst rel {worst:.1e}")

    print("\n== 2. omega = 0.11: wpbeh09(s) vs stock wpbeh at the preimage of eq.9(s), same library ==")
    rng = np.random.default_rng(3)
    rho = 10 ** rng.uniform(-5, 2, 4000)
    s = 10 ** rng.uniform(-3, np.log10(14.0), 4000)  # preimage exact below s_bar(15-)
    a = ev(func("gga_x_wpbeh09", 0.11), rho, grid(rho, s), 0)["zk"]
    sp_ = ws._preimage_scaling2(s3(s))
    b = ev(func("gga_x_wpbeh", 0.11), rho, grid(rho, sp_), 0)["zk"]
    print(f"  max rel |exc diff| = {np.max(np.abs(a - b) / np.abs(b)):.1e} over {s.size} points")

    print("\n== 3a. continuity across the old seams and the new branch point (rho = 1, omega = 0.11) ==")
    f = func("gga_x_wpbeh09", 0.11)
    for s0 in (0.08, 0.13, 1.0, SMAX, 15.0):
        r = ev(f, np.array([1.0, 1.0]), grid(1.0, np.array([s0 * (1 - 1e-10), s0 * (1 + 1e-10)])))
        jumps = {k: (r[k][1] - r[k][0]) / max(abs(r[k][0]), 1e-300) for k in ("zk", "vrho", "vsigma", "v2rho2", "v2sigma2")}
        print(f"  s0={s0:8.5f} " + " ".join(f"{k}={v:+.1e}" for k, v in jumps.items()))
    g = func("gga_x_wpbeh", 0.11)
    r = ev(g, np.array([1.0, 1.0]), grid(1.0, np.array([1 - 1e-10, 1 + 1e-10])), 0)["zk"]
    print(f"  (stock wpbeh at s=1 for contrast: zk jump {(r[1] - r[0]) / abs(r[0]):+.1e})")

    print("\n== 3b. analytic derivatives vs central FD of exc (omega = 0.11) ==")
    rho = 10 ** rng.uniform(-3, 1.5, 500)
    s = 10 ** rng.uniform(-1.5, 1.4, 500)
    sig = grid(rho, s)
    r = ev(f, rho, sig)
    E = lambda rr, ss: rr * ev(f, rr, ss, 0)["zk"]  # noqa: E731
    hr, hs = 1e-5 * rho, 1e-5 * sig
    fr = (E(rho + hr, sig) - E(rho - hr, sig)) / (2 * hr)
    fs = (E(rho, sig + hs) - E(rho, sig - hs)) / (2 * hs)
    Vs = lambda ss: ev(f, rho, ss, 1)["vsigma"]  # noqa: E731
    f2 = (Vs(sig + hs) - Vs(sig - hs)) / (2 * hs)
    print(f"  vrho     max rel {np.max(np.abs(r['vrho'] - fr) / np.abs(fr)):.1e}")
    print(f"  vsigma   max rel {np.max(np.abs(r['vsigma'] - fs) / np.abs(fs)):.1e}")
    print(f"  v2sigma2 max rel {np.median(np.abs(r['v2sigma2'] - f2) / np.abs(f2)):.1e} (median)")

    print("\n== 3c. tails: huge s must stay finite and approach the s_bar = smax limit ==")
    big = np.array([20.0, 1e2, 1e4, 1e8, 1e30, 1e100])
    rr = ev(f, np.ones(big.size), grid(1.0, big))
    lim = ev(f, np.array([1.0]), grid(1.0, np.array([1e200])), 0)["zk"][0]
    for sv, z, vs in zip(big, rr["zk"], rr["vsigma"]):
        print(f"  s={sv:8.1e} zk={z:.15f} (limit {lim:.15f}) vsigma={vs:.2e} finite={np.isfinite(z) and np.isfinite(vs)}")


def molecule():
    from pyscf import dft, gto
    import wpbeh_r2_fd as r2

    hyb = pylibxc.LibXCFunctional("hyb_gga_xc_hse06_09", "unpolarized")
    print("cam (omega, alpha, beta):", hyb.get_cam_coef())

    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        rho = np.asarray(rho)
        r = hyb.compute({"rho": rho[0], "sigma": (rho[1:4] ** 2).sum(0)}, do_exc=True, do_vxc=True)
        return (np.asarray(r["zk"]).ravel(),
                (np.asarray(r["vrho"]).ravel(), np.asarray(r["vsigma"]).ravel(), None, None), None, None)

    def mf_lib(mol):
        mf = dft.RKS(mol).define_xc_(eval_xc, "GGA", hyb=0, rsh=(0.11, 0.0, 0.25))
        mf.xc = "HSE06"
        mf.grids.atom_grid = r2.GRID
        mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-12, 1e-8, 0
        return mf

    mol = gto.M(atom=ws.MOLECULES["H2O2"], basis="6-31g*", verbose=0)
    a = mf_lib(mol)
    ea = a.kernel()
    ga = a.nuc_grad_method().set(grid_response=True).kernel()
    b = r2.make_mf(mol, "hjss09")
    eb = b.kernel()
    gb = b.nuc_grad_method().set(grid_response=True).kernel()
    print(f"hse06_09 (new libxc) vs emulation: dE = {ea - eb:+.2e} Eh, max|dG| = {abs(ga - gb).max():.2e} Ha/Bohr")

    dm = a.make_rdm1()
    for d in (1e-4, 5e-4, 2e-3):
        worst = 0.0
        for at in range(mol.natm):
            for x in range(3):
                e = {}
                for k in (-2, -1, 1, 2):
                    m = mf_lib(r2.displaced(mol, at, x, k * d))
                    e[k] = m.kernel(dm0=dm)
                fd = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * d)
                worst = max(worst, abs(fd - ga[at, x]))
        print(f"FD vs analytic, hse06_09 from the new libxc, d={d:g}: max {worst:.2e} Ha/Bohr", flush=True)


if __name__ == "__main__":
    molecule() if "molecule" in sys.argv else points()
