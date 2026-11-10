"""R2: energy finite differences vs analytic gradients for HSE06 variants.

docs/WPBEH_SMOOTH_RESEARCH_SCOPE.md, phase R2. Each variant is HSE06 with a
different reduced-gradient scaling s_bar(s) in both wPBEh pieces:

  libxc   libxc's own s_scaling_2 (jumps at s = 1 and s = 15)
  hjss09  Henderson et al. 2009 eq. 9 (C-infinity)
  blend-h s_scaling_2 with its two branches C2-blended over [1-h, 1+h]

The variants run as PySCF custom functionals (define_xc_) whose potentials
are the exact chain rule through s_bar, so each analytic gradient is the
derivative of its own energy. A variant s_bar is fed to libxc as the gradient
whose s_scaling_2 image is s_bar (wpbeh_seams._preimage_scaling2).

Usage (tests/pyscf/.venv/bin/python):
  wpbeh_r2_fd.py --check     native custom == built-in HSE06; pointwise vxc vs FD
  wpbeh_r2_fd.py [mol]       FD table (default H2O2, C1)
"""

import sys

import numpy as np
from pyscf import dft, gto
from pyscf.dft import libxc

import wpbeh_seams as ws

SMAX = float(ws.SMAX)
OMEGA = 0.11
GRID = (150, 590)


def _logistic(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def g_hjss09(s):
    # ponytail: above s = 50, s_bar = smax to 1e-18 and the preimage clamp
    # already zeroes its slope; clipping only avoids exp overflow in the tails.
    s = np.minimum(s, 50.0)
    L = np.log1p(np.exp(s - SMAX))
    em = np.exp(-s)
    return s - (1 - em) * L, 1 - em * L - (1 - em) * _logistic(s - SMAX)


def g_blend(h):
    def g(s):
        s = np.minimum(s, 50.0)  # s > 15 returns smax below; avoid exp overflow
        L = np.log1p(np.exp(s - SMAX))
        x = np.clip((s - (1 - h)) / (2 * h), 0, 1)
        S = x**3 * (10 - 15 * x + 6 * x**2)
        dS = 30 * x**2 * (1 - x) ** 2 / (2 * h)
        val = s - S * L
        der = 1 - dS * L - S * _logistic(s - SMAX)
        big = s > 15
        return np.where(big, SMAX, val), np.where(big, 0.0, der)
    return g


VARIANTS = {"libxc": None, "hjss09": g_hjss09,
            "blend-1e-2": g_blend(1e-2), "blend-1e-3": g_blend(1e-3)}


def _wpbeh_pair(rho, sigma):
    """e = wPBEh(0) - 0.25 wPBEh(omega) per particle, and its rho/sigma derivatives."""
    r = np.zeros((4, rho.size))
    r[0], r[1] = rho, np.sqrt(sigma)
    e0, v0 = libxc.eval_xc("GGA_X_WPBEH", r, spin=0, deriv=1, omega=0.0)[:2]
    ew, vw = libxc.eval_xc("GGA_X_WPBEH", r, spin=0, deriv=1, omega=OMEGA)[:2]
    return e0 - 0.25 * ew, v0[0] - 0.25 * vw[0], v0[1] - 0.25 * vw[1]


def make_eval_xc(g):
    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        assert spin == 0
        rho = np.asarray(rho)
        rho0 = np.maximum(rho[0], 1e-300)
        sigma = (rho[1:4] ** 2).sum(0)
        ec, vc = libxc.eval_xc("GGA_C_PBE", rho[:4], spin=0, deriv=1)[:2]
        if g is None:
            ex, vr, vs = _wpbeh_pair(rho0, sigma)
        else:
            c = 2 * (3 * np.pi**2 * rho0) ** (1 / 3) * rho0  # s = sqrt(sigma)/c
            s = np.sqrt(sigma) / c
            gs, dg = g(s)
            p = ws._preimage_scaling2(gs)
            top = 15 - np.log1p(np.exp(15 - SMAX)) - 1e-12
            d2 = np.where(p < 1, 1.0, 1 - _logistic(p - SMAX))  # s_scaling_2'(p)
            dp = np.where(gs >= top, 0.0, dg / d2)
            sig_p = (c * p) ** 2
            ex, vr, vs = _wpbeh_pair(rho0, sig_p)
            ratio = np.where(s > 1e-12, p * dp / np.maximum(s, 1e-300), 1.0)
            dsig_dsig = ratio  # d sigma'/d sigma = p p'/s
            dsig_drho = 2 * c * (4 / 3) * c / rho0 * p * (p - dp * s)
            vr, vs = vr + vs * dsig_drho, vs * dsig_dsig
        exc = ex + ec
        vrho = vr + vc[0]
        vsigma = vs + vc[1]
        return exc, (vrho, vsigma, None, None), None, None
    return eval_xc


def make_mf(mol, variant, builtin=False):
    mf = dft.RKS(mol)
    if builtin:
        mf.xc = "HSE06"
    else:
        mf = mf.define_xc_(make_eval_xc(VARIANTS[variant]), "GGA", hyb=0, rsh=(OMEGA, 0.0, 0.25))
        # PySCF decides whether to add exact exchange from the xc string
        # (is_hybrid_xc), not from the overridden coefficients; the custom
        # eval_xc ignores the string.
        mf.xc = "HSE06"
    mf.grids.atom_grid = GRID
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.verbose = 0
    return mf


def energy(mol, variant, dm0=None):
    mf = make_mf(mol, variant)
    e = mf.kernel(dm0=dm0)
    assert mf.converged
    return e, mf.make_rdm1()


def gradient(mol, variant):
    mf = make_mf(mol, variant)
    mf.kernel()
    gr = mf.nuc_grad_method()
    gr.grid_response = True
    return gr.kernel(), mf.make_rdm1()


def displaced(mol, atom, xyz, d):
    c = mol.atom_coords().copy()  # Bohr
    c[atom, xyz] += d
    return mol.set_geom_(c, unit="Bohr", inplace=False)


def check():
    mol = gto.M(atom=ws.MOLECULES["H2O2"], basis="6-31g*", verbose=0)
    m1, m2 = make_mf(mol, "libxc", builtin=True), make_mf(mol, "libxc")
    e1, e2 = m1.kernel(), m2.kernel()
    g1 = m1.nuc_grad_method().set(grid_response=True).kernel()
    g2 = m2.nuc_grad_method().set(grid_response=True).kernel()
    print(f"native custom vs built-in HSE06: dE = {e2 - e1:.2e}, max|dG| = {abs(g2 - g1).max():.2e}")
    rng = np.random.default_rng(1)
    rho = 10 ** rng.uniform(-3, 0.5, 400)
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    s = rng.uniform(0.05, 12, 400)
    s = s[np.abs(s - 1) > 0.05]
    rho, kf = rho[: s.size], kf[: s.size]
    for name, g in VARIANTS.items():
        f = make_eval_xc(g)

        def E(r, sg):
            return r * f(None, np.vstack([r, np.sqrt(sg), 0 * r, 0 * r]))[0]
        sig = (2 * kf * rho * s) ** 2
        _, (vr, vs, _, _), _, _ = f(None, np.vstack([rho, np.sqrt(sig), 0 * rho, 0 * rho]))
        hr, hs = 1e-6 * rho, 1e-6 * sig
        fr = (E(rho + hr, sig) - E(rho - hr, sig)) / (2 * hr)
        fs = (E(rho, sig + hs) - E(rho, sig - hs)) / (2 * hs)
        print(f"{name:>11}: max rel |vrho-FD| {np.max(abs(vr - fr) / abs(fr)):.1e}, "
              f"|vsigma-FD| {np.max(abs(vs - fs) / np.maximum(abs(fs), 1e-12)):.1e}")


def fd_table(molname="H2O2", deltas=(1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)):
    """Fourth-order central differences: (8[E(+d)-E(-d)] - [E(+2d)-E(-2d)]) / 12d,
    every Cartesian component."""
    mol = gto.M(atom=ws.MOLECULES[molname], basis="6-31g*", verbose=0)
    comps = [(a, x) for a in range(mol.natm) for x in range(3)]
    print(f"== R2: {molname}/6-31G*, grid {GRID}, max |FD - analytic| over {len(comps)} components (Ha/Bohr) ==")
    print(f"{'variant':>11} " + " ".join(f"{'d=' + format(d, 'g'):>10}" for d in deltas))
    for name in VARIANTS:
        g_an, dm = gradient(mol, name)
        errs = []
        for d in deltas:
            worst = 0.0
            for a, x in comps:
                e = {k: energy(displaced(mol, a, x, k * d), name, dm)[0] for k in (-2, -1, 1, 2)}
                fd = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * d)
                worst = max(worst, abs(fd - g_an[a, x]))
            errs.append(worst)
        print(f"{name:>11} " + " ".join(f"{v:10.2e}" for v in errs), flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        fd_table(*[a for a in sys.argv[1:] if not a.startswith("-")])
