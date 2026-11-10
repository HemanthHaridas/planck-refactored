"""wPBEh (HSE) seam inventory: libxc vs an independent exchange-hole reference.

Research scope docs/WPBEH_SMOOTH_RESEARCH_SCOPE.md, phase R1.

Reference: the Ernzerhof-Perdew hole J(s, y) with the HSE fits H(s), F(s),
integrated numerically in mpmath against the exact erfc:

    F_x^SR(s, nu) = -(8/9) int_0^inf y J(s, y) erfc(nu y) dy,
    G(s) fixed by the normalization  int_0^inf y^2 J(s, y) dy = -3 pi / 4.

This shares only the hole model (A..E, H, F) with libxc. It does not share
libxc's closed form, its erfc fit (ea1..ea8, eb1), its small-s Taylor branch
(EGscut) or its large-nu branch (wcutoff). The reduced-gradient rescaling is
applied explicitly (Henderson's first scaling, as libxc's s_scaling_2).

libxc is reached through PySCF (libxc 7.0.0 in PySCF 2.13.0). Run with
tests/pyscf/.venv/bin/python.
"""

import mpmath as mp
import numpy as np
from pyscf.dft import libxc

mp.mp.dps = 30

A, B, C, D, E = (mp.mpf(x) for x in
                 ("1.0161144", "-0.37170836", "-0.077215461", "0.57786348", "-0.051955731"))
HA = [mp.mpf(x) for x in ("0.00979681", "0.0410834", "0.187440", "0.00120824", "0.0347188")]
FC1, FC2 = mp.mpf("6.4753871"), mp.mpf("0.47965830")
SMAX = mp.mpf("8.572844")


def H(s):
    return (HA[0] * s**2 + HA[1] * s**4) / (1 + HA[2] * s**4 + HA[3] * s**5 + HA[4] * s**6)


def Fs(s):
    return FC1 * H(s) + FC2


def s_scaling_2(s):
    """Henderson's first scaling as libxc writes it (discontinuous at 1 and 15)."""
    if s < 1:
        return s
    if s > 15:
        return SMAX
    return s - mp.log1p(mp.exp(s - SMAX))


def _pieces(s):
    """J = [lda(y) + (A/y^2 + B + C(1+s^2 F) y^2 + E y^4 (1 + s^2 G)) e^{-D y^2}] e^{-s^2 H y^2}."""
    a = s**2 * H(s)
    cf = C * (1 + s**2 * Fs(s))

    def yJ_noG(y):  # y*J without the G term; the A/y pieces combined to stay finite at y -> 0
        ed = mp.exp(-D * y**2)
        sing = A / y * (ed - 1 / (1 + 4 * A * y**2 / 9)) if y > 0 else mp.mpf(0)
        return (sing + y * (B + cf * y**2 + E * y**4) * ed) * mp.exp(-a * y**2)

    def yJ_G(y):  # coefficient of G in y*J
        return y * E * s**2 * y**4 * mp.exp(-(D + a) * y**2)

    return yJ_noG, yJ_G


def _quad(f):
    return mp.quad(f, [0, 0.5, 2, 6, mp.inf])


def G_of_s(s):
    yJ0, yJG = _pieces(s)
    norm0 = _quad(lambda y: y * yJ0(y))
    normG = _quad(lambda y: y * yJG(y))
    return (-3 * mp.pi / 4 - norm0) / normG


def F_ref(s_bar, nu):
    """Short-range enhancement factor at the *rescaled* gradient s_bar."""
    s_bar = mp.mpf(s_bar)
    yJ0, yJG = _pieces(s_bar)
    G = G_of_s(s_bar) if s_bar > 0 else mp.mpf(0)
    w = (lambda y: mp.erfc(nu * y)) if nu > 0 else (lambda y: 1)
    return -mp.mpf(8) / 9 * _quad(lambda y: (yJ0(y) + G * yJG(y)) * w(y))


# ---------------------------------------------------------------- libxc side
CX = 0.75 * (3 / np.pi) ** (1 / 3)


def libxc_eval(rho, s, omega, deriv=2):
    """Unpolarized GGA_X_WPBEH at density rho, reduced gradient s, range omega."""
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    g = 2 * kf * rho * np.asarray(s, dtype=float)
    r = np.zeros((4, g.size))
    r[0] = rho
    r[1] = g
    exc, vxc, fxc, _ = libxc.eval_xc("GGA_X_WPBEH", r, spin=0, deriv=deriv, omega=omega)
    F = exc / (-CX * rho ** (1 / 3))
    return F, vxc, fxc


def F_libxc(rho, s, nu):
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    return libxc_eval(rho, s, nu * kf, deriv=1)[0]


def jump(rho, s0, omega, eps=1e-10):
    """Relative jumps in e, v_sigma, v2sigma2 across s0 at fixed rho, omega."""
    F, v, f = libxc_eval(rho, [s0 * (1 - eps), s0 * (1 + eps)], omega)
    out = {}
    for name, arr in (("F", F), ("vrho", v[0]), ("vsigma", v[1]),
                      ("v2rho2", f[0]), ("v2sigma2", f[2])):
        out[name] = (arr[1] - arr[0]) / max(abs(arr[0]), 1e-300)
    return out


def jump_nu(rho, s, nu0, eps=1e-10):
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    vals = [libxc_eval(rho, [s], nu0 * (1 + d) * kf) for d in (-eps, eps)]
    out = {}
    for name, get in (("F", lambda t: t[0][0]), ("vrho", lambda t: t[1][0][0]),
                      ("vsigma", lambda t: t[1][1][0]), ("v2rho2", lambda t: t[2][0][0])):
        a, b = get(vals[0]), get(vals[1])
        out[name] = (b - a) / max(abs(a), 1e-300)
    return out


def main():
    print("== reference self-checks ==")
    print("F_ref(0, 0) (must be 1):", mp.nstr(F_ref(0, 0), 15))

    print("\n== libxc vs independent reference, away from seams ==")
    print(f"{'s':>6} {'nu':>7} {'libxc':>18} {'reference':>18} {'rel diff':>10}")
    for s in (0.05, 0.3, 0.7, 1.5, 3.0, 6.0):
        for nu in (0.0, 0.0356, 0.3, 1.0, 5.0):
            fl = F_libxc(1.0, s, nu)[0]
            fr = float(F_ref(s_scaling_2(mp.mpf(s)), mp.mpf(nu)))
            print(f"{s:6.2f} {nu:7.4f} {fl:18.12f} {fr:18.12f} {(fl - fr) / fr:10.2e}")

    print("\n== seams in s (rho, omega fixed; relative jumps, eps = 1e-10) ==")
    for s0, label in ((0.08, "EGscut"), (1.0, "s_scaling_2 s=1"), (15.0, "s_scaling_2 s=15")):
        for rho, omega in ((1.0, 0.11), (0.1, 0.11), (1e-3, 0.11)):
            j = jump(rho, s0, omega)
            print(f"{label:18s} rho={rho:<6g} " + " ".join(f"{k}={v:+.2e}" for k, v in j.items()))

    print("\n== seam in nu (wcutoff = 14) ==")
    for s in (0.5, 2.0, 5.0):
        j = jump_nu(1.0, s, 14.0)
        print(f"wcutoff s={s:<4} " + " ".join(f"{k}={v:+.2e}" for k, v in j.items()))

    print("\n== reference across the same seams (continuous by construction) ==")
    for s0 in (0.08, 1.0, 15.0):
        lo = F_ref(s_scaling_2(mp.mpf(s0) * (1 - mp.mpf("1e-10"))), mp.mpf("0.0356"))
        hi = F_ref(s_scaling_2(mp.mpf(s0) * (1 + mp.mpf("1e-10"))), mp.mpf("0.0356"))
        print(f"s0={s0:<5} ref F jump with s_scaling_2 = {mp.nstr((hi - lo) / lo, 3)}")


MOLECULES = {
    "H2O": "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    "H2O2": "O 0 0.7375 -0.0528; O 0 -0.7375 -0.0528; H 0.8190 0.8170 0.4220; H -0.8190 -0.8170 0.4220",
    "NH3": "N 0 0 0.1173; H 0 0.9377 -0.2738; H 0.8121 -0.4689 -0.2738; H -0.8121 -0.4689 -0.2738",
    "HF": "F 0 0 0; H 0 0 0.917",
    "N2": "N 0 0 0; N 0 0 1.098",
    "CO": "C 0 0 0; O 0 0 1.128",
    "C2H4": "C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; H 0 -0.9289 1.2321; "
            "H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321",
    "C6H6": "; ".join(
        f"C {1.3915 * np.cos(k * np.pi / 3):.6f} {1.3915 * np.sin(k * np.pi / 3):.6f} 0; "
        f"H {2.4715 * np.cos(k * np.pi / 3):.6f} {2.4715 * np.sin(k * np.pi / 3):.6f} 0"
        for k in range(6)),
}


def molecule_census(basis="6-31g*", grid_level=3, omega=0.11, band=1e-3):
    """Grid weight and energy carried near each seam at a converged HSE06 density.

    For the s = 1 seam the relevant quantity for finite differences is the
    energy a point would gain or lose by crossing, |w e_x^LDA rho dF|, summed
    over points within `band` of s = 1 (a point inside the band crosses for a
    geometry step that moves its s by `band`).
    """
    from pyscf import dft, gto

    # relative F jump at each seam (from main(); weakly rho/s dependent)
    seams = (("s=0.08", 0.08, "s", 7e-9), ("s=1", 1.0, "s", 1.2e-4),
             ("s=15", 15.0, "s", 2.4e-5), ("nu=14", 14.0, "nu", 1e-4))
    print(f"\n== molecule census: HSE06/{basis}, grid level {grid_level}, band |x/x0-1| < {band:g} ==")
    print("per seam: points in band / upper bound on the energy jump if all of them crossed (Eh)")
    print(f"{'mol':>5} {'npts':>7} " + " ".join(f"{lab:>17}" for lab, *_ in seams)
          + f" {'n(s>15)':>8} {'E(s>15)':>9} {'E(nu>14)':>9}")
    for name, geom in MOLECULES.items():
        mol = gto.M(atom=geom, basis=basis, verbose=0)
        mf = dft.RKS(mol, xc="HSE06")
        mf.grids.level = grid_level
        mf.conv_tol = 1e-10
        mf.kernel()
        g = mf.grids
        ao = dft.numint.eval_ao(mol, g.coords, deriv=1)
        r = dft.numint.eval_rho(mol, ao, mf.make_rdm1(), xctype="GGA")
        rho = np.maximum(r[0], 1e-300)
        kf = (3 * np.pi**2 * rho) ** (1 / 3)
        s = np.linalg.norm(r[1:4], axis=0) / (2 * kf * rho)
        nu = omega / kf
        exc = libxc.eval_xc("GGA_X_WPBEH", r, spin=0, deriv=0, omega=omega)[0]
        ex = g.weights * rho * (-CX * rho ** (1 / 3))  # LDA exchange energy per point
        F = exc / (-CX * rho ** (1 / 3))
        cols = []
        for _, x0, var, dF in seams:
            x = s if var == "s" else nu
            m = np.abs(x / x0 - 1) < band
            cols.append(f"{m.sum():6d} / {dF * np.abs(ex[m] * F[m]).sum():8.1e}")
        big, tail = s > 15, nu > 14
        print(f"{name:>5} {s.size:7d} " + " ".join(f"{c:>17}" for c in cols)
              + f" {big.sum():8d} {abs((ex[big] * F[big]).sum()):9.1e} {abs((ex[tail] * F[tail]).sum()):9.1e}")


STRANS, SCONST = 8.3, 18.79622316


def _preimage_scaling2(t):
    """s' with libxc's s_scaling_2(s') = t, so libxc evaluates the hole at t."""
    t = np.asarray(t, dtype=float)
    out = t.copy()
    m = t >= 1
    top = 15 - np.log1p(np.exp(15 - float(SMAX)))  # s_scaling_2 just below 15
    tt = np.minimum(t[m], top - 1e-12)
    x = tt.copy()
    for _ in range(60):  # Newton on x - log1p(exp(x - smax)) = t (monotone)
        e = np.exp(x - float(SMAX))
        x -= (x - np.log1p(e) - tt) / (1 - e / (1 + e))
    out[m] = x
    return out


def scaling_heyd(s):
    """Heyd's original scaling (libxc s_scaling_1; CP2K, QE): C0, kink at 8.3."""
    return np.where(s < STRANS, s, float(SMAX) - SCONST / np.maximum(s, STRANS) ** 2)


def scaling_hjss2009(s):
    """Henderson et al., JCP 131, 044108 (2009) eq. 9 (libxc s_scaling_3): C-infinity."""
    return s - (1 - np.exp(-s)) * np.log1p(np.exp(s - float(SMAX)))


def scaling_blend(h):
    """R3(b): libxc's s_scaling_2 outside [1-h, 1+h], a C2 smootherstep blend
    of its two branches inside, so values are untouched outside the window."""
    def f(s):
        lo = s - np.log1p(np.exp(s - float(SMAX)))
        x = np.clip((s - (1 - h)) / (2 * h), 0, 1)
        S = x**3 * (10 - 15 * x + 6 * x**2)
        return np.where(s > 15, float(SMAX), (1 - S) * s + S * lo)
    return f


def _wpbeh_x(rho, sigma, omega, scaling=None):
    """Exchange energy per volume of GGA_X_WPBEH. scaling=None is libxc's own
    s_scaling_2; a callable s -> s_bar is emulated by feeding libxc the
    gradient whose s_scaling_2 image is s_bar (valid for s_bar below
    s_scaling_2(15-) = smax - 1.6e-3; above that it is clamped, which only
    touches the s > ~15 tails)."""
    if scaling is not None:
        kf = (3 * np.pi**2 * rho) ** (1 / 3)
        s = np.sqrt(sigma) / np.maximum(2 * kf * rho, 1e-300)
        sigma = (2 * kf * rho * _preimage_scaling2(scaling(s))) ** 2
    r = np.zeros((4, rho.size))
    r[0], r[1] = rho, np.sqrt(sigma)
    return rho * libxc.eval_xc("GGA_X_WPBEH", r, spin=0, deriv=0, omega=omega)[0]


def _spin_channels(mf):
    """(2 rho_s, 4 |grad rho_s|^2) per spin channel, for exchange spin scaling."""
    from pyscf import dft
    mol, g = mf.mol, mf.grids
    ao = dft.numint.eval_ao(mol, g.coords, deriv=1)
    dm = mf.make_rdm1()
    dms = (dm / 2, dm / 2) if dm.ndim == 2 else dm
    out = []
    for d in dms:
        r = dft.numint.eval_rho(mol, ao, d, xctype="GGA")
        out.append((np.maximum(2 * r[0], 1e-300), 4 * (r[1:4] ** 2).sum(0)))
    return out


def code_variants(basis="6-31g*", grid_level=4, omega=0.11):
    """Non-self-consistent HSE06 exchange differences from libxc's HSE06, at the
    converged libxc HSE06 density. Each variant is (full-range piece, scaling
    of the short-range piece); the full-range piece is the hole at omega = 0
    with the same scaling, or real PBE exchange:
      libxc   : hole, s_scaling_2                (PySCF, Planck, Psi4)
      cp2k    : hole, Heyd's scaling            (CP2K XWPBE, SCALE_X0 1)
      qe      : PBE_x, Heyd's scaling           (QE pbexsr)
      hjss09  : hole, 2009 eq. 9 scaling        (Gaussian HSEH1PBE per the paper;
                                                 its exact erfc integral is
                                                 omitted: <= 1.2e-6 relative)
      blend-h : hole, s_scaling_2 C2-blended over [1-h, 1+h]   (R3(b))
    Atomization-energy differences in kcal/mol."""
    from pyscf import dft, gto
    atoms = {"H": 1, "C": 2, "N": 3, "O": 2, "F": 1}
    variants = {"cp2k": ("hole", scaling_heyd), "qe": ("pbe", scaling_heyd),
                "hjss09": ("hole", scaling_hjss2009),
                "blend-1e-2": ("hole", scaling_blend(1e-2)),
                "blend-1e-3": ("hole", scaling_blend(1e-3))}

    def ex(mf):
        tot = dict.fromkeys(["libxc", *variants], 0.0)
        w = mf.grids.weights
        for rho, sig in _spin_channels(mf):
            pbe = rho * libxc.eval_xc("GGA_X_PBE", np.vstack([rho, np.sqrt(sig), 0 * rho, 0 * rho]),
                                      spin=0, deriv=0)[0]
            tot["libxc"] += 0.5 * w @ (_wpbeh_x(rho, sig, 0.0) - 0.25 * _wpbeh_x(rho, sig, omega))
            for k, (full, sc) in variants.items():
                fr = pbe if full == "pbe" else _wpbeh_x(rho, sig, 0.0, sc)
                tot[k] += 0.5 * w @ (fr - 0.25 * _wpbeh_x(rho, sig, omega, sc))
        return tot

    def run(geom, spin=0):
        mol = gto.M(atom=geom, basis=basis, spin=spin, verbose=0)
        mf = (dft.RKS if spin == 0 else dft.UKS)(mol, xc="HSE06")
        mf.grids.level = grid_level
        mf.conv_tol = 1e-10
        mf.kernel()
        return ex(mf)

    at = {a: run(f"{a} 0 0 0", spin=sp) for a, sp in atoms.items()}
    print(f"\n== HSE06 variants minus libxc HSE06, at the libxc density ({basis}, grid {grid_level}) ==")
    print("Ex in mEh | atomization energy in kcal/mol")
    print(f"{'mol':>5} " + " ".join(f"{k:>11}" for k in variants) + " | "
          + " ".join(f"{k:>11}" for k in variants))
    for name, geom in MOLECULES.items():
        m = run(geom)
        mol = gto.M(atom=geom, basis=basis, verbose=0)
        sym = [mol.atom_symbol(i) for i in range(mol.natm)]
        d = {k: m[k] - m["libxc"] for k in variants}
        dae = {k: -(d[k] - sum(at[s][k] - at[s]["libxc"] for s in sym)) * 627.5095 for k in d}
        print(f"{name:>5} " + " ".join(f"{1e3 * d[k]:11.4f}" for k in variants) + " | "
              + " ".join(f"{dae[k]:11.4f}" for k in variants))


if __name__ == "__main__":
    import sys
    if "--census" in sys.argv:
        molecule_census()
    elif "--variants" in sys.argv:
        code_variants()
    else:
        main()
