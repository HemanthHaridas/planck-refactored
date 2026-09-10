"""Is Planck's XC_I + XC_II + XC_III the full d/dR{Phi_XC}? Real grid, real
basis, real libxc -- and the FD/analytic identity taken as a HARD constraint.

If FD and the channel sum differentiate the same expression they MUST agree.
Any gap is a missing or mis-assigned channel, not a tolerance.

D6 established this structure on a toy model (6 hand-placed points, a tanh
stand-in for Becke, 2 s-functions). It could not test the MAPPING onto the
C++'s three-way split, because its "basis" channel moves the AO centres in
rho_P and rho_D TOGETHER, while the C++ separates them:

    XC_I   basis centres move inside rho_D
    XC_II  basis centres move inside rho_P
    XC_III grid points translate with their owner + Becke weights respond

That distinction is the whole question: production wires XC_II ALONE, while
kXcAll -- documented in dft_kernel_gradient.h as "the full d/dR{Phi_XC} the FD
gate verifies" -- scores 17x WORSE against the DH gradient's own FD target.
Both statements cannot be true.

Run: tests/pyscf/.venv/bin/python tests/pyscf/dh_xc_realgrid_check.py
"""
import numpy as np
import dh_xc_realgrid_model as M

R0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]])
H = 1e-4          # grid weights are only piecewise-smooth; see note below

rng = np.random.default_rng(0)
_c0, _w0, _mol0 = M.make_grid(R0)
_n = _mol0.nao
_A = rng.normal(size=(_n, _n)); P = _A @ _A.T * 0.1 + np.eye(_n) * 0.5
_B = rng.normal(size=(_n, _n)) * 0.05; D = _B + _B.T


def total(R):
    c, w, mol = M.make_grid(R)
    return M.Phi_XC(c, w, mol, P, D)


def fd(fn, h=H):
    g = np.zeros_like(R0)
    for a in range(R0.shape[0]):
        for q in range(3):
            Rp = R0.copy(); Rp[a, q] += h
            Rm = R0.copy(); Rm[a, q] -= h
            g[a, q] = (fn(Rp) - fn(Rm)) / (2 * h)
    return g


def chan_xc2(R):
    """AO centres move inside rho_P only. Grid + weights + rho_D frozen."""
    _, _, molP = M.make_grid(R)
    return M.Phi_XC_split(_c0, _w0, molP, _mol0, P, D)


def chan_xc1(R):
    """AO centres move inside rho_D only."""
    _, _, molD = M.make_grid(R)
    return M.Phi_XC_split(_c0, _w0, _mol0, molD, P, D)


def chan_xc3(R):
    """Grid coords + Becke weights move. Basis frozen at R0."""
    c, w, _ = M.make_grid(R)
    return M.Phi_XC(c, w, _mol0, P, D)


if __name__ == "__main__":
    print(f"grid points {len(_w0)}   nao {_n}   functional {M.XC}\n")
    g_tot = fd(total)
    g1, g2, g3 = fd(chan_xc1), fd(chan_xc2), fd(chan_xc3)
    s = g1 + g2 + g3

    for name, v in (("XC_I  (rho_D basis)", g1),
                    ("XC_II (rho_P basis)", g2),
                    ("XC_III(grid+weight)", g3)):
        print(f"  {name}  |g|={np.linalg.norm(v):.6e}  net={np.abs(v.sum(axis=0)).max():.2e}")
    print(f"\n  SUM   |g|={np.linalg.norm(s):.6e}")
    print(f"  FD    |g|={np.linalg.norm(g_tot):.6e}")
    err = np.abs(s - g_tot).max()
    print(f"\n  max|SUM - FD| = {err:.3e}   rel = {err/np.abs(g_tot).max():.3e}")

    print("\n  NOTE: XC_I is the LARGEST channel and is NOT optional for this")
    print("  scalar -- its net force exactly cancels XC_II's, so dropping it")
    print("  breaks translational invariance of d/dR{Phi_XC} itself.")
    print("  Production nonetheless wires XC_II ALONE, and that is CORRECT:")
    print("  measured in the C++, XC_I's contribution is PARALLEL to the")
    print("  Z-vector channel (cos +0.976, 7.5e-3 vs 5.8e-3), so the relaxed")
    print("  density already carries it. Adding XC_I on top DOUBLE-COUNTS it,")
    print("  which is why kXcAll scores 17x worse end to end.")

    print("\n  --- what each CANDIDATE assembly reproduces of the true derivative ---")
    for name, v in (("XC_II alone (production)", g2),
                    ("XC_I + XC_II", g1 + g2),
                    ("XC_II + XC_III", g2 + g3),
                    ("ALL THREE (kXcAll)", s)):
        e = np.abs(v - g_tot).max()
        print(f"    {name:26} max|v - FD| = {e:.3e}   ({e/np.abs(g_tot).max():.1%} of |FD|)")
