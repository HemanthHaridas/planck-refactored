"""A6 -- VOID AS WRITTEN. Do not read the convention scores below as a result.

STATUS (2026-09-11): this model CANNOT discriminate the zeta ov/vo conventions,
for a structural reason found after it was written. It is kept because two
findings in it ARE sound (see FINDINGS), and because the rewrite it needs is
specific.

WHY IT IS VOID. Production builds the energy-weighted density from

    corr_relaxed_mo = dm1_corr_mo           # oo block = doo+doo^T
                                            # vv block = dvv+dvv^T
    corr_relaxed_mo.bottomLeftCorner = z    # ov/vo block = the Z-vector
    zeta_ao = W_ref + C (zeta_weights .* corr_relaxed_mo) C^T

(mp2_gradient.cpp:403-405, 285-288). doo/dvv are AMPLITUDE BILINEARS --
`2 t_ijca t_ijcb - t_ijca t_ijbc`, with doo returned negated because
correlation depletes the reference occupations (mp2_rmp2.cpp:89-124).

This model has no amplitudes, so it builds D from the Z-vector ALONE. Measured:
`max|D_vv| = 0.000e+00`. zeta's D7-verified oo/vv weights therefore multiply
NOTHING, while the target (the MO channel) contains their contribution in full.
The comparison is structurally unable to succeed, and indeed every convention
scores cos ~ 0 against the target (-0.0846 / -0.0600 / +0.0003 / 0.0000) at
2-5x the target magnitude. That is the model failing, NOT evidence about zeta.

WHAT A REAL A6 NEEDS: explicit amplitudes. A toy `E_pt2 = sum_ia |v_ia|^2 /
(eps_i - eps_a)` has implied `t_ia = v_ia / (eps_i - eps_a)`, so doo/dvv can be
DERIVED rather than invented, giving non-zero oo/vv blocks with zeta's ov/vo
weight as the only free variable.

FINDINGS THAT ARE SOUND, and were the point of running it:

1. The SCF orbital Hessian is SINGULAR BY CONSTRUCTION, not by conditioning.
   E_scf depends only on the occupied SUBSPACE, so oo and vv rotations leave it
   exactly invariant. Verified at four sizes, not just this toy:

       (NB,NO)     max|dE| oo   max|dE| vv   max|dE| ov
       (3,1)              n/a     0.00e+00     9.53e-01
       (6,2)         2.22e-16     0.00e+00     4.11e-01
       (8,3)         4.44e-16     0.00e+00     6.22e-01
       (10,4)        4.44e-16     0.00e+00     6.62e-01

   E_pt2 is NOT vv-invariant over the same rotations (-0.007457 -> -0.006932 at
   theta = 0 -> 0.3), which is the D9 corollary. Consequence: `H z = -L` must be
   solved on the **ov subspace alone**. Solving it over the full generator set
   divides L by machine noise -- measured cond 6.3e15 and max|z| = 1.03e+13.

2. **10.9% of the Lagrangian lies in that null space** (|L_vv|/|L| = 0.109),
   i.e. in a direction NO orbital response can carry. A4 left 17-33% of the
   *corr defect unexplained after scaling s_zeta. Whether these are the same
   object is UNMEASURED -- it needs a production vv-channel probe, which does
   not exist. Do not assert the correspondence from this toy.

Original docstring follows.

A6: derive zeta's ov/vo blocks, and test Planck's convention against it.

Planck (src/post_hf/mp2_gradient.cpp, build_rmp2_energy_weighted_density):

    zeta_weights(p,q)     = 0.5*(eps_p + eps_q)      # oo, vv
    zeta_weights(v,o)     = eps_occ(i)               # ov, vo  <-- THIS
    zeta_ao = W_ref + C (zeta_weights .* D_mo) C^T

and the ov/vo block of D_mo IS the Z-vector, so zeta's ov/vo content is
`eps_i * z_ai`. D4 flagged this as the one place Planck differs from Eq. 44
(an amplitude-integral W_ia); D7 verified only oo/vv and its own functional
read `diag(D)*eps`, so ov/vo were UNCONSTRAINED -- all three candidate
conventions passed it identically.

A4 then localised the *corr defect to exactly this term: scaling s_zeta
removes 67-83% of it, with a coefficient that DRIFTS with system size
(1.2 / 1.3 / 1.4). That is the signature of a wrong ov/vo weight, since those
blocks scale with the occ-virt gap differently from the diagonal ones.

This model settles which weight is right. It needs three things D7 lacked:
  * a functional that READS the off-diagonal D (D7's read only diag(D)*eps),
  * a genuine Z-vector (so the ov block is determined, not free), and
  * the D9 Lowdin projection, without which the MO channel is contaminated
    by an O(dS) norm change and every conclusion is noise.

Run: tests/pyscf/.venv/bin/python tests/pyscf/dh_zeta_ov_derivation_check.py
"""
import numpy as np
from scipy.linalg import logm, expm
import dh_z_derivation_model as M

R0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.35], [0.0, 1.25, 0.62]])
NO = M.NOCC
NB = len(M.ALPHA)
NV = NB - NO
C0, E0 = M.scf(R0)
S0 = M.S_ao(R0)
PAIRS = [(p, q) for p in range(NB) for q in range(p + 1, NB)]


def lowdin_to(C, S):
    O = C.T @ S @ C
    w, U = np.linalg.eigh(O)
    return C @ (U @ np.diag(w ** -0.5) @ U.T)


def _disp(a, q, s, h):
    Rd = R0.copy()
    Rd[a, q] += s * h
    Cd, _ = M.scf(Rd)
    for p in range(NB):
        if C0[:, p] @ (M.S_ao(Rd) @ Cd[:, p]) < 0:
            Cd[:, p] *= -1
    return Rd, Cd


# ---------------------------------------------------------------------------
# The Z-vector of this model, from the same implicit-function identity D9 used.
# ---------------------------------------------------------------------------
def E_scf(R, C):
    P = 2 * C[:, :NO] @ C[:, :NO].T
    return float(np.sum(P * M.H_ao(R)))


def _g_scf(R, C, h=1e-5):
    g = np.zeros(len(PAIRS))
    for u, (p, q) in enumerate(PAIRS):
        K = np.zeros((NB, NB)); K[q, p], K[p, q] = h, -h
        g[u] = (E_scf(R, C @ expm(K)) - E_scf(R, C @ expm(-K))) / (2 * h)
    return g


def scf_hessian(h=1e-4):
    n = len(PAIRS); H = np.zeros((n, n))
    for u, (pu, qu) in enumerate(PAIRS):
        for v, (pv, qv) in enumerate(PAIRS):
            def E(su, sv):
                K = np.zeros((NB, NB))
                K[qu, pu] += su * h; K[pu, qu] -= su * h
                K[qv, pv] += sv * h; K[pv, qv] -= sv * h
                return E_scf(R0, C0 @ expm(K))
            H[u, v] = (E(1, 1) - E(1, -1) - E(-1, 1) + E(-1, -1)) / (4 * h * h)
    return H


def lagrangian(h=1e-5):
    L = np.zeros(len(PAIRS))
    for u, (p, q) in enumerate(PAIRS):
        K = np.zeros((NB, NB)); K[q, p], K[p, q] = h, -h
        L[u] = (M.E_pt2(R0, C0 @ expm(K), E0) -
                M.E_pt2(R0, C0 @ expm(-K), E0)) / (2 * h)
    return L


OV = [u for u, (p, q) in enumerate(PAIRS) if p < NO <= q]
VV = [u for u, (p, q) in enumerate(PAIRS) if p >= NO]


def zvector():
    """z solves H z = -L on the **ov subspace alone**.

    The full-generator-set solve is WRONG and was this file's first defect:
    E_scf depends only on the occupied SUBSPACE, so oo and vv rotations leave
    it exactly invariant and H is singular BY CONSTRUCTION (measured cond
    6.3e15, max|z| = 1.03e+13 -- machine noise, which swamped all four
    conventions identically and made the run look like a result).
    """
    H, L = scf_hessian(), lagrangian()
    z = np.zeros(len(PAIRS))
    z[OV] = np.linalg.solve(H[np.ix_(OV, OV)], -L[OV])
    return z


# ---------------------------------------------------------------------------
# The quantity zeta must reproduce: the part of dE_pt2/dR carried by the
# ORTHONORMALITY constraint, i.e. contracted against dS/dR.
# ---------------------------------------------------------------------------
def dS_dR(a, q, h=1e-6):
    Rp = R0.copy(); Rp[a, q] += h
    Rm = R0.copy(); Rm[a, q] -= h
    return (M.S_ao(Rp) - M.S_ao(Rm)) / (2 * h)


def mo_channel(h=1e-6):
    """The exact MO channel, D9-clean (Lowdin-projected onto S(R0))."""
    g = np.zeros((3, 3))
    for a in range(3):
        for q in range(3):
            v = []
            for s in (+1, -1):
                _, Cd = _disp(a, q, s, h)
                v.append(M.E_pt2(R0, lowdin_to(Cd, S0), E0))
            g[a, q] = (v[0] - v[1]) / (2 * h)
    return g


def w_channel(zeta_mo):
    """-<W S^(x)> with W = C (zeta .* D_mo) C^T, for a candidate zeta."""
    z = zvector()
    D = np.zeros((NB, NB))
    for u, (p, q) in enumerate(PAIRS):
        D[q, p] = z[u]
        D[p, q] = z[u]
    W = C0 @ (zeta_mo * D) @ C0.T
    g = np.zeros((3, 3))
    for a in range(3):
        for q in range(3):
            g[a, q] = -float(np.sum(W * dS_dR(a, q)))
    return g


def build_zeta(ov_rule):
    """zeta_weights with the diagonal blocks fixed (D7-VERIFIED) and the
    ov/vo blocks set by `ov_rule(i, a)`."""
    z = np.zeros((NB, NB))
    for p in range(NB):
        for q in range(NB):
            z[p, q] = 0.5 * (E0[p] + E0[q])
    for i in range(NO):
        for a in range(NO, NB):
            z[a, i] = z[i, a] = ov_rule(i, a)
    return z


CONVENTIONS = {
    "Planck:  eps_i          ": lambda i, a: E0[i],
    "average: (eps_i+eps_a)/2": lambda i, a: 0.5 * (E0[i] + E0[a]),
    "virtual: eps_a          ": lambda i, a: E0[a],
    "zero                    ": lambda i, a: 0.0,
}

if __name__ == "__main__":
    print("A6 -- which zeta ov/vo convention reproduces the constrained")
    print("     (orthonormality) part of dE_pt2/dR?\n")

    mo = mo_channel()
    print(f"  exact MO channel |g| = {np.linalg.norm(mo):.6e}")
    print(f"  (D9 verified this is reproduced by the Z-vector route to 4.5e-11)\n")

    L = lagrangian()
    print(f"  |L_vv|/|L| = {np.linalg.norm(L[VV])/np.linalg.norm(L):.3f}"
          "   <- lies in the null space; NO orbital response carries it\n")

    print("  " + "!" * 66)
    print("  !! VOID: this model has NO AMPLITUDES, so doo/dvv are absent and")
    print("  !! max|D_vv| = 0. zeta's oo/vv weights multiply nothing while the")
    print("  !! target contains them in full. The scores below DISCRIMINATE")
    print("  !! NOTHING -- do not cite them. See the docstring.")
    print("  " + "!" * 66 + "\n")

    print(f"{'zeta ov/vo convention':>26}{'|w_channel|':>14}{'cos vs MO':>12}")
    f = lambda v: v - v.sum(axis=0) / v.shape[0]
    for name, rule in CONVENTIONS.items():
        w = w_channel(build_zeta(rule))
        wf, mf = f(w), f(mo)
        nw = np.linalg.norm(wf)
        cos = (wf * mf).sum() / (nw * np.linalg.norm(mf)) if nw > 0 else 0.0
        print(f"{name:>26}{np.linalg.norm(w):>14.6e}{cos:>12.4f}")
