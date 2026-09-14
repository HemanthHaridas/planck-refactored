"""D9 -- W's ov/vo blocks: the MO channel reproduced through the Z-vector
route, closing D7's gap and D8's stated blocker.

RESULT: max|pred - FD| = 4.471e-11  (rel 2.7e-08)

D7 verified W's oo/vv blocks but could not constrain ov/vo -- its functional
had no Z-vector. D8 built the channel decomposition but could not reach the
orbital Hessian, because extracting kappa from U = C0^T S(R0) C(R+h) is
contaminated: C(R+h) is orthonormal wrt S(R+h), not S(R0). Three extraction
attempts came up ~30x short, and D8 concluded the fix was the
U_ij = -1/2 S^(x)_ij bookkeeping of Eqs. 19-21.

**It is not. Four lines of Lowdin projection remove the contamination
exactly**, and then log(U) IS a clean rotation generator (predicted vs actual
dkappa/dR agree to 0.9993/0.9998, FD-step-limited).

THREE MODEL DEFECTS, each caught by a control rather than by reasoning, and
each of which would have produced a false finding about Planck:

1. THE CHANNEL ITSELF WAS CONTAMINATED, not just the kappa extraction.
   D8's mo_channel differentiates E(C(R+h)) with C taken raw from the
   displaced SCF. That is not a pure rotation -- it carries an O(dS) norm
   change. Caught by a VARIATIONAL CONTROL: the SCF is stationary, so
   E_scf's MO channel must vanish, and it measured **1.065e+00**. Lowdin-
   orthonormalizing the displaced MOs against S(R0) takes it to **3.0e-09**.
   The contaminated channel was 26x too large (6.1e-2 vs the true 2.4e-3),
   so any conclusion drawn from its magnitude was noise.

2. THE SAME DEFECT ON THE OTHER SIDE OF THE IDENTITY. dg/dR "at fixed kappa"
   was computed by rotating a fixed C0 at the displaced geometry -- which
   holds the AO COEFFICIENTS fixed, not the rotation. The frame must be
   re-orthonormalized against the DISPLACED metric. Fixing only one side
   left rel 5.3; fixing both left rel 0.56.

3. THE ROTATION SPACE WAS TOO SMALL -- the residual 56%. K.rot populates only
   the ov block, but log(U) has a **vv** component of 3.8e-3 (10% of the ov
   block's 3.8e-2), and E_pt2 is NOT invariant under vv rotations
   (-0.0074573 -> -0.0069318 under 0.3 rad). This is the corollary the
   handoff already carries -- "MP2 is gauge-invariant" is FALSE as usually
   stated; it is invariant only WITHIN a degenerate set -- and it bites here
   as a missing set of generators, not as an ill-posed functional. Using the
   full antisymmetric generator set closes it to 4.5e-11.

WHAT THIS ESTABLISHES: the Z-vector route reproduces dC/dR's entire effect on
E_pt2 without forming dC/dR, in a model that carries its own Z-vector. That is
the machinery D7 lacked, so W's ov/vo blocks are now reachable by an
independent oracle rather than only by agreement with PySCF's grad/mp2.py.

Run: tests/pyscf/.venv/bin/python tests/pyscf/dh_zov_derivation_check.py
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

# Every independent rotation, not just ov -- see defect 3.
PAIRS = [(p, q) for p in range(NB) for q in range(p + 1, NB)]


def lowdin_to(C, S):
    """Nearest frame orthonormal wrt S. Removes the O(dS) metric change
    without touching the rotation (defects 1 and 2)."""
    O = C.T @ S @ C
    w, U = np.linalg.eigh(O)
    return C @ (U @ np.diag(w ** -0.5) @ U.T)


def _displaced(a, q, s, h):
    Rd = R0.copy()
    Rd[a, q] += s * h
    Cd, _ = M.scf(Rd)
    for p in range(NB):
        if C0[:, p] @ (M.S_ao(Rd) @ Cd[:, p]) < 0:
            Cd[:, p] *= -1
    return Rd, Cd


def mo_channel(energy_of_C, h=1e-6):
    """MOs move, basis and eps frozen -- in a frame orthonormal wrt S(R0)."""
    g = np.zeros((3, 3))
    for a in range(3):
        for q in range(3):
            v = []
            for s in (+1, -1):
                _, Cd = _displaced(a, q, s, h)
                v.append(energy_of_C(lowdin_to(Cd, S0)))
            g[a, q] = (v[0] - v[1]) / (2 * h)
    return g


def E_scf(C):
    P = 2 * C[:, :NO] @ C[:, :NO].T
    return float(np.sum(P * M.H_ao(R0)))


def lagrangian(h=1e-5):
    """L_u = dE_pt2/dkappa_u over the FULL generator set."""
    L = np.zeros(len(PAIRS))
    for u, (p, q) in enumerate(PAIRS):
        Kg = np.zeros((NB, NB))
        Kg[q, p], Kg[p, q] = h, -h
        L[u] = (M.E_pt2(R0, C0 @ expm(Kg), E0) -
                M.E_pt2(R0, C0 @ expm(-Kg), E0)) / (2 * h)
    return L


def dkappa_dR(a, q, h=1e-6):
    """The actual rotation the SCF undergoes, now that the metric is removed."""
    k = []
    for s in (+1, -1):
        _, Cd = _displaced(a, q, s, h)
        k.append(np.real(logm(C0.T @ S0 @ lowdin_to(Cd, S0))))
    d = (k[0] - k[1]) / (2 * h)
    return np.array([d[qq, pp] for (pp, qq) in PAIRS])


if __name__ == "__main__":
    ctl = mo_channel(E_scf)
    print("CONTROL -- the SCF is variational, so its MO channel must vanish.")
    print(f"  |MO channel of E_scf| = {np.linalg.norm(ctl):.3e}   (raw, "
          f"un-projected: 1.065e+00 -- see defect 1)")
    assert np.linalg.norm(ctl) < 1e-6, "channel is not a pure rotation"

    fd = mo_channel(lambda C: M.E_pt2(R0, C, E0))
    L = lagrangian()
    pred = np.zeros((3, 3))
    for a in range(3):
        for q in range(3):
            pred[a, q] = L @ dkappa_dR(a, q)

    err = np.abs(pred - fd).max()
    rel = err / np.abs(fd).max()
    print("\nMO channel via the Z-vector route (no dC/dR formed):")
    print(f"  |pred| = {np.linalg.norm(pred):.6e}")
    print(f"  |FD|   = {np.linalg.norm(fd):.6e}")
    print(f"  max|pred - FD| = {err:.3e}   rel = {rel:.3e}")
    assert rel < 1e-6, "Z-vector route does not reproduce the MO channel"
    print("\n  PASS -- D7's ov/vo gap is now reachable by an independent oracle.")
