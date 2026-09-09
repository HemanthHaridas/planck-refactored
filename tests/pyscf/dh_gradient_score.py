"""Score a candidate gradient term against the DH-gradient FD targets.

The instrument for the N3.5.7.11+ hunt for the remaining ~31% of the
double-hybrid PT2 gradient. See docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md.

    import dh_gradient_score as S
    S.score("my candidate", {1: contrib_fixture1, 2: contrib_fixture2})

where contrib is the candidate's CONTRIBUTION (analytic-with minus
analytic-without) as a 4x3 array per fixture.

Two properties make this trustworthy, both verified (N3.5.7.12):
  * POSITIVE control -- it recovers the known answer XC_II at cos 0.946/0.960
    and scale 1.078/1.053 across the two fixtures.
  * NEGATIVE control -- random vectors reach cos 0.49 on a single fixture but
    are rejected by the cross-fixture scale spread (175% / 128% / 30%).

The single-fixture cos is NOT decisive: a random vector can score 0.5.
Requiring the fitted coefficient to agree on two independent geometries is
what does the work; it killed three candidates in N3.5.7.11.

CAVEAT: random control #3 landed at 30% spread against a 25% threshold, so the
margin is thin. Treat spread in the 20-30% band as unproven, not as a pass.

Run directly to execute the positive control.
"""
import numpy as np

# Exact FD targets for *corr (PySCF, 0.27*dE_corr/dR), from
# tests/pyscf/h2o2{,b}_b2plyp_dh_gradient_fd.py
FD_TOT = {
    1: np.array([[0.0282807311, -0.0040897949, 0.0104669488],
                 [-0.1369469959, -0.0088664716, 0.0603711531],
                 [-0.0239522730, -0.0078739732, -0.0135072430],
                 [0.1326185388, 0.0208302504, -0.0573308823]]),
    2: np.array([[0.0946246944, -0.0194067318, 0.0547109003],
                 [-0.2981357716, -0.0109979096, 0.1654090941],
                 [-0.0780060422, -0.0123248611, -0.0517833256],
                 [0.2815171205, 0.0427295203, -0.1683367052]]),
}

# Planck analytic totals WITH XC_II wired (the current production path).
PLANCK = {
    1: np.array([[0.02822906, -0.00424534, 0.01036380],
                 [-0.13691422, -0.00861179, 0.06031316],
                 [-0.02434094, -0.00777107, -0.01375307],
                 [0.13286442, 0.02068303, -0.05743995]]),
    2: np.array([[0.09455721, -0.01946671, 0.05460470],
                 [-0.29809868, -0.01083147, 0.16533484],
                 [-0.07827494, -0.01228345, -0.05199666],
                 [0.28160731, 0.04254779, -0.16838794]]),
}

REMAINDER = {k: FD_TOT[k] - PLANCK[k] for k in FD_TOT}


def _free(v):
    """Translation-free part: the physically meaningful subspace, since the
    target has zero net force."""
    return v - v.sum(axis=0) / v.shape[0]


def score(name, contrib):
    """contrib: {1: 4x3, 2: 4x3} candidate contribution per fixture."""
    rows = []
    for k in sorted(contrib):
        c, R = _free(np.asarray(contrib[k])), _free(REMAINDER[k])
        nc = np.linalg.norm(c)
        if nc == 0:
            rows.append((k, 0.0, 0.0, np.abs(REMAINDER[k]).max()))
            continue
        cos = (c * R).sum() / (nc * np.linalg.norm(R))
        a = (c * R).sum() / (c * c).sum()
        resid = np.abs(REMAINDER[k] - a * np.asarray(contrib[k])).max()
        rows.append((k, cos, a, resid))
    print(f"\n=== {name} ===")
    print(f"{'fix':>4}{'cos':>10}{'scale':>10}{'resid@best':>13}{'baseline':>12}")
    for k, cos, a, resid in rows:
        print(f"{k:>4}{cos:>10.4f}{a:>10.4f}{resid:>13.3e}{np.abs(REMAINDER[k]).max():>12.3e}")
    if len(rows) == 2:
        a1, a2 = rows[0][2], rows[1][2]
        spread = abs(a1 - a2) / max(abs(a1), abs(a2), 1e-30)
        verdict = "CONSISTENT -> real term" if spread < 0.25 else "INCONSISTENT -> fit artifact"
        print(f"  scale spread {spread:.1%}  =>  {verdict}")
    return rows


if __name__ == "__main__":
    # Self-check: XC_II is the KNOWN answer. Its contribution is
    # (analytic with XC_II) - (analytic without), measured via parts=4 which is
    # XC_III-only and numerically ~= the no-XC baseline.
    XC3 = {1: np.array([[0.02858352, -0.00527690, 0.01056685],
                        [-0.13714448, -0.00762597, 0.06028943],
                        [-0.02400492, -0.00769557, -0.01355196],
                        [0.13256593, 0.02059844, -0.05730414]]),
           2: np.array([[0.09476557, -0.02030206, 0.05481105],
                        [-0.29816175, -0.01001727, 0.16524024],
                        [-0.07798141, -0.01216100, -0.05180831],
                        [0.28137770, 0.04248036, -0.16824278]])}
    xc2 = {k: PLANCK[k] - XC3[k] for k in PLANCK}
    print("SELF-CHECK: scoring XC_II against the FULL target (pre-XC_II).")
    print("Expect cos ~0.94, scale ~1.0 on BOTH fixtures -- if not, harness is wrong.")
    for k in (1, 2):
        c, T = _free(xc2[k]), _free(REMAINDER[k] + xc2[k])
        cos = (c * T).sum() / (np.linalg.norm(c) * np.linalg.norm(T))
        a = (c * T).sum() / (c * c).sum()
        print(f"  fixture {k}: cos {cos:+.4f}  scale {a:.4f}")
