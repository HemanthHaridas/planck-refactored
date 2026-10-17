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

# MISLABEL CORRECTED 2026-09-10. These are the FD **TOTAL DH gradient**
# (dE_total/dR), NOT 0.27*dE_corr/dR -- the name FD_TOT is right, the old
# comment above it was wrong, and they differ by ~10x in magnitude.
#
# Verified directly: Planck's own FD of 0.27*E_corr reproduces the FD script's
# printed `0.27*d(E_corr)/dR` block to **6.4e-08**, while these values match
# the script's TOTAL block. Every score computed here compares Planck's TOTAL
# analytic gradient against the FD TOTAL, which is the correct pairing --
# so no past conclusion is invalidated, but the comment would mislead anyone
# who trusted it when adding a new candidate.
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

# ---------------------------------------------------------------------------
# A1 (2026-09-10): the CORRECT target -- *corr against Planck's OWN FD of
# 0.27*E_corr, per fixture.
#
# Everything above scores TOTAL vs TOTAL. That pairing is self-consistent, but
# the KS and *corr halves partially CANCEL in the total, so it understates the
# real defect by 3.2x. Measuring the halves separately
# (PLANCK_DEBUG_DH_CHANNELS) gives KS correct to 1.02e-6 and *corr wrong by
# 1.24e-3 on a 1.88e-2 signal -- 100% of the defect is in *corr.
#
# Score candidates HERE, not against REMAINDER. See
# docs/DH_CORR_GRADIENT_DEFECT_SCOPE.md.
#
# Fixture 3 (h4) is s-only: every basis function is l=0, so it excludes every
# angular-momentum-dependent explanation in a single run. It is the cheapest
# discriminator in the set -- run a candidate there FIRST.

# Planck's own FD of 0.27*E_corr (same binary, same grid, same settings).
CORR_FD = {
    1: np.array([[0.0108844600, -0.0187604400, 0.0065567400],
                 [-0.0102507400, 0.0184097800, 0.0024252200],
                 [-0.0087019800, -0.0000123000, -0.0053664700],
                 [0.0080682600, 0.0003629600, -0.0036155000]]),
    2: np.array([[0.0078815125, -0.0150027296, 0.0070023639],
                 [-0.0067138829, 0.0163644613, 0.0020907527],
                 [-0.0074556836, -0.0015844095, -0.0051435760],
                 [0.0062880540, 0.0002226778, -0.0039495141]]),
    3: np.array([[0.0000074349, 0.0000148434, 0.0045079813],
                 [0.0000138380, 0.0000059797, -0.0045066848],
                 [-0.0016812489, 0.0045728849, -0.0013108778],
                 [0.0016599495, -0.0045937080, 0.0013095548]]),
}

# Planck's ANALYTIC *corr (PLANCK_DEBUG_DH_CHANNELS, production wiring).
CORR_PLANCK = {
    1: np.array([[0.0111504543, -0.0199570213, 0.0066452966],
                 [-0.0104591819, 0.0196528322, 0.0023482398],
                 [-0.0087220334, 0.0001756308, -0.0053973313],
                 [0.0080307610, 0.0001285584, -0.0035962051]]),
    2: np.array([[0.0080280676, -0.0159046947, 0.0071064410],
                 [-0.0067729454, 0.0173471682, 0.0019501078],
                 [-0.0074354537, -0.0014186093, -0.0051732614],
                 [0.0061803315, -0.0000238642, -0.0038832874]]),
    3: np.array([[0.0000091749, 0.0000218250, 0.0041595988],
                 [0.0000151660, 0.0000097194, -0.0041587480],
                 [-0.0015893998, 0.0042825881, -0.0012319334],
                 [0.0015650588, -0.0043141326, 0.0012310826]]),
}

# The defect a candidate must explain: analytic MINUS FD, so a candidate whose
# contribution matches it at scale +1 is the term to SUBTRACT.
CORR_DEFECT = {k: CORR_PLANCK[k] - CORR_FD[k] for k in CORR_FD}

FIXTURE_NAME = {1: "h2o2 C1", 2: "h2o2b C1", 3: "H4 (s-only)"}


def _free(v):
    """Translation-free part: the physically meaningful subspace, since the
    target has zero net force."""
    return v - v.sum(axis=0) / v.shape[0]


def score_corr(name, contrib, verbose=True):
    """Score a candidate against the *corr defect -- the A1 target.

    `contrib` is the candidate's CONTRIBUTION to Planck's analytic *corr, as
    {fixture: 4x3}. A candidate that IS the defect scores cos ~ +1 at scale
    ~ +1 with a consistent coefficient across fixtures.

    Three properties are checked, not one, because the scale spread alone has
    a thin margin (the total-gradient scorer's own docstring records a random
    control landing at 30% against a 25% threshold):

      cos      direction match in the translation-free subspace
      scale    fitted coefficient; must AGREE across fixtures
      tfree    the candidate must be translation-free like the defect is
               (the defect's net force is <= 2.7e-8, i.e. 0.0% of its size)

    Scoring on fixture 3 (s-only) is the cheap angular-momentum discriminator:
    a candidate that vanishes there cannot explain a defect that does not.
    """
    rows = []
    for k in sorted(contrib):
        c = np.asarray(contrib[k], dtype=float)
        D = CORR_DEFECT[k]
        cf, Df = _free(c), _free(D)
        nc = np.linalg.norm(cf)
        if nc == 0.0:
            rows.append((k, 0.0, 0.0, np.abs(D).max(), 0.0))
            continue
        cos = float((cf * Df).sum() / (nc * np.linalg.norm(Df)))
        a = float((cf * Df).sum() / (cf * cf).sum())
        resid = float(np.abs(D - a * c).max())
        netf = float(np.abs(c.sum(axis=0)).max() / max(np.abs(c).max(), 1e-30))
        rows.append((k, cos, a, resid, netf))

    if verbose:
        print(f"\n=== {name}  [target: the *corr defect] ===")
        print(f"{'fixture':>13}{'cos':>9}{'scale':>11}{'resid@best':>13}"
              f"{'baseline':>12}{'net/|c|':>10}")
        for k, cos, a, resid, netf in rows:
            print(f"{FIXTURE_NAME.get(k, k):>13}{cos:>9.4f}{a:>11.4f}"
                  f"{resid:>13.3e}{np.abs(CORR_DEFECT[k]).max():>12.3e}"
                  f"{netf:>10.1e}")
        # Every fixture counts, INCLUDING one where the candidate vanishes.
        # Dropping zero scales here let a heavy-atom-only candidate -- which
        # explains nothing on the s-only fixture -- report "CONSISTENT".
        scales = [r[2] for r in rows]
        dead = [FIXTURE_NAME.get(r[0], r[0]) for r in rows if r[2] == 0.0]
        if len(scales) >= 2:
            denom = max(abs(s_) for s_ in scales)
            spread = ((max(scales) - min(scales)) / denom) if denom > 0 else 0.0
            if dead:
                verdict = (f"REJECTED -> vanishes on {', '.join(dead)}, "
                           "where the defect does not")
            elif spread < 0.25:
                verdict = "CONSISTENT -> real term"
            else:
                verdict = "INCONSISTENT -> fit artifact"
            print(f"  scale spread {spread:.1%}  =>  {verdict}")
        # Threshold CALIBRATED against the defect itself, which scores up to
        # 7.6e-05 on this relative measure (net force / max|component|) purely
        # from the rounding of the committed reference blocks. An absolute
        # 1e-6 bound flagged the positive control -- i.e. it called the known
        # right answer wrong. 1e-3 is two orders above the defect's own value
        # and still an order below anything with a real net force.
        if any(r[4] > 1e-3 for r in rows):
            print("  WARNING: candidate is NOT translation-free; the defect is "
                  "(net/|c| <= 7.6e-5). A term that does not conserve momentum "
                  "cannot be the whole explanation.")
        if 3 in dict((r[0], r) for r in rows):
            r3 = dict((r[0], r) for r in rows)[3]
            if abs(r3[1]) < 0.2:
                print("  NOTE: near-zero cos on the s-only fixture. The defect "
                      "is 7.6% there, so a candidate that vanishes with no "
                      "p-shells cannot explain it.")
    return rows


def random_control(n=400, seed=0):
    """What cos/spread does a MEANINGLESS candidate get? Run this before
    believing any score -- the thresholds are only interpretable against it."""
    rng = np.random.default_rng(seed)
    cs, sps = [], []
    for _ in range(n):
        v = rng.normal(size=(4, 3))
        rows = score_corr("rand", {k: v for k in CORR_DEFECT}, verbose=False)
        cs.append(abs(rows[0][1]))
        sc = [r[2] for r in rows]
        sps.append((max(sc) - min(sc)) / max(abs(x) for x in sc))
    cs, sps = np.array(cs), np.array(sps)
    print(f"\nRANDOM CONTROL ({n} draws), scoring against the *corr defect:")
    print(f"  |cos|        median {np.median(cs):.3f}   90th pct {np.percentile(cs, 90):.3f}")
    print(f"  scale spread median {np.median(sps):.1%}   10th pct {np.percentile(sps, 10):.1%}")
    print(f"  fraction passing the 25% spread test by luck: {(sps < 0.25).mean():.1%}")
    return cs, sps


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


def _selfcheck_corr():
    """A1 self-check on the *corr layer, three ways.

    POSITIVE: the defect scores itself at cos +1, scale +1, zero residual on
    every fixture. This only proves the arithmetic, so it is necessary and
    far from sufficient -- hence the other two.

    NEGATIVE: a random direction, 400 draws, so the cos and spread thresholds
    are interpretable rather than asserted.

    DISCRIMINATING: a candidate that lives only on the heavy atoms (zero on
    fixture 3) is REJECTED, which is the angular-momentum family the s-only
    fixture exists to exclude.
    """
    print("=" * 70)
    print("A1 SELF-CHECK -- the *corr scoring layer")
    print("=" * 70)
    print("\nThe defect itself, per fixture (what a candidate must explain):")
    for k in sorted(CORR_DEFECT):
        D = CORR_DEFECT[k]
        rel = np.abs(D).max() / np.abs(CORR_FD[k]).max()
        print(f"  {FIXTURE_NAME[k]:>13}: max|d| = {np.abs(D).max():.4e}"
              f"   rel = {rel:.2%}"
              f"   net force = {np.abs(D.sum(axis=0)).max():.1e}")

    score_corr("POSITIVE CONTROL: the defect scoring itself",
               {k: CORR_DEFECT[k] for k in CORR_DEFECT})

    # The positive control above is invariant to a sign flip in CORR_DEFECT
    # and to dropping the translation projection -- both mutations scored a
    # clean cos +1 through it, because a vector scored against ITSELF cancels
    # either error. Two fixed reference vectors close that:
    #
    #  SIGN     a candidate equal to +CORR_PLANCK must score the SAME SIGN as
    #           the defect's own direction. Flip the defect and this flips.
    #  PROJECT  a pure TRANSLATION (every atom displaced identically) is
    #           entirely removed by _free, so it must score cos 0. Without
    #           the projection it scores nonzero.
    print("\n--- mutation guards (fixed vectors, not self-scoring) ---")
    sign_probe = score_corr("SIGN GUARD: +CORR_PLANCK as a candidate",
                            {k: CORR_PLANCK[k] for k in CORR_PLANCK},
                            verbose=False)
    # MEASURED, not asserted: +CORR_PLANCK scores +0.8513 against the defect
    # on fixture 1. The value itself is not meaningful -- what is checked is
    # that it is POSITIVE, which flips if CORR_DEFECT's subtraction order is
    # reversed. (An earlier version of this line carried a guessed expected
    # value of -0.5148 and was simply wrong; the guard is only worth having
    # if its reference comes from a run.)
    SIGN_GUARD_REF = +0.8513
    got = sign_probe[0][1]
    ok = (got > 0) and abs(got - SIGN_GUARD_REF) < 1e-3
    print(f"  cos on {FIXTURE_NAME[1]} = {got:+.4f}   "
          f"expect {SIGN_GUARD_REF:+.4f}   "
          f"{'OK' if ok else 'FAIL -- CORR_DEFECT sign/definition changed'}")
    # A PURE translation is a vacuous projection guard: it is orthogonal to
    # the defect whether or not _free is applied, so dropping the projection
    # scored a clean 0.0000 through it. The probe must be a vector whose
    # translational PART matters -- defect direction plus a large uniform
    # shift. With _free the shift is removed and cos returns to the defect's
    # own +1.0000; without it the shift dominates and cos collapses.
    biased = {k: CORR_DEFECT[k] + np.tile(np.array([0.02, -0.05, 0.03]), (4, 1))
              for k in CORR_DEFECT}
    pr = score_corr("PROJECTION GUARD: defect + a large uniform shift",
                    biased, verbose=False)
    got = pr[0][1]
    ok = abs(got - 1.0) < 1e-6
    print(f"  cos on {FIXTURE_NAME[1]} = {got:+.4f}   expect +1.0000   "
          f"{'OK' if ok else 'FAIL -- translation projection not applied'}")

    zero3 = {1: CORR_DEFECT[1], 2: CORR_DEFECT[2],
             3: np.zeros((4, 3))}
    score_corr("DISCRIMINATING CONTROL: a heavy-atom-only candidate", zero3)

    random_control()


if __name__ == "__main__":
    _selfcheck_corr()
    print()
    print("=" * 70)
    print("LEGACY total-gradient self-check (kept: it is the positive control")
    print("for the OLD layer, and still passes)")
    print("=" * 70)
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
