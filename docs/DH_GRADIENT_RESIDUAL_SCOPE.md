# The Double-Hybrid Gradient's Last ~31%: Two Scoped Investigations

In-flight scope. Fold into `docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md` (or
its answer-doc successor) when this lands. Canonical status lives in
`vault/Status/`.

**Question this scopes:** after N3.5.7.10 identified and wired Eq. 33's XC term
(XC_II, 69% of the missing quantity), a residual of **3.887e-4 Ha/Bohr**
(C1 H2O2) / **2.057e-4** (water) remains between Planck's analytic double-hybrid
gradient and finite difference. Every DH-specific object in the paper's
Eqs. 40-47 has been verified or excluded against a two-fixture target
(N3.5.7.8 through .14). Two hypotheses survive. They need different work, and
**H2 is much cheaper and should be run first.**

## What is already established (do not re-derive)

| fact | where | value |
|---|---|---|
| Planck's `*corr` vs exact FD `0.27 dE_corr/dR` | N3.5.7.9 | 2.42e-4 (water), and the total residual equals it -- the error is entirely in `*corr` |
| XC_II is Eq. 33's XC term | N3.5.7.10/.14 | cos 0.946/0.942, scale 1.078/1.015 across two fixtures; term-matched to Eq. 23 |
| PT2 assembly is internally sound | N3.5.7.9 | every accumulator translationally invariant to 1e-14 |
| XC_I, XC_III, both frame variants, the density choice, the spin factor, any rescaling of XC_II | N3.5.7.8-.11 | all excluded |
| whole `vhf_s1occ` family | N3.5.7.13 | bounded: any multiple leaves >=72% of the remainder |
| Eq. 42's `R^XC(D)` in isolation | N3.5.7.14 | 7% WORSE, cos -0.53/-0.44 |
| Eq. 39's `1/(1+delta_ij)` convention | N3.5.7.14 | not a defect; Planck's unrestricted sum is equivalent |
| PySCF `grad/mp2.py` as a reference | N3.5.7.9 | INVALID (nonlinear `get_veff` -> singular CPHF, cond 1.6e18) |

**Instruments, all committed:** two C1 fixtures with exact FD targets
(`tests/inputs/exploratory/dh_gradient/`, `tests/pyscf/h2o2{,b}_b2plyp_dh_gradient_fd.py`),
the scoring harness with positive and negative controls
(`tests/pyscf/dh_gradient_score.py`), and the `PLANCK_DFT_DH_XC_PARTS` /
`_SCALE` component probe.

## Three measurements taken while scoping (they shape both investigations)

**1. The residual is NOT grid noise.** Planck's DH gradient at `grid ultrafine`
is converged to **9.3e-5** against `fine` -- 4x below the 3.9e-4 residual.
(`normal` differs by 1.4e-3 and would swamp it, so ultrafine is mandatory for
any measurement in this arc.)

**2. The KS-only gradient is 11x more accurate than the PT2 remainder.**
Planck's hybrid-part gradient (`KS-total`, no PT2) against the same PySCF FD:

```
max|Planck_KS - FD_KS| = 3.657e-05 Ha/Bohr
PT2 remainder          = 3.887e-04 Ha/Bohr        ratio 0.09
```

**This is the single most constraining number for H2** and it was not available
before. A shared upstream defect would have to cancel to 3.7e-5 in the KS path
while surviving at 11x in the PT2 path.

**3. The residual sits below the paper's own validation threshold on stiff
coordinates, and above it on soft ones.** Converting force error to bond-length
error via `dR = f/k`:

| mode | k (Ha/Bohr^2) | dR from 3.887e-4 |
|---|---|---|
| stiff X-H stretch | 0.50 | **0.041 pm** |
| typical X-Y stretch | 0.30 | 0.069 pm |
| bend | 0.10 | 0.206 pm |
| soft mode / torsion | 0.03 | **0.686 pm** |

The paper reports B2-PLYP MAD **0.3 pm** (light set) / **0.6 pm** (heavy set)
against experiment, and cross-checks ORCA against TURBOMOLE only at
"less than 0.01 mEh" in **total energy** -- never a force component. So the
residual is invisible to every validation the paper performed on stiff
coordinates, and marginal-to-visible on soft ones.

---

## H2 -- an upstream defect in Planck's KS-side assembly (DO THIS FIRST)

**Claim.** The residual is a Planck defect in how KS quantities feed the PT2
path, not a gap in the paper's theory.

**Why it is still live despite measurement 2.** `*corr = full - ref_grad`, and
both halves are built on KS orbitals. A defect in a quantity the KS gradient
uses *differently* (or not at all) would not show in the KS-only comparison. The
KS gradient never touches: the MO-basis response machinery, the Z-vector
solution, the `zeta`/`imat` MO intermediates, or `rmp2_kernel` run on KS
orbitals. **Measurement 2 excludes a shared AO-level defect, not an
MO-side one.**

**Why first.** Every step is hours, uses instruments that already exist, and
each has a decisive pass/fail. H1 needs a geometry-optimization campaign.

### H2.1 -- RUN. The residual is LINEAR THROUGH THE ORIGIN to 97%: the defect is inside the PT2 term

Planck's gradient is `G_KS + c_pt2 * corr` and the FD reference is
`FD_KS + c_pt2 * dEcorr/dR`, so the residual is
`(G_KS - FD_KS) + c_pt2 * (corr - dEcorr/dR)` -- an intercept plus a slope, and
both are already measured. Linearity confirmed numerically by running Planck at
`c_pt2` scaled 0.5x / 1x / 2x (`max|[G(2)-G(1)] - 2[G(1)-G(0.5)]| = 1.0e-8`).

| piece | value | share |
|---|---|---|
| intercept (= the KS-only error, measurement 2) | `3.66e-5` | **3%** |
| slope (the PT2 part) | `1.24e-3` | **97%** |

**The `ref_grad` subtraction is not leaking**, which independently confirms
N3.5.7.9's per-term invariance check from a different direction. Whatever is
missing is a term in `E_PT2^x`, exactly as every candidate has assumed.

<details><summary>original plan (superseded by the above)</summary>

`*corr` is built from a Lagrangian scaled by `c_pt2 = 0.27`. If the residual is
**linear in `c_pt2`**, it lives in the PT2 correction proper. If it has a
`c_pt2`-independent piece, something is leaking from the reference side.

Run the two fixtures at `c_pt2` scaled to 0.5x and 2x (the
`PLANCK_DFT_DH_GRADIENT_SELFCHECK` probe already re-runs
`build_pt2_mo_intermediates` at `2*c_pt2` -- reuse that path, and note its
existing gotcha: it must be passed the same `ks_veff`). Compare each against an
FD reference recomputed at the same scaling (the FD script takes `0.27` as a
literal; parameterize it).

- **Linear through the origin** -> the defect is inside the PT2 term. Proceed to
  H2.2.
- **Non-zero intercept** -> the `ref_grad` subtraction is leaking. That
  contradicts N3.5.7.9's per-term invariance check, so re-examine that first.

**Falsifiable:** a defect that is *not* linear in `c_pt2` cannot be a missing
term in `E_PT2^x`, which is what every candidate so far has assumed.
</details>

### H2.2 -- RUN. Amplitudes agree ELEMENTWISE to 8e-9; exonerated

Dumped Planck's `t2` (`[i,j,a,b]` row-major) with its own `mo_coeff`/`mo_energy`,
then rebuilt PySCF's amplitudes **using Planck's C**, so MO phase and ordering
match by construction and no phase-invariant dodge is needed.

```
max|t2_planck - t2_pyscf| = 7.96e-09   (|t2|max 6.96e-02, rel 1.1e-07)
max|eps_planck - eps_pyscf| = 2.43e-06
```

The 8e-9 is fully accounted for by the 2.4e-6 orbital-energy difference between
the two independently-converged SCF solutions -- it is not an MP2-kernel error.
**Amplitudes are exonerated, and with them `Gamma^NS`.** This was the failure
mode `CCGEN_SPIN_ADAPT_DEFAULT` recorded (energy right because it is a
contraction, amplitudes wrong); it does not apply here.

<details><summary>original plan (superseded by the above)</summary>

Never checked. The energy agrees with PySCF to 6.3e-8 (N3.5.7.10), but the
**energy is a contraction of amplitudes** and can be right while individual
amplitudes are wrong in a way that cancels in the trace but not in the gradient
-- the exact failure mode `CCGEN_SPIN_ADAPT_DEFAULT` recorded for CC kernels.

Dump `t2` from Planck and from PySCF's `mp.MP2(ks)` on the same KS orbitals and
compare **elementwise** (MO phase is fixed by using Planck's own `mo_coeff` in
the PySCF run, so no phase-invariance dodge is needed). Tolerance 1e-10.

- **Mismatch** -> found it; the whole downstream chain inherits it.
- **Match** -> amplitudes are exonerated, and with them `Gamma^NS`.
</details>

### H2.3 -- RUN. Exact solve, symmetric well-conditioned operator; exonerated -- but the Z-vector has REAL LEVERAGE

**Part 1 is answered by inspection: there is no convergence tolerance to be
loose.** `dh_relaxed_density.cpp` assembles the Hessian densely and solves with
`colPivHouseholderQr` -- a direct solve, not an iterative one.

Probe results on both fixtures:

```
dim=27  ||Az-rhs||=6.8e-18  max|A-A^T|=1.0e-15 (rel 5.2e-17)
        eig[min,max]=[3.07e-01, 1.99e+01]   (SPD, cond ~65)
```

Machine-precision solve, symmetric to 5e-17 relative, positive definite. Part 2
(independent construction) done by cross-checking `A` against
`build_rhf_cphf_matrix`: they differ by 1.9% overall but **33% in the
off-diagonal block**, which is the expected structural difference from replacing
full HF exchange with `0.53*K + R^XC`. (The diagonal difference is larger in
absolute terms only because the diagonal is dominated by `eps_a - eps_i` ~ 20
while off-diagonals are ~0.2 -- an initial reading of this as "suspiciously
diagonal" was wrong.)

**But the Z-vector matters far more than expected, which is worth carrying.**
Switching the operator to HF-CPHF (`PLANCK_DFT_DH_ZVECTOR_HFCPHF`) moves the
final gradient by **1.39e-3 / 1.18e-3** -- **3.6x the residual** -- and makes it
worse (residual 3.89e-4 -> 1.59e-3). So the KS operator is both load-bearing and
the better choice, and a *subtle* error in it could comfortably carry 3.9e-4.
The checks above say it is well-formed, not that every coefficient in it is
right.

<details><summary>original plan (superseded by the above)</summary>

The DH path solves the Z-vector with `build_ks_orbital_hessian_op` (N2). Two
checks, neither done:
1. **Residual norm.** Print `||(eps_a - eps_i) Z + R(Z) + L||` at exit. If the
   solver is stopping loose, a 1e-4 force error is exactly what a partially
   converged Z produces. **Check this before anything else in H2.3** -- it is
   ten lines and could end the entire investigation.
2. **Independent solve.** Rebuild the same linear system densely (as the
   N3.5.7.9 PySCF probe did -- `nvir*nocc` is tiny here) and solve by
   `np.linalg.solve`. Compare `Z` elementwise.

**Falsifiable and cheap.** A tight residual plus a matching dense solve
exonerates the Z-vector entirely.
</details>

### H2.4 -- the `ks_veff` closure's ground density -- CHECKED WHILE SCOPING, REFUTED

The candidate was: `ground_density_veff = calculator._info._scf.alpha.density`
looks like an **alpha** (half) density, while
`compute_analytic_xc_hessian_vector_product` wants the **total** -- a
factor-of-2 in the kernel's *argument*, which no prefactor screen can see.

**It is not a defect.** For RKS, `make_spin_density` is called with
`doubled_occupancy = true` (`driver.cpp:648`), so `alpha.density = 2 C_occ
C_occ^T` -- the TOTAL density. `ks_orbital_hessian.h:38` documents its own
`density` field as "Converged KS **total** density (AO)" and SOSCF (validated)
passes exactly this quantity. Confirmed at runtime: `tr(P S) = 18.000000` on
H2O2, i.e. all 18 electrons, not 9.

Recorded rather than deleted because the naming genuinely invites the mistake --
`alpha.density` holding a total density is a trap a future reader will hit
again.

### H2 -- OUTCOME: all four steps run, all clean. H1 is now the live hypothesis.

| step | result |
|---|---|
| H2.1 `c_pt2` linearity | residual is **97% slope, 3% intercept** -- the defect is inside `E_PT2^x`; `ref_grad` does not leak |
| H2.2 amplitudes | elementwise to **8e-9**; exonerated, with `Gamma^NS` |
| H2.3 Z-vector | exact QR solve (resid 7e-18), symmetric (5e-17 rel), SPD; operator well-formed |
| H2.4 density convention | refuted during scoping (`alpha.density` IS the total for RKS) |

**H2 is exonerated to the precision available**, and the scope's own instruction
applies: record it and do not invent an H2.5. The one caveat worth carrying is
H2.3's leverage finding -- the Z-vector operator moves the gradient by 3.6x the
residual, so "well-formed" is not the same as "every coefficient correct". If
H1.2 ever shows Planck (not the paper) is wrong, the KS orbital Hessian's
coefficients are where to look first, despite passing every structural check
here.

**Next: H1.1**, which the scope already marks as the highest-value step in
either investigation and to be run regardless of cause.

---

## H1 -- the paper's equations do not account for the residual at this magnitude

**Claim.** Planck implements Eqs. 40-47 correctly and the residual is a genuine
limitation of the published formalism (or of the closed-shell reduction in
Sec. III.A), at a magnitude the paper never tested.

**Why it is plausible.** Measurement 3: the paper validates against experimental
**geometries** at 0.3-0.6 pm MAD and cross-checks codes only on **total
energy**. A 3.9e-4 Ha/Bohr force error is 0.04 pm on a stiff bond -- invisible to
everything the authors measured. **No published check would have caught it.**

**Why this is hard to prove, and what would count.** "The paper is incomplete" is
unfalsifiable as stated. It becomes testable only via an independent
implementation of the same equations.

### H1.1 -- the geometry-level consequence test (~1 day, do this even if H1 is never resolved)

**Independently of the cause, establish whether the residual matters.** Optimize
both fixtures plus water to convergence with the DH gradient and compare against
an FD-driven (numerical-gradient) optimization of the same energy.

- If optimized geometries agree to **<0.05 pm** on stiff coordinates, the
  gradient is **fit for purpose** for geometry optimization, and N3.6 can lift
  the workflow gate with the residual documented as a known bound rather than
  a blocker.
- Soft coordinates (H2O2's dihedral is ideal -- measurement 3 predicts
  ~0.7 pm there) are where it would bite; measure the dihedral specifically.

**This is the highest-value step in either investigation**, because it decides
whether the remaining 31% is a shipping blocker or a footnote. It does not
require knowing the cause.

### H1.2 -- cross-code check against ORCA (~1 day, needs ORCA access)

The paper's implementation *is* ORCA. If ORCA's B2-PLYP analytic gradient on
fixture 1 shows the **same** deviation from FD, the residual is in the published
equations and Planck reproduces them faithfully. If ORCA is FD-clean, it is
Planck's.

**This is the only decisive test of H1** and it needs a licence. Run
`! B2PLYP def2-SVP EnGrad` plus a numerical-gradient job on the identical
geometry; compare each to its own FD, not to Planck.

**If ORCA is unavailable, H1 cannot be settled**, and the honest outcome is to
document the residual as a bound and ship behind the existing flag. Say so
rather than substituting a weaker test.

### H1.3 -- re-derive the closed-shell reduction (~2-3 days, only if H1.2 says the paper is wrong)

Sec. III.A collapses the spin-unrestricted Eqs. 22-35 to Eqs. 40-47. That
collapse is stated, not derived, in the paper. Re-derive it from the
unrestricted form and compare term-by-term against Planck's contractions --
particularly Eq. 46's `1/2 PP - 1/4 PP + DP - 1/2 DP`, whose four coefficients
are the collapse's most error-prone output.

**Do not start here.** It is the most expensive step and only worth doing once
H1.2 has established that the closed-shell path is where the discrepancy lives.

---

## Order, and the stop condition

1. **H2.3's residual-norm print** (10 lines) -- could end everything.
2. **H2.2** (amplitudes elementwise vs PySCF) -- the one quantity never checked
   elementwise, and the failure mode this project has already been bitten by.
3. **H2.1** (`c_pt2` linearity) -- exonerates or localises.
4. **H1.1** -- decides whether any of this blocks shipping. **Run regardless.**
5. **H1.2** if ORCA is available; otherwise document and stop.

(H2.4 was checked during scoping and refuted -- see above.)

**Stop condition for the whole arc:** if H2 is exonerated and H1.1 shows the
geometry consequence is below 0.05 pm on stiff coordinates, **ship the gradient
behind the existing flag with the residual documented**, and stop hunting. The
remaining 31% would then be a known, bounded, non-blocking limitation -- which
is a legitimate outcome, not a failure.

**Method rule carried from N3.5.7.8-.14:** no probe without a derivation behind
it. Every productive step in this arc came from reading a source (the paper, or
Planck's own code); every speculative probe was negative.
