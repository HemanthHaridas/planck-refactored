# f/g/h-Shell (L>=3) Spherical Accuracy Fix

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Why did spherical RHF energies for f-and-higher shells (L>=3) disagree with PySCF by ~2e-5 Eh (below the variational minimum), and what fixed it?**

## Short answer

The `cart_to_sph_block` harmonic-combination matrices for L >= 3 carried integer coefficients for raw Cartesian monomials `x^lx y^ly z^lz`, but the integral engine feeds unit-normalized Cartesian components (each scaled by `N = 1/sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!)`). Since that per-component normalization differs across components (e.g. `xxx` vs `xyz` for f), applying raw-monomial coefficients directly to unit-normalized components produced spherical functions that were not pure harmonics in the physical monomials — a contaminated l-subspace. The fix, `normalized_pseudoinverse(T, L)` in `src/basis/spherical.cpp`, scales each Cartesian row of `T` by its normalization factor before the pseudoinverse, applied to L = 3, 4, 5. Water/cc-pVTZ RHF moved from -76.0571775438 to -76.0571561532 against PySCF's -76.0571561486 (diff 4.6e-9, a 4600x improvement over the prior 2.14e-5), and iteration count stabilized from an unstable 83-142 down to 90.

## Where the logic lives

- `src/basis/spherical.cpp` — `normalized_pseudoinverse`, `cart_to_sph_block` (production transform, L = 3, 4, 5)
- `src/basis/spherical_recurrence.cpp` — `cart_to_sph_block_recurrence` (independent oracle, all L, and the sole production path for L = 6)
- `tests/spherical_transform.cpp` — harmonic-purity and row-space-projector checks
- `tests/spherical_density_lift.cpp` — lift-identity unit test
- Regression cases: `water_rhf_spherical_ccpvtz_fshell`, `ne_rhf_spherical_ccpvqz_gshell`, `ne_rhf_spherical_ccpv5z_hshell`
- Unregistered exploratory input: `tests/inputs/regression/spherical/ne_rhf_spherical_ccpv6z_ishell.hfinp`

## What invariants matter

### 1. A cart-to-spherical transform must act on the same normalization convention the integral engine feeds it

The integral engine emits unit-normalized Cartesian components, not raw monomials. Any transform matrix built from raw-monomial harmonic coefficients (as opposed to hand-derived blocks that already bake normalization in, true only for L <= 2) must rescale each Cartesian row by its own normalization factor before use. Skipping this silently produces spherical functions that are valid in *span* but impure in *harmonic content* — a defect invisible to shape/rank checks and even to the density-lift identity test, since that identity holds for any consistent basis, including a wrong one.

Design rule:

- Any new or edited cart-to-sph block must be checked for harmonic purity in the *physical* (unit-normalized-component) basis, not just algebraic rank or span.

### 2. Convention differences between codes make raw-matrix comparison meaningless

Planck and PySCF normalize real solid harmonics differently, so raw spherical matrices (S_sph eigenvalues, `tr(H_sph)`) can differ by O(1) while being physically equivalent — confirmed on cc-pVDZ, where a large S_sph-eigenvalue difference (0.73) coexists with matching energy. Only convention-invariant quantities are valid cross-code comparisons: generalized eigenvalue spectra, `tr(P.H)` for a fixed physical density, or the total energy itself.

Design rule:

- Never diff raw AO matrices across codes as a correctness check. Compare generalized-eigenvalue spectra or the energy.

### 3. "Below the variational minimum in a same-spanning basis" is a specific, diagnostic anomaly

An energy that is *lower* than a reference in a basis that spans the same space (rather than merely different) means the integrals fed to SCF are internally inconsistent, not just a different valid orthonormal choice — SCF cannot vary itself below the true minimum on a consistent `tr(P.H)`/`S` pair. This was the signal that distinguished "wrong basis choice" (energy-invariant) from "corrupted integrals" (not invariant) early in the investigation.

Design rule:

- Treat any spherical-basis energy that lands below a same-span reference as evidence of an integral/metric inconsistency, not a legitimate alternate stationary point.

### 4. A span-correct, harmonic-pure transform is not automatically metric-consistent with runtime row-normalization

An earlier metric-correct (Loewdin) rewrite of the transform was a no-op for the energy because the runtime row-normalization step re-imposed its own scaling afterward, undoing the change. The eventual fix worked specifically because `normalized_pseudoinverse` corrects the transform at the point where it is built, before the pseudoinverse, rather than trying to patch the result post hoc.

Design rule:

- When fixing a transform-metric defect, verify the fix survives any downstream re-normalization step; a change that a later normalization silently undoes will look like a no-op and can be mistaken for evidence the hypothesis was wrong.

## What was fixed

1. `normalized_pseudoinverse(T, L)` added to `src/basis/spherical.cpp`: scales each Cartesian row of `T` by its normalization factor before the pseudoinverse, applied to L = 3 (f), 4 (g), 5 (h). L <= 2 already baked normalization into their hand-derived blocks and needed no change. L = 6 (i) already used the recurrence oracle (`cart_to_sph_block_recurrence`), which was already normalization-correct (`spherical_recurrence.cpp:279-296` performs the same `c_bare*sqrt(s)` scaling plus row unit-normalization) — so the fix makes the L = 3,4,5 production path consistent with the oracle it is cross-checked against, rather than changing the oracle.
2. Confirmed post-fix that production and oracle span the identical space for L = 3,4,5 (row-space projector diff ~1e-16); the raw matrices still differ by ~0.7, which is only the intra-span basis/m-ordering convention and is energy-invariant.
3. **FU1 + FU4 — cross-code energy gates at f, g, h added.** Three PySCF-anchored spherical RHF gates (`extended`+`spherical` suite):

   | shell | case | PySCF | Planck | delta | iters |
   |---|---|---|---|---|---|
   | f (L=3) | `water_rhf_spherical_ccpvtz_fshell` | -76.0571274203 | -76.0571274250 | **4.7e-9** | 19 |
   | g (L=4) | `ne_rhf_spherical_ccpvqz_gshell` | -128.5434696591 | all 10 digits | **0.0e+00** | 11 |
   | h (L=5) | `ne_rhf_spherical_ccpv5z_hshell` | -128.5467701295 | all 10 digits | **0.0e+00** | 12 |

   The f case is the tripwire that was previously missing: pre-fix it gave -76.0571775438 in 83-142 iterations. The g and h cases close FU1 — they had ridden the fixed path on a span-match argument with no end-to-end number until this landed. References were built from Planck's own GBS via `pyscf.gto.basis.parse_gaussian`, removing the basis-coefficient confound rather than assuming it away; they agree with PySCF's built-in cc-pVTZ/QZ/5Z, which is the check that the two basis sets match. Note: Planck's GBS is Gaussian94 format, so `gto.basis.parse` (NWChem) raises `BasisNotFoundError` on it directly, and PySCF 2.13.0 has no built-in cc-pV6Z at all.
4. **FU3 — convention-invariant transform tests added to `tests/spherical_transform.cpp`, applied for all L <= 6:**
   - Harmonic purity in the physical monomials: `max_laplacian` already undid the unit-normalization weighting before differentiating, but it had only ever been pointed at the oracle, never at production.
   - Row-space projector equality with the oracle, invariant to the m-ordering convention. The matrices are not compared element-wise, since they are different valid bases of the same space.

   Both checks were verified falsifiable in situ, by deliberately dropping the row scaling in `normalized_pseudoinverse` and rebuilding:

   | L | max|laplacian| | max|delta projector| |
   |---|---|---|
   | 0-2 | 0 | 0 (unaffected — normalization is baked into the hand-derived blocks) |
   | 3 | **2.2e-1** | **3.4e-1** |
   | 4 | **1.8e-1** | **3.0e-1** |
   | 5 | **8.4e-2** | **2.7e-1** |
   | 6 | 0 | 0 (unaffected — delegates to the oracle) |

   Post-fix, all of L <= 6 sits at ~1e-16. The failures land on exactly the three shells the fix touched, and the pre-existing shape/rank check stayed green throughout, which is precisely why the defect shipped originally. This also corrected a false statement in the file's own prior header, which claimed the L >= 3 pseudoinverse "legitimately carries r^2-contamination" — that described the defect itself, not a property of the transform.

## Validation strategy that should remain in place

- `water_rhf_spherical_ccpvtz_fshell`, `ne_rhf_spherical_ccpvqz_gshell`, `ne_rhf_spherical_ccpv5z_hshell` (PySCF-anchored, `extended`+`spherical` suite)
- `tests/spherical_transform.cpp` harmonic-purity and row-space-projector checks for all L <= 6, mutation-verified against the deliberately-broken transform
- `tests/spherical_density_lift.cpp` lift-identity check (necessary but not sufficient on its own — see invariant 1)
- No-regression checks on d-shell and below: cc-pVDZ unchanged at 3.3e-9; spherical 6-31g* regression suite (`be_ccsd`/`be_fci`/`water_rhf_freq`/`geomoptfreq` spherical) all pass

## Remaining architecture concern: FU2, the i-shell (L=6) oracle path is unchecked against another code

`tests/inputs/regression/spherical/ne_rhf_spherical_ccpv6z_ishell.hfinp` exists and carries its reference inline (PySCF 2.13.0 RHF/cc-pV6Z spherical Ne = -128.5470611007 Eh), but is deliberately not registered as a regression case: Ne/cc-pV6Z is 140 spherical AOs, and the conventional `nb^4` ERI build makes it far too heavy for the suite (the h case, at only 91 AOs, already takes ~37 s).

L = 6 is a structurally different code path from the L = 3,4,5 fix: it delegates entirely to the recurrence oracle (`cart_to_sph_block_recurrence`) and never touches `normalized_pseudoinverse`. A disagreement there would implicate `spherical_recurrence.cpp`, not the pseudoinverse fix this document describes. `MAX_L = 6` makes this the top of the currently supported angular-momentum range, so closing FU2 would complete cross-code validation across the entire supported range. Expected accuracy, by analogy with f/g/h, is ~1e-9.

## Historical investigation trail (falsified hypotheses, kept as guardrails)

Two earlier hypotheses were investigated and disproven before the root cause above was found. They are kept here because they name traps worth avoiding on any future spherical-basis defect.

**H1: basis coefficient precision.** True only for STO-3G (10-digit GBS vs 8-digit PySCF coefficients), and proven benign by coefficient-matching to 1.8e-14. Not the f-shell bug.

**H2: cart-to-sph transform contamination / metric-free pseudoinverse.** The measurement that `C.M.C^T != I` (the transform is not metric-orthonormal under the Cartesian metric) was real, but energy-irrelevant: the transform is span-preserving and SCF re-orthogonalizes via `S`. The transform was independently proven span-correct and pure-harmonic under the projector/Laplacian checks. A metric-correct (Loewdin) rewrite of `cart_to_sph_block` was implemented and found to be a no-op for the energy (see invariant 4), because the runtime row-normalization step re-imposes its own scaling and undoes it. The lesson: check energy-relevance (span, invariant spectrum) before trusting a raw-matrix "defect" as the cause.

Two other facts established along the way, proven and not to be re-litigated: the Cartesian one-electron integrals were always correct (Planck-vs-PySCF Cartesian `S^-1 T`, `S^-1 V`, `S^-1 H_core` generalized eigenvalue spectra agree to ~1e-6), and Planck's spherical overlap `S` was always positive-definite and well-conditioned (min eigenvalue 2.6e-3), ruling out a linear-dependence artifact as the cause.
