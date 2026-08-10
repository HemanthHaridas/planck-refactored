# f-shell (L≥3) Spherical Accuracy Defect — FIXED

## Status: ROOT CAUSE FOUND + FIXED (2026-08-09)

**Root cause:** the `cart_to_sph_block` harmonic-combination matrices for L ≥ 3 carry
integer coefficients for RAW Cartesian monomials `x^lx y^ly z^lz`, but the integral
engine feeds **unit-normalized** Cartesian components (each scaled by
`N = 1/√((2lx-1)!!(2ly-1)!!(2lz-1)!!)`). The per-component N differs across components
(e.g. `xxx` vs `xyz` for f), so applying raw-monomial coefficients directly to unit
components produces spherical functions that are NOT pure harmonics in the physical
monomials — a contaminated ℓ-subspace. This shifted f-and-up energies (~2.14e-5 for
water/cc-pVTZ, *below* the variational minimum) and stressed SCF convergence
(83–142 iters). The hand-derived L ≤ 2 blocks already baked the normalization in
(their 1/√3 factors); L = 3,4,5 (raw `pinv`) did not. The recurrence oracle (L ≥ 6)
was not exercised by the tested basis but should be checked separately.

**Fix** (`src/basis/spherical.cpp`): `normalized_pseudoinverse(T, L)` scales each
Cartesian row of T by its normalization factor before the pseudoinverse, so the
transform acts correctly on unit-normalized components. Applied to L = 3,4,5.

**Validation:**
- water/cc-pVTZ RHF: −76.0571775438 → **−76.0571561532** (PySCF −76.0571561486,
  diff **4.6e-9**; was 2.14e-5 — a 4600× improvement).
- Convergence: 90 iters, stable across guesses (was 83–142, unstable).
- No d-shell regression: cc-pVDZ unchanged at 3.3e-9; spherical 6-31g* regression
  suite (be_ccsd/be_fci/water_rhf_freq/geomoptfreq spherical) all pass.
- Spherical transform + density-lift unit tests still pass.

**Test gap (why it shipped, G4):** the `spherical_density_lift` test checks the
lift identity `tr(M_sph·X_sph) = tr(M_cart·X_cart)` — which holds for ANY C
(it is `tr(C M Cᵀ · C X Cᵀ)` self-consistency), so it passes for a wrong subspace.
The `spherical_transform` test checks harmonic purity only for L ≤ 2 and rank for
L ≥ 3. Neither checks that the L ≥ 3 spherical functions are pure harmonics in the
PHYSICAL (unit-normalized-component) basis. TODO: add a test that the production
`cart_to_sph_block(L)`, acting on unit-normalized Cartesian components, yields rows
whose Laplacian (in raw monomials) is 0 for all L ≤ 6 — the check that would have
caught this. Also add a cross-code f-shell energy gate (water/cc-pVTZ spherical RHF
vs PySCF ~1e-8); the existing high-L gates are cross-ENGINE (all engines share the
transform, so they agreed while wrong).

---

## (Historical, resolved) Investigation trail

Two prior hypotheses were **falsified** (see history at the bottom). This v2 rescopes
against what is now proven, with a clean entry point.

## The defect (unchanged, reconfirmed)

Water RHF, identical geometry, matched basis coefficients:

| Basis | shells | Planck vs PySCF |
|---|---|---|
| STO-3G, 6-31G*, cc-pVDZ | ≤ d | ~1e-9 (clean) |
| **cc-pVTZ** | +**f** | **2.14e-5** |

Planck's cc-pVTZ RHF = −76.0571775438 is **stable** (identical across hcore/SAD
guess and DIIS dim ∈ {8,12}) and **2.14e-5 BELOW** PySCF's −76.0571561486.
**Below the variational minimum** in a same-spanning basis is the key anomaly — it
means the energy is not a consistent `tr(P·H)` with a matching `S`; the integrals
Planck feeds SCF are subtly inconsistent, not merely a different valid basis.

## What is PROVEN (do not re-litigate)

1. **The f cart→sph transform is correct.** Planck's f-shell span is identical to
   PySCF's (row-space projector difference 1e-16); its transform rows are pure ℓ=3
   harmonics (Laplacian = 0); and the transform choice is energy-INVARIANT (SCF
   re-orthogonalizes via S). The `C·M·Cᵀ ≠ I` non-orthonormality measured earlier
   is a valid non-orthonormal basis of the CORRECT space — SCF handles it. A
   metric-correct (Löwdin) rewrite of the transform was implemented and is a
   **no-op for the energy** (reverted).

2. **The Cartesian 1e integrals are correct.** Planck-vs-PySCF Cartesian `S⁻¹T`,
   `S⁻¹V`, `S⁻¹H_core` generalized eigenvalue spectra (normalization-invariant)
   agree to ~1e-6.

3. **The defect is already in the 1-ELECTRON spherical problem.** Solving the core
   generalized problem `H_core c = ε S c` in Python from Planck's dumped SPHERICAL
   S and H_core:
   - core-guess 1e energy `tr(P·H)`: Planck −138.72298678 vs PySCF −138.72300069
     → **1.4e-5 off, before any 2e or SCF**.
   - core orbital-energy spectrum: ~3e-5 on occupied levels, up to 3e-2 on
     f-heavy virtuals.
   - Planck's S is positive-definite (min eig 2.6e-3, well-conditioned) — not a
     linear-dependence artifact.

So: **Cartesian integrals correct + transform span-correct, but the SPHERICAL
1e matrices are off ~1e-4/1.4e-5.** The inconsistency is introduced between the
(correct) Cartesian integrals and the (correct-span) transform — most likely in
HOW the transform is applied/normalized at runtime, in a way that makes S_sph and
H_sph mutually inconsistent for f.

## The measurement confound to respect

Raw-matrix cross-code comparisons (S_sph eigenvalues, tr(H_sph)) are **confounded**
by the spherical-harmonic normalization CONVENTION: Planck and PySCF normalize the
real solid harmonics differently, so raw matrices differ by O(1) while being
physically equivalent. Confirmed: cc-pVDZ shows a large S_sph-eigenvalue difference
(0.73) yet matching energy. ONLY convention-invariant quantities are valid:
generalized eigenvalue spectra, `tr(P·H)` for a FIXED physical density, or the
energy itself. Do NOT diff raw AO matrices across codes.

## Investigation steps

### G1 — Isolate: does Planck's SPHERICAL S,H reproduce its own energy, and is it self-consistent? (~S)

The 1e core spectrum already shows the ~1.4e-5 (step 3 above). Extend it: verify
`S_sph = C · S_cart · Cᵀ` and `H_sph = C · H_cart · Cᵀ` with the SAME runtime C,
by recomputing both in Python from Planck's dumped Cartesian S/H and Planck's C
(dump `_cart_to_sph` too), and comparing to Planck's dumped S_sph/H_sph.
- **If the recomputed S_sph/H_sph match Planck's dumped ones** → the transform is
  applied consistently; the error is in the Cartesian→spherical mapping being
  metric-inconsistent for f (see G2).
- **If they DON'T match** → the runtime applies C differently to S vs H (e.g. the
  row-normalization at `hf_driver.cpp:656` scales S but not H consistently). That
  would be the bug directly.

### G2 — The metric-inconsistency test (~S, the likely core)

The runtime row-normalizes C so `diag(C·S_cart·Cᵀ) = 1` (`hf_driver.cpp:652-660`:
`norm2 = (C S_cart Cᵀ)_mm`, then `C.row(m) /= sqrt(norm2)`). This normalizes each
spherical function against the CARTESIAN overlap. But the f cart→sph rows are NOT
orthogonal in the Cartesian metric (the `C·M·Cᵀ` off-diagonals ≠ 0, proven). So
after per-row normalization, `S_sph` has unit diagonal but **off-diagonal f-f
overlaps that a metric-correct spherical basis would not have** — and crucially,
the SAME C applied to H_cart gives an H_sph that is NOT the H of an orthonormal
spherical basis. The generalized problem `H_sph c = ε S_sph c` is still solvable,
but `tr(P·H_sph)` is taken against this metric-inconsistent pair.

- **Test:** build the CORRECT spherical S,H by transforming Planck's Cartesian
  S,H with the METRIC-correct C (Löwdin under the Cartesian metric M, computed in
  Python), solve the core problem, and check the 1e spectrum now matches PySCF
  ~1e-9. If yes: the fix is to apply the metric-correct C **to the integrals**
  (not the energy-invariant basis choice tried in the reverted F3 — the
  difference is that here it changes the actual S_sph/H_sph fed to SCF, not just
  the intra-span orthogonalization).
- This reconciles the F3 "no-op" result: F3 changed `_cart_to_sph` but the runtime
  RE-normalized it back via `hf_driver.cpp:656`, undoing the fix. The real fix
  must survive (or replace) that row-normalization.

### G3 — Confirm end-to-end + fix (~M)

Once G1/G2 name the exact inconsistency (transform application vs row-normalization
vs metric), fix it and verify:
- water/cc-pVTZ RHF matches PySCF ~1e-9 (from 2.14e-5).
- Planck cc-pVTZ converges in ~normal iteration count (currently 83–142, a sign of
  the metric inconsistency stressing DIIS).
- No regression on d-shell spherical cases (6-31g*, cc-pVDZ) or the spherical
  regression suite.

### G4 — Tests (~S)

- The spherical transform tests (`tests/spherical_transform.cpp`) only check
  harmonic purity (L≤2) + rank (L≥3) — too weak. Add a check that the PRODUCTION
  transform, applied to a reference Cartesian overlap, yields an S_sph whose
  generalized spectrum matches the recurrence-oracle transform's (an
  energy-relevant, convention-invariant check).
- Add a cross-code f-shell energy gate: water/cc-pVTZ spherical RHF PySCF-anchored
  to ~1e-8. Currently the spherical gates stop at d, and the high-L gates are
  cross-ENGINE (all engines share the transform, so they agree while wrong) —
  a cross-CODE f gate is the missing tripwire.

## Prime hypothesis (to confirm/kill in G2)

The runtime per-row normalization of `_cart_to_sph` against the Cartesian overlap
(`hf_driver.cpp:656`) makes `diag(S_sph)=1` but leaves the f functions
metric-inconsistent — H_sph and S_sph are transformed by a C that is not a proper
metric-orthonormal map, so the 1e problem they define sits ~1e-5 off (and below
variational). The fix operates on the integral transform, and must not be undone
by the row-normalization — which is exactly why the earlier `cart_to_sph_block`
edit (F3) was a no-op.

## Non-goals

- Re-testing the transform's span/harmonic purity — proven correct.
- The cc-pVTZ Cartesian non-convergence (separate near-linear-dependence issue).

## History (falsified hypotheses, kept as guardrails)

- **H1: basis coefficient precision** — true for STO-3G only (10-digit GBS vs
  8-digit PySCF), proven benign by coefficient-matching to 1.8e-14. NOT the f bug.
- **H2: cart→sph transform contamination / metric-free pinv** — the `C·M·Cᵀ≠I`
  measurement was real but energy-IRRELEVANT (span-preserving; SCF re-orthogonalizes).
  The transform is span-correct and pure-harmonic. A metric-correct rewrite of
  `cart_to_sph_block` was a no-op (reverted) because the runtime row-normalization
  re-imposes its own scaling. The lesson: check energy-RELEVANCE (span, invariant
  spectrum) before trusting a raw-matrix "defect".
