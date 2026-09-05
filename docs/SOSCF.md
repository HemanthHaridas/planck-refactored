# RHF SOSCF Architecture Note

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Does a second-order (augmented-Hessian) orbital step reduce RHF SCF iteration count without changing the converged energy, and what does it take to extend it beyond RHF?**

## Short answer

Yes. SOSCF is landed for RHF: it reuses the existing CASSCF augmented-Hessian
(CIAH) solver and the existing RHF orbital Hessian (`build_rhf_cphf_matrix`)
completely unchanged, needing only the callbacks that supply them with a
gradient and a Hessian-vector product. Three real defects were found and
fixed by direct measurement during the work — none of them in the reused
machinery itself, all in how this new caller drove it. SOSCF composes
cleanly with the SAD initial guess (verified, no interaction). It does not
yet extend to UHF, ROHF, or DFT — UHF is a mechanical port of code that
already exists in a different wrapper, DFT needs an existing XC-kernel
builder wired into the Hessian-vector product, and ROHF needs new theory
that does not exist anywhere in this codebase yet.

## Where the logic lives

- `src/scf/scf.cpp` — the SOSCF branch inside `run_rhf`
- `src/post_hf/rhf_response.cpp` — `build_rhf_cphf_matrix`, the RHF orbital Hessian (reused unchanged)
- `src/post_hf/casscf/aug-hessian.h` — `solve_augmented_hessian`, the generic CIAH solver (reused unchanged)
- `src/post_hf/casscf/orbital.cpp` — `apply_orbital_rotation`, the Cayley-transform-plus-Löwdin-cleanup step (reused unchanged)
- `src/base/types.h` — `scf_soscf_start`, `scf_soscf_cycles`, `scf_soscf_diis_tol`, `scf_soscf_min_iter`
- `src/post_hf/uhf_response.cpp` — `solve_uhf_cphf`, the UHF analogue (not yet factored out for reuse)

## What invariants matter

### 1. The orbital gradient and Hessian must be evaluated in the previous iteration's basis against the current Fock

Diagonalizing the current Fock first and then reading the occupied-virtual
block off the result gives a gradient that is zero by construction — that is
what diagonalizing means — regardless of how far the SCF actually is from
convergence. A run built this way "worked" in the sense of reaching the
right energy, but only because the resulting Newton step was real but
minuscule; it took *more* iterations than plain DIIS, the opposite of the
point.

Design rule:

- Persist the MO basis from the previous iteration (`C_soscf_prev`,
  `eps_soscf_prev`) and build the gradient/Hessian against it, contracted
  with the *current* Fock. The Newton step produces the new `C` directly;
  no diagonalization happens on a SOSCF iteration at all.

### 2. The gradient and Hessian must be scaled consistently with each other, not just individually plausible

A Newton step depends only on the ratio `g/H`. Deriving the gradient
prefactor by pattern-matching a reference implementation's own convention
(PySCF's `g = 2·F_mo`) while pairing it with a *different* Hessian (this
codebase's `build_rhf_cphf_matrix`, built for the MP2 Z-vector equation, not
independently re-derived to match PySCF's own Hessian scale) silently
produces a step that is off by whatever factor separates the two
conventions — in this case exactly 2x, which compounds every iteration.

Design rule:

- Do not trust a gradient/Hessian pairing derived by matching each side to a
  different external reference independently. Verify the pairing directly:
  perturb the orbitals by a small `κ` in one direction, compute the actual
  energy `E(κ)` numerically, and check that the analytic gradient and
  Hessian reproduce the finite-difference gradient and second derivative of
  that same numerical energy. This is the only check that cannot be fooled
  by two individually-plausible-looking but mutually-inconsistent
  conventions. Kept as a permanent, opt-in regression probe
  (`PLANCK_SOSCF_FD_CHECK`) rather than deleted after use.

### 3. A generic solver's tuning defaults do not transfer across callers with different problem scales

`aug-hessian.h`'s `ah_start_tol` default (`2.5`) is a fixed absolute residual
threshold tuned for CASSCF's orbital gradients, which run `O(1–10)`. RHF
SOSCF's gradient is typically `O(0.001–1)` — two to three orders of
magnitude smaller — so the same constant is satisfied after exactly one
Krylov iteration on every single call, regardless of how far from converged
the run actually is. The AH solver never gets to refine its subspace into a
genuine multi-vector Newton direction, so what looks like "the Newton
method" is actually a sequence of single-vector Krylov approximations —
correct in direction, but converging only linearly rather than the
superlinear rate a real Newton step gives near the minimum. Measured
directly: pure SOSCF took roughly 290 iterations to reach machine precision
on water/6-31G before this was found and fixed; 8 iterations after.

Design rule:

- Scale a reused solver's own internal tolerances to the caller's actual
  problem scale (here, `0.1·‖g‖`) rather than trusting a default tuned for
  a different caller. A generic solver's callback interface being reusable
  does not mean its tuning constants are.

### 4. A missing trust-region cap lets an early, unreliable Newton step diverge

Every existing CASSCF caller of `solve_augmented_hessian` caps the returned
step element-wise before applying it, because the Krylov subspace can
produce a large step where the local quadratic model is a poor
approximation to the true energy surface — expected far from convergence,
which is exactly where SOSCF's early iterations operate. Omitting the cap
let early iterations take unboundedly large rotations that made the
gradient worse each step rather than better.

Design rule:

- Any new caller of `solve_augmented_hessian` must cap the step the same way
  the existing callers do (element-wise at a fixed bound, `0.20` here,
  matching CASSCF's own `mcscf_max_rot` default) rather than assuming the
  solver self-limits.

### 5. SOSCF is a transient accelerator, not a permanent replacement for DIIS

Matching ORCA's own SOSCF/DIIS handoff: SCF starts with DIIS, switches to a
small, fixed number of second-order iterations once triggered, then hands
control back to DIIS to finish. Run to full, unbounded convergence with no
handoff at all, SOSCF does reach the same energy as DIIS (verified on
water/6-31G and H2/6-31G) — but converges only linearly in that mode even
after the tolerance fix above, so the transient-window default remains
faster in practice on the systems checked so far.

Design rule:

- DIIS's subspace must be cleared on handoff back from SOSCF — its stored
  `(F, error)` history predates the orbital rotations SOSCF just applied and
  would otherwise corrupt the next extrapolation.

## What was fixed

The three defects above were each confirmed by an actual divergent or
slowly-converging run, not found by static code review:

1. SOSCF was originally layered on top of DIIS's own diagonalization every
   iteration (double-stepping the orbitals) instead of replacing it for the
   duration of the window.
2. The orbital gradient/Hessian pairing was corrected from
   `g = 2·F_mo` (paired with an unscaled Hessian, a 2x-too-large step) to
   `g = F_mo` paired with the same unscaled Hessian — settled by the
   finite-difference check in invariant 2, not by re-deriving from a
   reference implementation a third time.
3. `aug-hessian.h`'s `ah_start_tol` is now scaled to `max(1e-8, 0.1·‖g‖)`
   for the SOSCF caller specifically, rather than left at the CASSCF-tuned
   fixed default.

## Validation strategy that should remain in place

- Same-energy check to all 10 printed digits against pure DIIS, on every
  mode: fixed-iteration window, DIIS-error-criterion window, and pure
  unbounded SOSCF. Verified on water/6-31G and H2/6-31G.
- The `PLANCK_SOSCF_FD_CHECK` finite-difference probe (invariant 2) — cheap,
  opt-in, and the only check that would have caught the scale-mismatch
  defect; keep it rather than deleting it now that the defect is fixed.
- Smoke regression suite unaffected with SOSCF off (`scf_soscf_start=0`,
  `scf_soscf_diis_tol=0.0`, the unconditional defaults): 33/35, matching the
  pre-existing baseline exactly (2 known unrelated CC/`rccsdt` routing
  failures on this build).
- SAD + SOSCF composition verified directly: converges to the identical
  energy as SAD + DIIS on water/6-31G. SAD only sets the initial density
  before the iteration loop starts; SOSCF's own state is independent of how
  the density was initialized, so no interaction was expected or found.

## What was measured but not resolved

**The full `scale.json` baseline ladder (nb ∈ {104, 156, 208, 312, 416},
serial, water chains) was not reproduced this session** — the source data
came from a 32-core cluster node (`notch386`), and the largest points
(HF nb=416 at 6263 s serial, DFT nb=312/416 at 15505 s/58683 s serial) are
not reproducible on a laptop in reasonable time. Correctness was instead
verified directly against finite differences of the true energy (invariant
2) and against exact energy agreement with DIIS, which does not depend on
reproducing the cluster ladder. The DIIS-error switch criterion (S3's
original goal) was verified on small systems only: water/6-31G (14 vs 15
DIIS iterations) and H2/6-31G (5 vs 11) — satisfying the "does not regress
small systems" requirement, but the large-`nb` iteration-count reduction
this work was originally motivated by has not been measured.

## What remains: UHF, ROHF, DFT

Investigated directly by reading the relevant code, not assumed:

- **UHF — a mechanical extension, not a new derivation.** `solve_uhf_cphf`
  (`src/post_hf/uhf_response.cpp`) already materializes the full dense
  coupled α/β Jacobian matrix internally — the same object
  `build_rhf_cphf_matrix` is for RHF — but it is currently wrapped inside a
  Z-vector solve (`A·z = -rhs` for the MP2 gradient) rather than exposed as
  a standalone builder. The work is the same shape as the RHF split between
  `build_rhf_cphf_matrix` and `solve_rhf_cphf`: factor the matrix build out,
  then pack a joint α/β gradient and step. The Cayley rotation and
  semicanonicalization would need separate α and β blocks throughout.
- **ROHF — new theory, not a port.** There is no ROHF orbital-response or
  CPHF machinery anywhere in this codebase (confirmed by direct search),
  consistent with ROHF-MP2, ROHF stability, and ROHF PCM all remaining
  unsupported for the same underlying reason. ROHF orbitals diagonalize the
  effective Roothaan Fock, not separate per-spin canonical Focks — the same
  subtlety that already forced the ROHF analytic *gradient* to use its own
  energy-weighted-density form (`W = P^α F^α P^α + P^β F^β P^β`) instead of
  reusing UHF's. A ROHF orbital Hessian needs its own derivation; nothing
  here transfers from RHF or UHF.
- **DFT — the hardest missing piece already exists, but isn't wired in.**
  The KS SCF loop (`src/dft/driver.cpp`) is a separate implementation with
  its own DIIS state, not routed through `run_rhf`, so SOSCF is not
  reachable there yet at all. But an XC kernel builder
  (`build_closed_shell_xc_kernel_blocks` / `build_unrestricted_xc_kernel_blocks`)
  already exists in-tree, built for TDDFT's linear response — it is exactly
  the extra term a DFT orbital Hessian needs added to the RHF/UHF form. The
  remaining work is wiring the KS loop through a shared SOSCF path (or
  duplicating the branch there) and reusing this existing kernel, not
  deriving a new one.

## What NOT to do

- **Do not write a second augmented-Hessian solver.** `aug-hessian.h` is
  generic by construction and validated by the 11/11 CASSCF gate suite. If
  it needs a change to serve a new caller, change it there.
- **Do not touch the DIIS path's behavior when SOSCF is off.** The default
  emit (`scf_soscf_start=0`, `scf_soscf_diis_tol=0.0`) must stay bitwise
  identical; every existing regression depends on it.
- **Do not judge success on wall-clock alone.** The claim is iterations;
  wall-clock mixes in the per-iteration cost of the Hessian build.
- **Do not assume the UHF or ROHF orbital Hessian is a straightforward
  generalization of the RHF one without checking.** ROHF specifically is
  not — the same trap the ROHF analytic gradient already hit once.
- **Do not skip the finite-difference check when extending this to a new
  reference type or method.** It is the only check that catches a
  gradient/Hessian pairing that is individually plausible on each side but
  mutually inconsistent — exactly the defect that caused the 2x overshoot
  here, and it would not have been caught by code review alone.

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
