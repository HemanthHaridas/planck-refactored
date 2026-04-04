# SA-CASSCF Path Forward (Deliverables + Regression Gates)

Date: 2026-04-03

This is a concrete execution plan derived from:
- `docs/casscf_remaining_work.md`
- `docs/plan_unified_sa_casscf.md`

It is written to match the current repository implementation and testing
infrastructure.

---

## Current reality (what the code does today)

The SA optimizer in `src/post_hf/casscf/casscf.cpp` is root-resolved through the
macro/micro scaffold and now has a production coupled orbital/CI candidate.

However, SA convergence/plateau behavior is still dominated by *root screens*:
- `weighted_root_gnorm` (weighted sum of per-root `||g_I||_inf`)
- `max_root_gnorm` (max over roots of `||g_I||_inf`)

These are useful diagnostics/guards, but they are **not** the SA stationarity
condition for the state-averaged objective.

---

## Core target (what “correct SA-CASSCF” should mean)

We minimize the state-averaged energy:

```
E_SA = Σ_I w_I E_I
```

The orbital stationarity condition is:

```
g_SA = Σ_I w_I g_I = 0
```

This **does not require** `g_I = 0` for every root.

Therefore the convergence check for SA must be expressed in terms of `||g_SA||`
(with rootwise gradients retained for diagnostics/guards, not as the definition
of stationarity).

---

## Milestones (ordered, with concrete deliverables)

### Milestone 1 — Fix SA convergence semantics (critical unblocker)

**Deliverables**
- In `src/post_hf/casscf/casscf.cpp`, update stopping/plateau logic so the
  gradient convergence criterion is based on the **SA gradient norm**
  `||g_SA||`, i.e. the norm of the already-built state-averaged gradient matrix
  `st_current.g_orb`.
- Keep `weighted_root_gnorm` and `max_root_gnorm` as diagnostics (and optionally
  as a “do not explode a root” safety guard), but do not require them to be
  `< tol_mcscf_grad` to declare SA stationarity.
- Update the iteration table semantics so the “Grad” quantity reported to users
  corresponds to `||g_SA||` (today the iteration table prints root-screen values
  in the CASSCF columns, which is misleading for SA runs).

**Acceptance**
- `nroots=1` results unchanged on existing CASSCF fixtures.
- `nroots>1` no longer fails to converge *purely* because `max_root_gnorm` is
  nonzero while `||g_SA||` is small.

---

### Milestone 2 — True SA coupled solve with a shared orbital step (shared κ)

Today the `sa-coupled` direction is built by:
1. Solving a per-root coupled orbital/CI step
2. Reducing/averaging those per-root orbital steps by weights

This is not equivalent to solving the SA stationarity system with a single
shared `κ`.

**Deliverables**
- Implement a solver path that computes **one shared `κ`** coupled to the CI
  response of all roots (matrix-free is fine; keep diagonal blocks as
  preconditioners).
- Replace the current “weighted sum of per-root coupled steps” with the shared-κ
  SA coupled solve as the production `sa-coupled` direction.

**Acceptance**
- A 2-root SA ethylene smoke converges to a multiroot stationary point (does not
  merely “fail honestly” by running out the macro iteration budget).

---

### Milestone 3 — Orbital Hessian action upgrade (production δg_SA[·])

The diagonal orbital “Hessian” is currently used as a preconditioner, not a
faithful OO block model. After Milestone 2 is in place, the next quality jump is
to provide a real Hessian *action* for the SA gradient.

**Deliverables**
- Add a production Hessian-vector action:
  - `R -> δg_SA[R]`
  - including core response, active density response, `Q` derivative, and
    commutator terms (per the unified SA plan).
- Use it inside the coupled SA solve (e.g., Krylov method with diagonal
  preconditioner).

**Acceptance**
- A finite-difference Hessian-vector check passes on a small active-space
  reference system.

---

### Milestone 4 — Simplify optimizer once the coupled SA solve is robust

The current driver still carries multiple escape hatches (numeric Newton on
small spaces, stagnation probes, etc.). These are currently justified while the
coupled solve hardens, but they complicate regressions.

**Deliverables**
- Make `numeric-newton`, probe steps, and other stagnation rescues debug-only
  (or remove) once the shared-κ solve has an explicit and reliable
  globalization/trust-region story.
- Ensure transcripts clearly identify which algorithmic path was used.

**Acceptance**
- Regressions become attributable to one main production solver path.

---

### Milestone 5 — Add a small trusted external reference suite (PySCF-backed)

This is the most important missing safety net for future refactors.

**Deliverables**
- Add optional (skip-if-missing) PySCF-backed comparisons for:
  - CASCI energies
  - CASSCF energies
  - SA energies (≥ 2 roots)
  - at least one additional invariant (e.g., natural occupations, or an orbital
    gradient finite-difference check)
  - root tracking across a small geometry distortion / near-crossing example

Recommended systems (from remaining-work notes):
- H2 CAS(2,2)
- LiH CAS(2,2)
- H2O CAS(4,4) or CAS(6,4)
- twisted ethylene
- one near-crossing SA example

**Acceptance**
- Reference tests pass within tolerances and catch regressions in SA behavior.

---

## Regression gates (what to run, and what to add)

### Existing mandatory gates (keep them mandatory)

1. **Internal unit harness**
   - Target: `planck-casscf-internal`
   - Source: `tests/casscf_internal.cpp`
   - Runs via CTest when `BUILD_TESTING=ON`.

2. **Manual CASSCF fixture gate**
   - Folder: `tests/inputs/casscf_tests`
   - Requirement: run all `.hfinp` inputs after any solver-facing CASSCF change.

---

### Add immediately: an SA expected-failure reproducer (then flip to pass)

Rationale: today there is no checked-in `nroots > 1` end-to-end fixture; add one
early so the SA plateau/stall behavior is tracked.

**Add**
- An SA input such as:
  - `tests/inputs/sa_ethylene_casscf_cas44_sto3g_2root.hfinp`
  - with `nactele 4`, `nactorb 4`, `nroots 2`, and `weights 0.5 0.5`

**Regression case lifecycle**
- Initially: mark as **expected failure**
  - expect exit code `1`
  - require output containing a clear non-convergence diagnostic (e.g.
    `"CASSCF : Failed :"` and `"did not converge"`).
- After Milestone 1+2: flip to **expected success**
  - expect exit code `0`
  - require `"CASSCF :                      Converged."`

---

### Make the regression runner SA-aware (high leverage)

The current end-to-end regression runner (`tests/run_regressions.py`) already
extracts CASSCF energies. For SA work, it should also extract the explicit SA
gradient diagnostics that the driver prints, so we can assert the correct
convergence semantics in a stable way.

**Add metrics to parse**
- `sa_g`
- `root_screen_g`
- `max_root_g`

**Then assert**
- `sa_g <= tol_mcscf_grad` (this is the stationarity check)
- keep root metrics for monitoring/diagnostics only

