# SA-CASSCF Path Forward (Deliverables + Regression Gates)

Date: 2026-04-05 (updated from 2026-04-03 original)

Derived from:
- `docs/casscf_remaining_work.md`
- `docs/plan_unified_sa_casscf.md`

---

## Bugs confirmed by PySCF reference run (2026-04-05) — fix first

### Bug 1 — `guess hcore` + `use_symm true` breaks the HF for water CAS(4,4)

The three single-root water inputs all use `guess hcore` + `use_symm true`. PySCF
shows the Planck HF converges to −74.4581 Eh instead of the correct −74.9629 Eh
(~0.5 Eh error). The CASSCF inherits this wrong orbital basis and gives wrong
energies by ~0.5 Eh. The SA-2 water input (`use_symm false`, no guess) gives the
correct HF energy, confirming the bug is in the hcore+symm interaction.

**Action:** remove `guess hcore` from `water_cas44_sto3g.hfinp`,
`water_cas44_631g.hfinp`, `water_cas44_ccpvdz.hfinp`; re-run; update gate values.

Correct PySCF targets (Cartesian basis): −75.0084054420, −76.0370099226, −76.0781256226 Eh.

### Bug 2 — SA optimizer converges to wrong local minimum (water SA-2)

Both Planck and PySCF start from the correct RHF (E ≈ −74.9629 Eh), same CAS(4,4)
active space. PySCF finds E_SA = −74.7877865139 Eh; Planck finds −74.7751279351 Eh
(0.013 Eh higher). PySCF finds a better SA minimum. This is consistent with the
per-root averaged κ issue (Milestone 2 below) — the optimizer does not solve the
true SA stationarity system.

**Action:** treat as validation that Milestone 2 is needed. After M2 lands, verify
water SA-2 converges to ≈ −74.7877 Eh.

### Bug 3 — Twisted ethylene SA-2 active space uncertain

The ethylene SA-2 input uses `guess hcore`, giving Planck HF = −76.6806 vs correct
−76.7930 Eh. Planck finds E_SA = −77.0034 Eh (0.010 Eh lower than PySCF −76.9930).
The lower Planck energy likely reflects a different active-orbital basis from the
wrong HF starting point, not a genuinely better minimum.

**Action:** re-run without `guess hcore`; compare to PySCF target −76.9930595776 Eh.

---

## Current reality (what the code does today)

The SA optimizer in `src/post_hf/casscf/casscf.cpp` is root-resolved through the
macro/micro scaffold and has a production coupled orbital/CI candidate plus the
correct SA convergence semantics.

Convergence is now gated on `||g_SA||_inf < tol` where `g_SA = Σ_I w_I g_I` is the
state-averaged orbital gradient (built in `build_weighted_root_orbital_gradient`).
The iteration table prints this as "SA Grad"; per-root screens appear as "MaxRootG"
for diagnostics only.

The remaining problem is in the *step construction*: the coupled orbital/CI step is
still built per-root and then averaged, not solved with one shared `κ` across all
roots simultaneously. This is architecturally correct for state-specific CASSCF but
is not the true SA second-order step, which couples a single `κ` to all roots' CI
responses at once.

---

## Milestones

### Milestone 1 — Fix SA convergence semantics ✓ DONE (c97a754, 2026-04-04)

**What was done**
- `sa_gradient_converged()` at `casscf.cpp:321` now tests `sa_gnorm < tol`.
- `st.gnorm` is set to `st.g_orb.cwiseAbs().maxCoeff()` where `st.g_orb` is the
  state-averaged gradient `Σ_I w_I g_I` (line 698 + 708).
- Convergence gates at lines 904–911 and 1306–1313 both use `sa_gradient_converged`.
- Stagnation detection (`sa_gradient_progress_flat`) also uses the SA gradient norm.
- Iteration table column renamed to "SA Grad"/"MaxRootG".
- Per-macroiteration wall-clock timing added via `std::chrono` (line 1265–1269).

**Validation**
- All 11 benchmark cases pass, including both 2-root SA cases
  (`water_cas44_sto3g_sa2`, `ethylene_cas44_sto3g_sa2`).
- `nroots=1` results unchanged.

---

### Milestone 2 — True SA coupled solve with a shared orbital step (shared κ)

**Current state (NOT DONE)**

`build_root_resolved_coupled_step_set` (lines 933–991) solves a per-root coupled
orbital/CI step for each root independently, then reduces to a weighted sum:

```
kappa_SA = Σ_I w_I kappa_I
```

This is not a shared-κ solve. The per-root solves each minimize their own root's
response residual; the average is taken only after. A true SA coupled solve would
hold one `κ` fixed across all roots and simultaneously minimize the residual of the
SA stationarity system:

```
R(κ) = g_SA + H_oo κ + Σ_I w_I G_oc c1_I(κ) = 0
```

where `c1_I(κ)` is the CI response of root I to the shared `κ`.

Note also that the acceptance merit function at line 1169 still uses
`trial.weighted_root_gnorm` (a per-root gradient screen), not the SA gradient
`trial.gnorm`. This is inconsistent with the convergence semantics fixed in M1.

**Deliverables**
- Implement a solver path that holds one shared `κ` and solves all roots' CI
  responses to that shared `κ` simultaneously.
- The SA orbital residual is `g_SA + H_oo κ + Σ_I w_I G_oc c1_I`; minimize this
  over `κ` (matrix-free is fine, keep diagonal blocks as preconditioners).
- Replace `build_root_resolved_coupled_step_set` with the shared-κ SA solve as the
  production `sa-coupled` direction.
- Fix the acceptance merit function to use the SA gradient norm (`trial.gnorm`), not
  the per-root weighted screen (`trial.weighted_root_gnorm`).

**Acceptance**
- The 2-root SA ethylene gate converges to a genuinely multiroot stationary point
  where `||g_SA||_inf < tol` rather than relying on the stagnation plateau escape.
- `nroots=1` results unchanged.

---

### Milestone 3 — Orbital Hessian action upgrade (production δg_SA[·])

**Current state (NOT DONE)**

`hessian_action()` in `orbital.cpp:178` is diagonal:
```
(HR)_pq = (F_sum[p,p] - F_sum[q,q]) * R_pq
```
This is used as a preconditioner inside the coupled solve, not a faithful OO block
model. The `δg_SA[R]` action that would include core response, active density
response, Q-matrix derivative, and commutator terms is not implemented.

**Deliverables**
- Add a production Hessian-vector action `R → δg_SA[R]` including:
  - core/active density response to the orbital rotation R
  - Q-matrix derivative
  - commutator terms (per the unified SA plan)
- Use it inside the shared-κ coupled SA solve (Krylov method with diagonal
  preconditioner for its inner iterations).

**Acceptance**
- A finite-difference Hessian-vector check passes on a small active-space reference
  system (add to `tests/casscf_internal.cpp`).
- Convergence rate for the coupled solve improves relative to the diagonal-only
  preconditioner baseline.

---

### Milestone 4 — Simplify optimizer once the coupled SA solve is robust

**Current state (PARTIAL — first simplification pass landed in 2c5038a)**

The production candidate family has been reduced from the larger always-on AH/mix/
gradient-variant family to:
- `sa-coupled` (primary)
- `sa-grad-fallback`
- `numeric-newton` as small-space escape hatch (`npairs <= 64`)
- `sa-diag-fallback`, per-root rescue candidates, and pair probes under stagnation

The large always-on family is gone. The remaining escape hatches are justified while
the coupled solve is being hardened against the current benchmark suite.

**Remaining deliverables**
- Once M2 is in place, decide whether `numeric-newton` can be promoted to
  debug-only for small spaces without regressing the solver gate.
- Once M2+M3 are in place, reduce or eliminate the stagnation-only rescue family
  (`sa-diag-fallback`, per-root candidates, pair probes).
- Fix the acceptance merit function (also needed for M2): use SA gradient in merit,
  not per-root weighted screen.
- Ensure transcript clearly distinguishes debug vs production algorithmic paths.

**Acceptance**
- Production convergence trajectory is attributable to one main step (shared-κ SA
  coupled) plus one explicit fallback.
- Regressions become attributable to a single optimizer path rather than several
  interacting candidates.

---

### Milestone 5 — Add a small trusted external reference suite (PySCF-backed)

**Current state (NOT DONE — scaffold present, values TBD)**

`tests/inputs/casscf_tests/pyscf_reference_energies.md` has been created and
contains the PySCF script template and tolerance spec. All SA entries are TBD.
No automated PySCF comparison exists.

**Deliverables**
- Run the PySCF script from `pyscf_reference_energies.md` and fill in the TBD
  entries for at least:
  - H₂ CAS(2,2)/STO-3G (SS)
  - LiH CAS(2,2)/STO-3G (SS)
  - H₂O CAS(4,4)/STO-3G (SS and SA-2)
  - twisted ethylene CAS(4,4)/STO-3G (SA-2)
- Add automated `ctest`-runnable comparisons (skip if PySCF unavailable).
- Assert SA-weighted energy matches PySCF within 1e-5 Eh.

**Acceptance**
- Reference tests pass and catch SA convergence regressions.

---

## Regression gates

### Existing mandatory gates (keep mandatory)

1. **Internal unit harness** — `planck-casscf-internal` (`tests/casscf_internal.cpp`)
2. **Manual CASSCF fixture gate** — all `.hfinp` inputs in `tests/inputs/casscf_tests/`

Current gate status (2026-04-04 rerun). Entries marked ⚠ are confirmed wrong by
PySCF cross-check and must be updated after the hcore+symm bug is fixed:

| Input | E(CASSCF) / Eh | PySCF ref / Eh | Note |
|---|---|---|---|
| ethylene_cas44_sto3g_sa2 (SA-2) | −77.0034974301 | −76.9930595776 | ⚠ hcore bug |
| ethylene_casscf_321g | −77.5145223872 | −77.5145223959 | OK |
| ethylene_casscf_321g_nroot2 | −77.5145223872 | — | OK |
| ethylene_casscf_ccpvdz | −77.9524855976 | −77.9524856210 | OK |
| h2_cas22_sto3g | −1.1372838351 | −1.1372838345 | OK |
| lih_cas22_sto3g | −7.8811184797 | −7.8811184639 | OK |
| water_cas44_631g | −75.5497490402 | −76.0370099226 | ⚠ wrong (hcore+symm) |
| water_cas44_b1 | −74.2879452324 | — | not yet cross-checked |
| water_cas44_ccpvdz | −75.6045806122 | −76.0781256226 | ⚠ wrong (hcore+symm) |
| water_cas44_sto3g | −74.4700757755 | −75.0084054420 | ⚠ wrong (hcore+symm) |
| water_cas44_sto3g_sa2 (SA-2) | −74.7751279351 | −74.7877865139 | ⚠ SA local-min gap |

The ⚠ entries must not be used as convergence targets until re-run with correct inputs. The
"expected-failure reproducer → flip to pass" lifecycle from the original plan
was skipped; the fixtures went straight to passing after M1.

### Add once M2 lands: a strict SA stationarity assertion

Currently the SA fixtures converge, but it is not verified that convergence was
declared because `||g_SA||_inf < tol` rather than because the stagnation plateau
escape fired. Once M2 is done, add a check:

- Verify the last logged `sa_g=` diagnostic is `< tol_mcscf_grad`.
- Verify the last macro iteration did not fire the plateau escape path.

### Make the regression runner SA-aware (still needed)

`tests/run_regressions.py` does not yet parse SA-specific diagnostics from the
macro-iteration log.

**Add metrics to parse:**
- `sa_g` — from `sa_g=` in macro diagnostic line
- `root_screen_g` — from `root_screen_g=` in macro diagnostic line
- `max_root_g` — from `max_root_g=` in macro diagnostic line

**Then assert:**
- `sa_g <= tol_mcscf_grad` (this is the SA stationarity check)
- Keep root metrics for diagnostics only

---

## Required verification loop after each solver-facing change

```bash
cmake --build build --target hartree-fock planck-casscf-internal -j4
./build/planck-casscf-internal
for input in tests/inputs/casscf_tests/*.hfinp; do
    ./build/hartree-fock "$input" | tee "${input%.hfinp}.rerun.log"
done
```

Review every case. Do not call work done if one case improves while another regresses.
