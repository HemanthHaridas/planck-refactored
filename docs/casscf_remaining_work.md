# CASSCF Remaining Work

## Purpose

This note is the current handoff for the next CASSCF implementation pass. It
supersedes the older patch-plan notes now that most of the P0 cleanup and the
large modular refactor have already landed in the tree.

## Current Status

The code is in much better shape than the original monolithic prototype, but it
is still not a true coupled second-order CASSCF implementation.

What has landed already:

- The old monolith has been split into dedicated `strings`, `ci`, `rdm`,
  `response`, `orbital`, and driver modules.
- RASSCF determinant screening is now enforced on the combined alpha+beta
  determinant, not independently on the spin strings.
- CI symmetry screening now uses an explicit Abelian irrep product table rather
  than the old XOR shortcut.
- Root tracking uses overlap-based reassignment with a global Hungarian-style
  maximum-overlap match instead of greedy local swaps.
- CI bitstring helpers now guard full-width shifts and the driver rejects
  active spaces beyond the supported `uint64_t` encoding limits.
- The one-body CI sigma build and the Slater-Condon Hamiltonian path now share
  the same ket-to-bra operator convention, with unit coverage for consistency.
- The public CI-response Davidson solve now has projection, restart/collapse,
  bounded-subspace handling, and best-residual reporting on truncated exits.
- The single-step CI-response fallback no longer reports false convergence.
- Active-space `Q` construction now reuses a cached transformed-integral
  container instead of retransformation inside the micro-iterations.
- The CI solver can now use a direct sigma-vector Davidson path, and this has
  internal agreement tests against dense CI on small spaces.
- The driver now carries a per-root `StateSpecificData` record through the
  macro loop, including per-root CI vectors/energies, 1-RDMs, 2-RDMs, active
  Fock contributions, `Q` contributions, and orbital-gradient contributions.
- The current averaged `gamma`, `Gamma`, `F_A`, and `g_orb` objects are now
  rebuilt as explicit weighted sums of those per-root records instead of being
  formed only from early averaged inputs.
- Response-side bilinear `Gamma1` accumulation is now expressed as an explicit
  weighted sum of per-root contributions.
- The SA micro-iteration now keeps more of the orbital-update path root-first:
  each root carries its own response RHS, CI-response vector, first-order
  2-RDM, `Q1`, CI-driven orbital correction, and updated orbital-gradient
  contribution before the driver rebuilds the weighted state-averaged gradient.
- The AH-like orbital step proposal is now also built root-first: each root
  forms its own preconditioned orbital step from its own `g_orb` and `F_A`
  data, and the driver reduces those proposals to one SA step only after that
  per-root solve. The fallback gradient step follows the same root-first
  reduction pattern.
- Numeric Newton is no longer part of the normal production path; it is behind
  `mcscf_debug_numeric_newton`.
- `calc._cas_mo_coefficients` now stays in the converged optimization basis,
  avoiding the old mismatch between stored orbitals and the solved CI gauge.
- The response layer now has an explicit `ResponseRHSMode` split between:
  - approximate commutator-only response
  - exact active-space orbital-derivative response
- The exact response RHS is now built analytically inside `response.cpp` from
  the active-space one- and two-electron Hamiltonian derivatives, and the
  response API no longer carries the older finite-difference context bundle.
- The driver now defaults to the exact orbital-derivative response RHS and
  keeps the commutator-only shortcut only behind the explicit
  `mcscf_debug_commutator_rhs` debug option.
- `tests/casscf_internal.cpp` now checks that the analytic exact RHS matches a
  finite-difference active-space Hamiltonian rotation on a small reference
  problem, and that it differs from the commutator-only shortcut when the
  two-electron Hamiltonian responds.
- `tests/casscf_internal.cpp` now also checks weighted per-root
  pair-priority/probe-ranking inputs and weighted quadratic-model diagnostics
  on small synthetic SA examples.
- `tests/casscf_internal.cpp` also covers candidate-screen sign flips and
  weighted pair-ranking edge cases where the state-averaged priority differs
  from a single root's dominant direction.
- The shared SA candidate screen now also keeps per-root AH-like and gradient
  fallback proposals as explicit trial steps, so strong single-root directions
  can still reach full reevaluation even when the immediate weighted proposal
  damps them.
- The `planck-casscf-internal` target now links the orbital/integral
  implementation it exercises, and the internal harness no longer crashes in
  the trailing root-reduction coverage block.

What is still true:

- The default optimizer is still an approximate macro/micro scaffold.
- State-averaged runs now stay root-resolved later in the micro-iteration and
  the response layer can drive an exact-RHS code path, but the overall
  optimizer is still not a true coupled SA second-order implementation.
- The exact RHS is now the default theory path, but it is still used inside the
  current diagonal-response scaffold rather than a genuine coupled orbital/CI
  step.
- The orbital Hessian used in production is still only a diagonal model.
- The default orbital-step logic still mixes several heuristics instead of one
  clean coupled solver and one explicit globalization strategy.

## Highest-Priority Next Steps

### 1. Finish the per-root state-specific SA refactor

The root-resolved scaffolding is now substantially deeper than it was in the
original handoff, but it is still not complete yet.

Today the driver keeps per-root CI/RDM/orbital-intermediate records and then
rebuilds the current averaged `gamma`, `Gamma`, `F_A`, and `g_orb` from those
records. The micro-iteration now also keeps the response-side RHS, `c1`,
`Gamma1`, `Q1`, CI-driven orbital correction, and AH-like/fallback orbital
step proposals root-resolved before reducing back to one weighted update. The
candidate screen now also keeps those per-root AH-like/fallback proposals as
explicit reevaluation candidates instead of screening only the weighted step.
That is a meaningful improvement, but the overall SA optimizer path still
collapses to shared acceptance diagnostics and one diagonal-response scaffold
too early for accurate state-averaged second-order theory.

Needed work:

- Keep `StateSpecificData` as the source of truth and extend it with the
  remaining response/coupled-step intermediates that are still missing.
- Move the remaining optimizer pieces to operate on root-resolved data first,
  then perform weighted reduction only where the theory actually permits it.
- Remove the remaining places where the SA path still falls back to shared
  acceptance diagnostics or one diagonal preconditioner too early, especially
  around coupled-step construction and merit/acceptance logic.
- Keep the current overlap-based root tracking, but make the later optimizer
  stages consume root-resolved quantities directly instead of only rebuilt
  averaged objects.

Acceptance target:

- Single-state behavior stays numerically unchanged.
- Multi-root runs no longer depend on early averaging anywhere in the orbital
  update, first candidate-generation path, or response path.

### 2. Finish integrating the exact CI-response RHS cleanly through the remaining scaffold

The response layer no longer has only the commutator shortcut: it now exposes
both an explicit approximate mode and an explicit exact orbital-derivative RHS
builder, and the driver now defaults to the exact path. The exact builder is
now analytic rather than finite-difference, so the remaining work is to keep
that exact path threaded cleanly through the rest of the SA response scaffold
instead of treating the old implementation choice as an open design decision.

Needed work:

- Continue threading the exact RHS through the now-root-resolved SA plumbing
  instead of dropping back immediately to one averaged perturbation.
- Make logs and internal control flow distinguish clearly between:
  - approximate commutator-only response
  - exact orbital-derivative response
- Keep the exact-RHS path as the production theory path while making sure
  response-side fallbacks and diagnostics do not blur it back together with the
  diagonal-response scaffold.

Acceptance target:

- The code no longer silently treats the approximate RHS as if it were the full
  second-order response theory, and approximate response stays an explicitly
  requested debug path only.
- Small reference problems continue to compare the analytic exact-response RHS
  against a finite-difference active-space Hamiltonian rotation.
- The exact path is the default theory path wherever the current SA plumbing is
  intended to approximate second-order behavior.

### 3. Build a true coupled orbital/CI step and demote the diagonal Hessian to a preconditioner

This is the core missing algorithmic step.

The production path still uses a diagonal orbital Hessian model plus response
corrections and several fallback heuristics. That is acceptable as scaffolding,
but not as the final solver architecture.

Needed work:

- Recast the current diagonal orbital Hessian as a preconditioner only.
- Implement one real coupled step:
  - matrix-free coupled Newton, or
  - genuine augmented-Hessian / norm-extended step
- Introduce explicit orbital-orbital, orbital-CI, CI-orbital, and CI-CI blocks
  in the step construction.
- Reduce the default optimizer to one main step builder plus one explicit
  globalization/fallback path.

Acceptance target:

- The driver can honestly distinguish:
  - approximate prototype mode
  - diagonal-response approximation
  - true coupled second-order mode
- Convergence behavior becomes easier to reason about after small code changes.

### 4. Simplify the default optimizer structure once the coupled step exists

The current loop still mixes:

- approximate AH-like orbital steps
- gradient fallback steps
- multi-candidate merit comparison
- damping/backtracking logic
- optional numeric Newton debug validation

Needed work:

- Separate theory/intermediate construction from step construction.
- Choose one default globalization strategy and make it explicit.
- Move any remaining experiment-only heuristics behind debug flags.

Acceptance target:

- The transcript makes it obvious which algorithmic path was used.
- Regressions become attributable to one optimizer path instead of several
  interacting heuristics.

## Important Secondary Work

### 5. Decide whether to export active-space natural orbitals explicitly

The dangerous old behavior is fixed: `_cas_mo_coefficients` now stores the
optimized orbital basis, not a silently rotated natural-orbital basis.

What is still missing:

- separate storage for active-space natural-orbital coefficients if downstream
  tools want them explicitly
- checkpoint/version support for those coefficients
- clearer exporter/UI behavior around “optimized orbitals” vs “active NOs”

Recommended shape:

- Keep `_cas_mo_coefficients` as the optimization basis.
- Add a separate result field for natural-orbital coefficients only if there is
  a concrete consumer for them.

### 6. Revisit RDM performance only after the theory path is stable

The large CI refactor items from the older plans are mostly done, so the main
remaining performance hotspot is the exported RDM work.

Still useful later:

- optimized 1-RDM accumulation
- optimized 2-RDM accumulation
- optimized bilinear 2-RDM accumulation
- cross-checks against the current reference kernels on small spaces

This is lower priority than the per-root, exact-response, and coupled-step
work above.

### 7. Profile the direct CI sigma path before doing more CI restructuring

The direct sigma-vector Davidson path is already in the tree and covered by
agreement tests. Do not reopen major CI-driver refactors until profiling shows
where the remaining cost actually is.

If follow-up work is needed, focus on:

- determinant-connectivity hot spots in `apply_ci_hamiltonian`
- convergence diagnostics for large direct-Davidson runs
- dense-vs-direct crossover tuning

## Validation Still Needed

### 8. Add a small trusted external reference suite

This remains one of the most important missing safety nets.

Needed work:

- Add PySCF-backed reference comparisons for:
  - CASCI energies
  - CASSCF energies
  - state-averaged energies
  - orbital gradients or finite-difference checks
  - natural occupations
  - root tracking across distorted geometries

Priority systems:

- H2 CAS(2,2)
- LiH CAS(2,2)
- H2O CAS(4,4) or CAS(6,4)
- twisted ethylene
- one near-crossing SA example

### 9. Expand internal invariance and robustness coverage

Still needed:

- invariance checks under active-active rotations after convergence
- anti-symmetry checks for orbital gradients and `kappa`
- stress cases for root tracking near crossings
- response-solver restart and truncation cases at larger dimensions
- checkpoint/restart consistency tests for stored CASSCF orbitals
- broader root-resolved unit checks for weighted per-root orbital intermediates
  (`F_A`, `Q`, `g_orb`) plus additional quadratic-model, probe-ranking, and
  candidate-retention edge cases beyond the current small internal reduction
  identity coverage

## Mandatory Change Gate

### 10. Treat `tests/inputs/casscf_tests` as a required solver gate

Every meaningful CASSCF code change should still be validated against the full
manual fixture folder:

- `tests/inputs/casscf_tests/water_cas44_sto3g.hfinp`
- `tests/inputs/casscf_tests/water_cas44_b1.hfinp`
- `tests/inputs/casscf_tests/water_cas44_631g.hfinp`
- `tests/inputs/casscf_tests/water_cas44_ccpvdz.hfinp`
- `tests/inputs/casscf_tests/ethylene_casscf_321g.hfinp`
- `tests/inputs/casscf_tests/ethylene_casscf_ccpvdz.hfinp`

Observed rerun status on 2026-04-02 after commit `1cc6ec2`:

- `water_cas44_sto3g.rerun.log` converges
- `water_cas44_b1.rerun.log` converges
- `water_cas44_631g.rerun.log` converges
- `water_cas44_ccpvdz.rerun.log` converges
- `ethylene_casscf_321g.rerun.log` converges
- `ethylene_casscf_ccpvdz.rerun.log` converges

Required verification loop after each solver change:

```bash
cmake --build build --target hartree-fock -j4
for input in tests/inputs/casscf_tests/*.hfinp; do
    ./build/hartree-fock "$input" | tee "${input%.hfinp}.rerun.log"
done
```

Completion gate for any solver-facing CASSCF change:

- Run all inputs in `tests/inputs/casscf_tests`.
- Review every case, not just the one that motivated the edit.
- Do not call the work done if one case improves while another regresses.

## Recommended Implementation Order

1. Build the coupled orbital/CI step and simplify the default optimizer around
   that step instead of the current diagonal scaffold.
2. Add PySCF-backed reference tests for a small, trusted system set.
3. Expand the internal invariance/root-robustness coverage once the coupled
   step starts landing.
4. Revisit separate natural-orbital export only if a real downstream consumer
   needs it.
5. Profile remaining performance hot spots before reopening major CI/RDM work.

## What Not To Do Next

- Do not spend another pass on RAS filtering, symmetry-table wiring, root
  tracking, or bitstring guards unless a new regression is found; those old P0
  items are already implemented.
- Do not relabel the current default solver as full second-order.
- Do not add more heuristic step mixing to the production path.
- Do not overload `_cas_mo_coefficients` with another orbital representation.
- Do not prioritize more file splitting for its own sake; the important next
  work is theory and validation, not more mechanical refactoring.

## Relevant Files For The Next Pass

- `src/post_hf/casscf/casscf.cpp`
- `src/post_hf/casscf/response.cpp`
- `src/post_hf/casscf/response.h`
- `src/post_hf/casscf/orbital.cpp`
- `src/post_hf/casscf/orbital.h`
- `src/post_hf/casscf/rdm.cpp`
- `src/post_hf/casscf/ci.cpp`
- `src/post_hf/casscf_internal.h`
- `src/base/types.h`
- `src/io/checkpoint.cpp`
- `tests/casscf_internal.cpp`
- `tests/regression_cases.json`
- `tests/inputs/casscf_tests/*`

## Issue Mapping

- The old patch-plan items around modularization, CI consistency, root
  tracking, direct sigma, and cache plumbing should now be treated as landed.
- The remaining real implementation effort is the deeper second-order work:
  finishing the root-resolved SA path, building a true coupled orbital/CI
  solver, and adding stronger external and internal validation around that
  theory path.
