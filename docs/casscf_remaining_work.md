# CASSCF Remaining Work

## Purpose

This note is the current handoff for the next CASSCF implementation pass. It
reflects the repository state on 2026-04-03 after the first production-enabled
coupled orbital/CI solve landed on top of the root-resolved SA optimizer work,
and it supersedes the older patch-plan notes.

## Current Status

The code is substantially better than the original monolithic prototype, but it
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
  the same ket-to-bra operator convention, with internal agreement coverage.
- The public CI-response Davidson solve now has projection, restart/collapse,
  bounded-subspace handling, and best-residual reporting on truncated exits.
- The single-step CI-response fallback no longer reports false convergence.
- Active-space `Q` construction now reuses a cached transformed-integral
  container instead of retransformation inside the micro-iterations.
- The CI solver can now use a direct sigma-vector Davidson path, and this has
  agreement tests against dense CI on small spaces.
- The driver now carries a per-root `StateSpecificData` record through the
  macro loop, including per-root CI vectors/energies, 1-RDMs, 2-RDMs, active
  Fock contributions, `Q` contributions, orbital gradients, CI-response data,
  first-order 2-RDMs, `Q1`, and CI-driven orbital corrections.
- The state-averaged `gamma`, `Gamma`, `F_A`, and `g_orb` objects are now
  rebuilt as explicit weighted sums of those per-root records instead of being
  formed only from early averaged inputs.
- Response-side bilinear `Gamma1` accumulation is now expressed as an explicit
  weighted sum of per-root contributions.
- The SA micro-iteration now keeps the response RHS, `c1`, `Gamma1`, `Q1`,
  CI-driven orbital correction, and updated orbital-gradient contribution
  root-resolved before reducing back to one weighted update.
- The AH-like orbital step proposal is now built root-first: each root forms
  its own preconditioned orbital step from its own `g_orb` and `F_A` data, and
  the driver reduces those proposals to one SA step only after that per-root
  solve. The fallback gradient step follows the same root-first reduction
  pattern.
- The shared SA candidate screen now keeps explicit per-root AH-like and
  gradient-fallback proposals as trial steps, so strong single-root directions
  can still reach full reevaluation even when the immediate weighted proposal
  damps them.
- The SA acceptance/merit screen now also uses a root-resolved orbital-gradient
  screen: a weighted sum of per-root `||g_orb||_inf` values for merit ranking,
  plus an explicit max-root guard so one lagging root cannot hide behind the
  averaged SA gradient.
- Probe-ranking now uses a weighted per-root priority/signed signal rather than
  ranking only from one already-averaged gradient vector.
- Macro diagnostics now report both `root_model_spread` and the winning
  `candidate=...` label for the accepted orbital proposal, along with the SA,
  weighted-root-screen, and max-root gradient diagnostics.
- The late macro-loop plateau and convergence gates now also follow the
  weighted and max-root orbital-gradient screens instead of collapsing back to
  the averaged SA gradient when deciding whether a multistate run is truly
  stationary.
- The root-resolved SA optimizer path is now complete for the current
  scaffold: orbital update, candidate generation, response, acceptance, and
  late plateau/convergence logic all keep the multiroot state-specific data
  alive instead of dropping back to early averaging.
- Numeric Newton is no longer a peer in the large default candidate family,
  but the production optimizer still keeps it as a small-space escape hatch
  (and as an explicit debug path via `mcscf_debug_numeric_newton`) while the
  new coupled solve is being hardened.
- `calc._cas_mo_coefficients` now stays in the converged optimization basis,
  avoiding the old mismatch between stored orbitals and the solved CI gauge.
- The response layer now has an explicit `ResponseRHSMode` split between:
  - approximate commutator-only response
  - exact active-space orbital-derivative response
- The exact response RHS is now built analytically in `response.cpp` from the
  active-space one- and two-electron Hamiltonian derivatives, and the response
  API no longer carries the older finite-difference context bundle.
- The driver now defaults to the exact orbital-derivative response RHS and
  keeps the commutator-only shortcut only behind the explicit
  `mcscf_debug_commutator_rhs` debug option.
- The codebase now has a diagonal-preconditioned orbital-step helper plus unit
  coverage for it, so the diagonal orbital model is no longer only a step
  model; it is now also used explicitly as a preconditioner inside the coupled
  orbital/CI solve.
- The response path now also packages the explicit coupled blocks per root:
  orbital-to-CI RHS, CI solve result, CI residual, first-order `Gamma1/Q1`,
  and the resulting CI-driven orbital correction.
- The driver now uses those coupled blocks in a matrix-free block iteration per
  root: it seeds from an orbital-preconditioned plus CI-response Schur step,
  refines the coupled residual with explicit OO/OC/CO/CC block actions, and
  promotes the weighted `sa-coupled` direction into the normal candidate screen
  instead of keeping it behind a disabled rescue gate.
- The first optimizer-simplification pass is now in place: the production
  candidate family has been cut down to the coupled step, gradient fallback,
  a small-space numeric-Newton escape hatch, and stagnation-only diagonal /
  probe / per-root rescue candidates. The larger always-on AH/mix/gradient
  variant family is no longer part of the normal path, and the full checked-in
  single-root manual fixture gate still converges at the expected energies.
- `tests/casscf_internal.cpp` now checks that the analytic exact RHS matches a
  finite-difference active-space Hamiltonian rotation on a small reference
  problem, and that it differs from the commutator-only shortcut when the
  two-electron Hamiltonian responds.
- `tests/casscf_internal.cpp` now also covers weighted per-root
  pair-priority/probe-ranking inputs, weighted quadratic-model diagnostics,
  candidate-screen sign flips, and edge cases where the state-averaged pair
  priority differs from a single root's dominant direction.
- The `planck-casscf-internal` target now links the orbital/integral
  implementation it exercises, and the internal harness no longer crashes in
  the trailing root-reduction coverage block.
- `tests/casscf_internal.cpp` now also checks coupled-block invariants for
  `build_coupled_response_blocks(...)` and verifies that the new coupled
  orbital/CI solve does not worsen the seeded coupled residual on a synthetic
  small-space problem.
- The full checked-in single-root manual CASSCF gate still converges after the
  coupled-step change:
  - `ethylene_casscf_321g` -> `-77.5145223872 Eh`
  - `ethylene_casscf_ccpvdz` -> `-77.9524855977 Eh`
  - `water_cas44_631g` -> `-75.5497490402 Eh`
  - `water_cas44_b1` -> `-74.2879452324 Eh`
  - `water_cas44_ccpvdz` -> `-75.6045806122 Eh`
  - `water_cas44_sto3g` -> `-74.4700757755 Eh`

What is still true:

- The default optimizer is no longer purely diagonal-response: it now includes
  a production matrix-free coupled orbital/CI candidate built from explicit
  coupled residual blocks, but it still lives inside the older macro candidate
  screen rather than a single clean solver path.
- The state-averaged optimizer is root-resolved through the full current
  scaffold, and it now has a real coupled step, but the overall optimizer is
  still not a polished final SA second-order method.
- The exact RHS is now the default theory path and is threaded through the new
  coupled solve, but the fallback heuristics around that solve are still richer
  than they should be in the final architecture.
- The orbital Hessian used in production is still only a diagonal
  preconditioner, not a full OO block model.
- The default orbital-step logic is much smaller than before, but it still has
  more than one production escape hatch: the coupled step is primary, yet
  small-space numeric Newton and stagnation-only rescue candidates are still
  present while the coupled solve is being hardened.

## Remaining Work

### 1. Keep the exact CI-response RHS as the real theory path through the remaining scaffold

The response layer no longer has only the commutator shortcut: it now exposes
both an explicit approximate mode and an explicit exact orbital-derivative RHS
builder, and the driver defaults to the exact path.

The remaining work here is no longer “decide analytic vs finite-difference”.
That choice is done. The remaining work is to keep the exact path threaded
cleanly through the rest of the SA scaffold instead of blurring it back into
the approximate diagonal-response machinery.

Needed work:

- Continue threading the exact RHS through the now-root-resolved SA plumbing
  instead of dropping back immediately to one averaged perturbation.
- Keep logs and control flow clear about the distinction between:
  - approximate commutator-only response
  - exact orbital-derivative response
- Keep the exact-RHS path as the production theory path while making sure
  response fallbacks and diagnostics do not silently relabel the approximate
  path as if it were the full second-order method.

Acceptance target:

- Approximate response remains an explicitly requested debug path only.
- Small reference problems continue to compare the analytic exact-response RHS
  against a finite-difference active-space Hamiltonian rotation.
- The exact path is the default theory path wherever the current scaffold is
  meant to approximate second-order behavior.

### 2. Complete and harden the new coupled orbital/CI step

The core solver milestone has moved forward: the code now has a production
matrix-free coupled orbital/CI step instead of only parked scaffolding.

The current implementation:

- seeds from an orbital-preconditioned plus CI-response Schur-like step
- evaluates explicit OO, OC, CO, and CC residual pieces
- refines them by block iteration with diagonal orbital/CI preconditioners
- contributes the weighted `sa-coupled` direction directly to the normal
  candidate screen

That is a real coupled step, but it is still an early coupled implementation
rather than the final solver architecture.

Needed work:

- Keep the diagonal orbital Hessian only as a preconditioner.
- Strengthen the matrix-free coupled solve itself:
  - better residual scaling and stopping criteria
  - cleaner globalization/trust-region behavior
  - less dependence on the surrounding legacy candidate family
- Decide whether the end state should stay matrix-free block-iterative or move
  to a more explicit augmented-Hessian / norm-extended coupled solve.
- Reduce the default optimizer to one main step builder plus one explicit
  globalization/fallback path.

Acceptance target:

- The driver can honestly distinguish:
  - debug approximate mode
  - production coupled mode
- Convergence behavior becomes easier to reason about after small code changes.

### 3. Finish simplifying the default optimizer around the coupled step

The current loop still mixes:

- the main coupled orbital/CI candidate
- a gradient fallback
- a small-space numeric-Newton escape hatch
- stagnation-only diagonal/per-root/probe rescue candidates
- merit comparison plus damping/backtracking logic

Needed work:

- Decide whether small-space numeric Newton should remain a production rescue
  path or become debug-only once the coupled solve is robust enough.
- Reduce or eliminate the stagnation-only rescue family once the coupled step
  has a cleaner globalization / trust-region story.
- Separate theory/intermediate construction from step construction.
- Choose one default globalization strategy and make it explicit.
- Move any remaining experiment-only heuristics behind debug flags.

Acceptance target:

- The transcript makes it obvious which algorithmic path was used.
- Regressions become attributable to one optimizer path instead of several
  interacting heuristics.

## Important Secondary Work

### 4. Add a small trusted external reference suite

This is still one of the most important missing safety nets.

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

### 5. Expand internal invariance and robustness coverage

Still needed:

- invariance checks under active-active rotations after convergence
- anti-symmetry checks for orbital gradients and `kappa`
- stress cases for root tracking near crossings
- response-solver restart and truncation cases at larger dimensions
- checkpoint/restart consistency tests for stored CASSCF orbitals
- at least one checked-in multiroot end-to-end fixture; the current manual
  `tests/inputs/casscf_tests` folder is still entirely `nroots 1`
- broader root-resolved unit checks for weighted per-root orbital intermediates
  (`F_A`, `Q`, `g_orb`) plus additional quadratic-model, probe-ranking,
  candidate-retention, and acceptance-screen edge cases beyond the current
  small synthetic coverage

### 6. Decide whether to export active-space natural orbitals explicitly

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

### 7. Revisit RDM performance only after the theory path is stable

The large CI refactor items from the older plans are mostly done, so the main
remaining performance hotspot is the exported RDM work.

Still useful later:

- optimized 1-RDM accumulation
- optimized 2-RDM accumulation
- optimized bilinear 2-RDM accumulation
- cross-checks against the current reference kernels on small spaces

This is lower priority than the per-root, exact-response, and coupled-step
work above.

### 8. Profile the direct CI sigma path before doing more CI restructuring

The direct sigma-vector Davidson path is already in the tree and covered by
agreement tests. Do not reopen major CI-driver refactors until profiling shows
where the remaining cost actually is.

If follow-up work is needed, focus on:

- determinant-connectivity hot spots in `apply_ci_hamiltonian`
- convergence diagnostics for large direct-Davidson runs
- dense-vs-direct crossover tuning

## Mandatory Change Gate

### 9. Treat `tests/inputs/casscf_tests` as a required solver gate

Every meaningful CASSCF code change should still be validated against the full
manual fixture folder:

- `tests/inputs/casscf_tests/water_cas44_sto3g.hfinp`
- `tests/inputs/casscf_tests/water_cas44_b1.hfinp`
- `tests/inputs/casscf_tests/water_cas44_631g.hfinp`
- `tests/inputs/casscf_tests/water_cas44_ccpvdz.hfinp`
- `tests/inputs/casscf_tests/ethylene_casscf_321g.hfinp`
- `tests/inputs/casscf_tests/ethylene_casscf_ccpvdz.hfinp`

Observed rerun status on 2026-04-03 after the optimizer-simplification pass:

- `ethylene_casscf_321g.rerun.log` converges to `-77.5145223871 Eh`
- `ethylene_casscf_ccpvdz.rerun.log` converges to `-77.9524855977 Eh`
- `water_cas44_631g.rerun.log` converges to `-75.5497490402 Eh`
- `water_cas44_b1.rerun.log` converges to `-74.2879452324 Eh`
- `water_cas44_ccpvdz.rerun.log` converges to `-75.6045806122 Eh`
- `water_cas44_sto3g.rerun.log` converges to `-74.4700757755 Eh`

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

1. Simplify the default optimizer around the new coupled orbital/CI step.
2. Add PySCF-backed reference tests for a small, trusted system set.
3. Expand the internal invariance/root-robustness coverage once the coupled
   step starts landing.
4. Revisit separate natural-orbital export only if a real downstream consumer
   needs it.
5. Profile remaining performance hot spots before reopening major CI/RDM work.

## What Not To Do Next

- Do not relabel the current default solver as full second-order.
- Do not add more heuristic step mixing to the production path.
- Do not reopen old P0 items such as RAS filtering, symmetry-table wiring,
  root tracking, or bitstring guards unless a new regression appears.
- Do not overload `_cas_mo_coefficients` with another orbital representation.
- Do not prioritize more file splitting for its own sake; the important next
  work is theory and validation, not more mechanical refactoring.

## Relevant Files For The Next Pass

- `src/post_hf/casscf/casscf.cpp`
- `src/post_hf/casscf/response.cpp`
- `src/post_hf/casscf/orbital.cpp`
- `tests/casscf_internal.cpp`
