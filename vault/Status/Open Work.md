---
name: Open Work
description: Canonical summary of known gaps, risks, and follow-up work in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, open-work, canonical, roadmap]
---

# Open Work

Last updated: 2026-06-04

This is the canonical open-work document for the repository.
Use it with `vault/Status/Completion.md`. Older status snapshots and handoff
notes may still exist for design history, but they are no longer the source of
truth for what remains.

## Highest-priority correctness and robustness work

- (none currently — the ROHF MO-energy bookkeeping inconsistency is resolved;
  see Completion)

## Verification and regression gaps

- Strengthen the end-to-end spherical full-symmetry direct-SCF regression ladder beyond the current focused infrastructure tests and committed NH3/CH4 ladder
- Add durable regression coverage for remaining full-symmetry edge cases called out in the design notes:
  D3h, Oh, linear-group interplay, and lone-atom behavior
- Revalidate the CASSCF/PySCF gate suite after future optimizer work; the current tree matches the documented state, but the 11/11 suite was not freshly rerun during the May 25 consolidation review
- Keep documentation comments aligned with the implemented spherical symmetry representation; stale comments have already drifted once

## Spherical-basis work still intentionally guarded off

- Spherical analytic gradients (and therefore geomopt / freq) for the post-HF
  correlated paths (RMP2 / UMP2). RHF, UHF, and ROHF spherical gradients,
  geomopt, and frequencies are all landed (ROHF via the same build-W-in-the-
  spherical-basis-then-lift-once pattern the RHF/UHF paths use). MP2 gradients
  still need the response-machinery audit before the same lift adapter (Phase 1)
  can be wired through `compute_rmp2_gradient` / `compute_ump2_gradient`.
  Boundary markers: `water_rmp2_spherical_{gradient,geomopt}_rejected`.
- Spherical PCM
- Spherical DFT and TDDFT
- Any additional spherical workflows not already covered by the landed
  single-point, RHF/UHF-gradient, and RHF/UHF-geomopt-and-freq allow-list

## Symmetry follow-up

- Conventional-path symmetry-unique ERI storage remains out of scope; current full-group reduction is a direct-SCF feature
- ROHF is still outside the full-symmetry direct-SCF implementation scope
- The full-symmetry performance story still has room to improve even after the persisted-skeleton and monomial-operator wins; the remaining major option is a true memory-direct contraction that avoids materializing the dense `nb^4` buffer

## DFT and response-method gaps

- Double-hybrid functionals remain single-point only; analytic gradients,
  geometry optimization, frequencies, and TDDFT are still unimplemented there
- For range-separated functionals, `ImaginaryFollow` and `LinearResponse`
  (TDDFT) remain gated / unvalidated even though gradient-driven workflows are
  now landed
- Analytic Hessian remains unimplemented; frequencies are currently semi-numerical
- DFT imaginary-mode following is not implemented
### TICKET: MPI rank-split the DFT grid layer (Gap 2) — the measured DFT scaling cap

**Priority: highest DFT-HPC item.** This is the single change that gives DFT
HF-like MPI scaling.

**Measured problem** (`scale.json`, Notchpeak notch460, post-#151, 6-31g/os):
DFT strong-scaling walls at **3.5× on 16 ranks (22% efficiency) at nb=208 and
degrades to 20% at nb=312**, while HF on the same ladder holds **10× / 63% and
rises with size**. The DFT/HF per-iteration ratio grows 4.8×→10× from nb=104 to
416 — because #151 distributed the J/K but the grid is still replicated, so the
grid's share of DFT wall time rises with both system size and rank count.

**Root cause:** `grep -rn "Mpi::\|USE_MPI" src/dft/` is empty. Every rank
rebuilds the full grid and evaluates the full XC. The grid loops are
OMP-parallel as of #152 (`xc_grid.cpp:83`, `if (!omp_in_parallel())`) but have
**no MPI rank split** — OMP threads work within a rank; nothing distributes
across ranks. (This supersedes the older "grid loops are still serial" note —
they are threaded now; what's missing is the rank split.)

**The work:**
- Partition grid points (or whole Becke atomic batches) by rank in
  `evaluate_density_on_grid` / the `xc_grid.cpp` density+XC loops.
- Reduce the XC matrix (`nb²`) across ranks with `Mpi::allreduce_inplace`,
  alongside the existing Fock reduce — one more reduction, not a new pattern.
- **Determinism constraint (load-bearing):** the DFT XC reduction is the
  historical jitter site (see the DFT XC Reduction Determinism note). The
  cross-rank sum MUST be in fixed rank order, never completion order, never
  `omp critical`. This is the medium-risk part of an otherwise ~M change.

**Acceptance:**
- `energy(-n k) == energy(serial)` bitwise across k ∈ {1,2,4,8,16} on a DFT
  case at nb where a partition bug bites (16-water B3LYP, nb=208). This is also
  Gap 3's missing CI tripwire — land them together.
- DFT strong-scaling efficiency at 16 ranks rises materially off the measured
  22% baseline; HF-like (>60%) is the target, grid being the last serial piece.

**Not in scope:** grid layer already OMP-threaded (#152); this is MPI only.
The `277ba10` (#151-only, pre-#152) attribution split is write-up-only and does
not block this. Full measured rescope in `docs/HPC_REMAINING_SCOPE.md`.
- Coarse/low-quality DFT grids can still show noticeable orientation sensitivity
  under symmetry reorientation; the validated symmetry-on gradient regression is
  intentionally pinned to `grid ultrafine`

## SCF, post-HF, and workflow gaps

- ROHF post-HF: FCI, CASSCF, and RASSCF now accept ROHF references; RMP2/UMP2
  and the coupled-cluster paths remain RHF/UHF only for ROHF inputs
- ROHF CASSCF/RASSCF only support a closed, doubly-occupied inactive core; a
  spin-polarized open inactive core (distinct alpha/beta core orbitals, with the
  unrestricted core Fock, core energy, and response-block changes it implies)
  is out of scope and stays rejected by the parity guard
- ROHF stability analysis and PCM remain incomplete (ROHF analytic gradients,
  and the geomopt / frequency workflows built on them, are now landed
  Cartesian-side — see Completion)
- The ccgen `TensorOptimized` RCCSDT backend is still treated in-tree as an experimental / phase-4 path

## ccgen dressed intermediates

### Problem

The ccgen-generated CC kernels (`build/generated/cc/*_planck_generated.cpp`,
consumed by the arbitrary-order tensor solver) do **not** carry the
dressed-operator intermediates the hand-written `tensor_backend.cpp`
CCSD/CCSDT paths use — the Stanton–Gauss `τ`/`τ̃`, `Fae`/`Fmi`/`Fme`, and
`Wmnij`/`Wabef`/`Wmbej`. What ccgen *does* extract
(`optimization/intermediates.py::detect_intermediates`) is **syntactic CSE**:
any leaf sub-contraction recurring ≥ `--intermediate-threshold` (default 5)
times is hoisted into a generic shape-tagged builder (`build_W_oov`,
`build_W_ovv_3`, …). That removes duplicate leaves but leaves the residual
*term structure* un-factored, so the generated residual keeps the higher-cost
contraction pattern. Same energy, worse asymptotic scaling. This is the real
content behind the "experimental / phase-4" status of the generated path.

CSE cannot produce dressing: CSE is term-local verbatim-pattern hoisting,
whereas dressing is a global refactorization — introduce a *new named operator*
(`τ = t2 + ½·P(t1t1)`, `Wmnij = ⟨mn|ij⟩ + P(t1·…) + τ⟨mn|ef⟩`), then rewrite
every residual term that is (part of) its expansion to reference it, including
terms only equivalent *after* the substitution.

### Option A — automatic factorization (this note's scope)

Discover the dressed factorization from the canonicalized residual terms with
no per-method human recipe. Pipeline insertion:
`… → canonicalize → [NEW: factorize] → detect_intermediates(CSE) → rewrite →
lower → emit` (CSE becomes cleanup on whatever the factorizer leaves).

Four sub-problems, effort/risk as scoped:

- **A1 — τ-recognition (substitution matching). LANDED (behind a flag).**
  Detect `t2 + ½·P(t1t1)` groupings and collapse to `τ`. Implemented in
  `python/ccgen/optimization/tau.py` + `tests/test_tau.py`, in eight verifiable
  steps: τ spec + external-skeleton fingerprint (A1.0/A1.1), the bare-half
  predicate `match_t1t1_half` (A1.2), residue-based pairing `find_tau_matches`
  (A1.3), the exact-coefficient firewall `validate_tau_match` (A1.4 — proved the
  written t1t1 rep carries weight **2** per unit τ, not ½, and that
  `canonicalize_term` is not idempotent when its relabel reorders an antisym
  factor, handled by a fixed-point loop), the rewrite `apply_tau` (A1.5), the
  offline algebra-preservation gate `tau_rewrite_preserves_algebra` (A1.6 — the
  proof that wiring it in cannot change answers), and emit wiring (A1.7/A1.8).
  τ rides the **existing** intermediate machinery: `factorize_tau_equations`
  synthesizes a `tau` `IntermediateSpec`, so the emitter's `build_<name>` path
  materializes `build_tau` and the residual `tau` factors resolve to the local
  (one-line `_map_factor` hook). Gated by `print_cpp_planck(factorize_tau=…)`,
  CLI `--factorize-tau`, and CMake `PLANCK_CC_FACTORIZE_TAU` — **all default
  OFF**, so the default build is byte-identical. The tau-on generated CCSD
  compiles against the real CC headers (verified in-test). A full bit-identical
  *energy* run is not yet possible because `ccsd_planck_generated.cpp` is not
  `#include`d into any binary (only ccsdt/ccsdtq are); when it is wired, add the
  energy gate. τ alone barely moves flops — its value is proving the
  factorization→emit plumbing end-to-end before A2/A3 add the load-bearing
  `Wmnij`/`Wabef` operators.
- **A2 — dressed-operator hypothesis generation.** (a) seeded from the known
  finite `Ŵ`/`F̂` family per rank (tractable, edges toward Option B), or
  (b) discovered by clustering terms on shared sub-contraction topology (the
  genuinely automatic, research-grade part). **XL, high risk** for (b).
- **A3 — subgraph-isomorphism rewrite (core).** For each hypothesized operator,
  find every residual term that is part of its expansion and rewrite to a
  reference. Graph isomorphism over `Tensor`-factor sets with index-space
  typing, permutation/antisymmetry bookkeeping, and `Fraction`-coefficient
  consistency — the same equivalence machinery as `canonicalize.py` +
  `_wickaccel.cpp`, applied to sub-terms. **XL, high risk.**
- **A4 — dependency-ordered emit.** Dressed ops form a DAG (`Wmnij` needs `τ`).
  Topo-sort builders, thread through kernel signature, manage lifetimes; beyond
  today's append-CSE-blob model in `emit/planck_tensor_cpp.py`. **M, low risk.**

Supporting: a scaling-aware cost model (extend `estimated_build_flops` to model
the flop *exponent*, not element count); a bit-identical energy gate vs the
hand-written CCSD/CCSDT on the Be/H2O CC regression set; an asymptotic-scaling
assertion (dressed residual must drop the exponent, else the pass was
cosmetic); and a safe-fallback guarantee (degrade to CSE-only output when a
term can't be factored — never emit wrong algebra).

De-risked ordering: build A1, A3, A4 and the gates against a **seeded** operator
set (A2a), prove bit-identical + scaling win on CCSD, then attempt A2b as a
research spike. If A2b doesn't converge, working dressed CCSD/CCSDT still ships
and the operator set is just "curated."

### ccgen parallel generation is not equivalence-safe (separate defect)

`generate_cc_equations(method, parallel_workers=N>1)` produces a **different**
equation set than the serial (`workers=1`, default) path — not just reordered,
genuinely different coefficients/term counts (ccsd: singles 24 vs 27–29,
doubles 200 vs 154). Two independent order-dependent defects, each internally
deterministic:

1. **`_wickaccel` is not spawn-safe.** The C extension's `apply_deltas_layout`
   / `analyze_signature` return divergent index-layout results in a
   freshly-spawned worker vs the parent, corrupting relabeled terms (the energy
   manifold gets factors desynced from their summed-index lists, e.g.
   `f(i,a) t1(b,j)` with summed `(i,a)`). A `CCGEN_NO_ACCEL` env hook (added to
   `wick.py` / `canonicalize.py`) forces the pure-Python path and is inherited
   by spawned workers; it fixes the energy manifold but not defect 2.
2. **Pre-canonical exact merge is partition-local.**
   `merge_exact_term_into_buckets` dedups raw terms within a chunk before
   canonicalization; raw terms that combine when co-located in one chunk
   survive separately when split across chunks (singles: `-1/4` vs two `-1/8`).
   Making it global would defeat its streaming-memory purpose on large BCH
   expansions.

The default path is serial and *is* deterministic + correct; parallel is an
opt-in speed feature. The regression `test_parallel_generation_matches_serial`
is marked `@unittest.expectedFailure` with the root cause inline, and
`test_serial_generation_is_deterministic` pins the guarantee that holds. Real
fix = make the extension spawn-safe (rebuild `_wickaccel.cpp`) and lift the
raw-merge global; deferred as parallel generation is unused by the default
build. No bearing on the dressed-intermediate work above (that runs on the
serial path).

### Option B — curated dressing templates (fallback, not this note's scope)

Hand-encode the known Stanton–Gauss factorization per method (CCSD, CCSDT) as a
substitution recipe ccgen applies mechanically. Drops A3 from XL to M; days per
method; deterministic. How PySCF/CFOUR actually got there. Kept as the fallback
if Option A's automatic-discovery core (A2b) stalls.

## BSSE follow-up

- DFT ghost / counterpoise support
- N-body counterpoise beyond two fragments
- Counterpoise-corrected gradients and geometry optimization
- Post-HF ghost-reference verification beyond the current SCF-level validated scope

## CASSCF

### Remaining work

#### P2: Optimizer simplification pass — mostly resolved; only cosmetic remainder

A suite-wide sweep of every CAS input recorded which candidate the merit
selector actually accepts. Result:

- **Per-root candidates** (`root*-coupled` / `root*-grad-fallback`): accepted
  **zero** times, yet cost a full per-root coupled solve every stagnant macro.
  **Removed** (see Completion). Dead weight, no behavior change.
- **`numeric-newton`**: the dominant accepted fallback (~125 accepted steps
  across the suite). **Load-bearing — must NOT be demoted.** The original P2
  deliverable to demote it behind `mcscf_debug_numeric_newton` was wrong.
- **Single-pair probes**: accepted exactly once, but that once is the
  load-bearing `probe-pair6-favored[uphill]` step on the SAD-uphill SA-2 canary.
  **Must NOT be removed.**

So the original P2 deliverables (demote numeric-newton, remove probes) are
disproven; only the per-root removal was correct, and it is done.

Cosmetic remainder (low value): make every transcript step label uniquely
identify the path taken. Not required for correctness or performance.

### Future hardening

- Plateau-escape convergence path (`casscf.cpp`, the `Treating the stationary
  orbital plateau as converged` branch) is **correct and load-bearing**, not a
  hack to retire. It is the only exit for a genuinely converged
  state-averaged solution: at an SA stationary point the gating quantity
  `sa_g = Σ_I w_I g_I` goes to ~1e-10 while the per-root screens
  (`root_screen_g` / `max_root_g`) plateau at an O(1e-2) nonzero value, because
  state-averaging makes only the *weighted* gradient stationary, not each
  individual root. With `mcscf_accept_uphill` the per-root convergence screen
  then never passes, so the plateau branch is the correct way to recognize "SA
  gradient converged, energy and step flat → done." This is exercised by
  `water_casscf_sa2_sto3g_sad_guess_uphill` (the only one of the four SA-2
  cases that uses it; the other three converge through the normal gate at
  `sa_g < 1e-5`).
- Keep the two water SA-2 SAD-start regressions, because they intentionally protect two distinct optimizer policies

## Performance and maintenance opportunities

- The DFT Coulomb/exchange (`build_coulomb_from_eri` / `build_exchange_from_eri`)
  contractions are now parallel and verified thread-count-invariant (see the
  Integral Engine note). The remaining DFT parallelization target is the grid
  layer, tracked above under the DFT gaps.
- Rework shell-pair construction to operate at shell granularity rather than per Cartesian AO component
- Eliminate remaining reversed-shell-pair reconstruction churn in gradient paths outside the already-fixed RHF path
- Deduplicate the full-group AO-transform machinery that still exists in both `group_operations.cpp` and `mo_symmetry.cpp`
- Extract a shared `SpatialQuartetLayout` (6-axis dims + strides +
  `spatial_index` + `resize_for_quartet`) and retrofit the OS, HGP, and Rys
  per-quartet scratch onto it. All three now carry near-duplicate per-quartet
  scratch structs — OS's `_eri_scratch`, HGP's `g_hgp_scratch`, and (as of
  PR #126) Rys's `RysScratch` — so the three concrete call sites exist to shape
  the shared interface. Only the spatial-layout core is common; the Boys `m`
  axis, HGP's `a0c0_accum`, OS's no-zero-init policy, and the differing
  accessors stay engine-specific. Bitwise-gate across all three engines
  (`planck-compute-2e`, `planck-hgp-engine-smoke`, plus the OS path via the
  existing ERI gates).
- Refactor `Calculator` only where it buys real safety or clarity: the leading candidates are grouping the loose MP2/UMP2 result cache and introducing a geometry-derived working-state object with a single invalidation point
