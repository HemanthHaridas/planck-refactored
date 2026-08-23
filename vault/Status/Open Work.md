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

## docs/ hygiene — two ccgen scope docs still owe an architecture rewrite

A file in `docs/` answers one architecture question or is a teaching guide; scoping **in-flight**
work is the only exception, and it expires when that work lands.

**Three of the original five are done (2026-08-16).** The CCSDTQ trio collapsed into one answer, as
predicted — they were split by *effort* and merged once regrouped by *question*:

| retired | into |
|---|---|
| `CCGEN_R3_HIGHER_RANK_BRIDGE_SCOPE.md` (295) | `docs/CCGEN_CCSDTQ_MULTISECTOR.md` |
| `CCGEN_KERNEL_WIRING_MULTISECTOR_SCOPE.md` (225) | same |
| `CCGEN_CCSDTQ_FCI_VERIFICATION_SCOPE.md` (128) | same |
| `CCGEN_TENSOR_ACCESSOR_FIX_SCOPE.md` (181) | `docs/CCGEN_TENSOR_ACCESSOR.md` |

All three CCSDTQ docs carried **stale headers contradicting their own content** — the bridge doc
advertised a rank-8 `xfail` that no longer exists in the code, the verification doc kept a "Why it
is still RED" section under a GREEN status line, and the wiring doc claimed "two gaps, both open"
when both were closed. Verified before rewriting: 12 bridge tests pass, the Be CCSDTQ==FCI oracle
passes (12m01s), and `be_rccsdtq_sto3g` passes end-to-end. **Do not trust a status header without
running its gate** — four such headers were found false in one session.

Remaining, deliberately deferred until the UCC work (U1–U5) lands:

| doc | lines | landed work |
|---|---|---|
| `CCGEN_SPIN_ADAPTATION_SCOPE.md` | 892 | S0–S4 |
| `CCGEN_KERNEL_WIRING_AND_BENCHMARK_SCOPE.md` | 331 | kernel wiring + benchmarks |

**Four became due 2026-08-22.** The UCC numeric ladder completed (F1/F2/F3), and then U1 completed
on top of it. `CCGEN_UCC_NUMERIC_FIXTURE_SCOPE.md`, `CCGEN_UCC_RESIDUAL_EVALUATOR_SCOPE.md` and
`CCGEN_U1_UCC_ADAPT_SCOPE.md` are all finished work, so the scoping exemption has expired for each.
(`CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` stays exempt — U2–U5 are genuinely in flight.
The rank-6 gap closed, so that handoff has become
`CCGEN_UCC_NUMERIC_VALIDATION.md` — the target the three older UCC scope docs should merge into.) They should merge into **one** answer — they are two halves of a single question, split by
effort exactly as the CCSDTQ trio was:

- **`CCGEN_UCC_NUMERIC_VALIDATION.md`** — how do you check that a spin-block CC residual is right?
  The U1 doc merges in here: its answer is the same question one layer up (how do you know the
  GCC→UCC *adaptation* is right), and its four PySCF-interface defects belong with the fixture
  lessons rather than in a step ladder.

Keep, because each cost real investigation: the per-target-pairing correction to the closed-shell
oracle; the closure relations and why F1's fixture cannot serve it; the `(occ…,vir…)` vs
`(vir…,occ…)` transpose; the `f_ov`-on-both-sides measurement table with the falsified first
hypothesis; and both vacuous-pass traps (converged amplitudes, OH/STO-3G). Drop the F-numbering, the
per-step *Verify:* lines, and the three-option F2.0 design table now that A is built and proven.
Deferring only until U1.2 has consumed the evaluator, in case that surfaces more.

`CCGEN_SPIN_ADAPTATION_SCOPE.md` is the reference U1 works against, so rewriting it now risks
discarding scope still load-bearing for unstarted work. Target questions:

- **`CCGEN_SPIN_ADAPTATION.md`** — how does a spin-orbital CC equation become a spatial one?
  (the rank-4 multi-sector half of this is already answered in `CCGEN_CCSDTQ_MULTISECTOR.md`)
- **`CCGEN_KERNEL_WIRING.md`** — how does a generated kernel reach a runnable binary, and what does
  it cost?

When doing it: read `CCGEN_SPIN_ADAPTATION_SCOPE.md` in full first, and move any still-live UCC
scope into `CCGEN_U1_UCC_ADAPT_SCOPE.md` rather than dropping it. Keep the measured numbers, the
ruled-out hypotheses and the wrong turns — they are part of each answer. Drop step numbering,
gates-to-write and sequencing diagrams.

Judged compliant in the same audit, for the record: `CCGEN_TEACHING_GUIDE`, `CCGEN_REPORT`,
`CCGEN_GENERATION_AND_VALIDATION` (teaching/report); `CCGEN_HIGHER_OPERATOR_REUSE`,
`CCGEN_DIAGRAM_REPRESENTATION_SCOPE`, `CCGEN_INTERMEDIATE_MEMORY_LOCALITY_SCOPE` (already
question-shaped, work unstarted); `CCGEN_ARBITRARY_ORDER_UCC_SCOPE`, `CCGEN_U1_UCC_ADAPT_SCOPE`
(genuine in-flight scope).

`CCGEN_DRESSED_KERNEL_VALIDATION_SCOPE` was in that list and has been **deleted** (2026-08-16): it
scoped V2–V6 for the dressed route, which is **retired** (see Completion — dressing and spin
adaptation do not compose, 52 % short on Be). The doc never acknowledged the retirement, so it read
as live scope inviting work the project has decided against — the "resumes an abandoned route" harm
this rule exists to prevent, and worse than a stale header because a full ladder looks actionable.
Its two still-binding design constraints (U1 must accept an already-dressed manifold; block-keyed
intermediate naming) were moved into `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md`, where they apply; the
retirement answer `CCGEN_DRESSING_AND_SPIN_ADAPTATION.md` already records what was kept and what to
check first if dressing is ever revisited.

### Active ccgen scopes, audited 2026-08-16 (verified against code, not headers)

| scope | state |
|---|---|
| `CCGEN_ARBITRARY_ORDER_UCC_SCOPE` + `CCGEN_U1_UCC_ADAPT_SCOPE` | **U0 + U1 COMPLETE and numerically validated; U2 IN PROGRESS.** **UCC now reaches the FCI limit directly** — `U15UccReachesFciLimitTests` solves the generated manifold to self-consistency on LiH+/6-31g (3-electron doublet, so CCSDT is exact) and hits FCI to **3.7e-14**; the obvious system Li/STO-3G is a *vacuous* gate there (t3 worth 0, a broken T3 passes), LiH+ makes the triples worth 8.1e-8. Also verified at rank 4 vs PySCF UCCSD (~6e-16) and rank 6 vs GCC-sliced (**1.6e-17**). **The gap that actually mattered: the GCC→UCC adaptation had 22 call sites and NO numeric gate** (structural checks only); `U14c3UccIsGccSlicedAtRankSixTests` closes it. U1.3 is DEAD (U1.1 designed its hazard out). **U2 is STRUCTURALLY COMPLETE** — U2.1 landed `build_ucc_block_denominator`; **U2.2 landed** `build_ucc_denominator_cache` + `ArbitraryOrderDenominatorCache::{sectors,sector_tensor}`, removing the B4 assumption that a sector reuses its rank's reference denominator (true for RHF where eps is spin-free, false under UHF, where `abab` differs in *shape* too). One code path, not two: `sector_tensor` falls back to `tensor(rank)` when no per-block entry is stored, so RHF is bit-identical — verified by building with and against the change (`be_rccsdtq_sto3g`, the only landed method carrying a sector, `-14.4036550465` to every digit; extended suite 107/107). Gate verified falsifiable against three mutations. **The scope doc's remaining U2 item — make the state's reference a variant — should NOT be done**: measured, the generated kernels touch only `f_oo`/`f_ov`/`f_vv` and `orbital_partition`, never `RHFReference` as a type, so a variant changes every kernel signature and every generated TU for no gain. Those three Fock blocks need the same spin split the ERIs do — that is U3, one change, not two. **Next is U3 + its emitter half.** **The rank-6 PySCF gap is CLOSED** (triples 2.3e-15, was rel 1.9e-3): the defect was in the comparison harness — `update_amps_uccsdt_tri_` updates t1/t2 **in place** before building the T3 intermediates, so `R = (t_new − t)·D` recovered through it is the residual at *different amplitudes*. Fixed by calling `compute_r3_tri_uhf` directly. Nine convention hypotheses had been falsified against it; none could have been right. Full answer: **`docs/CCGEN_UCC_NUMERIC_VALIDATION.md`** |
| `CCGEN_ARBITRARY_ORDER_UCC_SCOPE` (U3/U4/U5) | **U3 AND U4 LANDED; only U5 remains.** U3.0–U3.4 spin-blocked ERIs/Fock + emitter routing + open-shell MP2 limit; U4.0–U4.3 an ALL-SECTORS runtime bundle + the `--ucc` switch. Emitted UCC TU has **zero** untagged reads; RCC emit byte-identical throughout (SHA-256 pinned). **Four scope corrections worth carrying.** (1) U3's per-tag canonical block set had to be **derived** — a mixed block's orbits reach only 11 of 16 patterns, so the first restricted emit raised `NotImplementedError` on `vovv`; 6 arrays same-spin, **10** mixed. (2) U3.4 needed **no solver** (first-order MP2 amplitudes are closed-form), so it did not depend on U5 as scoped. (3) **U4.1 was not work at all** — pack/unpack and the update loop already tolerated an empty `by_rank`; U4.0's own gate had *asserted* the update was unreachable, inferring it from `max_rank()==0`, and the inference was wrong. (4) U4 was not a guard question: `validate_kernel_bundle` *required* `residuals_by_rank.size() == max_excitation_rank` while UCC pushes 0, so a UCC bundle was rejected before it ran; promoting one block per rank into the reference slot cannot fix it, because `rank_dims` gives one shape per rank while `aaaa` and `abab` differ in shape. **U4.2 fixed a real out-of-bounds read** (`by_rank[rank-1]` on a state with no reference blocks) that U4.0 had made reachable — removing the guard segfaults, exit 139. **Gate lessons**: a gate that re-implements the routing inline measures a *simulation* of the old code and cannot observe the fix (stayed red at 37 after U3.2 landed); a spin-blind-permutation mutation SURVIVED an array-name gate because what moves is index *order*; and two guards rejecting the same fixture means "it was rejected" asserts nothing — name the guard. **U5 is rescoped ~S → ~M**: `build_ucc_{spin_block_cache,fock_blocks,denominator_cache}` have **no production callers** (measured — tests only), and `prepare_generated_arbitrary_order_state` still builds the RHF reference and RHF cache unconditionally. So U5 is prepare-path wiring plus a UHF reference and a `rebind_physicist` counterpart (the oovv↔ovov cross-source is spin-sensitive), not a keyword. Land `ucc2` vs hand-written UCCSD before the FCI gate, and check the FCI system is not vacuous (Li/STO-3G makes t3 worth 0) |
| `CCGEN_UCC_NUMERIC_FIXTURE_SCOPE` + `CCGEN_UCC_RESIDUAL_EVALUATOR_SCOPE` | **COMPLETE — F1, F2.0–F2.4 and F3 all landed.** The UCC residuals are validated against PySCF UCCSD (CH3/STO-3G, all five blocks) to **~6e-16** — machine precision, gated at 1e-13 rather than the scoped 1e-10. Until this, every landed UCC residual was gated structurally only. Three scope claims were corrected by building it: the closed-shell oracle is a **per-target pairing, not a block sum**; the PySCF amplitude mapping is a **transpose, not a pure rename** (PySCF is `(occ…,vir…)`, ccgen is `(vir…,occ…)`); and **`f_ov` must be zeroed on BOTH sides** — one-sided zeroing is *worse* than neither (8e-9 → 9e-9 → 6e-17), since Planck CC kernels are canonical-Fock by construction while PySCF's `f_ov` is convergence noise that `update_amps` uses. Both vacuous-pass traps avoided and asserted. **U1.2 is unblocked; U1.3–U5 are the remaining UCC work** |
| `CCGEN_ARBITRARY_HARNESS_COST_SCOPE` | **research, not started** — H0 profile is blocking |
| `CCGEN_DRESSING_VS_PRODUCTION_CODES_SCOPE` | **research, D0 answered** — opened by "CFOUR/MRCC ship dressing as their only route, why did ccgen's fail?". D0 found the *derivation* route (`factorize.py`) also fails value preservation, **on GCC**, where there is no spin adaptation to blame: 23/66 `ccsd` doubles terms do not reproduce their source (‖diff‖/‖R‖ = 3.73e-01). So the retirement's decision stands but its stated reason does not. **The factorizer has no numeric gate** — its 47 tests compare factor `Counter`s, which cannot see index order. D1–D3 open |
| `CCGEN_KERNEL_SCALING_SCOPE` | **research, partly open** — H1 (memory-bound) untestable on the current ladder (tops out at 0.49 MiB `t3`); overlaps the cost scope, which hands off to it |

Two docs carried self-contradicting status lines ("nothing here is landed" above a LANDED entry) and
were corrected in the same pass: `CCGEN_ARBITRARY_ORDER_UCC_SCOPE` (U0) and
`CCGEN_KERNEL_WIRING_AND_BENCHMARK_SCOPE` (W0).

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

## ccgen generated-kernel performance

The dominant cost — the out-of-line, allocating tensor accessors — is fixed (see Completion).
What remains is the **scaling defect** the six-point ladder exposed: the generated-vs-hand-written
ratio grows from 21.8× to 50.1× with no plateau, and the generated cost does not obey a single
`o^a v^b` power law (21.4% residual, concentrated at high `v`). Full measurement in
`docs/CCGEN_KERNEL_SCALING_SCOPE.md`.

- **Enumerate the terms whose contraction order is wrong.** The high-`v` residual structure points
  at multiple contraction regimes — different residual terms wanting different orders, with the
  emitter picking none. `docs/CCGEN_HIGHER_OPERATOR_REUSE.md` already records `t2·t3·v` as `o⁵v⁵`
  n-ary against `o³v⁴` factored, superlinear in both indices and consistent with the measured
  `o^0.93 v^0.34`. Do this term-level enumeration **before** any emitter change.
- **Then consume `_optimal_contraction_order` in the emitter.** `python/ccgen/tensor_ir.py`
  defines `BLASHint` (`:66`), `_detect_gemm` (`:198`), and `_optimal_contraction_order` (`:283`),
  and `grep BLASHint python/ccgen/emit/planck_tensor_cpp.py` returns nothing — the emitter computes
  and discards all of it. This is the asymptotic fix. It outranks loop fusion, which was measured
  at 0.62× (i.e. no gain) at small size.
- **Firm up the exponents.** `o` spans only 4→8 across six points and the fit still leans on its
  endpoints (leave-one-out moves `o` across +0.40..+1.18, though it keeps its sign in all six
  variants). Two or three points in `o=8..12` would settle it. Treat `o^0.9 v^0.3` as indicative,
  not settled, until then.
- **The memory-bound hypothesis is untested, not refuted.** The whole reachable ladder stays under
  0.85 MiB `t3`, inside L2, so a cache transition cannot fire on it. Testing needs cc-pVDZ-class
  systems (H2O/cc-pVDZ is 6.5 MiB `t3`); at ~50× generated-kernel slowdown that run should be
  time-boxed before committing to it. Not exclusive with the scaling defect — it could add a term
  on top once the working set spills.
- **Rank 4 has no point on the ladder.** Different tensor types, different code path, plus the
  `-O1` registry pin (`CMakeLists.txt:402`) that rank 3 does not carry. The fixed-rank-only
  accessor pass already demonstrated rank 3 is not a proxy for rank 4 — do not assume the rank-3
  exponents transfer. The standing follow-on behind that pin (chunk the giant residual kernels in
  the ccgen emit so any optimization level stays cheap) is now worth re-costing, since the accessor
  no longer dominates.
- **Ladder-design constraint, for whoever extends this.** `choose_determinant_backstop`
  (`src/post_hf/cc/tensor_backend.cpp:241`) routes any case with `nso ≤ 16` **and** `ndet ≤ 10000`
  to the determinant-space teaching backstop, which never calls the generated tensor kernel. Such a
  case produces **no timing at all**, silently, regardless of `PLANCK_RCCSDT_BACKEND`. Any new
  ladder point must satisfy `nso > 16 || ndet > 10000`.

## ccgen dressed intermediates

**LANDED. Only the UCC follow-on remains.** See `docs/CCGEN_DRESSED_KERNEL_PIPELINE.md` for the
full record.

The problem this section used to describe — generated kernels carrying only *syntactic* CSE, never
the Stanton-Gauss dressed operators — is solved. Dressed CC kernels now generate from the build
(`-DPLANCK_CC_DRESS_OPERATORS=ON`), compile, link, and run, reproducing the undressed correlation
energy **and** iteration count at rank 3, pinned by the
`dressed_kernel_equivalence_rccsdt` regression case.

Route note, because this section previously scoped the wrong one: the retired plan was Option A's
exact-cover term algebra (A1-A4). What actually shipped is diagrammatic recognition — dressed
operators are matched as a topological subgraph property, which made A3's subgraph-isomorphism
problem the *mechanism* rather than an obstacle. `dressing.py`/`dressed_equation.py` carry it;
the old `tau.py` exact-cover route is history.

What remains open on the dressed path: **nothing in V1**. The follow-on is UCC
(`docs/CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md`, `docs/CCGEN_U1_UCC_ADAPT_SCOPE.md`) — U0 landed, U1
scoped as U1.0-U1.5, U2-U5 (the C++ side) ahead.

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
