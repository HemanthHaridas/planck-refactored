# Shell-Pair Granularity — Remaining Work

Companion to `docs/SHELLPAIR_GRANULARITY_HANDOFF.md` (the architecture note).

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**What is left to do in the shell-pair-granularity (Rys hoist) effort, and why did the CASSCF g-basis speed investigation it surfaced rule out the "cache the Fock build" lever?**

None of the items below block the landed work.

## Short answer

The Rys hoist (`RysHoistedQuartet`) is wired into the Auto dispatch path (B-3) but not yet into explicit `engine rys` (B-4), the symmetry direct-SCF skeletons (Phase C), the gradient derivative-ERI path (Phase D), or the shell-pair data structure itself (Phase E) — each is an independent, unstarted follow-on. Separately, a CASSCF-on-g-basis slowness investigation this work surfaced found that the bottleneck is a memory ceiling (the 1.4 GB materialized AO ERI tensor), not a time cost from redundant Fock rebuilds — both "direct Fock" and "cache the core Fock" speed levers were implemented or measured and disproven; the only remaining cheap lever is trimming the optimizer cascade.

## Where the logic lives

- `RysHoistedQuartet`, `_contracted_eri_block_hoisted_views` (Rys hoist)
- `src/post_hf/casscf/casscf.cpp` — MCSCF `evaluate` lambda, `build_inactive_fock_mo`
- `src/integrals/os_symm.cpp`, `hgp_symm.cpp`, `rys_symm.cpp` — symmetry direct-SCF skeletons
- `compute_eri_deriv_dispatch` — gradient derivative-ERI dispatch loop
- `build_shellpairs`, `expand_shell_groups_to_ao_pairs` — shell-pair adapter
- Tests: `planck-compute-2e`, `planck-os-block-kernel`, `planck-hgp-engine-smoke`, `planck-rys-box-invariance`, `scripts/fit_auto_dispatch.py`
- Regression cases: `ne_rhf_ccpvqz_highL_pyscf`, `ne_rhf_ccpvqz_highL_engine_os_vs_rys`

## What invariants matter

### 1. A cached-tensor contraction is not the same cost class as recomputing integrals

The MCSCF `evaluate` lambda rebuilds the inactive/active Fock every candidate step via `ObaraSaika::_compute_fock_rhf(eri, D, nbasis)`. That looks like an O(n_AO⁴) integral cost repeated per candidate, but the tensor `eri` is built once (`ensure_eri`) and each `evaluate` call only re-contracts against the cached tensor — a cache-friendly, fast operation. The real cc-pVQZ blocker is the one-time 1.4 GB materialization, a memory ceiling, not a per-call time cost. Conflating the two led directly to the disproven "Phase A" direct-Fock attempt below.

Design rule:

- Before proposing to "avoid rebuilding the Fock," check whether the expensive step is the integral build (paid once) or the contraction (paid per call). Only the former benefits from direct/RI approaches; the correct lever for the latter is different.

### 2. A caching lever needs a stable key, not just a hypothesis about physical drift

Lever B assumed the inactive-core Fock could be cached across candidate evaluations because the core orbitals "move far less than full `C`." A characterization counter on `build_inactive_fock_mo` (water CAS(4,4)/6-31G) measured 1913 `_compute_fock_rhf` builds with core-column drift `‖C_prevᵀ C_core − I‖_F ≈ 0.65–0.73` between *consecutive* calls — not small. The redundancy is real but structural (no stable cache key exists across step-scales/roots/probe directions within a macro), not incremental.

Design rule:

- Before implementing a cache, measure the actual drift between consecutive calls on the real candidate-search pattern, not on an idealized "orbitals barely move" assumption.

### 3. A basin change on the SA-2 uphill canary is not automatically a bug

Routing the *active* Fock through the direct path (the A2 detour) flipped the basin-sensitive STO-3G uphill SA-2 canary to a different valid stationary point. This was bit-identical across reruns and threads, and traced to the load-bearing `numeric-newton` finite-difference Hessian amplifying a ~1e-15 direct-vs-tensor contraction-order difference by `/2ε`. It is a real sensitivity of that optimizer path, not nondeterminism.

Design rule:

- When a numerically-tiny (~1e-15) change flips which stationary point `numeric-newton` lands on, treat it as amplification of a real FD sensitivity, and confine the responsible code path with a basis-size or engine gate rather than trying to bit-match it away.

## What remains to be done

1. **B-4 — explicit `engine rys` (`_compute_2e`).** The Rys hoist is wired into the Auto path (B-3) but not into the non-Auto Rys tensor/Fock builds, so explicit `engine rys` is still per-component. Lower value since Auto is the default and explicit `engine rys` is rare. It adopts the same hoisted (reordered) order as B-3, so it must be gated at the `1e-13` ERI bar, not bitwise. The existing committed SA-2 `engine rys` uphill gate runs `_compute_2e`, so B-4 is the step that exercises that specific gate — kept robust by the already-landed B-2.5 plateau hardening and the B-2.5c reorder-robustness measurement (a ~1e-15 reorder left the uphill SA-2 case on its intended basin). Gate: `planck-hgp-engine-smoke`-style Rys self-consistency on g shells plus the SA-2 uphill case under `engine rys`.
2. **Phase C — symmetry direct-SCF skeletons** (`os_symm` / `hgp_symm` / `rys_symm`): still per-AO. Convert to shell-quartet carrying the signed-AO orbit per component (same pattern as the `sym_ops` branch). Now unblocked, since the block kernels exist for all three engines.
3. **Phase D — gradient derivative ERIs** (`compute_eri_deriv_dispatch` + loop): still per-AO. Auto gradients fall back to OS by design (the dispatcher only recognizes explicit `HeadGordonPople`); Rys has no derivative kernel at all, so true per-quartet Auto gradients need one written from scratch first. This is the largest remaining piece and was an explicit non-goal of the landed work.
4. **Phase E — retire the per-AO adapter.** Change `build_shellpairs` to emit shell-level `ShellPair` objects and drop `expand_shell_groups_to_ao_pairs`. This is the only step that changes the shared `ShellPair` contract, so it must come last and resolve the per-component `_component_norm` folding (a handoff invariant) across OS/HGP/Rys simultaneously.
5. **Cleanup — shared `SpatialQuartetLayout`.** OS's `EriScratch`, HGP's `EriScratch` / `g_hgp_scratch`, and Rys's `RysScratch` / `RysMaxBoxLayout` carry near-duplicate per-quartet 6-axis scratch. Extract the shared layout; the triangular-vs-rectangular distinction noted in the design should inform the interface. Pure refactor, no behavior change, gated by the existing cross-engine equality tests.

## Validation strategy that should remain in place

```
# from build/
./planck-compute-2e
./planck-os-block-kernel
./planck-hgp-engine-smoke
./planck-rys-box-invariance
# from repo root
python3 tests/run_regressions.py --suite all
python3 tests/engine_scf_energy_compare.py <input>   # OS == HGP == Rys == Auto
python3 tests/run_regressions.py --suite all --case ne_rhf_ccpvqz_highL_pyscf
python3 tests/run_regressions.py --suite all --case ne_rhf_ccpvqz_highL_engine_os_vs_rys
# Auto dispatch re-fit (0 median disagreements):
python3 scripts/fit_auto_dispatch.py
```

## Related but separate outcome: CASSCF on a g-basis is slow

This was not a Rys task — it surfaced when the B-3 `engine auto` cc-pVQZ SA-CASSCF gate proved unrunnable (>15 min, no macroiteration completed, on both a laptop and a larger machine). That gate was dropped: B-3 is an integral-path change, so its correctness is already proven by the *integral* equality gate `ne_rhf_ccpvqz_highL_engine_os_vs_rys` (OS == RYS == AUTO SCF energy on g-shells, ~14 s, now exercising the Auto-Rys hoist), and its basin robustness is already measured by B-2.5c.

**Profiled root cause.** The MCSCF `evaluate` lambda (`casscf.cpp`, called per candidate orbital step, not just per macro) rebuilds the inactive and active Fock via `ObaraSaika::_compute_fock_rhf(eri, D, nbasis)` — a dense O(n_AO⁴) contraction over the full materialized AO ERI tensor (`ensure_eri`, ~1.4 GB at cc-pVQZ water). Cost is n_AO⁴ × (≈7 step scales + stagnation probes) × macros, so it explodes with basis size even though the active space is tiny. The AO Fock build is already OpenMP-parallel, so more cores help only linearly.

**Correction to the root-cause framing.** The dense `_compute_fock_rhf(eri, D)` contraction is O(n_AO⁴), but the tensor `eri` is built once (`ensure_eri`) and then re-contracted each `evaluate` — a cache-friendly, fast operation. It is the 1.4 GB allocation that is the cc-pVQZ blocker, i.e. a memory problem, not a time problem. The time cost per `evaluate` is the contraction, not a tensor rebuild. This distinction is what disproved lever A below.

### Levers considered, ranked

- **A (direct Fock) — disproven for speed.** See "Phase A" below. Direct/RI only ever helps memory (dropping the 1.4 GB tensor), and only via RI (approximate), not a direct rebuild.
- **B (medium) — premise disproven as a cache; real fix is optimizer restructuring.** The idea was that the inactive core Fock depends only on the core orbitals, which "move far less than full `C`," so it could be cached across candidate evaluations. Measured false (see invariant 2 above): the redundancy (~1913 builds) is structural, not incremental, since there is no stable cache key. A genuine win therefore requires restructuring the candidate search so it does not re-enter the full `evaluate` lambda (and its O(n⁴) inactive-Fock rebuild) per step-scale — bigger than the original ~1-day estimate (~2–3 days), and it carries basin-stability risk on the SA-2 uphill canary (the numeric-newton FD Hessian amplifies ~1e-15 perturbations). Reusing a stale accepted-macro `F_I_ao` across candidates is an approximation, not a refactor, and would need its own validation.
- **C (low, already planned)** — trim the macro-optimizer cascade (see `vault/Status/Open Work.md`, "CASSCF P2"). `numeric-newton` is load-bearing (some cases only converge with it) and must not be demoted. The trim is limited to removing redundant per-root candidates / pair probes, not the numeric-newton path.

### Phase A — direct Fock in MCSCF: attempted, reverted (disproven for speed)

The hypothesis was that building the inactive/active Fock directly from shell-pair ERIs (tensor-free) would beat the per-`evaluate` dense contraction of the cached tensor. It was implemented (A0 seam + equivalence gate, A1 inactive Fock wired in Cartesian mode), passed all 10 CASSCF cases (the direct build is the same `J − ½K` operator to ≤1.8e-15), then benchmarked and reverted because it is slower.

**Benchmark (water CAS(4,4), direct vs forced-tensor inactive Fock):**

| basis | AOs | direct | tensor | direct/tensor |
|---|---|---|---|---|
| STO-3G | 7 | 0.62 s | 0.32 s | 1.9x slower |
| 6-31G | 13 | 4.01 s | 1.42 s | 2.8x slower |
| cc-pVDZ | 25 | 70.8 s | 17.1 s | 4.1x slower |

**Why the premise was wrong.** The cached tensor `eri` is built once (`ensure_eri`) and the per-`evaluate` cost is a dense contraction against it — cache-friendly and fast. The direct path re-runs the full HGP VRR/HRR machinery on every `evaluate` call (dozens per macro x many macros), so it pays the integral cost repeatedly instead of amortizing it. Recompute-each-time only beats reuse-cached when the tensor cannot be built/held at all, which is far larger than the systems CASSCF active spaces reach. The 1.4 GB at cc-pVQZ is a memory ceiling, not a time cost; conflating the two is what drove the bad scope.

**The A2 detour (also reverted, instructive).** Before the speed benchmark, routing the active Fock direct (A2) was found to flip only the basin-sensitive STO-3G uphill SA-2 canary (-74.7877864784 -> the other valid stationary point -74.7751377977), via the load-bearing `numeric-newton` FD Hessian amplifying the ~1e-15 reorder by `/2ε`. Not a bug (bit-identical across reruns and threads); a fundamental direct-vs-tensor contraction-order difference unfixable without bit-matching. That alone would have confined direct to a basis-size gate; the speed benchmark then removed the case for direct Fock entirely.

**What remains for CASSCF speed.** Direct Fock is off the table for time. Lever B as a cache is disproven; its surviving form is an optimizer restructuring (~2-3 days, basin-risk on the SA-2 canary), not a drop-in cache. That leaves C (trim the optimizer cascade's redundant candidates/probes, keeping numeric-newton) as the one cheap live lever. The only direct/RI motivation left is memory (dropping the 1.4 GB tensor), which needs RI (approximate, opt-in behind a `casscf_ri` keyword), never the direct rebuild that this attempt disproved.
