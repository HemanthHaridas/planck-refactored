# Shell-pair Granularity (H-10) — Remaining & Deferred Work

Companion to `docs/SHELLPAIR_GRANULARITY_HANDOFF.md` (the architecture note).
Canonical status is `vault/Status/Completion.md` + `vault/Status/Open Work.md`;
this file holds the design-level detail for what is *not* yet done in the
shell-pair-granularity effort and the CASSCF speedups it surfaced.

None of these block the landed work.

## Rys hoist — B-4 (the one direct follow-on)

- **B-4 — explicit `engine rys` (`_compute_2e`).** The Rys hoist
  (`RysHoistedQuartet`, `_contracted_eri_block_hoisted_views`) is wired into the
  Auto path (B-3) but **not** into the non-Auto Rys tensor/Fock builds, so explicit
  `engine rys` is still per-component. Lower value (Auto is the default; explicit
  `engine rys` is rare). It adopts the same hoisted (reordered) order as B-3, so it
  is gated at the `1e-13` ERI bar, not bitwise. The *existing committed* SA-2
  `engine rys` uphill gate runs `_compute_2e`, so B-4 is the step that exercises
  that specific gate — kept robust by the already-landed B-2.5 plateau hardening
  and the B-2.5c reorder-robustness measurement (a ~1e-15 reorder left the uphill
  SA-2 case on its intended basin). Gate: `planck-hgp-engine-smoke`-style Rys
  self-consistency on g shells + the SA-2 uphill case under `engine rys`.

## Rys hoist — downstream phases (each independent)

- **Phase C — symmetry direct-SCF skeletons** (`os_symm` / `hgp_symm` /
  `rys_symm`): still per-AO. Convert to shell-quartet carrying the signed-AO orbit
  per component (same pattern as the `sym_ops` branch). Now unblocked — the block
  kernels exist for all three engines.
- **Phase D — gradient derivative ERIs** (`compute_eri_deriv_dispatch` + loop):
  still per-AO. Auto gradients fall back to OS by design (the dispatcher only
  recognizes explicit `HeadGordonPople`); Rys has **no derivative kernel at all**,
  so true per-quartet Auto gradients need one written from scratch first. Largest
  remaining piece; was an explicit non-goal.
- **Phase E — retire the per-AO adapter.** Change `build_shellpairs` to emit
  shell-level `ShellPair` objects and drop `expand_shell_groups_to_ao_pairs`. The
  only step that changes the shared `ShellPair` contract, so it must come **last**
  and resolve the per-component `_component_norm` folding (handoff invariant 2)
  across OS/HGP/Rys simultaneously.

## Cleanup

- **Shared `SpatialQuartetLayout`** (also in vault Open Work): OS's `EriScratch`,
  HGP's `EriScratch` / `g_hgp_scratch`, and Rys's `RysScratch` / `RysMaxBoxLayout`
  carry near-duplicate per-quartet 6-axis scratch. Extract the shared layout; the
  triangular-vs-rectangular distinction from the §4 design blocker should inform
  the interface. Pure refactor, no behavior change; gated by the existing
  cross-engine equality tests.

---

## CASSCF on a g-basis is slow (investigation)

Not a Rys task — surfaced when the B-3 `engine auto` cc-pVQZ SA-CASSCF gate proved
unrunnable (>15 min, no macroiteration completed, on both a laptop and a larger
machine). That gate was dropped: B-3 is an integral-path change, so its correctness
is the *integral* equality already proven by `ne_rhf_ccpvqz_highL_engine_os_vs_rys`
(OS==RYS==AUTO SCF energy on g-shells, ~14 s, now exercising the Auto-Rys hoist),
and its basin robustness is already measured by B-2.5c.

**Profiled root cause.** The MCSCF `evaluate` lambda (`casscf.cpp`, called *per
candidate orbital step*, not just per macro) rebuilds the inactive and active Fock
via `ObaraSaika::_compute_fock_rhf(eri, D, nbasis)` — a dense O(n_AO⁴) contraction
over the full materialized AO ERI tensor (`ensure_eri`, ~1.4 GB at cc-pVQZ water).
Cost is n_AO⁴ × (≈7 step scales + stagnation probes) × macros, so it explodes with
basis size even though the active space is tiny. The AO Fock build is *already*
OpenMP-parallel, so more cores help only linearly.

**Correction to the root-cause framing.** The dense `_compute_fock_rhf(eri, D)`
contraction is O(n_AO⁴), but the tensor `eri` is built **once** (`ensure_eri`) and
then *re-contracted* each `evaluate` — a cache-friendly, fast operation. It is the
1.4 GB allocation that is the cc-pVQZ blocker, i.e. a **memory** problem, not a
**time** problem. The time cost per `evaluate` is the contraction, not a tensor
rebuild. This distinction is what disproved lever A below.

**Levers, ranked.**

- **A (direct Fock) — DISPROVEN for speed; see "Phase A" below.** Rebuilding the
  Fock directly from shell-pair ERIs each `evaluate` *recomputes* integrals every
  call instead of reusing the cached tensor, and benchmarked **1.9×–4.1× slower**,
  worsening with basis size. Direct/RI only ever helps **memory** (dropping the
  1.4 GB tensor), and only via RI (approximate), not direct.
- **B (medium) — premise DISPROVEN as a cache; real fix is optimizer restructuring.**
  The idea was that the inactive core Fock depends only on the *core* orbitals,
  which "move far less than full `C`", so it could be cached across candidate
  evaluations. **Measured false.** A step-0 characterization counter on
  `build_inactive_fock_mo` (water CAS(4,4)/6-31G) logged **1913** `_compute_fock_rhf`
  builds, with core-column drift `‖C_prevᵀ C_core − I‖_F ≈ 0.65–0.73` between
  *consecutive* calls — not small. Consecutive calls are different trial step-scales /
  roots / probe directions within a macro, each with a substantially different core
  basis, so a drift-tolerance AO-Fock cache would hit ~0% of the time. The redundancy
  is real (~1913 builds) but **structural, not incremental**: there is no stable key.
  A genuine win therefore requires **restructuring the candidate search** so it does
  not re-enter the full `evaluate` lambda (and its O(n⁴) inactive-Fock rebuild) per
  step-scale — bigger than the original ~1-day estimate (~2–3 days, Bucket 2), and it
  carries basin-stability risk on the SA-2 uphill canary (the numeric-newton FD
  Hessian amplifies ~1e-15 perturbations). Reusing a stale accepted-macro `F_I_ao`
  across candidates is an *approximation*, not a refactor, and would need its own
  validation. Not the cheap lever it looked like.
- **C (low, already planned):** trim the macro-optimizer cascade (vault Open Work
  "CASSCF P2"). NOTE: this previously read "demote numeric-newton to debug-only" —
  **`numeric-newton` is load-bearing** (some cases only converge with it), so it
  must NOT be demoted. The trim is limited to removing redundant per-root
  candidates / pair probes, not the numeric-newton path.

### Phase A — direct Fock in MCSCF: ATTEMPTED, REVERTED (disproven for speed)

The hypothesis was that building the inactive/active Fock directly from shell-pair
ERIs (tensor-free) would beat the per-`evaluate` dense contraction of the cached
tensor. It was implemented (A0 seam + equivalence gate, A1 inactive Fock wired in
Cartesian mode), passed all 10 CASSCF cases (the direct build is the same `J − ½K`
operator to ≤1.8e-15), then **benchmarked and reverted** because it is slower.

**Benchmark (water CAS(4,4), direct vs forced-tensor inactive Fock):**

| basis | AOs | direct | tensor | direct/tensor |
|---|---|---|---|---|
| STO-3G | 7 | 0.62 s | 0.32 s | 1.9× slower |
| 6-31G | 13 | 4.01 s | 1.42 s | 2.8× slower |
| cc-pVDZ | 25 | 70.8 s | 17.1 s | 4.1× slower |

**Why the premise was wrong.** The cached tensor `eri` is built **once**
(`ensure_eri`) and the per-`evaluate` cost is a dense contraction *against* it —
cache-friendly and fast. The direct path re-runs the full HGP VRR/HRR machinery on
**every** `evaluate` call (dozens per macro × many macros), so it pays the integral
cost repeatedly instead of amortizing it. Recompute-each-time only beats
reuse-cached when the tensor cannot be built/held at all — far larger than the
systems CASSCF active spaces reach. The 1.4 GB at cc-pVQZ is a **memory** ceiling,
not a time cost; conflating the two is what drove the bad scope.

**The A2 detour (also reverted, instructive).** Before the speed benchmark, routing
the *active* Fock direct (A2) was found to flip only the basin-sensitive STO-3G
uphill SA-2 canary (−74.7877864784 → the other valid stationary point
−74.7751377977), via the load-bearing `numeric-newton` FD Hessian amplifying the
~1e-15 reorder by `/2ε`. Not a bug (bit-identical across reruns and threads); a
fundamental direct-vs-tensor contraction-order difference unfixable without
bit-matching. That alone would have confined direct to a basis-size gate; the speed
benchmark then removed the case for direct Fock entirely.

**What remains for CASSCF speed.** Direct Fock is off the table for time. Lever **B**
as a *cache* is disproven (see above — consecutive core bases differ by ~0.65–0.73,
no stable key); its surviving form is an **optimizer restructuring** (~2–3 days,
basin-risk on the SA-2 canary), not a drop-in cache. That leaves **C** (trim the
optimizer cascade's redundant candidates/probes, keeping numeric-newton) as the one
cheap live lever. The only direct/RI motivation left is **memory** (dropping the
1.4 GB tensor), which needs **RI** (approximate, opt-in behind a `casscf_ri`
keyword), never the direct rebuild that this attempt disproved.

## How to verify the landed state

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
