# Why don't direct or cached Fock builds speed up CASSCF on a large basis?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`.

This file answers one architecture question:

**SA-CASSCF on a g-shell basis (water/cc-pVQZ) was unrunnable. Is the cost the
repeated inactive and active Fock builds inside the MCSCF candidate search, and
would building them directly, or caching them, fix it?**

## Short answer

No. The repeated Fock builds are dense contractions against an AO ERI tensor
that is built **once**. Those contractions are cache-friendly and fast, so
recomputing integrals directly each time is strictly slower: measured 1.9×,
2.8× and 4.1× slower on STO-3G, 6-31G and cc-pVDZ. Caching the inactive Fock
fails too. Core orbitals drift too much between consecutive candidate calls
for any stable cache key (`‖C_prevᵀ C_core − I‖_F ≈ 0.65–0.73`). The real
cc-pVQZ blocker is the one-time ~1.4 GB tensor, a **memory** ceiling, not a
time cost. That has a lever that already exists: `mp2_use_ri` routes every
ERI-dependent piece of the CASSCF evaluation through RI, and the AO tensor is
never built (`src/post_hf/casscf/casscf.cpp`).

The investigation surfaced while gating the Rys shell-quartet hoist
(`docs/SHELLPAIR_GRANULARITY_HANDOFF.md`). The cc-pVQZ SA-CASSCF gate it
planned (>15 min, no macroiteration completed) was dropped in favour of the
integral-level gate `ne_rhf_ccpvqz_highL_engine_os_vs_rys`.

## Where the logic lives

- `src/post_hf/casscf/casscf.cpp`: the MCSCF `evaluate` lambda, called per
  candidate orbital step, and `build_inactive_fock_mo`. Its RI branch is taken
  when `mp2_use_ri` is set.
- `ObaraSaika::_compute_fock_rhf(eri, D, nbasis)`: the dense contraction
  against the cached tensor.
- `HartreeFock::Correlation::ensure_eri`: the one-time tensor build.

## What was found

**Where the time goes.** The MCSCF `evaluate` lambda (`casscf.cpp`, called per candidate orbital step, not just per macro) rebuilds the inactive and active Fock via `ObaraSaika::_compute_fock_rhf(eri, D, nbasis)` — a dense O(n_AO⁴) contraction over the full materialized AO ERI tensor (`ensure_eri`, ~1.4 GB at cc-pVQZ water). Cost is n_AO⁴ × (≈7 step scales + stagnation probes) × macros, so it explodes with basis size even though the active space is tiny. The AO Fock build is already OpenMP-parallel, so more cores help only linearly.

**Correction to the root-cause framing.** The dense `_compute_fock_rhf(eri, D)` contraction is O(n_AO⁴), but the tensor `eri` is built once (`ensure_eri`) and then re-contracted each `evaluate` — a cache-friendly, fast operation. It is the 1.4 GB allocation that is the cc-pVQZ blocker, i.e. a memory problem, not a time problem. The time cost per `evaluate` is the contraction, not a tensor rebuild. This distinction is what disproved lever A.

### Lever A: direct Fock, attempted and reverted

The hypothesis was that building the inactive/active Fock directly from shell-pair ERIs (tensor-free) would beat the per-`evaluate` dense contraction of the cached tensor. It was implemented (A0 seam + equivalence gate, A1 inactive Fock wired in Cartesian mode), passed all 10 CASSCF cases (the direct build is the same `J − ½K` operator to ≤1.8e-15), then benchmarked and reverted because it is slower.

**Benchmark (water CAS(4,4), direct vs forced-tensor inactive Fock):**

| basis | AOs | direct | tensor | direct/tensor |
|---|---|---|---|---|
| STO-3G | 7 | 0.62 s | 0.32 s | 1.9x slower |
| 6-31G | 13 | 4.01 s | 1.42 s | 2.8x slower |
| cc-pVDZ | 25 | 70.8 s | 17.1 s | 4.1x slower |

**Why the premise was wrong.** The cached tensor `eri` is built once (`ensure_eri`) and the per-`evaluate` cost is a dense contraction against it — cache-friendly and fast. The direct path re-runs the full HGP VRR/HRR machinery on every `evaluate` call (dozens per macro x many macros), so it pays the integral cost repeatedly instead of amortizing it. Recompute-each-time only beats reuse-cached when the tensor cannot be built or held at all. The 1.4 GB at cc-pVQZ is a memory ceiling, not a time cost, and the answer to a memory ceiling is RI (`mp2_use_ri`), which drops the tensor entirely, not a direct rebuild.

**The A2 detour (also reverted, instructive).** Before the speed benchmark, routing the active Fock direct (A2) was found to flip only the basin-sensitive STO-3G uphill SA-2 canary (-74.7877864784 -> the other valid stationary point -74.7751377977), via the load-bearing `numeric-newton` FD Hessian amplifying the ~1e-15 reorder by `/2ε`. Not a bug (bit-identical across reruns and threads); a fundamental direct-vs-tensor contraction-order difference unfixable without bit-matching. That alone would have confined direct to a basis-size gate; the speed benchmark then removed the case for direct Fock entirely.

### Lever B: cache the inactive Fock, premise disproven

The idea was that the inactive-core Fock depends only on the core orbitals,
which were assumed to move far less than full `C`, so it could be reused across
candidate evaluations. Measured on water CAS(4,4)/6-31G, it is false: 1913
`_compute_fock_rhf` builds, with core-column drift of 0.65–0.73 between
*consecutive* calls. The redundancy is structural (step scales, roots and probe
directions within one macroiteration), not incremental. A genuine win would
need the candidate search restructured so it does not re-enter the full
`evaluate` lambda per step scale. That is a 2–3 day change with basin risk on
the SA-2 uphill canary. Reusing a stale accepted-macro `F_I_ao` across
candidates is an approximation, not a refactor, and would need its own
validation.

### Lever C: trim the optimizer cascade, done

Per-root candidate steps were removed: a suite-wide sweep showed they never won
a merit selection. `numeric-newton` and single-pair probes were kept because
both are load-bearing (see the CASSCF entries in `vault/Status/Completion.md`).

## What invariants matter

### 1. A cached-tensor contraction is not the same cost class as recomputing integrals

The MCSCF `evaluate` lambda rebuilds the inactive/active Fock every candidate step via `ObaraSaika::_compute_fock_rhf(eri, D, nbasis)`. That looks like an O(n_AO⁴) integral cost repeated per candidate, but the tensor `eri` is built once (`ensure_eri`) and each `evaluate` call only re-contracts against the cached tensor — a cache-friendly, fast operation. The real cc-pVQZ blocker is the one-time 1.4 GB materialization, a memory ceiling, not a per-call time cost. Conflating the two led directly to the disproven direct-Fock attempt (lever A above).

Design rule:

- Before proposing to "avoid rebuilding the Fock," check whether the expensive step is the integral build (paid once) or the contraction (paid per call). Only the former benefits from direct/RI approaches; the correct lever for the latter is different.

### 2. A caching lever needs a stable key, not just a hypothesis about physical drift

Lever B (above) assumed the inactive-core Fock could be cached across candidate evaluations because the core orbitals "move far less than full `C`." A characterization counter on `build_inactive_fock_mo` (water CAS(4,4)/6-31G) measured 1913 `_compute_fock_rhf` builds with core-column drift `‖C_prevᵀ C_core − I‖_F ≈ 0.65–0.73` between *consecutive* calls — not small. The redundancy is real but structural (no stable cache key exists across step-scales/roots/probe directions within a macro), not incremental.

Design rule:

- Before implementing a cache, measure the actual drift between consecutive calls on the real candidate-search pattern, not on an idealized "orbitals barely move" assumption.

### 3. A basin change on the SA-2 uphill canary is not automatically a bug

Routing the *active* Fock through the direct path (the A2 detour) flipped the basin-sensitive STO-3G uphill SA-2 canary to a different valid stationary point. This was bit-identical across reruns and threads, and traced to the load-bearing `numeric-newton` finite-difference Hessian amplifying a ~1e-15 direct-vs-tensor contraction-order difference by `/2ε`. It is a real sensitivity of that optimizer path, not nondeterminism.

Design rule:

- When a numerically-tiny (~1e-15) change flips which stationary point `numeric-newton` lands on, treat it as amplification of a real FD sensitivity, and confine the responsible code path with a basis-size or engine gate rather than trying to bit-match it away.

## Validation strategy that should remain in place

- `ne_rhf_ccpvqz_highL_engine_os_vs_rys` — OS == Rys == Auto SCF energy on
  g shells (~14 s); the integral-level gate that replaced the unrunnable
  cc-pVQZ SA-CASSCF gate
- the four SA-2 CASSCF regressions, especially
  `water_casscf_sa2_sto3g_sad_guess_uphill`, as the basin-sensitivity canary
  for any change to Fock-build order inside MCSCF
