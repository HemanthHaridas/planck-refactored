# ERI Engine Performance

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Planck's ERI Fock build is ~200x slower than libcint. Where is that gap, and which speed levers are worth pulling?**

## Short answer

The gap is genuine algorithmic work executed at near-peak hardware efficiency by a scalar-per-quartet Obara-Saika engine. It is not a fixable inefficiency in the current engine. Four candidate optimizations were each scoped and then disproven by measurement (profile, disassembly, or hardware counters) before any of them was implemented. The only win that survived is a ~4% vector-Boys seed. Closing the remaining gap requires a different algorithm — a fundamentally batched (libcint-class) engine — which the measurements also show is not worth the hermeticity/complexity cost unless a real d/f production workload is demonstrably Fock-bound.

## Where the logic lives

- `src/integrals/os.cpp` (`_compute_2e_fock_direct`, `_os_contract_a0c0`, `_os_eri_hrr_to_eri`) — the OS direct-SCF Fock build and VRR/HRR
- `src/integrals/hgp.cpp`, `src/integrals/rys.cpp` — alternative engines profiled for comparison
- `Lookup::boys` / `Lookup::boys_vec` — the Boys function seed, the one landed optimization
- `src/integrals/shellpair.cpp` — ShellPair construction, measured off the critical path
- `tests/benchmarks/pyscf_bench.py` — the `eri_slow` column (Planck Fock-build vs PySCF libcint `get_jk`), the instrument for deciding whether to revisit this

## What invariants matter

### 1. A flame-graph profile share is not sufficient evidence to scope an optimization

Every one of the four candidate optimizations below looked promising from a profile share alone and was killed by a second, sharper measurement — disassembly, hardware counters, or a forced-on A/B comparison. A profile share tells you where time is spent; it does not tell you whether that time is removable.

Design rule:

- Before committing to any VRR/engine rewrite, take the sharper measurement first: the `eri_slow` benchmark column, a `sample` profile of the specific frame in question, or an `xctrace` IPC read. Do not scope work from a flame-graph share alone.

### 2. High load/store instruction count does not imply memory-bound

The VRR disassembly showed 808 loads/stores against ~296 fp ops (2.7:1), which looked like a memory-bound signature. Hardware counters (`xctrace "CPU Counters"` on cc-pVTZ, IPC differenced across 30,256 P-core intervals) instead measured median IPC 5.68 against an Apple P-core peak of ~6-7, with cache-miss counters at ~0.01-0.08/cycle — the VRR is compute/issue-bound, not memory-stalled (a stalled loop would show IPC ~1-2).

Design rule:

- Confirm memory-boundedness with actual hardware performance counters (IPC, cache-miss rate), not instruction-mix counting from a disassembly.

### 3. A per-quartet inefficiency that never executes on the critical path is not worth fixing

Shell-granular ShellPair construction rebuilds the primitive-pair loop redundantly per Cartesian-component pair, but measured at 0 profile frames at both 6-31g* and cc-pVTZ. Construction is `O(nprim^2)`, while the VRR it feeds is `O(nprim^2 * n_a * n_c * MMAX)`, so the VRR dwarfs it by the angular-momentum factor even at deep contraction, and reducing the VRR invocation *count* is impossible by this route since the count equals the number of non-screened component quartets.

Design rule:

- Verify a redundancy is actually on the critical path (via profiling) before optimizing it away, even when the redundancy is real and easy to see in the source.

### 4. A hoisted/batched kernel needs a measured crossover region, not just a plausible mechanism

A libint-class contract-once-per-shell-quartet hoisted kernel (`_contracted_eri_block_hoisted`) was built and the missing dispatch piece (`hoist_wins(L, n_comp, n_prim)`) was hypothesized to make it pay off at higher angular momentum. Forcing it on (`PLANCK_FORCE_HOIST`) across sto-3g through cc-pVDZ found no region where it won — it was slower at every AM/contraction tested, including the deepest contraction (cc-pVDZ), because snapshot-copy and per-component sub-box gather overhead exceeded the re-contraction it saved. It also crashed (SIGBUS) on f shells, since it was never production-reachable in the OS engine (Auto routes OS-chosen quartets to the per-component path).

Design rule:

- A hoisting/batching optimization must be validated by measuring an actual crossover point across the real angular-momentum/contraction range before being wired into a dispatcher — a plausible mechanism for why it should win at "large enough" AM is not sufficient.

## What was measured

### Profile: where the time actually goes

Profiled on the OS direct-SCF Fock build (`_compute_2e_fock_direct`), `sample` at -O3, 6-31g* and cc-pVTZ single water:

| component | share | notes |
|---|---|---|
| `_os_contract_a0c0` / VRR | **~60-74%** | the per-primitive-pair vertical recurrence |
| `Lookup::boys()` | ~26% (6-31g*), ~8% (cc-pVTZ) | inside the VRR seed |
| `_os_eri_hrr_to_eri` (HRR) | ~3.5% | negligible |
| ShellPair construction | **0 frames** | invisible at both low and high contraction |

OS and HGP profile within ~10% of each other — a faster kernel (which is all Libint would supply) hits the same wall. The lever, if one exists, is the VRR.

### Four candidate optimizations, each disproven

1. **Strided-pointer VRR** (remove per-element `spatial_index` math). Hypothesis: `EriScratch::v()` re-derives a strided index (6 mul + 5 add) per element, ~30-40 int ops per FLOP; replace with a base-pointer walk. Disproven by disassembly: at -O3 `spatial_index` is fully inlined into `_eri_vrr` and does not appear as its own frame — samples land on the recurrence lines. The compiler already strength-reduces the constant strides.
2. **Shell-granular ShellPair** (build the `nprim^2` loop once per shell pair). Disproven by profiling: 0 profile frames at both 6-31g* and cc-pVTZ (see invariant 3 above).
3. **SIMD-vectorize the VRR**. Disproven by hardware counters showing IPC 5.68, near peak, not memory-stalled (see invariant 2 above). The m-loop is already auto-vectorized (162 `fmul.2d`). Primitive-batched SoA SIMD would fight for the last ~15% of a saturated core on 2-wide NEON — not worth 1-2 weeks. (A wider ISA like AVX-512 could change this, but the target is arm64 and the codebase is portable-hermetic.)
4. **A libint-class engine** (AM-dispatched contract-once block). Audited against Planck's existing structure: canonical-ERI reuse over the 8-fold permutation is already done (`fused_fock.h` + `fock_accumulate`); Schwarz screening is already done (`qmax` block bound); contract-once-per-shell-quartet with HRR per component is built (`_contracted_eri_block_hoisted`) but unwired; only the AM-dispatched batching piece was missing. Disproven by the crossover measurement (see invariant 4 above and the table below).

### Hoisted-kernel crossover measurement

`PLANCK_FORCE_HOIST` forced on, sto-3g through cc-pVDZ, energies bitwise-matched against the default path:

| case | AM / contraction | hoist / default |
|---|---|---|
| sto-3g 3w | s,p | 1.46x slower |
| 6-31g* 1w | +d | 1.62x slower |
| 6-31g* 2w | +d | 1.14x slower |
| cc-pVDZ 1w | +d, ~8-prim contracted s | 1.23x slower |
| cc-pVTZ 1w | +f | **SIGBUS** |

There is no `hoist_wins` region at any AM/contraction that runs.

## What was built and landed

1. **`Lookup::boys_vec`** (PR #148) — the VRR seed fills `F_0..F_MMAX` in one call instead of a scalar `boys(m,T)` per order, sharing the table-index setup. Bitwise-identical to the scalar path; ~4% improvement on the 6-31g* ERI build.
2. **The `eri_slow` benchmark column** in `tests/benchmarks/pyscf_bench.py`, comparing Planck's Fock build against PySCF's libcint `get_jk` — the instrument for deciding whether a real production workload justifies revisiting this.
3. **The f-shell HGP-hoisted gate.** The live HGP/Rys hoisted kernels (reached via the `_compute_2e` Auto path) do work correctly on f/h shells and are gated there. The OS hoisted kernel, which crashed on f shells and was never production-reachable, has since been deleted as dead code.

## Validation strategy that should remain in place

- `tests/benchmarks/pyscf_bench.py --eri_slow`, watched for a real d/f production workload becoming demonstrably Fock-bound
- Re-measure with hardware counters (`xctrace` IPC, cache-miss rate) rather than a flame-graph share alone before scoping any future ERI optimization

## Remaining architecture concern

The only lever the measurements leave is a wholesale batched engine — either importing Libint (which drags in codegen, a non-hermetic build, and an exception/pointer API for capabilities Planck mostly already has) or a hand-written SoA-batched VRR on a wide-SIMD target. Neither is justified by the current workload set (regressions top out at 6-31g*). Revisit only if `pyscf_bench.py --eri_slow` shows a real d/f production workload is Fock-bound, which no current case is.
