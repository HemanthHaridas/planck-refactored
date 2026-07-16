# ERI Engine Performance — Architecture Note

Canonical status lives in `vault/Status/Completion.md` and `vault/Status/Open
Work.md`. This file answers one architecture question:

**Planck's ERI Fock build is ~200× slower than libcint. Where is that gap, and
which speed levers are worth pulling?**

## Short answer

The gap is **genuine algorithmic work executed at near-peak hardware efficiency**
by a scalar-per-quartet Obara-Saika engine. It is *not* a fixable inefficiency in
the current engine. Four candidate optimizations were each scoped and then
**disproven by measurement** (profile, disassembly, or hardware counters) before
any of them was implemented. The only win that survived is a ~4% vector-Boys seed.

Closing the remaining gap requires a *different algorithm* — a fundamentally
batched (libcint-class) engine — which the measurements also show is not worth the
hermeticity/complexity cost unless a real d/f production workload is demonstrably
Fock-bound. The `eri_slow` column in `tests/benchmarks/pyscf_bench.py` (Planck
Fock-build vs PySCF libcint `get_jk`) is the instrument to prove that first.

## Where the time actually goes

Profiled on the OS direct-SCF Fock build (`_compute_2e_fock_direct`), `sample`
at -O3, 6-31g* and cc-pVTZ single water:

| component | share | notes |
|---|---|---|
| `_os_contract_a0c0` / VRR | **~60–74%** | the per-primitive-pair vertical recurrence |
| `Lookup::boys()` | ~26% (6-31g*), ~8% (cc-pVTZ) | inside the VRR seed |
| `_os_eri_hrr_to_eri` (HRR) | ~3.5% | negligible |
| ShellPair construction | **0 frames** | invisible at both low and high contraction |

OS and HGP profile within ~10% of each other — a *faster kernel* (which is all
Libint would supply) hits the same wall. The lever, if one exists, is the VRR.

## The four disproven levers

Each was scoped as a small, reversible, bitwise-gated change, then killed by a
measurement. They are recorded here so none is re-proposed.

### 1. Strided-pointer VRR (remove per-element `spatial_index` math)

**Hypothesis:** `EriScratch::v()` re-derives a strided index (6 mul + 5 add) per
element, ~30–40 int ops per FLOP; replace with a base-pointer walk.

**Disproven by disassembly:** at -O3 `spatial_index` is fully inlined into
`_eri_vrr` and does not appear as its own frame — samples land on the recurrence
lines. The compiler already strength-reduces the constant strides. A manual
pointer walk removes nothing. (Also: "avoid raw pointers" is an API/ownership
rule, not the cause — the hot loops already use `double*` / `double(*)[13][13]`.)

### 2. Shell-granular ShellPair (build the `nprim²` loop once per shell pair)

**Hypothesis:** `ShellPair spCD(cvC,cvD)` rebuilds the primitive-pair loop
(`std::pow`/`std::exp`/product centers) per Cartesian-component pair —
redundant, since only one scalar norm factor differs across components.

**Disproven by profiling:** ShellPair construction is **0 profile frames** at both
6-31g* and cc-pVTZ. Construction is O(nprim²); the VRR it feeds is O(nprim²·n_a·
n_c·MMAX), so the VRR dwarfs it by the angular factor even at deep contraction.
The redundancy is real but off the critical path. And reducing the *count* of VRR
invocations (the hoped-for win) is impossible here: the count equals the number of
non-screened component quartets, which shell granularity does not change.

### 3. SIMD-vectorize the VRR

**Hypothesis:** vectorize the recurrence; a disassembly showed 808 loads/stores
vs ~296 fp ops (2.7:1), suggesting memory-bound.

**Disproven by hardware counters:** `xctrace "CPU Counters"` on cc-pVTZ, IPC
differenced across 30,256 P-core intervals → **median IPC = 5.68** (Apple P-core
peak ~6–7). The VRR retires at ~85–95% of peak — compute/issue-bound, *not*
memory-stalled (cache-miss counters ~0.01–0.08/cycle; a stalled loop would be IPC
~1–2). The 808 loads/stores hit L1 and pipeline cleanly. The m-loop is *already*
auto-vectorized (162 `fmul.2d`). Buffer-locality tuning has no problem to fix, and
primitive-batched SoA SIMD fights for the last ~15% of a saturated core on 2-wide
NEON — not worth 1–2 weeks. (A wider ISA like AVX-512 could change this, but the
target is arm64 and the codebase is portable-hermetic.)

### 4. A libint-class engine (AM-dispatched contract-once block)

Audited against Planck, libint's four structural moves reduce to: (1) compute each
canonical ERI once + reuse over the 8-fold permutation — **already done**
(`fused_fock.h` + `fock_accumulate`); (2) shell-pair Schwarz screening — **already
done** (`qmax` block bound); (3) contract (a0|c0) once per shell quartet, HRR each
component out — **built** (`_contracted_eri_block_hoisted`) but unwired; (4) batch
shell blocks through a path chosen by angular momentum — the only missing piece.

**Hypothesis:** move #4 (`hoist_wins(L, n_comp, n_prim)` dispatch) finally makes #3
pay by taking the hoisted path only where it wins (d/f) and the cheap
per-component path on sp.

**Disproven by the crossover measurement** (`PLANCK_FORCE_HOIST` forced on,
sto-3g → cc-pVDZ, energies bitwise-matched):

| case | AM / contraction | hoist ÷ default |
|---|---|---|
| sto-3g 3w | s,p | 1.46× slower |
| 6-31g* 1w | +d | 1.62× slower |
| 6-31g* 2w | +d | 1.14× slower |
| cc-pVDZ 1w | +d, ~8-prim contracted s | 1.23× slower |
| cc-pVTZ 1w | +f | **SIGBUS** |

There is **no `hoist_wins` region**: the hoisted block is slower at every AM /
contraction that runs — including cc-pVDZ, the deepest contraction tested —
because the snapshot-copy + per-component sub-box gather overhead exceeds the
re-contraction it saves before any break-even. The OS hoisted kernel also crashed
on f shells (untested past d); it was never production-reachable (Auto routes
OS-chosen quartets to the per-component path), so it was pure dead code and has
since been deleted. The live HGP/Rys hoisted kernels (reached via the `_compute_2e`
Auto path) *do* work on f/h shells and are now gated there.

## What landed, and what remains

**Landed** (PR #148): `Lookup::boys_vec` — the VRR seed fills `F_0..F_MMAX` in one
call instead of a scalar `boys(m,T)` per order, sharing the table-index setup.
Bitwise-identical to the scalar path; ~4% on the 6-31g* ERI build. Plus the
`eri_slow` benchmark column and the f-shell HGP-hoisted gate.

**Remaining (all high-cost, low-certainty):** the only lever the measurements
leave is a wholesale batched engine — either importing Libint (drags in codegen, a
non-hermetic build, and an exception/pointer API for capabilities Planck mostly
already has) or a hand-written SoA-batched VRR on a wide-SIMD target. Neither is
justified by the current workload set (regressions top out at 6-31g*). **Revisit
only if `pyscf_bench.py --eri_slow` shows a real d/f production workload is
Fock-bound**, which no current case is.

## Rule for future ERI-speed work

Do not scope an ERI optimization from a flame-graph share alone. Every candidate
here looked promising from the profile and was killed by a *second, sharper*
measurement (disassembly, hardware counters, or a forced-on A/B). Before
committing to any VRR/engine rewrite, take the sharper measurement first — the
`eri_slow` column, a `sample` of the specific frame, or an `xctrace` IPC read.
