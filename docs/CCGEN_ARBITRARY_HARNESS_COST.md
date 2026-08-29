# What does the generated rank-3 CC path cost, and where does the time go?

**Answered by profiling, 2026-08-29. The harness is ~1 %. The cost is one kernel
call, and two thirds of *that* was redundant operator construction — now fixed,
1.76x. The largest remaining lever is that CC has no OpenMP at all.**

| finding | measured | status |
|---|---|---|
| the harness (all-ranks loop, DIIS, energy) | **~1 %** | H2/H3/H4 **dead** |
| one call to the rank-3 kernel | **98.8 %** | H1 confirmed |
| …of which `build_W_*` operator builders | **67.7 %** | **FIXED** (H5) — 1.76x |
| …rebuilt once per `_partN` chunk | 1112 calls / 278 ops (rank 3); **16 092 / 894** (rank 4) | **FIXED** — 20.8x fewer calls at rank 4 |
| OpenMP anywhere in CC | **none**; 98.8 % CPU on 8 cores | **OPEN** (H6) — modelled 3.86x |

Opened by the option-2 decision in `CCGEN_RANK3_KERNEL_AND_SOLVER.md`: the
generated rank-3 kernels run in the arbitrary-order harness, which is correct
(CH4/STO-3G +1.49e-08 vs PySCF `rccsdt`) but far more expensive than the
hand-written tensor backend.

**Two framing corrections that this work established, before any number below is
quoted:**

1. **The generated/hand-written ratio is not like-for-like and is not a defect
   size.** Different solvers — wedge-packed vs dense amplitudes, cheap dressed
   intermediates vs a full generated kernel per rank, **40 vs 16 iterations on
   CH4**. A ratio across that boundary prices two algorithms. **Never set a target
   from it.**
2. **Profile generated-vs-generated.** Every measurement here compares
   configurations of the *same* solver. Two investigations were damaged by
   comparing across the solver boundary before this rule was written down.

---

## H0 — the profile

Two `steady_clock` bracket sets (~20 lines), then one `sample`. CH4/STO-3G,
Release, `PLANCK_RCCSDT_BACKEND=optimized`.

**H0a — phase attribution.** Consistent across all 15 iterations:

```
t=1.913s   t_res=1.913s   t_upd=0.000s   t_ene=0.000s
```

The residual evaluation is ~100 % of the iteration. **H4 (dense vs wedge DIIS
packing) is dead** — `t_upd` is unmeasurable, so the ~6x data through DIIS costs
nothing.

**H0b — per-rank** (`PLANCK_CC_RANK_TIME=1`):

| rank | elements | time | share |
|---|---|---|---|
| 1 | 20 | 0.0008 s | 0.04 % |
| 2 | 400 | 0.0237 s | 1.1 % |
| **3** | **8000** | **2.0775 s** | **98.8 %** |

**H2 (the harness evaluates every rank each iteration) is dead** — ranks 1+2 are
~1.2 % combined, so caching or skipping them buys at most 1 %.

**H0c — inside the kernel** (`sample`, 20 s, 16 882 leaf samples):

| | samples | share |
|---|---|---|
| **`build_W_*` operator builders** | **11 422** | **67.7 %** |
| `compute_ccsdt_triples_residual_part*` | 5 334 | 31.6 % |
| everything else | 126 | 0.7 % |

Dominant families: `t2t2v_oooovv` 33.9 %, `t1t3v_oooovv` 14.7 %, `t1t1t2v_oooovv`
9.5 %.

**H3 (intermediates for spin-adapted emission) collapses into H1** — its value was
reducing recomputation across the residual, and that recomputation is entirely
inside the rank-3 kernel.

**This retired the document's own framing.** It is not an arbitrary-harness cost
problem: the harness is ~1 %.

### Why brackets before sampling

The iteration loop's three phases were already delimited by function calls, so a
sampling profile would have spent its resolution rediscovering a boundary the
source states exactly. Brackets also persist in the log, so any later run is
self-documenting. Sampling earned its place at *"where inside the residual"*,
which is H0c.

---

## H5 — operators rebuilt once per chunk (FIXED, 1.76x)

`_emit_chunked_kernel` emitted every dressed operator inside **every** `_partN`
function. **The duplication factor equals the part count:**

| kernel | parts | distinct ops | builder calls | after H5 | reduction |
|---|---|---|---|---|---|
| `ccsdt` triples | 4 | 278 | 1112 | 278 | 4.0x |
| **`ccsdtq` quadruples** | **18** | **894** | **16 092** | **894** | **18.0x** |
| ccsdtq TU total | — | — | **50 601** | **2431** | **20.8x** |

**The waste scaled with kernel size — worst precisely at the production target.**
The rank-3 ratio did not transfer; rank 4 is 5x larger.

**The fix.** Build each operator once in the main kernel into a generated
`<kernel>_ops` struct, passed to the parts by `const&`. Each part binds
`const auto &W_x = ops.W_x;` so every term body is unchanged. Struct members use
`_tensor_type(spec.rank)` — the same expression the builder's own definition uses
for its return type, so the two cannot drift.

**Measured** (CH4, best-of-2, same binary configuration apart from H5):

| | before | after | |
|---|---|---|---|
| wall | 29.59 s | **16.81 s** | **1.76x** |
| builder share | 67.7 % | 45.1 % | |
| builder time | 20.03 s | 7.58 s | 2.64x |
| residual time | 9.56 s | 9.23 s | 1.04x |

BH3: 9.52 s → 5.57 s (1.71x). `E_corr` **bitwise identical** on both generated
gates — hoisting reassociates nothing, so this was exactly bitwise as predicted.
Rank-4 TU shrinks **12.8 → 10.5 MB** with **48 170 fewer call sites**, a
compile-time win on an `-O1`-pinned TU. The undressed path is byte-identical (no
operators to hoist).

**The residual at 1.04x is the check that the decomposition is sound:** H5 does
not touch the residual, so it must be flat, and it is.

### What the prediction got wrong

Predicted builders 67.7 % → ~17 % and ~50 % faster. Measured **45.1 %** and 43 %
faster. Builder time fell **2.64x**, not the **4x** the call count implies — so the
eliminated builds were **cheaper than average**, plausibly the small-rank
operators, since a part's 270 builds span every rank while its 59-118 *used*
operators skew toward what its terms need.

**Builders are still 45 % of runtime after removing 75 % of the calls.** The
remaining builds are each genuinely needed once, so further gain is about what a
single build *costs*, not how often it runs — which is H6.

### Why the chunking assumption was reasonable and became false

`planck_tensor_cpp.py`: *"the intermediate builds … are re-emitted per part —
cheap, local, and keeps each part self-contained."* True when chunking was
introduced for an **undressed** emit, where `required_intermediates` is empty and
the statement costs nothing. Dressing populated that list and nothing re-examined
the claim. **Self-containment was the goal; the price went unmeasured until H0c.**

### Not done

A rank-4 dressed **end-to-end run** — that means compiling a 10.5 MB TU plus the
`-O1` registry. The mechanical and structural evidence is unambiguous and the
rank-3 correctness gate passed bitwise. Recorded as not done rather than implied:
if a rank-4 dressed tree is ever built for another reason, run `be_rccsdtq_sto3g`
and check `E_corr` against `-14.4036550465`.

---

## Hotspot ranking (post-H5, HF/6-31G, 25 483 leaf samples)

Profiled on the largest tractable case rather than CH4, so the shares reflect a
size where the residual matters most.

| # | hotspot | share | fixable? | lever |
|---|---|---|---|---|
| 1 | **triples residual, `part1`** | **44.8 %** | **hard** | it is the `o²v²`-deep terms; H1's two models are spent |
| 2 | **`build_W_t2t2v_oooovv` (38 builders)** | **20.4 %** | **YES — mechanism exists** | `merge_transposes`, built and value-gated, **never threaded** |
| 3 | triples residual, `part0` | 13.8 % | hard | same as (1) |
| 4 | `build_W_t1t3v_oooovv` | 8.9 % | yes, same lever as (2) | |
| 5 | `build_W_t1t1t2v_oooovv` | 5.6 % | yes, same lever as (2) | |
| 6 | remaining builders (137 of 141) | 4.0 % | tail | |
| 7 | singles/doubles residual | 1.0 % | no | already ~free |
| — | **everything, at once** | **100 %** | **YES** | **H6 — no OpenMP anywhere in CC** |

### Why the parts are unequal — it is not chunk size

All three heavy parts hold **256 nests each**, but their modelled cost at `o=5 v=6`
differs by 18x:

| part | nests | modelled cost | deepest inner sum |
|---|---|---|---|
| part0 | 256 | 2.46e+08 | `o²v¹` |
| **part1** | **256** | **6.68e+08** | **`o²v²`** |
| part2 | 256 | 3.72e+07 | `o²v⁰` |
| part3 | 38 | 5.13e+06 | `o¹v⁰` |

**Chunking splits by term COUNT, not by cost.** `part1` collects the `o²v²` terms
and is 2.7x `part0` and 18x `part2`. That is not itself a defect — the same total
work is done — but it means any future per-part parallelism would be badly
unbalanced, and a cost-weighted split is the cheap fix if that is ever needed.

### Hotspot 2 is the actionable one: 38 builders, ONE contraction

The `t2t2v_oooovv` family is **38 distinct emitted builders**, each writing a
**rank-6 result over a 9-deep loop nest** (`i,j,k,l,b,c,m,d,e`). Normalizing index
names away, **all 38 are the same contraction**:

```
t2(....) * t2(....) * mo_blocks.oovv(....)
```

They differ only in index placement:

```
t2({i,j,b,d}) * t2({m,k,e,c}) * oovv(l,m,d,e)
t2({i,j,b,d}) * t2({m,k,e,c}) * oovv(l,m,e,d)
t2({i,j,d,b}) * t2({m,k,e,c}) * oovv(l,m,d,e)
t2({i,j,d,b}) * t2({m,k,e,c}) * oovv(l,m,e,d)
```

**`merge_transposes` already solves exactly this.** `operator_identity.py` decides
transpose-equivalence symbolically, is exact against a numeric oracle, and is
value-gated **0/2536 at rank 4**. It is **not threaded into the production dressing
path** — `CCGEN_MERGE_TRANSPOSES_SCOPE.md` scopes the wiring and was deferred on
the grounds that its modelled FLOP saving is only 1.02x-1.20x and the likely win is
compile time.

**That estimate now looks wrong, and this profile is why.** It was derived from an
operator-count model. Measured, this one family is **20.4 % of runtime**, and the
top three mergeable families are **34.9 % combined**. Merging cannot remove all of
that — the merged operator must still be built once — but it can remove the
duplicate builds, and 38→1 is the same shape of win H5 just delivered at 1.76x.

**Recommended: re-cost `merge_transposes` against this profile before anything
else**, because the mechanism is built, gated, and unthreaded. That is the same
position the derivation route was in before it turned out to be worth 3.6x.

## H6 — OpenMP: the largest remaining lever (OPEN)

**There is zero OpenMP anywhere in CC** — none in `src/post_hf/cc/*.cpp`, none in
the generated kernels, and the emitter never emits a pragma. Confirmed at runtime:
a CH4 solve with `OMP_NUM_THREADS=8` on an 8-core machine runs at **98.8 % CPU** —
one core busy, seven idle. Every other hot path in Planck (ERI, Fock builds, the
4-index transforms, the DFT J/K builds) is threaded; **CC is the exception.**

Amdahl on H5's measured split (builders 45.1 %, residual 53.7 %, other 1.2 %). This
machine has 4 performance cores / 8 logical, so `n=4` is the realistic ceiling:

| threads | both parallel | residual only | builders only |
|---|---|---|---|
| 2 | 1.98x | 1.37x | 1.29x |
| **4** | **3.86x** | 1.67x | 1.51x |
| 8 | 7.38x | 1.89x | 1.65x |

**3.86x at 4 threads is larger than every lever found so far combined** (dressing
3.6x, H5 1.76x), and it is the only remaining item that addresses both halves of
the split at once.

**Both sites are reduction-free, and the builders are the better shape:**

- **Builders (45.1 %) — embarrassingly parallel.** 270 independent calls, each
  writing its own freshly-allocated tensor. **No write sharing at all.** No
  reduction, no ordering.
- **Residual nests (53.7 %) — parallel but coarser.** Each nest's outer `i` writes
  disjoint `result(i,...)` slices, so no reduction either. But the trip count is
  `no` = 4-8, giving 1-2 iterations per thread at `n=4`. **Collapse `i,j`**
  (`o²` = 16-64 trips); writes stay disjoint.

**Why this is lower-risk than the DFT precedent.** The historical DFT jitter came
from a **cross-thread reduction summed in completion order**
(`dft_xc_reduction_determinism`). Neither site here has a reduction — which is the
same property that made the DFT J/K builds bitwise-invariant across thread counts.
**Verify it the same way: energies bitwise identical across `OMP_NUM_THREADS` =
1/2/4/8**, not assumed from the argument.

**Sequencing: builders first.** Better granularity, no write sharing, 45 % of the
time, and one `#pragma omp parallel for` over the emitted build list in the main
kernel — **where H5 has already collected them into one place.** Before H5 the
builds were scattered across four `_partN` functions.

**Not yet measured:** whether thread overhead is amortized at these sizes. CH4's
`o³v³` is 8000 elements and a per-builder task is small. The honest first step is
one `parallel for` over the builder list, measured — not a threading strategy
designed in advance.

---

## What this says about method

| lever | how found | outcome |
|---|---|---|
| contraction order | census + FLOP model | **hit** (3.6x) |
| loop fusion | census + traffic model | **miss** (~0 %) |
| **chunk rebuilds (H5)** | **`sample` profile** | **hit** (1.76x) |
| **no OpenMP (H6)** | **asking whether it was threaded** | modelled 3.86x |

The two models cost days and went 1-for-2. The profile cost ~20 minutes and found
a defect neither model could see, because **neither modelled work that should not
happen at all** — both priced the residual's arithmetic while two thirds of the
time was redundant operator construction outside it.

**A misreading worth naming.** `CCGEN_KERNEL_SCALING_SCOPE.md`'s "measurement route
is closed" forbids comparing generated and hand-written *residuals*, and makes
their wall-clock non-comparable. It never prevented profiling the generated path
**against itself** — which is what found both remaining causes. It had been treated
as a reason the whole question needed modelling.

---

## Acceptance, for anything that continues this

- **No target number from the generated/hand-written ratio.** It is a ratio across
  a solver boundary.
- **Correctness preserved bitwise**: `ch4_rccsdt_generated_sto3g` and
  `lih_rccsdt_generated_sto3g` with identical `E_corr`. Any change that
  reassociates floating-point accumulation is not evaluation-order-preserving and
  needs its own justification rather than absorption into a tolerance.
- **"There is no lever here" is a legitimate outcome** and cheaper to establish
  than the two refuted models were.

## Constraints

- **Do not reintroduce the hybrid.** Mixing hand-written and generated residual
  sources produced a self-consistent wrong answer (−7.56e-05) and is rejected on
  correctness, not cost.
- **`make -j4`.** These TUs are large enough that a full-width build is disruptive.
- **Explicit `CMAKE_BUILD_TYPE`.** The repo's `build/` has it empty, which drops
  `-DNDEBUG`, re-enables the CC tensor bounds asserts, and makes every timing
  meaningless — this cost a wrong diagnosis once already.
- **Non-square, backstop-clearing test system.** CH4/STO-3G (`no=5 nv=4`,
  `nso=18 ndet=43758`).
- **Do not compare absolute `sample` counts across runs** of different duration.
  One attempt reported the residual "falling" 0.60x under a change that does not
  touch it. Use wall-clock and within-run shares.
- **Do not use `git stash` to obtain a historical emitter.** An attempt popped an
  unrelated stash and produced conflicts. `git archive <commit> python` into a
  scratch tree instead.

## Key code locations

| what | where |
|---|---|
| phase brackets (H0a) | `run_generated_arbitrary_order_iterations`, `generated_arbitrary_runtime.cpp` |
| per-rank timing (H0b) | `PLANCK_CC_RANK_TIME`, `evaluate_generated_arbitrary_order_residuals`, same file |
| the hoist (H5) | `_emit_chunked_kernel`, `python/ccgen/emit/planck_tensor_cpp.py` |
| the chunking assumption it corrected | same function's header comment |
| Jacobi/DIIS update (H4, dead) | `update_amplitudes_with_jacobi_diis`, `solver_arbitrary.cpp` |
| why the correct path is this harness | `docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md` |
| the emitter-side levers, one hit one miss | `docs/CCGEN_WHY_GENERATED_IS_SLOW.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
