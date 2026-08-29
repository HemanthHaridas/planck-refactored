# Where does the time go in the generated rank-3 CC path?

**Not in the harness — it is ~1 %.** Effectively all of it is one call to the
rank-3 kernel, and two thirds of *that* was redundant operator construction.
Profiled 2026-08-29; two fixes have since landed and one lever remains.

| finding | measured | status |
|---|---|---|
| the harness (all-ranks loop, DIIS, energy) | **~1 %** | dead end |
| one call to the rank-3 kernel | **98.8 %** | the whole cost |
| …of which `build_W_*` operator builders | **67.7 %** | **fixed** — chunk rebuilds, 1.76x |
| …of which duplicate transpose-equivalent builders | 34.9 % across 3 families | **fixed** — the merge, 1.42x-1.52x |
| OpenMP anywhere in CC | **none**; 98.8 % CPU on 8 cores | **OPEN** — modelled 3.86x |

Opened by the option-2 decision in `CCGEN_RANK3_KERNEL_AND_SOLVER.md`: the
generated rank-3 kernels run in the arbitrary-order harness, which is correct
(CH4/STO-3G +1.49e-08 vs PySCF `rccsdt`) but far more expensive than the
hand-written tensor backend. **The title's premise turned out to be wrong** — the
harness is not the cost — and that is the first result, not a caveat.

**Two framing rules this work established, before any number below is quoted:**

1. **The generated/hand-written ratio is not like-for-like and is not a defect
   size.** Different solvers — wedge-packed vs dense amplitudes, cheap dressed
   intermediates vs a full generated kernel per rank, **40 vs 16 iterations on
   CH4**. A ratio across that boundary prices two algorithms. **Never set a target
   from it.**
2. **Profile generated-vs-generated.** Every measurement here compares
   configurations of the *same* solver. Two investigations were damaged by
   comparing across the solver boundary before this rule was written down.

## The profile

Two `steady_clock` bracket sets (~20 lines), then one `sample`. CH4/STO-3G,
Release, `PLANCK_RCCSDT_BACKEND=optimized`.

**Phase attribution**, consistent across all 15 iterations:

```
t=1.913s   t_res=1.913s   t_upd=0.000s   t_ene=0.000s
```

Residual evaluation is ~100 % of the iteration, so **DIIS packing costs nothing** —
`t_upd` is unmeasurable, and the ~6x data through DIIS is free.

**Per rank** (`PLANCK_CC_RANK_TIME=1`):

| rank | elements | time | share |
|---|---|---|---|
| 1 | 20 | 0.0008 s | 0.04 % |
| 2 | 400 | 0.0237 s | 1.1 % |
| **3** | **8000** | **2.0775 s** | **98.8 %** |

So **evaluating every rank each iteration is not a defect worth fixing** — ranks
1+2 are ~1.2 % combined, and caching or skipping them buys at most 1 %.

**Inside the kernel** (`sample`, 20 s, 16 882 leaf samples):

| | samples | share |
|---|---|---|
| **`build_W_*` operator builders** | **11 422** | **67.7 %** |
| `compute_ccsdt_triples_residual_part*` | 5 334 | 31.6 % |
| everything else | 126 | 0.7 % |

Dominant families: `t2t2v_oooovv` 33.9 %, `t1t3v_oooovv` 14.7 %, `t1t1t2v_oooovv`
9.5 %.

**Why brackets before sampling.** The iteration loop's three phases were already
delimited by function calls, so a sampling profile would have spent its resolution
rediscovering a boundary the source states exactly. Brackets also persist in the
log, so any later run is self-documenting. Sampling earned its place at *"where
inside the residual"*.

## Fix 1 — operators were rebuilt once per chunk (1.76x)

`_emit_chunked_kernel` emitted every dressed operator inside **every** `_partN`
function. **The duplication factor equals the part count:**

| kernel | parts | distinct ops | builder calls | after | reduction |
|---|---|---|---|---|---|
| `ccsdt` triples | 4 | 278 | 1112 | 278 | 4.0x |
| **`ccsdtq` quadruples** | **18** | **894** | **16 092** | **894** | **18.0x** |
| ccsdtq TU total | — | — | **50 601** | **2431** | **20.8x** |

**The waste scaled with kernel size — worst precisely at the production target.**
The rank-3 ratio did not transfer; rank 4 is 5x larger.

Each operator is now built once in the main kernel into a generated `<kernel>_ops`
struct, passed to the parts by `const&`; each part binds `const auto &W_x =
ops.W_x;` so every term body is unchanged. Struct members use
`_tensor_type(spec.rank)` — the same expression the builder's own definition uses
for its return type, so the two cannot drift.

**Measured** (CH4, best-of-2, same configuration apart from this change):

| | before | after | |
|---|---|---|---|
| wall | 29.59 s | **16.81 s** | **1.76x** |
| builder share | 67.7 % | 45.1 % | |
| builder time | 20.03 s | 7.58 s | 2.64x |
| residual time | 9.56 s | 9.23 s | 1.04x |

BH3: 9.52 s -> 5.57 s (1.71x). `E_corr` **bitwise identical** on both generated
gates — hoisting reassociates nothing. Rank-4 TU shrinks **12.8 -> 10.5 MB** with
**48 170 fewer call sites**. The undressed path is byte-identical (no operators to
hoist). **The residual at 1.04x is the check that the decomposition is sound:** the
fix does not touch the residual, so it must be flat, and it is.

**What the prediction got wrong.** Predicted builders 67.7 % -> ~17 % and ~50 %
faster; measured **45.1 %** and 43 % faster. Builder time fell **2.64x**, not the
**4x** the call count implies — the eliminated builds were **cheaper than
average**, plausibly the small-rank operators, since a part's 270 builds span every
rank while its 59-118 *used* operators skew toward what its terms need. **That 0.66
realization factor is specific to what this fix removed and does not transfer** —
applying it to the merge estimate below made that estimate too low, not
conservative.

**Why the assumption was reasonable and became false.** `planck_tensor_cpp.py`:
*"the intermediate builds … are re-emitted per part — cheap, local, and keeps each
part self-contained."* True when chunking was introduced for an **undressed** emit,
where `required_intermediates` is empty and the statement costs nothing. Dressing
populated that list and nothing re-examined the claim. **Self-containment was the
goal; the price went unmeasured until the profile.**

## Fix 2 — 38 builders that were one contraction (1.42x-1.52x)

The post-fix-1 ranking (HF/6-31G, 25 483 leaf samples — the largest tractable
case, so the shares reflect a size where the residual matters most):

| # | hotspot | share | outcome |
|---|---|---|---|
| 1 | **triples residual, `part1`** | **44.8 %** | **hard** — the `o²v²`-deep terms; both emitter models spent |
| 2 | `build_W_t2t2v_oooovv` (38 builders) | 20.4 % | **FIXED** — merged 38 -> 4 |
| 3 | triples residual, `part0` | 13.8 % | hard, same as (1) |
| 4 | `build_W_t1t3v_oooovv` | 8.9 % | **FIXED** — 19 -> 11 |
| 5 | `build_W_t1t1t2v_oooovv` | 5.6 % | **FIXED** — 12 -> 2 |
| 6 | remaining builders (137 of 141) | 4.0 % | tail |
| 7 | singles/doubles residual | 1.0 % | already ~free |
| — | **everything, at once** | **100 %** | **OPEN — no OpenMP anywhere in CC** |

Hotspot 2 was **38 distinct emitted builders each writing a rank-6 result over a
9-deep nest**, and with index names normalized away, **all 38 are the same
contraction** `t2(....) * t2(....) * oovv(....)`, differing only in slot placement.
`merge_transposes` already decided exactly that equivalence symbolically, exact
against a numeric oracle and value-gated 0/2536 at rank 4 — it just had no
production caller. Threading it measured **1.42x (LiH) / 1.52x (CH4)**, and the
attribution confirmed the causal story family by family: `t2t2v_oooovv` 23.3 % ->
4.1 %, `t1t1t2v` 6.6 % -> 1.7 %, and the negative control `t1t3v` (merges only
1.7x) 9.7 % -> 9.4 %. Full record: `CCGEN_MERGE_TRANSPOSES.md`.

**Its deferral is the same error as fix 1's prediction, twice over:** an
operator-count model said 1.02x-1.20x ("compile time, not speed"), a
profile-weighted re-cost said 1.21x-1.36x, measurement said 1.42x-1.52x. Both
models priced operators as equal-cost.

**Why the parts are unequal — it is not chunk size.** All three heavy parts hold
**256 nests each**, but their modelled cost at `o=5 v=6` differs by 18x:

| part | nests | modelled cost | deepest inner sum |
|---|---|---|---|
| part0 | 256 | 2.46e+08 | `o²v¹` |
| **part1** | **256** | **6.68e+08** | **`o²v²`** |
| part2 | 256 | 3.72e+07 | `o²v⁰` |
| part3 | 38 | 5.13e+06 | `o¹v⁰` |

**Chunking splits by term COUNT, not by cost.** Not itself a defect — the same
total work is done — but any future per-part parallelism would be badly
unbalanced, and a cost-weighted split is the cheap fix if that is ever needed.

## What remains: CC has no OpenMP at all

**There is zero OpenMP anywhere in CC** — none in `src/post_hf/cc/*.cpp`, none in
the generated kernels, and the emitter never emits a pragma. Confirmed at runtime:
a CH4 solve with `OMP_NUM_THREADS=8` on an 8-core machine runs at **98.8 % CPU** —
one core busy, seven idle. Every other hot path in Planck (ERI, Fock builds, the
4-index transforms, the DFT J/K builds) is threaded; **CC is the exception.**

Amdahl on the post-fix-1 split (builders 45.1 %, residual 53.7 %, other 1.2 %).
This machine has 4 performance cores / 8 logical, so `n=4` is the realistic
ceiling:

| threads | both parallel | residual only | builders only |
|---|---|---|---|
| 2 | 1.98x | 1.37x | 1.29x |
| **4** | **3.86x** | 1.67x | 1.51x |
| 8 | 7.38x | 1.89x | 1.65x |

**3.86x at 4 threads is larger than every lever found so far combined** (dressing
3.6x, chunk hoist 1.76x, merge 1.5x), and it is the only remaining item addressing
both halves of the split at once. Note the split has since shifted — the merge cut
builder work further — so re-measure the shares before relying on these exact
figures.

**Both sites are reduction-free, and the builders are the better shape:**

- **Builders — embarrassingly parallel.** Independent calls, each writing its own
  freshly-allocated tensor. **No write sharing at all**, no reduction, no ordering.
- **Residual nests — parallel but coarser.** Each nest's outer `i` writes disjoint
  `result(i,...)` slices, so no reduction either. But the trip count is `no` = 4-8,
  giving 1-2 iterations per thread at `n=4`. **Collapse `i,j`** (`o²` = 16-64
  trips); writes stay disjoint.

**Why this is lower-risk than the DFT precedent.** The historical DFT jitter came
from a **cross-thread reduction summed in completion order**
(`dft_xc_reduction_determinism`). Neither site here has a reduction — the same
property that made the DFT J/K builds bitwise-invariant across thread counts.
**Verify it the same way — energies bitwise identical across `OMP_NUM_THREADS` =
1/2/4/8 — rather than assuming it from the argument.**

**Start with the builders:** better granularity, no write sharing, and one
`#pragma omp parallel for` over the emitted build list in the main kernel, where
fix 1 has already collected them into one place. Before that fix they were
scattered across four `_partN` functions. **Unmeasured:** whether thread overhead
is amortized at these sizes — CH4's `o³v³` is 8000 elements and a per-builder task
is small. The honest first step is one `parallel for` over the builder list,
measured.

## What this says about method

| lever | how found | outcome |
|---|---|---|
| contraction order | census + FLOP model | **hit** (3.6x) |
| loop fusion | census + traffic model | **miss** (~0 %) |
| chunk rebuilds | **`sample` profile** | **hit** (1.76x) |
| duplicate transposes | **`sample` profile** | **hit** (1.5x) |
| no OpenMP | **asking whether it was threaded** | modelled 3.86x |

The two models cost days and went 1-for-2. The profile cost ~20 minutes and found
a defect neither model could see, because **neither modelled work that should not
happen at all** — both priced the residual's arithmetic while two thirds of the
time was redundant operator construction outside it.

**Every model here underestimated, none overestimated**, and always by treating
units of work as interchangeable — operators as equal-cost, eliminated builds as
average-cost. When a model prices *count*, ask what the individual items cost
before trusting the ratio.

**A misreading worth naming.** `CCGEN_KERNEL_SCALING_SCOPE.md`'s "measurement route
is closed" forbids comparing generated and hand-written *residuals*, and makes
their wall-clock non-comparable. It never prevented profiling the generated path
**against itself** — which is what found both fixes. It had been treated as a
reason the whole question needed modelling.

## Traps for anything that continues this

- **No target number from the generated/hand-written ratio.** It is a ratio across
  a solver boundary.
- **Correctness is bitwise**: `ch4_rccsdt_generated_sto3g` and
  `lih_rccsdt_generated_sto3g` with identical `E_corr`. A change that reassociates
  floating-point accumulation is not evaluation-order-preserving and needs its own
  justification rather than absorption into a tolerance.
- **"There is no lever here" is a legitimate outcome** and cheaper to establish
  than the two refuted models were.
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
- **The codegen target re-fires on every `make`.** A runtime patch to the emitter
  does not survive a build — it regenerates over your tree mid-measurement. Make
  the toggle a build option, and check a structural count (symbols, TU bytes), not
  only timings, or a raced A/B reports a clean null result.

## Key code locations

| what | where |
|---|---|
| phase brackets | `run_generated_arbitrary_order_iterations`, `generated_arbitrary_runtime.cpp` |
| per-rank timing | `PLANCK_CC_RANK_TIME`, `evaluate_generated_arbitrary_order_residuals`, same file |
| the chunk hoist | `_emit_chunked_kernel`, `python/ccgen/emit/planck_tensor_cpp.py` |
| the chunking assumption it corrected | same function's header comment |
| the transpose merge | `docs/CCGEN_MERGE_TRANSPOSES.md` |
| why the correct path is this harness | `docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md` |
| the emitter-side levers, one hit one miss | `docs/CCGEN_WHY_GENERATED_IS_SLOW.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
