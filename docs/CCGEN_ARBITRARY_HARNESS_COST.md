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
| OpenMP anywhere in CC | **was none** — flat 1->4 threads on an OpenMP build, while direct-SCF HF in the same binary went 3.3x | **DONE** — 3.22x at 4 threads (`CCGEN_CC_OPENMP`) |

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
| — | **everything, at once** | **100 %** | **DONE — 3.22x, `CCGEN_CC_OPENMP`** |

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

## The last lever: CC had no OpenMP at all (now landed)

**There is zero OpenMP in CC** — 0 `pragma omp` in `src/post_hf/cc/*.{cpp,h}` and
0 in the emitted kernels, against 8+ other files under `src/` that carry them.
Every other hot path in Planck (ERI, Fock builds, the 4-index transforms, the DFT
J/K builds) is threaded; **CC is the exception.**

**Rescoped 2026-08-29 with fresh measurements — the estimate and the
recommendation below both moved.** See `docs/CCGEN_CC_OPENMP.md`. Two
things this section previously got wrong:

- **It advised starting with the builders.** That was right at 45.1 %; after the
  transpose merge they are **13.8 %** and cap at **1.12x**. The residual is now
  **86.2 %**, concentrated in `triples_residual_part1` at **63.6 %**. Threading
  the triples parts modelled 2.74x and **measured 3.22x** once the emitter also
  covered singles/doubles. The builders were never threaded at all: in the chunked
  kernels they turned out to be **building every operator twice**, and deleting the
  dead set beat threading it (2.6 % to every build, with no OpenMP).
- **Its "98.8 % CPU on 8 cores" evidence did not distinguish its own claim.**
  That run used a tree where `OpenMP_CXX_FLAGS` is `NOTFOUND` and `-DUSE_OPENMP`
  never reaches the compile line, so *every* Planck pragma was inert in it, not
  just CC's absent ones. **Now measured properly** on `build-full`, which is
  genuinely threaded: CC is flat from 1 to 4 threads (**78.03 s -> 78.49 s**,
  99.1 % CPU at 8) while direct-SCF HF/cc-pVTZ in the **same binary** goes
  **2.70 s -> 0.81 s (3.3x)**. The positive control is what turns the claim from
  an inference into a measurement — and it has to be a case genuinely bound by a
  threaded path, measured warm.

The determinism argument stands unchanged and is the part worth keeping: neither
site has a **cross-thread reduction** (builders write private tensors; residual
nests accumulate into a thread-private `acc` and write disjoint `result(...)`
slices), which is the property that made the DFT J/K builds bitwise
thread-count-invariant, unlike the DFT grid reduction that summed in completion
order. **That is a reason to expect determinism, not evidence of it — verify
bitwise across `OMP_NUM_THREADS` = 1/2/4/8.**

## What this says about method

| lever | how found | outcome |
|---|---|---|
| contraction order | census + FLOP model | **hit** (3.6x) |
| loop fusion | census + traffic model | **miss** (~0 %) |
| chunk rebuilds | **`sample` profile** | **hit** (1.76x) |
| duplicate transposes | **`sample` profile** | **hit** (1.5x) |
| no OpenMP | **asking whether it was threaded** | **3.22x measured** |

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
