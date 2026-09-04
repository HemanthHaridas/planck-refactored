# ccgen Generated CC Kernel Performance Causes

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**What makes the generated CC kernels slower than the hand-written ones?**

## Short answer

Four causes were examined. Two were confirmed and fixed (3.6x and 1.76x, compounding to ~6.4x), one was built and refuted, one is scoped and untouched.

| # | cause | status | worth |
|---|---|---|---|
| 1 | **contraction order** — 391 of 824 terms evaluated n-arily at `o⁵v⁵` | **FIXED**, `--dressing derived` | modelled 10x–18x FLOPs; **measured 3.6x** |
| 2 | **one loop nest per term** — 806 nests vs the hand-written kernel's one | **BUILT** (`CCGEN_FUSE_LOOPS`), 806 → 15 | **~0 % runtime**; real compile-time/code-size win |
| 3 | **operators rebuilt once per chunk** — 1080 builder calls for 270 operators | **FIXED** (H5) | **1.76x**; **20.8x** fewer calls at rank 4 |
| 4 | **no OpenMP anywhere in CC** — 98.8 % CPU on 8 cores | **SCOPED** (H6) | modelled **3.86x** at 4 threads |

Cause 3 is the one the census-and-FLOP-model method missed. It was found by a `sample` profile in ~20 minutes, after two rounds of code-census-plus-FLOP-model had produced one hit and one miss. Cause 4 was found by asking whether the code was threaded at all. Neither needed a model.

**The "generated vs hand-written" ratio in this document is NOT a like-for-like comparison, and an earlier revision wrongly treated it as one.** The two paths are *different solvers*, not two implementations of one algorithm:

| | hand-written (`tensor`) | generated (`optimized`) |
|---|---|---|
| amplitude storage | **wedge-packed** (`i<=j<=k`), rebuilt via `restore` | **dense**, every index stored |
| r1/r2 each iteration | cheap hand-written dressed intermediates | full generated kernels, every rank |
| iterations on CH4 | **40** | **16** |

`CCGEN_RANK3_KERNEL_AND_SOLVER.md` establishes the storage difference is a *coupled convention*, and the same reasoning that forbids comparing their residuals elementwise applies to their wall-clock: a ratio between them prices two different algorithms, not two codegen strategies. Quote it as "the generated production path costs Nx the hand-written one end to end" — a real and useful operational fact — never as "the generated kernel is Nx slower".

Cause 1 and cause 2 are unaffected by this caveat: both were measured **generated-vs-generated**, one flag apart, same solver.

Established by code-level census of the emitted C++, a FLOP model, and end-to-end timing — after the isolated-kernel measurement route was closed (`CCGEN_KERNEL_SCALING_SCOPE.md`).

**Read this before acting on it:** the census-and-FLOP-model method produced one correct prediction (cause 1) and one confidently wrong one (cause 2 — predicted 32x, delivered nothing). Its record here is 1 for 2.

## Where the logic lives

- `build/generated/cc/ccsdt_planck_generated.cpp` — undressed kernel, 824 nests
- `src/post_hf/cc/tensor_backend.cpp` (`build_dressed_triples_residual`) — hand-written kernel, one nest
- `python/ccgen/emit/planck_tensor_cpp.py` (`_emit_terms`, `emit_planck_fused_group`, `_emit_chunked_kernel`) — fusion and the chunked emit path
- `docs/CCGEN_ARBITRARY_HARNESS_COST.md` — causes 3 and 4 (chunk rebuilds, OpenMP) in full
- `docs/CCGEN_KERNEL_SCALING_SCOPE.md` — the isolated-kernel ladder
- `docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md` — establishes the amplitude-storage convention difference
- `docs/CCGEN_HIGHER_OPERATOR_REUSE.md` — predicted the `t2·t3·v` reuse case that cause 1 confirms

## What invariants matter

### 1. The generated-vs-hand-written ratio is not a codegen quality measure

The two solvers differ in amplitude storage (wedge-packed vs dense), per-iteration intermediate cost, and iteration count (40 vs 16 on CH4). A wall-clock ratio between them prices the whole solver design, not the code generator's output quality.

Design rule:

- Only compare generated-vs-generated (one flag or setting apart, same solver) when trying to attribute a runtime effect to a specific codegen change. Never quote a generated-vs-hand-written ratio as "the generated kernel is Nx slower" — quote it as an end-to-end operational cost.

### 2. A FLOP or traffic model predicts direction, not magnitude, on these kernels

The contraction-order model predicted 10x–18x and measured 3.62x (right sign, factor of ~3 off) — direct evidence the generated path is not purely FLOP-bound. The loop-fusion traffic model predicted a real win from 32x fewer `o³v³` traversals and measured ~0 %, because it priced memory traffic that was already being served from L2 cache; `t3` never leaves L2 at any reachable ladder size.

Design rule:

- Trust a code-census/FLOP model for scaling exponents (it reproduced the measured `o` exponent to 0.05), never for the magnitude of a speedup or slowdown ratio.
- State an explicit falsification criterion before trusting a traffic or FLOP model, and re-test it at a size that can actually falsify it (BH3 is L1-resident and cannot test a cache-traffic claim; HF/6-31G at 3.4x the working set can).

### 3. Profile before modelling — a profile finds work a model does not know to price

Two rounds of census-plus-FLOP-modelling found one hit (cause 1) and one miss (cause 2), costing days. A `sample` profile taken in twenty minutes found cause 3 (67.7% of runtime in redundant `build_W_*` builder reconstruction) because it measures what actually happens rather than pricing an assumed set of operations. Neither model had "operators rebuilt once per chunk" or "zero OpenMP" as a term to price at all.

Design rule:

- Profile the actual generated-vs-generated binary before building a cost model. A model can only be wrong about things it decided to model; a profile cannot miss what is actually running.

### 4. Flags carried over from a previous build silently redefine what is being measured

The first fusion build was configured without `PLANCK_CC_DRESS_OPERATORS=ON`, fusing the undressed manifold — a repeat of the same class of mistake documented in the `SPIN_ADAPT=OFF` investigation, where flags were carried forward from a previous command and the one defining the manifold under test was dropped.

Design rule:

- Run `grep '^PLANCK_CC' <build>/CMakeCache.txt` before trusting any number from a new build tree.

## What was found

1. **Cause 1 (contraction order) — fixed.** 391 of 824 rank-3 triples-residual nests carried a four-index inner sum (`o²v²`), each costing `o⁵v⁵`, in four families: `t1·t2·t2·oovv` (172), `t2·t3·oovv` (151), `t1·t1·t3·oovv` (44), `t1·t1·t1·t2·oovv` (24) — these are 83–90% of generated FLOPs. `--dressing derived` eliminates all 391 (824 nests → 414, zero four-deep), moving the exponents from `o^4.92 v^4.94` to `o^4.42 v^4.40` against a hand-written `o^3.94 v^4.18`. Measured 3.62x on CH4 against a modelled 11.2x.
2. **Cause 2 (one loop nest per term) — built via `CCGEN_FUSE_LOOPS=N`, refuted as a runtime lever.** 806 nests share only 15 distinct `(free, summed)` loop signatures. Fusion reduces 806 → 15 nests (54x), halving the TU and dropping 845 KB of binary, but changes runtime by 0–3% (inside noise, non-monotonic) both before and after cause 3's fix — tested at three sizes spanning 7x in `t3` (BH3 0.031 MiB, CH4 0.061 MiB, HF/6-31G 0.21 MiB). The traffic-savings model (32x fewer `o³v³` passes) is refuted by its own falsification criterion: HF/6-31G is 3.4x BH3's working set and shows +0.35%, not the predicted "material" saving.
3. **Cause 3 (operators rebuilt once per chunk) — fixed (H5).** `_emit_chunked_kernel` emitted every dressed operator inside every `_partN` function, so the duplication factor equalled the part count: 1112 builder calls for 278 distinct operators on the rank-3 triples kernel (4 parts), 16 092 calls for 894 operators on the rank-4 quadruples kernel (18 parts) — the waste scaled with kernel size, worst at the production target. Fixed by building each operator once into a `<kernel>_ops` struct passed by `const&`. Measured on CH4: 29.59 s → 16.81 s (1.76x), `E_corr` bitwise identical, rank-4 TU 12.8 → 10.5 MB.
4. **Cause 4 (no OpenMP) — scoped as H6, not yet built.** Zero `#pragma omp` anywhere in `src/post_hf/cc/*.cpp`, the generated kernels, or the emitter. A CH4 solve with `OMP_NUM_THREADS=8` runs at 98.8% CPU (one core, seven idle) while every other hot path in Planck is threaded. Amdahl on the post-H5 split (builders 45.1%, residual 53.7%) gives a modelled 3.86x at 4 threads; both sites are reduction-free (builders write private tensors, residual nests write disjoint `result(i,...)` slices).

## Measured cost of each path

`PLANCK_RCCSDT_BACKEND=optimized` (generated, arbitrary-order harness) against `tensor` (hand-written), same input, same binary configuration apart from dressing. Energies identical to all ten digits across all arms. These are two solvers' end-to-end costs, not a codegen ratio — the undressed/dressed columns are like-for-like (one flag apart, same solver); the dressed/hand column is an operational fact about which production path is cheaper, not a measure of emitted-code quality.

| case | undressed | dressed | hand-written | dressed/hand |
|---|---|---|---|---|
| BH3/STO-3G | 33.70 s | 9.34 s | **0.10 s** | **93x** |
| CH4/STO-3G | 103.86 s | 28.67 s | **0.19 s** | **151x** |

This is not the 21.8x–50.1x from `CCGEN_KERNEL_SCALING_SCOPE.md` — that ladder timed the isolated triples residual. End to end the gap is 337x–547x, matching `CCGEN_ARBITRARY_HARNESS_COST.md`'s independently recorded ~500x. These are different quantities and should not be quoted interchangeably.

## What was ruled out

- **Loop-invariant work.** Zero of 824 nests have an accumulation ignoring any of `i,j,k,a,b,c`, so no nest can be hoisted — the emitted FLOPs are real work.
- **Accessor overhead.** Fixed and gated separately (`CCGEN_TENSOR_ACCESSOR.md`, 206x on rank 3).
- **Memory traffic from nest count.** Measured at three sizes spanning 7x in `t3`; fusion buys ~0% at every size, both before and after cause 3's fix.

## What was built

| item | description |
|---|---|
| `CCGEN_FUSE_LOOPS=N` | fuse the N largest loop-signature groups; 0 (default) is byte-identical to the pre-fusion emit |
| `term_loop_signature`, `group_terms_by_loop_signature` | the `(free, summed)` grouping key |
| `emit_planck_fused_group` | one nest header, N accumulations into a shared `acc` |
| `test_loop_fusion_grouping.py` | pins the grouping against the emitted text, not against the helper's own logic |

Gates: `ch4_rccsdt_generated_sto3g` and `lih_rccsdt_generated_sto3g` pass at every fusion level with bit-identical `E_corr` (`-0.0533629208` on BH3, `-0.0791116825` on CH4, matching PySCF to 1.4e-08); the default emit is byte-identical on both the dressed and undressed paths.

Three traps found while building fusion:

1. **The triples kernel does not use the obvious emit path.** It exceeds `_KERNEL_CHUNK_TERMS` and goes through `_emit_chunked_kernel`, which had its own term loop. The first wiring changed three small kernels and left triples untouched — fusion "applied" and bought nothing.
2. **Chunks are contiguous slices**, so groups must be reordered before chunking or a group straddles two `_partN` functions and silently un-fuses.
3. **Routing the chunked path through the shared helper broke byte-identity** (it never emitted `// Term N` or trailing blanks). Caught by re-running the grouping gate on the next change.

## Validation strategy that should remain in place

- `ch4_rccsdt_generated_sto3g` and `lih_rccsdt_generated_sto3g` regression cases, at every `CCGEN_FUSE_LOOPS` level
- `test_loop_fusion_grouping.py`, gated against the emitted text
- `grep '^PLANCK_CC' <build>/CMakeCache.txt` before trusting a number from a new build tree
- Re-profile (not re-model) after any change to operand residency or chunking, since H5's builder hoist measurably changed the residual's share of runtime and could change whether other levers (like fusion) matter

## Related but separate outcome: what profiling method this investigation demonstrates

| lever | how found | outcome |
|---|---|---|
| 1 contraction order | census + FLOP model | **hit** (3.6x) |
| 2 loop fusion | census + traffic model | **miss** (~0%) |
| 3 chunk rebuilds | `sample` profile | **hit** (1.76x) |
| 4 no OpenMP | asking whether it was threaded | **modelled 3.86x** |

The two models cost days and went 1-for-2. The profile cost twenty minutes and found a defect neither model could see, because neither modelled work that should not happen at all — they both priced the arithmetic of the residual, while two-thirds of the time was redundant operator construction outside it. This is the transferable result: profile before modelling.

## Remaining architecture concern

Cause 4 (no OpenMP in CC, modelled 3.86x at 4 threads) is scoped as H6 and not yet built. The isolated-kernel measurement route closed by `CCGEN_KERNEL_SCALING_SCOPE.md` remains narrow in scope: it forbids comparing the generated and hand-written *residuals* elementwise and makes their wall-clock ratio non-comparable, but it does not prevent profiling the generated path against itself — which is how causes 3 and 4 were found. An earlier revision of this document mistakenly treated that constraint as a reason the whole performance question had to be answered by modelling alone.
