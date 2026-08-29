# What does the correct rank-3 CC path cost, and can it be made cheap?

**H0 is DONE (2026-08-29), and it found an actionable defect.** The cost is
**98.8 % one call to the generated rank-3 kernel** (H2/H3/H4 dead, harness ~1 %),
and inside that call **67.7 % is `build_W_*` operator builders being rebuilt once
per `_partN` chunk** — 1418 builder calls against 288 distinct operators. The
emitter assumed those rebuilds were "cheap"; they are two thirds of the runtime.
Fix scoped as **H5** below: hoist them, expect ~2x. Opened by the option-2 decision in
`CCGEN_RANK3_KERNEL_AND_SOLVER.md`: the generated rank-3 kernels now run in the
arbitrary-order harness, which is **correct** (CH4/STO-3G +1.49e-08 vs PySCF
`rccsdt`) but far more expensive than the hand-written tensor backend.

## What changed (2026-08-29), before anything below is acted on

`docs/CCGEN_WHY_GENERATED_IS_SLOW.md` measured the emitter-side levers and closed
two of them. That moves this document from "one of several candidates" to **the
only remaining lead**, and it invalidates parts of the framing below.

**1. The ~500x is NOT a like-for-like ratio, and must not be treated as a defect
size.** The two paths are different solvers: wedge-packed vs dense amplitudes,
cheap dressed intermediates vs a full generated kernel per rank, and **40 vs 16
iterations on CH4**. A ratio between them prices two algorithms. It is an
operational fact about which production path is cheaper — not a measure of how
much is recoverable. **Do not set a target from it.**

**2. H1 can no longer "hand off to `CCGEN_KERNEL_SCALING_SCOPE.md`".** That
document's measurement route is closed and its lead item is now known to be
largely redundant: `--dressing derived` already eliminates the 391 four-deep terms
`_optimal_contraction_order` would target. If H0 says the rank-3 kernel dominates,
**that is a finding to act on here**, not a handoff.

**3. H3 is partly answered, in the direction that matters.** Emitting dressed
operators (`--dressing derived`) is the intermediates lever seen from the other
side, and it is **wired, value-gated, and measured at 3.6x end to end**. So H3 is
no longer "research behind a compile-time problem" — a working version exists.
What remains unmeasured is whether *more* intermediates help beyond that.

**4. The obvious codegen hypotheses are spent.** The generated path is measurably
**not FLOP-bound** (a modelled 11.2x FLOP reduction realised as 3.62x) and **not
traffic-bound** (fusing 806 loop nests to 15 changed runtime by ~0 %). Both were
measured generated-vs-generated. That is the strongest available evidence that the
remaining cost is in **the harness rather than the emitted kernel** — which is what
this document is for.

**Profile generated-vs-generated.** Every measurement here must compare
configurations of the same solver, never against `tensor`. Two investigations have
now been damaged by comparing across the solver boundary.

## The measurement, and what it does and does not say

All on CH4/STO-3G (`no=5 nv=4`), one Release build, same input:

| path | wall | iterations | per-iteration | correct? |
|---|---|---|---|---|
| hand-written tensor backend | **0.19 s** | 24 | ~0.008 s | yes (+1.45e-08) |
| generated, arbitrary harness | **~100 s** | 14 | **~7.0 s** | yes (+1.49e-08) |
| generated, old `tensor_backend` hybrid | 31.3 s | 23 | ~1.4 s | **no** (−7.56e-05) |

Two things follow immediately:

- **The gap is per-iteration, not convergence.** The arbitrary harness converges in *fewer*
  iterations (14 vs 24); it is ~875× more expensive per iteration and claws back ~1.7×.
- **The 31.3 s hybrid is not a valid baseline for "generated is 3× slower".** An earlier revision
  of the fix scope made exactly that comparison and reported ~3.1×; it was comparing two
  *generated* configurations and mislabelling one as the hand-written path. The honest number
  against the correct hand-written baseline is **~500×**.

Do not carry the older "~180×" figure that circulated in the retired rank-3 defect notes either — it
predates the tensor-accessor fix and never recorded its dimensions.

## Where the cost is — hypotheses, none yet measured

**This section is explicitly unmeasured.** The rank-3 investigation that preceded this doc falsified
five hypotheses formed by inspection; the rule that emerged is that a profile decides, not a reading.
Nothing below should be acted on before H0.

- **H1 — the generated kernel itself.** `ccsdt_planck_generated.cpp` carries **1015** separate
  `acc` accumulation nests for the triples residual, against 186 accessor call sites in the
  hand-written kernel. `CCGEN_KERNEL_SCALING_SCOPE.md` already measured the generated-vs-hand
  residual ratio at 21.8×→50.1× and growing with size, and attributed it to contraction order
  (the emitter computes `_optimal_contraction_order` and discards it). If the harness is merely
  paying that, the fix is the emitter work already scoped there, not anything in this doc.
- **H2 — the harness evaluates every rank, every iteration.**
  `evaluate_generated_arbitrary_order_residuals` loops `rank = 1..max` and calls a full generated
  kernel for each. `tensor_backend` instead builds r1/r2 from cheap hand-written dressed
  intermediates and only r3 from a kernel. At rank 3 that is three generated kernels per iteration
  against one.
- **H3 — no intermediates.** The arbitrary path runs with `PLANCK_CC_INCLUDE_INTERMEDIATES` off for
  spin-adapted emission (`ccgen_spin_adapt_no_intermediates`), so common subexpressions that the
  hand-written backend materializes once (`w_vvvo`, `w_vooo`, the dressed SD intermediates) are
  recomputed inside every nest.
- **H4 — dense packing.** The harness packs full dense tensors for DIIS where `tensor_backend` packs
  the unique wedge (`i<=j<=k`), i.e. ~6× more data through the DIIS solve. Cheap relative to the
  residual, so this is a tail item, not a lead.

H1 and H3 are not independent: absent intermediates *are* one reason the emitted nest count is high.

## The work

### H0 — **DONE (2026-08-29). H2, H3 and H4 are all dead. It is H1.**

Two bracket sets, ~20 lines, no sampling needed. CH4/STO-3G, Release,
`PLANCK_RCCSDT_BACKEND=optimized`.

**H0a — phase attribution.** Consistent across all 15 iterations:

```
t=1.913s   t_res=1.913s   t_upd=0.000s   t_ene=0.000s
```

**The residual evaluation is ~100 % of the iteration.** The Jacobi/DIIS update and
the energy evaluation are unmeasurable.

**H0b — per-rank attribution** (`PLANCK_CC_RANK_TIME=1`):

| rank | elements | time | share |
|---|---|---|---|
| 1 | 20 | 0.0008 s | 0.04 % |
| 2 | 400 | 0.0237 s | 1.1 % |
| **3** | **8000** | **2.0775 s** | **98.8 %** |

### What this kills

| hypothesis | verdict |
|---|---|
| **H2** — harness evaluates every rank each iteration | **DEAD.** Ranks 1+2 are ~1.2 % combined. Caching or skipping them buys at most 1 %. |
| **H4** — dense vs wedge DIIS packing | **DEAD.** `t_upd` is 0.000 s. The ~6x data through DIIS costs nothing measurable. |
| **H3** — intermediates for spin-adapted emission | **Not worth pursuing as stated.** Its value was in reducing recomputation across the residual; that recomputation is entirely inside the rank-3 kernel, so H3 collapses into H1. |
| **H1** — the rank-3 kernel itself | **CONFIRMED, and it is the whole cost.** |

**This retires the framing this document was opened with.** It is not an
arbitrary-harness cost problem: the harness is ~1 %. The 2.08 s is one call to one
generated kernel, and everything else in the iteration is free.

### H0c — **DONE (2026-08-29). 67.7 % is redundant operator rebuilds, not the residual.**

`sample` on a running CH4 solve, 20 s, leaf attribution over 16 882 samples:

| | samples | share |
|---|---|---|
| `build_W_*` operator builders (165 distinct in the profile) | **11 422** | **67.7 %** |
| `compute_ccsdt_triples_residual_part*` | 5 334 | 31.6 % |
| everything else | 126 | 0.7 % |

Dominant builder families: `t2t2v_oooovv` **33.9 %**, `t1t3v_oooovv` 14.7 %,
`t1t1t2v_oooovv` 9.5 %.

**The cause is chunking.** The dressed kernel is split into `_partN` functions,
and **each part rebuilds the entire operator set from scratch**. On the triples
kernel specifically:

```
1080 builder calls   (4 parts x an identical 270)
 270 distinct operators actually used
 810 calls are pure duplication          = 75% waste
```

(TU-wide the count is 1418 calls across 288 distinct builders; the 1080/270 above
is the triples kernel alone, which H0b showed is 98.8 % of the time.)

`planck_tensor_cpp.py:1193` states the assumption directly — *"the intermediate
builds and amplitude-view bindings are re-emitted per part — **cheap**, local, and
keeps each part self-contained."* **They are 67.7 % of runtime.** The assumption
was reasonable when chunking was introduced for an *undressed* emit, where there
are no operators to rebuild; dressing made it false, and nothing re-examined it.

### H5 — hoist the builders out of the parts

**The defect, measured exactly on the triples kernel:**

```
1080 builder calls   (4 parts x 270)
 270 distinct operators actually used
 810 calls are pure duplication          = 75% waste
```

Every part emits the **identical** 270 builds — verified: the four parts' builder
sets are set-equal. Per part only 59/81/118/21 are read, but the union over parts
is exactly 270, so **there are no built-but-unused operators at kernel level**; the
waste is entirely the 4x duplication, not over-building. (An earlier reading of
"270 built, 59 used in part0" as 78 % over-building was wrong — the other 211 are
used by sibling parts.)

#### H5.1 / H5.2 / H5.3 — **DONE (2026-08-29). 1.76x, energies bitwise identical.**

`_emit_chunked_kernel` now emits each operator **once** in the main kernel into a
generated `<kernel>_ops` struct, and passes it to the parts by `const&`. Each part
binds `const auto &W_x = ops.W_x;` so every term body is unchanged.

Struct members are typed with `_tensor_type(spec.rank)` — the same expression the
builder's own definition uses for its return type, so the two cannot drift.
(`decltype(builder(std::declval<...>()))` also works but drags `<utility>` into
every generated TU for nothing.)

**H5.1 — mechanical gate, no run required:**

| | TU builder calls | inside triples parts |
|---|---|---|
| before | 1418 | **1080** |
| after | **338** | **0** |

**H5.2 — correctness:** both generated gates pass with `E_corr` **bit-identical to
all ten digits** (`-0.0533629208`, `-0.0791116825`). As predicted this is exactly
bitwise — hoisting reassociates nothing.

**H5.3 — measured (CH4, best-of-2, same binary configuration apart from H5):**

| | before | after | |
|---|---|---|---|
| wall | 29.59 s | **16.81 s** | **1.76x** |
| builder share | 67.7 % | 45.1 % | |
| builder time | 20.03 s | **7.58 s** | **2.64x** |
| residual time | 9.56 s | 9.23 s | 1.04x |

BH3: 9.52 s → 5.57 s (**1.71x**).

**The residual at 1.04x is the check that the decomposition is sound** — H5 does
not touch the residual, so it should be flat, and it is. That also invalidates a
cross-run normalization attempted first: absolute sample counts between two
`sample` runs of different duration are not comparable (it reported the residual
"falling" 0.60x, which is impossible). **Use wall-clock times and within-run
shares; do not compare absolute sample counts across runs.**

#### What the prediction got wrong, and what that reveals

Predicted: builders 67.7 % → ~17 %, total ~50 % faster. Measured: **45.1 %**, total
**43 % faster**. The wall-clock prediction was close; the share prediction was not.

Builder time fell **2.64x**, not the **4x** the call-count reduction implies. So
the eliminated builds were **cheaper than average** — plausibly the small-rank
operators, since a part's 270 builds span every rank while its 59-118 *used*
operators skew toward the ones its terms actually need.

**Builders are still 45 % of runtime after removing 75 % of the calls.** The 270
remaining builds are irreducible in the sense that each is genuinely needed once —
so further gain here is not about *how many times* operators are built but about
*what a single build costs*. That is a different question, and this document does
not have it scoped.

#### H6 — OpenMP: the largest remaining lever, and it is untouched

**Measured premise:** there is **zero OpenMP anywhere in CC** — none in
`src/post_hf/cc/*.cpp`, none in the generated kernels, and the emitter never emits
a pragma. Confirmed at runtime: a CH4 solve with `OMP_NUM_THREADS=8` on an 8-core
machine runs at **98.8 % CPU** — one core busy, seven idle. Every other hot path in
Planck (ERI, Fock builds, the 4-index transforms, the DFT J/K builds) is threaded;
CC is the exception.

**Amdahl on H5's measured split** (CH4 post-H5: builders 45.1 %, residual 53.7 %,
other 1.2 %). This machine has 4 performance cores / 8 logical, so `n=4` is the
realistic ceiling:

| threads | both parallel | residual only | builders only |
|---|---|---|---|
| 2 | 1.98x | 1.37x | 1.29x |
| **4** | **3.86x** | 1.67x | 1.51x |
| 8 | 7.38x | 1.89x | 1.65x |

**~3.9x at 4 threads is larger than every lever this investigation has found
combined** (dressing 3.6x, H5 1.76x). It is also the only remaining item that
addresses *both* halves of the split at once.

**Both sites are cleanly parallel, and the builders are the better shape:**

- **Builders (45.1 %) — embarrassingly parallel.** 270 independent calls, each
  writing its own freshly-allocated tensor. **No write sharing at all.** A parallel
  loop over the build list needs no reduction and no ordering.
- **Residual nests (53.7 %) — parallel but coarser.** Each nest's outer `i` writes
  disjoint `result(i,...)` slices, so no reduction is needed either. But the trip
  count is `no` = 4-8, giving 1-2 iterations per thread at `n=4` — coarse and
  unbalanced. **Collapse `i,j`** (`o²` = 16-64 trips) for a usable shape; writes
  stay disjoint.

**Why this is lower-risk than the DFT grid precedent.** The historical DFT jitter
came from a **cross-thread reduction summed in completion order**
(`dft_xc_reduction_determinism`). Neither site here has a reduction: builders write
private tensors, residual nests write disjoint slices. That is the same property
that made the DFT J/K builds bitwise-invariant across thread counts, and it should
be **verified the same way — energies bitwise identical across `OMP_NUM_THREADS`
= 1/2/4/8**, not assumed from the argument.

**Sequencing:** do the builders first. Better granularity, no write sharing, 45 %
of the time, and one `#pragma omp parallel for` over the emitted build list in the
main kernel — where H5 has already collected them into one place. **H5 is what
makes this easy**; before it the builds were scattered across four `_partN`
functions.

**Not yet measured:** whether thread overhead is amortized at these sizes. CH4's
`o³v³` is 8000 elements; a per-builder task is small. The honest first step is one
`parallel for` over the builder list, measured — not a threading strategy designed
in advance.

#### H5.4 — check the other kernels and ranks (~S)

The same chunking applies wherever `len(terms) > _KERNEL_CHUNK_TERMS`. Rank 4
(CCSDTQ) is chunked far more aggressively and is the production target.

*Verify:* builder-call counts before/after for every chunked kernel in a rank-4
dressed emit. **Do not assume the rank-3 ratio transfers** — this codebase has
twice shown rank 3 is not a proxy for rank 4.

#### What H5 does not do

- It does not touch the undressed path (no operators to hoist), which stays
  byte-identical.
- It is independent of `CCGEN_FUSE_LOOPS`: fusion changes nest count within a
  part, hoisting changes how many times operators are built. Fusion's measured
  ~0 % stands and is not revisited by this.
- It is a **~2x lever on the generated path**, not a recovery of the ~500x — that
  is a ratio across a solver boundary, not a defect size.

#### Why the chunking assumption was reasonable and became false

`planck_tensor_cpp.py:1193`: *"the intermediate builds ... are re-emitted per part
— cheap, local, and keeps each part self-contained."* True when chunking was
introduced for an **undressed** emit, where `required_intermediates` is empty and
the statement costs nothing. Dressing populated that list and nothing re-examined
the claim. **Self-containment was the goal; the price was never measured until
H0c.**

### The measurement discipline this must follow

**Generated-vs-generated, always.** Compare configurations of the arbitrary
harness against each other — dressing on/off, fusion on/off, intermediates on/off
— never against `PLANCK_RCCSDT_BACKEND=tensor`. The two paths are different
solvers with different amplitude storage and different iteration counts (40 vs 16
on CH4); a ratio across that boundary prices two algorithms rather than one
change. **Two investigations have already been damaged by ignoring this.**

**Same-binary where possible.** H0a/H0b are internal brackets, so both arms are
the same binary and the comparison is exact. Where a rebuild is unavoidable
(intermediates, dressing), diff `grep '^PLANCK_CC' <build>/CMakeCache.txt` between
the trees and confirm exactly one flag differs — a build flag silently deciding
what is measured has cost this project twice.

**Energies bit-identical**, or the arms are not the same calculation.

### H1-path — **SUPERSEDED by H0c.** "The rank-3 kernel dominates" was right, but the cost inside it is redundant operator rebuilds (67.7 %), not the residual arithmetic the emitter-side models targeted. See H0c and H5.

### H2-path / H3-path / H4-path — **REMOVED.** H0 measured all three at ~1 % or less; see above. The reasoning they contained (do not reintroduce the hand-written r1/r2 hybrid — it produced a self-consistent wrong answer at −7.56e-05) is preserved under Constraints.

## Acceptance

There is no target number, and **the ~500x must not be used as one** — it is a
ratio across a solver boundary, not a defect size (see "What changed"). What the
work must produce:

- **A profile** (H0a/H0b) with measured shares that sum to the printed iteration
  time.
- **A decision** recorded against that profile: which path is taken, which are
  dropped. H0 is worth doing *even if every path is then dropped* — "the cost is
  diffuse and there is no lever" is a legitimate and useful outcome, and cheaper
  to establish than the two refuted models were.
- **Correctness preserved bitwise**: `ch4_rccsdt_generated_sto3g` and
  `lih_rccsdt_generated_sto3g` must still pass with identical `E_corr`. Any change
  that reassociates floating-point accumulation is not evaluation-order-preserving
  and needs its own justification rather than absorption into a tolerance.

## Constraints

- **Do not reintroduce the hybrid** in any form. Mixing hand-written and generated residual sources
  is a correctness defect regardless of what it buys.
- **`make -j4`.** The generated TUs are large enough that a full-width build is disruptive.
- **Explicit `CMAKE_BUILD_TYPE`.** The repo's `build/` has it empty, which drops `-DNDEBUG`,
  re-enables the CC tensor bounds asserts, and makes every timing meaningless — this cost a wrong
  diagnosis once already.
- **Non-square, backstop-clearing test system.** CH4/STO-3G (`no=5 nv=4`, `nso=18 ndet=43758`) is the
  only in-tree rank-3 case that reaches the generated kernels at all.

## Key code locations

| what | where |
|---|---|
| harness loop over all ranks (H2) | `generated_arbitrary_runtime.cpp`, `evaluate_generated_arbitrary_order_residuals` |
| Jacobi/DIIS update, dense pack (H4) | `solver_arbitrary.cpp`, `update_amplitudes_with_jacobi_diis` |
| generated rank-3 TU, 1015 accumulation nests (H1) | `build/generated/cc/ccsdt_planck_generated.cpp` |
| unused contraction-order analysis (H1's fix) | `python/ccgen/tensor_ir.py:283` |
| intermediates-off rationale (H3) | `ccgen_spin_adapt_no_intermediates`, `CMakeLists.txt:402` |
| the scaling defect this may reduce to | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |
| why the correct path is the arbitrary harness | `docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md` |
