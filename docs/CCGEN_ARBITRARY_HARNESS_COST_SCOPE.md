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

#### H5.1 — emit the builds once, pass them in (~M)

`_emit_chunked_kernel` currently emits `required_intermediates` inside every part
(`planck_tensor_cpp.py:1266-1270`). Emit them once in the main kernel instead and
pass them to the parts, exactly as `result` already travels by reference.

The parts' signature grows by one parameter — a struct of `const` operator
references, or the operators individually. Prefer a struct: 270 parameters is not
a signature, and a struct keeps the part signature stable as the operator set
changes.

*Verify — mechanical, no run required:*
- **builder calls in the emitted TU: 1080 → 270.** `grep -c "= build_W_"` on the
  triples kernel.
- every part still compiles (the operators it reads are now parameters).

#### H5.2 — correctness (~S)

*Verify:* `ch4_rccsdt_generated_sto3g` and `lih_rccsdt_generated_sto3g` pass with
**bit-identical** `E_corr` against the pre-H5 numbers (`-0.0791116825`,
and BH3 `-0.0533629208`).

**This must be exactly bitwise, unlike fusion.** Hoisting reassociates nothing —
each operator is built by the same code from the same `(reference, mo_blocks,
denominators, amplitudes)`. **If the energies move, stop:** it means an operator
was being built from state that differs between parts, which is a correctness
question about the current emit, not a tolerance question about the new one.

#### H5.3 — measure (~S)

*Verify:* re-run H0a/H0b (`t_res`) and the H0c `sample`.

Predicted: builders fall from 67.7 % toward ~17 % (a quarter of the calls),
i.e. total iteration time down **~50 %**. Recorded as a prediction so it can be
wrong — if the builders do not fall proportionally, the per-call cost is not
uniform and the profile will say which family still dominates.

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
