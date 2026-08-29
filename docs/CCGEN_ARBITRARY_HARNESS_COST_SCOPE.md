# What does the correct rank-3 CC path cost, and can it be made cheap?

**H0 is DONE (2026-08-29) and it retired most of this document.** The cost is
**98.8 % one call to the generated rank-3 kernel**; the harness itself is ~1 %.
H2, H3 and H4 are dead. What remains is H1, and the two obvious models of it are
already spent — see "Where that leaves H1". Opened by the option-2 decision in
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

### Where that leaves H1 — awkwardly, and honestly

`docs/CCGEN_WHY_GENERATED_IS_SLOW.md` has already measured the two obvious levers
**on that exact kernel**:

- **not FLOP-bound** — a modelled 11.2x FLOP reduction (`--dressing derived`)
  realised as 3.62x;
- **not traffic-bound** — fusing 806 loop nests to 15 changed runtime by ~0 %.

So H0 has localized the cost precisely (one kernel, 98.8 %) while both models of
*why that kernel is expensive* are spent. **The next step is H0c — sample inside
the rank-3 kernel** — and it is now well-posed in a way it was not before: a
30-second `sample` on a CH4 solve spends essentially all its samples in the
kernel, so the profile is not diluted by harness noise.

*Verify:* a symbol/line attribution within `compute_ccsdt_triples_residual`. What
to look for, given what is already excluded: instruction-level stalls that neither
a FLOP count nor a traffic count models — dependency chains on the accumulator,
failure to vectorize the inner contraction, or register pressure from the
generated form's operand count.

**Do not build another cost model first.** The record on this kernel is 1-for-2,
and the two failures were both models applied where a measurement was available.

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

### H1-path — if the rank-3 kernel dominates (~L, and it is NO LONGER a handoff)

This section previously said to record the share and hand off to
`CCGEN_KERNEL_SCALING_SCOPE.md`. **That is no longer available**, and the change
matters:

- that document's measurement route is **closed** (its probe sits on a rerouted
  code path; the two arms have no residual-level agreement gate), and
- its lead item — consuming `_optimal_contraction_order` — is **largely redundant**,
  because `--dressing derived` already eliminates the 391 four-deep terms it
  targets.

So if H0b says rank 3 dominates, the honest reading is that **the two obvious
emitter levers are already spent**: the path is not FLOP-bound (a modelled 11.2x
realised as 3.62x) and not traffic-bound (806 nests fused to 15 for ~0 %). A
rank-3-dominant profile then means the kernel is expensive for a reason *neither*
model captured, and the next step is H0c — sampling **inside** the residual
evaluation — not another cost model. `docs/CCGEN_WHY_GENERATED_IS_SLOW.md` records
that the modelling approach went 1-for-2.

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
