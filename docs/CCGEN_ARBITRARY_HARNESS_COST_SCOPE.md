# What does the correct rank-3 CC path cost, and can it be made cheap?

**Scope. Not started.** Opened by the option-2 decision in
`CCGEN_RANK3_KERNEL_AND_SOLVER.md`: the generated rank-3 kernels now run in the
arbitrary-order harness, which is **correct** (CH4/STO-3G +1.49e-08 vs PySCF `rccsdt`) but
**~500× slower** than the hand-written tensor backend. This doc is about closing that gap.

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

### H0 — profile before touching anything (~S, blocking)

Profile one converged CH4 rank-3 solve on the arbitrary harness (`correlation cc3`, Release,
`-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON`) and attribute wall time to: the rank-1 kernel, the rank-2
kernel, the rank-3 kernel, the Jacobi/DIIS update, and everything else.

*Verify:* a table of measured shares that sums to the wall time. `sample(1)` on the running process
is sufficient and was already used in this investigation to identify call sites correctly.

**This is blocking.** H2 predicts rank-1+rank-2 are a large share; H1 predicts rank-3 dominates and
the harness is incidental. They imply completely different work, and the profile separates them in
one run.

### H1-path — if the rank-3 kernel dominates (~L, and already scoped elsewhere)

Then this is not an arbitrary-harness problem at all: it is the generated-kernel scaling defect, and
`CCGEN_KERNEL_SCALING_SCOPE.md` owns it. The lead item there is consuming
`_optimal_contraction_order` in the emitter (`python/ccgen/tensor_ir.py:283`, computed and
discarded; `grep BLASHint python/ccgen/emit/planck_tensor_cpp.py` returns nothing).

Do **not** duplicate that scope here. Record the profile share and hand off.

### H2-path — if the lower-rank kernels are a large share (~M)

`tensor_backend` demonstrates that r1/r2 do not need a generated kernel per iteration to be correct
— but it demonstrates it in a *scheme* that is incompatible with the generated representation, which
is precisely the defect option 2 stepped away from. So the options are narrower than they look:

- **Cache what is rank-independent** across iterations inside the harness. Requires knowing what the
  generated kernels re-derive per call — an emitter question, not a solver one.
- **Emit intermediates for the spin-adapted path** (H3), which is the same lever seen from the other
  side.

Explicitly **rejected**: reintroducing hand-written r1/r2 alongside generated r3. That is the hybrid
that produced a self-consistent wrong answer (−7.56e-05), and it is rejected on correctness, not
cost.

### H3-path — intermediates for spin-adapted emission (~L, research)

`ccgen_spin_adapt_no_intermediates` records why they are off: CSE mislabels occ/vir on spatial
spin-adapted terms (unvalidated), and ~1544 `build_W_*` functions made the registry TU compile in
~28 min at `-O3`. Both are real, and the second interacts with the `-O1` registry pin
(`CMakeLists.txt:402`).

So this is not "turn the flag on". It needs the CSE labelling validated for spatial terms first, and
a compile-time story. Sequence it behind H0 and only if the profile says intermediates would matter.

### H4-path — dense vs wedge DIIS (~S, tail)

Only worth doing once the residual cost is addressed; ~6× on a component that is not the bottleneck
changes nothing.

## Acceptance

There is no target number yet, and inventing one before H0 would be arbitrary. What the work must
produce:

- **A profile** (H0) with measured shares.
- **A decision** recorded against that profile: which path is taken, and which are dropped.
- **Correctness preserved bitwise**: `ch4_rccsdt_sto3g` in both modes must still pass. Any change
  that reassociates floating-point accumulation is not evaluation-order-preserving and needs its own
  justification rather than absorption into a tolerance.

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
