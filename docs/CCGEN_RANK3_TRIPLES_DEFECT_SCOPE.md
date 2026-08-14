# The generated rank-3 triples kernel disagrees with the hand-written one — scope

Scopes the fix for a **measured, reproducible** numerical defect, plus a second smaller one it
uncovered. In-flight scope: to be rewritten as an architecture answer once the work lands.

---

## Status (updated after T1b)

**T1b landed and was most of D-A.** Rebinding `mo_blocks` to the physicist convention for the
generated rank-3 kernel took the error from **1.82e-4 → ~2.7e-5 Eh**.

**D-A is not closed, and the failure mode changed.** The solve now *oscillates* instead of
converging to a wrong value:

```
iter 33  E_corr=-0.0533339197  dE=+8.465e-07  rms(R3)=2.566e-04  rms(R1[T3])=4.240e-04
iter 38  E_corr=-0.0533366904  dE=-2.510e-06  rms(R3)=7.388e-04  rms(R1[T3])=5.191e-04
```

`dE` alternates sign and `rms(R3)` **grows**. That is more diagnostic than the original symptom:
a residual that is merely scaled or permuted converges to a wrong number, whereas one that is
inconsistent with the amplitude update cycles. **T0 is now the right next step**, and its value
went up — it separates "the residual value is still wrong" from "the values agree but the solver
path diverges".

## The measurement

`bh3_rccsdt_sto3g`, three builds differing only as noted:

| build | kernel | E_corr | iters | wall |
|---|---|---|---|---|
| hand-written | `native` | **−0.0533629199** | 26 | 7.1 s |
| generated, undressed | `ccgen` | **−0.0531812197** | 24 | 1261 s |
| generated, dressed | `ccgen` | **−0.0531789160** | 27 | 1283 s |

**Two independent defects, not one:**

- **D-A (large, 1.82e-4 Eh):** generated-undressed vs hand-written. Dressing is OFF, so this is a
  defect in the generated rank-3 triples kernel itself.
- **D-B (small, 2.3e-6 Eh):** dressed vs undressed generated. ~80× smaller than D-A and possibly
  downstream of it.
- **D-C (performance, ~180×):** 7 s → ~1270 s. Affects both generated builds, so it belongs with
  D-A. Correctness first.

## Why this was never caught

`compute_ccsdt_triples_residual` had **no caller**. `choose_rccsdt_backend` returns only
`DeterminantPrototype` or `TensorProduction`; the one call site is guarded by
`use_generated_triples_kernel`, and both `run_tensor_rccsdt_impl` callers passed `false`. The
kernel was generated, compiled, and linked — never executed.

The CCSDTQ==FCI validation does **not** cover it: that exercises
`compute_ccsdtq_triples_residual`, a different function in a different TU (arbitrary-order runtime).
Same generator, different emitted code.

**Ruled out, measured:** spin-adaptation. A build matched to the reference on `SPIN_ADAPT=ON`,
`MAXORDER=4`, `ENGINE=diagram` reproduced the identical wrong energy, so the three config
mismatches in the first attempt were real but not causal.

## What the A/B isolates

The two branches at `tensor_backend.cpp:2321` share *everything* except the triples residual —
same reference, `mo_blocks`, denominators, amplitudes, same hand-written
`add_dressed_triples_feedback_into_sd_residuals` for the R1/R2 feedback.

So the differing `rms(R1[T3])` (native **1.610e-4** vs ccgen **3.159e-5**, ~5×) is a *consequence*
of a wrong T3, not a separate defect. **One function is at fault.**

One asymmetry stands out: the generated branch calls `restore_restricted_t3_structure`
(`apply_restricted_t3_permutation_symmetry` → `apply_restricted_t3_p3_full` →
`purify_restricted_t3`) and the hand-written branch does not. That is a post-hoc symmetrization of
the generated result, and it is the first thing to check — it could be repairing a genuine
structural difference, or corrupting an already-correct residual.

---

## Steps

### T0 — compare the two residuals at identical amplitudes (~S, do first)

Call both branches on the *same* `amps` and diff the T3 tensors elementwise, before any solve.

**Why first:** it separates "the kernel computes a different function" from "the kernel is fine but
the solve diverges", and it is the difference between debugging one function and debugging an
iterative loop. A full solve is ~21 minutes; this is one evaluation.

*Gate:* a recorded max/rms elementwise difference, and the index pattern where it is largest —
whether it concentrates in a spin block, an occ/vir pattern, or the repeated-index elements
(`i==j`, `a==b`) the restricted-T3 convention pre-scales.

**Sharpened by the post-T1b oscillation.** Two outcomes, and they point in opposite directions:

- **Residuals differ at fixed amplitudes** → a value error remains in the generated kernel; the
  index pattern localizes the term class (→ T2).
- **Residuals AGREE at fixed amplitudes** → the kernel is correct and the divergence is in the
  solver path. Then the suspect is the restricted-T3 *convention*: both branches call
  `restore_restricted_t3_structure`, but the hand-written kernel produces a residual already in
  the repeated-index-prescaled form the ×6 sum expects, while the generated one may not. That
  would be consistent and self-cancelling for one branch and inconsistent for the other — and it
  would explain oscillation rather than a fixed offset, since the amplitude update would then keep
  re-introducing what the symmetrization removes.

Run the comparison **both with and without** `restore_restricted_t3_structure` applied, since the
difference between those two tells you whether the convention is the remaining gap.

### T1 — is `restore_restricted_t3_structure` the cause? — **ANSWERED: no**

Investigated while implementing T1b; both halves of the suspicion are disproven, recorded so
neither is re-run.

**It is not generated-only.** `restore_restricted_t3_structure` is called on the hand-written
branch too (`tensor_backend.cpp:2360, 2650, 2657, 2674`), and that path converges to the reference
value. The asymmetry I claimed in the original scoping does not exist — I had matched on the
generated call site at 2339 and not checked the others.

**Its non-idempotent 6× is not a bug.** `apply_restricted_t3_permutation_symmetry` sums over all 6
simultaneous permutations without dividing, so it returns 6× an already-symmetric tensor (verified
numerically). But that is compensated by the repeated-index pre-scaling at line ~2010 (`1/6` for
`i==j==k`, `1/2` for one repeat), so the convention is self-consistent across both branches.

*Original scoping retained below for the reasoning.*

#### T1 — original scoping

Re-run T0 with the call bypassed on the generated branch.

Three outcomes, each conclusive:
- difference **vanishes** → the symmetrization is corrupting a correct residual; fix is to drop or
  correct it, not to touch ccgen;
- difference **unchanged** → the symmetrization is neutral; the generated algebra is wrong (→ T2);
- difference **grows** → it is partially repairing a genuinely different structure; the generated
  kernel emits a different T3 convention (→ T2, but with the convention as prime suspect).

*Gate:* the outcome recorded explicitly. **Do not skip to T2** — this is one recompile and it
decides whether ccgen is implicated at all.

### T1b — the missing physicist rebind — **LANDED, most of D-A** (`eb1c611`)

Confirmed by measurement: 1.82e-4 → ~2.7e-5 Eh on bh3.

`rebind_physicist` was **exposed** from its anonymous namespace via
`generated_arbitrary_runtime.h` rather than copied — the `oovv`↔`ovov` sources cross, so an
independent re-derivation is easy to get subtly wrong. It rebinds into a **local** cache; the
shared `state.mo_blocks` stays chemists', because `build_spin_orbital_blocks` and the hand-written
branch both read it.

The rebind is hoisted **outside** the iteration loop. A first version used a function-local
`static thread_local` inside `update_restricted_rccsdt_amplitudes_once`, which runs per iteration —
that would have repeated the transform and leaked one molecule's integrals into the next run in the
same process.

*Original scoping retained below.*

#### T1b — original scoping

`grep -c rebind_physicist src/post_hf/cc/tensor_backend.cpp` returns **0**. The definition and its
only call site live in `generated_arbitrary_prepare.cpp`, so:

- the **arbitrary-order** path rebinds `mo_blocks` to physicist `<pq|rs>` before invoking generated
  kernels — which is why the CCSDTQ==FCI validation passes;
- the **plain rank-3** path passes chemists' `(pq|rs)` straight into
  `compute_ccsdt_triples_residual`, which ccgen emits against the physicist convention.

This is the B5 defect (`generated_ccsdtq_energy_wrong`) in the one consumer that never executed.
It also fits the error's *shape*: a convention mismatch permutes which integral each contraction
reads, producing a wrong-but-plausible T3 that still converges — consistent with 1.8e-4 Eh rather
than a divergence.

**If this is the cause, ccgen is not implicated at all** and the fix is on the C++ consumer side:
rebind, exactly as the arbitrary path already does. That would also explain why D-B is ~80× smaller
— both generated builds read the same mis-bound integrals, so dressing's delta rides on top.

*Gate:* rebind `mo_blocks` in the rank-3 consumer and re-run bh3. Expect −0.0533629199 (the
hand-written value) if this is the whole of D-A.

**Do this before T2.** It is a small change to one call site, and it is the difference between a
C++ wiring fix and a generator investigation.

### T0 — **ANSWERED: the generated residual value is wrong**

Measured with the `PLANCK_CC_T3_DIFF=1` probe on bh3, both residuals from identical amplitudes:

```
raw (no restore): max|gen-hand| = 2.045e-02   rms = 1.222e-03   max|hand| = 2.034e-02
after restore   : max|gen-hand| = 1.865e-03   rms = 1.576e-04   max|hand| = 2.779e-02
```

**The raw residuals differ by ~100 % of the hand-written magnitude.** That is not a scale factor, a convention, or a permutation — the generated kernel computes a substantially different function. So the remaining gap is a **value error in the generated kernel** (→ T2), not a C++ consumer issue, and the alternative branch of T0's gate (residuals agree, solver path diverges) is ruled out.

`restore_restricted_t3_structure` **masks** it: the visible difference drops ~11× to 1.87e-3, about 6.7 % of magnitude. That explains the whole earlier picture — the converged energy was only 2.7e-5 off despite a wholly different residual, because the symmetrization projects out most of the error, and the surviving 6.7 % is what makes the solve oscillate instead of settling.

**Corollary worth keeping:** any future gate on this kernel must compare the **raw** residual. A post-`restore` comparison understates the error by an order of magnitude and would have called a badly wrong kernel nearly right.

### T2 — locate the defect in the generated algebra (~M — NOW THE ACTIVE STEP)

Ruled out so far, by measurement:

- **The emit is faithful to the generator.** A fresh `print_cpp_planck("ccsdt", spin_adapt=True, dress_operators=True)` reproduces the built TU's amplitude-read counts exactly (t1 465, t2 935, t3 217), so the defect is upstream of emit — in the equations or in lowering, not in codegen.
- **`_ERI_SYMMETRY_PERMUTATIONS` is already the corrected +1-only form** from B5, so suspect 2 in the original list is out.
- **Dressing is not implicated**: dressed and raw rank-3 residuals agree to 8.4e-13 symbolically.

**What is still needed is an oracle**, and this is the honest blocker. Comparisons attempted so far were not decisive: GCC vs spin-adapted residuals are different objects (spin-orbital vs spatial) and not directly comparable, and the arbitrary-order vs plain rank-3 emits are close in size with similar term counts, so neither localizes the error.

The tractable oracle is the one the CCSDTQ==FCI gate already relies on: the **arbitrary-order** rank-3 path is exercised as the CCSDTQ warm start and is validated, whereas the **plain** rank-3 TU is the one that had no caller. Both are emitted from the same equations. So the decisive experiment is to run the *arbitrary-order* rank-3 kernel on bh3 (`-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON`) and
compare its T3 residual against the hand-written one with the same probe:

- **arbitrary agrees with hand-written** → the equations are right and the defect is specific to the plain TU's lowering/emit path;
- **arbitrary also differs** → the defect is in the shared rank-3 equations, and the CCSDTQ gate does not cover it because CCSDTQ's own triples residual is a separately-emitted function.

*Gate:* that comparison run, and a term class named from where the raw difference concentrates.

#### T2 — original suspect list (retained)

Compare the generated rank-3 triples residual against a Python oracle at fixed amplitudes, the way
`residual_eval` already does for the symbolic manifolds, and bisect by term class.

Prime suspects, ordered by prior:

1. **The `_ERI_SYMMETRY_PERMUTATIONS` +1/−1 issue**, also from B5: antisymmetric permutations are
   invalid for non-antisymmetrized spatial blocks.
3. **T3 storage convention** — whether the emitted kernel writes the same slice layout the tensor
   solver reads.

*Gate:* a term class identified and a numeric before/after at fixed amplitudes.

### T3 — D-B, the dressing delta (~S, after D-A is fixed)

2.3e-6 Eh between dressed and undressed generated. My symbolic check says dressed ≡ raw to
**8.4e-13** on the rank-3 triples manifold (random symmetry-correct tensors), so the algebra is
equivalent and the divergence is in the **emit**, not the equations — the same layer as V1.2.2's
declared-vs-built layout miscompile.

**Re-measure after D-A lands**: if the two generated kernels agree once D-A is fixed, D-B was
downstream and there is nothing separate to fix.

*Gate:* dressed == undressed generated to solver tolerance, or a named emit-layer cause.

### T4 — D-C, the ~180× slowdown (~M, last)

7 s → 1270 s on both generated builds. Correctness first: the fix for D-A may change the kernel
shape entirely. Likely candidates are intermediates rebuilt inside loops rather than hoisted, and
the absence of CSE (dressing forces `include_intermediates` off).

*Gate:* a profile naming the dominant cost, not a guess.

### T5 — correct the false claims (~S, can start now)

`vault/Status/Completion.md`, `docs/CCGEN_DRESSED_KERNEL_PIPELINE.md`, and the
`dressed_kernel_equivalence_rccsdt` regression case all assert a verified dressed==undressed
equivalence at rank 3. **That verification did not happen** — both builds ran hand-written code —
and the real comparison now *fails*. These are actively misleading and should be corrected
independently of the fix.

The gate itself is now correct (it asserts `RCCSDT[OPT]` + `kernels=ccgen-generated` before
believing a number) and currently reports FAIL, which is the honest state.

---

## Sequencing

```
T0 (diff at fixed amps)          ~S  ← one evaluation, not a 21-min solve
 └→ T1  (bypass restore_*)       ~S  ← decides whether ccgen is implicated
 └→ T1b (missing physicist rebind) ~S  ← MEASURED; likely all of D-A
      └→ T2 (locate in ccgen)    ~M  ← only if T1b does not resolve it
           └→ T3 (D-B dressing delta)  ~S  ← re-measure; may vanish
                └→ T4 (D-C slowdown)   ~M
T5 (correct the false docs)      ~S  ← independent, start immediately
```

## What NOT to do

- **Do not assume the CCSDTQ==FCI result covers this kernel.** Different function, different TU.
- **Do not debug through the solve.** T0 compares one evaluation; the solve costs 21 minutes and
  conflates kernel error with convergence path.
- **Do not attribute D-B to dressing without re-measuring after D-A.** The symbolic algebra is
  equivalent to 8e-13; a 2.3e-6 delta on top of a 1.8e-4 error is more likely downstream.
- **Do not touch the ~180× slowdown before correctness.** The fix may change the kernel entirely.
- **Do not "fix" this by reverting the backend wiring.** Reaching the generated kernel is what
  exposed a real defect; hiding it again restores a false green.
- **Do not leave the docs asserting the old claim** while the fix is in progress. T5 is
  independent for that reason.

---

## Honest status

The dressed-kernel pipeline through *emit, compile, and link* is real and validated. **The run
claim is not** — it was measured on hand-written code in both builds. The generated rank-3 triples
kernel has now executed for the first time and is wrong by 1.8e-4 Eh, independent of dressing.
