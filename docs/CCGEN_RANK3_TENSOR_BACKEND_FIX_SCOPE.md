# Fixing the rank-3 generated kernel under `tensor_backend`

**Scope for the fix. Not started.** Continues `CCGEN_RANK3_SURFACE_INVESTIGATION_SCOPE.md`, which
located the defect; this doc is about closing it. Rewrite both into one architecture answer when it
lands.

## What is established, and what it costs to ignore

Four converged CH4/STO-3G runs (`no=5 nv=4`, the smallest in-tree system that clears the determinant
backstop), against `pyscf.cc.rccsdt` = **−39.8058445240**:

| arm | residual producer | harness | Δ vs PySCF | converged |
|---|---|---|---|---|
| A | generated | `tensor_backend` | **−7.56e-05** | yes |
| B | generated | arbitrary runtime | **+1.49e-08** | yes |
| E | hand-written | `tensor_backend` | **+1.45e-08** | yes |
| C / D | generated, `restore` removed / halved | `tensor_backend` | — | **no** |

Read as a 2×2 (producer × harness), three cells are correct and one is wrong. That is the whole
constraint set, and it kills the obvious explanations:

- **The generated equations are right.** Arm B runs exactly them (verified: the two rank-3 TUs share
  811/811 normalized terms, 0 differences either way) and reproduces an independent code.
- **`restore_restricted_t3_structure` is right.** Arm E applies it in the same solver and lands at
  1.45e-08, agreeing with arm B to 4.0e-10.
- **So the defect is in the *pairing*** — what `tensor_backend` hands the generated kernel, or what
  it does with what comes back. Neither component is wrong alone.

**Do not re-open any of these three.** Each was closed by a converged run against an external oracle,
not by inspection. Five earlier fixes in this investigation passed a structural gate and made the
physics worse; the R2 double-symmetrization hypothesis in the predecessor doc was itself refuted this
way, after a numerical model appeared to confirm it. The model assumed a property of the residual
(`G == perm_sym(x)`) that was never measured.

## The prime suspect: the two harnesses disagree about T3 amplitude convention

Both call sites pass four analogous arguments (`reference`, blocks, `denominators`, amplitudes), so
the difference is in what those objects *contain*. The one asymmetry that survives inspection:

- **Arm B never touches its amplitudes.** `solve_arbitrary` / `generated_arbitrary_runtime.cpp`
  contain no `restore`, no `purify`, no symmetrization of any kind. The kernel is handed exactly what
  the Jacobi/DIIS update produced.
- **`tensor_backend` projects T3 with `restore_restricted_t3_structure`** at `:2826` and `:2843`,
  *before* DIIS and again after extrapolation, explicitly "so the subspace vectors are consistent
  with what the next iteration will see."

If the generated kernel expects raw amplitudes and the hand-written one expects projected amplitudes
— or vice versa — the pairing fails exactly as observed while each component is individually correct.

**Caveat that must be resolved first, because it decides whether this suspect is even live.** Those
two projections are in `run_staged_tensor_triples_iterations` (`:2689`). Arm A logged
`RCCSDT[TENSOR-R]`, i.e. `run_restricted_tensor_rccsdt_no_fallback` (`:2581`), which reaches the
kernel through `update_restricted_rccsdt_amplitudes_once` (`:2263`) and updates T3 via
`update_restricted_t3_from_r3_jacobi` (`:2223`) — a plain Jacobi step with **no projection**. So the
projections may be on a path arm A never executed. **F1 settles this before any fix is written.**

Second candidate, if T3 convention comes back clean: the **T3 → singles/doubles feedback** at
`:2291` (`add_dressed_triples_feedback_into_sd_residuals`), hand-written and consuming `amps.t3`
directly. Arms C and D stalled with `rms(R3)`, `rms(R1[T3])` and `rms(R2[T3])` plateauing together
at ~1e-4 while `rms(SD)` stayed small — a coupled-feedback signature, not a residual-magnitude one.

## The work

### F1 — ANSWERED: the amplitude-projection suspect is on a DEAD path. T3 is never projected in arm A

`run_tensor_rccsdt` dispatches on `backstop.enabled`, and the two branches are **mutually
exclusive** (`tensor_backend.cpp:3025` vs `:3065`):

- `!backstop.enabled` → `run_restricted_tensor_rccsdt_no_fallback` (`:2581`) — **arm A's path**, and
  it `return`s from inside the branch.
- `backstop.enabled` → `run_staged_tensor_triples_iterations` (`:2689`) — where the amplitude
  projections at `:2826`/`:2843` live.

CH4 clears the backstop, so arm A takes the first branch and **never executes `:2826`/`:2843`**. The
prime suspect named when this doc was drafted — "arm A projects its T3 amplitudes, arm B does not" —
is **wrong**: neither harness projects amplitudes on the paths actually compared.

The complete list of T3 transforms on arm A's executed path, which is the F1 deliverable:

| stage | transform |
|---|---|
| residual out of the generated kernel (`:2338`) | `restore_restricted_t3_structure` (`:2507`) — **and nothing else** |
| amplitude update (`:2619` → `:2263` → `update_restricted_t3_from_r3_jacobi`, `:2223`) | plain Jacobi step, **no projection** |
| solver loop `run_restricted_tensor_rccsdt_no_fallback` (`:2581`–`:2688`) | **no T3 transform at all** |

(`:2502`/`:2503` are the R2.1 probe branch, inert unless `PLANCK_CC_T3_COMPACT=1`; `:2471`/`:2476`
are probe-local; `:2529` is the hand-written branch's own residual restore, i.e. arm E's.)

**So the amplitudes entering the kernel are identical in kind between the two harnesses, and the
only post-kernel transform on arm A is a `restore` that arm E proves correct.** That is a sharp
constraint: the difference must be in the *other* two arguments — the ERI blocks or the denominators
— or in the T3→SD feedback consuming the result.

This raises the second candidate to prime. `add_dressed_triples_feedback_into_sd_residuals`
(`:2291`) runs **before** the kernel call on every iteration and is hand-written; it consumes
`amps.t3` and mutates the SD residuals that the same iteration's T1/T2 update then uses. Arms C and D
stalled with `rms(R3)`, `rms(R1[T3])`, `rms(R2[T3])` plateauing together while `rms(SD)` stayed
small — a coupled-feedback signature.

Also worth checking cheaply, now that amplitudes are excluded: `physicist_blocks` is rebound by the
caller for the generated branch while `state.mo_blocks` stays chemists' for the hand-written one
(`:2336` comment). Arm B's blocks are rebound by `prepare_generated_arbitrary_order_state`
(`generated_arbitrary_prepare.cpp:88`) through the *same* `rebind_physicist`. Confirm the two
rebinds produce identical blocks rather than assuming it — this is the convention that was already
found missing once at rank 3 (`eb1c611`), and `rebind_physicist` cross-sources `oovv`↔`ovov`, which
is exactly the kind of pairing that survives a square system and fails on a non-square one.

### F2 — the stride-mismatch answer below is **WITHDRAWN**. It was wrong; read this first

**Retracted 2026-08-16.** The section that follows concluded the generated kernel receives a
spin-orbital `RCCSDTAmplitudes` while looping over spatial extents. **It does not.** Arm A's loop
builds a *local* spatial `amps` via `project_rccsd_warm_start_to_restricted`
(`tensor_backend.cpp:2594`), whose `t3` is
`Tensor6D(n_occ,n_occ,n_occ, n_virt,n_virt,n_virt)` — spatial, matching the kernel exactly — and
**that** local is what reaches `compute_ccsdt_triples_residual`, not `state.triples.amplitudes`.

The error: the `T3~=3.91 MiB` figure used as proof comes from the memory report for
`state.triples.amplitudes`, the spin-orbital allocation belonging to the **staged** path
(`run_staged_tensor_triples_iterations`) — the branch F1 had already established arm A does not
execute. A log line from one code path was attributed to another. Checking which object the report
described would have caught it, as would noticing that it contradicted F1.

So: layouts match, and **F2 is unresolved**. The kernel's four arguments are now *all* accounted for
as equivalent between the harnesses:

| argument | verdict |
|---|---|
| `reference` | same builder, same calculator — identical |
| blocks | same `rebind_physicist(build_tensor_cc_block_cache(...))` — identical |
| `denominators` | differ in construction but **inert**: zero `denominators.` reads in the generated TU |
| amplitudes | **both spatial** — identical in kind |

That is a genuinely surprising state: every input matches, the equations match (811/811 terms), the
post-kernel `restore` is correct (arm E), and yet arm A is wrong by 7.56e-05 and arm B is right to
1.49e-08. **One of those four "identical" claims is false, or the difference is in something not yet
enumerated** — the T3→SD feedback at `:2291`, the DIIS/packing path
(`pack_restricted_unique_rccsdt_amplitudes` packs only the unique triangle), or the convergence
driver.

**Next measurement, and this time do it directly rather than by elimination:** the fixed-amplitude
diff originally scoped as F2 — evaluate `compute_ccsdt_triples_residual` on *bitwise-identical*
inputs in both harnesses and compare element by element. Elimination-by-inspection has now produced
two wrong answers in this investigation (R2's double symmetrization, F2's stride mismatch); a direct
comparison at fixed inputs has produced every correct one.

<details>
<summary>Withdrawn reasoning, kept because the retraction is the lesson (click to expand)</summary>

#### (WITHDRAWN) the defect is a STRIDE mismatch

Found by elimination over the kernel's four arguments, which is cheaper than the fixed-amplitude
diff originally scoped here and gives a sharper answer:

| argument | arm A | arm B | same? |
|---|---|---|---|
| `reference` | `build_canonical_rhf_cc_reference(calculator)` | same function, same calculator | **yes** |
| blocks | `rebind_physicist(state.mo_blocks)` (`:2611`) | `rebind_physicist(build_tensor_cc_block_cache(...))` | **yes** |
| `denominators` | `build_denominator_cache(..., false)` | `build_arbitrary_order_denominator_cache(...)` | differ, but **inert** — the generated TU never reads them (all 12 occurrences are parameter declarations; zero `denominators.` accesses) |
| **amplitudes** | **spin-orbital dims** | **spatial dims** | **NO — this is the defect** |

`prepare_tensor_rccsdt` allocates the triples amplitudes at **spin-orbital** extents
(`tensor_backend_state.cpp:299-307`):

```cpp
const int nocc_so  = 2 * partition.n_occ;
const int nvirt_so = 2 * partition.n_virt;
state.triples.amplitudes = RCCSDTAmplitudes{ .t3 = Tensor6D(nocc_so,nocc_so,nocc_so, nvirt_so,nvirt_so,nvirt_so, 0.0), ... };
```

while the generated kernel is a **spatial** (spin-adapted) kernel whose loops run to
`reference.orbital_partition.n_occ` / `n_virt` and which indexes `amplitudes.t3(i,j,k,a,b,c)`
directly (`ccsdt_planck_generated.cpp:2690`).

Confirmed numerically from arm A's own log — the memory report prints `T3~=3.91 MiB`:

| layout | extents (CH4, `no=5 nv=4`) | elements | size |
|---|---|---|---|
| spatial (what the kernel assumes) | 5×5×5×4×4×4 | 8,000 | 0.06 MiB |
| **spin-orbital (what is allocated)** | 10×10×10×8×8×8 | 512,000 | **3.91 MiB** ✓ |

So the kernel walks a `Tensor6D` whose **strides are twice the extents it is looping over**. Every
`t3(i,j,k,a,b,c)` read lands at the wrong offset, and every write scatters into the wrong element —
while staying comfortably in bounds, because the allocation is 64× larger than the region addressed.
It cannot fault; it silently reads a sparse, mostly-zero sub-lattice of the spin-orbital tensor.

**This explains every recorded symptom of the "rank-3 defect", none of which needed a symmetry
hypothesis:**

- **~45 % of elements unwritten, generated-live a strict subset of hand-live** — the kernel only ever
  addresses the `[0,no)×…×[0,nv)` corner of a `2no`/`2nv` tensor.
- **Values "largely unrelated", ratios spanning −149 to +66 with no dominant bucket** — each read is
  a different wrong element, so no constant factor can emerge. This is exactly why "constant
  mis-weight (½, 2, …)" was ruled out early.
- **31 % sign flips** — arbitrary wrong elements carry arbitrary signs.
- **Why `restore` looked guilty** — it is the only transform between the corrupt residual and the
  update, so removing (arm C) or halving (arm D) it changed the wrongness without curing it.
- **Why arm B is correct** — `make_zero_rcc_amplitudes` sizes every rank from `rank_dims`, which is
  spatial `n_occ`/`n_virt` (`amplitudes.cpp`), matching the kernel exactly.
- **Why the square-system warning kept recurring in these docs** — this is precisely the class of bug
  that hides when `no == nv`.

Note `make_zero_rccsdt_amplitudes` (`amplitudes.cpp:306`) already builds the **spatial** RCCSDT
layout. So the tree contains two allocators for the same struct with different conventions, and
`prepare_tensor_rccsdt` picks the spin-orbital one. The hand-written kernel is correct *because it
was written against the spin-orbital layout* — arm E proves the pairing is self-consistent, not that
the layout is canonical.

</details>

### F2 (original scope) — the fixed-amplitude diff, now the live step

R1 compared *converged solves*, which conflates kernel behavior with convergence path. Instead:
take arm B's converged amplitudes, evaluate the generated residual **once** in each harness, and
diff elementwise.

This is the measurement the whole investigation has lacked. Every prior comparison was
generated-vs-hand *within one harness*; this is one-kernel-two-harnesses at fixed input, so any
difference is purely the convention.

*Verify:* if the residuals agree, the defect is in the amplitude **update**, not the kernel call —
and F3 moves to the update path. If they differ, dump the disagreeing elements' index structure
(diagonal vs off-diagonal, occ-permutation orbit) to name the convention.

**Use `nv != no`.** CH4 is `no=5 nv=4`. A square system lets a wrongly-ordered read stay in bounds
and fail silently — the trap that has bitten this investigation twice.

#### F2 — ANSWERED (measured): the kernel is exonerated bitwise; the defect is a MISSING/DOUBLED T3→T3 feedback

Both harnesses were made to evaluate the identical kernel from **bitwise-identical inputs**
(`PLANCK_CC_T3_FINGERPRINT=1 PLANCK_CC_T3_COLD=2`: a deterministic index-derived pattern
`0.01·sin(i+1)` injected into t1/t2/t3 in both, applied to a local copy on the arbitrary side so the
live solve is untouched):

| quantity | arm A (`tensor_backend`) | arm B (arbitrary) | match |
|---|---|---|---|
| t1 / t2 / t3 | +9.982218844197820e-03 / +9.705584301094699e-03 / +1.354084077088470e-02 | same | **bitwise** |
| all 7 ERI blocks | — | same | **bitwise** |
| **R3_out sum** | **−3.015320877582520e-01** | **same** | **bitwise** |
| **R3_out sumsq** | **3.629457228548760e+01** | **same** | **bitwise** |

Sum **and** sum-of-squares agreeing to 16 digits rules out a permutation of the same values, which a
single checksum would miss. A first attempt using zeroed amplitudes was **uninformative** — R3_out
is then trivially zero in both arms — which is why the non-trivial pattern was added; recorded
because "inputs identical" is not sufficient, the output must also be non-degenerate.

**So the generated kernel, its inputs, and its output are all correct inside `tensor_backend`.**
Combined with arm B (equations correct vs PySCF) and arm E (`restore` correct), every component is
individually verified and only the composition fails.

**The structural difference, found by diffing the two branches of
`update_restricted_rccsdt_amplitudes_once`:**

```
hand-written branch (:2555)          generated branch (:2338)
  build_dressed_triples_intermediates  compute_ccsdt_triples_residual(...)
  add_dressed_triples_feedback_into_     <-- NO EQUIVALENT
      triples_intermediates(amps.t3)
  build_dressed_triples_residual
```

`add_dressed_triples_feedback_into_triples_intermediates` (`:1761`) folds a `t3·⟨lm‖de⟩`
contribution into `w_vooo` and `w_vvvo` before the hand-written residual is built. The generated
branch never calls it — **and should not need to**, because those terms are already emitted inline:
195 `t3·oovv` accumulations appear in the generated triples residual.

This explains every observation:

- R3_out identical → the kernel computes its own full residual either way.
- Arm B correct → the arbitrary harness has no dressed-intermediate layer at all, so nothing is
  dropped or double-counted.
- `rms(R1[T3])` = `rms(R2[T3])` = **exactly 0** at iteration 1, nonzero from iteration 2 → the
  T3-dependent coupling only engages once `t3 ≠ 0`.
- Arm A descends *through* the correct energy between iterations 2 and 3 and converges smoothly to a
  different fixed point (−4.27e-05 past it) → a self-consistent residual equation with a term
  missing or counted twice, not arithmetic corruption.

**Still undetermined, and it decides the fix's direction:** whether the generated branch is *missing*
this contribution, or whether the shared SD feedback at `:2291` — which runs for **both** branches,
before the split — now *double-counts* it for the generated path, since the generated kernel already
carries those terms internally. The two imply opposite fixes. Directly measurable: evaluate arm A's
residual with and without the shared feedback and score both against arm B's converged answer.

### F3 — the defect is a HYBRID RESIDUAL. Fix by making the branch consistent, not by toggling a term

**F2's follow-up measurement, and the diagnosis it forces.** `PLANCK_CC_T3_NO_SD_FEEDBACK=1` skips
the shared T3→SD feedback for the generated branch only (control run reproduced arm A's
−39.8059200873 exactly, so the delta is attributable):

| arm | config | energy | Δ vs PySCF |
|---|---|---|---|
| A | generated, unchanged | −39.8059200873 | **−7.56e-05** (overshoots) |
| H | generated, SD feedback **off** | −39.8056549607 | **+1.90e-04** (undershoots) |
| B | arbitrary harness | −39.8058445091 | +1.49e-08 |

Removing it overshoots the other way and is 2.5× worse, so **it is not a pure double count** — but
it moved the energy by 2.65e-04, an order of magnitude larger than the 7.56e-05 defect, and moved it
*through* the correct answer. Nothing else tested has that property. The term is implicated; the
naive on/off framing is what is wrong.

**The structural reason, which makes the on/off question ill-posed.** `tensor_backend`'s generated
branch does not run a generated CC iteration at all — it runs the **hand-written** iteration with
*only the triples residual* swapped for the generated kernel:

| residual | arm A (`tensor_backend` "generated") | arm B (arbitrary) |
|---|---|---|
| r1, r2 | **hand-written** `build_dressed_sd_residuals` (`:2289`), shared code above the branch | **generated** rank-1, rank-2 kernels |
| r3 | generated kernel | generated rank-3 kernel |
| T3→SD feedback (`:2291`, mutates `r1`/`r2`) | applied | none |
| T3→triples-intermediates feedback (`:1761`, mutates `w_vooo`/`w_vvvo`) | **skipped** | none |

The two feedbacks touch **disjoint** targets (`r1`/`r2` vs `w_vooo`/`w_vvvo`), so they are not
alternatives and toggling one cannot substitute for the other.

Arm A is therefore a **hybrid**: hand-written singles/doubles equations closed against a generated
triples residual. Each half is correct in its own scheme — arm B proves the generated set is
self-consistent, arm E proves the hand-written set is — but the two schemes partition the same
physics differently, so mixing them double-counts some contributions and drops others. That is
exactly the observed signature: a smooth descent to a self-consistent *wrong* fixed point, bracketed
by the two feedback settings rather than cured by either.

**Consequence: there is no correct setting of the existing toggles.** A fix that flips a flag is
choosing between two wrong hybrids. The branch has to be made internally consistent — every residual
from one scheme.

Options, in preference order:

1. **Use the generated kernels for all three ranks in `tensor_backend`'s generated branch**, i.e.
   make it a genuine generated iteration rather than a triples-only swap. This is what arm B does
   and what makes arm B correct. It also deletes the hybrid rather than parameterising it.
2. **Delete the generated branch from `tensor_backend` entirely** and route rank-3 generated work
   through the arbitrary harness (`cc3`/`ccsdt_gen`, which already exists and is already correct).
   The simplest change by far, at the measured ~500× cost — acceptable only if this path is not a
   production route.
3. **Derive the missing/duplicated terms and patch the hybrid.** Rejected: it keeps two partitioning
   schemes coupled through hand-maintained correction terms, which is the spaghetti outcome the
   project rule forbids — and every term added has to be re-derived whenever ccgen's factorisation
   changes.

Option 1 is the mechanism fix; option 2 is the honest deletion. Both are consistent with the
standing constraints below. Option 3 is not.

#### Option 1 was TRIED and does NOT fix it — the hybrid diagnosis above is refuted

`PLANCK_CC_T3_ALL_GEN=1` wires `compute_ccsdt_{singles,doubles}_residual` into the branch and skips
the two hand-written completions that would then double-count (the T3→SD feedback and the r2
`(ij|ab)↔(ji|ba)` symmetrization). Control reproduced the hybrid exactly, so the delta is
attributable.

| arm | r1/r2 | r3 | energy | Δ vs PySCF |
|---|---|---|---|---|
| A | hand-written | generated | −39.8059200873 | −7.56e-05 |
| H | hand-written, SD-fb off | generated | −39.8056549607 | +1.90e-04 |
| **I** | **generated** | **generated** | **−39.8057622521** | **+8.23e-05** |
| B | generated (arbitrary harness) | generated | −39.8058445091 | +1.49e-08 |

Arm I **converged cleanly** — 21 iterations, `rms(R3)=7.9e-10`, `norm(d tamps)=1.9e-10` — to a
genuine fixed point that is still wrong, and marginally worse than the hybrid it replaced.

**So "which residuals are used" is not the defect.** The same three generated kernels that are
correct in arm B are wrong in `tensor_backend` even when all three are used together. Every
composition of residual sources tried (hand+gen, hand-minus-feedback+gen, gen+gen) lands at a
different wrong fixed point, while arm B — identical kernels, different surrounding solver — is
right to 1.49e-08.

That leaves the **amplitude update** as the difference, and it is now the only untested layer:

| | arm A (`tensor_backend`) | arm B (arbitrary) |
|---|---|---|
| t1/t2/t3 update | `update_restricted_rccsdt_amplitudes_once` → `update_restricted_t3_from_r3_jacobi`, denominators recomputed on demand via `restricted_d1/d2/d3` | `update_amplitudes_with_jacobi_diis` reading `ArbitraryOrderDenominatorCache` |
| DIIS vector | unique `i<=j<=k` triangle, rebuilt through `restore_restricted_t{2,3}_from_unique` | full dense tensors |
| post-residual transform | `restore_restricted_t3_structure` | none |

Note the denominators were dismissed earlier as "inert" — correct for the *kernel*, which never
reads them, but the **update** does, and arm A and arm B build them with different functions
(`build_denominator_cache` vs `build_arbitrary_order_denominator_cache`). That dismissal was scoped
to the wrong consumer.

#### Denominators cleared, and arm J closes the loop: `restore` and the DIIS packing are ONE convention

**Denominators are not the defect** (read, not inferred): `restricted_d3` computes
`Σε_occ − Σε_virt` on demand, `build_arbitrary_order_denominator_cache` tabulates the *same*
formula, and both arms apply it as the identical Jacobi step `t += damping·R/D` with the same
`1e-12` guard.

**Arm J** = all-generated residuals **and** `restore` skipped — the first configuration that matches
arm B's scheme end to end inside `tensor_backend`. It **diverges** (64 iterations, no convergence,
best `rms(R3)=1.05e-04`).

| arm | r1/r2 | r3 | `restore` | result |
|---|---|---|---|---|
| A | hand | gen | yes | converges, −7.56e-05 |
| I | gen | gen | yes | converges, +8.23e-05 |
| **J** | **gen** | **gen** | **no** | **diverges** |
| C | hand | gen | no | diverges |
| B | gen | gen | n/a (no such step) | correct, +1.49e-08 |

**The pattern across C and J: removing `restore` diverges regardless of residual source.** That is
not a coincidence — `restore` is load-bearing for a reason that has nothing to do with the residual:

`tensor_backend`'s DIIS packs the **unique wedge only** (`pack_restricted_unique_rccsdt_amplitudes`:
`i<=j` for t2, `i<=j<=k` for t3) and `unpack_restricted_unique_rccsdt_amplitudes` rebuilds
everything off-wedge via `restore_restricted_t{2,3}_from_unique`. That round trip is only
information-preserving if the amplitudes carry full permutational symmetry — which is exactly the
property `restore_restricted_t3_structure` imposes on the residual each iteration.

**So `restore` + unique-wedge DIIS are a single coupled convention, not two independent choices.**
Deleting either half breaks the other. Arm B needs neither because it packs dense tensors.

This reframes the whole surface: the defect is not in any one component but in the fact that
`tensor_backend`'s solver is **built around a symmetry-packed amplitude representation**, and the
generated kernels do not produce residuals in that representation. Arms A/H/I are three ways of
mixing the two representations; each converges to a different wrong fixed point. Arms C/J remove the
symmetrization without removing the packing that depends on it, and diverge.

**Consequence for the fix, and it invalidates options 1 and 3 both.** There is no combination of the
existing flags that is correct, because the representation mismatch is structural. The remaining
honest choices are:

- **Convert the generated residual into the symmetry-packed convention** the solver requires — i.e.
  determine what transform maps the generated (fully-permutation-explicit) residual onto the wedge
  representation, and apply it in place of `restore`. R2.1's `p3_full`+`purify` was a first attempt
  at exactly this and failed; the correct transform has not been derived.
- **Replace the unique-wedge DIIS with dense packing** in `tensor_backend`, making its solver
  representation-compatible with the generated kernels. Larger change, but it removes the coupling
  rather than bridging it, and would let `restore` be dropped honestly.
- **Option 2 (delete the branch)** — unchanged, and now more attractive: the arbitrary harness
  already implements exactly the second bullet.

**Option 1 is cheap, because the kernels are already there and already unused.** The plain rank-3
TU that `tensor_backend.cpp` `#include`s emits `compute_ccsdt_singles_residual` (`:76`) and
`compute_ccsdt_doubles_residual` (`:576`) with signatures **identical** to the triples kernel's —
`(CanonicalRHFCCReference, TensorCCBlockCache, DenominatorCache, RCCSDTAmplitudes)`, exactly what
arm A's call site already has in hand. Neither has a caller anywhere in `src/`.

That is the *same defect class as the original one*: generated, compiled, linked, never executed.
`CCGEN_RANK3_TRIPLES_DEFECT.md` recorded the lesson as "linkage is not execution" after
`compute_ccsdt_triples_residual` sat callerless for months. Wiring the triples kernel alone is what
created the hybrid — the singles and doubles kernels were left behind, and the hand-written ones
kept running in their place.

**Open question for whoever implements it:** whether `tensor_backend`'s generated branch has a
reason to exist once option 1 is done, since it would then duplicate the arbitrary harness's job
with a different (and much faster) surrounding solver. If the answer is "no", option 2 is the real
fix and option 1 is a waypoint.

What survives regardless, and is the standing constraint on the fix:

- **Fix the mechanism, not the call site.** `rebind_physicist` is the in-tree precedent: it fixed
  the block convention at one call site, left the contract implicit, and rank 3 stayed wrong. One
  mechanism with a parameter beats two parallel paths; a conversion that exists only to patch a
  single call site is spaghetti and must not be added.
- **Prefer making a contract un-violable over documenting it.** If the defect is a convention
  mismatch, encoding the convention in a type (so the wrong pairing does not compile) outranks a
  named conversion function, which outranks a comment.
- **Whatever it is, check whether the same boundary is what blocks C3** in
  `CC_AMPLITUDE_CHECKPOINT_REMAINING_SCOPE.md` (hand-written spin-orbital solvers cannot write the
  spatial `.ccamp` sidecar). If one conversion serves both consumers, it is a mechanism worth
  building; if it serves only the kernel call, it is a patch. Note
  `project_rccsd_warm_start_to_restricted` (`:1487`) is **already** a spin-orbital→spatial
  projection for t1/t2 — C3 likely needs its t3 counterpart and an inverse, not a new bridge.

*Verify:* `ch4_rccsdt_sto3g` in `optimized` mode reaches −39.8058445240 ± 1e-7 **and** converges;
the same case in default mode stays bitwise at −39.8058445095 (arm E unchanged). Both halves of
F4's gate then pass, which is the fix's definition of done.

### F4 — the gate that should already exist (~S, do even if the fix slips)

**Every rank-3 CC regression case routes to the determinant backstop**, so the hand-written tensor
solver and the generated kernel under `tensor_backend` are both unreachable from CI:

| case | nso | ndet | path exercised |
|---|---|---|---|
| `h2_rccsdt_sto3g` | 4 | 6 | determinant prototype |
| `lih_rccsdt_sto3g` | 12 | 495 | determinant prototype |
| `water_rccsdt_sto3g` | 14 | 1001 | determinant backstop |
| `be_rccsdtq_sto3g` | 10 | 210 | backstop-eligible |

`water_rccsdt_sto3g` *asserts* the handoff string `RCCSDT[TENSOR] : Using the determinant-space
CCSDT backstop` — a gate pinning that the tensor path declines to run. The PySCF-validated CC suite
validates the determinant prototype; it has never covered the tensor path.

Add `ch4_rccsdt_sto3g` (`nso=18 ndet=43758`, `no=5 nv=4`, input already at
`tests/inputs/investigation/ch4_ccsdt.hfinp`), pinned to PySCF **−39.8058445240**, in both backend
modes:

- `PLANCK_RCCSDT_BACKEND=tensor` → hand-written (currently passing — pins arm E so a future change
  cannot silently regress the only correct `tensor_backend` configuration)
- `PLANCK_RCCSDT_BACKEND=optimized` → generated (currently failing — **this is the fix's gate**)

Land the hand-written half **now**, independent of the fix: it is green today and closes a coverage
gap that has been open for the life of the tensor backend.

Runtime is ~31 s (hand-written) and ~100 s (generated) on a Release build, so tag `extended`.

## Alternative that is available today: route rank ≤ 3 to the arbitrary harness

Arm B is correct, and the routing barrier is one compile-time constant —
`generated_floor = PLANCK_CC_ARBITRARY_LOWER_RANKS ? 3 : 4`
(`generated_kernel_registry.cpp:48`), with the `case 3:` dispatch already written.

Cost, measured on CH4/STO-3G, all three on the same Release build and input:

| path | wall time |
|---|---|
| hand-written tensor (`PLANCK_RCCSDT_BACKEND` unset → `TensorProduction`) | **0.19 s** |
| generated under `tensor_backend` (`=optimized`) | 31.3 s |
| generated under the arbitrary harness | ~100 s |

**An earlier revision of this doc put the reroute cost at "≈3.1× (100 s vs 31 s)". That was wrong:
the 31 s baseline was the *generated* kernel under `tensor_backend`, not the hand-written path.**
Against the correct baseline the reroute is **~500×**, and the correction matters because it
inverts the recommendation.

| | correctness | cost vs hand-written |
|---|---|---|
| route rank ≤ 3 to arbitrary | proven (1.49e-08) | **~500× slower** |
| fix `tensor_backend` (F3) | mechanism now known (F2) | keeps the 0.19 s path |

**Recommendation: fix F3.** With F2 naming a one-line stride mismatch, the fix is small, and the
reroute's cost is no longer close to acceptable as a default. Keep the reroute only as an opt-in
escape hatch. Two things would still need settling before it could ever be a default:

- **Rank 2 has never executed.** Parity P1 validated rank-2 *equations* against PySCF; the emitted
  rank-2 TU has no consumer. Routing `ccsd` there needs its own numeric gate first — the exact
  "linkage is not execution" trap that hid this defect.
- **The existing rank-3 gates assert determinant markers**, so rerouting changes what they exercise.
  F4's new case must land first, or the reroute removes coverage while appearing to add it.

## What NOT to do

- **Do not re-litigate the three closed cells** (generated equations, `restore`, hand-written
  producer). Each is closed by a converged run against PySCF.
- **Do not gate on a converged energy alone for a residual defect** — `restore` masks raw residual
  error 11–29×. F2 compares at fixed amplitudes.
- **Do not fix per call site.** Four crossings are known; a targeted patch re-arms the trap.
- **Do not trust a probe number without the backend marker.** `RCCSDT[OPT]` /
  `kernels=ccgen-generated` for the generated path; a build silently selects another backend
  otherwise.
- **Do not use a system under the backstop** (`nso <= 16 && ndet <= 10000`): it converges correctly
  via a path that never calls the kernel, which looks exactly like a pass. LiH cost one wasted run
  here.
- **Do not use a square system.** `no == nv` hides ordering errors.
- **Do not build without `CMAKE_BUILD_TYPE`.** The repo's `build/` has it empty, which drops
  `-DNDEBUG`, re-enables the accessor bounds asserts, and makes every CC timing meaningless.

## Key code locations

| what | where |
|---|---|
| generated kernel call (arm A) | `src/post_hf/cc/tensor_backend.cpp:2338` |
| generated kernel call (arm B) | `src/post_hf/cc/generated_arbitrary_runtime.cpp:138` |
| arm A's solver loop | `run_restricted_tensor_rccsdt_no_fallback`, `:2581` |
| arm A's amplitude update (no projection) | `update_restricted_t3_from_r3_jacobi`, `:2223` |
| T3 amplitude projections (other loop — F1 must confirm reachability) | `:2826`, `:2843` |
| T3 → SD feedback (second suspect) | `add_dressed_triples_feedback_into_sd_residuals`, `:2291` |
| `restore_restricted_t3_structure` (correct — arm E) | `:1977` |
| backstop gate that hides all of this from CI | `choose_determinant_backstop`, `:241` |
| routing constant for the reroute alternative | `generated_kernel_registry.cpp:48` |
| CH4 inputs | `tests/inputs/investigation/ch4_ccsdt.hfinp`, `ch4_cc4_sto3g.hfinp` |
