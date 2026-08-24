# The remaining UCCSD defect: `ucc2` is 3.8% off, and it is visible at first order

Scopes ONE question: **why does the generated `ucc2` correlation energy disagree with
hand-written UCCSD, after the ERI antisymmetrization fix (`fe744e6`) closed the larger
half of the gap?**

**Status, 2026-08-24. R0-R4 DONE — THE DEFECT IS FOUND, and it is the exchange fix I landed in
`fe744e6`.** `_map_eri_tensor` applies the exchange by swapping the **last two slots**, which is
the ket pair only for `oooo`/`oovv`/`vvvv`. On `ooov`/`ovov`/`ovvv` — 90 of the 180 emitted
exchange pairs — it swaps an occupied index with a virtual one, reading a different space
pattern out of the array. First order was clean because its only term is `oovv`, where the
swap happens to be right. **`test_ucc_eri_convention.py` asserted the swap was "the LAST TWO
slots only" — the gate encoded the bug.** Fix: derive the ket pair from the block's space
pattern, not from position, and re-gate on an `ovov` term.

*Superseded status lines, for the record:* **R0-R2 DONE. The first-order layer is CORRECT; the
defect is in the higher-order terms.** R2 found that the exact `0.800000` R0 and R1 were chasing is the
**`cc_damping` default of 0.8**, not a defect: with `cc_damping 1.0`, iteration 1 reproduces
UMP2 to ten digits on both fixtures. That clears the stored ERI blocks, the per-block
denominators, the rebind, the write-back and the energy coefficients — by measurement. **R4 is
next**, and the amplitude-antisymmetry suspect is LIVE AGAIN, since R0 only ruled it out at
first order.

*Superseded status lines, for the record:* **R0 and R1 DONE. The deficit is in the `abab`
channel.** R0 showed the
first-order algebra is textbook-correct (so the defect is in the data, not the equations, and
the amplitude-antisymmetry suspect is dead). R1 measured the three channels on two systems and
found the ratio is **0.800000 on both** — structural — with the deficit in **`abab`**, not in
the same-spin channels this doc predicted. **R2 is next**, rewritten around that.

*Original status line, for the record:* **R0 DONE — and it killed the prime suspect.** The first-order algebra is
textbook-correct, so the defect is in the **data** reaching the kernels (the stored `v` blocks,
the per-block denominators, or the amplitude written back), **not** in the equations and **not**
in the amplitude-antisymmetry convention this doc originally suspected. **R1 is next and is now
the step that names it.** Every measurement below is reproducible with the committed `build-ucc`
tree and costs seconds.

---

## The single measurement that shapes everything

The defect **does not need the solver to reproduce**. Iteration 1 from a zero start is
the first-order (MP2) amplitude, and it is already wrong — by an exact rational factor:

```
UMP2 E_corr (hand-written, same geometry/basis)   -0.0190946435
ucc2 iteration-1 E_corr                           -0.0152757148
ratio                                              0.800000        <- exactly 4/5
```

**An exact 0.8 is a structural defect — a wrong coefficient or a dropped channel — not
accumulated numerical error.** And because first-order amplitudes are closed-form
(`t2 = <ij||ab> / D`), reproducing it needs no iteration, no DIIS and no convergence
argument.

This is the same lever U3.4 used, and it is why that step "turned out to need no solver
at all". The investigation should live entirely at first order until the ratio is 1.

### Corroborating trace

The converged number is the *smaller* part of the story, but it rules out a diverging solve:

```
iter 1   -0.0152757148
iter 4   -0.0399385985     <- passes very close to the reference
iter 5   -0.0417424885     <- overshoots
iter 100 -0.0418040291     residual plateaus ~5e-8, never converges
reference (hand-written UCCSD)  -0.0402694793      (+3.81%)
```

A residual that plateaus at 5e-8 rather than diverging is a *consistent* set of equations
converging to the *wrong* fixed point. That is consistent with a coefficient defect and
inconsistent with, say, a shape or indexing error, which would blow up or land nowhere near.

---

## What is already ruled out, by measurement

Do not re-derive these:

| Ruled out | Evidence |
|---|---|
| Incomplete ERI exchange emit | Zero bare same-spin `v` reads at either tag; 90 exchange pairs per same-spin block; zero on `abab`; zero on Fock (`test_ucc_eri_convention.py`) |
| Wrong exchange slot pairing | The emitted pair swaps the LAST TWO slots only, asserted per-read |
| Denominator tag convention | `build_ucc_block_denominator` reads "occ half then vir half"; measured against the amplitude free-index layout, `doubles_abab` → occ,occ,vir,vir with spins a,b,a,b → `abab`. Matches |
| Spin-blind loop bounds / shapes | U3b.2; the TU compiles and every bound is spin-resolved |
| A simple double-count | The pre-fix ratio was 1.7515 and the post-fix ratio is 1.0381 — neither is 2 |

---

## The prime suspect, and why

> **LIVE AGAIN as of R2 — R0's dismissal was correct but narrower than it read.** R0 ruled this
> out *at first order*, which is right and which R2 then confirmed by reproducing UMP2 exactly.
> But R2 also showed the defect lives entirely in the HIGHER-ORDER terms — precisely where this
> convention operates. Nothing enforces or checks it. Treat it as a leading suspect for R4, not
> a closed question.

**The amplitude-side antisymmetry convention, which is the exact analogue of the ERI one
just fixed — one layer over.**

`ucc_amplitude_blocks` (`amplitudes.cpp:315`) states the assumption outright:

> "the tag is alpha-before-beta per half (**the within-half antisymmetry folds slot
> permutations, so only the count matters**)"

So the C++ block vocabulary *assumes* `t2_aaaa` is antisymmetric under `i↔j` and `a↔b`.
**Nothing enforces it and nothing checks it** — `grep` for `antisym|restore_|symmetriz`
across `amplitudes.cpp`, `solver_arbitrary.cpp` and `generated_arbitrary_runtime.cpp`
returns only that comment.

Meanwhile the algebra depends on it: **18 of the 104 `doubles_aaaa` terms** carry a `1/2`
or `1/4` coefficient on a `t2_aaaa` factor, and those fractions are only correct if the
permutational copies they stand for are actually present in the stored amplitude.

This mirrors the ERI defect precisely: a convention held implicitly on both sides, correct
on each side in isolation, never pinned between them. It is also the same failure mode as
B5 (physicist-vs-chemists) and the RCC `restore_restricted_t3_structure` coupling, where a
symmetry-packed representation and its restore step are one convention that cannot be split.

**It is a suspect, not a conclusion.** It has not been tested.

---

## Steps

**R0 — reproduce the defect with no solver — DONE, and it KILLED THE PRIME SUSPECT.**

Two findings, both cheap and both changing the plan.

**(a) Iteration 1 IS the first-order energy — no probe needed.** `run_generated_arbitrary_order_iterations`
evaluates residuals from the *current* amplitudes, updates, then reports; amplitudes start at
zero (`make_zero_rcc_amplitudes`) and `run_uccgen` carries no warm start. So the already-printed
iteration-1 number `-0.0152757148` is exactly `t = R(0)/D` — the MP2 limit. R0 needs no new code
at all; the measurement was already on screen.

**(b) THE FIRST-ORDER ALGEBRA IS CORRECT, so the defect is in the DATA, not the equations.**
Measured on the manifold: at `t = 0` every doubles residual collapses to a single constant term —

```
doubles_aaaa   1 * v_aaaa        doubles_abab   1 * v_abab        doubles_bbbb   1 * v_bbbb
```

— and the energy's doubles terms are `1/4 t2_aaaa v_aaaa`, `1 t2_abab v_abab`, `1/4 t2_bbbb v_bbbb`.
Substituting `t2 = v/D` gives `1/4 Σ|<ij||ab>|²/D` for same-spin and `Σ|<ij|ab>|²/D` for mixed —
**textbook UMP2, exactly**. The singles contribute nothing at first order (their only constant
term is `f_ov`, which is zero for a canonical UHF reference — see the `cc_canonical_fock_only`
invariant).

**This kills the amplitude-antisymmetry hypothesis for this defect.** That convention governs how
permutational copies of `t2` combine in the *higher-order* terms; it cannot affect a first-order
energy assembled from one ERI term per block. The suspect named in the parent scope is wrong, and
should not be pursued first. (It may still be a real latent issue — nothing enforces it — but it
is not *this* 3.8%.)

*What remains, therefore:* the values reaching the kernels. Either the stored `v` blocks, the
per-block denominators `D`, or the amplitude written back between them.

**R1 — split the 0.8 by spin channel — DONE. The deficit is in `abab`, and BOTH predictions
this doc made were falsified.**

Measured with a temporary per-channel probe in the runtime loop (inserted, run, reverted —
not committed), on **two** systems:

```
                        aaaa          abab          bbbb        total      vs UMP2
B/STO-3G          0.0000000000  -0.0152757148  0.0000000000  -0.01527571   0.800000
H2O+/STO-3G (C1) -0.0005788702  -0.0205364423 -0.0006610720  -0.02177638   0.800000
```

**The ratio is 0.800000 on both**, to six digits, on systems with nothing in common — which
makes it structural rather than accidental, and rules out any system-specific cause.

**Falsification 1 — "same-spin scaled by k" is dead.** The same-spin share is 0.0% on boron
and 5.7% on the water cation, yet the ratio is identical. No scaling of the same-spin
channels can produce the same total ratio at two such different shares. The `k = 1/2`
prediction (from an assumed ~40% share) was wrong.

**Falsification 2 — the boron zero was a FIXTURE ARTIFACT, not a defect.** On B/STO-3G the
same-spin residuals are *exactly* zero and I briefly read that as the bug. It is not: the
stored `oovv_aaaa` block satisfies `v(i,j,a,b) == v(i,j,b,a)` identically there, so the
`<ij||ab>` exchange cancels — a real property of that high-symmetry atom with `nva = 2`
degenerate 2p virtuals, not of the code. On the low-symmetry water cation
`max|v(ijab) - v(ijba)| = 3.9e-2` and both same-spin residuals are non-zero. **The exchange
fix works.**

> **This is the seventh instance of this scope's fixture-vacuity trap, and the first where
> the fixture passed its own non-vacuity check.** `b_ucc2_sto3g` was verified to have
> asymmetric counts (`noa=3,nob=2,nva=2,nvb=3`) and a non-trivial `E_corr` — both true, and
> both insufficient: they say nothing about *degeneracy within a channel*. **Any assertion
> about same-spin behaviour must run on a C1 system**; add
> `noa=3,nob=2` to the list of things that does not make a fixture general.

**Where the deficit actually is:** `abab` carries it, on both systems.

```
H2O+   abab measured -0.0205364423   needed (UMP2 - aa - bb) -0.0259805385   ratio 0.790
B      abab measured -0.0152757148   needed (UMP2 total)     -0.0190946435   ratio 0.800
```

Since `abab` has no exchange partner and no fractional coefficient (`1 * t2_abab * v_abab`
in both the residual and the energy), the candidates are narrow: the stored `oovv_abab`
values, its denominator, or the index convention relating the two. **R2 is rewritten below
around that.**

*(The `k = 1/2` prediction that stood here was falsified by R1 — see above. Kept out of the
plan rather than silently deleted: it was wrong because it assumed the same-spin share was
~40%, and the measured shares are 0% and 5.7%.)*

**R2 — find the `abab` deficit — DONE, and it dissolved the premise of R0 and R1.**

**The exact 0.800000 was the DAMPING FACTOR.** `OptionsSCF::_cc_damping` defaults to `0.8`
(`types.h:429`), and the Jacobi update applies `delta = damping * R/D`, so iteration 1 lands at
exactly 80% of the MP2 amplitude. The probe caught it as `max rel |t - v/D| = 2.000e-01` on the
mixed block — a ratio too exact to be physics.

**With `cc_damping 1.0`, iteration 1 reproduces UMP2 to all ten digits, on both systems:**

```
B/STO-3G     iter 1  -0.0190946435    UMP2  -0.0190946435    max rel |t - v/D| = 0.000e+00
H2O+/STO-3G  iter 1  -0.0272204807    UMP2  -0.0272204807
```

**So the entire first-order layer is CORRECT and verified**: the stored `oovv` blocks (all three
tags), the per-block denominators including `abab`'s spin assignment, the physicist rebind, the
amplitude write-back, and the energy kernel's coefficients. None of R2's three candidate causes
is the defect; all three are cleared by this measurement.

**Damping does not move the fixed point** — measured, `cc_damping 1.0` converges to the same
`-0.0418041` as `0.8`. So the real defect is untouched by everything R0–R2 examined.

> **WHAT WENT WRONG WITH R0 AND R1, because it is the reusable part.** R0 asserted "iteration 1
> IS the first-order energy" from reading the loop (residuals from zero amplitudes → update →
> report) and **never checked the update for a scale factor**. Every number R0 and R1 built on
> was that damped value, and the "structural, exact 4/5 on two unrelated systems" conclusion was
> exactly right about the *exactness* and entirely wrong about the *cause* — a solver knob, not
> the algebra. The R1 channel split was still worth having (it is what proved the exchange fix
> works on a C1 system), but its headline finding was an artifact.
>
> **The lesson: an exact rational ratio is evidence of a CONSTANT, and a constant is as likely
> to be a configuration default as a coefficient bug. Grep the knobs before theorising about
> the equations.**

**R4 — DONE. The defect is MY EXCHANGE FIX: it swaps the wrong slots on five of the six
space blocks.**

**Localized by the cheap probe first.** With `cc_damping 1.0` and DIIS off, the generated and
hand-written solvers agree *exactly* at iteration 1 and diverge at iteration 2 — so the defect
is in the terms linear in the MP2 amplitude, not in the constant term:

```
iter    generated        hand-written       diff
   1  -0.0190946435    -0.0190946435    +0.000e+00
   2  -0.0251026330    -0.0252106915    +1.081e-04
   3  -0.0304876742    -0.0295950340    -8.926e-04
```

Same signature on the C1 fixture (identical at 1, `+7.07e-05` at 2). The sign flip between
iterations 2 and 3 already rules out a single uniform scale factor.

**The amplitude-antisymmetry suspect is DEAD, measured this time.** Probed directly:
`max|t(ijab) + t(jiab)|` and `max|t(ijab) + t(ijba)|` are `~1e-16` and `~0` for both `aaaa` and
`bbbb`, at iterations 1 and 2. The amplitudes are antisymmetric to machine precision, so the
convention `ucc_amplitude_blocks` asserts is in fact satisfied. It should now be closed as a
suspect for this defect — though it remains unenforced and unchecked, which is worth a gate on
its own terms.

**THE ACTUAL DEFECT.** `_block_needs_explicit_exchange` correctly identifies *which spin blocks*
need an exchange term, but `_map_eri_tensor` applies it with a **blind swap of the last two
slots**, and that is only the ket pair for some space patterns. Measured across the rank-2 UCC
TU, the exchange is emitted on all six:

```
oooo   4     oovv  80     vvvv   6      <- last-two swap stays within one space: OK
ooov  32     ovov  20     ovvv  38      <- last-two swap crosses occ/vir: WRONG
```

Concretely, the `doubles_abab` term `-1 * t2_abab * v_aaaa` binds `v_aaaa` as
`(i:occ, c:vir, k:occ, a:vir)` — an `ovov` read. My rule emits
`v_aaaa_ovov(i,c,k,a) - v_aaaa_ovov(i,c,a,k)`, and the second read has pattern `(o,v,v,o)`: a
different space block entirely, read out of the `ovov` array. Wrong values, and in-bounds only
by luck when the occupied and virtual extents happen to be compatible.

*This also explains why first order was clean:* the only constant term is `1 * v_aaaa` on
`oovv`, where the last-two swap IS the ket swap. Every gate written for the exchange fix
(`test_ucc_eri_convention.py`) asserted the swap was "the LAST TWO slots only" — which is what
the code does, and the wrong thing to assert. **The gate encoded the bug.**

**The fix must swap the two KET slots of the physicist integral, identified from the pattern,
not from position.** The emitter already knows each block's space pattern
(`_canonical_eri_blocks_for`) and the permutation used to reach it, so the ket pair is
derivable there rather than assumed. Re-gate on a term whose `v` is `ovov`, not only on the
`oovv` energy terms.

**R3 — pin the convention, whichever way R4 lands (~S).**
The ERI fix landed `_block_needs_explicit_exchange` as the single place that states its
convention. The amplitude convention deserves the same treatment, plus a numeric gate. **A
structural gate is not enough here** — that is precisely what let both of these through.

---

## Fixtures — use BOTH, and the second is not optional

**`h2o_cation_ucc2_sto3g.hfinp` (H2O+ doublet, C1) is the one to reason on.** Committed with
R1. `noa=5, nob=4, nva=2, nvb=3`; all three channels non-zero; no degeneracy in the same-spin
block (`max|v(ijab) - v(ijba)| = 3.9e-2`). Its UMP2 reference is `-0.0272204807`.

**`b_ucc2_sto3g.hfinp` (B/STO-3G doublet) is kept, but it is DEGENERATE for same-spin
questions** and R1 was briefly misled by it. Its `oovv_aaaa` block satisfies
`v(i,j,a,b) == v(i,j,b,a)` identically, so both same-spin channels are exactly zero at first
order and any same-spin assertion on it passes vacuously. Keep it as the *simplest* case and
because its zero same-spin cleanly isolates `abab`; never conclude from it alone.

Its non-vacuity check, for the record:

- `noa=3, nob=2, nva=2, nvb=3` — every count differs, so a spin-blind bound or a collapsed
  block changes the SHAPE, not merely values.
- `E_corr` is worth −0.0403 Eh, so agreement cannot be two near-zeros matching.
- Both same-spin channels AND the mixed channel are populated.

Reference numbers, all from the committed trees:

```
RHF                -24.1489886649
UMP2 (E_corr)       -0.0190946435
UCCSD (E_corr)      -0.0402694793     hand-written, 13 iters, converged
```

---

## Traps specific to this investigation

- **A same-spin block cannot discriminate spin-routing hypotheses.** This scope has hit
  that vacuity six times. Every assertion here must be made on `abab`, or on a comparison
  where a wrong hypothesis changes a *shape*.
- **Do not gate on the converged number while the first-order number is wrong.** A
  100-iteration solve that plateaus can be made to *look* better by loosening a tolerance;
  the first-order check cannot.
- **`-DPLANCK_CC_MAXORDER=2` does not build** — `tensor_backend.cpp` hard-includes
  `ccsdt_planck_generated.cpp`, so rank 3 is the floor. Use the committed `build-ucc`.
- **A failed `make` can still report exit code 0.** Check for the binary, not the code.
- **`BASIS_PATH` must be set** to run any input from a build tree (`BASIS_PATH=$PWD/basis-sets`).
