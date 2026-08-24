# The remaining UCCSD defect: `ucc2` is 3.8% off, and it is visible at first order

Scopes ONE question: **why does the generated `ucc2` correlation energy disagree with
hand-written UCCSD, after the ERI antisymmetrization fix (`fe744e6`) closed the larger
half of the gap?**

**Status, 2026-08-24. R0 DONE — and it killed the prime suspect.** The first-order algebra is
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

> **SUPERSEDED BY R0 — do not pursue this first.** The reasoning below is kept because it is
> sound about the *convention* (nothing does enforce it, and that is a real latent gap), but R0
> measured that the first-order algebra is exactly textbook UMP2, and a first-order energy built
> from one ERI term per block cannot be affected by how permutational copies of `t2` combine.
> This is not the cause of the 3.8%.

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

**R1 — split the 0.8 by spin channel (~S).**
Report the first-order energy per channel (`aaaa`, `abab`, `bbbb`) against the hand-written
UMP2's three channels. **This is the step that names the defect**: the total ratio is an
exact 4/5, so either one channel is scaled by a rational factor or one is absent, and three
numbers distinguish those cases immediately.

**A sharper, falsifiable prediction now that R0 has narrowed it.** If `abab` is correct and only
the same-spin channels are scaled by a factor `k`, then `(k·E_ss + E_os)/(E_ss + E_os) = 0.8`
pins `k` against the same-spin share:

```
E_ss share   0.30    0.35    0.40    0.45    0.50
k            0.333   0.429   0.500   0.556   0.600
```

**A same-spin share near 40% gives k = 1/2 exactly** — a clean convention slip, and the most
likely single cause. R1 measures the share and `k` together, so it either lands on an exact
rational (naming the defect) or does not (falsifying the whole "one channel scaled" family and
sending the search to the denominators).

*If `abab` is ALSO wrong, the transform or the denominator is the cause, not a coefficient.*

**R2 — test the amplitude-antisymmetry hypothesis directly (~S).**
Take the converged `t2_aaaa` out of the solver and measure `t2(i,j,a,b) + t2(j,i,a,b)` and
`t2(i,j,a,b) + t2(i,j,b,a)`. Both must be zero for the 18 fractional-coefficient terms to
be correct.

*If non-zero, the fix has the same two options the ERI one had, and the same reasoning
should decide it: enforce the symmetry in the solver's update (one mechanism, one place),
or emit the permutational copies explicitly (generated text only). **Prefer whichever keeps
one meaning per stored array**; do not split the vocabulary.*

**R3 — pin the convention, whichever way R2 lands (~S).**
The ERI fix landed `_block_needs_explicit_exchange` as the single place that states its
convention. The amplitude convention deserves the same treatment, plus a numeric gate. **A
structural gate is not enough here** — that is precisely what let both of these through.

---

## Fixture

`b_ucc2_sto3g.hfinp` (B/STO-3G doublet), already committed. **Verified non-vacuous before
use**, which matters given how often this scope has been burned by degenerate fixtures:

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
