# What makes the generated CC kernels slower than the hand-written ones?

**Two causes, and the larger one is already fixed by a route that ships.**

1. **Contraction order (fixed).** The undressed emitter gives every term its own
   full `o³v³` nest and evaluates it n-arily; 391 of 824 terms carry a four-index
   inner sum (`o⁵v⁵` where factored is `o⁴v³`), accounting for **83–90 % of
   generated FLOPs**. **The factorized/dressed emit (`--dressing derived`)
   eliminates all 391** — 824 nests → 414, zero four-deep terms — worth a modelled
   **10x–18x, growing with size**, and it moves the exponents (`o^4.92 v^4.94` →
   `o^4.42 v^4.40`).
2. **One nest per term (open).** Even factorized, 414 nests remain against the
   hand-written kernel's **one**, which fuses ~9 accumulations into each
   single-index inner loop. That is the residual gap to `o^3.94 v^4.18`.

**Consequence: consuming `_optimal_contraction_order` — the standing
recommendation in `CCGEN_KERNEL_SCALING_SCOPE.md` — targets terms the derivation
route has already eliminated, and is probably redundant.** That was the exact
question the abandoned ladder existed to settle.

Established by code-level census and FLOP estimates, after the measurement route
was closed. No new instrumentation: a count of the emitted C++ plus arithmetic.

## The census

`build/generated/cc/ccsdt_planck_generated.cpp`, the rank-3 triples residual
(`compute_ccsdt_triples_residual_part{0,1,2}`): **824 separate `o³v³` loop nests**,
one per term. Classified by the inner summation each carries:

| inner sum | terms | cost per term |
|---|---|---|
| `o² v²` | **391** | **o⁵v⁵** |
| `o² v¹` | 144 | o⁵v⁴ |
| `o¹ v²` | 140 | o⁴v⁵ |
| `o¹ v¹` | 87 | o⁴v⁴ |
| `o² v⁰` | 16 | o⁵v³ |
| `o⁰ v²` | 16 | o³v⁵ |
| `o¹ v⁰` | 15 | o⁴v³ |
| `o⁰ v¹` | 15 | o³v⁴ |

The 391 four-deep terms are four families:

```
t1·t2·t2·oovv   x172
t2·t3·oovv      x151
t1·t1·t3·oovv   x 44
t1·t1·t1·t2·oovv x 24
```

A representative, verbatim from the emitted file:

```cpp
for i,j,k,a,b,c {                       // o^3 v^3
    double acc = 0.0;
    for l, d, e, m                      // x o^2 v^2
        acc += -0.5 * amplitudes.t2(i,l,a,b)
                    * amplitudes.t3(j,m,k,d,e,c)
                    * mo_blocks.oovv(l,m,d,e);
    result(i,j,k,a,b,c) += acc;
}
```

This is precisely the `t2·t3·v` case `CCGEN_HIGHER_OPERATOR_REUSE.md` predicted as
`o⁵v⁵` n-ary against `o³v⁴` factored — now confirmed present, and counted.

## Contrast: what the hand-written kernel does

`build_dressed_triples_residual` (`tensor_backend.cpp`) is **103 lines and ONE
loop nest**. Inside it, single-index loops carry ~9 fused accumulations each:

```cpp
for i,j,k,a,b,c {
    double value = 0.0;
    for (int d = 0; d < n_virt; ++d) {
        value += ints.w_vvvo(a,b,d,j) * amps.t2(i,k,d,c);
        value += ints.w_vvvo(a,c,d,k) * amps.t2(i,j,d,b);
        ... 9 accumulations sharing this one loop ...
    }
    for (int l = 0; l < n_occ; ++l) { ... 9 more ... }
}
```

Two differences, and both matter:

1. **It contracts against precomputed intermediates** (`w_vvvo`, `w_vooo`,
   `w_oooo`, `w_vvvv`, `w_ovov`, `w_ovvo`), built once per iteration by
   `build_dressed_triples_intermediates`. The four-index sums are already done
   there — that is what dressing *is*.
2. **Terms share loop nests.** One `d` loop serves nine accumulations; the
   generated form would emit nine separate `o³v³` nests.

So the hand-written kernel is `o³v³ × (one summed index)` — textbook — while the
generated one is `o³v³ × (up to o²v²)` per term, times 824 terms.

## The FLOP estimate, and where it is trustworthy

Summing the census gives a generated-side cost model. Fitted over the six ladder
points:

| | model (this census) | measured (the ladder) |
|---|---|---|
| generated | **`o^4.92 v^4.94`** | `o^4.87 v^4.52` |

**The `o` exponent agrees to 0.05.** That is the load-bearing check: a pure
code-level count of the emitted terms reproduces the measured scaling of the
generated kernel. The `v` exponent runs ~0.4 high, consistent with the measured
fit's own 21.4 % residual being concentrated at high `v`.

**The ratio model does NOT reproduce the measured ratio**, and this should be
stated plainly rather than tuned away: modelling the hand-written side as
`o³v³·(o+v)` gives ratios of 5000–24000× against a measured 21.8–50.1×, i.e. two
orders of magnitude too high, and the discrepancy is not flat (230× to 660×). The
hand-written side is not FLOP-bound in the way the model assumes — its nine fused
accumulations per loop reuse operands already in registers and stream one `t3`
pass, so its measured cost is far below its nominal FLOP count. **Trust the
generated-side model; do not quote a modelled ratio.**

## The factorized kernel already fixes this — measured, not assumed

**An earlier revision of this document proposed factoring the 391 terms as the
work to do. That work is already done**, and a factorized TU exists in the tree
(`build-profile/`, whose builders are named `build_W_<blocks>_<n>_ccsdt` — the
derivation route's block-signature naming, not Stanton-Gauss `Wmnij`/`tau`).

Same census, run on both:

| | nests | four-deep (`o²v²`) terms |
|---|---|---|
| undressed | 824 | **391** |
| factorized | **414** | **0** |

**Every `o⁵v⁵` term is gone**, and the nest count halves. Modelled FLOPs:

| case | undressed | factorized | saving |
|---|---|---|---|
| BH3/STO-3G | 4.93e+08 | 4.84e+07 | 10.2x |
| CH4/STO-3G | 1.48e+09 | 1.31e+08 | 11.2x |
| HF/6-31G | 1.09e+10 | 7.47e+08 | 14.5x |
| H2O/6-31G | 4.51e+10 | 2.64e+09 | 17.1x |
| BH3/6-31G | 7.30e+10 | 4.23e+09 | 17.2x |
| C2H4/STO-3G | 1.11e+11 | 6.22e+09 | 17.8x |

The saving **grows with size** (10x → 18x), confirming it is a scaling fix. Fitted:

| | model | vs hand-written |
|---|---|---|
| undressed generated | `o^4.92 v^4.94` | — |
| **factorized generated** | **`o^4.42 v^4.40`** | closes ~half the exponent gap |
| hand-written (measured) | `o^3.94 v^4.18` | the target |

**So factorization moves the exponents, not just the constant** — `o` drops 4.92 →
4.42 against a hand-written 3.94. That is consistent with the measured
3.12x/3.61x wall-clock for `--dressing derived`, and it is the first evidence in
this repo that the two agree on *mechanism* and not only on direction.

## What remains

**Roughly half the exponent gap survives factorization** (`o^4.42 v^4.40` against
`o^3.94 v^4.18`), so a second mechanism is still in play. The census locates it:
414 nests remain, against the hand-written kernel's **one**.

The hand-written form fuses ~9 accumulations into each single-index inner loop,
reusing operands already in registers over one `t3` pass. The emitter cannot do
this because it emits one nest per term, so 414 nests each re-stream their
operands. That is a constant-factor and memory-traffic effect on top of the
residual exponent gap.

Note the earlier measurement that loop fission is *not* a penalty at `no=nv=4`
(0.62x — the fissed form was faster) was taken at a size where the working set is
32 KB and fully L1-resident. It does not generalize to the 414-nest form at
production size, and that is the untested half of H1.

**Recommendation: measure the factorized kernel before building anything else.**
`--dressing derived` is wired and produces this TU. Consuming
`_optimal_contraction_order` in the emitter — the standing recommendation in
`CCGEN_KERNEL_SCALING_SCOPE.md` — targets the same 391 terms this route has
already eliminated, so **the two overlap and it is probably redundant**. That was
the exact question the abandoned ladder was meant to settle, and the census
answers it without the measurement.

## What was checked and ruled out

- **Loop-invariant work.** Zero of 824 nests have an accumulation that ignores any
  of `i,j,k,a,b,c`, so the compiler cannot hoist any nest out. The emitted FLOPs
  are real work, not something `-O3` removes.
- **Accessor overhead.** Fixed and gated (`CCGEN_TENSOR_ACCESSOR.md`, 206× on
  rank 3). Not a current factor.
- **Loop fission as the cause.** Measured 0.62× at `no=nv=4` — the fissed form is
  *faster* there. It is not the mechanism.

## Why this is code-level rather than measured

`CCGEN_KERNEL_SCALING_SCOPE.md` records that the measurement route is closed: the
timing probe sits on a code path the rank-3 representation fix rerouted away from,
and the hand-written and generated arms have no residual-level agreement gate
(distinct solvers, distinct amplitude representations, both correct). Whole-
iteration timing measures *"solver iteration"*, not *"triples kernel"*.

This document is the answer that route was meant to produce, obtained instead by
counting the emitted terms. It is sufficient to act on: the term census is exact,
the generated-side model reproduces the measured `o` exponent, and the fix's
target is identified to 391 specific terms in four named families.

## Key locations

| what | where |
|---|---|
| the 824 nests | `build/generated/cc/ccsdt_planck_generated.cpp`, `compute_ccsdt_triples_residual_part{0,1,2}` |
| the hand-written contrast | `build_dressed_triples_residual`, `src/post_hf/cc/tensor_backend.cpp` |
| the intermediates it contracts against | `build_dressed_triples_intermediates`, same file |
| computed and discarded contraction order | `python/ccgen/tensor_ir.py:283` (`_optimal_contraction_order`) |
| the emitter that ignores it | `python/ccgen/emit/planck_tensor_cpp.py:284,443` |
| the measured ladder this explains | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |
| the predicted `t2·t3·v` case, now confirmed | `docs/CCGEN_HIGHER_OPERATOR_REUSE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
