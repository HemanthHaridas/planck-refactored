# What makes the generated CC kernels slower than the hand-written ones?

**Answer: the emitter gives every residual term its own full `o³v³` loop nest and
evaluates each one n-arily. 391 of 824 terms carry a four-index inner sum, making
them `o⁵v⁵` where a factored order is `o⁴v³`. Those 391 terms are 83–90 % of all
generated FLOPs.**

Established by code-level comparison and FLOP estimates, after the measurement
route was closed (see `CCGEN_KERNEL_SCALING_SCOPE.md`). No new instrumentation:
this is a census of the emitted C++ plus arithmetic.

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

## What to fix, in order

**1. Factor the 391 four-deep terms.** They are 83–90 % of generated FLOPs
(83.2 % on BH3/STO-3G, 89.9 % on C2H4/STO-3G, 88.4 % on BH3/6-31G). Modelled
saving from binary factorization alone:

| case | o | v | as emitted | factored | saving |
|---|---|---|---|---|---|
| BH3/STO-3G | 4 | 4 | 4.10e+08 | 1.28e+07 | **32x** |
| CH4/STO-3G | 5 | 4 | 1.25e+09 | 2.82e+07 | **44x** |
| HF/6-31G | 5 | 6 | 9.50e+09 | 1.16e+08 | **82x** |
| H2O/6-31G | 5 | 8 | 4.00e+10 | 3.25e+08 | **123x** |
| BH3/6-31G | 4 | 11 | 6.45e+10 | 5.00e+08 | **129x** |
| C2H4/STO-3G | 8 | 6 | 9.96e+10 | 6.05e+08 | **165x** |

**The saving grows with system size**, which is the signature of a scaling fix
rather than a constant one — and it is why this outranks everything else on the
list. These are modelled FLOP ratios, not predicted wall-clock.

The mechanism already exists: `_optimal_contraction_order` in
`python/ccgen/tensor_ir.py:283` computes exactly this and the emitter discards it
(`grep BLASHint python/ccgen/emit/planck_tensor_cpp.py` returns nothing).
Derivation dressing (`--dressing derived`) addresses the same terms by a different
route and is already wired.

**2. Fuse nests that share an outer index and an inner loop.** The hand-written
kernel's nine-accumulations-per-`d`-loop pattern is unavailable to the emitter
because it emits one nest per term. Secondary to (1) — it changes the constant,
not the exponent, and the earlier measurement found loop fission itself is *not* a
penalty at small size (0.62×, i.e. fissed was faster). Do (1) first.

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
