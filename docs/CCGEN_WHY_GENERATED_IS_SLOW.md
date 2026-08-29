# What makes the generated CC kernels slower than the hand-written ones?

**Two causes. The larger already ships; the next lever is loop fusion, worth 32x
fewer passes over the result tensor.**

| # | cause | status | size |
|---|---|---|---|
| 1 | **contraction order** — 391 of 824 terms evaluated n-arily at `o⁵v⁵` | **FIXED** by `--dressing derived` | 83–90 % of undressed FLOPs; modelled **10x–18x**, growing with size |
| 2 | **one loop nest per term** — 414 nests, only 13 distinct inner loops | **OPEN — the next lever** | **32x** fewer `o³v³` traversals |

Established by code-level census of the emitted C++ plus FLOP arithmetic, after
the measurement route was closed (`CCGEN_KERNEL_SCALING_SCOPE.md`). No new
instrumentation.

**Immediate consequence: consuming `_optimal_contraction_order` — the standing
recommendation in `CCGEN_KERNEL_SCALING_SCOPE.md` — is probably redundant.** It
targets exactly the 391 terms the derivation route already eliminates. That was
the precise question the abandoned ladder existed to settle, and the census
answers it without a measurement. **Fusion is what remains.**

---

## Cause 1: contraction order — fixed, quantified

The undressed emitter gives every residual term its own full `o³v³` nest and
evaluates it n-arily. Census of `build/generated/cc/ccsdt_planck_generated.cpp`
(`compute_ccsdt_triples_residual_part{0,1,2}`): **824 nests**, by inner summation:

| inner sum | terms | cost each |
|---|---|---|
| `o²v²` | **391** | **o⁵v⁵** |
| `o²v¹` | 144 | o⁵v⁴ |
| `o¹v²` | 140 | o⁴v⁵ |
| `o¹v¹` | 87 | o⁴v⁴ |
| others | 62 | ≤o⁵v³ |

The 391 four-deep terms are **83–90 % of all generated FLOPs** (83.2 % BH3/STO-3G,
89.9 % C2H4/STO-3G, 88.4 % BH3/6-31G), in four families: `t1·t2·t2·oovv` (172),
`t2·t3·oovv` (151), `t1·t1·t3·oovv` (44), `t1·t1·t1·t2·oovv` (24) — confirming the
`t2·t3·v` case `CCGEN_HIGHER_OPERATOR_REUSE.md` predicted.

**The factorized emit eliminates all of them.** Same census on a factorized TU
(`build-profile/`, builders named `build_W_<blocks>_<n>_ccsdt`):

| | nests | four-deep |
|---|---|---|
| undressed | 824 | **391** |
| factorized | **414** | **0** |

| case | undressed | factorized | saving |
|---|---|---|---|
| BH3/STO-3G | 4.93e+08 | 4.84e+07 | 10.2x |
| CH4/STO-3G | 1.48e+09 | 1.31e+08 | 11.2x |
| HF/6-31G | 1.09e+10 | 7.47e+08 | 14.5x |
| H2O/6-31G | 4.51e+10 | 2.64e+09 | 17.1x |
| BH3/6-31G | 7.30e+10 | 4.23e+09 | 17.2x |
| C2H4/STO-3G | 1.11e+11 | 6.22e+09 | 17.8x |

It moves the **exponents**, not the constant — `o^4.92 v^4.94` → `o^4.42 v^4.40`
against a hand-written `o^3.94 v^4.18` — consistent with the measured 3.12x/3.61x
wall-clock for `--dressing derived`, and the first evidence here that model and
measurement agree on *mechanism* rather than only direction.

**Census validation:** the undressed model fits `o^4.92 v^4.94` against the
ladder's measured `o^4.87 v^4.52` — **the `o` exponent agreeing to 0.05**. A pure
code count reproduces the measured scaling.

**What the model cannot do, stated rather than tuned away:** modelling the
hand-written side gives ratios of 5000–24000x against a measured 21.8–50.1x, two
orders of magnitude high and not flat (230x–660x). That kernel is not FLOP-bound
the way the model assumes. **Trust the generated-side model; never quote a
modelled ratio.** Cause 2 is why.

---

## Cause 2: one nest per term — the next lever

Even factorized, **414 nests remain against the hand-written kernel's one.**

The decisive number: those 414 nests have only **13 distinct inner-loop
signatures**, and **every one of the 414 shares its signature with at least one
other**.

| inner loop | nests sharing it |
|---|---|
| `l, m, e` | **81** |
| `l` | 54 |
| `l, e` | 45 |
| `d` | 42 |
| `m, d, e` | 42 |
| `m, d` | 36 |
| `l, d` | 33 |
| `l, m` | 21 |
| (5 more) | ≤20 each |

They are fusable in the literal sense — three of the 81 in the largest group,
verbatim from the emitted file:

```cpp
acc += 0.5  * W_oooovv_2(i,j,l,m,a,e) * amplitudes.t3(k,l,m,b,c,e);
acc += -0.5 * W_oooovv_2(i,j,l,m,b,e) * amplitudes.t3(k,l,m,a,c,e);
acc += 0.5  * W_oooovv_2(i,j,l,m,c,e) * amplitudes.t3(k,l,m,a,b,e);
```

Same operands, same loop, three separate `o³v³` traversals. **This is exactly the
hand-written kernel's shape**, which writes precisely this pattern as one nest:

```cpp
for i,j,k,a,b,c {
    double value = 0.0;
    for (int d = 0; d < n_virt; ++d) {
        value += ints.w_vvvo(a,b,d,j) * amps.t2(i,k,d,c);
        value += ints.w_vvvo(a,c,d,k) * amps.t2(i,j,d,b);
        ...  9 accumulations sharing this one loop ...
    }
    for (int l = 0; l < n_occ; ++l) { ... 9 more ... }
}
```

### What fusion is worth

**Fusion does not change FLOP count.** It changes how many times the `o³v³` result
tensor is traversed and its operands re-streamed:

- now: **414** traversals, one per nest
- fused: **13**, one per distinct inner signature
- **32x fewer passes over the result**

| case | `t3` size | traffic now | fused |
|---|---|---|---|
| BH3/STO-3G | 0.031 MiB | 12.9 MiB | **0.4 MiB** |
| CH4/STO-3G | 0.061 MiB | 25.3 MiB | **0.8 MiB** |
| H2O/6-31G | 0.488 MiB | 202.1 MiB | **6.3 MiB** |
| C2H4/STO-3G | 0.844 MiB | 349.3 MiB | **11.0 MiB** |

**This is a memory-traffic lever, not a FLOP lever** — which is exactly why the
FLOP model overpredicts the hand-written side by two orders of magnitude: that
kernel gets its nine accumulations per loop nearly free on operands already in
registers, while the generated form pays a full pass for each.

It also explains the shape of the measured residual.
`CCGEN_KERNEL_SCALING_SCOPE.md` records the generated fit's 21.4 % error as
**concentrated at high `v`** — exactly where `o³v³` traffic dominates and 414
passes hurt most.

### The caveat worth respecting

The earlier "loop fission is not a penalty" measurement (0.62x — the fissed form
was *faster*) was taken at `no=nv=4`, where the whole residual is 32 KB and
L1-resident. **At that size there is no traffic to save, so the result is real and
simply does not generalize.** The table above shows traffic reaching 349 MiB at
C2H4/STO-3G while `t3` itself stays under 1 MiB — the working set fits, the
*traffic* does not.

This is the untested half of H1 from the scaling scope, now with a concrete
mechanism and a number rather than a hypothesis.

---

## What to do, in order

1. **Measure the factorized kernel first, before any emitter work.**
   `--dressing derived` is wired and produces the 414-nest TU, and its 3.12x/3.61x
   is already measured end-to-end. Fusion must be measured against *that* baseline,
   not the undressed one — otherwise cause 1's saving gets counted twice.
2. **Fuse nests sharing an inner-loop signature.** 414 → ~13. The grouping is
   mechanical (the signature is already implicit in the emitted structure) and the
   target shape is the hand-written kernel's. Emitter-side:
   `planck_tensor_cpp.py:284,443` emit one nest per term.
3. **Do not consume `_optimal_contraction_order`** without re-checking — it targets
   the 391 terms cause 1 already eliminates.

## Ruled out

- **Loop-invariant work.** Zero of 824 nests have an accumulation ignoring any of
  `i,j,k,a,b,c`, so no nest can be hoisted out. The emitted FLOPs are real work.
- **Accessor overhead.** Fixed and gated (`CCGEN_TENSOR_ACCESSOR.md`, 206x on
  rank 3).
- **Loop fission per se.** 0.62x at `no=nv=4` — see the caveat above; a
  size-dependent result, not a refutation.

## Why this is code-level rather than measured

`CCGEN_KERNEL_SCALING_SCOPE.md` records the measurement route as closed: the
timing probe sits on a code path the rank-3 representation fix rerouted away from,
and the two arms have no residual-level agreement gate (distinct solvers, distinct
amplitude representations, both correct). Whole-iteration timing measures *"solver
iteration"*, not *"triples kernel"*.

This document is the answer that route was meant to produce, obtained by counting
emitted terms instead. It is sufficient to act on: the census is exact, the
generated-side model reproduces the measured `o` exponent, and both levers are
identified to specific term counts.

## Key locations

| what | where |
|---|---|
| undressed kernel, 824 nests | `build/generated/cc/ccsdt_planck_generated.cpp` |
| factorized kernel, 414 nests / 13 signatures | `build-profile/generated/cc/ccsdt_planck_generated.cpp` |
| hand-written kernel, ONE nest | `build_dressed_triples_residual`, `src/post_hf/cc/tensor_backend.cpp` |
| its intermediates | `build_dressed_triples_intermediates`, same file |
| one-nest-per-term emission (the fusion target) | `python/ccgen/emit/planck_tensor_cpp.py:284,443` |
| computed and discarded contraction order | `python/ccgen/tensor_ir.py:283` |
| the measured ladder this explains | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |
| the predicted `t2·t3·v` case, confirmed | `docs/CCGEN_HIGHER_OPERATOR_REUSE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
