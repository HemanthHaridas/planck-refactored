# What makes the generated CC kernels slower than the hand-written ones?

**One confirmed lever, one refuted, and ~100x still unexplained.**

| # | cause | status | size |
|---|---|---|---|
| 1 | **contraction order** — 391 of 824 terms evaluated n-arily at `o⁵v⁵` | **FIXED** by `--dressing derived` | modelled 10x–18x FLOPs; **measured 3.6x** wall-clock (F1) |
| 2 | **one loop nest per term** — 806 nests, 15 distinct inner loops | **DONE, and worth ~0 % at runtime** | 54x fewer traversals; **no measurable speedup at 3 sizes** |
| 3 | whatever accounts for the remaining **~100x** | **UNIDENTIFIED** | — |

**F1 measured the end-to-end gap at 337x–547x, not the 21.8x–50.1x this document
first cited** — that figure is from the isolated-triples-residual ladder, a
different quantity. Dressing closes 3.6x of it, leaving **93x–151x**. And cause
1's modelled 11.2x FLOP saving on CH4 realises as only 3.62x wall-clock, so **the
generated path is not FLOP-bound**. That reasoning pointed at a traffic lever
(cause 2) — which was then built, and **measured at ~0 %** (F4). Not FLOP-bound
and not traffic-bound at these sizes: the remaining ~100x is unidentified, and
this document no longer claims to explain it.

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

## The work, in small verifiable steps

Ordered so the cheapest step can kill the expensive ones. **F1 is a measurement
with no code change and it can invalidate everything after it.**

### F0 — what makes this tractable, established before scoping

Three properties of the emitted code, each checked rather than assumed:

| property | measured | why it matters |
|---|---|---|
| free-index order | **1 distinct order** across all 414 nests: `(i,j,k,a,b,c)` | grouping needs no reordering; the key is just `(free, summed)` |
| distinct `(free, summed)` groups | **13** | that is the fusion floor, directly |
| operands in the largest group | 81 terms read **5** distinct operands | no new plumbing; everything is already in scope per nest |

The emit seam is a plain `for term in emitted_terms: emit_planck_term(...)` loop
(`planck_tensor_cpp.py:980-985`). `emit_planck_term` emits free loops, summed
loops, then one accumulation — so fusion is *"group before the loop, emit the
shared header once"*, not a rewrite.

### F1 — **DONE (2026-08-29). Fusion is NOT killed; the gap is far larger than modelled.**

Two trees, cache-diffed to one differing flag, `ARBITRARY_LOWER_RANKS=ON`,
`PLANCK_RCCSDT_BACKEND=optimized` (routing confirmed as `RCCSDT[OPT]` through the
arbitrary harness on both):

| case | undressed | dressed | hand-written | dressed/hand |
|---|---|---|---|---|
| BH3/STO-3G | 33.70 s | 9.34 s | **0.10 s** | **93x** |
| CH4/STO-3G | 103.86 s | 28.67 s | **0.19 s** | **151x** |

**Energies identical to all ten digits across all three arms** (`-0.0533629208`,
`-0.0791116825`), so dressing is a pure rewrite and the arms are the same
calculation. CH4's 3.62x independently reproduces W5's recorded 3.61x from fresh
trees.

**Verdict: F2-F5 proceed.** The factorized kernel is nowhere near hand-written
time — dressing closes 3.6x of a **337x-547x** gap, leaving **93x-151x**.

**Two corrections this forces, and they matter more than the verdict:**

1. **The gap is ~10x larger than this document assumed.** The header cited
   21.8x-50.1x from `CCGEN_KERNEL_SCALING_SCOPE.md`, but that ladder timed the
   *isolated triples residual*. End to end the generated path is **337x-547x**
   slower, consistent with `CCGEN_ARBITRARY_HARNESS_COST_SCOPE.md`'s recorded
   ~500x. The two are different quantities and this document previously conflated
   them.
2. **Cause 1's saving does not translate.** The census predicted **11.2x** FLOPs
   on CH4; measured wall-clock is **3.62x**. So roughly two-thirds of the modelled
   FLOP win is not realised — which is itself evidence that the generated path is
   **not FLOP-bound**, and therefore that a traffic lever (cause 2) is the right
   next target rather than more FLOP reduction.

**Baseline for F5: the DRESSED column** (9.34 s / 28.67 s), never the undressed
one.

**Scope note.** Only 2 of 6 ladder points were run. The remaining four were
estimated at **~6 h** of wall-clock (`hf_631g` alone is ~13 min undressed) and
abandoned deliberately: F1's question is "is cause 2 worth fixing", a 93x-151x
residual gap answers it at any size, and the four points would refine an exponent
F1 does not need. They belong in F5, measured against the dressed baseline, where
the size trend is the actual deliverable.

### F2 — **DONE (2026-08-29). Byte-identical, and the fusion ratio is larger than stated.**

`term_loop_signature` / `group_terms_by_loop_signature` land beside
`emit_planck_term`, with `python/ccgen/tests/test_loop_fusion_grouping.py`.
Regenerating `ccsdt` with and without `--dressing derived` gives TUs
**byte-identical** to the F1 trees on all four files; the SHA-pin suite is green.

The load-bearing test is not the census but
`test_signature_equals_the_loops_emit_planck_term_writes`, which re-derives the
nests **by regex over the emitted text** and asserts the helper matches for every
term. The grouping is only meaningful if it describes what is actually emitted.

**A fixture correction the gate caught immediately.** The first version asserted
this document's 13 signatures / largest 81 and **failed at 8**. That census was
measured on the **factorized** TU; the undressed manifold is a different term set
(factorization splits contractions and introduces `W_*` operands). Measured:

| manifold | terms | signatures | largest | fusion ratio |
|---|---|---|---|---|
| undressed | 399 | **8** | 153 | **~50x** |
| factorized | 414 | 13 | 81 | ~32x |

**The undressed manifold fuses even more tightly**, so the 32x quoted above is the
conservative end. Asserting one fixture's numbers against another's is the same
class of mistake as the antisymmetrized-fixture traps recorded across ccgen; the
test now names its fixture and why.

### F3 — **DONE (2026-08-29). Fused, and energies are bit-identical.**

`emit_planck_fused_group` emits one nest header with N accumulations into a shared
`acc`; `_term_coeff_product` is shared with the per-term emitter so a term
contributes **identical text** either way, which makes fusion a pure regrouping
rather than a re-derivation. Switch: `CCGEN_FUSE_LOOPS=N`, default 0.

On the **dressed** manifold (the F1 baseline):

| | triples nests | fused groups | binary |
|---|---|---|---|
| nofuse | 806 | 0 | 7 391 056 B |
| `FUSE=1` | **286** | 4 | 6 945 232 B |

**Gate green.** `ch4_rccsdt_generated_sto3g` and `lih_rccsdt_generated_sto3g` pass
on both arms, and `E_corr` matches to all ten printed digits
(`-0.0533629208`, `-0.0791116825`). The default emit stays **byte-identical** on
both the dressed and undressed paths.

**Three things this step surfaced, each a real defect in the plan:**

1. **The triples kernel does not use the emit path F2 wired.** It exceeds
   `_KERNEL_CHUNK_TERMS` and goes through `_emit_chunked_kernel`, which had its own
   term loop. The first wiring changed three small kernels and left triples at 824
   nests — fusion "applied" and buying nothing.
2. **Chunks are contiguous slices, so groups must be reordered before chunking**,
   or a group straddles two `_partN` functions and silently un-fuses. Terms are now
   sorted by group before the split. This changes floating-point accumulation
   order, which is exactly what the gate exists to catch.
3. **Routing the chunked path through the shared helper broke F2's byte-identity**
   (that path never emitted `// Term N` or trailing blanks). Caught by re-running
   F2's gate on the very next change, fixed with an `annotate` flag.

**And a setup error worth recording, because it is a repeat.** The first F3 build
was configured **without `PLANCK_CC_DRESS_OPERATORS=ON`**, so it fused the
undressed manifold — wrong baseline (F1 says fusion must be measured against the
dressed kernel or cause 1's 3.6x is double-counted) and a weaker gate. The flags
were carried forward from the F1 command with the one that defines the manifold
dropped. Same class as the `SPIN_ADAPT=OFF` investigation: **a build flag silently
deciding what is under test.** Run `grep '^PLANCK_CC' <build>/CMakeCache.txt`
before trusting any number from a new tree.

The correct manifold also fuses far better — 2.8x (806->286) against 1.4x
(824->591) undressed.

### F4 — **DONE (2026-08-29). Fusion works and buys NOTHING. The traffic model is wrong.**

`CCGEN_FUSE_LOOPS=99` fuses every group. On the dressed manifold, triples:
**806 -> 15 nests (54x)**, TU 22 607 -> 11 971 lines, binary 7.39 -> 6.55 MB.
Both named gates pass; `E_corr` bit-identical at every fusion level.

**And the timings do not move.** Best-of-2, same binary configuration apart from
`CCGEN_FUSE_LOOPS`:

| case | `t3` | nofuse (806) | F3 (286) | F4 (15) |
|---|---|---|---|---|
| BH3/STO-3G | 0.031 MiB | 9.52 s | 9.73 s | 9.54 s |
| CH4/STO-3G | 0.061 MiB | 29.59 s | 28.88 s | 28.60 s |
| HF/6-31G | 0.21 MiB | **154.00 s** | — | **154.54 s** |

Three sizes, a 54x reduction in traversals, **0-3 % change — inside noise, and
not monotonic.**

**The traffic model predicted this wrong, and the prediction was specific.** It
said the saving should be negligible at BH3 (L1-resident) but material at larger
`t3`, with C2H4/STO-3G's 349 MiB -> 11 MiB the headline. HF/6-31G is 3.4x BH3's
working set and shows **+0.35 %**. The model's own falsification criterion —
"if the small case shows nothing and the large ones do, the mechanism is
confirmed" — is not met.

**Why the model was wrong.** It counted a full `o³v³` read+write per nest. That is
not what the hardware does: consecutive nests over the *same* `result` tensor hit
the same cache lines, and at these sizes `t3` never leaves L2 in the first place,
so the "traffic" the model priced was already being served from cache. Fusing
removes loop overhead and instruction count, not memory stalls that were not
occurring.

**This does not refute cause 2 as a description of the code** — 806 nests against
the hand-written kernel's one is still the structural difference, and fusion still
closes it. What is refuted is the claim that the difference is *worth 32x* by way
of memory traffic. It is worth ~0 % at every size reachable here.

**What F4 does buy, and it is not nothing:** TU size halves and the binary drops
445-845 KB. `generated_kernel_registry.cpp` is `-O1`-pinned precisely because
these TUs are pathological to compile, so fusion is a **compile-time and
code-size** lever. That is worth having; it is not the runtime lever this document
claimed.

### F5 — **CANCELLED.** F4 already answered it, on three sizes rather than six.

F5 was to measure fusion against F1's dressed baseline across the ladder and check
whether the saving grows with size. F4 measured it at three sizes spanning a 7x
range in `t3` and found no saving to grow. Running the remaining points (~6 h,
see F1) would refine an exponent on an effect that is not there.

**The residual 93x-151x gap is therefore NOT explained by this document.** Cause 1
is real and measured (3.6x). Cause 2 is real as a code-structure difference but
worth ~0 % in runtime. Whatever accounts for the remaining ~100x is unidentified,
and the honest state is that the census-plus-FLOP-model approach has now produced
one confirmed lever and one refuted one.

### Not in scope

- **Do not consume `_optimal_contraction_order`.** It targets the 391 terms cause 1
  already eliminates. Re-check only if F1 shows the factorized kernel is still
  FLOP-bound.
- **Do not reorder loops for stride.** A separate lever with its own risks; fusion
  changes traversal count, not layout.
- **Do not touch the undressed emit path.** It is the byte-identical default and
  cause 1's fix is opt-in; fusion should ride the same flag.

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
