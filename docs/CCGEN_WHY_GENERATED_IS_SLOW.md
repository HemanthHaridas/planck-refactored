# What makes the generated CC kernels slower than the hand-written ones?

**Four causes examined. Two confirmed and fixed (3.6x and 1.76x, compounding to
~6.4x), one built and refuted, one scoped and untouched.**

| # | cause | status | worth |
|---|---|---|---|
| 1 | **contraction order** — 391 of 824 terms evaluated n-arily at `o⁵v⁵` | **FIXED**, `--dressing derived` | modelled 10x–18x FLOPs; **measured 3.6x** |
| 2 | **one loop nest per term** — 806 nests vs the hand-written kernel's one | **BUILT** (`CCGEN_FUSE_LOOPS`), 806 → 15 | **~0 % runtime**; real compile-time/code-size win |
| 3 | **operators rebuilt once per chunk** — 1080 builder calls for 270 operators | **FIXED** (H5) | **1.76x**; **20.8x** fewer calls at rank 4 |
| 4 | **no OpenMP anywhere in CC** — 98.8 % CPU on 8 cores | **SCOPED** (H6) | modelled **3.86x** at 4 threads |

**Cause 3 is the one this document's own method missed.** It was found by a
`sample` profile in ~20 minutes, after two rounds of code-census-plus-FLOP-model
had produced one hit and one miss. Cause 4 was found by asking whether the code was
threaded at all. Neither needed a model.

> **The "generated vs hand-written" ratio in this document is NOT a like-for-like
> comparison, and an earlier revision wrongly treated it as one.** The two paths
> are *different solvers*, not two implementations of one algorithm:
>
> | | hand-written (`tensor`) | generated (`optimized`) |
> |---|---|---|
> | amplitude storage | **wedge-packed** (`i<=j<=k`), rebuilt via `restore` | **dense**, every index stored |
> | r1/r2 each iteration | cheap hand-written dressed intermediates | full generated kernels, every rank |
> | iterations on CH4 | **40** | **16** |
>
> `CCGEN_RANK3_KERNEL_AND_SOLVER.md` establishes the storage difference is a
> *coupled convention*, and the same reasoning that forbids comparing their
> residuals elementwise applies to their wall-clock: **a ratio between them prices
> two different algorithms, not two codegen strategies.** Quote it as "the
> generated production path costs Nx the hand-written one end to end" — a real and
> useful operational fact — never as "the generated kernel is Nx slower".
>
> Cause 1 and cause 2 are unaffected: both were measured **generated-vs-generated**,
> one flag apart, same solver.**

Established by code-level census of the emitted C++, a FLOP model, and end-to-end
timing — after the isolated-kernel measurement route was closed
(`CCGEN_KERNEL_SCALING_SCOPE.md`).

**Read this before acting on it:** the census-and-FLOP-model method produced one
correct prediction (cause 1) and one confidently wrong one (cause 2 — predicted
32x, delivered nothing). Its record here is 1 for 2.

---

## The measured cost of each path

`PLANCK_RCCSDT_BACKEND=optimized` (generated, arbitrary-order harness) against
`tensor` (hand-written), same input, same binary configuration apart from
dressing. Energies identical to all ten digits across all arms.

**These are two solvers' end-to-end costs, not a codegen ratio** — see the caveat
above. The undressed/dressed columns *are* like-for-like (one flag apart, same
solver); the `dressed/hand` column is an operational fact about which production
path is cheaper, not a measure of emitted-code quality.

| case | undressed | dressed | hand-written | dressed/hand |
|---|---|---|---|---|
| BH3/STO-3G | 33.70 s | 9.34 s | **0.10 s** | **93x** |
| CH4/STO-3G | 103.86 s | 28.67 s | **0.19 s** | **151x** |

**This is not the 21.8x–50.1x from `CCGEN_KERNEL_SCALING_SCOPE.md`** — that ladder
timed the *isolated triples residual*. End to end the gap is 337x–547x, matching
`CCGEN_ARBITRARY_HARNESS_COST_SCOPE.md`'s independently recorded ~500x. Different
quantities; do not quote them interchangeably.

---

## Cause 1: contraction order — confirmed

The undressed emitter gives every residual term its own full `o³v³` nest and
evaluates it n-arily. Census of the rank-3 triples residual: **824 nests**, by
inner summation:

| inner sum | terms | cost each |
|---|---|---|
| `o²v²` | **391** | **o⁵v⁵** |
| `o²v¹` | 144 | o⁵v⁴ |
| `o¹v²` | 140 | o⁴v⁵ |
| `o¹v¹` | 87 | o⁴v⁴ |
| others | 62 | ≤o⁵v³ |

The 391 four-deep terms are **83–90 % of generated FLOPs**, in four families:
`t1·t2·t2·oovv` (172), `t2·t3·oovv` (151), `t1·t1·t3·oovv` (44),
`t1·t1·t1·t2·oovv` (24) — confirming the `t2·t3·v` case
`CCGEN_HIGHER_OPERATOR_REUSE.md` predicted. Representative, verbatim:

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

**`--dressing derived` eliminates all 391** — 824 nests → 414, zero four-deep.
Modelled saving 10.2x (BH3/STO-3G) → 17.8x (C2H4/STO-3G), growing with size, and
it moves the exponents rather than the constant: `o^4.92 v^4.94` →
`o^4.42 v^4.40` against a hand-written `o^3.94 v^4.18`.

**Measured: 3.62x**, not the modelled 11.2x on CH4. Right sign, wrong magnitude —
roughly two-thirds of the predicted FLOP win does not materialise, which is direct
evidence **the generated path is not FLOP-bound.**

**Census validation:** the undressed model fits `o^4.92 v^4.94` against the
ladder's measured `o^4.87 v^4.52` — the `o` exponent agreeing to 0.05. A pure code
count reproduces the measured *scaling*, even where it misses the magnitude.

**What the model cannot do:** modelling the hand-written side gives ratios of
5000–24000x against a measured 21.8–50.1x, two orders of magnitude high and not
flat. That kernel is not FLOP-bound either. **Trust the generated-side model for
scaling; never quote a modelled ratio.**

---

## Cause 2: one nest per term — built, and worth nothing at runtime

Even factorized, **806 nests remain against the hand-written kernel's one**, and
they carry only **15 distinct `(free, summed)` loop signatures** — every nest
shares its signature with at least one other. They are fusable in the literal
sense; three from the largest group, verbatim:

```cpp
acc += 0.5  * W_oooovv_2(i,j,l,m,a,e) * amplitudes.t3(k,l,m,b,c,e);
acc += -0.5 * W_oooovv_2(i,j,l,m,b,e) * amplitudes.t3(k,l,m,a,c,e);
acc += 0.5  * W_oooovv_2(i,j,l,m,c,e) * amplitudes.t3(k,l,m,a,b,e);
```

Same operands, same loop, three separate `o³v³` traversals — exactly what the
hand-written kernel writes as one nest with ~9 accumulations per inner loop.

Fusion is implemented (`CCGEN_FUSE_LOOPS=N`, default 0) and reduces **806 → 15
nests (54x)**, halving the TU and dropping 845 KB of binary. **It does not change
runtime:**

| case | `t3` | 806 nests | 286 | 15 |
|---|---|---|---|---|
| BH3/STO-3G | 0.031 MiB | 9.52 s | 9.73 s | 9.54 s |
| CH4/STO-3G | 0.061 MiB | 29.59 s | 28.88 s | 28.60 s |
| HF/6-31G | 0.21 MiB | **154.00 s** | — | **154.54 s** |

0–3 %, inside noise, not monotonic. Energies bit-identical at every level.

### The traffic model, and why it was wrong

The prediction was that fusion saves memory traffic: 414 traversals of the `o³v³`
result → 13, i.e. **32x fewer passes**, with C2H4/STO-3G's 349 MiB → 11 MiB the
headline. It carried an explicit falsification criterion — *negligible at BH3
(L1-resident), material at larger `t3`*.

**HF/6-31G is 3.4x BH3's working set and shows +0.35 %.** The criterion is not met.

The model counted a full `o³v³` read+write per nest. That is not what the hardware
does: consecutive nests over the *same* `result` tensor hit the same cache lines,
and `t3` never leaves L2 at any reachable size anyway. **The model priced traffic
that was already being served from cache.** Fusion removes loop overhead and
instruction count, not stalls that were not occurring.

This also settles a question left open earlier: the "loop fission is not a
penalty" measurement (0.62x at `no=nv=4`) was read as size-limited and therefore
not generalisable. It generalises.

**What survives:** cause 2 is a real structural difference and fusion genuinely
closes it. What is refuted is that it is worth anything in runtime.

**What it is worth instead:** `generated_kernel_registry.cpp` is `-O1`-pinned
because these TUs are pathological to compile. Halving TU size is a real
compile-time and code-size win — that is the basis on which to keep or wire
fusion, not speed.

---

## Causes 3 and 4: what profiling found that modelling did not

**Cause 3 — operators rebuilt once per chunk (FIXED, 1.76x).** A `sample` profile
put **67.7 % of runtime in `build_W_*` builders**, not in the residual kernel at
all. `_emit_chunked_kernel` emitted every dressed operator inside *every* `_partN`
function, so the duplication factor equalled the part count:

| kernel | parts | distinct ops | builder calls | after H5 |
|---|---|---|---|---|
| `ccsdt` triples | 4 | 278 | 1112 | 278 |
| **`ccsdtq` quadruples** | **18** | **894** | **16 092** | **894** |

**The waste scaled with kernel size — worst at the production target.** The
emitter said the rebuilds were "cheap, local, and keeps each part self-contained";
true for an *undressed* emit where there are no operators, false once dressing
populated the list, and never re-measured.

Fixed by building each operator once into a `<kernel>_ops` struct passed by
`const&`. Measured on CH4: **29.59 s → 16.81 s (1.76x)**, `E_corr` **bitwise
identical**, rank-4 TU 12.8 → 10.5 MB. Full record:
`docs/CCGEN_ARBITRARY_HARNESS_COST_SCOPE.md` H5.

**Cause 4 — no OpenMP (SCOPED, modelled 3.86x).** There is **zero OpenMP anywhere
in CC** — not in `src/post_hf/cc/*.cpp`, not in the generated kernels, not in the
emitter. A CH4 solve with `OMP_NUM_THREADS=8` runs at **98.8 % CPU**: one core,
seven idle. Every other hot path in Planck is threaded. Amdahl on the post-H5 split
(builders 45.1 %, residual 53.7 %) gives **3.86x at 4 threads**, and both sites are
reduction-free — builders write private tensors, residual nests write disjoint
`result(i,...)` slices. Scoped as H6.

### What this says about method

| lever | how found | outcome |
|---|---|---|
| 1 contraction order | census + FLOP model | **hit** (3.6x) |
| 2 loop fusion | census + traffic model | **miss** (~0 %) |
| **3 chunk rebuilds** | **`sample` profile** | **hit** (1.76x) |
| **4 no OpenMP** | **asking whether it was threaded** | **modelled 3.86x** |

The two models cost days and went 1-for-2. The profile cost twenty minutes and
found a defect neither model could see, because **neither modelled work that
should not happen at all** — they both priced the arithmetic of the residual,
while two thirds of the time was redundant operator construction outside it.

**Profile before modelling.** That is the transferable result here.

## Ruled out

- **Loop-invariant work.** Zero of 824 nests have an accumulation ignoring any of
  `i,j,k,a,b,c`, so no nest can be hoisted. The emitted FLOPs are real work.
- **Accessor overhead.** Fixed and gated (`CCGEN_TENSOR_ACCESSOR.md`, 206x on
  rank 3).
- **Memory traffic from nest count.** Measured at three sizes spanning 7x in `t3`.

## What was built

| | |
|---|---|
| `CCGEN_FUSE_LOOPS=N` | fuse the N largest loop-signature groups; 0 (default) is byte-identical to the pre-fusion emit |
| `term_loop_signature`, `group_terms_by_loop_signature` | the `(free, summed)` grouping key |
| `emit_planck_fused_group` | one nest header, N accumulations into a shared `acc` |
| `test_loop_fusion_grouping.py` | pins the grouping against **the emitted text**, not against the helper's own logic |

Gates: `ch4_rccsdt_generated_sto3g` and `lih_rccsdt_generated_sto3g` pass at every
fusion level with bit-identical `E_corr` (`-0.0533629208` on BH3,
`-0.0791116825` on CH4, matching PySCF to 1.4e-08); the default emit is byte-identical on
both the dressed and undressed paths.

Three traps found while building it:

1. **The triples kernel does not use the obvious emit path.** It exceeds
   `_KERNEL_CHUNK_TERMS` and goes through `_emit_chunked_kernel`, which had its
   own term loop. The first wiring changed three small kernels and left triples
   untouched — fusion "applied" and buying nothing.
2. **Chunks are contiguous slices**, so groups must be reordered before chunking
   or a group straddles two `_partN` functions and silently un-fuses.
3. **Routing the chunked path through the shared helper broke byte-identity** (it
   never emitted `// Term N` or trailing blanks). Caught by re-running the
   grouping gate on the next change.

And one setup error, a repeat of a known class: the first fusion build was
configured **without `PLANCK_CC_DRESS_OPERATORS=ON`**, fusing the undressed
manifold — wrong baseline and a weaker gate. Flags were carried forward from a
previous command with the one defining the manifold dropped, exactly as in the
`SPIN_ADAPT=OFF` investigation. **`grep '^PLANCK_CC' <build>/CMakeCache.txt`
before trusting any number from a new tree.**

## On the measurement constraint

`CCGEN_KERNEL_SCALING_SCOPE.md` records the *isolated-kernel* measurement route as
closed: its timing probe sits on a code path the rank-3 representation fix rerouted
away from, and the two arms have no residual-level agreement gate (distinct
solvers, distinct amplitude representations, both correct).

**That constraint is narrower than it was read as.** It forbids comparing the
generated and hand-written *residuals*, and it makes their wall-clock ratio
non-comparable. It never prevented profiling the generated path **against itself**
— which is what found causes 3 and 4. An earlier revision of this document treated
the constraint as a reason the whole question had to be answered by modelling. It
was not.

## Key locations

| what | where |
|---|---|
| undressed kernel, 824 nests | `build/generated/cc/ccsdt_planck_generated.cpp` |
| factorized kernel | regenerate with `--dressing derived` |
| hand-written kernel, ONE nest | `build_dressed_triples_residual`, `src/post_hf/cc/tensor_backend.cpp` |
| fusion | `_emit_terms`, `emit_planck_fused_group`, `python/ccgen/emit/planck_tensor_cpp.py` |
| the chunked path that also needed wiring | `_emit_chunked_kernel`, same file |
| causes 3 and 4 (chunk rebuilds, OpenMP) in full | `docs/CCGEN_ARBITRARY_HARNESS_COST_SCOPE.md` |
| the isolated-kernel ladder | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
