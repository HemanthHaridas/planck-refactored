# Why do the generated CC kernels run slower than hand-written ones?

**Status: measured, and the dominant cause is fixed.** The answer is the tensor accessor, not the
loop structure.

The carried figure was "~180× slower, attributed to intermediates rebuilt inside loops and CSE
being disabled". Measured on a current build the gap was **~37.6×**, and both the number and the
attributed cause were wrong. The dominant cost was `Tensor*::operator()` — an out-of-line call
that **heap-allocated `std::vector<int>` per element access**. Loop fission, the thing this doc
originally hypothesized, is **not** a penalty at the measured size.

The fix landed (`docs/CCGEN_TENSOR_ACCESSOR_FIX_SCOPE.md`): all fixed-rank *and* runtime-rank
(`TensorND` / `DenseTensorView` / `ConstDenseTensorView`) hot accessors are now inlined in
`common.h`. Measured result, energies bitwise-identical throughout:

| | before | after | speedup |
|---|---|---|---|
| rank-3 generated T3 residual (`bh3`) | 6.40 s | **0.031 s** | 206× |
| rank-3 hand-written T3 residual | 0.170 s | **0.0014 s** | 121× |
| generated-vs-hand ratio | 37.6× | **22×** | — |
| **rank-4 CCSDTQ per iteration** (Be/STO-3G) | 38.5 s | **11.4 s** | 3.4× |
| `water_rccsdt_sto3g` regression case | 44.6 s | **0.39 s** | 114× |

**The runtime-rank accessors were the load-bearing half at rank 4.** An earlier pass fixed only
the fixed-rank `Tensor{2,4,6}D` accessors and left rank-4 CCSDTQ *completely unchanged* at ~38 s
per iteration, because the generated arbitrary-order (rank ≥ 4) kernels index exclusively through
braced lists on `TensorND` / `DenseTensorView` — **23,338 such accesses per residual evaluation**,
each one allocating a `std::vector<int>` via `to_vector` *before* the out-of-line call, i.e.
strictly worse than the fixed-rank path. Excluding them as "not on the hot path" was a reading of
signatures, not a measurement, and it was wrong.

## P1 — the number

`bh3` RCCSDT / STO-3G, `no=4 nv=4`, Release `-O3`, `OMP_NUM_THREADS=1`, clang (Darwin 25.5.0).
Both residuals evaluated from identical amplitudes in one pass via the existing
`PLANCK_CC_T3_DIFF=1` probe, six consecutive iterations:

| | generated | hand-written | + its intermediates | ratio |
|---|---|---|---|---|
| T3 residual | **6.40 s** | **0.170 s** | 0.014 s | **37.6× ± 0.4** |

Stable to ±0.4× across iterations. **The previously carried ~180× does not reproduce** — it has
no recorded `o`/`v`/compiler/flags anywhere in the tree, so it is not clear what it measured; treat
37.6× at this size as the number, and re-measure rather than cite 180×.

*Ruled out as confounds:* neither kernel is threaded (0 OpenMP pragmas in either) and neither
reaches BLAS/Eigen, so parallelism and vendor-BLAS are not variables. The `-O1` compile pinned on
`generated_kernel_registry.cpp` (`CMakeLists.txt:402`) does **not** apply here — the rank-3 triples
TU is `#include`d into `tensor_backend.cpp:19`, so **both kernels are `-O3`**. The optimization
asymmetry is real for rank 4+, but it does not confound this measurement.

## P2 — which mechanism

Two microbenchmarks at the real kernel's size (`no=nv=4`, 1063 sweeps), isolating one variable each.

**The accessor (dominant, 13.5×).** Replicating `common.cpp`'s implementation exactly —
out-of-line call, two `std::vector<int>` constructed per access, `std::expected` returned — against
a plain inlined flat index:

| accessor | time |
|---|---|
| checked + vector-allocating (as shipped) | 0.0446 s |
| inlined flat index | 0.0033 s |
| | **13.5×** |

**Loop fission (not a penalty, 0.62×).** Same inlined accessor for both, differing *only* in 1063
separate `o³v³` sweeps vs one fused sweep:

| shape | time |
|---|---|
| fissed (1063 sweeps, generated shape) | 0.0018 s |
| fused (1 sweep, hand-written shape) | 0.0029 s |
| | **0.62× — fissed is FASTER** |

13.5× (accessor) × ~2.8× (residual structural factor) ≈ the observed 37.6×.

### H1 was wrong, and why

This doc originally hypothesized the kernel was **memory-bound by loop fission** — 1063 sweeps
each streaming the residual from RAM. That is falsified: at `no=nv=4` the residual is 4096 doubles
= **32 KB, fully L1-resident**, so there is no RAM traffic to save, and the fissed form actually
vectorizes better. The hypothesis reasoned about traffic without checking the working-set size
against cache. H1 may still hold at production `o`/`v` where the working set leaves cache — that
is exactly what P3 tests — but it is not what makes `bh3` slow.

### The real mechanism

Both kernels called the same expensive accessor; the generated one just called it far more often.

| | generated triples | hand-written triples |
|---|---|---|
| tensor accessor call sites (rank 3) | **3416** | 186 |
| braced-index accesses (rank 4 quadruples) | **23338** | — |

`Tensor6D::operator()` was declared in `common.h` and defined in `common.cpp`, with **no LTO or
IPO configured anywhere** (`CMakeCache.txt` has no `IPO`/`flto`), making it an unavoidable cross-TU
call. Each invocation:

1. called out-of-line (could not inline),
2. constructed `std::vector<int>{dim1..dim6}` and `std::vector<int>{i..n}` — **two heap allocations**,
3. built a `std::expected<std::size_t, std::string>`,
4. then indexed.

Two `malloc`/`free` pairs per tensor element read, on the innermost loop of an `o³v³` kernel.

The runtime-rank path was worse still: `TensorND::operator()(initializer_list)` forwarded through
`to_vector(indices)` — copying the list into a fresh `std::vector<int>` — *before* the out-of-line
`flatten_index` call. That is the path the rank ≥ 4 generated kernels use exclusively, which is why
fixing only the fixed-rank accessors moved rank 3 by 76× and rank 4 by nothing at all.

## What to do, in ladder order

1. ~~**Make the accessor cheap.**~~ **Done** — inlined in `common.h`, fixed-rank *and* runtime-rank.
   One mechanism covering every rank and both kernel families at once. See
   `docs/CCGEN_TENSOR_ACCESSOR_FIX_SCOPE.md` for the invariants it had to preserve.
2. **Re-measure the ratio at production `o`/`v` (P3).** ← now the next step. With the accessor
   fixed, the remaining 22× is whatever is genuinely structural. H1 may reappear here.
3. **Only then consider fusing / consuming the IR hints.** `tensor_ir.py` defines `BLASHint`
   (`:66`), `_detect_gemm` (`:198`), and `_optimal_contraction_order` (`:283`), and
   `grep BLASHint python/ccgen/emit/planck_tensor_cpp.py` **returns nothing** — the emitter
   discards all of it. Real, but it was not the bottleneck, and the measurement above says fusion
   alone would buy nothing at small size.

Fix the emitter, not the emitted files — the generated TUs are build artifacts, and a patch to one
rank's output re-arms the defect at every other rank.

## Still open

- **P3 — ratio vs system size.** Scoped in `docs/CCGEN_KERNEL_SCALING_SCOPE.md`. All ratio numbers
  above come from one point (`bh3`/STO-3G, `nocc=8 nvirt=8`, square), which cannot distinguish a
  constant 22× tax from a scaling defect — and H1 and H3 make opposite predictions there.
- **Rank 4 is still subject to the `-O1` registry pin** (`CMakeLists.txt:402`), which the rank-3
  path is not. Now that the accessor no longer dominates, that asymmetry is worth re-checking —
  the pin exists because a ~230k-line TU is super-linear to optimize at `-O3`, and the standing
  follow-on is to chunk the giant residual kernels in the emit so any level stays cheap.

## Reproducing

```bash
cmake -B build-profile -DCMAKE_BUILD_TYPE=Release -DPLANCK_CC_MAXORDER=3
make -C build-profile -j4 hartree-fock
BASIS_PATH=$PWD/basis-sets ./build-profile/hartree-fock \
  tests/inputs/regression/post_hf/bh3_rccsdt_sto3g.hfinp
```

Rank 4 (the production target) needs `-DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_SPIN_ADAPT=ON` and
`tests/inputs/regression/post_hf/be_rccsdtq_sto3g.hfinp`; read the per-iteration `t=` on the
`RCCSDTQ[TENSOR]` lines. Build with `make -j4` — the generated TUs are large enough that a
full-width build is disruptive.

`BASIS_PATH` is required if the build's compiled-in basis path points at a stale install prefix.

The P1/P2 numbers were taken with temporary timing instrumentation hung off the pre-existing
`PLANCK_CC_T3_DIFF=1` probe (which already evaluates the generated and hand-written residuals once
each from identical amplitudes). That instrumentation has been removed; re-add it the same way if
the comparison is needed again. The two microbenchmarks are throwaway and described inline in P2.

**Do not benchmark against a stale build tree.** Both misreads during this investigation came from
comparing binaries built from different source states or different `CMAKE_BUILD_TYPE`s — an
`ethylene_rhf_stability_unstable` failure initially attributed to the accessor change turned out to
reproduce with the change reverted, and a reported CCSDTQ non-convergence turned out to be a build
with `spin_adapt` *and* intermediates on (the known 0.25 defect; see
[[ccgen_spin_adapt_no_intermediates]]). A/B in one configure, rebuilding both arms.

## Key code locations

| what | where |
|---|---|
| the accessors (now inlined) | `src/post_hf/cc/common.h` (`detail::fixed_rank_index_valid`, `nd_flat_index`) |
| layout gate | `tests/cc_tensor_index.cpp` (`planck-cc-tensor-index`) |
| hand-written triples (1-nest reference) | `src/post_hf/cc/tensor_backend.cpp:1800` |
| generated-vs-hand branch (+ `T3_DIFF` probe) | `src/post_hf/cc/tensor_backend.cpp:2324` |
| rank-3 triples TU (`-O3`, via `#include`) | `src/post_hf/cc/tensor_backend.cpp:19` |
| `-O1` pin (rank 4+ registry only) | `CMakeLists.txt:402` |
| one-nest-per-term emission | `python/ccgen/emit/planck_tensor_cpp.py:284`, `:443` |
| unused GEMM / contraction-order analysis | `python/ccgen/tensor_ir.py:66,198,261,283` |

## Related, deliberately separate

- `CCGEN_HIGHER_OPERATOR_REUSE.md` — factorizing contractions to cut FLOP *scaling*. Changes the
  asymptotics; this doc's finding changes a constant factor.
- `CCGEN_INTERMEDIATE_MEMORY_LOCALITY_SCOPE.md` — locality of *materialized intermediates*. Landed,
  and orthogonal: the default triples emit materializes none, which is also why the old
  "intermediates rebuilt inside loops" attribution could not have been right.
