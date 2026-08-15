# Making the CC tensor accessor cheap

**LANDED.** Scope and record for the fix identified in `CCGEN_KERNEL_PERFORMANCE_SCOPE.md`. That
doc measured the generated-vs-hand-written CC gap at ~37.6× and found the dominant cause is not
loop structure but the tensor accessors themselves: **13.5×**, measured in isolation.

Result, energies bitwise-identical throughout:

| | before | after | speedup |
|---|---|---|---|
| rank-3 generated T3 residual (`bh3`) | 6.40 s | 0.031 s | 206× |
| rank-3 hand-written T3 residual | 0.170 s | 0.0014 s | 121× |
| generated-vs-hand ratio | 37.6× | 22× | — |
| rank-4 CCSDTQ per iteration (Be/STO-3G) | 38.5 s | 11.4 s | 3.4× |
| `water_rccsdt_sto3g` regression case | 44.6 s | 0.39 s | 114× |

This doc's original draft predicted a post-fix ratio of ~2.8×; the actual is 22×. The prediction
assumed the accessor cost was proportionally equal on both sides. It is not — the hand-written
kernel is *more* accessor-dominated (fewer FLOPs per access), so it gained more and the ratio
briefly *widened* to 59× after the fixed-rank-only pass, before the runtime-rank half brought it to
22×. The absolute win is large on both paths; the residual 22× is genuinely structural and is what
P3 should measure.

## What was wrong

`Tensor6D::operator()` is declared in `common.h:68` and defined in `common.cpp:271`. There is **no
LTO or IPO configured** (no `IPO`/`flto` in `CMakeCache.txt` or `CMakeLists.txt`), so it is an
unavoidable cross-TU call. Per element access it:

1. calls out-of-line (cannot inline),
2. constructs `std::vector<int>{dim1..dim6}` and `std::vector<int>{i..n}` — **two heap allocations**,
3. builds a `std::expected<std::size_t, std::string>` via `checked_fixed_rank_index`,
4. then indexes.

Two `malloc`/`free` pairs per tensor element read, on the innermost loop of an `o³v³` kernel. The
generated rank-3 triples has **3416** accessor call sites against the hand-written kernel's 186 —
same accessor, far more calls, which is the entire measured gap.

Measured cost, replicating the shipped implementation exactly at the real kernel's size
(`no=nv=4`, 1063 sweeps):

| accessor | time |
|---|---|
| checked + vector-allocating (as shipped) | 0.0446 s |
| inlined flat index | 0.0033 s |
| | **13.5×** |

## The fix

Move the fixed-rank `operator()` bodies into `common.h` as inline flat-index computations, with no
allocation on the hot path. Shape:

```cpp
inline double &Tensor6D::operator()(int i,int j,int k,int l,int m,int n) noexcept {
    assert(in_bounds(...) && "Tensor6D index out of bounds");   // NDEBUG-only
    return data[(((((std::size_t)i*dim2 + j)*dim3 + k)*dim4 + l)*dim5 + m)*dim6 + n];
}
```

Applies to `Tensor2D`, `Tensor4D`, and `Tensor6D` — const and non-const, six pairs. The index
arithmetic is already row-major in `flatten_index`; the flat form must reproduce it exactly.

### The runtime-rank accessors are also hot — this was the load-bearing half at rank 4

**An earlier draft of this scope excluded them, and that was wrong.** It read the signatures
(`std::vector<int>` / `initializer_list`), concluded they "cannot go allocation-free without an
interface change," and asserted they were "not on the hot path." The second claim was never
measured. Fixing only the fixed-rank accessors sped rank 3 by 76× and left **rank-4 CCSDTQ
completely unchanged at ~38 s per iteration**.

The rank ≥ 4 generated kernels index *exclusively* through braced lists on `TensorND` /
`DenseTensorView` — **23,338 accesses per residual evaluation** — and
`operator()(initializer_list)` forwarded through `to_vector(indices)`, copying the list into a
fresh `std::vector<int>` *before* the out-of-line `flatten_index` call. Strictly worse than the
fixed-rank path it was excluded in favour of.

The interface-change objection was also unfounded: inlining the `initializer_list` overloads to
index directly needs no signature change and no call-site change. The `vector<int>` overloads stay
out-of-line for the handful of non-hot callers that already hold a vector — which conveniently
makes them an independent implementation to cross-check against in the gate.

So the fix covers `TensorND`, `DenseTensorView`, and `ConstDenseTensorView` braced-index accessors
as well, via `detail::nd_flat_index` / `detail::nd_index_valid`.

**Generalizable lesson:** rank 3 is not a proxy for rank 4. They use different tensor types and
different code paths, and rank 4 is the production target. Measure the target, not the convenient
small case.

Two other levers from the performance doc are explicitly **not** in scope: emitter loop fusion
(measured at 0.62×, i.e. no gain) and consuming the unused `tensor_ir.py` BLAS hints (real, but
a separate question and not the bottleneck). Re-measure after this lands before touching either.

## What the bounds check currently catches, and what replaces it

The check is **not** fully redundant with the constructors, and this is the part to get right.

`checked_fixed_rank_index` enforces two things: per-index `0 <= idx < dim`, and
`offset < data.size()` ("tensor storage is smaller than the declared dimensions"). The
constructors (`common.cpp`, e.g. `Tensor6D::Tensor6D`) do validate `data.size() == product(dims)`
at construction — but on failure they **zero the dims and leave `data` populated**, and more
importantly `data` is a **public member that call sites assign directly** after construction:

```
src/post_hf/cc/tensor_backend.cpp:197   out.t1.data = src.t1.data;
src/post_hf/cc/tensor_backend.cpp:198   out.t2.data = src.t2.data;
```

(also `common.cpp:452–508`, and the `raw_generated.data = ...` / `restored_*.data = ...` pattern in
the `T3_DIFF` probe). So the size invariant is breakable post-construction by design, and the
accessor's storage check is the only thing that currently catches it.

Decision for the fix: keep **both** conditions in the debug assert — index range *and*
`offset < data.size()`. Debug builds keep exactly today's detection; release builds trade it for
speed, which is the point. Do not silently drop the storage half on the theory that the ctor covers
it; it does not.

**Behavior change, stated plainly:** today an out-of-bounds access in a release build returns a
reference to a shared `tensor_error_slot()` and continues; after the fix it is UB. This is a real
narrowing of safety and should be a deliberate choice, not a side effect. Two things make it
acceptable: `tensor_error_slot` is file-local to `common.cpp` with **no consumers anywhere in
`src/` or `tests/`**, so no caller can observe or depend on the fallback; and reaching it already
fires `assert(false)` in debug, i.e. it is already treated as a bug, not a supported path.

## Gates

- **Bitwise-identical energies.** This changes only how an index is computed, not the arithmetic or
  its order, so every CC energy must be unchanged to the last digit — a stronger gate than the
  usual tolerance-based one, and it should be asserted as such. The existing CC regressions
  (`bh3_rccsdt_sto3g`, `bh3_rccsd_sto3g`, the Be CCSDTQ/FCI case) carry this.
- **One new unit assertion** that the inline index equals `flatten_index`'s result across the
  dimension shapes actually used (non-square, and each rank), so the row-major reproduction is
  pinned rather than eyeballed. Non-square is load-bearing: `bh3`/STO-3G is `no == nv == 4`, which
  the rank-3 defect work already flagged as able to let a wrongly-ordered read stay in bounds and
  fail silently.
- **Re-measure the 37.6×** and report the residual gap. Prediction was ~2.8×; **actual 22×** — see
  the note at the top on why the prediction was wrong.
- **The hand-written kernels must speed up too** (186 call sites of the same accessor). If they
  do not, the accessor was not actually on their hot path and the model is wrong.
- **Rank 4 must move.** Added after the fixed-rank-only pass left it at 38 s/iteration. Rank 3
  passing all its gates said nothing about rank 4.

Full-suite regression run required, not just the CC subset — `common.h` is included well beyond
the CC module.

### Outcome

| gate | result |
|---|---|
| bitwise-identical energies | **pass** — `bh3` `E_corr` and `T3-DIFF` match to the last digit; rank-4 CCSDTQ `-0.0517746458` in 12 iterations, unchanged |
| layout unit gate (`planck-cc-tensor-index`) | **pass** — non-square shapes each rank, permuted-dims `swap_mid_axes` pattern, ND braced-vs-`vector` overload agreement, view-vs-owner |
| 6 CC regression cases | **pass** (incl. `be_rccsdtq_sto3g`, the generated-kernel/FCI gate) |
| CC unit tests | **pass** (3/3) |
| hand-written kernels sped up | **pass** — 0.170 s → 0.0014 s |
| rank 4 moved | **pass** — 38.5 s → 11.4 s per iteration |
| unit suite | 39/41; the 2 failures are pre-existing, confirmed by rebuilding the same tree with the change reverted |

`ethylene_rhf_stability_unstable` fails on the current tree **with or without** this change — an
RHF→UHF triplet-instability follow whose trajectory is sensitive to build configuration. It was
briefly misattributed to this work by comparing against a stale binary from a different source
state and a different `CMAKE_BUILD_TYPE`; the controlled A/B (one configure, both arms rebuilt)
clears it.

## Why this ordering

This is one mechanism in one header, fixing every rank and both kernel families at once — as
against the emitter work, which changes generated output per rank. It also has to land *first*:
until the accessor is cheap, any measurement of structural cost is swamped by allocation noise, so
P3 (ratio vs system size) cannot be read meaningfully before it.

## Key code locations

| what | where |
|---|---|
| inlined fixed-rank accessors + `detail::fixed_rank_index_valid` | `src/post_hf/cc/common.h` |
| inlined ND braced-index accessors + `detail::nd_flat_index` / `nd_index_valid` | `src/post_hf/cc/common.h` |
| remaining out-of-line `vector<int>` overloads (non-hot, and the gate's cross-check) | `src/post_hf/cc/common.cpp` |
| layout gate | `tests/cc_tensor_index.cpp` (`planck-cc-tensor-index`) |
| ctor size validation (not sufficient alone) | `src/post_hf/cc/common.cpp` (`Tensor6D::Tensor6D`) |
| direct `.data` assignment breaking the invariant | `src/post_hf/cc/tensor_backend.cpp:197,198` |
| rank-4 consumers of the ND path | `generated_arbitrary_runtime.cpp` (`to_tensor_nd`), `generated_arbitrary_prepare.cpp` (`swap_mid_axes`) |
| measurement record | `docs/CCGEN_KERNEL_PERFORMANCE_SCOPE.md` |
