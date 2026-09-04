# CC Tensor Accessor Performance

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Why were the CC tensor accessors the dominant cost in every CC kernel, and what replaced them?**

## Short answer

Every CC kernel reads tensor elements through `operator()`. Those accessors were defined out-of-line in `common.cpp` with no LTO configured, so each element access was a cross-TU call that heap-allocated one or two `std::vector<int>` and built a `std::expected` before indexing — two `malloc`/`free` pairs per element read, on the innermost loop of an `o³v³` kernel. Inlining them as flat row-major index computations in `common.h` removed the allocator from the hot path entirely, with energies bitwise identical throughout.

| | before | after | speedup |
|---|---|---|---|
| rank-3 generated T3 residual (`bh3`) | 6.40 s | 0.031 s | **206×** |
| rank-3 hand-written T3 residual | 0.170 s | 0.0014 s | **121×** |
| rank-4 CCSDTQ per iteration (Be/STO-3G) | 38.5 s | 11.4 s | 3.4× |
| `water_rccsdt_sto3g` regression | 44.6 s | 0.39 s | 114× |

Measured in isolation at the real kernel's size (`no=nv=4`, 1063 sweeps): checked+allocating 0.0446 s vs inlined flat index 0.0033 s — 13.5×.

## Where the logic lives

- `src/post_hf/cc/common.h` — inlined fixed-rank accessors + `detail::fixed_rank_index_valid`; inlined ND braced-index accessors + `detail::nd_flat_index` / `nd_index_valid`
- `src/post_hf/cc/common.cpp` — remaining out-of-line `vector<int>` overloads (non-hot; the gate's cross-check)
- `src/post_hf/cc/tensor_backend.cpp:197,198` — direct `.data` assignment that breaks the size invariant
- `tests/cc_tensor_index.cpp` — the layout gate, `planck-cc-tensor-index`

## What invariants matter

### 1. Fixing only the fixed-rank accessors does not cover rank ≥ 4

Rank 3 is not a proxy for rank 4. Fixing only the fixed-rank accessors sped rank 3 by 76× and left rank-4 CCSDTQ completely unchanged at ~38 s per iteration. The rank ≥ 4 generated kernels index *exclusively* through braced lists on `TensorND` / `DenseTensorView` — 23,338 accesses per residual evaluation — and `operator()(initializer_list)` forwarded through `to_vector(indices)`, copying into a fresh vector *before* the out-of-line `flatten_index` call. Strictly worse than the fixed-rank path it was excluded in favour of.

That exclusion rested on two claims, both wrong: that the `vector<int>`/`initializer_list` signatures "cannot go allocation-free without an interface change" (inlining the braced overloads needs no signature or call-site change), and that they were "not on the hot path" (never measured).

Design rule:

- Measure the target rank/code path directly rather than assuming a fix at one rank transfers to another. Different ranks can use entirely different tensor types and code paths.

### 2. A cost model must be per-call-site, not per-kernel-average

The cost is per *access*, so it scales with call sites, and generated code has far more of them: 3416 accessor call sites in the generated rank-3 triples residual against 186 in the hand-written kernel. That count difference is the entire measured gap.

Counterintuitively the generated-vs-hand-written ratio *widened* before it narrowed. The hand-written kernel is more accessor-dominated (fewer FLOPs per access), so it gained more from the fix: the ratio went 37.6× → 59× after the fixed-rank-only pass, then to 22× once the runtime-rank half landed. An early draft predicted ~2.8× by assuming the accessor cost was proportionally equal on both sides; it is not.

Design rule:

- Do not estimate a fix's effect on a ratio between two implementations by assuming the fixed cost is proportionally equal on both sides — measure each side independently.

### 3. A bounds check protecting a breakable post-construction invariant must not be dropped silently

`checked_fixed_rank_index` enforced two things: per-index `0 <= idx < dim`, and `offset < data.size()`. The storage half is not redundant with the constructors — `data` is a public member that call sites assign directly after construction (`tensor_backend.cpp:197-198`), so the size invariant is breakable post-construction by design, and the accessor's check was the only thing catching it.

The fix keeps both conditions in a debug assert. In release they compile out, so an out-of-bounds access becomes UB rather than returning a shared `tensor_error_slot`. That is a real narrowing of safety, and it was a deliberate trade rather than a side effect: `tensor_error_slot` is file-local with no consumers anywhere in `src/` or `tests/`, and reaching it already fired `assert(false)` — it was already treated as a bug, not a supported path.

Design rule:

- A build without an explicit `CMAKE_BUILD_TYPE` gets no `-DNDEBUG`, which re-enables these asserts and effectively reverts this entire fix. Anyone timing CC code should confirm the build type first — the repo's `build/` has been in exactly that state and has produced at least one wrong performance diagnosis as a result.

## What was fixed

1. Inlined the fixed-rank tensor accessors (`Tensor2D`/`Tensor4D`/`Tensor6D`) as flat row-major index computations directly in `common.h`, removing the cross-TU call and its two heap allocations per access.
2. Inlined the runtime-rank braced-index accessors (`TensorND` / `DenseTensorView` / `ConstDenseTensorView`) the same way, closing the rank ≥ 4 gap that the fixed-rank-only pass left untouched.
3. Kept both bounds-check conditions (per-index range and `offset < data.size()`) in a debug assert rather than dropping either.
4. Dropped the now-unused `to_vector` and `checked_fixed_rank_index`.

## What this did not fix

The residual ~22× generated-vs-hand-written gap is structural — loop shape and contraction order, not element fetching. It was subsequently characterised as a scaling defect (21.8× → 50.1× across a six-point ladder, no plateau); see `docs/CCGEN_KERNEL_SCALING_SCOPE.md`.

Two levers were explicitly ruled out at the time: emitter loop fusion (measured 0.62×, i.e. no gain) and consuming the unused `tensor_ir.py` BLAS hints (real, but a separate question).

## Validation strategy that should remain in place

- `planck-cc-tensor-index` (`tests/cc_tensor_index.cpp`) pins the flat index against an independent row-major reference on non-square shapes at each rank, covers the permuted-dims `swap_mid_axes` pattern used by `rebind_physicist`, and cross-checks the braced-index overload against the still out-of-line `vector<int>` one. Non-square is load-bearing: `bh3`/STO-3G is `no == nv == 4`, which lets a wrongly-ordered read stay in bounds and fail silently.
- Confirm the build type (`-DCMAKE_BUILD_TYPE=Release`, i.e. `-DNDEBUG` present) before trusting any CC timing measurement.
