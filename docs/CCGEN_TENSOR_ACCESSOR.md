# Why were the CC tensor accessors the dominant cost, and what replaced them?

Every CC kernel reads tensor elements through `operator()`. Those accessors were defined
**out-of-line in `common.cpp`** with no LTO configured, so each element access was a cross-TU call
that heap-allocated one or two `std::vector<int>` and built a `std::expected` before indexing —
**two `malloc`/`free` pairs per element read, on the innermost loop of an `o³v³` kernel**.

Inlining them as flat row-major index computations in `common.h` gave, with energies bitwise
identical throughout:

| | before | after | speedup |
|---|---|---|---|
| rank-3 generated T3 residual (`bh3`) | 6.40 s | 0.031 s | **206×** |
| rank-3 hand-written T3 residual | 0.170 s | 0.0014 s | **121×** |
| rank-4 CCSDTQ per iteration (Be/STO-3G) | 38.5 s | 11.4 s | 3.4× |
| `water_rccsdt_sto3g` regression | 44.6 s | 0.39 s | 114× |

Measured in isolation at the real kernel's size (`no=nv=4`, 1063 sweeps): checked+allocating
0.0446 s vs inlined flat index 0.0033 s — **13.5×**.

## Why the generated kernels suffered most

The cost is per *access*, so it scales with call sites, and generated code has far more of them:
**3416** accessor call sites in the generated rank-3 triples residual against **186** in the
hand-written kernel. That count difference is the entire measured gap.

Counterintuitively the ratio *widened* before it narrowed. The hand-written kernel is **more**
accessor-dominated (fewer FLOPs per access), so it gained more from the fix: the
generated-vs-hand-written ratio went 37.6× → 59× after the fixed-rank-only pass, then to 22× once
the runtime-rank half landed. An early draft predicted ~2.8× by assuming the accessor cost was
proportionally equal on both sides; it is not.

## Rank 3 is not a proxy for rank 4

**Fixing only the fixed-rank accessors sped rank 3 by 76× and left rank-4 CCSDTQ completely
unchanged at ~38 s per iteration.** The rank ≥ 4 generated kernels index *exclusively* through
braced lists on `TensorND` / `DenseTensorView` — **23,338 accesses per residual evaluation** — and
`operator()(initializer_list)` forwarded through `to_vector(indices)`, copying into a fresh vector
*before* the out-of-line `flatten_index` call. Strictly worse than the fixed-rank path it was
excluded in favour of.

That exclusion rested on two claims, both wrong: that the `vector<int>`/`initializer_list`
signatures "cannot go allocation-free without an interface change" (inlining the braced overloads
needs no signature or call-site change), and that they were "not on the hot path" (never measured).

**The lesson generalises: measure the target, not the convenient small case.** Different ranks use
different tensor types and different code paths.

## The bounds check that was traded away

`checked_fixed_rank_index` enforced two things: per-index `0 <= idx < dim`, **and**
`offset < data.size()`. The storage half is *not* redundant with the constructors — `data` is a
public member that call sites assign directly after construction (`tensor_backend.cpp:197-198`), so
the size invariant is breakable post-construction by design, and the accessor's check was the only
thing catching it.

The fix keeps **both** conditions in a debug assert. In release they compile out, so an
out-of-bounds access becomes UB rather than returning a shared `tensor_error_slot`. That is a real
narrowing of safety, and it was a deliberate trade rather than a side effect: `tensor_error_slot` is
file-local with **no consumers anywhere** in `src/` or `tests/`, and reaching it already fired
`assert(false)` — it was already treated as a bug, not a supported path.

**Consequence for anyone timing CC code:** a build without an explicit `CMAKE_BUILD_TYPE` gets no
`-DNDEBUG`, which re-enables these asserts and effectively reverts this entire fix. The repo's
`build/` is in exactly that state; it has produced at least one wrong performance diagnosis.

## What this did not fix

The residual 22× generated-vs-hand-written gap is **structural** — loop shape and contraction order,
not element fetching. It was subsequently characterised as a *scaling* defect (21.8× → 50.1× across
a six-point ladder, no plateau); see `CCGEN_KERNEL_SCALING_SCOPE.md`.

Two levers were explicitly ruled out at the time: emitter loop fusion (measured 0.62×, i.e. no gain)
and consuming the unused `tensor_ir.py` BLAS hints (real, but a separate question).

## Gates

`planck-cc-tensor-index` (`tests/cc_tensor_index.cpp`) pins the flat index against an independent
row-major reference on **non-square** shapes at each rank, covers the permuted-dims `swap_mid_axes`
pattern used by `rebind_physicist`, and cross-checks the braced-index overload against the still
out-of-line `vector<int>` one.

Non-square is load-bearing: `bh3`/STO-3G is `no == nv == 4`, which lets a wrongly-ordered read stay
in bounds and fail silently.

## Key code locations

| what | where |
|---|---|
| inlined fixed-rank accessors + `detail::fixed_rank_index_valid` | `src/post_hf/cc/common.h` |
| inlined ND braced-index accessors + `detail::nd_flat_index` / `nd_index_valid` | same header |
| remaining out-of-line `vector<int>` overloads (non-hot; the gate's cross-check) | `src/post_hf/cc/common.cpp` |
| direct `.data` assignment that breaks the size invariant | `src/post_hf/cc/tensor_backend.cpp:197,198` |
| layout gate | `tests/cc_tensor_index.cpp` |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`, which are canonical.
