# CASSCF Active-Cache Optimization Patch Series

Last updated: 2026-04-08
Branch context: `codex/casscf-active-cache-patch1`

## Purpose

This document turns the active-integral-cache optimization idea into a concrete,
reviewable patch series for the CASSCF code path. The goal is to reduce the cost
of rebuilding the mixed-basis `(p,u,v,w)` tensor used by the orbital-gradient
and response machinery, without destabilizing the existing CASSCF convergence
logic.

The intended audience is a developer who wants to land the work incrementally,
measure each step, and stop at any patch boundary if the payoff has already
become good enough.

## Why This Is The Right Target

`xctrace` Time Profiler was run on:

- executable: `build/hartree-fock`
- input: `tests/inputs/casscf_tests/ethylene_cas44_sto3g_sa2.hfinp`
- environment: `OMP_NUM_THREADS=4`
- trace duration: `173.888 s`
- application wall time: `169.555 s`
- trace artifact: `/tmp/ethylene_sa2_time.trace`

Grouping samples by the first project frame in each call stack gave:

| Hot frame | Sample share |
| --- | ---: |
| `HartreeFock::Correlation::transform_eri(...)` | 75.67% |
| `HartreeFock::ObaraSaika::_compute_fock_rhf(...)` | 15.93% |
| `HartreeFock::Correlation::CASSCF::solve_ci_dense(...)` | 2.26% |
| `compute_2rdm_impl(...)` | 1.15% |
| `HartreeFock::Rys::rys_roots_weights(...)` | 1.04% |
| `HartreeFock::Correlation::CASSCF::apply_orbital_rotation(...)` | 1.03% |

The dominant path is:

1. `run_mcscf_loop()` reevaluates a trial MO basis in `evaluate(...)`
2. `evaluate(...)` rebuilds `st.ga` with `transform_eri_internal(...)`
3. `evaluate(...)` rebuilds `st.active_integrals` with `build_active_integral_cache(...)`
4. `build_active_integral_cache(...)` calls the generic `transform_eri(...)`
5. `compute_Q_matrix(...)` later contracts the cached `(p,u,v,w)` tensor with the active 2-RDM

Relevant source locations:

- `src/post_hf/casscf/casscf.cpp`
- `src/post_hf/casscf/orbital.cpp`
- `src/post_hf/casscf_internal.h`
- `src/post_hf/integrals.cpp`

The present implementation uses the fully generic four-leg AO->MO transform even
though the active-cache path always has the same shape:

- first leg: all MOs, size `nbasis`
- remaining legs: active space only, size `n_act`

That shape is specific enough that a dedicated implementation should outperform
the generic scalar quarter-transform materially.

## Current Data Flow

### Entry points

- `evaluate(...)` in `src/post_hf/casscf/casscf.cpp`
- `build_active_integral_cache(...)` in `src/post_hf/casscf/orbital.cpp`
- `compute_Q_matrix(...)` in `src/post_hf/casscf/orbital.cpp`

### Current active-cache build

`build_active_integral_cache(...)` does:

```text
C_act = C.middleCols(n_core, n_act)
cache.puvw = transform_eri(eri, nbasis, C, C_act, C_act, C_act)
```

The generic implementation in `src/post_hf/integrals.cpp` then performs four
serial quarter-transforms:

- `T1[i,nu,lam,sig]`
- `T2[i,a,lam,sig]`
- `T3[i,a,j,sig]`
- `out[i,a,j,b]`

This has three major issues for the CASSCF active-cache path:

1. It allocates large generic intermediates even though three legs are small.
2. It uses scalar indexing in deep nested loops, creating poor cache behavior.
3. It does not exploit the natural output partitioning needed for OpenMP.

## Optimization Goals

### Primary goals

- Reduce total wall time for SA-CASSCF ethylene CAS(4,4) by improving the
  active-cache build.
- Preserve numerical results to the current tolerance envelope.
- Keep the work reviewable and bisectable.
- Avoid invasive optimizer changes while the tensor path is being reworked.

### Secondary goals

- Improve memory locality in the active-cache builder.
- Reduce transient allocations and allocator churn.
- Make future MPI or blocked distributed work easier, not harder.

### Non-goals

- Replacing the whole CASSCF optimizer.
- Rewriting CI or RDM code in the same series.
- Changing convergence criteria or acceptance logic.
- Removing the generic `transform_eri(...)` path used by MP2 and other modules.

## Design Constraints

1. The cached tensor layout must remain compatible with `compute_Q_matrix(...)`
   unless a later patch intentionally updates both sides together.
2. The generic `transform_eri(...)` API should remain available for non-CASSCF
   callers.
3. Every optimization patch should have a correctness gate before performance
   claims are accepted.
4. OpenMP should be introduced only at loop levels where each thread owns a
   disjoint output slice.
5. Nested OpenMP oversubscription must be avoided because `_compute_fock_rhf()`
   already has internal OpenMP parallelism.

## Proposed Patch Series

## Patch 1: Split The API And Isolate The Active-Cache Path

### Goal

Create a dedicated entry point for the active-cache transform without changing
the generic transform yet.

### Expected files

- `src/post_hf/integrals.h`
- `src/post_hf/integrals.cpp`
- `src/post_hf/casscf/orbital.cpp`

### Changes

- Add a new function with a name that states its intent clearly, for example:
  - `transform_eri_active_cache(...)`
  - or `transform_eri_puvw(...)`
- Keep the output layout exactly equal to the current cache layout:
  `[(p,u,v,w)]`.
- Switch `build_active_integral_cache(...)` to the new entry point.
- Leave `transform_eri(...)` untouched for all other callers.

### Why this patch stands alone

This separates "which code path is hot" from "how it is optimized". It makes
later performance patches easier to review because they no longer affect every
AO->MO transform user in the codebase.

### Validation

- Build succeeds.
- Existing CASSCF tests reproduce the same energies.
- Add a targeted parity test:
  - build active cache with generic transform
  - build active cache with the new entry point
  - compare elementwise within tight tolerance

## Patch 2: Replace The Generic Quarter-Transform With A CASSCF-Specific Scalar Kernel

### Goal

Implement a scalar but shape-aware kernel for the active-cache path.

### Expected files

- `src/post_hf/integrals.cpp`
- `tests/casscf_internal.cpp`

### Changes

- Implement the dedicated active-cache transform directly for the shape:
  `AO(nb,nb,nb,nb) -> MO(nb,n_act,n_act,n_act)`.
- Remove the generic `T1/T2/T3` staging for this path.
- Use a blocked or slab-oriented traversal so the last three active indices stay
  small and contiguous in the output.
- Keep the implementation simple and obviously correct before adding OpenMP.

### Recommended structure

Use one of these patterns:

- outer loop on `p`, accumulate one full `(u,v,w)` block at a time
- outer loop on `(p,u)` tiles, accumulate contiguous `(v,w)` slices

The key requirement is that the output slice assigned to a unit of work must be
independent.

### Why this patch stands alone

It should show whether shape specialization alone gives a benefit even before
threading. If it does not, the later OpenMP patch will be easier to interpret.

### Validation

- Reuse the parity test from Patch 1.
- Benchmark wall time on the ethylene SA-2 input.
- Rerun `xctrace` and confirm the hotspot is still the same function but with a
  lower total cost.

## Patch 3: Add OpenMP Parallelism Over Output-Owned Work Units

### Goal

Parallelize the active-cache builder safely.

### Expected files

- `src/post_hf/integrals.cpp`

### Changes

- Add guarded OpenMP pragmas to the dedicated active-cache transform only.
- Parallelize over disjoint output-owned units such as:
  - `p`
  - `p x u`
  - `p` slabs with static scheduling
- Avoid reductions entirely.
- Prefer `schedule(static)` first unless profiling shows strong imbalance.

### Loop ownership rule

Each thread should own a unique region of `puvw` and write to no shared output
outside that region. If a thread needs scratch storage, it should be private or
thread-local.

### OpenMP guidance

Use the existing project pattern:

```cpp
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
```

### Risk

If the work unit is too small, scheduling overhead may dominate. If it is too
large, some threads may sit idle for small active spaces.

### Validation

- Numerical parity against the scalar specialized kernel.
- Repeat the ethylene SA-2 profile with `OMP_NUM_THREADS=1` and `4`.
- Confirm the dedicated transform scales without changing energies.

## Patch 4: Reduce Allocation Pressure And Reuse Scratch Storage

### Goal

Lower allocator overhead and improve locality.

### Expected files

- `src/post_hf/integrals.cpp`
- possibly `src/post_hf/integrals.h`

### Changes

- Remove repeated `clear()` plus `shrink_to_fit()` style behavior from the hot
  path.
- Reuse one or two work buffers instead of allocating several full intermediates.
- If blocking is used, allocate thread-local scratch once per thread.

### Why this patch matters

The current generic implementation creates large transient buffers. Even when
the arithmetic is acceptable, these allocations amplify cache misses and memory
traffic. This patch should also reduce noise in the profiler, where many tiny
accessor and allocator frames currently appear.

### Validation

- Same numerical tests as earlier patches.
- Compare runtime variability over several repeated runs.
- If possible, compare memory behavior informally via wall-time stability and
  profiler sample concentration.

## Patch 5: Align Cache Layout And Q Contraction For Contiguous Access

### Goal

Make the producer and consumer agree on a cache-friendly traversal.

### Expected files

- `src/post_hf/casscf_internal.h`
- `src/post_hf/casscf/orbital.cpp`
- `src/post_hf/integrals.cpp`
- `tests/casscf_internal.cpp`

### Changes

- Revisit the exact memory order of `puvw`.
- Ensure the hot loops in `compute_Q_matrix(...)` traverse the innermost
  contiguous dimension linearly.
- If the present layout is already best for `Q`, keep it and document why.
- If a better layout exists, update both producer and consumer in the same patch.

### Why this patch is separate

Changing layout is correctness-sensitive and affects every user of
`ActiveIntegralCache`. It deserves its own review boundary.

### Validation

- Add a direct test of `compute_Q_matrix(...)` parity across old and new layouts.
- Run at least:
  - `h2_cas22_sto3g`
  - `water_cas44_sto3g`
  - `ethylene_cas44_sto3g_sa2`

## Patch 6: Optional Experimental Fused Path For Direct Q Construction

### Goal

Test whether materializing `puvw` is still the right choice once the dedicated
path exists.

### Expected files

- `src/post_hf/casscf/orbital.cpp`
- `src/post_hf/casscf_internal.h`
- `src/post_hf/integrals.cpp`

### Changes

- Add an experimental path, likely behind a debug or internal option, that
  computes `Q` directly from AO ERIs, `C`, and `Gamma` without storing the full
  `puvw` tensor.
- Keep the cached path as default until the fused path is proven better.

### Why it is optional

The cache is reused in more than one place. A fused path may save memory but
lose reuse. This is an experiment, not a guaranteed win.

### Validation

- Numerical parity against cached `Q`.
- Compare total runtime, not just the transform subroutine.
- Remove or keep behind a flag based on measured outcome.

## Patch 7: Cleanup, Documentation, And Performance Gates

### Goal

Turn the optimized path into a maintainable, documented feature.

### Expected files

- `docs/CASSCF_STATUS.md`
- `tests/README.md` or another benchmark note if useful
- code comments near the new transform implementation

### Changes

- Document the active-cache path and why it is specialized.
- Record the benchmark protocol and the baseline numbers.
- Add a note on OpenMP ownership and why nested reductions are avoided.

### Validation

- Final profile on the ethylene SA-2 case.
- Confirm no regression on simpler CASSCF cases.

## Suggested Review Boundaries

The series should be reviewed as:

1. API split
2. specialized scalar transform
3. OpenMP on the specialized transform
4. scratch and allocation cleanup
5. cache layout and Q contraction alignment
6. optional fused-Q experiment
7. documentation and status update

This order keeps correctness-sensitive behavior changes small and measurable.

## Detailed Implementation Notes

## Data layout recommendation

For the active cache, prefer an explicit documented row-major mapping:

```text
index(p,u,v,w) = ((p * n_act + u) * n_act + v) * n_act + w
```

This matches the current `compute_Q_matrix(...)` access pattern and makes it
easy to reason about output ownership.

## Preferred parallel decomposition

The safest first decomposition is over `p`.

Reasons:

- each `p` owns a contiguous block of `n_act^3` output values
- no reductions are required
- thread ownership is easy to inspect
- the downstream tensor layout already groups data by `p`

If `n_act` is very small and `nbasis` is also small, `p x u` tiles may provide
better granularity, but `p` should be the first implementation.

## Why Not Parallelize Inner Accumulations

Parallelizing the innermost contraction loops would require reductions into the
same output entries. That adds synchronization, complicates correctness review,
and usually fights cache locality. The output-owned outer decomposition is much
safer and more maintainable.

## Interaction With Existing OpenMP

The active-cache transform should be parallelized, but it should not itself call
other OpenMP-heavy kernels inside the same parallel region. This avoids nested
oversubscription on systems where OpenMP nesting is enabled or not controlled.

## Numerical Validation Strategy

Each patch that changes arithmetic should pass all of the following:

1. Elementwise parity for the active cache on at least one small case.
2. Elementwise parity for `Q`.
3. End-to-end energy agreement on the existing CASSCF benchmark inputs.
4. Convergence behavior that is unchanged except for runtime.

Recommended tolerances:

- active-cache tensor parity: `1e-10` absolute on small cases
- Q-matrix parity: `1e-10` absolute
- final CASSCF energies: existing test tolerances

## Performance Validation Strategy

Use the same benchmark protocol before and after each performance patch:

```bash
OMP_NUM_THREADS=4 ./build/hartree-fock \
  tests/inputs/casscf_tests/ethylene_cas44_sto3g_sa2.hfinp
```

When a patch is claimed to improve performance, rerun:

```bash
xcrun xctrace record --template "Time Profiler" ...
```

Track at least:

- total wall time
- active-cache transform sample share
- `_compute_fock_rhf(...)` sample share
- whether the next bottleneck becomes dominant

## Expected Payoff

The active-cache transform currently dominates the runtime so strongly that even
a modest improvement there should reduce total wall time noticeably.

Reasonable expectations for the series are:

- Patch 2: visible improvement from specialization alone
- Patch 3: additional gain from OpenMP parallelism
- Patch 4: better consistency and lower memory overhead
- Patch 5: smaller but still meaningful locality gain

It is realistic to aim for a large reduction in the `transform_eri(...)` share.
It is not realistic to expect the entire program to scale linearly, because the
Fock build and the rest of the CASSCF machinery will become relatively more
visible as the main hotspot shrinks.

## Rollback Plan

If any patch destabilizes convergence or creates hard-to-debug numerical drift:

1. keep the generic `transform_eri(...)` path intact
2. switch `build_active_integral_cache(...)` back to the generic path
3. retain only the benchmarks and parity tests

That fallback is one reason Patch 1 is worth landing first.

## Deliverables Checklist

- [ ] dedicated active-cache transform API
- [ ] parity test for active-cache tensor
- [ ] parity test for `Q`
- [ ] specialized scalar implementation
- [ ] OpenMP parallel active-cache implementation
- [ ] reduced scratch allocation strategy
- [ ] documented cache layout
- [ ] updated benchmark and status notes

## Recommended First Implementation Step

Start with Patch 1 and Patch 2 only. Those two patches should answer the most
important question quickly:

Can a CASSCF-specific active-cache transform outperform the generic quarter-
transform enough to justify the rest of the series?

If the answer is yes, continue with OpenMP. If the answer is only marginally
yes, re-evaluate whether a fused `Q` path is the better direction.
