# ERI Engine — Data Alignment and Cache Behaviour

Companion to `docs/ERI_PERFORMANCE_SCOPE.md`, which established that the ~200×
gap to libcint is genuine algorithmic work at near-peak IPC and that four
candidate optimizations were each disproven by measurement. This note answers the
remaining hardware-level question:

**Is the ERI engine's data alignment good, and is it losing time to cache
misses?**

## Short answer

**Alignment is fine and there are effectively no cache misses.** The VRR's entire
per-quartet working set fits in L1 with room to spare — 1.8 KB for a (dd|dd)
quartet, 5.0 KB for (ff|ff), 10.8 KB for (gg|gg), against a 128 KB L1d. Measured
cost per element is flat (~0.45–0.79 ns) across all of those sizes, which is the
signature of pure L1 residency: a cache-bound loop's cost rises with working-set
size, and this one does not.

There is no cache optimization available here because there is no cache problem.

## Alignment

`EriScratch`'s three buffers (`vrr`, `hrr`, `a0c0_accum`) are `std::vector<double>`,
so the standard only guarantees `max_align_t` (16 B). Measured actual alignment
from the allocator at realistic sizes:

| buffer elements | vrr | hrr | a0c0 |
|---|---|---|---|
| 13 | 128 B | 16 B | 16 B |
| 169 | 128 B | 128 B | 128 B |
| 2197 | 128 B | 128 B | 128 B |
| 28561 | 128 B | 128 B | 128 B |

At every size the engine actually uses (≥169 elements), malloc's size-class bins
hand back **≥128-byte-aligned** blocks — two cache lines. Only a degenerate
13-element buffer falls back to 16 B, and that case is one cache line total, so
alignment is irrelevant to it.

**No `alignas(64)` / `aligned_alloc` change is warranted.** It would be a no-op at
every size that matters. (It would matter on a 64 B-vector ISA like AVX-512 if the
allocator were *not* already over-aligning — it is.)

## Working set vs L1

The layout (`quartet_layout.h::configure`) is row-major with `cz` unit-stride at
the spatial level, and the Boys `m` axis innermost in `vrr_data`
(`spatial_index × m_dim + m`). So the inner `m`-loop walks **contiguous doubles**.

| quartet | spatial | m_dim | VRR buffer | fits 128 KB L1 |
|---|---|---|---|---|
| (ss\|ss) | 1 | 1 | 0.0 KB | yes |
| (sp\|sp) | 4 | 3 | 0.1 KB | yes |
| (pp\|pp) | 9 | 5 | 0.4 KB | yes |
| (dd\|dd) 6-31g* | 25 | 9 | **1.8 KB** | yes |
| (ff\|ff) cc-pVTZ | 49 | 13 | **5.0 KB** | yes |
| (gg\|gg) | 81 | 17 | 10.8 KB | yes |

The recurrence's `ax-1` / `ax-2` back-references jump 5.6 cache lines (dd) to 11.4
(ff), which *looks* far — but those jumps land inside a buffer that is only a few
KB total, so they are L1 hits, not misses. Distance in cache lines is the wrong
metric when the whole array is resident.

## Measured: flat cost = no misses

Microbenchmark of the A-VRR x-axis sweep against its real strided layout:

| quartet | VRR KB | ns/element |
|---|---|---|
| (sp\|sp) | 0.1 | 0.57 |
| (pp\|pp) | 0.4 | 0.79 |
| (dd\|dd) | 1.8 | 0.45 |
| (ff\|ff) | 5.0 | 0.60 |
| (gg\|gg) | 10.8 | 0.55 |

**Flat across a 100× range of working-set size.** A cache-bound loop cannot do
that.

For contrast, a pointer-chase on the same machine (prefetch defeated) shows what
the hierarchy actually costs:

| working set | ns/access |
|---|---|
| L1 (8–128 KB) | 1.7–2.4 |
| L2 (1–16 MB) | 7.2–15.1 |
| RAM (128 MB) | **86.3** |

The VRR's 0.45–0.79 ns/element is *below even the L1 pointer-chase figure*,
because its accesses are contiguous and pipelined rather than dependent. It is
running in the best case the memory system offers.

## Consistency with the earlier IPC result

`docs/ERI_PERFORMANCE_SCOPE.md` measured median **IPC 5.68** on P-cores (peak
~6–7) during the cc-pVTZ Fock build, with cache-miss-shaped PMU counters at
~0.01–0.08 per cycle. That is the same conclusion from the other direction: a
memory-stalled loop sits at IPC ~1–2 with high miss rates. The 808 loads/stores
seen in the `_eri_vrr` disassembly are all L1 hits that pipeline cleanly.

**High load/store count ≠ memory-bound.** Both measurements agree.

## What this rules out

- `alignas(64)` / aligned allocation for the scratch buffers — no-op at every
  real size.
- Buffer-layout reordering or tiling for locality (the "Tier 3b" idea) — there is
  no locality problem to fix.
- Blocking / cache-aware restructuring of the recurrence — the working set is
  already 100× smaller than L1.

## Caveat

Measured on Apple Silicon (arm64, 128 KB L1d, 64 B lines). On an x86 server with
a 32–48 KB L1d the margin narrows but the conclusion holds: (gg|gg) at 10.8 KB
still fits, and every basis in the regression suite tops out far below that. Re-run
`/tmp/vrr_cache.cpp`-style sizing if a much larger-L target ever appears.
