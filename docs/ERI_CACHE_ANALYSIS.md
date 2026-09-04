# ERI Engine Data Alignment and Cache Behaviour

Companion to `docs/ERI_PERFORMANCE_SCOPE.md`, which established that the ~200x gap to libcint is genuine algorithmic work at near-peak IPC and that four candidate optimizations were each disproven by measurement.

This file answers a narrower architecture question:

**Is the ERI engine's data alignment good, and is it losing time to cache misses?**

## Short answer

Alignment is fine and there are effectively no cache misses. The VRR's entire per-quartet working set fits in L1 with room to spare — 1.8 KB for a (dd|dd) quartet, 5.0 KB for (ff|ff), 10.8 KB for (gg|gg), against a 128 KB L1d. Measured cost per element is flat (~0.45-0.79 ns) across all of those sizes, which is the signature of pure L1 residency: a cache-bound loop's cost rises with working-set size, and this one does not. There is no cache optimization available here because there is no cache problem.

## Where the logic lives

- `src/integrals/os.cpp` — `EriScratch` (`vrr`, `hrr`, `a0c0_accum` buffers)
- `src/integrals/quartet_layout.h` (`configure`) — the row-major spatial layout with the Boys `m` axis innermost
- `docs/ERI_PERFORMANCE_SCOPE.md` — the companion IPC/algorithmic-cost measurement this note follows up on

## What invariants matter

### 1. `std::vector<double>` allocation is already sufficiently aligned at real sizes

`EriScratch`'s three buffers are `std::vector<double>`, so the standard only guarantees `max_align_t` (16 B). Measured actual alignment from the allocator: at every buffer size the engine actually uses (>=169 elements), malloc's size-class bins hand back at least 128-byte-aligned blocks (two cache lines). Only a degenerate 13-element buffer falls back to 16 B, and that case is one cache line total, so alignment is irrelevant to it.

Design rule:

- Do not add `alignas(64)` / `aligned_alloc` to these scratch buffers — measured to be a no-op at every size that matters. It would only matter on a 64 B-vector ISA like AVX-512 if the allocator were not already over-aligning, and it is.

### 2. Distance in cache lines is the wrong metric when the whole working set is resident

The recurrence's `ax-1` / `ax-2` back-references jump 5.6 cache lines (dd) to 11.4 (ff), which looks far, but those jumps land inside a buffer that is only a few KB total — an L1 hit, not a miss. The whole per-quartet VRR buffer stays under 11 KB even at (gg|gg), against a 128 KB L1d.

Design rule:

- When judging whether a recurrence's back-reference pattern is a cache risk, compare the total working-set size against L1 capacity first — do not reason from stride distance in isolation.

### 3. Flat per-element cost across a 100x working-set range is direct evidence of no cache misses

Microbenchmarking the A-VRR x-axis sweep against its real strided layout gives ~0.45-0.79 ns/element flat across (sp|sp) at 0.1 KB through (gg|gg) at 10.8 KB — a 100x range in working-set size with no corresponding rise in cost. A cache-bound loop cannot produce a flat curve like that; its cost rises with working-set size once it crosses L1/L2/RAM boundaries.

Design rule:

- To test whether a hot loop is cache-bound, measure per-element cost across a range of working-set sizes and look for a flat line (not memory-bound) versus a step at each cache-level boundary (memory-bound) — do not infer memory-boundedness from load/store instruction count alone.

## What was measured

1. **Alignment** across representative buffer sizes (13, 169, 2197, 28561 elements): all three scratch buffers (`vrr`, `hrr`, `a0c0`) land on at least 128-byte boundaries at every size the engine actually uses.
2. **Working-set size per quartet** against the 128 KB L1d: (ss|ss) 0.0 KB, (sp|sp) 0.1 KB, (pp|pp) 0.4 KB, (dd|dd) 6-31g* 1.8 KB, (ff|ff) cc-pVTZ 5.0 KB, (gg|gg) 10.8 KB — every quartet fits comfortably.
3. **Per-element cost of the VRR A-VRR x-axis sweep**, flat at 0.45-0.79 ns/element from (sp|sp) through (gg|gg) — below even the L1 figure from an independent pointer-chase benchmark on the same machine (L1 1.7-2.4 ns/access, L2 7.2-15.1 ns/access, RAM 86.3 ns/access), because the VRR's accesses are contiguous and pipelined rather than dependent.
4. **Cross-check against the earlier IPC measurement.** `docs/ERI_PERFORMANCE_SCOPE.md` measured median IPC 5.68 on P-cores (peak ~6-7) during the cc-pVTZ Fock build, with cache-miss-shaped PMU counters at ~0.01-0.08 per cycle — a memory-stalled loop would instead sit at IPC ~1-2 with high miss rates. The 808 loads/stores seen in the `_eri_vrr` disassembly are all L1 hits that pipeline cleanly. High load/store count does not imply memory-bound; both measurements agree on that conclusion from independent directions.

## What was ruled out

- `alignas(64)` / aligned allocation for the scratch buffers — no-op at every real size.
- Buffer-layout reordering or tiling for locality (the "Tier 3b" idea) — there is no locality problem to fix.
- Blocking / cache-aware restructuring of the recurrence — the working set is already 100x smaller than L1.

## Validation strategy that should remain in place

- Re-run the sizing microbenchmark (`/tmp/vrr_cache.cpp`-style) if a much larger angular-momentum target ever appears, since the margin on a smaller-L1 platform narrows.

## Remaining architecture concern

Measured on Apple Silicon (arm64, 128 KB L1d, 64 B lines). On an x86 server with a 32-48 KB L1d the margin narrows but the conclusion holds: (gg|gg) at 10.8 KB still fits, and every basis in the regression suite tops out far below that. If a target basis set with much higher angular momentum than currently supported is ever added, re-run the sizing analysis rather than assuming the margin still holds.
