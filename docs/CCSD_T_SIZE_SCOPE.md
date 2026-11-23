# What limits the size of a CCSD(T) calculation, and what removes it?

Scope, opened 2026-11-20. Status lives in `vault/Status/Open Work.md`
(priority 1). Parent: `docs/CCSD_T_SCOPE.md`.

## Answer (2026-11-23)

Memory, not the CC algebra, limits the size. The generated spatial RCCSD
that `correlation ccsd(t)` runs is 48–59× faster than the hand-written
spin-orbital one with 2.5–4× less memory. What filled memory was the AO
integral handling and avoidable copies around it. Most of that is gone. What
is left, on the water dimer in cc-pVTZ:

| MiB | Object | Removed by |
|---|---|---|
| 1381 | spherical AO ERI tensor, alive through the whole CC | step 5 (open) |
| 963 | `vvvv` block | needed by the CCSD |
| ~600 | other CC blocks, amplitudes, DIIS | — |

The CC phase then needs roughly 1.5–2 GB on the dimer, down from 5.5 GB.
That figure is extrapolated: the last dimer run (3.93 GB peak) predates the
move fix in the table below, which removed a second 963 MiB `vvvv`.

| Water dimer / cc-pVTZ `ccsd(t)` (nb 116 sph / 130 cart, o 10, v 106) | |
|---|---|
| Total energy | −152.7012785674 Eh (no PySCF reference yet) |
| E_corr(CCSD), E(T) | −0.5642828768, −0.0160404999 |
| Peak footprint | 6.7 GB (did not finish) → 5.5 GB → **3.93 GB** (before the move fix) |
| Whole run | 519 s, 4 threads, still compressing ~2.5 GB of other processes' memory |
| (T) | 42–57 % of the CC time on the measured rungs: the next time lever |

## What changed

| # | Change | Where | Effect |
|---|---|---|---|
| 1 | One SCF copy of the ERI tensor; the spherical transform frees its input | `build_conventional_eri` (`src/scf/scf.cpp`), `transform_eri_cart_to_sph` | `885109ce`. Water cc-pVTZ Cartesian SCF 466 → 330 MiB |
| 2 | `transform_eri` batched over its leading index | `src/post_hf/integrals.cpp` | `25714026`. Removes two n1·nb³ / n1·n2·nb² intermediates (1.32 + 1.21 GB on the dimer); bitwise identical, no time cost |
| 3 | (T) reuses the CCSD's converged integral blocks | `compute_rccsd_t_energy(calculator, state)` | `94fb61b5`. E(T) bitwise unchanged; dimer (T) 379 → 354 s |
| 3b | Re-exec with libmalloc's large-block cache off | `src/base/malloc_env.h`, both `main`s | `22589e59`. Water cc-pVTZ peak footprint 624 → 262 MiB |
| 4 | Free the engines' thread-local ERI scratch after a one-shot tensor build | `_compute_2e` in `src/integrals/base.h` | `22589e59`. ~130 MiB on water cc-pVTZ |
| 6 | Spherical tensor built per shell quartet, never the Cartesian one | `RysQuad::_compute_2e_spherical` (`src/integrals/rys.cpp`) | SCF peak nc⁴ + ns·nc³ (4.1 GB on the dimer) → ns⁴ (1.38 GB) |
| — | CC solver moves its state instead of copying it | `run_generated_arbitrary_order_iterations` | One copy of every integral block fewer (963 + 91 MiB on the dimer). Water cc-pVTZ peak live 245 → 186 MiB |

Energies are identical to every printed digit after every change. The core
suite passes (178/178) with all of them.

**Step 6** in detail. Each canonical shell quartet's Cartesian block comes
from the same kernels `_compute_2e` uses (Auto per quartet, HGP or Rys
hoisted blocks, per-component OS). It gets the same per-component Schwarz
screen, then four one-index transforms with the shells' blocks of `C`. Only
canonical spherical elements are written, with their 8 permutations, so the
tensor is exactly symmetric. A canonical element can have its ket shell pair
before its bra pair when both start on the same shell, (P1 P2 | P1 P1)
holding (3,6|4,4), so the ket loop starts at (sa, sa). The builder checks
that `C` is block-diagonal by shell instead of assuming it. With integral
symmetry operations, which act on Cartesian AOs, it falls back to
build-then-transform. Gated in `planck-compute-2e`: equal to
build-then-transform to ≤ 6.7e-16 (cc-pVDZ, all four engines) and 2.1e-15
(cc-pVTZ, Auto), against integrals up to 4.9, and exactly 8-fold symmetric.
The gate's first run caught the missing (sa, sa) kets (4592 elements, all
unwritten). The SCF's spherical build no longer has the transform's hidden
150-Cartesian-AO limit, matching `ensure_eri` and the Cartesian build.

## Open, in order

1. **Step 5: free the AO tensor once the CC blocks exist** (1.38 GB on the
   dimer). Neither the CCSD iterations nor (T) read it. Check every later
   reader first (post-CC properties, checkpoint, any `ensure_eri` after the
   CC); if one exists, free it only on the `ccsd(t)` path.
2. **Step 7: re-measure the ladder.** Footprint-watched and timed in
   separate runs (`vmmap` pauses the process). Add a PySCF CCSD(T)
   reference for the dimer in cc-pVTZ if it fits there. Decides between 8
   and 9.
3. **Step 8: reshape the (T) kernels.** The block kernels loop one term at a
   time through generic accessors. Each (a, b, c) block is o³·(o + v) work
   over 30 terms; an occupied-index GEMM per term is the candidate. Gates:
   `ccsd_t_oracle_diff` and bitwise thread invariance.
4. **Step 9: CCSD time.** Profile first. Known levers, unmeasured at rank 2:
   the doubles residual is symmetric under (ia)(jb) (5.6e-16), an order-2
   group the external-symmetry fold could use; loop order in the generated
   nests (up to 1.7× in a one-nest benchmark,
   `docs/CCGEN_GENERATED_CC_COST.md`). A hand-written spatial RCCSD is the
   fallback only if the generated route loses by a factor that grows with
   size.

Beyond these: build the MO blocks from a direct or RI transform without
storing the AO tensor at all. RI-CCSD would be its own scope.

## Findings

- **macOS libmalloc held up to half the footprint.** It keeps a freed large
  block dirty in its large-block cache and returns it whole for a later,
  smaller request. The freed 2.18 GB Cartesian tensor became the 1.45 GB
  spherical tensor's block, pinning a dirty 0.73 GB tail for the whole run.
  A standalone C program reproduces it: freeing a touched 1000 MiB block
  leaves the footprint at 1001 MiB, and a 632 MiB request gets the same
  pointer back.
  - Ruled out: `malloc_zone_pressure_relief(NULL, 0)` (releases 0 bytes),
    `madvise(MADV_FREE_REUSABLE)` or `MADV_FREE` before `free` (footprint
    unchanged), `MallocXzoneEnabled=0` (no change).
  - Works: `MallocLargeCache=0`, read only at process start, hence the
    re-exec. A user-set value is kept, a failed exec continues unchanged,
    `planck-mpi` does not re-exec (unsafe under an MPI launcher). No cost:
    water cc-pVTZ 5.82–5.86 s with the cache, 5.87–5.92 s without.
  - Linux: a no-op. glibc serves blocks above its mmap threshold (at most
    32 MiB) with `mmap` and returns them on `free`. From glibc's documented
    defaults, not measured.
- **The live-byte total read off a profile depends on `malloc_size`.** An
  earlier profile concluded the CCSD's ≥ 1 MiB allocations summed to less
  than the SCF peak (4.26 GB) and looked for small allocations. They summed
  to 5.34 GB, because the spherical tensor occupied a 2179 MiB block. Small
  allocations were under 20 MiB of 5.36 GB live. Track requested size next
  to `malloc_size` and the reuse shows as slack.
- **An earlier attribution of the dimer's peak to (T) was wrong.** (T)
  rebuilt every integral block (step 3 removed that), but the footprint
  peak had already been reached in the CCSD, at 73 s.
- **Measure footprint, not RSS, near the RAM limit.** On a paging run RSS
  read 0.1–1.3 GiB while the footprint was 5–6.7 GB; an RSS watchdog never
  fired, and `/usr/bin/time` max-RSS understates the same way.
- **`vmmap` pauses the process it inspects** (water cc-pVTZ 6.3 → 7.9 s at
  a 2 s poll). The profiler reads `task_info` in-process instead.
- **The spin-orbital RCCSD was the first limit.** `run_rccsd` works in 2o
  and 2v spin orbitals: 16 v⁴ `vvvv`, 16 o²v² t2, a 64 o²v⁴ leading
  contraction. Moving `ccsd(t)` to the generated spatial RCCSD removed it.
- **The generated route's speed is the dressing.** Undressed, rank 2 ran the
  same 26 iterations in 45.5 s on C1 H2O2, slower than hand-written; dressed
  (`--dressing derived`, every residual term a binary contraction) 0.54 s.
  So rank 2 is emitted dressed in every build.

## Tools

The allocation profiler is a `DYLD_INSERT_LIBRARIES` interposer over
malloc/calloc/realloc/free/posix_memalign/aligned_alloc. Every N seconds it
logs live bytes by log2 size class, slack (`malloc_size` minus requested)
for blocks ≥ 1 MiB, the malloc zone's in-use and allocated bytes, and the
physical footprint and compressed bytes from `task_info`. At exit it writes
the backtraces, requested and actual sizes of the ≥ 1 MiB blocks alive at
the live peak after an arm time. Build it with gcc-15 (the system clang's
linker cannot read the macOS 27 SDK) and launch the binary directly: SIP
strips `DYLD_*` through `/usr/bin/time`.

## Evidence

### Hand-written against generated RCCSD

HEAD `a1692526`, 4 threads, 3 repeats each, whole-run wall and peak RSS.

| | hand-written RCCSD + (T) | generated RCCSD + (T) |
|---|---|---|
| C1 H2O2 6-31G | 32.0–32.4 s, 83–86 MiB, 22 iter | 0.54–0.58 s, 21 MiB, 26 iter |
| water cc-pVDZ spherical | 14.5–14.8 s, 100–106 MiB, 15 iter | 0.30–0.31 s, 37–40 MiB, 19 iter |

E(T) agrees to every printed digit, and both totals equal the PySCF-pinned
gates. Plain `ccsd` still runs the hand-written RCCSD.

### The ladder

`build-default`, spherical, conventional SCF, 4 threads. Measured before
steps 3b–6; step 7 repeats it.

| Rung | nb sph / cart | o / v | CCSD iters × s/iter | (T) |
|---|---|---|---|---|
| water cc-pVDZ | 24 / 25 | 5 / 19 | 19 × 0.010 s | 0.04 s |
| water cc-pVTZ | 58 / 65 | 5 / 53 | 16 × 0.188 s | 1.8 s |
| water dimer cc-pVDZ | 48 / 50 | 10 / 38 | 20 × 0.457 s | 4.0 s |
| water dimer cc-pVTZ | 116 / 130 | 10 / 106 | 18 × 15.3 s | 354 s |

### Dimer in cc-pVTZ: where the memory was

Live blocks ≥ 1 MiB during the CCSD, from the profiler:

| MiB | Object | After 3b, 4, 6 | After the move fix |
|---|---|---|---|
| 2179 | spherical AO tensor in the recycled Cartesian block (1448 requested) | 1381 | 1381 |
| 963 | `vvvv`, `rebind_physicist` | 963 | 963 |
| 963 | `vvvv` copy, `TensorCCBlockCache` copy in the solver | 963 | gone |
| 91 + 91 | `ovvv` and its copy | 91 + 91 | 91 |
| ~140 | thread-local ERI scratch | gone | gone |
| | **peak live / footprint** | 3.89 / 3.93 GB | not re-run |

Before 3b–6 the SCF also held the Cartesian tensor (2179 MiB) with the
transform's first quarter (1944 MiB), 4.26 GB at ~23 s.

### Water cc-pVTZ `ccsd(t)` through the changes

| | peak live | peak footprint |
|---|---|---|
| after step 3 | 424 MiB (`malloc_size`) | 624 MiB |
| + 3b, 4 | 245 MiB | 262 MiB |
| + 6 | 245 MiB (SCF phase 235 → 185) | 260 MiB |
| + move fix | 186 MiB | 213 MiB |

## Not in this scope

RI-CCSD and RI-(T), frozen core, MPI distribution of the CC solve, and the
generated rank-3 CCSDT's cost (`docs/CCGEN_GENERATED_CC_COST.md`).
