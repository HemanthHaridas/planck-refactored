# What limits the size of a CCSD(T) calculation, and what removes it?

Scope, opened 2026-11-20. Status lives in `vault/Status/Open Work.md`
(priority 1). Parent: `docs/CCSD_T_SCOPE.md`.

## Answer (2026-11-23)

Memory, not the CC algebra, limited the size, and it no longer binds at the
water dimer in cc-pVTZ. The generated spatial RCCSD that `correlation
ccsd(t)` runs is 48–59× faster than the hand-written spin-orbital one with
2.5–4× less memory. What filled memory was the AO integral handling and
avoidable copies around it. Each phase now holds one large object:

| Phase | Largest object | Peak footprint |
|---|---|---|
| SCF | spherical AO ERI tensor, 1.38 GB (ns⁴) | 1.31 GB |
| CCSD | `vvvv`, 0.96 GB (v⁴), with the other blocks and amplitudes | 1.43 GB |
| (T) | `vvvv` and blocks, kept by the state | 1.14 GB |

The next memory wall is the ns⁴ AO tensor in the SCF and v⁴ `vvvv` in the
CCSD. Removing either is a different method (direct or RI integrals, an
AO-driven or RI `vvvv` contraction), not a fix.

| Water dimer / cc-pVTZ `ccsd(t)` (nb 116 sph / 130 cart, o 10, v 106) | |
|---|---|
| Total energy | −152.7012785674 Eh, equal to PySCF 2.15 to every printed digit |
| E_corr(CCSD), E(T) | −0.5642828768, −0.0160404999 |
| Peak footprint | 6.7 GB (did not finish) → 5.5 GB → 3.93 GB → 2.83 GB → **1.43 GB** |
| CCSD / (T) / whole run | 18 × 11.73 s / 284 s / 512 s, 4 threads |
| Against PySCF 2.15 (4 threads) | CCSD 213 s vs 7.4 s (29×), (T) 283 s vs 6.0 s (47×); the gap grows with size (step 7) |

## What changed

| # | Change | Where | Effect |
|---|---|---|---|
| 1 | One SCF copy of the ERI tensor; the spherical transform frees its input | `build_conventional_eri` (`src/scf/scf.cpp`), `transform_eri_cart_to_sph` | `885109ce`. Water cc-pVTZ Cartesian SCF 466 → 330 MiB |
| 2 | `transform_eri` batched over its leading index | `src/post_hf/integrals.cpp` | `25714026`. Removes two n1·nb³ / n1·n2·nb² intermediates (1.32 + 1.21 GB on the dimer); bitwise identical, no time cost |
| 3 | (T) reuses the CCSD's converged integral blocks | `compute_rccsd_t_energy(calculator, state)` | `94fb61b5`. E(T) bitwise unchanged; dimer (T) 379 → 354 s |
| 3b | Re-exec with libmalloc's large-block cache off | `src/base/malloc_env.h`, both `main`s | `22589e59`. Water cc-pVTZ peak footprint 624 → 262 MiB |
| 4 | Free the engines' thread-local ERI scratch after a one-shot tensor build | `_compute_2e` in `src/integrals/base.h` | `22589e59`. ~130 MiB on water cc-pVTZ |
| 6 | Spherical tensor built per shell quartet, never the Cartesian one | `RysQuad::_compute_2e_spherical` (`src/integrals/rys.cpp`) | SCF peak nc⁴ + ns·nc³ (4.1 GB on the dimer) → ns⁴ (1.38 GB) |
| — | CC solver moves its state instead of copying it | `run_generated_arbitrary_order_iterations` | `5be4a29e`. One copy of every integral block fewer (963 + 91 MiB on the dimer). Water cc-pVTZ peak live 245 → 186 MiB |
| 5 | Free the AO tensor before the CC iterations | `Correlation::release_eri`, called in `rccgen.cpp` and `uccgen.cpp` | Dimer peak footprint 2.83 → 1.43 GB |

Energies are identical to every printed digit after every change. The core
suite passes (178/178) with all of them.

**Step 5** frees `calculator._eri` right before a generated CC route's
iterations. The iterations and (T) read only the CC blocks, and every
rank's `prepare`, the warm-start recursion's included, runs before any rank
iterates. A later `ensure_eri` rebuilds the tensor, so a missed reader
would cost time, not correctness. Both routes are single-point only.
Gated by the RCC cases on `build-default` and the UCC and generated CCSDT
cases (`b_ucc{2,3,4}`, `*_uccsd_t_*`, `*_rccsdt_generated_*`) on the fully
featured `build/`.

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

Step 7 (the ladder re-measure, below) found time, not memory, is now the
limit, and that Planck loses to PySCF 2.15 by a factor that grows with
size: CCSD ~2× → 29×, (T) ~10× → 47× from water cc-pVDZ to the dimer in
cc-pVTZ. On the dimer (T) is 55 % of the run.

1. **Step 8: reshape the (T) kernels.** The larger and faster-growing gap
   (283 s against PySCF's 6.0 s on the dimer). The block kernels loop one
   term at a time through generic accessors. Each (a, b, c) block is
   o³·(o + v) work over 30 terms; an occupied-index GEMM per term is the
   candidate. Profile first. Gates: `ccsd_t_oracle_diff` and bitwise thread
   invariance.
2. **Step 9: CCSD time** (213 s against 7.4 s on the dimer). Profile
   first. Known levers, unmeasured at rank 2: the doubles residual is
   symmetric under (ia)(jb) (5.6e-16), an order-2 group the
   external-symmetry fold could use; loop order in the generated nests (up
   to 1.7× in a one-nest benchmark, `docs/CCGEN_GENERATED_CC_COST.md`). The
   gap grows with size, which was this scope's condition for a
   hand-written spatial RCCSD (PySCF `rccsd` form) as the fallback; decide
   after the profile whether the generated route can close a 29× gap.

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

Step 7, at HEAD `d9f5fb69`. `build-default`, spherical, conventional SCF,
4 threads. Timing from plain runs (3 repeats; one for the dimer in cc-pVTZ,
whose profiled run took the same 512 s); footprint from separate profiled
runs. Energies identical across repeats and equal to the gates.

| Rung | nb sph / cart | o / v | CCSD | (T) | Whole run | Peak footprint |
|---|---|---|---|---|---|---|
| water cc-pVDZ | 24 / 25 | 5 / 19 | 19 × 0.010 s | 0.03 s | 0.26 s | ~33 MiB |
| water cc-pVTZ | 58 / 65 | 5 / 53 | 16 × 0.19 s | 1.8–1.9 s | 5.8–6.1 s | 212 MiB |
| water dimer cc-pVDZ | 48 / 50 | 10 / 38 | 20 × 0.45 s | 4.0–4.1 s | 13.4 s | 84 MiB |
| water dimer cc-pVTZ | 116 / 130 | 10 / 106 | 18 × 11.8 s | 283 s | 512 s | 1.43 GB |

Per-iteration times are unchanged by the memory work except on the dimer
in cc-pVTZ, which no longer pages (CCSD 15.3 → 11.8 s per iteration, (T)
354 → 283 s).

### Against PySCF

PySCF 2.15.0, RHF + RCCSD + `ccsd_t.kernel`, Planck's own GBS files through
`parse_gaussian`, `cart=False`, `conv_tol` 1e-11, CCSD `conv_tol` 1e-10, 4
threads (`OMP_NUM_THREADS=4`), same geometries. Script:
`tests/pyscf/ccsd_t_ladder_reference.py <input> <gbs>`. PySCF
times are wall clock to 0.1 s.

| Rung | E total, PySCF | Δ Planck | CCSD Planck / PySCF | (T) Planck / PySCF |
|---|---|---|---|---|
| water cc-pVDZ | −76.2431380866 | 5e-10 | 0.19 / 0.1 s (~2×) | 0.03 / < 0.05 s |
| water cc-pVTZ | −76.3457669118 | 7e-10 | 3.1 / 0.2 s (~16×) | 1.9 / 0.2 s (~10×) |
| water dimer cc-pVDZ | −152.4977225012 | 2e-10 | 9.1 / 0.4 s (~23×) | 4.0 / 0.2 s (~20×) |
| water dimer cc-pVTZ | −152.7012785674 | 0 | 213 / 7.4 s (29×) | 283 / 6.0 s (47×) |

PySCF's peak footprint on the dimer in cc-pVTZ is 1.94 GB, against
Planck's 1.43 GB. The Δ on the small rungs is CCSD convergence (Planck
stops at `tol_energy`), not a defect.

### Dimer in cc-pVTZ: where the memory was

Live blocks ≥ 1 MiB during the CCSD, from the profiler:

| MiB | Object | After 3b, 4, 6 | + move fix | + step 5 |
|---|---|---|---|---|
| 2179 | spherical AO tensor in the recycled Cartesian block (1448 requested) | 1381 | 1381 | gone |
| 963 | `vvvv`, `rebind_physicist` | 963 | 963 | 963 |
| 963 | `vvvv` copy, `TensorCCBlockCache` copy in the solver | 963 | gone | gone |
| 91 + 91 | `ovvv` and its copy | 91 + 91 | 91 | 91 |
| ~140 | thread-local ERI scratch | gone | gone | gone |
| | **peak live / footprint** | 3.89 / 3.93 GB | 2.80 / 2.83 GB | 1.41 / 1.43 GB |
| | **through (T)** | 2.49 GB live | 2.49 GB live | 1.11 GB live |

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
