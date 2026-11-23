# What limits the size of a CCSD(T) calculation, and what removes it?

Scope, opened 2026-11-20. Status lives in `vault/Status/Open Work.md`
(priority 1). Parent: `docs/CCSD_T_SCOPE.md`.

## Current result (2026-11-23)

- **The CCSD is no longer the limit.** `correlation ccsd(t)` runs the
  generated spatial RCCSD, 48–59× faster than the hand-written spin-orbital
  one with 2.5–4× less memory.
- **The AO integral handling is the limit.** Copies of the AO ERI tensor
  and full-size transform intermediates, not CC amplitudes, set the peak.
- **Water dimer / cc-pVTZ now completes on this 8 GiB machine**, but only
  just: footprint peak 5.5 GB, with ~3.5 GB of it paged out. Before steps 1
  and 2 it did not fit (6.7 GB, stopped).
- **The macOS allocator was holding up to half the footprint** (step 3b).
  libmalloc kept freed large blocks dirty and handed them back whole for
  smaller requests: the freed 2.18 GB Cartesian tensor became the 1.45 GB
  spherical tensor's block, pinning a 0.73 GB tail for the whole run. Both
  binaries now re-exec themselves with `MallocLargeCache=0`. Water cc-pVTZ
  `ccsd(t)` peak footprint 624 → 262 MiB, same energy and time. Not yet
  re-measured on the dimer (step 7).
- **Thread-local ERI scratch is freed after the tensor build** (step 4),
  ~130 MiB on water cc-pVTZ.
- **The remaining live-tensor peak is the SCF's Cartesian→spherical
  transform**: Cartesian tensor 2.18 GB + first quarter-transformed tensor
  1.94 GB at once (step 6 removes it).
- **(T) is now 42–57 % of the CC time**, so once memory is settled (T)'s
  kernels are the next time lever.

| Water dimer / cc-pVTZ `ccsd(t)` (nb 116 sph / 130 cart, o 10, v 106) | |
|---|---|
| Total energy | −152.7012785674 Eh (no PySCF reference yet) |
| E_corr(CCSD), E(T) | −0.5642828768, −0.0160404999 |
| SCF / CCSD / (T) | 20 iterations / 18 × 15.3 s = 276 s / 354 s (after step 3; 379 s before) |
| Whole run | 542 s without the watchdog (661–694 s with it) |
| Footprint, before step 3b | 4.3 GB at the SCF transform (~23 s), 5.2–5.5 GB through the CCSD (44–251 s), 3.7 GB through (T) |

## Steps

Done steps first, then open ones in the recommended order. "Needs" lists
the steps that must land first.

| # | Step | State | Needs |
|---|---|---|---|
| 0 | Measure the limit on the generated route | done | — |
| 1 | One SCF copy of the ERI tensor; transform consumes its input | done, `885109ce` | 0 |
| 2 | Batch `transform_eri` over its leading index | done, `25714026` | 0 |
| 3 | (T) reuses the CCSD's integral blocks | done, `94fb61b5` | — |
| 3a | Allocation profile of the dimer in cc-pVTZ: which phase sets the peak | done (corrected by 3b) | 3 |
| 3b | Find what fills the CCSD-phase footprint; re-exec with `MallocLargeCache=0` | done | 3a |
| 4 | Free the thread-local ERI scratch after the build | done | — |
| 6 | Direct spherical ERI build (per-block transform) | next; removes the live-tensor peak | 3a |
| 5 | Free the AO tensor once the CC blocks exist | open | 3 |
| 7 | Re-measure the ladder, footprint-watched, clean timings | open | 3b, 4, 5, 6 |
| 8 | Reshape the (T) kernels | open, conditional | 7 |
| 9 | CCSD time levers | open, conditional | 7 |

### 0. Measure the limit — done

The ladder below ("Evidence") found the AO tensor binding through its
copies, not the CC.

### 1. One SCF copy; the transform consumes its input — done (`885109ce`)

- `build_conventional_eri` (`src/scf/scf.cpp`) builds the tensor straight
  into `calculator._eri` for RHF, UHF and ROHF, replacing three copies of
  the same block. Each SCF reads it by reference, and a previous geometry's
  tensor is freed before the new one is built.
- `transform_eri_cart_to_sph` gained an rvalue overload that frees the
  Cartesian tensor after the first quarter step; both overloads share one
  implementation. The SCF and `ensure_eri` move into it; the per-iteration
  full-symmetry skeleton callers keep the `const&` overload.
- Result: Cartesian water cc-pVTZ SCF 466 → 330 MiB (one 142.8 MB
  tensor); spherical dimer cc-pVDZ SCF 165 → 122 MiB. Energies unchanged;
  46 spherical and Cartesian HF/post-HF regression cases pass.

### 2. Batch `transform_eri` — done (`25714026`)

- `transform_eri` (`src/post_hf/integrals.cpp`) runs all four quarter
  transforms per leading index i into that i's output slice. The
  intermediates become slabs (nb³ + n2·nb² + n2·n3·nb doubles, ~27 MB on
  the dimer in cc-pVTZ) instead of n1·nb³ and n1·n2·nb² tensors (1.32 +
  1.21 GB when building `vvvv` there). Each element sums its contracted
  index in the same ascending order, so the result is bitwise identical.
- Callers: MP2, the CC block builders (RCC, UCC, determinant-space),
  stability, and the DFT driver (TDDFT, double hybrids). CASSCF does not
  call it.
- Result: dimer cc-pVTZ completes (footprint 6.7 → 4.8 GB through the
  CCSD, 5.5 GB peak in (T)). No time cost (A/B below), energies identical,
  `planck-transform-eri` passes at 1 and 4 threads.
- Regression: 110 of the 114 cases that reach the transform (MP2, CC, UCC,
  stability, TDDFT, double hybrids) pass on the fully featured `build/`;
  the other four are the known `rccsdt` routing cases.

### 3. (T) reuses the CCSD's integral blocks — done (`94fb61b5`)

- **Before:** `compute_rccsd_t_energy` called
  `prepare_generated_arbitrary_order_state` again and rebuilt the
  reference, every integral block and the denominators from the AO tensor,
  though `run_rccgen` held all of them in its converged state.
- **Change:** `compute_rccsd_t_energy(calculator, state)` takes the CCSD's
  converged `ArbitraryOrderTensorCCState`, the same signature the UCC (T)
  (`compute_uccsd_t_energy`) already had; `run_rccgen` passes
  `solve_res->state`. The solver only reads the blocks (`const` uses), and
  both builds came from the same `prepare` call, so (T) sees the same data.
- **Result:** E(T) bitwise unchanged on every rung; the four `*_rccsd_t_*`
  gates pass, including `ccsd_t_oracle_diff`. (T) time: water cc-pVTZ
  2.17 → 1.83 s, dimer cc-pVDZ 4.17 → 3.96 s, dimer cc-pVTZ 379 → 354 s.
- **The dimer cc-pVTZ footprint peak did not move** (5.5 GB before and
  after). The earlier attribution of that peak to the rebuild was wrong: the
  step-3 run reached 5.3 GB within 73 s, before (T). That is step 3a.

### 3a. Allocation profile of the dimer in cc-pVTZ — done

- Same interposed allocator, dimer cc-pVTZ, footprint polled every 20 s,
  no watchdog. (A first attempt launched the binary through `/usr/bin/time`;
  macOS SIP strips `DYLD_INSERT_LIBRARIES` from system binaries and their
  children, so the profiler never loaded. Launch the binary directly.)
- **Live allocations ≥ 1 MiB peaked at 4.26 GB, in the SCF:**

  | MiB | Object |
  |---|---|
  | 2179 | Cartesian AO ERI tensor (`_compute_2e_auto` in `build_conventional_eri`), 130⁴ |
  | 1944 | first quarter-transformed tensor (`transform_eri_cart_to_sph`), 116·130³ |
  | ~140 | six thread-local `EriScratch` buffers |

- **Footprint timeline:** 2.3 GB at 2 s, 4.3 GB peak by ~23 s (the
  transform), 5.2–5.3 GB from 44 s to 251 s (the CC block build and the
  CCSD iterations), 3.7 GB from ~270 s to the end ((T)). Wall 542 s;
  energy unchanged.
- **So:** the live-tensor peak is the SCF transform (step 6 removes it);
  the footprint peak is the CCSD phase, and at least ~1 GB of it is not
  large live tensors. Step 3b finds it.
- **Corrected by 3b:** "the CCSD's ≥ 1 MiB allocations sum to less than
  4.26 GB" was wrong. They sum to 5.34 GB, because the spherical tensor's
  block is 2179 MiB, not 1448 (below).

### 3b. What fills the CCSD-phase footprint — done

- **Tool:** a rewritten interposer (`DYLD_INSERT_LIBRARIES`, built with
  gcc-15: the system clang's linker cannot read the macOS 27 SDK). Every
  5 s it logs live malloc bytes by log2 size class, the malloc zone's
  in-use and allocated bytes, and the physical footprint from `task_info`,
  so nothing pauses the process. At exit it writes the backtraces of the
  ≥ 1 MiB blocks alive at the live peak, with requested and actual sizes.
- **Not small allocations, not a leak.** Through the CCSD: 5.36 GB live in
  ~16 k blocks, under 20 MiB of it below 1 MiB; footprint 5.49 GB; the zone
  holds only ~40 MB more than is in use.
- **The allocator reused a freed block whole.** The spherical tensor (1448
  MiB requested) sat in a 2179 MiB block allocated inside
  `transform_eri_cart_to_sph`: libmalloc kept the freed Cartesian tensor's
  block dirty in its large-block cache and returned it for the smaller
  request. The 731 MiB tail, still dirty with Cartesian data, counted in
  the footprint until the end of the run. Same on water cc-pVTZ (86.3 MiB
  in a 136.2 MiB block; a 78.9 MiB request in 121.5 MiB).
- **A standalone test reproduces it without Planck.** `free` of a touched
  1000 MiB block leaves the footprint at 1001 MiB, and a 632 MiB request
  gets the same pointer back at 1000 MiB.
  - `malloc_zone_pressure_relief(NULL, 0)` releases 0 bytes.
  - `madvise(MADV_FREE_REUSABLE)` or `MADV_FREE` before the `free` leaves
    the footprint unchanged.
  - `MallocXzoneEnabled=0` changes nothing.
  - `MallocLargeCache=0` returns the block on `free` (footprint 1 MiB), and
    the smaller request gets its own block.
- **Fix:** libmalloc reads `MallocLargeCache` only at process start, so
  `HartreeFock::reexec_without_malloc_large_cache` (`src/base/malloc_env.h`),
  called first in the `hartree-fock` and `planck-dft` `main`s, sets it to 0
  and re-execs. A value the user set is kept, a failed exec continues
  unchanged, and `planck-mpi` does not call it (re-exec under an MPI
  launcher is unsafe). It is a no-op off macOS. glibc serves blocks above
  its mmap threshold (at most 32 MiB) with `mmap` and returns them on
  `free`; that is from glibc's documented defaults, not measured on Linux.
- **Result**, water cc-pVTZ `ccsd(t)`: peak footprint 624 → 262 MiB, end
  footprint 624 → 187 MiB. Energy identical. Wall time, 3 runs each,
  alternated: 5.82–5.86 s with the cache, 5.87–5.92 s without.

### 4. Free the thread-local ERI scratch — done

- **Before:** grow-only thread-local scratch (OS `_eri_scratch`, HGP
  `g_hgp_scratch` and `g_hgp_hoist_comp_scratch`, Rys `g_rys_scratch`)
  stayed at its largest quartet's size for the whole run. That was ~140 MiB
  over six threads on the dimer in cc-pVTZ.
- **Change:** each engine has `release_thread_scratch()`. The dispatcher
  `_compute_2e` (`src/integrals/base.h`), the one-shot full-tensor build
  behind the SCF, `ensure_eri` and TDDFT, frees it in a short
  `#pragma omp parallel` after the build, so each pool thread frees its own.
  A thread the pool did not reuse only keeps its scratch, as before. Direct
  Fock builds do not go through `_compute_2e` and keep their scratch across
  SCF iterations.
- **Result**, water cc-pVTZ `ccsd(t)`: live bytes ~130 MiB lower from the
  tensor build on, with no `EriScratch` blocks left at the peak. Energy
  identical.
- **Regression** (steps 3b and 4 together): core suite 178/178 on
  `build-default`.

### 5. Free the AO tensor once the CC blocks exist — open

- The spherical AO tensor (nb⁴, 1.45 GB on the dimer in cc-pVTZ) is not
  read by the CCSD iterations, and since step 3 not by (T) either: (T)
  takes the CCSD's integral blocks instead of rebuilding them.
- Check every later reader before freeing it (post-CC properties,
  checkpoint, anything calling `ensure_eri` after the CC). If one exists,
  free it only on the `ccsd(t)` path.

### 6. Direct spherical ERI build — open

- The SCF's Cartesian→spherical transform holds the Cartesian tensor plus
  the first quarter-transformed one, nc⁴(1 + f) with f = ns/nc, ≈ 4.3 GB
  on the dimer in cc-pVTZ. Step 3a measured it as the largest live-tensor
  peak of the whole run (4.26 GB at ~23 s).
- Transforming each Cartesian shell-quartet block to spherical as it is
  computed never holds the Cartesian tensor; the peak is the ns⁴ output
  (1.45 GB).
- **Needs step 3a** (done), which showed this peak binds now; no need to
  wait for step 7.

### 7. Re-measure the ladder — after 3b, 4, 5, 6

- Same rungs, footprint-watched (`vmmap` "Physical footprint (peak)"), and
  timed in a separate run without the watchdog, which pauses the process.
- Add a PySCF CCSD(T) reference for the dimer in cc-pVTZ if it fits there.
- This decides whether step 8 or 9 comes next.

### 8. Reshape the (T) kernels — conditional

- The block kernels loop one term at a time through generic accessors; they
  are not BLAS-shaped. Each (a, b, c) block is o³·(o + v) work over 30
  terms; an occupied-index GEMM per term is the candidate.
- **Needs step 7** for clean timings at size. (T) was 42–57 % of the CC
  time on the measured rungs.
- Gates: `ccsd_t_oracle_diff` and bitwise thread invariance, both
  unchanged.

### 9. CCSD time levers — conditional

- Profile first. Known levers, unmeasured at rank 2: the doubles residual is
  symmetric under (ia)(jb) (5.6e-16), an order-2 group the
  external-symmetry fold could use; loop order in the generated nests (up to
  1.7× in a one-nest benchmark, `docs/CCGEN_GENERATED_CC_COST.md`).
- **Needs step 7.** A hand-written spatial RCCSD (PySCF `rccsd` form) is
  the fallback only if the generated route loses by a factor that grows
  with size.

Beyond these: build the MO blocks from a direct or RI transform without
storing the AO tensor at all. RI-CCSD would be its own scope.

## Evidence

### Hand-written against generated RCCSD

HEAD `a1692526`, 4 threads, 3 repeats each, one run at a time; whole-run
wall and peak RSS. The generated column is the same in a default build and
a fully dressed one.

| | hand-written RCCSD + (T) | generated RCCSD + (T) |
|---|---|---|
| C1 H2O2 6-31G | 32.0–32.4 s, 83–86 MiB, 22 iter | 0.54–0.58 s, 21 MiB, 26 iter |
| water cc-pVDZ spherical | 14.5–14.8 s, 100–106 MiB, 15 iter | 0.30–0.31 s, 37–40 MiB, 19 iter |

E(T) agrees to every printed digit (E_corr to 1e-10 on H2O2), and both
totals equal the PySCF-pinned gates. Plain `ccsd` still runs the
hand-written RCCSD.

### The ladder

`build-default`, spherical unless marked, conventional SCF, 4 threads, one
run at a time. Small rungs: peak RSS from `/usr/bin/time -l`. Dimer
cc-pVTZ: physical footprint from `vmmap`. "SCF only" is the same input
without `correlation`.

| Rung | nb sph / cart | o / v | CCSD iters × s/iter | (T) |
|---|---|---|---|---|
| water cc-pVDZ | 24 / 25 | 5 / 19 | 19 × 0.010 s | 0.04 s |
| water cc-pVTZ | 58 / 65 | 5 / 53 | 16 × 0.188 s | 2.2 s |
| water dimer cc-pVDZ | 48 / 50 | 10 / 38 | 20 × 0.457 s | 4.2 s |
| water dimer cc-pVTZ | 116 / 130 | 10 / 106 | 18 × 15.9 s (paging) | 379 s (incl. rebuild) |

| Peak memory | before step 1 | after step 1 | after step 2 |
|---|---|---|---|
| water cc-pVTZ, Cartesian, SCF only | 466 MiB | 330 MiB | |
| water cc-pVTZ, SCF only | 447 MiB | 453 MiB | |
| water cc-pVTZ, `ccsd(t)` | 699 MiB | 686 MiB | 661 MiB |
| water dimer cc-pVDZ, SCF only | 165 MiB | 122 MiB | |
| water dimer cc-pVDZ, `ccsd(t)` | 257 MiB | 265 MiB | 227 MiB |
| water dimer cc-pVTZ, `ccsd(t)` | did not run | 6.7 GB footprint, stopped in (T) | completes, 5.5 GB footprint |

Direct SCF did not help before step 1 (water cc-pVTZ 672 MiB, dimer
cc-pVDZ 294 MiB, same energies): the CC setup (`ensure_eri`) rebuilds the
tensor through the same path.

### Profile: where the peak is

An allocator interposed with `DYLD_INSERT_LIBRARIES` recorded every live
allocation ≥ 1 MiB, with its backtrace, at the moment their total peaked.
Energies were unchanged under it. Water cc-pVTZ `ccsd(t)` after step 1,
421 MiB live in 12 allocations:

| MiB | Object | Scales as |
|---|---|---|
| 86.3 | spherical AO ERI tensor (`calculator._eri`) | nb⁴ |
| 78.9 | `transform_eri` intermediate inside `build_tensor_cc_block_cache` | nb³·v |
| 72.1 | `transform_eri` intermediate, alive at the same time | nb²·v² |
| 60.2 | `vvvv` block (`TensorCCBlockCache`) | v⁴ |
| ~112 | six thread-local `EriScratch` buffers from the ERI build | system-independent |
| ~11 | smaller CC blocks | |

The dimer cc-pVDZ (141 MiB, 18 allocations) has the same shape. Scaled to
the dimer in cc-pVTZ these are 1.45 + 1.32 + 1.21 + 1.01 GB ≈ 5 GB at once,
consistent with the 6.7 GB footprint before step 2. Step 2 removed the two
intermediates.

### Step 2 A/B

Both binaries from one `build-default` tree, alternated, 3 repeats each,
4 threads. Differences are within ±0.03 s apart from one cold first
repeat; every energy is identical.

| Case | unbatched wall | batched wall | unbatched RSS | batched RSS |
|---|---|---|---|---|
| water cc-pVTZ `ccsd(t)` | 7.07, 6.05, 6.05 s | 6.60, 6.08, 6.07 s | 667–701 MiB | 657–659 MiB |
| water dimer cc-pVDZ `ccsd(t)` | 13.24, 13.32, 13.28 s | 13.60, 13.29, 13.33 s | 267–268 MiB | 224–226 MiB |
| water cc-pVTZ RMP2 | 0.91, 0.83, 0.82 s | 0.85, 0.85, 0.85 s | 452–473 MiB | 439–474 MiB |
| water cc-pVTZ CASSCF(4,4) | 1.41, –, 1.48 s | 1.41, –, 1.50 s | 184 MiB | 184 MiB |

### The dimer cc-pVTZ runs

- **After step 1:** got through the SCF (20 iterations, 1448.5 MB
  spherical tensor) and the CCSD (~12.1 s per iteration), stopped in (T)
  at 7 minutes; footprint peak 6.7 GB, ~5 GB swapped or compressed.
- **After step 2:** completed. Footprint 4.8 GB through the CCSD, 5.5 GB
  peak during (T)'s rebuild. About 3.5 GB stayed swapped out from the
  SCF-phase transform peak on, with 0.1–1.5 GB resident, while the CPU ran
  at 300–400 %: the CCSD and the (T) kernels do not touch the paged AO
  tensor, and (T) does not read `vvvv`.

## Findings

- **Measure footprint, not RSS, near the RAM limit.** On the step-1 dimer
  run resident memory read 0.1–1.3 GiB while the footprint was 5–6.7 GB:
  macOS compresses and swaps pages out of RSS. An RSS watchdog never fired,
  and `/usr/bin/time`'s max-RSS understates a paging run the same way. The
  small rungs ran far from the limit, so their numbers stand.
- **`vmmap` pauses the process it inspects.** A watchdog polling it every
  2 s inflated wall times (water cc-pVTZ 6.3 → 7.9 s), so footprint
  watching and timing must be separate runs.
- **The spin-orbital RCCSD was the first limit.** `run_rccsd` works in 2o
  occupied and 2v virtual spin orbitals: 16 v⁴ `vvvv`, 16 o²v² t2, and a
  64 o²v⁴ leading contraction. Moving `ccsd(t)` to the generated spatial
  RCCSD removed it.
- **The generated route's speed is the dressing.** Undressed, the rank-2
  kernel ran the same 26 iterations to the same energy in 45.5 s on C1
  H2O2, slower than the hand-written route; dressing only rank 2
  (`--dressing derived`) in the same default build gave 0.54 s. The derived
  dressing makes every residual term a binary contraction. So rank 2 is
  emitted dressed in every build, by its own generator call; default and
  fully dressed builds emit byte-identical rank-2 files and time the same,
  and the hand-written `ccsd` control is 32.2 s in both.
- **The generated route's first timing was slower than hand-written**
  (55.4 s against 32.1 s on C1 H2O2), before fixed-rank amplitude views,
  factored builders and threaded kernels. Generated and hand-written are
  different solvers, so only end-to-end ratios are meaningful.
- **The spherical transform freed two of its intermediates already.** Its
  peak before step 1 was the Cartesian input plus `t1` and `t2`,
  nc⁴(1 + f + f²) ≈ 2.7 nc⁴, not all four tensors as an earlier version of
  this doc said.
- **The generated route had printed the wrong label** (`Total RCCSD
  Energy` for a total including (T)); the driver now prints
  `Total RCCSD(T) Energy`.

## Not in this scope

RI-CCSD and RI-(T), frozen core, MPI distribution of the CC solve, and the
generated rank-3 CCSDT's cost (`docs/CCGEN_GENERATED_CC_COST.md`).
