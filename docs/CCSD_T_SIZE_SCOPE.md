# What limits the size of a CCSD(T) calculation, and what removes it?

Scope, opened 2026-11-20. Status lives in `vault/Status/Open Work.md`
(priority 1). Parent: `docs/CCSD_T_SCOPE.md`.

## Current result (2026-11-22)

- **The CCSD is no longer the limit.** `correlation ccsd(t)` runs the
  generated spatial RCCSD, 48–59× faster than the hand-written spin-orbital
  one with 2.5–4× less memory.
- **The AO integral handling is the limit.** Copies of the AO ERI tensor
  and full-size transform intermediates, not CC amplitudes, set the peak.
- **Water dimer / cc-pVTZ now completes on this 8 GiB machine**, but only
  just: footprint peak 5.5 GB, with ~3.5 GB of it paged out. Before steps 1
  and 2 it did not fit (6.7 GB, stopped).
- **Which phase sets that 5.5 GB peak is not known.** Step 3 removed
  (T)'s duplicate block build and the peak did not move; the step-3 run
  reached 5.3 GB in its first 73 s, before (T). The largest known transient
  is the SCF's Cartesian→spherical transform (~4.3 GB of tensors, step 6),
  but footprint also counts freed-but-cached heap and compressor accounting,
  so the next measurement is an allocation profile of the dimer itself.
- **(T) is now 42–57 % of the CC time**, so once memory is settled (T)'s
  kernels are the next time lever.

| Water dimer / cc-pVTZ `ccsd(t)` (nb 116 sph / 130 cart, o 10, v 106) | |
|---|---|
| Total energy | −152.7012785674 Eh (no PySCF reference yet) |
| E_corr(CCSD), E(T) | −0.5642828768, −0.0160404999 |
| SCF / CCSD / (T) | 20 iterations / 18 × 15.3 s = 276 s / 354 s (after step 3; 379 s before) |
| Whole run | 661 s (inflated: watchdog pauses and paging) |
| Footprint peak | 5.5 GB, both before and after step 3; phase not known |

## Steps

Done steps first, then open ones in the recommended order. "Needs" lists
the steps that must land first.

| # | Step | State | Needs |
|---|---|---|---|
| 0 | Measure the limit on the generated route | done | — |
| 1 | One SCF copy of the ERI tensor; transform consumes its input | done, `885109ce` | 0 |
| 2 | Batch `transform_eri` over its leading index | done, `25714026` | 0 |
| 3 | (T) reuses the CCSD's integral blocks | done, uncommitted | — |
| 3a | Allocation profile of the dimer in cc-pVTZ: which phase sets the peak | next | 3 |
| 4 | Free the thread-local ERI scratch after the build | open | — |
| 5 | Free the AO tensor once the CC blocks exist | open | 3 |
| 6 | Direct spherical ERI build (per-block transform) | open, conditional | 3a, 7 |
| 7 | Re-measure the ladder, footprint-watched, clean timings | open | 3, 4, 5 |
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

### 3. (T) reuses the CCSD's integral blocks — done, uncommitted

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

### 3a. Allocation profile of the dimer in cc-pVTZ — next

- **Why:** the small-rung profile located the live tensors, but on the dimer
  the footprint also counts freed-but-cached heap and compressor
  accounting, and the peak phase is unknown.
- **How:** the same interposed allocator (every live allocation ≥ 1 MiB,
  with backtraces, at the moment their total peaks), on the dimer, without
  the `vmmap` watchdog. Its live-peak total against the 5.5 GB footprint
  also says how much of the footprint is heap overhead rather than live
  tensors.
- **Decides** between step 6 (if the SCF transform holds the peak) and
  steps 4–5 (if later phases do).

### 4. Free the thread-local ERI scratch — open

- Six thread-local `EriScratch` buffers from the ERI build, ~112 MiB on
  water cc-pVTZ, stay alive for the whole run. Their size is set by the
  basis's angular momentum and contraction, not by the system size.
- Release them after the tensor build (or shrink them on exit from the
  build). Independent of the other steps.

### 5. Free the AO tensor once the CC blocks exist — open

- The spherical AO tensor (nb⁴, 1.45 GB on the dimer in cc-pVTZ) is not
  read by the CCSD iterations or by (T) once step 3 lands.
- **Needs step 3**: today (T)'s rebuild reads it.
- Check every later reader before freeing it (post-CC properties,
  checkpoint, anything calling `ensure_eri` after the CC). If one exists,
  free it only on the `ccsd(t)` path.

### 6. Direct spherical ERI build — conditional

- The SCF's Cartesian→spherical transform holds the Cartesian tensor plus
  the first quarter-transformed one, nc⁴(1 + f) with f = ns/nc, ≈ 4.3 GB
  on the dimer in cc-pVTZ. After steps 3–5 this is the largest transient
  there.
- Transforming each Cartesian shell-quartet block to spherical as it is
  computed never holds the Cartesian tensor; the peak is the ns⁴ output
  (1.45 GB).
- **Needs step 7**: take it only if the re-measured ladder shows the SCF
  transform still binding at the sizes that matter.

### 7. Re-measure the ladder — after 3, 4, 5

- Same rungs, footprint-watched (`vmmap` "Physical footprint (peak)"), and
  timed in a separate run without the watchdog, which pauses the process.
- Add a PySCF CCSD(T) reference for the dimer in cc-pVTZ if it fits there.
- This decides whether step 6, 8 or 9 comes next.

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
