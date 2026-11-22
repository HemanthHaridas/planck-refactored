# What limits the size of a CCSD(T) calculation, and what removes it?

Scope, opened 2026-11-20. The first limit, the hand-written spin-orbital
RCCSD, is removed. The next one is measured: copies of the AO ERI tensor
(step 0). Status lives in
`vault/Status/Open Work.md` (priority 1). Parent: `docs/CCSD_T_SCOPE.md`.

## Where it stands (2026-11-22)

`correlation ccsd(t)` runs the generated spatial RCCSD (`cc2`), which is
built, and dressed, in every build. Against the hand-written spin-orbital
RCCSD it replaced (HEAD `a1692526`, 4 threads, 3 repeats each, run one at a
time; whole-run wall and peak RSS; the generated column is the same in a
default build and a fully dressed one):

| | hand-written RCCSD + (T) | generated RCCSD + (T) |
|---|---|---|
| C1 H2O2 6-31G | 32.0–32.4 s, 83–86 MiB, 22 iter | 0.54–0.58 s, 21 MiB, 26 iter |
| water cc-pVDZ spherical | 14.5–14.8 s, 100–106 MiB, 15 iter | 0.30–0.31 s, 37–40 MiB, 19 iter |

- 48–59× faster end to end and 2.5–4× less memory, despite 4 more
  iterations. Per iteration, about 1.45 s against 0.02 s on H2O2.
- E(T) agrees to every printed digit (E_corr to 1e-10 on H2O2), and both
  totals equal the PySCF-pinned gates.
- (T) is 0.06 s of the 0.54 s on H2O2.

Plain `ccsd` still runs the hand-written RCCSD.

**What binds next is the AO integral handling, not the CC.** Step 0
measured it; step 1 (in progress, uncommitted) removes the redundant copies
of the AO ERI tensor. Whether spherical water dimer / cc-pVTZ now fits in
8 GiB is being measured.

## Where it stands after step 1 (2026-11-22, in progress)

`build-default`, spherical unless marked, conventional SCF, 4 threads, one
run at a time; peak RSS from `/usr/bin/time -l`. "SCF only" is the same
input without `correlation`.

| Run | before step 1 | after step 1 |
|---|---|---|
| water cc-pVTZ, Cartesian, SCF only | 466 MiB | 330 MiB |
| water cc-pVTZ, SCF only | 447 MiB | 453 MiB |
| water cc-pVTZ, `ccsd(t)` | 699 MiB | 686 MiB |
| water dimer cc-pVDZ, SCF only | 165 MiB | 122 MiB |
| water dimer cc-pVDZ, `ccsd(t)` | 257 MiB | 265 MiB |
| water dimer cc-pVTZ, `ccsd(t)` | does not run | does not run: footprint peak 6.7 GB, paged (stopped in (T)) |

- **The Cartesian SCF** drops by exactly one tensor (136 MiB against
  142.8 MB).
- **The spherical cc-pVTZ SCF does not move.** Its peak is not the
  transform; a malloc-level profile is next (an interposed allocator that
  records every live allocation ≥ 1 MiB at the peak).
- **CC peaks barely move**, so on these rungs the CC phase, not the SCF,
  sets the whole-run peak.
- **Energies are unchanged** on every rung, and 46 spherical and Cartesian
  HF/post-HF regression cases pass.
- **The dimer in cc-pVTZ still does not fit.** It got through the SCF
  (20 iterations, 1448.5 MB spherical tensor) and the CCSD (~12.1 s per
  iteration) and was stopped in (T) at 7 minutes. Its physical footprint
  peaked at 6.7 GB, about 5 GB of it swapped or compressed, so its timings
  are not usable. Which phase set that peak is not known yet.
- **Measure footprint, not RSS.** Resident memory read 0.1–1.3 GiB on that
  run while the footprint was 5–6.7 GB: macOS compresses and swaps pages
  out of RSS. The watchdog polled RSS and never fired, and `/usr/bin/time`'s
  max-RSS understates a paging run the same way. The smaller rungs ran far
  from the limit, so their numbers stand; larger rungs need the footprint
  (`vmmap --summary`, "Physical footprint (peak)").

### Step 1 changes

- **One SCF copy.** `build_conventional_eri` (`src/scf/scf.cpp`) builds
  the tensor straight into `calculator._eri` for RHF, UHF and ROHF, replacing
  three copies of the same block. Each SCF reads it through a reference, so
  the local copy and the old `calculator._eri = eri` copy are gone, and a
  previous geometry's tensor is freed before the new one is built.
- **The transform can consume its input.** `transform_eri_cart_to_sph`
  gained an rvalue overload that frees the Cartesian tensor after the first
  quarter step; both overloads share one implementation. The SCF and
  `ensure_eri` move into it. The per-iteration full-symmetry skeleton
  callers keep the `const&` overload, which does not copy.

## Step 0 result: the ladder

Same setup, before step 1.

| Rung | nb sph / cart | o / v | SCF only | + CCSD | + (T) | CCSD iters × s/iter | (T) |
|---|---|---|---|---|---|---|---|
| water cc-pVDZ | 24 / 25 | 5 / 19 | — | — | 37 MiB | 19 × 0.010 s | 0.04 s |
| water cc-pVTZ | 58 / 65 | 5 / 53 | 447 MiB | 650 MiB | 699 MiB | 16 × 0.188 s | 2.2 s |
| water dimer cc-pVDZ | 48 / 50 | 10 / 38 | 165 MiB | 258 MiB | 257 MiB | 20 × 0.457 s | 4.2 s |
| water dimer cc-pVTZ | 116 / 130 | 10 / 106 | did not run (projected ~6–9 GB) | | | | |

Whole-run wall: 0.35 s, 6.4 s, 13.7 s. Direct SCF does not help (water
cc-pVTZ 672 MiB, dimer cc-pVDZ 294 MiB, same energies): the CC setup
(`ensure_eri`) rebuilds the tensor through the same path.

**The copies found.** Measured against a direct Cartesian SCF on water
cc-pVTZ (149 MiB, no tensor):
- **`run_rhf` kept the tensor twice.** It built `eri` locally and copied
  it into `calculator._eri` for post-HF use, and both lived through the
  whole SCF. The Cartesian conventional SCF peaked at 466 MiB.
- **The spherical transform held its input to the end.**
  `transform_eri_cart_to_sph` (`src/basis/spherical.cpp`) already freed
  `t1` and `t2` as it went, but took the Cartesian input by `const&`, so
  its peak was the input plus `t1` and `t2` during step 2:
  nc⁴(1 + f + f²), f = ns/nc, about 2.7 nc⁴. (An earlier version of this
  doc said all four tensors were held at once; the code frees two.)

The CC itself adds about 200 MiB on water cc-pVTZ and 90 MiB on the dimer
cc-pVDZ, and (T) about 50 MiB.

**Time.** (T) is no longer negligible: 42 % of the CC time on the dimer
cc-pVDZ (4.2 s against 9.1 s of CCSD) and 42 % on water cc-pVTZ (2.2 s
against 3.0 s). So step 3 ((T) kernel shaping) moves up once memory is
fixed.

## What could bind

The generated route stores spatial t1/t2 and the seven spatial MO blocks
(`TensorCCBlockCache`, `vvvv` = v⁴), built from the conventional AO ERI
tensor (`ensure_eri`, nb⁴) through `prepare_generated_arbitrary_order_state`.
The hand-written path's extra full MO tensor (`MOBlockCache::full`, n_mo⁴)
is no longer on the `ccsd(t)` path. The candidates were:
- **The AO ERI tensor**, nb⁴: 4.1 GB at nb = 150. Measured: this binds
  first, through its extra copies (above).
- **`vvvv`**, v⁴: 63 MB at water cc-pVTZ (v = 53), 1.2 GB at v = 110.
- **CCSD time.** The leading contraction is o²v⁴ per iteration.

## Steps

### 0. Measure the limit on the generated route — done

See "Step 0 result" above. The AO tensor binds, through its copies.

### 1. The AO tensor binds: drop the extra copies — in progress

The two cheap changes are made (above, "Step 1 changes"). Remaining:
- **What sets the peak: the CC integral-block build** (profiled
  2026-11-22, below). Next, in order of memory saved on the dimer in
  cc-pVTZ:
  1. **Batch `transform_eri` for the CC blocks.** Building `vvvv` keeps
     full nb³·v and nb²·v² intermediates (1.32 + 1.21 GB there). Batched
     over the leading index they become slabs.
  2. **Let (T) reuse the CCSD's blocks.** `compute_rccsd_t_energy`
     (`ccsd_t.cpp:334`) calls `prepare_generated_arbitrary_order_state`
     again and rebuilds every block from the AO tensor: repeated work, and
     a second build's intermediates.
  3. **Release the thread-local `EriScratch`** after the ERI build:
     ~112 MiB that stays alive for the whole run, whatever the system size.
  4. **Then the AO tensor** (nb⁴, 1.45 GB) could be freed once the blocks
     exist, if nothing later reads it.

### Profile: where the peak is

An allocator interposed with `DYLD_INSERT_LIBRARIES` recorded every live
allocation ≥ 1 MiB, with its backtrace, at the moment their total peaked.
Energies are unchanged under it. Water cc-pVTZ (nb = 58, o = 5, v = 53),
`ccsd(t)`, 421 MiB live in 12 allocations:

| MiB | Object | Scales as |
|---|---|---|
| 86.3 | spherical AO ERI tensor (`calculator._eri`) | nb⁴ |
| 78.9 | `transform_eri` intermediate inside `build_tensor_cc_block_cache` | nb³·v |
| 72.1 | `transform_eri` intermediate, alive at the same time | nb²·v² |
| 60.2 | `vvvv` block (`TensorCCBlockCache`) | v⁴ |
| ~112 | six thread-local `EriScratch` buffers from the ERI build | system-independent |
| ~11 | smaller CC blocks | |

The dimer cc-pVDZ (141 MiB, 18 allocations) has the same shape: AO tensor
40.5, `transform_eri` intermediates 32.1 and 25.4, `vvvv` 15.9 MiB. So the
peak is the CC block build, not the SCF. Scaled to the dimer in cc-pVTZ
(nb = 116, v = 106) that is 1.45 + 1.32 + 1.21 + 1.01 GB ≈ 5 GB at once,
consistent with the 6.7 GB footprint once heap overhead and the rest are
added. The SCF's transform peak there is lower, ~4.3 GB.
- **Then decide on a direct spherical build.** Transforming each Cartesian
  shell-quartet block to spherical as it is computed would never hold the
  Cartesian tensor; the peak is the ns⁴ output alone (1.45 GB on the dimer
  in cc-pVTZ). Only if the dimer in cc-pVTZ is still out of reach.

Beyond that, build the MO blocks from a direct or RI transform without
storing the AO tensor. RI-CCSD would be its own scope.

### 2. If CCSD time binds

Profile first. Known levers, unmeasured at rank 2:
- the doubles residual is symmetric under (ia)(jb) (5.6e-16), an order-2
  group the external-symmetry fold could use;
- loop order in the generated nests (summed index innermost and strided;
  up to 1.7× in a one-nest benchmark, `docs/CCGEN_GENERATED_CC_COST.md`).

A hand-written spatial RCCSD (PySCF `rccsd` form) is the fallback only if
the generated route loses by a factor that grows with size; it goes against
the project direction of generated kernels replacing hand-written ones.

### 3. If (T) becomes a measurable share

Reshape the (T) block kernels (an occupied-index GEMM per term; today they
loop one term at a time through generic accessors). Gate: the
batched-vs-full-tensor check (`ccsd_t_oracle_diff`) and thread invariance,
both unchanged.

## Findings

- **The spin-orbital RCCSD was the first limit.** `run_rccsd` works in 2o
  occupied and 2v virtual spin orbitals: (2v)⁴ = 16 v⁴ `vvvv`, 16 o²v² t2,
  and a 64 o²v⁴ leading contraction (1.0 GB of `vvvv` at water cc-pVTZ,
  against 63 MB spatial). Moving `ccsd(t)` to the generated spatial RCCSD
  removed it.
- **The generated route's cost was the risk, and it turned out not to bind.**
  Its first timing, before fixed-rank amplitude views, factored builders and
  threaded generated kernels, was slower than hand-written (55.4 s against
  32.1 s on C1 H2O2, 19.6 s against 14.8 s on spherical water). After those
  changes it is the table above. Generated and hand-written are different
  solvers (dense against packed storage, different iteration counts), so
  only end-to-end ratios are meaningful.
- **The speed is the dressing, measured with only that changed.** The
  table above is a dressed build. In a default build (every option at its
  default), the undressed rank-2 kernel ran the same 26 iterations to the
  same energy in 45.5 s on C1 H2O2, slower than the hand-written route.
  Emitting only rank 2 with `--dressing derived` in that same build brought
  it to 0.54–0.55 s. The derived dressing makes every residual term a binary
  contraction; the plain emit keeps n-ary terms. So the rank-2 companion is
  emitted dressed in every build, by its own generator call, while higher
  ranks in a default build stay undressed. With that, the default and fully
  dressed builds emit byte-identical rank-2 files and run `ccsd(t)` in the
  same time (H2O2 0.54–0.55 s in both; spherical water 0.30–0.32 s), and
  the hand-written `ccsd` control takes 32.2 s in both, so the other build
  options do not move the timing.
- **The generated route had printed the wrong label.** It printed
  `Total RCCSD Energy` for a total including (T), which the runner metric
  `rccsd_t_total_energy` would not parse. The driver now prints
  `Total RCCSD(T) Energy`.

## Not in this scope

RI-CCSD and RI-(T), frozen core, MPI distribution of the CC solve, and the
generated rank-3 CCSDT's cost (`docs/CCGEN_GENERATED_CC_COST.md`).
