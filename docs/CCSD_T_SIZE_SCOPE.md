# What limits the size of a CCSD(T) calculation, and what removes it?

Scope, opened 2026-11-20. Nothing here is built or measured yet. It expires
when the limit is moved; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 1). Parent scope:
`docs/CCSD_T_SCOPE.md` ("What remains", step 3).

## Why it matters

`correlation ccsd(t)` equals PySCF, but on small molecules only. The
largest gate, C1 H2O2 6-31G (o = 9, v = 13 spatial), takes 31 s, and
almost all of it is the CCSD:

| C1 H2O2 6-31G, 4 threads | |
|---|---|
| RCCSD (hand-written, spin-orbital) | 22 iterations × 1.39 s ≈ 30.6 s |
| (T) (batched) | 0.27 s |

The (T) is therefore not the limit at these sizes. Users want CCSD(T) on
something like a water dimer in cc-pVTZ.

## What the hand-written RCCSD costs, on paper

`run_rccsd` (`src/post_hf/cc/ccsd.cpp`) works in spin orbitals: 2o
occupied and 2v virtual. It stores seven antisymmetrized blocks
(`SpinOrbitalBlocks`, oooo through vvvv) and spin-orbital t1/t2. The
spatial equations need far less:

| | spin-orbital (now) | spatial |
|---|---|---|
| `vvvv` storage | (2v)⁴ = 16 v⁴ | v⁴ |
| t2 storage | (2o)²(2v)² = 16 o²v² | o²v² |
| leading contraction (`vvvv` × τ) | (2o)²(2v)⁴ = 64 o²v⁴ | o²v⁴ |

At water cc-pVTZ (o = 5, v = 53), the spin-orbital `vvvv` is 1.0 GB against
63 MB spatial. That 16× in memory and 64× in leading FLOPs is the size
limit. It is an upper bound on the gain: the hand-written code may already
skip zero spin blocks inside its loops. Step 0 measures it.

The integral side may bind first. Checked in the code:
- `prepare_rccsd` calls `build_mo_block_cache`, which takes the conventional
  AO ERI tensor (`ensure_eri`, nb⁴) and also keeps a full MO-basis (pq|rs)
  tensor (`MOBlockCache::full`, n_mo⁴) "for the pedagogical path".
- (T) then builds a second, spatial block cache
  (`build_tensor_cc_block_cache`, through
  `prepare_generated_arbitrary_order_state`) from the same AO tensor.

So a `ccsd(t)` run holds at least two nb⁴-sized tensors at once. At
nb = 150 each is 4.1 GB, past this machine's 8 GiB together. Step 0 also
finds out whether these bind before the amplitudes do.

## Routes

### A. Generated spatial RCCSD (recommended to try first)

The spatial rank-2 kernel already exists. With
`-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON`, ccgen emits
`ccsd_arbitrary_planck_generated.cpp`: a spatial CCSD residual on
`ArbitraryOrderRCCAmplitudes`. Nothing runs it, because the generated floor
is 3 (`generated_kernel_registry.cpp`, `rccgen.cpp`). The arbitrary-order
harness already solves rank 2 for UCC (`ucc2`), so that is proven. What
route A needs:
- **Floor.** Lower the RCC generated floor to 2 when the companion is
  built, and include it in the registry.
- **Keyword.** Add a rank-2 RCC keyword. The teaching guide says "there is
  no generated rank-2 RCC keyword, because hand-written RCCSD covers it — a
  generated `cc2` would have no consumer". (T) is now that consumer, so the
  premise has changed.
- **Feeding (T).** Point `ccsd(t)` at the generated route.
  `compute_rccsd_t_energy` already takes spatial `ArbitraryOrderRCCAmplitudes`,
  so no projection step is needed.
- **Build cost.** Decide whether the companion is generated in every build,
  as the (T) TU is (84 KB), or only with the lower-ranks option.

The known risk is the generated route's cost.
`docs/CCGEN_WHY_GENERATED_IS_SLOW.md` sets the terms:
- The generated and hand-written paths are different solvers (dense versus
  packed storage, different intermediates and iteration counts). Their
  wall-clock ratio is an end-to-end cost, never "the generated kernel is N×
  slower". The rank-3 CCSDT end-to-end gap was 93–151× with dressing, on BH3
  and CH4.
- Generated-kernel OpenMP has been on by default since 2026-11-20
  (`PLANCK_CC_OMP_COLLAPSE`, default 3; `docs/CCGEN_CC_OPENMP.md`).
- Rank 2 has never been measured, generated against hand-written.

Until 2026-11-20 the generated kernels had no `#pragma omp`, so `ucc2`
and route A ran on one core. With the option wired through CMake and on by
default, `oh_uccsd_t_631g` went from 12.25 s to 3.59 s at 4 threads, with
results bitwise unchanged. Even threaded, spherical water⁺ cc-pVDZ UCCSD
takes 9.4 s per iteration: the generated route's cost is real and has to
be measured against the hand-written one before route A is chosen.

A 64× FLOP reduction could be eaten by that gap. The answer is a profile
and a measurement, not a model: the profile-first rule in that doc.

### B. Hand-written spatial RCCSD

This is the PySCF `rccsd` form: spatial t2 with the closed-shell spin sum
written out. It would be fast per operation, but it is new hand-written
solver code, which goes against the project's direction (generated kernels
replace hand-written ones). Take it only if A is measured too slow.

### C. The (T) kernel

The block kernels loop one term at a time through generic accessors; they
are not BLAS-shaped. Each (a,b,c) block is o³·(o + v) work over 30 terms.
Reshaping them (an occupied-index GEMM per term) only matters once the CCSD
fits larger systems. Measure (T)'s share after A or B first.

## Steps

### 0. Measure the current limit

Ladder: water cc-pVDZ, water cc-pVTZ, water dimer cc-pVDZ, water dimer
cc-pVTZ, `ccsd(t)` at 4 threads. For each, record:
- peak RSS, split into AO ERI, MO blocks and amplitudes (from the memory
  report the CC setup prints, or by instrumenting);
- RCCSD seconds per iteration and iteration count, recorded separately;
- (T) seconds.

A run that does not fit in 8 GiB is recorded as "does not run", not left to
swap (`benchmark_ladder_design` memory). Larger rungs go to the cluster.

This step decides which object binds: the AO tensor, the MO blocks, or the
spin-orbital amplitudes and blocks.

### 1. Route A on the same ladder

Lower the floor and add the keyword, then compare:
- **Energy.** The generated rank-2 E_corr must equal the hand-written RCCSD
  to every printed digit on the three `*_rccsd_t_*` inputs and the spherical
  case.
- **Cost.** Seconds per iteration, iteration count and peak RSS, rung by
  rung.

Decide from the numbers. If A wins at the larger rungs, `ccsd(t)` switches
to it. If A loses by a constant that grows with size, open B.

### 2. If an nb⁴ tensor binds

A spatial RCCSD does not help if the integrals bind first. Route A already
drops the hand-written path's `MOBlockCache::full`, because the generated
path uses only the seven `TensorCCBlockCache` blocks. The AO tensor stays.
Beyond that, the work is to build the MO blocks from a direct or RI
transform without storing the AO tensor. RI-CCSD is out of scope here; it
would be its own scope. Record the limit and stop.

### 3. (T) BLAS shaping

Only if step 1 moves the limit enough that (T) becomes a measurable share.
Gate: the batched-vs-full-tensor check (`ccsd_t_oracle_diff`) and thread
invariance, both unchanged.

## Not in this scope

RI-CCSD and RI-(T), frozen core, MPI distribution of the CC solve, and the
generated rank-3 CCSDT's scaling defect (its own scope).
