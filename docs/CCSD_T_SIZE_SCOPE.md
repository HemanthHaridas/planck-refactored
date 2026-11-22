# What limits the size of a CCSD(T) calculation, and what removes it?

Scope, opened 2026-11-20. The first limit, the hand-written spin-orbital
RCCSD, is removed. The next one is not yet measured. Status lives in
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

**Not yet measured:** how large a `ccsd(t)` now runs, and which object
binds first. That is step 0 below.

## What may bind next

The generated route stores spatial t1/t2 and the seven spatial MO blocks
(`TensorCCBlockCache`, `vvvv` = v⁴), built from the conventional AO ERI
tensor (`ensure_eri`, nb⁴) through `prepare_generated_arbitrary_order_state`.
The hand-written path's extra full MO tensor (`MOBlockCache::full`, n_mo⁴)
is no longer on the `ccsd(t)` path.

So the candidates are:
- **The AO ERI tensor**, nb⁴: 4.1 GB at nb = 150, half of this machine's
  8 GiB.
- **`vvvv`**, v⁴: 63 MB at water cc-pVTZ (v = 53), 1.2 GB at v = 110.
- **CCSD time.** The leading contraction is o²v⁴ per iteration.

## Steps

### 0. Measure the limit on the generated route

Ladder: water cc-pVDZ, water cc-pVTZ, water dimer cc-pVDZ, water dimer
cc-pVTZ, `ccsd(t)` at 4 threads. For each, record:
- peak RSS, split into AO ERI, MO blocks and amplitudes (from the memory
  report the CC setup prints, or by instrumenting);
- CCSD seconds per iteration and iteration count, separately;
- (T) seconds.

A run that does not fit in 8 GiB is recorded as "does not run", not left to
swap. Larger rungs go to the cluster. This step decides which object binds.

### 1. If the AO tensor binds

Build the MO blocks from a direct or RI transform without storing the AO
tensor. RI-CCSD would be its own scope. Record the limit and stop.

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
