# FCIQMC Scalability

## Current status — 2026-11-08

R0 qualified the HF-to-FCIQMC handoff on the cases below. R1 is measuring
where time and memory go on the current solver. The next decision is whether
the evidence favors a propagation change, a setup change, or a larger
admissible active space. The canonical population, parent-derived RNG, E1
selector, and M1 parallel merge are in production. E4 is still test-only.

The upstream audit began on 2026-11-06 at `22da4338`. The November 8
measurements use `cf74abe2` plus the working-tree handoff and profiling
changes. Earlier timing tables retain their original configurations.

### Upstream changes that affect this investigation

| Change verified in the tree | Consequence for FCIQMC | Required response |
|---|---|---|
| HF stores eigenpairs of the true converged Fock after DIIS (`ea251d62`, `src/scf/scf.cpp`); direct-SCF screening and integral kernels also changed. | MO coefficients and therefore `h_eff`/`ga` can change even when total HF energies are very close. A fixed seed does not imply the same stochastic trajectory for changed integrals. | Separate same-Hamiltonian kernel reproducibility from end-to-end upstream compatibility. Recheck exact FCI and S5 before attributing any moved pin to propagation. |
| Shared `transform_eri` now uses the unit-stride contraction (`bc36af7b`, `src/post_hf/integrals.cpp`). `transform_eri_internal` calls it. | FCIQMC inherits the transform improvement directly; floating-point evaluation changes may also affect its Hamiltonian. | Measure the current transform once per calculation; do not add the CASSCF speedup again as a proposed QMC improvement. |
| Spherical `ensure_eri` constructs and transforms a Cartesian tensor when none is cached (`5da0d7fe`); transform normalization moved to basis read (`e565e4fd`). | Direct spherical SCF now has a post-HF tensor path, but can incur a dense Cartesian allocation followed by a spherical tensor and MO-transform scratch. Fast direct SCF does not make post-HF setup memory-free. | Validate both cached and uncached routes; measure peak setup RSS independently of walker RSS. |
| `build_all_mo_ci_setup` now uses `total_nuclear_charge()` (`100675e4`); `_hcore` includes ECP integrals. | FCIQMC inherits effective-charge electron counting and the ECP one-electron Hamiltonian. R0 exercises both with a two-electron ECP control. | Check effective electron counts, spin split, nuclear energy, and small-space spectra against an independent reference. Avoid counting removed ECP electrons again as an inactive core. |
| CASSCF now has an occupied half transform, core folding, and a much faster orbital solver (`src/post_hf/casscf/{casscf,orbital}.cpp`). | Useful building blocks exist for fixed-orbital active-space FCIQMC. The current all-MO driver does not consume that active-space setup. | Scope extraction of the Hamiltonian preparation only if full-MO limits block the target; do not require a large deterministic CASSCF solve before QMC can start. |
| HF stability/SOSCF and spherical reference handling have changed (`100675e4`, `8745c07a`). | Stability following can change the reference type/orbitals. A final UHF reference has distinct alpha and beta orbitals, while the CI engine consumes one spatial MO set. | The shared setup checks the final reference. R0 covers triplet ROHF and RHF followed into UHF. |

New spherical DFT, TDDFT, and gradient capabilities do not automatically enter
the FCIQMC path: it is dispatched by `src/hf_driver.cpp`. Changes to the shared
integrals and working-basis representation do enter it. Gradient and response
timings are not evidence of faster QMC propagation.

### R0 — Hamiltonian and reference results

FCI and FCIQMC accept a converged RHF or ROHF reference. Both validate the
MO coefficients, overlap and core Hamiltonian against `working_nbasis()`,
and check the dense ERI count. An RHF stability follow that ends in UHF fails
before CI integral preparation. The triplet O₂ case exercises unequal spin
populations with a common ROHF orbital set.

The four new molecular gates use 10,000 target walkers, a 0.001 timestep,
20,000 equilibration and 30,000 sampling steps, seed 20250901 and two threads.
Energies and blocked standard errors are in hartree. The last column identifies
the Hamiltonian on this build.

| System and route | MOs, α/β | Determinants | Exact FCI | Shift ± SE | Projected ± SE | Hamiltonian ID |
|---|---|---:|---:|---:|---:|---|
| Be, spherical 6-31G*, cached | 14, 2/2 | 8,281 | −14.613942546561 | −14.61833 ± 0.0269 | −14.62005 ± 0.00446 | `79831605c5eb80f2` |
| Be, spherical 6-31G*, direct | 14, 2/2 | 8,281 | −14.613942546561 | −14.61748 ± 0.0250 | −14.61839 ± 0.00381 | `cc8c91f608326e28` |
| O₂, Cartesian STO-3G, triplet ROHF | 10, 9/7 | 1,200 | −147.744188578194 | −147.72373 ± 0.0282 | −147.73861 ± 0.00436 | `0fd036a02a465db5` |
| RbH⁸⁺, spherical def2-SVP, Rb ECP | 29, 1/1 | 841 | −8.772519943865 | −8.75497 ± 0.0508 | −8.75916 ± 0.00592 | `4eea2c079072b5b6` |

`tests/pyscf/fciqmc_handoff_references.py` independently builds the integrals
and solves FCI with PySCF 2.13.0 using the repository basis/ECP data. The charged RbH fixture
keeps two explicit electrons after removing the Rb 28-electron core. Its
effective-charge nuclear repulsion is 1.984414540886 Eh. Both Be routes give
the same deterministic energy at printed precision. Direct SCF builds the
spherical ERIs from 15 Cartesian AOs after SCF. Its Hamiltonian differs at
bit level from the cached route, so each has its own fixed-seed trajectory.
All four gates check the nuclear offset, maximum absolute element of
`CᵀSC − I` (observed at most 3.7 × 10⁻¹⁵), the spin sector, both energies
within five blocked standard errors, and absence of the sign-instability flag.

The existing Cartesian H₂/N₂ energy gates and N₂ S5 pins remain in place.
With `fciqmc_profile true`, `phase=hamiltonian` fingerprints the dimensions,
spin counts, nuclear offset and raw doubles of `h_eff` and `ga`.
`phase=result` fingerprints the raw shift and valid projected sample series,
the projected numerator/denominator sums, the final reference determinant
and the final sorted walker population. These are 64-bit FNV-1a fingerprints
for comparing runs. The short N₂ gate compares 1/2/4/8 threads, then changes
the seed and checks that only the result ID changes. Its Hamiltonian ID was
`3d0ae86188394164` and result ID was `af01f83434290758`.
The gate compares IDs within a build. For a future kernel A/B, keep the input
and seed fixed and require matching Hamiltonian IDs before comparing results.

### R1 — rebaseline setup, propagation, and memory separately

On hydrogen fluoride/6-31G (50,000 walkers, four threads), the current
solver took **170.099 s** for one full trajectory. The shared CI Hamiltonian
build took 0.000186 s after conventional SCF, equilibration took 68.133 s
and sampling took 101.955 s. Those phases account for 170.088 s. Sampling
worker time was 52.701 s, merge 24.521 s and diagonal prefill 20.463 s.
Merge includes the pairwise tree. Peak solver RSS over the complete run was
48.5 MiB (macOS child-process `ru_maxrss`). The projected energy was
−100.1155051 ± 0.00841 Eh, against the fixture’s exact −100.1156979 Eh.
Hamiltonian ID `f032498d77f973b5`, result ID `ea152a6472c1789b`.
One four-thread observation cannot establish the HF scaling curve or a change
from the historical median.

On cached spherical Be/6-31G*, two threads gave the lowest median wall time.
Each of three repetitions ran 1, 2, 4 and 8 threads in that order. All twelve
runs had the same Hamiltonian ID, sampling parent count (50,650,414) and
result ID (`db8df0c70201c473`). Times below are seconds.

| Threads | Median wall | Wall range | Sampling worker median | Sampling merge median |
|---:|---:|---:|---:|---:|
| 1 | 12.002 | 11.615–12.017 | 5.675 | 0.942 |
| 2 | 10.296 | 10.150–10.395 | 3.230 | 2.115 |
| 4 | 11.146 | 11.115–11.351 | 2.496 | 3.475 |
| 8 | 14.723 | 14.176–14.793 | 2.623 | 5.406 |

From two to four threads the sampling worker saved 0.734 s and the merge
cost an additional 1.360 s. The small case averaged 1,688 parent determinants
per sampling step. Hydrogen fluoride averaged 48,485. The production merge
opens a parallel loop at each tree level, down to the last pair. A focused
A/B can use the existing sequential pair merge for small levels and retain
the same reduction order. Measure the cutoff on both occupancies before
changing production.

To reproduce the Be ladder from the repository root:

```sh
OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  tests/run_fciqmc_e1_profile_ladder.sh build/hartree-fock \
  tests/inputs/regression/post_hf/be_fciqmc_spherical_631gd.hfinp \
  /tmp/planck-be-scaling 3 '1 2 4 8'
```

Both November 8 benchmarks used macOS 27.2 arm64, g++ 15.2.0 Release
(`-O3 -DNDEBUG`), libgomp OpenMP 4.5, Eigen 3.4.0 with internal parallelism
disabled, and Boys core 1.2.1. `OMP_DYNAMIC=false`, OpenBLAS and Accelerate
thread limits were one. Both fixtures select the OS integral engine, hcore
guess and no symmetry. Binary SHA-256
`8991e03fc53a48db0096bccd0bf819fab363755f0ac54818b110aaa8c8dfa1a5`.
Generated hydrogen fluoride input SHA-256
`72da8718359d16c06f7450afeac39f2bbe322ca681a0f7e3c59f606e08ce4fde`.

`build_s` times the complete post-HF all-MO setup and excludes the diagnostic
hash and orthogonality check. `phase_wall_s` covers each complete propagation
phase, including its existing nested counters. The runner saves the generated
input beside its raw log.

#### Remaining R1 measurements

Reuse `tests/run_fciqmc_scaling_profile.sh`,
`tests/run_fciqmc_cpu_profile.sh`, and the E1 ladder for the remaining measurements.

1. Record source diff and binary hash, compiler flags, Boys-library version,
   basis/ECP files, engine, SCF tolerances, symmetry, cached/direct/RI route,
   generated input, hardware/core topology, and numerical-library thread counts.
   Persist generated inputs; use distinct directories for every repetition.
2. Time input/basis/one-electron setup, SCF, AO ERI materialization, AO-to-MO
   transformation, equilibration, and sampling separately. The current
   `FCIQMC_PROFILE setup_s` measures propagation scaffolding, **not** SCF or
   `build_all_mo_ci_setup`. Reconcile disjoint phase times with total wall time;
   do not double-count nested merge counters.
3. Keep HF/6-31G 50k as the continuity fixture at normal verbosity, with the
   same stochastic settings. Its input pins `engine os`; newer Auto/HGP/Rys
   improvements cannot be inferred from that case. Compare an Auto setup run
   separately and validate its Hamiltonian before interpreting the difference.
4. Refresh 1/2/4/8-thread results with at least three repeats, then the existing
   10k/25k/50k/100k population ladder where informative. Four threads were best
   historically, not necessarily on the current build/host. Track actual live
   occupancy, attempts, rejected/zero spawns, and per-bin work, not just weight.
5. Add one larger, reachable throughput case alongside the measured spherical control.
   Screen timestep stability, reference occupancy, and equilibration before
   long timing runs. Keep correctness, occupancy scaling, and orbital/electron
   scaling as separate experiments.
6. Report peak process RSS plus canonical population capacity, accumulator and
   tagged-merge temporary storage, diagonal-cache size/misses, and retained
   estimator/verbose data. The diagonal memo still retains visited determinants;
   canonicalizing the live population did not bound it. The current driver
   creates a fresh canonical `next_pop` each step: measure vector allocation
   and merge-level temporaries, not the retired map-node allocation cost.

The CPU sampling script currently waits one second before sampling. On a
larger spherical/ECP setup that may profile SCF or integral construction;
identify the sampled phase or trigger sampling at an explicit phase boundary.
The ladder accepts a space-separated list of thread counts for one binary. Alternate separately built binaries
as well when testing E4 or another change. Check probe overhead with profiling
disabled and retain dispersion and host-load observations.

Exit: current phase/memory tables and a decision whether the limiting resource
is setup, worker propagation, merging, or problem admission. Do not fit a scaling
exponent across different electron counts or quote the old wall times as a new
baseline.

### R2 — select a bounded optimization from the refreshed evidence

| Measured limit | Next candidate | Acceptance evidence |
|---|---|---|
| Worker bit selection remains material | Uncontended paired E4 retest; keep production selector unchanged until it wins. | Existing E0/E4 exact oracles, identical current Hamiltonian/trajectory, localized worker gain and end-to-end threshold. |
| Worker matrix elements or sort/fold | Excitation-aware Hamiltonian evaluation or local accumulator work, one at a time. | Current phase plus CPU attribution; shared CI/CASSCF and open-shell checks for Hamiltonian changes. |
| Prefill/cache | Carry a diagonal with each parent and/or bound the cache if measured churn requires it. | No parallel cache writes, unchanged values and trajectory, measured memory/lookup tradeoff. |
| Tagged merge/vector allocation | Reuse bounded vector/tree storage or refine the current level-parallel tree. | Same source-bin association and raw-bit merge oracle; account for peak simultaneous levels and final population. No return to map/sharded-sort proposals without new evidence. |
| Projected estimator | Cache the fixed reference connections/elements after re-anchoring. | Material estimator ledger cost; unchanged validity filtering, ordered sums, and ratio-of-sums result. |
| AO/MO setup | Optimize or reuse shared integral preparation; assess RI as a separate Hamiltonian approximation. | Same-Hamiltonian exact-path comparison; quantify RI error separately from QMC error and preserve deterministic FCI/CASSCF consumers. |

Retain the existing 10% end-to-end / 5% control-regression thresholds or a
demonstrated capacity-enabling memory reduction. Also report propagation-only
gain so an upstream setup improvement is not credited to the QMC kernel.
The E4 1.130x local result is not an end-to-end prediction. Correlated phase
slowdowns suggest host contention, but do not prove it; altered cache/thermal
behavior is another possibility. Resolve that with paired evidence.

### R3 — capacity branch, now grounded in the current CASSCF machinery

If the target is blocked by whole-MO dimensionality rather than step rate,
prioritize a **fixed-orbital active-space Hamiltonian adapter** over another
selector change. This is a separate implementation deliverable in the rescope;
it is not a request to replace the CASSCF orbital optimizer with QMC.

- Reuse/extract `build_occupied_half_transform`, `occupied_two_electron_ao`,
  `transform_occupied_half`, and `compute_core_energy` contracts. Return active
  `h_eff`, `ga`, spin counts, and an explicit core+nuclear offset without
  constructing `CIDeterminantSpace` or solving deterministic CI for the target.
  Use HF orbitals first; optimized orbitals from an affordable smaller CAS are
  optional, not a prerequisite requiring the large calculation already solved.
- Gate zero-core/all-MO equivalence and a small closed-core CAS against
  deterministic CI at the **same fixed orbitals**, including spherical and ECP
  cases. Keep all unpaired electrons active for the supported closed inactive
  core. Validate selected orbital indices and the energy offset explicitly.
- Separate stochastic resource admission from deterministic `ci_max_dim`.
  Audit its input range and dimension arithmetic; retain deterministic FCI's
  allocation guard. The packed 31-orbital limit currently applies to the full
  MO space and would apply to the selected active space after adaptation.
- Account for AO tensor/half-transform memory before promising larger bases:
  the existing half-transform machinery still consumes dense AO integrals.
  Benchmark integral storage and walkers separately; RI still returns dense
  active integrals here and is not a factorized propagator.

The old Cr2 deterministic runtime estimates predate substantial CASSCF/shared
transform work. Re-time affordable deterministic rungs and size the intended
CAS without claiming a new large-space runtime from those small benchmarks.
Only choose the expensive target after this capacity audit.

Deliverables, in order: R0 compatibility evidence; R1 reproducible current
baseline; one evidence-selected R2 prototype **or** an R3 adapter design when
admission is the blocker. Replica throughput, GPU/MPI, stochastic orbital
optimization, and new sampling laws remain outside this first implementation
cycle.

## Purpose

This document records the production FCIQMC propagation design, its historical
performance measurements, and the evidence required before another optimization
is accepted. The November rescope above defines the next work; the records below
explain landed and deferred decisions rather than adding competing work plans.

The propagation optimization scope is one fixed-seed FCIQMC trajectory at a
fixed Hamiltonian. An accepted optimization must
not change its sampling law, support, `p_gen`, deterministic reduction order,
or fixed-seed trajectory. Replica throughput and MPI/GPU execution are separate
projects. Active-space/frozen-core FCIQMC is the conditional R3 capability
branch above, separate from propagation optimization.

## Production architecture

`CanonicalWalkerPopulation` stores the live wavefunction as a sorted vector of
`(DetKey, weight)` entries. This gives deterministic traversal, binary-search
lookup for initiator and estimator queries, and in-place compaction.

One propagation step is:

1. Partition parent determinants into 64 fixed bins by parent hash. The bin
   count does not depend on the OpenMP thread count.
2. Derive each parent's random stream from its determinant. Excitation and
   stochastic-rounding draws therefore do not depend on container layout or
   visitation order.
3. Accumulate each bin privately, then sort and fold each bin in canonical key
   order.
4. Merge the 64 finalized streams through a tagged, balanced pairwise tree.
   Equal keys retain ascending source-bin association at every tree node.
5. Store the sorted result in the next canonical population, compact it, then
   update population control and the projected estimator.

The production merge is
`merge_finalized_spawn_bins_pairwise_parallel` in
`src/post_hf/ci/spawn_accumulator.h`. Independent pair merges at a tree level
run in parallel, but each result is written to a predetermined slot; the tree
shape and floating-point association are unchanged. The sequential pairwise
tree is retained only as a test oracle.

## Correctness contract

The following properties are non-negotiable.

- Parent partitioning, not child partitioning, defines independent work.
- The 64-bin layout and ascending-bin equal-key reduction association are
  invariant across OpenMP thread counts.
- Parent-keyed RNG streams make generated excitations independent of traversal
  order.
- `p_gen` equals the probability for the exact generated excitation; support
  and observed frequency are tested separately.
- A mechanical optimization reproduces the fixed-seed trajectory bitwise. Any
  intentional statistical change needs a new, explicitly documented reference.

`planck-fciqmc-walkers` covers excitation support, `p_gen`, literal sampler
equivalence, fixed-seed reproducibility, canonical-population propagation, and
the parallel-tree raw-bit oracle. `planck-fciqmc-accumulator` covers local
ordered folding and merge behavior. N2 S5 runs at one and four threads are the
fixed-seed integration gate; H2/N2 exact-FCI cases remain statistical controls.

## Historical performance before the November handoff checks

The representative throughput fixture is
`tests/inputs/exploratory/fciqmc/validation/hf_pop_50000.hfinp`: HF/6-31G,
50,000 walkers, 20,000 equilibration steps, and 30,000 sampling steps. Each
accepted figure uses the same seed, timestep, walker target, Hamiltonian, and
step counts. Timings are three-run alternating medians unless stated otherwise.

| configuration | 1 thread wall | 2 thread wall | 4 thread wall | 8 thread wall |
|---|---:|---:|---:|---:|
| map population baseline | 565.88 s | 374.18 s | 284.03 s | 258.41 s |
| canonical population + pairwise merge | 474.54 s | 294.88 s | 203.221 s | 225.07 s |
| E1 rank-select + M1 parallel tree | 403.748 s | — | **163.958 s** | — |

Four threads are the best measured configuration for this workload. This is a
recommendation, not a runtime limit; callers continue to select
`OMP_NUM_THREADS`.

At four threads, M1 changed the phase ledger as follows:

| phase | E1 median | M1 median | change |
|---|---:|---:|---:|
| sampling worker | 50.233810 s | 51.357996 s | +2.2% |
| sampling pairwise merge | 38.811332 s | 24.049039 s | -38.0% |
| full wall | 182.431 s | 163.958 s | -10.1% |

All three M1 runs have identical parent, attempt, raw-record, folded-record,
and final-energy values. The small worker increase is the cost of materializing
tree levels and is more than recovered by the parallel merge.

## Landed optimization decisions

| item | decision | evidence |
|---|---|---|
| Canonical population | landed | removes hash-layout traversal dependence |
| Parent-derived RNG | landed | deterministic trajectory across layout and threads |
| Tagged pairwise merge | landed | raw-bit equality with the heap/reference fold; faster than heap merge |
| E0 literal excitation oracle | landed, test-only | exact excitation, `p_gen`, phase, validity, and RNG-state equality |
| E1 rank selection | landed | 1.112x focused benchmark; 10.2% four-thread full-run wall reduction before M1 |
| M1 level-parallel pairwise tree | landed | sequential-tree raw-bit oracle; N2 S5 1/4-thread gate; 10.1% additional wall reduction |
| E2 table dispatch | rejected | exact supplied-input descriptor oracle, but 0.968x versus the switch |
| E4 byte-table selector | deferred for uncontended A/B | exact local oracle and 1.130x direct win; loaded-host ladder is inconclusive |
| Reused map populations | rejected | bucket layout changes parent traversal |
| PMR node recycling | rejected | negligible end-to-end return |
| Child-hash sharded merge | rejected | second sorting stage loses performance |

E1 replaces a scan-to-the-selected-bit implementation of `nth_set_bit` with
repeated least-significant-set-bit clearing followed by `countr_zero`. It does
not change excitation classes, RNG calls, signs, or `p_gen` arithmetic order.

## Profiling and acceptance workflow

Use the phase ledger to establish the dominant phase, then use a CPU sample to
identify the responsible routine. A stack sample is attribution evidence, not
a performance result. Do not modify a candidate until both identify a concrete
mechanism.

```sh
tests/run_fciqmc_cpu_profile.sh \
  build/hartree-fock \
  tests/inputs/exploratory/fciqmc/validation/hf_pop_50000.hfinp \
  1 30 /private/tmp/fciqmc-profile

tests/run_fciqmc_cpu_profile.sh \
  build/hartree-fock \
  tests/inputs/exploratory/fciqmc/validation/hf_pop_50000.hfinp \
  4 30 /private/tmp/fciqmc-profile
```

An accepted change requires:

1. an isolated, deterministic oracle for the altered operation;
2. walker and accumulator invariant tests plus fixed-seed N2 S5 at one and
   four threads;
3. H2/N2 statistical regression controls when sampling behavior is involved;
4. three alternating representative fixture runs; and
5. at least 10% end-to-end improvement beyond noise, no unexplained control
   regression above 5%, or a documented memory reduction that enables a
   previously infeasible calculation.

## Post-M1 attribution and next decision

Fresh 30-second one- and four-thread CPU samples were collected after M1 on
the representative fixture. Their ledgers preserve the expected trajectory
counts. At four threads, sampling spends `49.528 s` in worker propagation and
`23.355 s` in pairwise merging; M1 therefore remains effective, but worker
propagation is again the dominant scalable cost.

In the one-thread sample, `draw_excitation` accounts for 7,135 of 16,064
propagation-stack samples (44.4%); at four threads it accounts for 3,544 of
8,320 (42.6%). `slater_condon_element` and `SpawnAccumulator::finalize()` /
sorting remain visible secondary costs. This authorizes an excitation-generator
attribution benchmark, but not a production dispatch rewrite: the sampled
frame still includes rank selection, random draws, orbital updates, parity,
and class selection.

### E2 — decomposed excitation-generator benchmark

Build a deterministic, test-only benchmark that measures the current literal
generator and a candidate dispatch mapping separately from orbital updates,
parity, and RNG generation. Both paths must consume the same supplied class
and rank values and produce exactly the same excitation descriptor and
`p_gen`. The benchmark must report a digest and per-component times over the
same closed-shell, open-shell, and edge active-space corpus used by E0.

Only if that benchmark attributes a material fraction to dispatch may a
branch-only production candidate be written. It must preserve class order, RNG
calls, signs, and `p_gen` operation order, then pass the normal exact and
full-trajectory gates.

**Closed as a non-candidate (2026-10-20):** `planck-fciqmc-walkers` constructs
plans with a supplied non-empty class and within-class rank. A literal `switch`
and a table-dispatch candidate receive the same plans; neither path draws RNG,
selects orbitals, evaluates parity, or updates a determinant. The invariant
requires raw descriptor and `p_gen` equality, then materializes both results
to confirm equal excitations. The opt-in benchmark is:

```sh
PLANCK_FCIQMC_DISPATCH_BENCHMARK=1 build/planck-fciqmc-walkers
```

`PLANCK_FCIQMC_DISPATCH_BENCHMARK_DRAWS` and
`PLANCK_FCIQMC_DISPATCH_BENCHMARK_ROUNDS` override the defaults. Five
alternating 10,000,000-draw rounds gave `0.033586 s` for the literal switch
and `0.034704 s` for table dispatch: **0.968x**, with a matching digest. The
indirect call loses, so production retains the switch and dispatch is closed.

### E3 — generator-component attribution

**Measured (2026-10-25):** the same supplied-input corpus times five test-only
stages independently: mask/popcount and class-size setup, rank decomposition,
rank-to-orbital bit selection, parity/determinant materialization, and the
recomposed no-RNG pipeline. The recomposed pipeline must produce the same
final-excitation digest as the precomputed staged path.

```sh
PLANCK_FCIQMC_COMPONENT_BENCHMARK=1 build/planck-fciqmc-walkers
```

`PLANCK_FCIQMC_COMPONENT_BENCHMARK_DRAWS` and
`PLANCK_FCIQMC_COMPONENT_BENCHMARK_ROUNDS` override the defaults. Five
5,000,000-draw rounds measured `0.030784 s` setup, `0.015322 s` rank
decomposition, `0.062204 s` bit selection, `0.025833 s` materialization, and
`0.109556 s` for the recomposed pipeline. The staged and recomposed final
digests match. Isolated timings are not additive, but bit selection is clearly
the largest measured component and is the only authorized local-generator
candidate.

### E4 — byte-table bit-selection oracle

The current selector clears low set bits once per requested rank, then uses
`countr_zero`. Arm64 has no PDEP, but a 256-by-8 byte lookup table can locate
the requested bit within a selected byte after a fixed eight-byte popcount
prefix scan. The test-only oracle now checks the byte table against the current
selector over the complete E0 active-space corpus and all valid ranks, then
times both over the E3 selection corpus:

```sh
PLANCK_FCIQMC_SELECTOR_BENCHMARK=1 build/planck-fciqmc-walkers
```

`PLANCK_FCIQMC_SELECTOR_BENCHMARK_DRAWS` and
`PLANCK_FCIQMC_SELECTOR_BENCHMARK_ROUNDS` override the defaults. Reject the
candidate if table lookup and the prefix scan lose to the compact current loop
or introduce a cache cost that erases the local gain.

**Deferred for an uncontended paired retest (2026-10-25):** five alternating
10,000,000-selection rounds measured `0.017161 s` for the
clear-lowest-set-bit selector and `0.015188 s` for the byte table: **1.130x**,
with a matching digest. The literal sampler and N2 S5 one-/four-thread gates
also passed. However, the three-run full fixture gave a four-thread median
sampling worker time of `54.713 s`, merge time of `25.329 s`, and wall time of
`171.763 s`, versus M1's `51.358 s`, `24.049 s`, and `163.958 s`.

All stochastic counters remained identical, but worker, merge, and prefill
time moved together. This suggests host load, but does not establish causality:
cache and thermal interactions can affect unchanged code too. The run is
inconclusive for candidate attribution; the R1 paired retest is required.

Production retains the compact clear-lowest-set-bit loop while the candidate
remains test-only. On an uncontended machine, apply the byte-table selector in
a disposable candidate worktree, build baseline and candidate binaries
separately, and perform a paired A/B ladder in alternating order on the same
host. Accept it only if the three-pair median improves the four-thread wall
time by the documented threshold and its phase ledger localizes the gain to
worker propagation.

The following table describes the decision after E3:

| observed limit | permitted next investigation | required isolation |
|---|---|---|
| setup, bit selection, or parity | local generator candidate | literal sampler oracle and full fixed-seed trajectory gate |
| orbital updates or parity | local generator work, not dispatch rewrite | literal excitation oracle |
| `SpawnAccumulator::finalize()` / sorting | local folding or allocation prototype | ordered-fold raw-bit oracle and memory accounting |
| `slater_condon_element` | separate shared-Hamiltonian project | CI, CASSCF, open-shell, and FCIQMC validation |
| residual pairwise merge | larger-population merge study | same tagged tree and explicit temporary-memory bound |

Descriptor caching is not authorized without a physical-population prototype
that reports retained bytes and lookup cost. The reference estimator is not a
candidate unless the ledger shows it is material.

## Out of scope

- Replica trajectories improve time-to-error-bar, not a single-chain step
  rate, and must be compared at equal total core counts.
- MPI and GPU work require an explicit CPU or memory target beyond this
  measured envelope.
- Active-space/frozen-core FCIQMC is outside propagation micro-optimization;
  its conditional investigation and validation are specified in R3 above.
