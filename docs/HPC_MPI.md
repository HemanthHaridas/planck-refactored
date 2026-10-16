# How does `planck-mpi` distribute an SCF, and how well does it scale?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md` (the "HPC campaign — what is left" entry tracks the
open gaps). This file answers one architecture question:

**What does the MPI front end actually distribute, what stays replicated, and
what do the cluster measurements show?**

## Short answer

`planck-mpi` is a third executable, built only with `-DBUILD_MPI=ON`. It parses
the input once and dispatches to the same HF or DFT driver the serial binaries
use. Only two things are distributed:

1. the **direct-SCF Fock build** (HF Fock in #146; DFT J/K in #151), which
   stripes the bra triangle of the fused quartet loop across ranks, and
2. the **DFT SCF-energy grid** (#156), where each rank takes a contiguous slice
   of grid points.

Each is followed by one dense `nb²` `MPI_Allreduce`, summed in fixed rank order.
Diagonalization, setup and everything else are replicated. Every distributed
result is **bitwise equal** to the serial binary.

On Notchpeak (32 ranks, 6-31G water chains), HF runs at 42–46 % parallel
efficiency, near-linear to 16 ranks. DFT runs at 46–65 %, and its efficiency
*rises* with system size. DFT memory grows as `nb^1.9`, not `nb⁴`.

## Where the logic lives

| Piece | Location |
|---|---|
| Front end | `src/mpi/main.cpp` — `MPI_Init`, parse, then `calculator.is_dft_run() ? DFT::Driver::run : HartreeFock::Driver::run` |
| MPI surface | `src/base/mpi_env.h` — `Mpi::rank`, `size`, `distributed`, `allreduce_inplace`; no-ops in the serial binaries |
| HF/DFT Fock stripe + reduce | `src/integrals/fused_fock.h` (stripe on `mpi_rank`/`mpi_size`, `allreduce_inplace` on `G` / `Ga`,`Gb`) |
| DFT J and K | `_compute_2e_j_direct`, `_compute_2e_k_direct` (`src/integrals/base.h`), accumulators in `src/integrals/fock_accumulate.h` |
| DFT grid slice | `src/dft/driver.cpp` (rank/size slice of the grid, XC matrix and scalars reduced in rank order) |
| Build | `option(BUILD_MPI ...)` in `CMakeLists.txt` |
| Gates | `water_rhf_mpi_smoke`, `water_dft_mpi_smoke`, `water_uks_mpi_grid_smoke` |
| Scaling harness | `tests/benchmarks/scale_bench.py`; data `scale.json` (post-#156) and `docs/scale_notchpeak.json` (pre-#151) |

## Why it is a third binary, not an MPI-enabled one

`hartree-fock` and `planck-dft` always shared one parser. Their `main()`
functions differed only after parsing, so the split was convenience, not an
architectural boundary. Two refactors made a single front end possible. Both
landed on the serial binaries first, gated by bitwise-identical energies:

- `Calculator::is_dft_run()`, set when `%begin_dft` is present, so the input
  declares the method rather than the executable name;
- the inline HF driver extracted into `HartreeFock::Driver::run(Calculator&)`
  (`src/hf_driver.cpp`), the symmetric peer of `DFT::Driver::run`.

The serial binaries do not link MPI. An `MPI_Init` in a binary people run
interactively is a regression, and `mpi_env.h` compiling to no-ops keeps the
kernels' source identical across all three binaries.

## What is distributed, and the invariants that keep it correct

### 1. The fused Fock loop is the partition point

The integral kernels are stateless functions with `thread_local` scratch, so
they are rank-safe as they stand. MPI lives entirely at the loop-partition and
reduce layer: each rank takes `bra % nranks == rank` of the canonical quartet
triangle, OpenMP threads within the rank, and the `nb²` partial is allreduced.
No ERI is ever redistributed.

### 2. Reductions sum in fixed rank order — never completion order

A wrong or reordered reduction gives a plausible but wrong energy, not a crash.
Every distributed path is gated on `energy(-n k) == energy(serial)` **bitwise**
and on rank-count invariance (`-n 2 == -n 4`). This matters most for the DFT XC
grid reduction, which is the historical jitter site: it is summed in rank order,
and it is reduced *before* the J/K combine so nothing is double-counted.

### 3. DFT needs J and K separately, with **unscaled** K

The HF fused entry returns `G = J − ½K`. DFT cannot use that. It needs J alone,
and for hybrids K alone, scaled by the functional's own coefficient. A
range-separated hybrid needs two different K's (full-range and erf short-range)
added to one J. So #151 split the accumulator:

```
coulomb_accumulate:  J(a,b) += P(c,d) * val
exchange_accumulate: K(a,c) += P(b,d) * val     // no 0.5, no sign
```

The RHF wrapper re-applies its own `-0.5`, keeping HF byte-identical. Each DFT
call site keeps its own prefactor: `-0.5 × c` for RKS and `-c` for UKS. **The
½ must not be folded into the shared accumulator.** RKS and UKS use different
exchange prefactors (RHF ½, UHF 1), so folding it in gives RKS hybrids half
their exact exchange while UKS stays correct. The only symptom is the two spin
cases disagreeing.

The split preserves the orbit-deduplication correctness argument term by term:
it is a property of the orbit enumeration, not of which term is accumulated.

### 4. TDDFT is the one dense-tensor consumer left

Linear response still builds the `nb⁴` AO tensor for its AO→MO transform. For
range-separated functionals it builds a second, short-range one too. That is
correct, but it means TDDFT neither saves memory nor distributes. The DFT
analytic-gradient grid and the TDDFT kernel builds also run on the whole grid on
every rank: only the two SCF-energy grid call sites opt into the slice.

### 5. Diagonalization is replicated, and that is the soft ceiling

Every rank diagonalizes the same reduced Fock matrix. That is correct, and the
cost is invisible at the measured sizes. It is the fixed serial fraction that
makes the curves taper from about 10× at 16 ranks to about 14× (not 32×) at
32 ranks, and it is expected to dominate near `nb ≈ 2000`. A distributed
eigensolver (ScaLAPACK/ELPA) is deliberately deferred until a target forces it.

## What the measurements show

All runs: Notchpeak (CHPC Utah), 6-31G, `engine os`, 1 thread per rank, water
chains, B3LYP on grid `normal` for DFT.

### Read the wall clock, not Planck's `Time(s)` column

Under `mpirun`, the SCF table is printed by one rank about its own timer. It
was bit-identical (`1.371943 s`) at 1–16 ranks while the real per-iteration time
fell from 15.75 to 1.62 s. Read that way, MPI appears not to scale. The
harness therefore uses `wall_s / n_iters`, measured outside the launcher, and
keeps the scraped value only as `per_iter_rank_s`.

### Before the DFT J/K fix (`docs/scale_notchpeak.json`, notch389)

| water | nb | HF best | DFT best | DFT peak RSS (1 rank) |
|---|---|---|---|---|
| 8 | 104 | 8.46× @16 | 1.19× @16 | 1.3 GB |
| 16 | 208 | 9.70× @16 | 1.19× @8 | 15.8 GB |
| 24 | 312 | 9.71× @16 | 1.04× @4 | 76.3 GB |
| 32 | 416 | 10.04× @16 | serial only | 235 GB |

DFT then still built the `nb⁴` tensor serially for its KS Coulomb term. It was
pinned at about 1.2×, grew as `nb⁴`, and died of OOM at 24 water on 8 ranks
because every rank replicated a 76 GB allocation. That single cause is why J/K
was routed through the fused loop.

### After #151 and #156 (`scale.json`, notch386)

Speedup at 32 ranks and parallel efficiency:

| | nb=104 | nb=208 | nb=312 | nb=416 |
|---|---|---|---|---|
| HF | 13.59× (42 %) | 13.37× (42 %) | — | 14.85× (46 %) |
| DFT | 14.62× (46 %) | 15.09× (47 %) | 20.90× (**65 %**) | did not converge |

| nb | HF RSS | DFT RSS |
|---|---|---|
| 104 | 15 MB | 332 MB |
| 208 | 35 MB | 1243 MB |
| 312 | 68 MB | 2741 MB |
| 416 | 115 MB | 4829 MB |

Locally on 8 water, the J/K fix cut peak RSS from 1360 to 427 MB with an
identical energy. The 933 MB freed is exactly `104⁴ × 8 B`.

Three readings are worth carrying:

- **The DFT/HF memory ratio (42×) is the wrong discriminator.** The DFT RSS
  exponent is about 1.9, which is clean `nb²` grid and AO-on-grid storage that
  HF does not need. Per-rank RSS grows only 9 % from 1 to 32 ranks. The harness
  now keys its memory verdict on the exponent (flag above `nb^3`), and its
  scaling verdict on the Karp–Flatt serial fraction rather than efficiency at
  maximum ranks.
- **DFT efficiency rising with size** is the signature of genuinely distributed
  grid work: the larger the grid, the better it amortizes.
- **The nb=416 DFT failure is SCF convergence, not scaling.** The serial
  baseline hit `max_cycles` at 4.8 GB, so there is no anchor for a speedup.
  That is a robustness problem with large water chains, separate from MPI.

`scale.json` reflects #151, #152 and #156 together. Attributing the win to any
one PR would need a run at its parent commit.

## What a workstation can and cannot tell you

**It can prove correctness.** MPI bugs are logic bugs: a double-counted quartet
or a stale broadcast reproduces at `mpirun -n 2` on a six-atom molecule against
the serial oracle, and `--oversubscribe -n 16` exercises the partition
arithmetic beyond the core count.

**It cannot measure scaling.** At `-n 4`, each rank spawns all the node's
OpenMP threads unless pinned, so wall time *rises*. Holding total threads
constant leaves the curve flat, and single-node `mpirun` uses shared-memory
transport. The control is that HF measures flat on an 8-core laptop while
scaling about 10× on the cluster. Build bitwise-correct locally, then measure
the payoff on a cluster.

A workstation also cannot surface defects that exist only at scale, such as a
pair index overflowing `int` or imbalance at high rank counts. That is a small,
nameable set, but it is why the CI gap below matters.

## What remains (tracked in Open Work)

- **No CI scale gate.** The largest regression input is 4 atoms, so nothing
  fails if distribution regresses to replication. The cheap fix is one
  16-water / 6-31G (nb=208) case asserting `-n 2 == -n 4 == serial` bitwise.
- **HGP and Rys conventional `_compute_2e` replicate.** Each rank builds the
  full tensor (`hgp.cpp:1392`, `rys.cpp:1460`). The result is correct but the
  extra ranks are wasted. Either stripe them (the OS pattern is six lines plus
  one reduce) or reject `-n > 1` for them. Direct SCF, which is what cluster
  runs use, is striped for every engine.
- **Post-HF is not distributed.** Generated CC kernels are OpenMP-threaded
  behind `CCGEN_OMP_COLLAPSE`, but the hand-written CC and MP2 paths are serial,
  and no post-HF path is MPI-striped. Dense post-HF can be rank-striped on the
  outer MO index without RI: RI is a *memory* strategy, not a distribution
  prerequisite. Distributing the RI 3-center tensor by auxiliary block is the
  separate, memory-motivated tail.
- **Before reaching for MPI, ask whether a fat node and RI suffice.** RI breaks
  the `nb⁴` memory wall on one node with no new executable. MPI pays off only
  once one node is saturated. Distribute the memory-lean algorithm, not the
  dense one.
