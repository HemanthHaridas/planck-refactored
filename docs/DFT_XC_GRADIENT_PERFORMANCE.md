# Why was a V4-grid KS gradient single-threaded and 8 GB, and what fixed it?

Answered 2026-11-18 (opened as a scope the same day, from the default-grid
decision in `docs/DFT_DEFAULT_GRID_SCOPE.md`, whose cost measurement needed
it). Status lives in `vault/Status/`.

## Answer

The XC nuclear gradient (`src/dft/dft_gradient.cpp`) worked one grid point
at a time on one thread. At each point it built three nb × nb matrices and
contracted each against the density. It also held ten npts × nb AO arrays:
values, three gradients and six Hessian components. On V4, with about 9×
the points of the default grid, this loop dominated the run.

Five changes fixed it. All terms are unchanged; the gradients agree with the
old code to the printed digits:

| Step | Change |
|---|---|
| X1 | 128-point blocks; the per-point matrices become block GEMMs |
| X2 | GGA AO Hessian evaluated per block, never stored |
| X2.5 | no nb × nb accumulators at all; the result comes from row and column sums of an elementwise block product |
| X3 | blocks threaded through 64 fixed bins, summed in bin order: bitwise identical at any thread count. AO evaluators threaded over columns |
| X3b | Becke owner-weight derivative O(natoms²) per point instead of O(natoms³) |

B3LYP/6-31G\*, gradient run, on an 8-thread M3 with 8 GiB:

| | before | after X2 | after X3b |
|---|---:|---:|---:|
| benzene V4: wall | 117 s (1 core) | 25.0 s | **15.1 s** |
| benzene V4: peak | 8.1 GB | 3.6 GB | 3.6 GB |
| benzene legacy `fine`: wall | 15.0 s | 8.2 s | **5.8 s** |
| water12 chain (3 Å) legacy `fine`: wall | | 110.7 s | **73.6 s** |
| water12 legacy `fine`: peak | | 4.8 GB | 4.8 GB |

Two things are not done here, and both are now steps of the joint plan in
`docs/DFT_DEFAULT_GRID_SCOPE.md`. A profile of the remaining time is S2.
X4, streaming the SCF's stored AO arrays (the last memory lever), is S7.

## The starting measurement

Benzene, B3LYP/6-31G\* (Cartesian, nb = 102), `grid_policy g4-3-v4`,
135 radial shells, RKS analytic gradient, one rotated orientation, idle
machine, `sample` at ~1.3 ms. Wall 117 s, user CPU 118 s (one core of
eight on average), peak footprint 8.1 GB.

| Phase | Share of wall | Threaded |
|---|---:|---|
| `compute_xc_nuclear_gradient_rks` | ~68 % | no |
| of which `becke_partition_owner_derivatives` | ~4 % | no |
| of which large-allocation zeroing (`madvise`) | ~1.5 % | no |
| `evaluate_ao_hessian_on_grid` | 2.3 % | no |
| HF-like gradient terms (`compute_rks_gradient`) | 0.7 % | yes |
| SCF, grid build, AO values (everything else) | ~26 % | partly |

The cost was not specific to V4. `legacy` runs the same loop, but V4 has ~9×
the points (~1M here), which made a hidden cost the dominant one.

## Why it was slow and large

**Time.** At every point the loop allocated and zeroed three nb × nb
matrices, added six rank-1 updates (GGA), and reduced each against the
density with a full nb² elementwise sum: ~15 nb² memory operations per
point, ~1.5e11 here, on one thread.

**Memory.** Ten dense npts × nb arrays: ~1M × 102 × 8 B × 10 ≈ 8.1 GB, the
measured peak. The SCF keeps four of them (values and gradients, 3.2 GB);
the gradient added the six Hessian components.

## What changed

All of it is in `xc_nuclear_gradient`, one driver for RKS and both UKS spin
channels, plus `becke_partition_owner_derivatives` and the AO evaluators in
`src/dft/ao_grid.h`.

**X1 (`77558b99`).** The per-point vtmp_q = g_q aowᵀ (+ aow_q φᵀ, GGA) is
never formed. Over a 128-point block its sum is a GEMM pair, and its
contraction with P (the owner-atom term) is a row dot product against A·P
and Φ·P.

**X2 (`88487b04`).** The GGA AO Hessian is evaluated per block inside the
driver, so `compute_xc_nuclear_gradient_{rks,uks}` no longer take an
`AOGridHessian`. The old code built six zeroed npts × nb arrays even for
LDA. The DH paths still build their own Hessians.

**X2.5 (`89e774f4`).** After X1 the nb × nb accumulators M_q were only read
as M_q.row(μ)·P.row(μ). With P symmetric:

```
M_q(μ,:)·P(μ,:) = Σ_p g_q(p,μ) (A P)(p,μ)  +  Σ_p A_q(p,μ) (Φ P)(p,μ)
```

So per block the loop forms E_q = g_q ∘ AP (+ A_q ∘ ΦP), elementwise and
n × nb. Its row sums are the owner-atom term. Its column sums, gathered by
the atom that carries each AO, are the basis-motion term. Only the A·P and
Φ·P GEMMs remain: 2 n nb² per block and channel instead of 8 (GGA). Each
block's whole contribution shrinks to natoms × 3 plus nb × 3 numbers, which
is what makes X3 cheap.

**X3 (`89e774f4`).** Contiguous blocks go into 64 fixed bins
(`kXCGradientBins`), and the bins run under
`omp parallel for schedule(dynamic, 1)`. Each bin sums its blocks
serially, and the bins are added in bin order after the loop. Bins never
depend on thread ids, so the gradient is bitwise identical at any thread
count (the DFT XC determinism rule). Errors from the AO Hessian or the
Becke derivative cannot `return` from inside the loop; they are recorded
per bin and returned afterwards. The two AO evaluators first check for
null shells, then run `parallel for` over AO columns. The writes are
disjoint, so the result is bitwise; `if (!omp_in_parallel())` keeps the
per-block Hessian call inside X3 serial.

**X3b (`89e774f4`).** The old Becke owner derivative scaled a natoms × 3
block for every atom pair (`dprod(i) *= sij`): ~6 natoms³ flops and four heap
allocations per point. Now every s_kj and its derivative (nonzero only in
the owner, i and j rows) is computed once per pair, and

```
dP_k = Σ_j (Π_{l≠j} s_kl) ds_kj
```

uses prefix and suffix products, with no division, so s → 0 is safe. Scratch
is `thread_local`; the signature is unchanged, so the DH callers
(`dh_pt2_gradient.cpp`, `udh_xc_ii*.cpp`) gain without edits.

## How it was checked

- **Printed digits.** The changes reorder sums, so they are not bitwise
  against the old loop. Checked against the previous binary at every step:
  - Nine C1 H2O2 inputs (RKS and UKS × LDA/PBE/B3LYP/HSE06, plus RKS B3LYP
    on V4), at 1, 4 and 8 threads. All are identical or off by one unit in
    the 12th printed decimal. That flip comes from the SCF: its energy
    differs at 1e-15 relative between 1 and 8 threads, in the old binary
    too.
  - Benzene (both grids) and water12 gradients are identical to every
    printed digit.
- **`planck-dft-xc-gradient-threads`** (ctest, 0.8 s):
  - RKS/UKS × LDA/GGA on a C1 water cation, 6-31G\*, `Normal` grid, with
    more blocks than bins: the gradient is bitwise equal at 1/2/3/4/8
    threads.
  - The new Becke derivative against the old O(natoms³) routine, kept in
    the test as the oracle: all 17,952 points of a C1 7-atom fixture agree
    to 4.4e-16 (max |ref| 1.25).
- **Mutations**, each failing its gate:
  - Dropping the GGA owner A_q term, or halving the LDA weight (X1):
    0.2–0.35 Ha/Bohr.
  - Dropping A_q ∘ ΦP from the column sums (X2.5): 0.14 Ha/Bohr.
  - Tying the bin count to `omp_get_max_threads()` (X3): all 16
    thread-identity checks fail (≤ 3.3e-15).
  - Flipping the sign of the s(j,k) = 1 − s(k,j) term (X3b): the oracle
    fails (1.36).
- **Regression cases.** The 28 DFT gradient/geomopt/frequency cases pass
  (`--suite all`), including the FD gate
  `water_dft_camb3lyp_gradient_symm_ultrafine_fd`, the PCM and D3 cases, and
  the two UKS DH V4 cases that use the new Becke derivative.
- **ctest `planck-(udh|dh|dft)`:** 42/42 pass. Running the group found one
  stale test. `planck-dh-workflow-runner` had failed since `33944891`
  (2026-11-10), which moved the frequency reference rebuild from the DFT
  wrapper into `compute_hessian`; the test still looked for it in
  `driver.cpp`. It now checks the order in `hessian.cpp`, and removing the
  rebuild fails it.

## What the measurements taught

- **X1 alone could not be timed on V4.** At 8.2 GB on an 8 GiB machine it
  swapped (83.6 s). X1 and X2 belong together: the memory had to go before
  the time could be measured.
- **Per-block partials were planned, fixed bins were built.** One partial
  per block is deterministic too, but its memory grows with the grid (water12
  V4: ~23k blocks). 64 contiguous bins keep the order fixed with constant
  memory.
- **X3 changed wall time, not CPU time.** Benzene V4 user CPU went
  100.7 → 101.2 s while wall fell 25.0 → 15.1 s. The XC gradient was ~10 s
  of serial wall and now runs on all cores.
- **The Becke prediction was grid-specific.** "~10 min serial at 36 atoms"
  was estimated for V4. At legacy `fine`, water12 saved 37 s.

## Not done

- **The profile of what remains** was done as S2 of the joint plan
  (`DFT_DEFAULT_GRID_SCOPE.md`), and led to S2b (blocked XC matrix).
- **X4, stream the SCF's AO arrays: superseded by AO screening** (S7 of
  the joint plan, 2026-11-19). The stored arrays are now kept per 128-point
  block with only the significant AO columns (`ScreenedAOGrid`), which the
  density, XC matrix and this gradient read directly. Water12
  `legacy@fine` went from 48.5 s / 4.8 GB to 15.5 s / 1.37 GB with
  unchanged printed results. The dense consumers (SOSCF, stability, TDDFT
  kernel, DH paths) still read dense arrays, built from the blocks on first
  use; porting them is S7b.
- **The DH paths' own per-point loops** (`dh_pt2_gradient.cpp`,
  `udh_xc_ii*.cpp`). They gain from X3b but are not restructured.
- **Grid construction** stays serial. The SCF XC build was already threaded
  (`assemble_xc_matrix`).
