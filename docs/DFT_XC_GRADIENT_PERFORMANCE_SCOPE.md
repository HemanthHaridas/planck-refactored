# Why is a V4-grid KS gradient single-threaded and 8 GB, and what fixes it?

Scope, opened 2026-11-18. It expires when the work lands; rewrite it then as
the answer. Status lives in `vault/Status/Open Work.md`. Opened from the
default-grid decision (`docs/DFT_DEFAULT_GRID_SCOPE.md`), whose cost
measurement depends on it.

## The measurement

Benzene, B3LYP/6-31G* (Cartesian, nb = 102), `grid_policy g4-3-v4`,
135 radial shells, RKS analytic gradient, one rotated orientation, run alone
on an idle 8-thread M3 (8 GiB), profiled with `sample` at ~1.3 ms:

| | |
|---|---|
| Wall | 117 s |
| User CPU | 118 s (one core on average, of eight) |
| Peak memory footprint | **8.1 GB** |

Main-thread samples (92,271 = 117 s):

| Phase | Share of wall | Threaded |
|---|---:|---|
| `compute_xc_nuclear_gradient_rks` | **~68 %** | no |
| of which `becke_partition_owner_derivatives` | ~4 % | no |
| of which large-allocation zeroing (`madvise`) | ~1.5 % | no |
| `evaluate_ao_hessian_on_grid` | 2.3 % | no |
| HF-like gradient terms (`compute_rks_gradient`) | 0.7 % | yes |
| SCF, grid build, AO values (everything else) | ~26 % | partly |

The fix is not specific to V4. `legacy` runs the same loop; V4 just has ~9×
the points (~1M here), which turns a hidden cost into the dominant one.

## Why

**Time.** The point loop (`src/dft/dft_gradient.cpp:372`, and its UKS
twin at `:524`) allocates and zeroes three nb × nb matrices *per grid point*,
adds six rank-1 updates into them (GGA), and then reduces each one against
the density with a full nb² `cwiseProduct().sum()`. That is ~15 nb² memory
operations per point, about 1.5e11 here, one point at a time on one thread.
But the algebra is two GEMMs over a block of points:

```
vxc1_q  -=  G_qᵀ A  +  A_qᵀ Φ          (Φ, G_q: block × nb AO values/gradients;
                                         A, A_q: the weighted aow / aow_q rows)
grad(owner(p), q) += 2 (a_pᵀ P g_qp + φ_pᵀ P a_qp)
```

The owner-atom term needs only P Φᵀ and P G_qᵀ per block, plus row dot
products. No per-point nb × nb matrix is needed at all.
`becke_partition_owner_derivatives` allocates `natoms` matrices of size
natoms × 3 per point; that scratch can be hoisted.

**Memory.** The gradient holds ten dense npts × nb arrays: AO values, three
gradients, and six Hessian components. ~1M × 102 × 8 B × 10 ≈ 8.1 GB, which
is the measured peak. The SCF itself keeps four of them (3.2 GB here).
With V4, a 12-water chain cannot run a gradient on this machine at all.

## Plan

1. **X1 — blocked GEMM rewrite** of `compute_xc_nuclear_gradient_{rks,uks}`.
   Same terms, ~128-point blocks, hoisted Becke-derivative scratch. Serial.
   Expected: the 68 % phase drops to a few %, so the run becomes SCF-bound.
2. **X2 — stream AO derivatives per block** inside that loop instead of
   taking stored npts × nb Hessians (and gradients, if the SCF arrays are
   not reused). Removes six of the ten arrays (~4.9 GB here). Do it with X1,
   since both restructure the loop into blocks.
3. **X3 — thread the blocks.** Partials keyed by a fixed function of the
   block index and summed in bin order, never by thread id and never with
   `omp critical` (the DFT XC determinism rule: the gradient must be identical
   at 1/2/4/8 threads). Thread `evaluate_ao_*_on_grid` over AO columns
   (disjoint writes, bitwise).
4. **X4 — only if still needed:** stream the SCF's four stored AO arrays
   (recompute per iteration against 3.2 GB here). This is a cost trade, so
   measure it after X1–X3.

## Gates

- Summation order changes, so not bitwise against the old loop. Every DFT
  gradient, geomopt and frequency regression case must match to printed
  digits (`--suite all`), and the existing FD gates
  (`water_dft_hse06_gradient_symm_ultrafine_fd`, the PCM and D3 gradient
  cases) must pass.
- A unit test that compares the blocked contraction against the per-point
  form on one small fixture (RKS + UKS, LDA + GGA) to 1e-12 relative, and
  checks gradient identity across thread counts. Mutation-check it.
- Re-time this benzene case and record wall, CPU and peak footprint.

## Out of scope

- The DH paths (`dh_pt2_gradient.cpp`, `udh_xc_ii*.cpp`) call
  `becke_partition_owner_derivatives` per point too. They benefit from its
  scratch fix, but their own loops are separate work.
- Grid construction, and the SCF XC build (already threaded,
  `assemble_xc_matrix`).
- Which grid becomes the default. That decision waits on X1–X3, because
  V4's cost is mostly this loop today.
