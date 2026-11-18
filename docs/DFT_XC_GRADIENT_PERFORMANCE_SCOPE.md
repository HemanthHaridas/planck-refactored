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

## Progress (2026-11-18)

**X1 landed** (`77558b99`). One driver, `xc_nuclear_gradient`, serves RKS
and both UKS spins: per 128-point block, `M_q += G_qᵀA + A_qᵀΦ` and the
owner term as row dots against `A·P` and `Φ·P`. The Becke owner-weight
derivative keeps one scratch buffer and updates only the owner/i/j rows.

**X2 landed** (next commit after X1). The GGA AO Hessian is
evaluated per block inside the driver; `compute_xc_nuclear_gradient_{rks,uks}`
no longer take an `AOGridHessian`, and the KS gradient no longer builds one.
It had built six zeroed npts × nb arrays even for LDA. The DH paths still
build their own (out of scope).

| | before | X1 | X2 |
|---|---:|---:|---:|
| benzene V4: wall | 117 s | 83.6 s (swapping) | **27.5 s** |
| benzene V4: peak footprint | 8.1 GB | 8.2 GB | **3.6 GB** |
| benzene legacy `fine`: wall | 14.96 s | 8.59 s | |
| benzene legacy `fine`: XC phase | 8.4 s | 1.8 s | |

Benzene B3LYP/6-31G*, nb = 102, one 8-thread M3, idle. X1 alone could not
be timed honestly on V4: at 8.2 GB on 8 GiB it swapped. The XC phase is
now GEMM-bound on one thread (21 % of the legacy-`fine` run); the next
serial pieces are the Becke derivative (~15 % of the phase) and AO
evaluation.

Checks: gradients equal to the old code at all 12 printed decimals on
RKS/UKS × LDA/GGA/global/range-separated inputs, legacy and V4 (benzene:
two last-digit changes after X1, none from X2). Mutations dropping the GGA
owner `A_q` term or halving the LDA weight fail by 0.2–0.35 Ha/Bohr. The
28 DFT gradient/geomopt/frequency regression cases pass after X1.

## Gates

- Summation order changes, so not bitwise against the old loop. Every DFT
  gradient, geomopt and frequency regression case must match to printed
  digits (`--suite all`), and the existing FD gates
  (`water_dft_hse06_gradient_symm_ultrafine_fd`, the PCM and D3 gradient
  cases) must pass.
- For X3: a unit test that checks gradient identity across thread counts
  (RKS + UKS, LDA + GGA). Mutation-check it. (X1/X2 were gated by
  full-precision before/after gradients instead, above.)
- Re-time this benzene case and record wall, CPU and peak footprint.

## Out of scope

- The DH paths (`dh_pt2_gradient.cpp`, `udh_xc_ii*.cpp`) call
  `becke_partition_owner_derivatives` per point too. They benefit from its
  scratch fix, but their own loops are separate work.
- Grid construction, and the SCF XC build (already threaded,
  `assemble_xc_matrix`).
- Which grid becomes the default. That decision waits on X1–X3, because
  V4's cost is mostly this loop today.
