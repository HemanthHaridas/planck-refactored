# ERI / Fock BLAS-LAPACK Migration — Scope

Status: **scope, nothing landed** (2026-10-17). In-flight work, so this file is
allowed to be scope-shaped; rewrite it as an answer when the work lands.

**Question:** Would routing the ERI and Fock builds through BLAS/LAPACK make
them faster, and where?

## Short answer

No for the ERI kernels and for the dense Fock builds Planck runs by default.
Yes for the RI-JK Fock build, though the largest win there does not involve
BLAS at all. In priority order:

| step | change | measured lever | needs BLAS? |
|---|---|---|---|
| B0 | Cache the RI pair factors and the unpacked `B_Q` once per geometry | **61 % + 8 %** of RI-JK Fock time is rebuilding them twice per iteration | no |
| B1 | Optional `USE_BLAS` build: `EIGEN_USE_BLAS` + Accelerate / system BLAS | RI-K GEMMs **4.3–7.8×** | yes |
| B2 | RI-K from `C_occ` instead of `D` on the SCF caller | **4.4–4.8×** serial on RI-K | no |
| B3 | LAPACK `dsyevd` wrapper for SCF diagonalization | 2.5–7× at n ≥ 300, but diag is **0.08 %** of an iteration today | yes |

The ERI kernels, direct-SCF digestion, the conventional `nb⁴` contraction and
the Cartesian-to-spherical transform were each measured and gain nothing. See
"Ruled out" below.

## Measurements

Machine: Apple M3 (4P + 4E), macOS, `build-full` (g++-15, `-O3`, OpenMP),
Accelerate. Microbenchmarks live in the session scratchpad and are not committed.
They are reproducible from the shapes given below.

### Where an SCF iteration goes (`PLANCK_PHASE_TIMING=1`, 4 threads)

| case | mode | fock | diis | diag |
|---|---|---|---|---|
| 8 H2O / 6-31G (nb 104) | direct | 0.888 s | 0.0005 s | 0.0007 s |
| 4 H2O / 6-31G* (nb 76) | direct | 0.743 s | 0.0003 s | 0.0003 s |
| 4 H2O / 6-31G* + `scf_ri_jk` | RI-JK | 0.062 s | 0.0002 s | 0.0003 s |
| 10 H2O / 6-31G* + `scf_ri_jk`, 1 thread | RI-JK | 2.14 s | 0.003 s | 0.004 s |

The Fock build is >99 % of every iteration, so diagonalization (B3) cannot
matter until nb is in the thousands. `docs/HPC_MPI.md` §5 already expects the
replicated diag to become relevant only near nb ≈ 2000.

### Inside the RI-JK Fock build (`sample`, 10 H2O / 6-31G*, 1 thread)

| inclusive share | what |
|---|---|
| **~61 %** | `build_ri_pair_factors`, called twice per iteration: once via `build_ri_3index_unpacked`, once via `build_ri_j` |
| ~29 % | `build_ri_k`, the `naux × 2` nb³ GEMMs |
| ~8 % | `build_ri_3index_unpacked`, the copy itself |
| ~1.5 % | `build_ri_j` arithmetic |

`build_ri_pair_factors` (`src/post_hf/ri/ri_eri.cpp`) computes `j3c · Lᵀ`, or
a triangular solve. That is density-independent, but it is recomputed on every
call, and there are 8 call sites. It is the single largest item in RI-JK, and
removing it needs no BLAS.

### RI-K GEMM shape: `K += (B_Q D) B_Qᵀ` over Q (naux = 3 nb, nocc = nb/5)

| nb | threads | Eigen (D) | Accelerate (D) | Eigen (C_occ) |
|---|---|---|---|---|
| 150 | 1 | 0.158 s | 0.025 s (**6.3×**) | 0.036 s (4.4×) |
| 300 | 1 | 2.405 s | 0.308 s (**7.8×**) | 0.502 s (4.8×) |
| 150 | 4 | 0.128 s | 0.027 s (4.8×) | 0.086 s (1.5×) |
| 300 | 4 | 1.296 s | 0.301 s (4.3×) | 0.406 s (3.2×) |

Relative differences are below 4e-14. Accelerate times do not change with
thread count, because the AMX unit is shared per cluster, so the advantage
narrows as Eigen gains threads. B1 and B2 multiply.

### Eigensolver: Eigen `SelfAdjointEigenSolver` vs Accelerate `dsyevd`

| n | Eigen | dsyevd | |
|---|---|---|---|
| 100 | 0.0006 s | 0.0061 s | **0.1×, slower** |
| 300 | 0.0122 s | 0.0049 s | 2.5× |
| 1000 | 0.412 s | 0.075 s | 5.5× |
| 2000 | 3.42 s | 0.495 s | 6.9× |

So any wrapper needs a size cutoff (around n ≈ 200). Most `SelfAdjointEigenSolver`
calls are on small SAO blocks, and those must stay on Eigen.

### Determinism

Accelerate `dgemm` and `dsyevd` are **bitwise identical at
`VECLIB_MAXIMUM_THREADS` = 1/2/4/8** and across repeat runs (n = 700 hash
check). They are **not** bitwise equal to Eigen, so a BLAS build and an Eigen
build differ in the last digits. The `metric_close` references are fine at
their tolerances, but no cross-build `atol = 0.0` comparison can hold.
OpenBLAS/MKL thread-count invariance is **unmeasured**; check it on the cluster
before trusting the FCIQMC/FCI thread-pair gates in a Linux BLAS build.

## Steps

### B0 — Cache the density-independent RI factors (no BLAS)

- Store `pair_factors` (and the unpacked `B_Q`, or have `build_ri_k` read the
  packed factors directly) on `Calculator` next to `_ri_j3c`, populated in the
  same `ensure_ri_3c_ready` call.
- **Invalidation is the risk.** `_ri_j3c` is cleared at
  `ri_eri.cpp:820` on geometry/basis change, and the RI-MP2 gradient work found
  that this cache's invalidation is load-bearing. The new cache must be cleared
  by the **same** statement, never by a parallel mechanism, or geomopt and
  frequency runs will silently reuse stale factors.
- Memory: unpacked `B` is `naux · nb²` doubles (10 H2O / 6-31G*: about 0.25 GB).
  If that is too much, keep only `pair_factors` cached and unpack per Q inside
  `build_ri_k`.
- Gate: same arithmetic, just hoisted, so energies must be **bitwise identical**
  to pre-B0. `planck-ri-jk-equivalence` must pass. **Add an RI-JK regression
  case**: none exists today (`scf_ri_jk` appears in no regression input). Add an
  RI-JK geomopt case too, because it is what exercises invalidation.
- Expected: about 3× on the RI-JK Fock build.

### B1 — Optional BLAS build

- New CMake option `USE_BLAS` that defines `EIGEN_USE_BLAS` and links
  Accelerate on Apple or `find_package(BLAS)` elsewhere. Eigen uses its own
  BLAS prototypes, so the fact that `Accelerate.h` does not compile under
  g++-15 (measured: `vBasicOps.h` vector-cast errors) should not matter. Verify
  that on the first build.
- **Hermeticity:** "no system installs" is a stated project property. Accelerate
  is always present on macOS; Linux needs a system or fetched OpenBLAS. Default
  **OFF** on Linux. Whether to default ON for Apple is a decision for the user.
- This speeds up every large Eigen GEMM, not just RI: RI-K, the pair-factor
  build (if B0 has not already removed it), DFT AO-grid products, RI-MP2.
- Gates: smoke + extended suites at existing tolerances; `h2_fciqmc_threads1/4`
  and `n2_fciqmc_s5_threads1/4` in the BLAS build; `planck-ri-jk-equivalence`.
  Re-pin nothing to a BLAS build. The Eigen build stays the reference.

### B2 — RI-K from occupied orbitals (no BLAS)

- `K = Σ_Q (B_Q C_occ)(B_Q C_occ)ᵀ` does `nocc/nb` of the work. It is exact
  **only for idempotent densities** `D = C_occ C_occᵀ`.
- Five non-SCF call sites pass non-idempotent densities: CASSCF `D`/`D_act`, the MP2-gradient
  relaxed density, and the UHF response trial densities. Add an overload taking
  `C_occ` and use it only at the two SCF call sites (`scf.cpp` RHF and UHF). Do
  not change `build_ri_k(D)`.
- Gate: RI-JK SCF energies within 1e-10 of the D form (the summation order
  changes), plus the new B0 regression case.

### B3 — LAPACK eigensolver (low priority)

- One wrapper (`dsyevd`, size cutoff around 200, Eigen below it) at the
  SCF/orthogonalizer diagonalization sites only. `EIGEN_USE_LAPACKE` cannot be
  used because Accelerate ships no `lapacke.h`.
- Risk: eigenvector signs and rotations within degenerate subspaces differ from
  Eigen. Energies are unaffected, but MO-coefficient comparisons, irrep labeling
  of degenerate sets, and checkpoint-restart cases could move. Run the full
  symmetry and checkpoint ladders.
- Not worth doing until a run at nb ≳ 1000 shows diag in the phase timer.

## Ruled out (measured, do not revisit without new evidence)

- **ERI kernels (VRR / HRR / Boys).** They aren't GEMM-shaped, and they already
  run at about 85–95 % of peak IPC (`docs/ERI_PERFORMANCE_SCOPE.md`). BLAS has
  nothing to replace.
- **Direct-SCF Fock digestion** (`fock_accumulate.h`, `fused_fock.h`). Each
  quartet scatters into J/K with 1–9 component blocks on s/p bases, so a BLAS
  call's overhead exceeds the work. The direct build is also 99.8 % ERI
  evaluation, not digestion.
- **Conventional `_compute_fock_rhf`** (`os.cpp`). It's a bandwidth-bound GEMV
  over the stored tensor. Replacing it with two `dsymv`s ran 0.72× / 3.0× /
  1.03× at nb 40/70/100 on 1 thread and 1.69× / 0.93× / 0.79× on 4 threads, with
  no consistent win. The K half also needs a second, permuted `nb⁴` copy, and
  conventional mode is capped at nb ≤ 100 (`_threshold`), where one contraction
  takes about 0.04 s. The exception is **many densities at once** (CPHF,
  stability, response), which turns the GEMV into a GEMM. That is a separate
  question and is not scoped here.
- **Cartesian-to-spherical ERI transform as a dense `dgemm`.** It ran
  **0.14–0.62× (slower)**. `C` is only 2.5–4 % nonzero (block-diagonal per
  shell), and the existing `c == 0.0` skip loop already exploits that. A
  per-shell-block GEMM might pay, but the transform runs once per SCF.
