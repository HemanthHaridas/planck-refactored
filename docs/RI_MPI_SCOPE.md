# RI under MPI — scope

**Question.** How should `planck-mpi` split the RI work (RI-JK SCF, RI-MP2,
RI gradients) across ranks, and in what order?

Status: **M0 and M1 landed 2026-11-17** (see "M1 as implemented" at the
end). M2 is not started and is still scope.

## Where it stands

No file under `src/post_hf/ri/` refers to MPI. The MPI-aware code is the
fused direct Fock build, the OS/HGP/Rys/`auto` conventional ERI builds, the
DFT grid and the UKS double-hybrid driver (`docs/HPC_MPI.md`). RI-JK bypasses
the fused Fock loop, so none of that reaches it: under `mpirun -n k`, every
rank runs the full RI calculation.

The answer is correct. `water_rhf_rijk_631gd` gives −76.0076329813 Eh from
`hartree-fock` and from `planck-mpi` at 1, 2 and 3 ranks. But extra ranks buy
nothing, and nothing tells the user so. None of the ten `*_mpi_*` regression
cases uses RI.

Measured on 16 waters / 6-31G / def2-universal-JKFIT (the
`water16_rhf_631g_auto_direct_mpi_scale` geometry with `scf_ri_jk`), one
OpenMP thread per rank, 2026-11-17 (pre-B2 build):

| ranks | wall | user CPU | energy |
|---|---|---|---|
| 1 | 159.7 s | 152.0 s | −1215.7175270085 |
| 2 | 186.2 s | 341.8 s | −1215.7175270085 |

Two ranks are slower and burn twice the CPU: each rank repeats the whole
calculation, and the two compete for memory bandwidth.

## Why the split is easy

Every RI contraction is a sum over auxiliary functions Q:

| Quantity | Form | Per-rank piece |
|---|---|---|
| 3-center tensor | `(μν\|P)` per aux shell | the rank's aux shells |
| RI-J | `J = Σ_Q B_Q c_Q`, `c_Q = Σ B_Q·D` | `c_Q` and the J sum over the rank's Q |
| RI-K | `K = Σ_Q (B_Q D) B_Qᵀ` | the rank's Q |
| RI-MP2 energy | `(ia\|jb) = Σ_Q B_ia,Q B_jb,Q` | stripe the occupied pair (i, j) instead |
| RI gradient | one partial per aux shell, summed in shell order (G6) | the rank's aux shells |

So each one is the pattern the conventional builds already use: stripe, then
one `Mpi::allreduce_inplace`. The reduced objects are small (nb², natoms×3,
a scalar), except the 3-center tensor itself.

The one thing that is not a sum over Q is the metric. The fitted factors
`B = (μν|P) V^{-1/2}` (or the Cholesky solve) mix aux columns. That is the
only design question, and it only matters for the memory phase.

## Steps

### M0 — gate and warn (small)

- Regression cases `water_rhf_rijk_631gd_mpi_smoke` (2 and 3 ranks), plus
  one RI-MP2 energy and one RKS RI-JK case, each requiring serial and MPI to
  agree on every printed digit. They pass today, because every rank computes
  everything. They are the baseline every later step must keep.
- A one-line `[WRN]` when `Mpi::distributed()` and RI is on: "RI is not
  distributed; every rank does the full RI work." It is deleted by M1.

### M1 — split the work, keep the data replicated (the time win)

Each rank still holds the full `npair × naux` tensor. Only the work is split.

1. `compute_3c_eri`: each rank computes its aux shells' columns, and the
   others stay zero. One allreduce of `npair × naux`. Then the metric is
   applied on every rank, as now. The allreduce of the full tensor is the
   price of keeping B replicated, and it happens once per geometry.
2. `build_ri_j` / `build_ri_k` / `build_ri_fock_{rhf,uhf}`: the Q loop runs
   over the rank's Q, then one allreduce of J (and of K, or of both spins).
   `build_ri_3index_unpacked` builds only the rank's `B_Q`. That is the first
   memory saving: the `naux × nb²` unpacked set drops by 1/k.
3. RI-MP2 energy: stripe the outer occupied index, allreduce the scalar.
4. RI gradients (`for_each_3c_deriv`, `for_each_2c_deriv`): stripe the aux
   shells, allreduce `natoms × 3`. G6 already sums per-shell partials in
   shell order, so the serial result is unchanged.
5. The fitted orbital-Hessian action (SOSCF, stability, CPHF) goes through
   `two_electron_fock`, so it inherits step 2 with no extra code.

**Gate:** the M0 cases, plus a mutation check like the one the `auto` stripe
got (dropping one allreduce must break the energy). Serial output must stay
byte-identical: every split is behind `Mpi::distributed()`.

**Bitwise across rank counts is not a goal.** MPI_Allreduce does not fix its
summation order. The existing MPI gates require agreement on every printed
digit, not bitwise equality, and M1 keeps that bar.

### M2 — distribute the tensor (the memory win)

M1 still keeps the full B (`npair × naux` doubles) on every rank. With a JKFIT
set naux ≈ 3·nb, that is about 12 GB at nb = 1000 and about 96 GB at
nb = 2000, so M2 matters only past nb ≈ 1000–1500 on a normal node. Below
that, one node with RI already removes the nb⁴ wall, which is
`docs/HPC_MPI.md`'s standing advice.

Design to settle first (a decision, not code):

- **Keep `(μν|P)` unfitted and distributed by P.** J needs only the replicated
  fitted density `d = V^{-1} c` (naux, small). For K, build
  `X_{μi,P} = Σ_λ (μλ|P) C_λi` locally from `C_occ`, then apply V⁻¹ across
  ranks with a reduce-scatter on `nb·n_occ × naux`, which is smaller than B
  by n_occ / nb. B2 (`docs/ERI_FOCK_BLAS_SCOPE.md`, landed 2026-11-17)
  already computes RI-K from a low-rank factor X of D, which is the
  `C_occ` this formulation needs, for any density.
- RI-MP2 needs `B_ov` (o·v·naux), which is small enough to keep replicated
  after one distributed build.
- Gradients would need the 3-index density striped the same way.

## M1 as implemented (2026-11-17)

One ownership rule, `Mpi::Stripe` (`src/base/mpi_env.h`: rank r owns index
i when i % size == r; a serial build owns everything), and one reduce,
`sum_over_ranks` (`ri_eri.cpp`, a no-op when serial). Each split function
reduces its own result, so no caller changed.

| Function | Split over | Reduced |
|---|---|---|
| `compute_3c_eri` | aux shells | the Cartesian 3c tensor, before the transforms |
| `ensure_ri_3c_ready` (B = J V^{-1/2}) | AO pair rows (serial path unchanged) | B |
| `build_ri_j` | AO pair rows, both passes | c, then J |
| `build_ri_3index_unpacked` | aux Q: other ranks' B_Q stay empty (0×0) | — |
| `build_ri_k` | the B_Q this rank holds | K |
| `build_ri_mo_block` | AO pair rows | B_pq |
| `fitted_gram` (new; MP2 ovov/ovOV and `transform_eri_ri`) | aux Q | the Gram |
| RMP2 OVOV tile stream (`mp2_internal.h`) | aux Q | each tile, before the consumer |
| `for_each_pair_block_w` → `build_ri_imat_from_pair_blocks`, `build_ri_gamma3_from_pair_blocks` | aux Q | imat, gamma3 |
| `build_ri_scf_two_electron_gradient` (gamma3) | aux Q | gamma3 |
| `for_each_{3c,2c}_deriv` → `build_ri_two_electron_gradient` | aux shells | the gradient |

Differences from the plan above:

- **RI-MP2 energy** is split over aux Q in the `B_ov B_ovᵀ` products, not
  over the occupied index. The MP2 kernels consume the ovov tensor or tile
  stream; splitting where it is built leaves them untouched.
- **No M0 warning.** M1 makes it untrue, so it was never written.
- **Still replicated:** the 2-center metric and its factorization (naux²,
  small), and `build_ri_x_pair_factors` (one npair·naux² product per
  gradient). Split them only if a profile shows them.
- `compute_3c_eri_deriv` / `compute_2c_eri_deriv` (test-only full tensors)
  pass `Stripe{0, 1}` and stay whole on every rank.

Contract: every rank must reach every reduce. None of these functions may
be called from a rank-dependent branch or inside an OpenMP region.

Gates (`tests/regression_cases.json`, tag `ri`, through
`tests/mpi_smoke_compare.py`, which now also compares printed MP2 energies
and gradient rows): `water_rhf_rijk_631gd`, `water_cation_uhf_rijk_gradient_631gd`,
`water_rks_b3lyp_rijk_631gd`, `water_rhf_rijk_soscf_stability_631gd`,
`water_cation_rimp2_ccpvdz`, `water_ri_rmp2_gradient_631gd` and
`water_casscf_ri_ccpvdz`, each `_mpi_smoke`, serial against 2 (and 3) ranks.
Tolerance is 2e-10 for energies and 2e-8 where gradient rows (8 decimals)
are compared: the partial sums reassociate, so this is not bitwise.

### Measured

- **Serial unchanged:** all 45 RI regression inputs print byte-identical
  output to the build before M1; the nine `planck-ri-*` ctest gates pass.
- **MPI gates:** all seven pass. In practice every energy, MP2 energy and
  gradient row matched serial exactly (|Δ| = 0) at 2 and 3 ranks.
- **Speed**, 16 H2O / 6-31G / def2-universal-JKFIT RHF, one OpenMP thread
  per rank (post-B2 build):

  | ranks | wall | speedup | energy |
  |---|---|---|---|
  | 1 | 31.8 s | 1.00× | −1215.7175270085 |
  | 2 | 19.6 s | 1.62× | −1215.7175270085 |
  | 4 | 16.5 s | 1.93× | −1215.7175270085 |

  The work still replicated (metric factorization, diagonalization, SCF
  bookkeeping) caps it; profile before splitting more.
- **Mutation:** removing the sum in `build_ri_k` does **not** give a wrong
  energy. It gives a hang. Each rank holds a different K, the SCFs diverge
  and stop making matching collective calls, and rank 0 waits forever in
  `MPI_Allreduce`. `tests/mpi_smoke_compare.py` therefore now has a per-run
  `--timeout` (SIGTERM first, so `mpirun` takes its ranks down) and names
  the likely cause; the RI gates use 30 s. Against the mutation the gate
  fails in 31 s with no ranks left behind. Lesson for any future split:
  **a missing reduce shows up as a deadlock, not a wrong number**, so every
  MPI gate needs a timeout.

## Not in this scope

- Distributed dense post-HF (rank-striping CC/MP2 on the outer MO index;
  `docs/HPC_MPI.md`). That is independent of RI.
- RI-DH (O9) and attenuated RI-K. When they land they inherit M1's stripe if
  they are written as sums over Q.
