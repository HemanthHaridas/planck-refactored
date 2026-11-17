# RI under MPI — scope

**Question.** How should `planck-mpi` split the RI work (RI-JK SCF, RI-MP2,
RI gradients) across ranks, and in what order?

Status: scope, 2026-11-17. Nothing here is built.

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

## Not in this scope

- Distributed dense post-HF (rank-striping CC/MP2 on the outer MO index;
  `docs/HPC_MPI.md`). That is independent of RI.
- RI-DH (O9) and attenuated RI-K. When they land they inherit M1's stripe if
  they are written as sums over Q.
