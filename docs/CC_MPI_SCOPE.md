# How should `planck-mpi` distribute coupled cluster and (T)?

Scope, opened 2026-11-20. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md`. Related: `docs/HPC_MPI.md` (how the SCF is
distributed), `docs/CCSD_T_SIZE_SCOPE.md` (what limits CCSD(T) size today),
`docs/RI_MPI_SCOPE.md` (the RI pattern this would follow).

## The state

- `planck-mpi` compiles all of `POSTHF_SRC` and runs the same driver, so CC
  already runs under `mpirun`. No CC code makes an MPI call, though, so
  every rank runs the full solve redundantly.
- The AO ERI build underneath is already striped and reduced (#181,
  `docs/HPC_MPI.md`); the CC that consumes it is not.
- The building blocks exist in `src/base/mpi_env.h`: `Mpi::Stripe`
  (round-robin ownership of an index range, `owns(i)`) and
  `allreduce_inplace`. RI uses them for the 3c build, RI-J/K and the RI-MP2
  loops (`docs/RI_MPI_SCOPE.md`, M0–M1).
- `planck-mpi` takes the same CC compile definitions as `hartree-fock`
  (one list, `_planck_posthf_cc_defs`; since 2026-11-25). Before that it
  lacked `PLANCK_CC_UCC` and `PLANCK_CC_DRESS_OPERATORS`, so it had no
  `uccsd(t)` and ran a different CC path from the serial binary.
- `tests/mpi_smoke_compare.py` compares `Total Energy`, every correlated
  total (`Total <label> Energy`, so `Total RCCSD(T) Energy` and
  `Total UCCSD(T) Energy` too) and gradient rows. The baseline gates
  `water_rccsd_t_631g_mpi_smoke` and `oh_uccsd_t_631g_mpi_smoke` pass at 2
  and 3 ranks with `--atol 0` (replicated CC, 2026-11-25).

## Two different things "distributed" can mean

| | split the work | split the memory |
|---|---|---|
| data | every rank holds full t1/t2, MO blocks and AO tensor | amplitudes and large blocks (`vvvv`) spread across ranks |
| per iteration | each rank computes the residual for its slice of an outer index; one allreduce | contractions need communication between ranks |
| buys | wall time | system size |
| cost | small: the SCF and RI pattern | a distributed-tensor design; the emitter assumes whole local tensors (`t2({i,j,a,b})`, `mo_blocks.ovvv(...)`) |

Since the memory work in `docs/CCSD_T_SIZE_SCOPE.md`, time is the limit,
not memory: the water dimer in cc-pVTZ peaks at 1.43 GB and takes about 37 s
(4 threads). Splitting the work therefore pays now. It still fits nothing
new; splitting the memory is the only MPI lever on size, and the next
memory walls (the ns⁴ AO tensor in the SCF, v⁴ `vvvv` in the CCSD) have
single-node fixes first (direct or RI integrals). `docs/HPC_MPI.md` also
says to distribute the memory-lean algorithm, not the dense one.

## What should come first

These are cheaper and some are prerequisites:

1. **Thread the generated kernels by default.** Done 2026-11-20:
   `PLANCK_CC_OMP_COLLAPSE` (default 3) is passed to the generator. The
   results are bitwise unchanged, and `oh_uccsd_t_631g` runs 3.4× faster at
   4 threads (`docs/CCGEN_GENERATED_CC_COST.md`). MPI ranks on one node now compete
   with these threads, so a node should run one rank with threads, not
   many single-threaded ranks.
2. **A spatial RCCSD.** Done 2026-11-22: `ccsd(t)` runs the generated
   spatial RCCSD, 48–59× faster than the spin-orbital one and with 2.5–4×
   less memory (`docs/CCSD_T_SIZE_SCOPE.md`).
3. **RI** for the integral side, if the nb⁴ tensors bind before the
   amplitudes. RI breaks the nb⁴ wall on one node with no new executable.
   Not needed yet: at the largest measured size, time binds first.

## Steps

### M0. (T) across ranks (cheap; its condition is now met)

The batched (T) is already shaped for this. Its work items, one per virtual
triple (or per block and triple for UHF), are independent, and each writes
its own slot in `e_triple` / `e_item`:
- each rank fills only the slots it `owns(t)` and leaves the rest zero;
- `allreduce_inplace` on the slot vector, where every slot has one nonzero
  contributor, so the sum is x + 0 + … + 0 = x exactly;
- the existing ordered serial sum over slots, on every rank.

E(T) is then bitwise identical to serial at any rank count, with no
dependence on MPI's reduction order. The cost is one allreduce of a vector
of about v³/6 doubles.

Gate: `*_rccsd_t_*` and the spherical case at 2 and 3 ranks, every printed
digit equal to serial. Add a mutation: drop the allreduce. Per
`mpi_missing_reduce_deadlocks`, that is expected to give a wrong energy
here, not a hang, because no later collective waits on it. Use
`tests/mpi_smoke_compare.py` with its timeout anyway.

(T) is under 1 % of the C1 H2O2 6-31G run and about 12 % of the water
dimer cc-pVTZ run (4.6 of ~37 s, after size-scope step 1a). It is a
measurable share at size, but smaller than the CCSD that M1 splits.

### M1. Split the work for the CCSD residual

Apply to the CCSD residual the same pattern as the SCF and RI:
- **Where.** Generated route (the spatial RCCSD behind `ccsd(t)`): stripe the
  outermost free index of each residual nest, then one `allreduce_inplace`
  per residual tensor per iteration (o²v² for doubles). The emitter already
  writes each nest as a loop over disjoint `result(i, ...)` slices, which is
  what made the OpenMP collapse reduction-free. A rank stripe is the same
  shape one level up.
- **Not the hand-written spin-orbital RCCSD.** It is the solver to replace
  (size scope), so do not invest in distributing it.
- **Determinism.** One rank owns each residual element and all others
  contribute 0, so the allreduce is exact, as in M0. Then the update, DIIS
  and the energy run replicated.

Gate: serial versus 2 and 3 ranks, every printed digit, on `ccsd(t)` and
`uccsd(t)` cases, plus the drop-the-reduce mutation. `planck-mpi` has the
UCC and dressing options, and the comparator reads the CC totals, since
2026-11-25; extend the two `*_mpi_smoke` CC cases above.

Measure before claiming a speedup. Per-iteration time includes replicated
DIIS and the energy, which do not shrink with the rank count.

### M2. Split the memory (only if M1 and the cheaper levers are not enough)

This is a separate design question, scoped here only to what it would
require:
- **Ownership.** Distributed ownership of t2 and `vvvv`. For example, own
  the `vvvv` blocks by the first virtual index and the t2 blocks by the
  first occupied index.
- **Kernels.** The generated kernels would read non-local blocks through a
  fetch layer (one-sided gets or a batched exchange) instead of direct
  accessors, which is an emitter change. Alternatively, contract through an
  external distributed-tensor library such as Cyclops or TiledArray. That
  would be a new dependency, and the build doc's hermetic-dependency
  rule applies.
- **Rank-count independence.** Communication patterns that keep the energy
  independent of the rank count (fixed-order reductions throughout).

Do not start M2 without measurements from the size scope showing that a
single node (spatial RCCSD, RI, threads) is saturated.

## Not in this scope

Distributed CCSDT and higher, distributed CC gradients, and MPI for MP2
(it would follow M1's pattern but is its own small scope).
