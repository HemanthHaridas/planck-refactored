# Why does a 32-water RKS run not converge, and what makes it converge?

Scope, opened 2026-11-10. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 2).

## The one data point

`docs/SCALE_RESULTS_NOTCHPEAK.md` (2026-07-20, commit `3ccbdc4e`): 32 water,
B3LYP/6-31G, nb = 416, 1 rank, `engine os`, `grid normal`, **`guess hcore`**,
DIIS 8, `tol_density 1e-7` → `RKS did not converge in 100 iterations`. HF
converged every case of the same ladder. Inputs come from
`tests/benchmarks/scale_bench.py`.

That run predates the memory-direct KS J/K. Open Work records a later
serial run that still hits `max_cycles` at 4.8 GB ("it ran and iterated"),
so the failure survived the J/K change. It is not recorded whether that run
also predates Boys v1.2.1, the true-Fock re-diagonalization at convergence
and the incremental Fock build's screen-off tail.

## What exists

- `max_cycles` defaults by basis size: 50 / 100 / 200 / 300 at
  nb ≤ 250 / 500 / 1000 / above (`max_cycles_for_nbasis`,
  `src/base/types.h`). 32 water lands on 100.
- SAD guess (`guess sad`), DIIS with `diis_restart`, level shift (RKS and
  UKS), SOSCF for RKS/UKS with an automatic stall trigger
  (`scf_soscf_auto_stall`), all opt-in. The default guess is `hcore`.

## Steps

### 1. Reproduce on the current build

Run the identical input with `max_cycles 300`. Record per-iteration ΔE and
DIIS error. Three outcomes:

- converges in > 100 iterations → slow, not stuck; go to step 2;
- DIIS error plateaus or oscillates → stuck; go to step 3;
- now converges in ≤ 100 → the defect was fixed in passing; write that down
  and stop at step 4.

4.8 GB is close to this machine's 8 GiB: do not time it here
(`benchmark_ladder_design` memory). Iteration counts are fine locally; timings
belong on the cluster.

### 2. Guess

Rerun with `guess sad`. An `hcore` guess on 96 atoms is a poor start, and
the HF ladder converging says little about it (HF has no grid). If SAD fixes
it, the decision is whether `planck-dft` should default to SAD. That
reopens the 2026-11-09 decision to keep `hcore` (which was argued for HF
open-shell cases); scope it as a DFT-only default and run the full DFT suite
(77 cases) under it.

### 3. Accelerator

If SAD alone does not converge it: try `scf_soscf_auto_stall`, then a
level shift, then EDIIS/ADIIS (not implemented; the standard cure for
DIIS stagnation far from convergence). Implement EDIIS+DIIS only if the first
two fail, in `src/scf/diis.cpp`, shared by HF and DFT.

### 4. Gate

A medium-size regression case cheap enough for the extended suite: the
smallest water chain that reproduces the failure (try 12–16 water with
6-31G), run with defaults, asserting convergence and an energy pinned to
PySCF (`guess` irrelevant at convergence). Add a 32-water run to the cluster
runbook, not the suite.

## Not in this scope

Fractional occupation / smearing (metals, not water), and the MPI scaling
of DFT (`docs/HPC_MPI.md`).
