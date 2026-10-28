# Why was an analytic gradient 20–120× the SCF it follows?

**Answer: none of the cost was intrinsic. Every analytic gradient in the HF
binary (RHF, UHF, ROHF, RKS/UKS through the same loops, RMP2 and UMP2) now
costs 1.1–4.3× the SCF it follows, down from 20–120× and, for the MP2 cases,
minutes.** There was no single cause. The derivative ERIs were evaluated per AO
component, over the full square of AO pairs, on one thread. The MP2 gradients
carried an unread O(N⁸) density and rebuilt N⁵ intermediates inside a loop
they did not depend on. The UMP2 Z-vector built its whole orbital Hessian to
solve one linear system, and its back-transforms ran as scalar n_occ² sums.
The 1e block, pasted six times, re-ran the nuclear recurrence for every axis.

Measured on M3, `build-full`, direct SCF, `engine auto`, SAD, no symmetry,
four threads, best of three, whole run including the SCF:

| system | SCF | gradient run 2026-10-25 | **now** | now ÷ SCF |
|---|---|---|---|---|
| 4 H2O / 6-31G\* (nb 76) | 0.38 s | 53.9 s | **0.50 s** | 1.3× |
| 8 H2O / 6-31G (nb 104) | 0.56 s | 68.0 s | **0.78 s** | 1.4× |
| 1 H2O / cc-pVTZ (nb 65 Cart.) | 0.54 s | 114.9 s | **0.59 s** | 1.1× |
| 1 H2O / 6-31G\* RMP2 (conventional) | 0.08 s | 7.30 s | **0.12 s** | 1.5× |
| 2 H2O / 6-31G\* RMP2 (conventional) | 0.15 s | > 5 min | **0.65 s** | 4.3× |
| 2 H2O⁺ / 6-31G\* UMP2 (conventional) | 0.16 s | 136 s (after G3) | **0.53 s** | 3.3× |

The 2026-10-25 runs had a concurrent compile on the machine (±20 %) and the
SCF got faster in between, so compare gradient columns, not ratios across
dates. No step changed a printed gradient, and every step is identical at any
thread count.

The step labels (G1–G5, G7) match the commits. This work follows
`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, which made the memory-direct Fock
build fast.

## The starting point (2026-10-25)

- The gradient sat at ~99 % CPU at four threads: nothing in `src/gradient/`
  or `mp2_gradient.cpp` carried a pragma.
- 4 H2O / 6-31G\*, one thread: `engine os` 71.4 s, `engine hgp` 47.0 s.
- Profile (8 H2O / 6-31G, one thread, 15 s): `_compute_eri_deriv_elem`
  98.5 % inclusive of `compute_rhf_gradient`; self time led by `_eri_vrr`
  36 %, HRR 16 %, `a0c0` build 13 %, Boys 5 %,
  `EriScratch::resize_for_quartet` 3 %, memmove/memset 7 %, emulated TLS 3 %.
  Nuclear-derivative 1e terms ~5 %.

## The causes, in the order they were removed

### 1. Dead work and the wrong engine (G1)

- `compute_eri_deriv_dispatch` sent `engine auto` to OS. It now resolves Auto
  to HGP (precedent: `full_symmetry_engine`). 4 H2O / 6-31G\* RHF gradient,
  four threads: 56.4 → 29.7 s, 1.90× against an estimated 1.5×.
- A second RMP2 derivative sweep fed only the `PLANCK_DEBUG_RMP2_TERMS`
  print; it now runs only when that flag is set. 1 H2O / 6-31G\* RMP2
  6.78 → 4.27 s.
- The MP2 sweeps built two `nao²` AO `ShellPair`s per innermost iteration;
  they were built once instead (later superseded by G4). With all of G1 the
  RMP2 case was 6.78 → 3.17 s.

### 2. A square, serial loop (G2)

UHF and ROHF had their own copies of the 2e loop. All references now share
one loop that visits each unordered pair of AO pairs once (`q ≥ p`). It folds
Γ over the index swaps and threads over 64 fixed bins. The unused
`assume_pair_exchange_symmetry = false` fallback (a full `nb²` ordered pair
list) is deleted. UKS had run its long-range exchange sweep three times; it
now runs once.

4 H2O / 6-31G\* RHF gradient, `engine auto`:

| | 1 thread | 2 | 4 | 8 |
|---|---|---|---|---|
| G1 | | | 30.5 s | |
| G2 | 12.8 s | 6.45 s | 3.42 s | 2.43 s |

One thread beats G1 at four by 2.4×. G1 never used more than one core, and
the triangle halves the work.

RKS and UKS also ran a second full Coulomb-only sweep on every gradient, to
fill a `WavefunctionGradientBreakdown` that only the
`PLANCK_DFT_GRADIENT_DEBUG` report reads. `compute_{rks,uks}_gradient` now
take `with_breakdown` (default false). 4 H2O / 6-31G\* B3LYP gradient run
7.94 → 5.36 s.

### 3. Per-component derivative integrals (G3)

The per-component derivative evaluates up to 24 contracted ERIs per AO
quartet, each with its own primitive loop. Two shell-quartet block kernels now
compute every component from one primitive pass:

- `HgpDerivBlockKernel` (`hgp_deriv_block_eri.h`): one VRR to
  (L_A+L_B+1 | L_C+L_D+1) contracted with weights 1, 2α, 2β, 2γ, then the
  shared `hgp_block::hrr` on each raised and lowered quartet.
- `RysDerivBlockKernel` (`rys_deriv_block_eri.h`): per-axis differentiation
  of the 2D Rys tables, libcint's route.

Both take D from translational invariance. A block the chain declines falls
back to the per-component derivative.

**Kernel accuracy.** `planck-eri-derivative-kernels` checks every component
and all 12 derivatives against per-component OS on water in 6-31G, 6-31G\*,
cc-pVDZ and cc-pVTZ (D included, which OS computes explicitly), relative to
the sweep's largest derivative:

| | 6-31G / 6-31G\* | cc-pVDZ | cc-pVTZ |
|---|---|---|---|
| HGP block | 2.6e-15 | 5.9e-15 | 1.8e-13 |
| Rys2D block | 2.8e-15 | 4.9e-15 | 1.0e-13 |
| per-component OS vs HGP | | | 2.7e-13 |

A 1e-13 bar was below the floor: on cc-pVTZ (f raised to g) the two
per-component engines differ by more than that, and both block kernels sit
inside the spread. The gate is 5e-13. Mutations (2β → β, lowered term
dropped, D without C) each fail on every basis.

**Crossover, measured rather than assumed** (whole run, four threads, best
of two):

| | energy crossover rule | HGP only | Rys2D everywhere |
|---|---|---|---|
| 4 H2O / 6-31G\* | **0.57 s** | 0.58 s | 0.72 s |
| 4 H2O / cc-pVDZ | **1.46 s** | 1.63 s | 2.02 s |
| 1 H2O / cc-pVTZ | **0.62 s** | 0.68 s | 0.69 s |

The energy build's rule (Rys2D for one-primitive blocks at L ≥ 4 and
three-quartet blocks at L ≥ 7, HGP for the rest) also wins for derivatives.

Whole run against G2, four threads: 4 H2O / 6-31G\* 4.75 → 1.21 s, 1 H2O /
cc-pVTZ 3.58 → 0.67 s (per-component OS: 10.43 s, 12.23 s).

### 4. MP2 gradients outside the shared loop (G4)

The 2e loop moved into `src/gradient/two_electron_gradient.h`, a pure move.

- **RMP2.** A profile of the 2 H2O / 6-31G\* gradient put every busy sample
  in `build_rmp2_gradient_intermediates`' own loops, none in derivative
  integrals. The cause was an AO pair density over μνλσ × ijab, O(N⁸), stored
  in a field (`Gamma_pair_ao`) nothing read. It is deleted, along with the
  equally unread `two_e_{non,}separable_gradient` fields. The per-atom
  derivative sweep is replaced by one `two_electron_gradient` call with
  Γ = 2 dm2buf (dense only) plus the separable term. 358 → 0.65 s.
- **UMP2.** Its sweep rebuilt the separable densities D_σ[p,v,r,s] inside
  the q loop, which they do not depend on: O(nao⁵ nocc²). They are now built
  once, Imat_σ is a separate nao⁵ contraction threaded over q (or
  `build_ri_imat` under RI), and the 2e term is one `two_electron_gradient`
  call. 136 → 3.5 s.

Why the MP2 terms fit a loop written for SCF: they were a bra-only
contraction Σ_{p on A} X ∂_p(pq|rs). With X symmetrized over r ↔ s and
8-fold symmetric, which holds for both the pair density and the separable
part (T̃ = 4t − 2t is symmetric under (i,a) ↔ (j,b)), that equals the loop's
all-centre form with Γ = X. No explicit symmetrization was needed.

### 5. The UMP2 orbital side (G7)

After G4 almost none of the UMP2 gradient was derivative integrals. Profile
(2 H2O⁺ / 6-31G\*, ~3,040 main-thread samples in `compute_ump2_gradient`):
59 % `solve_uhf_cphf` (1,443 samples in direct UHF Fock builds, 333
recomputing the Schwarz table inside each one), 23 % D_σ and Imat, 14 % serial
`part_dm2`, < 1 % the 2e derivative term.

- **Matrix-free UHF Z-vector.** `build_uhf_cphf_matrix` formed the
  (n_ov^α + n_ov^β)² orbital Hessian one unit column at a time, one full
  direct UHF Fock build each (28·10 + 29·9 = 541 here), then QR-solved it.
  The column body is now one operator, `UHFOrbitalHessian`
  (`uhf_response.cpp`). The dense builder still applies it to unit vectors,
  because SOSCF needs the matrix. `solve_uhf_cphf` hands the operator to
  `DFT::Response::gmres`, the solver the DH Z-vector already uses, with the
  ε_a − ε_i diagonal as preconditioner. The operator is symmetric and
  positive definite at a stable minimum, so CG would also work, but GMRES
  already checks its unpreconditioned residual with a fresh action and a
  second solver would save only Krylov storage. 16 iterations, **33 Fock
  builds instead of 541**; run 3.5 → 1.83 s. RHF is unchanged:
  `solve_rhf_cphf` builds its matrix from MO ERIs, with no Fock build per
  column.
- **Back-transforms as matrix products, one mechanism for both MP2
  flavours.** Three helpers in `mp2_gradient.cpp` replace RMP2's serial
  `dm2buf_full` loop and UMP2's `part_dm2` + D_σ loops:
  - `pair_half_density`: C_l T_ij C_rᵀ per (i,j), threaded over pairs;
  - `back_transform_half_density`: two reshaped GEMMs, nao⁴·n_occ instead
    of nao⁴·n_occ²;
  - `add_symmetrized`: the p↔q / r↔s / bra↔ket image sums, threaded over p.

  D_β's αβ term is B_αβ read bra↔ket, so one B_αβ serves both spins. Run
  1.83 → **0.55 s**. Peak memory grows by one nao⁴ buffer during the D_σ
  build.
- Imat as a GEMM was planned and dropped: it was 1.3 % after the Z-vector
  fix.

The comparison against the dense Z-vector needs care. At GMRES tolerance 1e-10
z agrees with the dense QR to 1.2e-8 of max|z|; tightening to 1e-12/1e-14
floors at ~5e-10. That floor is the dense reference's own noise, 541 screened
direct builds. The printed gradient is identical either way. On STO-3G
(cation / triplet / RI) the agreement is 1.6e-15 / 2.0e-15 / 1.3e-14. The new
back-transforms agree with the old loops to ≤ 6.1e-16 of the tensor's largest
element on the STO-3G cation, triplet, RI-UMP2 and RMP2 inputs, and 1.9e-15
on 2 H2O⁺ / 6-31G\*.

### 6. Six copies of the 1e block (G5)

Priced before building: on the RHF gradients the 1e term was 14–32 % of the
gradient call (main-thread samples: 8 H2O / 6-31G 64 of 203, 4 H2O / 6-31G\*
29 of 108, 1 H2O / cc-pVTZ 8 of 57). Almost all of it was the
nucleus-position derivative, which ran the whole nuclear VRR once per (atom,
axis).

- `Gradient::one_electron_gradient(calc, ao_pairs, D, W)` replaces six
  copies:
  - four in `gradient.cpp` (RHF/RKS, UHF, ROHF, UKS);
  - the MP2 helper, which allocated nine nao² matrices per atom and re-ran
    the C-derivative over all AO pairs per (atom, axis);
  - the DH Eq. 33 h/S contraction in `dft/driver.cpp`, which walked ordered
    nao² pairs and recomputed the all-nuclei A-derivative per (atom, axis).

  The unused public `contract_rmp2_one_electron_gradient` is deleted.
- `_compute_nuclear_deriv_C_xyz` shares the Boys values and V VRR across the
  three directions. The recurrences are unchanged, so each direction is
  bitwise the old kernel.
- Threaded over AO pairs in 64 fixed bins.

Whole run: 8 H2O / 6-31G 0.87 → 0.78 s, 4 H2O / 6-31G\* 0.54 → 0.50 s;
cc-pVTZ and UMP2 unchanged within noise. 1e main-thread samples on 8 H2O
64 → 19. The gain matched the price measured beforehand. One symmetry-zero
component on 8 H2O flips the sign of its printed zero.

## How the gradient runs now

The 2e loop lives in `src/gradient/two_electron_gradient.h`, the
SCF-reference gradients and the 1e block in `src/gradient/gradient.cpp`, and
the MP2 ones in `src/post_hf/mp2_gradient.cpp`.

- **One 2e loop for every reference.** RHF, RKS, UHF, ROHF, UKS, RMP2 and
  UMP2 all call `two_electron_gradient(calc, shell_pairs, gamma_fn)` with
  their own Γ. The result is ¼ Σ over ordered AO quartets of
  Γ(p,q,r,s) d(pq|rs)/dR.
- **Permutations only change Γ.** In every index swap of a quartet, each
  centre's derivative still goes to that centre's own atom, so
  `pair_permuted_gamma` sums Γ over the (ij)/(kl) swaps and the bra–ket swap,
  and one derivative evaluation serves them all. This is exact for any Γ. A zero Γ
  skips the derivative entirely.
- **`engine auto`, Coulomb kernel: shell quartets.**
  `accumulate_block_eri_gradient` walks shell-group pairs Q ≥ P and visits
  exactly the AO quartets the AO loop would (i ≤ j within a group; each
  unordered AO pair of pairs once in a diagonal block). Each block's 12
  derivatives per component come from one primitive pass of
  `auto_deriv_kernel()` (`src/integrals/auto_deriv_block_eri.h`).
- **Explicit engines and screened kernels: AO pairs.** The same `q ≥ p`
  triangle over AO pairs with per-component `compute_eri_deriv_dispatch`
  (HGP for `hgp`, OS otherwise). This keeps the `--engines os,auto`
  comparators independent of the kernels they check.
- **One 1e block for every reference**, `one_electron_gradient`. It returns
  the kinetic, basis-function nuclear, nucleus-position nuclear and overlap
  parts separately, and only the symmetric parts of D and W enter.
- **MP2 orbital side:** the UHF Z-vector is matrix-free GMRES, and the
  T2 → AO back-transforms are the shared helpers above.
- **Threads.** Every loop deals its work units into 64 fixed bins (unit
  mod 64), each with its own `natoms × 3` partial summed in bin order, so the
  result is identical at any thread count.

## Gates

- Six `*_engine_os_vs_auto` gradient comparators
  (`tests/engine_gradient_compare.py --engines os,auto`): RHF STO-3G and
  Cartesian 6-31G\*, UHF, ROHF (C1 CH3), B3LYP, HSE06/cc-pVTZ (block Coulomb,
  AO-loop long range). Two more for RMP2 and UMP2. Re-introducing
  "D without C" fails them end to end.
- The PySCF-gated spherical 6-31G\* RHF/ROHF gradient, geomopt and freq cases
  run Auto, so they cover the block path.
- `planck-eri-derivative-kernels` for the kernels themselves.
- At the end of G5: 72 gradient / geomopt / freq / FD / MP2 / DH regression
  cases and 35 DH/MP2/gradient/derivative ctests pass. Three Eigen
  `conjugate_gradient_*` tests report Not Run, the known Eigen test
  registration issue.
- The dense-Z-vector and old-loop comparisons in G7 were temporary probes: no
  C++ test harness runs an SCF.

## Found along the way

- **A DIIS defect in the SCF.** G2's new summation order moved an H2
  geometry-optimization step by 1e-16, and `h2_optfreq_smoke` started
  failing. RHF, UHF, RKS and UKS stored the eigenpairs of the
  DIIS-extrapolated Fock at convergence. For H2/STO-3G every DIIS error
  vector is ~1e-16, the DIIS weights are arbitrary, and the extrapolated
  matrix carries other iterations' eigenvalues with the right eigenvectors.
  The gradient's `W = Σ εᵢ Cᵢ Cᵢᵀ` then had the wrong εᵢ: the 1e/Pulay term
  was off by 1.6e-3 Ha/Bohr and the line search failed. It was hidden
  whenever the error came out exactly zero. Every path now re-diagonalizes
  the true Fock once at convergence when DIIS extrapolated, as PySCF does
  (teaching guide §7).
- **A coverage gap in the regression runner.** `run_regressions.py` passed
  only the input path to `.py` cases, and every comparator defaulted
  `--build-dir` to `./build`. Under `--build-dir build-full` the engine
  comparators (gradient, geomopt, SCF energy, MPI smoke) had been running
  whatever stale binary sat in `./build`. The runner now passes `--build-dir`
  and a case's optional `args`.
- **Deliberately left alone.** The two overlap-only MP2 loops: their weight
  includes `vhf_s1occ`, so symmetrizing it is not obviously safe, and they
  are cheap.

## Not done

Tracked in `vault/Status/Open Work.md`, none needed at these sizes:

- **RI three-centre integrals (G6, measure first).** `compute_3c_eri`,
  `compute_2c_eri` and `compute_3c_deriv_elem` (`src/post_hf/ri/ri_eri.cpp`)
  run per component, serial and unscreened. Price their share on an RI-MP2
  gradient and an RI-JK SCF before building. D3 (RI pair factors recomputed
  per iteration, `docs/ERI_FOCK_BLAS_SCOPE.md`) is the larger RI-JK lever.
- **Block-level density screening** in the 2e loop
  (`Q_ab Q_cd · max|Γ_block| < tol`; the Schwarz table itself stays
  unscreened, invariant 8), and **screened (range-separated) kernels** in the
  block derivative kernels: HSE06's long-range term still takes the AO loop.
- **The UMP2 residue.** After G7 the 2 H2O⁺ gradient is ~35 % Z-vector,
  ~16 % ERI tensor, ~7 % each the 2e derivative term and Imat. The
  conventional path already holds the AO ERI tensor, so the Z-vector action
  could contract against it instead of 33 direct Fock builds. The dense
  `nao⁴` tensors (D_α, D_β, the ERI) cap the conventional UMP2 gradient's
  system size; RI-MP2 is the route past that.
- The stored `nb⁴` tensor (`_compute_2e_auto`, used by MP2/CC/CASSCF/TDDFT
  and conventional SCF) still uses the shell-group hoist, not the block
  kernels. Measure its share of a real MP2/CC run first.
- Analytic Hessians.

## Invariants

From `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`: fixed work partition for
determinism (4), price on the production loop (5), restructured code is not
bitwise under g++ (7), screening never reaches the Schwarz table (8). From
the norm-factor gotcha: `Nc` is already in `_coefficients`, and the
derivative raise acts on the bare angular part, so the component norm of the
**unraised** function applies (the RI `normC` bug is the same trap).
