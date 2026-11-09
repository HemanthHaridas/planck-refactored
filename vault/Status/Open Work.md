---
name: Open Work
description: Canonical summary of known gaps, risks, and follow-up work in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, open-work, canonical, roadmap]
---

# Open Work

Last updated: 2026-11-08 (rewrite at `8570139a`).

This is the canonical list of what remains. What is done lives in
`vault/Status/Completion.md`. Each item names its scope or answer doc, which
holds the measurements and the ruled-out routes. Superseded plans are in
[the historical archive](History/Open%20Work%20investigations%20through%202026-11-06.md),
which is not a work queue.

This rewrite removed what landed since the 2026-11-07 one: D3 dispersion
(#183, merged) and the whole PCM geometry scope (smooth cavity, spherical
PCM, analytic gradients, geometry optimization and frequencies;
`docs/PCM_GEOMETRY.md`). Every item below was re-checked on 2026-11-08
against the tree, and the upstream and CI items against GitHub. The core
suite was 132/132 at `1b3ffc9b`.

## 1. Correctness decisions and external dependencies

- **Yukawa / Gaussian screened exchange is evaluated with the erf kernel.**
  `cam_coefficients()` (`src/dft/base/wrapper.h`) folds
  `XC_HYB_YUKAWA_SR` / `XC_HYB_GAUSSIAN_SR` into the short-range coefficient,
  and the SCF builds it with `ERIKernel::ShortRange`. PySCF does the same
  (water/STO-3G `camyb3lyp` agrees to 1.4e-6 Eh), so nothing shows that the
  energy is wrong, but it is not the kernel libxc defines. TDDFT already
  rejects these functionals (`water_tddft_camy_b3lyp_rejected_sto3g`).
  **Decide:** reject them for every workflow, or accept and document the erf
  substitution.

- **The open-shell `hcore` default can land on a symmetry-trapped saddle.**
  Example: water⁺ UHF and UKS land on ²A₁, 0.069 Eh above ²B₁. The hcore
  guess fills 1a₁ 2a₁ 1b₂ 1b₁ 3a₁, and the descent direction breaks C₂ᵥ.
  Both energies match PySCF. `stability_check` + `stability_follow` (HF and
  KS, Cartesian, spherical and ECP) find the minimum. The same trap hit the
  spherical water⁺ PCM gate (λ_min −0.0715 from `hcore`, in PySCF too),
  which now starts from `guess sad`. **Decide:** whether the default guess,
  or a warning, should change. Making SAD the open-shell default was not
  pursued. Answer: `docs/KS_STABILITY.md`.

- **HSE06 energy finite differences are unreliable by construction.** libxc's
  wPBEh `s_scaling_2` jumps by 5.15e-4 at s = 1, and grid points crossing the
  jump between FD geometries carry the whole FD error. This is not a Planck
  defect: the analytic gradient matches PySCF to ≤ 1.2e-10. **Decide**
  whether a smooth variant is ever needed (`docs/WPBEH_S_SCALING_SCOPE.md`).
  The research question of characterizing every seam is separate:
  `docs/WPBEH_SMOOTH_RESEARCH_SCOPE.md`. Found there (2026-11-08): the jump
  is eq. 8 of Henderson et al., JCP 131, 044108 (2009), which the same
  paper replaces with a smooth scaling (eq. 9, libxc's unused
  `s_scaling_3`; Gaussian 09 `HSEH1PBE`). So libxc's HSE06 is the
  superseded version. Eq. 9 moves atomization energies by 0.006–0.03
  kcal/mol; a C² blend over |s − 1| < 1e-2 moves them by ≤ 2e-3. Also:
  libxc 7.1.0 fixed a second wPBEh seam at small s (`EGscut`, issue #627),
  which Planck's 2023 pin lacks.

- **libecpint is pinned to an unmerged upstream PR** (still open on
  2026-11-08). Planck pins
  robashaw/libecpint#81 at `4e8767fc` on `loriab/libecpint`, because stock
  libecpint's derivatives miss FD by 4.6e-5 relative on HI/def2-QZVP. When #81
  merges: repin to the merge commit, then rerun `planck-ecp-gradient-fd` and
  every `*_ecp` regression case. If the fork branch disappears first, the
  commit is still reachable as upstream `refs/pull/81/head`.

- **PySCF's DFT gradient is wrong on ECP atoms.** `grad/rks.py`
  `grids_response_becke` looks up Bragg radii by `atom_charges()` (Z − n_core).
  On AuH·H₂O B3LYP it is 1.1e-5 Ha/Bohr off PySCF's own FD. R1–R3 are done
  (2026-11-07): it hits every ECP element except I, Te and Y (whose Bragg
  radius at Z − n_core happens to equal the one at Z), diatomics included,
  RKS and UKS, all functional families, and the default and Stratmann
  partitions. It falls with grid level (4.3e-5 at level 2, 6.3e-7 at
  level 7). R4 is done too. The patch and test are in
  `tests/pyscf/upstream/`; all six tests fail on master and pass with the
  patch. R4 also found that ghost atoms trigger the same mismatch on
  PySCF ≥ 2.14 (#3302 updated `radi.py` but not the gradient): 1.4e-5
  Ha/Bohr on a counterpoise water dimer. R5 filed
  2026-11-07: issue pyscf/pyscf#3597, PR pyscf/pyscf#3598 (both open,
  review required, on 2026-11-08). Remaining: follow the review. Once a
  release carries the fix, R6: drop
  `patch_pyscf_grid_response_radii` and pin that PySCF version
  (`docs/PYSCF_ECP_GRID_RESPONSE_SCOPE.md`).
  Planck's reference script patches it at runtime meanwhile.

## 2. Capability gaps

### DFT, in the recommended order

- **Meta-GGA** (`docs/META_GGA_SCOPE.md`). Rejected in `xc_grid.cpp`. Step 1
  is τ and V^τ in the SCF (AO gradients only). Steps 2–4 are gradients (AO
  Hessian, no CPHF), the response kernel, and VV10. Step 1 must make every
  non-SCF workflow refuse meta-GGA by name: the KS gradient picks its terms by
  `vsigma.cols()` and would otherwise silently drop the τ term.
- **PCM leftovers** (`docs/PCM_GEOMETRY.md`, now an answer; see
  Completion). Imaginary-mode following under PCM is rejected (the loops
  would carry the cavity, but no case gates it). UHF/UKS and spherical
  geomopt/frequencies, and planck-dft frequencies, run through the gated
  code paths but have no case of their own. Deferred from step 3:
  tolerance-linked pair screening, and a cheaper φ_el contraction.
- **Double hybrids** (`docs/DH_IMPLEMENTATION.md` §9):
  - UKS DH geometry optimization and frequencies are rejected. The gradient
    they would call already passes FD.
  - The UKS DH path is dense: full α/β MO ERIs, a dense joint Jacobian, and an
    `N_AO⁴` pair density. It is qualified only on small molecules.
  - No committed regression case runs the production UKS DH gradient on the
    V4 grid. The evidence lives in the `tests/g4_3_*` runners, so routing or
    grid breakage would pass the suite.
  - O8 response throughput: paired J/K, blocked XC action, then batched
    Krylov only if a benchmark shows it pays.
  - O9 RI-DH is a separate energy backend. It is the only route to dropping
    the conventional AO ERI tensor.
  - Not covered: range-separated DH gradients, solvent, SCS/DOS spin scaling,
    meta-GGA, frozen-core or RI PT2 in the gradient, DH TDDFT, and spherical
    DH gradients (refused by `water_rks_b2plyp_gradient_spherical_rejected`).
- **The default grid is not qualified for differentiated-grid properties.**
  `legacy` failed the UKS DH rotational-covariance check (2.2e-5 Ha/Bohr
  against a 2e-6 limit), where `g4-3-v4` passes (1.1e-7). Migrating the
  default is a separate, open decision (`docs/DFT_GRID_CONVERGENCE.md`).
  Coarse grids also show orientation sensitivity under symmetry
  reorientation, which is why the symmetry-on FD gradient gate uses
  `grid ultrafine`.
- **Not implemented:** an analytic Hessian (frequencies are semi-numerical),
  `ImaginaryFollow` in `planck-dft`, and the DFT KS J/K incremental build.
- **Large water chains fail to converge.** 32-water RKS hits `max_cycles` at
  serial (4.8 GB; it ran and iterated). This is an SCF-robustness item, not
  a scaling defect.
- **D3 follow-ups** (`docs/DISPERSION_D3.md`, Not done). D3(BJ), D3(0) and
  ATM landed for `planck-dft` (#183). None of these is wanted yet:
  - HF-D3: the same two call sites in the HF SCF and gradient; `hf`
    parameters exist.
  - Modified damping (`d3mbj`, `d3mzero`, `d3op`): one branch each in
    `DFT::Dispersion::compute`.
  - D4, which needs atomic partial charges.
  - A separate D3 field in the results JSON. `electronic_energy` there is
    total minus nuclear repulsion, so it currently includes D3.
  - Regenerating the references needs a hand-built shared simple-dftd3
    (`DFTD3_LIBRARY`): pyscf-dispersion 1.0.0 does not load on macOS arm64.
    Replace the workaround when a fixed wheel ships.

### HF and post-HF

- **ROHF:** RMP2/UMP2, coupled cluster, stability, PCM and SOSCF reject ROHF.
  SOSCF and stability need new orbital-response machinery for the effective
  Roothaan Fock, which does not exist in the tree. ROHF CASSCF and RASSCF
  support only a closed inactive core, and the parity guard rejects a
  spin-polarized one. ROHF is also outside full-symmetry direct SCF.
- **Spherical MP2/UMP2 gradients** (and the geomopt and frequency workflows
  that need them) wait on a response-machinery audit before the lift adapter
  can go through `compute_rmp2_gradient` / `compute_ump2_gradient`. Refused
  by `water_rmp2_spherical_{gradient,geomopt}_rejected`.
- **BSSE:** DFT ghost atoms and counterpoise (D3 already leaves ghost
  atoms out, which is the counterpoise-consistent choice), N-body counterpoise beyond two
  fragments, counterpoise-corrected gradients and geomopt, and post-HF
  ghost-reference verification beyond the SCF-level gates.
- **Symmetry:** conventional-path symmetry-unique ERI storage is out of scope
  (full-group reduction is a direct-SCF feature).
- **`TensorOptimized` RCCSDT** is still treated in-tree as experimental.

## 3. Build, test and CI

- **The libxc pin sits on an API upstream withdrew** (assessed 2026-11-08
  against 7.1.2). Planck pins `4bd0e1e3` (5.0.0-799, Feb 2023), a devel
  commit from between 6.0.0 and 6.1.0. Its multi-term hybrid API
  (`hyb_type`/`hyb_coeff`/`hyb_omega`, `xc_hyb_type`, the
  `XC_HYB_FOCK`/`PT2`/`ERF_SR`/`YUKAWA_SR`/`GAUSSIAN_SR` terms) and every
  double-hybrid definition (B2PLYP etc.) were removed by libxc `8add6f17d`
  (2024-02-28, "Undo breaking API changes ... rethink the whole interface
  ... for double hybrids"). No release has them, 7.1.2 included. Double
  hybrids return only in libxc 8.0.0, with a redesigned C++ API. A bump
  therefore needs:
  - **Hybrid API port.** Port `src/dft/base/wrapper.h` to
    `xc_hyb_cam_coef` / `xc_hyb_exx_coef` and the `XC_FLAGS_HYB_CAM(Y)`
    flags. Gaussian-attenuated SR has no release equivalent.
  - **Planck-side double hybrids.** Planck would define the DH functionals
    itself (components plus PT2 coefficient), as PySCF does, and re-gate
    the 39 B2PLYP inputs.
  - **Build arguments.** Pass `-DMAXORDER=3` explicitly: `-DDISABLE_KXC=OFF`
    alone leaves order 2, compiles out kxc, and the DH gradient hard-exits.
    Drop `ENABLE_XHOST`, which was removed upstream along with
    `-march=native`, ending the SIGILL hazard. CMake ≥ 3.21 is required.

  **Values, libxc 7.0.0 → 7.1.2** (PySCF 2.13 vs 2.15, which reproduces
  the pin to 1e-10):
  - PBE, B3LYP, PBE0, CAM-B3LYP, ωB97X, CAMY-B3LYP, SVWN and BLYP move only
    at roundoff (exc ≤ 8e-11 relative, SCF ≤ 4e-13 Eh).
  - HSE06 moves 1.5–2.7e-10 Eh, from the `EGscut` fix (libxc #627). At
    s < 0.08 the pin's v_σ is off by up to 4e-6 and f_σσ by up to 7e-4
    against an exact-hole reference; 7.1.2 agrees with it to about 1e-9.

  **Version choice.** Use 7.1.2, which PySCF ≥ 2.14 ships. Avoid 7.1.0: a
  10–43× HSE/wPBEh slowdown, and its deprecated `DISABLE_*` switches capped
  the order at 2.

- **Every executable recompiles the shared sources**
  (`docs/BUILD_SHARED_OBJECTS_SCOPE.md`). The build has no library of Planck
  sources, so 82 distinct `src/` files cost 860 compile jobs (90 %
  duplicates) on a header change, and hand-written per-test source lists
  break silently (three on 2026-11-07). Plan: one static `planck_core` plus
  a small `planck_dft`. `USE_OPENMP`, `USING_Libxc`, `USE_MPI` and the
  `PLANCK_CC_*` options currently vary per target and decide what can be
  shared. `planck-mpi` also omits two `PLANCK_CC_*` options the other
  binaries define; check whether that was intended.
- **No build-and-test CI runs.** The "CMake on multiple platforms"
  workflow (`.github/workflows/cmake-multi-platform.yml`) is
  `disabled_manually`; its last run was 2026-04-26. It did not run on #181
  or #183, so the workflow fixes in #181 (build every target, `BASIS_PATH`
  for every test) have never executed on CI. **Decide:** whether to
  re-enable it. A first run is also the first Linux build of simple-dftd3
  with gfortran, and it needs the `PLANCK_LICENSE` secret for the Boys core.
  Separately, "Callgraph Profiling" failed on the 2026-11-07 `devel` push
  because the runner ran out of disk, not because of the code.

## 4. Verification gaps

- **i-shell (L = 6) spherical path has no cross-code check** (FU2,
  `docs/SPHERICAL_F_SHELL_ACCURACY_SCOPE.md`). It bypasses
  `normalized_pseudoinverse` for the recurrence oracle. The input
  `tests/inputs/regression/spherical/ne_rhf_spherical_ccpv6z_ishell.hfinp`
  and its reference (PySCF −128.5470611007 Eh) are committed but
  unregistered, because they are too heavy for the suite.
- **Full-symmetry edge cases:** D3h, Oh, linear groups and lone atoms need
  durable regression cases. The spherical full-symmetry direct-SCF ladder
  should grow beyond NH3/CH4. Keep in-code comments on the spherical symmetry
  representation in line with the code; they have drifted once.
- **CASSCF optimizer changes must rerun the PySCF gates and both SA-2 SAD
  policies.** Both `water_casscf_sa2_sto3g_sad_guess_uphill` (asserts the
  plateau exit, `true`) and `water_casscf_sa2_sto3g_sad_guess` (`false`) are
  tagged `extended`. The uphill case is registered in `tests/regression_cases.json`
  and uses `tests/benchmarks/casscf/pyscf_reference/water_cas44_sto3g_sa2_sad_uphill.hfinp`.
  Run them with `--suite all --case <id>`, because the
  default `core` filter skips them even when `--case` is given. Preserve the
  plateau exit (`sa_g ≤ max(1e-6, tol_mcscf_grad)`): at an SA stationary
  point individual root gradients need not vanish.
- **Validation records to reconcile, not redo blindly:** the Auto-table refit
  (`45ef3cef`) against `planck-compute-2e`, `engine_scf_energy_compare.py`
  (including He₂/cc-pV5Z) and the suites; and the Boys v1.2.1 re-pin
  checklist. Check later runs first, and rerun only the coverage that is
  missing.

## 5. Performance

None of these blocks correctness. Take each up only with a current profile.

- **SCF and ERI kernels** (`docs/ERI_KERNEL_VS_LIBCINT.md`,
  `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, `docs/SCF_VS_PYSCF.md`):
  - Rerun the Planck-vs-PySCF Fock table and the end-to-end
    `pyscf_bench.py` table. Both predate Rys2D and the direct-vs-direct
    harness fix.
  - Route the conventional Auto build's 6D hoist
    (`_contracted_eri_block_hoisted_views`, 50–210× libcint) through Rys2D,
    and delete the hoist.
  - libcint is still 2.3–2.6× faster on the Rys2D blocks. Four layouts have
    already lost (doc §4).
  - G0 general contraction: build it only if a closed-form-class probe gains
    ≥ 1.10× on the whole build with Pople bases unchanged (doc §5).
  - R1b per-AO readout, C5 screened block kernels, and F3 block-fed persisted
    full-group skeleton.
  - Check the `-RIFIT` files for redundant primitives.
- **Shell-pair granularity** (`docs/SHELLPAIR_GRANULARITY_HANDOFF.md`):
  - B-4: hoist explicit `engine rys`. Gate at 1e-13 plus the SA-2 uphill case.
  - Phase C: shell-quartet symmetry skeletons (`{os,hgp,rys}_symm`).
  - Screened and explicit-engine derivative block fallbacks.
  - Phase E: shell-level `ShellPair`. Last, because it changes the shared
    contract.
- **RI-JK and BLAS** (`docs/ERI_FOCK_BLAS_SCOPE.md`). Nothing has landed.
  - B0: cache the density-independent `build_ri_pair_factors`, which is ~61 %
    of the RI-JK Fock build. Add the first RI-JK regression case with it.
  - B1: optional `USE_BLAS`.
  - B2: RI-K from `C_occ`.
  - B3 (`dsyevd`) is deferred.
- **Gradients** (`docs/GRADIENT_INTEGRAL_PERFORMANCE.md`):
  - G6: profile RI 2c/3c derivatives.
  - Block-level density screening and screened derivative block kernels.
  - Stored-ERI UMP2 response, only if its share pays.
  - Conventional tensor construction on real post-HF workloads.
- **Post-HF threading and MPI** (`docs/HPC_MPI.md` Gap 5). The hand-written
  `tensor_backend.cpp` has no OpenMP. Profile MP2 and CC at a realistic size
  first. The dense post-HF MPI stripe (outer MO index) is untouched. RI is a
  memory strategy, not a prerequisite for it.
- **CASSCF** (`docs/CASSCF_PERFORMANCE.md`). Stopped at 1.3× behind PySCF at
  water cc-pVTZ C2v, which was accepted. Leftovers, only with a new target:
  re-profile cc-pVTZ C2v, a better relaxed-solve preconditioner, and the RI
  path, which has none of the speed-ups.
- **FCI F2:** `occupied_orbitals` still returns a vector
  (`src/post_hf/ci/ci.cpp`, ~0.1–0.2 %). Validate any sigma change on an
  iterative case (`o2_fci_rohf_sto3g`, `be_fci_spherical_631gd`).
- **FCIQMC** (`docs/FCIQMC_SCALING_INVESTIGATION.md`), in order:
  - R1 is complete on the M3: HF/6-31G at 50k was fastest at eight threads
    (151.51 s; 2.80× over one); small spherical Be was fastest at two.
    The 10k–100k HF and 50k water controls separate memory and time from setup.
  - R2 passed on the M3. Skipping serial diagonal lookups for surviving
    parents cut full water wall time by 25.5% across matched pairs (median),
    HF by 17.4% and small Be by 5.7%. All trajectories matched.
  - The small-tree merge passed. Running the last three levels sequentially
    for up to 4,096 tagged entries cut eight-thread Be wall time by 28.9%.
    Four threads gave the fastest Be median (8.08 s). Use eight for HF 50k
    on this M3. E4 stays test-only.
  - R3 admits Cr₂ CAS(12,18) at 1.68 Å with 18 frozen RHF orbitals.
    All 15 runs in the population and spawn-attempt sweeps fit the M3 budget.
    Raising the population to 50k widened the estimator gap at one draw per
    parent. Four draws at 10k walkers retained the reference across three
    seeds. Next, repeat that setting with 10k equilibration steps and compare
    each seed against its 2k baseline. The energy remains unconverged.
- **ccgen generated kernels** (`docs/CCGEN_CC_OPENMP.md`,
  `docs/CCGEN_KERNEL_PERFORMANCE.md`):
  - Re-profile the generated production configuration, separating
    per-iteration cost from iteration count.
  - Measure rank 4 on its own, including the registry `-O1` pin. Rank 3 is
    not a proxy.
  - Extend occupied-size and working-set coverage if cache effects matter.
  - Confirm the actual route for new fixtures: the determinant backstop
    belongs to `tensor_backend.cpp` only.

## 6. ccgen

- UCC: `wick`-engine coverage and a durable amplitude-antisymmetry gate.
- O6: spin-tag-aware operator reuse and dressing, with numeric UCC
  validation.
- Hand-written tensor-solver DIIS fragility, and a review of the small Be
  rank-4 reference offset before changing its pin.
- Factorizer: decide whether a canonical tie-break should make factorization
  order-invariant (`docs/CCGEN_RED_TESTS.md`).
- **Parallel generation is not equivalent to serial.**
  `generate_cc_equations(parallel_workers > 1)` gives different terms. There
  are two causes: `_wickaccel` is not spawn-safe, and the pre-canonical exact
  merge is partition-local. Serial (the default) is deterministic and correct.
  `test_parallel_generation_matches_serial` is `expectedFailure`. Fixing it
  means making the extension spawn-safe and the raw merge global.
- The dressed spin-adapted route is retired, and its seven tests are
  `expectedFailure`. An unexpected pass means the composition was fixed
  (`docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`).

## 7. CASSCF

- **Unreproduced OpenMP hang** (once, 2026-10-28, `water_cas44_631g`, 4
  threads, g++-15 libgomp, macOS): all threads parked at the barrier of the
  `parallel for` in `compute_orbital_gradient`. 60 reruns did not hang.
  Suspect libgomp first. If it recurs, capture `sample <pid>`.
- Transcript labels should uniquely name the selected candidate (cosmetic).
  Do not delete the SA-2 single-pair escape probes as cleanup.

## 8. Maintenance

- Extract a shared `SpatialQuartetLayout` for the OS, HGP and Rys
  per-quartet scratch. Only the spatial core is shared. Bitwise-gate it
  across all three engines.
- Deduplicate the full-group AO-transform code in `group_operations.cpp` and
  `mo_symmetry.cpp`.
- Audit any remaining reversed-shell-pair reconstruction in the
  explicit-engine and screened gradient fallbacks.
- A geometry-derived working-state object with one invalidation point:
  justify it only by a concrete invalidation defect.
