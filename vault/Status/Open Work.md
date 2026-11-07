---
name: Open Work
description: Canonical summary of known gaps, risks, and follow-up work in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, open-work, canonical, roadmap]
---

# Open Work

Last updated: 2026-11-07 (rewrite at `631d5c6b`).

This is the canonical list of what remains. What is done lives in
`vault/Status/Completion.md`. Each item names its scope or answer doc, which
holds the measurements and the ruled-out routes. Superseded plans are in
[the historical archive](History/Open%20Work%20investigations%20through%202026-11-06.md),
which is not a work queue.

The 2026-11-07 rewrite removed items that have landed: def2/ECP (all steps),
spherical DFT, KS stability, the installed basis path, and the RCCSD F_me
fix. D3 dispersion landed later the same day (#183) and its follow-ups are
listed below. It also re-checked the items below against the tree. No test suite was
rerun for it.

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
  KS, Cartesian, spherical and ECP) find the minimum. **Decide:** whether the
  default guess, or a warning, should change. Making SAD the open-shell
  default was not pursued. Answer: `docs/KS_STABILITY.md`.

- **HSE06 energy finite differences are unreliable by construction.** libxc's
  wPBEh `s_scaling_2` jumps by 5.15e-4 at s = 1, and grid points crossing the
  jump between FD geometries carry the whole FD error. This is not a Planck
  defect: the analytic gradient matches PySCF to ≤ 1.2e-10. **Decide**
  whether a smooth variant is ever needed (`docs/WPBEH_S_SCALING_SCOPE.md`).
  The research question of characterizing every seam is separate:
  `docs/WPBEH_SMOOTH_RESEARCH_SCOPE.md`.

- **libecpint is pinned to an unmerged upstream PR.** Planck pins
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
  2026-11-07: issue pyscf/pyscf#3597, PR pyscf/pyscf#3598. Remaining: follow
  the review. Once a release carries the fix, R6: drop
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
- **PCM geometry optimizations and frequencies**
  (`docs/PCM_GEOMETRY_SCOPE.md`). Step 1, the smooth SWIG cavity, landed
  2026-11-07 (see Completion). Next is step 2, the analytic gradient. PCM is still
  single-point only, in both binaries and in both bases, and spherical PCM is
  refused.
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
  default is a separate, open decision (`docs/DFT_GRID_CONVERGENCE_SCOPE.md`).
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

- **Every executable recompiles the shared sources**
  (`docs/BUILD_SHARED_OBJECTS_SCOPE.md`). The build has no library of Planck
  sources, so 82 distinct `src/` files cost 860 compile jobs (90 %
  duplicates) on a header change, and hand-written per-test source lists
  break silently (three on 2026-11-07). Plan: one static `planck_core` plus
  a small `planck_dft`. `USE_OPENMP`, `USING_Libxc`, `USE_MPI` and the
  `PLANCK_CC_*` options currently vary per target and decide what can be
  shared. `planck-mpi` also omits two `PLANCK_CC_*` options the other
  binaries define; check whether that was intended.
- **CI's first full-test run is unverified.** The workflow builds every
  target and every test now gets `BASIS_PATH` from the source tree (#181).
  #181 is now in `devel`, so the first CI run on a PR into `devel` (#183,
  D3) is the first full-test run, and also the first that installs gfortran
  and builds simple-dftd3 on Ubuntu. Unknowns: the Boys-core license
  (`PLANCK_LICENSE`) and the Fortran link on Linux.

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
  tagged `extended`. Run them with `--suite all --case <id>`, because the
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
  - R0: qualify the HF-to-CI handoff after the true-Fock, spherical and ECP
    changes, and explain any moved fixed-seed pins.
  - R1: remeasure on an identified build.
  - R2: choose a kernel from current attribution. E2 was rejected. Keep E4
    test-only until a paired retest passes.
  - R3, conditional: a fixed-orbital active-space adapter.
  - A scientific target and a resource budget are still unselected.
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
