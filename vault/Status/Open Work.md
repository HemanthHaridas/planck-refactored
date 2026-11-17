---
name: Open Work
description: Canonical summary of known gaps, risks, and follow-up work in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, open-work, canonical, roadmap]
---

# Open Work

Last updated: 2026-11-16 (rewrite: RI-JK answered and removed from the
priority list; libecpint repinned to upstream, #82 filed; PySCF #3598 merged
upstream).

This is the canonical list of what remains. What is done lives in
`vault/Status/Completion.md`. Each item names its scope or answer doc, which
holds the measurements and the ruled-out routes. Superseded plans are in
[the historical archive](History/Open%20Work%20investigations%20through%202026-11-06.md),
which is not a work queue.

This rewrite removed what landed since 2026-11-08: the RI-JK routine-use
scope (`docs/RI_JK_ROUTINE.md`, now an answer), the RI-MP2 gradient response
fix and spherical RI-MP2 gradients (`docs/RI_MP2_GRADIENT_RESPONSE.md`), the
matrix-free HF orbital Hessian (`docs/SCF_ORBITAL_HESSIAN_OPERATOR.md`), G6
and the factored RI-MP2 pair density (`docs/GRADIENT_INTEGRAL_PERFORMANCE.md`).
The upstream items were re-checked against GitHub on 2026-11-16.

## Priority for a final product (2026-11-10)

A ranking of the gaps below, for a general-purpose program that users would
pick for routine work: single points, geometries, frequencies and interaction
energies with standard spherical basis sets. The sections that follow keep the
detail; this is only the order. A narrower target (a double-hybrid showcase, a
teaching code) reorders it.

**Necessary.** Without these the product is broken or misleading.

1. **CCSD(T).** Not implemented: CCSD, CCSDT and higher exist, perturbative
   triples do not. It is the benchmark method users ask for first. Scope: `docs/CCSD_T_SCOPE.md`.
2. **SCF robustness on medium systems.** 32-water RKS hits `max_cycles`
   (§2, DFT). Scope: `docs/SCF_CONVERGENCE_MEDIUM_SYSTEMS_SCOPE.md`.
3. **Meta-GGA**, at least SCF energies with a by-name refusal everywhere else,
   then gradients (§2, DFT). r2SCAN, ωB97M-V, M06-2X and TPSSh are rejected
   today. Scope: `docs/META_GGA_SCOPE.md`.
4. **Default DFT grid qualified for gradients** (§2, DFT). `legacy` fails the
   rotational-covariance check, so default-settings geometries and
   frequencies carry grid noise. A decision and a migration, not new code. Scope: `docs/DFT_DEFAULT_GRID_SCOPE.md`.
5. **CI running again** (§3). Blocked on the shared-objects build change. Scope: `docs/CI_SCOPE.md`.
RI-JK for routine HF and DFT is no longer on this list: it is answered
(`docs/RI_JK_ROUTINE.md`).

**Expected, with a workaround.**

6. ROHF-MP2 and ROHF coupled cluster; UHF-based versions exist. Matters more
    once CCSD(T) exists. Scope: `docs/ROHF_CORRELATION_SCOPE.md`.
7. Analytic Hessian; semi-numerical frequencies work, this is cost on large
    systems. Scope: `docs/ANALYTIC_HESSIAN_SCOPE.md`.
8. Spherical double-hybrid gradients, frozen-core PT2 in the gradient, RI-DH.
    Necessary if double hybrids are a headline feature; spherical first. Scope: `docs/DH_GRADIENT_EXTENSIONS_SCOPE.md`.
9. D4 dispersion; D3(BJ) covers most uses. Scope: `docs/DISPERSION_D4_SCOPE.md`.
10. Range-separated functionals with RI-JK (attenuated erf 2c/3c integrals);
    refused by name today, and the exact path works. §2, DFT.
11. RI under MPI, steps M0–M1: today every `planck-mpi` rank runs the
    full RI calculation (correct, no speedup, no warning, no MPI gate).
    M0 adds the gate and a warning; M1 splits the 3c build, RI-J/K, RI-MP2
    and RI gradients over aux shells with one allreduce each. Ranked here
    because one node with RI already removes the nb⁴ wall. Scope:
    `docs/RI_MPI_SCOPE.md`.

**Nice to have.** Imaginary-mode following in `planck-dft` and under PCM;
HF-D3 and modified D3 damping; counterpoise-corrected gradients and N-body
counterpoise; ROHF stability, SOSCF and PCM; a spin-polarized inactive core in
ROHF CASSCF; range-separated, solvent, SCS/DOS, meta-GGA and TDDFT
double-hybrid variants; the incremental KS J/K build and PCM performance
leftovers; `TensorOptimized` RCCSDT; conventional symmetry-unique ERI storage;
further post-HF ghost-atom checks; RI under MPI step M2 (distribute the
3-center tensor; matters past nb ≈ 1000–1500).

## 1. Correctness decisions and external dependencies

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

- **libecpint thread safety upstream** (robashaw/libecpint#82, filed
  2026-11-16). Since #78 a shared `ECPIntegral` races; Planck builds one per
  thread, so nothing here depends on the outcome. Follow the issue; if
  upstream restores re-entrant `const` methods, no change is needed.

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
  Ha/Bohr on a counterpoise water dimer. R5: issue pyscf/pyscf#3597 and PR
  pyscf/pyscf#3598, **merged 2026-11-15** (`6eb9abd1`). The latest release,
  v2.15.0 (2026-11-05), predates it. Remaining, R6, once a release carries
  the fix: drop `patch_pyscf_grid_response_radii` and pin that PySCF version
  (`docs/PYSCF_ECP_GRID_RESPONSE_SCOPE.md`). Planck's reference script
  patches it at runtime meanwhile.

## 2. Capability gaps

### DFT, in the recommended order

- **Meta-GGA** (`docs/META_GGA_SCOPE.md`). Rejected in `xc_grid.cpp`. Step 1
  is τ and V^τ in the SCF (AO gradients only). Steps 2–4 are gradients (AO
  Hessian, no CPHF), the response kernel, and VV10. Step 1 must make every
  non-SCF workflow refuse meta-GGA by name: the KS gradient picks its terms by
  `vsigma.cols()` and would otherwise silently drop the τ term.
- **Rydberg states** (`docs/RYDBERG_ORBITALS_SCOPE.md`). Diffuse-basis
  TDDFT runs, but no regression case uses a diffuse basis. On water
  aug-cc-pVDZ CAM-B3LYP, the `legacy` grid's angular pruning misses PySCF by
  7.2e-7 Eh and 0.7–2.2 meV per root; `g4-3-v4` is within 4.4e-9 Eh. The wall
  is dense TDDFT: two nb⁴ tensors for range-separated functionals, so benzene
  aug-cc-pVDZ needs 28 GB. Matrix-free TDDFT can reuse the KS stability (A+B)
  operators and their Davidson solver. Canonical orthogonalization is not
  needed yet (benzene aug-cc-pVTZ λ_min 7.7e-8, against a 1e-8 wall).
- **Range-separated functionals with RI-JK.** `planck-dft` refuses them by
  name: the RI 2c/3c integrals are Coulomb-kernel only. Needs attenuated
  (erf) 2c/3c integrals and their derivatives (`docs/RI_JK_ROUTINE.md`, Not
  in this scope).
- **PCM leftovers** (`docs/PCM_GEOMETRY.md`, now an answer; see
  Completion). Imaginary-mode following under PCM is rejected (the loops
  would carry the cavity, but no case gates it). Deferred from step 3:
  tolerance-linked pair screening, and a cheaper φ_el contraction.
- **Double hybrids** (`docs/DH_IMPLEMENTATION.md` §9):
  - The UKS DH gradient is qualified only on small molecules (STO-3G and
    6-31G fixtures; `docs/UDH_GRADIENT_ONE_ARCHITECTURE.md`).
  - O8 response throughput: paired J/K, blocked XC action, then batched
    Krylov only if a benchmark shows it pays.
  - O9 RI-DH is a separate energy backend. It is the only route to dropping
    the conventional AO ERI tensor. It also takes the UKS amplitude tile source
    (resident `t2_aa/ab/bb` today; design in
    `docs/UDH_GRADIENT_ONE_ARCHITECTURE.md`, "Deferred to RI-DH"), whose saving only shows once the AO tensor is gone.
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
  - Regenerating the references needs a hand-built shared simple-dftd3
    (`DFTD3_LIBRARY`): pyscf-dispersion 1.0.0 does not load on macOS arm64.
    Replace the workaround when a fixed wheel ships.

### HF and post-HF

- **CCSD(T) is not implemented.** The CC stack has CCSD, CCSDT, CCSDTQ and
  arbitrary order, but no perturbative triples correction.
- **ROHF:** RMP2/UMP2, coupled cluster, stability, PCM and SOSCF reject ROHF.
  SOSCF and stability need new orbital-response machinery for the effective
  Roothaan Fock, which does not exist in the tree. ROHF CASSCF and RASSCF
  support only a closed inactive core, and the parity guard rejects a
  spin-polarized one. ROHF is also outside full-symmetry direct SCF.
- **PySCF < 2.15 analytic MP2 gradients miss their own energy FD** by up to
  4.2e-7 Ha/Bohr on C1 fixtures. It is fixed in PySCF 2.15.0 by its Krylov /
  CPHF early-termination fix (pyscf/pyscf#3334); 2.14.0 still has it.
  Planck agrees with the FD. `tests/pyscf/.venv` is 2.15.0 since
  2026-11-11 (CH3 gap 6.5e-9), so PySCF's analytic MP2 gradient is usable
  again; the committed MP2 gates keep their energy-FD references.
- **Frozen-core MP2 gradients** (RMP2 and UMP2) are refused by name. The
  builders lack the frozen–active occupied and virtual relaxation blocks and
  see only the active orbitals; the scope doc says why and what to add
  (PySCF `grad/mp2.py` L124–132). UMP2 had no refusal and printed a gradient
  4.5e-2 Ha/Bohr off until 2026-11-11.
- **BSSE:** N-body counterpoise beyond two fragments,
  counterpoise-corrected gradients and geomopt (HF and DFT), counterpoise
  under PCM (refused), and post-HF ghost-reference verification beyond the
  SCF-level gates. HF and KS counterpoise single points are done
  (`docs/DFT_COUNTERPOISE.md`).
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
- **No build-and-test CI runs, because the runners ran out of disk.** The
  "CMake on multiple platforms" workflow
  (`.github/workflows/cmake-multi-platform.yml`) is `disabled_manually`; its
  last run was 2026-04-26. Its runs failed for lack of space on GitHub, and
  since #181 it builds every target in two `PLANCK_CC_MAXORDER` jobs, so
  re-enabling it as it stands would fail the same way. The workflow fixes in
  #181 (build every target, `BASIS_PATH` for every test) have therefore never
  executed on CI. **Blocked on** cutting the build's disk use, which the
  shared-objects item above does by removing the duplicate objects. A first
  run is also the first Linux build of simple-dftd3 with gfortran. The
  `PLANCK_LICENSE` secret it needs is set. "Callgraph Profiling" hit the same
  limit on the 2026-11-07 `devel` push and is disabled too.

## 4. Verification gaps

- **Full-symmetry edge cases:** D3h, Oh, linear groups and lone atoms are
  gated at the energy level (`*_fullsym_direct`). D∞h, C∞v and Kh do not use
  the full-group ERI reduction. The spherical full-symmetry direct-SCF ladder
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
- **RI-JK and BLAS** (`docs/ERI_FOCK_BLAS_SCOPE.md`).
  - B0 landed for the pair factors (cached per geometry, 2026-11-16); the
    unpacked per-aux `B_Q` matrices are still rebuilt on every Fock build
    (~8 % of the RI-JK Fock build when measured).
  - B1: optional `USE_BLAS` (RI-K GEMMs 4.3–7.8×).
  - B3 (`dsyevd`) is deferred.
- **Gradients** (`docs/GRADIENT_INTEGRAL_PERFORMANCE.md`):
  - RI shell-block 3c/2c kernels (G6 threaded the per-component loops;
    benzene RI-JK gradient 27.4 → 7.9 s). Only if RI gradients grow.
  - Block-level density screening and screened derivative block kernels.
  - Stored-ERI UMP2 response, only if its share pays.
  - Conventional tensor construction on real post-HF workloads.
- **Post-HF threading and MPI** (`docs/HPC_MPI.md` Gap 5). The hand-written
  `tensor_backend.cpp` has no OpenMP. Profile MP2 and CC at a realistic size
  first. The dense post-HF MPI stripe (outer MO index) is untouched. RI is a
  memory strategy, not a prerequisite for it. RI itself is not distributed
  either: `docs/RI_MPI_SCOPE.md` (priority #11 and Nice to have).
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
    The three 25k projections at τ = 5–6 exceed the variational upper
    bound of −2064.896580 Eh by 0.108–1.092 Eh. Extending the seed that
    retained its reference to τ = 10 increased its minimum error to
    0.263 Eh, with 5,483 usable projections in 10,000 steps. Compare a fixed
    multideterminant projection with the single-reference estimator on
    identical trajectories. Two SC-NEVPT2 estimates differ by 19 mEh.
  - Low-weight parents account for 93–96% of visits across three Cr₂
    profiles. Opt-in probabilistic spawning passed the November 16
    current-build correctness and thread-identity checks. Five CAS(12,10)
    pairs are complete: projected RMSE 2.48 → 1.40 mEh, shift RMSE
    5.92 → 6.45 mEh, with 3.14% more mean wall time. Keep it experimental.
    Next: matched CAS(12,18) cost pilot. Default adoption needs tighter
    runtime matching and more exact-control seeds.
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
