---
name: Open Work
description: Canonical summary of known gaps, risks, and follow-up work in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, open-work, canonical, roadmap]
---

# Open Work

Last updated: 2026-11-06 (landed-work audit at `22da4338` plus local changes)

This is the canonical open-work document for the repository.
Use it with `vault/Status/Completion.md`. Older status snapshots and handoff
notes may still exist for design history, but they are no longer the source of
truth for what remains.

This audit removes superseded implementation plans from the active list. Their
original text is preserved in [the historical investigation archive](History/Open%20Work%20investigations%20through%202026-11-06.md),
which is not a work queue. “Landed” means present in commit history; local ECP
and E3/E4 additions are identified separately. No test suite was rerun for this
documentation audit; unresolved validation items remain open.

## Highest-priority correctness and robustness work

- **Yukawa / Gaussian screened-exchange functionals are evaluated with the
  erf kernel everywhere except TDDFT (found 2026-10-16).** `cam_coefficients()`
  (`src/dft/base/wrapper.h`) folds `XC_HYB_YUKAWA_SR` / `XC_HYB_GAUSSIAN_SR`
  into the short-range coefficient and the SCF builds it with
  `ERIKernel::ShortRange` (erf). PySCF does the same: water/STO-3G
  `camyb3lyp` agrees to 1.4e-6 Eh (-75.4581507 vs -75.4581521), so there is no
  independent reference showing the energy is wrong -- but it is not the
  kernel libxc defines. TDDFT now rejects these by name
  (`Functional::has_non_erf_short_range_exchange`, gated by
  `water_tddft_camy_b3lyp_rejected_sto3g`). Decide whether functional
  initialization should reject them for every workflow, or accept and document
  the erf substitution.

- **Open-shell SCF from the default `hcore` guess can converge to an
  excited, symmetry-trapped saddle (investigated 2026-11-04; UKS can now
  detect and follow it, 2026-11-05 — see Completion).** Found as UKS HSE06 water⁺/cc-pVTZ sitting 0.066 Eh above
  PySCF. Not an energy defect: Planck's state is the ²A₁ excited state
  (beta hole in 3a₁), and both states match PySCF to the grid floor (²A₁
  7.2e-6, ²B₁ from `guess sad` 6.5e-6). Plain UHF does the same: from
  `hcore` it returns ²A₁ at −75.5798415962, PySCF's ²A₁ to every digit,
  0.069 Eh above the ²B₁ ground state (−75.6490088941).
  - Cause: the hcore guess fills the lowest 5 alpha / 4 beta core-Hamiltonian
    orbitals, which order 1a₁ 2a₁ 1b₂ 1b₁ 3a₁, so the guess already has the
    ²A₁ hole. ²A₁ is a saddle (PySCF UKS internal stability: lowest
    orbital-Hessian eigenvalue −0.128), but its descent direction mixes a₁
    and b₁, which breaks C₂ᵥ, so a symmetric density never leaves it.
    PySCF's own unconstrained SCF started from the ²A₁ density stays there
    too; its `1e` guess reaching ²B₁ is trajectory luck (its guess has the
    same symmetric 5/4 occupation, and its symmetry breaking only applies
    to spin-0 guesses).
  - `hcore` is the default guess (`SCFGuess::HCore`, `types.h`), for both
    binaries.
  - UHF already has the remedy: `stability_check` + `stability_follow` finds
    λ_min = −0.069 and re-converges to −75.6490088941. UKS and RKS now
    also have stability analysis and following; the remaining issue is the
    default guess and whether users enable the available checks.
  - Remedy chosen: Kohn-Sham stability analysis and follow
    (`docs/KS_STABILITY.md`). UKS, RKS (both channels) and R0 (the HF
    RHF → UHF channel) landed; see Completion. Spherical KS stability
    (`SPHERICAL_DFT.md` step 4) and KS stability with ECPs
    (`DEF2_ECP.md` step 6) landed 2026-11-06, so KS stability is
    complete; the HF-side ECP gates landed
    2026-11-06 (`hi_cation_uhf_def2svp_ecp_stability`,
    `hi_rmp2_def2svp_ecp`, `hi_rccsd_def2svp_ecp`; spherical HF gates in
    S0.6). Only HSE06/cc-pVTZ traps among the
    water⁺ variants tried. The S1 sweep found one unintended saddle among 38
    open-shell regression inputs (`h2o2_cation_uhf_tight_tol_sto3g`, kept: it
    gates DIIS termination). SAD as the open-shell default was not pursued.

- **HSE06 energy finite differences are unreliable by construction (answered
  2026-11-06; one decision open).** Not a Planck defect. libxc's HSE06 rescales
  the reduced gradient with `s_scaling_2` (`maple/scaling.mpl`, "first version
  of the scaling by TM Henderson, apparently used by Gaussian"), which jumps by
  5.15e-4 at s = 1, so the energy density jumps by ~1e-4 relative there. A grid
  point whose s crosses 1 between FD geometries adds its weighted jump to the
  difference. On C1 H2O2/6-31G* (δ = 5e-4 Bohr) 7 of 354000 points cross and
  carry 2.796e-8 Eh, the whole FD error; removing them leaves 4.4e-8 Ha/Bohr
  of ordinary step error (`tests/pyscf/hse06_s_scaling_jump.py`). Planck's
  energy and analytic gradient match PySCF to ≤ 1.2e-10, and PySCF reproduces
  the FD miss to 1e-12. `water_dft_hse06_gradient_symm_ultrafine_fd` passed
  only because none of its points happened to cross; it is now the CAM-B3LYP
  `water_dft_camb3lyp_gradient_symm_ultrafine_fd`. Upstream kept the jump on
  purpose (libxc `173000399`, 2020, "follow the old convention"), and it
  affects every `GGA_X_WPBEH`-based hybrid. Fix scoped in
  `docs/WPBEH_S_SCALING_SCOPE.md`: gate around it now (PySCF comparisons,
  an FD warning), measure the effect on frequencies, geomopt and SCF, then
  decide whether a smooth variant is ever needed. The scientific question
  (characterize every libxc wPBEh seam, measure consequences, construct a
  C² variant that agrees with HSE06) is a separate, publishable research
  scope: `docs/WPBEH_SMOOTH_RESEARCH_SCOPE.md`.

## Routine-use gaps (scoped 2026-10-26)

What blocks Planck from routine DFT use, each with its own scope doc. Order
is the recommended one; none blocks another except where noted.

- **Spherical basis in `planck-dft` — DONE 2026-11-06** (`docs/SPHERICAL_DFT.md`;
  see Completion). Every DFT workflow runs spherical except PCM and
  double-hybrid gradients, which are refused by name. ECP DFT
  (`DEF2_ECP.md` step 6) landed 2026-11-06 as well (see Completion).
- **PySCF's DFT gradient is wrong on ECP atoms; report upstream once the
  research lands** (`docs/PYSCF_ECP_GRID_RESPONSE_SCOPE.md`, found
  2026-11-06). `grad/rks.py` `grids_response_becke` looks Bragg radii up by
  `mol.atom_charges()` (Z − n_core) while the energy's partition uses the
  proton number: AuH·H₂O B3LYP 1.1e-5 Ha/Bohr off PySCF's own energy FD,
  2.4e-10 from Planck once corrected. Still on master and hotfix; no issue or
  PR. Planck's reference script patches it at runtime meanwhile.
- **D3 dispersion** (`docs/DISPERSION_D3_SCOPE.md`). None in `src/`. A
  geometry-only energy and gradient beside nuclear repulsion. Licence
  checked 2026-10-26: simple-dftd3 is LGPL-3.0-or-later. Planck ships source
  only, so fetching it at build time (as libxc is) carries no LGPL
  obligations for the project; a native port would put LGPL files in the
  tree. Recommended: `ExternalProject`, optional because it needs a Fortran
  compiler.
- **def2/ECP: done 2026-11-07** (`docs/DEF2_ECP.md`, now an answer doc; see
  Completion). Only the libecpint repin below remains.
- **libecpint is pinned to an UNMERGED upstream PR — recheck when it merges
  (decided 2026-11-04).** Stock libecpint (upstream HEAD `95203c68`; no
  release since v1.0.7) has ECP derivatives that miss finite differences by
  4.6e-5 relative on HI/def2-QZVP and 2.6e-7 on I2/def2-TZVP.
  robashaw/libecpint#81 fixes seven type-2/quadrature defects (including the
  6-digit kernel constants Planck used to patch; `cmake/patch_libecpint.cmake`
  is deleted) and passes all five FD cases at ≤4e-9. `LIBECPINT_GIT_TAG` is
  `4e8767fc` on `loriab/libecpint`. **When #81 merges: repin to the upstream
  merge commit (review may change it), then rerun `planck-ecp-gradient-fd` and
  every `*_ecp` regression case.** If the fork branch disappears first, the
  commit is still reachable as upstream `refs/pull/81/head`.
- **Gradient performance follow-ups** (`docs/GRADIENT_INTEGRAL_PERFORMANCE.md`).
  G1–G5 and G7 O1/O2 have landed; the historical 20–120× gap is closed.
  Remaining measured-work candidates are listed under Performance below.
- **Meta-GGA** (`docs/META_GGA_SCOPE.md`). Rejected in `xc_grid.cpp`. τ and
  V^τ need only the AO gradients the GGA path has; nuclear gradients need the
  AO Hessian, which exists. TDDFT/SOSCF kernels and VV10 are separate steps.
- **PCM geometry optimizations and frequencies**
  (`docs/PCM_GEOMETRY_SCOPE.md`). The cavity hard-prunes buried points, so
  the energy jumps with geometry; a smooth (switching-function, Gaussian
  charge) cavity matching PySCF's comes first, then the analytic gradient.

## Build and packaging

- **`ctest` runs Eigen's own test suite and reports 14 failures (found
  2026-10-25).** `fetchcontent_makeavailable(eigen)` registers Eigen's tests
  (`resize`, the `sblat*`/`dblat*`/`cblat*`/`zblat*` BLAS tests) in Planck's
  CTest tree, but nothing builds them, so every one fails in ~0 s and buries
  real failures in the summary. Fix: keep Eigen's testing off at fetch
  (e.g. `set(EIGEN_BUILD_TESTING OFF)` / `BUILD_TESTING` scoped around the
  `fetchcontent_makeavailable` call, depending on what Eigen 3.4.0 honors).

- **Result-changing environment overrides still reach production binaries
  (scoped 2026-10-12, `docs/RUNTIME_ENV_OVERRIDES_SCOPE.md`).** Six unused ones
  were deleted, then E1 (`PLANCK_DFT_ALLOW_RS_WORKFLOWS`) and E4 (the
  unreachable generated-kernel branch in `tensor_backend.cpp`, carrying
  `PLANCK_CC_T3_TIME`) landed 2026-10-12. Remaining: E2 turn `PLANCK_RCCSDT_BACKEND` into an input keyword; E3
  `PLANCK_DFT_DH_HESSIAN_PROBE_LOG`: delete the DH swap probe (recommended)
  or gate it at compile time. It replaces the returned DH gradient. It exists
  in the current `src/dft/driver.cpp`; remove or compile-gate it here rather
  than treating it only as a future branch-merge blocker. The two generator steps E4 orphaned (the
  spin-orbital warm-start `.inc` and the plain rank-2/3 RCC TUs) were removed
  the same day.

- **Installed basis path: resolved**, not an open build defect.
  `CMakeLists.txt` now uses `${CMAKE_INSTALL_PREFIX}/share/basis-sets`,
  matching the install destination (`0106e0a3`). An uninstalled build may
  still need `BASIS_PATH`; that is distinct from the repaired prefix mismatch.

- **Release builds compile with `-DNDEBUG`, silently disabling every
  `assert()`-based unit test in the suite (found 2026-09-04, while gating
  X5.0/CC amplitude checkpoint work).** Every hand-rolled test binary in
  `tests/` (`cc_amplitude_checkpoint.cpp`, `cc_spatial_amplitude_projection.cpp`,
  `cc_amplitude_sector_seed.cpp`, and by extension every other `assert()`-based
  gate in the tree) passes vacuously under the default `Release` build type —
  `ctest` reports green regardless of whether the assertions inside actually
  ran. Concretely reproduced: `cc_spatial_amplitude_projection` passed under
  the normal `build-full` (`-DNDEBUG` set) while its underlying fixture had a
  real bug that only surfaced when rebuilt standalone with `-UNDEBUG`. Every
  gate built this session was independently re-verified with a `-UNDEBUG`
  standalone compile plus a mutation test before being trusted — see the CC
  amplitude checkpoint entry in Completion for the pattern. **Not fixed**:
  either force `-UNDEBUG` for these hand-rolled test binaries specifically, or
  replace `assert()` with a check that does not compile out in Release. Left
  as a project-wide gap by explicit user decision (noted, not fixed, when
  found).

- **CI's `cmake-multi-platform.yml` builds only the `hartree-fock` target
  before running `ctest`, never the test binaries themselves (found
  2026-09-04, same session).** `ctest` runs against whatever binaries already
  exist in the build tree; since the workflow's `Build` step passes
  `--target hartree-fock` explicitly, none of the `add_executable(planck-*)`
  test targets are ever built there. What `ctest --build-config Release` does
  in CI as a result is unverified from this finding alone — it may pass
  trivially (no test binaries found) or run stale binaries from a prior step,
  neither of which is the intended coverage. **Not fixed**, same reasoning as
  the NDEBUG gap above: real, project-wide, out of scope for the change that
  surfaced it, left for a deliberate look rather than a byproduct fix.

## SCF convergence follow-ups

RHF/UHF/RKS/UKS SOSCF, hybrid/range-separated KS response, and spherical
SOSCF/stability have landed. See Completion, `docs/SOSCF*.md`, and
`docs/SPHERICAL_DFT.md`. Remaining:

- ROHF SOSCF and orbital response: the effective Roothaan Fock needs its own
  response machinery; this is also the gap behind ROHF stability/PCM.
- Reproduce the large-`nb` `scale.json` iteration-count cliff on the intended
  cluster. Small-system correctness does not establish that scaling result.
- DFT SOSCF with PCM or SAO/symmetry: reaction-field and block-diagonal
  response remain separate work; spherical AO support does not resolve them.

## FCI performance follow-up

F1 allocation removal and F3 deterministic threading have landed; the sigma
build is not serial. See `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` and Completion.
Only F2 remains as low-priority hygiene: `occupied_orbitals` still returns a
vector in `src/post_hf/ci/ci.cpp` (historically 0.1–0.2% of the profile).
Any sigma change must exercise an iterative case such as
`o2_fci_rohf_sto3g` or `be_fci_spherical_631gd`; H2/water STO-3G use the dense
path and cannot validate it.

## FCIQMC — current rescope and deferred capability work

The method, statistical gates, threading, canonical population, parent-keyed
RNG, E1 rank selection, and M1 level-parallel merge are implemented. Canonical
population and E1/M1 landed in `f38f2f0b` and `fc687b3d`. There is no remaining
F1–F5 implementation task, missing RNG prerequisite, or unbuilt S5 gate.
The detailed current scope is `docs/FCIQMC_SCALING_INVESTIGATION.md`.

Remaining, in the November 6 order:

1. **R0:** qualify the current HF-to-CI Hamiltonian handoff after the true-Fock,
   spherical-integral, and ECP changes; resolve reference eligibility and
   explain any moved fixed-seed pins before changing them.
2. **R1:** remeasure setup, propagation, and memory on an identified current
   build. The historical HF/6-31G 50k result was 403.75 s at one thread and
   163.96 s at four; it is not a fresh timing of this checkout.
3. **R2:** choose a kernel candidate from current attribution. E2 dispatch was
   rejected. The local E3 measurements identify bit selection; local E4 has a
   1.130x selector-only win and inconclusive whole-run evidence. Keep E4
   test-only until a paired baseline/candidate retest passes the gates.
4. **R3, conditional:** if problem admission is the blocker, scope a fixed-orbital
   active-space Hamiltonian adapter using the current CASSCF helpers, with
   core-energy, spherical/ECP, and small-CAS equivalence checks.

A concrete scientific target and resource budget still need selection. Replica
throughput, MPI/GPU, stochastic orbital optimization, and new sampling laws
remain separate/deferred work. Do not reopen retired map/PMR/sharded-sort
experiments without a new measured reason. Earlier FCIQMC investigations are
in the historical archive and the FCIQMC answer documents.

## Verification and regression gaps

- **ccgen factorizer reproducibility:** the nine historical Python failures
  were fixed (2026-08-29); they are not an open correctness queue. The remaining
  decision is whether a canonical tie-break should make factorization
  order-invariant. See `docs/CCGEN_RED_TESTS.md`.

- Strengthen the end-to-end spherical full-symmetry direct-SCF regression ladder beyond the current focused infrastructure tests and committed NH3/CH4 ladder
- Add durable regression coverage for remaining full-symmetry edge cases called out in the design notes:
  D3h, Oh, linear-group interplay, and lone-atom behavior
- Preserve and rerun the CASSCF/PySCF gates when changing the optimizer. The May consolidation verification gap is superseded by the October/November validation recorded in Completion and `docs/CASSCF_PERFORMANCE.md`; this audit does not claim a fresh suite run.
- Keep documentation comments aligned with the implemented spherical symmetry representation; stale comments have already drifted once
- **FU2 — the i-shell (L=6) spherical path has never been checked against another
  code.** L=6 is the only production angular momentum that bypasses
  `normalized_pseudoinverse` entirely and delegates to the recurrence oracle
  (`cart_to_sph_block_recurrence`), so it is a distinct code path from the f/g/h
  gates landed 2026-08-28. The input and its reference are committed but
  **deliberately unregistered** — Ne/cc-pV6Z is 140 spherical AOs and the
  conventional `nb⁴` ERI build makes it far too heavy for the suite (h, at 91 AOs,
  already takes ~37 s):
  `tests/inputs/regression/spherical/ne_rhf_spherical_ccpv6z_ishell.hfinp`,
  PySCF 2.13.0 RHF/cc-pV6Z spherical Ne = `-128.5470611007` Eh, expect ~1e-9.
  A disagreement implicates `spherical_recurrence.cpp`, not the pseudoinverse fix.
  See `docs/SPHERICAL_F_SHELL_ACCURACY_SCOPE.md` FU2.

## ccgen follow-ups

The scope-document rewrites, UCC U0–U5.5 implementation, numeric UCC validation,
derived dressing wiring, transpose merging, spin-adapted default, and generated
OpenMP kernels have landed. Do not reopen the old “no production caller”,
“no OpenMP”, or six failing selection-model-gate tasks. See Completion and
`docs/CCGEN_{UNRESTRICTED_CC,WIRING_THE_DERIVATION_ROUTE,MERGE_TRANSPOSES,CC_OPENMP}.md`.

Remaining/deferred items from those investigations:

- UCC `wick`-engine coverage and a durable amplitude-antisymmetry gate.
- Spin-tag-aware operator reuse/dressing (O6), with UCC numeric validation;
  this is distinct from the already-complete UCC solver implementation.
- Rank-4 end-to-end performance evidence and the generated rank-4 `-O1` pin.
- Hand-written tensor-solver DIIS fragility and review of the small historical
  Be rank-4 reference offset before changing its pin.
- Generated-kernel scaling and hand-written post-HF threading, as scoped below.
  The old dressed-vs-undressed six-point probe route was retired, not left
  waiting to run. Optional external implementation/literature comparisons are
  not blockers.

These remaining research/coverage questions were not rebenchmarked by this
status audit.

## Spherical-basis work still intentionally guarded off

- Spherical analytic gradients (and therefore geomopt / freq) for the post-HF
  correlated paths (RMP2 / UMP2). RHF, UHF, and ROHF spherical gradients,
  geomopt, and frequencies are all landed (ROHF via the same build-W-in-the-
  spherical-basis-then-lift-once pattern the RHF/UHF paths use). MP2 gradients
  still need the response-machinery audit before the same lift adapter (Phase 1)
  can be wired through `compute_rmp2_gradient` / `compute_ump2_gradient`.
  Boundary markers: `water_rmp2_spherical_{gradient,geomopt}_rejected`.
- Spherical PCM (both binaries)
- Spherical double-hybrid gradients (the DH gradient code is Cartesian-only);
  boundary marker `water_rks_b2plyp_gradient_spherical_rejected`

## Symmetry follow-up

- Conventional-path symmetry-unique ERI storage remains out of scope; current full-group reduction is a direct-SCF feature
- ROHF is still outside the full-symmetry direct-SCF implementation scope
- The above-threshold memory-direct full-symmetry path has landed (`92838e03`); the remaining F3 candidate feeds the below-threshold persisted skeleton from block kernels. See `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`.

## DFT and response-method gaps

- **Double-hybrid analytic gradients: landed (see Completion); what remains is
  scope and throughput, not correctness.** RKS global DH supports gradient,
  geomopt, frequencies and geomopt+freq; UKS global DH supports `calculation
  gradient` only, on the pinned V4 grid. Remaining, in
  `docs/DH_IMPLEMENTATION.md` §9:
  - **UKS DH geometry optimization and frequencies** are rejected until their
    workflow contracts are accepted (the gradient they would call already
    passes FD).
  - **The UKS path is dense** -- full alpha/beta MO ERIs, a dense joint Jacobian
    solved by pivoted QR, and an `N_AO^4` AO pair density. None of the RKS
    streaming / matrix-free work (O7) has been carried over; it is qualified
    only on small molecules.
  - **O8 response throughput** (performance only): paired raw J/K traversal for
    hybrid response actions (O8.1), blocked fixed-grid XC action (O8.2), then a
    batched Krylov interface only if a benchmark shows it pays (O8.3).
  - **O9 RI-DH** is a separate energy backend with its own derivative and FD
    gates, not a speed switch -- it is the only route to dropping the remaining
    conventional `N_AO^4` AO ERI tensor.
  - **Not covered:** range-separated DH gradients, solvent response,
    SCS/DOS-style independent spin scaling, meta-GGA, frozen-core/RI PT2 in the
    gradient, and DH TDDFT. Each needs its own response and validation
    contract.
  - The diagnostic `PLANCK_DFT_DH_HESSIAN_PROBE_LOG` swap is tracked under
    Build and packaging (E3); it remains in the current tree.
- **The default DFT grid is not qualified for differentiated-grid
  properties.** Radial counts were fixed (PySCF `RAD_GRIDS`; the old formula
  gave 44 points where ~75 are needed), but the default `legacy` angular
  pruning failed the UKS DH rotational-covariance check (2.2e-5 Ha/Bohr on C1
  H2O2+, limit 2e-6) where the qualified `g4-3-v4` policy passes (1.1e-7).
  V4 is opt-in and mandatory only for UKS DH gradients. Whether to migrate the
  default is open and deliberately separate
  (`docs/DFT_GRID_CONVERGENCE_SCOPE.md`). **No committed regression case runs
  the production UKS DH gradient on V4**: the qualification evidence lives in
  the `tests/g4_3_*` runners, not `tests/regression_cases.json`, so routing or
  grid-contract breakage would go unnoticed by the suite.
- Double-hybrid TDDFT is unimplemented; UKS double-hybrid geometry optimization
  and frequencies are rejected (above)
- For range-separated functionals, `ImaginaryFollow` remains gated (the DFT
  driver does not implement it for any functional)
- Analytic Hessian remains unimplemented; frequencies are currently semi-numerical
- DFT imaginary-mode following is not implemented
### HPC campaign — what is left (verified against the tree 2026-08-30)

Architecture and measurements in `docs/HPC_MPI.md`. **Tiers 3 and 1 are done
and measured** for the default engine, HF and DFT: the `planck-mpi` front end, the
distributed direct-SCF Fock build (#146), the DFT J/K build (#151), and the **DFT
grid rank-split (#156 — Gap 2, CLOSED)**; `src/dft/driver.cpp` carries the
`Mpi::rank`/`size`/`allreduce_inplace` split. This entry previously described Gap 2
as the highest open DFT-HPC item and gave it a full ticket; it had already landed.

Measured on `scale.json` (notch386, 6-31g, os, 1 thread/rank): HF 42-46 %
efficiency at 32 ranks and near-linear to 16; DFT 46-65 % and **rising** with
system size. Both DFT walls — memory and scaling — are closed.

Remaining, in the doc's value order:

- **Gap 3 — no scale-proving fixture in CI (~S, the #1 item).** Every regression
  input is <= 6 atoms, so nothing fails if distribution silently regresses to
  replication. The cheap fix is one 16-water/6-31g (nb=208) case asserting
  `energy(-n 2) == energy(-n 4) == serial` **bitwise** — a correctness tripwire at
  a size where a partition bug can actually manifest, not a speed measurement.
  It is what stands between "DFT scales" (measured by a hand-run harness) and
  "DFT scaling cannot silently regress".
- **Gap 1 — two of four engines silently replicate (~S).** The one-shot
  `_compute_2e` tensor builds in HGP and Rys are not MPI-distributed; both carry
  an explicit in-tree admission (`hgp.cpp:1392`, `rys.cpp:1460`). `mpirun -n 8`
  with `engine hgp` in conventional mode is *correct but pointless* — every rank
  does 100 % of the work. OS already has the pattern (6 lines plus one
  `allreduce_inplace`). **Decide one of two — stripe them, or reject `-n > 1` with
  a clear message — but do not leave it silent.** Arguably YAGNI: HPC runs use
  direct SCF, which is already striped for every engine.
- **Gap 5a — post-HF OpenMP: PARTIALLY CLOSED (2026-08-30).** The gap was that
  every `src/post_hf/cc/*.cpp` backend and every generated kernel had **zero**
  `#pragma omp`. The **generated** side is now threaded: `planck_tensor_cpp.py`
  emits `collapse(3)` behind `CCGEN_OMP_COLLAPSE`, measured **3.22x at 4 threads**
  with energies bitwise identical across thread counts
  (`docs/CCGEN_CC_OPENMP.md`). **Still open: the hand-written path** —
  `tensor_backend.cpp` has 0 pragmas, and MP2 is unprofiled. The gap's own advice
  still applies: profile first, because which post-HF path is the real wall-time
  sink at a realistic size has never been measured. Its determinism constraint
  (scalar-accumulator terms are not thread-invariant under a naive `reduction`)
  was handled on the generated side by keeping the inner summed loop serial
  within a thread.
- **Gap 5b — dense post-HF MPI stripe (~M).** Untouched: `grep -rl "Mpi::"
  src/post_hf/` is empty. The reclassification is the useful part — RI is a
  *memory* strategy, not a distribution prerequisite, so the dense MP2/CC
  contractions can be rank-striped on the outer MO index today. RI-post-HF
  distribution stays the separate, memory-motivated tail (Tier 2).
- **DFT nb=416 fails to CONVERGE, not OOM.** 32-water RKS hits max_cycles at the
  serial baseline (4.8 GB, it ran and iterated). A convergence-robustness item
  (guess/DIIS/damping at large water chains), **wholly separate from the HPC
  track** — do not conflate it with a scaling defect.

**Closed since that doc was written:** Gap 4 (`scale_bench.py`'s Q2 verdict now
keys on the DFT RSS nb-exponent rather than a raw >50x ratio, `2b764b21`) and the
Q1 verdict (Karp-Flatt serial fraction rather than efficiency-at-max-ranks,
`ebf8ae5c`).
- Coarse/low-quality DFT grids can still show noticeable orientation sensitivity
  under symmetry reorientation; the validated symmetry-on gradient regression is
  intentionally pinned to `grid ultrafine`

## SCF, post-HF, and workflow gaps

- ROHF post-HF: FCI, CASSCF, and RASSCF now accept ROHF references; RMP2/UMP2
  and the coupled-cluster paths remain RHF/UHF only for ROHF inputs
- ROHF CASSCF/RASSCF only support a closed, doubly-occupied inactive core; a
  spin-polarized open inactive core (distinct alpha/beta core orbitals, with the
  unrestricted core Fock, core energy, and response-block changes it implies)
  is out of scope and stays rejected by the parity guard
- ROHF stability analysis and PCM remain incomplete (ROHF analytic gradients,
  and the geomopt / frequency workflows built on them, are now landed
  in both Cartesian and spherical bases — see Completion)
- The ccgen `TensorOptimized` RCCSDT backend is still treated in-tree as an experimental / phase-4 path


## ccgen generated-kernel performance

Accessor fixes, derived contraction factoring, hoisted operator construction,
transpose merging, and O2–O4 generated OpenMP emission are landed. The emitter
supports `CCGEN_OMP_COLLAPSE`; the historical four-thread result was 3.22x.
This does not imply every build enables it or that the hand-written tensor
backend is threaded. See `docs/CCGEN_CC_OPENMP.md` and Completion.

Remaining investigations, only with a current workload/profile:

- Re-profile the generated production configuration after these fixes; separate
  per-iteration cost from iteration count. Hand-written and generated solvers
  have different amplitude representations, so whole-run ratios are not
  residual-kernel ratios.
- Extend occupied-size and working-set coverage if scaling exponents or cache
  effects matter. The historical ladder stayed in cache and its fitted
  exponents are not a general production law.
- Measure rank 4 independently, including its registry `-O1` pin; consider
  chunking only after measuring compile/runtime cost. Rank 3 is not a proxy.
- Reassess unused contraction-order hints only if they eliminate work that
  derived dressing does not already remove. Loop fusion was measured neutral
  for runtime and remains a code-size/compile-time option, not an open speed fix.

The dressed-versus-undressed residual probe route was retired. The old
`PLANCK_CC_T3_TIME` branch was subsequently deleted; do not revive it as a
pending benchmark. Verify routing for new fixtures: the determinant-backstop
size guard belongs to the hand-written `tensor_backend.cpp` path, not every
generated-route run. Confirm the actual generated route instead of imposing
that guard universally.

## ccgen dressed intermediates

Production derived dressing and UCC solver support are landed. The remaining
spin-tag-aware dressing/operator-reuse question is O6 under ccgen follow-ups;
U1–U5 are not pending. Historical recognition-route and exact-cover proposals
are preserved in the archive, not active implementation plans.

### ccgen parallel generation is not equivalence-safe (separate defect)

`generate_cc_equations(method, parallel_workers=N>1)` produces a **different**
equation set than the serial (`workers=1`, default) path — not just reordered,
genuinely different coefficients/term counts (ccsd: singles 24 vs 27–29,
doubles 200 vs 154). Two independent order-dependent defects, each internally
deterministic:

1. **`_wickaccel` is not spawn-safe.** The C extension's `apply_deltas_layout`
   / `analyze_signature` return divergent index-layout results in a
   freshly-spawned worker vs the parent, corrupting relabeled terms (the energy
   manifold gets factors desynced from their summed-index lists, e.g.
   `f(i,a) t1(b,j)` with summed `(i,a)`). A `CCGEN_NO_ACCEL` env hook (added to
   `wick.py` / `canonicalize.py`) forces the pure-Python path and is inherited
   by spawned workers; it fixes the energy manifold but not defect 2.
2. **Pre-canonical exact merge is partition-local.**
   `merge_exact_term_into_buckets` dedups raw terms within a chunk before
   canonicalization; raw terms that combine when co-located in one chunk
   survive separately when split across chunks (singles: `-1/4` vs two `-1/8`).
   Making it global would defeat its streaming-memory purpose on large BCH
   expansions.

The default path is serial and *is* deterministic + correct; parallel is an
opt-in speed feature. The regression `test_parallel_generation_matches_serial`
is marked `@unittest.expectedFailure` with the root cause inline, and
`test_serial_generation_is_deterministic` pins the guarantee that holds. Real
fix = make the extension spawn-safe (rebuild `_wickaccel.cpp`) and lift the
raw-merge global; deferred as parallel generation is unused by the default
build. No bearing on the dressed-intermediate work above (that runs on the
serial path).

## BSSE follow-up

- DFT ghost / counterpoise support
- N-body counterpoise beyond two fragments
- Counterpoise-corrected gradients and geometry optimization
- Post-HF ghost-reference verification beyond the current SCF-level validated scope

## CASSCF

### Remaining work

#### Performance: leftovers, not scheduled (answered 2026-11-01)

The performance question is answered in `docs/CASSCF_PERFORMANCE.md` (see
Completion). The user accepted 1.3x behind PySCF at water cc-pVTZ C2v and
stopped the work. Not scheduled; take up only with a new target:

- re-profile cc-pVTZ C2v (where the last 1.3x goes is unmeasured);
- a better preconditioner for the relaxed solve (28-64 vectors per macro at
  cc-pVTZ; the diagonal is F_I + F_A differences only);
- the RI path has none of the speed-ups (FD orbital Hessian, full-leg-first
  transform, dense-density Fock builds; `water_casscf_ri_ccpvdz` 4.1 s);
- delete the temporary `CallProbe` (`PLANCK_CASSCF_CALL_PROBE`).

#### Unreproduced OpenMP hang in CASSCF (seen once, 2026-10-28)

One `water_cas44_631g` run (4 threads, `build-full`, g++-15 libgomp, macOS)
hung at 0 % CPU: every thread parked in `gomp_team_barrier_wait` at the end
of the `parallel for` in `compute_orbital_gradient` (`orbital.cpp`), reached
from `select_best_candidate_step` -> `evaluate`, with `nr_used = 1` (no
nesting). 60 reruns (probe on, off, and a pre-probe build) did not hang, so
the rate is around 1 in 60 or rarer. Nothing in that loop is shared-mutable,
so suspect libgomp on macOS before Planck code. If it recurs, capture
`sample <pid>` and check whether other `parallel for` regions in `orbital.cpp`
are involved.

#### Optimizer transcript labels — cosmetic only

The per-root candidate removal is complete. A low-priority remainder is to
make transcript labels uniquely identify the selected path. The older claim
that numeric Newton is the dominant accepted fallback predates P3–N9; the
current exact coupled solve is the production step and numeric Newton remains
an explicit/unreliable-step/stagnation fallback. Do not reopen the completed
optimizer rewrite or delete the SA-2 single-pair escape probes as cleanup.

### Future hardening

- Preserve the stationary-plateau exit in `src/post_hf/casscf/casscf.cpp`:
  it requires stagnation, small energy change, a flat accepted step, the escape
  condition, and weighted SA gradient below `max(1e-6, tol_mcscf_grad)`.
  An SA stationary point need not make every individual root gradient zero;
  this is a supported convergence route, not a claim that all SA solutions
  require it.
- **Test existence checked 2026-11-06 at `22da4338` plus local changes:**
  `water_casscf_sa2_sto3g_sad_guess_uphill` is a **regression case ID** in
  `tests/regression_cases.json`, not an input filename. Its tracked input is
  `tests/benchmarks/casscf/pyscf_reference/water_cas44_sto3g_sa2_sad_uphill.hfinp`
  (`engine rys`, `guess sad`, `mcscf_accept_uphill .true.`). The case asserts
  `casscf_converged_via_plateau=true`; its SAD companion
  `water_casscf_sa2_sto3g_sad_guess` asserts `false`. Preserve both policies.
  Both cases are tagged **extended**, so the runner's default `core` filter
  excludes them even when `--case` is supplied. Verify registration with
  the listing below (`--list` itself lists all cases without filtering):

  ```sh
  python3 tests/run_regressions.py --list | \
    rg '^water_casscf_sa2_sto3g_sad_guess_uphill:'
  ```

  To execute the single case:

  ```sh
  python3 tests/run_regressions.py --suite all \
    --case water_casscf_sa2_sto3g_sad_guess_uphill --build-dir build
  ```

  This existence/registration check does not claim a fresh solver-test pass.

## Performance and maintenance opportunities

- **RHF vs PySCF — answered 2026-10-25; S1 and S3 landed, B2.2 declined**
  (`docs/SCF_VS_PYSCF.md`). With matched tolerances the iteration counts are
  identical. **The "11–16× kernel gap" was a benchmark artifact:** PySCF
  stores the ERI tensor in memory whenever it fits, regardless of
  `direct_scf`, so it had been running conventional SCF.
  `tests/benchmarks/pyscf_bench.py` now forces it direct and asserts it.
  Direct against direct, a full Fock build is level or faster in Planck on
  6-31G\* (0.27–0.95×) and 1.5–3.7× slower on cc-pVXZ. It is not general
  contraction (B2.2 declined). Remaining open:
  - **The end-to-end SCF table** in `docs/SCF_VS_PYSCF.md` is still the
    in-core one; rerun the fixed harness.
  - **The cc-pVXZ gap — answered 2026-10-26** (`docs/ERI_KERNEL_VS_LIBCINT.md`,
    rewritten from its scope). It was a benchmark artifact, a Boys defect,
    redundant basis primitives, and high-L single-primitive kernel cost,
    now served by the 2D Rys block kernel (cc-pVTZ Fock builds 1.11–1.16×;
    see Completion). Still open:
    - **Close the Auto-table validation record** (`45ef3cef`): check later
      results for `planck-compute-2e`, `engine_scf_energy_compare.py` including
      He₂/cc-pV5Z, and the regression suites; rerun only missing coverage.
      The refit itself is landed.
    - **Rerun the Planck-vs-PySCF Fock table** (doc §3); it predates Rys2D.
    - Gradient G3 is closed: HGP/Rys2D derivative block kernels and the shared
      shell-quartet loop landed in `ae5baf02` / `cbe1cd50`. Remaining derivative
      work is listed below, not another G3 rescope.
    - **Route the conventional Auto build's 6D hoist**
      (`_contracted_eri_block_hoisted_views`, 50–210× libcint) through
      Rys2D and delete the hoist.
    - **libcint is still 2.3–2.6× faster** on the blocks Rys2D owns; compare
      its `g`-tensor layout (`g0`/`g2d` strides in `cint2e.c`) only if that
      matters. Four obvious layouts already lost (doc §4).
    - **General contraction (G0):** rerun the family probe restricted to
      closed-form classes on 4 and 8 H2O cc-pVDZ at 1 and 4 threads; build
      only if the whole build gains ≥ 1.10× with Pople unchanged (doc §5).
    - **Smaller:** rerun `pyscf_bench.py` end to end; check the `-RIFIT`
      files for the same redundant primitives (JK-fit sets are usually
      uncontracted); the non-kernel half of the 6-31G\* fused loop (low
      priority, Planck already beats PySCF there).
  - **DFT:** the KS J/K direct builds (`_compute_2e_j_direct` / `_k_direct`)
    do not yet build incrementally.
  - **Boys v1.2.1 validation record:** reconcile the original re-pin/full-suite
    checklist with later runs before scheduling it again. The correction is
    landed; this audit has not established completion of that specific checklist.
- **Gradient performance — core work landed.** G1–G5 and G7 O1/O2 are committed
  (`e851749e` through `32f0819f`); derivative ERIs are no longer an exclusively
  serial per-AO path, and UMP2 has matrix-free response and GEMM back-transforms.
  See `docs/GRADIENT_INTEGRAL_PERFORMANCE.md` for the measured 1.1–4.3× gradient/
  SCF ratios and conditions. G7 O3 was dropped, not left pending. Remaining:
  - G6: profile RI 2c/3c and derivative-integral cost before optimizing it.
  - Block-level density screening and screened derivative block kernels.
  - Consider stored-ERI UMP2 response actions only if their measured share pays.
  - Measure conventional stored-tensor construction on real post-HF workloads.
  - Preserve the separate analytic-Hessian and spherical-MP2 capability gaps.
  The two overlap-only MP2 loops were deliberately left alone, not missed.

- **ERI/Fock BLAS-LAPACK migration — scoped 2026-10-17, nothing landed**
  (`docs/ERI_FOCK_BLAS_SCOPE.md`). BLAS buys nothing for the ERI kernels,
  direct-SCF digestion, the conventional `nb⁴` contraction (bandwidth-bound,
  `dsymv` 0.8–1.7× at 4 threads) or the cart→sph transform (dense `dgemm`
  0.14–0.62×, `C` is 2.5–4 % nonzero). The real lever is **RI-JK**:
  `build_ri_pair_factors` is density-independent but recomputed twice per
  iteration, **~61 %** of the RI-JK Fock build (B0, no BLAS; invalidate with
  `_ri_j3c`, and add the first RI-JK regression case). Then B1 optional
  `USE_BLAS` (Accelerate RI-K GEMMs 4.3–7.8×, bitwise thread-count-invariant
  on M3; OpenBLAS unmeasured), B2 RI-K from `C_occ` on the SCF caller only
  (4.4–4.8×), B3 `dsyevd` (diag is 0.08 % of an iteration today; deferred).
- **ERI/Fock shell-quartet work — landed, with bounded follow-ons.** Closed-form
  kernels, HGP lanes, dense digestion, deterministic dynamic bins, batched Boys,
  and grow-only scratch are committed. The unused family-quartet traversal was
  deleted (`1e047aa2`); it is not a dormant working-tree feature. The Auto table
  was refit (`45ef3cef`), though outstanding validation is tracked above.
  Remaining in `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`:
  - R1b: per-AO readout table, only if the production-loop threshold is met.
  - C5: screened block kernels.
  - F3: block-fed persisted full-group skeleton, a one-time setup optimization.
  - Conditional same-class batching/four-raise kernels only after profiling;
    RI-JK pair-factor reuse is tracked separately under BLAS/LAPACK.
  Historical measurements and rejected B2/R1/S2 proposals are archived.

- **Shell-pair granularity follow-ons** (architecture in
  `docs/SHELLPAIR_GRANULARITY_HANDOFF.md`; none block the landed Auto-path Rys
  hoist):
  - **B-4 — explicit `engine rys`.** The hoist is wired into Auto (B-3) but not
    the non-Auto Rys tensor/Fock builds. It reorders accumulation, so gate at the
    `1e-13` ERI bar, not bitwise, plus the SA-2 uphill case under `engine rys`.
  - **Phase C — symmetry direct-SCF skeletons** (`os_symm` / `hgp_symm` /
    `rys_symm`) are still per-AO; convert to shell-quartet carrying the signed-AO
    orbit per component. Unblocked: block kernels exist for all three engines.
  - **Phase D is closed for Auto/Coulomb:** HGP and Rys2D derivative blocks
    landed under G3. Explicit-engine and screened-kernel AO fallbacks remain;
    screened block support is the narrower open item.
  - **Phase E — retire the per-AO adapter.** Make `build_shellpairs` emit
    shell-level `ShellPair` objects and drop `expand_shell_groups_to_ao_pairs`.
    Last, because it changes the shared `ShellPair` contract and must resolve
    per-component `_component_norm` folding across OS/HGP/Rys at once.
- Audit any remaining reversed-shell-pair reconstruction in explicit-engine/
  screened gradient fallbacks before proposing cleanup; the shared Auto block
  path and shared SCF/MP2 gradient loops have already replaced the old copies.
- Deduplicate the full-group AO-transform machinery that still exists in both `group_operations.cpp` and `mo_symmetry.cpp`
- Extract a shared `SpatialQuartetLayout` (6-axis dims + strides +
  `spatial_index` + `resize_for_quartet`) and retrofit the OS, HGP, and Rys
  per-quartet scratch onto it. All three now carry near-duplicate per-quartet
  scratch structs — OS's `_eri_scratch`, HGP's `g_hgp_scratch`, and (as of
  PR #126) Rys's `RysScratch` — so the three concrete call sites exist to shape
  the shared interface. Only the spatial-layout core is common; the Boys `m`
  axis, HGP's `a0c0_accum`, OS's no-zero-init policy, and the differing
  accessors stay engine-specific. Bitwise-gate across all three engines
  (`planck-compute-2e`, `planck-hgp-engine-smoke`, plus the OS path via the
  existing ERI gates).
- `Calculator` already groups cached MP2/UMP2 results in `MP2Result`
  (`src/base/types.h`); that refactor is not pending. A geometry-derived
  working-state object with one invalidation point remains a separate design
  candidate, to justify by concrete invalidation defects.
