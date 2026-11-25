---
name: Open Work
description: Canonical summary of known gaps, risks, and follow-up work in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, open-work, canonical, roadmap]
---

# Open Work

Last updated: 2026-11-20 (restricted and UHF CCSD(T); RKS DH
gradient refuses frozen-core and RI PT2; CI is not blocked on the
shared-objects change). Previous rewrite 2026-11-16:
RI-JK answered and removed from the priority list; libecpint repinned to
upstream, #82 filed; PySCF #3598 merged upstream.

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

1. **CCSD(T).** It is the benchmark method users ask for first.
   Restricted and UHF (T) landed 2026-11-20, batched and threaded, and equal
   PySCF in Cartesian and spherical bases. Since 2026-11-22 `ccsd(t)` runs
   the generated spatial RCCSD (48–59× faster than the spin-orbital one).
   Remaining: the size ladder (where the generated RCCSD and UCCSD stop),
   and ROHF-(T).
   Scope: `docs/CCSD_T_SCOPE.md`.
2. **SCF robustness on medium systems.** 32-water RKS hits `max_cycles`
   (§2, DFT). Scope: `docs/SCF_CONVERGENCE_MEDIUM_SYSTEMS_SCOPE.md`.
3. **Meta-GGA**, at least SCF energies with a by-name refusal everywhere else,
   then gradients (§2, DFT). r2SCAN, ωB97M-V, M06-2X and TPSSh are rejected
   today. Scope: `docs/META_GGA_SCOPE.md`.
4. **Default DFT grid qualified for gradients** (§2, DFT). `legacy` fails the
   rotational-covariance check, so default-settings geometries and
   frequencies carry grid noise. Measured 2026-11-18: at the default grid
   the gradient depends on orientation at 4–13e-4 Ha/Bohr for ordinary
   functionals; V4 fixes isolated molecules but degrades with neighbouring
   molecules (water tetramer 1.2e-6 Eh) and costs 9× the points. Rescoped
   the same day: two tiers (default ≤ 3e-5, tight ≤ 1.5e-6 Ha/Bohr), the
   existing policies measured first, and V5 only if tight-tier clusters
   are needed. The ordered plan S1–S10 covers this and the XC-gradient
   scope. 2026-11-19: S3/S3b done. No `normal`-level setting passes (they
   fail naphthalene); `pyscf-nwchem-2.13@fine` passes the default tier with
   ~10× margin but not the tight tier. S7 stage 1 (AO screening) landed so
   the larger runs fit. Tight tier: V4 misses it on naphthalene (3.0e-6);
   `pyscf-nwchem-2.13@ultrafine` meets it (8.2e-7) with fewer points.
   Its noise does not grow from naphthalene to anthracene (STO-3G).
   Next: S4 (frequency noise), S5 (cost at size). Scope:
   `docs/DFT_DEFAULT_GRID_SCOPE.md`.
5. **CI running again** (§3). Next: measure the build's disk use (CI scope
   step 1). Runner cleanup, cached external projects and `-g0` may be enough
   on their own; the shared-objects change is the durable fix, not a
   prerequisite. Scope: `docs/CI_SCOPE.md`.
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

**Nice to have.** Imaginary-mode following in `planck-dft` and under PCM;
HF-D3 and modified D3 damping; counterpoise-corrected gradients and N-body
counterpoise; ROHF stability, SOSCF and PCM; a spin-polarized inactive core in
ROHF CASSCF; range-separated, solvent, SCS/DOS, meta-GGA and TDDFT
double-hybrid variants; the incremental KS J/K build and PCM performance
leftovers; `TensorOptimized` RCCSDT; conventional symmetry-unique ERI storage;
further post-HF ghost-atom checks; RI under MPI step M2 (distribute the
3-center tensor; matters past nb ≈ 1000–1500; M0–M1 landed,
`docs/RI_MPI_SCOPE.md`).

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
    Frozen-core and RI PT2 are refused by name for RKS and UKS since
    2026-11-20 (`water_rks_b2plyp_{frozen,ri}_gradient_rejected_sto3g`).
  - `mp2_level_shift` is parsed but nothing reads it: no MP2, UMP2 or DH
    kernel applies it. Either wire it or delete the keyword.
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

- **CCSD(T): restricted and UHF are done** (`docs/CCSD_T_SCOPE.md`).
  `correlation ccsd(t)` and `uccsd(t)` equal PySCF in Cartesian and
  spherical bases, with batched, threaded (T). `uccsd(t)` needs
  `-DPLANCK_CC_UCC=ON`. System size is now limited by the CCSD, not by (T).
  Open:
  - ROHF-(T), after semicanonical ROHF-CC;
  - size (`docs/CCSD_T_SIZE_SCOPE.md`). Memory binds, not the CC algebra.
    Water dimer / cc-pVTZ `ccsd(t)` peak footprint 5.5 → 1.43 GB on this
    8 GiB machine, energy identical. The changes: one SCF tensor copy,
    batched `transform_eri`, (T) reusing the CCSD's blocks, re-exec with
    macOS `MallocLargeCache=0` (libmalloc reused freed large blocks whole),
    freed ERI thread scratch, a spherical tensor built per shell quartet
    (never the Cartesian one), the CC solver moving its state instead of
    copying every integral block, and the AO tensor freed before the CC
    iterations. Each phase now holds one large object: the ns⁴ AO tensor
    (SCF) or v⁴ `vvvv` (CCSD). 7 (ladder re-measure): the dimer energy
    equals PySCF 2.15 to every digit, but time is now the limit and the
    gap to PySCF grows with size, CCSD ~2× → 29× and (T) ~10× → 47× from
    water cc-pVDZ to the dimer in cc-pVTZ (213 / 283 s against 7.4 / 6.0 s).
    8: the restricted (T) computes W once per virtual triple for all six
    orderings (24 distinct contractions instead of 144 evaluations), (T)
    5–6× faster: dimer 283 → 49.9 s, 8× behind PySCF; whole run 278 s.
    Next: 9 CCSD time (`docs/CCSD_T_SIZE_SCOPE.md`).
    The gap was more arithmetic times slower arithmetic. Generator changes
    closed the arithmetic to 1.4× PySCF (a j ≥ i half range, a
    common-operand merge on t2 and on v) and most of the rate by lowering
    every large binary contraction to an Eigen GEMM
    (`src/post_hf/cc/gemm_contract.h`). Dimer cc-pVTZ CCSD per iteration
    11.8 → 0.85 s, 2.7× PySCF, and the ratio no longer grows with the basis;
    whole `ccsd(t)` 278 → 81 s. Step 10 halved the (T) blocks by t2
    symmetry and runs the `t2 · ovvv` blocks as small GEMMs per triple:
    (T) 49.8 → 8.25 s. Step 1a (2026-11-25) made every (T) block a GEMM
    on contiguous layouts built once per (T): (T) 4.5–4.8 s, faster than
    PySCF's 6.0 s; whole run ~37 s. R3b (staircase GEMMs on j ≥ i) done:
    CCSD 1.04–1.08× per iteration. UCC (1c): the GEMM lowering reaches
    UCC, water⁺ cc-pVTZ UCC CCSD 0.339 → 0.145 s per iteration (with U1); the GEMM
    helper's parallel regions now run serially on tiny calls, which also
    made the small CC gates 3–6× faster. Next for UCC: batched t1·vvvv
    builders, half range and merges, the UCC (T) kernels.
    Open, in order: R3c
    Accelerate behind the helper (2.5–4× on the GEMMs; B1 build decision),
    R2c shared builder combinations, R4 re-measure. The hand-written
    fallback is not justified.
  - CCSD[T], CCSDT(Q), CCSDTQ(P) and the Λ variants
    (`docs/CCSDT_Q_SCOPE.md`). The non-Λ projections generate in ccgen
    today; (Q) is limited by the generated CCSDT's cost, and Λ needs new
    ccgen machinery.
  - CC2, CC3, CCn and CCSDT-n (`docs/CC_APPROXIMATE_MODELS_SCOPE.md`): one
    per-term perturbation-order filter in ccgen; worth building with an
    excited-state CC scope. Its step 0 is a naming defect today: `ccN` /
    `uccN` mean rank N, so `cc3` runs full CCSDT. Spelled rank names now
    exist on both routes (`PostHF::RCCGEN` / `UCCGEN`, one keyword table);
    left are migrating seven inputs and refusing `ccN` (user decision).
  - **Generated RCCSDT: reading amplitudes through a fixed-rank view made
    it 5–6× faster per iteration on one thread** (`docs/CCGEN_GENERATED_CC_COST.md`).
    Hot nests read t2/t3 as `t3({...})`, a runtime-rank offset loop with
    the index array rebuilt in memory on every read: about 40 instructions
    per multiply-add. The emitter now binds every amplitude as
    `fixed_rank_view<N>` (`src/post_hf/cc/common.h`) and emits
    `t2(i, j, a, b)`. Iterations and energies are bitwise unchanged; all 9
    generated CC regression cases pass. On an 8-rung ladder, generated is
    1.4–2.3× the hand-written backend per iteration on one thread (was
    2.2–5.1×; ratio fit o^0.38 · v^−0.32, LOO 11 %), and faster
    wall-clock at four threads on every rung but CH4 (2.9–4.2× from
    H2O/6-31G up). A 1.0e-8 Eh gap on C2H2 was the
    hand-written solver stopping early (a fixed 1e-6 step tolerance let
    |dE| alone decide); it now uses `tol_density`, and the routes agree to
    ≤ 4e-10 (C2H4 1.7e-9, convergence noise). Open: re-run the ladder's
    hand-written timings (it now takes more iterations), and its residual
    plateaus near 1.7e-10. The registry's `-O1`
    pin was measured and stays: `-O3` gains 1.15–1.23× for an ~21 min
    compile.
    - Before this, generated RCCSDT was 1.3–2.6× the hand-written one
      wall-clock at 4 threads (within 2× on 6 of 8 rungs) and about 5.7×
      on one thread; CH4 was 19× at 4 threads. Four fixes got it there:
      every residual term binary (the emit path had kept matching the
      retired route's seeded operators, leaving 112 terms n-ary), threaded
      builders, the triples residual once per aab symmetry orbit, and
      factored builder bodies (M3.0 had never reached the production
      emit). The per-iteration ratio went from o^0.78 · v^−0.24 to
      o^0.11 · v^−0.63, so the gap no longer grows with size.
    - Also open: whether this lands, and the default `ccsdt` routing of a
      dressed build.
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
  shared. `planck-mpi` now takes `hartree-fock`'s CC definitions from one
  list (it lacked UCC and dressing until 2026-11-25, unintended).
- **No build-and-test CI runs, because the runners ran out of disk.** The
  "CMake on multiple platforms" workflow
  (`.github/workflows/cmake-multi-platform.yml`) is `disabled_manually`; its
  last run was 2026-04-26. Its runs failed for lack of space on GitHub, and
  since #181 it builds every target in two `PLANCK_CC_MAXORDER` jobs, so
  re-enabling it as it stands would fail the same way. The workflow fixes in
  #181 (build every target, `BASIS_PATH` for every test) have therefore never
  executed on CI. It needs the build's disk use cut: first measure it, then
  try runner cleanup, cached external projects and `-g0` (`docs/CI_SCOPE.md`
  steps 1–2); the shared-objects item above is the durable fix. A first
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

- **DFT AO arrays: S7b, only if needed.** AO screening per point block
  (S7 stage 1, 2026-11-19) replaced the planned X4 streaming. What remains
  (`docs/DFT_DEFAULT_GRID_SCOPE.md`, S7):
  - port the dense consumers (SOSCF, stability, TDDFT, DH) to blocks and
    delete the dense arrays; until then those runs allocate them on top;
  - spatial sorting of grid points (water12 27 % → 21 % kept);
  - shell-extent prescreening of the AO evaluation.
- **DFT J/K.** After S7, the J and K direct builds are ~41 % of a
  naphthalene B3LYP gradient run, and are built separately each iteration.
  A joint J+K pass is the next DFT time lever; not scoped.
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
  memory strategy, not a prerequisite for it. RI work is split across ranks
  with the data replicated (M1, 2026-11-17); distributing the 3-center
  tensor (M2) is open: `docs/RI_MPI_SCOPE.md`.
  - **CC under MPI** (`docs/CC_MPI_SCOPE.md`). Every rank currently runs
    the full CC redundantly. In order:
    - M0: (T) across ranks; bitwise equal to serial through owned slots.
      (T) is ~21 % of the water dimer cc-pVTZ run now, so it pays.
    - M1: split the generated CCSD residual by outer index.
    - M2: split the memory; only once a single node is saturated.

    Both prerequisites are met (spatial RCCSD, threaded generated kernels);
    RI is not needed while time, not memory, binds.
    `planck-mpi` has `uccsd(t)` since 2026-11-25 (same CC definitions as
    `hartree-fock`), and `tests/mpi_smoke_compare.py` compares the CC
    totals; baseline gates `{water_rccsd_t,oh_uccsd_t}_631g_mpi_smoke`.
    Order and status: `docs/CC_REMAINING_WORK_HANDOFF.md`.
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
    bound of −2064.896580 Eh by 0.108–1.092 Eh. A code review
    (2026-11-17, "Code review findings" in the doc) found four defects
    that confound this result. All four are fixed and pass the unit tests, the statistical gate and the 13 FCIQMC
    regressions; thread identity holds at 1, 2, 4 and 8 threads.
    `n2_fciqmc_s5_threads1` was re-pinned. The four defects:
    1. Surviving parent weights were never stochastically rounded, so
       sub-walker determinants piled up (93–96 % low-weight visits).
    2. The projected error bar blocked per-step ratios, not the reported
       ratio of sums; it understated the seed spread about 4×.
    3. Blocking had no plateau check. It now flags
       `[ERROR BAR UNCONVERGED]`.
    4. The projected sums dropped steps with |c₀| < 3.

    The re-anchored reference is now chosen from averaged weights.
    The five-seed CAS(12,10) control after the fixes: projected SD across
    seeds / RMS reported SE 4.15 / 3.87 mEh (was 9.82 / 2.48), RMSE
    5.2 mEh (was ≈10.6). H₂ and LiH regressions were lengthened to
    128k / 240k steps after a 20-seed study showed their projected errors
    were 30 % too small at the old lengths.
    On the fixed code the CAS(12,18) projections still exceed the bound
    (+0.165–0.231 Eh at τ = 5–6), and τ = 55 at 25k walkers does not
    converge: the shift keeps falling and the anchor wanders. In exact
    CASCI the RHF determinant holds 54 %, 0.55 % and 0.02 % of the
    wavefunction in 10, 12 and 13 orbitals. Next: walker scan
    (25k/100k/400k) on exact CAS(12,13), −2064.746875 Eh; the inputs are
    `tests/inputs/exploratory/fciqmc/cr2_cas12_13_walkers_*.hfinp`. Too
    slow for a laptop at 400k; run it elsewhere.
    Open side questions: a −4 mEh mean offset of both estimators on
    CAS(12,10) (five seeds, ~2 SEM), and a shift error bar 2–6× larger
    than the spread across seeds. Two SC-NEVPT2 estimates differ by 19 mEh.
  - Opt-in probabilistic spawning from low-weight parents passed the
    November 16 correctness and thread-identity checks. Five CAS(12,10)
    pairs: projected RMSE 2.48 → 1.40 mEh, shift RMSE 5.92 → 6.45 mEh,
    3.14 % more mean wall time. It treated a symptom of defect 1 above;
    after the rounding fix no parent is below one walker, so the option has
    no effect and can be removed.
- **ccgen generated kernels** (`docs/CCGEN_GENERATED_CC_COST.md`). Threading has been on by default
  since 2026-11-20 (`PLANCK_CC_OMP_COLLAPSE`, default 3):
  - Re-profile the generated production configuration, separating
    per-iteration cost from iteration count.
  - Measure rank 4 on its own, including the registry `-O1` pin. Rank 3 is
    not a proxy. Threaded rank 4 is bitwise correct, but slower on
    Be/STO-3G (thousands of short nests); time it on a system that matters.
  - Extend occupied-size and working-set coverage if cache effects matter.
  - Confirm the actual route for new fixtures: the determinant backstop
    belongs to `tensor_backend.cpp` only.

## 6. ccgen

- UCC: `wick`-engine coverage and a durable amplitude-antisymmetry gate.
- **Derived operators on UCC (O6), wired 2026-11-22**
  (`docs/CCGEN_UNRESTRICTED_CC.md`, "the derived operators reach UCC"). A
  dressed build now dresses the UCC emit too, and the factorizer's UCC output
  is value-gated (`UCCRewritePreservesValueTests`); the UCC regression
  gates pass on the dressed build. The UCC orbit fold landed on top of it
  (`docs/CCGEN_UCC_ORBIT_FOLD.md`). Open:
  - a timing of dressed against undressed UCC from the same commit;
  - the order-576 symmetrize pass (rank-4 same-spin quadruples) profiled
    on a system larger than B/STO-3G;
  - spin-resolved `tensor_symmetries` entries (`v_aaaa`, `v_abab`), which
    would add merges.
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
