---
name: Completion Status
description: Canonical summary of what is implemented and validated in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, completion, validated, canonical]
---

# Completion Status

Last updated: 2026-11-12

This is the canonical completion-status document for the repository.
Subsystem handoff, plan, benchmark, and fix-summary notes may still exist for
historical design context, but they are no longer the source of truth for
"what is done". Use this file together with `vault/Status/Open Work.md`.

## Fully Implemented and Validated

### HF / SCF core

- RHF, UHF, and ROHF SCF with DIIS (`src/scf/scf.cpp`)
- ROHF MO-energy bookkeeping is consistent: both spin channels store the
  canonical per-spin Fock diagonals (`epsa`/`epsb`), a matched pair aligned to
  the `C` column order that `_reorder_rohf_orbitals` sorts by `epsa`. Previously
  the alpha slot held the effective Roothaan eigenvalues (a mislabeled pairing
  with canonical `epsb`); the effective set was a convergence device only and is
  not read after the reorder. The two ROHF-reachable consumers (CASSCF/FCI
  active-space selection) use these energies for ordering only, so the change is
  behavior-neutral for them; it fixes the user-facing MO-energy printout. Planck
  now matches PySCF 2.13.0 ROHF `Cᵀ Fα C` (epsa) exactly. Gated by
  `homo_energy`/`lumo_energy` `metric_close` on `water_radical_cation_rohf_sto3g`
- H_core and SAD initial guesses
- Same-basis checkpoint restart and density restart
- Symmetry detection, MO irrep labeling, and SAO-blocked Fock diagonalization
- Wavefunction stability analysis for RHF/UHF, plus optional instability following
- Mulliken, Lowdin, Mayer, dipole, quadrupole, and related property reporting
- C-PCM solvation for RHF/UHF energies, analytic gradients, geometry optimization and frequencies
- **SOSCF for RHF, UHF, RKS, and UKS** (second-order, augmented-Hessian
  orbital step convergence acceleration), reusing the CASSCF CIAH solver
  and `apply_orbital_rotation` unchanged for all four. Runs as a transient
  window that hands back to DIIS by default (matching ORCA's own handoff),
  or to full unbounded convergence (same energy). Switch trigger is a fixed
  iteration or the DIIS error norm. Off by default; composes cleanly with
  SAD. All four reach fully-converged DIIS's energy to all 10 digits on
  genuinely open-shell (UHF/UKS) or closed-shell (RHF/RKS) systems, with
  superlinear gradient shrinkage.
  - **RHF/UHF** (`docs/SOSCF.md`, `docs/SOSCF_UHF.md`):
    reuse `build_rhf_cphf_matrix` / `build_uhf_cphf_matrix` unchanged
    (`build_uhf_cphf_matrix` split out of `solve_uhf_cphf` for this). UHF
    is FD-verified `g_true=2·g_used`, `H_true=2·Amat`; per-spin Cayley
    rotation + semicanonicalization; mutually exclusive with an active
    level shift.
  - **RKS/UKS** (`docs/SOSCF_DFT.md`, built on `docs/DFT_ANALYTIC_FXC_HESSIAN.md`):
    the KS orbital Hessian's XC piece
    is an **analytic** `fxc` Hessian-vector product
    (`compute_analytic_xc_hessian_vector_product{,_polarized}`,
    `src/dft/analytic_hessian.cpp`, LDA + GGA), NOT the
    `O(n_occ·n_virt)`-grid-pass FD-kernel oracle. `h_op` composes it with
    the orbital-energy-difference diagonal, a Coulomb response, and — for
    hybrids — an exact-exchange `K` response, all from the trial density.
    **All hybrids are supported now — RKS and UKS, global (B3LYP, PBE0)
    and range-separated (HSE06)**; only PCM and SAO fall back to plain
    DIIS with a warning. Three related fixes landed alongside the hybrid
    work: the RKS kernel term is scaled `2×` relative to the diagonal
    (`4×`/`8×` in the raw κ-parametrization — the old single `4×`
    half-weighted it and made RKS PBE/LDA converge *linearly*;
    `SOSCF_DFT.md` invariant 2); the analytic XC Hessian was
    double-counting a correlation `fxc` for combined exchange-correlation
    functionals (`DFT_ANALYTIC_FXC_HESSIAN.md` invariant 3a); and a step
    deadband lets pure-SOSCF terminate at tight `tol_density` instead of
    hitting `max_cycles` on its limit cycle (invariant 7). Gates:
    `water_rks_{lda,b3lyp,hse06,pbe_soscf_puredeadband}_soscf_631g`,
    `h2o_cation_uks_pbe0_soscf_631g`. The UKS analytic path is a
    measurable 3–5× per-Newton-step wall-clock win at every size tested;
    the RKS path is correct and correctly-scaling but not reliably faster
    than the FD-kernel alternative at the two modest sizes measured. Two
    pre-existing limits carried, both unrelated to the K term: the
    polarized GGA `fxc` for B88/LYP-based functionals shows ~1e-4..1e-3
    FD scatter on small directions in a whole-molecule probe (the
    point-level `δV_xc` is exact — `dft_gga_hessian_selfcheck` now covers
    B88/LYP); and triplet-radical UHF landscapes are shallow enough that
    SOSCF can find a lower basin than DIIS, so a UKS hybrid gate needs a
    clean single-minimum open-shell system.
  - **ROHF SOSCF is not done** — needs new ROHF orbital-response/CPHF
    machinery (no such code in the tree), the same gap behind
    ROHF-MP2/stability/PCM. See Open Work.

### Direct SCF and full point-group symmetry

- Obara-Saika, Rys, and HGP direct Fock engines, with auto-dispatch by angular momentum
- Full point-group ERI reduction for direct RHF/UHF in the Cartesian basis
- Full point-group ERI reduction for direct RHF/UHF in the spherical-harmonic basis
- Metric-correct spherical group operators
  `O_sph = S_sph^{-1} (C S_cart O_cart C^T)`
- Focused validation of the full-symmetry machinery:
  `planck-group-operations`, `planck-fock-symmetrization`,
  `planck-symm-fock-equivalence`
- Committed direct full-symmetry regression ladder through Td, including spherical
  NH3/CH4 cases
- Persisted full-symmetry skeleton ERI across SCF iterations (C1), so the
  density-independent skeleton is built once and reused during the SCF cycle
- Monomial-operator fast path in full-group symmetrization for operations that
  reduce to signed AO permutations

### Spherical-harmonic basis support

- Spherical single-point RHF, UHF, and ROHF
- Conventional and direct SCF in the spherical working basis
- Spherical property reporting
- Same-basis and cross-basis checkpoint restart in the spherical working basis
- Spherical MP2, CASSCF, RASSCF, FCI, FCIDUMP export, and coupled-cluster energies
- Spherical analytic gradients for RHF, UHF, and ROHF, via the
  `lift_density_sph_to_cart` adapter that maps the spherical SCF density and
  energy-weighted density back to the Cartesian basis so the Cartesian
  derivative-integral engine can be reused unchanged (the energy is invariant
  under the basis change, so the lift carries no approximation). For ROHF the
  energy-weighted density `W = Pa·Fa·Pa + Pb·Fb·Pb` is built in the spherical
  basis and lifted **once** — never from separately-lifted factors, since the
  cart←sph transform C is non-square (C·Cᵀ ≠ I), so
  `lift(Pa·Fa·Pa) ≠ lift(Pa)·lift(Fa)·lift(Pa)`. PySCF-validated to ~1e-7
  Ha/Bohr on water/6-31g* (RHF), OH/STO-3G (UHF), and the water-cation doublet
  6-31g* (ROHF, `h2o_cation_rohf_spherical_gradient_631gd`)
- Spherical geometry optimization (IC-BFGS + Cartesian L-BFGS) and
  semi-numerical frequencies for RHF, UHF, and ROHF, plus geomopt+frequency and
  imaginary-mode following. Driven by a shared
  `HartreeFock::SCF::rebuild_basis_dependent_state` helper that re-runs the
  spherical `_cart_to_sph` row-normalization and the `C·(T+V)·Cᵀ` working-basis
  lift at every displaced geometry, keeping the geomopt/freq inner loops in
  lockstep with the driver's startup setup. PySCF-validated to <0.1 cm⁻¹ on
  water/6-31g* vibrational frequencies and ~6e-8 Eh on the IC-BFGS optimized
  energy
- Spherical symmetry support at both the SAO-blocking level and the full-group
  direct-SCF level
- Regression coverage for the landed spherical single-point feature set plus
  RHF gradient, geomopt, frequency, geomopt+frequency cases (all PySCF-gated;
  see `tests/regression_cases.json` entries `water_rhf_spherical_{gradient,
  geomopt,freq,geomoptfreq}_631gd`), with unsupported workflows hard-gated
  rather than allowed to return wrong answers (boundary markers:
  `water_rmp2_spherical_{gradient,geomopt}_rejected`)
- High angular momentum (f/g/h) spherical transform validated **cross-CODE**
  (2026-08-28). The `L >= 3` `cart_to_sph_block` matrices carried integer
  coefficients for RAW Cartesian monomials while the integral engine feeds
  **unit-normalized** components, so the spherical functions were not pure
  harmonics in the physical monomials — a contaminated l-subspace that put
  water/cc-pVTZ 2.14e-5 **below** the variational minimum in a same-spanning
  basis. Fixed earlier by `normalized_pseudoinverse` (`src/basis/spherical.cpp`);
  what landed now is the coverage that was missing, and the reason it shipped:
  every high-L gate was **Cartesian and cross-ENGINE**, and since all engines
  share the transform they agreed with each other while all being wrong.
  Three PySCF-anchored spherical RHF gates close it —
  `water_rhf_spherical_ccpvtz_fshell` (f, 4.7e-9, 19 iters against 83-142
  pre-fix), `ne_rhf_spherical_ccpvqz_gshell` (g) and
  `ne_rhf_spherical_ccpv5z_hshell` (h), the latter two matching PySCF to all ten
  printed digits. References were built from Planck's **own** GBS via
  `pyscf.gto.basis.parse_gaussian`, removing the basis-coefficient confound
  rather than assuming it away. `tests/spherical_transform.cpp` additionally
  pins the two convention-invariant properties the old shape/rank check could
  not see — harmonic purity in the physical monomials, and row-space equality
  with the recurrence oracle — both mutation-verified: re-introducing the defect
  fails exactly L = 3,4,5 (max|∇²| 2.2e-1/1.8e-1/8.4e-2) while L <= 2 and L = 6
  stay green. The i-shell (L=6) path bypasses the pseudoinverse for the
  recurrence oracle; Ne/cc-pV6Z matches PySCF to every printed digit
  (−128.5470611007 Eh, 2026-11-09, run by hand because it is too heavy for the suite).

### Post-HF methods

- RMP2 and UMP2 correlation energies
- Analytic RHF, UHF, RMP2, and UMP2 gradients
- FCI over the full MO space for small RHF/ROHF references
- CASSCF, SA-CASSCF, and RASSCF from RHF or ROHF references (open-shell
  supported when the unpaired electrons live in the active space, so the
  inactive core stays closed-shell; PySCF-gated by `o2_casscf_rohf_sto3g` and
  `oh_casscf_rohf_sto3g`)
- Coupled-cluster support for RCCSD, UCCSD, RCCSDT, UCCSDT, RCCSDTQ
- Arbitrary-order RCC solver via ccgen-generated residuals
- **Arbitrary-order UNRESTRICTED CC (UCC) from ccgen-generated residuals**, behind
  `-DPLANCK_CC_UCC=ON` (default OFF, so the default build is unaffected and emits no
  UCC translation unit). Keywords `ucc2`…`ucc6` plus `uccsd_gen`/`uccsdt_gen`/
  `uccsdtq_gen`, routing through `PostHF::UCCGEN` → `run_uccgen`. Validated against
  three independent references on B/STO-3G (doublet, 3α/2β):

  ```
  ucc2  -24.1892581442   == hand-written UCCSD, exactly
  ucc3  -24.1892636163   T3 recovers 80.1% of the UCCSD->FCI gap
  ucc4  -24.1892649766   == in-tree FCI, all ten digits
  ```

  `ucc4 == FCI` on an **open-shell** system is the strongest UCC gate: CCSDTQ is exact
  there because T5 is unreachable in the basis (only 2 alpha virtuals for 3 alpha
  electrons), not because the electron count is small. Gated by `b_ucc{2,3,4}_sto3g`,
  which skip cleanly in a default build via the runner's `requires_build_option`.

  UCC is RCC minus the spatial collapse — the same generator, bridge, runtime and
  sector machinery, with the spin blocks kept resolved. What that costs is one
  vocabulary per layer instead of one object: 24 ERI arrays (7 `aaaa` / 10 `abab` /
  7 `bbbb`), per-block denominators (`abab` differs in *shape*), per-spin Fock blocks,
  four orbital counts, and a distinct kernel namespace so both sets link into one
  binary. See `docs/CCGEN_UNRESTRICTED_CC.md` and
  `docs/CCGEN_UCC_ERI_ANTISYMMETRY.md`.
- Tensor-backed and determinant-space coupled-cluster paths, including the
  optimized RCCSDT warm-start route
- **Dressed ccgen CC kernels — RETIRED as a production route, deliberately.**
  The Stanton-Gauss `Wmnij`/`Wabef`/`Wmbej` + `tau`/`tau_c` operators are
  recognized diagrammatically, generate from the build, compile, and link
  (`-DPLANCK_CC_DRESS_OPERATORS=ON`; default OFF, so the default build is
  byte-identical and unaffected).

  **The spin-adapted dressed path is unsupported: its RCC kernels are wrong.**
  Dressing and spin adaptation do not compose — each transform is correct alone
  and the composition is wrong in **either** order, because recognition
  subtracts what an operator absorbs against a term set that adaptation then
  changes. Measured: dressed Be/STO-3G CCSDTQ `E_corr` = −0.0247182895 against
  an exact −0.0517746319 (52 % short).

  Retired rather than fixed because the payoff does not justify the work: the
  measured saving is ~1.2–1.5× (spin-orbital, actual) against a research task —
  deriving a spatial operator set plus a spatial-capable matcher — for a bounded
  ~1.9–2.8× expected. The generated-vs-hand-written slowdown is a larger
  untouched lever (this entry previously cited it as "~180×"; that figure never
  carried its dimensions and did not reproduce — see the tensor-accessor entry
  below for the measured numbers). Five fix attempts each passed their gate and
  made the energy worse. Full answer, with the measurements and what was kept:
  `docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`.

  **The seven tests of this route are `@unittest.expectedFailure` as of
  2026-08-22**, with the reason inline in each. Four are the composition defect
  itself (`test_dressed_numeric_oracle`, `test_dressed_spatial_equivalence`);
  three are the F0 builder-vs-usage layout defect on the mixed-space operators
  `tau`/`Wmbej` (`test_intermediate_layout_agreement`). All seven were confirmed
  pre-existing on a clean tree. They are marked rather than fixed (that means
  investing in an abandoned route) and rather than deleted (they are numeric
  instruments this note keeps deliberately). **An unexpected PASS is the signal
  that the composition was genuinely fixed** — at which point this entry, and
  the notes in those files, should be revisited.

  **A second route exists and also fails, which the retirement never considered.**
  `factorize.py` *derives* operators from contraction structure rather than
  matching hand-seeded fingerprints, and is basis-agnostic (GCC and spatial
  `ccsd` doubles both yield the same 20 operators). Measured 2026-08-22, it does
  not preserve the residual's value either — **on GCC**, where there is no spin
  adaptation to blame: 23 of 66 `ccsd` doubles terms disagree
  (‖diff‖/‖R‖ = 3.73e-01). So the retirement's *decision* is better supported
  than when it was taken, but its stated *reason* ("dressing and spin adaptation
  do not compose") is not what the second route demonstrates. The factorizer has
  **no numeric gate** — its 47 tests compare factor `Counter`s, which are blind
  to index order by construction. Full record and the three discarded
  hypotheses: `docs/CCGEN_TWO_DRESSING_ROUTES.md`.

  An earlier version of this entry claimed a verified rank-3 equivalence
  (`h2` 12/12, `lih` 16/16, `bh3` 26/26). That comparison never ran the
  generated kernel — `compute_ccsdt_triples_residual` had no caller until
  `64d0074`, so both builds executed hand-written code and agreed vacuously.

### CASSCF / SA-CASSCF status

- **Performance (2026-10-28..11-01, P1-N9, through `a224881a`):** water
  CAS(4,4) is faster than PySCF at 6-31G (0.069 vs 0.34 s), cc-pVDZ (0.177 vs
  0.27 s; C1 0.266 vs 0.43 s) and cc-pVTZ C1 (1.86 vs 2.30 s), and 1.3x
  behind at cc-pVTZ C2v (1.49 vs 1.16 s); it started at 13.4 s for cc-pVDZ and
  ~176 s per macro at cc-pVTZ. Exact CI-relaxed orbital Hessian (two missing
  coupling terms restored), solved matrix-free with a seed vector so it still
  leaves zero-gradient saddles; trust region from the exact model; the Newton
  step taken as soon as it improves (22 -> 2 evaluations per macro); and one
  half transform (mu nu|lam i) per evaluation from which every Fock build and
  mixed integral block is an O(nb^3) contraction. RDM reductions made
  thread-count invariant. Two saddle pins re-pinned to minima (water SA-2,
  `oh_casscf_rohf_sto3g`). Answer: `docs/CASSCF_PERFORMANCE.md`.

- Shared-kappa state-averaged coupled orbital/CI solve is the primary production path
- Exact CI-response RHS is the default
- FD-based SA orbital Hessian action (`delta_g_sa_action`) is implemented and wired
- Active-integral-cache transform is landed and benchmarked
- Per-root SA total-energy reporting is fixed
- SA diagnostics are parsed by the regression runner
- SAD-start uphill-enabled water SA-2 basin is retained as a separate
  regression mode. Reaching it used to depend on ~1e-14 roundoff (`diis_dim`
  6/7/10 and the integral engine flipped it); fixed 2026-10-23 by a noise
  floor in the candidate selector (no gradient-only acceptance below
  `tol_mcscf_grad`) and a plateau exit that counts "no step" as flat and, under
  `mcscf_accept_uphill`, needs the escape probes offered first. All seven
  variations now land at -74.7877864784
  (-74.787786527 since the 2026-11-01 Bohr-constant fix)
- Plateau-escape convergence branch is hardened: the old rounding-sensitive
  `100·tol_mcscf_grad` screen is replaced by an explicit `sa_g`-stationarity
  bound `plateau_sa_g_bound = max(1e-6, tol_mcscf_grad)`, and a
  `casscf_converged_via_plateau` diagnostic is emitted and asserted by the
  runner — `false` for `water_casscf_sa2_sto3g`,
  `water_casscf_sa2_sto3g_sad_guess`, and `ethylene_casscf_sa2_sto3g`; `true`
  only for `water_casscf_sa2_sto3g_sad_guess_uphill` (all four green)
- Stagnation-cascade trim: the per-root candidate steps (`root*-coupled` /
  `root*-grad-fallback`) and their generator `build_root_resolved_coupled_step_set`
  are removed. A suite-wide accepted-candidate sweep showed they never won a
  merit selection while costing a per-root coupled solve every stagnant macro.
  `numeric-newton` (dominant fallback) and single-pair probes (load-bearing on
  the SAD-uphill SA-2 canary) are kept. Verified zero behavior change: 121/121
  regressions green including all 11 PySCF CASSCF gates, and the SAD-uphill case
  accepts the identical candidate sequence as before

### Gradients, optimization, and frequencies

- Analytic gradients for RHF, UHF, ROHF, RMP2, and UMP2
- Geometry optimization in Cartesian and internal coordinates
- Semi-numerical Hessian / vibrational frequencies
- Imaginary-frequency following
- Constrained geometry optimization
- ROHF analytic gradients (Cartesian basis), plus the geometry-optimization
  and frequency workflows built on them. The gradient is structurally the UHF
  gradient — same Hellmann-Feynman + Pulay + 2e + Vnn terms over the alpha/beta
  densities — with one ROHF-specific piece: the energy-weighted density
  `W = Pa·Fa·Pa + Pb·Fb·Pb` (`build_rohf_energy_weighted_density` in
  `src/gradient/gradient.cpp`, built from the spin Fock matrices the SCF already
  persists). This is exactly PySCF's ROHF `make_rdm1e` (`W_a + W_b`) and is
  required because ROHF orbitals are canonical for the effective Roothaan Fock,
  not for the individual spin Focks, so the UHF `Σ ε_i C_i C_iᵀ` form is wrong.
  No CPHF/Z-vector solve — ROHF SCF is variational, so the SCF gradient needs
  none, same as RHF/UHF. PySCF-gated by `oh_rohf_gradient_sto3g`,
  `ch3_radical_rohf_gradient_sto3g` (low-symmetry C1, all 12 gradient components
  non-zero, matches PySCF analytic to ~8e-8), `oh_rohf_geomopt_sto3g`
  (E_opt Δ ~2e-8 Eh), and `oh_rohf_freq_sto3g` (stretch Δ ~0.07 cm⁻¹ vs a
  PySCF FD-of-analytic-gradient Hessian). Spherical ROHF gradients, geomopt,
  and frequencies are also landed (same build-W-in-spherical-then-lift-once
  pattern; PySCF-gated by `h2o_cation_rohf_spherical_{gradient,geomopt,freq}_631gd`
  to ~1e-7 Ha/Bohr, ~1e-7 Eh, and <0.1 cm⁻¹). ROHF-MP2 gradients, ROHF
  stability, and ROHF PCM remain out of scope.

### DFT

- RKS and UKS
- LDA, GGA, global hybrids, and arbitrary libxc functional selection
- **Grimme D3 dispersion** (2026-11-07, `docs/DISPERSION_D3.md`):
  `dispersion d3bj|d3zero|d3bj-atm|d3zero-atm` through simple-dftd3 v1.7.0
  (LGPL, fetched at build time, needs Fortran; `PLANCK_WITH_DFTD3`). It is
  added once per KS scaffold call and once in `compute_analytic_ks_gradient`,
  so single points, gradients, geomopt, frequencies and the KS half of DH
  gradients all carry it. Parameter names are derived from the resolved libxc
  ids; a functional without parameters is rejected by name. Against PySCF
  `mf.disp` over a separately built simple-dftd3 v1.0.0: D3 energies to every
  printed digit; totals ≤ 3e-11; water-dimer gradient ≤ 8.2e-10 Ha/Bohr, also
  under Cs reorientation; geomopt E_opt 1.0e-8. Eight cases, `*d3*_sto3g`;
  mutation-checked. pyscf-dispersion 1.0.0 does not load on macOS arm64
  (numpy 2 loader call, x86_64 dylib), and its wrapper turns unknown names
  into a zero energy; the reference script works around both.
- Range-separated libxc functionals for single-point, analytic-gradient,
  geometry-optimization, frequency, and geomopt+frequency workflows
- Double-hybrid libxc functionals for single-point energies (RKS and UKS;
  range-separated double hybrids single-point only)
- **Restricted global double-hybrid analytic gradient** (landed 2026-09-12,
  `647aeaa7`), plus geomopt, frequencies (central differences of analytic
  gradients) and geomopt+freq on the same callback. Neese-Schwabe-Grimme (2007)
  stationary Lagrangian, `src/dft/dh_pt2_gradient.{h,cpp}`. Analytic vs
  total-energy FD at two steps: water 1.70e-8, C1 H2O2 1.34e-8 Ha/Bohr
  (B2PLYP/STO-3G, limit 5e-8). Production is matrix-free and streamed:
  GMRES Z-vector (`solve_dh_zvector`, no dense fallback), direct h/S/ERI
  derivative contractions, Eq. 47 pair density streamed in bounded AO tiles,
  only five MO-ERI sectors formed, scratch-backed pair-packed T2 tiles
  (O7 closed 2026-10-16, `6a621425`). The remaining `N_AO^4` object is the
  conventional AO ERI tensor. The long debugging arc that preceded this
  carries three lessons:
  - **Two separate omissions.** Wiring XC-II (the fixed-matrix,
    moving-function geometry derivative of `D:V_xc[P]`, including Becke and
    owner-point grid motion) removed 69% of the residual. The rest was in the
    PT2 half and closed by implementing the paper's objects literally: the
    four-coefficient amplitude derivative, the literal Eq. 47 RHS/overlap, and
    the hybrid Eq. 46 complement (driver comment in
    `src/dft/driver.cpp`).
  - **Measure the KS and PT2 halves separately.** They partially cancel in
    the total, which understated the real defect by 3.2x.
  - **Count a fixture's independent gradient components before scoring
    candidate terms.** Water/C2v has only 3, so any three candidates fit
    exactly; four attempts were reverted on that artifact.

  PySCF's `grad/mp2.py` is not a valid DH reference (its response uses the
  nonlinear `get_veff` on a non-idempotent trial density). Validation is
  Planck's gradient against Planck's own energy FD.
- **UKS global double-hybrid analytic gradient** (`73ce3866`, promoted with the
  grid in `039bd2df`); geometry optimization and frequencies since
  2026-11-12 (`docs/UKS_DH_GEOMOPT_FREQ.md`). Same Lagrangian with
  per-spin densities, three amplitude blocks (c, 2c, c), alpha/beta coupling
  only through J and the cross-spin XC kernel, and a joint alpha-then-beta
  Z-vector (`src/dft/udh_*.cpp`). Partly dense (see Open Work).
  Scope: conventional all-active UMP2 with stored T2, no solvent, no range
  separation, and the pinned grid `grid_policy g4-3-v4` + `grid_radial_points
  135` (`validate_qualified_uks_dh_grid`). Fourth-order FD on C1 H2O2+, HO2 and
  an asymmetric water cation (B2PLYP/STO-3G): worst 2.68e-9 Ha/Bohr; translation
  below 5.9e-13.
- **DFT grid: radial counts fixed and a qualified V4 pruning policy.** Radial
  shells now come from PySCF's `RAD_GRIDS` period x level table (presets map to
  levels 1/3/5/7). The old ORCA-style formula gave 44 points for a second-row
  atom where ~75 are needed, and the energy still moved 2.9e-5 Eh at
  "ultrafine". `grid_policy g4-3-v4` was selected on fixed-density XC
  energy/potential/kernel L1 limits and passed a blind suite frozen before its
  first run. On live UKS DH gradients it cut the rotational-covariance error
  from 2.2e-5 (legacy) to 1.1e-7 Ha/Bohr. Opt-in; `legacy` stays the default
  (`docs/DFT_GRID_CONVERGENCE.md`).
- Treutler-Ahlrichs radial grid, Lebedev angular grid, and Becke partitioning
- Grid quality levels: Coarse, Normal, Fine, UltraFine, with explicit
  `grid_radial_points` / `grid_angular_points` overrides and versioned
  `grid_policy` pruning schedules
- C-PCM solvation for RKS/UKS energies, analytic gradients, geometry optimization and frequencies
- Linear-response TDDFT / Casida and TDA excited states
- Hybrid and range-separated TDDFT (2026-10-16, `docs/RS_TDDFT.md`).
  The full-Casida B-matrix exact-exchange term used `(ia|bj)` instead of
  `(ib|ja)` in both RKS and UKS (B3LYP root 3 off by 7.5 meV), ungated because
  every TDDFT case was PBE; fixed in one shared helper
  (`src/dft/tddft_exchange.h`). Range-separated response adds
  `c_sr K[erf, omega]` from the same coefficient source as the SCF. HSE06,
  CAM-B3LYP and wB97X agree with PySCF 2.13.0 on water/STO-3G within
  4e-4 eV per root across RKS singlet/triplet, UKS spin-conserving, TDA and
  Casida; the residual is grid/kernel (UKS PBE, with no exact exchange, is
  already 1.9e-4 eV off). Gated by `water_tddft_{b3lyp_casida,b3lyp_tda,
  hse06_casida,hse06_triplet_casida,camb3lyp_casida,wb97x_tda}_sto3g`,
  `water_triplet_uks_tddft_{b3lyp,camb3lyp}_sto3g` and the Yukawa rejection
  case
- DFT single-point, gradient, geometry optimization, frequency, and geomopt+frequency workflows
- DFT checkpoint/restart and symmetry+SAO blocking
- Symmetry-enabled DFT gradient/frame handling fixed by synchronizing
  `_coordinates` to the symmetry-standardized frame before grid construction;
  covered by the `water_dft_hse06_gradient_symm_ultrafine_fd` regression
- HSE06 analytic-gradient validation against both finite differences and PySCF,
  including the long-range exchange contribution and a symmetry-on ultrafine
  finite-difference regression for water

### BSSE / counterpoise

- Ghost atoms, including multiple input syntaxes
- Automated two-fragment SCF-level Boys-Bernardi counterpoise driver
- Per-fragment charge and multiplicity handling
- PySCF-validated He2/cc-pVDZ counterpoise decomposition
- **DFT ghost atoms and counterpoise (2026-11-11, `docs/DFT_COUNTERPOISE.md`).**
  `run_counterpoise` takes the sub-calculation runner as a parameter, and
  `planck-dft` passes its KS single point, so HF and KS share one driver.
  Before this, `planck-dft` silently ignored `%begin_bsse`. PCM with
  counterpoise is now refused in both binaries (it used to be dropped
  silently). Other changes:
  - **Ghost grids** use the element's Z, as in PySCF ≥ 2.14 (#3302).
  - **Symmetry** is now turned off when ghosts are present; He + Gh(He) had
    been detected as D∞h and the SCF did not converge.
  - **KS convergence:** both KS loops now feed the DIIS residual into
    `is_converged`, with a KS stall floor of 1e-8 (HF keeps 1e-10). A lone He
    from `hcore` had stopped at a stalled step 6.2e-7 Eh high. The floor must
    stay above the SOSCF deadband residual (4.1e-10,
    `water_rks_pbe_soscf_puredeadband_631g`).

  Gates, all matching PySCF to every printed digit:
  - `water_dimer_ghost_{rks_pbe,rks_b3lyp,cation_uks_pbe}_631gd`
  - `{he2,water_dimer}_rks_b3lyp_cp_ccpvdz`
  - `water_dimer_rks_b3lyp_d3bj_cp_ccpvdz`: its BSSE equals the D3-free
    value, because ghosts carry no D3.

### Recent fixes now considered landed

- **RI-JK SCF gated against PySCF (2026-11-12, `docs/RI_JK_ROUTINE_SCOPE.md`
  step 0).** `scf_ri_jk` had no regression case. Water, water⁺, C1 H2O2 and
  H2O2⁺ (RHF/UHF, Cartesian 6-31G*, cc-pVDZ-RIFIT) now equal PySCF
  `density_fit` to every printed digit, with both bases read from Planck's GBS
  files (`tests/pyscf/ri_jk_references.py`); `*_rijk_631gd`, tagged core. No
  code changed. The MP2 fitting set costs 1.0–1.7e-3 Eh against the exact
  energy. Step 1 added `def2-universal-JKFIT` and `cc-pVTZ-JKFIT` (the first
  contracted auxiliary sets), also equal to PySCF on all four: fitting error
  ≤ 8.9e-5 / 5.9e-5 Eh. Gated by `*_rijk_jkfit_631gd`. No separate
  `scf_ri_basis` keyword: RI-JK and RI-MP2 share one fitting cache. Step 2
  lifted the spherical refusal: the fitted factors were already in the
  working basis, so no code was needed; `*_rijk_spherical_631gd`.
  Then the fitted aux was made to follow `basis_type`, as in PySCF
  (`docs/RI_SPHERICAL_AUX.md`): a per-shell transform on the metric and the
  3-center tensor, 14–20 % fewer fitted functions, and spherical RI-JK and
  RI-MP2 equal plain PySCF density fitting to every printed digit (before,
  a Cartesian aux put them 1e-5..2e-4 Eh off every other code). Cartesian
  RI results unchanged. Step 3: `scf_ri_jk` now refuses gradient, geomopt
  and frequency workflows; before, they ran with the exact-integral gradient
  at RI orbitals, 9.9e-6 Ha/Bohr off the RI energy (water, JKFIT), silently
  (`water_rhf_rijk_gradient_rejected`).
  Step 5: RI-JK ROHF (one branch: ROHF builds the UHF (Ga, Gb)); water⁺
  equals PySCF ROHF density fitting, Cartesian and spherical.
  Step 8a: AgCl/def2-SVP with ECP and def2-universal-JKFIT (two l = 6 aux
  shells on Ag) equals PySCF to every printed digit; no code changed.
  Step 4: SOSCF and stability use the fitted operator under `scf_ri_jk`
  (RI-MP2's CPHF route and `transform_eri_ri`); before, SOSCF rebuilt the AO
  tensor every step and stability used the exact Hessian. Eigenvalues equal
  PySCF stability on a density-fitted SCF (RHF and UHF).
- **RI-JK in `planck-dft` for pure and global-hybrid single points
  (2026-11-16, `docs/RI_JK_ROUTINE_SCOPE.md` steps 12–13).** The KS assembly
  adds the fitted J and K in the working basis after projecting the other
  terms. Water RKS/UKS PBE, spherical water RKS PBE, and water RKS, water⁺
  UKS and C1 H2O2 RKS B3LYP equal PySCF `density_fit()` to every printed
  digit (6-31G*, def2-universal-JKFIT). KS SOSCF and stability use the
  fitted operator (step 14): stability eigenvalues equal PySCF on
  `density_fit()` to all seven digits; the exact-integral Hessian was up to
  6.7e-6 off. Cartesian RKS/UKS gradients and geomopt (step 15) equal
  PySCF's density-fitted gradient with grid response to ≤ 7e-11 Ha/Bohr and
  E_opt to every digit. Range-separated functionals, spherical derivatives,
  TDDFT, PCM and counterpoise with `scf_ri_jk` are refused by name.

- **RI pair factors cached per geometry (2026-11-16, B0;
  `docs/RI_JK_ROUTINE_SCOPE.md` step 11).** B = J V^{-1/2} is built once by
  `ensure_ri_3c_ready` into `_ri_pair_factors` and reset with `_ri_j3c`,
  instead of re-solved on every J/K call. Benzene 6-31G*/JKFIT RI-JK energy
  1.5× faster end to end, energies bitwise identical.
  - **`planck-ri-3c-eri` fixed (test side).** It had failed since K1
    (2026-10-25): its PySCF export parsed the raw cc-pVDZ GBS while
    `read_gbs_basis` drops redundant primitives, so it compared different
    contracted functions (same span, so no energy gate noticed). The export
    script now mirrors `drop_redundant_primitives`; passes at 1e-10. The
    script was also untracked (`scripts/*` is ignored); it is now allowlisted
    in `.gitignore`.

- **RI-JK HF gradient, geometry optimization and frequencies (2026-11-16,
  `docs/RI_JK_ROUTINE_SCOPE.md` steps 9–10).** One helper,
  `build_ri_scf_two_electron_gradient`, gives the fitted 2e term for the Γ
  RHF, UHF, ROHF and RKS share; the three SCF gradients call it under
  `scf_ri_jk`. Cartesian RHF is gated: water and C1 H2O2 gradients, water
  geomopt (E_opt −76.0107180372) and frequencies equal PySCF density fitting
  to every printed digit (6-31G*, def2-universal-JKFIT).
  - **Found.** X = J V⁻¹ was built from an explicit V⁻¹. The Cartesian JKFIT
    metric has cond 8e11, which put the gradient 2.1e-6 Ha/Bohr off its own
    energy FD; two triangular solves fix it. The RI-MP2 gradient shares the
    function; it had been gated only with the better-conditioned RIFIT sets.
  - **Memory.** Benzene 6-31G*/JKFIT peaks at 243 MiB; the stored derivative
    tensors alone would have been 1.04 GiB.
  - **UHF and ROHF (step 10, same day).** Already wired through the same
    helper; gated on water⁺ UHF/ROHF and C1 H2O2⁺ UHF, every printed digit
    against PySCF.
  - **Still refused:** spherical (16), correlated and PCM RI-JK derivatives.

- **RI-MP2 frequencies, and a streamed RI derivative contraction
  (2026-11-16, `docs/RI_JK_ROUTINE_SCOPE.md` steps 6–8).**
  - **Frequencies.** RI-RMP2/UMP2 `frequency`, `geomoptfreq` and
    `imaginaryfollow` run; no new code, only the refusal removed. PySCF has
    no DF-MP2 gradient, so the reference is a fourth-order FD of PySCF's
    RI-MP2 energy, central-differenced at Planck's 5e-3 Bohr step: water
    1480.60/5000.85/5418.23 cm⁻¹ and water⁺ 1347.72/4689.10/4868.83, equal
    to every printed digit (one last-digit rounding). Gated by
    `water_ri_rmp2_freq_sto3g` and `water_radical_cation_ri_ump2_freq_sto3g`;
    disabling the RI cache invalidation fails both by up to 41 cm⁻¹.
  - **Streamed contraction.** The RI 2e gradient term contracted
    3·natoms stored npair × naux derivative tensors (≈ 32 GB at 20 atoms,
    nb = 300, JKFIT). One loop per class (`for_each_3c_deriv`,
    `for_each_2c_deriv`) now feeds it element by element; equal to the
    stored-tensor formula to 1.4e-14, RI-MP2 results unchanged.
  - **`planck-dft` refuses `scf_ri_jk`.** It parsed the keyword and ran
    exact integrals silently (`water_rks_pbe_rijk_rejected`).

- **HF frequency SCFs warm-start from the reference density (2026-11-12,
  `docs/ANALYTIC_HESSIAN_SCOPE.md` lever 1).** 19–20 → 14 iterations per
  displaced SCF, but only 1.04× on C1 cyclohexane/6-31G (SCF is 14 % of the
  run) and 1.14× on the 12 HF frequency cases; frequencies unchanged. HF
  frequency cost is gradient cost.

- **UKS double-hybrid geometry optimization and frequencies (2026-11-12,
  `docs/UKS_DH_GEOMOPT_FREQ.md`).**
  - **One gradient for every caller.** The UKS DH assembly moved out of the
    `Gradient` case into `compute_analytic_udh_gradient`, which the
    per-geometry callback (optimizer and semi-numerical Hessian) now calls
    too. Widening the gate alone would have run UKS through the RKS DH path.
  - **Gates.** `tests/dh_workflow_audit.py` (now RKS or UKS; its UKS-refusal
    assertion was removed) on HO2 B2PLYP/STO-3G, V4 grid: Hessian vs 18
    independent gradients 2.8e-9; energy FD vs gradient at the optimum
    2.2e-9 Ha/Bohr. Water RKS and `scf_type uhf` frequencies agree to every
    printed digit. Regression `ho2_uks_b2plyp_v4_geomoptfreq_sto3g`
    (extended, ~6 min) pins E_opt and the three frequencies.
  - **Found.** OH is a poor fixture: its β hole in a degenerate π pair is a
    zero mode (λ_min −2.4e-8), and one SCF drifted along it for 168
    iterations at fixed energy. A property of the molecule, not a
    convergence defect; HO2 has no such pair.

- **RMP2/UMP2 frequencies, Cartesian and spherical (2026-11-12,
  `docs/SPHERICAL_MP2_GRADIENT_SCOPE.md` step 5).**
  - **One per-geometry routine.** Geometry optimization and the
    semi-numerical Hessian each had their own SCF → gradient step, and only
    the geomopt one ran MP2. Both now call `Opt::run_scf_gradient`
    (`src/opt/geomopt.cpp`). Geomopt warm-starts it; the Hessian runs it
    cold. `frequency`, `geomoptfreq` and `imaginaryfollow` work with
    `correlation rmp2|ump2`.
  - **Gates.** Both match PySCF 2.15.0 to every printed digit: central
    differences of its analytic MP2 gradient at h = 5e-3 Bohr.
    - `water_radical_cation_ump2_freq_sto3g` (Cartesian UMP2; was the
      `_rejected` case)
    - `water_rmp2_spherical_freq_631gd` (spherical RMP2)
  - **Then refused.** RI-MP2 frequencies (`water_ri_rmp2_freq_rejected`);
    lifted 2026-11-16 (entry above).

- **Spherical RMP2/UMP2 gradients and geometry optimization (2026-11-11,
  `docs/SPHERICAL_MP2_GRADIENT_SCOPE.md` steps 1–4).**
  - **Design.** The MP2 Lagrangian stays in the working basis. Everything
    the Cartesian derivative kernels contract is lifted once:
    - the 1e density, the overlap-term matrices and the separable-Γ factors
      directly;
    - the pair densities by rebuilding them from Tᵀ C.
  - **Shared helpers.** One working-basis Fock helper
    (`SCF::two_electron_fock{,_uhf}`) and one MO lift
    (`BasisFunctions::mos_to_cartesian`), each replacing copies.
  - **Gates** (6-31G* spherical): water and C1 H2O2 RMP2, water⁺ and C1
    CH3 UMP2 gradients against a fourth-order FD of PySCF's MP2 energy
    (≤ 1e-8, every component). Water RMP2 geomopt E_opt −76.1955706453
    equals PySCF `geometric` GAU_TIGHT to every printed digit.
  - **Refusals.** Spherical RI-MP2 gradients stay refused
    (`water_ri_rmp2_spherical_gradient_rejected`). MP2 frequencies stay
    refused in every basis (step 5).
  - **Found.** PySCF < 2.15's analytic `mp2.Gradients` misses its own
    energy FD by up to 4.2e-7 Ha/Bohr on C1 fixtures, so it was not used as
    the reference. That is a Krylov/CPHF early-termination bug, fixed in
    2.15.0 (pyscf/pyscf#3334); 2.14.0 still has it.

- **Frozen-core UMP2 gradients are refused (2026-11-11).** With
  `mp2_frozen`, the UMP2 gradient used to run and print a gradient
  4.5e-2 Ha/Bohr off the energy FD, with no warning (water cation STO-3G,
  O z: −0.1647 vs −0.1201). The builders have no frozen-core terms: no
  frozen–active relaxation blocks, and they see only the active orbitals
  (`docs/SPHERICAL_MP2_GRADIENT_SCOPE.md`). RMP2 already refused it.
  - **One shared check.** All three builders now call
    `has_frozen_orbitals`, which compares the active count with the SCF MO
    count. RMP2's old comparison with `nbasis()` would have refused every
    spherical gradient.
  - **Gates.** `water_rmp2_frozen_gradient_rejected` and
    `water_radical_cation_ump2_frozen_gradient_rejected`.

- **Every PCM workflow gated against PySCF; three defects fixed on the way
  (2026-11-10, `docs/PCM_GATING.md`).** Six new water/6-31G* cases: UKS and
  spherical RKS gradients (≤ 3.6e-11 Ha/Bohr), and UHF, spherical RHF, RKS
  and UKS geomopt + frequencies (energies ≤ 1e-9 Eh, frequencies ≤ 0.9
  cm⁻¹ against PySCF's analytic PCM Hessian). Removing the PCM gradient term
  fails all six. Fixed: `compute_hessian` now rebuilds the reference state
  (HF frequency runs had published the last displaced geometry to the results
  JSON); the KS SCF stores the D3/PCM energies so DFT geomopt no longer writes
  `solvation_energy` 0; and the geometry-optimization force threshold is the
  new `%begin_geom` keyword `geomopt_grad_tol` (default 3e-4 unchanged).

- **Results JSON reports D3 and PCM energies separately (2026-11-10,
  `docs/RESULTS_JSON_ENERGY_TERMS.md`).** `electronic_energy` had silently
  included the D3 energy. It now excludes D3 and still includes PCM, and
  `dispersion_energy` and `solvation_energy` are their own fields (0 when
  off), carried on the `Calculator` so both binaries write them. The runner
  gained `json_checks` (it runs the binary with `--json`). The PCM case pins
  `solvation_energy` to PySCF's G_pcm to every printed digit, and the D3 case
  pins `dispersion_energy` to the PySCF D3 reference. Both cases also check
  that the energy terms add up to the total.

- **UKS double-hybrid gradient: one architecture with RKS, accepted
  (2026-11-10, `docs/UDH_GRADIENT_ONE_ARCHITECTURE.md`, now an answer).**
  RKS and UKS share the derivative contraction, pair stream, GMRES Z-vector
  and Eq. 40 MO sectors; production keeps no dense path. The full
  `udh_molecular_audit.py` passes on the V4 grid: every open-shell fixture
  ≤ 4.3e-9 Ha/Bohr (water cation 6-31G 1.1e-9), closed-shell UKS = RKS to
  1.1e-11, RKS full-coordinate FD ≤ 1.2e-8, all nine scope refusals. Two
  findings on the way: three refusal expectations in the audit predated
  production support and aborted it; and the CN fixture had started from
  `guess hcore` on a UKS saddle (λ_min −0.0173, 9.7 mEh above the SAD-guess
  ground state), where it missed by ~8e-8 along the bond. From `guess sad`
  it passes at 7.8e-10. UKS amplitude tiles are deferred to RI-DH.

- **No dense double-hybrid gradient path in production (2026-11-10,
  `docs/UDH_GRADIENT_ONE_ARCHITECTURE.md` step 1).** The RKS DH code
  kept dense reference adapters that only tests reached: n_ao^4 Eq. 46/47
  tensors behind a `DHGammaStorage` switch, dense h/S/ERI derivative arrays,
  a direct Eq. 47 backtransform, a one-argument reference contract and a
  `mo_eri` stationary overload. All are gone (`dh_pt2_gradient.{h,cpp}` net
  -450 lines), with `build_dh_ks_fixed_density_fock_derivative`, which had no
  caller. A contract built without the direct callbacks now fails with a named
  error instead of falling back to a dense path. Tests rebuild the dense views
  from the production stream in `tests/dh_dense_oracle.h`. RKS B2PLYP gradients
  on water and C1 H2O2 are identical to every printed digit; 29 DH/UDH ctest
  entries and 99 Python contracts pass; a mutation of the Eq. 46 exchange
  factor fails the rewritten oracle comparisons.

- **A `types.h` edit no longer recompiles the CC kernel registry (2026-11-10,
  `aa51df70`).** `cc/common.h` and `cc/tensor_backend.h` included `types.h`
  (directly and through `shellpair.h`) only for `Calculator&` and
  `ShellPair` reference parameters; both now forward-declare them. The
  registry, the largest TU, was recompiled once per target on every
  `types.h` edit.

- **UMP2 no longer copies its amplitudes into the Calculator (2026-11-10,
  `bea7dd8b`, `docs/UDH_GRADIENT_ONE_ARCHITECTURE.md` step 0).**
  `apply_ump2_result` copied `t2_aa/ab/bb` into `calculator._mp2_result` on
  every UMP2 run and nothing read them: a full extra amplitude set per run,
  for every UMP2 caller. Deleted with the three `types.h` fields. The 11 UMP2
  and UKS double-hybrid regression cases pass.

- **UKS double hybrids read Eq. 40 MO sectors, not full MO tensors
  (2026-11-10, `docs/UDH_GRADIENT_ONE_ARCHITECTURE.md` step 3).** The
  UKS gradient transformed three full `n⁴` MO ERI tensors (αα, αβ, ββ).
  `build_dh_eq40_mo_eri_blocks_direct` now takes a separate frame per
  electron and builds, for each spin pair, the same five sectors RKS uses
  (ovov, vvov, ooov, ovvv, ovoo). Gradients and energies are bitwise
  identical on all six `uks_u0` fixtures, the regression case and both RKS
  inputs. By element count the MO integrals shrink 8.6× for a water cation
  in cc-pVTZ (409 → 48 MiB); the AO tensor is now the largest UKS object.

- **One Z-vector solver for RKS and UKS double hybrids (2026-11-10,
  `docs/UDH_GRADIENT_ONE_ARCHITECTURE.md` step 2).** The UKS gradient
  built its joint α/β KS Jacobian column by column (1 + 2n response actions)
  and solved it by pivoted QR. Both spins now call `solve_packed_zvector`, the
  matrix-free GMRES the RKS path already used. On C1 H2O2⁺ that is 46 actions
  instead of 119, and the count now tracks GMRES iterations rather than pair
  count. UKS gradients moved ≤ 6.5e-13 Ha/Bohr on all six `uks_u0` fixtures;
  RKS is bitwise unchanged. The solver takes `A z = b` (the old UKS solve was
  `A^T z`); the physical Hessian is self-adjoint, and the tests check that
  against a dense oracle (`tests/dh_zvector_oracle.h`), which replaces the
  deleted `DHZVectorBackend::DenseReference`.

- **UKS double-hybrid gradient restored on one streamed path (2026-11-09,
  `docs/UDH_GRADIENT_ONE_ARCHITECTURE.md`).** Every UKS DH gradient had
  failed since `d72742df` (2026-10-12), which left the dense UKS consumer reading
  derivative arrays that were no longer filled. No regression case ran the path.
  RKS and UKS now share one spin-general pair stream
  (`for_each_dh_pair_ao_block`), one two-particle source and one assembly
  (`build_dh_eq33_direct_non_xc_gradient`), and UKS keeps none of its six `n⁴`
  tensors. Fourth-order energy FD on the V4 grid: C1 H2O2⁺ 4.3e-9, HO2 1.8e-9,
  asymmetric water cation 9.1e-10 Ha/Bohr; translation ≤ 3.8e-13. RKS gradients
  are bitwise unchanged. Closed-shell `scf_type uhf` is within 2.4e-11 Ha/Bohr of
  RKS. Four mutations fail the contract test. Gated by
  `h2o2_cation_uks_b2plyp_v4_gradient_sto3g`.

- **Open decisions and full-symmetry gates settled (2026-11-09).** Three
  changes, core suite 137/137:
  - Yukawa and Gaussian screened exchange are refused by name at functional
    setup, so no workflow evaluates the erf kernel under their name. PySCF does
    make that substitution, which is why agreeing with it proved nothing. Gated
    by `water_{scf,tddft}_camy_b3lyp_rejected_sto3g`.
  - The default guess stays `hcore`. Both binaries warn when a UHF or UKS run
    with multiplicity above 1 starts from it without `stability_check`
    (`SCF::warn_open_shell_hcore_guess`). Checked on one UHF and one UKS case,
    and absent on a case that runs the stability check.
  - `RYS_MAX_ROOTS` is 13, not 11. (ii|ii) needs 13 roots, so any quartet of
    total L ≥ 22 indexed past the root table and aborted on the default engine.
    Only L = 6 reaches it. `planck-rys-roots` covers n = 12 and 13 with the
    existing bounds (moment error ≤ 1.3e-13). This is what let the i-shell
    cross-code check run: Ne/cc-pV6Z equals PySCF to every printed digit.
  - New RHF/STO-3G gates at D3h, Oh, D∞h, C∞v and Kh match PySCF to ≤ 1.3e-9 Eh.
    The full-group ERI reduction is active for D3h and Oh only. D∞h, C∞v and Kh
    detect the group and converge without it.
  - `test_dh_o5_backtransform.py` had been stale since `d72742df` and nothing ran
    it. Its assertions now match the tiled streams, and ctest runs it as
    `planck-dh-o5-backtransform-contract`.

- **PCM geometry optimization and frequencies (2026-11-08,
  `docs/PCM_GEOMETRY.md` step 5; the scope is now an answer).** The
  HF geomopt step, every Hessian displacement and the post-optimization
  symmetry SCF each build the cavity at their own geometry, pass it to the
  SCF, and add `Gradient::pcm_gradient`. Before this, they ran without PCM
  and nothing reported it. planck-dft already rebuilt the cavity in
  `prepare_current_geometry`. Against PySCF 2.13.0 (`geometric_solver`
  gau_tight, analytic PCM Hessian): water RHF/6-31G* E_opt 1.3e-9 Eh,
  frequencies within 0.31 cm⁻¹; water RKS PBE E_opt to every printed digit
  (`water_rhf_pcm_swig_geomoptfreq_631gd`,
  `water_rks_pbe_pcm_swig_geomopt_631gd`). Removing the cavity from any of
  the three loops, or the PCM term from the geomopt gradient, fails the
  gate. Imaginary-mode following under PCM stays rejected (not gated).

- **Analytic PCM gradients (2026-11-08, `docs/PCM_GEOMETRY.md`
  step 4).** `calculation gradient` with C-PCM runs for RHF, UHF, RKS and
  UKS, Cartesian and spherical. `Gradient::pcm_gradient` re-solves the
  charges at the converged density and adds three terms. `grad_nuc` and
  `grad_solver` (the cavity terms, through the tessera positions and the
  SWIG switching F_i) live in `pcm.cpp`. `grad_qv` is a shell-block
  derivative kernel, `_compute_electronic_potential_gradient`, that reuses
  the point-charge VRR with 2α/2β-weighted slices and takes the
  charge-centre term from translation invariance. Each term matches FD
  (`planck-pcm-cavity-gradient-fd`, `planck-pcm-qv-gradient-fd`), and each
  matches PySCF 2.13.0's `grad_nuc`/`grad_qv`/`grad_solver` to all ten
  printed digits on C1 H2O2. Totals match PySCF to every printed digit on
  four cases (`*_pcm_swig_gradient_*`; DFT 1.6e-11), and a five-point FD of
  the SCF energy to 4.2e-8 Ha/Bohr. Dropping any one term fails every
  gate. Cost: 0.46 s of a 33 s anthracene/6-31G* run. Geomopt and
  frequencies followed the same day (step 5, entry above). Found on the way:
  `planck-esp-points` had failed since the SWIG cavity, because its
  hand-built tesserae left ξ = 0; fixed in the test.

- **Spherical PCM single points (2026-11-08, `docs/PCM_GEOMETRY.md`
  step 6).** `evaluate_pcm_reaction_field` lifts the working density to
  Cartesian for φ_el and returns V_rxn in the working basis, so every
  caller (HF SCF, DFT SCF, ESP reaction field) works unchanged. The
  refusals in both binaries are removed. Water RHF, water⁺ UHF and water
  RKS PBE/6-31G* match PySCF 2.13.0 `cart=False` C-PCM to every printed
  digit (`*_pcm_swig_spherical_631gd`). The Cartesian path is unchanged:
  both lifts are the identity there.

- **Smooth SWIG PCM cavity (2026-11-07, `docs/PCM_GEOMETRY.md` step
  1).** The cavity used Fibonacci points with a hard bury test, so the
  energy jumped as points crossed sphere boundaries. It is now SWIG, as in
  PySCF's C-PCM: Lebedev points, a smooth switching function, and Gaussian
  surface charges. `_os_nuclear_primitive` takes the Gaussian exponent; point
  charges print identical output on eight non-PCM inputs. Three things a
  user sees change:
  - f(ε) is (ε−1)/ε (the old factor was COSMO's);
  - H uses the modified Bondi radius 1.1 Å;
  - `surface_points_per_atom` is a Lebedev size, default 302.

  Four new 6-31G* gates (water RHF, C1 H2O2 RHF, water⁺ UHF, water RKS PBE)
  match PySCF 2.13.0 to every printed digit. An O–H stretch gate
  (`water_rhf_pcm_smoothness_sto3g`) holds the fourth difference of E to
  2.3e-9; the old cavity fails it at 4.4e-5. The five existing PCM cases
  were re-pinned. The orientation dependence (5.2e-6 Eh at 302 points, the
  same as PySCF) is accepted: `docs/PCM_ROTATIONAL_INVARIANCE.md`.

  Step 2 followed: the cavity stores no AO matrices. S is factored once,
  and φ_el and V_rxn are one fused integral sweep each per SCF iteration.
  It trades time for memory. With direct SCF and the two binaries run back
  to back: naphthalene/6-31G* went from 14.3 s and 0.63 GB to 30.1 s and
  0.13 GB, and anthracene from 37.4 s and 1.40 GB to 76.6 s and 0.20 GB,
  with the same energies and iteration counts. (A first report of
  157 → 118 s came from a conventional run whose 6 GB ERI tensor put this
  8 GiB machine under memory pressure. It was wrong.)

  Step 3 won the time back. Point-charge integrals (V_ne, V_rxn, ESP, φ_el)
  are now built per shell pair: one VRR per primitive pair and charge,
  contracted over primitives (and charges), one HRR per component pair, and
  batched Boys. φ_el is dynamically scheduled. Per iteration, φ_el is
  2.6–2.8× faster and V_rxn 4.8–5.3× faster. Back to back, naphthalene
  and anthracene PCM runs take 13.7–13.9 s and 35.8–35.9 s, against
  13.5–13.7 s and 35.2–35.3 s for the stored-matrix design, with 5–7× less
  memory. Non-PCM output is unchanged to printed digits, apart from signed
  zeros and two basin selections. HI⁺'s degenerate π hole picks a different
  orientation at the same energy. `ethylene_rhf_stability_unstable` was
  exact D2d twisted ethylene, whose basin roundoff chose (in PySCF too).
  Its input now pyramidalizes one CH₂ by 0.20 Å and uses SAD, and is
  re-pinned to −77.3844126695, matching PySCF to every printed digit.
  Core 132/132; extended 215/219, the failures being the four known RCCSDT
  routing cases.

- **Test infrastructure: live asserts, no Eigen tests, CI builds the tests
  (2026-11-07).** Three fixes in `CMakeLists.txt` and
  `.github/workflows/cmake-multi-platform.yml`:
  - Release's `-DNDEBUG` compiled every `assert()` in a test out, so those
    gates passed vacuously. Seen on `cc_spatial_amplitude_projection`, whose
    fixture bug surfaced only under `-UNDEBUG`. Every `tests/*.cpp` source now
    gets `-UNDEBUG` as a source-file property. The production sources linked
    into the test binaries keep `-DNDEBUG`. Only three tests use `assert()`
    (`cc_amplitude_checkpoint`, `cc_amplitude_sector_seed`,
    `cc_spatial_amplitude_projection`); the rest use their own checks.
  - Eigen reads Planck's own `BUILD_TESTING` switch, so
    `fetchcontent_makeavailable(eigen)` registered 109 Eigen tests that
    nothing built (14 failed in ~0 s). A normal variable now shadows the
    switch around the fetch. `ctest -N` lists 94 tests, none from Eigen.
  - CI built only `--target hartree-fock`, so no `planck-*` test binary was
    built before `ctest`. It now builds every target. The `-j$(nprocs)` typo
    (not a command, so `-j` got no value) is now `-j$(nproc)`.

  - The first full build found three test targets that had never linked,
    because CI never built them. `planck-multipole-integrals` and
    `planck-population-analysis` lacked `spherical_recurrence.cpp`.
    `planck-working-state-rebuild` broke when `project_checkpoint_density`
    moved into `working_state.cpp`; the function now lives, unchanged, in
    `src/scf/checkpoint_projection.cpp`.
  - Tests read basis sets from the install prefix, which CI never populates
    and an older install can lack (27 def2 cases failed on it locally). Every
    CTest test now gets `BASIS_PATH=<source>/basis-sets`, appended to its
    environment.

  Verification (2026-11-07, `build-full`, PR #181): 92 tests. Every unit
  test passes with live asserts. Regression core 126/126, smoke 54/56,
  extended 202/206; the failures are only the four known RCCSDT routing
  cases (`h2`/`lih`/`water`/`ch4_rccsdt_sto3g`). CI has not run yet (#181 is
  stacked on #180).

- **Result-changing environment overrides removed; HGP/Rys/auto ERI builds
  striped under MPI (2026-11-07, PR #181).**
  - E2/E3 (`docs/RUNTIME_ENV_OVERRIDES.md`, now an answer):
    - `PLANCK_RCCSDT_BACKEND` is the input keyword
      `rccsdt_backend auto|determinant|tensor|optimized`, and
      `ch4_rccsdt_generated_sto3g` has its own input.
    - The `PLANCK_DFT_DH_HESSIAN_PROBE_LOG` swap probe, which returned a
      different DH gradient, is deleted, as are the `PLANCK_CASSCF_CALL_PROBE`
      counters.
    - Every remaining `getenv` in `src/` only changes printed output.
  - MPI (`docs/HPC_MPI.md`):
    - The conventional ERI tensor builds for HGP, Rys and **`auto` (the
      default engine)** ran in full on every rank. They now stripe bras
      across ranks and merge with one allreduce, as OS does.
    - Gated by water/6-31G* conventional cases per engine, exact against
      serial at 2 and 3 ranks. The exact energy cannot tell the stripe from
      plain replication, so it was checked by mutation: without the reduce,
      2 ranks give -137.47 Eh against -76.01.
    - The scale gate `water16_rhf_631g_auto_direct_mpi_scale` (nb=208, direct
      SCF): serial, 2 and 4 ranks all give -1215.7179682094 Eh; 11.7 s for
      all three runs on a laptop, 73 MB serial peak.

- **RCCSD doubles equation was missing its F_me terms (2026-11-07).**
  Stanton-Gauss T2 uses `F_be − ½ t_m^b F_me` and `F_mj + ½ t_j^e F_me`;
  `build_residuals` (`src/post_hf/cc/ccsd.cpp`) used bare `F_be`/`F_mj`.
  Canonical RHF has f_ov = 0, but `F_me = t_n^f <mn||ef>` is not, so the
  error is O(t1² t2): small, growing with system size, and invisible to the
  gates because they were pinned to Planck's own output. Fixed there and in
  the identical RCCSDT stage-1 warm start (`tensor_backend.cpp`; the generated
  RCCSDT route never called it). Now every printed digit of PySCF: water/6-31G
  (was 5.4e-7 off), HI/def2-SVP with ECP (7.4e-8, new gate
  `hi_rccsd_def2svp_ecp`); `water_spherical_ccsd_sto3g` (6.2e-7) and
  `be_ccsd_spherical_631gd` (1.4e-6, tightened to 1e-9) re-pinned to PySCF.
  It also converges faster (water/6-31G 22 → 15 iterations). Found by the
  def2 post-HF gates; the "HBr 1.1e-6" gap recorded then was this bug, not
  an ECP effect.

- **def2 / ECP work complete (2026-11-07, `docs/DEF2_ECP.md`).** Closing
  items: HI CAS(2,2) gate; the ZnCl₂/AgCl ~1e-9 offsets were Planck's default
  `tol_eri` 1e-10 screening (at 1e-12 every printed digit of PySCF; both
  inputs tightened, gates 2e-9 → 5e-10); `planck-compute-2e` goldens
  re-pinned after the 2026-11-01 Å→Bohr fix (the one-centre element did not
  move); `planck-population-analysis` linked `spherical.cpp`. Planck ctest
  (`-R ^planck`): 94 tests, failures only the four known RCCSDT routing
  cases. Open: the libecpint repin when PR #81 merges.

- **HF RHF → UHF stability channel fixed (2026-11-05,
  `docs/KS_STABILITY.md` R0).** The "external triplet" channel built
  A^T − B^T, which is the complex-orbital test again, so the triplet row
  repeated the complex row and stretched H₂/STO-3G was reported stable
  (+0.0535) where A^T + B^T is −0.5109 (PySCF `rhf_external` −0.51090553).
  Now A^T + B^T; the follow reaches PySCF's broken-symmetry UHF to every digit
  (−0.9338672048 at 2.5 Å). Water's triplet value is now PySCF's +0.3988887.
  Gated by `h2_rhf_triplet_{instability,follow}_sto3g` and tightened
  `water_rhf_stability_stable`, all failing on the old algebra.

- **Spherical HF SOSCF and stability gated (2026-11-05,
  `docs/SPHERICAL_DFT.md` S0.6).** RHF SOSCF's CPHF Hessian indexed the
  spherical ERI tensor with Cartesian strides. |g| grew and every
  augmented-Hessian solve failed, and only the DIIS handback converged it. UHF
  SOSCF passed spherical densities to the Cartesian Fock build. Both are now in
  the working basis and converge superlinearly. Gated with the spherical
  stability fix from `100675e4` by `water_rhf_stability_spherical_631gd`,
  `water_cation_uhf_stability_follow_spherical_631gd` and
  `water_{rhf,cation_uhf}_soscf_spherical_631gd` (PySCF `cart=False`), all
  failing on the old code.

- **DFT with ECPs (2026-11-06, `docs/DEF2_ECP.md` step 6).** Lifting
  the refusal was the only code change: H_core, the electron counts and
  `one_electron_gradient` were already ECP-aware. HI RKS PBE energy and HI
  B3LYP TDDFT match PySCF `cart=False` to every printed digit, the C1 AuH·H₂O
  B3LYP gradient to 2.4e-10 Ha/Bohr (fourth-order energy FD 1.6e-8), and HI⁺
  UKS stability gives the π-hole zero mode (+2.3e-9 vs PySCF −1.4e-8), which
  closes the KS stability work (`docs/KS_STABILITY.md`). Found on the way: a
  PySCF defect in its DFT gradient grid response on ECP atoms
  (`docs/PYSCF_ECP_GRID_RESPONSE_SCOPE.md`, see Open Work).

- **Spherical DFT complete: double-hybrid single points and checkpoint restart
  (2026-11-06, `docs/SPHERICAL_DFT.md`, now an answer doc).** Double-hybrid
  single points needed only the refusal narrowed (PT2 is the HF MP2 kernel on
  the spherical ERI tensor): water RKS and water⁺ UKS B2PLYP/6-31G* match
  PySCF `cart=False` to every printed digit; DH gradients stay refused by name.
  Checkpoint restart: the DFT and HF cross-basis projections were two copies,
  and the DFT one built its cross overlap in Cartesian against a working-basis
  orthogonalizer, lacked the ECP-core check, and dropped the beta part of a UHF
  checkpoint loaded into RKS. Both drivers now call one
  `SCF::project_checkpoint_density` (`src/scf/working_state.cpp`). Spherical
  projected and same-basis restarts reach the fresh-run energy to 1e-8, and a
  spherical restart from a Cartesian checkpoint is refused by name. The
  regression runner's seeding assumed `.hfchk` and now follows the executable
  (`planck-dft` writes `.dftchk`). A stale smoke case
  (`water_rks_spherical_gradient_rejected`, failing since step 2) was removed.

- **Spherical TDDFT, and an analytic TDDFT XC kernel (2026-11-06,
  `docs/SPHERICAL_DFT.md` step 3).** `run_linear_response` lifts the MO
  coefficients (TᵀC) and densities once and builds the Coulomb ERIs in
  Cartesian; `n_virt` now comes from the MO count (spherical TDDFT was not
  refused before and read past the MO matrix). The TDDFT XC kernel was a
  central difference of V_xc; on libxc's discontinuous wPBEh (HSE) potential
  it shifted spherical water/6-31G* HSE06 roots by up to 4.4e-4 eV. It now
  applies the analytic f_xc (`prepare_uks_xc_kernel`) and the FD helper is
  deleted. Water RKS B3LYP singlet/triplet, HSE06 and UKS B3LYP match PySCF
  `cart=False` to 1e-7 eV in every root; the 13 Cartesian TDDFT cases are
  unchanged.

- **Spherical KS gradients, geometry optimization and frequencies
  (2026-11-06, `docs/SPHERICAL_DFT.md` step 2).** One change in
  `compute_analytic_ks_gradient`: the stored densities are lifted with
  `to_cartesian` before the XC nuclear gradient (the HF-like terms and the XC
  evaluation already lifted). Against PySCF `cart=False` with grid response on
  C1 H₂O₂/6-31G*: RKS B3LYP 1.7e-10, RKS HSE06 1.2e-10, UKS PBE 3.4e-10
  Ha/Bohr; UKS energy FD 1.2e-9. Water B3LYP geomopt E_opt 3.2e-7 above
  PySCF's gau_tight minimum (Planck's default 3e-4 force threshold),
  frequencies within 1.6 cm⁻¹. A 3.2e-8 UKS gap first blamed on Planck was
  PySCF's default `conv_tol_grad` (√conv_tol). New runner check
  `gradient_close` (every component). An HSE06 energy-FD miss that surfaced
  on the way is libxc's HSE06 `s_scaling_2` discontinuity at s = 1, not
  Planck (see Open Work).

- **Spherical KS stability and SOSCF (2026-11-06,
  `docs/SPHERICAL_DFT.md` step 4).** The two orbital-Hessian builders
  (`src/dft/ks_orbital_hessian.cpp`), shared by SOSCF and stability, take a
  `basis` pointer and lift the ground density and the MO coefficients
  (Cᵀ C_mo) once; the trial densities and the MO packing are then exact in
  Cartesian with no change to the operator bodies. The SOSCF and stability
  refusals are gone. λ_min matches PySCF `cart=False` to ≤ 1.2e-7 on water
  RKS B3LYP and water⁺ UKS PBE (cc-pVTZ) and H₂O₂⁺ UKS PBE (6-31G*); the
  stretched-H₂ cc-pVDZ triplet follow lands on PySCF's broken-symmetry UKS
  to every digit (−0.9997290788). SOSCF reaches the DIIS energy with the same
  gradient decay as the Cartesian control. Cartesian SOSCF/stability cases
  unchanged. Six cases (`*_stability_spherical_*`, `*_soscf_spherical_631gd`,
  `h2_rks_pbe_triplet_follow_spherical_ccpvdz`) and runner metric
  `stability_uks_internal`; references from
  `tests/pyscf/spherical_ks_stability_references.py`.

- **RKS stability analysis and following in `planck-dft` (2026-11-05,
  `docs/KS_STABILITY.md` S2).** Both RKS channels, `RKS -> RKS` (singlet)
  and `RKS -> UKS` (triplet), from one prepared UKS operator at the
  closed-shell reference, on [κ, κ] and [κ, −κ]. The triplet follow promotes to
  UKS through `promote_rhf_to_uhf_along_triplet_mode` (split out of the HF
  follow). After the promotion the run re-initializes polarized functionals and
  re-validates the workflow. Stretched H₂/STO-3G PBE and B3LYP: triplet −0.1448
  / −0.2231 (PySCF ≤ 2e-5, the grid floor), and the follow reaches PySCF's
  broken-symmetry UKS. Water stays stable in both channels (B3LYP, HSE06). PySCF
  `gen_g_hop_rhf` is 2× Planck's singlet convention, not 4×. Gated by
  `water_rks_hse06_stability_631gd` (replaces `water_rks_stability_rejected`)
  and `h2_rks_{pbe,b3lyp}_triplet_follow_sto3g`; a mutation (triplet built as
  singlet) fails all three.

- **Stability sweep over open-shell regression inputs; spherical HF stability
  and ECP electron counts fixed (2026-11-05, `docs/KS_STABILITY.md`
  S1).** 38 `scf_type uhf` inputs re-run with `stability_check`. One saddle the
  suite did not intend: `h2o2_cation_uhf_tight_tol_sto3g`, λ −0.0738, same as
  PySCF. Two defects in the check:
  - **Spherical HF stability** sized the ERI transform with the Cartesian
    `nbasis()`. Li failed, water⁺ reported λ −15 [UNSTABLE], HI⁺ segfaulted.
    Now `working_nbasis()`; water⁺ matches PySCF to 1e-8.
  - **Eleven Σ Z electron counts** that ignored ECP cores (and ghosts), in
    stability, RHF response, MP2, CC, FCI, CASSCF, TDDFT and the KS scaffold.
    **HI/def2-SVP RMP2 was 24× too small with no warning** (E_corr −0.00598 vs
    PySCF −0.14342); it now matches. All use `total_nuclear_charge()`.

- **UKS stability analysis and following in `planck-dft` (2026-11-05,
  `docs/KS_STABILITY.md`).** `stability_check` / `stability_follow` now
  work for UKS: the lowest eigenvalue of the spin-conserving A + B by Davidson
  on the UKS SOSCF operator (lifted into `build_uks_orbital_hessian_op`), and a
  follow through `rotate_along_internal_mode` (split out of the HF follow).
  Water⁺/cc-pVTZ UKS HSE06 from `hcore` lands on the ²A₁ saddle; the check
  reports λ_min −0.06385 (PySCF `gen_g_hop_uhf` −0.06381) and the follow
  reaches the ²B₁ ground state, −75.9126471254, re-checked stable. Cost cut
  172 → 14 s by caching Davidson products (the solver recomputed every basis
  vector each iteration), a 1e-4 residual, and a prepared polarized XC kernel
  (`UKSXCKernel`, GEMM point blocks; also used by the UKS DH response). Gated
  by `water_cation_uks_hse06_stability_{ccpvtz,631gd}` and
  `water_rks_stability_rejected` (since replaced by the RKS gates); UKS SOSCF bitwise through the builder before
  the kernel change.

- **Spherical prerequisites S0.3–S0.5 (2026-11-04, `docs/SPHERICAL_DFT.md`).**
  One pair of shape-checked `BasisFunctions::to_cartesian` / `to_working`
  helpers replaces seven hand-written C·M·Cᵀ copies (S0.3); the RKS/UKS
  gradients lift P and W like the HF ones (S0.4, Cartesian byte-identical);
  and C is row-normalized once in `read_gbs_basis` from the closed-form
  same-centre overlap instead of by three copies that built S_cart at every
  geometry (S0.5). The per-geometry premise was false: C is block-diagonal by
  shell, so its normalization is geometry-independent (renormalized C
  byte-identical at two water/cc-pVTZ geometries). Spherical outputs move at
  roundoff only.

- **Spherical post-HF after a direct SCF no longer crashes (2026-11-04,
  `docs/SPHERICAL_DFT.md` S0.2).** `Correlation::ensure_eri` returned
  an empty tensor in spherical mode when the SCF had cached none, and its
  ~20 callers index it unchecked: `scf_mode direct` + `basis_type spherical`
  + RMP2 died with SIGSEGV. It now builds the Cartesian tensor and
  transforms it to spherical. Gated by
  `{water_rmp2,h2o_cation_ump2}_spherical_631gd_direct` (same energies as
  the conventional cases).

- **Spherical-basis ESP fixed (2026-11-04, `docs/SPHERICAL_DFT.md`
  S0.1).** `SCF::log_esp_report` handed the spherical (working-basis) density
  to Cartesian ESP integrals; `electrostatic_potential` only checked
  squareness, so the wrong AO elements were read with no error (water/6-31G*
  HF Connolly charges O −2.94, H 1.461/1.474; now O −0.811). The density is
  lifted first, and `electrostatic_potential` now refuses a density whose
  size differs from the AO count of its shell pairs. Gated by
  `water_rhf_spherical_esp_points_631gd` (PySCF `cart=False` `int1e_grids`,
  ≤1.8e-9), which fails on the old code.

- **ECP gradients, geometry optimization and frequencies (2026-11-04, branch
  `feat/def2-ecp`, `docs/DEF2_ECP.md` step 5).**
  `ECP::_compute_ecp_gradient` contracts libecpint's shell-pair derivatives
  with the density; `one_electron_gradient` adds it inside `core()`, so
  RHF/UHF/ROHF (and the RKS/UKS/MP2/DH paths once those accept ECPs) inherit
  it with no per-caller code. Stock libecpint's derivatives miss FD by 4.6e-5
  relative on HI/def2-QZVP; libecpint is pinned to unmerged upstream PR #81,
  which passes (see Open Work for the repin). PySCF gates: HI and C1 AuH·H2O
  RHF gradients (≤3e-8 Ha/Bohr), HI⁺ UHF/ROHF (every printed digit), HI
  geomopt (E_opt 1.6e-9 Eh) and frequency (0.03 cm⁻¹); plus
  `planck-ecp-gradient-fd` (ctest) and an AuH·H2O energy-FD case. Found on
  the way: four FD drivers kept a stale Å→Bohr literal (fixed).

- **Checkpoint and FCIDUMP with ECPs (2026-11-01, branch `feat/def2-ecp`,
  `docs/DEF2_ECP.md` step 4b).** Checkpoint version 8 stores each
  atom's ECP core count; a load or cross-basis projection between different
  cores is refused (v7 files read as all-electron, which is exact). FCIDUMP
  needed no change: PySCF reproduces its ECP RHF energy from the file to
  2e-12. Found on the way: (1) a failed cross-basis projection in any
  spherical run kept the Cartesian S/H_core and diverged; the fallback now
  rebuilds them; (2) six Σ Z electron counts the step-2 audit missed (HF MO
  printout, four in the DFT driver) now use `total_nuclear_charge()`;
  (3) `chkdump` reports valence electrons and fchk Nuclear charges
  Z − n_core. Gates: `hi_rhf_def2svp_ecp_{fcidump,projection_refused,
  restart_full}`.

- **ECP integrals in H_core (2026-11-01, branch `feat/def2-ecp`,
  `docs/DEF2_ECP.md` step 4).** def2 bases from Rb onward run
  single points through libecpint (MIT, `ExternalProject`).
  `_compute_potential` (`src/integrals/base.h`) is V_ne + U_ECP at every
  H_core site; SAD runs ECP atoms with Z − n_core electrons. libecpint's
  code generator wrote its type-2 angular prefactors at 6 digits
  (157.914 for 16π²), putting every off-centre semilocal integral 2.09e-6
  too large; patched at fetch (since 2026-11-04 carried by the pinned upstream PR #81 instead). Found by a
  brute-force radial × Lebedev integral that agreed with PySCF to 12
  digits. Five PySCF gates (HI SVP/TZVP+SAD+symmetry, AgCl, PdCO,
  AuH·H2O): 0 to 1.1e-9 Eh, matrices to ≤1.3e-9. Gradients (step 5) and
  DFT (step 6) refuse ECP bases.

- **Å→Bohr conversion constant fixed (2026-11-01, branch `feat/def2-ecp`).**
  `src/base/tables.h` paired `BOHR_TO_ANGSTROM = 0.529177210903` (CODATA
  2018) with `ANGSTROM_TO_BOHR = 1.8897259886`, an older literal 7.2e-8
  smaller than its reciprocal, so every Ångström input ran at a geometry
  7.2e-8 too small and Å → Bohr → Å round trips drifted. Now
  `ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM`; two other literal copies
  (`types.h`, two RI tests) use the constants. Found because the new def2
  gates missed PySCF by up to 1.3e-8 while E_nn on water was 6.6e-7 off;
  after the fix four of the five match PySCF to every printed digit
  (`docs/DEF2_ECP.md` step 1b). 26 extended cases pinned to Planck's
  own output moved 2e-9..1.9e-7 Eh and were re-pinned; water SA-2 CASSCF
  moved from 3.6e-8 to 1.1e-8 off PySCF's SAD-start minimum. Ten core
  cases moved too and were missed until step 3 (only the extended suite
  was run): water/STO-3G energies (2.6e-8) and printed ESP charges and
  values in the 8th decimal; the Mayer O–H bond order moved toward PySCF
  (0.76017799 → 0.76017801, PySCF 0.76017810). Run `--suite all` after a
  change like this, not one tag. Rebuild `planck-mpi` with the other
  binaries, or the MPI smoke cases compare old against new.

- **Converged SCF orbitals come from the true Fock, not the DIIS
  extrapolation (2026-10-27).** RHF, UHF, RKS and UKS stored the eigenpairs of
  the DIIS-extrapolated Fock at convergence. These usually agree with the true
  Fock to the DIIS residual, but when every error vector is ~0 (H2/STO-3G: any
  symmetric density commutes with F) the DIIS weights are arbitrary and the
  extrapolated matrix carries other iterations' eigenvalues with the same
  eigenvectors. The gradient's energy-weighted density W is built from those
  energies: in the H2 geometry optimization, the step-1 1e/Pulay term was off
  by 1.6e-3 Ha/Bohr and the line search failed. It was hidden whenever the
  error came out exactly 0, and surfaced when G2's new summation order moved
  the geometry by 1e-16 (G1 with `engine hgp` failed the same way). Now each
  path re-diagonalizes the true Fock once at convergence when DIIS
  extrapolated, and rebuilds the density from those orbitals, as PySCF does
  (UHF already did this for level shift; UKS only for the DH PT2 handoff).
  ROHF was unaffected (its stored energies are diag(Cᵀ F_α C) and its W is
  built from Fock matrices). Gated by `h2_optfreq_smoke`; extended suite
  135/139 (the known rccsdt cases).

- **2D Rys block kernel in Auto's Fock build (K2.1b/K2.1c, 2026-10-26).**
  `RysBlockKernel` (`src/integrals/rys_block_eri.h`) is libcint's
  formulation: per root, a 1D VRR and a per-axis 1D HRR to
  I_q(a,b,c,d), then each component as sum_r w_r I_x I_y I_z. The old
  per-component path summed the roots into a 6D box before its HRR (the
  hoisted block was 50-210x libcint). On the K = 1, L >= 4 classes of
  2 H2O cc-pVTZ it is 2.00x the HGP block (every class faster; libcint
  still 2.3-2.6x faster). The one change that paid in tuning was caching
  component offsets per class (1.0 us of a 5 us (dd|dd)); four layout
  variants lost. HGP wins under contraction (it contracts before its
  HRR), so `auto_block_eri.h` chains closed forms (L <= 3), Rys2D in
  crossover mode (K = 1 at L >= 4, K <= 3 at L >= 7), then the HGP block,
  defined once for rys.cpp, the tests and the benchmark. RHF direct per
  iteration: cc-pVTZ 1.11-1.16x, cc-pVDZ 1.01-1.04x, 6-31G* neutral,
  energies and iteration counts unchanged. Gated by
  `planck-hgp-engine-smoke` (Rys2D vs per-component HGP <= 6.4e-15 on
  water 6-31G..cc-pVTZ; four mutations caught, the HRR sign only from
  cc-pVDZ up, since 6-31G* H has no p) and `planck-fock-accumulate`
  (thread-count invariance now includes cc-pVTZ). Core 80/80, smoke
  43/45, extended 135/139 (known rccsdt cases). Record:
  `docs/ERI_KERNEL_VS_LIBCINT.md`.

- **Fast Rys roots and weights (K2.1a, 2026-10-25).** `rys_roots_weights`
  is now Chebyshev segment fits below T = 35 + 7n and the Laguerre(−½)
  asymptote above, built per n on first use from a new reference generator
  (discretized Stieltjes with compensated sums and Christoffel weights,
  `rys_roots_weights_reference`). It replaces the moment-based
  Golub–Welsch solver, whose nodes were off by up to 8e-3 at n = 11 even
  after the recursion fix (moment problems are ill-conditioned; the rule
  still reproduced moments, which is why integrals survived). Found along
  the way: `long double` is `double` on Apple silicon, so nothing had
  extended precision; naive sums over the 1200-point discretization left
  1e-14 of noise until compensated. 32–43 ns per call for every n, against
  0.46–3.04 µs; per-component Rys 2.1–3.2× faster (OS/HGP control
  0.94–1.02×); end to end with `engine rys`, identical energies,
  ethylene cc-pVDZ 2.22× and 2 H2O cc-pVTZ 3.48× (241 → 69 s). The default
  engine is unaffected by design (4 H2O cc-pVDZ `engine auto` 1.00×). Gated by
  `planck-rys-roots` (fast vs reference ≤ 6.3e-15 for n ≤ 5, 4.2e-14 at
  n = 11; four mutations caught) and `planck-rys-box-invariance`, which
  the first, noisier generator failed. `planck-auto-dispatch-benchmark`
  segfaulted at startup (no Boys loader, like nine others earlier); fixed.
  Extended suite 135/139 (the known rccsdt cases). Record:
  `docs/ERI_KERNEL_VS_LIBCINT.md` §4.

- **Rys roots and weights fixed at T >= 1 and at small T (2026-10-25).**
  Found while scoping K2.1a. `_boys_moment` (`rys_roots.cpp`) built the
  moments by upward recursion from T = 1, which is stable only for T >= m;
  at T in [1, ~5] the F_k up to k = 21 lost every digit. Rules for n >= 7
  roots then had weights that did not sum to F_0 (up to 413 of 300k
  sampled T at n = 11, moment errors to 1.6e2), and n = 4-6 were degraded
  to 5e-12 - 4.6e-8. Fixed with the positive-term series below T = m. The
  closed-form one-root rule (`rys_1pt`) also cancelled at small T (1.3e-10
  relative at T = 2e-6); it now uses the same series below T = 1. Exposure:
  explicit `engine rys` everywhere (~1e-10 integrals on d shells), and the
  Auto Rys buckets (L_AB, L_CD) = (7, 8) / (8, 8) on molecules, i.e. g-shell
  quartets in cc-pVQZ and larger, where the old rules were badly wrong. It
  went unseen because nothing tested the rule against an independent
  reference: the Rys regression cases stop at n = 5, the high-L Rys case
  is a single Ne atom (T = 0 for every quartet), and the box-invariance
  test compares Rys with Rys. New gate `planck-rys-roots`: every rule
  n = 1..11 on T in [0, 2000] (dense over [0.5, 6]) must reproduce the
  moments from long-double quadrature (4e-14 for n <= 6, up to 4.8e-11 at
  n = 11, the moment method's conditioning); the old solver fails it from
  n = 1.

- **Redundant primitives dropped at basis-read time (K1, 2026-10-25).**
  PySCF's cc-pVXZ files use the optimized general contraction: an outer
  primitive that is also its own uncontracted shell of the same L is left out
  of the contraction. The BSE GBS files Planck reads keep it, so Planck ran
  28 primitives per water in cc-pVDZ against PySCF's 24 (42 against 34 in
  cc-pVTZ) for the same span. `drop_redundant_primitives`
  (`src/basis/gaussian.cpp`) now removes them for every orbital basis; Pople
  sets are unchanged and RI auxiliary bases are not touched. Fock build
  2.05–2.34× faster on cc-pVDZ water chains and 1.47–1.79× on cc-pVTZ, 1 and
  4 threads, energies within the 1e-9 SCF tolerance and exactly equal to
  hand-converted files. MO coefficients and population analyses change,
  because the contracted functions do. Found by the K1 census, which also
  refuted the scope's lead hypothesis (PySCF's density prescreen removes only
  5–8 %, like Planck's) and showed B2.2 had been declined on an in-core test.
  Extended suite 135/139 (the four known rccsdt routing cases); no case
  moved. Record: `docs/ERI_KERNEL_VS_LIBCINT.md` §3.

- **Boys core v1.2.1: high-order values fixed (2026-10-25).** The kernel
  comparison against libcint (`docs/ERI_KERNEL_VS_LIBCINT.md` §2)
  found integral errors of up to 6.6e-8 (6-31G\*), 4.0e-8 (cc-pVDZ) and
  7.2e-6 (cc-pVTZ, pure f quartets), growing with L. Planck's Rys engine,
  which does not use the Boys function, matched libcint to 2e-16, while OS,
  HGP and the block kernel matched each other. The out-of-tree core returned
  high-order values with too few significant digits over part of its
  argument range. v1.2.1 has max relative error 1.45e-13 against mpmath
  (orders 0–59), gated on the macOS and Linux release builds by a new
  accuracy check. After it, K0 agrees to 3.4e-14 / 2.6e-14 / 7.2e-13, and all
  four Planck paths agree to 1e-16. Released, encrypted and served; the dev
  license and the `PLANCK_LICENSE` CI secret are v1.2.1. Not bitwise against
  v1.2.0. This bug sits under every OS/HGP/closed-form integral, so part of
  the old ~1e-8 "precision floor" against PySCF on d/f bases may have been
  this defect (`docs/INTEGRAL_PRECISION_FLOOR_SCOPE.md`, correction note).

- **`pyscf_bench.py` now compares direct with direct (2026-10-25).**
  PySCF's `RHF.get_jk` keeps the ERI tensor in memory whenever it fits,
  whatever `direct_scf` says. So every Planck-vs-PySCF timing in the harness
  had compared Planck's direct SCF with PySCF's conventional SCF (4 H2O
  cc-pVDZ, one build: 0.030 s in-core, 0.205 s direct). The harness now
  forces direct and asserts `mf._eri is None`. The matched `conv_tol_grad`
  default and a `--spherical` flag are also new.

- **Incremental direct Fock build with density-weighted block screening (S3,
  2026-10-25).** The direct RHF/UHF/ROHF loops build `G_prev + G[ΔP]`, and
  the fused loop skips a block when `Q_ab Q_cd · max|D| < tol_eri / 10`
  (Häser–Ahlrichs, both traversals; off in replication mode). Fock build
  1.19–1.33× on water chains in 6-31G\* / cc-pVDZ / cc-pVTZ, with
  |ΔE| ≤ 6e-10 against the full build and unchanged iteration counts. At
  `tol_eri` itself the energy random-walks at 1e-8 and DIIS stalls (15 → 24
  iterations); a periodic rebuild does not fix it, because the error is
  inconsistency, not accumulation. Core 80/80, extended 135/139 (the known
  rccsdt cases). Record: `docs/SCF_VS_PYSCF.md`.
  **Follow-up the same day: the screen is now dropped for the SCF tail.**
  "Unchanged iteration counts" held only at the bench's `tol_density 1e-7`.
  At the default 1e-10 the `/10` screen still random-walked (2 H2O /
  cc-pVTZ 88 iterations against 18), and planck-mpi landed 6.7e-9 from
  serial, failing `water4_rhf_ccpvdz_auto_direct_mpi_smoke`. Once the DIIS
  error is below 1e-8, `IncrementalFock::observe` (`src/scf/scf.cpp`) turns
  the screen off and forces one full rebuild. The flag is threaded through
  `fused_fock_build` and the engine direct entries as `density_screen`
  (default on, so DFT and every other caller are unchanged). Energies now
  equal the unscreened build's, serial and MPI agree bitwise, and the gain
  is 1.17–1.26× (bench tolerances) / 1.05–1.28× (default).

- **`planck-dft-coulomb-response` fixed: its t2 fixture lacked the pair
  symmetry (2026-10-25).** The two Eq. 33 full-geometry FDs (LDA, complete
  GGA) had missed by ~1e-6 since at least 2026-10-09. A channel probe put the
  whole gap in the D-AO XC-II term: `relaxed.symmetric_mo` was asymmetric
  (8.5e-5), because the synthetic t2 had no `t_ij^ab = t_ji^ba`, and the
  tiled D' builder (`7239ebeb`) gives a symmetric D' only for amplitudes that
  have it. `drho_channel` collapses its double sum assuming a symmetric
  density. Production is unaffected (real RHF amplitudes always carry the
  symmetry, and `RMP2Result`'s packed form stores only one of each pair);
  the fixture now does too, and the failure messages print FD and analytic
  values.

- **UHF stability analysis now uses the real-rotation Hessian, and following
  escapes the saddle (2026-10-25).** The UHF→UHF check combined A − B
  same-spin blocks with an A + B cross-spin block. The result was neither
  test, its lowest mode was not a descent direction, and following it fell
  back to the saddle. All blocks are now A + B. Internal follows use the same
  adaptive step as the RHF→UHF branch (π/4 when λ < −1e-2), and the follow
  returns the total energy instead of the electronic one. 4 H2O⁺ / cc-pVDZ
  from hcore: before, a saddle at −303.6695 Eh, above ROHF (−303.6871); now
  the follow re-converges at −303.7136061802, matching PySCF's hcore+follow
  (−303.7136061785) to 1.7e-9. Core suite 80/80. Note: a lower stable UHF
  minimum exists (PySCF minao: −303.7248), since stability guarantees a local
  minimum only.

- **Nuclear-attraction setup no longer zero-fills a 246 KB table per
  primitive (S1, 2026-10-25).** `_os_nuclear_primitive` cleared
  `double V[13][13][13][14]` on every primitive pair × charge × AO component,
  whether the call came from H_core, PCM or ESP. That was 2.4 s of a 5.6 s RHF
  on 12 H2O / 6-31G\*. The initializers on `V`, `V0` and the derivative's `dV0`
  are dropped: every read is of an entry written first, so the result is
  bitwise. Setup is now 0.05 s, and that run takes 3.3 s against PySCF's 6.3 s.
  Gated by identical outputs on SCF / PCM / ESP / CHELPG / spherical-gradient
  / RKS-PCM inputs and the core suite (80/80). Record:
  `docs/SCF_VS_PYSCF.md`.

- **Batched Boys calls and grow-only kernel scratch in the fused Fock loop
  (R2 `102f1d01`, `13c60826`, `0e3e074d`, 2026-10-23/24).** The closed forms
  and the HGP block kernel make one batched Boys call per block / lane group
  (core v1.2.0, now required: a v1.1.0 license fails at load naming
  v1.2.0): 1.01-1.11x at one thread, 1.03-1.11x at four, not bitwise only
  by g++ contracting one multiply-add (<= 1.1e-16). Block-kernel scratch is
  grow-only (`grow_to`), removing per-block `vector::resize` and re-zeroed
  tails (2.4-4.7 % of samples -> <= 0.1 %): bitwise, 1.03-1.04x on
  6-31G\* / cc-pVTZ, neutral elsewhere. R1 (precomputed closed-form readout)
  was measured slower and declined. Record:
  `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`.

- **The fused Fock loop is thread-count invariant (`ea353a83`,
  2026-10-24).** Partials were keyed by thread, so G differed in its last
  bits between 1 and 4 threads. They are now keyed by bin (bra mod n_bins,
  n_bins a function of nb only), summed per element in fixed bin order, and
  the bins are scheduled dynamically: G is identical at 1/2/4/8 threads
  (28 fingerprints), and the load imbalance under symmetry is recovered
  (1.06x at four threads on naphthalene). Record:
  `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`.

- **Dense digestion per block in the fused Fock loop (K1 `5fd08496`, K1.4
  `f2849647`, 2026-10-22).** Instead of running every component through the
  canonical filter, orbit dedup and up to eight scattered writes, the loop
  visits each unordered block pair once and digests the whole block, with
  three shell-level degeneracy factors and one final symmetrization. Taken
  per build when the block kernel `covers()` every total L of the basis
  (Auto, Coulomb, shells up to f) and every density is symmetric. One
  thread 1.11-1.22x, four 1.09-1.17x. K1.4 extends it to skeleton-mode (S1)
  builds: a per-build table of each op's image of each shell pair decides
  every op per block with two integer comparisons, identity-map ops get a
  sign test, and only ops fixing the block are applied per component.
  Naphthalene / 6-31G D2h 0.273 -> 0.206 s (1.32x), 1.33x at four threads.
  Gate: `check_dense_digest` (every `FusedTerm` mode, RHF/UHF, four bases,
  two tolerances, water's real C2v with the op count asserted, a counting
  kernel proving the dense path ran), 5 + 8 mutations; MPI 1/2/4 ranks
  match. Record: `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`.

- **Lanes across primitive quartets in the HGP block kernel (2b,
  2026-10-21).** `hgp_block::vrr_lanes<B>` runs the VRR for four primitive
  quartets at once in blocks with K >= 3 (scalar below). Fock build, one
  thread: cc-pVTZ 1.38x, 6-31G\* 1.23x, cc-pVDZ 1.21x, sp-only 1.05-1.07x.
  Not bitwise; block sweep <= 1.1e-14 on four bases, mutation-tested.
  Record: `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`.

- **No per-iteration skeleton rebuild on the full point-group path (F1,
  2026-10-21).** Above `threshold` the SAO-blocked direct SCF rebuilt its
  n_b⁴ skeleton every iteration and was slower than no symmetry. It now
  takes the fused direct build with the geometry's D2h-subgroup signed
  operations (an SAO density is symmetric under any subgroup): benzene /
  6-31G\* 17.7 -> 4.08 s, naphthalene / 6-31G 66.6 -> 6.35 s, energies
  equal to the persisted full-group path; the ten full-symmetry inputs at
  `threshold 1` match their default energies. Record:
  `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`.

- **`engine auto` uses HGP on the full point-group path (F2, 2026-10-20).**
  The `{os,hgp,rys}_symm.cpp` skeleton loops have no Auto mode, so the three
  dispatchers in `src/scf/scf.cpp` sent Auto to OS. `full_symmetry_engine`
  now resolves it to HGP, the faster per-component engine there: per rebuilt
  skeleton, benzene/6-31G\* 1.40 -> 1.31 s, naphthalene/6-31G 3.93 -> 3.51 s
  (Rys 6.9 s on benzene). Energies unchanged. Record:
  `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`.

- **`guess sad` works with full point-group symmetry (F4, 2026-10-20).** It
  used to abort on C3v/D6h with "initial density is not symmetry-adapted"
  (NH3/6-31G\* 8.0e-5, benzene 2.6e-3). The raw SAD density is invariant;
  the guess was not, because making it idempotent kept the top n natural
  orbitals, which is ill-posed when a degenerate set straddles the cutoff (a
  spherically averaged open-shell atom has one: O's three 2p at 1/3 beta).
  The first fix (`d0224110`) group-averaged the guess. A later attempt
  corrected the projection's metric (it used `S^{-1/2} P S^{-1/2}`, the
  Fock-like transform) and made things worse: the wrong metric had been an
  accidental tie-break, and triplet water fell into the UHF state 0.103 Eh
  above the ground state (four regression cases). Now (2026-10-23) the guess
  is the raw density scaled to the target count,
  `P = (n / Tr(P_raw S)) P_raw`, not idempotent; open shell uses the
  spin-summed raw density scaled to n_alpha and n_beta (`scale_raw_sad`,
  `src/scf/sad.cpp`). The projection and `symmetrize_density` are deleted.
  Gated by `nh3_c3v_fullsym_631gd_sad_direct` and the four
  `water_triplet_*` cases. Record:
  `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`.

- **CC amplitude checkpoint (`.ccamp`): hand-written RCCSD can now write a
  spatial sidecar, a sector-drop defect in the shipped format is fixed, and
  UCC has full write/restart support (2026-09-04/05).**
  `docs/CC_AMPLITUDE_CHECKPOINT.md` is the single answer doc for all of it
  (the three prior scope docs it replaced are deleted). PR #164,
  branch `ccamp-x5-spin-orbital-projection`.

  **X5.0/X5.1 — spin-orbital -> spatial projection.**
  `project_rccsd_amplitudes_to_spatial` (`amplitudes.{h,cpp}`) converts
  `run_rccsd`'s converged spin-orbital `t1`/`t2` to the spatial RCC layout
  the sidecar and generated arbitrary-order runtime both use. The
  closed-shell relation -- `t1` is either spin channel (identical at closed
  shell, cross-spin blocks exactly zero); `t2` is the opposite-spin
  spin-orbital block (`t2_aa = t2_ab - t2_ab.swap(a,b)` is the dependent
  same-spin combination, not independent data) -- was **numerically
  verified against this codebase's own converged BH3/STO-3G amplitudes**
  via a temporary probe, not trusted from memory. Gated by
  `tests/cc_spatial_amplitude_projection.cpp`, mutation-verified. Caught a
  real fixture bug along the way that Release's `-DNDEBUG` had made
  invisible -- see Open Work for the NDEBUG gap this surfaced project-wide.
  `run_rccsd` was worth doing because it has no generated-arbitrary-order
  sibling (the registry floor is rank 3); verified end-to-end on
  BH3/STO-3G, same energy, values read back bit-for-bit correct.

  **C0 -- a real correctness defect in the already-shipped format, fixed on
  both sides.** `save_cc_amplitudes` only ever wrote `amplitudes.by_rank`,
  never `amplitudes.sectors` -- the independent higher Sz sector blocks a
  rank->=4 amplitude set carries (e.g. `(4, "aaabaaab")` for CCSDTQ, proven
  genuinely independent by R3.1.3). A `cc4` restart therefore always seeded
  that sector at zero -- silent because it still converges (Jacobi/DIIS
  pulls the zero sector up), so no energy-only gate ever caught it. Fixed
  on the file-format side (`CCAMP_VERSION` bumped to 2, sectors appended via
  a shared `write_tensor`/`read_tensor` codec, version-1 backward
  compatibility preserved and correctly distinguished from mid-field
  truncation) and on the seed-application side
  (`seed_arbitrary_order_amplitudes` now applies sectors by `(rank, tag)`
  key, skipping rather than failing on a shape mismatch or missing
  counterpart) -- fixing only one side would have left the other silently
  inert. Also folds in a one-byte `reference_type` field for a future UCC
  sidecar, so it needs no version 3. Both halves independently
  mutation-verified.

  **C1 -- the sidecar is now validated against the live run before
  seeding.** Previously only a per-rank shape check existed; two different
  bases (or molecules) can share identical `n_occ`/`n_virt`, so a
  same-shaped stale sidecar could silently seed a wrong basin. Now compares
  `basis_name` and `n_occ`/`n_virt` at the read site, degrading to a warned
  cold-start on mismatch -- the same policy every other sidecar problem
  here uses. The validating function lives in an anonymous namespace and
  cannot be unit-tested directly; verified end-to-end on LiH/STO-3G instead
  (`ccsdt_gen`, rank 3): two independent hand-corrupted sidecars each
  correctly cold-start at the same final energy a from-scratch run reaches,
  while an untouched sidecar still warm-starts in 1 iteration.

  **C2 -- the `ccsdt_gen`->`cc4` cross-rank restart, verified end to end.**
  Ran the actual gate on Be/STO-3G at loose tolerance
  (`PLANCK_CC_ARBITRARY_LOWER_RANKS=ON`), with `cc_warm_start .false.` on
  every run to isolate the disk-restart claim from W6's in-memory
  recursion (which otherwise recurses to a converged rank-3 in memory
  regardless of whether a sidecar exists, making a "cold" comparison
  meaningless without it). Genuinely cold `cc4`: 6 iterations,
  `E_corr = -0.0518049788`. `ccsdt_gen` write (rank 3): 6 iterations,
  `E_corr = -0.0517514344`. `cc4` restart from that sidecar: **1 iteration**,
  `E_corr = -0.0517592490`, log line confirms
  `Warm-started rank 4 from CC amplitude checkpoint ... (seeded 3 rank(s),
  method 'cc3')`. No warnings during restart -- the empty-sectors-seed-
  nothing path (a rank-3 source has no sectors of its own) is a silent
  no-op exactly as reasoned, not an error. Both cold and restart runs land
  within loose-tolerance spread of the tight-tolerance reference
  `-14.4036551081` (`be_rccsdtq_sto3g`'s own committed value), confirming
  the correct basin rather than a coincidentally-plausible wrong answer.
  **X4 is formally closed.** Not added as a committed regression case (ad
  hoc `.hfinp` files, not wired into `regression_cases.json` -- see the
  REMAINING doc for what a permanent gate would need).

  **C4 -- corrected finding, not fixed.** The REMAINING doc previously said
  C4 was blocked on arbitrary-order UCC landing at all; that premise was
  wrong -- UCC already exists in the tree, complete and validated
  (`ucc2`/`ucc3`/`ucc4`, `-DPLANCK_CC_UCC=ON`). Reading `uccgen.cpp`
  directly found its own comment already correctly diagnoses the real
  blocker: `.ccamp` persistence was deliberately omitted because
  `CCAmplitudeCheckpointMeta` carries one `n_occ`/`n_virt` pair, but
  `UHFReference` needs four independent counts
  (`n_occ_alpha`/`n_occ_beta`/`n_virt_alpha`/`n_virt_beta`) -- a shape gap
  C0's `reference_type` byte does not fix (it lets a reader tell RHF from
  UHF, not represent UHF's counts). Also confirmed: UCC leaves `by_rank`
  empty entirely (`prepare_generated_ucc_state`'s own comment) and is
  sectors-only, while its sector tags already fit C0's `(rank, tag)` keying
  unchanged. This needs a version-3 header design decision, not a
  mechanical write/read follow-on -- deliberately not built.

  **C4 rescoped 2026-09-04 as U0-U3; ALL FOUR STEPS LANDED 2026-09-05.**
  `.ccamp` bumped to version 3, adding
  two independent things merged into one bump (a writer-only fix for the
  first was tried and found broken on round-trip -- see below): an
  `n_by_rank` field independent of `max_rank` (previously the reader
  derived its `by_rank` read-loop trip count from `max_rank`, which held
  for every RCC caller by construction but breaks the moment `by_rank` is
  legitimately empty -- exactly UCC's shape), and UCC's four occupation
  counts on `CCAmplitudeCheckpointMeta`, always written, defaulting to 0
  for RHF -- following the precedent this codebase already used for the
  identical problem at the state-struct level (`CanonicalRHFCCReference`
  adds the same four fields alongside `orbital_partition` rather than
  repurposing it). `save_cc_amplitudes`'s emptiness check moved to
  `by_rank.empty() && sectors.empty()`, so a sectors-only write is no
  longer rejected outright. Version-2 files load with `n_by_rank`
  defaulted to that file's own `max_rank` (not 0 -- 0 would silently
  discard every existing version-2 sidecar's `by_rank` data).

  **The doc's own U0 plan was tried first and found broken by testing, not
  by re-reading it.** A writer-only fix (accept an empty `by_rank`, switch
  `max_rank`'s write-time source) compiled clean and the write call
  succeeded, but a real save-then-load round-trip on the exact UCC shape
  failed: the reader's `by_rank` loop was still silently coupled to
  `max_rank`, corrupting on the very next field (`n_sectors`'s bytes read
  as `by_rank` data). U0 and U1 were merged into one version-3 step as a
  result -- both needed the same bump anyway.

  **A second, independent bug was found and fixed while verifying, not
  scoped.** The pre-existing version check (`version != 1 && version !=
  CCAMP_VERSION`) rejected every version strictly between 1 and current --
  harmless while only 1 and 2 existed, and silently broke every real
  version-2 file the moment `CCAMP_VERSION` became 3. Caught by the new
  hand-built version-2-file test, not by inspection. Fixed to accept the
  full `1..CCAMP_VERSION` range.

  Verified: byte-for-byte inertness on a real BH3/STO-3G RCC `.ccamp`
  (every field through `reference_type` identical, only the version number
  and the inserted 36-byte block differ, the entire `by_rank`/`sectors`
  payload byte-identical); the UCC-shaped round-trip itself (empty
  `by_rank`, two sectors, non-zero UHF counts, all bytewise-equal after
  round-trip); three-tier version compatibility (1, 2, 3) in one test
  binary; every new assertion mutation-verified (reverting the emptiness
  check, the version-range check, or the `n_by_rank` default each makes
  exactly the test built to catch it fail). End-to-end beyond the unit
  gate: a real BH3/STO-3G RCCSD run still converges to the same energy and
  writes a valid version-3 file; C2's `ccsdt_gen`->`cc4` cross-rank restart
  on Be still warm-starts in 1 iteration at the same energy; C1's
  basis-mismatch rejection still cold-starts correctly against a
  hand-corrupted version-3 file.

  **U2/U3 LANDED 2026-09-05, mechanical as predicted.** `run_uccgen`
  (`uccgen.cpp`) now has a write site (`meta.reference_type = UHF`, the four
  counts from `state.reference`, same `_save_checkpoint`/`_checkpoint_path`
  gate and warn-not-fail policy as `run_rccgen`) and a restart site
  (`try_restart_ucc_from_sidecar`, mirroring `try_restart_from_sidecar` with
  a `reference_type` check ahead of the four-count comparison, wired in
  *after* `ensure_amplitude_sectors` since seeding matches an incoming
  sector to its live counterpart by key). `seed_arbitrary_order_amplitudes`
  needed no change, confirmed rather than assumed. Verified end-to-end on
  `b_ucc2_sto3g` (B/STO-3G doublet): a cold run converges in 12 iterations
  and writes the sidecar; a `guess density` restart in the same directory
  warm-starts and converges in **1 iteration** to the identical
  `E_corr = -0.0402694793`. `b_ucc4_sto3g` (the `ucc4 == FCI` flagship gate)
  also writes correctly, still converging to `-24.1892649766`. All three
  negative cases verified against hand-byte-patched copies of a real written
  sidecar: a corrupted `basis_name`, a corrupted occupation count, and a
  flipped `reference_type` (UHF->RHF) each independently cold-start with a
  warning naming the specific mismatch, never failing the run. **The whole
  C4 UCC-sidecar arc (U0-U3) is complete.**

  **C3 (`run_rccsdt`'s hand-written rank-3 participation) investigated and
  declined 2026-09-05**, not merely deferred: only the tensor-production
  backend ever holds dense amplitudes, and only when the system is large
  enough to skip the determinant backstop (in-tree, only `ch4_rccsdt_sto3g`
  qualifies) -- every other case re-solves via spin-orbital
  determinant-space CC with no dense amplitudes surviving to project. Even
  the qualifying case's amplitudes are spin-orbital and local to
  `run_tensor_rccsdt_impl`, never returned to a caller. `ccsdt_gen` already
  covers the gap through the route this project intends to keep long-term.
  Full record in `docs/CC_AMPLITUDE_CHECKPOINT.md`, which also replaces the
  three prior scope docs for this whole feature area (all deleted).

- **FCIQMC: the method is implemented and runs from an input file (2026-08-31 to
  2026-09-01).** `correlation fciqmc` is a working calculation. Walker state and
  RNG, the excitation generator with a consistent `p_gen`, spawn/death/
  annihilation, shift control with the initiator approximation, both energy
  estimators with blocked error bars, and the driver entry point with eleven input
  keywords. Gated by `planck-fciqmc-walkers` (~34 s) and the
  `h2_fciqmc_sto3g` regression case.

  **On H2/STO-3G against exact FCI `-1.1372744062`:** shift `-1.1375360199` ±
  2.76e-03 (**0.09σ**), projected `-1.1373278832` ± 1.58e-04 (**0.34σ**).

  **What is NOT yet demonstrated: a chemically interesting answer.** Every gate
  above H2 runs on a *synthetic* Hamiltonian — one that respects a real
  Hamiltonian's sparsity and is checked against exact diagonalization, but is not
  a molecule. H2's 4 determinants are far below where sampling means anything
  (the walker population covers the whole space). **The N2/STO-3G gate is the
  first real test and is not built**; see `vault/Status/Open Work.md`.

  **The design decision that makes future comparisons meaningful:**
  `build_all_mo_ci_setup` was *extracted* from `run_fci` — a move, not a copy —
  and both paths call it, with the Hamiltonian callbacks wrapping the same
  `slater_condon_element`. The stochastic and deterministic paths therefore cannot
  disagree about the Hamiltonian, only about how they solve it, so a disagreement
  on a larger system is attributable to sampling rather than plumbing.

  **Two prerequisites answered before any of it was written.** Q1 (is there a
  calculation this codebase cannot do?) — yes, **Cr₂ CAS(12,18)**: two atoms,
  344.6M determinants, 2.76 GB per CI vector, the first case where both the time
  and memory walls fail, and inside the existing 31-orbital representation. Q2
  (can a stochastic method be validated in a codebase gated on exactness?) — yes,
  via the G1–G4 machinery (`metric_within_sigma`, a blocking analysis validated
  against analytic AR(1), a fixed-seed harness, and an end-to-end estimator), all
  built *before* any FCIQMC code and reusable independently of it.

  **N2/STO-3G validation landed (2026-09-02).** `n2_fciqmc_sto3g` (extended, 69 s)
  gates both estimators against exact FCI `-107.6529998854` at **0.32 sigma**
  (shift) and **0.41 sigma** (projected), at 0.69 walkers per determinant so the
  population is a genuine sample. Verified non-vacuous: an unstable timestep fails
  it on three independent grounds.

  **The finding worth carrying: a good-looking number hid a broken run.** At
  `dt = 0.010` the shift energy read 0.14 sigma from exact while the reference
  determinant was oscillating in sign (mean `|c_0|` 91.75 against mean signed
  `c_0` -7.50). The shift cannot see it — it responds to the total population —
  so **a single-estimator implementation would have reported a perfect-looking
  answer.** The projected energy and the cross-check caught it. The driver now
  warns on sign instability directly and the gate asserts
  `not_contains: SIGN-UNSTABLE`.

  **Threading determinism decided, not discovered:** no exception. Partitioning the
  parents by `hash(parent) % kBins` and merging in fixed bin order keeps bitwise
  thread-count invariance, gated by `h2_fciqmc_threads1/4` before any threading
  exists.

  Answers: `docs/FCIQMC_SAMPLING_AND_DYNAMICS.md`,
  `docs/FCIQMC_POPULATION_CONTROL.md`,
  `docs/FCIQMC_DRIVER_AND_VALIDATION.md`.

- **FCIQMC: two estimator defects found and fixed by comparing the sampled
  WAVEFUNCTION against deterministic FCI, not the energy (2026-09-02).** Both were
  invisible to every existing gate, which compares energies only. The instrument
  that found them is a coefficient-ratio dump on both paths: `run_fci` prints
  `C_I/C_0` and `run_fciqmc` prints `<N_I>/<N_0>` over the accumulated SIGNED
  per-determinant weight, in one format, at `verbosity verbose`.

  **(1) The reference determinant was built by occupying the LOWEST-INDEX
  orbitals, and index order is not energy order.** On N2/STO-3G the Aufbau
  determinant is `0xbf` (orbitals [0,1,2,3,4,5,**7**]) because MO 6 lies above
  MO 7 in the converged SCF ordering; the old code used `0x7f`. The true reference
  carried **14.2x** the weight of the one everything was normalised against. It
  survived because the failure is silent: the projected energy
  `E = H_00 + (sum_j H_0j c_j)/c_0` anchors on it, so a weak anchor inflates the
  VARIANCE rather than producing a wrong-looking number -- N2's projected error bar
  ran ~20x the shift's and read as ordinary noise. Fixed by minimising
  `ops.diagonal` over single occupied->virtual swaps from the Aufbau guess, using
  the SAME `slater_condon_element` the propagator uses so the reference cannot
  disagree with the Hamiltonian being sampled. Measured on N2: shared determinants
  between the two dumps **0 -> 16**, projected error bar **3.27e-01 -> 1.34e-02
  (24x tighter)**, shift **0.92 -> 0.11 sigma**.

  **The bad reference corrupted the SHIFT too, indirectly** -- it is also the
  starting population, so the run spent equilibration migrating away from a
  near-empty determinant. An earlier note here claimed the shift "never touches the
  reference, so this is a separate defect"; that was wrong.

  **(2) On a DEGENERATE ground state the projected energy decays with sampling
  time.** Any mixture of degenerate eigenstates is itself an eigenstate at the same
  energy, so the imaginary-time dynamics apply **no restoring force** within the
  manifold and the population random-walks between partners. Measured on
  C2/STO-3G (FCI roots 0 and 1 both `-74.6406501646`, partners `0x3f/0x6f` and
  `0x6f/0x3f` at +/-1.000000), varying ONLY the equilibration length:

  | equil | partner/anchor | E_proj | sigma vs exact |
  |---|---|---|---|
  | 20000 | -0.861 | -74.6172886 | +2.62 |
  | 40000 | -1.674 | -74.6413697 | -0.06 |
  | 60000 | **-3.833** | **-74.7503958** | **-5.57** |

  By 60000 the anchor holds a quarter of the partner's weight while the numerator
  still samples the whole manifold, so the ratio inflates negatively and reports an
  energy **5.6 sigma BELOW the variational minimum**. **Nothing is unstable** --
  the sign is steady, the population is controlled, the reference holds 743
  walkers. The run is fine and the ESTIMATOR is measuring the wrong thing.
  **LONGER equilibration makes it WORSE**, inverting the usual intuition: more time
  to converge is also more time to drift.

  Fixed by warning on drift (>2x the seeded reference) and re-anchoring the
  projection onto the largest-weight determinant **once**, at the end of
  equilibration -- a reference that moved during sampling would change what the
  accumulated ratio-of-sums means partway through. The scan breaks ties on the
  bitstrings because the population is a hash map and an order-dependent anchor
  would break fixed-seed reproducibility. Verified as a mutation test: the
  known-broken `eq60` case goes **-5.57 -> +2.27 sigma** with the shift
  **bit-identical** (proving only the projection changed), and non-degenerate N2 is
  **bitwise unchanged** end to end -- inert where there is no drift.

  **The shift is immune** (-1.05/-0.14/-0.49 sigma across the same three runs)
  because it responds to total population growth, which is indifferent to how
  weight is distributed inside the manifold. That is the **mirror image** of the N2
  sign-instability finding, where the projected energy caught what the shift could
  not. **Neither estimator dominates, which is the actual argument for computing
  both.**

  **Fixture consequence: C2/STO-3G is unsuitable for validating the projected
  energy** and is replaced by **HF/6-31G** (11 orbitals, 5a/5b, **ndet = 213444**,
  4.8x C2). Non-degeneracy was **measured, not assumed** -- roots 0 and 1 at
  `-100.1156979102` / `-99.7369526906`, a **10.31 eV gap**, against C2's
  `0.00e+00`. Inferring non-degeneracy from the "closed-shell singlet" label is
  exactly the mistake C2 punished. A population sweep there spans 0.047-0.469
  walkers/determinant, entirely below saturation (C2 saturates at 44100 walkers).
  Fixtures and the sweep driver `tests/fciqmc_validation.py` are committed but
  **deliberately unregistered**: each sweep is minutes-to-hours, which no CI budget
  absorbs.

- **FCI sigma build: 4.8x from removing the allocator, then 3.54x more from
  threading it (2026-08-30).** The iterative `apply_ci_hamiltonian`
  (`src/post_hf/ci/ci.cpp`) — the path taken by every determinant space above
  `dense_threshold = 500` — was both allocation-bound and single-threaded.

  **F1:** `get_excitation` returned a pair of heap vectors for values holding at
  most two entries each (a Slater-Condon element vanishes beyond a double
  excitation); it now returns a fixed-capacity struct. N2/STO-3G **125.7 s ->
  26.3 s**, and the `malloc`/`free` profile share went **53 % -> 0.1 %** — the
  allocator is eliminated, not reduced. Worth keeping: **the 4.8x exceeded what
  the profile implied** (a 53 % share caps the direct saving at ~2.1x by Amdahl),
  because per-element churn also cost cache pressure and bookkeeping attributed to
  other frames. **A profile share is a lower bound on what removing that work is
  worth.**

  **F3:** the loop is now threaded, **3.54x at 4 threads** (N2 26.3 s -> 7.79 s;
  1/2/4/8 threads = 27.59/14.66/7.79/5.71 s) against a modelled ~3.7x ceiling.
  `be_fci_spherical_631gd` 7.6 s -> 3.42 s. Energies are **byte-identical across
  `OMP_NUM_THREADS` = 1/2/4/8** and equal to the serial values
  (`-107.6529998854`, `-14.6139425466`, `-147.7441885517`).

  **The route the scope recommended — inverting the scatter to a gather — was
  built, is numerically correct, and is 2.2-2.4x SLOWER.** The outer
  `|c| < 1e-15` skip is not a summation-order detail but a sparsity exploitation
  carrying asymptotic weight: in the scatter it elides the entire 126-line
  enumeration for a negligible ket, and the trial vectors really are sparse
  (Davidson starts from unit vectors; `solve_ci` reconstructs a dense `H` one
  `Unit(dim, j)` column at a time, which is O(dim) enumerations scattered and
  **O(dim^2)** gathered). The gather is stashed, not committed.

  **So the scatter is kept, with partial vectors summed in fixed order — and the
  determinism half is the part worth carrying: a fixed-order reduction is
  NECESSARY BUT NOT SUFFICIENT.** Two defects, each caught only by byte-diffing
  outputs across thread counts, neither by reasoning: `schedule(dynamic)` (which
  the scope recommended for load balance) gives a buffer a different *subset* of
  terms per run so its sums reassociate — two different last digits across 5
  identical 4-thread runs; and keying buffers by `omp_get_thread_num()` makes their
  *contents* depend on the thread count, so 8 threads disagreed with 1/2/4 even
  under `schedule(static)`. **What must be deterministic is the partition of work
  into accumulators, not merely the order they are summed.** The fix bins by a
  fixed function of `j` (`partials[j / bin_size]`, `constexpr kBins = 64`) with
  `schedule(static, bin_size)` so one chunk is exactly one bin. Memory is
  `kBins x dim x 8`, **independent of thread count** (7.4 MB at N2). Never
  `omp atomic`, never completion-order — the DFT-grid jitter defect.

  **19/19 green** — every FCI, CASSCF, RASSCF, FCIDUMP and RI case in the repo,
  including both iterative FCI gates and all four SA-CASSCF cases with the
  `water_casscf_sa2_sto3g_sad_guess_uphill` canary. The 126-line excitation
  enumeration is untouched; the only deleted lines are the old `accumulate`
  lambda. See `docs/FCI_SIGMA_BUILD_PERFORMANCE.md`. F2 (the `occupied_orbitals` allocation)
  remains open and independent.

- **ccgen `merge_transposes` threaded into production — 1.42x / 1.52x measured
  (M1-M5, 2026-08-29).** Transpose-equivalent derived dressed operators now fold
  onto one shared array, which the call sites read through a permutation instead
  of each building its own copy. On spatial `ccsdt` that is **288 -> 91 builders
  (3.16x)**, with the hottest family `t2t2v_oooovv` going **38 -> 4 (9.5x)**.
  Unconditional for `--dressing derived`: no flag, no CMake option — the change is
  **10 lines in one file, 8 of them comment**, since the mechanism was already
  built, symbolically exact and value-gated 0/2536 at rank 4.

  Measured on the two generated-route cases, 3 runs each: `lih_rccsdt_generated_sto3g`
  **1.02 s -> 0.72 s (1.42x)**, `ch4_rccsdt_sto3g` **16.65 s -> 10.98 s (1.52x)**,
  spread ±0.03 s. **Energies bitwise identical** — LiH agrees on `E_corr`, `dE`,
  `rms(res)` and `rms(step)` at every one of 62 iterations, CH4's full output
  diffs empty with timers stripped — and iteration counts are unchanged (62, 15),
  so this is per-iteration work removed rather than faster convergence. Compile
  time and size improve too: registry TU (`-O1`-pinned) 11.68 s -> 10.44 s and its
  object 1.50x smaller, `tensor_backend.cpp` 38.86 s -> 35.80 s, binary 1.11x
  smaller.

  **The estimate was wrong twice, in the same direction, for the same reason.** An
  operator-count FLOP model said 1.02x-1.20x ("compile time, not speed"); a
  profile-weighted re-cost said 1.21x-1.36x; measurement said 1.42x-1.52x. Both
  models treated operators as equal-cost. The re-cost additionally applied H5's
  measured 0.66 realization factor, which does not transfer — H5's discount existed
  because the builds *it* removed were cheaper than average, whereas the builds
  removed here are the **expensive** ones (`t2t2v_oooovv` writes a rank-6 result
  over a 9-deep nest). **A realization factor encodes which work a transform
  removed; it is not a general haircut for the next estimate.**

  **The causal story was verified, not just the total.** Leaf-sample attribution on
  CH4: `t2t2v_oooovv` 23.3 % -> 4.1 % (merges 9.5x), `t1t1t2v_oooovv` 6.6 % -> 1.7 %
  (6.0x), `t1t3v_oooovv` 9.7 % -> 9.4 % (1.7x). The families that merge hard
  collapse and the one that barely merges barely moves — the negative control that
  separates "the number moved" from "the number moved for the stated reason".

  New gate `python/ccgen/tests/test_merged_call_sites.py`: the existing builder gate
  is definition-only and cannot see a call site reading a merged array through the
  wrong permutation (the D4 failure shape), so this one emits the residual **both
  ways and requires equality**. Mutation-verified — three independent call-site
  perturbations go red at 1.5e-03 to 8.6e-03. **Two defects in the gate itself are
  worth carrying:** covering only singles+doubles made it blind to the entire point
  (**all 56 `t2t2v_oooovv` reads live in the H5 `_partN` triples chunks**, none in
  doubles; the perturbation moved it 2.2e-16), and a fixture can be too **general**
  — a random `t2` made 17 doubles terms look broken at 6.4e-02, because the merge
  is justified by `t2(a,b,i,j) == t2(b,a,j,i)`, a symmetry real amplitudes have
  (2.8e-17 once imposed). Withholding a symmetry the physical object **has**
  manufactures failures, the inverse of the usual vacuity trap.

  **Rank 4 needed no new mechanism** — `-DPLANCK_CC_MAXORDER=4` appends `ccsdtq`
  to the same method list through the same emit path, and merging is
  unconditional, so it already applies. The reduction roughly **doubles** with
  rank: **1615 -> 239 builders (6.8x)** against rank 3's 3.2x, TU 11.0 MB ->
  6.6 MB, with families as extreme as `t2t2v_ooooovvv` 95 -> 1 and
  `t2t3v_ooooovvv` 313 -> 15. The M5 gate runs at both ranks and is
  mutation-verified at both (rank-4 perturbations red at 3.7e-03 / 6.7e-03).
  Extending it exposed two rank-4-only emitter conventions, each failing with a
  message naming neither the rank nor the cause: **rank >= 7 targets use the
  BRACED runtime-rank accessor** (`result({i, j, ...})`, which numpy rejects as
  non-letter subscripts) and **rank >= 7 builders return `TensorND`**, invisible
  to `build_emitted_operators`'s fixed-rank regex, so a `ccsdtq` residual could
  not be evaluated at all until a local pass materialized them. What remains
  untried at rank 4 is only an end-to-end RUN — building the 6.6 MB dressed TU
  against the `-O1`-pinned registry and timing a solve. Full record:
  `docs/CCGEN_MERGE_TRANSPOSES.md`.

- CC tensor element accessors inlined (`src/post_hf/cc/common.h`), the dominant
  cost in every CC kernel. `Tensor{2,4,6}D::operator()` and the runtime-rank
  `TensorND` / `DenseTensorView` / `ConstDenseTensorView` braced-index accessors
  were defined out-of-line in `common.cpp`; with no LTO configured they could
  not be inlined away, so each element access was a cross-TU call that
  heap-allocated one or two `std::vector<int>` and built a `std::expected`
  before indexing. The generated kernels are dominated by this — 3416 accessor
  call sites in the rank-3 triples residual, 23338 braced-index accesses in the
  rank-4 quadruples residual (where the `initializer_list` overload additionally
  copied into a vector via `to_vector` first). Now flat row-major index
  computations in the header, with the debug assert retaining both conditions
  the old `checked_fixed_rank_index` enforced: per-index range **and**
  `offset < data.size()`. The storage half is not redundant with the
  constructors — `data` is a public member that call sites assign directly after
  construction (`tensor_backend.cpp:197-198`), so the size invariant is
  breakable post-construction by design. In release the check compiles out and
  an out-of-bounds access becomes UB rather than returning a shared
  `tensor_error_slot`; acceptable because that slot has no consumers anywhere in
  `src/` or `tests/` and reaching it already fired `assert(false)`. Measured,
  energies bitwise-identical throughout: rank-3 generated T3 residual 6.40 s →
  0.031 s (206×), rank-3 hand-written 0.170 s → 0.0014 s (121×), rank-4 CCSDTQ
  38.5 s → 11.4 s per iteration (3.4×), `water_rccsdt_sto3g` 44.6 s → 0.39 s.
  **Fixing only the fixed-rank accessors left rank 4 completely unchanged** —
  the rank ≥ 4 generated kernels index exclusively through the runtime-rank
  types — so both halves were required; rank 3 is not a proxy for rank 4. Gated
  by `planck-cc-tensor-index`, which pins the flat index against an independent
  row-major reference on **non-square** shapes (a square fixture cannot catch a
  transposed index), covers the permuted-dims `swap_mid_axes` pattern used by
  `rebind_physicist`, and cross-checks the braced-index overload against the
  still-out-of-line `vector<int>` one. Also drops the now-unused `to_vector` and
  `checked_fixed_rank_index`. See `docs/CCGEN_TENSOR_ACCESSOR.md`.
- Generated rank-3 CCSDT fixed, and the kernel was never the defect. The ccgen
  rank-3 triples residual reproduces PySCF `rccsdt` to +1.49e-08 and is
  **bitwise identical** across both harnesses at identical inputs; what was wrong
  is the `tensor_backend` solver wrapped around it. That solver uses a
  symmetry-packed amplitude representation — DIIS packs only the unique wedge
  (`i<=j` for t2, `i<=j<=k` for t3) and rebuilds the rest via
  `restore_restricted_t{2,3}_from_unique`, valid only if the amplitudes carry full
  permutational symmetry, which `restore_restricted_t3_structure` imposes each
  iteration. The ccgen kernels emit every index permutation explicitly instead, so
  they never produce residuals in that representation. The wedge packing and
  `restore` are one coupled convention: removing either half diverges (measured
  with both residual sources), and no combination of residual sources inside
  `tensor_backend` converges correctly — hand r1/r2 + gen r3 gives −7.56e-05, all
  three generated gives +8.23e-05. Fixed by routing generated rank-3 to the
  arbitrary-order harness, the representation the kernels are emitted for:
  `optimized` was recorded as landing at +1.44e-08 (5247× error reduction), agreeing
  with the hand-written path to 1.0e-10. **That does not reproduce (2026-08-26).**
  Commit `1986d0c` was checked out, built with the same configuration and run on the
  same input: `E_corr = -0.0565650696` against the hand-written `-0.0791116825`,
  never converging within 100 iterations — **bit-identical to HEAD**, so this is not a
  regression, the result was never reproducible. The gate that commit added,
  `ch4_rccsdt_sto3g`, asserts `kernels=hand-optimized` and so has been green
  throughout while never running the generated kernel it was added to protect.
  Rank 2 and rank 4 both work through the same harness, localizing the defect to the
  rank-3 kernel. Scoped in `docs/CCGEN_SPIN_ADAPT_DEFAULT.md`; it blocks
  W4 and therefore the derivation-route wiring. Without
  `-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON` it fails with an actionable message
  instead of a wrong number. The hand-written path is untouched and bitwise
  unchanged. **The rank-parity hypothesis is dead** — its premise (rank 3 is
  wrong) was false; the arbitrary harness is correct at ranks 2, 3 and 4, and
  ranks 5/6 still have no numeric gate. New gate `ch4_rccsdt_sto3g` (`nso=18
  ndet=43758`, `no=5 != nv=4`, PySCF −39.8058445240) is the **only** in-tree
  rank-3 case that clears `choose_determinant_backstop` and therefore the only one
  that reaches the tensor path at all — every other CC case routes to the
  determinant prototype, and `water_rccsdt_sto3g` even *asserts* the backstop
  handoff, so the hand-written tensor solver had no regression gate for its entire
  life. Verified falsifiable before being trusted. Cost is not addressed: the
  correct path is ~500× slower (0.19 s vs ~100 s on CH4), per-iteration rather
  than convergence; see `docs/CCGEN_ARBITRARY_HARNESS_COST.md`. Full answer,
  with the eight ruled-out hypotheses:
  `docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md`.
- The generated-vs-hand-written CC kernel gap is characterized: it is a
  **scaling defect, not a constant tax**. Six rank-3 ladder points (BH3/STO-3G
  21.8× → C2H4/STO-3G 50.1×, no plateau) with the generated and hand-written
  residuals evaluated from identical amplitudes. Hand-written fits
  `o^3.94 v^4.18` at 4.5% residual — textbook `o³v³` output × one contracted
  index. Generated fits `o^4.87 v^4.52` but at 21.4%, and that residual is
  **concentrated at high `v`** (the four lowest-`v` points fit to ≤5.5%; `v=8`
  is +21%, `v=11` is −10%), so a single power law does not describe the
  generated cost — evidence of multiple contraction regimes, consistent with
  different residual terms having different optimal orders and the emitter
  picking none of them. The earlier carried "~180× on `bh3`" figure never
  recorded its dimensions and did not reproduce (measured 37.6× on actual `bh3`
  pre-accessor-fix); it can be neither explained nor dismissed from this data,
  and should not be cited. Adds the opt-in `PLANCK_CC_T3_TIME=N` probe (inert
  when unset). Two constraints found and recorded: `choose_determinant_backstop`
  (`tensor_backend.cpp:241`) routes `nso ≤ 16 && ndet ≤ 10000` to the
  determinant-space backstop, which never calls the generated kernel — so
  `water_rccsdt_sto3g` silently yields no timing at all and any ladder point
  needs `nso > 16 || ndet > 10000`; and the whole reachable ladder stays under
  0.85 MiB `t3`, inside L2, so the memory-bound hypothesis is untested rather
  than refuted. See `docs/CCGEN_KERNEL_SCALING_SCOPE.md` and
  `docs/CCGEN_KERNEL_PERFORMANCE.md`.
- SAD isolated-atom false-convergence fixed in the SCF convergence gate
  (`is_converged`, `src/scf/scf.cpp`). For small lone closed-shell atoms
  (He/cc-pVDZ) the SAD guess drove DIIS to extrapolate a Fock whose
  diagonalized density exactly reproduced the previous one (ΔP → 0) while the
  DIIS residual FPS-SPF was still ~1e-3, so the ΔE+ΔP gate declared convergence
  in a wrong basin (-2.8551548739 vs the true -2.8551604772). `is_converged`
  now also requires the DIIS error below `_tol_density`; `IterationMetrics`
  carries `diis_error` (set at the RHF/UHF/ROHF call sites from the already-
  computed `diis_err`), and it is 0 when DIIS is inactive so non-DIIS paths are
  unaffected. Full regression suite unchanged (71/71); new gate
  `he_sad_ccpvdz` pins the He/cc-pVDZ SAD energy to -2.8551604772. The earlier
  BSSE HCore workaround is no longer required for correctness.
- Rys 6D ERI accumulator sized per quartet (PR #126). `_rys_sum_buf` in
  `src/integrals/rys.cpp` was a thread-local `double[2·MAX_L+1]^6 = [13]^6 =
  38.5 MB` sized off the global `MAX_L=6`; on the g++-15 / emulated-TLS build
  emutls `calloc`s the full block on each thread's first Rys-kernel access, so
  every Rys-active worker thread paid 38.5 MB despite the reachable angular
  momentum being far lower (Auto only routes `L_AB+L_CD≤1` to Rys; explicit
  `engine rys` tops out at F in the suite). Replaced with a thread-local
  `RysScratch` struct sized per quartet via `resize_for_quartet` with flat
  `index()`/`at()` accessors, reused across quartets — mirroring the HGP/OS
  `EriScratch` pattern but minimal (spatial-only, no Boys `m` axis). No L
  ceiling and no rejection guard: explicit `engine rys` at g/h just sizes the
  vector larger. Footprint ~KB/thread under Auto, ≤0.94 MB at the explicit-rys
  F worst case. Index layout unchanged, so bitwise-identical — gated by
  `planck-compute-2e` + `planck-hgp-engine-smoke` and the full extended
  regression suite (RYS/OS/Auto engine comparisons). `MAX_L` and `os.cpp`
  untouched. Follow-up (Open Work): factor the shared 6-axis spatial layout out
  of the OS/HGP/Rys scratch structs.
- DIIS coefficient-solve conditioning guard (shared across RHF/ROHF and UHF).
  Both `extrapolate()` paths now route through
  `HartreeFock::solve_diis_coefficients` (`src/base/types.h`), which drops the
  oldest vector and retries only on genuine numerical breakdown — an indefinite
  error-overlap (Gram) block or non-finite/explosive coefficients — rather than
  on the Gram condition number, since healthy near-converged SCF routinely has
  ~1e-29 smallest Gram eigenvalues while staying positive-definite. It is a
  no-op on well-behaved SCF (energies and iteration counts unchanged). Gated by
  the `planck-diis-conditioning` unit test (well-conditioned / benign near-
  converged / singular-Gram). Merged via PR #124.
- Mayer bond-order open-shell factor-of-2 fix. The unrestricted branch of
  `mayer_bond_order_analysis` (`src/populations/bond-order.cpp`) was missing
  the leading `2` on the spin-resolved `(P^α S)² + (P^β S)²` contraction, so
  every open-shell bond order came out at half its correct value; the
  closed-shell total-density branch was already correct. PySCF-anchored
  (H2 RHF B(H–H)=1.0, H2O+ UHF B(O–H)=0.76017799 vs 0.76017810). New gates:
  `h2_rhf_mayer_bond_order_sto3g`, `h2o_cation_uhf_mayer_bond_order_sto3g`,
  plus a closed-shell unit assertion that was previously absent. See the
  Mayer Bond Order Density Convention gotcha.
- Two previously-open robustness items were verified already-resolved in the
  tree (no new work, doc was stale): the developer-specific absolute basis
  path is not committed (`src/base/basis.h` is git-ignored; the tracked
  `basis.h.in` template uses `@BASIS_INSTALL_PATH@` + `$BASIS_PATH`), and the
  CASSCF orbital-action solver already warns and falls back to the diagonal
  preconditioner when >20% of orbital-Hessian eigenvalues are clamped
  (`src/post_hf/casscf/response.cpp`).
- ccgen dressed-operator recognition was quadratic in manifold size and is fixed.
  `hypothesis_is_consistent` rebuilt `raw_multiset(residual_terms)` on every call
  — 7461 times on `ccsdt` triples, over an input that never changes — so the cost
  was `n_hypotheses × n_terms`. Hoisted into `find_operator_occurrences` and
  threaded down (not memoized: the redundancy is structural, and a cache keyed on
  large term tuples would live for the process). Triples 94.7 s → 6.9 s with the
  `raw_multiset` call count flat at ~19 regardless of term count; rank-3
  end-to-end 293.7 s → 9.1 s, and rank 4 went from ">25 min, abandoned" to
  61.6 s, which is what made rank-4 dressing viable at all. Output byte-identical.
  Gated by `test_dressing_scaling` on the **call count**, not wall-clock —
  deterministic, and it names this defect if it returns. Two dead ends recorded in
  `docs/CCGEN_DRESSING_COST.md`: `_eri_canonical` showed the largest
  profile number (864 s *cumulative*) but memoizing it bought 6 %, and the
  self-time ranking is diffuse — the win was structural, not micro-optimization.
- ERI / transform parallelization pass (profiled, all bitwise-verified):
  the two serial 4-index transforms (`Correlation::transform_eri`,
  `BasisFunctions::transform_eri_cart_to_sph`) are now parallel; the one-shot
  triangular `_compute_2e` loop is flattened + `schedule(dynamic)` across all
  four engines (OS / HGP / Rys / Rys-auto); and the DFT `build_coulomb_from_eri`
  / `build_exchange_from_eri` KS builds are parallel. Idle-wait on the profiled
  cases dropped from ~72%→~7% (spherical RHF), ~72%→~0% (MP2), and 60%→12%
  (B3LYP), with all energies unchanged. New gates: `planck-transform-eri`,
  `planck-transform-eri-sph`, `planck-compute-2e`. The DFT J/K builds were
  confirmed thread-count-invariant (no repeat of the grid-reduction jitter).
  Those two dense-tensor builders were later deleted outright (2026-07-20):
  the KS J/K now run memory-direct through the fused Fock loop
  (`_compute_2e_j_direct` / `_compute_2e_k_direct`), cutting 8-water/6-31g
  B3LYP peak RSS 1360 -> 427 MB at an identical energy; see `docs/HPC_MPI.md`.
  See the Integral Engine and DFT implementation notes.
- ROHF references enabled for CASSCF and RASSCF. ROHF stores a single common
  spatial-orbital set in the alpha channel, so the MCSCF loop and CI engine
  consume it unchanged; the work was a guard relaxation in `src/driver.cpp` and
  `src/post_hf/casscf/casscf.cpp`, not new MCSCF machinery. Open-shell systems
  are supported when all unpaired electrons sit inside the active space (closed,
  doubly-occupied inactive core), enforced by the existing
  `(n_elec - nactele)` parity guard; a spin-polarized open inactive core stays
  rejected. PySCF-gated to ~1e-8 Eh by `o2_casscf_rohf_sto3g` (triplet O2
  CAS(8,6)) and `oh_casscf_rohf_sto3g` (doublet OH CAS(5,4)); the RHF CASSCF
  gate suite is unchanged. See [[CASSCF and SA-CASSCF]].
- RMP2 analytic gradient response-path fix, validated against finite differences
  and PySCF
- UMP2 gradient cross-check on the radical-cation path, with no code fix required
- BSSE / ghost-atom infrastructure and CP driver
- Full-symmetry direct-SCF performance improvements: persisted skeleton ERI and
  monomial-group-operator fast path
- HGP screened-derivative correctness: `hgp_vrr` now scales the C-VRR
  `inv_2_delta` cross-coupling term by `screen.boys_scale` for non-Coulomb
  kernels, matching OS. The screened-kernel OS fallback inside HGP
  `_contracted_eri_elem` is removed, and the gradient dispatcher's
  Coulomb-only HGP guard is lifted. Net effect: range-separated DFT
  gradients (HSE06 etc.) now run natively through HGP when the engine is
  selected. Gated by a 2352-quartet OS↔HGP sweep on water/STO-3G (max diff
  ~4e-15) plus four end-to-end cross-engine comparison regressions:
  `water_{rhf,b3lyp,hse06}_gradient_engine_os_vs_hgp` and
  `water_rhf_geomopt_engine_os_vs_hgp`. See [[HGP Screened inv_2_delta]].
- HGP screened Fock builds: the three OS fallbacks in
  `HeadGordonPople::_compute_2e{,_fock,_fock_uhf}` (lines 914 / 997 / 1031
  pre-lift) are removed. Screened-kernel SCF Fock builds — closed-shell
  and unrestricted, conventional and direct, C1 and full-symmetry —
  now run native HGP end-to-end when `engine hgp` is selected. Gated by
  five end-to-end SCF-energy regressions
  (`water_{rhf,hse06,uhf_triplet,uks_hse06,hse06_symm}_scf_energy_engine_os_vs_hgp`),
  each comparing the OS and HGP `Total Energy` / `DFT Energy` to ≤ 5e-9 Eh.
  Comparator: `tests/engine_scf_energy_compare.py`.

## CASSCF PySCF Gate Table

Suite status: **10/10 passing, 1 re-pinned** (2026-10-28). The water SA-2
hcore reference, -74.7751378, is a measured saddle of the SA-2 energy (one
negative CI-relaxed Hessian eigenvalue). With the exact Hessian (CASSCF
performance scope, P3c) Planck descends to the minimum, -74.7877864783 (-74.7877865251 since the
2026-11-01 Bohr-constant fix), which
is also PySCF's own SAD-start answer. See
`tests/benchmarks/casscf/pyscf_reference/pyscf_reference_energies.md`.

PySCF version: 2.12.1. All scripts use `mol.cart = True` to match Planck
Cartesian-basis references. Tolerance: `1e-5 Eh`.

| Case | Active space | Basis | PySCF / Eh | Planck / Eh | Delta / Eh | Status |
|---|---|---|---|---|---|---|
| h2_cas22_sto3g | CAS(2e,2o) | STO-3G | -1.1372838345 | -1.1372838351 | 6.0e-10 | PASS |
| lih_cas22_sto3g | CAS(2e,2o) | STO-3G | -7.8811184639 | -7.8811184797 | 1.6e-08 | PASS |
| water_cas44_sto3g | CAS(4e,4o) | STO-3G | -74.9760171635 | -74.9760171760 | 1.2e-08 | PASS |
| water_cas44_631g | CAS(4e,4o) | 6-31G | -75.9998609866 | -75.9998609785 | 8.1e-09 | PASS |
| water_cas44_ccpvdz | CAS(4e,4o) | cc-pVDZ | -76.0440109036 | -76.0440109052 | 1.6e-09 | PASS |
| water_cas44_b1 | CAS(4e,4o) | STO-3G | -74.5856164513 | -74.5856163677 | 8.4e-08 | PASS |
| ethylene_casscf_321g | CAS(2e,2o) | 3-21G | -77.5145223959 | -77.5145223872 | 8.7e-09 | PASS |
| ethylene_casscf_321g_nroot2 | CAS(2e,2o) | 3-21G | -77.5145223959 | -77.5145223872 | 8.7e-09 | PASS |
| ethylene_casscf_ccpvdz | CAS(2e,2o) | cc-pVDZ | -77.9524856209 | -77.9524855977 | 2.3e-08 | PASS |
| water_cas44_sto3g_sa2 | CAS(4e,4o) SA-2 | STO-3G | -74.7751378317 | -74.7877864783 | 1.26e-02 | RE-PINNED (PySCF value is a saddle) |
| ethylene_cas44_sto3g_sa2 | CAS(4e,4o) SA-2 | STO-3G | -77.0034974774 | -77.0034974301 | 4.7e-08 | PASS |

### CASSCF validation notes

- The committed gate suite is still the clearest validation point for the CASSCF stack.
- Two ROHF CASSCF cases are gated alongside the RHF table: `o2_casscf_rohf_sto3g`
  (triplet O2 CAS(8,6), Δ ~9e-9 Eh) and `oh_casscf_rohf_sto3g` (doublet OH
  CAS(5,4), Δ ~7e-9 Eh; re-pinned 2026-11-01 from a saddle at -74.3854277 to
  the minimum -74.3867176233), both vs PySCF 2.13.0 with `mol.cart = True`.
- The water SA-2 SAD-start uphill-enabled case reaches the PySCF SAD-start basin
  within `3.6e-08 Eh`.
- The baseline monotone SAD-start landing is also intentionally preserved as a
  separate regression because it exercises a different optimizer policy.
