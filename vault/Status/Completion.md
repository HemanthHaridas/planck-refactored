---
name: Completion Status
description: Canonical summary of what is implemented and validated in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, completion, validated, canonical]
---

# Completion Status

Last updated: 2026-06-04

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
- PCM solvation for single-point RHF/UHF runs

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

- Shared-kappa state-averaged coupled orbital/CI solve is the primary production path
- Exact CI-response RHS is the default
- FD-based SA orbital Hessian action (`delta_g_sa_action`) is implemented and wired
- Active-integral-cache transform is landed and benchmarked
- Per-root SA total-energy reporting is fixed
- SA diagnostics are parsed by the regression runner
- SAD-start uphill-enabled water SA-2 basin is validated and retained as a separate
  regression mode
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
- Range-separated libxc functionals for single-point, analytic-gradient,
  geometry-optimization, frequency, and geomopt+frequency workflows
- Double-hybrid libxc functionals for single-point energies
- Treutler-Ahlrichs radial grid, Lebedev angular grid, and Becke partitioning
- Grid quality levels: Coarse, Normal, Fine, UltraFine
- Single-point PCM solvation for RKS/UKS
- Linear-response TDDFT / Casida and TDA excited states
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

### Recent fixes now considered landed

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
  `optimized` now lands at +1.44e-08 (5247× error reduction) and agrees with the
  hand-written path to 1.0e-10; without
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
  than convergence; see `docs/CCGEN_ARBITRARY_HARNESS_COST_SCOPE.md`. Full answer,
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
  `docs/CCGEN_KERNEL_PERFORMANCE_SCOPE.md`.
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

Suite status: **11/11 passing**

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
| water_cas44_sto3g_sa2 | CAS(4e,4o) SA-2 | STO-3G | -74.7751378317 | -74.7751377977 | 3.4e-08 | PASS |
| ethylene_cas44_sto3g_sa2 | CAS(4e,4o) SA-2 | STO-3G | -77.0034974774 | -77.0034974301 | 4.7e-08 | PASS |

### CASSCF validation notes

- The committed gate suite is still the clearest validation point for the CASSCF stack.
- Two ROHF CASSCF cases are gated alongside the RHF table: `o2_casscf_rohf_sto3g`
  (triplet O2 CAS(8,6), Δ ~9e-9 Eh) and `oh_casscf_rohf_sto3g` (doublet OH
  CAS(5,4), Δ ~1.6e-8 Eh), both vs PySCF 2.13.0 with `mol.cart = True`.
- The water SA-2 SAD-start uphill-enabled case reaches the PySCF SAD-start basin
  within `3.6e-08 Eh`.
- The baseline monotone SAD-start landing is also intentionally preserved as a
  separate regression because it exercises a different optimizer policy.
