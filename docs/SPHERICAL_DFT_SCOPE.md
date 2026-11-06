# Can `planck-dft` run in a spherical-harmonic basis?

Research scope, opened 2026-10-26. The shared prerequisites (S0.1–S0.6),
step 1 (RKS/UKS single points) and step 4 (SOSCF and KS stability) have
landed; steps 2, 3, 5 and 6 are open, so this is still a scope. Rewrite it as the
answer when they land. Status lives in `vault/Status/Open Work.md`.

## Why it matters

Routine DFT is run in spherical (pure) functions: cc-pVXZ and def2 are defined
that way, and a Cartesian d shell carries an extra s-type function, so the two
give different energies. `planck-dft` rejects `basis_type spherical` at the top
of `DFT::Driver::run` (`src/dft/driver.cpp`), while the HF binary already runs
spherical SCF, gradients, geometry optimization and frequencies. Spherical DFT
is also the gate for DFT with ECPs (`docs/DEF2_ECP_SCOPE.md` step 6), since
def2 is spherical.

## Answer in one paragraph

Keep everything that touches the grid or the integral engines Cartesian, and
move only the SCF scaffold (S, H_core, orthogonalizer, guess, DIIS,
diagonalization, densities, MO coefficients, checkpoint) to the working basis,
as HF does. The grid layer is linear in the AOs, so with P_cart = Cᵀ P C the
density, its gradient and V_xc are exact, V_xc,sph = C V_xc,cart Cᵀ. The same
holds for J, K, the XC response kernel and MO coefficients (c_cart = Cᵀ c_sph).
The shared machinery is in place: one pair of conversion helpers, a
normalized C on every basis read, spherical-safe ERIs for post-SCF methods,
and lifted RKS/UKS gradients. What remains is placing the boundary at about
ten call sites in `driver.cpp`, then gating each workflow.

## Current state (2026-11-04)

What `planck-dft` can rely on now:

- **`Basis::_cart_to_sph` (C, n_sph × n_cart)** is built and row-normalized by
  `read_gbs_basis`, so diag(C S Cᵀ) = 1 at any geometry. Nothing downstream
  renormalizes it (S0.5).
- **`BasisFunctions::to_cartesian(basis, M)`** (Cᵀ M C) and
  **`to_working(basis, M)`** (C M Cᵀ), in `src/basis/spherical.{h,cpp}`. They
  are shape-checked, return `expected`, and return M unchanged in Cartesian
  mode. HF, the gradient, SAD, multipoles and ESP all go through them (S0.3).
- **`SCF::rebuild_basis_dependent_state`** (`src/scf/working_state.cpp`)
  builds shell pairs, refreshes integral symmetry, and stores working-basis
  `_overlap` and `_hcore` (V_ne + ECP). The DFT driver does not call it yet.
- **`Correlation::ensure_eri`** returns a spherical tensor in spherical mode
  even when the SCF cached none (S0.2). TDDFT and double-hybrid PT2 need this,
  because the DFT SCF never caches a tensor.
- **`compute_rks_gradient` / `compute_uks_gradient`** lift P and build W in
  the working basis and lift it once (S0.4).
- **Already spherical-aware:** the SAD guess, SAO blocking
  (`build_sao_basis`), the checkpoint (stores the basis type), multipoles,
  ESP (S0.1) and `one_electron_gradient`.

## What remains

| site (`src/dft/driver.cpp`) | today | needed |
|---|---|---|
| `validate_spherical_support` | admits SinglePoint, SOSCF and stability analysis; refuses other workflows, DH, PCM and checkpoint restart by name | widen per step |
| checkpoint restart (`load_ks_restart_state`, `project_ks_restart_density`) | refused in spherical | the cross-basis projection needs the spherical overlaps, as HF's does |
| XC nuclear gradient (`compute_analytic_ks_gradient`) | stored densities to `compute_xc_nuclear_gradient_{rks,uks}` | `to_cartesian` |
| TDDFT (`run_linear_response`) | `n_virt = nbasis − n_occ` | `n_virt` from `mo_coefficients.cols()`; tensor from `ensure_eri` |
| PCM reaction field | stored density | stays rejected (HF rejects spherical PCM too) |
| DH gradients (`dh_pt2_gradient.cpp`, `udh_*`) | require a Cartesian basis | stay rejected |

The other 30 `nbasis()` sites in `src/dft/` (`ao_grid.h`, `xc_grid.cpp`,
`ks_matrix.cpp`, `dft_gradient.cpp`, `analytic_hessian.cpp`, `dh_*`, `udh_*`)
check against the AO grid's dimension and stay Cartesian. They need no edit
once their inputs arrive lifted, and their dimension checks turn a missed lift
into an error. **Audit rule:** a site that sizes a grid or integral object
stays Cartesian; a site that sizes an SCF object becomes `working_nbasis()`.

**Remaining, in order:** step 2 (gradients, geomopt, frequencies: the XC
nuclear-gradient lift), step 3 (TDDFT: `n_virt` from `mo_coefficients.cols()`,
tensor from `ensure_eri`), step 5 (DH single points), then checkpoint restart
(the cross-basis projection needs spherical overlaps, as HF's does). Step 6
(PCM) stays rejected. After step 2, `DEF2_ECP_SCOPE.md` step 6 can lift the
ECP refusal for single points and gradients together. The SOSCF paths size
`n_virt` from `working_nbasis()`, which is safe only because no
linear-dependence removal exists; switch them to `mo_coefficients.cols()` if
that ever lands.

### Steps

1. **RKS and UKS single points — LANDED 2026-11-04.** The DFT driver's 1e
   copy is deleted in favour of `SCF::compute_working_one_electron` (split out
   of `rebuild_basis_dependent_state`); SCF objects are sized by
   `working_nbasis()` (also in `Calculator::initialize`, while `scf_mode`
   and `max_cycles` stay on the Cartesian count because the ERIs are built
   in Cartesian); `evaluate_current_density_and_xc` and
   `assemble_current_ks_potential` lift the densities and project every
   matrix of `KSPotentialMatrices`; `validate_spherical_support` replaces the
   blanket rejection. Measured:
   - Gate A, exact, 6-31G: spherical = Cartesian to ≤3.4e-13 on water and C1
     H2O2, PBE / B3LYP / HSE06, RKS and UKS (12 combinations).
   - Gate B, PySCF 2.13.0: at `fine` both bases sit 1e-6..2e-5 from PySCF, a
     floor set by the two codes' different pruned grids, with the spherical
     gap tracking the Cartesian one. With matched unpruned grids (150 × 590)
     the floor vanishes: water/cc-pVTZ PBE, B3LYP, HSE06 RKS and PBE UKS, and
     C1 H2O2 B3LYP RKS / H2O2⁺ PBE UKS at 6-31G*, all ≤2.5e-12 in both
     bases. The regression cases use those grids and pin PySCF at 1e-9.
   - Gate C: with SAO on (C2v water), spherical = Cartesian exactly in 6-31G
     and the symmetric-minus-C1 shift is ≤2e-13 in both bases; ESP at four
     points through `planck-dft` is identical in both bases in 6-31G.
   - Cartesian DFT is byte-identical on all 42 `tests/inputs/regression/dft`
     inputs (double hybrids excluded) against the previous `planck-dft`.
   - Refusals verified for gradient, TDDFT, double hybrid, PCM, restart and
     SOSCF.

   Cases: `water_rks_b3lyp_spherical_631g_exact` (smoke),
   `water_rks_b3lyp_spherical_ccpvtz`, `water_rks_hse06_spherical_ccpvtz`,
   `water_cation_uks_pbe_spherical_ccpvtz`, `h2o2_rks_b3lyp_spherical_631gd`,
   `h2o2_cation_uks_pbe_spherical_631gd`,
   `water_rks_spherical_gradient_rejected` (smoke).
2. **Gradients, geometry optimization, frequencies.** The working basis is
   rebuilt per geometry by step 1 and the gradient lift is in place; add the
   XC-nuclear-gradient lift. Gates: energy FD (`tests/dft_gradient_fd.py`),
   PySCF `cart=False` gradients for B3LYP and HSE06 (the long-range
   derivative), water/6-31G* geomopt E_opt against PySCF, one frequency.
3. **TDDFT.** Gate: PySCF `cart=False` TDA and Casida, RKS singlet/triplet and
   UKS, B3LYP and HSE06, at the existing tolerances (4e-4 eV,
   `docs/RS_TDDFT.md`).
4. **SOSCF and KS stability — LANDED 2026-11-06.** Both go through
   `build_ks_orbital_hessian_op` and `build_uks_orbital_hessian_op`
   (`src/dft/ks_orbital_hessian.cpp`); the RKS stability check uses the UKS
   one at the closed-shell reference.
   - **Lift: the MO coefficients, not each trial density.** The builders take
     a `basis` pointer and, when spherical, copy their inputs once with the
     ground density lifted by `to_cartesian` and every MO block replaced by
     Cᵀ C_mo. Then C_vᵀ x C_o in Cartesian *is* the lifted trial density, and
     the packing (Cᵀ C_mo)ᵀ M_cart (Cᵀ C_mo) = C_moᵀ to_working(M_cart) C_mo,
     so the operator bodies needed no change. MO row counts are checked
     against the working basis. Both builders now return `expected` and own
     their inputs (they used to capture them by reference).
   - **Refusals** for SOSCF and stability are gone from
     `validate_spherical_support`.
   - **Cartesian unchanged:** all ten DFT SOSCF / stability / follow cases
     pass.
   - **Stability vs PySCF `cart=False`** (Planck's GBS and Bohr, 150 × 590
     unpruned; `tests/pyscf/spherical_ks_stability_references.py`), all
     ≤ 1.2e-7, the precision of the 1e-4 Davidson residual:

     | case | channel | Planck | PySCF |
     |---|---|---|---|
     | water/cc-pVTZ RKS B3LYP | singlet / triplet | +0.2999874 / +0.2569170 | +0.2999875 / +0.2569171 |
     | water⁺/cc-pVTZ UKS PBE | UKS → UKS | +0.06595028 | +0.0659503 |
     | H₂O₂⁺/6-31G* UKS PBE | UKS → UKS | +0.08105482 | +0.0810548 |
     | H₂ 2.5 Å cc-pVDZ RKS PBE | triplet | −0.09923338 | −0.0992334 |
     | | after follow (UKS) | −0.9997290788 | −0.9997290788 |

   - **SOSCF** (6-31G* H₂O₂ RKS B3LYP and H₂O₂⁺ UKS PBE, 3-step window):
     energies equal the PySCF-pinned DIIS energies to 10 digits. The energy
     alone is not the gate, since DIIS takes over after the window and would
     converge with a wrong Hessian (the S0.6 HF defect hid that way). The
     gradient is: RKS 9.8e-2 → 9.5e-3 → 5.9e-4, UKS 1.6e-1 → 1.5e-2 →
     4.8e-3, against a Cartesian control of 1.4e-1 → 1.0e-2 → 8.8e-4 (RKS)
     and a worse UKS one (trust cap fires, last |g| 5.0e-2).
   - Cases: `water_rks_b3lyp_stability_spherical_ccpvtz`,
     `water_cation_uks_pbe_stability_spherical_ccpvtz` (extended),
     `h2o2_cation_uks_pbe_stability_spherical_631gd`,
     `h2_rks_pbe_triplet_follow_spherical_ccpvdz`,
     `h2o2_{rks_b3lyp,cation_uks_pbe}_soscf_spherical_631gd` (core). New runner
     metric `stability_uks_internal`.
   - KS stability with ECPs now waits only on `DEF2_ECP_SCOPE.md` step 6.
5. **Double-hybrid single points.** Gate: PySCF `cart=False` B2PLYP. DH
   gradients stay rejected with a message naming the spherical basis.
6. **PCM** stays rejected in spherical mode for both binaries; a separate
   item, since HF needs it too.

Then `DEF2_ECP_SCOPE.md` step 6 is a one-line change: remove the ECP refusal
in `read_basis_and_initialize`, because H_core already includes the ECP and
the ECP gradient sits in `one_electron_gradient`.

## Design

One boundary, two helpers, nothing spherical below it.

- Stored densities, Fock matrices and MO coefficients stay in the working
  basis, as in HF. Every density-to-grid and density-to-integral call goes
  through `to_cartesian` on the way in and `to_working` on the way out.
- No per-geometry transform work: C is normalized when the basis is read,
  and the DFT driver already re-reads the basis at each geometry.
- Cost: two n_sph × n_cart × n_cart products per build, negligible next to
  J/K. The grid keeps n_cart AO columns, about 1.2× the work of spherical AOs
  at cc-pVTZ. Evaluating spherical AOs on the grid is not in scope.

## Risks

- **A missed lift that happens to fit.** The grid layer's dimension checks
  catch most mistakes, but a working-basis matrix in an s/p-only basis has
  the Cartesian size and passes silently. That is why Gate A cannot be the
  only gate: Gate B must include d shells.
- **MO count.** With linear-dependence removal the MO count can be below
  `working_nbasis()`. Size occupied/virtual spaces from
  `mo_coefficients.cols()`, never from a basis count.
- **Roundoff, not bitwise.** Moving C·M·Cᵀ between translation units changed
  last-place rounding in S0.3 and S0.5. Cartesian runs must stay bitwise;
  spherical changes are judged by the gates, not by diffs.

## Findings on the way (S0)

Each fixed one shared mechanism that DFT would otherwise have worked around.

**S0.1 — HF spherical ESP charges were silently wrong.** `log_esp_report`
contracted the working-basis density against Cartesian integrals, and
`electrostatic_potential` only checked that the density was square, so the
wrong elements were read, possibly out of bounds. Water/6-31G* RHF spherical
gave O −2.94 and H 1.461 / 1.474 on a symmetric molecule; it now gives
O −0.811 (Cartesian −0.812). The density is lifted first, and
`electrostatic_potential` refuses a density whose size differs from its shell
pairs' AO count. Gate `water_rhf_spherical_esp_points_631gd` matches PySCF
`cart=False` `int1e_grids` to ≤1.8e-9 and fails on the old code. Never caught
because every ESP case was Cartesian and the spherical feature gate screens
calculation types, not property reports.

**S0.2 — spherical post-HF after a direct SCF crashed.** `ensure_eri` logged
an error and returned an empty tensor, and none of its ~20 callers checks for
that: RMP2 after `scf_mode direct` died with SIGSEGV after the SCF converged,
the error lost in the crash. It now builds the Cartesian tensor and
transforms it with `transform_eri_cart_to_sph` (exact; the 150-AO cap is
lifted at this call, since an error return would crash the caller). Gates
`{water_rmp2,h2o_cation_ump2}_spherical_631gd_direct` reproduce the
conventional energies (RMP2 to 1e-13 Eh). Never caught because `auto` picks
conventional SCF below 100 basis functions, so the cache was always there.

**S0.3 — seven hand-written copies of C·M·Cᵀ.** HF's `spherical_direct_fock`
and its UHF twin, `gradient.cpp`'s `lift_ao_matrix_if_spherical`, the 1e
transforms in `hf_driver.cpp` and `working_state.cpp`, multipoles, and SAD's
`transform_density_cart_to_sph` now use the two helpers. Cartesian runs are
bitwise. Spherical runs moved at roundoff: 2 of 1444 S/H_core elements by one
ulp on sih4/def2-SVP. Of 36 spherical/ECP inputs diffed against the old
binary, 32 were identical and 4 moved only in late residuals and the last
kcal/mol digit (~1e-13 Eh). It was not Eigen evaluating `C * (T + V) * Cᵀ`
differently from a materialized sum (bit-identical in isolation), and not
`scf.cpp` codegen (an old-`scf.cpp` hybrid still differed). It is the same
product compiled in a different translation unit.

**S0.4 — the RKS/UKS gradients contracted unlifted P and W.** They now match
the HF paths. Cartesian DFT gradients are byte-identical against the previous
`planck-dft` (B3LYP, HSE06 with long-range exchange, symmetry on, geomopt+freq,
UKS doublet).

**S0.5 — C was renormalized at every geometry for no reason.** Three copies
scaled each row by 1/√((C S_cart Cᵀ)_mm), each building a full Cartesian
overlap first, on the stated premise that the scaling moves with geometry.
It does not: C is block-diagonal by shell, so the row norm involves only
same-centre overlaps. Measured on water/cc-pVTZ with the H atoms moved:
`S_cart` changes, the renormalized C is byte-identical. `read_gbs_basis` now
normalizes C once from the closed-form same-centre overlap,
Σ_pq c_p c_q N_p N_q I(a+a′) I(b+b′) I(c+c′) with I(n) = (n−1)!!/(2p)^{n/2}·√(π/p)
for even n and 0 for odd n, and the copies are deleted.
`planck-working-state-rebuild` checks, on water/cc-pV5Z (s..h) at two
geometries, that the read-time C matches the old normalization to ≤1e-14
relative per row and is geometry-independent; removing the call fails both
checks at 0.999 relative.

**S0.6 — HF spherical stability and HF spherical SOSCF were wrong (found
2026-11-05, `docs/KS_STABILITY_SCOPE.md`).** Both are the audit rule above: a
site that sizes an SCF-basis object must use `working_nbasis()`, and an
operator built in Cartesian must lift its inputs.

- **Stability** (`100675e4`). `analyze_{rhf,uhf}_stability` and
  `follow_instability_and_rerun` sized the ERI transform with the Cartesian
  `_shells.nbasis()`, while `ensure_eri` returns the spherical tensor:
  - Li/6-31G* failed to diagonalize;
  - water⁺/6-31G* reported λ = −15 [UNSTABLE];
  - HI⁺/def2-SVP segfaulted.

  Now `working_nbasis()`.
- **SOSCF.** The "harmless" `rhf_response.cpp` site was live, through RHF
  SOSCF (`build_rhf_cphf_matrix`):
  - It indexed the spherical ERI tensor with Cartesian strides. |g| grew
    0.23 → 0.86 → 0.64, every augmented-Hessian solve hit 30 iterations with
    residual > 1, and only the DIIS handback converged the SCF (19 iterations
    instead of 14). Now `working_nbasis()`.
  - `build_uhf_cphf_matrix` passed working-basis densities to the Cartesian
    `_compute_2e_fock_uhf`. It now wraps the build in `to_cartesian` /
    `to_working`; the RI branch is unchanged.
  - Both now converge superlinearly: RHF 0.23 → 0.039 → 0.0025, UHF 0.26 →
    0.030 → 0.0022. The MP2 gradient Z-vector shares these builders, which
    matters once spherical MP2 gradients are allowed.

PySCF 2.13.0 `cart=False` on water/6-31G*:

| quantity | Planck | PySCF |
|---|---|---|
| RHF real / complex / triplet | +0.3659592 / +0.3352054 / +0.2886527 | +0.3659592 / +0.3352055 / +0.2886527 |
| UHF (water⁺) after the hcore-saddle follow | −75.6104747910 | −75.6104747768 |
| UHF λ at that state (from `guess sad`) | +0.0844190 | +0.0844190 |

Gates, all four failing on the old code:
- `water_rhf_stability_spherical_631gd`;
- `water_cation_uhf_stability_follow_spherical_631gd` (hcore saddle λ −0.0854,
  followed to the ground state);
- `water_{rhf,cation_uhf}_soscf_spherical_631gd` (new `soscf_last_gradient`
  metric ≤ 1e-2, energy equal to DIIS).

## Step log

| step | commit | what |
|---|---|---|
| S0.1 | `3714a4ae` | spherical ESP lifted; `electrostatic_potential` size check; PySCF gate |
| S0.2 | `5da0d7fe` | `ensure_eri` builds the spherical tensor; direct-SCF post-HF gates |
| S0.3 | `eeb6eb34` | `to_cartesian` / `to_working`; seven copies folded |
| S0.4 | `50cbc3b9` | RKS/UKS gradient lift |
| S0.5 | `e565e4fd` | C normalized once at read; three copies deleted |
| S0.6 | `100675e4`, `8745c07a` | HF stability and RHF/UHF CPHF (SOSCF) in the working basis; four gates |
| 1 | `8bc9c7c6` | spherical RKS/UKS single points; PySCF-gated |
| 4 | — (working tree) | SOSCF and KS stability: MO-lifted orbital-Hessian builders; six gates |
| 2, 3, 5, 6 | — | gradients, TDDFT, DH single points, PCM (open) |

## Not in this scope

Spherical AOs evaluated on the grid; DH gradients in a spherical basis; PCM in
a spherical basis; RI-J/K for DFT.

## Depends on / enables

Nothing blocks it. It enables def2 DFT and ECP DFT
(`docs/DEF2_ECP_SCOPE.md` step 6). Step 4 (landed) gave spherical KS
stability (`docs/KS_STABILITY_SCOPE.md`); with ECP step 6 that covers ECP
bases too.
