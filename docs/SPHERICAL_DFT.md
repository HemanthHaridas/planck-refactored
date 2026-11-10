# How does `planck-dft` run in a spherical-harmonic basis?

Answered 2026-11-06 (opened as a scope 2026-10-26). Status lives in
`vault/Status/`.

## Answer

**Everything that touches the grid or the integral engines stays Cartesian.
Only the SCF scaffold lives in the working (spherical) basis**, as in HF: S,
H_core, the orthogonalizer, guess, DIIS, diagonalization, densities, MO
coefficients and the checkpoint. The grid and the integrals are linear in the
AOs, so with C = `Basis::_cart_to_sph` (n_sph × n_cart) every crossing is exact:

| crossing | transform |
|---|---|
| density, energy-weighted density, any MO-built matrix | P_cart = Cᵀ P C (`to_cartesian`) |
| Fock, potential, 1e matrices | M_working = C M_cart Cᵀ (`to_working`) |
| MO coefficients | c_cart = Cᵀ c |

Lifting the **MO coefficients** once is what kept the response code unchanged:
a trial density built from lifted MOs is already Cartesian, and
(Cᵀc)ᵀ M_cart (Cᵀc) = cᵀ to_working(M_cart) c, so projecting a Cartesian
response back onto MO pairs needs no extra transform.

**Supported:** RKS and UKS single points (double hybrids included), analytic
gradients, geometry optimization, frequencies, TDDFT (Casida and TDA, RKS
singlet/triplet and UKS), SOSCF, KS stability analysis and following, and
checkpoint restart (same basis and cross-basis projection).

**Refused by name:** PCM (HF refuses spherical PCM too), double-hybrid
gradient workflows (the DH gradient code is Cartesian-only), and
ImaginaryFollow (not implemented in `planck-dft` in any basis).
`validate_spherical_support` in `src/dft/driver.cpp` is the single gate.

**Audit rule.** A site that sizes a grid or integral object stays Cartesian
(`_shells.nbasis()`); a site that sizes an SCF object uses `working_nbasis()`,
and occupied/virtual spaces come from `mo_coefficients.cols()`. The ~30
`nbasis()` sites in `src/dft/` that check against the AO grid stay Cartesian,
and their dimension checks turn a missed lift into an error.

## Shared machinery

- **C** is built and row-normalized once in `read_gbs_basis` from the
  closed-form same-centre overlap. C is block-diagonal by shell, so its
  normalization does not depend on geometry (S0.5).
- **`BasisFunctions::to_cartesian` / `to_working`** (`src/basis/spherical.h`):
  shape-checked, return `expected`, identity in Cartesian mode. Every crossing
  in HF and DFT goes through them (S0.3).
- **`SCF::compute_working_one_electron`** (`src/scf/working_state.cpp`): the
  working-basis S and H_core (V_ne + ECP), used by both drivers and by every
  per-geometry rebuild.
- **`SCF::project_checkpoint_density`** (same file): the cross-basis restart
  projection, with the cross overlap in the working basis on both sides.
  Shared by `hartree-fock` and `planck-dft`; it refuses on an ECP-core
  mismatch, or in spherical mode on a non-spherical checkpoint.
- **`Correlation::ensure_eri`** returns the spherical tensor in spherical mode,
  building it when the SCF cached none (S0.2). Double-hybrid PT2 needs this
  because the DFT SCF is direct.

## What each workflow needed

| workflow | change |
|---|---|
| single points | `compute_working_one_electron` replaced the DFT driver's 1e copy; SCF objects sized by `working_nbasis()`; the XC evaluation and KS potential lift densities and project matrices |
| gradients, geomopt, frequencies | the XC nuclear gradient receives lifted densities; the HF-like terms already lifted P and W (S0.4) and the per-geometry rebuild already used the working basis |
| SOSCF, KS stability | the two orbital-Hessian builders (`src/dft/ks_orbital_hessian.cpp`) lift the ground density and MO coefficients once; operator bodies unchanged |
| TDDFT | `run_linear_response` lifts MOs and densities, builds Coulomb ERIs in Cartesian, sizes `n_virt` from the MO count; the XC kernel is now analytic (below) |
| double-hybrid single points | nothing beyond the refusal: PT2 is the HF `rmp2_kernel`/`ump2_kernel` on `ensure_eri`'s spherical tensor |
| checkpoint restart | the DFT and HF cross-basis projections became one shared function |

## Validation

All references are PySCF 2.13.0 `cart=False` with Planck's own GBS files, its
CODATA 2018 Bohr, and a matched unpruned 150 × 590 grid. At Planck's default
`fine` grid both bases sit 1e-6 to 2e-5 Eh from PySCF, a floor set by the two
codes' different pruned grids; matching the grids removes it.

| workflow | system | Planck vs PySCF | script |
|---|---|---|---|
| single points | water/cc-pVTZ PBE, B3LYP, HSE06 RKS, PBE UKS; C1 H₂O₂ B3LYP RKS and H₂O₂⁺ PBE UKS /6-31G* | ≤ 2.5e-12 Eh | — |
| gradients | C1 H₂O₂/6-31G*: RKS B3LYP / HSE06, UKS PBE (12 independent components, grid response on) | 1.7e-10 / 1.2e-10 / 3.4e-10 Ha/Bohr | `spherical_ks_gradient_references.py` |
| geomopt + freq | water RKS B3LYP/6-31G* | E_opt 3.2e-7 Eh, freqs ≤ 1.6 cm⁻¹ | same |
| TDDFT | water/6-31G* B3LYP singlet Casida, triplet TDA, HSE06 Casida, UKS water⁺ B3LYP | ≤ 1e-7 eV, oscillator strengths included | `spherical_tddft_references.py` |
| KS stability | water/cc-pVTZ RKS B3LYP, water⁺/cc-pVTZ and H₂O₂⁺/6-31G* UKS PBE, stretched H₂/cc-pVDZ | ≤ 1.2e-7 (Davidson residual); H₂ follow to every digit | `spherical_ks_stability_references.py` |
| double hybrids | water RKS and water⁺ UKS B2PLYP/6-31G* | every printed digit | `spherical_dh_references.py` |

Also checked:
- **Spherical = Cartesian exactly** where the bases span the same space
  (6-31G): ≤ 3.4e-13 Eh over water and C1 H₂O₂, PBE/B3LYP/HSE06, RKS/UKS. With
  SAO on (C₂ᵥ water) the symmetric-minus-C1 shift is ≤ 2e-13 in both bases.
- **Energy finite differences** (fourth order): UKS PBE gradient 1.2e-9
  Ha/Bohr.
- **SOSCF**: same energy as DIIS to 10 digits, and the gradient falls at the
  rate of the Cartesian control (RKS 9.8e-2 → 9.5e-3 → 5.9e-4). The energy
  alone cannot gate SOSCF, since DIIS takes over after the window and would
  converge with a wrong Hessian.
- **Restart**: an STO-3G → 6-31G* projected restart and a same-basis `guess
  full` restart reach the fresh-run energy to 1e-8; a spherical restart from
  a Cartesian checkpoint is refused by name and falls back to H_core.
- **Cartesian unchanged** throughout: byte-identical single points on the 42
  DFT regression inputs at step 1, and every Cartesian SOSCF, stability,
  TDDFT and gradient case still passing after each later step.

Gates: the `*_spherical_*` cases in `tests/regression_cases.json`
(`spherical` and `dft` tags), with `water_rks_b2plyp_gradient_spherical_rejected`
holding the refusal boundary.

## Findings on the way

Each fixed one shared mechanism instead of a DFT-side workaround.

- **S0.1, HF spherical ESP charges were silently wrong.** `log_esp_report`
  contracted the working-basis density against Cartesian integrals (water
  6-31G* O −2.94 instead of −0.811). The density is lifted, and
  `electrostatic_potential` refuses a mis-sized density. Gate
  `water_rhf_spherical_esp_points_631gd`.
- **S0.2, spherical post-HF after a direct SCF crashed.** `ensure_eri` returned
  an empty tensor and RMP2 segfaulted; it now builds the spherical tensor.
  Never caught because `auto` picks conventional SCF below 100 functions.
- **S0.3, seven hand-written copies of C·M·Cᵀ** folded into the two helpers;
  Cartesian bitwise, spherical moved at roundoff (the same product compiled in
  a different translation unit).
- **S0.4, the RKS/UKS gradients contracted unlifted P and W.**
- **S0.5, C was renormalized at every geometry for no reason**: three copies
  rebuilt the Cartesian overlap per geometry on a false premise; now once at
  read, verified to ≤ 1e-14 relative on water/cc-pV5Z at two geometries.
- **S0.6, HF spherical stability and SOSCF were wrong**: ERI transforms sized
  with the Cartesian count (Li failed, water⁺ reported λ = −15, HI⁺
  segfaulted) and the CPHF Hessian indexed the spherical tensor with Cartesian
  strides (only the DIIS handback converged it).
- **TDDFT read past the MO matrix in spherical**: it was never refused, and
  `n_virt` came from the Cartesian count.
- **The TDDFT XC kernel was a finite difference of V_xc.** On libxc's
  discontinuous wPBEh (HSE) potential, grid points within one step of s = 1
  shifted spherical HSE06 roots by up to 4.4e-4 eV. Now the analytic f_xc
  (`prepare_uks_xc_kernel`) for every functional. See
  `docs/WPBEH_S_SCALING_SCOPE.md`.
- **HSE06 energy finite differences cannot validate HSE gradients**, in any
  basis: libxc's `s_scaling_2` jumps at s = 1 (8.0e-6 Ha/Bohr Cartesian miss,
  fully accounted for by `tests/pyscf/hse06_s_scaling_jump.py`). HSE gradients
  are gated against PySCF instead.
- **A 3.2e-8 UKS gradient gap was PySCF's.** Its default `conv_tol_grad` is
  √`conv_tol`; with 1e-9 the gap is 3.4e-10. Every PySCF gradient reference
  here sets it.
- **Two copies of the cross-basis restart projection.** The DFT copy built
  the cross overlap in Cartesian against a working-basis orthogonalizer, had
  no ECP-core check, and put only the alpha part of a UHF checkpoint into an
  RKS density. Replaced by HF's logic, now shared.
- **The regression runner assumed `.hfchk`.** `planck-dft` writes `.dftchk`,
  so no DFT case could be seeded; `checkpoint_path_for` now follows the
  executable.
- **A stale boundary case.** `water_rks_spherical_gradient_rejected` asserted
  the refusal step 2 removed and failed silently until the next smoke run;
  deleted, the DH-gradient refusal holds the boundary now.

## Risks that remain

- **A missed lift that happens to fit.** A working-basis matrix in an s/p-only
  basis has the Cartesian size and passes the dimension checks. Gates with d
  shells (6-31G*, cc-pVTZ) are what catch it.
- **MO count.** The SOSCF paths size `n_virt` from `working_nbasis()`, safe
  only because there is no linear-dependence removal. Switch them to
  `mo_coefficients.cols()` if that lands.
- **Grid cost.** The grid evaluates n_cart AOs, about 1.2× the spherical count
  at cc-pVTZ. Spherical AOs on the grid were not in scope.

## Commit log

| step | commit | what |
|---|---|---|
| S0.1–S0.5 | `3714a4ae`, `5da0d7fe`, `eeb6eb34`, `50cbc3b9`, `e565e4fd` | ESP lift, spherical `ensure_eri`, the two helpers, gradient lift, C at read |
| S0.6 | `100675e4`, `8745c07a` | HF stability and CPHF in the working basis |
| single points | `8bc9c7c6` | spherical RKS/UKS |
| SOSCF + stability | `7cf32683` | MO-lifted orbital-Hessian builders |
| gradients | `a39f5ee5` | XC nuclear-gradient lift |
| TDDFT | `10ed19ef` | lifted MOs, analytic f_xc kernel |
| DH + restart | (this commit) | DH single points; shared restart projection |

## Not done

PCM in a spherical basis (both binaries); double-hybrid gradients in a
spherical basis; RI-J/K for DFT. DFT with ECPs is unblocked for every
supported workflow (`docs/DEF2_ECP.md` step 6).
