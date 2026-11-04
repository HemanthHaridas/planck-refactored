# Can `planck-dft` run in a spherical-harmonic basis?

Research scope, opened 2026-10-26, rewritten 2026-11-04 after a code audit.
S0.1 and S0.2 have landed; the rest is open. It expires when the work lands; rewrite it then as the
answer. Status lives in `vault/Status/Open Work.md`.

## Why it matters

Routine DFT is run in spherical (pure) functions: cc-pVXZ and def2 are defined
that way, and a Cartesian d shell carries an extra s-type function, so the two
give different energies. `planck-dft` rejects `basis_type spherical` at the top
of `DFT::Driver::run` (`src/dft/driver.cpp`). The HF binary already runs
spherical SCF, gradients, geometry optimization and frequencies. Spherical DFT
is also the gate for DFT with ECPs (`docs/DEF2_ECP_SCOPE.md` step 6), since
def2 is spherical.

## Answer in one paragraph

Keep everything that touches the grid or the integral engines Cartesian, and
move only the SCF scaffold (S, H_core, orthogonalizer, guess, DIIS,
diagonalization, densities, MO coefficients, checkpoint) to the working basis,
as HF does. The grid layer is linear in the AOs, so with P_cart = Cᵀ P C the
density, its gradient and V_xc are exact, V_xc,sph = C V_xc,cart Cᵀ. The same
holds for J, K, the XC response kernel, PCM and MO coefficients
(c_cart = Cᵀ c_sph). The work is placing that boundary at about ten call sites
in `driver.cpp`, plus four shared fixes outside `src/dft/` (S0 below). One of
those is a wrong answer HF ships today.

## What exists (verified 2026-11-04)

- **The transform.** `read_gbs_basis` builds `Basis::_cart_to_sph` (C, n_sph ×
  n_cart) whenever `basis_type spherical`, so the DFT driver already gets it.
  It must be renormalized against the Cartesian overlap at every geometry.
- **`SCF::rebuild_basis_dependent_state`** (`src/scf/working_state.cpp`) does
  the renormalization, refreshes integral symmetry, and stores working-basis
  `_overlap` and `_hcore` (V_ne + ECP via `_compute_potential`). It returns the
  shell pairs. The DFT driver does not call it: `compute_one_electron_terms`
  (`driver.cpp`) is a Cartesian-only copy of its body.
- **Lift helpers.** `BasisFunctions::lift_density_sph_to_cart` (public,
  `src/basis/spherical.h`) does Cᵀ M C. There is no public projection back
  (C M Cᵀ): HF has a file-local `spherical_direct_fock` template plus an
  inline UHF copy in `src/scf/scf.cpp`, and `gradient.cpp` has its own
  `lift_ao_matrix_if_spherical`.
- **Already spherical-aware:** the SAD guess (it maps into the working basis
  against `_overlap`), SAO blocking (`build_sao_basis`), the checkpoint
  (stores the basis type), multipoles (`populations/multipole.cpp`), and
  `one_electron_gradient` (its callers pass lifted densities).

## What is missing

| site | today | needed |
|---|---|---|
| `run()` entry | rejects spherical | remove, per step |
| `prepare_current_geometry` | re-reads the basis, then `build_shellpairs` + `compute_one_electron_terms` (Cartesian S, H) | call `rebuild_basis_dependent_state`; delete `compute_one_electron_terms` |
| `prepare_current_geometry` (3), `run_ks_scf_scaffold`, `diagonalize_in_ao_basis`, `project_ks_restart_density` | size by `_shells.nbasis()` | `working_nbasis()` |
| `evaluate_current_density_and_xc` | passes the stored density to the grid | lift at entry |
| `assemble_current_ks_potential` | J, K (all kernels) and V_xc from the stored density | lift densities at entry, project J/K/V_xc at exit; builders stay Cartesian |
| SOSCF window (`run_ks_scf_scaffold`) | trial δP into the analytic XC Hessian and direct J/K response | same lift/project around each linear-in-density build |
| PCM reaction field | stored density | lift / project; HF rejects spherical PCM, so keep it rejected in both until gated |
| `compute_rks_gradient` / `compute_uks_gradient` (`src/gradient/gradient.cpp`) | P and W **not lifted**, unlike `compute_rhf_gradient` | lift P, W (W built in the working basis, lifted once) |
| XC nuclear gradient (`compute_analytic_ks_gradient`) | stored densities to `compute_xc_nuclear_gradient_{rks,uks}` | lift |
| TDDFT (`run_linear_response`) | `n_virt = nbasis − n_occ`; MO slices; `ensure_eri` | `n_virt` from `C.cols()`; the spherical tensor from S0.2 (or lift C and use the Cartesian tensor) |
| double-hybrid PT2 (`rmp2_kernel` / `ump2_kernel`) | `ensure_eri` | S0.2 |
| DH gradients (`dh_pt2_gradient.cpp`, `udh_*`) | require a Cartesian basis explicitly | keep rejected in this scope |
| ESP report (`log_esp_report_dft`) | shared `log_esp_report` | S0.1 |

The other 30 `nbasis()` sites in `src/dft/` (`ao_grid.h`, `xc_grid.cpp`,
`ks_matrix.cpp`, `dft_gradient.cpp`, `analytic_hessian.cpp`, `dh_*`, `udh_*`)
check against the AO grid's dimension and stay Cartesian. They need no edit
once their inputs arrive lifted, and their existing dimension checks catch a
missed lift as an error, not as undefined behaviour. **Audit rule:** a site
that sizes a grid or integral object stays Cartesian; a site that sizes an SCF
object becomes `working_nbasis()`.

## Design

One boundary, two helpers, nothing spherical below it.

- `to_cartesian(calc, M) = Cᵀ M C` and `to_working(calc, M) = C M Cᵀ`, in
  `src/basis/spherical.{h,cpp}` beside `lift_density_sph_to_cart`. They
  return M unchanged in Cartesian mode, so every Cartesian run stays bitwise
  identical (the gate for step 1). Then fold HF's `spherical_direct_fock`, its
  UHF inline copy and `gradient.cpp`'s `lift_ao_matrix_if_spherical` onto
  them, so there is one implementation instead of four.
- The stored densities, Fock matrices and MO coefficients stay in the working
  basis, as in HF. Every density-to-grid and density-to-integral call lifts on
  the way in and projects on the way out.
- Cost: two n_sph × n_cart × n_cart products per build, negligible next to
  J/K. The grid keeps n_cart AO columns, about 1.2× the work of spherical AOs
  at cc-pVTZ. Evaluating spherical AOs on the grid directly is not in scope.

## Shared fixes first (S0)

Each fixes one mechanism that DFT would otherwise work around:

1. **HF spherical ESP was wrong — FIXED 2026-11-04.** `SCF::log_esp_report`
   (`src/populations/esp.cpp`) contracts the working-basis density against
   Cartesian integrals. Measured on water/6-31G* RHF, `basis_type spherical`:
   O −2.94, H 1.461 / 1.474 (asymmetric on a symmetric molecule); Cartesian
   gives O −0.81. No error is raised (only squareness is checked), and the
   index mismatch may read past the end. Fix: lift the total density in
   `log_esp_report`, and have `electrostatic_potential` check the density
   dimension against the Cartesian basis. CHELPG goes through the same
   function. Landed as written: `log_esp_report` lifts the total density,
   and `electrostatic_potential` refuses a density whose size differs from the
   AO count its shell pairs span. Gate `water_rhf_spherical_esp_points_631gd`
   (smoke) matches PySCF `cart=False` `int1e_grids` to ≤1.8e-9 at five
   off-axis points, and fails on the pre-fix code. The Connolly charges are
   now O −0.811 (Cartesian −0.812).
2. **`Correlation::ensure_eri` in spherical mode with nothing cached — FIXED
   2026-11-04** (`src/post_hf/integrals.cpp`). It logged an error and returned
   an empty tensor, and none of its ~20 callers checks for that: spherical
   RMP2 after `scf_mode direct` died with SIGSEGV after a converged SCF, the
   error message lost in the crash. It now builds the Cartesian tensor and
   transforms it with `transform_eri_cart_to_sph` (exact; the 150-AO cap is
   lifted there, since the Cartesian n⁴ build is the dominant cost and an
   error return would crash the caller). Gates
   `water_rmp2_spherical_631gd_direct` and
   `h2o_cation_ump2_spherical_631gd_direct` reproduce the conventional
   energies (RMP2 to 1e-13 Eh) and crash on the old code. This serves TDDFT,
   double-hybrid PT2, and every spherical post-HF method after a direct SCF.
3. **The two helpers above**, with the HF consolidation (bitwise on the HF
   spherical gates: water/6-31g* gradient, geomopt, freq, f/g/h shells).
4. **RKS/UKS gradient lift** in `compute_rks_gradient` /
   `compute_uks_gradient` (identity in Cartesian, so no DFT case moves).

## Steps

1. **S0.3–S0.4** (S0.1, S0.2 done). HF-side gates as listed. All Cartesian DFT cases stay
   bitwise.
2. **RKS and UKS single points**: LDA, GGA, global and range-separated
   hybrids, with SAD and SAO symmetry. Switch `prepare_current_geometry` to
   `rebuild_basis_dependent_state`, apply the sizing table, put the boundary
   on the two seams, and lift the entry rejection for SinglePoint only.
   - Gate A, exact: in a basis with only s and p shells (6-31G), spherical and
     Cartesian span the same space and the grid is identical, so energies must
     agree to ≤1e-10, RKS and UKS.
   - Gate B, PySCF 2.13.0 `cart=False`: water and a C1 molecule, 6-31G* and
     cc-pVTZ, PBE / B3LYP / HSE06, RKS and a UKS doublet. Set the tolerance by
     first running the same comparison with `cart=True`, which measures the
     grid floor between the two codes. Spherical must land inside that floor.
   - Gate C: the ESP case from S0.1 in `planck-dft`, and a spherical SAO
     (`use_symm`) run equal to C1.
3. **Gradients, geometry optimization, frequencies.** Inherited from
   `rebuild_basis_dependent_state` and the S0.4 lift; also lift at the XC
   nuclear gradient. Gates: energy FD (`tests/dft_gradient_fd.py`), PySCF
   `cart=False` gradients for B3LYP and HSE06 (the long-range derivative),
   water/6-31G* geomopt E_opt against PySCF, and one frequency.
4. **TDDFT.** `n_virt` from `C.cols()`, and the tensor from S0.2. Gate: PySCF
   `cart=False` TDA and Casida, RKS singlet/triplet and UKS, B3LYP and HSE06,
   at the existing TDDFT tolerances (4e-4 eV, `docs/RS_TDDFT.md`).
5. **SOSCF**: lift/project inside the Newton step's h_op. Gate: same energy
   as DIIS to 10 digits, RKS and UKS, as in `docs/SOSCF_DFT.md`.
6. **Double-hybrid single points** through S0.2. Gate: PySCF `cart=False`
   B2PLYP. DH gradients stay rejected with a message naming the spherical
   basis.
7. **PCM** stays rejected in spherical mode for both binaries; it is a
   separate item (it also needs HF).

Then `DEF2_ECP_SCOPE.md` step 6 is a one-line change: remove the ECP refusal
in `read_basis_and_initialize`, because H_core already includes the ECP and
step 5 put the ECP gradient into `one_electron_gradient`.

## Risks

- **A missed lift.** The grid layer's dimension checks turn most mistakes
  into errors, but a matrix that happens to be square of the right size, for
  example a working-basis matrix in an s/p-only basis, would pass silently.
  This is why Gate A cannot be the only gate: run Gate B with d shells.
- **The renormalized C.** Every lift must use the C that
  `rebuild_basis_dependent_state` just renormalized. A stale copy taken before
  that call silently breaks every element (the trap the HF geomopt and freq
  loops hit; vault `Implementation/Gradients and GeomOpt`, "Geomopt + freq
  inner-loop rebuild").
- **MO count.** With linear-dependence removal the MO count can be below
  `working_nbasis()`. Size occupied/virtual spaces from `C.cols()`, never
  from a basis count.

## Not in this scope

Spherical AOs evaluated on the grid; DH gradients in a spherical basis; PCM in
a spherical basis; RI-J/K for DFT.

## Depends on / enables

Nothing blocks it. It enables def2 DFT and ECP DFT (`docs/DEF2_ECP_SCOPE.md`
step 6), and S0.1 fixes a wrong HF answer independently.
