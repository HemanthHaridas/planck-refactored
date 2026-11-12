# What does RI-JK need before it is usable for routine HF and DFT?

## Headline (2026-11-12)

**RI-JK HF energies are done and exact against PySCF.** `scf_ri_jk` gives
RHF, UHF and ROHF energies, Cartesian or spherical, that equal PySCF density
fitting to every printed digit on every gate molecule. With a JKFIT set the
fitting error is ≤ 2.2e-5 Eh per atom (def2-universal-JKFIT) and ≤ 1.5e-5
(cc-pVTZ-JKFIT), 40–60× below the MP2 RIFIT set it used to need. The fitted
auxiliary basis follows `basis_type`, as in PySCF.

**Not yet usable beyond HF energies.** Derivative workflows are refused (no
RI-JK gradient exists, and the exact-integral one was silently 9.9e-6
Ha/Bohr off). SOSCF and stability still build the full n⁴ tensor RI exists
to avoid. `planck-dft` has no RI-JK at all.

| | Status |
|---|---|
| RHF / UHF / ROHF energy, Cartesian and spherical | **Done**, = PySCF to every printed digit (steps 0, 2, 5) |
| JKFIT fitting sets (def2-universal, cc-pVTZ) | **Done** (step 1) |
| Spherical auxiliary basis for spherical runs | **Done** (`docs/RI_SPHERICAL_AUX.md`) |
| Gradient, geomopt, frequencies | **Refused** by name (step 3); RI-JK gradient is step 7 |
| Full point-group symmetry | Refused; out of scope |
| SOSCF, stability | Run, but build the dense tensor (step 4) |
| `planck-dft` | Not started (step 6) |
| Heavy elements, ECPs, aux L = 6 | **Done**, AgCl = PySCF (step 8a) |

Recommended order for what is left: 4, 6, 7.

Scope, opened 2026-11-10. It expires when the work lands; rewrite it then as
the answer. Status lives in `vault/Status/Open Work.md` (priority 6). The
speed half (pair-factor cache, BLAS, RI-K from C_occ) is scoped in
`docs/ERI_FOCK_BLAS_SCOPE.md`; this doc is about correctness and reach.

How `scf_ri_jk` is wired: the RHF/UHF/ROHF loops in `src/scf/scf.cpp` call
`build_ri_fock_rhf/uhf` (`src/post_hf/ri/ri_eri.cpp`) instead of the dense
or direct Fock build; the fitting set is `mp2_ri_basis`; the boundary is the
RI-JK gate in `src/hf_driver.cpp`. Nothing else checks the flag: every other
consumer of the two-electron operator runs with exact integrals on RI
orbitals (steps 4 and 6).

## Done

### 0. A gate for what exists

Water, water⁺, C1 H2O2 and H2O2⁺, Cartesian 6-31G*, `scf_ri_jk` with
cc-pVDZ-RIFIT. All four equal PySCF `density_fit` to every printed digit,
with the orbital and auxiliary bases both read from Planck's GBS files
(`tests/pyscf/ri_jk_references.py`). Regression cases
`{water,water_cation,h2o2,h2o2_cation}_{rhf,uhf}_rijk_631gd`, tagged core.
No code changed.

| Case | E (RI-JK) | E (exact) | fitting error |
|---|---|---|---|
| water RHF | −76.0076329813 | −76.0066326441 | −1.00e-3 |
| water⁺ UHF | −75.6043871772 | −75.6033570420 | −1.03e-3 |
| H2O2 RHF | −150.7554893339 | −150.7539338384 | −1.56e-3 |
| H2O2⁺ UHF | −150.3709559632 | −150.3692659085 | −1.69e-3 |

One false alarm on the way: PySCF's `1e` guess puts H2O2⁺ UHF on a saddle
0.023 Eh up (unstable), where Planck's `hcore` run is on the stable minimum
(λ_min +0.073). The reference script follows internal UHF instabilities
before comparing.

### 1. JK fitting bases

`basis-sets/def2-universal-JKFIT` (86 elements, H–Rn) and
`basis-sets/cc-pVTZ-JKFIT` (16 elements), from the Basis Set Exchange
(gaussian94). BSE has no cc-pVDZ-JKFIT; cc-pVTZ-JKFIT is the usual fit for
cc-pVDZ too. These are the first contracted auxiliary sets in the tree
(RIFIT is uncontracted). Planck equals PySCF `density_fit` with each set on
all four gate molecules to every printed digit
(`tests/pyscf/ri_jk_references.py <set>`). Fitting error against the exact
energy, Eh:

| Case | cc-pVDZ-RIFIT | def2-universal-JKFIT | cc-pVTZ-JKFIT |
|---|---|---|---|
| water RHF | −1.00e-3 | +2.67e-5 | +1.52e-5 |
| water⁺ UHF | −1.03e-3 | +1.95e-5 | +9.31e-6 |
| H2O2 RHF | −1.56e-3 | +8.94e-5 | +5.89e-5 |
| H2O2⁺ UHF | −1.69e-3 | +8.21e-5 | +5.54e-5 |

Gated by `{water_rhf,h2o2_cation_uhf}_rijk_jkfit_631gd`
(def2-universal-JKFIT).

**No `scf_ri_basis` keyword, deliberately.** The RI-JK and RI-MP2 consumers
share one fitting cache (`_ri_aux_basis`, the 2c metric factor, the 3c
tensor; `ensure_ri_metric_ready`), and `types.h` records one fitting basis
per run as a decision ("splitting it into two keywords would let them
silently disagree"). A JKFIT set already works through `mp2_ri_basis`. A
separate keyword is only worth it for a run that wants RI-JK *and* RI-MP2
each with its own set, and that needs a second cache, not just a field.

### 2. Spherical, by lifting the refusal

Neither planned route was needed. `compute_3c_eri` already transforms the
orbital indices of (μν|Q) to the spherical working basis
(`unpack_transform_repack_3c`), so the fitted factors, `build_ri_j`/`build_ri_k`
and both Fock builders already take and return working-basis matrices. The
refusal in `hf_driver.cpp` was the only obstacle. (A lift inside the Fock
builders was written first and reverted before it ran: it would have
applied the Cartesian→spherical transform twice.)

At first the aux basis was always Cartesian, while PySCF makes it follow
`mol.cart`, so the gates needed a Cartesian-aux PySCF reference. The same day
the fitted aux was made to follow `basis_type` (`docs/RI_SPHERICAL_AUX.md`):
all four gate molecules now equal plain PySCF `density_fit` to every printed
digit (def2-universal-JKFIT, spherical 6-31G*; `*_rijk_spherical_631gd`,
core). Fitting errors 1.6e-5 to 9.3e-5 Eh.

Found on the way: spherical water⁺ UHF from `hcore` lands on a saddle
(λ_min −0.072, RI and exact alike, 0.072 Eh up). That input uses
`guess sad`.

### 3. Refuse the inconsistent gradient

`calculation gradient` with `scf_ri_jk` used to run and return the
exact-integral gradient at the RI orbitals. Water RHF, def2-universal-JKFIT:
it missed a central-difference gradient of the RI energy by **9.9e-6
Ha/Bohr**, which passes the default geometry-optimization threshold (3e-4)
silently; with the MP2 RIFIT set (fitting error ~50× larger) it would
approach it, and frequencies inherit it. The `hf_driver.cpp` gate now
refuses Gradient, GeomOpt, Frequency, GeomOptFrequency and ImaginaryFollow,
naming the reason (checked on gradient, geomopt and frequency; energies
unchanged). Gated by `water_rhf_rijk_gradient_rejected` (core). Lift when
step 7 lands.

### 5. ROHF, one branch

The refusal said the Roothaan Fock "is not wired", but `run_rohf` builds the
same (Ga, Gb) as UHF before assembling the effective Fock. It now primes the
RI cache, skips the dense tensor and calls `build_ri_fock_uhf`, as UHF does;
the refusal is gone. Water⁺ ROHF, def2-universal-JKFIT, equals PySCF
`ROHF(...).density_fit()` to every printed digit, Cartesian
(−75.5989675667) and spherical (−75.5972324349); fitting errors 1.9e-5 and
1.6e-5 Eh. `water_cation_rohf_rijk{,_spherical}_631gd`, core.

## Remaining

### 4. SOSCF and stability (HF): use the RI operator, not the dense tensor

With `scf_ri_jk`, both still build the full AO ERI tensor that RI exists to
avoid:

- **SOSCF** (`scf_soscf_start`) rebuilds it at **every** SOSCF step
  (`RHF CPHF : Building AO ERI tensor`, once per step in the log), for the
  CPHF Hessian action. It converges to the RI energy, because the gradient is
  RI, but at n⁴ memory and per-step cost.
- **Stability** (`stability_check`) builds it once and diagonalizes the
  exact Hessian at the RI orbitals, so λ is off by the fitting error and the
  memory is n⁴.

The RI response already exists: under `mp2_use_ri`, `uhf_response.cpp`
builds the response J/K with `build_ri_fock_uhf`. Route the SOSCF Hessian
action the same way under `scf_ri_jk` (RHF and UHF). For stability, the
explicit A/B matrices come from MO integrals, so build those from the fitted
factors (B_ia,Q products, the RI-MP2 pattern) instead of `ensure_eri`. Gate:
SOSCF and stability on a `*_rijk_*` input log no `Building AO ERI tensor`,
SOSCF reaches the DIIS energy, and λ_min matches PySCF's stability analysis
on a `density_fit()` SCF.

Post-HF methods (MP2, CC, CASSCF) on RI-JK orbitals with exact integrals are
a legitimate combination and stay as they are.

### 6. `planck-dft`

The KS J and the hybrid K go through `_compute_2e_j_direct` /
`_compute_2e_k_direct` (`src/dft/driver.cpp` ~4693–4790). Points the plan
has to respect:

- **Basis.** KS J, K and V_xc are built in Cartesian from lifted densities
  and projected to the working basis at the end. The RI builders take and
  return working-basis matrices (the 3c tensor is already in the working
  basis), so the RI branch uses the working densities
  (`calculator._info._scf`) and adds its J/K *after* the projection; it must
  not be fed the lifted ones.
- **Prefactors.** Keep the unscaled-K contract: each caller applies its own
  prefactor (−½c for RKS, −c per spin for UKS); the exchange energy is
  computed from the same matrices.
- **Range separation.** The RI 2c/3c integrals are Coulomb-kernel only
  (`ri_eri.h`). Short-range erf exchange (HSE06, CAM-B3LYP, ωB97X) needs
  attenuated metric and 3c integrals, the screened-kernel Boys scaling the OS
  and HGP engines already carry. Refuse range-separated functionals with
  `scf_ri_jk` by name first; attenuated RI-K is its own step.
- **Other KS consumers.** `ks_orbital_hessian.cpp` (SOSCF and stability) and
  the DH response/gradient paths build J/K directly; apply step 4's rule
  (route through RI). TDDFT builds a dense tensor: refuse with `scf_ri_jk`
  until a decision on RI response. DH gradients fall under step 3's refusal.
- **MPI.** The RI builders have no MPI distribution, so under `planck-mpi`
  every rank does the whole RI build: correct, no speed-up. Record it; do not
  block on it.

Gates against PySCF `dft.RKS/UKS(...).density_fit()` on Planck's unpruned
grid convention (as the PCM and DH scripts do): water PBE (RI-J only), water
and H2O2 B3LYP RKS, water⁺ B3LYP UKS, one spherical; the range-separated
refusal.

### 7. RI-JK gradient (HF, then KS)

The derivative machinery exists from the RI-MP2 gradient:
`compute_3c_eri_deriv`, `compute_2c_eri_deriv`, and the contraction
`build_ri_two_electron_gradient(gamma3, x_proj, dJ, dV)`, which already
carries the metric-derivative term. What is missing is the SCF three-index
density γ_{μν,P} for J (D_μν·c_P with c = V⁻¹(P|λσ)D_λσ) and for K (from the
occupied fitted factors). Cartesian first: `compute_3c_eri_deriv` refuses a
spherical orbital basis, and the spherical case would lift γ on the orbital
index and apply T_auxᵀ on the aux index (both contractions are linear). Then
UHF/ROHF, then the KS J/K terms. Gate: PySCF `df` gradients
(`mf.density_fit().nuc_grad_method()`) and an FD of the RI energy, as in
step 3; then lift step 3's refusal and add a geomopt case.

### 8. Small and independent

- **Heavy elements — done 2026-11-12.** def2-universal-JKFIT carries i
  functions (L = 6) for the transition metals and heavier (Zn, Ag, Au, …;
  iodine stops at h, so HI would not have tested L = 6). AgCl, def2-SVP with
  the Ag ECP, spherical: two l = 6 aux shells, 383 aux functions; equals
  PySCF `density_fit` (ECP and bases from Planck's GBS) to every printed
  digit, −605.4799497321, fitting error 4.9e-5 Eh.
  `agcl_rhf_rijk_def2svp_ecp`, core. No code changed.
- **`scf_ri_basis`.** Still open (step 1): only worth a second fitting cache
  if a run wants RI-JK and RI-MP2 with different sets.

## Not in this scope

Full-symmetry RI-JK (symmetry-orbit folding), COSX/seminumerical exchange,
attenuated (range-separated) RI-K beyond the refusal, and the performance
work in `ERI_FOCK_BLAS_SCOPE.md`.
