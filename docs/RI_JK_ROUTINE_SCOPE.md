# What does RI-JK need before it is usable for routine HF and DFT?

Scope, opened 2026-11-10. Steps 0–2 are done (2026-11-12); steps 3–8 are
scoped below. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 6). The speed half (pair-factor cache,
BLAS, RI-K from C_occ) is scoped in `docs/ERI_FOCK_BLAS_SCOPE.md`; this doc
is about correctness and reach.

## The state

`scf_ri_jk` (`src/scf/scf.cpp`, `src/hf_driver.cpp`) replaces the J/K build
in the RHF/UHF SCF with RI, Cartesian or spherical, with any fitting set
given as `mp2_ri_basis` (JKFIT sets since step 1). It is refused for ROHF and
full point-group symmetry. It equals PySCF density fitting to every printed
digit (steps 0–2). `planck-dft` has no RI-JK at all, and every consumer
outside the SCF loop still uses exact integrals (see Remaining work).

## Steps

### 0. A gate for what exists — done 2026-11-12

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

The MP2 fitting set costs ~5e-4 Eh per heavy atom, the number step 1's JKFIT
sets have to beat. One false alarm on the way: PySCF's `1e` guess puts
H2O2⁺ UHF on a saddle 0.023 Eh up (unstable), where Planck's `hcore` run is
on the stable minimum (λ_min +0.073). The reference script follows internal
UHF instabilities before comparing.

### 1. JK fitting bases and their own keyword

**Sets and measurement done 2026-11-12; the keyword is open.**

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

So ≤ 2.2e-5 Eh/atom (def2) and ≤ 1.5e-5 (cc-pVTZ), near the expected
~1e-5, and 40–60× better than the MP2 set. Gated by
`{water_rhf,h2o2_cation_uhf}_rijk_jkfit_631gd` (def2-universal-JKFIT). Not
tested: elements past Ne, where the set reaches i functions (L = 6).

**The keyword `scf_ri_basis` is not added, deliberately.** The RI-JK and
RI-MP2 consumers share one fitting cache (`_ri_aux_basis`, the 2c metric
factor, the 3c tensor; `ensure_ri_metric_ready`), and `types.h` records one
fitting basis per run as a decision ("splitting it into two keywords would
let them silently disagree"). A JKFIT set already works through
`mp2_ri_basis`. A separate keyword is only worth it for a run that wants
RI-JK *and* RI-MP2 each with its own set, and that needs a second cache,
not just a field. Decide that when such a run is wanted.

### 2. Spherical — done 2026-11-12, by lifting the refusal

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

## Remaining work (scoped 2026-11-12)

Measured before writing: `scf_ri_jk` is checked only by the SCF loops,
`hf_driver.cpp` and the RI cache (`ri_eri.cpp`). Every other consumer of the
two-electron operator runs with exact integrals on RI orbitals and says
nothing. In order of cost, cheapest first.

### 3. Refuse the inconsistent gradient (now; ~10 lines)

`calculation gradient` with `scf_ri_jk` runs today and returns the
exact-integral gradient at the RI orbitals. Water RHF, def2-universal-JKFIT:
it misses a central-difference gradient of the RI energy by **9.9e-6
Ha/Bohr**. That passes the default geometry-optimization threshold (3e-4)
silently, and with the MP2 RIFIT set (fitting error ~50× larger) it would
approach it; frequencies inherit it. Refuse `scf_ri_jk` for Gradient,
GeomOpt, Frequency, GeomOptFrequency and ImaginaryFollow in the existing
`hf_driver.cpp` gate, naming the reason. Gate: one `*_rijk_gradient_rejected`
case. Lift when step 7 lands.

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

### 5. ROHF (one branch)

The refusal says the Roothaan Fock "is not wired", but `run_rohf` builds the
same (Ga, Gb) as UHF (`_compute_fock_uhf` / `direct_fock_uhf`,
`scf.cpp` ~2170) before assembling the effective Fock. Add the same
`build_ri_fock_uhf` branch and drop the ROHF refusal. Gate: water⁺ ROHF
against PySCF `ROHF(...).density_fit()`.

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
(`mf.density_fit().nuc_grad_method()`) and an FD of the RI energy, as
measured above; then lift step 3's refusal and add a geomopt case.

### 8. Small and independent

- **Heavy elements.** def2-universal-JKFIT reaches i functions (L = 6) past
  Ne; nothing gates it. One def2/ECP molecule (HI/def2-SVP) against PySCF.
- **`scf_ri_basis`.** Still open (step 1): only worth a second fitting cache
  if a run wants RI-JK and RI-MP2 with different sets.

Recommended order: 3 (correctness, minutes), 5 and 8a (small), 4, 6, 7.

## Not in this scope

Full-symmetry RI-JK (symmetry-orbit folding), COSX/seminumerical exchange,
attenuated (range-separated) RI-K beyond the refusal, and the performance
work in `ERI_FOCK_BLAS_SCOPE.md`.
