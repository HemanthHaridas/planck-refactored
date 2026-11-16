# What does RI-JK need before it is usable for routine HF and DFT?

Scope, opened 2026-11-10, last revised 2026-11-16. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 6). The speed half (BLAS, RI-K from
C_occ) is in `docs/ERI_FOCK_BLAS_SCOPE.md`; this doc is about correctness
and reach, plus the one speed item (B0) that KS RI-JK cannot do without.

## Where it stands (2026-11-16)

**HF energies are done and exact against PySCF.** `scf_ri_jk` gives RHF, UHF
and ROHF energies, Cartesian or spherical, equal to PySCF density fitting to
every printed digit on every gate molecule, heavy elements with ECPs
included. With a JKFIT set the fitting error is ≤ 2.2e-5 Eh per atom
(def2-universal-JKFIT) and ≤ 1.5e-5 (cc-pVTZ-JKFIT), 40–60× below the MP2
RIFIT set. SOSCF and stability use the fitted operator.

**RHF gradients, geometry optimization and frequencies work** for a
Cartesian basis, equal to PySCF's density-fitted gradient to every printed
digit. UHF, ROHF and spherical RI-JK derivatives are still refused by name.
`planck-dft` refuses `scf_ri_jk` (it used to ignore it).

| | Status |
|---|---|
| RHF / UHF / ROHF energy, Cartesian and spherical | **Done**, = PySCF (steps 0, 2, 5) |
| JKFIT fitting sets (def2-universal, cc-pVTZ) | **Done** (step 1) |
| Spherical auxiliary basis for spherical runs | **Done** (`docs/RI_SPHERICAL_AUX.md`) |
| Heavy elements, ECPs, aux L = 6 | **Done**, AgCl = PySCF (step 5a) |
| SOSCF, stability | **Done**, eigenvalues = PySCF DF (step 4) |
| Streamed RI derivative contraction | **Done**, shared with RI-MP2 (step 7) |
| RI-MP2 frequencies | **Done** (step 8) |
| RI-JK RHF gradient, geomopt, frequencies (Cartesian) | **Done**, = PySCF (step 9) |
| RI-JK UHF/ROHF and spherical derivatives | **Refused** by name; steps 10, 16 |
| `planck-dft` | **Refused** by name (step 6); steps 11–15 |
| Full point-group symmetry | Refused; out of scope |

## How `scf_ri_jk` is wired

The RHF/UHF/ROHF loops in `src/scf/scf.cpp` call `build_ri_fock_rhf/uhf`
(`src/post_hf/ri/ri_eri.cpp`) instead of the dense or direct Fock build. The
fitting set is `mp2_ri_basis`, and RI-JK and RI-MP2 share one fitting cache
(`_ri_aux_basis`, the 2c metric factor, the 3c tensor), which invalidates
itself when the geometry moves. The RI builders take and return
working-basis matrices: `compute_3c_eri` already transforms the orbital
indices to the spherical working basis.

The boundaries: the RI-JK gate in `src/hf_driver.cpp` (full symmetry, and
derivative workflows other than Cartesian RHF without correlation or PCM) and the refusal at the top of `DFT::Driver::run`. HF
SOSCF and stability check the flag (step 4). Post-HF methods (MP2, CC,
CASSCF) on RI-JK orbitals keep exact integrals, which is a legitimate
combination.

## Remaining, in dependency order

| Step | What | Needs |
|---|---|---|
| 10 | RI-JK gradient, UHF and ROHF | — |
| 11 | B0: cache the RI pair factors | — |
| 12 | KS RI-J, pure functionals | 6, 11 |
| 13 | KS RI-K, global hybrids | 12 |
| 14 | KS SOSCF and stability on RI; refusals for TDDFT and DH | 13 |
| 15 | KS RI-JK gradient, Cartesian | 13 |
| 16 | Spherical RI gradients: RI-JK HF and KS, and RI-MP2 | 10, 15 |

10 and 11 have no open prerequisites; 11 is on the critical path to KS.

### 10. RI-JK gradient, UHF and ROHF

Already wired: `compute_uhf_gradient` and `compute_rohf_gradient` call step
9's `build_ri_scf_two_electron_gradient(calc, P_a, P_b)` under `_ri_jk`. What
is missing is the gate. Gates on water⁺ (UHF, ROHF) and C1 H2O2⁺ (UHF) with
`tests/pyscf/ri_jk_gradient_references.py` (it already reads `scf_type`);
then narrow the `hf_driver.cpp` refusal and replace
`water_cation_uhf_rijk_gradient_rejected`.

### 11. B0: cache the RI pair factors

`build_ri_j` and `build_ri_k` call `build_ri_pair_factors` on every call,
~61 % of the RI-JK Fock build (`docs/ERI_FOCK_BLAS_SCOPE.md`). It does not
depend on the density. KS RI-J would pay it every iteration and probably not
beat the direct build, so it goes before step 12. The `*_rijk_*` cases gate
it (energies unchanged).

### 12. KS RI-J, pure functionals

In the KS assembly (`src/dft/driver.cpp`, the `_compute_2e_j_direct` call),
under `_ri_jk` replace the direct Coulomb by `build_ri_j(calculator,
P_working_total)`. Prime the cache once with `ensure_ri_3c_ready` before the
SCF loop, as `scf.cpp` does. Narrow step 6's refusal to what is still
missing: hybrids, and gradient, geomopt and frequency workflows (the exact KS
gradient at RI orbitals would be inconsistent, as it was for HF in step 3).

- **Basis.** KS J, K and V_xc are built in Cartesian from lifted densities
  and projected to the working basis at the end. The RI builders work in the
  working basis, so the RI branch takes the working densities
  (`calculator._info._scf`) and adds its J *after* the projection. Feeding it
  the lifted densities, or projecting its J again, is wrong.
  `combine_ks_potential` needs the Coulomb passed separately, or the
  projection loop must skip it.
- **MPI.** The RI builders have no MPI distribution, so under `planck-mpi`
  every rank does the whole RI build: correct, no speed-up. Record it; do not
  block on it.

Gates against PySCF `dft.RKS/UKS(...).density_fit()` on Planck's unpruned
grid convention (as the PCM and DH scripts do): water PBE RKS, water⁺ PBE
UKS, one spherical.

### 13. KS RI-K, global hybrids

`build_ri_k` per spin, unscaled. Keep the unscaled-K contract: each caller
applies its own prefactor (−½c for RKS, −c per spin for UKS), and the
exchange energy comes from the same matrices. Share one
`build_ri_3index_unpacked` between J and K. Range-separated functionals stay
refused: the RI 2c/3c integrals are Coulomb-kernel only, and attenuated RI-K
is outside this scope. Gates: water and H2O2 B3LYP RKS, water⁺ B3LYP UKS, the
range-separated refusal.

### 14. KS consumers

`ks_orbital_hessian.cpp` (SOSCF and stability) builds J/K directly at four
sites: route them through RI under `_ri_jk`, as step 4 did for HF. Refuse by
name with `scf_ri_jk`: TDDFT (`run_linear_response` builds a dense ERI
tensor; RI response needs its own decision) and double hybrids
(`dh_pt2_gradient.cpp`, `udh_ks_response.cpp`; RI-DH is its own item). Gates:
as step 4, against PySCF stability on `density_fit()`.

### 15. KS RI-JK gradient, Cartesian

Already wired for global hybrids: `compute_rks_gradient` passes its exact-
exchange fraction to the closed-shell builder, whose RI branch calls step 9's
helper with it; the UKS gradient needs the same. The XC terms are untouched. Gate: PySCF
`density_fit()` KS gradients and an FD of the RI energy. Lift the KS
derivative-workflow refusal kept since step 12.

### 16. Spherical RI gradients

The derivative loops refuse a spherical orbital basis (`for_each_3c_deriv`)
and a spherical aux basis (`for_each_2c_deriv`). Lift γ3 on the orbital
index (Tᵀ γ T per aux column) and apply T_auxᵀ on the aux index before the
Cartesian contraction; both contractions are linear. Done once in step 7's
contraction, it serves RI-JK HF, RI-JK KS and RI-MP2 alike. Lift the
spherical refusals together, including
`water_ri_rmp2_spherical_gradient_rejected`.

### After this scope

- **G6, profile the RI 2c/3c integrals**
  (`docs/GRADIENT_INTEGRAL_PERFORMANCE.md`). The derivative loops are still
  serial: benzene 6-31G*/JKFIT takes 29 s for one gradient (step 9).
- **RI-DH (O9)**: needs steps 12–15.
- **`scf_ri_basis`**: see step 1.

## Done

### 0. A gate for what existed

Water, water⁺, C1 H2O2 and H2O2⁺, Cartesian 6-31G*, `scf_ri_jk` with
cc-pVDZ-RIFIT. All four equal PySCF `density_fit` to every printed digit,
with the orbital and auxiliary bases both read from Planck's GBS files
(`tests/pyscf/ri_jk_references.py`). Regression cases
`{water,water_cation,h2o2,h2o2_cation}_{rhf,uhf}_rijk_631gd`, core. No code
changed.

| Case | E (RI-JK) | E (exact) | fitting error |
|---|---|---|---|
| water RHF | −76.0076329813 | −76.0066326441 | −1.00e-3 |
| water⁺ UHF | −75.6043871772 | −75.6033570420 | −1.03e-3 |
| H2O2 RHF | −150.7554893339 | −150.7539338384 | −1.56e-3 |
| H2O2⁺ UHF | −150.3709559632 | −150.3692659085 | −1.69e-3 |

A false alarm on the way: PySCF's `1e` guess puts H2O2⁺ UHF on a saddle
0.023 Eh up, where Planck's `hcore` run is on the stable minimum (λ_min
+0.073). The reference script follows internal UHF instabilities before
comparing.

### 1. JK fitting bases

`basis-sets/def2-universal-JKFIT` (86 elements, H–Rn) and
`basis-sets/cc-pVTZ-JKFIT` (16 elements), from the Basis Set Exchange
(gaussian94). BSE has no cc-pVDZ-JKFIT; cc-pVTZ-JKFIT is the usual fit for
cc-pVDZ too. These are the first contracted auxiliary sets in the tree
(RIFIT is uncontracted). Planck equals PySCF `density_fit` with each set on
all four gate molecules to every printed digit. Fitting error against the
exact energy, Eh:

| Case | cc-pVDZ-RIFIT | def2-universal-JKFIT | cc-pVTZ-JKFIT |
|---|---|---|---|
| water RHF | −1.00e-3 | +2.67e-5 | +1.52e-5 |
| water⁺ UHF | −1.03e-3 | +1.95e-5 | +9.31e-6 |
| H2O2 RHF | −1.56e-3 | +8.94e-5 | +5.89e-5 |
| H2O2⁺ UHF | −1.69e-3 | +8.21e-5 | +5.54e-5 |

Gated by `{water_rhf,h2o2_cation_uhf}_rijk_jkfit_631gd`.

**No `scf_ri_basis` keyword, deliberately.** RI-JK and RI-MP2 share one
fitting cache, and `types.h` records one fitting basis per run as a decision
("splitting it into two keywords would let them silently disagree"). A JKFIT
set already works through `mp2_ri_basis`. A separate keyword is worth it only
for a run that wants RI-JK and RI-MP2 each with its own set, and that needs a
second cache, not just a field.

### 2. Spherical, by lifting the refusal

Neither planned route was needed. `compute_3c_eri` already transforms the
orbital indices of (μν|Q) to the spherical working basis, so the fitted
factors, `build_ri_j`/`build_ri_k` and both Fock builders already worked in
the working basis. The refusal in `hf_driver.cpp` was the only obstacle. A
lift inside the Fock builders was written first and reverted before it ran:
it would have applied the Cartesian→spherical transform twice.

The fitted aux was then made to follow `basis_type`, as PySCF's follows
`mol.cart` (`docs/RI_SPHERICAL_AUX.md`). All four gate molecules equal plain
PySCF `density_fit` to every printed digit (def2-universal-JKFIT, spherical
6-31G*; `*_rijk_spherical_631gd`, core). Fitting errors 1.6e-5 to 9.3e-5 Eh.

Found on the way: spherical water⁺ UHF from `hcore` lands on a saddle
(λ_min −0.072, RI and exact alike, 0.072 Eh up). That input uses `guess sad`.

### 3. Refuse the inconsistent gradient

`calculation gradient` with `scf_ri_jk` used to return the exact-integral
gradient at the RI orbitals. Water RHF, def2-universal-JKFIT: it missed a
central-difference gradient of the RI energy by **9.9e-6 Ha/Bohr**, inside
the default geometry-optimization threshold (3e-4), so silent; with the MP2
RIFIT set (fitting error ~50× larger) it approaches it, and frequencies
inherit it. The `hf_driver.cpp` gate now refuses gradient, geomopt,
frequency, geomoptfreq and imaginaryfollow, naming the reason. Gated by
`water_rhf_rijk_gradient_rejected` (core). Lifted by steps 9–10.

### 4. SOSCF and stability use the fitted operator

Before, RHF SOSCF rebuilt the full AO ERI tensor at every step for its CPHF
Hessian, UHF SOSCF used exact direct J/K, and stability diagonalized the
exact Hessian at the RI orbitals, ~1e-6 off the fitted one. Both now use
machinery RI-MP2 already had:

- The CPHF builders (`rhf_response.cpp`, `uhf_response.cpp`) take their
  fitted route under `RI::fitted_response(calc)` = `mp2_use_ri || scf_ri_jk`.
- Stability (`stability.cpp`) reads its MO blocks through `transform_eri_ri`
  under `scf_ri_jk`. It keys on `scf_ri_jk` alone, because it analyses the
  SCF operator, which RI-MP2 does not change.

Water RHF (def2-universal-JKFIT): the three eigenvalues equal PySCF's
stability analysis on `mf.density_fit()` to the 7 printed digits (real
internal 0.3825292 = PySCF 1.5301167/4, complex 0.3534208, triplet
0.3084010). H2O2⁺ UHF: 0.07322713 = PySCF `uhf_internal` 0.14645447/2. SOSCF
reaches the RI energy, and neither log builds the AO tensor. Gated by
`{water_rhf,h2o2_cation_uhf}_rijk_soscf_stability_631gd` (core); the old code
fails both.

### 5. ROHF, one branch

The refusal said the Roothaan Fock "is not wired", but `run_rohf` builds the
same (Ga, Gb) as UHF before assembling the effective Fock. It now primes the
RI cache and calls `build_ri_fock_uhf`, as UHF does. Water⁺ ROHF,
def2-universal-JKFIT, equals PySCF `ROHF(...).density_fit()` to every
printed digit, Cartesian (−75.5989675667) and spherical (−75.5972324349);
fitting errors 1.9e-5 and 1.6e-5 Eh. `water_cation_rohf_rijk{,_spherical}_631gd`,
core.

### 5a. Heavy elements, ECPs, aux L = 6

Recorded in `vault/Status/Completion.md` as "step 8a", from an earlier
numbering. def2-universal-JKFIT carries i functions (L = 6) for the
transition metals and heavier (Zn, Ag, Au, …); iodine stops at h, so HI would
not have tested L = 6. AgCl, def2-SVP with the Ag ECP, spherical: two l = 6
aux shells, 383 aux functions; equals PySCF `density_fit` (ECP and bases from
Planck's GBS) to every printed digit, −605.4799497321, fitting error 4.9e-5
Eh. `agcl_rhf_rijk_def2svp_ecp`, core. No code changed.

### 6. `planck-dft` refuses `scf_ri_jk`

The keyword was parsed in the shared `io.cpp`, but nothing under `src/dft/`
read it: a KS input with `scf_ri_jk true` ran exact integrals and said
nothing. `DFT::Driver::run` now refuses it by name before anything else runs.
`water_rks_pbe_rijk_rejected` (core). Narrowed by step 12.

### 7. Streamed RI derivative contraction

`compute_3c_eri_deriv` stored 3·natoms matrices of npair × naux, and
`compute_2c_eri_deriv` 3·natoms of naux × naux, before
`build_ri_two_electron_gradient` contracted them. At a routine size (20
atoms, nb = 300, naux ≈ 1500 JKFIT) dJ alone is ≈ 32 GB.

Each class now has one loop, `for_each_3c_deriv` / `for_each_2c_deriv`
(`ri_eri.cpp`), that hands every derivative element to a callback.
`build_ri_two_electron_gradient(calculator, gamma3, x_proj, bra_prefolded)`
contracts through them and stores nothing beyond natoms × 3. The
stored-tensor functions are thin scatters over the same loops, kept for the
FD test (`planck-ri-3c-derivative-fd`) and as an oracle. The loops check only
that the aux basis is loaded, not `mp2_use_ri`, so RI-JK can call them. RI-MP2
and RI-UMP2 (`mp2_gradient.cpp`) use the streamed form.

`planck-ri-two-electron-gradient` compares the streamed contraction with the
stored-tensor formula: 1.4e-14 on values ~2.6; weakening the metric ½ by
0.02 % fails it (6.7e-5). RI-MP2/UMP2 gradients and the RI-MP2 geomopt are
identical to the printed digits before and after (one kcal/mol digit moves
at 8e-13). The memory saving shows only on an RI-JK gradient, because the
RI-MP2 path still holds its own nao⁴ `dm2buf`; step 9 measures it.

### 8. RI-MP2 frequencies

No new code: the semi-numerical Hessian runs the RI-MP2 gradient through
`Opt::run_scf_gradient`, the per-geometry step geometry optimization already
uses, and the RI cache invalidates when the geometry moves. The frequency
guard in `hf_driver.cpp` is gone, and the RI-MP2 workflow gate admits
frequency, geomoptfreq and imaginaryfollow for RHF and UHF.

PySCF 2.15.0 still has no DF-MP2 gradient, so the reference
(`tests/pyscf/ri_mp2_frequency_references.py`) builds one as a fourth-order
FD (1e-3 Bohr) of the RI-MP2 energy (dense SCF + DF-MP2, aux from Planck's
GBS), then central-differences it at Planck's h = 5e-3 Bohr. A plain second
difference of the energy has a different O(h²) error and was 0.08 cm⁻¹ off
at the same step (1480.52 against 1480.60; 1480.70 at h = 1e-3), which is why
the reference uses Planck's own scheme.

| Case | Reference, cm⁻¹ | Planck |
|---|---|---|
| water RI-RMP2/STO-3G | 1480.60 / 5000.85 / 5418.23 | identical |
| water⁺ RI-UMP2/STO-3G | 1347.72 / 4689.10 / 4868.83 | 1347.73 (last digit), rest identical |

RI-MP2 energies equal to every printed digit. Gated by
`water_ri_rmp2_freq_sto3g` (was `water_ri_rmp2_freq_rejected`) and
`water_radical_cation_ri_ump2_freq_sto3g`, ±0.05 cm⁻¹. With the RI cache
invalidation disabled, both fail by up to 41 cm⁻¹. All 33 RI regression
cases pass.

### 9. RI-JK gradient, RHF, Cartesian

Every SCF gradient takes its 2e term from `two_electron_gradient(gamma_fn)`,
which takes Γ quartet by quartet and cannot carry RI. One sibling,
`build_ri_scf_two_electron_gradient(calc, Pa, Pb, k)` (`ri_eri.cpp`), covers
the Γ all three references share, Γ = 2 Pt⊗Pt − 2k (Pa·Pa + Pb·Pb) (RHF:
Pa = Pb = P/2; k = 1 for HF, the exact-exchange fraction for a hybrid). It
forms

  γ3_{μν,Q} = Pt_μν c_Q − k (Pa X[Q] Pa + Pb X[Q] Pb)_μν,  c_Q = Σ Pt_λσ X_{λσ,Q},

that is Γ/2 against X = J V⁻¹, because `build_ri_two_electron_gradient`
returns ½ Σ G d(fitted ERI) and the SCF term is ¼ Σ Γ d. The closed-shell
builder (RHF, RKS), UHF and ROHF gradients call it under `_ri_jk` in place of
the exact 2e term; core/Pulay and V_nn are unchanged, since W comes from the
RI Fock the orbitals diagonalize.

**Found: V⁻¹ was formed explicitly.** `build_ri_x_pair_factors` (shared with
the RI-MP2 gradient) built X = J (L⁻ᵀL⁻¹). The first RI-JK gradient missed
its energy FD by 2.1e-6 Ha/Bohr with def2-universal-JKFIT and the two H atoms
differed by 3e-8, while every orbital basis with the uncontracted
cc-pVDZ-RIFIT set matched to 3e-8 (FD noise). Ruled out on the way: a pruned
metric (both sets take the Cholesky branch), the derivative integrals on a
contracted aux set (`planck-ri-3c-derivative-fd` with JKFIT: 1.3e-8, FD
floor), aux column counts (`nfit()` = `nfunctions` for Cartesian aux), and
the formula (PySCF's analytic DF gradient matches its own FD to 8e-11 on the
same setup). The cause is conditioning: the Cartesian JKFIT metric has cond
8e11 (RIFIT 7e8; spherical JKFIT 2e7). X is now two triangular solves,
Xᵀ = L⁻ᵀ(L⁻¹Jᵀ), the same factor the energy applies. RI-MP2 gradients use
the same function and stay inside their gates.

| Case (6-31G*, def2-universal-JKFIT) | Reference | Planck |
|---|---|---|
| water gradient | PySCF analytic (= its FD to 8.3e-11) | every printed digit |
| C1 H2O2 gradient, 12 components | PySCF analytic (= its FD to 2.2e-9) | every printed digit |
| water geomopt | PySCF geometric GAU_TIGHT, E_opt −76.0107180372 | identical |
| water frequencies | central differences of PySCF's DF gradient at 5e-3 Bohr: 1578.68 / 4585.52 / 4766.02 cm⁻¹ | identical |

Gated by `water_rhf_rijk_gradient_631gd` (was
`water_rhf_rijk_gradient_rejected`), `h2o2_rhf_rijk_gradient_631gd`,
`water_rhf_rijk_geomopt_631gd` (core) and `water_rhf_rijk_freq_631gd`
(extended); the explicit-inverse X fails the gradient cases by 100×. The
`hf_driver.cpp` gate admits derivative workflows for Cartesian RHF without
correlation (the MP2 gradient's reference terms assume exact integrals) or
PCM (not gated); `water_cation_uhf_rijk_gradient_rejected` marks the rest.
All 37 RI cases pass.

Memory: benzene 6-31G*/JKFIT (nb = 102, naux = 654) peaks at 243 MiB for the
whole gradient run; the stored derivative tensors of step 7 would have been
1.04 GiB on their own. 29 s, almost all in the serial derivative loops (G6).

## Not in this scope

Full-symmetry RI-JK (symmetry-orbit folding), COSX/seminumerical exchange,
attenuated (range-separated) RI-K beyond the refusal, and the rest of the
performance work in `ERI_FOCK_BLAS_SCOPE.md` (B1 BLAS, B2 RI-K from C_occ).
