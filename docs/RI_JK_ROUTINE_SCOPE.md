# What does RI-JK need before it is usable for routine HF and DFT?

The answer, as of 2026-11-16 (scoped 2026-11-10; steps 0–16 landed). Status
lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`
(priority 6). The speed half (BLAS, RI-K from C_occ) is in
`docs/ERI_FOCK_BLAS_SCOPE.md`; this doc is about correctness and reach, plus
the one speed item (B0) that KS RI-JK could not do without.

## Answer

**Nothing more, for HF and for pure and global-hybrid DFT.** With
`scf_ri_jk`, every one of these equals PySCF density fitting to every printed
digit on its gate molecules, Cartesian and spherical:

- **HF** (RHF, UHF, ROHF): energies, SOSCF, stability, analytic gradients,
  geometry optimization and frequencies; heavy elements with ECPs included.
- **Kohn-Sham** (RKS, UKS), pure and global-hybrid functionals: energies,
  SOSCF, stability, analytic gradients and geometry optimization.

With a JKFIT set the fitting error is ≤ 2.2e-5 Eh per atom
(def2-universal-JKFIT) and ≤ 1.5e-5 (cc-pVTZ-JKFIT), 40–60× below the MP2
RIFIT set.

**Refused by name**, each for a stated reason: range-separated functionals
(the RI integrals are Coulomb-kernel only), TDDFT, PCM and counterpoise with
RI-JK; full point-group symmetry; a correlated gradient on RI-JK orbitals;
and spherical RI-MP2 gradients, which wait on the RI-MP2 gradient defect
found in step 16 (below).

| | Status |
|---|---|
| RHF / UHF / ROHF energy, Cartesian and spherical | **Done**, = PySCF (steps 0, 2, 5) |
| JKFIT fitting sets (def2-universal, cc-pVTZ) | **Done** (step 1) |
| Spherical auxiliary basis for spherical runs | **Done** (`docs/RI_SPHERICAL_AUX.md`) |
| Heavy elements, ECPs, aux L = 6 | **Done**, AgCl = PySCF (step 5a) |
| SOSCF, stability | **Done**, eigenvalues = PySCF DF (step 4) |
| Streamed RI derivative contraction | **Done**, shared with RI-MP2 (step 7) |
| RI pair factors cached per geometry (B0) | **Done** (step 11) |
| RI-MP2 frequencies | **Done** (step 8) |
| RI-JK RHF/UHF/ROHF gradient, geomopt, frequencies | **Done**, = PySCF, Cartesian and spherical (steps 9–10, 16) |
| `planck-dft` pure and global-hybrid energies (RI-J, RI-K) | **Done**, = PySCF (steps 12–13) |
| `planck-dft` SOSCF, stability | **Done**, eigenvalues = PySCF DF (step 14) |
| `planck-dft` gradients, geomopt (RKS, UKS) | **Done**, = PySCF, Cartesian and spherical (steps 15–16) |
| Spherical RI-MP2 gradients | **Refused**: the RI-MP2 gradient misses its own FD by ~4e-6 (Cartesian too) |
| `planck-dft` range-separated with RI-JK | **Refused** by name (attenuated RI not implemented) |
| Full point-group symmetry | Refused |

## How `scf_ri_jk` is wired

The RHF/UHF/ROHF loops in `src/scf/scf.cpp` call `build_ri_fock_rhf/uhf`
(`src/post_hf/ri/ri_eri.cpp`) instead of the dense or direct Fock build. The
fitting set is `mp2_ri_basis`, and RI-JK and RI-MP2 share one fitting cache
(`_ri_aux_basis`, the 2c metric factor, the 3c tensor), which invalidates
itself when the geometry moves. The RI builders take and return
working-basis matrices: `compute_3c_eri` already transforms the orbital
indices to the spherical working basis.

The boundaries: the RI-JK gate in `src/hf_driver.cpp` (full symmetry, and
derivative workflows with correlation or PCM), and `validate_ri_jk_support`
in `src/dft/driver.cpp` (range separation, double hybrids, TDDFT, PCM;
counterpoise at the top of `DFT::Driver::run`). Every SCF and KS gradient
takes its 2e term from `build_ri_scf_two_electron_gradient` under RI, and
`build_ri_two_electron_gradient` lifts the working-basis three-index density
to the Cartesian derivative loops. In `planck-dft`, `assemble_current_ks_potential` adds
the RI J and K in the working basis after projecting the other KS terms. HF
SOSCF and stability check the flag (step 4). Post-HF methods (MP2, CC,
CASSCF) on RI-JK orbitals keep exact integrals, which is a legitimate
combination.

## Still open

- **The RI-MP2 gradient misses its own energy FD by ~4e-6 Ha/Bohr**, found
  in step 16. Cartesian water, 6-31G*/cc-pVDZ-RIFIT: RMP2 3.9e-6, UMP2
  1.6e-6, independent of the FD step (4.16/3.93/3.93e-6 at 2e-3/1e-3/5e-4
  Bohr); dense RMP2 on the same input 1.5e-7. So the error is in the
  RI-specific terms (RI Lagrangian, RI CPHF or the 2e term's RI-MP2
  density), not in the derivative integrals or the spherical lift, which the
  HF/KS gates pin exactly. The existing gate (`water_ri_rmp2_gradient_fd`,
  STO-3G, tolerance 3e-4) was too loose to see it. Cartesian RI-MP2
  gradients ship with it today; spherical stays refused until it is fixed.
  Cause found: the CPHF Hessian and `veff` use the fitted operator while the
  SCF is exact; keying them on `scf_ri_jk` brings both to the FD floor
  (1.3e-7 / 1.2e-7). Scope: `docs/RI_MP2_GRADIENT_RESPONSE_SCOPE.md`.
- **RI-DH (O9)**: RI-JK in `planck-dft` and the streamed contraction are
  in place.
- **G6, profile the RI 2c/3c integrals**
  (`docs/GRADIENT_INTEGRAL_PERFORMANCE.md`). The derivative loops are still
  serial: benzene 6-31G*/JKFIT takes 29 s for one gradient (step 9).
- **Attenuated (range-separated) RI-K**, so range-separated functionals can
  use RI-JK.
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

### 10. RI-JK gradient, UHF and ROHF

No code beyond the gate: the UHF and ROHF gradients already called step 9's
helper with (P_a, P_b), and ROHF needs nothing more because its
energy-weighted density comes from the spin Fock matrices the RI SCF stores.
The `hf_driver.cpp` refusal now covers spherical, correlated and PCM RI-JK
derivatives only.

| Case (6-31G*, def2-universal-JKFIT) | PySCF analytic vs its own FD | Planck |
|---|---|---|
| water⁺ UHF gradient | 4.9e-11 | every printed digit |
| water⁺ ROHF gradient | 8.5e-11 | every printed digit |
| C1 H2O2⁺ UHF gradient, 12 components | 1.4e-9 | every printed digit; Planck's own FD 3.7e-8 |

Gated by `water_cation_uhf_rijk_gradient_631gd` (was
`water_cation_uhf_rijk_gradient_rejected`),
`water_cation_rohf_rijk_gradient_631gd`, `h2o2_cation_uhf_rijk_gradient_631gd`
(core) and the new boundary marker
`water_rhf_rijk_spherical_gradient_rejected`. Forcing the closed-shell
shortcut (2 P_a X P_a, ignoring P_b) fails all three open-shell cases and
leaves RHF green. All 40 RI cases pass.

Found on the way, in the reference script only: PySCF's `1e` guess puts
H2O2⁺ UHF on a saddle, so the script follows the instability. Its
displaced-geometry SCFs then changed basin and its FD came out 1.18 off;
seeding them from the central density fixed that (1.4e-9). Its SCF also needs
more than the default 50 cycles to reach `conv_tol_grad` 1e-9.

### 11. B0: the RI pair factors are cached

`build_ri_j` and `build_ri_k` re-solved B = J V^{-1/2} from `_ri_j3c` on every
call, twice per SCF iteration (`docs/ERI_FOCK_BLAS_SCOPE.md` B0 measured it
at ~61 % of the RI-JK Fock build). The factors do not depend on the density.
`ensure_ri_3c_ready` now builds them once into `Calculator::_ri_pair_factors`
right after `_ri_j3c`, and `build_ri_pair_factors` returns a reference to
that cache; the five callers that copied it now bind a reference.

Invalidation, the risk the B0 scope named, uses the existing mechanism: the
factors are reset in the same block of `ri_invalidate_if_geometry_moved` that
clears `_ri_j3c`, and again whenever `ensure_ri_3c_ready` rebuilds the tensor.
`planck-ri-cache-invalidation` now also checks that, after the geometry
moves, the cached factors are the moved tensor with the metric applied
(Δ = 0). Removing both resets fails that test and the RI-JK geomopt and
frequency cases; either reset alone keeps them correct.

Benzene 6-31G*/def2-universal-JKFIT RI-JK energy, whole run: user time
4.3–4.5 s → 2.9 s (1.5×; setup included, so the Fock build gains more),
peak RSS unchanged (~184 MiB). Energies and every SCF iteration line are
bitwise identical. The unpacked per-aux B_Q (the other 8 %) is still built
per call; caching it would cost nb²·naux, twice the packed factors.

Found and fixed on the way: `planck-ri-3c-eri` failed its live PySCF
comparison of `compute_3c_eri` (native −1.22 against −0.0317 on O's second s
function). The integrals were right; the test compared different functions.
Since K1 (2026-10-25, `docs/ERI_KERNEL_VS_LIBCINT.md` §3) `read_gbs_basis`
drops redundant primitives from cc-pVXZ contractions, which keeps the span
but changes the contracted functions, while `scripts/export_ri_3c_pyscf.py`
parsed the raw GBS file. Every energy-level gate is span-invariant, so
nothing else noticed. The export script now mirrors
`drop_redundant_primitives` for the orbital basis (opt-in flag; the four
RI-MP2 benchmark scripts that import its loader compare energies and keep
the raw parse), and the test passes at 1e-10. The script itself was never
in the repository (`.gitignore` ignores `scripts/*` except an allowlist), so
on a fresh clone the test could not even find its oracle; it is now on the
allowlist.

### 12. KS RI-J, pure functionals

`assemble_current_ks_potential` skips the direct Coulomb build under
`scf_ri_jk` and, after the Cartesian-to-working projection of the other KS
terms, adds `build_ri_j` of the stored working density (alpha + beta for UKS)
to `coulomb`, `alpha` and `beta`. The SCF energy uses ½ Σ P·`coulomb`, so it
follows with no other change. The cache is primed there (a no-op after the
first call). The only other exact J/K builders in `planck-dft` are TDDFT and
the KS orbital Hessian (SOSCF, stability), both refused.

`validate_ri_jk_support` (next to `validate_workflow_support`, once the
functional is known) admits only pure-functional single points: hybrids,
double hybrids, every non-single-point workflow, SOSCF, stability and PCM
are refused by name, and counterpoise at the top of `DFT::Driver::run`.
Under `planck-mpi` every rank builds the whole RI J: correct, no speed-up.

| Case (6-31G*, def2-universal-JKFIT, unpruned 150 × 590) | PySCF RI | exact | Planck |
|---|---|---|---|
| water RKS PBE | −76.3126823494 | −76.3126584396 | every printed digit |
| water⁺ UKS PBE | −75.8534904952 | −75.8534653188 | every printed digit |
| water RKS PBE, spherical | −76.3107746348 | −76.3107325601 | every printed digit |

The 2.4–4.2e-5 Eh gap to the exact energy shows the RI branch is the one
running. Gated by `water_rks_pbe_rijk_631gd` (was
`water_rks_pbe_rijk_rejected`), `water_cation_uks_pbe_rijk_631gd`,
`water_rks_pbe_rijk_spherical_631gd`, and the markers
`water_rks_b3lyp_rijk_rejected` and `water_rks_pbe_rijk_gradient_rejected`
(the KS gradient would otherwise use exact J at RI orbitals, the step-3
defect). References: `tests/pyscf/ri_jk_dft_references.py`.

### 13. KS RI-K, global hybrids

The post-projection RI block in `assemble_current_ks_potential` now also adds
the exchange under `scf_ri_jk`: one `build_ri_3index_unpacked` shared by the
spins, `build_ri_k` unscaled, and the direct path's prefactors and energy
unchanged (RKS −½c K with E = −¼c tr(P K); UKS −c K_σ per spin with
E = −½c Σ_σ tr(P_σ K_σ)). The direct exchange is skipped under RI.
`validate_ri_jk_support` admits global hybrids and refuses range-separated
functionals by name: the RI 2c/3c integrals are Coulomb-kernel only.

| Case (B3LYP, 6-31G*, def2-universal-JKFIT, 150 × 590) | PySCF RI | exact | Planck |
|---|---|---|---|
| water RKS | −76.4009488436 | −76.4009350142 | every printed digit |
| water⁺ UKS | −75.9401245123 | −75.9401084943 | every printed digit |
| C1 H2O2 RKS | −151.5278433929 | −151.5278251054 | every printed digit |

Gated by `water_rks_b3lyp_rijk_631gd` (was `water_rks_b3lyp_rijk_rejected`),
`water_cation_uks_b3lyp_rijk_631gd`, `h2o2_rks_b3lyp_rijk_631gd` and the
marker `water_rks_hse06_rijk_rejected`.

### 14. KS SOSCF and stability use the fitted operator

`KsOrbitalHessianInputs` and `UksOrbitalHessianInputs`
(`src/dft/ks_orbital_hessian.h`) carry an optional `ri` Calculator pointer,
set at the four call sites under `scf_ri_jk`. With it, the J and K responses
come from `build_ri_j`/`build_ri_k` on a working-basis trial density, packed
with the working MOs (the RI builders work in the working basis), with the
direct path's prefactors (RKS J − ½c K under `kernel_scale`; UKS J − c K_σ).
The unpacked factors are built once per operator, not per product. The XC
kernel term stays on its Cartesian path. `validate_ri_jk_support` no longer
refuses SOSCF or stability.

| Case (6-31G*, def2-universal-JKFIT, 150 × 590) | PySCF on `density_fit()` | Planck | exact-integral Hessian |
|---|---|---|---|
| water RKS B3LYP, singlet (gen_g_hop_rhf/2) | +0.3242326 | +0.3242326 | +0.3242393 |
| water RKS B3LYP, triplet (rhf_external) | +0.2702975 | +0.2702975 | +0.2702970 |
| water⁺ UKS PBE (gen_g_hop_uhf) | +0.0688955 | +0.0688955 | +0.0689020 |

SOSCF reaches the RI DIIS energy with superlinear gradient decay (RKS
3.5e-3 → 2.3e-4 → 1.6e-5). Gated by
`{water_rks_b3lyp,water_cation_uks_pbe}_rijk_soscf_stability_631gd` (core) at
atol 2e-7, which the exact-integral Hessian fails on every eigenvalue (the
existing KS stability gates' 1e-6 would miss the triplet's 5e-7). The SOSCF
energy check cannot tell the two Hessians apart: both converge to the RI
minimum.

### 15. KS RI-JK gradient, Cartesian

Both KS gradients already had the Γ step 9's helper covers,
2 P_t⊗P_t − 2c (P_a·P_a + P_b·P_b). `compute_rks_gradient` reached it
through the closed-shell builder's RI branch with its exchange fraction since
step 9; `compute_uks_gradient` gets the same branch now. The XC terms and the
grid response are unchanged. `validate_ri_jk_support` admits gradient,
geomopt, frequency and geomopt+frequency workflows, and refuses them in a
spherical basis by name (step 16). (The DFT gradient debug breakdown still
splits Coulomb and exchange with exact integrals; it is a diagnostic print,
not part of the gradient.)

| Case (6-31G*, def2-universal-JKFIT, 150 × 590) | Reference | Planck |
|---|---|---|
| water RKS PBE gradient | PySCF DF analytic, grid response | ≤ 7e-11 Ha/Bohr, every component |
| C1 H2O2 RKS B3LYP gradient, 12 components | same | ≤ 7e-11 |
| water⁺ UKS B3LYP gradient | same | ≤ 7e-11 |
| water RKS PBE geomopt | PySCF geometric GAU_TIGHT, E_opt −76.3225206089 | identical |

PySCF's `geometric_solver.optimize(mf)` builds its own gradient object with
grid response off, so the reference passes the configured Gradients object
instead. Gated by `water_rks_pbe_rijk_gradient_631gd` (was
`water_rks_pbe_rijk_gradient_rejected`), `h2o2_rks_b3lyp_rijk_gradient_631gd`,
`water_cation_uks_b3lyp_rijk_gradient_631gd`, `water_rks_pbe_rijk_geomopt_631gd`
and the marker `water_rks_pbe_rijk_spherical_gradient_rejected`.

### 16. Spherical RI gradients

`build_ri_two_electron_gradient` now takes the three-index density in the
working space (working AO pairs × fitted aux) and lifts it before streaming:
each aux column is unpacked to the symmetric G_Q (the packed `w·γ3`,
halved off the diagonal, so pre-folded and single-ordering densities give the
same G), lifted Cᵀ G_Q C, and repacked with the Cartesian pair weight; the
aux index is multiplied by T_aux; and the metric charge becomes
T_auxᵀ γ_PQ T_aux. In a Cartesian run both lifts are skipped. The derivative
loops no longer refuse spherical bases, and `build_ri_scf_two_electron_gradient`
takes the stored working-basis densities (the gradient's P is the Cartesian
lift; in a Cartesian run they are the same matrix).

| Case (6-31G*, def2-universal-JKFIT, spherical) | Reference | Planck |
|---|---|---|
| water RHF gradient | PySCF DF analytic, `cart=False` (= its FD to 9.0e-11) | every printed digit |
| C1 H2O2⁺ UHF gradient, 12 components | same (= its FD to 9.7e-10) | every printed digit |
| water RKS PBE gradient (150 × 590) | PySCF DF analytic with grid response | ~1e-12 |

Gated by `water_rhf_rijk_spherical_gradient_631gd` (was
`water_rhf_rijk_spherical_gradient_rejected`),
`h2o2_cation_uhf_rijk_spherical_gradient_631gd` and
`water_rks_pbe_rijk_spherical_gradient_631gd` (was
`water_rks_pbe_rijk_spherical_gradient_rejected`). Doubling the off-diagonal
unpack factor fails both HF spherical cases and leaves the Cartesian ones
green.

**Spherical RI-MP2 stays refused.** The lift serves it too (its γ3 comes
from the working-basis pair density), but its FD check failed by 3.5e-6
(RMP2) / 1.6e-6 (UMP2), and the same check in a Cartesian basis fails by the
same amount: the RI-MP2 gradient itself is off (see Still open).
`water_ri_rmp2_spherical_gradient_rejected` now names that reason.

## Not in this scope

Full-symmetry RI-JK (symmetry-orbit folding), COSX/seminumerical exchange,
attenuated (range-separated) RI-K beyond the refusal, and the rest of the
performance work in `ERI_FOCK_BLAS_SCOPE.md` (B1 BLAS, B2 RI-K from C_occ).
