# What does RI-JK need before it is usable for routine HF and DFT?

## Headline (2026-11-12)

**RI-JK HF energies, SOSCF and stability are done and exact against PySCF.** `scf_ri_jk` gives
RHF, UHF and ROHF energies, Cartesian or spherical, that equal PySCF density
fitting to every printed digit on every gate molecule. With a JKFIT set the
fitting error is ≤ 2.2e-5 Eh per atom (def2-universal-JKFIT) and ≤ 1.5e-5
(cc-pVTZ-JKFIT), 40–60× below the MP2 RIFIT set it used to need. The fitted
auxiliary basis follows `basis_type`, as in PySCF.

**Not yet usable beyond HF energies.** Derivative workflows are refused (no
RI-JK gradient exists, and the exact-integral one was silently 9.9e-6
Ha/Bohr off). `planck-dft` has no RI-JK at all.

| | Status |
|---|---|
| RHF / UHF / ROHF energy, Cartesian and spherical | **Done**, = PySCF to every printed digit (steps 0, 2, 5) |
| JKFIT fitting sets (def2-universal, cc-pVTZ) | **Done** (step 1) |
| Spherical auxiliary basis for spherical runs | **Done** (`docs/RI_SPHERICAL_AUX.md`) |
| Gradient, geomopt, frequencies | **Refused** by name (step 3); RI-JK gradient is steps 9–10, 15–16 |
| Full point-group symmetry | Refused; out of scope |
| SOSCF, stability | **Done**, fitted operator; eigenvalues = PySCF DF (step 4) |
| `planck-dft` | Not started; `scf_ri_jk` silently ignored (steps 6, 12–15) |
| Heavy elements, ECPs, aux L = 6 | **Done**, AgCl = PySCF (step 5a) |

What is left, in dependency order, is the table under "Remaining".

Scope, opened 2026-11-10. It expires when the work lands; rewrite it then as
the answer. Status lives in `vault/Status/Open Work.md` (priority 6). The
speed half (pair-factor cache, BLAS, RI-K from C_occ) is scoped in
`docs/ERI_FOCK_BLAS_SCOPE.md`; this doc is about correctness and reach.

How `scf_ri_jk` is wired: the RHF/UHF/ROHF loops in `src/scf/scf.cpp` call
`build_ri_fock_rhf/uhf` (`src/post_hf/ri/ri_eri.cpp`) instead of the dense
or direct Fock build; the fitting set is `mp2_ri_basis`; the boundary is the
RI-JK gate in `src/hf_driver.cpp`. Nothing else checks the flag: every other
consumer of the two-electron operator runs with exact integrals on RI
orbitals (steps 12–14; HF SOSCF and stability were fixed in step 4).

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
steps 9–10 land.

### 4. SOSCF and stability use the fitted operator

Before, with `scf_ri_jk` both still used exact integrals: RHF **SOSCF**
rebuilt the full AO ERI tensor at every SOSCF step for its CPHF Hessian (UHF
SOSCF used exact direct J/K), and **stability** built the tensor once and
diagonalized the exact Hessian at the RI orbitals, ~1e-6 off the fitted one.

Both now use machinery that already existed for RI-MP2:

- The CPHF builders (`rhf_response.cpp`, `uhf_response.cpp`) take their
  fitted route under `RI::fitted_response(calc)` = `mp2_use_ri || scf_ri_jk`
  (`ri_eri.h`); before, only `mp2_use_ri`.
- Stability (`stability.cpp`) gets its MO blocks from a `MOBlocks` helper:
  `transform_eri_ri` (fitted, same layout) under `scf_ri_jk`, the dense
  transform otherwise. It keys on `scf_ri_jk` alone: it analyses the SCF
  operator, which RI-MP2 does not change.

Water RHF (def2-universal-JKFIT): the three eigenvalues equal PySCF's
stability analysis on `mf.density_fit()` to the 7 printed digits (real
internal 0.3825292 = PySCF 1.5301167/4, complex 0.3534208, triplet
0.3084010). H2O2⁺ UHF: 0.07322713 = PySCF `uhf_internal` 0.14645447/2. SOSCF
reaches the RI energy; neither log builds the AO tensor. Gated by
`{water_rhf,h2o2_cation_uhf}_rijk_soscf_stability_631gd` (core), which check
the eigenvalues and `not_contains: Building AO ERI tensor`; the old code fails
both. Post-HF methods (MP2, CC, CASSCF) on RI-JK orbitals keep exact
integrals: a legitimate combination.

### 5. ROHF, one branch

The refusal said the Roothaan Fock "is not wired", but `run_rohf` builds the
same (Ga, Gb) as UHF before assembling the effective Fock. It now primes the
RI cache, skips the dense tensor and calls `build_ri_fock_uhf`, as UHF does;
the refusal is gone. Water⁺ ROHF, def2-universal-JKFIT, equals PySCF
`ROHF(...).density_fit()` to every printed digit, Cartesian
(−75.5989675667) and spherical (−75.5972324349); fitting errors 1.9e-5 and
1.6e-5 Eh. `water_cation_rohf_rijk{,_spherical}_631gd`, core.

### 5a. Heavy elements, ECPs, aux L = 6 (recorded elsewhere as step 8a)

def2-universal-JKFIT carries i functions (L = 6) for the transition metals
and heavier (Zn, Ag, Au, …; iodine stops at h, so HI would not have tested
L = 6). AgCl, def2-SVP with the Ag ECP, spherical: two l = 6 aux shells, 383
aux functions; equals PySCF `density_fit` (ECP and bases from Planck's GBS)
to every printed digit, −605.4799497321, fitting error 4.9e-5 Eh.
`agcl_rhf_rijk_def2svp_ecp`, core. No code changed (2026-11-12).

## Remaining, in the order it can be done

Re-scoped 2026-11-16 against the tree. Each step names what it needs; every
step's prerequisites come before it. This list also places the other RI
items from Open Work priority 6 (spherical RI-MP2 gradients, RI-MP2
frequencies, B0). Steps 0–5 keep their numbers; the heavy-element check
recorded elsewhere as "step 8a" is under Done.

| Step | What | Needs |
|---|---|---|
| 6 | Refuse `scf_ri_jk` in `planck-dft` — **done 2026-11-16** | — |
| 7 | Streamed RI derivative contraction; RI-MP2 moved onto it — **done 2026-11-16** | — |
| 8 | RI-MP2 frequencies | 7 |
| 9 | RI-JK gradient, RHF, Cartesian; lift step 3's refusal for RHF | 7 |
| 10 | RI-JK gradient, UHF and ROHF | 9 |
| 11 | B0: cache the RI pair factors | — (must precede 12) |
| 12 | KS RI-J, pure functionals | 6, 11 |
| 13 | KS RI-K, global hybrids | 12 |
| 14 | KS SOSCF and stability on RI; refusals for TDDFT, DH, range separation | 13 |
| 15 | KS RI-JK gradient, Cartesian | 10, 13 |
| 16 | Spherical RI gradients: RI-JK HF and KS, and RI-MP2 | 7, 10, 15 |

11 has no prerequisites and can run in parallel with 8–10.

### 6. Refuse `scf_ri_jk` in `planck-dft` — done 2026-11-16

`scf_ri_jk` was parsed in the shared `io.cpp`, but nothing under `src/dft/`
read it, so a KS input with `scf_ri_jk true` ran exact integrals and said
nothing. `DFT::Driver::run` now refuses it by name before anything else
runs. Gated by `water_rks_pbe_rijk_rejected` (core). Lifted for pure
functionals by step 12.

### 7. Streamed RI derivative contraction — done 2026-11-16

`compute_3c_eri_deriv` stored 3·natoms matrices of npair × naux, and
`compute_2c_eri_deriv` 3·natoms of naux × naux, before
`build_ri_two_electron_gradient` contracted them. At a routine size (20 atoms,
nb = 300, naux ≈ 1500 JKFIT) dJ alone is ≈ 32 GB.

Now each class has one loop, `for_each_3c_deriv` / `for_each_2c_deriv`
(`ri_eri.cpp`), that hands every derivative element to a callback.
`build_ri_two_electron_gradient(calculator, gamma3, x_proj, bra_prefolded)`
contracts through them and stores nothing beyond natoms × 3. The stored-tensor
functions are thin scatters over the same loops, kept for the FD test
(`planck-ri-3c-derivative-fd`) and as an oracle. Neither loop checks
`_mp2.use_ri` any more (only that the aux basis is loaded), so RI-JK can call
them. RI-MP2 and RI-UMP2 (`mp2_gradient.cpp`) call the streamed form.

Checks: `planck-ri-two-electron-gradient` compares the streamed contraction
with the stored-tensor formula, 1.4e-14 on values ~2.6; weakening the metric
½ by 0.02 % fails it (6.7e-5). RI-RMP2/UMP2 gradients and the RI-RMP2 geomopt
are identical to the printed digits before and after (one kcal/mol digit
moves at 8e-13). All six RI-MP2 gradient and geomopt regression cases pass.
Still serial, as before (G6). The memory saving shows only on an RI-JK
gradient: the RI-MP2 path still holds its nao⁴ `dm2buf`. Measure peak RSS in
step 9.

### 8. RI-MP2 frequencies

Refused by name (`water_ri_rmp2_freq_rejected`). The semi-numerical Hessian
runs the RI-MP2 gradient through `Opt::run_scf_gradient`; it needs only a
gate against a finite difference of the RI-MP2 gradient
(`docs/SPHERICAL_MP2_GRADIENTS.md`). After step 7, because step 7 changes
that gradient.

### 9. RI-JK gradient, RHF, Cartesian

Every SCF gradient (`compute_closed_shell_gradient_from_density`,
`compute_uhf_gradient`, `compute_rohf_gradient`) gets its 2e term from one
shared `two_electron_gradient(gamma_fn)`
(`src/gradient/two_electron_gradient.h`), which takes Γ quartet by quartet.
RI cannot use that shape (it would need nao⁴), so add one sibling,
`ri_scf_two_electron_gradient(calc, Pa, Pb, k_coeff)`, called instead of it
under `_ri_jk` at those three sites, and feeding step 7's contraction
(`build_ri_two_electron_gradient(calculator, gamma3, x_proj)`).
Core/Pulay and V_nn are unchanged: W is built from the RI Fock, which the
orbitals diagonalize, so it is already consistent.

- J: γ3_{(μν),P} = ½·D_μν·(XᵀD)_P (rank one per pair), with
  c = V⁻¹(P|λσ)D_λσ.
- K: γ3_{(μν),P} = −¼·k·Σ_σ(D^σ X[P] D^σ)_μν per spin (RHF: D = P/2 per
  spin). nb²·naux, no nao⁴.
- Pin the ½/¼ convention against the RI-MP2 call
  (`mp2_gradient.cpp:860`, `0.5 * build_ri_two_electron_gradient`) and an FD
  of the RI energy, not by reasoning.

Gate: water and C1 H2O2 RHF, def2-universal-JKFIT, against PySCF
`mf.density_fit().nuc_grad_method()` (every component, `gradient_close`)
and a central-difference FD of the RI energy. Then lift step 3's refusal for
RHF and add a geomopt case against PySCF `geometric`. Record peak RSS on
one larger input against the stored-tensor estimate (step 7).

### 10. RI-JK gradient, UHF and ROHF

Same helper, two densities. Gates on water⁺ (UHF, ROHF) and C1 H2O2⁺ (UHF);
lift step 3's refusal for both.

### 11. B0: cache the RI pair factors

`build_ri_j` and `build_ri_k` call `build_ri_pair_factors` on every call
(`ri_eri.cpp:1509`, `:1542`), ~61 % of the RI-JK Fock build
(`docs/ERI_FOCK_BLAS_SCOPE.md`). It is density-independent. KS RI-J (step
12) would pay it every iteration and likely not beat the direct build, so
this goes first. The `*_rijk_*` cases gate it (energies unchanged).

### 12. KS RI-J, pure functionals

In the KS assembly (`src/dft/driver.cpp` ~4686), under `_ri_jk` replace the
direct `coulomb` by `build_ri_j(calculator, P_working_total)`. Prime the
cache once with `ensure_ri_3c_ready` before the SCF loop, as `scf.cpp` does.
Lift step 6's refusal for pure-functional energies only: keep refusing
hybrids, and keep refusing gradient, geomopt and frequency workflows as
step 3 does for HF (the exact KS gradient at RI orbitals would be
inconsistent, as it was for HF).

- **Basis.** KS J, K and V_xc are built in Cartesian from lifted densities
  and projected to the working basis at the end (~4799). The RI builders take
  and return working-basis matrices (the 3c tensor is already in the working
  basis), so the RI branch uses the working densities
  (`calculator._info._scf`) and adds its J *after* the projection; it must
  not be fed the lifted ones. `combine_ks_potential` needs the Coulomb passed
  separately, or the projection loop must skip it.
- **MPI.** The RI builders have no MPI distribution, so under `planck-mpi`
  every rank does the whole RI build: correct, no speed-up. Record it; do
  not block on it.

Gates against PySCF `dft.RKS/UKS(...).density_fit()` on Planck's unpruned
grid convention (as the PCM and DH scripts do): water PBE RKS, water⁺ PBE
UKS, one spherical.

### 13. KS RI-K, global hybrids

`build_ri_k` per spin, unscaled. Keep the unscaled-K contract: each caller
applies its own prefactor (−½c for RKS, −c per spin for UKS), and the
exchange energy is computed from the same matrices. Share one
`build_ri_3index_unpacked` between J and K. Range-separated functionals stay
refused: the RI 2c/3c integrals are Coulomb-kernel only (`ri_eri.h`), and
attenuated RI-K is outside this scope. Gates: water and H2O2 B3LYP RKS,
water⁺ B3LYP UKS, and the range-separated refusal.

### 14. KS consumers

`ks_orbital_hessian.cpp` (SOSCF and stability) builds J/K directly at four
sites: route them through RI under `_ri_jk`, as step 4 did for HF. Refuse by
name with `scf_ri_jk`: TDDFT (`run_linear_response` builds a dense
`_compute_2e` tensor; needs a decision on RI response) and double hybrids
(`dh_pt2_gradient.cpp`, `udh_ks_response.cpp`; RI-DH is its own item).
Gates: as step 4, against PySCF stability on `density_fit()`.

### 15. KS RI-JK gradient, Cartesian

J (and global-hybrid K) through step 9's helper, from
`compute_analytic_ks_gradient`; the XC terms are untouched. Gate: PySCF
`density_fit()` KS gradients and an FD of the RI energy; lift the KS
derivative-workflow refusal kept since step 12.

### 16. Spherical RI gradients

`compute_3c_eri_deriv` refuses a spherical orbital basis and
`compute_2c_eri_deriv` a spherical aux basis. Lift γ3 on the orbital index
(T ᵀ γ T per aux column) and apply T_auxᵀ on the aux index before the
Cartesian derivative contraction (both contractions are linear). Done once
in step 7's contraction, this serves RI-JK HF, RI-JK KS and RI-MP2 alike;
lift the spherical refusals together, including
`water_ri_rmp2_spherical_gradient_rejected`.

### After this scope

- **G6, profile the RI 2c/3c integrals**
  (`docs/GRADIENT_INTEGRAL_PERFORMANCE.md`): step 7 rewrites the derivative
  loop it would profile, so profile after step 7.
- **RI-DH (O9)**: needs steps 12–15.
- **`scf_ri_basis`**: only worth a second fitting cache if a run wants RI-JK
  and RI-MP2 with different sets (step 1). Open until such a run exists.

## Not in this scope

Full-symmetry RI-JK (symmetry-orbit folding), COSX/seminumerical exchange,
attenuated (range-separated) RI-K beyond the refusal, and the rest of the
performance work in `ERI_FOCK_BLAS_SCOPE.md` (B1 BLAS, B2 RI-K from C_occ;
B0 is step 11).
