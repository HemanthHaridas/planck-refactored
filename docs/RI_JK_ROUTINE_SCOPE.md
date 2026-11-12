# What does RI-JK need before it is usable for routine HF and DFT?

Scope, opened 2026-11-10. Step 0 is done (2026-11-12). It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 6). The speed half (pair-factor cache,
BLAS, RI-K from C_occ) is scoped in `docs/ERI_FOCK_BLAS_SCOPE.md`; this doc
is about correctness and reach.

## The state

`scf_ri_jk` (`src/scf/scf.cpp`, `src/hf_driver.cpp`) replaces the J/K build
in the HF SCF with RI. It is refused for ROHF, the spherical basis and full
point-group symmetry (named rejections in `hf_driver.cpp`). It reuses the
MP2 fitting basis (`mp2_ri_basis`); the only fitting sets in `basis-sets/`
are `cc-pVDZ-RIFIT`, `cc-pVTZ-RIFIT` and `toy-ri`. `planck-dft` has no RI-JK
at all.

So RI-JK today is Cartesian HF with the wrong kind of fitting basis
(RIFIT sets are fitted for MP2 pair densities, not for the Coulomb/exchange
of the SCF density). Since step 0 it is at least correct against PySCF.

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

Add `def2-universal-jkfit` (Weigend 2008, covers H–Rn with def2 ECPs) and
the cc-pVXZ-JKFIT sets to `basis-sets/`, and a keyword `scf_ri_basis`
separate from `mp2_ri_basis` (fall back to it for compatibility). Quantify
the fitting error on the gate cases: RI-JK total energies are expected
within ~1e-5 Eh/atom of the exact ones with a JKFIT set; record the actual
number.

### 2. Spherical

Every standard basis is spherical. Two choices: run the RI builders on the
lifted Cartesian density and project the result with `to_working`, as
spherical PCM does (`evaluate_pcm_reaction_field`); or build spherical 3c
integrals. Do the first; it is one mechanism and already proven for PCM.
Gate against PySCF `cart=False`.

### 3. `planck-dft`

The KS J and the exact-exchange K for hybrids go through
`_compute_2e_j_direct` / `_compute_2e_k_direct`. Route them through the RI
builders under the same flag, keeping the unscaled-K contract (each caller
applies its own prefactor; see the DFT J/K notes in CLAUDE.md). Gate PBE
(RI-J only) and B3LYP RKS/UKS against PySCF `density_fit()`.

### 4. Gradients

RI-JK SCF with an exact-integral gradient is inconsistent at the fitting
error (~1e-5 Ha/Bohr). Refuse `scf_ri_jk` + gradient workflows by name until
an RI-JK gradient exists (the RI 2c/3c derivative integrals exist for
RI-MP2). The RI-JK gradient is a separate scope.

## Not in this scope

ROHF and full-symmetry RI-JK, COSX/seminumerical exchange, and the
performance work in `ERI_FOCK_BLAS_SCOPE.md`.
