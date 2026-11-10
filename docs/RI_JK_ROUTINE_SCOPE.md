# What does RI-JK need before it is usable for routine HF and DFT?

Scope, opened 2026-11-10. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 9). The speed half (pair-factor cache,
BLAS, RI-K from C_occ) is scoped in `docs/ERI_FOCK_BLAS_SCOPE.md`; this doc
is about correctness and reach.

## The state

`scf_ri_jk` (`src/scf/scf.cpp`, `src/hf_driver.cpp`) replaces the J/K build
in the HF SCF with RI. It is refused for ROHF, the spherical basis and full
point-group symmetry (named rejections in `hf_driver.cpp`). It reuses the
MP2 fitting basis (`mp2_ri_basis`); the only fitting sets in `basis-sets/`
are `cc-pVDZ-RIFIT`, `cc-pVTZ-RIFIT` and `toy-ri`. `planck-dft` has no RI-JK
at all. No regression case uses `scf_ri_jk`.

So RI-JK today is Cartesian HF with the wrong kind of fitting basis
(RIFIT sets are fitted for MP2 pair densities, not for the Coulomb/exchange
of the SCF density), and nothing checks it.

## Steps

### 0. A gate for what exists

Water and C1 H2O2, RHF and UHF, Cartesian 6-31G*, `scf_ri_jk` with
cc-pVDZ-RIFIT, against PySCF `density_fit(auxbasis=…)` with the same
auxiliary basis read from Planck's file. Every printed digit. This pins the
current code before anything changes, and is the "first RI-JK regression
case" the BLAS scope's B0 also asks for.

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
