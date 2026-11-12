# What does RI-JK need before it is usable for routine HF and DFT?

Scope, opened 2026-11-10. Steps 0–1 are done (2026-11-12; no separate keyword). It expires when the work
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
