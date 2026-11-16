# Why does the RI-MP2 gradient miss its own energy FD by ~4e-6 Ha/Bohr?

Scope, opened 2026-11-16. It expires when the fix lands; rewrite it then as
the answer. Found in `docs/RI_JK_ROUTINE_SCOPE.md` step 16; status in
`vault/Status/Open Work.md` (priority 6).

## Answer (measured, not yet fixed)

**The orbital-response terms use the fitted operator, but the SCF they
respond to is exact.** Planck's RI-MP2 energy is the exact-integral HF
energy plus a density-fitted correlation energy (`mp2_use_ri` fits only the
correlation; the SCF stays exact unless `scf_ri_jk` is also set). Each term
of the gradient must use the operator of the energy it differentiates:

| Gradient piece | Differentiates | Must use | Uses today |
|---|---|---|---|
| 2e derivative term of the pair density (`build_ri_two_electron_gradient`) | E_corr | fitted | fitted ✓ |
| Orbital Lagrangian `imat` (`build_ri_imat`) | E_corr | fitted | fitted ✓ |
| CPHF / Z-vector orbital Hessian A (`rhf_response.cpp`, `uhf_response.cpp`) | the SCF | **exact** | fitted, via `RI::fitted_response` = `mp2_use_ri \|\| scf_ri_jk` |
| `veff` of the relaxed density (`build_veff_from_density`, `build_veff_from_spin_densities` in `mp2_gradient.cpp`) | the SCF Fock | **exact** | fitted, under `mp2_use_ri` |
| HF-reference derivative terms (`vhf1`, separable Γ) | the SCF | exact | exact ✓ |

The last two rows of fitted operators leave the gradient off by roughly the
fitting error of the HF operator times the orbital response.

## Evidence

Water, 6-31G*, cc-pVDZ-RIFIT, `tests/rmp2_gradient_fd.py` (central FD of
Planck's own RI-MP2 energy), max |analytic − FD|, Ha/Bohr:

| Run | RI-RMP2 | RI-UMP2 (water⁺) |
|---|---|---|
| Cartesian, today | 3.93e-6 | 1.58e-6 |
| Spherical, with the step-16 lift | 3.50e-6 | 1.63e-6 |
| Cartesian, FD step 2e-3 / 1e-3 / 5e-4 Bohr | 4.16e-6 / 3.93e-6 / 3.93e-6 | — |
| Dense MP2 (no RI), same input | 1.5e-7 | — |
| **Experiment: CPHF and `veff` keyed on `scf_ri_jk`** | **1.3e-7** | **1.2e-7** |

- **Not FD noise:** the error does not shrink with the step.
- **Not the spherical lift:** the Cartesian error is the same size, and the
  HF/KS spherical RI gradients equal PySCF exactly.
- **The experiment** keyed the response operator on `scf_ri_jk` alone (the
  operator the SCF actually used). It brings both errors to the dense-MP2 FD
  floor, so these two sites are the whole defect. It was reverted; nothing is
  committed.

**Why it shipped.** The RI-MP2 gradient was designed as "every stage fitted"
(RG3.1/RG3.2/RG4.2). That is right when the SCF is also fitted, and wrong
for Planck's exact-SCF RI-MP2. PySCF has no DF-MP2 gradient, so the only
check was Planck's own FD on STO-3G (`water_ri_rmp2_gradient_fd`, 3.2e-7
against a 3e-4 tolerance), where the HF fitting error is too small to show.
Step 4 of the RI-JK scope later widened `fitted_response` to include
`scf_ri_jk` and kept the `mp2_use_ri` half.

## Fix

**Done (2026-11-16) by the recommended route; F1–F3 landed.** Both response
consumers go through `SCF::two_electron_fock{,_uhf}`, which is exact unless
the SCF ran RI-JK. RI-MP2 gradients now meet their own energy FD at 3e-7
Ha/Bohr (water RI-RMP2 6-31G* 1.3e-7, water⁺ RI-UMP2 1.2e-7; spherical
1.1e-7 / 1.5e-7), and spherical RI-MP2 gradients are no longer refused.
Status and gates: `docs/SCF_ORBITAL_HESSIAN_OPERATOR_SCOPE.md`.

**Recommended route:** `docs/SCF_ORBITAL_HESSIAN_OPERATOR_SCOPE.md`. It
makes `SCF::two_electron_fock{,_uhf}` the one place the fitted/exact choice
is made, and puts RHF and UHF on one matrix-free orbital-Hessian operator.
That does F1 below as a by-product, and F4 does not arise. F1 as written is
the minimal version if that refactor is not wanted.

**F1. The response operator follows the SCF.** One rule at three sites:
- `RI::fitted_response` (`src/post_hf/ri/ri_eri.h`) returns
  `calculator._scf._ri_jk` only. It serves the RHF/UHF CPHF builders and the
  HF SOSCF Hessian. HF SOSCF under `mp2_use_ri` alone currently uses a
  fitted Hessian for an exact SCF; that converges, but not quadratically.
- `build_veff_from_density` and `build_veff_from_spin_densities`
  (`src/post_hf/mp2_gradient.cpp`) key on `scf_ri_jk`, not `mp2_use_ri`.
- Rewrite the RG3.2/RG4.2 comments that call the fitted response
  "RI-consistent", and the `fitted_response` comment.

**F2. Gates at a tolerance that can see it.**
- New: water RI-RMP2 and water⁺ RI-UMP2 gradients, Cartesian 6-31G*,
  cc-pVDZ-RIFIT, `tests/rmp2_gradient_fd.py --atol 3e-7` (the dense floor is
  1.5e-7). Both fail today by ~4e-6 / 1.6e-6 and pass with F1.
- Tighten `water_ri_rmp2_gradient_fd` and
  `water_radical_cation_ri_ump2_gradient_fd` from 3e-4 to 3e-7.
- Re-run the RI-MP2 geomopt and frequency cases. E_opt moves only to second
  order. The step-8 frequency references come from PySCF's RI-MP2 energy,
  independent of Planck's gradient, so they should hold or improve.

**F3. Spherical RI-MP2 gradients.** The step-16 lift already serves them.
Lift the refusal in `hf_driver.cpp`, turn
`water_ri_rmp2_spherical_gradient_rejected` into an FD gate, and add the
water⁺ RI-UMP2 spherical case, both at 3e-7.

**F4 (cost; only if a profile says so).** The exact RHF CPHF builds an
explicit A from the full AO ERI tensor and an nao⁴ MO transform
(`build_rhf_cphf_matrix`), which the RI-MP2 gradient avoided until now. The
gradient already holds an nao⁴ pair density (`dm2buf`), so the asymptotic
memory does not change, but time and peak memory grow. The UHF path is
already matrix-free through `SCF::two_electron_fock_uhf`. If it matters, give
RHF the same matrix-free form through `SCF::two_electron_fock`.

## Not in this scope

- **RI-MP2 on RI-JK orbitals** (`scf_ri_jk` with a correlated gradient) stays
  refused. With F1 its response would already be fitted, which is correct,
  but its HF-reference derivative terms would also have to become fitted.
- **Frozen-core RI-MP2 gradients** (refused with the dense ones).
