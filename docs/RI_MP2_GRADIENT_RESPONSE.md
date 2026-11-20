# Why did the RI-MP2 gradient miss its own energy FD by ~4e-6 Ha/Bohr?

Answered 2026-11-16 (opened as a scope 2026-11-16, from
`docs/RI_JK_ROUTINE.md` step 16). Status lives in `vault/Status/`.

## Answer

**The orbital-response terms used the fitted operator, but the SCF they
respond to is exact. Both now use the SCF's own operator, and the gradient
meets its energy FD at the dense-MP2 floor (1.1–1.5e-7 Ha/Bohr).** The fix
is `docs/SCF_ORBITAL_HESSIAN_OPERATOR.md` (commit `9e9c2623`).

Planck's RI-MP2 energy is the exact-integral HF energy plus a density-fitted
correlation energy (`mp2_use_ri` fits only the correlation; the SCF stays
exact unless `scf_ri_jk` is also set). Each term of the gradient must use
the operator of the energy it differentiates:

| Gradient piece | Differentiates | Must use | Before the fix |
|---|---|---|---|
| 2e derivative term of the pair density (`build_ri_two_electron_gradient`) | E_corr | fitted | fitted ✓ |
| Orbital Lagrangian `imat` (`build_ri_imat`) | E_corr | fitted | fitted ✓ |
| CPHF / Z-vector orbital Hessian | the SCF | **exact** | fitted, via `RI::fitted_response` = `mp2_use_ri \|\| scf_ri_jk` |
| `veff` of the relaxed density (`build_veff_from_density{,_spin_densities}`) | the SCF Fock | **exact** | fitted, under `mp2_use_ri` |
| HF-reference derivative terms (`vhf1`, separable Γ) | the SCF | exact | exact ✓ |

The two fitted response operators put the gradient off by roughly the
fitting error of the HF operator times the orbital response.

**The fix:** both response consumers now apply `SCF::two_electron_fock{,_uhf}`,
the one place the fitted/exact choice is made, keyed on the operator the SCF
actually used (`scf_ri_jk`). `RI::fitted_response` is deleted.

## Evidence

Water, 6-31G*, cc-pVDZ-RIFIT, `tests/rmp2_gradient_fd.py` (central FD of
Planck's own RI-MP2 energy), max |analytic − FD|, Ha/Bohr:

| Run | RI-RMP2 | RI-UMP2 (water⁺) |
|---|---|---|
| Cartesian, before | 3.93e-6 | 1.58e-6 |
| Spherical (step-16 lift), before | 3.50e-6 | 1.63e-6 |
| Cartesian, FD step 2e-3 / 1e-3 / 5e-4 Bohr, before | 4.16e-6 / 3.93e-6 / 3.93e-6 | — |
| Dense MP2 (no RI), same input | 1.5e-7 | — |
| **Cartesian, after** | **1.3e-7** | **1.2e-7** |
| **Spherical, after** | **1.1e-7** | **1.5e-7** |

- **Not FD noise:** the error did not shrink with the step.
- **Not the spherical lift:** the Cartesian error was the same size, and the
  HF/KS spherical RI gradients equal PySCF exactly.
- **These two sites were the whole defect:** fixing them brings both cases
  to the dense-MP2 FD floor.

**Why it shipped.** The RI-MP2 gradient was designed as "every stage fitted"
(RG3.1/RG3.2/RG4.2). That is right when the SCF is also fitted, and wrong
for Planck's exact-SCF RI-MP2. PySCF has no DF-MP2 gradient, so the only
check was Planck's own FD on STO-3G (`water_ri_rmp2_gradient_fd`, 3.2e-7
against a 3e-4 tolerance), where the HF fitting error is too small to show.
Step 4 of the RI-JK scope later widened `fitted_response` to include
`scf_ri_jk` and kept the `mp2_use_ri` half.

## Gates

- `water_ri_rmp2_gradient_631gd_fd`, `water_cation_ri_ump2_gradient_631gd_fd`
  (Cartesian) and `water_ri_rmp2_spherical_gradient_631gd_fd`,
  `water_cation_ri_ump2_spherical_gradient_631gd_fd`: FD of the RI-MP2
  energy at 3e-7. All four fail on the fitted response.
- `water_ri_rmp2_gradient_fd` and `water_radical_cation_ri_ump2_gradient_fd`
  (STO-3G) tightened from 3e-4 to 3e-7 (now 2.4e-7 / 1.6e-7).
- The RI-MP2 geomopt and frequency pins held without re-pinning (E_opt
  moves at second order; the frequency references come from PySCF's RI-MP2
  energy).
- Spherical RI-MP2 gradients are no longer refused
  (`water_ri_rmp2_spherical_gradient_rejected` retired).

## Not done

Nothing in this scope.

## Not in this scope

- **RI-MP2 on RI-JK orbitals** (`scf_ri_jk` with a correlated gradient) stays
  refused. Its response is now fitted, which is correct, but its HF-reference
  derivative terms would also have to become fitted.
- **Frozen-core RI-MP2 gradients** (refused with the dense ones).
