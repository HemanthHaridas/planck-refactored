# Should Planck offer a spherical auxiliary (fitting) basis?

Answered 2026-11-12: yes, and it is not an option but the rule. **The fitted
auxiliary functions follow `basis_type`**, as in PySCF: a spherical orbital
basis fits with a spherical aux basis, a Cartesian one with a Cartesian aux.
No keyword. Context: `docs/RI_JK_ROUTINE_SCOPE.md` step 2.

## Why

Before this, Planck's aux basis was always Cartesian. PySCF's `make_auxmol`
copies `mol.cart`, and the published fitting sets (RIFIT, JKFIT, from
Turbomole) were optimized as spherical sets. So spherical RI energies
differed from every other code by a convention. Measured with PySCF 2.15,
spherical orbitals (`tests/pyscf/ri_aux_cart_vs_sph.py`):

| Orbital / aux | Molecule | naux cart | naux sph | ratio | fit err cart | fit err sph | sph − cart |
|---|---|---|---|---|---|---|---|
| 6-31G* / def2-universal-JKFIT | water | 133 | 113 | 1.18 | +2.68e-5 | +2.10e-5 | −5.8e-6 |
| | H2O2 | 226 | 190 | 1.19 | +8.95e-5 | +9.31e-5 | +3.6e-6 |
| | benzene | 654 | 558 | 1.17 | +1.07e-4 | +1.13e-4 | +6.8e-6 |
| cc-pVDZ / cc-pVTZ-JKFIT | water | 166 | 139 | 1.19 | +1.20e-5 | +3.24e-6 | −8.7e-6 |
| | H2O2 | 262 | 218 | 1.20 | +4.80e-5 | +2.23e-5 | −2.6e-5 |
| | benzene | 786 | 654 | 1.20 | +6.34e-5 | +7.81e-5 | +1.5e-5 |
| cc-pVDZ / cc-pVDZ-RIFIT | water | 96 | 84 | 1.14 | −9.84e-4 | −1.07e-3 | −8.9e-5 |
| | H2O2 | 162 | 140 | 1.16 | −1.63e-3 | −1.80e-3 | −1.8e-4 |
| | benzene | 486 | 420 | 1.16 | −4.00e-3 | −4.12e-3 | −1.2e-4 |

RHF energies in Eh; fit error is RI minus exact.

- **Cost.** The Cartesian aux had 14–20 % more functions: RI-JK time and
  memory scale with naux, the metric factorization with naux³.
- **Accuracy.** Neither is consistently better; the difference is of the
  order of the fitting error itself, with either sign (RI-K is not
  variational in the aux space).
- **Comparability.** The Cartesian convention put spherical RI energies
  1e-5..2e-4 Eh off PySCF, ORCA and Turbomole, against the 1e-9 agreement
  Planck holds elsewhere.

## How

Aux shells are still read Cartesian (`read_ri_basis`), and every integral
loop still runs over Cartesian aux components. With a spherical orbital
basis, `ensure_ri_metric_ready` also builds `AuxBasis::cart_to_sph`, a
block-diagonal map assembled from `cart_to_sph_block(L)` per aux shell (L = 6
goes through the recurrence oracle; def2-universal-JKFIT reaches i on heavy
elements). It is applied once at each output:

- metric: `V → T V Tᵀ` at the end of `compute_2c_eri`;
- 3-center tensor: `J3c → J3c Tᵀ` at the end of `compute_3c_eri`, after the
  existing orbital-index transform.

`AuxBasis::nfit()` is the fitted count; `nfunctions` stays the Cartesian
count the loops and offsets use. Everything downstream (RI-JK, RI-MP2,
RI-CASSCF) sees only a smaller naux. The fit is invariant to the row scale of
T, so the bare blocks serve; only their span matters. `compute_2c_eri_deriv`
refuses a spherical aux by name; `compute_3c_eri_deriv` already refused a
spherical orbital basis, and the rule keeps the aux Cartesian whenever
derivatives can run.

## Gates

- The four `*_rijk_spherical_631gd` RI-JK cases, re-pinned: each equals
  plain PySCF `density_fit` (spherical aux) to every printed digit. The
  `cart_aux_jk` workaround left the reference script; it survives only in
  the measurement script above.
- New `water_rimp2_spherical_ccpvdz`: spherical RI-MP2 equals PySCF DF-MP2
  (RHF to every digit, MP2 −76.2230990337 vs −76.2230990336).
- Every Cartesian RI case (RI-JK, RI-MP2, RI-CASSCF, RI-FCI) is unchanged:
  17/17 `ri` cases and the 9 `planck-ri*` unit tests pass.

## Not in this scope

Default fitting sets per orbital basis. (Auxiliary-basis derivatives in a
spherical basis landed later, as `docs/RI_JK_ROUTINE_SCOPE.md` step 16.)
