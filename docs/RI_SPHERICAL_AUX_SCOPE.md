# Should Planck offer a spherical auxiliary (fitting) basis?

Scope, opened 2026-11-12. Nothing here is built. It expires when the decision
is made; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 6, RI-JK). Context:
`docs/RI_JK_ROUTINE_SCOPE.md` step 2.

## The question

Planck's auxiliary basis is always Cartesian (`read_ri_basis`,
`AuxBasis::cartesian = true`), even when the orbital basis is spherical.
PySCF makes the auxiliary basis follow the orbital basis (`make_auxmol`
copies `mol.cart`), and the published fitting sets (RIFIT, JKFIT, all from
Turbomole) were optimized as spherical sets. So for a spherical orbital
basis, Planck's RI energies differ from every other code's by a convention,
not a defect. Matching PySCF in the spherical RI-JK gates needed a custom
reference (`tests/pyscf/ri_jk_references.py`, `cart_aux_jk`).

## What it costs and changes (measured, PySCF 2.15, spherical orbitals)

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

RHF energies in Eh; fit error is RI minus exact. Script:
`tests/pyscf/ri_aux_cart_vs_sph.py`.

Read off:

- **Cost.** The Cartesian aux has 14–20 % more functions. RI-JK time and
  memory scale with naux (3-center tensor nb²·naux, K build nb²·naux·nocc)
  and the metric factorization with naux³, so spherical aux is roughly
  1.2× cheaper per Fock build and 1.7× on the factorization.
- **Accuracy.** Neither is consistently better. The difference is of the
  order of the fitting error itself (1e-6 to 3e-5 Eh with JKFIT, 1e-4 with
  RIFIT), with no fixed sign: the RI-K energy is not variational in the aux
  space, so more functions do not guarantee a smaller error.
- **Comparability.** A user checking a spherical RI energy against PySCF,
  ORCA or Turbomole sees up to ~3e-5 Eh (JKFIT) or ~2e-4 Eh (RIFIT) of
  difference from the convention alone, larger than the 1e-8..1e-9 Eh
  agreement Planck holds itself to elsewhere.

## What it needs

The aux basis is consumed only in `src/post_hf/ri/ri_eri.cpp`. The fit is
invariant to any full-rank linear map within the aux span, so only the
*span* of the transform matters, not its normalization:

1. A block-diagonal aux transform T_aux (n_sph_aux × n_cart_aux), one
   `cart_to_sph_block(L)` per aux shell, built once in `read_ri_basis` when
   the orbital basis is spherical. Reuse the orbital machinery: its
   pure-harmonic property is gated through L = 6
   (`tests/spherical_transform.cpp`), which matters because
   def2-universal-JKFIT reaches i functions on heavy elements.
2. Apply it in two places: the metric (`J2c → T J2c Tᵀ`, in
   `ensure_ri_metric_ready` / `compute_2c_eri`) and the 3-center columns
   (`J3c → J3c Tᵀ`, at the end of `compute_3c_eri`, next to the existing
   orbital-index transform). Everything downstream (`build_ri_j/k`, RI-MP2,
   RI-CASSCF) works on the fitted factors and sees only a smaller naux.
3. The derivative integrals (`compute_3c_eri_deriv`, `compute_2c_eri_deriv`)
   already refuse a spherical orbital basis, and the rule below keeps the
   aux Cartesian for Cartesian orbitals, so they need nothing.

Rule: **the aux basis follows `basis_type`**, as in PySCF. No keyword: a
Cartesian aux with spherical orbitals has no use case beyond reproducing
today's numbers.

Gates: the four `*_rijk_spherical_631gd` cases re-pinned to plain PySCF
`density_fit` (and `cart_aux_jk` deleted from the reference script, which
is the point); the Cartesian RI-JK and RI-MP2 cases bitwise unchanged; one
spherical RI-MP2 energy against PySCF `mp2.dfmp2` with the same sets.

## Timing

Only the four spherical RI-JK cases added on 2026-11-12 pin the current
convention (plus `water_ri_rmp2_spherical_gradient_rejected`, a refusal).
RI-JK step 3 (`planck-dft`) will add spherical RI cases; deciding before it
avoids re-pinning those too.

## Recommendation

Yes: switch spherical orbital runs to a spherical aux basis, before
RI-JK step 3. It is ~1.2× cheaper, no less accurate on these cases, and
removes a 1e-5..1e-4 Eh convention gap to every other code. The change is a
transform in two places plus a re-pin of four cases.

## Not in this scope

Auxiliary-basis derivatives in a spherical orbital basis (spherical RI-MP2
gradients, `docs/SPHERICAL_MP2_GRADIENTS.md`), and choosing default fitting
sets per orbital basis.
