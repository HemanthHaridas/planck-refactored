# Can `planck-dft` run in a spherical-harmonic basis?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Why it matters

Routine DFT is run in spherical (pure) functions: cc-pVXZ and def2 bases are
defined that way, and a Cartesian d shell carries an extra s-type function,
so Cartesian and spherical energies differ. `planck-dft` rejects
`basis_type spherical` at its entry point (`src/dft/driver.cpp`, "Spherical
basis (basis_type spherical) is not supported by planck-dft"). The HF binary
already runs spherical SCF, gradients, geometry optimization and frequencies,
so the machinery exists; it has not been carried into the KS driver.

## What exists

- **The basis transform.** `Basis::_cart_to_sph` is the block-diagonal
  `C` (n_sph × n_cart). Spherical AOs are `φ_sph = C φ_cart`.
- **The HF skin** (`src/scf/scf.cpp`, `direct_fock_rhf/uhf`): lift the
  density, `D_cart = Cᵀ D_sph C`, build in Cartesian, project the result,
  `G_sph = C G_cart Cᵀ`. The integral engines never see spherical functions.
- **The gradient lift** (`BasisFunctions::lift_density_sph_to_cart`,
  `src/basis/spherical.{h,cpp}`): lift P and W once and reuse the Cartesian
  derivative integrals. For ROHF, W is built in the
  spherical basis and lifted once, never assembled from lifted factors.
- **The working-basis rebuild** (`SCF::rebuild_basis_dependent_state`,
  `src/scf/working_state.cpp`): renormalizes `C` against the new Cartesian
  overlap and stores `S_sph`, `H_sph` at every displaced geometry. The DFT
  driver does not call it: it builds `S` and `T + V` in Cartesian directly.
- **Spherical MP2** in the HF binary, which the double-hybrid PT2 step needs.

## Why the skin covers DFT too

Everything on the grid is linear in the AOs. With `P_cart = Cᵀ P_sph C`,
the density, its gradient and τ are identical to the spherical ones, and the
XC matrix projects exactly: `V_xc,sph = C V_xc,cart Cᵀ`. So the XC layer
(`xc_grid.cpp`, `ao_grid.h`) can stay Cartesian. The same holds for J, K,
the PCM reaction field and every MO-based quantity: MO coefficients lift as
`c_cart = Cᵀ c_sph`, after which TDDFT's MO integrals
(`Correlation::transform_eri` on the Cartesian tensor) are already right.

The risk is not the algebra but the bookkeeping. About 48 sites in
`src/dft/` read `_shells.nbasis()` or `ao_grid.nbasis()` (38 in `driver.cpp`
alone). Each is either an AO-grid/integral dimension (stays Cartesian) or a
working-basis dimension (becomes `working_nbasis()`). A wrong choice is a
dimension mismatch that Release builds turn into undefined behaviour, not an
error.

## Design

One boundary, one helper pair, nothing spherical below it:

- `to_cart(M_sph) = Cᵀ M C` and `to_working(M_cart) = C M Cᵀ` (identity in
  Cartesian mode), used at the KS-matrix assembly point for every
  density-to-matrix build: J, K (global, long-range, short-range), V_xc,
  V_rxn.
- The SCF scaffold (orthogonalizer, guess, DIIS, diagonalization,
  checkpoint) works in the working basis, sized by `working_nbasis()`.
- The DFT driver's one-electron setup calls `rebuild_basis_dependent_state`
  instead of building `S` and `H` itself, so geometry optimization and
  frequencies get the renormalized transform for free.

## Steps

1. **Audit and classify** the ~48 dimension sites as grid/integral
   (Cartesian) or working. No behaviour change; Cartesian runs stay bitwise.
2. **RKS and UKS single points** (LDA, GGA, global and range-separated
   hybrids) through the boundary helpers.
   - Gate 1, exact: in a basis with only s and p shells (6-31G) spherical
     and Cartesian span the same space, so energies must agree to 1e-10.
   - Gate 2, PySCF `cart=False`: water and a C1 molecule, 6-31G\* and
     cc-pVTZ, B3LYP / PBE / HSE06. Set the tolerance by first running the
     same comparison with `cart=True`, which measures the grid floor between
     the two codes; spherical must land inside that floor.
3. **Analytic gradients, geometry optimization, frequencies** through the
   existing lift. Gates: FD of the energy, PySCF `cart=False` gradients, and
   a spherical geomopt of water/6-31G\* matching PySCF's optimized energy.
4. **TDDFT** with lifted MO coefficients. Gate: PySCF `cart=False` TDA and
   Casida excitation energies at the existing TDDFT tolerances.
5. **SOSCF, PCM, symmetry (SAO blocking), double hybrids.** Each either goes
   through the same helpers or keeps an explicit rejection. Double-hybrid
   gradients have their own dense machinery and are the likeliest to stay
   rejected in this scope.

## Not in this scope

Evaluating spherical AOs directly on the grid. It would save the transform
but duplicate the AO layer, and the skin is exact.

## Depends on / enables

Nothing blocks it. It is a prerequisite for useful def2 DFT
(`docs/DEF2_ECP_SCOPE.md`).
