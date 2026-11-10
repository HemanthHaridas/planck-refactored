# What does UKS double-hybrid geometry optimization and frequencies need?

Scope, opened 2026-11-10. Driver wiring and gates. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 13). The gradient is answered in
`docs/UDH_GRADIENT_ONE_ARCHITECTURE.md`.

## The state

The UKS global-double-hybrid analytic gradient is accepted: ≤ 4.3e-9 Ha/Bohr
against fourth-order energy FD on the V4 grid. `validate_workflow_support`
(`src/dft/driver.cpp`) admits it for `CalculationType::Gradient` only; GeomOpt,
Frequency and GeomOptFrequency fall through to the final rejection. The RKS
DH branch admits all four, and RKS frequencies are central differences of
the analytic gradient on the same callback.

## Steps

### 1. Admit the workflows

Extend the UKS DH branch's `case` list to GeomOpt, Frequency and
GeomOptFrequency, keeping all its conditions (no solvent, all-active
conventional PT2 with stored T2, `validate_qualified_uks_dh_grid`). Check
the geomopt and frequency loops call the same gradient callback the
Gradient workflow uses, and that the V4 grid policy survives
`prepare_current_geometry` at every displaced geometry (the grid is rebuilt
per geometry; the policy must not reset to the default).

### 2. Gates

PySCF has no DH gradient (`grad/mp2.py` is not a valid DH reference; see
CLAUDE.md, DFT). Use the RKS DH validation approach:

- **geomopt**: OH or HO2 radical, B2PLYP/STO-3G, V4 grid. Check that the
  final gradient is below threshold and that the energy at the optimized
  geometry is a minimum along each Cartesian direction by energy FD (Planck
  against Planck). Pin E_opt.
- **frequencies**: the same molecule at its optimized geometry; compare the
  Hessian from FD of analytic gradients with the Hessian from second
  differences of energies (fourth-order stencil), ≤ 1 cm⁻¹.
- Closed-shell control: `scf_type uhf` on water must reproduce the RKS DH
  frequencies (the gradients already agree to 2.4e-11).

### 3. Cost

A UKS DH frequency run is 6N UKS DH gradients on the V4 grid. Time one on
HO2/6-31G before putting any case in `core`; it may belong in `extended`.
The warm-start lever in `docs/ANALYTIC_HESSIAN_SCOPE.md` helps here too.

## Not in this scope

Imaginary-mode following (refused for DFT generally), solvent, frozen core,
and larger-basis qualification of the UKS gradient
(`docs/DH_GRADIENT_EXTENSIONS_SCOPE.md`).
