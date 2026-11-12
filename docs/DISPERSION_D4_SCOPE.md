# What does `planck-dft` need for D4 dispersion?

Scope, opened 2026-11-10. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 10). D3 is answered in
`docs/DISPERSION_D3.md`.

## Why it matters, and why it is last

D4 is the current Grimme default (r2SCAN-3c, ωB97X-3c, and recommended over
D3 for new work). D3(BJ) covers most uses, so this is "expected", not
"necessary".

## What D3 already provides

`DFT::Dispersion::compute` (`src/dft/dispersion.{h,cpp}`) is called once per
KS scaffold call and once in `compute_analytic_ks_gradient`; ghost atoms are
excluded there; parameter names come from the resolved libxc ids. D4 should
plug into the same two call sites through the same function, with a
`dispersion d4` keyword branch. Nothing else in the driver changes.

## The dependency

dftd4 (Grimme group, LGPL-3.0, Fortran, CMake/meson) is the reference
implementation. It needs **atomic partial charges**; it computes its own
electronegativity-equilibration (EEQ) charges via the `multicharge` library,
so Planck does not have to supply them. Build it as an `ExternalProject`
like simple-dftd3 (which already brought a Fortran toolchain into the
build). It also depends on mctc-lib and multicharge; check whether
simple-dftd3's mctc-lib can be shared, or whether two copies conflict at
link time.

## Steps

1. Build dftd4 hermetically next to simple-dftd3; a `PLANCK_WITH_DFTD4`
   option, off if the Fortran build fails, like D3.
2. Branch in `Dispersion::compute` for `d4` (and `d4-atm` is the default
   three-body in D4; follow dftd4's defaults rather than D3's naming).
3. Gate: D4 energies and water-dimer gradient against `dftd4`'s Python API
   (or PySCF's `mf.disp = 'd4'` if pyscf-dispersion loads; it did not on
   macOS arm64 for D3, see `DISPERSION_D3.md`). Every printed digit, plus a
   geomopt E_opt case, as for D3.

## Not in this scope

Charge-dependent D4 using Planck's own SCF charges (dftd4 uses EEQ by
design), and VV10.
