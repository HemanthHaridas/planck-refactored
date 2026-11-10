# What does `planck-dft` need for ghost atoms and counterpoise?

Scope, opened 2026-11-10. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 6). The HF design is answered in
`docs/BSSE_PLAN.md`.

## Why it matters

Interaction energies are mostly computed with DFT. Planck's counterpoise
driver (`src/bsse/counterpoise.cpp`) is HF-only: it builds five sub-
`Calculator`s and calls `SCF::run_rhf` / `run_uhf`.

## The silent part, first

`planck-dft` never reads `calculator._bsse`. The parser fills it from
`%begin_bsse` for either binary, and only `src/hf_driver.cpp` acts on it. A
`planck-dft` input with a `%begin_bsse` block therefore runs the plain
molecule and prints no counterpoise result, without a word. **Step 0 is a
refusal by name** in the DFT driver, with a regression case, before anything
else.

## What already holds for ghost atoms in DFT

Since the 2026-11 ECP and ghost audits the electron counts in the KS scaffold
use `total_nuclear_charge()`; D3 leaves ghosts out (`dispersion.cpp`). The
grid is built over all `atomic_numbers`, ghosts included, which is what
PySCF does (ghost atoms carry grid points so their basis functions are
integrated). Nothing gates any of it: no DFT regression case has a ghost
atom.

## Steps

### 1. Ghost-atom DFT single points

Gate RKS and UKS single points with ghost atoms against PySCF (`ghost-O`
syntax, `mol.cart = True`): water dimer with one monomer ghosted, PBE and
B3LYP, 6-31G*. Check what to compare: PySCF ≥ 2.14 gives ghosts their own
Bragg radius in the Becke partition (#3302); match whatever Planck's
`radial.h` table does for index 0 (radius 1.0) or change it. A ghost's grid
treatment changes the energy at the 1e-6 level, so this choice must be made
deliberately and written down.

### 2. Counterpoise driver for KS

The orchestration is reference-agnostic except for the SCF call. Make the
sub-calculation runner a function parameter (or dispatch on the presence of
a DFT functional) so the same five-calculation driver calls the KS SCF.
One mechanism, not a second copy of `counterpoise.cpp`. The KS SCF entry the
driver needs is whatever `DFT::Driver` uses for a single point; if it is not
callable on a sub-`Calculator`, extracting it is the main work.

Gate: He2 and water dimer, B3LYP/cc-pVDZ (spherical), CP-corrected
interaction energy against PySCF to 1e-8 Eh. Add D3 to one case (ghosts must
contribute nothing).

### 3. Symmetry

The HF driver forces symmetry off when ghosts are present
(`hf_driver.cpp`); the DFT driver needs the same switch. Verify, do not
assume.

## Not in this scope

Counterpoise-corrected gradients and geometry optimization, N-body
counterpoise, and post-HF counterpoise (nice-to-have in Open Work).
