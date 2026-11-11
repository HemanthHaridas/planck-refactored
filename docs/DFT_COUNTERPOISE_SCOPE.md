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
else. *Done 2026-11-11:* `DFT::Driver::run` (the CLI entry `planck-dft` and
`planck-mpi` share) refuses `_bsse._enabled`; gated by
`he2_rks_pbe_bsse_rejected_sto3g`. Step 2 replaces the refusal.

## What already holds for ghost atoms in DFT

Since the 2026-11 ECP and ghost audits the electron counts in the KS scaffold
use `total_nuclear_charge()`; D3 leaves ghosts out (`dispersion.cpp`). The
grid is built over all `atomic_numbers`, ghosts included, which is what
PySCF does (ghost atoms carry grid points so their basis functions are
integrated). Nothing gates any of it: no DFT regression case has a ghost
atom.

## Steps

### 1. Ghost-atom DFT single points — done 2026-11-11

**Decision: a ghost atom's grid is built with its element's Z** (radial
count, Treutler ξ, pruning, Becke radius), exactly as for the real atom.
Planck already did this (the grid reads `atomic_numbers`, which ghosts keep;
the `radial.h` index-0 entry is never reached for a ghost). Two reasons: the
ghost carries the element's tight basis functions, which a Z = 0 (H-like)
grid integrates poorly; and the dimer and ghosted-monomer runs then use the
same grid, so its error cancels in the counterpoise difference. PySCF ≥ 2.14
does the same (pyscf/pyscf#3302, checked against Q-Chem). PySCF 2.13 built
ghost grids at Z = 0, which #3302 fixed as a bug; its gradient side is
pyscf/pyscf#3597 / #3598.

Gates (water dimer, second monomer ghosted, 6-31G*, cart, 150 × 590):
`water_dimer_ghost_rks_{pbe,b3lyp}_631gd`,
`water_dimer_ghost_cation_uks_pbe_631gd`. All three match PySCF to every
printed digit (`tests/pyscf/ghost_dft_references.py`, which on 2.13 builds
the grid on the molecule with ghosts made real). PySCF 2.13's own Z = 0 ghost
grid is 3.1e-8 Eh off on the PBE case, so the 1e-9 gate tells them apart.

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

### 3. Symmetry — done 2026-11-11

The DFT driver lacked the HF driver's switch: He + Gh(He) was detected as
D∞h and the RKS SCF diverged. `setup_symmetry` (`src/dft/driver.cpp`) now
forces symmetry off when ghosts are present, with the HF warning;
`water_dimer_ghost_rks_pbe_631gd` runs with `use_symm .true.` and asserts it.

## Not in this scope

Counterpoise-corrected gradients and geometry optimization, N-body
counterpoise, and post-HF counterpoise (nice-to-have in Open Work).
