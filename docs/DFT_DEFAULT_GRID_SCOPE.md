# Should `planck-dft` change its default grid policy, and to what?

Scope, opened 2026-11-10. A decision, not new code. It expires when the
decision is made; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 4). The grid itself is answered in
`docs/DFT_GRID_CONVERGENCE.md`.

## The question

The default `grid_policy` is `legacy` (`src/base/types.h`, `_grid_policy`).
On the hard UKS double-hybrid cases it fails rotational covariance (2.2e-5
Ha/Bohr on C1 H2O2⁺, against a 2e-6 limit). The qualified opt-in `g4-3-v4`
passes (1.1e-7). Default-settings geometries and frequencies therefore carry
grid noise that a user cannot see. `DFT_GRID_CONVERGENCE.md` says the
migration needs "a separate review with a compact calibration manifest" and
that whether legacy's error matters for ordinary (non-DH) workflows "has not
been measured".

## What to measure

All on the default preset (`grid normal`) unless stated, both policies:

1. **Rotational covariance of ordinary gradients.** PBE, B3LYP and HSE06 RKS
   and one UKS, on C1 H2O2 and two medium molecules, at three random
   orientations. Metric: max component deviation after rotating back. This is
   the number that says whether legacy's failure is DH-only.
2. **Frequency noise.** One small molecule with a low mode (H2O2 torsion):
   frequencies under the same three orientations.
3. **Cost.** Grid points per atom and SCF wall time on a 4-water and a
   12-water chain, 6-31G* and cc-pVTZ. V4's frozen-atlas maximum is 64,998
   points per atom; compare it with legacy at each preset.
4. **Energies.** How far every existing DFT regression energy moves (77
   cases). They will move; the question is whether any moves past its
   tolerance for a reason other than "the grid changed".

## Findings so far (2026-11-18)

Driver: `tests/dft_grid_orientation_audit.py` (input orientation plus three
rigid rotations; gradients rotated back; max over components and rotations).
6-31G*, Cartesian, `grid normal`. V4 always runs at its qualified 135 shells.

### (1) Rotational covariance: legacy fails on every functional

| Case | legacy max\|Δg\| | V4 max\|Δg\| | legacy max\|ΔE\| | V4 max\|ΔE\| |
|---|---:|---:|---:|---:|
| C1 H2O2 PBE | 5.5e-4 | 2.0e-7 | 9.6e-5 | 1.1e-8 |
| C1 H2O2 B3LYP | 4.5e-4 | 1.3e-7 | 8.9e-5 | 1.3e-8 |
| C1 H2O2 HSE06 | 5.0e-4 | 2.3e-7 | 7.8e-5 | 1.3e-7 |
| C1 H2O2⁺ UKS PBE | 4.4e-4 | 1.8e-7 | 1.0e-4 | 1.0e-8 |
| benzene B3LYP | 4.9e-4 | 8.4e-7 | 1.3e-4 | 1.8e-7 |
| water4 chain (3 Å) B3LYP | 1.3e-3 | 1.0e-6 | 1.3e-3 | 1.2e-6 |

Gradients in Ha/Bohr, energies in Eh. The default grid's gradient depends
on orientation at 5e-4–1.3e-3 Ha/Bohr, above the default geomopt force
threshold (3e-4). So legacy's failure is not DH-only.

**Why the candidate evaluation missed it.** `legacy` was only ever measured
as a comparison column on the UKS DH fixtures, at elevated presets
(2.2e-5 there, matching `legacy ultrafine` here, 1.4e-5). The shipped
`grid normal` was never measured. Of 31 DFT derivative regression inputs,
20 override `grid_angular_points` (bypassing pruning), 2 use V4, and none
rotates the input.

### legacy's error is its angular pruning

H2O2 PBE, legacy radial counts unless noted:

| Angular grid | max\|Δg\| | max\|ΔE\| |
|---|---:|---:|
| pruned `normal` (default) | 5.5e-4 | 9.6e-5 |
| pruned `fine` | 2.7e-4 | 3.8e-5 |
| pruned `ultrafine` | 1.4e-5 | 2.0e-6 |
| unpruned 302 | 1.1e-5 | 1.3e-6 |
| unpruned 590 | 1.3e-6 | 9.1e-8 |
| unpruned 590, 135 shells | 1.4e-6 | 9.1e-8 |

Going from 75 to 135 radial shells changes nothing.

### V4's error grows with neighbouring molecules, not atom count

V4 energy, water chain, B3LYP:

| System | max\|ΔE\| |
|---|---:|
| monomer | 1.2e-9 |
| dimer (3 Å) | 5.6e-7 |
| tetramer (3 Å) | 1.2e-6 |
| tetramer (6 Å) | 8.6e-8 |

One neighbour accounts for the jump (480×); doubling again adds 2×; moving
the molecules apart cuts it 14×. V4 sets the angular order by u = r/r_Bragg,
with small outer bands (second row 170, H 434). Every selection and blind
fixture was a single small molecule, whose outer shells hold only a decaying
tail. In a cluster those shells pass through the neighbouring molecules.

Unpruned grids, 135 shells, energy (water4 at 3 Å):

| Angular | water1 | water4 | water4 / (4 × water1) |
|---:|---:|---:|---:|
| 590 | 3.55e-8 | 6.53e-8 | 0.46 |
| 974 | 8.44e-9 | 3.69e-8 | 1.09 |
| 1454 | 7.6e-10 | not run (5.7 GB, swaps) | |

V4 on the same water4 is 1.2e-6, so its pruning, not the angular order of
its valence band, causes the cluster error. 200 radial shells at 590 give
6.6e-8 (unchanged). PySCF on the same grids (Becke + Treutler adjustment,
`tests/pyscf/becke_floor_orientation.py`) gives water1 3.551e-8 and water4
6.535e-8 at 590, matching Planck to three digits.

**No partition floor.** All four waters share one orientation, so a
rotation moves each molecule's quadrature error the same way and they add:
water4 ≈ 4 × water1 is the no-cluster-effect expectation, and 974 meets it.
The monomer converges fast (3.6e-8 → 8.4e-9 → 7.6e-10). An earlier draft
called the water4 590→974 decline a Becke-partition floor; that compared
cluster grids without the monomer control at the same grid, and was wrong.
Not run: water4 at 1454 (needs the X2 memory fix).

### (3) Cost, provisional

H2O2/6-31G* points: legacy normal 35,180; fine 78,148; ultrafine 147,396;
V4 322,080 (9.2× normal). V4 wall time is today dominated by the serial,
per-point KS XC gradient and its ten stored npts × nb arrays (benzene:
117 s on one core, 8.1 GB). Fix scoped in
`docs/DFT_XC_GRADIENT_PERFORMANCE_SCOPE.md`; the cost comparison waits on it.

### What this means for the candidates

- **Not a re-derivation.** The method and the qualification protocol stand.
  V4 passes on isolated molecules (1e-9); its atlas never contained the
  intermolecular regime. Re-select (a V5) with cluster fixtures (water
  dimer/tetramer at H-bond and 3–6 Å spacings, benzene) in both the
  selection atlas and the blind suite, keeping the Z-and-u-only contract.
  Expect larger outer bands.
- **The partition needs no change.** Unpruned grids converge on the
  cluster as fast as on the monomer, and match PySCF.
- **Still to measure:** gradient (not only energy) covariance on the
  clusters for a V5 candidate, which needs the X1/X2 memory fix to run
  at 1454-class grids on this machine.

## Decision rule

- If (1) shows legacy above ~1e-6 Ha/Bohr on ordinary functionals, or (2)
  shows orientation-dependent frequencies above ~1 cm⁻¹, and V4 costs under
  ~1.5× → make V4 the default.
- If legacy is fine outside DH → keep it, and make DH gradients select V4
  automatically instead of refusing (`validate_qualified_uks_dh_grid`
  currently refuses without it).
- Otherwise the cost decides; record the trade-off.

## Migration if V4 wins

One line in `types.h`. Then re-pin the DFT regression references that moved
(each against PySCF on a matching grid where one exists, or against the new
Planck value with the old one in the commit message), keep `legacy` as a
named policy, and update `DFT_GRID_CONVERGENCE.md` and the vault. Run
`--suite all` (the Bohr-constant lesson: one tag is not enough).

