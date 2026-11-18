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
V4 322,080 (9.2× normal). V4 wall time was dominated by the serial,
per-point KS XC gradient and its ten stored npts × nb arrays (benzene:
117 s on one core, 8.1 GB). X1–X2 of
`docs/DFT_XC_GRADIENT_PERFORMANCE_SCOPE.md` brought that to 27.5 s and
3.6 GB, and X3 (threading) to 15.1 s at 8 threads (legacy `fine`:
5.8 s).

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

## Remaining work (scoped 2026-11-18, after X3)

The XC-gradient performance work has landed (X1–X3b,
`docs/DFT_XC_GRADIENT_PERFORMANCE_SCOPE.md`). Benzene B3LYP/6-31G\* at
8 threads: V4 gradient 15.1 s, legacy `fine` 5.8 s. Peak memory is still
set by the stored SCF AO arrays (V4 benzene 3.6 GB). Three things follow.

**1. The decision rule below cannot be met as written.** "V4 costs under
~1.5×" is not reachable: V4 has 9.2× the points of `normal` on H2O2, and the
first wall-time comparison is 2.6× `fine`. "Legacy above ~1e-6 Ha/Bohr" is
already true at 4–13e-4, so its first branch is decided and only cost is
left. Replace it with two tiers, tied to the optimizer's own thresholds:

| Tier | Gradient covariance target | Who needs it |
|---|---:|---|
| default (`grid normal`) | ≤ 3e-5 Ha/Bohr (1/10 of the default `geomopt_grad_tol`, 3e-4) | every geometry and frequency run |
| tight (opt-in preset, or the DH gate) | ≤ 1.5e-6 Ha/Bohr (1/10 of the `gau_tight` 1.5e-5) | DH gradients, tight optimizations, PySCF-matched gates |

plus frequency noise ≤ 1 cm⁻¹ at the default tier. On the measurements so
far no single policy covers both tiers at an acceptable cost: legacy
`normal` misses tier 1 by 20×, and V4 meets tier 2 only on isolated
molecules (water4 at 3 Å: 1.0e-6 Ha/Bohr, at the limit).

**2. Several candidates need no new code.** `src/dft/base/grid.h` already
has `legacy-unpruned-max` and `pyscf-nwchem-2.13` next to `legacy` and V4.
The H2O2 table above already puts unpruned 302 (legacy radial) at
1.1e-5 Ha/Bohr, inside tier 1, and pruned `ultrafine` at 1.4e-5. Measure
before designing anything:

- **G1, candidate matrix.** Policies `legacy` at normal/fine/ultrafine,
  `legacy-unpruned-max` at normal/fine, `pyscf-nwchem-2.13` at its default
  level (it shows whether Planck's default is simply worse than PySCF's or
  both share the problem), and V4. Fixtures: the six above plus a water
  dimer at its H-bond distance (O–O ≈ 2.9 Å), the case V4 was never
  selected on. Per cell: max |Δg|, max |ΔE|, grid points/atom, wall time
  (8 threads) and peak footprint. The audit driver
  (`tests/dft_grid_orientation_audit.py`) needs two small additions: record
  points, wall time and peak memory per run, and the water-dimer fixture.
  About 8 policy/preset pairs × 7 fixtures × 4 orientations ≈ 220 runs,
  minutes each at most; V4 water4 is the largest.
- **G2, frequency noise** (the open item (2) above). H2O2 B3LYP/6-31G\*
  frequencies at the input orientation and three rotations, for the
  candidates that pass G1 tier 1, and for legacy `normal` as the baseline.
  The low torsion mode is the one to watch.
- **G3, cost at size.** For the two or three G1 survivors: points/atom and
  SCF + gradient wall time on water4 and water12 chains, 6-31G\* (and
  cc-pVTZ on water4). Baseline already measured: water12 legacy `fine`
  gradient 73.6 s at 8 threads, peak 4.8 GB, so memory, not time, is the
  first limit at this size. Water12 on V4 needs ~22 GB (the X4 item), so on this
  8 GiB machine V4 water12 is "does not run"; record it as such rather than
  swapping.

**3. Decide from G1–G3.** Expected outcome, to be confirmed: a cheap
default (unpruned or lightly pruned at 302) for tier 1, and the tight tier
kept as an explicit preset.

- If an existing policy meets tier 1 on every fixture at ≤ ~2× legacy
  `normal` cost, make it the default (the migration section below).
- **Tight tier.** V4 meets it on isolated molecules only, so it stays as is
  for DH (already enforced) and gets a user-facing name. Re-selecting it on
  clusters (V5) is a separate scope, opened only if a tight-tier user needs
  clusters: it needs cluster fixtures in a new selection atlas and a new
  blind suite under the `DFT_GRID_CONVERGENCE.md` protocol, and is expected
  to come out with larger outer bands and more points.
- **DH:** independent of the default, make UKS DH gradients select V4
  automatically instead of refusing (`validate_qualified_uks_dh_grid`).
- **(4), energy movement**, only after a choice: run the 77 DFT
  regression energies on the new default and classify every move, as in
  the migration section.

## Plan: both DFT scopes, in order

This merges the open items of this scope and of
`docs/DFT_XC_GRADIENT_PERFORMANCE_SCOPE.md` (X1–X3b done). Each step names
what it needs first and what it produces.

| Step | Needs | Produces |
|---|---|---|
| S1 audit driver | — | the measuring tool for S3–S5 |
| S2 profile | — | where time and memory go now |
| S3 G1 matrix | S1 | which policies meet tier 1 |
| S4 G2 frequencies | S3 | frequency noise of the survivors |
| S5 G3 cost at size | S3, S2 | cost and memory of the survivors |
| S6 choose | S3–S5 | default policy, tight preset |
| S7 X4 go/no-go | S5, S6 | whether to stream the AO arrays |
| S8 migration | S6 | the new default, re-pinned suite |
| S9 DH auto-select | S6 | DH gradients pick the tight preset |
| S10 V5 | S6 | only if tight-tier clusters are needed |

**S1, extend `tests/dft_grid_orientation_audit.py`.** Small code change.
Per run, also record grid points (turn on `print_grid_summary` and parse it),
wall time, and peak footprint (`/usr/bin/time -l` on macOS). Add the
H-bonded water dimer (O–O ≈ 2.9 Å) to `CASES`. Add a `--threads` option,
default 8, so cost columns are comparable. Check it on one cell against the
2026-11-18 H2O2 PBE numbers above (legacy 5.5e-4, V4 2.0e-7).

**S2, profile today's cost.** Benzene B3LYP/6-31G\* V4 gradient (15.1 s)
and water12 legacy `fine` gradient (73.6 s), 8 threads, with `sample`.
Report the phase shares (grid build, AO evaluation, SCF XC build, J/K, XC
gradient, HF-like gradient) and the memory owners (the four stored AO
arrays, at about npts × nb × 32 B, against the 4.8 GB water12 peak). This
does not block S3. It does decide whether a cheap performance fix belongs
before S5. For example, if a serial phase dominates, S5 would be measuring
that phase, not the grid.

**S3, G1 candidate matrix.** As in "Remaining work" item 2. Policies:
`legacy` at normal/fine/ultrafine, `legacy-unpruned-max` at normal/fine,
`pyscf-nwchem-2.13`, and V4. Fixtures: the six above plus the dimer. Pass
mark: tier 1 (≤ 3e-5 Ha/Bohr) on every fixture. Record tier 2 as a column.
Output: the survivor list, with points/atom and the small-molecule cost.

**S4, G2 frequency noise.** H2O2 B3LYP/6-31G\* frequencies at four
orientations, for the S3 survivors and legacy `normal`. Pass mark: ≤ 1 cm⁻¹,
watching the torsion. Needs a `--frequency` mode in the S1 driver (the
modes compare directly; no rotation back).

**S5, G3 cost at size.** S3 survivors that also pass S4, on water4 and
water12 (6-31G\*) and water4 cc-pVTZ: SCF and gradient wall time, and peak
footprint. Read the numbers against S2's phase shares. Record a run that
would not fit in 8 GiB as "does not run" rather than letting it swap.

**S6, choose.** Make the cheapest policy that passes S3–S4 the default if
it costs ≤ ~2× legacy `normal` at water12. Otherwise record the trade-off
and pick explicitly. Give the tight tier a user-facing preset name that
maps to V4 at 135 shells. Write the outcome here and in
`DFT_GRID_CONVERGENCE.md` ("Not done").

**S7, X4 go/no-go** (the perf scope's item 4). Go only if S5 shows the
chosen default, or the tight preset at a size a user needs, does not fit
in 8 GiB. Then X4 becomes its own scope. It is cross-cutting: a per-block
AO provider for `xc_grid.cpp`, `ks_matrix.cpp`, `analytic_hessian.cpp`, the
DH paths and the driver (11 files, ~48 sites), plus AO re-evaluation every
SCF iteration. Its design starts from S2's memory owners.

**S8, migration.** Do the "Migration if V4 wins" section, with the S6
policy in place of V4: one default change, the 77 DFT regression energies
classified, references re-pinned, `--suite all`.

**S9, DH auto-select.** `validate_qualified_uks_dh_grid` selects the tight
preset instead of refusing an unqualified grid, and says so in the output.
It needs S6 only for the preset name, so it can follow S6 directly,
before S8.

**S10, V5.** Only if S6 finds a user who needs the tight tier on clusters
(V4's water4 is 1.0e-6, at the limit). Its own scope, under the
`DFT_GRID_CONVERGENCE.md` protocol: cluster fixtures in a new selection
atlas and a new blind suite.

S1 and S2 can run in parallel. After S3, the critical path is S4 → S5 → S6.

## Decision rule (superseded by "Remaining work" above)

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

