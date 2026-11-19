# Should `planck-dft` change its default grid policy, and to what?

Scope, opened 2026-11-10. A decision: what needs writing is a one-line
default change, a tight preset name, and re-pinned references. Rewrite this
as the answer once the decision is made. Status lives in
`vault/Status/Open Work.md` (priority 4). The pruning qualification is
answered in `docs/DFT_GRID_CONVERGENCE.md`, the gradient cost work in
`docs/DFT_XC_GRADIENT_PERFORMANCE.md`.

## The question

The default `grid_policy` is `legacy` (`src/base/types.h`, `_grid_policy`)
at `grid normal`. Its gradients depend on molecular orientation by
4–13e-4 Ha/Bohr, above the default geometry-optimization force threshold
(3e-4). Default-settings geometries and frequencies therefore carry grid
noise the user cannot see.

## The requirement

The noise only has to be small against what the optimizer resolves:

| Tier | Gradient orientation noise | For |
|---|---:|---|
| default (`grid normal`) | ≤ 3e-5 Ha/Bohr (1/10 of the default `geomopt_grad_tol`, 3e-4) | every geometry and frequency run |
| tight (opt-in preset; DH gate) | ≤ 1.5e-6 Ha/Bohr (1/10 of `gau_tight`, 1.5e-5) | DH gradients, tight optimizations, PySCF-matched gates |

Plus frequency noise ≤ 1 cm⁻¹ at the default tier. The default needs
headroom, not a bare pass: the fixtures are small molecules, and noise grows
with system size (benzene → naphthalene doubles it).

## Where the answer stands

- **Default: `pyscf-nwchem-2.13@fine` is the only candidate.** It passes
  the default tier on every fixture with ≥ 9× margin. Nothing at `normal`
  level passes: the two cheapest settings fail naphthalene. Open: frequency
  noise (S4) and cost at size (S5).
- **Tight preset: `pyscf-nwchem-2.13@ultrafine`.** It meets 1.5e-6 on
  benzene (4.9e-7) and naphthalene (8.2e-7) with fewer points than V4,
  which misses naphthalene (3.0e-6). Its margin on naphthalene is only
  1.8×, and noise grows with aromatic size, so a larger aromatic should be
  checked before it is named (S3e). Open also: re-qualifying the UKS DH
  gradient on it (S9).
- **Memory no longer limits the measurements** at the sizes in this scope,
  since AO screening (S7, below). V4 on naphthalene in 6-31G\* (~6.5 GB) is
  the exception on an 8 GiB machine.

## Evidence

### Orientation noise

Driver: `tests/dft_grid_orientation_audit.py`. It runs the input orientation
plus three rigid rotations, rotates the gradients back, and reports the max
over components and rotations, with grid points, wall time and peak
footprint. 6-31G\*, Cartesian, 8 threads; V4 at its qualified 135 radial
shells. Raw rows: `tests/benchmarks/dft/grid_orientation/*.{json,log}`.

The settings, as the driver spells them:
- `pyscf-nwchem-2.13@normal` is PySCF's default grid: `normal` maps to PySCF
  level-3 radial counts, and the policy is PySCF 2.13 `_default_ang` +
  `nwchem_prune` without the light-atom reduction (`grid.h`,
  `resolve_grid_spec`). `@fine` is level 5, `@ultrafine` level 7.
- `legacy-unpruned-max` puts the scheme's largest region on every shell:
  302 for second-row atoms at `normal`, 434 at `fine`.

max|Δg| in Ha/Bohr; ✗ fails the default tier, **bold** meets the tight tier.
"pts" is the point count relative to `legacy@normal`.

| Setting | H2O2 PBE | H2O2 B3LYP | H2O2 HSE06 | H2O2⁺ UKS PBE | water dimer | water4 (3 Å) | benzene | naphthalene | pts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy@normal` | 5.5e-4 ✗ | 4.5e-4 ✗ | 5.0e-4 ✗ | 4.4e-4 ✗ | 1.2e-3 ✗ | 1.3e-3 ✗ | 4.9e-4 ✗ | 6.6e-4 ✗ | 1 |
| `legacy@fine` | 2.7e-4 ✗ | | | | | 8.3e-5 ✗ | | | 2.2 |
| `legacy@ultrafine` | 1.4e-5 | 1.4e-5 | 1.2e-5 | 1.1e-5 | 1.5e-5 | 1.0e-5 | 2.8e-5 | | 4.2 |
| `legacy-unpruned-max@normal` | 1.1e-5 | 1.2e-5 | 1.3e-5 | 8.5e-6 | 1.5e-5 | 1.0e-5 | 3.0e-5 | 3.2e-5 ✗ | 1.8 |
| `legacy-unpruned-max@fine` | 2.5e-6 | 2.4e-6 | 2.4e-6 | 3.4e-6 | 2.6e-6 | **1.4e-6** | 8.7e-6 | | 3.8 |
| `pyscf-nwchem-2.13@normal` | 1.1e-5 | 1.2e-5 | 9.4e-6 | 1.0e-5 | 5.5e-6 | 9.1e-6 | 3.0e-5 | 3.7e-5 ✗ | 1.3–1.5 |
| `pyscf-nwchem-2.13@fine` | **5.0e-7** | **3.4e-7** | **9.4e-7** | **5.2e-7** | **4.8e-7** | **2.3e-7** | 1.5e-6 | 3.1e-6 | 3.7–4.0 |
| `pyscf-nwchem-2.13@ultrafine` | | | | | | | **4.9e-7** | **8.2e-7** † | 7.6–7.8 |
| `g4-3-v4` | **2.0e-7** | **1.3e-7** | **2.3e-7** | **1.8e-7** | **8.2e-8** | **1.0e-6** | **8.4e-7** | 3.0e-6 † | 8.7–10.7 |

† 6-31G, not 6-31G\*: V4 in 6-31G\* needs ~6.5 GB and thrashed on the
8 GiB machine; `@ultrafine` was run in 6-31G to match. The basis barely matters here: in 6-31G legacy is 6.75e-4
and `pyscf-nwchem@fine` 3.28e-6, within 5 % of their 6-31G\* values.

Values at a limit, in full: benzene `legacy@ultrafine` 2.816e-5,
`legacy-unpruned-max@normal` 2.961e-5, `pyscf-nwchem@normal` 2.985e-5,
`pyscf-nwchem@fine` 1.528e-6. Energies follow the same order
(`pyscf-nwchem@fine` max|ΔE| ≤ 2.7e-7 Eh on every fixture; V4 1.2e-6 on
water4, 5.2e-7 on naphthalene).

What the table says:
- **Legacy fails because of its angular pruning** (next section), 15–40×
  over the default tier on every functional.
- **The aromatics bind, not the clusters.** Every setting that passes has
  margin on water4; the cheap ones reach the limit on benzene and fail
  naphthalene.
- **`pyscf-nwchem@fine` dominates the other robust settings.** Against
  `legacy-unpruned-max@fine` (same points) it has 3–17× less noise. Against
  V4 it has ~2.5× fewer points, is better on both water clusters, and equal
  on naphthalene.
- **The tight tier needs `pyscf-nwchem@ultrafine`.** V4 and
  `pyscf-nwchem@fine` both double from benzene to naphthalene and miss it
  there; `@ultrafine` grows 1.7× (4.9e-7 → 8.2e-7) and passes. At about
  the same points as V4 it is 3.7× quieter on naphthalene. Why V4 does
  worse on aromatics is not measured; its outer bands (second row 170
  beyond 4.5 r_Bragg) are the first suspect, as on clusters.

### Legacy's error is its angular pruning

H2O2 PBE, legacy radial counts unless noted:

| Angular grid | max\|Δg\| | max\|ΔE\| |
|---|---:|---:|
| pruned `normal` (default) | 5.5e-4 | 9.6e-5 |
| pruned `fine` | 2.7e-4 | 3.8e-5 |
| pruned `ultrafine` | 1.4e-5 | 2.0e-6 |
| unpruned 302 | 1.1e-5 | 1.3e-6 |
| unpruned 590 | 1.3e-6 | 9.1e-8 |
| unpruned 590, 135 shells | 1.4e-6 | 9.1e-8 |

75 → 135 radial shells changes nothing.

### V4's error grows with neighbouring molecules

V4 energy, water chain, B3LYP:

| System | max\|ΔE\| |
|---|---:|
| monomer | 1.2e-9 |
| dimer (3 Å) | 5.6e-7 |
| tetramer (3 Å) | 1.2e-6 |
| tetramer (6 Å) | 8.6e-8 |

V4 sets the angular order by u = r/r_Bragg, with small outer bands (second
row 170, H 434). Its selection and blind fixtures were single small
molecules, whose outer shells hold only a decaying tail; in a cluster those
shells pass through the neighbours. Unpruned grids at 135 shells (energy,
water4 at 3 Å):

| Angular | water1 | water4 | water4 / (4 × water1) |
|---:|---:|---:|---:|
| 590 | 3.55e-8 | 6.53e-8 | 0.46 |
| 974 | 8.44e-9 | 3.69e-8 | 1.09 |

So the cluster error is V4's pruning, not the order of its valence band.
PySCF on the same grids (`tests/pyscf/becke_floor_orientation.py`) matches
Planck to three digits. `pyscf-nwchem@fine`, with larger outer bands, is
4.6× better than V4 on water4.

### Cost

Gradient runs, 8 threads, B3LYP, with S7's AO screening:

| | points | wall | peak |
|---|---:|---:|---:|
| naphthalene 6-31G\* `legacy@normal` | 166,004 | 24.0 s | 1.03 GB |
| naphthalene 6-31G\* `pyscf-nwchem@normal` | 219,644 | 27.3 s | 1.35 GB |
| naphthalene 6-31G\* `pyscf-nwchem@fine` | 618,100 | 35.3 s | 3.05 GB |
| naphthalene 6-31G `pyscf-nwchem@fine` | 618,100 | 11.5 s | 2.11 GB |
| naphthalene 6-31G `pyscf-nwchem@ultrafine` | 1,266,276 | 22.9 s | 4.28 GB |
| naphthalene 6-31G V4 | 1,444,380 | 24.9 s | 4.92 GB |
| benzene 6-31G\* `pyscf-nwchem@ultrafine` | 819,972 | 8.5 s | 2.80 GB |
| water12 6-31G\* `legacy@fine` | 610,680 | 15.5 s | 1.37 GB |
| water12 6-31G\* `pyscf-nwchem@fine` | 1,080,696 | 28.9 s | 2.67 GB |

(The naphthalene `@normal` rows predate S7; S7 changes them little, since
naphthalene keeps 83 % of its AO entries.) At these sizes most of the run
is not grid work, so `pyscf-nwchem@fine` costs 1.5–1.9× legacy's wall time
for 3.7× the points.

### Where the time and memory go

Naphthalene B3LYP/6-31G\* gradient, `pyscf-nwchem@fine`, 8 threads, under
`sample`, before S7 (main-thread shares):

| Phase | share |
|---|---:|
| XC matrix (`assemble_xc_matrix`) | 21.4 % |
| density on the grid | 20.4 % |
| J build (fused, direct) | 20.4 % |
| K build (fused, direct) | 20.4 % |
| ERI gradient | 9.2 % |
| XC nuclear gradient | 4.3 % |
| grid build + AO evaluation | 2.6 % |

The four stored AO arrays (npts × nb × 32 B) were 91 % of the peak. Two
changes made in this scope address the grid half:
- **S2b** (`b601a9e7`): `assemble_xc_matrix` is one GEMM per 128-point
  block and spin in fixed bins, so the KS SCF is thread-invariant.
- **S7** (`1920b058`): AO values and gradients are stored per 128-point
  block, keeping only the AOs above 1e-12 somewhere in the block
  (`ScreenedAOGrid`, `src/dft/ao_grid.h`). Density, XC matrix and XC
  gradient are each one template over `AOBlockView`; SOSCF, stability,
  TDDFT and the DH paths get dense arrays built from the blocks on first use
  (`PreparedSystem::dense_ao()`). Water12 `legacy@fine` gradient 48.5 s /
  4.8 GB → 15.5 s / 1.37 GB, identical printed results to a no-screening
  build. Checks: `planck-dft-xc-gradient-threads` (separated dimer,
  screened vs dense ≤ 5e-16, bitwise at 1–8 threads, three mutations
  caught); all 114 DFT regression cases.

Fraction of AO entries kept at 1e-12 (PySCF probe on the same grids):

| | as ordered | points sorted spatially |
|---|---:|---:|
| naphthalene, 618k points | 0.86 | 0.68 |
| water12, 1.08M points | 0.27 | 0.21 |

J/K (41 % on naphthalene) is the other half of the cost and is not a grid
question; it is tracked in the vault's performance list.

## Open work, in order

**S3e, `pyscf-nwchem@ultrafine` on anthracene.** Its naphthalene margin
is 1.8× and its noise grew 1.7× from benzene; one more ring tells whether
it holds. Anthracene 6-31G (nb ~ 140): ~1.7 M points, ~4.5 GB with the
naphthalene keep fraction. If it fails, the tight preset is valid only up
to naphthalene-sized aromatics, and S10 decides whether that is enough.

**S4, frequency noise.** H2O2 B3LYP/6-31G\* frequencies at four
orientations, compared directly (no rotating back). Pass mark ≤ 1 cm⁻¹.
- **An optimized fixture first.** At the unoptimized audit geometry, legacy
  `normal` frequencies move by up to 37.7 cm⁻¹ with orientation: residual
  forces leak into the projected Hessian. Optimize H2O2 with V4 and
  `geomopt_grad_tol 1.5e-5`, add it to `CASES` as `h2o2_b3lyp_opt`, and use
  that one geometry for every setting.
- **Control:** V4 must be ≤ ~0.1 cm⁻¹; otherwise the remaining noise is the
  fixture or the FD step.
- **Then** `pyscf-nwchem@fine` and `legacy@normal`. Report the torsion
  (~350 cm⁻¹) separately: its curvature is small, so gradient noise is the
  largest fraction of it.

**S5, cost at size.** `pyscf-nwchem@fine` against `legacy@normal` on water4
and water12 (6-31G\*) and water4 cc-pVTZ: SCF and gradient wall time and
peak. A run that does not fit in 8 GiB is recorded as "does not run", not
left to swap.

**S6, choose.**
- Default: `pyscf-nwchem-2.13@fine` if it passes S4 and costs ≤ ~2× legacy
  `normal` at water12. Otherwise record the trade-off and pick explicitly;
  no cheaper setting passes.
- Tight preset: a user-facing name for `pyscf-nwchem-2.13@ultrafine`
  (if S3e holds), replacing V4 in that role.
- Rewrite this scope as the answer; update `DFT_GRID_CONVERGENCE.md`.

**S8, migration.**
- One line in `types.h`; keep `legacy` as a named policy.
- Run the DFT regression cases and classify every energy that moves: only
  "the grid changed" is acceptable.
- Re-pin each moved reference against PySCF on a matching grid where one
  exists (a `pyscf-nwchem` default makes this direct); otherwise use the new
  Planck value, with the old one in the commit message.
- Update `DFT_GRID_CONVERGENCE.md` and the vault; run `--suite all` (one tag
  is not enough: the Bohr-constant lesson).

**S9, DH auto-select.** `validate_qualified_uks_dh_grid` selects the tight
preset instead of refusing an unqualified grid, and says so. It needs only
the preset name, so it can follow S6 directly. The preset is no longer V4,
so the UKS DH gradient must first be re-qualified on
`pyscf-nwchem@ultrafine` (V4's qualification does not transfer).

**S10, V5.** Only if S3e fails and the tight tier must hold for larger
aromatics. Its own scope under the `DFT_GRID_CONVERGENCE.md` protocol, with
aromatic and cluster fixtures in the selection atlas and the blind suite.

**S7b, AO screening stage 2.** Only if a needed size does not fit, or a
SOSCF, stability, TDDFT or DH run at size needs the memory:
- port the dense consumers to blocks and delete `AOGridEvaluation` and
  `dense_ao()` (until then those runs allocate the dense arrays on top);
- sort grid points spatially before blocking (table above); every per-point
  array follows the order, including the owner/weight bookkeeping and the
  MPI slices;
- prescreen shells by extent instead of evaluating every AO to decide what
  to keep (AO evaluation is ~1 % of a run).

## Ruled out

- **V4 as the default.** The original rule ("default to V4 if legacy is
  above ~1e-6 and V4 costs under ~1.5×") cannot be met: V4 has 9–11× the
  points. The two tiers replace it.
- **Any `normal`-level setting as the default.** `pyscf-nwchem@normal` and
  `legacy-unpruned-max@normal` pass benzene by ≤ 1 % and fail naphthalene.
- **`legacy@fine` and `legacy@ultrafine` as the default.** `fine` fails
  water4; `ultrafine` passes but has 2.3× the points of
  `legacy-unpruned-max@normal` for no less noise.
- **`pyscf-nwchem@fine` and V4 as the tight preset.** Both ~3e-6 on
  naphthalene. V4 costs 2.3× the points of `@fine` for 8 % less gradient
  noise, and more points than `@ultrafine` for 3.7× more.
- **More radial shells.** 75 → 135, and 200 at 590, change nothing.
  Angular pruning is the whole effect.
- **A Becke-partition error floor on clusters.** An earlier reading of the
  water4 590 → 974 decline as a floor compared cluster grids without the
  monomer. All four waters share one orientation, so their quadrature
  errors add; water4 ≈ 4 × water1 is the expectation, and 974 meets it.
- **Streaming the AO arrays (X4).** Planned as the memory fix; AO screening
  (S7) removed the memory wall without re-evaluating AOs every iteration.

Why the problem went unseen: `legacy` had been measured only as a comparison
column on the UKS DH fixtures, at elevated presets. Of the 31 DFT derivative
regression inputs, 20 override `grid_angular_points` (bypassing pruning),
2 use V4, and none rotates its input.
