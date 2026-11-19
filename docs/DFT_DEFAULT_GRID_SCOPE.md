# Should `planck-dft` change its default grid policy, and to what?

Scope, opened 2026-11-10, rewritten 2026-11-18 and 2026-11-19 (after S3).
A decision: what needs writing is a one-line default change, a tight preset
name, and re-pinned references. The scope expires when the decision is
made; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 4). The pruning qualification itself
is answered in `docs/DFT_GRID_CONVERGENCE.md`, and the gradient cost work in
`docs/DFT_XC_GRADIENT_PERFORMANCE.md`.

## The question

The default `grid_policy` is `legacy` (`src/base/types.h`, `_grid_policy`)
at `grid normal`. Its gradients depend on molecular orientation by
4–13e-4 Ha/Bohr. That is above the default geometry-optimization force
threshold (3e-4), so default-settings geometries and frequencies carry grid
noise the user cannot see. The qualified V4 policy (`g4-3-v4`) removes it,
but costs ~9× the points and loses accuracy once there are neighbouring
molecules.

## The requirement: two tiers

The noise only has to be small against what the optimizer resolves:

| Tier | Gradient orientation noise | For |
|---|---:|---|
| default (`grid normal`) | ≤ 3e-5 Ha/Bohr (1/10 of the default `geomopt_grad_tol`, 3e-4) | every geometry and frequency run |
| tight (opt-in preset; DH gate) | ≤ 1.5e-6 Ha/Bohr (1/10 of `gau_tight`, 1.5e-5) | DH gradients, tight optimizations, PySCF-matched gates |

Plus frequency noise ≤ 1 cm⁻¹ at the default tier. The default needs
headroom, not a bare pass: the fixtures are small, and a setting at the
limit on benzene will fail on larger systems.

## Where it stands (2026-11-19)

| Step | Needs | Produces | |
|---|---|---|---|
| S1 audit driver | — | the measuring tool | done |
| S2 profile | — | where time and memory go | done |
| S2b XC matrix build | S2 | blocked, thread-invariant `assemble_xc_matrix` | done |
| S3 candidate matrix | S1 | which policies meet the default tier | done |
| S3b naphthalene | S3 | whether the marginal passes hold on a larger aromatic | done: they do not |
| S3c V4 on naphthalene | S3b | whether V4 holds the tight tier | done: it does not |
| S3d `pyscf-nwchem@ultrafine` | S3c | whether an existing level meets the tight tier | next |
| S4 frequency noise | S3 | frequency noise of the survivors | next |
| S5 cost at size | S3, S2b | cost and memory of the survivors | next |
| S6 choose | S3–S5 | default policy, tight preset | |
| S7 AO screening, stage 1 | S2b | screened AO blocks for the SCF and gradient | done |
| S7b AO screening, stage 2 | S7 | the remaining consumers on blocks; dense arrays deleted | |
| S8 migration | S6 | the new default, re-pinned suite | |
| S9 DH auto-select | S6 | DH gradients pick the tight preset | |
| S10 V5 | S3d | only if no existing setting meets the tight tier | |

This plan also holds the open items of `DFT_XC_GRADIENT_PERFORMANCE.md`
(S2, S7).

## S3: the candidate matrix

Driver: `tests/dft_grid_orientation_audit.py`. It runs the input
orientation plus three rigid rotations, rotates the gradients back, and
reports the max over components and rotations, with grid points, wall time
and peak footprint. Build `b601a9e7`, 8 threads, 6-31G\*, Cartesian; V4 at
its qualified 135 radial shells. Raw rows and logs:
`tests/benchmarks/dft/grid_orientation/s3_pass{1,2,3}.{json,log}`.

The settings, as the driver spells them:
- `pyscf-nwchem-2.13@normal` is PySCF's default grid: `normal` maps to
  PySCF level-3 radial counts, and the policy is PySCF 2.13
  `_default_ang` + `nwchem_prune` without the light-atom reduction
  (`grid.h`, `resolve_grid_spec`). `@fine` is level 5.
- `legacy-unpruned-max` puts the scheme's largest region on every shell:
  302 for second-row atoms at `normal`, 434 at `fine`.

Pass 1 ran all settings on H2O2 PBE and water4, pass 2 the survivors on
the other five fixtures. `pyscf-nwchem-2.13@fine` was added afterwards
(pass 3), because `@normal` sat at the limit. The control cells reproduce
earlier measurements (legacy 5.46e-4 / 1.32e-3, V4 2.03e-7 / 1.04e-6).

max|Δg| in Ha/Bohr; ✗ fails the default tier, **bold** meets the tight
tier:

| Setting | H2O2 PBE | H2O2 B3LYP | H2O2 HSE06 | H2O2⁺ UKS PBE | benzene B3LYP | water dimer | water4 (3 Å) | points / legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy@normal` | 5.5e-4 ✗ | 4.5e-4 ✗ | 5.0e-4 ✗ | 4.4e-4 ✗ | 4.9e-4 ✗ | 1.2e-3 ✗ | 1.3e-3 ✗ | 1 |
| `legacy@fine` | 2.7e-4 ✗ | | | | | | 8.3e-5 ✗ | 2.2–2.3 |
| `legacy@ultrafine` | 1.4e-5 | 1.4e-5 | 1.2e-5 | 1.1e-5 | 2.8e-5 | 1.5e-5 | 1.0e-5 | 4.2–4.3 |
| `legacy-unpruned-max@normal` | 1.1e-5 | 1.2e-5 | 1.3e-5 | 8.5e-6 | 3.0e-5 | 1.5e-5 | 1.0e-5 | 1.8–1.9 |
| `legacy-unpruned-max@fine` | 2.5e-6 | 2.4e-6 | 2.4e-6 | 3.4e-6 | 8.7e-6 | 2.6e-6 | **1.4e-6** | 3.8–3.9 |
| `pyscf-nwchem-2.13@normal` | 1.1e-5 | 1.2e-5 | 9.4e-6 | 1.0e-5 | 3.0e-5 | 5.5e-6 | 9.1e-6 | 1.4–1.5 |
| `pyscf-nwchem-2.13@fine` | **5.0e-7** | **3.4e-7** | **9.4e-7** | **5.2e-7** | 1.5e-6 | **4.8e-7** | **2.3e-7** | 3.8–4.0 |
| `g4-3-v4` | **2.0e-7** | **1.3e-7** | **2.3e-7** | **1.8e-7** | **8.4e-7** | **8.2e-8** | **1.0e-6** | 9.2–10.7 |

The benzene values that sit at a limit, in full: `legacy@ultrafine`
2.816e-5, `legacy-unpruned-max@normal` 2.961e-5,
`pyscf-nwchem-2.13@normal` 2.985e-5, `pyscf-nwchem-2.13@fine` 1.528e-6.
Energies follow the same order (`pyscf-nwchem@fine` max|ΔE| ≤ 2.7e-7 Eh on
all seven; V4 1.2e-6 on water4).

What this says:

- **Legacy fails because of its angular pruning.** It is 15–40× over the
  default tier on every functional. Removing the pruning (`unpruned-max`)
  or raising the level fixes it; more radial shells do not (below).
- **Benzene decided, not the cluster.** Every survivor has margin on water4
  (≤ 1.0e-5). On benzene the three cheap survivors pass within 1–6 % of
  3e-5. They are marginal, not clean passes.
- **`legacy@ultrafine` is dominated.** It has 2.3× the points of
  `legacy-unpruned-max@normal` and no less noise.
- **`pyscf-nwchem-2.13@fine` is the finding.** At the same points as
  `legacy-unpruned-max@fine` it has 3–17× less noise. Against V4 it has
  2.4–2.7× fewer points, is better on both water clusters, and misses the
  tight tier only on benzene, by 2 %. It is a candidate for either tier: a
  default with ~20× margin, or the tight preset in place of V4.
- **Wall time at these sizes says little.** Benzene takes 4.0 s (legacy),
  6.1 s (`pyscf-nwchem@fine`), 11.9 s (V4): the non-grid work dominates.
  Cost is S5's question.

## S3b: naphthalene

Same driver and build; naphthalene B3LYP/6-31G\* (D2h, the geometry of
`tests/eri_class_share_benchmark.cpp`, now `naphthalene_b3lyp` in `CASES`).
Raw rows: `tests/benchmarks/dft/grid_orientation/s3b_naphthalene.{json,log}`.

| Setting | max\|Δg\| | max\|ΔE\| | points | / legacy | wall (8 thr) | peak |
|---|---:|---:|---:|---:|---:|---:|
| `legacy@normal` | 6.6e-4 ✗ | 9.6e-5 | 166,004 | 1 | 24.0 s | 1.03 GB |
| `pyscf-nwchem-2.13@normal` | 3.7e-5 ✗ | 1.4e-5 | 219,644 | 1.32 | 27.3 s | 1.35 GB |
| `legacy-unpruned-max@normal` | 3.2e-5 ✗ | 1.5e-5 | 304,100 | 1.83 | 28.2 s | 1.81 GB |
| `pyscf-nwchem-2.13@fine` | 3.1e-6 | 1.2e-7 | 618,100 | 3.72 | 38.4 s | 3.62 GB |

- **The two marginal settings fail**, as S3 predicted: 3.7e-5 and 3.2e-5
  against 3e-5. Nothing at `normal` level is a default.
- **`pyscf-nwchem-2.13@fine` passes with ~10× margin**, at 1.6× legacy's
  wall time here (most of the run is not grid work).
- **It does not meet the tight tier on naphthalene** (3.1e-6, 2× over),
  and its noise grew faster than `@normal`'s from benzene (1.53e-6 → 3.13e-6
  against 2.99e-5 → 3.74e-5). V4 does not either (S3c).

Into S4–S5: `pyscf-nwchem-2.13@fine` as the default candidate,
`legacy-unpruned-max@fine` as its fallback (benzene 8.7e-6, naphthalene not
run), V4 as the tight control.

## S3c: V4 on naphthalene

V4 in 6-31G\* needs ~6.5 GB of AO blocks after S7 and thrashed in swap on
this 8 GiB machine (footprint 6.49 GB, 160 MB resident), so it was stopped.
S3c runs in 6-31G instead (nb 106, not 166), with the S3b rows repeated
there. Raw rows:
`tests/benchmarks/dft/grid_orientation/s3c_naphthalene_631g.{json,log}`.

| naphthalene B3LYP/6-31G | max\|Δg\| | max\|ΔE\| | points | wall (8 thr) | peak |
|---|---:|---:|---:|---:|---:|
| `legacy@normal` | 6.7e-4 ✗ | 9.1e-5 | 166,004 | 7.7 s | 0.58 GB |
| `pyscf-nwchem-2.13@fine` | 3.3e-6 | 1.2e-7 | 618,100 | 11.5 s | 2.11 GB |
| `g4-3-v4` | 3.0e-6 | 5.2e-7 | 1,444,380 | 24.9 s | 4.92 GB |

- **The basis barely matters here.** Against the 6-31G\* rows of S3b,
  legacy moves 6.56e-4 → 6.75e-4 and `pyscf-nwchem@fine` 3.13e-6 →
  3.28e-6: within 5 %. The noise is the grid's.
- **V4 fails the tight tier on naphthalene** (3.0e-6, 2× over), like
  `pyscf-nwchem@fine`. With 2.3× the points it is only 8 % quieter in the
  gradient and 4× noisier in the energy.
- **So no measured setting meets the tight tier beyond benzene.** Both
  candidates double from benzene to naphthalene, so the tier is not met on
  extended aromatics by adding points to these schedules.

Before designing a V5 (S10), one existing setting is unmeasured: S3d.

## Background measurements

### Why earlier evaluation missed it

`legacy` was only measured as a comparison column on the UKS DH fixtures,
at elevated presets (2.2e-5 there, close to `legacy ultrafine` here). The
shipped `grid normal` was never measured. Of the 31 DFT derivative
regression inputs, 20 override `grid_angular_points` (bypassing pruning),
2 use V4, and none rotates its input.

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

Going from 75 to 135 radial shells changes nothing.

### V4's error grows with neighbouring molecules

V4 energy, water chain, B3LYP:

| System | max\|ΔE\| |
|---|---:|
| monomer | 1.2e-9 |
| dimer (3 Å) | 5.6e-7 |
| tetramer (3 Å) | 1.2e-6 |
| tetramer (6 Å) | 8.6e-8 |

The first neighbour causes the jump (480×). V4 sets the angular order by
u = r/r_Bragg, with small outer bands (second row 170, H 434). Every
selection and blind fixture was a single small molecule, whose outer shells
hold only a decaying tail; in a cluster those shells pass through the
neighbours. Unpruned grids at 135 shells (energy, water4 at 3 Å):

| Angular | water1 | water4 | water4 / (4 × water1) |
|---:|---:|---:|---:|
| 590 | 3.55e-8 | 6.53e-8 | 0.46 |
| 974 | 8.44e-9 | 3.69e-8 | 1.09 |

So the cluster error comes from V4's pruning, not the angular order of its
valence band. PySCF on the same grids (`tests/pyscf/becke_floor_orientation.py`)
matches Planck to three digits. S3 agrees: `pyscf-nwchem@fine`, whose
outer bands are larger, is 4.6× better than V4 on water4.

### Where the time and memory go (S2, S2b)

Profiled at 8 threads under `sample` (benzene V4 and water12 legacy `fine`
gradients). Before S2b, the SCF's XC matrix assembly was 42–55 % of the
main thread, with 16–21 % barrier wait: a rank-1 update of nb × nb at every
point, with partials keyed by thread, so the KS SCF was not thread-invariant.

S2b (`b601a9e7`) made `assemble_xc_matrix` (`src/dft/ks_matrix.cpp`) one
GEMM per 128-point block and spin (M += ΦᵀA, V = M + Mᵀ), with contiguous
blocks summed in fixed bins (≤ 64, set by nb, spin count and block count,
never by thread count) under `schedule(dynamic, 1)`; the density
evaluation's slabs went dynamic too. Checks: ≤ 3.5e-14 against a per-point
oracle and bitwise at 1/2/3/4/8 threads (`planck-dft-xc-gradient-threads`),
two mutations caught, all 114 DFT regression cases pass, and the KS SCF is
now identical at 1 and 8 threads.

| 8 threads | before S2b | after S2b | peak |
|---|---:|---:|---:|
| benzene V4 gradient | 16.4 s | 11.1 s | 3.6 GB |
| water12 legacy `fine` gradient | 72.2 s | 48.5 s | 4.8 GB |

After S2b, water12: XC matrix 30.7 %, density on the grid 29.3 %, J/K
~10 %, grid build (serial, built twice) ~5 % each. Both XC phases are dense
GEMMs over every AO at every point. Memory is the four stored AO arrays
(npts × nb × 32 B), 87–93 % of the peak (3.14 GB of 3.6 for benzene V4,
4.46 of 4.8 for water12). V4 at water12 is estimated at ~22 GB. The next
lever for both time and memory is AO screening per block of points (S7,
below).

## S7: AO screening per block (stage 1, 2026-11-19)

Done ahead of S3c, which would otherwise not fit in memory.

**Profile first.** Naphthalene B3LYP/6-31G\* gradient,
`pyscf-nwchem-2.13@fine`, 8 threads, under `sample` (53.8 s, 3.61 GB):

| Phase (main thread) | share |
|---|---:|
| XC matrix (`assemble_xc_matrix`) | 21.4 % |
| density on the grid | 20.4 % |
| J build (fused, direct) | 20.4 % |
| K build (fused, direct) | 20.4 % |
| ERI gradient | 9.2 % |
| XC nuclear gradient | 4.3 % |
| grid build + AO evaluation | 2.6 % |

The four stored AO arrays (618,100 × 166 × 32 B = 3.28 GB) were 91 % of
the peak. The grid work (42 %) and the memory both scale with the AO
arrays; J/K (41 %) is the other half and is not a grid question.

**How much screening can drop.** A PySCF probe on the same grids
(level 5, Cartesian 6-31G\*): the fraction of AO columns significant
anywhere in a 128-point block.

| | @1e-10 | @1e-12 | points sorted spatially, @1e-12 |
|---|---:|---:|---:|
| naphthalene, 618k points | 0.82 | 0.86 | 0.68 |
| water12, 1.08M points | 0.26 | 0.27 | 0.21 |

Compact molecules gain little; extended ones gain ~4× in memory and up to
~14× in XC flops (the GEMMs scale with kept²).

**What landed.**
- `ScreenedAOGrid` (`src/dft/ao_grid.h`): fixed 128-point blocks
  (`kAOBlock`), each keeping the AOs whose value or gradient exceeds
  `kAOScreenTolerance` = 1e-12 anywhere in the block. One contiguous
  buffer: pass 1 screens, pass 2 evaluates only the kept columns into it.
  Per-block allocations were tried first and fragmented the heap by ~15 %
  of the stored size.
- `AOBlockView`: one block's φ and ∇φ plus its kept-column list (`nullptr`
  = all). Both `ScreenedAOGrid` and the dense `AOGridEvaluation` provide it,
  so density (`xc_grid.cpp`), XC matrix (`ks_matrix.cpp`) and XC nuclear
  gradient (`dft_gradient.cpp`) are each one template over block views. A
  screened block gathers P(s, s), evaluates the GGA AO Hessian for its kept
  columns only, and scatters its k × k (matrix) or k-vector (basis-motion
  gradient) result back. The fixed-bin, thread-invariant summation of S2b is
  unchanged.
- `PreparedSystem` holds `ao_blocks`. The consumers not yet ported (SOSCF
  and stability Hessian-vector products, TDDFT, the DH and UDH paths) call
  `dense_ao()`, which builds dense arrays from the blocks on first use, so
  they see the same AOs as the SCF and allocate them only when they run.
- The grid summary prints the kept fraction ("DFT AO screening :").

**What dropping costs.** A dropped AO is removed entirely, including from
the GGA gradient's Hessian term, where (ΦP)_μ is not zero. At the 1e-12
tolerance this is invisible: water12 `legacy@fine` with screening and with
a no-screening build gives the same energy to all 17 printed digits and
the same gradient to every printed digit (1e-12). At a loose 1e-4 it is
6.6e-9 relative on the test dimer, which is why the same-AO check below
excludes the GGA gradient.

**Checks** (`planck-dft-xc-gradient-threads`, extended):
- A C1 water-cation dimer with its monomers 6 Å apart (49 % kept at 1e-4),
  RKS/UKS × LDA/GGA. Screened blocks against dense arrays holding the same
  AOs: XC matrix ≤ 1.5e-16, E_xc ≤ 3.6e-15, LDA gradient ≤ 5.3e-16
  (relative). Default screening against unscreened dense: gradient
  ≤ 1.1e-15. Bitwise at 1/2/3/4/8 threads.
- Three mutations, each fails the test: the XC-matrix scatter (error 1.0),
  the gradient scatter (0.03), the density gather of P(s, s) (0.03–0.10).
- The orientation audit reproduces S3 exactly (water4 and H2O2 rows).
- The 114 DFT regression cases pass (below).

**Result** (8 threads, unprofiled, gradient runs):

| | before S7 | after S7 |
|---|---:|---:|
| naphthalene `pyscf-nwchem@fine` | 38.4 s, 3.61 GB | 35.3 s, 3.05 GB |
| water4 `pyscf-nwchem@fine` | 2.76 s, 1.06 GB | 2.37 s, 0.77 GB |
| water12 `legacy@fine` | 48.5 s, 4.8 GB | 15.5 s, 1.37 GB |
| water12 `pyscf-nwchem@fine` | does not fit (~8 GB of arrays) | 28.9 s, 2.67 GB |

The no-screening build of the same code takes 42.1 s and 4.77 GB on water12
`legacy@fine`, so the gain is the screening, not the restructuring.

**Regression run.** All 114 DFT cases pass (`--suite all`, each case
whose executable is `planck-dft` or whose input has `%begin_dft`):
energies, gradients, geomopt, frequencies, TDDFT, stability, SOSCF, PCM,
D3, counterpoise, MPI smoke and the DH/UDH gradients, which exercise the
`dense_ao()` path.

**Left for S7b**, only if a target needs it:
- Port the dense consumers to blocks and delete `AOGridEvaluation` and
  `dense_ao()`. Until then a SOSCF, stability, TDDFT or DH run pays the full
  dense arrays on top of the blocks.
- Sort grid points spatially before blocking (naphthalene 0.86 → 0.68,
  water12 0.27 → 0.21 kept). Every per-point array follows the order, so it
  touches the grid owner/weight bookkeeping and the MPI slices.
- Prescreen shells by extent instead of evaluating every AO in pass 1
  (AO evaluation is ~1 % of a run today).

## Next steps

**S3d, `pyscf-nwchem-2.13@ultrafine`.** PySCF level 7 with the same
pruning: naphthalene 6-31G and benzene 6-31G\*, with `@fine` as the
control. If it is ≤ 1.5e-6 on both, it is the tight preset and V4 is
retired from that role. If it is not, the open question becomes whether
the tight tier (1/10 of `gau_tight`) is the right requirement for
aromatics, or whether S10 is needed; decide that before designing V5.

**S4, frequency noise.** H2O2 B3LYP/6-31G\* frequencies at four
orientations; frequencies are compared directly, without rotating back.
Pass mark ≤ 1 cm⁻¹.
- **S4a, an optimized fixture.** S1 found 37.7 cm⁻¹ of orientation noise
  for legacy `normal` at the unoptimized geometry, where residual forces
  leak into the projected Hessian. Optimize H2O2 B3LYP/6-31G\* with V4 and
  `geomopt_grad_tol 1.5e-5`, and add it to `CASES` as `h2o2_b3lyp_opt`.
  Use the one geometry for every setting.
- **Control first.** V4 must be ≤ ~0.1 cm⁻¹; otherwise the remaining noise
  is the fixture or the FD step, and 1 cm⁻¹ cannot be judged.
- **Then** the S3 survivors and `legacy@normal` as the baseline. Report the
  torsion (~350 cm⁻¹) separately: its curvature is small, so gradient noise
  is the largest fraction of it.

**S5, cost at size.** The survivors on water4 and water12 (6-31G\*) and
water4 cc-pVTZ: SCF and gradient wall time and peak footprint, against
`legacy@normal`. A run that would not fit in 8 GiB is recorded as "does
not run", not left to swap. Since S7, `pyscf-nwchem@fine` on water12
runs (1.08 M points, 27.5 % kept, 28.9 s, 2.67 GB); that row is the
starting point.

**S6, choose.**
- Default: `pyscf-nwchem-2.13@fine` if it passes S4 and costs ≤ ~2× legacy
  `normal` at water12 (naphthalene: 3.7× points, 1.6× wall). Otherwise
  record the trade-off and pick explicitly; no cheaper setting passes
  (S3b).
- Tight preset: a user-facing name for whichever setting S3d finds; neither
  V4 nor `pyscf-nwchem@fine` meets the tier on naphthalene (S3b, S3c).
- Write the outcome here (this scope becomes the answer) and in
  `DFT_GRID_CONVERGENCE.md` ("Not done").

**S7b, AO screening stage 2.** Only if S5 or S3c shows a needed size that
does not fit, or a SOSCF/stability/TDDFT/DH run at size needs the memory.
Items in the S7 section above. Full streaming (X4: re-evaluating AOs every
SCF iteration) is no longer planned: screening removed the memory wall it
was for.

**S8, migration.**
- One line in `types.h`; keep `legacy` as a named policy.
- Run the 77 DFT regression cases. Classify every energy that moves: only
  "the grid changed" is acceptable.
- Re-pin each moved reference against PySCF on a matching grid where one
  exists (a `pyscf-nwchem` default makes this direct). Otherwise use the
  new Planck value, with the old one in the commit message.
- Update `DFT_GRID_CONVERGENCE.md` and the vault.
- Run `--suite all`; one tag is not enough (the Bohr-constant lesson).

**S9, DH auto-select.** `validate_qualified_uks_dh_grid` selects the tight
preset instead of refusing an unqualified grid, and says so in the output.
It needs S6 only for the preset name, so it can follow S6 directly. If the
tight preset becomes `pyscf-nwchem@fine`, the UKS DH gradient must first be
re-qualified on it (the V4 qualification does not transfer).

**S10, V5.** Only if S3d finds no existing setting that meets the tight
tier and the tier itself stands. V4 does not meet it on naphthalene (S3c),
and `pyscf-nwchem@fine` already beats V4 on clusters. It would be its own scope under the
`DFT_GRID_CONVERGENCE.md` protocol, with cluster fixtures in the selection
atlas and blind suite.

## Ruled out or superseded

- **The original decision rule** ("make V4 the default if legacy is above
  ~1e-6 Ha/Bohr and V4 costs under ~1.5×") cannot be met. V4 has 9–11× the
  points. The two tiers replace it.
- **Any `normal`-level setting as the default.** `pyscf-nwchem-2.13@normal`
  and `legacy-unpruned-max@normal` passed benzene within 1 % and failed
  naphthalene (3.7e-5, 3.2e-5; S3b).
- **`pyscf-nwchem-2.13@fine` as the tight preset.** 1.5e-6 on benzene,
  3.1e-6 on naphthalene (S3b).
- **V4 as the tight preset.** 3.0e-6 on naphthalene (S3c), with 2.3× the
  points of `pyscf-nwchem@fine` for 8 % less gradient noise.
- **`legacy@fine` and `legacy@ultrafine` as the default.** `fine` fails
  (water4 8.3e-5); `ultrafine` passes but is dominated by
  `legacy-unpruned-max@normal` (S3).
- **The cluster as the binding fixture.** Expected from legacy's 2.4×
  water4/H2O2 ratio; S3 found benzene binding instead.
- **A Becke-partition error floor on clusters.** An earlier draft read the
  water4 590 → 974 decline as a floor. It compared cluster grids without
  the monomer at the same grid. All four waters share one orientation, so
  their quadrature errors add: water4 ≈ 4 × water1 is the expectation with
  no cluster effect, and 974 meets it. The partition needs no change.
- **More radial shells.** 75 → 135, and 200 at 590, change nothing.
  Angular pruning is the whole effect.
