# Should `planck-dft` change its default grid policy, and to what?

Scope, opened 2026-11-10, rewritten 2026-11-18. A decision: what needs
writing is a measuring tool and, at the end, a one-line default change with
re-pinned references. The scope expires when the decision is made; rewrite
it then as the answer. Status lives in `vault/Status/Open Work.md`
(priority 4). The pruning qualification itself is answered in
`docs/DFT_GRID_CONVERGENCE.md`, and the gradient cost work in
`docs/DFT_XC_GRADIENT_PERFORMANCE.md`.

## The question

The default `grid_policy` is `legacy` (`src/base/types.h`, `_grid_policy`)
at `grid normal`. Its gradients depend on molecular orientation by
4–13e-4 Ha/Bohr for ordinary functionals. That is above the default
geometry-optimization force threshold (3e-4), so default-settings geometries
and frequencies carry grid noise the user cannot see. The qualified V4
policy (`g4-3-v4`) removes it on isolated molecules, but costs ~9× the
points and loses accuracy once there are neighbouring molecules.

## The requirement: two tiers

The noise only has to be small against what the optimizer resolves:

| Tier | Gradient orientation noise | For |
|---|---:|---|
| default (`grid normal`) | ≤ 3e-5 Ha/Bohr (1/10 of the default `geomopt_grad_tol`, 3e-4) | every geometry and frequency run |
| tight (opt-in preset; DH gate) | ≤ 1.5e-6 Ha/Bohr (1/10 of `gau_tight`, 1.5e-5) | DH gradients, tight optimizations, PySCF-matched gates |

Plus frequency noise ≤ 1 cm⁻¹ at the default tier. Nothing measured so far
covers both tiers at an acceptable cost, so the outcome is expected to be
two settings: a cheaper default and an explicit tight preset.

## What is measured (2026-11-18)

Driver: `tests/dft_grid_orientation_audit.py`. It runs the input orientation
plus three rigid rotations, rotates the gradients back, and reports the max
over components and rotations. 6-31G\*, Cartesian, `grid normal`; V4 always
at its qualified 135 radial shells.

### Legacy fails the default tier on every functional

| Case | legacy max\|Δg\| | V4 max\|Δg\| | legacy max\|ΔE\| | V4 max\|ΔE\| |
|---|---:|---:|---:|---:|
| C1 H2O2 PBE | 5.5e-4 | 2.0e-7 | 9.6e-5 | 1.1e-8 |
| C1 H2O2 B3LYP | 4.5e-4 | 1.3e-7 | 8.9e-5 | 1.3e-8 |
| C1 H2O2 HSE06 | 5.0e-4 | 2.3e-7 | 7.8e-5 | 1.3e-7 |
| C1 H2O2⁺ UKS PBE | 4.4e-4 | 1.8e-7 | 1.0e-4 | 1.0e-8 |
| benzene B3LYP | 4.9e-4 | 8.4e-7 | 1.3e-4 | 1.8e-7 |
| water4 chain (3 Å) B3LYP | 1.3e-3 | 1.0e-6 | 1.3e-3 | 1.2e-6 |

Gradients in Ha/Bohr, energies in Eh. Legacy is 15–40× over the default
tier. V4 meets the tight tier, but only at its limit on water4.

**Why earlier evaluation missed it.** `legacy` was only measured as a
comparison column on the UKS DH fixtures, at elevated presets (2.2e-5
there, close to `legacy ultrafine` here, 1.4e-5). The shipped `grid normal`
was never measured. Of the 31 DFT derivative regression inputs, 20 override
`grid_angular_points` (bypassing pruning), 2 use V4, and none rotates its
input.

### Legacy's error is its angular pruning

H2O2 PBE, legacy radial counts unless noted:

| Angular grid | max\|Δg\| | max\|ΔE\| | tier |
|---|---:|---:|---|
| pruned `normal` (default) | 5.5e-4 | 9.6e-5 | neither |
| pruned `fine` | 2.7e-4 | 3.8e-5 | neither |
| pruned `ultrafine` | 1.4e-5 | 2.0e-6 | default |
| unpruned 302 | 1.1e-5 | 1.3e-6 | default |
| unpruned 590 | 1.3e-6 | 9.1e-8 | tight |
| unpruned 590, 135 shells | 1.4e-6 | 9.1e-8 | tight |

Going from 75 to 135 radial shells changes nothing.

### V4's error grows with neighbouring molecules

V4 energy, water chain, B3LYP:

| System | max\|ΔE\| |
|---|---:|
| monomer | 1.2e-9 |
| dimer (3 Å) | 5.6e-7 |
| tetramer (3 Å) | 1.2e-6 |
| tetramer (6 Å) | 8.6e-8 |

The first neighbour causes the jump (480×), doubling again adds 2×, and
moving the molecules apart cuts it 14×. V4 sets the angular order by
u = r/r_Bragg, with small outer bands (second row 170, H 434). Every
selection and blind fixture was a single small molecule, whose outer shells
hold only a decaying tail; in a cluster those shells pass through the
neighbours. Unpruned grids at 135 shells (energy, water4 at 3 Å):

| Angular | water1 | water4 | water4 / (4 × water1) |
|---:|---:|---:|---:|
| 590 | 3.55e-8 | 6.53e-8 | 0.46 |
| 974 | 8.44e-9 | 3.69e-8 | 1.09 |
| 1454 | 7.6e-10 | not run (5.7 GB at the time; retry after X2) | |

V4 on the same water4 is 1.2e-6, so the cluster error comes from its
pruning, not the angular order of its valence band. 200 radial shells at
590 give 6.6e-8 (unchanged). PySCF on the same grids (Becke + Treutler
adjustment, `tests/pyscf/becke_floor_orientation.py`) gives 3.551e-8 and
6.535e-8 at 590, matching Planck to three digits.

### Cost so far

Points on H2O2/6-31G\*: legacy normal 35,180, fine 78,148,
ultrafine 147,396; V4 322,080 (9.2× normal). With the XC gradient fixed
(`DFT_XC_GRADIENT_PERFORMANCE.md`), gradient runs at 8 threads take:

| | wall | peak |
|---|---:|---:|
| benzene, V4 | 15.1 s | 3.6 GB |
| benzene, legacy `fine` | 5.8 s | 0.9 GB |
| water12 (3 Å), legacy `fine` | 73.6 s | 4.8 GB |
| water12, V4 | not run (~22 GB estimated) | |

At water12, memory runs out before time does. Peak memory is set by the
SCF's four stored AO arrays (npts × nb × 32 B).

## Candidates already in the code

`src/dft/base/grid.h` has five policies: `legacy`, `legacy-no-light-reduction`,
`legacy-unpruned-max`, `pyscf-nwchem-2.13` and `g4-3-v4`. From the tables
above, pruned `ultrafine` and unpruned 302 already meet the default tier on
H2O2. `pyscf-nwchem-2.13` has not been measured: it shows whether Planck's
default is simply worse than PySCF's or whether both have the problem. A new
policy (V5) is only considered if none of these will do.

## Plan, in order

This plan also holds the open items of `DFT_XC_GRADIENT_PERFORMANCE.md`
(S2, S7). Each step names its prerequisites.

| Step | Needs | Produces |
|---|---|---|
| S1 audit driver | — | the measuring tool for S3–S5 (done) |
| S2 profile | — | where time and memory go now (done) |
| S2b XC matrix build | S2 | blocked, thread-invariant `assemble_xc_matrix` (done) |
| S3 candidate matrix | S1 | which policies meet the default tier |
| S4 frequency noise | S3 | frequency noise of the survivors |
| S5 cost at size | S3, S2b | cost and memory of the survivors |
| S6 choose | S3–S5 | default policy, tight preset |
| S7 X4 go/no-go | S5, S6 | whether to stream the AO arrays |
| S8 migration | S6 | the new default, re-pinned suite |
| S9 DH auto-select | S6 | DH gradients pick the tight preset |
| S10 V5 | S6 | only if tight-tier clusters are needed |

S1, S2 and S2b are done. Next is S3, then S4, S5 and S6.

**S1, extend `tests/dft_grid_orientation_audit.py`.** For every run, also
record:
- grid points (turn on `print_grid_summary` and parse it);
- wall time and peak footprint (`/usr/bin/time -l` on macOS).

Add the H-bonded water dimer (O–O ≈ 2.9 Å) to `CASES`, a `--threads` option
(default 8), and a `--frequency` mode for S4. Check one cell against the
H2O2 PBE row above (legacy 5.5e-4, V4 2.0e-7).

**S2, profile today's cost.** Benzene V4 gradient (15.1 s) and water12
legacy `fine` gradient (73.6 s), 8 threads, with `sample`. Report:
- phase shares: grid build, AO evaluation, SCF XC build, J/K, XC gradient,
  HF-like gradient;
- memory owners, in particular the stored AO arrays against the 4.8 GB
  water12 peak.

S2 does not block S3. It decides whether a cheap performance fix comes
before S5: if a serial phase dominates, S5 would be measuring that phase,
not the grid.

**S1 and S2 results (2026-11-18).**

S1 is done. The driver now records grid points, wall time and peak
footprint per run, and has the dimer, a water12 case, `--settings
policy@grid`, `--threads`, `--frequency` and `--json`. It reproduces the
H2O2 PBE row (legacy 5.46e-4, V4 2.03e-7 Ha/Bohr; 35,180 vs 322,080
points; 0.36 vs 1.21 s per run). A first frequency run gave legacy
`normal` H2O2 frequencies moving by up to 37.7 cm⁻¹ with orientation, at
the unoptimized fixture geometry, so S4 must use an optimized geometry with
V4 as the control.

S2 profiled two gradient runs at 8 threads under `sample`, which makes them
slower than unprofiled (20.9 s and 93.0 s). The table gives main-thread
shares; "wait" is time the main thread spends at the region's end barrier,
waiting for the other threads:

| Phase | benzene V4 | water12 legacy `fine` |
|---|---:|---:|
| XC matrix assembly (`assemble_xc_matrix`), per SCF iteration | 41.7 % (16.3 % wait) | 55.4 % (21.3 % wait) |
| density on the grid (`evaluate_density_on_grid`) | 18.1 % (5.4 % wait) | 21.4 % (6.2 % wait) |
| J/K (`fused_fock_build`) | 13.3 % | 6.5 % |
| grid build (`MakeMolecularGrid`, built twice, serial) | 5.3 % | 6.2 % |
| AO values | 3.0 % | < 2.5 % |
| XC nuclear gradient | 7.0 % | 4.7 % |
| HF-like gradient | 4.2 % | < 2.5 % |

| Memory | points | nb | four stored AO arrays | peak |
|---|---:|---:|---:|---:|
| benzene V4 | 962,640 | 102 | 3.14 GB | 3.6 GB |
| water12 legacy `fine` | 610,680 | 228 | 4.46 GB | 4.8 GB |

What this says:

- **The cost is now the SCF's XC work, not the gradient.**
  `assemble_xc_matrix` (`src/dft/ks_matrix.cpp`) is the pattern the
  gradient had before X1: a rank-1 update of nb × nb matrices at every grid
  point. Its partials are keyed by `omp_get_thread_num()`, so the KS SCF is
  bitwise reproducible only at a fixed thread count. That explains the
  last-digit SCF differences between 1 and 8 threads found during X3.
- **The barrier waits are a hypothesis to check.** A static contiguous
  split over points, of which the zero-weight ones are skipped, would leave
  some threads idle. The density evaluation has the same static split over
  2048-row slabs.
- **Memory is the stored AO arrays**, 87–93 % of the peak. On a 33 Å chain
  most AOs are negligible at most points, so screening AOs per block of
  points (dropping the negligible columns) is a cheaper candidate for the
  memory fix than streaming. It also cuts the XC flops.

The plan gains a step between S2 and S5:

**S2b, XC matrix build: blocked and thread-invariant. Done 2026-11-18.**

- **`assemble_xc_matrix` (`src/dft/ks_matrix.cpp`).** Per 128-point block,
  A = diag(w v_ρ/2) Φ + diag(w) G, where G = Σ_k c_k ∂Φ/∂k (GGA), so
  Σ_p [w v_ρ φφᵀ + w(φgᵀ + gφᵀ)] = ΦᵀA + AᵀΦ. One GEMM M += ΦᵀA per
  block and spin, and V = M + Mᵀ after the reduction. Zero-weight points get
  a zero row.
- **Bins.** Contiguous blocks go into fixed bins under
  `schedule(dynamic, 1)`, summed in bin order. The bin count is at most 64;
  it depends on nb and the spin count (a 256 MB cap on the nb × nb partials,
  at least 8 bins) and on the block count, never on the thread count.
- **Density evaluation.** Its 2048-row slabs now use
  `schedule(dynamic, 1)`. The writes are disjoint, so this changes no bits.

Checks:

- **`planck-dft-xc-gradient-threads`, extended** to the KS matrix,
  RKS/UKS × LDA/GGA:
  - against a per-point oracle (the old algorithm, written out in the
    test): ≤ 3.5e-14 with |V| ≈ 3;
  - bitwise identical at 1/2/3/4/8 threads.
- **Mutations.** Tying the bin count to the thread count fails the identity
  check; dropping the ½ on v_ρ fails the oracle by ~3.
- **Gate inputs.** On the nine H2O2 inputs, energies move ≤ 3e-13 Eh, and
  gradients are unchanged except one 1e-12 last-digit flip.
- **The KS SCF is now thread-invariant.** Energies and gradients are
  identical at 1 and 8 threads to all printed digits (17 for the energy).
  Before S2b they were not.
- **Regression cases.** All 114 DFT cases pass (`--suite all`):
  energies, gradients, geomopt, frequencies, TDDFT, stability, SOSCF, PCM,
  D3 and DH.

| 8 threads | X3b binary | S2b | peak |
|---|---:|---:|---:|
| benzene V4 gradient | 16.4 s | **11.1 s** | 3.6 GB |
| water12 legacy `fine` gradient | 72.2 s | **48.5 s** | 4.8 GB |

Gradients are identical between the two binaries. Re-profiled water12,
main-thread shares:

| Phase | before S2b | after S2b |
|---|---:|---:|
| XC matrix assembly | 55.4 % (21.3 % wait) | 30.7 % (no wait) |
| density on the grid | 21.4 % (6.2 % wait) | 29.3 % (no wait) |
| J/K | 6.5 % | ~10 % |
| grid build (serial) | 6.2 % | ~5 % per build |

Both XC phases are now dense GEMMs over every AO at every point. The next
lever for both time and memory is AO screening per block (S7). S5 can now
measure grids rather than a slow loop.

**S3, candidate matrix.** Settings: `legacy` at normal, fine and ultrafine;
`legacy-unpruned-max` at normal and fine; `pyscf-nwchem-2.13` at its
default level; V4. Fixtures: the six above plus the dimer. Pass mark: the
default tier on every fixture; record the tight tier as a column. About
8 × 7 × 4 ≈ 220 runs, each minutes at most; V4 water4 is the largest.
Output: the survivors, with points per atom and small-molecule cost.

**S4, frequency noise.** H2O2 B3LYP/6-31G\* frequencies at four
orientations, for the S3 survivors and for legacy `normal` as the baseline.
Frequencies need no rotating back, so they are compared directly. Pass mark
≤ 1 cm⁻¹; the low torsion mode is the one to watch.

**S5, cost at size.** Survivors of S3 and S4 on water4 and water12
(6-31G\*) and water4 cc-pVTZ: SCF and gradient wall time and peak footprint,
read against S2's phase shares. A run that would not fit in 8 GiB is
recorded as "does not run", not left to swap.

**S6, choose.**
- Make the cheapest policy that passes S3 and S4 the default if it costs
  ≤ ~2× legacy `normal` at water12. Otherwise record the trade-off and pick
  explicitly.
- Give the tight tier a user-facing preset name that maps to V4 at 135
  shells.
- Write the outcome here (this scope becomes the answer) and in
  `DFT_GRID_CONVERGENCE.md` ("Not done").

**S7, X4 go/no-go.** Go only if S5 shows that the chosen default, or the
tight preset at a size someone needs, does not fit in 8 GiB. X4 then gets
its own scope. It cuts across the code: a per-block AO provider for
`xc_grid.cpp`, `ks_matrix.cpp`, `analytic_hessian.cpp`, the DH paths and the
driver (11 files, ~48 sites), plus AO re-evaluation every SCF iteration.
Its design starts from S2's memory owners and the analysis in
`DFT_XC_GRADIENT_PERFORMANCE.md` ("Not done").

**S8, migration.**
- One line in `types.h`; keep `legacy` as a named policy.
- Run the 77 DFT regression cases. Classify every energy that moves: only
  "the grid changed" is acceptable.
- Re-pin each moved reference against PySCF on a matching grid where one
  exists. Otherwise use the new Planck value, with the old one in the
  commit message.
- Update `DFT_GRID_CONVERGENCE.md` and the vault.
- Run `--suite all`; one tag is not enough (the Bohr-constant lesson).

**S9, DH auto-select.** `validate_qualified_uks_dh_grid` selects the tight
preset instead of refusing an unqualified grid, and says so in the output.
It needs S6 only for the preset name, so it can follow S6 directly, before
S8.

**S10, V5.** Only if S6 finds a user who needs the tight tier on clusters
(V4's water4 is at the limit). It would be its own scope under the
`DFT_GRID_CONVERGENCE.md` protocol: cluster fixtures (water dimer and
tetramer at H-bond and 3–6 Å spacings, benzene) in a new selection atlas
and a new blind suite, keeping the Z-and-u-only contract. Expect larger
outer bands and more points. The qualification method itself stands.

## Ruled out or superseded

- **The original decision rule** ("make V4 the default if legacy is above
  ~1e-6 Ha/Bohr and V4 costs under ~1.5×") cannot be met. V4 has 9.2× the
  points, and its benzene gradient costs 2.6× legacy `fine`. The two tiers
  above replace it.
- **A Becke-partition error floor on clusters.** An earlier draft read the
  water4 590 → 974 decline as a floor. It compared cluster grids without the
  monomer at the same grid. All four waters share one orientation, so their
  quadrature errors add: water4 ≈ 4 × water1 is the expectation with no
  cluster effect, and 974 meets it. The partition needs no change.
- **More radial shells.** 75 → 135, and 200 at 590, change nothing.
  Angular pruning is the whole effect.
