# DFT grid accuracy: implementation scope and acceptance plan

Status: scoped, 2026-09-20. This plan does not change defaults or enable
production UKS DH gradients. The explicit angular override and its numerical
validation already exist; the numbered steps below are proposed. The user
owns rebuilds. No background build is part of this plan.

## 1. Established evidence and remaining uncertainty

This is a shared KS quadrature issue, not a remaining DH equation defect.
Use Planck's own energies/gradients and independent quadrature tests, not
PySCF double-hybrid analytic derivatives, as the acceptance oracles.

The old version of this document described a radial-count formula giving
only 44 oxygen shells at UltraFine. That is historical: `grid.h` now uses
the radial table in `radial.h`. Do not implement the old radial-only fix again.

| Preset | H radial shells | O radial shells | O angular sizes by index region |
| --- | ---: | ---: | --- |
| Coarse | 30 | 40 | 26, 50, 110, 194, 110 |
| Normal | 50 | 75 | 26, 110, 194, 302, 194 |
| Fine | 70 | 105 | 26, 194, 302, 434, 302 |
| UltraFine | 90 | 135 | 50, 302, 434, 590, 434 |

H/He use the next lower angular scheme: UltraFine H has 26, 194, 302, 434,
302 points. Region boundaries are radial-index fractions 0.15, 0.40, 0.70,
0.90, not physical radii. M4 radii are stored **outermost first**. Adding
radial shells does not guarantee sufficient angular resolution.

At fixed UltraFine radial quadrature, unpruned angular refinement closes
the tested orientation failures (total-gradient errors, Ha/Bohr):

| UKS B2PLYP case | Original first rotation | 1454 points, worst of two rotations |
| --- | ---: | ---: |
| C1 H2O2+ / STO-3G | 2.18696770e-5 | 4.62886414e-8 |
| HO2 / STO-3G | 4.29560331e-6 | 1.97937515e-8 |
| Water cation / 6-31G | 2.95964654e-6 | 5.09715928e-8 |

All pass the unchanged 2e-6 covariance bound at 1454 points, with
974-to-1454 refinement stability. The complete refined-grid U11 suite also
passes: six UKS cases, 108 coordinate/step comparisons, separate KS/PT2,
scaling, translation, closed-shell reduction, ten rejection checks and both
production RKS regressions. Worst UKS total FD error is 2.48330156e-9
Ha/Bohr against 5e-8. See the [DH acceptance ledger](DH_IMPLEMENTATION.md#complete-refined-grid-u11-molecular-acceptance-2026-09-20)
for the stencil, raw two-point truncation errors and logs.

This establishes angular under-resolution in the tested pruned grids. It
does **not** isolate pruning versus light-atom reduction versus maximum
angular order, establish radial convergence, or certify 1454 points for
every element, basis, functional and geometry. The previous claim that
pruning was ruled out is withdrawn: geometry-independent pruning can be
differentiable and still inaccurate.

## 2. Accuracy contract

Keep three independent tests:

1. Derivative consistency: the gradient differentiates the implemented
   finite-grid energy at the same stationary electronic state.
2. Quadrature convergence: energies and gradients stabilize under independent
   radial, angular and joint refinement.
3. Rigid motion: in row-coordinate convention, E(R Q)=E(R) and
   g(R Q)=g(R) Q. Finite lab-frame angular grids approximate this symmetry.

Preserve existing hard limits: FD 5e-8 Ha/Bohr for total/KS/PT2 separately;
rotation gradient 2e-6 Ha/Bohr and energy 2e-6 Ha; translation gradient
5e-8 Ha/Bohr and energy 1e-9 Ha. A quadrature tolerance must not relax
derivative consistency or SCF stationarity.

For a new accurate preset, propose additional calibration targets of
maximum gradient change <=1e-6 Ha/Bohr and absolute energy change <=1e-7 Ha
against each of two successive reference refinements at every tested
orientation/geometry. These are **proposed targets, not measured universal
guarantees**. The reference itself must stabilize to <=1e-7 Ha/Bohr and
<=1e-8 Ha before scoring a candidate. Extend the ladder or report unresolved
accuracy if it does not. Never call the largest available grid exact.
Larger-molecule energy scaling needs an explicit contract, not silent
conversion to a per-atom tolerance.

Electron integration is only an auxiliary diagnostic:

    N_grid = sum_g w_g rho_g
    N_AO   = Tr(P S)                     [RKS total density]
           = Tr((P_alpha + P_beta) S)    [UKS]

Compare the same density on both sides. Finite quadrature need not give
exact equality; integer targets are inappropriate for general fractional
occupations. Small electron-count error does not bound XC energy, kernels
or gradients. Do not require monotonic errors across non-nested rules or
renormalize weights/density to force this check to pass.

## 3. Architectural contract

Retain the atom-centered lab-frame rule:

    r_Akl = R_A + r_Ak * n_Al
    w_Akl = w_rad(A,k) * w_ang(A,k,l) * p_A(r_Akl; R)

For a selected element/basis/policy, radial/angular weights and shell
assignments are fixed. Only owner translations and the Becke partition
depend on geometry. Existing KS and complete XC-II point/partition
derivatives then remain applicable.

Introduce these proposed types:

- `GridRequest`: quality, policy version, independent radial/angular
  overrides, pruning selection and verification mode.
- `ResolvedGridSpec`: M4 rule/scale, supported element range, ordered
  radii/weights, Lebedev size per shell, pruning identifier, light-atom
  treatment and partition convention; stable fingerprint plus human/JSON
  reporting of every effective numerical setting.
- `MolecularGrid`: current geometry-dependent points, owners, atomic weights
  and partition weights built from that specification.

Resolve once and propagate through preparation, post-SCF quadrature,
displaced geometries and response. All four current driver rebuild sites
already forward `grid_angular_points`; centralize their interpretation.
Cache keys include specification, basis/order and geometry. Density changes
invalidate XC values; geometry changes invalidate AO/partition data.

Optimization and frequency stencils retain one rule family, radial counts
and per-shell angular schedule. Recompute points, AO values and partition
weights at each geometry; do not freeze their numerical values. Changing
resolution requires a fresh SCF/PT2/response solve. Never mix an energy,
gradient or Hessian column from different grid policies.

Excluded from the first fix:

- Molecule-aligned grid frames: they require frame derivatives and careful
  handling of axis/eigenvector degeneracies, and can hide orientation error
  without demonstrating quadrature convergence.
- Density-/geometry-driven shell switching during SCF, optimization or FD:
  its discrete changes are not differentiated by the present code.
- Replacement of Becke partitioning, M4 mapping or DH factors without
  independent evidence; functional-specific cancellation fitted to water.
- New meta-GGA, range-separated, ECP, heavy-element or MPI DH derivative
  support. Preserve existing guards and validate these separately.

## 4. Small, verifiable implementation steps

### G0 — Preserve a reproducible baseline

Commit compact test summaries/manifests for the six UKS and two RKS cases:
inputs, basis/functional, effective grid, revision, SCF controls, stencil,
raw/extrapolated FD errors, covariance and cost. Keep large generated logs
outside Git; workstation `/private/tmp` paths are not durable CI artifacts.

Verify: reproduce the original pruned-grid failing control and explicit-1454
passing control. Run source checks and grid invariants without compiling
inside a Python test. The negative control must still detect the old error.

### G1 — Centralize resolution without changing numbers

Implement the typed contract and route all consumers through it. Remove or
mark obsolete `int_acc`/`radial_row_factor` metadata: neither determines
today's radial count. Preserve preset and zero/nonzero angular behavior
exactly in this first refactor.

Verify: point/weight/owner equality, element shell counts, parse errors,
atomic-number/table limits, fingerprints, cache invalidation and all driver
rebuild paths. Reject unsupported elements in an accuracy-certified policy
instead of silently inheriting the final table entry.

### G2 — Independent radial control and reference oracles

Add a strict radial-count override (proposed `grid_radial_points`, zero for
preset; positive uniform per atom initially), retaining M4 scale and Becke
partitioning. Element-specific overrides can follow if calibration needs
them. Keep explicit unpruned angular operation available.

Verify exact angular moments, radial Gaussian integrals and off-center
molecular Gaussian integrals over several length scales; validate element
tables separately. Test descending radial order and shell-boundary cases.

Add a no-build convergence runner: radial N, ceil(1.25 N), ceil(1.5 N) at
fixed high angular order; angular 974/1454/2030 at fixed refined radial
counts; then joint refinement. These are starting ladders, not guaranteed
endpoints. Extend when needed, reject unsupported sizes, retain failures.

### G3 — Attribute angular error before choosing pruning

At converged radial resolution compare legacy pruning, the same scheme
without H/He reduction, unpruned legacy maximum order, and the converged
high-order reference. Log shell-resolved XC energy/response contractions
in a test harness, not a production debug branch.

Use fixed-density matrix/contraction differences to isolate quadrature,
then fresh SCF/PT2 gradients to include response amplification. Preserve
roots; compare alpha/beta matrices and multiple independent response
directions, not only a trace. For DH include complete XC-II P-side, D-side,
partition and owner-translation channels.

Exit: measured attribution among light-atom reduction, shell pruning and
angular order, with radial error separately bounded. Do not select a
schedule solely from three H/O rotation failures.

### G4 — Calibrate an accurate preset, then recover pruning

Use the independently converged unpruned rule as the correctness baseline.
Design fixed per-element schedules in physical radial regions (for example
r/xi), with explicit ordered boundaries. Physical-radius pruning is not
intrinsically accurate: choose boundaries/orders using G3 evidence and G2
references, starting conservatively and measuring each point reduction.

No automatic historical H/He reduction. Admit it only after the same tests,
including diffuse/polarized bases. Any basis dependence uses immutable basis
descriptors in the fingerprint and stays fixed over geometry displacements;
never prune on the changing SCF density.

Exit: Section 2 calibration targets pass across the matrix below, with
point count, AO-grid memory and timings alongside accuracy. A cheap candidate
that misses forces fails. If no efficient schedule passes, release the
explicitly documented unpruned accurate preset first and optimize pruning
separately without relaxing gates.

### G5 — Molecular derivatives and supported workflows

Run all-coordinate energy FD at both existing steps. Retain raw two-point
and existing fourth-order UKS results separately. Check stationary roots,
Z residuals, total/KS/PT2 channels, translation and at least two non-octahedral
rotations. Include near-linear and symmetry-breaking geometries to expose
topology/frame discontinuities.

Minimum calibration matrix:

- Current six UKS B2PLYP and production RKS water/C1 H2O2 controls.
- RKS/UKS LDA, pure GGA and global hybrid on small systems to isolate shared
  KS accuracy from DH response amplification.
- C/N/F molecules, a diffuse anion and a polarized basis; Si/P/S/Cl examples
  before claiming second-row coverage. Establish reproducible roots first;
  failed SCF is not evidence for or against quadrature accuracy.
- Perturbed bond lengths/angles, stretched and near-linear geometries;
  held-out acceptance geometries separate from pruning calibration cases.
- Existing RKS DH optimization and semi-numerical frequency audits: one
  resolved policy across every Hessian column, full-matrix comparison under
  further grid refinement, unchanged requested displacement sizes. UKS
  optimization/frequencies remain outside current support.

Register fast primitive checks in CTest. Run expensive convergence/FD
matrices as explicit acceptance jobs. Source-string tests cannot establish
numerical acceptance.

### G6 — Deliberate preset and production migration

After G5, publish the measured accurate preset generation and an explicit
legacy selector. A selected quality/policy applies equally to energies,
gradients, response and supported optimization/frequency workflows; never
silently upgrade only gradients. Keep `grid_angular_points 0` meaning the
selected preset's schedule and explicit overrides authoritative.

Switch default preset generation in a separate reviewed commit with release
notes, resolved-setting logs and intentional energy/gradient rebaselines.
Declare which preset meets tight-force targets; `normal` alone is not a
numerical guarantee. An opt-in verification run may restart with a stricter
fixed policy and report unresolved accuracy, not change nodes mid-solve.
Offline calibration is not a certificate for every user system.

Narrow UKS DH production promotion is separate, following these accuracy
policy decisions. Preserve all-active conventional unshifted PT2, global
LDA/GGA, Cartesian and no-solvent restrictions. The current 1454-point pass
removes the derivative blocker, not those scope limits or the need to
address default accuracy. Do not promise “1454 or larger always works.”

### G7 — Bound memory without altering the rule

After correctness, share identical Lebedev tables and resolved atomic rules;
batch AO/XC evaluations to avoid retaining all values/gradients/Hessians.
Preserve point order and owner/partition metadata, including distribution
across ranks where the consumer supports it.

Verify dense/batched energies, Fock and XC response matrices, complete
XC-II, gradients, thread/rank consistency, peak memory and wall time.
No density screening or point deletion is hidden in a memory optimization.
The serial/OpenMP U11 result does not imply MPI DH acceptance.

## 5. Delivery and stop conditions

G0/G1 precede numerical changes; G2/G3 precede pruning selection; G4/G5
precede defaults migration. Each step gets a small commit, tests and a
measured ledger. Wait for the user's rebuild before numerical execution.
Do not raise SCF caps or relax thresholds to finish an experiment.

Completion means a declared, reproducible preset meets derivative,
independent-refinement and rigid-motion targets on the expanded matrix,
behaves consistently in every supported consumer, and states its cost and
compatibility policy. More points on one molecule, FD on an orientation-
dependent grid, or changing a guard alone does not satisfy that contract.
