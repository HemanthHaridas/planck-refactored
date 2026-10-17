# How is a DFT grid pruning policy qualified, and why is V4 the qualified one?

This file answers one architecture question:

**What does it take for an angular-pruning schedule to be trusted for
calculations that differentiate the grid (UKS double-hybrid gradients), and
what is the qualified `g4-3-v4` policy?**

## Short answer

A candidate pruning rule may depend only on atomic number and a fixed physical
radial coordinate. It is accepted only if, on every fixture, it reproduces a
135-radial-shell × 2030-angular-point reference within non-cancelling L1 limits
on three fixed-density observables: XC energy, $D:V_{\rm XC}$, and
$D:f_{\rm XC}:D$. The evidence is tiered, and a holdout that has been looked
at can never be reused as independent evidence.

V3 failed one channel by a factor of 2 and is unqualified. V4 was found by an
unconstrained search over the whole development bank, reproduced exactly by the
C++ builder, and then passed a blind suite frozen before its first run. It also
passed live UKS/DH checks where the default (legacy) pruning failed rotational
covariance: 2.2e-5 against V4's 1.1e-7 Ha/Bohr on C1 H2O2⁺.

**Scope.** V4 is opt-in (`grid_policy g4-3-v4`); `legacy` stays the default.
UKS global-double-hybrid gradients *require* V4 with `grid_radial_points 135`
and no `grid_angular_points` override (`validate_qualified_uks_dh_grid`). The old
`g4-3-v4-test-only` spelling is an input-compatible alias for the same immutable
schedule.

## Where the logic and evidence live

- `src/dft/base/grid.h` — `GridPruningPolicy`, `G43V4QualifiedV1`, the resolver
- `src/dft/driver.cpp` — `validate_qualified_uks_dh_grid`
- `tests/g4_3_*.py`, `tests/g4_3_direct_grid.h` — search, direct-rescore, and holdout runners
- `tests/inputs/exploratory/dh_gradient/grid_calibration/artifacts/g4_3_9_revision_2026_10_03/` — raw evidence (the V4 holdout ledger is `v4_holdout/ledger.json`)

## What this does not claim

- It does not change a default grid.
- A direct-grid/atlas match proves the C++ builder realizes a rule, not that
  the rule integrates accurately.
- No other program's double-hybrid analytic gradient is used as an oracle.

## Fixed numerical contract

All candidate rules use the physical radial coordinate

```math
u = r_{\mathrm{M4}} / r_{\mathrm{Bragg}}(Z).
```

They may depend only on atomic number and this fixed radial coordinate. They
must not depend on density, molecular geometry, SCF iteration, or orbital
response.

| Family | Atomic numbers |
|---|---|
| H/He | 1--2 |
| Second row | 3--10 |
| Third row | 11--18 |

The fixed reference is 135 radial shells and 2030 angular points. A selected
rule is accepted only when every relevant fixture obeys non-cancelling L1
limits:

| Observable | Limit |
|---|---:|
| XC energy | 1e-8 Ha |
| $D:V_{\rm XC}$ | 1e-7 |
| $D:f_{\rm XC}:D$ | 1e-7 |

Each fixture uses two independent AO-density directions. Signed sums are
diagnostic only; a candidate cannot pass through cancellation between shells,
atoms, spin channels, or directions.

## Evidence tiers

### Frozen selection atlas

The original 14 fixture reports in **selection/** establish comparable cost and
base accuracy. They are immutable. Every candidate uses the same allowed
boundaries and Lebedev-order vocabulary against these reports.

### Development bank

Every v1--v3 holdout is consumed. It is useful diagnostic calibration evidence
but can never again be presented as independent qualification. The V4 search
must use the complete development bank, including prior passes and failures.

Missing normalized residual reports from prior runs must be regenerated and
stored under **v4_development/** before V4 enumeration begins. Existing reports
are retained verbatim.

### Blind qualification

Blind inputs are frozen only after a candidate has been selected from the
selection atlas and development bank. They must not be inspected, rescored, or
used to adjust the rule before the full batch has completed. Their request must
contain inputs, hashes, families, roles, reference ladder, limits, and the
selected candidate fingerprint.

## What was found

### Ruled out: V3

The previous candidate, test-only V3:

| Family | Boundaries in $u$ | Lebedev points, bands 0--4 |
|---|---|---|
| H/He | [0.4, 1.0, 3.5, 4.5] | [26, 146, 1454, 434, 434] |
| Second row | [0.4, 1.0, 2.5, 4.5] | [110, 434, 1454, 590, 170] |
| Third row | [0.1, 0.5, 3.5, 4.5] | [26, 146, 770, 230, 26] |

Its frozen-atlas cost is 64,998 maximum points per atom and 2,598,078 total
ledger points. The literal direct builder is
**--fixed-xc-g4-3-v3-direct-rescore**, with fingerprint
**g4-3-v3-constrained-selection-v1**.

All 14 selection fixtures passed literal direct-grid/atlas reconciliation. The
largest discrepancy was 1.2097574331182415e-10, below the fixed 1e-9
implementation-identity limit.

V3 nevertheless failed qualification and is unqualified:

| V3 qualification case | Reference ladder | Direct reconstruction | L1 result |
|---|---:|---:|---:|
| C1 H2O2/B2PLYP/STO-3G | pass | 1.57e-10 | **fail** |
| stretched HCl/B2PLYP/STO-3G | pass | 1.89e-11 | pass |
| displaced water-cation UKS/PBE0/6-31G | pass | 1.18e-10 | pass |

The only failed channel is H/He XC-energy L1 for H2O2:

```math
2.0703621109700418\times10^{-8} > 1.0\times10^{-8}.
```

All H2O2 potential/kernel channels and the second-row energy channel pass.
The selected outer H/He band, $u \ge 4.5$, at 434 points contributes
1.3117254654494327e-8 (63.4%) of that L1 value. At this fixed density, 590
points in that band would reduce the measured value to approximately 9.52e-9.
That result is a V4 diagnostic constraint, not permission to patch V3 and call
the same holdout independent.

### The V4 search

1. Validate and collect all selection and development residual reports.
2. Keep the allowed band boundaries and Lebedev-order vocabulary unchanged.
3. Enumerate one geometry-independent rule per periodic family over the union
   of the frozen selection atlas and the whole development bank.
4. Require the same rule to satisfy H/He midrange/boundary, H/He outer-band,
   and second-row-core evidence simultaneously. Do not impose a sequence of
   manual one-band patches.

Every survivor must meet the ordinary L1 limits on every selection and
development report. The first production-rule search is deliberately
cost-unconstrained: among survivors with the robustness margin, rank first by
maximum atom points, then total ledger points, then normalized L1 error. This
finds the cheapest rule that is already robust; optimizing its construction is
a later task.

For robustness, prefer only survivors with an internal twofold margin:

| Development observable | Internal target |
|---|---:|
| XC energy | 5e-9 Ha |
| response contractions | 5e-8 |

Do not relax physical acceptance limits or silently select a barely passing
rule. A uniform/high-resolution fallback remains preferable to an unqualified
pruning policy.

**Prior bounded probe:** the completed development-bank probe found
1,946,397 hard-limit-qualified H/He rules and 1,793,278 second-row rules, but
zero robust-margin survivors for either family under the ceiling. Third row has
222,011 robust survivors. The all-2030 H/He/second-row probe output is
diagnostic only; `g4-3-v4-search-status.json` preserves this bounded-search
result. The unconstrained run that followed removed that ceiling, and a robust
rule was required before any performance optimization or blind validation.

**Unconstrained V4 result:** the complete search found robust survivors for
all three families and selected the lowest-cost survivor in each:

| Family | Boundaries in $u$ | Lebedev points, bands 0--4 | Normalized worst L1 | Maximum points/atom |
|---|---|---|---:|---:|
| H/He | [0.25, 1.0, 2.5, 3.5] | [26, 146, 2030, 974, 590] | 0.4495606113 | 80,010 |
| Second row | [0.4, 1.0, 2.5, 4.5] | [110, 770, 2030, 590, 170] | 0.3965197292 | 82,170 |
| Third row | [0.4, 0.8, 2.5, 4.5] | [86, 590, 974, 590, 26] | 0.4832675922 | 53,538 |

The combined selection-atlas ledger has a maximum of 82,170 points/atom and
3,234,918 total points. This is 11.6% above the former 73,626-point ceiling,
not the uniform 2030-point fallback's 274,050 points/atom. It is a
development-selected rule only: direct-builder reconciliation and the frozen
blind qualification suite remain required before promotion.

### Literal realization

The V4 direct builder reconciled with all 14 frozen selection and all 9
consumed-development shell atlases. Every one of the 23 reports passed the 1e-9
identity limit, with a worst absolute channel difference of
1.9399352610416365e-10 (SiH4/B2PLYP/cc-pVDZ). This proves that the C++
physical-grid selector realizes the enumerated rule. It does not qualify the
rule's quadrature accuracy on unseen systems; the blind suite does that.

### V4 blind and live qualification

Freeze at least six unused input states before their first calculation:

1. Two non-symmetric RKS H/second-row geometries, spanning pure GGA and global
   double-hybrid XC.
2. Two H/third-row global-double-hybrid geometries with distinct elements or
   basis classes.
3. Two open-shell UKS states, spanning pure GGA and hybrid XC.

Every input state must differ from all selection and development inputs. The
qualification runner persists product-reference ladders, residual reports,
literal direct rescoring, and a single aggregate ledger.

**V4 blind result:** the frozen six-case suite passed in full: two non-symmetric
H/second-row RKS B2PLYP cases, two H/third-row RKS B2PLYP cases, a UKS/PBE
triplet, and a UKS/PBE0 cation. Every product reference was stable, every
literal direct reconstruction was below the 1e-9 identity limit, and every
fixed-density L1 certificate passed. The largest direct reconstruction
difference was 6.14204520577033e-10 (the UKS/PBE triplet). The durable ledger
is `artifacts/g4_3_9_revision_2026_10_03/v4_holdout/ledger.json`.

**Live stationary result:** V4 also passed the three difficult live UKS/DH
rotation fixtures. The largest V4 total-gradient covariance errors were
1.13e-7 (C1 H2O2+), 9.29e-8 (HO2), and 7.04e-7 Ha/Bohr (water cation/PBE0/
6-31G); each is below the 2e-6 live covariance limit. The corresponding
legacy rules failed all three cases at 2.17e-5, 6.86e-6, and 5.34e-6
Ha/Bohr. Every V4 run retained converged SCF and Z-vector residuals.

**Live molecular FD result:** all three difficult UKS/DH molecular checks pass
at the two fixed displacement sizes with the fourth-order total-energy
stencil. The worst total-gradient errors are 2.68e-9 (C1 H2O2+), 1.31e-9
(HO2), and 7.59e-10 Ha/Bohr (asymmetric water cation), versus the 5e-8
limit. Translation errors are below 5.9e-13 Ha/Bohr and rotational covariance
errors below 7.8e-8 Ha/Bohr. This validates V4 against live stationary
unrestricted double-hybrid response on the three historical difficult cases.

**Blind live UKS FD result:** the two frozen open-shell qualification inputs
also pass a fourth-order total-energy finite-difference check at 135 radial
shells under V4.  The PBE water-triplet/STO-3G case has a maximum analytic--FD
error of 3.24e-8 Ha/Bohr and the PBE0 water-cation/6-31G case 6.77e-9
Ha/Bohr, both below the fixed 5e-8 Ha/Bohr limit.  The checks use tightened
SCF energy and density tolerances (1e-12), inner/outer displacements of
5e-4/1e-3 Bohr, and retain full reported energy/gradient precision so the
finite-difference oracle itself is not quantized at the acceptance scale.
Together with the fixed-density blind ledger, these are the complete blind
UKS qualification evidence for the opt-in V4 policy.

Run the entire blind suite before inspecting its result. Qualification requires
every case to pass reference stability, direct reconstruction, XC-energy L1,
and both response L1 channels. A failure is classified before another search:

- a **known-regime failure** means the development search or report handling is
  inadequate; or
- a **new-regime failure** means the selector model needs an explicitly scoped
  extension.

Neither outcome authorizes an in-place patch to the candidate just tested.

## Promotion rules

Only an all-pass blind suite authorizes an opt-in GridPruningPolicy entry,
resolver/parser support, shell-size invariants, and default-legacy regression
checks. Production promotion remains separate from live stationary SCF/DH
controls, which must subsequently compare the candidate with refined grids at
fixed requested finite-difference steps.

V4 has satisfied those qualification prerequisites, including the two blind
live UKS molecular finite-difference checks.  The normal `planck-dft` route
(without the validation-ledger sidecar) must be rebuilt and smoke-tested on a
V4 UKS/B2PLYP gradient before the promotion implementation is declared
released.  That smoke test checks routing and the grid contract; it is not a
new selector fit or a reason to revise the frozen V4 rule.

Default-policy migration requires a separate review with a compact calibration
manifest: resolved rule/fingerprint, inputs and hashes, reference ladders,
channel errors, point counts, memory/timing, stationary-root diagnostics, and
live derivative results. No default changes before that review.

## What remains

- **No committed regression case exercises the production UKS double-hybrid
  gradient on V4.** The qualification evidence comes from runners under
  `tests/`, not from `tests/regression_cases.json`. The release smoke test
  described above is the obvious candidate to commit.
- **Default migration is a separate decision.** The legacy pruning failed
  rotational covariance on the hard UKS/DH cases. Whether that matters for
  workflows that do not differentiate the grid at this precision has not been
  measured.
