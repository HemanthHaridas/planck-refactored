# G4.3.9 revision evidence — 2026-10-04

This is the durable, repository-resident evidence set for the test-only
G4.3.9 second-row response-floor revision. It must not be regenerated merely
because `/private/tmp` is cleared.

- `selection/` contains the complete normalized 14-fixture frozen selection
  atlas. These reports are the inputs to future constrained-frontier searches.
- `g4-3-second-row-constrained-frontier.json` is the unique second-row
  response-floor frontier result.
- `g4-3-revised-combined-frontier.json` is the re-certified combined
  candidate. It is unqualified and must never be treated as a production grid
  policy.
- `g4-3-dual-floor-*-frontier.json` are the current dual-energy-floor family
  searches and re-certified combined candidate. This later candidate remains
  test-only: its literal-grid reconciliation is complete, but it has not
  undergone fresh holdouts.
- `dual_floor_direct_rescores/` contains the complete 14-fixture
  literal-grid/atlas reconciliation for the dual-energy-floor candidate. All
  reports pass the fixed `1e-9` tolerance; the largest discrepancy is
  `2.054687739394545e-10` (SiH4/cc-pVDZ). This establishes direct-builder and
  frozen-atlas identity only.
- `direct_rescores/` contains the six literal-grid/atlas reconciliations whose
  selected grids differ from the retired candidate. The remaining eight
  calibration fixtures use unchanged H/He and third-row rules and retain their
  earlier reconciliation evidence.
- `holdouts/` contains the normalized H2S pass and UKS/PBE failure ledgers,
  plus their reference, direct-grid, and raw shell-residual sidecars. The raw
  UKS/PBE residual is required to derive a later energy-floor repair.

The H2S geometry holdout passed. The UKS/PBE functional-spin holdout failed
the fixed `1e-8` XC-energy L1 limit, so this candidate is not eligible for
production promotion.

The dual-energy-floor candidate has completed its frozen selection-matrix
reconciliation. All three prior holdouts remain consumed, so its next required
evidence is a separately frozen, previously unused three-case holdout set;
do not promote it to a production policy before that qualification.

The v3 constrained candidate is stored in `g4-3-v3-combined-frontier.json`.
Its literal direct builder was independently reconciled against all fourteen
frozen selection-atlas fixtures on 2026-10-06. The normalized reports are in
`v3_direct_rescores/`; all pass the fixed `1e-9` direct-minus-atlas gate, with
largest error `1.2097574331182415e-10` (HO2). This proves that the physical
builder reproduces the selected atlas rule, not that the candidate is eligible
for production. The next evidence must be newly frozen holdouts; all v2 cases
remain consumed diagnostic data.

`holdouts_v3/` is the durable v3 qualification record. HCl/B2PLYP and
water-cation UKS/PBE0 pass all gates. C1 H2O2/B2PLYP fails only the H/He
XC-energy L1 gate (`2.0703621109700418e-8` versus `1e-8`); its reference ladder
and literal direct-grid reconciliation pass. The v3 candidate is unqualified.
The raw H2O2 residual identifies its outer H/He `u >= 4.5` band as the dominant
energy source, but this consumed holdout may only inform a subsequent diagnostic
constraint—not a direct production-rule edit.

The v2 request (`g4_3_dual_floor_holdout_request.json`) is now frozen with
displaced PBE water, displaced B2PLYP PH3, and displaced UKS/PBE triplet-water
inputs. Their reference, residual, and literal-direct results are to be kept
as separate qualification evidence and must not be used to re-select a rule.

`holdouts_v2/water-triplet-uks-pbe-displaced-sto3g/` is the completed
functional/spin qualification case. Its reference ladder and literal direct
reconstruction passed (the latter at `3.190266384400786e-10`), but its frozen
L1 certificate failed the unchanged H/He energy/kernel and second-row kernel
limits. The dual-energy-floor candidate is therefore unqualified; preserve
this case only as a diagnostic for a later, separately re-certified derivation.
