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

## Also stale

`DFT_GRID_CONVERGENCE.md` "Not done" says no committed case exercises the
production UKS DH gradient on V4. `h2o2_cation_uks_b2plyp_v4_gradient_sto3g`
now does; fix that line with the decision.
