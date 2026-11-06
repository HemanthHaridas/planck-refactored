# Why is PySCF's DFT gradient wrong on ECP atoms, and what is the upstream fix?

Research scope, opened 2026-11-06. The root cause is identified and confirmed;
nothing has been reported upstream yet. **When this work lands, it must be
reported to PySCF** (step R5). Status lives in `vault/Status/Open Work.md`.

## The finding

Found while gating ECP DFT in Planck (`docs/DEF2_ECP_SCOPE.md` step 6). On
C1 AuH·H₂O / def2-SVP (60-electron Au ECP), RKS B3LYP, 150 × 590 unpruned grid:

| quantity | Planck | PySCF 2.13.0 |
|---|---|---|
| energy | −212.7277364439 | −212.7277364439 |
| analytic gradient, Au x (grid response on) | +0.0043842546 | +0.0043752181 |
| energy finite difference, Au x (δ 5e-4 Bohr) | +0.0043842550 | +0.0043842534 |

The two codes agree on the energy to 1e-12 and on its finite difference to
2e-9. Planck's analytic gradient matches both finite differences (Planck's
fourth-order FD over all 15 components: ≤ 1.6e-8 Ha/Bohr); PySCF's is up to
1.1e-5 Ha/Bohr off, on Au and O. The PySCF HF gradient with the same ECP
matches Planck to 3e-8, so the defect is DFT-specific.

## Root cause (confirmed)

The Becke partition weights carry a Treutler (or Becke) atomic-size adjustment
built from Bragg radii, looked up per atom by nuclear charge. PySCF does that
lookup two different ways:

- **energy:** `dft/radi.py`, `treutler_atomic_radii_adjust` and
  `becke_atomic_radii_adjust` use
  `charges = [elements_proton(x) for x in mol.elements]`, the proton number
  (Au: 79);
- **gradient grid response:** `grad/rks.py`, `grids_response_becke`, nested
  `_radii_adjust`, uses `charges = mol.atom_charges()`. On an ECP atom that is
  Z − n_core (Au: 19, potassium's radius).

So the analytic gradient differentiates a different partition from the one the
energy was integrated on. Only ECP atoms are affected (elsewhere both give Z),
and only with grid response on and a radius adjustment, which is PySCF's
default (`radii_adjust = treutler_atomic_radii_adjust`).

**Proof:** replacing that one line with the proton-number lookup at runtime
makes PySCF's AuH·H₂O gradient agree with Planck's on all 15 components to
≤ 2.4e-10 Ha/Bohr (it was 1.1e-5), and therefore with both codes' energy
finite differences. Planck's reference script applies this workaround
(`patch_pyscf_grid_response_radii` in `tests/pyscf/def2_ecp_dft_references.py`).

## Upstream status (checked 2026-11-06)

- **master**: still `charges = mol.atom_charges()` in `grids_response_becke`
  (line 497). The file was last touched by `827a622d07` (2026-11-05, NumInt
  XC customization APIs) and `55b925a007` (#3525, 2026-10-18, "Fix saturated
  grid-response derivatives and guard small denominators"), neither of which
  changes this lookup.
- **hotfix** branch: same line (229).
- **Issues and PRs:** no match for `grids_response_becke`, for the radius
  lookup, or for ECP grid-response gradients. #3532 (merged 2026-11-02, "Fix
  numerical and API defects") touches ECP integrals and initialization, not
  this.
- Not checked: GPU4PySCF, which has its own grid-response code.

**Re-check all of the above immediately before filing (R5).**

## Questions this scope answers

1. **Reach.** Which PySCF paths call `grids_response_becke`: RKS and UKS
   gradients (does `grad/uks.py` reuse it?), hybrid, range-separated and
   double-hybrid gradients, TDDFT excited-state gradients, solvent, and the
   Stratmann partition (#3156)? Does any analytic Hessian with grid response
   repeat the lookup?
2. **Size.** How the error scales with the ECP core (I, Au, Ag, Pd: small and
   large cores), with grid size, with the Treutler vs Becke adjustment, and
   whether `grid_response = False` hides or changes it. Is it ever large
   enough to move a geometry optimization?
3. **Other `atom_charges()` uses near grids.** `hessian/rks.py:2091` picks the
   radial grid size from `any(mol.atom_charges() > 10)`, which undercounts ECP
   atoms the same way. That is a grid-size heuristic, not a wrong derivative,
   but it gives heavy ECP atoms a smaller grid than intended. Also check ghost
   atoms (charge 0) against `elements_proton`, and the PBC and solvent grid
   code.
4. **The fix.** Most likely the one-line change above (use the proton number,
   as `radi.py` does). Confirm there is no case where `atom_charges()` is
   intended (e.g. fractional-charge MM atoms), and decide whether to share one
   helper between `radi.py` and `grad/rks.py` so they cannot diverge again.

## Steps

- **R1. A minimal reproducer in pure PySCF**, no Planck: a small ECP molecule
  (HI or AgH·H₂O with a standard PySCF basis name), the analytic gradient with
  grid response, and the energy finite difference, showing the gap and its
  disappearance with the fix. This is the core of the upstream report.
- **R2. Reach and size** (questions 1–2): a table over elements, core sizes,
  functionals (LDA, GGA, hybrid, range-separated), RKS/UKS, radius schemes and
  grid sizes.
- **R3. Related lookups** (question 3).
- **R4. The patch with a test**, in PySCF's style: a test in
  `pyscf/grad/test/` comparing the ECP DFT gradient (grid response on) against
  a finite difference, failing on master and passing with the fix.
- **R5. Report upstream** once R1–R4 are done: an issue with the reproducer
  and the table, and a PR with the fix and test. Re-check upstream status
  first. Then record the issue and PR numbers here and in the vault.
- **R6. Planck follow-up.** When the fix is released, drop
  `patch_pyscf_grid_response_radii` (it already becomes a no-op when the line
  is gone) and pin the PySCF version that has it.

## Not in scope

Changing PySCF's grid defaults; GPU4PySCF (note in the report if it shares the
code); any change to Planck, whose gradient is already correct.
