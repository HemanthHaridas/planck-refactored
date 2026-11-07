# Why is PySCF's DFT gradient wrong on ECP atoms, and what is the upstream fix?

Research scope, opened 2026-11-06. The root cause is identified and confirmed.
R1–R4 are done (2026-11-07): a pure-PySCF reproducer, reach and size,
related lookups, and a patch with a test. R4 also found a second trigger on
PySCF ≥ 2.14: ghost atoms. R5 is prepared (2026-11-07). Upstream was
re-checked that day: no existing issue or PR, and master was still at
`830bf89`. Because PySCF's `AI_POLICY.md` requires a human to submit, the
issue and PR are filed by hand. The ECP and ghost cases are one PR: the same
lookup line causes both, and a single fix covers both.
- Branch: `fix-grid-response-ecp-radii` (`91d731796`) on
  `HemanthHaridas/pyscf`.
- Drafts: `tests/pyscf/upstream/ISSUE.md` and `PR.md`. The PR's Notes
  discloses the AI assistance, as the policy requires.
- Reproducer: `issue_repro.py`, prints 9.8e-6 Ha/Bohr on master and 6.8e-10
  with the fix.

**Filed 2026-11-07:** issue pyscf/pyscf#3597, PR pyscf/pyscf#3598 (head
`91d731796`). Remaining: R6 once a release carries the fix. Status lives in `vault/Status/Open Work.md`.

## The finding

Found while gating ECP DFT in Planck (`docs/DEF2_ECP.md` step 6). On
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
3. **Other `atom_charges()` uses near grids.** (Answered under R3: the claim
   below about the Hessian grid size was wrong; it is only a warning.)
   `hessian/rks.py:2091` picks the
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

- **R1. A minimal reproducer in pure PySCF — DONE 2026-11-06.**
  `tests/pyscf/pyscf_ecp_grid_response_repro.py`: PySCF built-ins only
  (`basis="def2-svp"`, `ecp={"Au": "def2-svp"}`), C1 AuH·H₂O, PBE, grid level
  5, about a minute. It prints the stock analytic gradient, the analytic
  gradient with the one-line fix, and a central difference of the energy
  (δ 1e-3 Bohr) for all 15 components. PySCF 2.13.0:

  | | max \|analytic − energy FD\| |
  |---|---|
  | stock | 4.3e-6 Ha/Bohr (worst on Au z and O z) |
  | one-line fix | 1.4e-7 Ha/Bohr (the δ² truncation of the central difference) |

  `atom_charges()` is 19 for Au against a proton number of 79. This is the
  core of the upstream report. (R1 also recorded that a diatomic, HI, does
  not reproduce the defect. R2 showed that was a property of iodine, not of
  diatomics.)
- **R2. Reach and size — DONE 2026-11-07.**
  `tests/pyscf/pyscf_ecp_grid_response_survey.py` (PySCF 2.13.0, def2-SVP +
  def2 ECP, about 10 minutes). It reports max |stock − fixed| over all
  gradient components, which is the error, since R1 shows the fixed gradient
  matches the energy FD.

  **Reach (from the code).** Every grid-response gradient goes through
  `grids_response_cc`: RKS and UKS (`grad/uks.py` calls
  `rks_grad.grids_response_cc`), ddCOSMO, MC-PDFT and L-PDFT, and SGX.
  `grids_response_cc` dispatches to `grids_response_becke` for the
  `original_becke` (default) and `stratmann` schemes, so both are affected.
  The `becke_lko` scheme goes through `_get_lko_partition_functions`, which
  calls the grid's real `radii_adjust` (proton numbers), so it is correct.
  No analytic Hessian path calls either response function. TDDFT
  excited-state gradients reuse the RKS/UKS XC gradient machinery and were
  not checked separately.

  **Which ECP atoms.** Any atom where the Bragg radius at Z − n_core differs
  from the one at Z. Diatomics are affected. **HI is immune only because
  Bragg(I) = Bragg(Mn), Z − n_core = 25**; Te (→ Cr) and Y (→ Na) are immune
  the same way. Every other def2-ECP element from Rb to Rn differs.

  | case (level 3 unless noted) | max \|stock − fixed\| (Ha/Bohr) |
  |---|---|
  | AuH, PBE | 2.9e-7 |
  | AgH, PBE | 4.7e-7 |
  | CsH, PBE (46-electron core; Cs gets F's radius) | 4.3e-6 |
  | HI, PBE (control) | 2.6e-21 |
  | AuH·H₂O RKS LDA / PBE / B3LYP | 1.8e-5 / 1.8e-5 / 1.8e-5 |
  | AuH·H₂O RKS ωB97X | 3.1e-5 |
  | AuH·H₂O⁺ UKS PBE (doublet) | 1.9e-5 |
  | AuH·H₂O PBE, level 2 / 3 / 5 / 7 | 4.3e-5 / 1.8e-5 / 4.4e-6 / 6.3e-7 |
  | Becke radius adjustment (not Treutler) | 1.2e-5 |
  | Stratmann scheme | 2.0e-5 |
  | no radius adjustment (control) | 2.2e-16 |
  | `becke_lko` scheme (control) | 0 |

  AuH bond component against a fourth-order energy FD (δ 2e-3 Bohr): stock
  −0.0077384079, fixed −0.0077381227, FD −0.0077381226. A second-order FD at
  1e-3 Bohr has ~1e-7 truncation and could not separate the two.

  The error does not depend on the functional family. It falls with grid
  size but does not vanish, and it is largest at PySCF's coarse default
  levels. `grid_response = False` does not call `grids_response_becke` at
  all, so it hides the defect, along with the grid-response term itself.
  Whether it moves a geometry optimization was not measured. At 1e-5
  Ha/Bohr it is below the usual 4.5e-4 force threshold, so the expected
  effect is a small shift in the converged geometry, not a failure.
- **R3. Related lookups — DONE 2026-11-07.** `grids_response_becke` is the
  only place a grid radius is looked up by `atom_charges()`.
  - The remaining `atom_charges()` uses near grids are nuclear-charge
    electrostatics, where Z − n_core is correct on an ECP atom:
    `dft/numint.py`, `solvent/grad/ddcosmo_grad.py`, `solvent/grad/pcm.py`
    and the PBC stress code.
  - Solvent cavity radii use `gto.charge(symbol)`, the proton number.
  - `hessian/rks.py` `_check_mgga_grids` uses `atom_charges() > 10` only to
    decide whether to warn that a meta-GGA Hessian grid may be too coarse. It
    does not choose the grid size (question 3 said it did). On ECP atoms it
    can miss the warning. Mention it in the report as a minor note.
  - Ghost atoms give 0 under both lookups (`GHOST-H`: `atom_charges()` 0,
    `elements_proton` 0), so they are consistent.
  - Question 4: no use of `atom_charges()` as a radius key looks
    intentional. MM point charges are not atoms in `mol` and do not reach
    this code. A shared helper with `radi.py` is the natural fix. That
    decision is part of R4.
- **R4. The patch with a test — DONE 2026-11-07.** The patch is
  `tests/pyscf/upstream/0001-Use-the-partition-s-radius-lookup-in-the-grid-respon.patch`,
  against PySCF master `830bf89` (2026-11-06, after v2.15.0). Apply it with
  `git am`.
  - **The fix.** A new `radi.atomic_radii_charges(mol)` returns the element
    proton numbers with the ghost prefix stripped.
    `becke_atomic_radii_adjust`, `treutler_atomic_radii_adjust` and
    `grids_response_becke` all call it, so the partition and its derivative
    cannot diverge again. That settles question 4's shared-helper choice.
  - **The test**, `pyscf/grad/test/test_grid_response_radii.py`. It uses the
    method of the existing `test_grid_response_cutoff.py`: the weight
    derivatives from `grids_response_cc` against a central difference of
    `gen_grid.get_partition`, on mocked atomic grids. There is no SCF, and it
    runs in under a second. It covers an ECP molecule (Au, H, O) with the
    Becke scheme (Treutler and Becke radii) and the Stratmann scheme, and a
    ghost molecule (O, ghost-O, H) with both schemes. The molecule needs O,
    not a second H: with Becke radii, the Au–H adjustment saturates at the
    ±0.5 clamp under both lookups and cannot tell them apart.
  - **Result.** On master, all six tests fail. With the patch, they pass,
    along with the existing `test_grid_response_cutoff.py`,
    `test_grid_response_lko.py`, `dft/test/test_h2o.py` (the #3302 ghost
    tests), `grad/test/test_rks.py` and `grad/test/test_uks.py`: 90 passed,
    16 deselected by PySCF's slow markers.
  - **Build note.** Master needs a newer libxc than the 2.13.0 wheel ships
    (the wheel lacks `xc_func_find_ext_params_name`). The PySCF C libraries
    were built from the clone with `BUILD_LIBXC=ON`, reusing the wheel's
    libcint and xcfun.
  - **Second trigger: ghost atoms, a regression in v2.14.0.** PySCF #3302
    (`84d558d3e`, 2026-07-09, "Fix the bad atomic radii assigned to ghost
    atom") changed `radi.py` and `gen_grid.py` to look ghost atoms up by
    their element. It did not change `grids_response_becke`, which still gets
    0 from `atom_charges()`. On a water dimer with one ghost monomer (6-31G,
    PBE, level 3), the O y gradient on master is 1.41e-5 Ha/Bohr off a
    fourth-order energy FD. With the patch it is 2.8e-10
    (`tests/pyscf/pyscf_ghost_grid_response_repro.py`). This hits
    counterpoise-corrected DFT gradients with grid response, with no ECP
    involved. PySCF 2.13.0, which Planck's venv uses, is not affected,
    because there both lookups give 0 for a ghost.
- **R5. Report upstream** once R1–R4 are done: an issue with the reproducer
  and the table, and a PR with the fix and test. Re-check upstream status
  first. Then record the issue and PR numbers here and in the vault.
- **R6. Planck follow-up.** When the fix is released, drop
  `patch_pyscf_grid_response_radii` (it already becomes a no-op when the line
  is gone) and pin the PySCF version that has it.

## Not in scope

Changing PySCF's grid defaults; GPU4PySCF (note in the report if it shares the
code); any change to Planck, whose gradient is already correct.
