# What does UKS double-hybrid geometry optimization and frequencies need?

Answered 2026-11-12. One shared gradient helper, the widened workflow gate, and
the existing workflow audit pointed at a UKS input. The gradient itself is
answered in `docs/UDH_GRADIENT_ONE_ARCHITECTURE.md`.

## The answer

The UKS global-double-hybrid gradient was already accepted (≤ 4.3e-9 Ha/Bohr
against fourth-order energy FD on the V4 grid). Two things kept geomopt and
frequencies out:

1. **The gate.** `validate_workflow_support` (`src/dft/driver.cpp`) admitted the
   UKS DH branch for `CalculationType::Gradient` only. It now admits GeomOpt,
   Frequency and GeomOptFrequency with the same conditions: no solvent,
   all-active conventional unshifted PT2 with stored T2, and the qualified grid
   (`validate_qualified_uks_dh_grid`).
2. **The per-geometry callback.** Widening the gate alone would have been wrong.
   `run_analytic_gradient_current_geometry`, which the optimizer and the
   semi-numerical Hessian call at every geometry, knew only the RKS DH
   gradient. The UKS assembly (UMP2 correction + KS gradient) lived inline in
   the `Gradient` case. It is now `compute_analytic_udh_gradient`, and both the
   `Gradient` case and the callback call it, so the optimizer and the Hessian
   see exactly the gradient the gradient gate pins.

The V4 grid policy lives on `calculator._dft` and `prepare_current_geometry`
rebuilds the grid from it at every geometry, so it does not fall back to
`legacy` under displacement.

## Validation

PySCF has no DH gradient, so validation is Planck against Planck, the standard
the RKS DH workflows were accepted on. `tests/dh_workflow_audit.py` now takes an
RKS or UKS input (its old assertion that UKS geomopt/freq are refused was
removed). On HO2/STO-3G B2PLYP, V4 grid
(`tests/inputs/exploratory/dh_gradient/ho2_uks_b2plyp_v4_workflow_sto3g.hfinp`):

| Check | Result | Limit |
|---|---|---|
| freq: callback Hessian vs 18 independent gradient runs | 2.8e-9 | 2e-6 |
| geomopt: energy FD vs gradient at the optimum | 2.2e-9 Ha/Bohr | 5e-8 |
| geomopt / freq: fresh energy and gradient at the final geometry | ≤ 1.4e-12 Eh, ≤ 1.4e-11 Ha/Bohr | 1e-8, 1e-7 |

Closed-shell control: water B2PLYP/STO-3G on the V4 grid gives the same
frequencies as RKS and as `scf_type uhf` to every printed digit (1389.12,
4956.05, 5372.67 cm⁻¹ at an unoptimized geometry).

Regression case `ho2_uks_b2plyp_v4_geomoptfreq_sto3g` (extended, ~6 min at 4
threads) pins E_opt −148.6568370557 Eh (6 L-BFGS steps) and the frequencies at
the optimum, 1331.76 / 1522.35 / 3567.63 cm⁻¹.

## Why HO2 and not OH

OH was the first fixture tried. It optimizes to the same answer with and without
symmetry (3684.44 cm⁻¹), but one SCF took 168 iterations. OH's β hole sits in a
degenerate π pair; rotating it about the bond costs no energy except grid error
(UKS stability λ_min −2.4e-8). DIIS has no curvature along that mode, so the
density drifts at fixed energy: the DIIS error reached 4e-15 while ΔP stayed at
1e-9..1e-8, and `tol_density 1e-11` was met only by chance. With
`use_symm .true.` (C∞v) the geomopt SCFs take 10 iterations, but the frequency
displacements still take 24–43, since a sideways displacement barely splits the
pair. This is the zero mode `hi_cation_uks_pbe_def2svp_ecp_stability` already
documents. It is a property of the molecule, not a convergence defect, so the
convergence gate was left alone: dropping the ΔP test at roundoff DIIS error
would change every HF and KS run, and the SA-CASSCF basin canary depends on the
HF iteration count. HO2 (Cs, no degenerate pair) converges in ~20 iterations at
the same tolerance.

## Cost

A UKS DH frequency run is 6N gradients. HO2: 267 s for the frequency run alone
at 4 threads; 351 s for geomopt + frequencies. The warm-start lever in
`docs/ANALYTIC_HESSIAN_SCOPE.md` applies here too.

## Not done

Imaginary-mode following (refused for DFT generally), solvent, frozen core, and
larger-basis qualification of the UKS gradient
(`docs/DH_GRADIENT_EXTENSIONS_SCOPE.md`). No RKS DH geomopt/freq regression case
exists either; RKS is gated by the same audit only.
