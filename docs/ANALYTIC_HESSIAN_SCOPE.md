# How should Planck make frequencies cheaper on large systems?

Scope, opened 2026-11-10. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 8).

## The state

Frequencies are semi-numerical: central differences of analytic gradients,
6N gradient evaluations (`src/freq/hessian.cpp`; the DFT driver does the
same). Each displaced point rebuilds the basis and resets the SCF state,
serially, one displacement after another. Since 2026-11-12 the HF SCF at
each displacement starts from the reference density (lever 1, below);
`planck-dft` starts from the previous displacement's density.

The question is "frequencies are too slow on large systems", and an analytic
Hessian is one answer, not the only one. Do the cheap ones first; they also
speed up the analytic path's CPHF later.

## Cheap levers, in order

1. **Warm start each displaced SCF. Done for HF, 2026-11-12; measured
   1.04×.** `compute_hessian` restores the reference α/β density before
   every displacement and calls `run_scf_gradient(c, true)`. Never chain
   from the previous displacement: +h and −h must start from the same
   density. The 2–4× expected here was wrong on both counts:
   - **Iterations:** 19–20 → 14 per displaced SCF (C1 cyclohexane/6-31G,
     18 atoms), not "a few". A 0.005 Bohr displacement leaves the start
     density ~1e-3 off, and DIIS still needs ~10 iterations to reach 1e-10.
   - **Share:** SCF iterations are 26 of 186 s there (14 %); the gradients
     are the rest. Wall time 186.2 → 179.3 s (1.04×). On the 12 small HF
     frequency regression cases, 12.8 → 11.2 s (1.14×).

   Frequencies are unchanged to every printed digit on all 12 cases and to
   1e-8 cm⁻¹ on cyclohexane. **So HF frequency cost is gradient cost;** only
   levers that remove or spread gradients (2, 3, an analytic Hessian) can
   move it.
2. **Symmetry-unique displacements.** Displace only symmetry-inequivalent
   atoms/directions and build the rest of the Hessian by the point-group
   operations (`symmetry/` already has the operations and
   `vibrational_symmetry`). Up to |G|× fewer gradients on symmetric
   molecules; nothing on C1.
3. **Displacements in parallel** across MPI ranks (embarrassingly parallel;
   the gradient code is MPI-aware). Not across OpenMP threads; each gradient
   is already threaded.

Lever 1 was measured as asked (above). The repo has no 20–40-atom input, so
the C1 fixture is cyclohexane with a fixed 0.05 Å distortion.

## The analytic RHF Hessian, if still needed

What it needs, none of which exists:

- **Second-derivative integrals**: overlap, kinetic, nuclear attraction
  (including the operator-centre terms) and ERI second derivatives. The OS
  derivative machinery (`_compute_eri_deriv_elem`) is first order only. This
  is the bulk of the work, and the ERI part is the expensive part.
- **3N CPHF solves** for the orbital response to each nuclear coordinate.
  `solve_rhf_cphf` and `build_rhf_cphf_matrix` (`rhf_response.h`) exist; the
  RHS (S^x, F^x in the MO basis) does not.
- Assembly: Hellmann-Feynman second derivatives + response + overlap terms.

Gate: PySCF `hessian.RHF` on water and C1 H2O2, every Hessian element, and
the frequencies of every RHF frequency regression case.

UHF, DFT (adds the XC second derivative and grid response) and PCM each
follow as their own step. The DFT case is the one users would want and the
largest.

## Recommendation

Lever 1 is done and buys little (1.04–1.14×) because gradients dominate. On
C1 molecules, lever 3 (MPI across displacements) is the only cheap lever
left. Decide between it and the analytic Hessian next.
