# How should Planck make frequencies cheaper on large systems?

Scope, opened 2026-11-10. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 12).

## The state

Frequencies are semi-numerical: central differences of analytic gradients,
6N gradient evaluations (`src/freq/hessian.cpp`; the DFT driver does the
same). Each displaced point rebuilds the basis, **resets the SCF state**
(`calc._info._scf = DataSCF(...)`) and runs the SCF from the default guess,
serially, one displacement after another.

The question is "frequencies are too slow on large systems", and an analytic
Hessian is one answer, not the only one. Do the cheap ones first; they also
speed up the analytic path's CPHF later.

## Cheap levers, in order

1. **Warm start each displaced SCF** from the reference density (or the
   previous displacement's). A 0.005 Bohr displacement converges in a few
   iterations from the reference density instead of ~15–20 from `hcore`.
   Expected 2–4× on the whole frequency run. Gate: identical frequencies
   (to the existing tolerances) on every frequency regression case, with the
   iteration counts logged.
2. **Symmetry-unique displacements.** Displace only symmetry-inequivalent
   atoms/directions and build the rest of the Hessian by the point-group
   operations (`symmetry/` already has the operations and
   `vibrational_symmetry`). Up to |G|× fewer gradients on symmetric
   molecules; nothing on C1.
3. **Displacements in parallel** across MPI ranks (embarrassingly parallel;
   the gradient code is MPI-aware). Not across OpenMP threads; each gradient
   is already threaded.

Measure 1 before building 2 or 3, on a 20–40 atom C1 molecule, and write the
number down.

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

Do lever 1 now; it is small and helps every frequency run. Decide on the
analytic Hessian with the lever-1 numbers in hand.
