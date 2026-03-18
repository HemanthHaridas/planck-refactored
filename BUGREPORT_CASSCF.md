# CASSCF Bug Report

## Summary

The RHF-based CASSCF implementation behaved correctly for `H2 CAS(2,2)` but failed for water active spaces:

- `H2O CAS(2,2)` did not converge within the configured macro-iterations.
- `H2O CAS(4,4)` could start from a non-variational CI energy above RHF in the broken path, and the orbital optimization could diverge or stall.

The expected behavior for a ground-state CASCI/CASSCF calculation built from RHF orbitals is that the initial CI energy is less than or equal to the RHF energy. In addition, the macro-iteration should either reduce the orbital residual or terminate cleanly once no improving orbital step exists.

## Affected Inputs

Reproduced with:

- `/Users/hemanthharidas/Desktop/codes/planck-refactored/inputs/h2_cas22.hfinp`
- `/Users/hemanthharidas/Desktop/codes/planck-refactored/inputs/water_cas22.hfinp`
- `/Users/hemanthharidas/Desktop/codes/planck-refactored/inputs/water_cas44_symm.hfinp`

## Symptoms Before Fix

### 1. Non-variational CI/CASSCF energy for larger active spaces

The original water runs could produce an initial CASSCF/CASCI energy that was inconsistent with the RHF reference, which is a clear sign that the CI eigenproblem and the density/energy reconstruction were not using a fully consistent formulation.

### 2. Orbital gradient did not reduce reliably

For water, the macro-iterations often showed one of two bad behaviors:

- stagnation with the energy nearly unchanged but the reported orbital gradient stuck at a non-negligible value
- unstable rotations that caused the energy and residual to grow dramatically

### 3. Missing search directions in orbital optimization

The inactive-virtual block was explicitly zeroed in the orbital gradient, even though those rotations are non-redundant in CASSCF. That removed physically relevant relaxation directions from the optimizer.

## Root Causes

### 1. Fragile CI density reconstruction

The original 1-RDM and 2-RDM builders used hand-coded case logic over determinant differences. That implementation was too brittle for larger active spaces and could make the CI eigenvalue, density matrices, and reconstructed energy inconsistent with one another.

### 2. Inconsistent energy evaluation path

The macro-iteration energy was reconstructed from the density matrices instead of being taken directly from the solved CI roots. When the RDM path is even slightly inconsistent, this can break the variational property and produce misleading macro-iteration energies.

### 3. Incorrect orbital-gradient masking

The orbital gradient code zeroed inactive-virtual rotations. Those are non-redundant CASSCF orbital rotations and must remain available to the optimizer.

### 4. No energy-aware macro-step control

The orbital update accepted large preconditioned rotations without checking whether they improved the CASSCF energy. That allowed overshooting and divergence.

## Fix Implemented

### 1. Exact determinant-based 1-RDM and 2-RDM evaluation

The CI density matrices are now constructed by applying creation/annihilation operators directly in the spin-orbital determinant basis with an exact determinant lookup. This removes the fragile case-by-case determinant-difference logic.

### 2. Energy taken directly from CI roots

The active-space contribution is now evaluated from the solved CI eigenvalues and state-averaging weights instead of relying on a separate back-contracted RDM energy path.

### 3. Restored inactive-virtual orbital rotations

The inactive-virtual block is no longer zeroed in the orbital gradient, so the macro-optimizer has access to the full set of non-redundant orbital rotations.

### 4. Added energy-aware backtracking and stagnation handling

The orbital macro-step now:

- tries scaled orbital rotations
- keeps only steps that improve or at least do not worsen the CASSCF energy
- resets DIIS when no acceptable extrapolated step exists
- terminates cleanly when the energy is stationary and no improving orbital step can be found

## Validation After Fix

Validated with:

```bash
./build/hartree-fock inputs/h2_cas22.hfinp
./build/hartree-fock inputs/water_cas22.hfinp
./build/hartree-fock inputs/water_cas44_symm.hfinp
```

Observed results:

- `H2 CAS(2,2)` converged with `E(CASSCF) = -1.1372744062 Eh`
- `H2O CAS(2,2)` converged with `E(CASSCF) = -74.9641865744 Eh`
- `H2O CAS(4,4)` converged with `E(CASSCF) = -75.9851092026 Eh`

For both water cases, the initial and final CASSCF energies are below the RHF reference, restoring the expected variational behavior.

## Remaining Note

The reported orbital-gradient residual is still not a fully trustworthy stationary indicator for the larger water cases. The current fix makes the implementation robust and variational again by using exact CI densities plus energy-aware macro-step acceptance, but the closed-form orbital-gradient expression would still benefit from a dedicated follow-up derivation and verification against a reference implementation.

## Files Involved

- `/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/casscf.cpp`
- `/Users/hemanthharidas/Desktop/codes/planck-refactored/BUGREPORT_CASSCF.md`
