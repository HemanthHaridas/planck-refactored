# Planck Teaching Guide

This document is a teaching-oriented guide to the `planck-refactored` codebase.
It explains what the program does, how the major algorithms work, how the
source tree is organized, and how the quantum-chemistry theory maps onto the
implementation.

The intended audience is:

- a student learning Hartree-Fock and post-Hartree-Fock methods
- a contributor trying to understand where to change the code
- a researcher who wants to audit the mathematical and software structure

The guide is written in the same order that a calculation usually happens in
the executable:

1. parse the input
2. build the molecule and basis
3. generate shell pairs and integrals
4. solve SCF
5. optionally run correlation methods
6. optionally compute gradients, optimize geometry, or build vibrational data

## 1. What Planck Is

Planck is a compact quantum chemistry program centered on Gaussian-basis
electronic structure methods. At the moment it implements:

- RHF and UHF self-consistent field theory
- Obara-Saika one- and two-electron integrals
- conventional and direct SCF modes
- DIIS acceleration
- point-group detection and MO irrep labeling
- RMP2 and UMP2 correlation energies
- RHF and UHF analytic nuclear gradients
- geometry optimization in Cartesian and internal coordinates
- semi-numerical Hessians and vibrational frequencies
- CASSCF and RASSCF active-space calculations
- binary checkpoint restart support

There is also in-progress infrastructure for a true analytic RMP2 gradient:

- RHF response matrix construction
- MP2 amplitudes
- MP2 unrelaxed density builders
- MP2 Z-vector and relaxed-density helpers

Important implementation status:

- RHF and UHF gradients are analytic
- the current public RMP2 gradient path is still central-difference total-energy
  differentiation, even though the codebase now contains some response-theory
  building blocks for a future analytic implementation

## 2. Big-Picture Architecture

At a high level, the executable in [src/driver.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/driver.cpp)
orchestrates everything.

The main data object is `HartreeFock::Calculator` in
[src/base/types.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/base/types.h).
It holds:

- input options
- molecule and coordinates
- basis and shell information
- SCF state
- integral tensors and one-electron matrices
- post-HF energies and active-space results
- gradient, Hessian, and geometry optimization state

Conceptually, `Calculator` is the shared world-state for one calculation.

### Directory Map

### `src/base`

- `types.h`: central enums, options, molecule/basis containers, SCF state, DIIS
  state, checkpoint-related data, and the `Calculator` object
- `tables.h`: chemical or optimization helper tables
- `basis.h.in` and generated `basis.h`: compiled-in basis path configuration

### `src/io`

- `io.cpp/.h`: input parsing from `.hfinp`
- `checkpoint.cpp/.h`: binary checkpoint load/save and projection support
- `logging.h`: formatted program output

### `src/lookup`

- periodic-table and Boys-function support data

### `src/basis`

- Gaussian primitives, contractions, and `.gbs` basis parsing

### `src/integrals`

- shell-pair generation
- Obara-Saika recursions for overlap, kinetic, nuclear attraction, ERIs
- derivative integral kernels for gradients

### `src/scf`

- orthogonalization
- initial density construction
- RHF and UHF SCF loops

### `src/symmetry`

- point-group detection through `libmsym`
- symmetry-adapted orbital basis support
- MO irrep assignment

### `src/post_hf`

- MP2 energy code
- MP2 gradient-response support code
- CASSCF and RASSCF code
- AO to MO integral transformations
- RHF response matrix builders

### `src/gradient`

- analytic RHF and UHF nuclear gradients
- current numerical RMP2 gradient driver
- shared closed-shell AO derivative contraction helper

### `src/opt`

- geometry optimization in Cartesian and internal coordinates
- generalized internal coordinate construction and constraint handling

### `src/freq`

- finite-difference Hessian and vibrational analysis

### `tests`

- manifest-driven executable regression tests

## 3. Core Data Structures

The most important single file is
[src/base/types.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/base/types.h).

### `Molecule`

The molecule object stores:

- atomic numbers
- charge and multiplicity
- input-frame coordinates
- standard-orientation coordinates
- flags such as whether coordinates are already in Bohr

Two coordinate representations are important:

- `coordinates`: user-facing geometry, often in Angstrom
- `_standard` and `_coordinates`: internal Bohr-space geometry used by the
  integrals and optimization machinery

### `Basis`

The basis object contains:

- shells
- basis functions / contracted Cartesian components
- bookkeeping like number of shells and number of basis functions

### `DataSCF` and `SpinChannel`

The SCF state stores, per spin channel:

- density matrix
- Fock matrix
- MO energies
- MO coefficients
- optional MO symmetry labels

This is enough to reconstruct most later quantities.

### `DIISState`

The DIIS object stores:

- a history of Fock matrices
- a history of Pulay error matrices
- the subspace dimension

It then solves the standard augmented Pulay linear system to build an
extrapolated Fock matrix.

### `Calculator`

`Calculator` gathers everything together and also owns:

- `_overlap` and `_hcore`
- `_eri` if conventional SCF has stored the AO ERI tensor
- energies
- gradient and Hessian storage
- active-space results such as natural occupations
- SAO blocking data for symmetry-aware diagonalization

If you want to understand “where state lives,” it is almost always here.

## 4. Theoretical Foundations

This section summarizes the theory behind the implemented methods and explains
how the code mirrors that theory.

## 5. Gaussian Basis Functions

The code works in an atom-centered Gaussian basis:

\[
\chi_\mu(\mathbf r) = \sum_p d_{p\mu} \, (x-A_x)^{l_x}(y-A_y)^{l_y}(z-A_z)^{l_z}
e^{-\alpha_p |\mathbf r-\mathbf A|^2}
\]

Why Gaussians?

- products of Gaussians on different centers are again Gaussians
- integrals can be reduced recursively
- this makes them practical for Hartree-Fock and post-HF methods

The implementation details are split across:

- [src/basis/basis.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/basis/basis.cpp)
- [src/basis/gaussian.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/basis/gaussian.cpp)
- [src/integrals/shellpair.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/integrals/shellpair.h)
- [src/integrals/shellpair.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/integrals/shellpair.cpp)

`ShellPair` precomputes Gaussian-product data such as:

- combined exponent
- prefactors
- Gaussian product center
- displacement vectors needed in recurrence relations

That is a classic quantum-chemistry optimization: precompute shell-pair data
once and reuse it many times.

## 6. One- and Two-Electron Integrals

Electronic structure in a Gaussian basis depends on several integral classes:

- overlap:
  \[
  S_{\mu\nu} = \langle \chi_\mu | \chi_\nu \rangle
  \]
- kinetic:
  \[
  T_{\mu\nu} = \left\langle \chi_\mu \left| -\frac{1}{2}\nabla^2 \right| \chi_\nu \right\rangle
  \]
- nuclear attraction:
  \[
  V_{\mu\nu} = \sum_A \left\langle \chi_\mu \left| -\frac{Z_A}{r_A} \right| \chi_\nu \right\rangle
  \]
- electron repulsion:
  \[
  (\mu\nu|\lambda\sigma) =
  \iint \chi_\mu(1)\chi_\nu(1)\frac{1}{r_{12}}\chi_\lambda(2)\chi_\sigma(2)\, d1\, d2
  \]

Planck computes these with Obara-Saika recursions in:

- [src/integrals/os.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/integrals/os.cpp)
- [src/integrals/os.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/integrals/os.h)

The important implementation split is:

- `_compute_1e(...)`: overlap and kinetic
- `_compute_nuclear_attraction(...)`: nuclear attraction
- `_compute_2e(...)`: full AO ERI tensor
- `_compute_fock_rhf(...)` and `_compute_fock_uhf(...)`: ERI contraction into
  Coulomb and exchange contributions

### Conventional vs Direct SCF

There are two major strategies for ERIs:

- conventional SCF:
  build the full AO ERI tensor once and reuse it
- direct SCF:
  recompute or contract on the fly each iteration

This is controlled in [src/scf/scf.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/scf/scf.cpp).

Tradeoff:

- conventional is faster per SCF iteration
- direct uses less memory

The code also uses Schwarz screening thresholds to skip negligible quartets.

## 7. Hartree-Fock Theory

Hartree-Fock approximates the many-electron wavefunction as a Slater
determinant built from molecular orbitals.

Each spatial orbital is expanded in the AO basis:

\[
\phi_i = \sum_\mu C_{\mu i}\chi_\mu
\]

The RHF variational equations become the Roothaan equations:

\[
\mathbf F \mathbf C = \mathbf S \mathbf C \boldsymbol \varepsilon
\]

where:

- `S` is the overlap matrix
- `F` is the Fock matrix
- `C` contains molecular orbital coefficients
- `ε` are orbital energies

For RHF, the closed-shell density is:

\[
P_{\mu\nu} = 2\sum_{i \in occ} C_{\mu i} C_{\nu i}
\]

and the Fock matrix is:

\[
F_{\mu\nu} = H_{\mu\nu}^{core} + G_{\mu\nu}
\]

with:

\[
G_{\mu\nu} = \sum_{\lambda\sigma} P_{\lambda\sigma}
\left[(\mu\nu|\lambda\sigma) - \frac{1}{2}(\mu\lambda|\nu\sigma)\right]
\]

For UHF, alpha and beta densities are separate, and the exchange is
spin-dependent.

### Where It Lives in Code

- [src/scf/scf.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/scf/scf.h)
- [src/scf/scf.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/scf/scf.cpp)

Important functions:

- `build_orthogonalizer`: constructs \(S^{-1/2}\)
- `initial_density`: core-Hamiltonian guess
- `run_rhf`
- `run_uhf`

### SCF Cycle

Each iteration does roughly this:

1. build `G`
2. form `F = H + G`
3. compute energy
4. optionally update DIIS
5. diagonalize in an orthonormal basis
6. rebuild density
7. test convergence

The orthogonalization step uses:

\[
\mathbf X = \mathbf S^{-1/2}
\]

and diagonalizes:

\[
\mathbf F' = \mathbf X^T \mathbf F \mathbf X
\]

then transforms back:

\[
\mathbf C = \mathbf X \mathbf C'
\]

## 8. DIIS Convergence Acceleration

Planck uses Pulay DIIS to accelerate SCF convergence.

The key idea is:

- store several recent Fock matrices
- store the corresponding error matrices
- solve for a linear combination of Fock matrices that minimizes the residual

The residual is the commutator in the orthonormal basis:

\[
\mathbf e = \mathbf X^T(\mathbf F \mathbf P \mathbf S - \mathbf S \mathbf P \mathbf F)\mathbf X
\]

When the SCF is converged, this should go to zero.

This is implemented in the `DIISState` object in
[src/base/types.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/base/types.h).

## 9. Symmetry

Planck can:

- detect molecular point groups
- rotate the molecule to a standard frame
- construct symmetry-adapted AO block structure
- label converged MOs by irreducible representation

This logic lives in:

- [src/symmetry/symmetry.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/symmetry/symmetry.cpp)
- [src/symmetry/mo_symmetry.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/symmetry/mo_symmetry.cpp)

The code uses `libmsym` for group theory tasks.

The main implementation point is not “using symmetry to derive the HF
equations,” but “using symmetry to block orbital spaces and label results in a
chemically meaningful way.”

## 10. MP2 Correlation Energy

Second-order Moller-Plesset perturbation theory adds a perturbative correction
to the Hartree-Fock energy.

For closed-shell RHF, the standard spatial-orbital expression is:

\[
E_{MP2} =
\sum_{ijab}
\frac{(ia|jb)\left[2(ia|jb) - (ib|ja)\right]}
{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
\]

Indices:

- `i, j`: occupied orbitals
- `a, b`: virtual orbitals

For UMP2, same-spin and opposite-spin channels are treated separately.

### Where It Lives

- [src/post_hf/mp2.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/mp2.cpp)
- [src/post_hf/mp2.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/mp2.h)
- [src/post_hf/integrals.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/integrals.cpp)
- [src/post_hf/integrals.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/integrals.h)

The post-HF integral helper code does AO to MO tensor transformations. That is
needed because MP2 is naturally expressed in MO-space two-electron integrals,
while SCF and integrals are generated in AO space.

### MP2 Gradient Infrastructure

Files:

- [src/post_hf/mp2_gradient.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/mp2_gradient.cpp)
- [src/post_hf/mp2_gradient.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/mp2_gradient.h)
- [src/post_hf/rhf_response.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/rhf_response.cpp)
- [src/post_hf/rhf_response.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/rhf_response.h)

This code now contains:

- RMP2 amplitudes
- unrelaxed MP2 one-particle density pieces
- RHF response matrix construction
- MP2 Z-vector source terms
- MP2 relaxed-density helpers

But the production `compute_rmp2_gradient` entry point still uses numerical
central differences, not the fully assembled analytic MP2 Lagrangian.

That distinction is important for both science and software maintenance.

## 11. Active-Space Methods: CASSCF and RASSCF

CASSCF combines:

- a CI problem in an active orbital space
- orbital optimization outside and inside that space

The core idea is that the wavefunction is no longer a single determinant:

\[
|\Psi\rangle = \sum_I c_I |D_I\rangle
\]

where the determinants `|D_I>` span the active space.

The energy is optimized with respect to:

- CI coefficients
- orbital rotations

Planck’s implementation is in:

- [src/post_hf/casscf.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/casscf.cpp)
- [src/post_hf/casscf.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/casscf.h)

Main ingredients:

- active-space selection
- determinant generation
- exact CI diagonalization in the active space
- 1-RDM and 2-RDM construction
- orbital gradient evaluation
- macro-iterations for orbital optimization

The recent fixes in this area centered on:

- making the initial CI energy variational relative to RHF
- repairing density reconstruction
- stabilizing orbital updates

From a teaching perspective, the most important concept is:

- HF optimizes a single determinant
- CASSCF optimizes both multiconfigurational CI coefficients and orbitals

## 12. Analytic Nuclear Gradients

For RHF and UHF, Planck computes analytic nuclear gradients.

The RHF gradient has several contributions:

1. one-electron derivative terms
2. nuclear-attraction center derivatives
3. two-electron ERI derivative terms
4. Pulay overlap terms
5. nuclear repulsion derivative

The code explains this directly in
[src/gradient/gradient.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/gradient/gradient.cpp).

Symbolically:

\[
\frac{dE}{dR_A}
=
\sum_{\mu\nu} P_{\mu\nu}\frac{dh_{\mu\nu}}{dR_A}
+
\frac{1}{2}\sum_{\mu\nu\lambda\sigma}\Gamma_{\mu\nu\lambda\sigma}
\frac{d(\mu\nu|\lambda\sigma)}{dR_A}
 E_{Pulay}
 \frac{dE_{nuc}}{dR_A}
\]

The Pulay term is needed because the AO basis itself depends on nuclear
positions. In a finite atom-centered basis, moving nuclei changes the basis
functions, so there are extra overlap-related contributions even when the
orbital coefficients are variationally optimized.

### Where It Lives

- [src/gradient/gradient.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/gradient/gradient.cpp)
- [src/integrals/os.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/integrals/os.cpp)

Recent refactoring added a reusable closed-shell derivative assembly helper that
takes:

- density matrix `P`
- energy-weighted density `W`
- a generic two-particle contraction function `Gamma`

That was done specifically to make future correlated gradients easier to plug
in without duplicating the entire RHF derivative engine.

### UHF Gradient

The UHF gradient is similar but uses separate alpha and beta densities and the
appropriate spin-resolved two-particle structure.

### Current RMP2 Gradient Status

The current public `RMP2` gradient path:

- recomputes the total MP2 energy at slightly displaced geometries
- forms a central difference

This is numerically useful but not a true analytic gradient.

## 13. Geometry Optimization

Planck supports geometry optimization in:

- Cartesian coordinates
- internal coordinates

Files:

- [src/opt/geomopt.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/opt/geomopt.cpp)
- [src/opt/intcoords.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/opt/intcoords.cpp)

### Cartesian Optimization

Uses an L-BFGS strategy on the flattened Cartesian coordinate vector.

Why L-BFGS?

- full BFGS stores an approximate Hessian explicitly
- that becomes expensive for larger systems
- L-BFGS stores only recent update history

### Internal-Coordinate Optimization

Uses generalized internal coordinates:

- bond stretches
- angle bends
- torsions

Why internal coordinates?

- they align better with chemically relevant motions
- optimization often converges in fewer steps
- constraints are much easier to impose naturally

The internal-coordinate machinery uses:

- a Wilson B matrix
- generalized inverse / back-transformation steps
- Schlegel-style microiterations

## 14. Vibrational Frequencies and Hessians

Files:

- [src/freq/hessian.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/freq/hessian.cpp)
- [src/freq/hessian.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/freq/hessian.h)

The Hessian is obtained by finite differences of analytic gradients.

That is a common compromise:

- gradients are much cheaper and more reliable than total-energy Hessians
- full analytic Hessians are substantially more complicated

Workflow:

1. displace each Cartesian coordinate positively and negatively
2. compute analytic gradients
3. build the Hessian by central differences
4. mass-weight
5. project translations and rotations
6. diagonalize
7. convert eigenvalues into frequencies

This yields:

- vibrational frequencies
- normal modes
- zero-point energy

## 15. Checkpointing and Restart

Files:

- [src/io/checkpoint.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/io/checkpoint.cpp)
- [src/io/checkpoint.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/io/checkpoint.h)

The checkpoint system stores:

- density data
- geometry
- basis-related data
- optimization metadata

It supports:

- density restart in the same basis
- full geometry + density restart
- cross-basis projection using overlap-based techniques

Pedagogically, this is a good example of how practical electronic structure
codes reduce wall time by reusing chemically meaningful state.

## 16. Execution Flow of a Typical Run

The main calculation path in
[src/driver.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/driver.cpp)
looks like this:

1. parse the input file
2. construct `Calculator`
3. convert coordinates to internal Bohr form
4. optionally restore checkpoint geometry
5. detect symmetry and standard orientation
6. read the basis set
7. initialize matrices and options
8. build shell pairs
9. compute one-electron integrals
10. run SCF
11. optionally run MP2 or CASSCF/RASSCF
12. optionally compute gradient
13. optionally run geometry optimization
14. optionally run frequency analysis
15. optionally write checkpoint

This ordering is why `driver.cpp` is the best “map file” for new contributors.

## 17. Teaching Map: How Theory Matches Functions

Use this section as a quick lookup table.

| Theory topic | Main implementation files |
|---|---|
| molecule/options/state | [src/base/types.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/base/types.h) |
| input parsing | [src/io/io.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/io/io.cpp) |
| basis parsing | [src/basis/basis.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/basis/basis.cpp) |
| shell-pair construction | [src/integrals/shellpair.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/integrals/shellpair.cpp) |
| one-electron integrals | [src/integrals/os.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/integrals/os.cpp) |
| ERIs and Fock builds | [src/integrals/os.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/integrals/os.cpp) |
| RHF/UHF SCF | [src/scf/scf.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/scf/scf.cpp) |
| symmetry and MO labels | [src/symmetry/symmetry.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/symmetry/symmetry.cpp), [src/symmetry/mo_symmetry.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/symmetry/mo_symmetry.cpp) |
| MP2 energy | [src/post_hf/mp2.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/mp2.cpp) |
| MP2 response groundwork | [src/post_hf/mp2_gradient.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/mp2_gradient.cpp), [src/post_hf/rhf_response.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/rhf_response.cpp) |
| CASSCF/RASSCF | [src/post_hf/casscf.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/casscf.cpp) |
| analytic HF gradients | [src/gradient/gradient.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/gradient/gradient.cpp) |
| geometry optimization | [src/opt/geomopt.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/opt/geomopt.cpp) |
| internal coordinates | [src/opt/intcoords.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/opt/intcoords.cpp) |
| vibrational analysis | [src/freq/hessian.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/freq/hessian.cpp) |
| checkpointing | [src/io/checkpoint.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/io/checkpoint.cpp) |

## 18. Limitations and Honest Status Notes

This is the section a teacher or contributor should read before presenting the
program as “finished.”

- RHF and UHF are mature enough for small teaching calculations.
- MP2 energies are implemented for RHF and UHF.
- RHF and UHF analytic gradients are implemented.
- The public RMP2 gradient path is still numerical, not fully analytic.
- The code contains partial response-theory infrastructure for future analytic
  RMP2 gradients, but the final Lagrangian assembly is not yet wired into the
  production gradient entry point.
- CASSCF and RASSCF are present and the recently fixed cases now behave
  variationally in the tested inputs.
- Hessians are finite-difference of gradients, not fully analytic second
  derivatives.

## 19. How to Study This Codebase Efficiently

Recommended reading order:

1. [src/base/types.h](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/base/types.h)
2. [src/driver.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/driver.cpp)
3. [src/io/io.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/io/io.cpp)
4. [src/scf/scf.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/scf/scf.cpp)
5. [src/integrals/os.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/integrals/os.cpp)
6. [src/gradient/gradient.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/gradient/gradient.cpp)
7. [src/post_hf/mp2.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/mp2.cpp)
8. [src/post_hf/casscf.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/post_hf/casscf.cpp)
9. [src/opt/geomopt.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/opt/geomopt.cpp)
10. [src/freq/hessian.cpp](/Users/hemanthharidas/Desktop/codes/planck-refactored/src/freq/hessian.cpp)

This order follows the dependency graph from basic state and control flow to
methods and then to downstream workflows.

## 20. Summary

Planck is a compact but nontrivial electronic-structure code. It is useful for
teaching because the theory-to-code mapping is still visible:

- basis functions become shell and primitive objects
- integrals become explicit recursive kernels
- SCF becomes a density-Fock fixed-point iteration
- DIIS becomes a small linear algebra problem on stored residuals
- MP2 becomes AO-to-MO integral transformation plus perturbative energy sums
- CASSCF becomes CI + orbital optimization + reduced density matrices
- gradients become derivative integral contractions plus Pulay terms
- geometry optimization becomes an iterative minimization around those
  gradients

That transparency is the main educational value of the codebase.
