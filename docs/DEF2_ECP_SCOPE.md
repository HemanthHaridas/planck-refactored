# What does Planck need to run def2 basis sets, including the ECPs for heavy elements?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Why it matters

def2-SVP and def2-TZVP(P) are the default bases of ORCA and Turbomole and the
most common choice for routine DFT. `basis-sets/` has Pople and Dunning
families but no def2. For Rb onwards the def2 family uses effective core
potentials (ECPs), and Planck has no ECP support: no parser for ECP blocks,
no ECP integrals, no core-electron bookkeeping. Without ECPs, heavy elements
are all-electron only, with no scalar-relativistic treatment.

## Split by element

- **H–Kr: all-electron.** def2 for these needs only basis files. The GBS
  reader (`src/basis/gbs_parser.h`, `read_gbs`) already handles Gaussian94
  shells including fused SP. This is cheap and useful immediately for HF
  and post-HF (which run spherical today) and for DFT once spherical DFT
  lands (`docs/SPHERICAL_DFT_SCOPE.md`). def2 is defined for spherical
  functions.
- **Rb onwards: ECP.** Needs everything below.

## What ECPs require

1. **Parsing.** BSE's Gaussian94 output appends ECP blocks per element
   (core-electron count, then per angular-momentum channel a list of
   `(power r^n, exponent, coefficient)` terms). `read_gbs` stops at shells.
2. **Core-electron bookkeeping.** Removing $n_\text{core}$ electrons means
   the atom's charge, as seen by every physical term, is $Z - n_\text{core}$.
   `Molecule::nuclear_charge(a)` is already the single source for that (it
   zeroes ghost atoms), and the comment above it forbids reading
   `atomic_numbers[a]` for physical quantities. The audit of the direct
   `atomic_numbers[...]` reads is done (step 2): 46 reads, 16 physical
   ones switched to `nuclear_charge`, the rest element lookups that keep Z.
3. **ECP integrals.** A semilocal ECP is a local part plus angular-momentum
   projectors. The one-electron matrix $U_{\mu\nu}$ has type-1 (local) and
   type-2 (projector) parts, both needing radial quadrature or recurrence
   schemes. It is added to H_core, and its nuclear derivative is needed for
   gradients.
4. **SAD guess** needs atomic densities consistent with the ECP (fewer
   electrons per atom), or a fallback to the core-Hamiltonian guess for ECP
   atoms.
5. **Checkpoint restart.** Cross-basis Löwdin projection must check that
   source and target agree on each atom's core-electron count, or it projects
   onto the wrong occupied space.

Data model: an ECP channel is a list of `(n, ζ, c)` primitives, so it carries
a radial power that `Shell` does not. Store ECPs per atom beside the shells
rather than as a `Shell` variant.

What needs no change: SCF, MP2, CC, CASSCF/RASSCF, DFT, TDDFT and PCM consume
H_core and an electron count and do not care why H_core has a third term,
provided steps 2–4 are right. Population analyses will show fewer core
electrons on ECP atoms; that is expected, not a defect.

### Boundaries until each piece lands

Fail cleanly, never return an incomplete answer:

- A basis file with an ECP block parses, but the driver rejects the run until
  ECP integrals are in H_core.
- Gradient-consuming workflows (gradient, geomopt, frequency, imaginary
  follow) reject ECP input until step 5, with a boundary-marker regression
  case like `water_rmp2_spherical_gradient_rejected`.
- Full point-group direct SCF (`src/symmetry/{os,rys,hgp}_symm.cpp`) builds
  its own Fock skeleton and would ignore the ECP operator: reject ECP with
  `use_symm` there until it is wired in.

PySCF's `gto/ecp.py` (`type1_by_shell`, `type2_by_shell`) in the test venv is
a readable reference for the local/semilocal split if libecpint's output
ever needs debugging.

## The decision: library or native ECP integrals

| option | for | against |
|---|---|---|
| **libecpint** (C++) via `ExternalProject` | complete type-1/type-2 integrals and first derivatives; used by Psi4 and PySCF-adjacent tools | a new dependency (licence checked below: compatible) |
| **native** | no dependency | type-2 integrals are the hard part of ECP work (angular projection, numerical radial quadrature with screening); long validation tail |

Recommendation: libecpint, because the integrals and their derivatives are
the risky part and it has them (its README lists analytic first and second
derivatives). This supersedes the earlier `ECP_SUPPORT_SCOPE.md` plan for a
native `src/integrals/ecp.{h,cpp}` engine; that doc is merged here and
deleted.

### Licence findings (checked 2026-10-26; not legal advice)

- **libecpint: MIT** (Copyright 2021 Robert A. Shaw). It bundles
  Faddeeva (MIT, MIT 2012) and fetches pugixml (MIT); GoogleTest
  (BSD-3-Clause) is for its own tests only. All permissive: compatible with
  Planck's PolyForm Noncommercial licence. The only obligation is keeping
  the copyright and permission notices with source and binary copies.
- **Basis data:** def2 files and their ECP blocks would come from the Basis
  Set Exchange (BSD-3-Clause), like the existing files in `basis-sets/`.
  Keep the BSE header in each file; it carries the citation.

## Steps

1. **def2 basis files for H–Kr — LANDED** (`6e5f191a`). def2-SVP, -SVPD,
   -TZVP, -TZVPP and -QZVP from BSE, elements 1–36 (no ECP blocks there).
   RI fitting sets not added; add them when density fitting needs them.
   Five spherical RHF gates against PySCF 2.13.0, basis parsed from the same
   GBS files (`tests/pyscf/def2_rhf_references.py`):

   | case | Planck − PySCF |
   |---|---|
   | water/def2-SVP | 0 |
   | water/def2-QZVP (g shells) | 0 |
   | H2S/def2-TZVP | 0 |
   | SiH4/def2-SVP | 0 |
   | ZnCl2/def2-SVP | 1.2e-9 |

   Pinned at 1e-9 (ZnCl2 2e-9). They first landed at 4e-10..1.3e-8 and
   5e-8 tolerances; step 1b found why.
1b. **Where the missing digits went — ANSWERED: the Å→Bohr constant, not
   the integrals.** The first guess, f shells, did not fit: water/def2-SVP
   has no f functions and missed at 3.7e-9 while H2S/def2-TZVP, which has
   one, was at 1.5e-9. Nuclear repulsion settled it: water E_nn differed by
   6.6e-7 Eh (Planck 9.1895344241, PySCF 9.1895337629). `src/base/tables.h`
   paired `BOHR_TO_ANGSTROM = 0.529177210903` (CODATA 2018) with
   `ANGSTROM_TO_BOHR = 1.8897259886`, an older literal 7.2e-8 smaller than
   the reciprocal — exactly the relative E_nn gap — so every Ångström input
   ran at a geometry 7.2e-8 too small, and Å → Bohr → Å round trips drifted
   by the same amount. Confirmed before fixing: PySCF at Planck's Bohr
   coordinates reproduced four cases to every printed digit.
   **Fix:** `ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM`, and three copies
   of either literal (`types.h`, two RI tests) now use the constants.
   PySCF defaults to the CODATA 2010 Bohr (0.52917721092, 3e-11 away), so
   the reference script passes `unit=0.529177210903`.
   Ruled out on the way: SCF convergence (both codes unchanged at tighter
   tolerances; PySCF also at `direct_scf_tol` 1e-16). Primitive dropping
   and the integrals needed no check once the geometry closed the gap.
   **ZnCl2's 1.2e-9 is not the constant** — it was 1.2e-9 before the fix
   too, survives tight convergence in both codes, and is 4.5e-13 relative
   to a 2696 Eh energy. Unexplained, left at a 2e-9 pin.
   **Suite impact:** 26 extended cases pinned to Planck's own past output
   moved 2e-9..1.9e-7 Eh and were re-pinned. The one with a PySCF anchor in
   reach moved toward it: water SA-2 CASSCF went from 3.6e-8 to 1.1e-8 off
   PySCF's SAD-start minimum (-74.7877865139). Five MPI smoke failures
   were a stale `planck-mpi` binary, not a regression.
2. **Nuclear-charge audit — LANDED** (`2d3956e5`). 46 direct
   `atomic_numbers[...]` reads in `src/` (not 51). 16 physical — nuclear
   repulsion and its gradient (`gradient.cpp`, `mp2_gradient.cpp`), the
   nuclear-attraction derivative (`gradient.cpp`, `os.cpp`), gradient
   electron counts, multipole nuclear charges — now read `nuclear_charge()`
   / `total_nuclear_charge()`. 30 are element lookups (basis, RI basis,
   SAD), printing, IO/checkpoint, symmetry atom matching or counterpoise
   copying, and keep Z. No change without ghost atoms (core suite 85/85);
   with ghosts the gradient and multipole code had been counting the
   ghost's charge. SAD, PCM and counterpoise still read Z and are revisited
   with step 3's core count.
3. **ECP parsing and storage** on the basis object; `nuclear_charge` returns
   $Z - n_\text{core}$; electron count follows.
4. **ECP integrals in H_core** via libecpint. Gate: the ECP matrix
   against PySCF's `mol.intor('ECPscalar')` element by element on e.g. HI,
   AgCl, a Pd complex; then SCF energies against PySCF to 1e-9 Eh.
5. **ECP gradients.** Gate: FD and PySCF gradients; geometry optimization of
   a small heavy-element molecule against PySCF.
6. **DFT with ECPs** after spherical DFT lands: same gates in `planck-dft`.

## Not in this scope

Spin-orbit ECP terms, all-electron relativistic Hamiltonians (X2C, DKH).
