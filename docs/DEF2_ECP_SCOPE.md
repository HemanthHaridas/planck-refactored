# What does Planck need to run def2 basis sets, including the ECPs for heavy elements?

Research scope, opened 2026-10-26. Steps 1–4b have landed: def2 runs for H–Rn
in `hartree-fock` single points. Steps 5 (ECP gradients) and 6 (DFT with ECPs)
are open, so this is still a scope. Rewrite it as the answer when they land.
Status lives in `vault/Status/Open Work.md`.

## Why it matters

def2-SVP and def2-TZVP(P) are the default bases of ORCA and Turbomole and the
most common choice for routine DFT. From Rb onward the def2 family replaces
the core electrons with effective core potentials (ECPs), which also carry
scalar-relativistic effects. Without ECP support, heavy elements in Planck
were all-electron only, with no relativistic treatment.

## Split by element

- **H–Kr: all-electron.** Only basis files were needed (step 1). def2 is
  defined for spherical functions, so this serves HF and post-HF today and DFT
  once spherical DFT lands (`docs/SPHERICAL_DFT_SCOPE.md`).
- **Rb onward: ECP.** Needs everything below.

## What ECPs require, and where each piece stands

1. **Parsing — done (step 3).** BSE's Gaussian94 output appends an ECP block
   per element: core-electron count, then for each angular-momentum channel a
   list of `(n, exponent, coefficient)` terms, with power r^(n−2). `read_gbs`
   returns them, and `read_gbs_basis` stores one `ECPPotential` per non-ghost
   atom on `Basis::_ecps`.
2. **Core-electron bookkeeping — done (steps 2, 3).** Every physical term sees
   a charge of Z − n_core. `Molecule::nuclear_charge(a)` is the single source
   (it also zeroes ghost atoms), and `Calculator::sync_ecp_core_electrons()`
   feeds it the core counts. Element lookups keep reading Z.
3. **ECP integrals — done for energies (step 4); derivatives open (step 5).**
   A semilocal ECP is a local part plus angular-momentum projectors, giving
   type-1 (local) and type-2 (projector) integrals. libecpint computes both,
   and `_compute_potential` adds U to H_core. The nuclear derivative is needed
   for gradients.
4. **SAD guess — done (step 4).** The atomic UHF carries the element's ECP and
   runs with Z − n_core electrons.
5. **Checkpoint restart and FCIDUMP — done (step 4b).** Checkpoint version 8
   stores the core counts and refuses to mix cores; FCIDUMP writes the
   valence-only Hamiltonian, checked against PySCF.

Data model: an ECP channel carries a radial power that `Shell` does not, so
ECPs are stored per atom beside the shells (`ECPPotential`, `src/base/types.h`)
rather than as a `Shell` variant.

What needed no change: SCF, MP2, CC, CASSCF/RASSCF and PCM consume H_core and
an electron count, and do not care why H_core has a third term. Population
analyses show fewer core electrons on ECP atoms; that is expected, not a
defect. Open-shell and post-HF energies with ECPs take this path but are not
gated yet.

### Boundaries until each piece lands

Fail cleanly, never return an incomplete answer:

- Gradient-consuming workflows (anything but a single point) refuse an ECP
  basis until step 5, gated by `hi_rhf_def2svp_ecp_gradient_rejected`.
- `planck-dft` refuses an ECP basis until step 6
  (`hi_rks_def2svp_ecp_rejected`). Its H_core already carries the ECP.
- Full point-group symmetry needs no gate: the symmetry skeleton covers the
  two-electron Fock build only, and H_core (with the ECP) is built in full.
  `hi_rhf_def2tzvp_ecp_sad_symm` runs with `use_symm .true.`.

For debugging libecpint, compare against PySCF (`mol.intor('ECPscalar')`) and,
when the two disagree, against a brute-force radial × Lebedev integral around
the ECP centre. That third answer is what located the bug in step 4.

## The decision: library or native ECP integrals — libecpint, taken

| option | for | against |
|---|---|---|
| **libecpint** (C++) via `ExternalProject` | complete type-1/type-2 integrals and first and second derivatives; used by Psi4 | a new dependency |
| **native** | no dependency | type-2 integrals are the hard part of ECP work (angular projection, radial quadrature with screening); long validation tail |

libecpint was chosen because the integrals and their derivatives are the risky
part and it has them. This superseded the earlier `ECP_SUPPORT_SCOPE.md` plan
for a native engine (that doc was merged here and deleted). The cost turned
out to be one upstream bug: its generator wrote kernel constants at 6 digits,
patched at fetch time (step 4).

### Licence (checked 2026-10-26; not legal advice)

- **libecpint: MIT** (Copyright 2021 Robert A. Shaw). It bundles Faddeeva
  (MIT); pugixml (MIT) is disabled in Planck's build; GoogleTest is for its
  own tests only. Compatible with Planck's PolyForm Noncommercial licence.
  The obligation is keeping the notice with source and binary copies: the
  fetched source carries it, and `CMakeLists.txt` installs it with the
  binaries as `share/doc/planck/libecpint-LICENSE`.
- **Basis data:** the def2 files and their ECP blocks come from the Basis Set
  Exchange (BSD-3-Clause), like the rest of `basis-sets/`. Each file keeps its
  BSE header, which carries the citation.

## Steps

1. **def2 basis files for H–Kr — LANDED** (`6e5f191a`). def2-SVP, -SVPD,
   -TZVP, -TZVPP and -QZVP from BSE (first H–Kr; extended to H–Rn with the
   ECP blocks at step 3).
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
   ghost's charge. SAD, PCM and counterpoise were fixed at step 4.
3. **ECP parsing and storage — LANDED** (`99f142d1`). The def2 files now cover H–Rn
   (50 ECP elements, Rb onward). `detail::read_gbs` returns a `GbsFile`
   (shells + ECP blocks); `read_gbs_basis` stores one `ECPPotential` per
   non-ghost atom on `Basis::_ecps` — center, core count, local channel
   U_L and semilocal U_l − U_L channels, with each term's `n` kept as
   written (r^(n−2), Gaussian94 convention; convert at step 4).
   `Calculator::sync_ecp_core_electrons()` copies the core counts to
   `Molecule::ecp_core_electrons` after the primary basis read in both
   drivers, so `nuclear_charge()` returns Z − n_core and every electron
   count (all of which go through `total_nuclear_charge()`) follows.
   Two traps found on the way:
   - **BSE writes the ECP section's headers uppercase** (`RB 0`,
     `RB-ECP`) while the shells use `Rb`; the case-sensitive element
     lookup failed every def2 run. The reader now normalizes symbols.
   - **SAD had a private copy of the GBS reader**, so it still failed on
     `RB` after the fix. The copy is deleted (−123 lines); SAD uses the
     shared reader, which is now the only one.
   The parsed counts ("I: 28 core electrons, Z_eff 25, L 3, terms
   4/7/8/10") are printed at startup and pinned by
   `hi_rhf_def2svp_ecp_gradient_rejected`.
4. **ECP integrals in H_core — LANDED** (`749c8355`). libecpint (MIT) is an
   `ExternalProject` pinned past v1.0.7 for upstream fixes #71/#72, built
   with Planck's C++ compiler (its API passes `std::vector`, so a libc++
   build would not link against libstdc++), and linked into every target,
   since `src/integrals/ecp.cpp` sits in the shared integral sources.
   `HartreeFock::ECP::_compute_ecp` builds the Cartesian U matrix:
   libecpint takes `n` as written in the file and treats the largest-l
   channel as local, so U_L goes in at l = L; coefficients are passed as
   (c_p Nc) N_p and `_component_norm` applied per element (same xx, xy, xz,
   ... order). `_compute_potential` (`src/integrals/base.h`) returns
   V_ne + U and replaces the nuclear-attraction call at every H_core site
   (HF driver both paths, `working_state`, DFT driver, counterpoise), so
   geomopt rebuilds and restarts inherit it. Also fixed: SAD attaches each
   element's ECP to its atomic basis and runs the atom with Z − n_core
   electrons; PCM's nuclear potential and counterpoise fragments read
   `nuclear_charge()` / the core counts.

   **libecpint's generated kernels carried 6-digit constants.** The first
   matrix check missed PySCF by up to 1.9e-7, on H–H elements only.
   Localised in five eliminations: a brute-force radial × Lebedev integral
   agreed with PySCF to 12 digits (so PySCF was right); libecpint's local
   and d channels matched, its s and p projectors did not; the error was
   blind to grid size and threshold; its Bessel table and Gauss–Chebyshev
   quadrature were exact to ~1e-14; and against an exact 1-D integral every
   off-centre type-2 integral was off by the same 1 + 2.09e-6 for every
   exponent and power of r (on-centre was exact). That is 157.914 /
   (16π²): the generator writes the angular prefactors into `Q*.cpp` at
   default ostream precision. `cmake/patch_libecpint.cmake` sets precision
   17 at fetch time. Worth reporting upstream.

   Gates (PySCF 2.13.0, basis and ECP both from Planck's GBS,
   `tests/pyscf/def2_ecp_references.py`, which converts the Gaussian94 ECP
   block PySCF's reader skips; run with a dump directory it also compares
   Planck's Cartesian V_ne + U element by element):

   | case | ECP | Planck − PySCF | max matrix diff |
   |---|---|---|---|
   | HI/def2-SVP (smoke) | I, 28e | 0 | 1.6e-11 |
   | HI/def2-TZVP, SAD, `use_symm` | I, 28e, f shells | 0 | 6.3e-11 |
   | AgCl/def2-SVP | Ag, 28e | 1.1e-9 | 1.3e-9 |
   | PdCO/def2-SVP, SAD | Pd, 28e | 5e-10 | 4.2e-10 |
   | AuH·H2O/def2-SVP, C1 | Au, 60e | 2e-10 | 9.2e-10 |

   Pinned at 1e-9, AgCl at 2e-9: like ZnCl2 in step 1b it is the case with
   Cl, ~4e-13 relative, unexplained. Not done: cross-basis checkpoint
   restart does not check that both bases agree on each atom's core count
   (only a guess is affected, but an ECP/all-electron mismatch projects
   onto the wrong occupied count).
4b. **Checkpointing and FCIDUMP with ECPs — LANDED.** Step 4 changed what
   H_core and the electron count mean on ECP atoms; this step checked every
   consumer of those that writes or reads them back.

   - **FCIDUMP needed no change.** `src/io/fcidump.cpp` builds h_pq from
     `calc._hcore` (so U_ECP is in it), NELEC from `total_nuclear_charge()`
     and ECORE from `_nuclear_repulsion`, both Z − n_core. The file is a
     consistent valence-only Hamiltonian. No note about the removed core is
     written: the format has no standard field for it, and nothing a reader
     computes from the file depends on it. PySCF reading the HI file
     reproduces its own ECP RHF energy to 2e-12, both from the reference
     determinant and from an RHF solved on the file
     (`def2_ecp_references.py --fcidump`).
   - **Checkpoint version 8 stores each atom's ECP core count**, right after
     its atomic number (`src/io/checkpoint.cpp`, `tools/chkdump.cpp`).
     Version 7 and older files read as all zero, which is exact: ECP runs
     were refused before version 8 existed. A same-basis load whose core
     counts differ from the current basis is refused, and cross-basis
     projection between different cores (an ECP basis and an all-electron
     one, or two core sizes) falls back to the core guess. `guess full`
     needed nothing: the core counts are re-synced from the re-read basis
     before any electron count is taken. `chkdump` now reports the valence
     electron count and writes Z − n_core as the fchk "Nuclear charges"
     (Gaussian's convention), keeping Z as "Atomic numbers".
   - **The projection fallback was broken in every spherical run.** When
     projection failed for any reason, the driver set the core guess but kept
     the Cartesian S and H_core the projection branch had computed, while the
     working basis was spherical. The SCF then ran on mismatched matrices and
     diverged (HI: no convergence in 50 iterations). The fallback now lets the
     normal one-electron block rebuild them.
   - **Six Σ Z electron counts the step-2 audit missed**: two in the HF MO
     printout (HI marked 27 orbitals occupied instead of 13), four in
     `src/dft/driver.cpp` (KS electron counts and the grid-integration check,
     which would have been wrong under ECP DFT at step 6). All now use
     `total_nuclear_charge()`; no `Σ atomic_numbers` loop is left in `src/`.

   Gates:

   | case | what it pins |
   |---|---|
   | `hi_rhf_def2svp_ecp_fcidump` (smoke) | NELEC 26; HOMO is orbital 13 at −0.384222 (PySCF's); RHF energy |
   | `hi_rhf_def2svp_ecp_projection_refused` (smoke) | a 3-21G checkpoint is not projected into def2-SVP, and the fallback converges to PySCF's energy |
   | `hi_rhf_def2svp_ecp_restart_full` | `guess full` from a v8 ECP checkpoint converges in 2 iterations to the same energy |

5. **ECP gradients — OPEN.** libecpint has the derivative:
   `ECPIntegral::compute_shell_pair_derivative` returns nine blocks in the
   order Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz (C the ECP centre). Each adds to
   its own atom's gradient, so when centres coincide the components sum, and
   translational invariance ties the three together. Construct the engine with `deriv = 1`, which raises the
   L ceiling to basis L + 1 ≤ `LIBECPINT_MAX_L` (5); g shells still fit.
   Contract with the density in the gradient's spherical-lift skin like the
   other 1e derivatives, then lift the step-4 rejection in `hf_driver.cpp`.
   Gate: FD and PySCF gradients; geometry optimization of a small
   heavy-element molecule against PySCF.
6. **DFT with ECPs** after spherical DFT lands: same gates in `planck-dft`.

## Not in this scope

Spin-orbit ECP terms, all-electron relativistic Hamiltonians (X2C, DKH).
