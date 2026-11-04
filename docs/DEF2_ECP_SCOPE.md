# What does Planck need to run def2 basis sets, including the ECPs for heavy elements?

Research scope, opened 2026-10-26. Steps 1–4b have landed (branch
`feat/def2-ecp`, PR #180); steps 5 (ECP gradients) and 6 (DFT with ECPs) are
open, so this is still a scope. Rewrite it as the answer when they land.
Status lives in `vault/Status/Open Work.md`.

## Current result

**def2 runs for H–Rn in `hartree-fock` single points, and matches PySCF to
≤1.2e-9 Eh.** From Rb onward the core electrons are replaced by the def2
effective core potentials (ECPs), computed by libecpint.

| works | refused, with a message |
|---|---|
| def2-SVP, -SVPD, -TZVP, -TZVPP, -QZVP for H–Rn, spherical | ECP gradients, geometry optimization, frequencies, imaginary follow (step 5) |
| RHF single points with ECPs, PySCF-gated | `planck-dft` with an ECP basis (step 6) |
| SAD guess, full point-group symmetry, with ECPs | A checkpoint load or cross-basis projection between different ECP cores (falls back to the core guess) |
| Checkpoint restart (version 8 stores core counts), FCIDUMP | |

UHF, ROHF and post-HF energies with ECPs go through the same H_core and
electron count but have no regression cases yet.

### Validation

All references are PySCF 2.13.0 with the basis (and ECP) parsed from Planck's
own GBS file, at Planck's CODATA 2018 Bohr:
`tests/pyscf/def2_rhf_references.py` (all-electron) and
`tests/pyscf/def2_ecp_references.py` (ECP; it converts the Gaussian94 ECP
block, which PySCF's reader skips).

| case | ECP | Planck − PySCF (Eh) | max ‖ΔV‖ |
|---|---|---|---|
| water/def2-SVP | — | 0 | |
| water/def2-QZVP (g shells) | — | 0 | |
| H2S/def2-TZVP | — | 0 | |
| SiH4/def2-SVP | — | 0 | |
| ZnCl2/def2-SVP | — | 1.2e-9 | |
| HI/def2-SVP (smoke) | I, 28e | 0 | 1.6e-11 |
| HI/def2-TZVP, SAD, `use_symm` | I, 28e, f shells | 0 | 6.3e-11 |
| AgCl/def2-SVP | Ag, 28e | 1.1e-9 | 1.3e-9 |
| PdCO/def2-SVP, SAD | Pd, 28e | 5e-10 | 4.2e-10 |
| AuH·H2O/def2-SVP, C1 | Au, 60e | 2e-10 | 9.2e-10 |

‖ΔV‖ compares Planck's dumped Cartesian V_ne + U against PySCF's
`int1e_nuc + ECPscalar`, element by element, after normalizing both by
√diag(S) (run the script with a dump directory and case name). All energies
are pinned at 1e-9 except the two Cl cases at 2e-9.

Restart, FCIDUMP and boundaries:

| case | what it pins |
|---|---|
| `hi_rhf_def2svp_ecp_fcidump` (smoke) | NELEC 26; HOMO is orbital 13 at −0.384222 (PySCF's). PySCF reading the file reproduces its ECP RHF energy to 2e-12, from the reference determinant and from an RHF solved on the file (`def2_ecp_references.py --fcidump`) |
| `hi_rhf_def2svp_ecp_projection_refused` (smoke) | a 3-21G checkpoint (all-electron iodine, 54 electrons) is not projected into def2-SVP (26), and the fallback converges to PySCF's energy |
| `hi_rhf_def2svp_ecp_restart_full` | `guess full` from a v8 ECP checkpoint converges in 2 iterations to the same energy |
| `hi_rhf_def2svp_ecp_gradient_rejected` (smoke) | gradients refused; pins the parsed ECP ("I: 28 core electrons, Z_eff 25, L 3, terms 4/7/8/10") |
| `hi_rks_def2svp_ecp_rejected` (smoke) | `planck-dft` refuses ECP bases |

Full regression suite at step 4b: 207/211, the four failures being the known
`*_rccsdt_sto3g` routing cases. `ctest` (the C++ unit binaries) has not been
run on the branch.

## What remains

1. **Step 5, ECP gradients.** libecpint has the derivative:
   `ECPIntegral::compute_shell_pair_derivative` returns nine blocks in the
   order Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz (C the ECP centre). Each adds to
   its own atom's gradient, so when centres coincide the components sum, and
   translational invariance ties the three together. Its header still says
   "derivs currently being implemented", so trust nothing before the FD gate.

   **One seam.** Every gradient (RHF, UHF, ROHF, RKS, UKS, RMP2, UMP2, the DH
   path) gets its 1e term from `Gradient::one_electron_gradient`, already
   called on the lifted Cartesian D. Adding the ECP term there reaches all of
   them, and geomopt, frequencies and imaginary follow call those gradients.
   Nothing per caller.

   - **5a. `ECP::_compute_ecp_gradient(basis, shell_atom, D)` → natoms × 3**
     (`src/integrals/ecp.cpp`), beside `_compute_ecp`, sharing its
     shell/ECP conversion (factor it out, do not copy it). Engine built with
     `deriv = 1`. Loop the same `b ≥ a` triangle with weight 2·D_μν off the
     diagonal (D_μμ on it), `_component_norm` per element; A and B blocks go
     to their shells' atoms, C to `ecp._atom_index`. Several pairs write one
     atom row, so accumulate in the fixed 64-bin pattern of
     `one_electron_gradient`, summed in bin order (thread-count invariant).
   - **5b. Wire it in.** `OneElectronGradient` gains an `ecp` field, included
     in `core()`; zero when the basis has no ECP, so every all-electron
     gradient is byte-identical (check on the core suite).
   - **5c. L ceiling.** `describe_ecps` takes the derivative order: libecpint
     needs basis L + 1 ≤ `LIBECPINT_MAX_L` (5) for gradients. def2 tops out
     at g, so it fits; the check is for safety, since libecpint only asserts.
   - **5d. Lift the refusal** in `src/hf_driver.cpp` for Gradient, GeomOpt,
     Frequency, GeomOptFrequency and ImaginaryFollow. MP2 gradients stay
     refused by the existing spherical gate (def2 is spherical); no ECP code
     needed for that. Confirm the geomopt inner loop moves the ECP centres
     (`read_gbs_basis` → `rebuild_basis_dependent_state`): the final geomopt
     energy must equal a single point at the printed geometry.
   - **5e. Gates**, PySCF 2.13.0 `cart=False` (`mf.nuc_grad_method()`
     includes the ECP term), extending `tests/pyscf/def2_ecp_references.py`:

     | case | why |
     |---|---|
     | HI/def2-SVP RHF gradient (smoke) | smallest; the C term on I is the bulk |
     | AuH·H2O/def2-SVP RHF gradient, C1 | all components nonzero, off-axis; H/O shells off the ECP centre exercise A, B and C separately |
     | HI/def2-SVP energy FD, two steps | a third answer that shares no code with libecpint's derivative or PySCF |
     | translational sum ≤ 1e-10 | ties A + B + C together cheaply |
     | HI or AgH geomopt vs PySCF `geometric_solver` | E_opt, bond length |
     | HI frequency vs PySCF FD-of-analytic Hessian | one stretch, < 0.1 cm⁻¹ |
     | HI⁺ UHF and ROHF gradient | closes the "gate UHF/ROHF with an ECP" item at the same time |

     Tolerance 1e-7 Ha/Bohr (print precision), as the other gradient gates.
   - **Order:** 5a+5b with the HI FD and PySCF checks first; 5c–5d only once
     those pass; geomopt/freq/UHF/ROHF gates last.
2. **Step 6, DFT with ECPs**, after spherical DFT lands
   (`docs/SPHERICAL_DFT_SCOPE.md`). H_core and the KS electron counts are
   already ECP-aware; lifting the rejection in `src/dft/driver.cpp` is the
   gate. Same PySCF gates in `planck-dft`.
3. **Smaller open items** (none blocks step 5; run `ctest` before starting
   it, since it has never run on the branch):
   - gate UHF, ROHF (folded into 5e) and post-HF energies (one RMP2 and one
     CASSCF energy on HI vs PySCF) with an ECP;
   - the two Cl cases (ZnCl2, AgCl) sit 1.1–1.2e-9 (~4e-13 relative) from
     PySCF for an unexplained reason. It is not the Å→Bohr constant (ZnCl2
     was 1.2e-9 before that fix too) and survives tight convergence in both
     codes;
   - when upstream PR #81 merges, repin `LIBECPINT_GIT_TAG` to the merge
     commit (the generator precision bug is in it, so nothing to report);
   - run `ctest` on the branch.

## How it works

**Split by element.** H–Kr def2 is all-electron and needed only basis files.
Rb onward needs ECPs. def2 is defined for spherical functions, so it serves HF
and post-HF today and DFT once spherical DFT lands.

**Data.** BSE's Gaussian94 output appends an ECP block per element: the core
count, then the local channel U_L and the semilocal channels U_l − U_L (already
differences), each a list of `(n, ζ, d)` terms d·r^(n−2)·e^(−ζr²). An ECP
channel carries a radial power that `Shell` does not, so ECPs are stored per
atom beside the shells (`ECPPotential` on `Basis::_ecps`, `src/base/types.h`)
rather than as a `Shell` variant, with `n` kept as written.

**Pipeline.**
1. `detail::read_gbs` returns shells and ECP blocks; `read_gbs_basis` stores
   one `ECPPotential` per non-ghost atom (`make_ecp_potential`).
2. `Calculator::sync_ecp_core_electrons()` copies the core counts to
   `Molecule::ecp_core_electrons`, so `nuclear_charge(a)` = Z − n_core and
   every electron count, which goes through `total_nuclear_charge()`,
   follows. Element lookups keep reading Z.
3. `HartreeFock::ECP::_compute_ecp` (`src/integrals/ecp.cpp`) builds the
   Cartesian U matrix with libecpint. libecpint takes `n` in the file
   convention and treats the largest-l channel as local, so U_L goes in at
   l = L. It integrates bare monomials, so coefficients go in as
   (c_p N_c) N_p and `_component_norm` is applied per element afterwards;
   both enumerate Cartesian components xx, xy, xz, …
4. `_compute_potential` (`src/integrals/base.h`) returns V_ne + U, and every
   H_core build calls it: both HF driver paths,
   `rebuild_basis_dependent_state` (so geometry steps inherit it), the DFT
   driver, counterpoise fragments, and SAD.
5. SAD attaches each element's ECP to its atomic basis and runs the atom with
   Z − n_core electrons. PCM's nuclear potential and counterpoise fragments
   use the reduced charges.
6. Checkpoint version 8 stores each atom's core count after its atomic number
   (`src/io/checkpoint.cpp`, `tools/chkdump.cpp`). Version 7 and older read as
   all zero, which is exact: ECP runs were refused before version 8. Loads and
   projections between different cores are refused. `chkdump` reports valence
   electrons and writes Z − n_core as the fchk "Nuclear charges", keeping Z as
   "Atomic numbers" (Gaussian's convention).
7. FCIDUMP needed no change: h_pq comes from `_hcore`, NELEC and ECORE from
   Z − n_core, so the file is a consistent valence-only Hamiltonian. No note
   about the removed core is written; the format has no field for it and
   nothing a reader computes depends on it.

What needed no change: SCF, MP2, CC, CASSCF/RASSCF and PCM consume H_core and
an electron count. Full point-group symmetry needed no change either: its
skeleton covers only the two-electron Fock build, and H_core is built in full.
Population analyses show fewer core electrons on ECP atoms, as expected.

**Boundaries.** Fail cleanly, never return an incomplete answer: gradient
workflows refuse ECP bases in `hf_driver.cpp` (step 5), `planck-dft` refuses
them (step 6), and `describe_ecps` refuses basis or ECP L above libecpint's
`LIBECPINT_MAX_L` (5), which libecpint only asserts (and asserts are off in
Release).

**Debugging ECP integrals.** Compare against PySCF (`mol.intor('ECPscalar')`),
and when the two disagree, against a brute-force radial × Lebedev integral
around the ECP centre. That third answer located the libecpint bug below.

### Library: libecpint

| option | for | against |
|---|---|---|
| **libecpint** (C++) via `ExternalProject` — taken | complete type-1/type-2 integrals and first and second derivatives; used by Psi4 | a new dependency |
| native | no dependency | type-2 integrals are the hard part of ECP work (angular projection, radial quadrature with screening); long validation tail |

The integrals and their derivatives are the risky part, and libecpint has
them. This superseded an earlier `ECP_SUPPORT_SCOPE.md` plan for a native
engine (merged here and deleted). Build details:
- pinned to the head of open upstream PR #81 (`loriab/libecpint`
  `4e8767fc`), which sits on upstream HEAD `95203c68` (itself past v1.0.7 for
  fixes #71 and #72) and fixes seven type-2/quadrature defects, our 6-digit
  constants among them. **Decision 2026-11-04: when PR #81 merges, repin to
  the upstream merge commit and rerun `planck-ecp-gradient-fd` and the
  `*_ecp` cases** (Findings, step 5a);
- built with Planck's C++ compiler, because its API passes `std::vector` and
  a libc++ build would not link against libstdc++;
- linked into every target, since `ecp.cpp` sits in the shared integral
  sources.

Licence (checked 2026-10-26; not legal advice): libecpint is MIT (Copyright
2021 Robert A. Shaw) and bundles Faddeeva (MIT); pugixml is disabled in
Planck's build and GoogleTest is for its own tests. Compatible with Planck's
PolyForm Noncommercial licence. The notice travels with the fetched source,
and `CMakeLists.txt` installs it with the binaries as
`share/doc/planck/libecpint-LICENSE`. The def2 files and ECP blocks come from
the Basis Set Exchange (BSD-3-Clause), each keeping its BSE header, which
carries the citation.

## Findings on the way

**The Å→Bohr constant was 7.2e-8 off its own reciprocal (step 1b).** The
first def2 gates landed at 4e-10..1.3e-8 from PySCF. The first guess, f
shells, did not fit: water/def2-SVP has no f functions and missed at 3.7e-9,
while H2S/def2-TZVP, which has one, was at 1.5e-9. Nuclear repulsion settled
it: water E_nn differed by 6.6e-7 Eh (Planck 9.1895344241, PySCF
9.1895337629). `src/base/tables.h` paired `BOHR_TO_ANGSTROM = 0.529177210903`
(CODATA 2018) with `ANGSTROM_TO_BOHR = 1.8897259886`, an older literal 7.2e-8
smaller than the reciprocal — exactly the relative E_nn gap — so every
Ångström input ran at a geometry 7.2e-8 too small. PySCF at Planck's Bohr
coordinates reproduced four cases to every printed digit before the fix. Now
`ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM`, and three copies of either
literal (`types.h`, two RI tests) use the constants. PySCF defaults to the
CODATA 2010 Bohr (3e-11 away), so the reference scripts pass
`unit=0.529177210903`. Ruled out on the way: SCF convergence (both codes
unchanged at tighter tolerances; PySCF also at `direct_scf_tol` 1e-16). Suite
impact: 26 extended and 10 core cases pinned to Planck's own past output
moved 2e-9..1.9e-7 Eh and were re-pinned; water SA-2 CASSCF moved from 3.6e-8
to 1.1e-8 off PySCF's SAD-start minimum.

**The nuclear-charge audit missed six electron counts (steps 2, 4b).** Step 2
switched 16 of 46 direct `atomic_numbers[...]` reads to `nuclear_charge()`
(nuclear repulsion and its gradient, the nuclear-attraction derivative,
gradient electron counts, multipole nuclear charges); with ghost atoms the
gradient and multipole code had been counting the ghost's charge. The other
30 are element lookups, printing, IO, symmetry matching or counterpoise
copying. But six `Σ atomic_numbers` loops were not indexed reads and escaped
it: two in the HF MO printout (HI marked 27 orbitals occupied instead of 13)
and four in `src/dft/driver.cpp` (KS electron counts and the grid-integration
check). All use `total_nuclear_charge()` now, and none is left in `src/`.
**Audit sums as well as indexed reads.**

**BSE writes the ECP headers uppercase (step 3).** The ECP section uses `RB 0`
and `RB-ECP` while the shells use `Rb`; the case-sensitive lookup failed
every def2 run. The reader normalizes symbols. SAD then still failed: it had a
private copy of the GBS reader. The copy is deleted (−123 lines), and SAD uses
the shared reader.

**libecpint's generated kernels carried 6-digit constants (step 4).** The first
matrix check missed PySCF by up to 1.9e-7, on H–H elements only (both shells
off the iodine centre); the HI energy was 2.3e-8 off. Five eliminations:
1. a brute-force radial × Lebedev integral agreed with PySCF to 12 digits, so
   PySCF was right;
2. libecpint's local and d channels matched; its s and p projectors did not;
3. the error did not move in any digit with 4× radial grids or a tighter
   threshold;
4. libecpint's Bessel table and Gauss–Chebyshev quadrature were each exact to
   ~1e-14 alone;
5. against an exact 1-D integral, every off-centre type-2 integral was too
   large by the same 1 + 2.09e-6 for every exponent and power of r; on-centre
   integrals were exact.

A constant relative factor is a constant: libecpint's generator writes the
type-2 angular prefactors into its `Q*.cpp` kernels at default stream
precision, `157.914` for 16π² = 157.91367…, and 157.914 / 16π² = 1 + 2.09e-6.
Planck patched the generator to 17 digits at fetch time (now carried by
PR #81, so the patch is deleted). After it, the
matrices agree to ~1e-11 relative and HI to every printed digit. **When two
codes disagree, build a third answer that shares nothing with either; and a
relative error that is the same for every input is a wrong constant.**

**A failed spherical projection diverged (step 4b).** Gating the ECP refusal
exposed an older bug: when cross-basis projection failed for any reason, the
driver switched to the core guess but kept the Cartesian S and H_core the
projection branch had computed, while the working basis was spherical. The
SCF ran on matrices of the wrong basis (HI: no convergence in 50 iterations).
No test had ever made a spherical projection fail. The fallback now lets the
normal one-electron block rebuild them.

**Stock libecpint's derivatives are wrong past def2-TZVP (step 5a).** No
release since v1.0.7 (2021); the pin, `95203c68`, is upstream HEAD, and the
"derivs being implemented" header comment is stale. `planck-ecp-gradient-fd`
(tr(D U) FD vs `_compute_ecp_gradient`, random D, five-point, basis re-read at
each displacement) on the pin: HI/def2-SVP and -TZVP agree to 7e-10 relative;
I2/def2-TZVP misses by 2.6e-7 and HI/def2-QZVP by 4.6e-5, the latter the same
at every step (a wrong derivative), while AuH·H2O worsens as h shrinks
(non-smooth U(R)). Open upstream PR #81 (loriab, `4e8767fc`, on top of the
pin) fixes seven type-2/quadrature defects, the first being our 6-digit
constants; against it all five cases pass at ≤4e-9 relative (h = 1e-3, 2e-3).
A ~1e-10 step in U(R) remains for AuH·H2O at h = 5e-4. The maintainer
acknowledged the PR on 2026-11-04 but warned review will be slow.
**Decision (2026-11-04): pin `LIBECPINT_GIT_TAG` to the PR head on the fork**
(`CMakeLists.txt`) and delete `cmake/patch_libecpint.cmake`, whose fix the PR
carries. All ten `*_ecp` cases pass unchanged at their 1e-9/2e-9 pins. When
the PR merges upstream, check the merged commit (the PR may change in review)
and repin to it.

## Step log

| step | commit | what |
|---|---|---|
| 1 | `6e5f191a` | def2-SVP/-SVPD/-TZVP/-TZVPP/-QZVP files (H–Kr, later H–Rn); five all-electron PySCF gates. RI fitting sets not added |
| 1b | `4ca06045`, `aedff720` | Å→Bohr constant fixed; suite re-pinned |
| 2 | `2d3956e5` | 16 physical `atomic_numbers` reads → `nuclear_charge()` |
| 3 | `99f142d1` | ECP parsing and storage; Z − n_core reaches the electron counts |
| 4 | `749c8355` | ECP integrals in H_core via libecpint; SAD/PCM/counterpoise; precision patch |
| 4b | `2c9009f0` | checkpoint v8 core counts; FCIDUMP checked; spherical projection fallback; six Σ Z counts |
| 5a | — (working tree) | `_compute_ecp_gradient`, `planck-ecp-gradient-fd`; stock libecpint fails it beyond TZVP (Findings) |
| 5 | — | ECP gradients (open) |
| 6 | — | DFT with ECPs (open, after spherical DFT) |

## Not in this scope

Spin-orbit ECP terms, all-electron relativistic Hamiltonians (X2C, DKH).
