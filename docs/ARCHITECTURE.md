# Planck — Architecture Reference

<div align="justify">

Planck is a C++23 quantum chemistry engine: Hartree-Fock (RHF, UHF, ROHF), post-HF correlation (MP2, coupled cluster to arbitrary order, CASSCF/RASSCF, FCI, FCIQMC), Kohn-Sham DFT (LDA through double hybrids, TDDFT), analytic gradients, geometry optimization, frequencies, PCM solvation, and counterpoise. It ships as three binaries that share one library of integrals, basis handling, symmetry, SCF, I/O and geometry code. Every subsystem receives a `Calculator &` (the root aggregate in `src/base/types.h`) and reads or writes its fields; larger subsystems (coupled cluster, CASSCF, double-hybrid gradients) keep method-local state objects on top of it. Recoverable failures propagate as `std::expected<T, std::string>` up to the driver, which logs them and exits; there are no exceptions on the hot paths.

This document is the map: what lives where, how the pieces connect, and the invariants that are not enforced by types. The *why* of individual design decisions, with measurements, lives in the topic documents under `docs/` that each section names. Canonical done/open status lives in `vault/Status/`.

</div>

---

## Binaries and Build (`CMakeLists.txt`)

| Binary | Entry | Purpose |
|---|---|---|
| `hartree-fock` | `src/driver.cpp` → `src/hf_driver.cpp` | HF SCF, MP2, CC, CASSCF/RASSCF, FCI, FCIQMC, gradients, geomopt, frequencies, PCM, counterpoise |
| `planck-dft` | `src/dft/main.cpp` → `src/dft/driver.cpp` | RKS/UKS, hybrids, range-separated and double hybrids, TDDFT, gradients, geomopt, frequencies, PCM |
| `planck-mpi` | `src/mpi/main.cpp` (`BUILD_MPI=ON`) | One binary for both; selects HF or DFT on `calculator.is_dft_run()`. Links MPI and defines `USE_MPI` when MPI is found, otherwise builds as a serial binary |
| `chkdump` | `tools/chkdump.cpp` (`BUILD_TOOLS=ON`) | Checkpoint inspector |

<div align="justify">

CMake 3.5+, C++23, extensions off. Sources are gathered per directory with `file(GLOB ...)`, so a new source file needs a reconfigure. The repository also defines about eighty unit-test executables (`planck-*`, under `tests/`), registered with CTest.

**Dependencies.**

</div>

| Dependency | How | Notes |
|---|---|---|
| Eigen 3.4.0 | `FetchContent` | Header-only. `EIGEN_DONT_PARALLELIZE` is defined globally: Planck owns its OpenMP parallelism, and Eigen's own thread pool nested against it (measured ~26 % slower on 8-thread conventional RHF) |
| libmsym | `ExternalProject_Add`, pinned commit | Point-group detection |
| libxc | `ExternalProject_Add`, pinned commit | XC functionals; `planck-dft` and `planck-mpi` only. Built without `-march=native` unless `LIBXC_ENABLE_XHOST=ON` (portability) |
| libecpint | `ExternalProject_Add`, pinned commit | ECP integrals, linked into every target (`src/integrals/ecp.cpp` is in the shared integral sources). Built with Planck's C++ compiler (its API passes `std::vector`). Pinned to the head of unmerged upstream PR #81 (fork `loriab/libecpint`), which fixes kernel constants and derivative accuracy; repin to upstream when it merges |
| OpenSSL (Crypto) | `find_package` | Licence-signature verification and decryption for the Boys library loader |
| Boys-function core | fetched at run time | Encrypted, licence-gated shared library; see `src/lookup/loader/` below |
| OpenMP | `find_package`, `USE_OPENMP=ON` default | Defines `USE_OPENMP` |
| MPI | `find_package`, `planck-mpi` only | Defines `USE_MPI` |
| Python 3 | `find_package` | Runs the `ccgen` generator at build time for the generated CC kernels |

<div align="justify">

**Generated code.** The coupled-cluster residual kernels for the generated arbitrary-order solvers are emitted by the `ccgen` Python package (`python/ccgen/`) at build time. CMake options select what is generated: `PLANCK_CC_SPIN_ADAPT` (spin-adapted closed-shell kernels; ON, and required for correct generated RCC energies), `PLANCK_CC_UCC` (unrestricted generated CC; OFF by default), `PLANCK_CC_ARBITRARY_LOWER_RANKS` (route generated rank 2/3 through the arbitrary-order harness), `PLANCK_CC_DRESS_OPERATORS` and `PLANCK_CC_INCLUDE_INTERMEDIATES` (dressing experiments; OFF). See `docs/CCGEN_KERNEL_WIRING.md`.

**Configure-time headers.** `src/base/basis.h` (from `basis.h.in`) bakes in the basis-set install path, overridable by `$BASIS_PATH`; `src/lookup/loader/license_path.h` and `cache_path.h` fix where the Boys licence and its cached blob live under the install prefix.

</div>

---

## Error Handling and Logging

<div align="justify">

Public functions return `std::expected<T, std::string>`; callers propagate with an early `return std::unexpected(...)` or the monadic `.and_then` / `.transform`. The driver prints the final error. Debug invariants use `assert`, which compiles out in Release builds (so `assert`-only unit tests pass vacuously there; see `vault/Status/Open Work.md`).

`src/io/logging.h` is a header-only logger with a global mutex and a `thread_local` silence depth. The `ScopedSilence` RAII guard mutes all output while an inner calculation runs (finite-difference Hessian points, SAD atoms, counterpoise sub-calculations, geometry-optimization inner SCFs). `map_enum<T>` turns enums into printable names.

</div>

---

## Central Type System (`src/base/`)

<div align="justify">

`types.h` (~1700 lines) is the one header every module includes: options structs, data structs, enums, and a few shared helpers (the SCF DIIS containers and their coefficient solve). `tables.h` holds constants, `sphere.h` the Fibonacci sphere points used by the ESP shells, `mpi_env.h` the whole MPI surface (below), and `basis.h.in` the basis path template.

</div>

### Enumerations

| Enum | Values |
|---|---|
| `ShellType` | `S` … `I` (L = 0 … 6) |
| `BasisType` | `Cartesian`, `Spherical` |
| `SCFType` | `RHF`, `ROHF`, `UHF` |
| `SCFMode` | `Conventional`, `Direct`, `Auto` |
| `SCFGuess` | `HCore`, `SAD`, `ReadDensity`, `ReadFull` |
| `IntegralMethod` | `ObaraSaika`, `RysQuadrature`, `HeadGordonPople`, `Auto` (default) |
| `ERIKernel` | `Coulomb`, `LongRange` (erf), `ShortRange` (erfc) |
| `PostHF` | `None`, `RMP2`, `UMP2`, `RCCSD`, `UCCSD`, `RCCSDT`, `UCCSDT`, `RCCSDTQ`, `UCCGEN`, `CASSCF`, `RASSCF`, `FCI`, `FCIQMC` |
| `CalculationType` | `SinglePoint`, `Gradient`, `GeomOpt`, `Frequency`, `GeomOptFrequency`, `ImaginaryFollow`, `LinearResponse` |
| `OptCoords` | `Cartesian`, `Internal` |
| `DFTGridQuality` | `Coarse`, `Normal`, `Fine`, `UltraFine` |
| `XCExchangeFunctional` / `XCCorrelationFunctional` | named aliases plus `Custom` (any libxc name or ID) |
| `LinearResponseMethod` / `LinearResponseSpin` | Casida or TDA; singlet, triplet, spin-conserving |
| `SolvationModel` | none or C-PCM |

### `Molecule` and the three coordinate frames

<div align="justify">

`coordinates` is the input geometry (Angstrom); `_coordinates` is the same in Bohr; `_standard` is the symmetry-reoriented geometry in Bohr, set by `detectSymmetry()` or copied from `_coordinates` when symmetry is off. Basis centres, nuclear repulsion and every integral use `_standard`. DFT additionally reads `_coordinates` for the grid, so after symmetry detection the DFT path calls `sync_coordinate_frames_from_standard()` to keep both in one frame (a split frame once produced symmetry-forbidden gradient components). Ghost atoms (for counterpoise) carry basis functions but no nuclear charge.

</div>

### `Shell`, `ContractedView`, `Basis`

<div align="justify">

`Shell` holds one contracted shell: centre, L, exponents, coefficients (with the contracted norm already folded in at read time), and primitive norms. `ContractedView` is a span-based view of one Cartesian component with its `(lx, ly, lz)`, component norm, and `_index`, its position in `Basis::_basis_functions`, which is how integral code places results in AO matrices. `Basis` holds the shells and the per-component views. For a spherical basis it also holds `_cart_to_sph`, the block-diagonal `[n_sph × n_cart]` transform; `working_nbasis()` is the size of the basis SCF actually works in (spherical or Cartesian) and `nbasis()` the Cartesian count the integral engines use.

</div>

### `ShellPair` and `PrimitivePair`

<div align="justify">

Built once per geometry by `build_shellpairs`: for every unique pair (upper triangle, row-major), the Gaussian product centres, prefactors and exponent sums of all primitive pairs, plus the Schwarz bound. AO placement always uses `sp.A._index` / `sp.B._index`, never arithmetic on the pair's position in the vector.

</div>

### SCF data and DIIS

<div align="justify">

`SpinChannel` (density, Fock, MO coefficients, energies, irrep labels) ×2 make `DataSCF`; ROHF and RHF keep one spatial MO set in both channels. `DIISState` / `UHFDIISState` implement Pulay DIIS with error `e = FPS − SPF`; both route their coefficient solve through `solve_diis_coefficients`, which drops the oldest vector only on a genuine breakdown (indefinite Gram block, non-finite coefficients), not on the Gram condition number.

</div>

### Options structs and `Calculator`

<div align="justify">

Each input section parses into an `Options*` struct (`OptionsSCF`, `OptionsBasis`, `OptionsGeometry`, `OptionsIntegral`, `OptionsDFT`, `OptionsSolvation`, `OptionsActiveSpace`, `OptionsFCIQMC`, `OptionsBSSE`, `OptionsESP`, `OptionsOutput`). `Calculator` aggregates all of them with the molecule, basis, shell pairs, overlap and core Hamiltonian, the stored ERI tensor (conventional mode), SCF results, post-HF results, gradient, Hessian and frequencies. `Calculator::initialize()` allocates matrices after the basis is loaded and must run before any integral or SCF code.

</div>

### MPI surface (`mpi_env.h`)

<div align="justify">

The only place `<mpi.h>` is included below the driver. Without `USE_MPI` every function is a no-op returning rank 0, size 1, so the compute kernels carry identical source in all three binaries. Distributed work splits the shell-quartet or grid-point loop by rank and reduces in fixed rank order, so results are independent of rank count. See `docs/HPC_MPI.md`.

</div>

---

## Input/Output (`src/io/`)

| File | Role |
|---|---|
| `io.{h,cpp}` | Section-based input parser (`%begin_control`, `%begin_scf`, `%begin_geom`, `%begin_coords`, `%begin_dft`, …) into the `Options*` structs; `std::expected` on every parse error, naming the keyword |
| `checkpoint.{h,cpp}` | Binary `.hfchk` (magic `PLNKCHK`, version 7, additive optional blocks): geometry, basis, SCF matrices, CASSCF MOs. Restart modes `ReadDensity`, `ReadFull`, and cross-basis Löwdin projection |
| `fcidump.{h,cpp}` | MOLPRO FCIDUMP export of the MO Hamiltonian for external FCI/DMRG/FCIQMC codes |
| `results_json.h` | `--json <path>`: the stable machine-readable results schema for `python/planck.py` |
| `external_libraries.h` | Startup table of every linked library with version, licence and role |
| `logging.h` | Logger (above) |

---

## Lookup and the Boys Library (`src/lookup/`)

<div align="justify">

`elements.{h,cpp}` holds atomic numbers, symbols, masses and covalent radii. `boys.h` declares `boys`, `boys_vec` and the batched `boys_batch`, which dispatch to function pointers resolved at start-up.

The Boys function is not compiled into Planck. `loader/` implements a licence-gated loader: `orchestrator` reads the local signed `license.json`, verifies its Ed25519 signature (`signing_key.cpp`, `license.cpp`), fetches the matching encrypted library blob for the platform (`fetch.cpp`, with a local cache in `cache.cpp`), decrypts it with AES-256-GCM (`decrypt.cpp`), and loads it from memory (`memory_load.cpp`). It runs once, first thing in `main`; failure is terminal. `planck-mpi` fetches once on one rank and broadcasts the ciphertext. Every integral that needs the Boys function (OS, HGP, closed forms, nuclear attraction) depends on this core; Rys quadrature uses its own roots.

</div>

---

## Basis Sets (`src/basis/`)

| File | Role |
|---|---|
| `gaussian.cpp`, `gbs_parser.{h,cpp}` | Gaussian94 `.gbs` reader. Folds the contracted norm into coefficients, stamps `_index`, and drops redundant primitives (an outer primitive that is also its own uncontracted shell of the same L, as in the BSE cc-pVXZ files), which matches PySCF's primitive count |
| `basis.{h,cpp}` | Primitive and contracted normalization, Cartesian component order |
| `spherical.{h,cpp}`, `spherical_recurrence.{h,cpp}` | Cartesian → real-spherical transform per shell (pseudo-inverse of the unit-normalized components for L = 3–5; recurrence for L = 6), and the density lift `M_cart = Cᵀ M_sph C` |
| `rifit.{h,cpp}` | RI / density-fitting auxiliary basis loader, kept separate from the orbital basis |

<div align="justify">

**Spherical bases** reuse the Cartesian integral engines unchanged: integrals are computed Cartesian and transformed at the skin (`_overlap = C S Cᵀ`, `_hcore = C (T+V) Cᵀ`), and gradients lift the spherical density and energy-weighted density back to Cartesian once. `SCF::rebuild_basis_dependent_state` (`src/scf/working_state.cpp`) redoes this, including the geometry-dependent row normalization of `_cart_to_sph`, at every displaced geometry in geomopt and frequency loops.

</div>

---

## Symmetry (`src/symmetry/`)

| File | Role |
|---|---|
| `symmetry.{h,cpp}`, `wrapper.h` | libmsym wrapper: point group, standard orientation (`_standard`), atom permutations |
| `mo_symmetry.{h,cpp}` | SAO basis and per-irrep blocked diagonalization; MO irrep labels (Abelian subgroup labels for non-Abelian groups) |
| `group_operations.{h,cpp}` | AO-basis group operators; spherical operators are `S_sph⁻¹ (C S_cart O_cart Cᵀ)` |
| `integral_symmetry.{h,cpp}` | `SignedAOSymOp`: signed AO permutations per operation |
| `skeleton_eri.h`, `{os,hgp,rys}_symm.cpp` | Full-point-group skeleton ERI build (petite list, orbit multiplicity) per engine |
| `fock_symmetrization.{h,cpp}` | Skeleton Fock → full Fock by group averaging, `F = (1/|G|) Σ_R O_Rᵀ F_skel O_R` |
| `vibrational_symmetry.{h,cpp}` | Mulliken labels for normal modes |

<div align="justify">

**Two symmetry paths in direct SCF.** Below `threshold` basis functions the full-point-group skeleton ERI is built once and persisted. Above it, the fused direct Fock loop (below) runs with the geometry's D2h-subgroup signed operations and skeleton accumulation, then symmetrizes; an SAO density is symmetric under any subgroup, so this is exact. Under `engine auto` the full-symmetry skeleton path uses HGP. The density convention is contravariant: `O P Oᵀ = P`.

</div>

---

## Integrals (`src/integrals/`)

<div align="justify">

Three ERI engines share one public surface (full tensor `_compute_2e`, two-phase and memory-direct Fock builds, single contracted quartet, 12-component derivative quartet), selected by `OptionsIntegral::_engine`:

</div>

| Engine | Files | Scheme |
|---|---|---|
| Obara-Saika | `os.{h,cpp}` | VRR + HRR per primitive quartet. Also owns every one-electron integral (overlap, kinetic, nuclear attraction, multipoles, cross-basis overlap, ESP) and their derivatives |
| Head-Gordon-Pople | `hgp.{h,cpp}` | VRR per primitive pair, contracted, then HRR once per contracted quartet. Serves Coulomb and the screened `LongRange`/`ShortRange` kernels natively |
| Rys quadrature | `rys.{h,cpp}`, `rys_roots.{h,cpp}`, `rys_detail.h` | Roots and weights from per-n Chebyshev segment fits plus the large-T asymptote, built on first use from a reference generator; per-quartet `RysScratch` |
| ECP (one-electron) | `ecp.{h,cpp}` | libecpint type-1/type-2 integrals into a Cartesian U matrix; `base.h` `_compute_potential` returns V_ne + U for every H_core build |
| Auto (default) | `base.h`, `auto_block_eri.h` | Block-kernel chain (below) in the fused Fock loop; a per-(L_AB, L_CD) engine table (`kAutoEngine`, fitted from benchmarks) for per-component work |

### Block kernels and the fused Fock loop

<div align="justify">

The SCF Fock build is memory-direct: `fused_fock.h` runs one shell-quartet loop over canonical quartets and contracts each one straight into the Fock matrix (`fock_accumulate.h`), so nothing of size nb⁴ is allocated. The loop is parameterized on the engine's per-quartet callable, so OS, HGP, Rys and Auto share it. Under symmetry, `quartet_orbit.h` maps a canonical quartet to its orbit.

A **block kernel** (`block_eri.h`) evaluates every Cartesian component of one shell quartet from a single primitive pass, with `prepare` / `value` / `covers`. `auto_block_eri.h` chains closed forms (`low_l_eri.h`, total L ≤ 3), the 2D Rys block (`rys_block_eri.h`; one primitive quartet at L ≥ 4, up to three at L ≥ 7) and the HGP block (`hgp_block_eri.h`, lanes across primitive quartets, up to L = 12). When a block kernel covers every total L in the basis and the densities are symmetric, the loop digests whole blocks densely with shell-level degeneracy factors. Partial Fock matrices are binned by bra shell pair and summed in fixed bin order, so the result is identical at any thread count.

The direct SCF builds incrementally, `G_prev + G[ΔP]`, with density-weighted block screening, which is switched off once the DIIS error is below 1e-8 so the final iterations are exact. Schwarz tables come from `screening.{h,cpp}` (HGP-evaluated, engine-independent).

The *why* and the measurements are in `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md` and `docs/ERI_KERNEL_VS_LIBCINT.md`.

</div>

### Derivatives

<div align="justify">

`hgp_deriv_block_eri.h`, `rys_deriv_block_eri.h` and `auto_deriv_block_eri.h` are the derivative counterparts, used by the shared two-electron gradient loop (see Gradients). `Gradient::compute_eri_deriv_dispatch` routes per-component derivative quartets to HGP or OS.

</div>

**Invariants.** Coefficients already carry the contracted norm; never multiply by it again. Any screened-kernel scaling in the OS recurrences must be mirrored in HGP's `hgp_vrr` (a missing `boys_scale` on the C-VRR coupling term once corrupted only matching-axis quartets).

---

## SCF (`src/scf/`)

| File | Role |
|---|---|
| `scf.{h,cpp}` | `run_rhf`, `run_uhf`, `run_rohf`: conventional or direct, SAO-blocked or plain, DIIS, level shift, damping, SOSCF, incremental direct Fock |
| `sad.{h,cpp}` | SAD guess: the raw superposed atomic density scaled to the electron count (not projected to idempotency; the projection was ill-posed for open-shell atoms) |
| `stability.{h,cpp}` | RHF/UHF stability: real/complex internal, external triplet, UHF real-rotation Hessian; optional instability following |
| `working_state.{h,cpp}` | `rebuild_basis_dependent_state`: shell pairs, 1e integrals and the spherical working-basis lift at a new geometry |

<div align="justify">

The loop is standard: build G, F = H + G (RHF), α/β Fock (UHF), or the Roothaan effective Fock (ROHF); DIIS on `FPS − SPF`; diagonalize in the orthogonal (or SAO) basis; rebuild the density; converge on ΔE, ΔP and the DIIS error (the last guards against a ΔP = 0 false convergence on SAD-started atoms). At convergence, when DIIS extrapolated, the true Fock is re-diagonalized once so the stored orbital energies belong to the true Fock; gradients' energy-weighted densities depend on that.

**SOSCF** (RHF, UHF, RKS, UKS) reuses the CASSCF augmented-Hessian solver (`src/post_hf/casscf/aug-hessian.h`) and Cayley rotation, as a transient window that hands back to DIIS by default. ROHF SOSCF is not implemented (there is no ROHF orbital response machinery). See `docs/SOSCF*.md`.

`Auto` mode picks conventional SCF below `threshold` basis functions (stored ERI tensor) and direct above.

</div>

---

## Solvation, Properties, Counterpoise

| Directory | Role |
|---|---|
| `src/solvation/pcm.{h,cpp}` | C-PCM on a SWIG cavity (Lebedev points, smooth switching function, Gaussian surface charges), apparent surface charges, reaction field added to the Fock matrix each iteration; shared by HF and DFT single points |
| `src/populations/` | Mulliken, Löwdin, Mayer bond orders, multipoles, ESP and CHELPG charges |
| `src/bsse/counterpoise.{h,cpp}` | Two-fragment Boys-Bernardi counterpoise: five silenced sub-SCFs with ghost atoms, per-fragment charge and multiplicity |

<div align="justify">

**Mayer bond order density convention:** the closed-shell branch contracts the total density with no prefactor (H₂ gives 1); the open-shell branch contracts per-spin densities and needs an explicit factor 2 to agree. C-PCM supports energies, analytic gradients, geometry optimization and frequencies; its SWIG cavity is smooth in the geometry, which is what makes the gradient possible (see `docs/PCM_GEOMETRY.md`).

</div>

---

## Post-HF Correlation (`src/post_hf/`)

### Integral transforms and RI (`integrals.{h,cpp}`, `ri/`)

<div align="justify">

`transform_eri(eri, nb, C1, C2, C3, C4)` is the shared quarter-by-quarter AO→MO transform: each quarter contracts one leg with the contracted index outside a unit-stride inner loop, parallel over disjoint output slabs, so the result is bitwise independent of thread count. `transform_eri_internal` (all four legs the same) and `transform_eri_active_cache` (`(pu|vw)`, active legs first) build on it. The RI path (`ri/ri_eri.{h,cpp}`, `transform_eri_ri`, `RI::build_ri_fock_rhf`) builds 2- and 3-centre integrals against the auxiliary basis and factors `(μν|λσ) ≈ Σ (μν|P) V⁻¹_PQ (Q|λσ)`; it serves RI-MP2 energies and gradients, an RI-JK Fock build, and RI-routed CASSCF and FCI.

</div>

### MP2 and orbital response

| File | Role |
|---|---|
| `mp2.h`, `mp2_rmp2.cpp`, `mp2_ump2.cpp`, `mp2_internal.{h,cpp}`, `mp2_t2_storage.cpp` | `rmp2_kernel` / `ump2_kernel` (energies, frozen core), `rmp2_make_natural_orbitals`; layouts mirror PySCF's `_ChemistsERIs` |
| `mp2_gradient.{h,cpp}` | RMP2 and UMP2 relaxed densities and gradient intermediates |
| `orbital_hessian.{h,cpp}` | `OrbitalHessian`: the matrix-free RHF/UHF orbital Hessian, one `SCF::two_electron_fock{,_uhf}` call per action (fitted under `scf_ri_jk`, exact otherwise); used by the Z-vector solves and HF SOSCF |
| `rhf_response.{h,cpp}`, `uhf_response.{h,cpp}` | CPHF / Z-vector solves (`solve_rhf_cphf`, `solve_uhf_cphf`): GMRES on `OrbitalHessian` with the orbital-gap diagonal preconditioner |

### CI engine, FCI and FCIQMC (`ci/`, `fci.{h,cpp}`, `fciqmc_driver.{h,cpp}`)

<div align="justify">

`ci/strings.{h,cpp}` enumerates determinants as packed `uint64_t` bit strings (up to 31 spatial orbitals per spin packed together); `ci/ci.{h,cpp}` holds the determinant space, Slater-Condon elements, a dense Hamiltonian below 500 determinants and an iterative sigma build with Davidson above. The sigma build is a scatter into 64 fixed bins by ket index, `schedule(static, bin_size)`, summed in bin order (bitwise thread-count invariant). `ci/rdm.{h,cpp}` builds 1- and 2-RDMs (and the bilinear 2-RDM for CI response) with the same binned, fixed-order reduction.

`run_fci` is CASCI over the whole MO space. `build_all_mo_ci_setup` produces the integrals and Hamiltonian callbacks shared by `run_fci` and `run_fciqmc`, so the two can only differ in how they solve. `ci/fciqmc.{h,cpp}` and `ci/spawn_accumulator.h` hold the walker population, a deterministic per-parent RNG, the excitation generator with consistent `p_gen`, spawn/death/annihilation, shift control with the initiator approximation, and both energy estimators with blocked error bars. See `docs/FCIQMC_*.md`.

</div>

### CASSCF and RASSCF (`casscf.h`, `casscf_internal.h`, `casscf/`)

| File | Role |
|---|---|
| `casscf/casscf.cpp` | `run_mcscf_loop`: macroiterations, candidate selection, trust region, convergence, output |
| `casscf/casscf_driver_internal.{h,cpp}` | `McscfState`, root tracking (Hungarian assignment), SA helpers, candidate assembly |
| `casscf/orbital.{h,cpp}` | Fock builds, orbital gradient, `OccupiedHalfTransform`, analytic orbital Hessian action |
| `casscf/response.{h,cpp}` | CI response, orbital-CI coupling, the matrix-free CI-relaxed Newton solve |
| `casscf/aug-hessian{,-orbital}.{h,cpp}` | Generic augmented-Hessian Davidson (also used by SCF SOSCF) |

<div align="justify">

Each energy evaluation (`evaluate` in `casscf.cpp`) builds the half-transformed integrals `H = (μν|λi)`, i over core and active MOs, once at the trial orbitals. F_I, F_A, (tu|vw) and the `(pu|vw)` cache are O(nb³) contractions of it (`occupied_two_electron_ao`, `transform_occupied_half`), because every density involved is `X Yᵀ + Y Xᵀ` with X the occupied columns of C. The orbital step is a Newton step on the exact CI-relaxed orbital Hessian (analytic fixed-CI action plus per-root CI response with both orbital-CI coupling terms), solved matrix-free by a preconditioned subspace solve with an inexact-Newton stop and a fixed seed vector (so it still sees negative modes the gradient is orthogonal to). Steps are scaled by a trust region driven by the exact model; the Newton step is taken as soon as it improves, otherwise further candidates and stagnation probes are scored by the merit `E + 0.1‖g‖²`. RHF and ROHF references are accepted (unpaired electrons must sit in the active space). RAS constraints act on the determinant strings only. RI runs keep the dense-density builds and a finite-difference Hessian. See `docs/CASSCF_PERFORMANCE.md`.

</div>

### Coupled cluster (`cc/`, namespace `HartreeFock::Correlation::CC`)

| Entry | Path |
|---|---|
| `run_rccsd`, `run_uccsd` (`ccsd.{h,cpp}`) | Hand-written CCSD; UCCSD and small cases via the determinant-space solver |
| `run_rccsdt`, `run_uccsdt` (`ccsdt.{h,cpp}`) | RCCSDT routes to the determinant prototype (small: `nso ≤ 16` and `ndet ≤ 10000`), the hand-written tensor backend (`tensor_backend*.cpp`), or the generated harness; `PLANCK_RCCSDT_BACKEND` overrides |
| `run_rccgen` (`rccgen.{h,cpp}`) | Generated arbitrary-order RCC (`cc2` … `cc6`, `ccsdt_gen`, RCCSDTQ) through `solver_arbitrary`, `generated_arbitrary_{prepare,runtime}` and `generated_kernel_registry` |
| `run_uccgen` (`uccgen.cpp`, `ucc_blocks.cpp`) | Generated unrestricted CC (`ucc2` … `ucc6`), `PLANCK_CC_UCC=ON` builds only |

<div align="justify">

Supporting files: `common.{h,cpp}` (tensor types with inlined row-major accessors, RHF/UHF references), `amplitudes.{h,cpp}` (amplitude and denominator packs, spin-orbital → spatial projection), `mo_blocks.{h,cpp}` (MO integral blocks), `diis.{h,cpp}` (amplitude DIIS), `determinant_space.{h,cpp}` (exact-in-the-space determinant CC for validation), `tensor_optimized.{h,cpp}`, and `cc_amplitude_checkpoint.{h,cpp}` (the `.ccamp` amplitude sidecar for generated RCC and UCC restarts, kept separate from `.hfchk`). Generated kernels index amplitudes `(occ…, virt…)`; ccgen's own fixtures are `(virt…, occ…)`. All CC references are canonical (`f_ov = 0`). See the `docs/CCGEN_*.md` answers.

</div>

---

## Analytic Gradients (`src/gradient/`)

<div align="justify">

`gradient.{h,cpp}` assembles nuclear gradients for RHF, UHF, ROHF, RKS, UKS (with the HF-like exchange of hybrids and range-separated hybrids), and, with `mp2_gradient.cpp`, RMP2 and UMP2. The one-electron part (Hellmann-Feynman plus Pulay with the energy-weighted density W) runs in one threaded block; the two-electron part for every reference goes through one loop, `two_electron_gradient.h`: the caller supplies Γ, the loop visits the `q ≥ p` shell-pair triangle, evaluates derivative blocks through the Auto derivative chain, and sums 64 fixed bins in order (thread-count invariant).

ROHF uses `W = Pα Fα Pα + Pβ Fβ Pβ`, built in the SCF's own basis and lifted to Cartesian once (lifting the factors separately is wrong because the cart←sph transform is not square). Spherical gradients for RHF/UHF/ROHF lift P and W at the skin; spherical MP2 gradients are rejected. See `docs/GRADIENT_INTEGRAL_PERFORMANCE.md`.

</div>

---

## Geometry Optimization and Frequencies (`src/opt/`, `src/freq/`)

<div align="justify">

`run_geomopt` is Cartesian L-BFGS with backtracking; `run_geomopt_ic` is BFGS in redundant internal coordinates (`intcoords.{h,cpp}`: bonds from covalent radii, bends, torsions, Wilson B matrix, iterative back-transform, constraints for bonds, angles, dihedrals and frozen atoms). Both take the energy-and-gradient as a callback, so any method with a gradient can be optimized. `compute_hessian` (`hessian.{h,cpp}`) builds the semi-numerical Hessian by central differences of analytic gradients (parallel over displacements), and `vibrational_analysis` mass-weights, projects out translations and rotations, and labels modes by symmetry. `ImaginaryFollow` displaces along the largest imaginary mode and re-optimizes (HF driver only). Each displaced geometry rebuilds the basis-dependent state through `rebuild_basis_dependent_state`.

</div>

---

## Kohn-Sham DFT (`src/dft/`)

| Area | Files |
|---|---|
| Driver | `driver.{h,cpp}` (`DFT::Driver::prepare` / `run`): KS-SCF, hybrids, PCM, TDDFT, SOSCF, double hybrids, gradient and workflow dispatch; `main.cpp` |
| Grid | `base/grid.h`, `radial.h` (radial counts from PySCF's period × level table, Treutler-Ahlrichs mapping), `angular.h` (Lebedev), Becke partition; versioned pruning policies (`legacy`, `g4-3-v4`) |
| libxc | `base/wrapper.h`: functional metadata (hybrid, range-separated, double hybrid, coefficients) |
| AO and XC on the grid | `ao_grid.h`, `xc_grid.{h,cpp}` (density and libxc evaluation in disjoint slabs, no cross-thread reduction), `ks_matrix.{h,cpp}` |
| Gradients | `dft_gradient.{h,cpp}`: XC nuclear gradient including Becke-weight motion |
| Analytic XC Hessian action | `analytic_hessian.{h,cpp}`, `ks_orbital_hessian.{h,cpp}`: fxc products for SOSCF (LDA, GGA, polarized) |
| TDDFT | in `driver.cpp`; `tddft_exchange.h` owns the exact-exchange response blocks (full-range and erf short-range) |
| Double-hybrid gradients | RKS: `dh_pt2_gradient.{h,cpp}`, `dh_zvector.h`, `response_gmres.h` (matrix-free GMRES Z-vector, streamed pair density); UKS: `udh_*.cpp` (dense implementation, pinned V4 grid) |
| Helpers | `response_packing.{h,cpp}`, `uks_level_shift.{h,cpp}`, diagnostic probes (`dh_hessian_probe.*`, `dh_probe_gmres.h`) |

<div align="justify">

The KS Coulomb and exact-exchange matrices are built memory-direct through the same fused Fock loop as HF (`_compute_2e_j_direct`, `_compute_2e_k_direct`), with unscaled K; each caller applies its own exchange prefactor. Range-separated hybrids add `ShortRange`/`LongRange` kernel builds. Double hybrids run the KS SCF and then a non-self-consistent MP2 on the KS orbitals. Spherical basis sets and meta-GGAs are rejected in `planck-dft` today (see `vault/Status/Open Work.md`).

</div>

---

## Entry Points and Data Flow

<div align="justify">

`main` in each binary first initializes the Boys library (licence check, fetch, decrypt, load), prints the banner and external-library table, parses arguments (`<input> [--json <path>]`) and the input, and hands the `Calculator` to `HartreeFock::Driver::run` (`hf_driver.cpp`) or `DFT::Driver::run`. `planck-mpi` does the same and picks the driver from the input.

</div>

```
input file
  └─ IO::parse_input()                → Calculator (options, molecule)
       └─ detectSymmetry()            → _standard (Bohr, reoriented), point group
            └─ read_gbs_basis()       → Basis (shells, _index, cart→sph)
                 └─ build_shellpairs, 1e integrals, SAO basis, ERI (conventional)
                      └─ run_rhf / run_uhf / run_rohf   (or DFT::Driver::run)
                           ├─ properties, populations, PCM, FCIDUMP, checkpoint
                           ├─ Gradient          → compute_*_gradient
                           ├─ GeomOpt           → run_geomopt / run_geomopt_ic
                           ├─ Frequency         → compute_hessian → vibrational_analysis
                           ├─ RMP2 / UMP2       → rmp2_kernel / ump2_kernel
                           ├─ CC                → run_rccsd[t] / run_uccsd[t] / run_rccgen / run_uccgen
                           ├─ CASSCF / RASSCF   → run_casscf / run_rasscf
                           ├─ FCI / FCIQMC      → run_fci / run_fciqmc
                           └─ counterpoise      → BSSE::run_counterpoise
```

---

## Determinism

<div align="justify">

Every parallel reduction in the tree is fixed-order: work is partitioned by a fixed function of the work index (never by thread id, never with dynamic scheduling into a shared accumulator), and partial results are summed in a fixed order. This covers the fused Fock loop, the gradient loop, the CI sigma build, the RDM builds, the DFT grid, the transforms and MPI reductions. Results are therefore bitwise identical at any thread or rank count, and regression pairs (`*_threads1/4`) hold several paths there. Merging per-thread partials under `omp critical` or with `omp atomic` reintroduces run-to-run differences and is not used.

</div>

---

## Key Invariants and Pitfalls

- **`_standard` must be in Bohr** before basis construction, nuclear repulsion and any integral. In DFT, also keep `_coordinates` in the same frame (`sync_coordinate_frames_from_standard`).
- **`ContractedView._index`** is the AO position used for placement; never recover AO positions from a shell pair's vector index (`invert_pair_index` is not valid for multi-function shells).
- **The contracted norm is folded into `Shell._coefficients`**; never apply it again. Primitive norms are handled inside the recurrences.
- **Spherical runs:** integral code is Cartesian (`nbasis()`); SCF state is sized by `working_nbasis()`. Rebuild `_cart_to_sph` normalization at every new geometry.
- **ROHF energy-weighted density** is `Pα Fα Pα + Pβ Fβ Pβ`, built in the working basis and lifted once.
- **Stored orbital energies at convergence** come from the true Fock, not the DIIS-extrapolated one; gradients depend on it.
- **K from the shared builders is unscaled** in DFT; the ½ (RKS) or 1 (UKS) and the exchange coefficient are applied by the caller.
- **`ActiveIntegralCache::puvw`** is row-major `(p,u,v,w)`; `compute_Q_matrix` assumes contiguous `n_act³` slabs.
- **Generated CC** needs `PLANCK_CC_SPIN_ADAPT=ON`; the OFF emit gives wrong correlation energies. Cases that depend on build options declare `requires_build_option` in `tests/regression_cases.json`.
- **Saddle points:** a symmetric start can sit on a stationary point whose negative mode the gradient cannot see. Iterative solvers seeded only from the gradient cannot leave it; that is why the CASSCF solve carries a seed vector.
- **Release builds define `NDEBUG`**, so `assert`-only tests do not run their checks there.
