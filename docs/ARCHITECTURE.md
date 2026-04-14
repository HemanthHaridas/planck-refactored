# Planck — Architecture Reference

<div align="justify">

Planck is a C++23 quantum chemistry engine implementing Hartree-Fock SCF, post-HF correlation, and Kohn-Sham DFT from first principles. The codebase is organized around two standalone binaries — `hartree-fock` and `planck-dft` — that share a common library of integrals, basis-set handling, symmetry, I/O, and geometry optimization. Every public interface propagates errors through `std::expected<T, std::string>` rather than exceptions, making failure paths explicit and composable throughout the call graph.

</div>

---

## Build System (`CMakeLists.txt`)

<div align="justify">

The project uses CMake 3.5+ with C++23 required and extensions disabled. Two primary executables are defined alongside an optional utility binary:

</div>

| Binary | Purpose |
|---|---|
| `hartree-fock` | RHF/UHF SCF, MP2, CASSCF/RASSCF, analytic gradients, geometry optimization, frequencies |
| `planck-dft` | Kohn-Sham DFT (RKS/UKS) using the same SCF infrastructure plus libxc and a numerical grid |
| `chkdump` | Checkpoint file inspector (optional, `BUILD_TOOLS=ON` by default) |

<div align="justify">

Dependencies are acquired at configure time through two mechanisms. **Eigen 3.4.0** is fetched via `FetchContent` and is header-only — it is never compiled separately. **libmsym** (point-group detection) and **libxc** (exchange-correlation functionals) are built as static archives via `ExternalProject_Add` and linked directly. This keeps all three dependencies hermetic: no system-level installs are needed, and the build is fully reproducible across machines.

Source files are collected with `file(GLOB ...)` per module directory (`BASE_SRC`, `IO_SRC`, `LOOKUP_SRC`, `SYMM_SRC`, `BASIS_SRC`, `INT_SRC`, `SCF_SRC`, `POSTHF_SRC`, `GRADIENT_SRC`, `OPT_SRC`, `FREQ_SRC`). `planck-dft` adds `DFT_SRC` and links `libxc.a`; `hartree-fock` does not. Both binaries compile with `USE_OPENMP` enabled by default — if `find_package(OpenMP)` succeeds, `OpenMP::OpenMP_CXX` is linked and `USE_OPENMP` is defined, which activates `#pragma omp parallel` regions in the ERI inner loops and in the finite-difference Hessian. CUDA acceleration is opt-in (`-DUSE_CUDA=ON`) and routes through a separate `gpu/` subdirectory that is not documented here. Basis sets are installed under `share/basis-sets/`; the configure step writes `src/base/basis.h` from `basis.h.in` so that `get_basis_path()` resolves at runtime without hard-coded paths.

</div>

**Trade-off:** `file(GLOB ...)` does not automatically detect new files added after `cmake ..` without re-running CMake. This is a deliberate ergonomic choice — the glob strategy avoids maintaining explicit file lists and is acceptable because source-file additions are comparatively rare and always require a reconfigure anyway.

---

## Central Type System (`src/base/`)

<div align="justify">

`types.h` is the single header that every other module includes. It defines all data-carrying structs and enums in the `HartreeFock` namespace. No business logic lives here — only data layout and a handful of trivial invariant-maintaining methods.

</div>

### Enumerations

<div align="justify">

All user-visible options are encoded as scoped enums, preventing accidental integer misuse and making `switch` exhaustiveness errors visible at compile time.

</div>

| Enum | Values |
|---|---|
| `ShellType` | `S(0)`, `P(1)`, `D(2)`, `F(3)`, `G(4)`, `H(5)` |
| `SCFType` | `RHF`, `UHF` |
| `PostHF` | `None`, `RMP2`, `UMP2`, `CASSCF`, `RASSCF` |
| `CalculationType` | `SinglePoint`, `Gradient`, `GeomOpt`, `Frequency`, `GeomOptFrequency`, `ImaginaryFollow` |
| `SCFMode` | `Conventional`, `Direct`, `Auto` |
| `IntegralMethod` | `ObaraSaika`, `RysQuadrature`, `Auto` |
| `SCFGuess` | `HCore`, `SAD`, `ReadDensity`, `ReadFull` |
| `OptCoords` | `Cartesian`, `Internal` |
| `DFTGridQuality` | `Coarse`, `Normal`, `Fine`, `UltraFine` |
| `XCExchangeFunctional` | `Custom`, `Slater`, `B88`, `PW91`, `PBE` |
| `XCCorrelationFunctional` | `Custom`, `VWN5`, `LYP`, `P86`, `PW91`, `PBE` |

### `Molecule`

<div align="justify">

Holds all per-molecule data. The coordinate layout follows a strict three-frame convention:

</div>

- `coordinates` — raw input geometry, in Angstrom as given by the user
- `_coordinates` — same geometry converted to Bohr by `prepare_coordinates()`
- `_standard` — symmetry-reoriented geometry in Bohr, set by `detectSymmetry()` (or equal to `_coordinates` when symmetry is disabled)

<div align="justify">

**Critical invariant:** All downstream code — basis set construction, nuclear repulsion, integral evaluation, and gradients — consumes `_standard`. The `_angstrom_to_bohr()` helper only converts `coordinates → _coordinates`; it does not touch `_standard`. This separation means symmetry reorientation is transparent to all downstream modules.

</div>

### `Shell` and `ContractedView`

<div align="justify">

`Shell` stores the raw data for one contracted Gaussian shell: center (Bohr), shell type, primitive exponents `_primitives`, contracted coefficients `_coefficients`, and per-primitive normalizations `_normalizations`. The contracted norm `Nc` is folded into `_coefficients` during GBS file parsing — there is no separate `Nc` factor anywhere in the integral code.

`ContractedView` is a lightweight span-based reference into a shell. It adds a Cartesian angular-momentum tuple `_cartesian = (lx, ly, lz)`, a component norm `_component_norm = 1/sqrt((2lx-1)!! (2ly-1)!! (2lz-1)!!)`, and crucially `_index` — the position of this basis function in `Basis::_basis_functions`. This index is what the integral engine uses to place each computed integral into the correct matrix element.

</div>

**Key invariant:** `ContractedView._index` must be set correctly for every basis function. The OS integral engine uses it directly to index into output matrices; an incorrect or default-zero index silently misplaces all integrals for that function.

### `Basis`

<div align="justify">

Holds `_shells` (one entry per contracted shell) and `_basis_functions` (one `ContractedView` per Cartesian basis function — multiple entries per shell for angular momentum L≥1). `nbasis()` returns the number of Cartesian basis functions; `nshells()` returns the number of shells.

</div>

### `ShellPair` and `PrimitivePair`

<div align="justify">

`ShellPair` is computed once before SCF and cached. Its constructor evaluates all `nA × nB` primitive-pair Gaussian products: the product center `P`, displacement vectors `pA = P − A` and `pB = P − B`, the exponential prefactor, and the product of contracted coefficients and normalizations. The Schwarz screening value is also pre-computed. `build_shellpairs` in `shellpair.h` generates all unique `(i, j)` pairs with `i ≥ j` in row-major order. To recover the shell indices from a `ShellPair`, use `sp.A._index` and `sp.B._index` directly — do not call `invert_pair_index` on a shell-pair position.

</div>

### `DIISState`

<div align="justify">

Implements Pulay's Direct Inversion in the Iterative Subspace inline in `types.h`. Each iteration `push(F, e)` appends the current Fock matrix and DIIS error `e = X^T (FPS − SPF) X` to a `std::deque` capped at `max_vecs` (default 8). `extrapolate()` solves the augmented linear system `B c = [0…0, −1]^T` where `B_{ij} = Tr(e_i^T e_j)` and returns the linear combination of Fock matrices minimizing the error norm subject to `Σ c_i = 1`. The DIIS restart factor in `OptionsSCF` causes the subspace to be cleared when the error grows by more than a threshold, preventing stagnation.

</div>

### `DataSCF` and `SpinChannel`

<div align="justify">

`SpinChannel` holds the density matrix, Fock matrix, MO energies, MO coefficients, and MO symmetry labels for one spin. `DataSCF` aggregates an alpha and a beta channel; for RHF the beta channel is allocated but unused. `Calculator::initialize()` allocates all matrices to `nbasis × nbasis` zero matrices after the basis is loaded.

</div>

### `Calculator`

<div align="justify">

The root aggregate. Every module receives a `Calculator &` and reads/writes its fields. The public fields include all `Options*` structs, the `Molecule`, the `Basis`, the SCF data (`_info._scf`), the stored ERI tensor (`_eri`), the overlap and core Hamiltonian matrices, CASSCF results, gradient, Hessian, frequencies, and SAO blocking data. For SA-CASSCF runs, `_cas_root_energies` stores per-root total CASSCF energies (not bare CI eigenvalues), so the final root table can be averaged directly with the configured SA weights. `_compute_nuclear_repulsion()` uses `_molecule._standard` — it must be called after `_standard` is populated in Bohr.

</div>

---

## Input/Output (`src/io/`)

### Parser — `io.h` / `io.cpp`

<div align="justify">

The input format is section-based. `_split_into_sections()` reads the file and builds a `SectionMap` (`unordered_map<string, vector<string>>`) keyed on section headers (`%begin_control`, `%begin_scf`, etc.). Each section parser receives its corresponding line vector and writes into the appropriate `Options*` struct. All section parsers return `std::expected<void, std::string>`, propagating errors to `parse_input()` which assembles the complete `Calculator`. Constraints are parsed from an optional `%begin_constraints` block into `vector<GeomConstraint>`.

**Design trade-off:** A single flat section map is simpler than a recursive grammar but requires every section name to be globally unique. This is acceptable for the modest vocabulary of a quantum chemistry input file.

</div>

### Logger — `logging.h`

<div align="justify">

A header-only namespace with a global `std::mutex log_mutex` and `thread_local int silence_depth`. The `ScopedSilence` RAII guard increments `silence_depth` on construction and decrements on destruction; all logging functions return immediately when `silence_depth > 0`. This mechanism is used during gradient finite-difference steps (the inner SCF runs silently) and during the SAD guess (atomic SCF runs are silenced). `map_enum<T>` specializations for all enum types live here, converting enum values to human-readable strings for log output.

</div>

### Checkpoint — `checkpoint.h` / `checkpoint.cpp`

<div align="justify">

Binary checkpoint files use the magic bytes `"PLNKCHK\0"` and a versioned layout (currently v6). The format stores everything needed to restart or continue a calculation: geometry in the standard frame (Bohr), basis shell data, SCF matrices, CASSCF MOs, and active natural occupations. Three restart modes exist:

</div>

1. **`ReadDensity`** — loads density/Fock/MOs only; geometry comes from the input file. Use when re-running the same system in a different basis or with different options.
2. **`ReadFull`** — `load_geometry()` is called first (before basis/symmetry setup) to restore the optimized geometry, charge, and multiplicity; then `load()` supplies the density. Use to restart from a converged geometry optimization.
3. **Basis projection** — `load_mos()` reads MO coefficients without enforcing `nbasis` agreement; `project_density()` then uses SVD Löwdin projection via the cross-overlap matrix to produce a starting density in the larger basis. Use to bootstrap a larger-basis calculation from a smaller-basis checkpoint.

<div align="justify">

**Trade-off:** The versioned binary format is compact and fast but not human-readable. The `chkdump` utility provides inspection. Each version increment is additive (old files remain loadable), but the reader must handle `has_*` flags that indicate whether optional blocks are present.

</div>

---

## Element Data (`src/lookup/`)

<div align="justify">

`elements.h` / `elements.cpp` provide lookup tables for atomic numbers, symbols, masses, and covalent radii. Used by the input parser to map element symbols to atomic numbers, by the SAD guess to identify atomic occupations, and by the geometry optimizer to infer connectivity for internal coordinate generation.

</div>

---

## Basis Set (`src/basis/`)

### Normalization — `basis.cpp`

<div align="justify">

`primitive_normalization(L, exponents)` computes the per-primitive normalization `N_i = (2α/π)^{3/4} (4α)^{L/2} / sqrt((2L-1)!!)` for each primitive in the shell. `contracted_normalization(L, exponents, coefficients, prim_norms)` computes the contracted-shell norm `Nc` such that `⟨χ|χ⟩ = 1`. The norm `Nc` is immediately multiplied into `Shell._coefficients`; downstream code never applies a separate `Nc` factor. `_cartesian_shell_order(L)` generates the canonical ordering of Cartesian angular-momentum tuples `(lx, ly, lz)` with `lx + ly + lz = L`, matching the convention used by Gaussian94 GBS files.

</div>

### GBS Reader — `gaussian.cpp`

<div align="justify">

`read_gbs_basis()` parses Gaussian94 `.gbs` format files. For each atom in the molecule it locates the matching element block, reads all shells, computes primitive and contracted normalizations, and populates `Shell._center` from `molecule._standard` (Bohr). After reading, it builds `Basis._basis_functions` by calling `_cartesian_shell_order` and stamping `ContractedView._index` with the running basis-function counter.

</div>

---

## Symmetry (`src/symmetry/`)

### Point-group detection — `symmetry.h` / `symmetry.cpp`

<div align="justify">

`detectSymmetry(molecule)` wraps libmsym via RAII `SymmetryContext` and `SymmetryElements` classes. libmsym identifies the molecular point group, returns the symmetry-adapted coordinates in the standard frame (`_standard`), and provides symmetry operations expressed as permutations of atoms. The detected point group string is stored in `molecule._point_group`. When symmetry is disabled, `_standard` is set to `_coordinates` unchanged.

</div>

### SAO blocking — `mo_symmetry.h`

<div align="justify">

`build_sao_basis()` assembles the symmetry-adapted AO basis. The SAO transform matrix `_sao_transform` is a unitary matrix `U` that block-diagonalizes the overlap and Fock matrices by irrep. During SCF, `run_rhf` / `run_uhf` use per-irrep diagonalization instead of a global eigensolve, ensuring that the resulting MOs carry clean irrep labels (`SpinChannel.mo_symmetry`). The irrep-per-column arrays `_sao_irrep_index` and `_sao_irrep_names` allow the CASSCF active-orbital picker to filter MOs by symmetry.

</div>

### Integral symmetry — `integral_symmetry.h`

<div align="justify">

`build_integral_symmetry_ops()` generates `SignedAOSymOp` records — permutation/phase tables on AO indices — for each symmetry operation of the molecular point group. When `_use_integral_symmetry` is active, the OS integral engine uses these to skip symmetry-equivalent shell quartets, reducing the ERI cost by approximately the order of the point group.

</div>

---

## Integrals (`src/integrals/`)

### Shell pairs — `shellpair.h`

<div align="justify">

`build_shellpairs(basis)` iterates over all unique `(i, j)` basis-function pairs with `i ≥ j` and constructs a `ShellPair` for each. The shell-pair index is `pair_index(i, j) = i(i+1)/2 + j`. `invert_pair_index(k)` inverts this mapping arithmetically, but to recover shell indices from a `ShellPair` in the ERI loop it is always better to read `sp.A._index` and `sp.B._index` directly from the stored references.

</div>

### Obara-Saika engine — `os.h` / `os.cpp`

<div align="justify">

The primary integral engine implements the Obara-Saika vertical and horizontal recurrence relations (VRR+HRR). It provides:

</div>

- `_compute_1e(shell_pairs, nbasis, sym_ops)` — overlap `S` and kinetic `T` matrices in one pass
- `_compute_nuclear_attraction(shell_pairs, nbasis, molecule, sym_ops)` — nuclear attraction `V` using the Boys function
- `_compute_2e(shell_pairs, nbasis, tol_eri, sym_ops)` — full AO ERI tensor with Schwarz screening; returns a flat `vector<double>` in chemist's notation `(μν|λσ)`
- `_compute_2e_fock(shell_pairs, density, nbasis, tol_eri, sym_ops)` — direct-SCF Fock contribution builder; it avoids storing the tensor on `Calculator`, but currently builds a local screened ERI tensor before the density contraction
- `_compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, tol_eri, sym_ops)` — UHF direct-SCF variant that builds one local spin-independent ERI tensor and contracts alpha/beta Fock contributions together
- `_compute_eri_deriv_elem(spAB, spCD)` — 12-element gradient of a contracted ERI quartet used by analytic gradient code
- `_compute_1e_deriv_A(sp)`, `_compute_nuclear_deriv_A_elem(sp, mol)`, `_compute_nuclear_deriv_C_elem(sp, C, Z, dir)` — 1e gradient derivatives
- `_compute_multipole_matrices(shell_pairs, nbasis, origin)` — dipole and quadrupole integral matrices
- `_compute_cross_overlap(large_basis, small_basis)` — cross-basis overlap for Löwdin basis-set projection

<div align="justify">

The four-center OS kernel uses a thread-local `EriScratch` object. Its `vrr` and `hrr` vectors are resized to the actual angular momentum extents of each primitive quartet, replacing the previous fixed worst-case per-thread VRR/HRR arrays. This keeps OpenMP scratch private without paying the memory cost of gigabyte-scale scratch buffers on every thread.

The Schwarz screening criterion discards quartets where `Q(AB) × Q(CD) < tol_eri`. `Q(AB) = sqrt((AB|AB))` is pre-computed per shell pair during ERI construction. The OS Schwarz table also respects AO integral symmetry operations by evaluating a canonical pair and assigning the same bound across its symmetry orbit. The stored-ERI and local-ERI Fock builders iterate only canonical pair quartets, scatter all eight ERI permutation slots, and use OpenMP atomic writes in `write_eri_permutations(...)` so overlapping permutation fills are race-free.

</div>

### Rys quadrature engine — `rys.h` / `rys.cpp`

<div align="justify">

The Rys backend provides the same public ERI and Fock-building surface as the OS engine: `_compute_2e`, `_compute_2e_fock`, `_compute_2e_fock_uhf`, plus `_auto` variants. At the primitive level, `_rys_eri_primitive(...)` converts the Boys-function dependence into Rys roots and weights, builds three 1D VRR tables per root, accumulates the root-summed six-dimensional spatial buffer `_rys_sum_buf`, and then applies AB/CD HRR to obtain the requested Cartesian quartet. `_rys_contracted_eri(...)` sums the primitive-pair products.

Root generation lives in `rys_roots.cpp`. The one-root case uses an exact closed form; multi-root cases build a Stieltjes-Jacobi recurrence from long-double Boys moments and diagonalize the Jacobi matrix with Eigen. The Gauss-Legendre tables are retained as a numerical fallback if the recurrence or diagonalization degenerates.

The Rys stored-ERI builders mirror the OS loop shape: build a Schwarz table, iterate pair quartets with `p <= q`, screen on `Q(AB) × Q(CD)`, compute the contracted ERI, and scatter the eight permutation-equivalent tensor slots. The Rys API accepts `sym_ops` for signature parity with OS, but quartet-orbit pruning from AO integral symmetry is currently only applied in the OS backend.

</div>

### Engine dispatch — `base.h`

<div align="justify">

`IntegralMethod` selects `ObaraSaika`, `RysQuadrature`, or `Auto` in `integrals/base.h`. `Auto` dispatches at the contracted quartet level through `_auto_contracted_eri(...)`: it estimates OS and Rys work in `_auto_prefers_rys(...)` and chooses Rys only when the Rys estimate is cheaper. The public `RYS_CROSSOVER_L = 4` constant documents the intended high-angular-momentum crossover, but the actual `Auto` branch is cost-model based rather than a hard `L >= 4` rule. In practice, `ObaraSaika` remains the default engine and reference path.

</div>

---

## SCF (`src/scf/`)

<div align="justify">

The SCF module implements `run_rhf()` and `run_uhf()`. Both functions share the same structural loop but differ in how the Fock matrix is built — RHF uses a single total density; UHF maintains separate alpha and beta densities and builds `G_alpha` and `G_beta` independently.

</div>

### SCF loop (conceptual)

<div align="justify">

The SCF loop (implemented in `src/scf/`) follows the standard algorithm:

1. Form initial density from core Hamiltonian diagonalization (H-core guess) or SAD guess
2. Each iteration:
   1. Form the two-electron Fock contribution G (conventional: from stored ERI; direct: recompute)
   2. Build F = H_core + G
   3. Compute DIIS error `e = X^T (F P S - S P F) X` and push to `DIISState`
   4. If DIIS is ready, replace F with the extrapolated Fock
   5. Transform to orthogonal basis: F' = X^T F X
   6. Diagonalize F' → C', ε
   7. Back-transform: C = X C'
   8. Form new density P = 2 C_occ C_occ^T
   9. Check convergence on ΔE and ΔP; if converged, break

For SAO-blocked SCF, step 2.6 is replaced by block-diagonal diagonalization in the symmetry-adapted basis, guaranteeing MOs with clean irrep labels.

</div>

### Conventional vs. direct SCF

<div align="justify">

In `Conventional` mode, `_compute_2e()` builds the full ERI tensor once before the SCF loop. Every iteration reads from the stored tensor in O(N^4) memory. In `Direct` mode, `_compute_2e_fock()` recomputes the two-electron contribution from shell pairs on every iteration, using O(N^2) memory at the cost of repeated integral evaluation. `Auto` mode selects `Conventional` when `nbasis ≤ threshold` (default 100) and `Direct` otherwise. This threshold is tunable via `OptionsSCF._threshold`.

**Trade-off:** Conventional SCF is faster for small systems (cache-friendly reads of the ERI array dominate) but becomes memory-prohibitive at ~250+ basis functions. Direct SCF has a higher per-iteration cost but constant memory.

</div>

### SAD guess

<div align="justify">

The Superposition of Atomic Densities guess runs a minimal-basis atomic SCF for each unique element in the molecule (with all output silenced via `ScopedSilence`). The resulting atomic densities are superimposed to form the initial molecular density. This provides a substantially better starting point than the H-core guess for transition metals and multi-reference systems.

</div>

---

## Post-HF Correlation (`src/post_hf/`)

### MP2 — `mp2.h` / `mp2.cpp`

#### RMP2 (`run_rmp2`)

<div align="justify">

The restricted MP2 energy is evaluated as:

</div>

```
E_MP2 = Σ_{i<j, a<b} |(ia|jb) - (ib|ja)|² / (ε_i + ε_j - ε_a - ε_b)
```

<div align="justify">

The AO→MO transformation uses a quarter-transform strategy. If `_eri` is populated from a conventional SCF, it is reused directly; otherwise the ERI tensor is rebuilt from shell pairs. The generic transform entry points in `src/post_hf/integrals.cpp` remain the shared path for MP2 and for the fully internal CASSCF integral builds. RMP2 natural orbitals are available through `compute_rmp2_natural_orbitals()`, which diagonalizes the MP2 one-particle density matrix.

</div>

#### UMP2 (`run_ump2`)

<div align="justify">

Unrestricted MP2 maintains separate α-α, β-β, and α-β orbital-pair contributions. The quarter-transform operates independently on the alpha and beta MO spaces.

</div>

### CASSCF / RASSCF — `casscf.h` and `casscf/`

<div align="justify">

`run_casscf()` and `run_rasscf()` are thin wrappers around a shared MCSCF driver that alternates CI diagonalization and orbital optimization until both the energy and orbital gradient converge. The active-space specification in `OptionsActiveSpace` controls the number of active electrons, active orbitals, number of CI roots (state averaging), SA weights, RAS partitioning, and convergence thresholds.

</div>

#### CI strings — `casscf_internal.h` / `casscf/strings.h`

<div align="justify">

Determinants are represented as `uint64_t` bitmasks (`CIString`). Each bit position corresponds to one active spatial orbital. Separate alpha and beta strings are enumerated by `generate_strings(n_orb, n_occ)` using fixed-popcount bitstring enumeration in ascending order. The `kMaxSeparateSpinOrbitals = 63` limit on the active space (64-bit type minus the sign bit) sets the hard ceiling on the active-orbital count. For RASSCF, `RASParams` encodes the three-space partition (RAS1/RAS2/RAS3) and screening caps (`max_holes`, `max_elec`); `ras1_holes()` and `ras3_elec()` enforce these constraints during determinant filtering.

</div>

#### CI Hamiltonian — `casscf/ci.h` / `ci.cpp`

<div align="justify">

`CIDeterminantSpace` bundles the filtered determinant list, the diagonal, a packed key array, a lookup table, and optionally the full dense Hamiltonian matrix. Matrix elements are computed by `slater_condon_element()` using the Slater-Condon rules applied to the bitmask strings. `build_ci_space()` assembles the space and applies the density threshold `dense_threshold` (default 500) to decide whether to materialize the full matrix or use iterative sigma application via `apply_ci_hamiltonian()`. `solve_ci()` is overloaded for the dense path (direct `SelfAdjointEigenSolver`) and the iterative path (Davidson method with a diagonal preconditioner).

</div>

#### Reduced density matrices — `casscf/rdm.h` / `rdm.cpp`

<div align="justify">

`compute_1rdm()` and `compute_2rdm()` build the state-averaged active-space 1-RDM and 2-RDM by traversing all determinant pairs and accumulating contributions weighted by the SA weights. `compute_2rdm_bilinear()` computes the bilinear form `Γ_{pq,rs} = Σ_I w_I ⟨Ψ^bra_I | â_p† â_r â_s â_q | Ψ^ket_I ⟩` needed by the CI response solver. Reference implementations (`_reference` suffix) use explicit spin-orbital traversal and serve as validation targets.

</div>

#### Orbital optimization — MCSCF loop (`casscf/casscf.h`)

<div align="justify">

The macro/micro iteration structure alternates: (1) CI solve to get CI vectors and state-averaged energy, (2) 1-RDM and 2-RDM construction, (3) orbital gradient and approximate Hessian computation, (4) orbital rotation by a Cayley-transformed antisymmetric matrix `κ`. The micro iterations apply a matrix-free orbital Hessian action to the orbital gradient via the Lagrangian formulation, with a diagonal preconditioner fallback and a small-space finite-difference Newton escape hatch. Convergence is assessed on both ΔE and the state-averaged orbital-gradient screen. Hungarian root tracking ensures consistent state labeling across macro iterations when multiple roots are computed.

The orbital-gradient and CI-response path relies on a mixed-basis active-integral cache built by `build_active_integral_cache(...)` in `casscf/orbital.cpp`. That cache is produced by the dedicated `transform_eri_active_cache(...)` kernel in `src/post_hf/integrals.cpp`, not by the fully generic four-leg AO→MO transform. The cached tensor is stored row-major as `(p,u,v,w)`, with one full-space MO index `p` and three active-space indices `u,v,w`. Each fixed-`p` slab is contiguous, so the builder can parallelize safely over `p` with OpenMP and reuse thread-local scratch buffers across repeated macro-iteration rebuilds.

On the consumer side, `compute_Q_matrix(...)` contracts the cached `(p,u,v,w)` slabs with the active-space 2-RDM to form the Q matrix used in the orbital gradient and response equations. Because both the `puvw` cache and `Γ[t,u,v,w]` are traversed as contiguous `n_act^3` slabs, the hot contraction path reduces to repeated dot products rather than repeated four-index address arithmetic.

</div>

**Trade-off:** The matrix-free Hessian avoids explicitly forming the O(N^4) orbital Hessian tensor but requires a CI response solve at each micro step. This is the correct asymptotic choice for large active spaces. For small active spaces (≤10 orbitals), the dense path may be faster in practice.

---

## Coupled Cluster (`src/post_hf/cc/`)

<div align="justify">

All coupled-cluster code lives under `src/post_hf/cc/` in namespace
`HartreeFock::Correlation::CC`. Four solver paths are available:
`run_rccsd`, `run_uccsd`, `run_rccsdt`, and `run_uccsdt`. Every path follows
the same two-phase `prepare` / `run` API: `prepare_X` validates the SCF
result, transforms integrals, and allocates amplitude tensors; `run_X` calls
`prepare_X` internally and then drives the iterative loop to convergence.

</div>

### File map

| File | Owns |
|------|------|
| `common.h` / `common.cpp` | Tensor types (`Tensor2D/4D/6D`), `RHFReference`, `UHFReference`, reference builders |
| `amplitudes.h` / `amplitudes.cpp` | `DenominatorCache`, `RCCSDAmplitudes`, `RCCSDTAmplitudes`, zero-amplitude factories |
| `mo_blocks.h` / `mo_blocks.cpp` | `MOBlockCache`, AO→MO four-index transform for the teaching solver |
| `diis.h` / `diis.cpp` | `AmplitudeDIIS` — DIIS on flattened amplitude vectors |
| `ccsd.h` / `ccsd.cpp` | `RCCSDState`, `UCCSDState`, `prepare_rccsd`, `run_rccsd`, `prepare_uccsd`, `run_uccsd` |
| `ccsdt.h` / `ccsdt.cpp` | `RCCSDTState`, `UCCSDTState`, `prepare_rccsdt`, `run_rccsdt`, `prepare_uccsdt`, `run_uccsdt` |
| `determinant_space.h` / `determinant_space.cpp` | `SpinOrbitalSystem`, `DeterminantCCSpinOrbitalSeed`, `build_rhf/uhf_spin_orbital_system`, `solve_determinant_cc` |
| `tensor_backend.h` / `tensor_backend.cpp` | `CanonicalRHFCCReference`, `TensorCCBlockCache`, `TensorTriplesWorkspace`, `TensorRCCSDTState`, `run_tensor_rccsdt`, `choose_rccsdt_backend` |

### Tensor types (`common.h`)

<div align="justify">

`Tensor2D`, `Tensor4D`, and `Tensor6D` store flat `std::vector<double>` with
row-major layout and expose a call-operator for multi-index access.
Indexing follows chemists' notation throughout: `Tensor4D(i,j,k,l)` stores
`(ij|kl)`, not the physicist antisymmetrized bracket. The antisymmetrized
form `<ij||kl> = (ij|kl) - (il|kj)` is computed on the fly where needed.

</div>

### Reference types (`common.h`)

<div align="justify">

`RHFReference` holds the canonical closed-shell partition: `C_occ` [n_ao×n_occ],
`C_virt` [n_ao×n_virt], `eps_occ`, `eps_virt`. All restricted CC paths use
this struct. `UHFReference` holds the full alpha and beta MO coefficient matrices
and orbital energies; the unrestricted determinant-space paths use this.
Both are produced by `build_rhf_reference` / `build_uhf_reference` in
`common.cpp`, which run basic dimension and occupation sanity checks before
returning.

</div>

### Amplitude and denominator storage (`amplitudes.h`)

<div align="justify">

`DenominatorCache` pre-computes the Møller-Plesset denominators `d1(i,a)`,
`d2(i,j,a,b)`, and optionally `d3(i,j,k,a,b,c)`. The `include_triples` flag
allows CCSD paths to skip the \(O(o^3 v^3)\) T3 allocation. `RCCSDAmplitudes`
bundles T1 and T2; `RCCSDTAmplitudes` adds T3 as a `Tensor6D`.

</div>

### MO integral blocks (`mo_blocks.h`)

<div align="justify">

`MOBlockCache` stores the AO→MO transformed ERIs as named sub-tensors
(`oooo`, `ooov`, `oovv`, `ovov`, `ovvo`, `ovvv`, `vvvv`) plus the full
spatial tensor `full`. `build_mo_block_cache` performs the quarter-transform
using `C_occ` and `C_virt` from `RHFReference`. The teaching paths use `full`
to build the antisymmetrized spin-orbital two-body tensor on the fly.
The production tensor backend (`TensorCCBlockCache`) omits `full` and stores
only the seven named blocks to reduce memory.

</div>

### DIIS for amplitudes (`diis.h`)

<div align="justify">

`AmplitudeDIIS` mirrors the SCF DIIS helper (`src/scf/diis.cpp`) but accepts
flattened `Eigen::VectorXd` amplitude vectors and residual vectors. T1 and T2
(and T3 for CCSDT) are concatenated into one vector before each `push` call,
so a single `AmplitudeDIIS` instance accelerates all amplitude blocks
simultaneously. Queue size defaults to 8.

</div>

### RCCSD solver (`ccsd.cpp`)

<div align="justify">

`run_rccsd` operates in the spin-orbital basis. Each iteration: (1) builds
`τ` and `τ̃`, (2) builds the five standard CCSD intermediates
`F_ae`, `F_mi`, `F_me`, `W_mnij`, `W_abef`, `W_mbej`, (3) forms `R1` and
`R2` by contracting amplitudes against those intermediates, (4) applies the
Jacobi update using `DenominatorCache`, (5) accelerates with
`AmplitudeDIIS`. The correlation energy uses the antisymmetrized form
`E = f_{ia} t_i^a + ¼ <ij||ab> t_{ij}^{ab} + ½ <ij||ab> t_i^a t_j^b`.

</div>

### RCCSDT backend dispatch (`ccsdt.cpp`, `tensor_backend.*`)

<div align="justify">

`run_rccsdt` calls `choose_rccsdt_backend(reference)` before doing any work.
The backend choice is based on system size:

- **`DeterminantPrototype`** — selected when n_spin_orb ≤ 12.
  Delegates to `solve_determinant_cc(max_rank=3)`.
- **`TensorProduction`** — selected for larger systems.
  Delegates to `run_tensor_rccsdt` in `tensor_backend.*`, which runs a
  staged pipeline: CCSD warm-start → T1-dress Fock and ERIs → restricted
  R1/R2/R3 equations → restricted T3 symmetry restoration → DIIS.
  For moderate systems (n_spin_orb ≤ 16, det count ≤ 10000) a determinant
  backstop cross-check is optionally run from the warm-started tensor
  amplitudes via `DeterminantCCSpinOrbitalSeed`.

</div>

### Determinant-space backend (`determinant_space.*`)

<div align="justify">

`SpinOrbitalSystem` holds the spin-orbital one-body tensor `h1` and
antisymmetrized two-body tensor `g2`. `build_rhf_spin_orbital_system` and
`build_uhf_spin_orbital_system` construct it from the respective reference
and MO blocks. `solve_determinant_cc(system, max_rank)` enumerates all unique
excitations up to `max_rank` out of the reference determinant, assembles the
cluster operator `T`, evaluates `exp(-T) H exp(T) |Φ₀⟩` by the finite
nilpotent series in the determinant basis, projects onto S/D/T manifolds to
get residuals, and iterates with Jacobi updates and DIIS. Hard limits:
`n_spin_orb ≤ 12`, det count ≤ 1200 (teaching / correctness use only).

</div>

### UCCSD and UCCSDT (`ccsd.cpp`, `ccsdt.cpp`)

<div align="justify">

Both unrestricted paths call `build_uhf_spin_orbital_system` to construct a
spin-orbital Hamiltonian from the UHF alpha and beta MO spaces, then forward
to `solve_determinant_cc` with `max_rank=2` (UCCSD) or `max_rank=3` (UCCSDT).
No separate tensor production path exists for the unrestricted methods; they
are teaching-scale solvers subject to the same determinant-space size limits.

</div>

---

## Analytic Gradients (`src/gradient/`)

<div align="justify">

`compute_rhf_gradient()` and `compute_uhf_gradient()` return the natoms×3 nuclear gradient in Ha/Bohr via the Hellmann-Feynman and Pulay contributions. The Hellmann-Feynman term includes dV/dR (nuclear-attraction derivative with respect to nucleus coordinates). The Pulay term accounts for the basis-function center dependence and requires the energy-weighted density matrix. All derivative integrals are computed analytically via `_compute_1e_deriv_A`, `_compute_nuclear_deriv_A_elem`, and `_compute_eri_deriv_elem`.

`compute_rmp2_gradient()` implements the relaxed MP2 density and Z-vector (coupled-perturbed HF / CPHF) response. The Z-vector equation is solved iteratively to avoid storing the full CPHF response matrices. The relaxed density then enters the same Hellmann-Feynman + Pulay gradient evaluation as the HF case.

</div>

---

## Geometry Optimization (`src/opt/`)

### Cartesian L-BFGS — `geomopt.h` / `geomopt.cpp`

<div align="justify">

`run_geomopt()` implements the two-loop L-BFGS recursion with Armijo backtracking line search. The gradient runner is a `std::function<Eigen::VectorXd(Calculator &)>` callback, allowing the optimizer to drive any level of theory without knowing the details of the energy and gradient computation. Convergence is assessed on the maximum absolute gradient element (`_geomopt_grad_tol`, default 3×10⁻⁴ Ha/Bohr). The L-BFGS history size `_geomopt_lbfgs_m` (default 10) controls the memory-precision trade-off. On convergence, `calc._molecule._standard` holds the optimized geometry in Bohr and the result struct contains the per-step energy trajectory.

</div>

### Internal-coordinate BFGS — `intcoords.h` / `geomopt.cpp`

<div align="justify">

`run_geomopt_ic()` optimizes in redundant generalized internal coordinates (GIC). `IntCoordSystem::build()` automatically generates all bonds (using covalent radii from the element lookup), all valence bends, and all proper torsions. The Wilson B-matrix `B_{qi,a} = ∂q_i/∂x_a` is evaluated analytically for each IC type. The BFGS Hessian update operates in the redundant IC space; the Cartesian back-transform uses the iterative Schlegel procedure (Newton steps in the pseudo-inverse of B until the IC values converge). Geometry constraints from `%begin_constraints` are enforced by projecting the gradient and Hessian step into the constraint-compliant subspace. `GeomConstraint::Type` supports `Bond`, `Angle`, `Dihedral`, and `FrozenAtom`.

**Trade-off:** IC-BFGS typically converges in fewer steps than Cartesian L-BFGS for systems with strained internal coordinates (rings, transition states), but the Schlegel back-transform adds iteration overhead per step.

</div>

---

## Vibrational Frequencies (`src/freq/`)

<div align="justify">

`compute_hessian()` builds the 3N×3N Cartesian Hessian by central finite differences of analytic gradients: `H_{ij} = (g_i(x + h ê_j) − g_i(x − h ê_j)) / (2h)` where `h = _hessian_step` (default 5×10⁻³ Bohr). The 6N gradient evaluations are parallelized with OpenMP when available. `vibrational_analysis()` mass-weights the Hessian, projects out translations and rotations using Eckart conditions, diagonalizes the resulting 3N−6 (or 3N−5 for linear molecules) subspace, converts eigenvalues to cm⁻¹, and computes the zero-point energy. When symmetry is active, normal modes are assigned Mulliken labels by projecting onto the SAO basis. Imaginary frequencies are reported as negative values in cm⁻¹.

`ImaginaryFollow` mode runs a frequency calculation, displaces the geometry along the largest imaginary normal mode by `_imag_follow_step` (default 0.2 Bohr), then runs a geometry optimization to locate the adjacent minimum.

</div>

---

## KS-DFT (`src/dft/`)

<div align="justify">

The `planck-dft` binary provides Kohn-Sham DFT by augmenting the shared SCF infrastructure with a numerical integration grid and an XC functional evaluator. The DFT pipeline is cleanly separated from the HF pipeline: `DFT::Driver::run()` calls `DFT::Driver::prepare()` to build the grid and AO-on-grid evaluations, then enters the KS-SCF loop.

</div>

### Driver — `dft/driver.h` / `driver.cpp`

<div align="justify">

`DFT::Driver::prepare()` constructs the `PreparedSystem` containing the shell pairs, the molecular grid, and the pre-evaluated AO basis functions on every grid point. `DFT::Driver::run()` then enters the KS-SCF loop, which follows the standard SCF structure but replaces the HF exchange contribution with the XC potential assembled from the grid. `evaluate_current_density_and_xc()` and `assemble_current_ks_potential()` are exposed as public functions so that external tools can evaluate XC contributions without running a full SCF loop — useful for testing grid integration quality independently.

</div>

### Grid — `dft/base/grid.h`, `radial.h`, `angular.h`

<div align="justify">

The molecular integration grid uses the Becke partitioning scheme to assign multi-center weights to a superposition of atom-centered grids. Each atom-centered grid combines a **Treutler-Ahlrichs M4** radial grid (which maps the semi-infinite radial interval onto [−1, 1] with a logarithmic transformation optimized for exponentially decaying functions) with a **Lebedev** angular grid. The `GridPreset` struct encodes the angular scheme (1–7) and radial accuracy factor for each of the four quality levels:

</div>

| Level | Angular scheme | Approx. points/atom |
|---|---|---|
| Coarse | 1 | ~3 000 |
| Normal | 3 | ~12 000 |
| Fine | 5 | ~30 000 |
| UltraFine | 7 | ~75 000 |

<div align="justify">

Five-region pruning reduces the angular order near the nucleus (where the density is nearly spherical) and far from the nucleus (where it is negligible), substantially reducing the total grid point count without affecting accuracy.

</div>

### libxc wrapper — `dft/base/wrapper.h`

<div align="justify">

`XC::Functional` is an RAII wrapper around a libxc `xc_func_type`. The `evaluate()` method takes a density array (and gradient array for GGA functionals) and fills the XC energy density and potential arrays. The libxc functional ID is resolved from the `XCExchangeFunctional` / `XCCorrelationFunctional` enums; `Custom` mode accepts a raw libxc integer ID for any functional in the libxc library. The `USING_Libxc` preprocessor guard ensures that this header (and its libxc dependency) is only compiled into `planck-dft` and not `hartree-fock`.

</div>

### AO evaluation on grid — `dft/ao_grid.h`

<div align="justify">

`evaluate_ao_basis_on_grid()` evaluates all `nbasis` contracted Cartesian Gaussian basis functions at every grid point and returns an `AOGridEvaluation` struct (`npoints × nbasis` matrix). This evaluation is done once before the SCF loop and reused every iteration. For GGA functionals, the gradient of the AO basis (`npoints × nbasis × 3`) is also evaluated and stored.

</div>

### XC on grid — `dft/xc_grid.h` / `xc_grid.cpp`

<div align="justify">

`evaluate_density_on_grid()` contracts the density matrix with the AO values to produce the electron density at each grid point: `ρ(r) = Σ_{μν} P_{μν} χ_μ(r) χ_ν(r)`. `evaluate_xc_on_grid()` passes ρ (and ∇ρ for GGA) to the libxc wrapper and returns the XC energy density and potential at each point. The integrated electron count `∫ρ dV` is checked as a grid quality diagnostic.

</div>

### KS potential matrix — `dft/ks_matrix.h` / `ks_matrix.cpp`

<div align="justify">

`assemble_xc_matrix()` integrates `V_{XC,μν} = ∫ v_xc(r) χ_μ(r) χ_ν(r) dV` using the pre-evaluated AO grid values and quadrature weights. `combine_ks_potential()` assembles the full KS Fock matrix `F_KS = H_core + J + V_XC` where J is the Coulomb contribution from the SCF infrastructure. The resulting `KSPotentialMatrices` are passed back to the SCF loop, which diagonalizes `F_KS` exactly as it would diagonalize a HF Fock matrix.

</div>

---

## Entry Points (`src/driver.cpp`, `src/dft/main.cpp`)

### HF driver — `src/driver.cpp`

<div align="justify">

`main()` orchestrates the full HF calculation pipeline:

</div>

1. Parse input → `Calculator`
2. Derive checkpoint path from input filename stem
3. Detect symmetry, build SAO basis and integral symmetry ops
4. Read GBS basis
5. If `ReadFull` guess: `load_geometry()` then repopulate basis and symmetry
6. If `ReadDensity` or `ReadFull`: `load()` checkpoint
7. `Calculator::initialize()` — allocate SCF matrices, set auto SCF mode
8. Build shell pairs
9. Compute 1e integrals (S, T, V → H_core); optionally compute ERI tensor
10. If `Conventional` mode: store ERI
11. Optionally build SAO blocking data
12. Run SCF via `run_rhf()` or `run_uhf()`
13. Save checkpoint
14. Dispatch to gradient, geomopt, frequency, or post-HF based on `CalculationType` and `PostHF`
15. Report multipole moments and timing

### DFT driver — `src/dft/main.cpp`

<div align="justify">

The DFT entry point follows the same setup sequence as the HF driver (parse → symmetry → basis → shell pairs → 1e integrals) but then calls `DFT::Driver::run()` instead of `run_rhf()`. Gradient, geometry optimization, and frequency analysis route through the same `src/gradient/`, `src/opt/`, and `src/freq/` modules, using the DFT energy and gradient as the callback.

</div>

---

## Data Flow Summary

```
Input file
    │
    ▼
IO::parse_input()          → Calculator (options, molecule)
    │
    ▼
Symmetry::detectSymmetry() → molecule._standard (Bohr, reoriented)
    │
    ▼
BasisFunctions::read_gbs_basis() → Basis (_shells, _basis_functions)
    │
    ▼
build_shellpairs()         → vector<ShellPair>
    │
    ▼
ObaraSaika::_compute_1e()  → overlap S, core Hamiltonian H
ObaraSaika::_compute_2e()  → ERI tensor (conventional mode)
    │
    ▼
run_rhf() / run_uhf()      → DataSCF (density, Fock, MOs, energies)
    │
    ├─ SinglePoint → energies + properties
    ├─ Gradient    → Gradient::compute_*_gradient()
    ├─ GeomOpt     → Opt::run_geomopt() / run_geomopt_ic()
    ├─ Frequency   → Freq::compute_hessian() → vibrational_analysis()
    ├─ RMP2/UMP2   → Correlation::run_rmp2() / run_ump2()
    └─ CASSCF      → Correlation::run_casscf()
                         │
                         ├─ CASSCF::build_ci_space()
                         ├─ CASSCF::solve_ci()
                         ├─ CASSCF::compute_1rdm() / compute_2rdm()
                         ├─ CASSCF::build_active_integral_cache()
                         ├─ CASSCF::compute_Q_matrix()
                         └─ orbital rotation loop
```

---

## Key Invariants and Pitfalls

<div align="justify">

The following invariants are not enforced by the type system and must be maintained by callers:

</div>

- **`_standard` must be in Bohr** before `read_gbs_basis()`, `_compute_nuclear_repulsion()`, and any integral evaluation. `detectSymmetry()` sets it in Bohr automatically. When symmetry is disabled, `prepare_coordinates()` must have been called first.
- **`ContractedView._index`** must equal the position of the basis function in `Basis._basis_functions`. A missing or zero-initialized index silently misplaces all integrals for that function.
- **`Nc` is folded into `Shell._coefficients`** during GBS parsing. No external `Nc` factor should ever be applied in integral code.
- **Shell-pair index recovery**: Use `sp.A._index` and `sp.B._index` from the stored `ContractedView` references. Do not use `invert_pair_index` on a shell-pair vector position to recover shell indices — it operates on triangular pair indices, not vector positions.
- **`Calculator::initialize()`** must be called after the basis is loaded and before any integral or SCF code runs. It allocates all matrix storage.
- **SAO blocking** is valid only after `build_sao_basis()` sets `_sao_transform` and related arrays. The SCF code checks `_use_sao_blocking` before attempting block-diagonal diagonalization.
- **DIIS restart**: `DIISState::clear()` is called automatically when the error grows by more than `_diis_restart_factor`. Callers should not manually clear the DIIS state unless explicitly restarting from scratch.
- **`ActiveIntegralCache::puvw` layout** is row-major `(p,u,v,w)`. Every producer and consumer must agree on that exact ordering; `compute_Q_matrix()` assumes each fixed-`p` slab is one contiguous block of length `n_act^3`.
- **Per-root SA energies** stored in `Calculator::_cas_root_energies` are full CASSCF total energies. Do not replace them with raw CI eigenvalues unless the logging and checkpoint consumers are updated accordingly.
