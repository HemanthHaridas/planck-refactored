# Bug Report — Planck Codebase Review

**Date:** 2026-04-08
**Reviewer:** Claude Code (automated review)
**Scope:** Full codebase — integrals, SCF, post-HF, CASSCF, gradients, DFT, I/O
**Branch:** codex/casscf-active-cache-patch1

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 2 |
| Major    | 4 |
| Minor    | 8 |
| Nitpick  | 3 |

---

## Critical

### C1 — ~1 GB per-thread VRR/HRR scratch buffers cause OOM on multi-threaded runs

**File:** `src/integrals/os.cpp:546–549`

**Description:**
`_vrr_buf` is declared as `thread_local double[13][13][13][13][13][13][26]` = 125,497,034 doubles = **1.00 GB per thread**. `_hrr_buf` adds another ~38 MB. With a typical OpenMP thread pool of 4–16 threads, this pre-allocates 4–16+ GB just for ERI scratch space, regardless of the actual angular momenta of the molecules being computed. On most workstations this causes an immediate OOM when `MAX_L >= 4` (i.e., any basis with f-functions).

**Reproduction:** Run any calculation with f-function basis (e.g., 6-311G**) with `OMP_NUM_THREADS=8`.

**Fix:** Replace fixed-size `thread_local` arrays with per-quartet heap allocations sized to the actual angular momenta of the quartet: `(lA + lB + lC + lD + 1)^n`. Alternatively, use a `std::vector` resized at the start of each quartet loop.

---

### C2 — Data race in `write_eri_permutations` under OpenMP

**File:** `src/integrals/os.cpp:1553–1601, 1648–1696, 1870–1917`

**Description:**
The canonical-representative guard prevents redundant computation of each unique quartet, but the 8-fold permutation expansion in `write_eri_permutations` can write to ERI tensor indices that overlap with another thread's 8-fold expansion of a different canonical quartet. Specifically, quartet `(i,j,k,l)` with `i < k` writes `eri[k,l,i,j]`, which is also the primary write target of quartet `(k,l,i,j)` in another thread. Even if both threads write the same value, concurrent writes to a non-atomic location are **undefined behavior in C++** and will produce incorrect results under TSan or aggressive compiler optimization.

**Fix:** Accumulate per-thread partial ERI tensors and reduce at the end of the parallel region, or protect writes with `#pragma omp atomic` / `#pragma omp critical` for the permutation fill.

---

## Major

### M1 — `_bohr_to_angstrom()` return type mismatch

**File:** `src/base/types.h:840–843`

**Description:**
`_molecule._standard` is `Eigen::MatrixXd` (natoms × 3). The expression `_molecule._standard * 0.529177210903` produces a `MatrixXd`, but the declared return type is `const Eigen::VectorXd`. Eigen does not implicitly convert a multi-column matrix to a vector; depending on Eigen version this either fails to compile or silently returns only the first column.

**Fix:**
```cpp
// Before
const Eigen::VectorXd _bohr_to_angstrom() const { ... }

// After
Eigen::MatrixXd _bohr_to_angstrom() const { ... }
```

---

### M2 — `set_max_cycles_auto` and `get_max_cycles` return inconsistent values

**File:** `src/base/types.h:289–306`

**Description:**
Two functions independently hardcode the max-cycle tier table and disagree on the value for `nbasis > 250`:

| Function | `nbasis > 250` |
|----------|---------------|
| `set_max_cycles_auto` | 100 |
| `get_max_cycles` | 150 |

This means a calculation that calls `set_max_cycles_auto` and then `get_max_cycles` to check the limit will see a different number than what was actually set.

**Fix:** Unify into a single `max_cycles_for_nbasis(int nbasis)` free function called by both.

---

### M3 — Gradient ERI loop is O(N⁴) with no Schwarz screening

**File:** `src/gradient/gradient.cpp:73–77, 142–170` (RHF); `271–275, 336–364` (UHF)

**Description:**
The gradient code constructs `all_pairs` as the full `nb × nb` Cartesian product of shell pairs and iterates over `all_pairs × all_pairs` with no:
- Upper-triangle permutation symmetry exploitation
- Schwarz pre-screening (`|ij| × |kl| < threshold`)

For `nb = 100` shells this produces ~100M ERI derivative evaluations. The ERI build code in `os.cpp` uses screened upper-triangle iteration reaching ~6M evaluations for the same system — a ~17× difference that grows as O(N²) with system size.

**Fix:** Mirror the screened shell-pair iteration from `os.cpp` in the gradient loop. The gradient permutation symmetry is slightly different (factor of 2 rather than 8) but the screening logic is identical.

---

### M4 — `Molecule::nelectrons` is dead code

**File:** `src/base/types.h:169`

**Description:**
The field `int nelectrons` is declared on `Molecule` but is never populated by the parser and never read by any downstream code. All electron counts are recomputed inline (e.g., from `charge` and atomic numbers). The field misleads readers into thinking it holds a valid count.

**Fix:** Remove the field, or populate and use it consistently.

---

## Minor

### m1 — Fermion operator utilities duplicated across three CASSCF files

**Files:**
- `src/post_hf/casscf/rdm.cpp:17–41`
- `src/post_hf/casscf/response.cpp:39–62`
- `src/post_hf/casscf/strings.cpp:208–225`

**Description:** `FermionOpResult` and the fermion creation/annihilation operator functions are copy-pasted into three separate anonymous namespaces. Any bug fix or behavioral change must be applied in three places.

**Fix:** Extract to a shared `src/post_hf/casscf/casscf_utils.h` internal header.

---

### m2 — `as_single_column_matrix` / `single_weight` helpers duplicated

**Files:**
- `src/post_hf/casscf/casscf.cpp:238–250`
- `src/post_hf/casscf/response.cpp:212–224`

**Description:** Same pattern as m1 — identical helper functions duplicated across files.

**Fix:** Move to the shared CASSCF utilities header (see m1).

---

### m3 — Several functions throw exceptions instead of returning `std::expected`

**Files and lines:**

| File | Line | Function | Exception thrown |
|------|------|----------|-----------------|
| `src/gradient/gradient.cpp` | 43 | `build_shell_atom_map` | `std::runtime_error` |
| `src/gradient/gradient.cpp` | 394 | `compute_rmp2_gradient` | (uncaught throw) |
| `src/io/io.cpp` | 62 | `toBool` | `std::invalid_argument` |
| `src/io/io.cpp` | 82, 96 | `parse_irrep_count_list` | `std::invalid_argument` |
| `src/post_hf/casscf/strings.cpp` | 427 | `reorder_mo_coefficients` | `std::invalid_argument` |

**Description:** These functions violate the project-wide convention that all fallible public functions return `std::expected<T, std::string>`. Exceptions escape into callers that expect the `std::expected` protocol, silently bypassing all error-propagation logic.

**Fix:** Convert each to return `std::expected<T, std::string>` and use `std::unexpected(message)` at each current throw site.

---

### m4 — `ShellPair` stores const references, creating lifetime fragility

**File:** `src/base/types.h:498–499`

**Description:**
`ShellPair` holds `const Shell& A` and `const Shell& B`. If the `vector<Shell>` that owns these shells is ever reallocated (e.g., via `push_back` after `ShellPair` objects are constructed), all existing `ShellPair` references silently dangle. There is currently no lifetime enforcement preventing this.

**Fix:** Store indices into the shell vector instead of references, or ensure `build_shellpairs` is always called after the shell vector is finalized and never modified.

---

### m5 — `_compute_nuclear_repulsion` is `noexcept` but silently produces garbage if called early

**File:** `src/base/types.h:802`

**Description:**
The function is marked `noexcept` and uses `_molecule._standard` (which must be Bohr). If called before `initialize()` → `detectSymmetry()` sets `_standard`, the result is wrong with no diagnostic. The `noexcept` annotation prevents callers from even considering that the result might be invalid.

**Fix:** Either add a precondition assertion (`assert(_standard_initialized)`), or document clearly in the declaration that this must only be called post-`initialize()`.

---

### m6 — `Molecule::clear()` does not reset `nelectrons`

**File:** `src/base/types.h:175–193`

**Description:**
`Molecule::clear()` resets atoms, coordinates, charge, multiplicity, and other fields but omits `nelectrons`. If the field is ever populated (fixing M4 above), `clear()` will leave a stale count.

**Fix:** Add `nelectrons = 0;` to the body of `clear()`.

---

### m7 — DFT KS matrix V_xc assembly loop is not parallelized

**File:** `src/dft/ks_matrix.cpp:98–155`

**Description:**
The V_xc integration loop (sum of `vxc * AO * AO` over grid points) is single-threaded, while the J/K ERI-based portion of the same KS matrix is OpenMP-parallelized. For large grids (UltraFine quality) or large basis sets, V_xc assembly becomes the bottleneck.

**Fix:** Add `#pragma omp parallel for reduction(+:Vxc)` over the grid-point loop; the loop body is embarrassingly parallel.

---

### m8 — `energy_sorted_indices` uses exact floating-point equality

**File:** `src/post_hf/casscf/strings.cpp:76`

**Description:**
Orbital energies are compared with `==` to detect degeneracies or duplicates. Floating-point orbital energies from the SCF are subject to rounding and will rarely be bitwise-identical even when physically degenerate.

**Fix:** Use a tolerance-based comparison: `std::abs(e_i - e_j) < 1e-10`.

---

## Nitpick

### n1 — Typo: "Hartee-Fock" in comment

**File:** `src/base/types.h:53`
**Fix:** "Hartee-Fock" → "Hartree-Fock"

---

### n2 — Typo: "Shwartz" in comment

**File:** `src/base/types.h:338`
**Fix:** "Shwartz" → "Schwarz"

---

### n3 — Unreachable `default` branch throws in `set_output_options`

**File:** `src/base/types.h:478`
**Description:** The `switch` covers all enum values but includes a `default: throw` branch. This is unreachable and misleading — it implies unknown values are possible when the type system already prevents them.
**Fix:** Remove the `default` branch (or replace with `__builtin_unreachable()` / `std::unreachable()` in C++23).

---

## Known Bug (pre-existing, tracked separately)

### Per-root total energy display stores CI eigenvalue, not total CASSCF energy

**File:** `src/post_hf/casscf/casscf.cpp:1418`
**Description:** The per-root energy stored at line 1418 is the CI eigenvalue (active-space energy relative to core), not the full total energy (`E_CI + E_nuc + E_core`). Energies used for convergence gating are correct; this is a display-only bug.
**Fix:** Replace stored value with `ci_eigenvalue + E_nuc + E_core_one_electron`.

---

*End of report.*
