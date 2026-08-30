# Why the FCI sigma build was slow, and what made it fast

`apply_ci_hamiltonian` (`src/post_hf/ci/ci.cpp`) is the iterative sigma build —
the path every determinant space above `dense_threshold = 500` takes. It was
**allocation-bound and single-threaded**. Two changes landed 2026-08-30:
**4.8x from removing the allocator, then 3.54x more from threading it**, for
~17x combined on N2/STO-3G.

| | N2/STO-3G (ndet 14 400) | `be_fci_spherical_631gd` (8 281) |
|---|---|---|
| before | 125.7 s | ~46.5 s |
| after F1 (allocation) | 26.3 s | 7.6 s |
| after F3 (threading, 4 threads) | **7.79 s** | **3.42 s** |

Energies are unchanged throughout — bitwise against the pre-change binary, and
byte-identical across `OMP_NUM_THREADS` = 1/2/4/8.

## 1. Half the time was in `malloc`, not arithmetic

The opening profile (21 048 leaf samples, 1 thread) was the surprise:

| frame | share |
|---|---|
| `malloc` / `free` family | **~53 %** |
| `apply_ci_hamiltonian` | 12.0 % |
| `get_excitation` | 5.9 % |

`get_excitation` returned `std::pair<std::vector<int>, std::vector<int>>` **by
value**, and `slater_condon_element` called it for both spin channels at six call
sites — up to four heap allocations per matrix element, for vectors holding **at
most two entries each** (a Slater-Condon element vanishes beyond a double
excitation, so 2 is a true bound, not a guess). It now returns a fixed-capacity
struct: `std::array<int,2> ann, cre` plus counts, with the bound asserted rather
than silently truncated.

**Result: 125.7 s -> 26.3 s (4.8x), and the malloc share went 53 % -> 0.1 %** —
the allocator is eliminated, not reduced.

**The 4.8x exceeded what the profile implied, and the reason generalizes.** A
53 % share caps the direct saving at ~2.1x by Amdahl. Getting 4.8x means the
allocator cost more than its own samples: per-element `malloc`/`free` churns the
heap, so removing it also removed cache pressure and bookkeeping attributed to
*other* frames. **A profile share is a lower bound on what removing that work is
worth, not an estimate of it** — the inverse of the CC transpose merge, where an
operator-count model *over*-promised by treating unequal work as equal.

## 2. Threading it: the obvious route is the wrong one

The loop scatters — it runs over ket determinants `j` and writes
`sigma(i) += ...` at indices found by a hash lookup, so two threads on different
`j` will target the same `sigma(i)`. That is not the disjoint-slice shape the CC
residual nests had, so the CC recipe does not transfer as written.

### The gather was recommended, built, and is 2.2-2.4x SLOWER

The natural fix is to invert the loop to run over **bras** and pull, making each
thread's writes disjoint. It was built and it is **numerically correct** — all
three references matched to every printed digit. It is also much slower:

| case | scatter | gather |
|---|---|---|
| `be_fci_spherical_631gd` | 7.6 s | **16.4 s** |
| N2/STO-3G | 26.3 s | **63.3 s** |

**Why: the `|c| < 1e-15` skip is not a summation-order detail. It is a sparsity
exploitation carrying asymptotic weight.** In the scatter the test sits on the
*outer* loop, so a negligible ket skips the entire 126-line enumeration —
hundreds of neighbours — in one comparison. In the gather the outer index is the
bra, whose `sigma(i)` must be computed regardless of `c(i)`, so **every** outer
iteration runs the full enumeration; the skip survives only as an inner test
saving one `slater_condon_element` call.

That matters because the vectors really are sparse:

- `davidson` (`ci.cpp:130-136`) starts from unit vectors on the lowest-diagonal
  determinants — `max(2*nroots, 4)` nonzeros in `dim`, and the QR of already-
  orthonormal unit columns keeps them sparse.
- Worse, `solve_ci` (`ci.cpp:743`) reconstructs a dense `H` by calling
  `sigma_apply` with `Eigen::VectorXd::Unit(dim, j)` — **exactly one nonzero** —
  once per column. That is O(dim) enumerations scattered and **O(dim²)**
  gathered.

**Generalizable: a cheap-looking guard on an outer loop can be carrying
asymptotic weight.** Before moving a skip inward to enable a restructuring,
measure what fraction of outer iterations it eliminates *on the actual inputs*.
No threading win (~3.7x ceiling at 4 threads) repays a 2.4x serial loss plus an
asymptotic change.

**Two facts the gather rested on are sound and need not be re-litigated** if
anyone revisits it: the reachability relation is symmetric (verified by direct
enumeration — **0 asymmetric edges** across n_act 4/5/6 including an open-shell
case), and `H` is real symmetric, which `build_ci_hamiltonian_dense` already
depends on, filling one triangle and assigning `H(i,j) = H(j,i) = v`
(`ci.cpp:264`). The gather is correct. It is just slower.

### What landed: keep the scatter, bin the accumulators

The scatter is kept — preserving the outer skip — and accumulates into partial
vectors summed in fixed order. **3.54x at 4 threads** against a modelled ~3.7x
ceiling:

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| N2/STO-3G | 27.59 s | 14.66 s | **7.79 s** | 5.71 s |
| speedup | 1.00x | 1.88x | **3.54x** | 4.83x |

## 3. A fixed-order reduction is necessary but NOT sufficient

This is the part most worth carrying. **Two determinism defects, each caught only
by byte-diffing outputs across thread counts — neither by reasoning:**

1. **`schedule(dynamic)`**, the natural choice for load balance and what the
   original scope recommended. Dynamic assignment gives a buffer a different
   *subset* of terms per run, so the sums inside it reassociate. Measured: two
   different last digits across **5 identical 4-thread runs**.
2. **Keying the buffers by `omp_get_thread_num()`.** Even under
   `schedule(static)`, each buffer's *contents* depend on the thread **count**,
   so 8 threads disagreed with 1/2/4.

The first version had the fixed-order reduction and was *still* non-deterministic.
**What must be deterministic is the partition of work into accumulators, not
merely the order they are summed.**

The fix bins by a fixed function of `j` — `partials[j / bin_size]` with a
`constexpr kBins = 64` — under `schedule(static, bin_size)`, so one chunk is
exactly one bin (which is also what gives a thread exclusive access to its bin).
Bin `b` then receives exactly the same `j` values in the same order regardless of
thread count, so every partial is bit-identical and so is the fixed-order sum
over bins.

Memory is `kBins × dim × 8` bytes — **independent of thread count**, 7.4 MB at
N2.

**Never `omp atomic` on `sigma`, and never a completion-order reduction.** That
is the DFT-grid jitter defect (`dft_xc_reduction_determinism`), and every other
parallel path in this codebase is bitwise thread-count-invariant by design.

## 4. Post-change profile: nothing cheap is left

**Serial is unchanged from post-F1, so the binning costs nothing measurable:**

| frame | post-F1 | post-F3, 1 thread |
|---|---|---|
| `apply_ci_hamiltonian` | 55.0 % | **55.4 %** |
| `slater_condon_element` | 19.6 % | **19.9 %** |
| `apply_creation` | 10.2 % | **9.9 %** |
| `parity_between` | 7.3 % | **6.7 %** |
| `apply_annihilation` | 6.3 % | **6.4 %** |

**The 7.4 MB per-call `partials` allocation was suspected and is refuted.** The
whole malloc family is **0.1 %** and the `memset` zeroing is **8 samples
(0.0 %)**, despite Davidson calling the sigma build once per subspace vector per
iteration. Eigen reuses the allocation and the zeroing is trivial next to the
enumeration. The serial reduction is likewise negligible: `n_bins × dim` = 921 600
adds per call, under a millisecond against 7.79 s.

At 4 threads the only new frame is the barrier, `__psynch_cvwait` at 3.0 %.
Per-thread idle is uneven — the residual imbalance:

| thread | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| idle | 0.4 % | 4.4 % | 2.5 % | **11.8 %** |

**4.8 % total idle**, so removing the imbalance entirely is worth only **~1.05x**
on top of 3.54x. Left as-is: determinism forbids `dynamic`, and 5 % does not
justify it. **If it is ever wanted, the move is more bins (smaller `kBins`) under
static scheduling — never `dynamic`.**

## What is left

**`occupied_orbitals` (`ci.cpp:24`) still returns a `std::vector<int>` by value**,
called twice per outer iteration. It was scoped as a companion to the
`get_excitation` fix and expected to be worth doing on its own merits — but that
expectation was formed when the allocator was 53 % of runtime. **It now profiles
at 0.1-0.2 %.** F1 removed the pressure that made it look expensive, so it is
code hygiene, not a performance item.

## Gate coverage — read this before writing a test here

Of the seven committed FCI regression cases, **only two reach the iterative sigma
path at all**:

| case | CI dim | path |
|---|---|---|
| `h2_fci_sto3g` | 4 | dense |
| `h2_fci_rimp2_ccpvdz` | 100 | dense |
| `water_fci_sto3g` | 441 | dense |
| **`o2_fci_rohf_sto3g`** | **1 200** | **iterative** |
| **`be_fci_spherical_631gd`** | **8 281** | **iterative** |
| `water_fcidump_*` | — | FCIDUMP export, not a solve |

**The two smallest and most obvious FCI cases run the dense path and would pass a
broken threaded sigma build unchanged** — the same trap that let
`ch4_rccsdt_sto3g` sit green for its entire life while never running the kernel
it was added to protect.

`slater_condon_element` is **public** (`ci.h:43`) and also consumed by
`src/post_hf/casscf/response.cpp`, so the CASSCF/RASSCF cases are downstream of
this code and must stay green. All 19 FCI/CASSCF/RASSCF/FCIDUMP/RI cases pass,
including both iterative gates and the `water_casscf_sa2_sto3g_sad_guess_uphill`
canary.

**The larger timing fixture** is
`tests/inputs/exploratory/fciqmc/n2_fci_sto3g.hfinp` (ndet 14 400), which is not
a registered regression case.

## Key code locations

| what | where |
|---|---|
| the sigma build | `apply_ci_hamiltonian`, `src/post_hf/ci/ci.cpp` |
| the single write seam every term funnels through | the `accumulate` lambda |
| the outer skip that must not move | `if (std::abs(cj) < 1e-15) continue;` |
| the symmetry the gather relied on | `build_ci_hamiltonian_dense`, `ci.cpp:264` |
| the sparse callers that killed the gather | `davidson` `ci.cpp:130-136`, `solve_ci` `ci.cpp:743` |
| the determinism precedent to imitate | DFT J/K builds, `src/dft/driver.cpp` |
| the determinism defect to avoid | `dft_xc_reduction_determinism` note |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
