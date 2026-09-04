# FCI Sigma Build Performance

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Why was the FCI sigma build slow, and what made it fast without breaking bitwise determinism?**

## Short answer

`apply_ci_hamiltonian` (`src/post_hf/ci/ci.cpp`) is the iterative sigma build —
the path every determinant space above `dense_threshold = 500` takes. It was
allocation-bound and single-threaded. Two changes landed 2026-08-30: 4.8x from
removing the allocator, then 3.54x more from threading it, for ~17x combined on
N2/STO-3G.

| | N2/STO-3G (ndet 14 400) | `be_fci_spherical_631gd` (8 281) |
|---|---|---|
| before | 125.7 s | ~46.5 s |
| after F1 (allocation) | 26.3 s | 7.6 s |
| after F3 (threading, 4 threads) | 7.79 s | 3.42 s |

Energies are unchanged throughout — bitwise against the pre-change binary, and
byte-identical across `OMP_NUM_THREADS` = 1/2/4/8.

## Where the logic lives

- `src/post_hf/ci/ci.cpp` — `apply_ci_hamiltonian` (the sigma build), the
  `accumulate` lambda (the single write seam every term funnels through), the
  outer skip `if (std::abs(cj) < 1e-15) continue;`, `build_ci_hamiltonian_dense`
  at line 264 (the symmetry the gather relied on), `davidson` at lines 130-136
  and `solve_ci` at line 743 (the sparse callers that killed the gather)
- `ci.h:43` — `slater_condon_element`, public, also consumed by
  `src/post_hf/casscf/response.cpp`
- `src/dft/driver.cpp` — the DFT J/K builds, the determinism precedent this
  fix imitates
- `dft_xc_reduction_determinism` note — the determinism defect this fix avoids
- `tests/inputs/exploratory/fciqmc/n2_fci_sto3g.hfinp` — the larger timing
  fixture (ndet 14 400), not a registered regression case

## What invariants matter

### 1. A profile share is a lower bound on what removing that work is worth, not an estimate of it

The opening profile (21 048 leaf samples, 1 thread) showed `malloc`/`free` at
~53 %, `apply_ci_hamiltonian` itself at 12.0 %, `get_excitation` at 5.9 %. A
53 % share caps the direct saving at ~2.1x by Amdahl, but the fix (below)
measured 4.8x. Per-element `malloc`/`free` churns the heap, so removing it also
removed cache pressure and bookkeeping attributed to other frames.

Design rule:

- Treat a measured profile share as a floor on the payoff of removing that
  work, not a ceiling — this is the inverse of the CC transpose merge, where
  an operator-count model over-promised by treating unequal work as equal.

### 2. A cheap-looking guard on an outer loop can be carrying asymptotic weight

The natural threading fix — invert the scatter loop to run over bras and pull,
making each thread's writes disjoint — was built, is numerically correct
(matched to every printed digit), and is 2.2-2.4x slower. The `|c| < 1e-15`
skip is not a summation-order detail, it is a sparsity exploitation: in the
scatter the test sits on the outer loop, so a negligible ket skips the entire
126-line enumeration in one comparison. In the gather the outer index is the
bra, whose `sigma(i)` must be computed regardless of `c(i)`, so every outer
iteration runs the full enumeration. This matters because the vectors really
are sparse — `davidson` starts from unit vectors (`max(2*nroots, 4)` nonzeros),
and `solve_ci` reconstructs a dense `H` one `Eigen::VectorXd::Unit(dim, j)`
column at a time, which is O(dim) enumerations scattered and O(dim^2) gathered.

Design rule:

- Before moving a skip inward to enable a restructuring, measure what
  fraction of outer iterations it eliminates on the actual inputs. No
  threading win repays an asymptotic change plus a serial slowdown.

### 3. A fixed-order reduction is necessary but not sufficient for determinism

Two determinism defects were found, each caught only by byte-diffing outputs
across thread counts, neither by reasoning: `schedule(dynamic)` (the natural
choice for load balance, and what the original scope recommended) gives a
buffer a different subset of terms per run, so its internal sums reassociate —
measured as two different last digits across 5 identical 4-thread runs. And
keying accumulator buffers by `omp_get_thread_num()` makes their contents
depend on the thread count itself, even under `schedule(static)` — 8 threads
disagreed with 1/2/4. The first version had a fixed-order reduction and was
still non-deterministic.

Design rule:

- What must be deterministic is the partition of work into accumulators, not
  merely the order they are summed. Bin by a fixed function of the loop
  index (`partials[j / bin_size]`, `constexpr kBins = 64`) under
  `schedule(static, bin_size)` so one chunk is exactly one bin — bin `b`
  then receives the same values in the same order regardless of thread
  count.
- Never use `omp atomic` on the accumulator, and never a completion-order
  reduction. That is the DFT-grid jitter defect; every other parallel path
  in this codebase is bitwise thread-count-invariant by design.

### 4. A removed bottleneck can make the next-most-obvious target stop mattering

`occupied_orbitals` (`ci.cpp:24`) still returns a `std::vector<int>` by value,
called twice per outer iteration. It was scoped as a companion fix to
`get_excitation`, expected to be worth doing on its own merits — but that
expectation was formed when the allocator was 53 % of runtime. Post-fix it
profiles at only 0.1-0.2 %.

Design rule:

- Re-price a planned follow-on fix after its sibling lands; removing one
  bottleneck can eliminate the pressure that made a second target look
  expensive.

### 5. The two smallest, most obvious test cases can be exactly the ones that never exercise the code being fixed

Of the seven committed FCI regression cases, only two reach the iterative
sigma path at all (`o2_fci_rohf_sto3g` at CI dim 1200, `be_fci_spherical_631gd`
at 8281); the smallest and most obvious (`h2_fci_sto3g` at 4,
`water_fci_sto3g` at 441) run the dense path and would pass a broken threaded
sigma build unchanged.

Design rule:

- When gating a fix to a specific code path, verify which committed test
  cases actually reach that path rather than assuming size or prominence
  implies coverage — the same trap left `ch4_rccsdt_sto3g` green for its
  entire life while never running the kernel it was added to protect.

## What was fixed

1. **`get_excitation` no longer allocates.** It returned
   `std::pair<std::vector<int>, std::vector<int>>` by value, and
   `slater_condon_element` called it for both spin channels at six call sites
   — up to four heap allocations per matrix element, for vectors holding at
   most two entries each (a Slater-Condon element vanishes beyond a double
   excitation, so 2 is a true bound, not a guess). It now returns a
   fixed-capacity struct: `std::array<int,2> ann, cre` plus counts, with the
   bound asserted rather than silently truncated. Result: 125.7 s -> 26.3 s
   (4.8x), and the malloc share went 53 % -> 0.1 % — the allocator is
   eliminated, not reduced.
2. **The sigma build is threaded via binned scatter, not a gather rewrite.**
   The scatter is kept (preserving the outer sparsity skip) and accumulates
   into partial vectors summed in fixed order, binned by `j / bin_size` under
   `schedule(static, bin_size)`. 3.54x at 4 threads against a modelled ~3.7x
   ceiling:

   | threads | 1 | 2 | 4 | 8 |
   |---|---|---|---|---|
   | N2/STO-3G | 27.59 s | 14.66 s | 7.79 s | 5.71 s |
   | speedup | 1.00x | 1.88x | 3.54x | 4.83x |

   Memory is `kBins x dim x 8` bytes — independent of thread count, 7.4 MB at
   N2.

## Validation strategy that should remain in place

- Bitwise comparison against the pre-change binary, and byte-identical output
  across `OMP_NUM_THREADS` = 1/2/4/8
- All 19 FCI/CASSCF/RASSCF/FCIDUMP/RI regression cases, including both
  iterative gates (`o2_fci_rohf_sto3g`, `be_fci_spherical_631gd`) and the
  `water_casscf_sa2_sto3g_sad_guess_uphill` canary — `slater_condon_element`
  is public and consumed by CASSCF/RASSCF response code, so those paths are
  downstream and must stay green
- Post-change reprofiling to confirm no new bottleneck was introduced

## What was measured after the fix

Serial profile is unchanged from post-F1, confirming the binning costs nothing
measurable:

| frame | post-F1 | post-F3, 1 thread |
|---|---|---|
| `apply_ci_hamiltonian` | 55.0 % | 55.4 % |
| `slater_condon_element` | 19.6 % | 19.9 % |
| `apply_creation` | 10.2 % | 9.9 % |
| `parity_between` | 7.3 % | 6.7 % |
| `apply_annihilation` | 6.3 % | 6.4 % |

The 7.4 MB per-call `partials` allocation was suspected and is refuted: the
whole malloc family is 0.1 % and `memset` zeroing is 8 samples (0.0 %), since
Eigen reuses the allocation and the zeroing is trivial next to the
enumeration. The serial reduction itself is negligible: `n_bins x dim` =
921 600 adds per call, under a millisecond against 7.79 s.

At 4 threads the only new frame is the barrier, `__psynch_cvwait` at 3.0 %.
Per-thread idle is uneven:

| thread | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| idle | 0.4 % | 4.4 % | 2.5 % | 11.8 % |

4.8 % total idle, so removing the imbalance entirely is worth only ~1.05x on
top of 3.54x — not worth it, since determinism forbids `dynamic` and 5 % does
not justify it. If it is ever wanted, the move is more bins (smaller `kBins`)
under static scheduling, never `dynamic`.

## Remaining architecture concern

Two facts the refuted gather rested on are sound and need not be re-litigated
if anyone revisits threading via a gather formulation: the reachability
relation is symmetric (verified by direct enumeration — 0 asymmetric edges
across n_act 4/5/6 including an open-shell case), and `H` is real symmetric,
which `build_ci_hamiltonian_dense` already depends on. The gather is correct,
just slower — nothing here rules it out permanently, only at the currently
reachable problem sizes.
