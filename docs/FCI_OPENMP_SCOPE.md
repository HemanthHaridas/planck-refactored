# Scope: threading the FCI sigma build

**Scope for in-flight work. Not started.** Opened 2026-08-30 from a measurement,
after answering "is FCI OpenMP enabled?" with "no, not on the path that matters".

## The measurement

`build-full` is genuinely OpenMP-enabled (`-fopenmp`, `-DUSE_OPENMP`, libgomp
linked). N2/STO-3G, ndet = 14 400:

```
OMP_NUM_THREADS=1    121.9 s
OMP_NUM_THREADS=4    123.6 s      <- flat
OMP_NUM_THREADS=8    100.0 % CPU  <- one core of eight
```

`apply_ci_hamiltonian` (`ci.cpp:437-622`) — the iterative sigma build used by
**every** determinant space above `dense_threshold = 500` — has **zero**
`#pragma omp`. The one pragma in the file (`:221`) is on the **dense** build,
which runs only *below* 500 determinants and therefore never fires when it would
matter. `rdm.cpp` is threaded (6 pragmas); `fci.cpp` and `strings.cpp` are not.

**Leaf-sample profile (21 048 samples, 1 thread) — and this is the surprise:**

| frame | share |
|---|---|
| `malloc` / `free` family | **~53 %** |
| `apply_ci_hamiltonian` | 12.0 % |
| `get_excitation` | 5.9 % |

**Half the time is allocation, not arithmetic.** That changes the order of work:
threading a function that spends half its time in the allocator parallelizes the
allocator, and `malloc` contention across threads is a well-known way to get
*negative* scaling. **Fix the allocation first** — it is also the smaller,
lower-risk change, and it is measurable on its own.

## The two allocation sites

Both are file-local (anonymous namespace in `ci.cpp`), so the blast radius is one
file:

**1. `get_excitation` (`ci.cpp:65`)** returns
`std::pair<std::vector<int>, std::vector<int>>` **by value**, and
`slater_condon_element` calls it for *both* spin channels at six call sites
(`:295, :317, :340, :361, :383, :384`). Up to four heap allocations per matrix
element — for vectors that hold **at most two entries each**, because a
Slater-Condon element vanishes beyond a double excitation.

**2. `occupied_orbitals` (`ci.cpp:24`)** returns `std::vector<int>` by value and
is called twice per outer iteration of the sigma loop (`:504-505`). It
`reserve`s `n_orb` and fills only `n_elec` — one allocation each, per
determinant, per sigma application.

## The loop structure, and why it is not the CC shape

```cpp
for (int j = 0; j < dim; ++j)          // over KET determinants
{
    ...
    accumulate(bra_a, bra_b, ket_a, ket_b, cj);   // -> sigma(it->second) += ...
}
```

The outer loop is over **columns** (kets `j`), and each iteration scatters into
`sigma(i)` at indices found by a hash lookup (`space.det_lookup`). **This is a
scatter, not the disjoint-slice write the CC residual nests had.** Two threads
processing different `j` can and will target the same `sigma(i)`.

So the CC recipe does **not** transfer. The options, in order of preference:

- **Per-thread partial vectors, summed in fixed thread order.** Each thread owns
  a private `sigma_t` of length `dim`; after the parallel region, sum them as
  `for t in 0..nthreads: sigma += sigma_t`. **Fixed index order makes it bitwise
  thread-count-invariant**, which is the DFT J/K discipline and the property every
  other threaded path in Planck holds. Cost: `nthreads × dim × 8` bytes — at
  ndet = 14 400 and 8 threads that is ~0.9 MB, trivial; at ndet = 10⁶ it is 64 MB,
  which is the point where this stops being free.
- **Gather (row-driven) reformulation.** Restructure so each thread *owns* a
  slice of `sigma` and pulls contributions in. No reduction at all, and it scales
  to any `dim` — but it is a genuine rewrite of the excitation enumeration, not a
  pragma.

**Never `#pragma omp atomic` on `sigma(...)`, and never a completion-order
reduction.** That is exactly the DFT-grid jitter defect
(`dft_xc_reduction_determinism`), and this codebase has spent real effort keeping
every other parallel path bitwise thread-count-invariant.

## Gate coverage — read this before writing a test

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
broken threaded sigma build unchanged.** A gate built on `h2_fci_sto3g` or
`water_fci_sto3g` proves nothing here — the same trap that let
`ch4_rccsdt_sto3g` sit green for its entire life while never running the kernel
it was added to protect.

`slater_condon_element` is **public** (`ci.h:43`) and also consumed by
`src/post_hf/casscf/response.cpp`, so the 12 CASSCF/RASSCF cases are downstream of
step F1 and must stay green.

## Steps

Ordered so the cheapest step can kill the expensive ones, and so each is
independently verifiable and independently revertible.

### F1 — **DONE (2026-08-30). 4.8x, and the allocator is gone, not merely reduced.**

Return a fixed-size result instead of two heap vectors: a small struct with
`std::array<int,2> ann, cre;` plus `int n_ann, n_cre;`. A Slater-Condon element is
zero beyond a double excitation, so **2 is the true maximum**, not a guess — but
assert it (`n_ann <= 2`) rather than truncating silently, because a violated
assumption here would produce a wrong matrix element rather than a crash.

Update the six call sites; all are in `ci.cpp`.

**Result.** `get_excitation` now returns a fixed-capacity `Excitation` struct
(`std::array<int,2> ann, cre` plus counts) instead of a pair of heap vectors.
Six call sites updated; all in `ci.cpp`.

| | before | after |
|---|---|---|
| N2/STO-3G, 1 thread | 125.7 s | **26.3 s** (**4.8x**) |
| `be_fci_spherical_631gd` | ~46.5 s | **7.6 s** |
| `malloc`/`free` share of profile | ~53 % | **0.1 %** |

**Correctness: bitwise identical.** `o2_fci_rohf_sto3g` `-147.7441885517` and
`be_fci_spherical_631gd` `-14.6139425466` match the pre-change binary digit for
digit, as does N2's correlation energy `-0.8864061248`. All 7 FCI and all 11
CASSCF/RASSCF cases pass. The extended suite is 111/115 with the 4 failures being
the pre-existing `PLANCK_CC_ARBITRARY_LOWER_RANKS=ON` determinant-routing
interaction in the **CC** cases — verified identical on a default build, and
untouched by a change confined to `ci.cpp`.

**The 4.8x exceeded the profile's implication, and the reason is worth keeping.**
A 53 % malloc share bounds the direct saving at ~2.1x by Amdahl. Getting 4.8x
means the allocator was costing more than its own samples: per-element
`malloc`/`free` churns the heap, and removing it also removed cache pressure and
allocator bookkeeping attributed to *other* frames. **A profile share is a lower
bound on what removing that work is worth, not an estimate of it** — the inverse
of the CC merge, where an operator-count model *over*-promised because it treated
unequal work as equal.

The post-F1 profile is now what one would want: `apply_ci_hamiltonian` 55.0 %,
`slater_condon_element` 19.6 %, `apply_creation` 10.2 %, `parity_between` 7.3 %,
`apply_annihilation` 6.3 %. Real arithmetic, no allocator.

### F2 — remove the `occupied_orbitals` allocation (~S, no threading)

Same treatment: fill a caller-provided buffer, or return a small fixed-capacity
type (`n_act <= 31` from `kMaxPackedSpatialOrbitals`, so a `std::array<int,31>`
plus a count is stack-friendly).

- **Verify:** as F1 — bitwise-identical energies, and a measured wall-time delta.

**F1 and F2 are worth doing on their own merits** even if the threading is never
done, and they should be landed and measured separately so each has its own
number.

### F3 — thread the sigma loop with per-thread partials (~M)

Only after F1/F2, so the allocator is out of the hot path and threading is not
fighting it.

`#pragma omp parallel for schedule(dynamic)` over `j`, each thread accumulating
into its own `sigma_t`, then a **serial, fixed-order** sum over threads.
`schedule(dynamic)` because the per-`j` work varies with occupation pattern.

- **Verify (correctness, non-negotiable):** `o2_fci_rohf_sto3g` and
  `be_fci_spherical_631gd` produce **bitwise-identical** energies at
  `OMP_NUM_THREADS` = 1/2/4/8 and against the serial baseline. Not "to 1e-10" —
  bitwise, as the DFT J/K builds and the CC kernels were verified.
- **Verify (the pragma fires):** CPU utilization above 100 % before any timing is
  interpreted. The unthreaded baseline is exactly 100.0 %, so this is a free check
  that costs one `ps` call and catches an inert pragma immediately.
- **Verify (speed):** N2/STO-3G at 1/2/4/8 threads against the post-F2 serial
  baseline.

### F4 — decide about memory, and about `dense_threshold` (~S)

The per-thread partials cost `nthreads × dim × 8` bytes. Record where that stops
being acceptable and either cap the thread count for large `dim` or document the
ceiling. **Do not silently allocate 64 MB of scratch at ndet = 10⁶.**

Separately: `dense_threshold = 500` was chosen when nothing was threaded. If the
threaded sigma build beats the dense path at some size, that constant should move
— but **measure it, do not assume**, and change it in its own commit so the
regression suite attributes any behaviour change correctly.

## What this must not do

- **Do not thread before F1/F2.** Half the runtime is in the allocator; threading
  that first parallelizes `malloc` contention and can scale *negatively*.
- **Do not use `omp atomic` or a completion-order reduction on `sigma`.** The
  DFT-grid jitter defect is exactly this, and it is the one determinism failure
  this codebase has already paid for.
- **Do not gate on `h2_fci_sto3g` or `water_fci_sto3g`.** Both run the dense path
  and cannot see a defect in the threaded sigma build.
- **Do not accept "energies match to 1e-10".** Every other threaded path here is
  bitwise thread-count-invariant; there is no reason for this one to be the
  exception, and accepting a tolerance hides exactly the reduction-order bug the
  fixed-order sum exists to prevent.
- **Do not change `slater_condon_element`'s signature.** It is public and CASSCF
  consumes it; F1 is an internal representation change behind it.

## Key code locations

| what | where |
|---|---|
| the unthreaded sigma build | `apply_ci_hamiltonian`, `src/post_hf/ci/ci.cpp:437-622` |
| the outer loop to thread | `ci.cpp:496` (`for j` over kets) |
| the scatter that forbids a naive pragma | `accumulate` lambda, `ci.cpp:484-493` |
| allocation site 1 | `get_excitation`, `ci.cpp:65` (6 call sites: `:295,317,340,361,383,384`) |
| allocation site 2 | `occupied_orbitals`, `ci.cpp:24` (called `:504-505`) |
| the dense path that is already threaded | `ci.cpp:221` (only below `dense_threshold = 500`) |
| the public entry CASSCF also uses | `slater_condon_element`, `src/post_hf/ci/ci.h:43` |
| the determinism precedent to imitate | DFT J/K builds, `src/dft/driver.cpp` |
| the determinism defect to avoid | `dft_xc_reduction_determinism` note |
| the two cases that actually gate this | `o2_fci_rohf_sto3g`, `be_fci_spherical_631gd` |
| a larger timing fixture | `tests/inputs/exploratory/fciqmc/n2_fci_sto3g.hfinp` (ndet 14 400) |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
