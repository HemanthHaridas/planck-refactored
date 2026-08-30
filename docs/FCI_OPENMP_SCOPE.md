# Scope: threading the FCI sigma build

**DONE (2026-08-30). F1 landed 4.8x; F3 landed 3.54x at 4 threads.** Opened from a
measurement, after answering "is FCI OpenMP enabled?" with "no, not on the path
that matters". Two of the three steps are landed and measured; F2 (the
`occupied_orbitals` allocation) is the only one left, and it is independent.

**The two findings most worth carrying out of this document:**

1. **The gather route this scope RECOMMENDED is refuted** — built, correct, and
   2.2-2.4x slower, because the outer `|c| < 1e-15` skip is a sparsity
   exploitation carrying asymptotic weight, not the summation-order detail R6
   took it for. See the REFUTED banner below.
2. **A fixed-order reduction is necessary but not sufficient for thread-count
   invariance.** The *partition of work into accumulators* must also be
   deterministic. Both the `schedule(dynamic)` this scope recommended and
   thread-id-keyed buffers break it. See the fallback section.

This doc keeps its scope shape because the F2 step is still open; when F2 lands it
should be rewritten as an answer to "how was the FCI sigma build made fast?"

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

## The loop structure, and why it is not the CC shape *as written*

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

So the CC recipe does **not** transfer *to this loop as written*. Two options:

- **Invert it to a gather** — run over bras and pull, so each thread owns a
  disjoint slice of `sigma` and the CC recipe applies after all. **This is the
  recommendation**; F3 below establishes that it is exact and that it costs less
  code, not more. An earlier revision of this section called it "a genuine rewrite
  of the excitation enumeration" — that was wrong: the enumeration is unchanged
  and only the `accumulate` lambda flips from write to read.
- **Keep the scatter and add per-thread partial vectors**, summed in fixed thread
  order. Correct, and bitwise thread-count-invariant, but it needs
  `nthreads × dim × 8` bytes of scratch and a reduction that the gather does not.
  The fallback, not the plan.

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

### F3 — thread the sigma loop (~M) — **invert it to a gather; the scatter is avoidable**

F1 changed the shape of this step. The allocator is gone (53 % -> 0.1 %), so the
profile is now real arithmetic — `apply_ci_hamiltonian` 55.0 %,
`slater_condon_element` 19.6 %, `apply_creation` 10.2 %, `parity_between` 7.3 %,
`apply_annihilation` 6.3 % — and **all of it is inside the loop F3 threads**.
Amdahl on ~98 % parallel gives ~3.7x at 4 threads. That is the target.

#### Blast radius — inventoried, not assumed

**Two callers, both outside any parallel region** (verified, because nested
parallelism would silently oversubscribe):

| caller | site | enclosing `omp` region? |
|---|---|---|
| the internal Davidson driver | `ci.cpp:775` (`sigma_apply` lambda) | none |
| CASSCF's `CISigmaApplier` | `casscf/casscf.cpp:894` | **no** — the nearest `parallel for` is at `:589` and closes at `:611` |

That second one had to be checked rather than assumed: `casscf.cpp` *does* carry
a `#pragma omp parallel for` over roots, and if it enclosed the sigma call this
would be nested parallelism. It does not.

**Downstream consumers of a changed `sigma`:** none directly — `sigma` is an
out-parameter written by this function and consumed by Davidson. The 11
CASSCF/RASSCF regression cases exercise the CASSCF caller and are the gate for it.

**State the loop body touches, and its mutability:**

| state | access | thread-safe as-is? |
|---|---|---|
| `space` (`dets`, `spin_dets`, `det_lookup`) | `const&`, read-only; `det_lookup.find` only | **yes** — concurrent `find` on `unordered_map` is safe; there is no `operator[]` |
| `a_strs`, `b_strs`, `h_eff`, `ga`, `n_act` | `const&` / by value | yes |
| `c` | `const&` | yes |
| **`sigma`** | **`+=` at hash-lookup indices** | **NO — this is the entire problem** |

**The only mutation in the loop body is `sigma(...) += ...`.** Everything else is
read-only. That is a small and well-defined risk surface.

#### MEASURED AND REFUTED (2026-08-30): the gather is 2.2-2.4x SLOWER. Do not build it.

F3a was implemented exactly as specified below and **measured**. It is
**numerically correct** — `o2_fci_rohf_sto3g` `-147.7441885517` and
`be_fci_spherical_631gd` `-14.6139425466` and N2's `-0.8864061248` all match the
pre-F3 values to **every printed digit** (the anticipated ~1e-12 summation-order
delta did not even materialize at print precision). But it is far slower:

| case | scatter (post-F1) | gather (F3a) | |
|---|---|---|---|
| `be_fci_spherical_631gd` | 7.6 s | **16.4 s** | 2.2x slower |
| N2/STO-3G (ndet 14 400) | 26.3 s | **63.3 s** | 2.4x slower |

The scope's own stop condition — *"if it is materially slower, stop"* — fired, and
the work stopped there. The change is stashed, not committed.

**Why: the `|c| < 1e-15` skip was not a summation-order detail. It was doing
algorithmic work, and the gather structurally cannot keep it.**

R6 below treats the moved skip purely as a reordering concern. That is wrong. In
the scatter the test sits on the OUTER loop, so a negligible ket skips the entire
126-line enumeration — hundreds of neighbours — in one comparison. In the gather
the outer index is the bra, whose `sigma(i)` must be computed regardless of
`c(i)`, so **every outer iteration runs the full enumeration** and the skip
survives only as an inner-loop test that saves one `slater_condon_element` call.

That matters because **the vectors this is applied to are extremely sparse**, which
the design never checked:

- `davidson` (`ci.cpp:130-136`) starts from unit vectors on the lowest-diagonal
  determinants — `V(order(k), k) = 1.0`, i.e. `init_cols = max(2*nroots, 4)`
  nonzeros in `dim`. The QR of already-orthonormal unit columns keeps them sparse.
- Worse, `solve_ci` (`ci.cpp:743`) reconstructs a dense `H` by calling
  `sigma_apply` with `Eigen::VectorXd::Unit(dim, j)` — **exactly one nonzero** —
  once per column. Under the scatter that is O(dim) enumerations total; under the
  gather it is O(dim^2).

So the scatter is not merely a different traversal of the same work: on sparse
input it does asymptotically less of it. The gather pays full price on every
application.

**What this does NOT invalidate.** The two facts the inversion rests on were
verified and both hold — reachability is symmetric (0 asymmetric edges), and `H`
is real symmetric (`build_ci_hamiltonian_dense` at `:264` already depends on it,
assigning `H(i,j) = H(j,i)` from one evaluation). The gather is *correct*; it is
just slower. Any future attempt does not need to re-litigate the symmetry.

**Consequence for threading: the fallback is now the plan.** Threading the sigma
build means the scatter with **per-thread partial vectors summed in fixed thread
order** (see "If the gather turns out to be wrong" below), which preserves the
outer skip. Its `nthreads x dim x 8` bytes is the price of keeping the sparsity
exploitation. Note that the memory bound quoted there is a worst case over a dense
`c`; with the outer skip intact, most threads touch few entries.

**Generalizable lesson.** A cheap-looking guard on an outer loop can be carrying
asymptotic weight. Before moving a skip inward to enable a restructuring, measure
what fraction of the outer iterations it eliminates on the *actual* inputs — here
the answer was "nearly all of them, in early Davidson iterations and in the entire
dense-reconstruction path", and no amount of threading (max ~3.7x at 4 threads)
would have repaid a 2.4x serial loss plus that asymptotic change.

#### The route: invert the loop to a gather

The loop currently pushes; invert it to pull:

```
for j:  for each i reachable from j:   sigma(i) += H(i,j) * c(j)      // scatter
for i:  for each j reachable from i:   sigma(i) += H(i,j) * c(j)      // gather
```

Then **each thread owns a disjoint slice of `sigma`** — the CC-residual shape,
where a bare `#pragma omp parallel for` is correct with no reduction, no partial
buffers, and bitwise thread-count invariance for free.

**Two facts make this exact, and both are already established in this codebase.**

**1. The reachability relation is symmetric.** The enumeration generates every
determinant reachable from the ket by <=2 excitations preserving the alpha and
beta counts. The inverse excitation (swap annihilated and created orbitals) maps
`i` back to `j` and is itself such an excitation, so it lies in the set generated
from `i`. Verified by direct enumeration, including an open-shell case:

| n_act | na/nb | dim | edges | **asymmetric** |
|---|---|---|---|---|
| 4 | 2/2 | 36 | 972 | **0** |
| 5 | 2/3 | 100 | 5 500 | **0** |
| 6 | 3/3 | 400 | 47 200 | **0** |

**2. `H` is real symmetric, and the codebase already depends on it.**
`build_ci_hamiltonian_dense` (`ci.cpp:241-267`) computes only the upper triangle
and assigns `H(i, j) = H(j, i) = v` at `:264`. If `slater_condon_element` were
not symmetric under bra/ket exchange, **the dense path would already be wrong** —
and it is gated by every FCI and CASSCF case.

Together: `sigma_i = Σ_j H_ij c_j` can be computed by enumerating **from `i`** and
gathering `c_j` — same enumeration, same matrix element, a read where there was a
write.

#### What the gather does to each risk

| risk | under the gather |
|---|---|
| **R1 — the scatter** | **gone.** Disjoint slices; two threads never touch the same `sigma(i)` |
| **R2 — determinism** | **gone.** No cross-thread reduction, so nothing to order. Still verified bitwise |
| **R3 — memory** | **gone.** No per-thread buffers (the scatter route needed up to 106 MB at water/6-31G) |
| **R4 — load imbalance** | **remains.** Work per `i` varies with occupation pattern; `schedule(dynamic)`, as the dense build at `:252` already uses |
| **R5 — the fallback path** | **remains, and stays serial.** `ci.cpp:490-511` is effectively dead: `det_lookup` is always populated at `:461`, so the `empty()` guard is false in normal operation. Threading a dead path adds a second shape to maintain for no measured benefit |
| **R6 — the moved skip** | **new, and the only real cost.** See below |

**R6 — the `|c| < 1e-15` skip moves, changing summation order.** The current loop
skips a ket *before* enumerating; a gather cannot, because it does not know `c_j`
until after the lookup. The skip moves inside the lambda, testing `c(it->second)`.
That changes *which* terms are skipped and therefore the order they are summed, so
**the gather is not expected to be bitwise identical to today's serial result** —
it is a different, equally valid summation.

**This is the one place the usual bitwise discipline cannot apply, so it is
handled explicitly rather than waived:** F3a establishes the gather's own serial
result as the new reference, confirms it agrees with the current one to ~1e-12,
and records the delta in the commit. Thread-count invariance is then verified
bitwise against *that*. **Do not silently rebaseline.**

#### Why this is less code, not more

The 126-line loop body is unchanged — it already enumerates the neighbours of one
determinant. Only `accumulate` changes:

```cpp
sigma(it->second) += hij * coeff;      // scatter: write to a looked-up index
acc              += hij * c(it->second);  // gather: read from it, accumulate locally
```

with the caller writing `sigma(i) = acc` once per outer iteration. **The scatter
lambda becomes a gather lambda; nothing else moves.** That the code already
funnels every write through a single lambda is what keeps this a small diff.

**Excluded shapes** — any of these means the change has gone wrong:

- a second copy of the excitation enumeration for the threaded case;
- an `if (threaded) ... else ...` branch inside the term bodies;
- threading the dead fallback path for symmetry;
- an `omp atomic` at the write site;
- a tunable for thread count or buffer size.

#### Steps

**F3a — invert the loop to a gather. Serial, no OpenMP.**
Same enumeration; `accumulate` becomes a gather; the `|c|` skip moves inside.

*Verify:*
- both iterative cases agree with the pre-F3 energies to ~1e-12
  (`o2_fci_rohf_sto3g` `-147.7441885517`, `be_fci_spherical_631gd`
  `-14.6139425466`);
- all 7 FCI + 11 CASSCF/RASSCF cases green;
- N2/STO-3G within noise of 26.3 s — this step is a reformulation, not a
  speed-up, and **if it is materially slower, stop**: the gather is doing more
  lookups than the scatter did and that needs understanding before threading
  hides it;
- **record the exact serial energies.** They are the reference for F3b.

**F3b — add the parallel region.**
`#pragma omp parallel for schedule(dynamic)` over `i`. No buffers, no reduction.

*Verify, in this order:*
1. **CPU > 100 % at 4 threads.** The serial baseline is exactly 100.0 %, so this
   costs one `ps` call and catches an inert pragma before any timing is read.
2. **Bitwise identical to F3a's serial result** at `OMP_NUM_THREADS` = 1/2/4/8.
3. All 7 FCI + 11 CASSCF/RASSCF cases green.
4. Speed against the post-F1 26.3 s. Expect up to ~3.7x at 4 threads; below ~2x,
   check `schedule` and imbalance before concluding the lever is absent.

**F3c — revisit `dense_threshold` (separate commit).**
`dense_threshold = 500` was chosen when nothing was threaded. If the threaded
sigma build now beats the dense path at some size, the constant should move — but
**measure it, and change it in its own commit** so the suite attributes any
behaviour change correctly.

#### The fallback — BUILT AND LANDED (2026-08-30). 3.54x at 4 threads.

The gather was refuted, so the fallback became the plan. It is the scatter with
partial vectors summed in fixed order — never `omp atomic`, never
completion-order, which is the DFT-grid jitter defect.

**Measured** (N2/STO-3G, ndet 14 400, against the F1 serial 26.3 s):

| threads | time | speedup |
|---|---|---|
| 1 | 27.59 s | 1.00x |
| 2 | 14.66 s | 1.88x |
| 4 | **7.79 s** | **3.54x** |
| 8 | 5.71 s | 4.83x |

3.54x at 4 threads against the modelled ~3.7x ceiling. `be_fci_spherical_631gd`
7.6 s -> 3.42 s. Energies **byte-identical at 1/2/4/8 threads** on all three
cases and equal to the F1 serial values (`-107.6529998854`, `-14.6139425466`,
`-147.7441885517`).

**This section previously said "per-thread partial vectors summed in fixed thread
order". That design was built first and it is WRONG — it is not thread-count
invariant.** Two separate determinism defects were found, each only by the
byte-diff, neither by reasoning:

1. **`schedule(dynamic)`** — which this scope recommended for load balance (R4).
   Dynamic assignment gives a buffer a *different subset* of terms per run, so the
   sums inside it reassociate. Measured: the same 4-thread config produced two
   different last digits across 5 identical runs of Be/6-31g*.
2. **Indexing the buffers by `omp_get_thread_num()`.** Even with
   `schedule(static)`, each buffer's *contents* depend on the thread COUNT, so 8
   threads disagreed with 1/2/4 in the last digit.

**The generalizable half: a fixed-order reduction is necessary but NOT
sufficient.** What has to be deterministic is the **partition of work into
accumulators**, not merely the order they are summed. The first version had the
fixed-order sum and was still non-deterministic.

**What landed instead:** bin by a fixed function of `j` — `partials[j / bin_size]`
with a constant `kBins = 64` — and `schedule(static, bin_size)` so one chunk is
exactly one bin (which is also what gives a thread exclusive access to its bin).
Bin `b` then receives exactly the same `j` values in the same order regardless of
thread count, so every partial is bit-identical and so is the fixed-order sum over
bins.

Memory is `kBins x dim x 8` bytes — **independent of thread count**, 7.4 MB at N2.
The earlier `nthreads x dim x 8` estimate (0.9 MB at N2, 106 MB at water/6-31G)
does not apply; note also that the 106 MB figure was for water/6-31G at ndet 1.66M,
which this scope elsewhere prices at tens to hundreds of hours for FCI, so it is
not a case anyone runs.

#### Post-F3 profile (N2/STO-3G, `sample`, same tool as the numbers above)

**Serial is unchanged from post-F1, so the binning costs nothing measurable:**

| frame | post-F1 | post-F3, 1 thread |
|---|---|---|
| `apply_ci_hamiltonian` | 55.0 % | **55.4 %** |
| `slater_condon_element` | 19.6 % | **19.9 %** |
| `apply_creation` | 10.2 % | **9.9 %** |
| `parity_between` | 7.3 % | **6.7 %** |
| `apply_annihilation` | 6.3 % | **6.4 %** |

**The 7.4 MB per-call `partials` allocation was suspected and is refuted**: the
whole malloc family is **0.1 %** and `memset` (zeroing the buffers) is **8 samples,
0.0 %**, despite Davidson calling `apply_ci_hamiltonian` once per subspace vector
per iteration. Eigen reuses the allocation and the zeroing is trivial next to the
enumeration. The serial reduction is likewise negligible — `n_bins x dim` = 921 600
adds per call, under a millisecond against 7.79 s.

**At 4 threads the only new frame is the barrier**, `__psynch_cvwait` at 3.0 %
overall. Per-thread idle is uneven, which is the residual imbalance R4 predicted:

| thread | idle | working |
|---|---|---|
| t0 | 0.4 % | 99.6 % |
| t1 | 4.4 % | 95.6 % |
| t2 | 2.5 % | 97.5 % |
| t3 | **11.8 %** | 88.2 % |

**4.8 % total idle across the four threads.** Removing the imbalance entirely would
be worth only **~1.05x** on top of the current 3.54x, which is why `schedule(static,
bin_size)` is left as-is: the determinism requirement forbids `dynamic`, and the
balance it would buy is ~5 %. **If that is ever wanted, the move is more bins
(smaller `kBins`) with static scheduling — never `dynamic`, which breaks
thread-count invariance.**

#### Acceptance

- **Bitwise-identical energies across `OMP_NUM_THREADS` = 1/2/4/8** — **MET**, on
  all three cases, byte-compared. This is the gate; a tolerance is not.
- Equal to the F1 serial energies — **MET exactly**, so the ~1e-12 allowance the
  gather needed was not required here (the scatter preserves summation order
  within a bin).
- All 7 FCI + 11 CASSCF/RASSCF cases green — **MET, 19/19**. Every FCI, CASSCF,
  RASSCF, FCIDUMP and RI case in the repo passes, including both iterative FCI
  gates (`o2_fci_rohf_sto3g`, `be_fci_spherical_631gd`), all four SA-CASSCF cases
  and the `water_casscf_sa2_sto3g_sad_guess_uphill` canary.
- The term-body code is **unchanged** — **MET**, verifiable by
  `git diff | grep '^-'`: the only deleted lines are the old `accumulate` lambda.
- No new tunable, no second code path, no `omp atomic` — **MET**. (`kBins` is a
  `constexpr` implementation detail, not a user-facing knob.) The "no per-thread
  buffers" clause was written for the gather and does not survive: the fallback
  necessarily has buffers, just keyed by bin rather than by thread.

## What this must not do

- **Do not thread before F1.** (F1 is done; this stands as the reason the order
  was chosen.) Half the runtime was in the allocator, and threading that first
  parallelizes `malloc` contention, which is a known route to *negative* scaling.
- **Do not touch the term bodies (`ci.cpp:527-652`).** The whole point of F3's
  design is that the 126 lines of excitation enumeration are untouched and the
  change is a wrapper around the single `accumulate` write point. A diff that
  reaches into those lines has become the second implementation this scope exists
  to avoid.
- **Do not use `omp atomic` or a completion-order reduction on `sigma`.** The
  DFT-grid jitter defect is exactly this, and it is the one determinism failure
  this codebase has already paid for.
- **Do not gate on `h2_fci_sto3g` or `water_fci_sto3g`.** Both run the dense path
  and cannot see a defect in the threaded sigma build.
- **Do not accept "energies match to 1e-10" for the THREADING.** Every other
  threaded path here is bitwise thread-count-invariant, and accepting a tolerance
  would hide exactly the reduction-order bug the design exists to prevent. The one
  documented exception is F3a's reformulation itself, where the moved `|c|` skip
  changes summation order — that comparison is ~1e-12 against the *pre-F3* result,
  once, with the delta recorded. Everything after F3a is bitwise against F3a.
- **Do not silently rebaseline.** If F3a's serial energies differ from the pre-F3
  ones by more than ~1e-12, that is a defect in the inversion, not a new normal.
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
