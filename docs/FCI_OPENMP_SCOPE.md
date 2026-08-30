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

### F3 — thread the sigma loop (~M) — **blast radius and risks inventoried first**

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

#### The risks, in order

**R1 — the scatter (the real one).** The loop runs over ket determinants `j` and
writes `sigma(it->second)` where `it` comes from a hash lookup. Two threads on
different `j` *will* target the same output element. This is **not** the
disjoint-slice write the CC residual nests had, so the CC recipe does not
transfer.

**R2 — determinism.** Every other threaded path in Planck is bitwise
thread-count-invariant, by design and by gate. A naive `reduction` or
`omp atomic` gives completion-order summation and therefore thread-count-dependent
rounding — the DFT-grid jitter defect exactly
(`dft_xc_reduction_determinism`). **A tolerance-based gate would hide this**,
which is why the acceptance below is bitwise.

**R3 — memory.** Per-thread partial vectors cost `nthreads × dim × 8` bytes:

| dim | 4 threads | 8 threads |
|---|---|---|
| 8 281 (`be_fci`) | 0.26 MB | 0.53 MB |
| 14 400 (N2) | 0.46 MB | 0.92 MB |
| 213 444 (HF/6-31G) | 6.8 MB | 13.7 MB |
| 1 656 369 (water/6-31G) | 53 MB | **106 MB** |

Free at every size the suite reaches; material only in the regime FCI cannot
afford anyway. **It must still be bounded rather than discovered.**

**R4 — load imbalance.** Work per `j` varies with occupation pattern and with the
`|cj| < 1e-15` skip, so a static schedule would imbalance. `schedule(dynamic)` is
the answer, and it is what the dense build at `:252` already uses.

**R5 — the fallback path.** `ci.cpp:490-511` has a different structure (dense
`O(dim²)`, inner loop over `i`). It is **effectively dead**: `det_lookup` is
always populated at `:461`, so the guard `det_lookup.empty()` is false in normal
operation. **Leave it serial.** Threading a dead path adds a second code shape to
maintain for no measured benefit.

#### The design — one shape, not a special case

The constraint that keeps this from becoming spaghetti: **the term-body code must
not change at all.** The 130 lines of excitation enumeration (`:527-652`) are the
part a reader needs to follow, and they are already correct.

So the entire change is a *wrapper* around them:

```cpp
// serial today:
Eigen::VectorXd &target = sigma;
for (int j = 0; j < dim; ++j) { ... accumulate(...) -> target(i) += ... }

// threaded: the loop body is IDENTICAL; only what `accumulate` writes into,
// and a fixed-order sum afterwards, are new.
```

Concretely: hoist the existing `accumulate` lambda to capture a
`Eigen::VectorXd &out` instead of `sigma` directly, give each thread its own
`out`, and sum them in **thread-index order** after the region. The 130 lines in
between are untouched, and `accumulate` — which already exists as the single
write point — is the natural and only seam. **That is why this is a small diff:
the code already funnels every write through one lambda.**

**What would be spaghetti, and is therefore excluded:**

- a second copy of the excitation enumeration for the threaded case;
- a `if (threaded) ... else ...` branch inside the term bodies;
- threading the dead fallback path to make it "symmetric";
- an `omp atomic` sprinkled at the write site (fast to type, breaks R2);
- a tunable for the number of partial buffers.

#### Steps

**F3a — hoist the write target (~S, no threading, no behaviour change).**
Change `accumulate` to write into a referenced accumulator that is `sigma` itself.
Pure refactor.
*Verify:* the two iterative cases bitwise identical; N2 timing unchanged within
noise. **This step must measure as a no-op** — if it does not, stop.

**F3b — add the parallel region (~M).**
`#pragma omp parallel for schedule(dynamic)` over `j`, per-thread `out`, serial
fixed-order sum.
*Verify, in this order:*
1. **CPU utilization > 100 %** at 4 threads. The serial baseline is exactly
   100.0 %, so this costs one `ps` call and catches an inert pragma before any
   timing is interpreted.
2. **Bitwise-identical energies** at `OMP_NUM_THREADS` = 1/2/4/8 **and** against
   the serial binary, on `o2_fci_rohf_sto3g` (`-147.7441885517`) and
   `be_fci_spherical_631gd` (`-14.6139425466`). Not "to 1e-10".
3. All 7 FCI + 11 CASSCF/RASSCF cases green.
4. Speed: N2/STO-3G at 1/2/4/8 against the post-F1 26.3 s. Expect up to ~3.7x at
   4; below ~2x, suspect imbalance and check the schedule before concluding.

**F3c — bound the memory (~S).**
Cap the partial-buffer count (fall back to serial, or to fewer threads) above a
documented `dim`, so R3 is a decision rather than an accident.

**F3d — revisit `dense_threshold` (~S, separate commit).**
`dense_threshold = 500` was chosen when nothing was threaded. If the threaded
sigma build now beats the dense path at some size, that constant should move —
but **measure it, and change it in its own commit** so the suite attributes any
behaviour change correctly.

#### Acceptance

- Bitwise-identical energies across `OMP_NUM_THREADS` = 1/2/4/8 and against
  serial, on both iterative cases. **This is the gate; a tolerance is not.**
- All 7 FCI + 11 CASSCF/RASSCF cases green.
- The term-body code (`:527-652`) is **unchanged** — verifiable by diff.
- No new tunable, no second code path, no `omp atomic`.

## What this must not do

- **Do not thread before F1.** (F1 is done; this stands as the reason the order
  was chosen.) Half the runtime was in the allocator, and threading that first
  parallelizes `malloc` contention, which is a known route to *negative* scaling.
- **Do not touch the term bodies (`ci.cpp:527-652`).** The whole point of F3's
  design is that the 130 lines of excitation enumeration are untouched and the
  change is a wrapper around the single `accumulate` write point. A diff that
  reaches into those lines has become the second implementation this scope exists
  to avoid.
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
