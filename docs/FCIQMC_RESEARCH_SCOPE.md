# Research scope: FCIQMC in Planck

**Research scope. Not started, and not yet justified — the first step is deciding
whether to build it at all.** Opened 2026-08-30.

FCIQMC (Booth/Thom/Alavi 2009) samples the FCI wavefunction with a population of
signed walkers evolving under the imaginary-time Schrödinger equation, rather than
storing a CI vector. It reaches determinant spaces that deterministic FCI cannot,
at the cost of a stochastic error bar.

## The one question this scope must answer first

**Does Planck have a problem that FCIQMC solves, and is a stochastic method
compatible with how this codebase validates itself?**

Both halves are in doubt, and the second is the harder one. **Do not start
implementing until Q1 and Q2 below are answered**, because a negative answer to
either kills the item, and both are cheap to answer.

### Q1 — where is the gap FCIQMC would fill?

The determinant ceiling is hard and known. `CIString` is a `std::uint64_t`
(`casscf_internal.h:21`), giving `kMaxPackedSpatialOrbitals = 31`. `fci.cpp`
fails loudly above it rather than truncating. Within that ceiling:

| n_act | ndet (half filling) | deterministic FCI |
|---|---|---|
| 10 | 63 504 | trivial |
| 14 | 1.2 × 10⁷ | feasible |
| 16 | 1.7 × 10⁸ | hard, direct-sigma only |
| 20 | 3.4 × 10¹⁰ | **out of reach** |
| 24 | 7.3 × 10¹² | **out of reach** |
| 31 | ~10¹⁷ | the packed ceiling |

So there *is* a window — roughly `n_act` 18-31 — where the determinant space is
addressable by the existing bitstring representation but far beyond a stored CI
vector. **That is the honest case for FCIQMC here.**

**But:** what actually wants that window? Planck's CASSCF is validated on active
spaces of CAS(8,6) and smaller; the whole regression suite tops out at 6 atoms.
**Name a target calculation before building anything** — a specific molecule and
active space someone wants and cannot currently run. If the answer is "none yet",
this is a capability looking for a use, and the right decision is to record that
and stop.

### Q2 — can a stochastic method live in this validation culture?

**This is the real obstacle, and it is cultural more than technical.** Planck's
regression discipline is built on exactness:

- **161 `metric_close` assertions**, the tightest at `atol 1e-9`.
- Every recent performance change in this codebase was gated on **bitwise
  identity** — the OpenMP work verified energies bit-for-bit across
  `OMP_NUM_THREADS`, the transpose merge and the operator-hoist likewise.
- There is **no RNG anywhere in `src/`** (`grep -rl "mt19937\|random_device"`
  returns nothing). FCIQMC would introduce the first stochastic component into a
  codebase with no precedent for testing one.

An FCIQMC energy is a mean with an error bar and a serial correlation time. It
cannot be gated at 1e-9 against a reference. Gating it requires machinery that
does not exist here:

- a **fixed-seed** reproducibility gate (same seed → same trajectory, bitwise),
  which is testable and should be mandatory;
- a **statistical** gate (energy within N σ of a deterministic FCI reference on a
  small system where both are computable), which needs a blocking analysis to get
  σ right — a naive standard error on correlated samples understates it badly.

**Answer Q2 by writing the gate first, against the existing FCI.** If a
fixed-seed + blocked-error gate cannot be made to pass reliably on a case where
FCI gives the exact answer, the method will not be maintainable here regardless of
how good the sampling is.

## The validation fixture — measured, 2026-08-30

Q2 needs a system where **both** methods run: FCI to give the exact answer, and
FCIQMC on a determinant space large enough that sampling is meaningful. Those pull
in opposite directions, so the window is narrow and worth pinning down before any
code.

**Measured on this machine** (`build/hartree-fock`, Release, serial):

| system | norb | na/nb | ndet | FCI wall |
|---|---|---|---|---|
| H2/STO-3G | 2 | 1/1 | 4 | instant |
| water/STO-3G | 7 | 5/5 | 441 | instant |
| H2/cc-pVDZ | 10 | 1/1 | 100 | instant |
| **Be/6-31g\*** | 14 | 2/2 | **8 281** | **46.5 s** |
| **N2/STO-3G** | 10 | 7/7 | **14 400** | **124.4 s** |
| C2/STO-3G | 10 | 6/6 | 44 100 | **> 10 min** (not run to completion) |
| water/6-31G | 13 | 5/5 | 1 656 369 | hours — see below |

**The recommendation: `N2/STO-3G` (ndet = 14 400, FCI in ~2 min).**

It is the smallest system that satisfies both constraints:

- **FCI is cheap enough to be a routine reference** — 2 minutes, so the gate can
  recompute it rather than hard-coding a number.
- **The determinant space is big enough for sampling to mean something.** This is
  the constraint people miss. Below a few thousand determinants a walker
  population of the usual size covers essentially the whole space, FCIQMC
  degenerates into a noisy exact diagonalization, and the gate proves nothing
  about sampling. At 14 400, a few-thousand-walker run is a genuine sample.
- **It has 7α/7β electrons**, so the excitation generator is exercised properly.
  Be/6-31g\* is comparable in ndet (8 281) but has only 2α/2β — a two-electron
  system cannot exercise double excitations between different occupied pairs,
  which is where `p_gen` bugs hide.

`Be/6-31g*` is the useful *second* fixture precisely because it differs in that
way: same order of ndet, very different electron count. If FCIQMC agrees on N2 and
disagrees on Be (or vice versa), the difference isolates the excitation generator.

**H2 and water/STO-3G are unsuitable as FCIQMC fixtures** despite being the
existing FCI regression cases: at 4 and 441 determinants the walker population
would exceed the space.

### The FCI reference cost grows fast — do not plan on a large one

Two measured points give an empirical **ndet^1.78** for the direct-sigma FCI, and
C2/STO-3G at 44 100 determinants already exceeded 10 minutes, so **1.78 is a
lower bound**. Extrapolating, water/6-31G (1.66 M determinants) is tens to
hundreds of hours.

**Consequence for the scope:** the deterministic reference is only available in
the ndet ≲ 10⁵ window. Any FCIQMC validation above that has no exact answer to
check against, which is exactly the regime FCIQMC exists for — and exactly why the
fixed-seed reproducibility gate matters more than the statistical one. **The
statistical gate can only ever run on small systems; the reproducibility gate runs
everywhere.**

Two caveats on those numbers, stated because the fit is thin: two points cannot
establish an exponent, and the two systems differ in *electron count* as well as
ndet (2+2 vs 7+7), so the fit conflates two variables. Treat it as
order-of-magnitude. `ci_max_dim` also defaults to 10 000 and must be raised
explicitly for anything larger — it fails loudly, which is correct.

Candidate inputs are committed under `tests/inputs/exploratory/fciqmc/`.

### The FCI reference is single-threaded, and half its time is `malloc`

Measured while establishing the fixture, because it changes what the reference
costs and it matters to FCIQMC directly (both share `slater_condon_element`).

**FCI does not thread.** On `build-full` (genuinely OpenMP-enabled: `-fopenmp`,
`-DUSE_OPENMP`, libgomp linked), N2/STO-3G:

```
OMP_NUM_THREADS=1    121.9 s
OMP_NUM_THREADS=4    123.6 s      <- flat
OMP_NUM_THREADS=8    100.0 % CPU  <- one core, seven idle
```

The code agrees: `apply_ci_hamiltonian` (`ci.cpp:437-622`) — the iterative
sigma build, which is what runs for any space above `dense_threshold = 500` —
contains **zero** `#pragma omp`. The one pragma in `ci.cpp` is on the **dense**
Hamiltonian build (`:221`), which only runs *below* 500 determinants. `rdm.cpp`
has six, so the RDM path is threaded; `fci.cpp` and `strings.cpp` have none.

**So every FCI case large enough to matter runs single-threaded.**

**And the hot path is allocation, not arithmetic.** Leaf-sample profile
(21 048 samples, N2/STO-3G, 1 thread):

| frame | share |
|---|---|
| `malloc`/`free` family combined | **~53 %** |
| `apply_ci_hamiltonian` | 12.0 % |
| `get_excitation` | 5.9 % |

The cause is visible in the source: `get_excitation` (`ci.cpp:65`) returns
`std::pair<std::vector<int>, std::vector<int>>` **by value**, and
`slater_condon_element` calls it for both spin channels on **every matrix
element**. That is up to four heap allocations per element — for vectors that
hold **at most two entries each**, since a Slater-Condon element is zero beyond a
double excitation. A `std::array<int,2>` plus a count removes the allocation
entirely.

**Why this belongs in an FCIQMC scope rather than a separate ticket:** FCIQMC's
spawning step calls `slater_condon_element` in its innermost loop, far more often
than FCI's sigma build does. Building FCIQMC on top of a function that heap-
allocates per call would inherit a ~2x penalty at the outset. **Fix this before
writing the spawn**, and the FCI reference gets faster at the same time — which
directly widens the ndet window where a deterministic reference is affordable.

Neither of these is scoped here. Both are recorded in
`vault/Status/Open Work.md` as standalone items, because they are worth doing
whether or not FCIQMC ever happens.

## What already exists, and what does not

Genuinely reusable:

| piece | where | note |
|---|---|---|
| bitstring determinants | `CIString`, `casscf_internal.h:21` | `uint64_t`, 31 spatial orbitals |
| occupation/parity helpers | `count_occupied_below`, `parity_between`, `strings.h:42-43` | exactly what excitation generation needs |
| **Slater-Condon matrix elements** | `slater_condon_element`, `ci.h:43` | the spawning step's `H_ij` |
| CI diagonal | `build_ci_diagonal`, `ci.h:101` | the death/cloning step's `H_ii` |
| active-space integrals | `h_eff` + `ga`, passed through the CI API | already transformed |
| a deterministic reference | `src/post_hf/fci.cpp` | the only way to validate at small size |

**Not present, and each is real work:**

- **No RNG, and no policy for one.** Needs a seeded, reproducible, per-thread
  generator. `std::mt19937_64` per walker-list shard with a seed derived from a
  single run seed is the conventional answer; the constraint is that a rerun with
  the same seed must reproduce the trajectory **bitwise**, which rules out any
  thread-count-dependent draw order.
- **No sparse walker container.** FCIQMC's state is a hash map from determinant to
  signed weight, with spawning, annihilation and compression each iteration. The
  existing `det_lookup` (`unordered_map<CIString,int>`) indexes a *fixed enumerated
  space*; FCIQMC needs a *dynamic* one.
- **No excitation generator.** Uniform or Cauchy-Schwarz-weighted random single/
  double excitation from a determinant, with the generation probability returned —
  the `p_gen` is as important as the excitation, and getting it inconsistent with
  the sampling is the classic silent-bias bug.
- **No population control.** Shift adjustment, initiator approximation (i-FCIQMC
  is effectively mandatory for anything but toy systems), walker-number targets.

## Why the shape of this codebase helps, and where it fights

**Helps:** the determinant layer is already factored out and shared by FCI and
CASSCF (`src/post_hf/ci/`), the integrals arrive pre-transformed, and
`std::expected` error propagation means a stochastic path can report
non-convergence honestly rather than returning a plausible number.

**Fights:** every parallel path in Planck is **bitwise thread-count-invariant** by
design and by explicit gate — the DFT J/K builds, the CC kernels, the ERI
transforms. FCIQMC's natural parallelization (distribute walkers, spawn across
ranks) is *not* order-invariant: the annihilation step's floating-point sum
depends on arrival order. Preserving the codebase's determinism discipline means
either a fixed-order reduction over ranks (costly, and it constrains the
communication pattern) or **explicitly documenting FCIQMC as the one path where
bitwise thread-invariance does not hold** — which is a real exception to a rule
this codebase has otherwise kept everywhere.

**Do not make that exception silently.** It is the kind of thing that becomes an
unpleasant surprise three investigations later.

## If it proceeds: suggested first cut

Deliberately minimal, aimed at answering Q2 rather than at being useful.

1. **Serial, single-thread, no initiator.** One walker list, `std::mt19937_64`,
   fixed seed. Target a system where FCI is computable (Be/STO-3G, LiH/STO-3G) so
   every number has an exact reference.
2. **Gate on fixed-seed reproducibility first**, before touching accuracy: same
   seed, same walker trajectory, bitwise. This is the gate that keeps the method
   maintainable, and it is independent of whether the physics is right.
3. **Then the statistical gate**, with a blocking analysis for σ. Assert the FCI
   energy lies within 3σ, and assert σ itself shrinks as √N_walkers — the second
   is what catches a systematically biased `p_gen`, which a mean-only check
   cannot see.
4. **Only then** consider the initiator approximation, parallelism, and the
   `n_act` 18-31 window that motivated the item.

**Steps 1-3 are the whole research question.** If they pass, FCIQMC is viable
here; if they cannot be made to pass, that is the finding, and it is worth
recording either way.

## What this must not do

- **Do not implement before answering Q1.** A method with no target calculation is
  a capability nobody asked for. Name the molecule and active space first.
- **Do not weaken an existing gate to accommodate stochastic output.** The
  statistical gate is a *new* kind of gate, not a relaxed tolerance on an old one.
  Loosening a 1e-9 assertion to fit a Monte Carlo result would damage a discipline
  that has caught real defects repeatedly in this codebase.
- **Do not reimplement the determinant layer.** `src/post_hf/ci/` is shared by FCI
  and CASSCF and is the single representation. If FCIQMC needs something from it,
  extend it there.
- **Do not skip `p_gen` validation.** An excitation generator whose returned
  probability disagrees with its actual sampling distribution produces a
  *plausible, converged, wrong* energy — the exact failure class this codebase has
  hit before (the spin-adapt default, the ERI symmetry table). Test the generator
  against a brute-force enumeration on a tiny space before trusting any energy.
- **Do not claim a speedup against FCI.** They compute different things: FCI gives
  the exact energy, FCIQMC gives an estimate with an error bar. The comparison is
  *reachable system size*, not wall-clock.

## Prior art worth reading before starting

- Booth, Thom, Alavi, *JCP* **131**, 054106 (2009) — the original method.
- Cleland, Booth, Alavi, *JCP* **132**, 041103 (2010) — the initiator
  approximation, which is what makes it practical.
- NECI (the Alavi group code) — the reference implementation; useful mostly for
  what the excitation generators and population control actually look like.

## Key code locations

| what | where |
|---|---|
| determinant representation and its 31-orbital ceiling | `CIString`, `kMaxPackedSpatialOrbitals`, `src/post_hf/casscf_internal.h:21-27` |
| occupation / parity helpers | `src/post_hf/ci/strings.h:42-43` |
| Slater-Condon element (the spawn) | `slater_condon_element`, `src/post_hf/ci/ci.h:43` |
| CI diagonal (the death step) | `build_ci_diagonal`, `src/post_hf/ci/ci.h:101` |
| the deterministic reference to validate against | `src/post_hf/fci.cpp` |
| the determinism discipline this would except | `dft_xc_reduction_determinism` note; the CC OpenMP gates |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
