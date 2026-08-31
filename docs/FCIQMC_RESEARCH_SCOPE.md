# Research scope: FCIQMC in Planck

**Research scope. Not started, and not yet justified — the first step is deciding
whether to build it at all.** Opened 2026-08-30; **fixture costs and the
prerequisite rescoped 2026-08-30** after the FCI sigma build got ~17x faster.

**What the rescope changed, in one line:** the deterministic reference is now
roughly an order of magnitude cheaper, so the small-system window where FCIQMC can
be validated at all is wider (C2/STO-3G went from "abandoned at >10 min" to
**47.7 s**), and the allocation prerequisite this scope named is **already
satisfied**. Neither Q1 nor Q2 below is affected — the case for building FCIQMC at
all is exactly as unproven as it was.

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
  `OMP_NUM_THREADS`, the transpose merge and the operator-hoist likewise. **The FCI
  threading that prompted this rescope is one more instance**: it was held to
  byte-identical output at 1/2/4/8 threads, and two determinism defects were
  rejected on exactly that basis. This discipline is not softening.
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

**Remeasured 2026-08-30 after the FCI sigma build got ~17x faster**
(`docs/FCI_SIGMA_BUILD_PERFORMANCE.md` — allocator removal 4.8x, then threading
3.54x at 4 threads). The original serial column is kept because it is what the
recommendation below was originally argued from, and the *ratios* between systems
are unchanged:

| system | norb | na/nb | ndet | FCI, was (serial) | FCI, now (4 threads) |
|---|---|---|---|---|---|
| H2/STO-3G | 2 | 1/1 | 4 | instant | instant |
| water/STO-3G | 7 | 5/5 | 441 | instant | instant |
| H2/cc-pVDZ | 10 | 1/1 | 100 | instant | instant |
| **Be/6-31g\*** | 14 | 2/2 | **8 281** | 46.5 s | **2.72 s** |
| **N2/STO-3G** | 10 | 7/7 | **14 400** | 124.4 s | **8.28 s** |
| **C2/STO-3G** | 10 | 6/6 | **44 100** | > 10 min (abandoned) | **47.7 s** |
| water/6-31G | 13 | 5/5 | 1 656 369 | hours | ~6 h (extrapolated) |

**C2/STO-3G is the headline change.** It was abandoned un-run at over ten minutes
and is now a **47.7 s** reference, which roughly **triples the usable ndet** and
makes it a viable fixture rather than an aspiration.

**The recommendation is unchanged: `N2/STO-3G` (ndet = 14 400), now ~8 s.**

It is the smallest system that satisfies both constraints:

- **FCI is cheap enough to be a routine reference** — now ~8 s at 4 threads (was
  2 minutes), so the gate can recompute it rather than hard-coding a number. It is
  now cheap enough to sit in the regression suite outright, which the 2-minute
  version was not.
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

**`C2/STO-3G` is now a viable *third* fixture** (44 100 determinants, 47.7 s),
where before it was abandoned un-run. It is the most valuable of the three for a
`p_gen` bug hunt: at 6α/6β it sits *between* Be (2/2) and N2 (7/7) at the same
orbital count as N2, so the three together vary electron count against a fixed
orbital count — three points to triangulate with, rather than a pair where ndet
and electron count move at once. It also has 3x N2's determinant space, so a
walker population that is a genuine sample on N2 is a sparser one there.

**H2 and water/STO-3G are unsuitable as FCIQMC fixtures** despite being the
existing FCI regression cases: at 4 and 441 determinants the walker population
would exceed the space.

### The FCI reference cost grows fast — do not plan on a large one

**Refit 2026-08-30 on three points** (Be 8 281 / 2.72 s, N2 14 400 / 8.28 s,
C2 44 100 / 47.69 s, all at 4 threads): **ndet^1.69**, with pairwise slopes of
2.01, 1.71 and 1.56. The old two-point figure was ndet^1.78, so the exponent is
about where it was — **the speedup moved the constant, not the scaling**, which is
what one expects from removing a per-element allocation and adding threads.

The fit is still thin, but one of the two original caveats is now partly
addressed: C2 (6α/6β) and N2 (7α/7β) have similar electron counts at the same
orbital count, so the Be↔N2 conflation of ndet with electron count no longer drives
the fit alone. The declining pairwise slope (2.01 → 1.56) is the usual sign of
fixed overhead being amortized, so **1.69 is more likely an over- than
under-estimate at large ndet**.

**Consequence for the scope — this is the part that actually moved.** The
deterministic reference used to be practical only below ndet ≈ 10⁵ *in principle*,
with C2 at 44 100 already unaffordable *in practice*. Now:

| ndet | FCI reference cost |
|---|---|
| 44 100 (C2/STO-3G) | **47.7 s — measured** |
| ~10⁵ | **~3 min — extrapolated** |
| 1.66 M (water/6-31G) | ~6 h — extrapolated |

So the window where a gate can *recompute* the exact answer rather than hard-code
it now comfortably includes ndet ~10⁵. **That does not change the strategic
conclusion**: FCIQMC exists for spaces of 10⁹ and beyond, which no deterministic
reference will ever reach, so the **fixed-seed reproducibility gate remains the
primary one** and the statistical gate remains restricted to small systems. What
changed is that the small-system regime is now roughly an order of magnitude
wider, and a *third* fixture (C2) is affordable — useful because it sits between
Be and N2 in electron count, giving three points rather than a pair to triangulate
an excitation-generator bug.

`ci_max_dim` defaults to 10 000 and must be raised explicitly for anything larger —
it fails loudly, which is correct.

Candidate inputs are committed under `tests/inputs/exploratory/fciqmc/`.

### The FCI reference used to be single-threaded and allocation-bound — both fixed

**This section previously read "The FCI reference is single-threaded, and half its
time is `malloc`". Both halves are now false**, and the fix landed 2026-08-30
(`docs/FCI_SIGMA_BUILD_PERFORMANCE.md`). It is kept because the *reason* it
mattered to FCIQMC still applies to the code FCIQMC would be built on.

What was measured, and what it is now (N2/STO-3G):

| | was | now |
|---|---|---|
| `malloc`/`free` share of profile | **~53 %** | **0.1 %** |
| threading | none — flat 121.9 s → 123.6 s at 1→4 threads | **3.54x at 4 threads** |
| wall | 124.4 s | **8.28 s** |

`get_excitation` returned `std::pair<std::vector<int>, std::vector<int>>` **by
value** for values holding at most two entries each; it now returns a
fixed-capacity struct. `apply_ci_hamiltonian` is now parallel over the ket loop,
with per-bin partial vectors summed in fixed order.

**Why this still belongs in an FCIQMC scope.** FCIQMC's spawning step calls
`slater_condon_element` in its innermost loop — far more often than FCI's sigma
build does. That function is **shared**, so the allocation fix is inherited by any
FCIQMC built on it: the ~2x penalty this scope warned about being baked in at the
outset is no longer there to inherit. **The prerequisite this section used to
name is satisfied.**

**Two constraints from that work bind any FCIQMC implementation here**, and they
are the concrete form of the Q2 tension below:

1. **A fixed-order reduction is necessary but NOT sufficient** for bitwise
   thread-count invariance. Two defects were found there, each only by byte-diffing
   across thread counts: `schedule(dynamic)` gives an accumulator a different
   *subset* of terms per run, and keying accumulators by `omp_get_thread_num()`
   makes their contents depend on the thread *count*. **What must be deterministic
   is the partition of work into accumulators, not merely the order they are
   summed.** An FCIQMC annihilation step accumulating per-thread will hit exactly
   this, and the working pattern is `partials[j / bin_size]` with a fixed bin count.
2. **A cheap-looking guard on an outer loop can carry asymptotic weight.** The
   recommended gather reformulation of the sigma build was built and was 2.2-2.4x
   *slower*, because moving a `|c| < 1e-15` test inward destroyed a sparsity
   exploitation. FCIQMC is built entirely on sparsity — walker lists are sparse by
   construction — so before restructuring any loop that skips negligible weight,
   measure what fraction of iterations the skip actually eliminates.

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

**The F3 threading work sharpens this rather than softening it.** It threaded
`apply_ci_hamiltonian` — the closest analogue in the tree to what FCIQMC's spawn
would do, a scatter into a shared vector — and **kept** bitwise thread-count
invariance, at the cost of `kBins × dim × 8` bytes of fixed-partition
accumulators. So the "costly fixed-order reduction" option above is not
hypothetical: it has a worked precedent in this exact file, with a measured price
(no serial cost, 4.8 % idle at 4 threads). That makes the exception **harder** to
justify, not easier — the burden is now to show why FCIQMC cannot do what the
sigma build did.

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
  *reachable system size*, not wall-clock. **Nor should the ~17x FCI speedup be
  read as weakening the case for FCIQMC**: it moved the reference constant, not the
  `ndet^1.7` scaling, and FCIQMC's argument was never about the constant. A 17x
  faster exponential is still exponential — it buys roughly one more `n_act` step,
  against the 18-31 window Q1 describes. What it genuinely changed is the
  *validation* budget, which is the only reason this rescope touched anything.

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
