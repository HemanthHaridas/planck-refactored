# Scope: threading FCIQMC

**Scope for in-flight work. T1 LANDED (`9ca61897`); T2 is next.** FCIQMC still has
zero `#pragma omp`. The determinism policy is already decided
(`FCIQMC_RESEARCH_SCOPE.md` §6): **no exception, bitwise thread-count invariance is
kept**, and the gate for it exists.

## The measurement that orders the work

Profiled on the N2/STO-3G gate case (`sample`, 15 s window, idle threads excluded,
12 689 work samples):

| frame | share |
|---|---|
| `slater_condon_element` | **40.1 %** |
| **`malloc` / `free` family** | **29.5 %** |
| `occupied` / `virtuals` | 10.1 % |
| `draw_excitation` | 2.7 % |
| `ordered_l1_norm` (sort) | 2.5 % |

**Nearly a third of the runtime is allocation**, and that changes the order of the
work. This is the same situation the FCI sigma build was in — 53 % malloc there —
where the recorded lesson was explicit: *threading a function that spends half its
time in the allocator parallelizes the allocator, and `malloc` contention across
threads is a well-known route to negative scaling.*

**Fix the allocation first.** It is smaller, lower-risk, independently measurable,
and it changes what the threading is worth.

## The allocation sites

All in the per-determinant spawn path, all returning small heap vectors by value:

| site | called | holds |
|---|---|---|
| `occupied(det, n_act)` | 2x per parent, again inside `draw_excitation` | ≤ n_act ints |
| `virtuals(det, n_act)` | 2x per parent, again inside `draw_excitation` | ≤ n_act ints |
| `live` class vector | 1x per `draw_excitation` | ≤ 5 pairs |

At ~14 000 occupied determinants × 30 000 iterations that is on the order of 10⁹
vector allocations for values that never exceed 31 entries — `n_act` is bounded by
`kMaxPackedSpatialOrbitals`.

## Steps

### T1 — remove the per-call allocations (no threading) — **DONE, 1.76x**

**Measured on N2/STO-3G at 1 thread: 71.63 s -> 40.81 s (1.76x), malloc share
29.5 % -> 1.4 %.** The allocator is effectively eliminated.
Output **bitwise identical** — every printed line, not only the energies — on both
`n2_fciqmc_sto3g` and `h2_fciqmc_sto3g` against a pre-T1 binary from the same tree.

**The "lower bound, not an estimate" instruction below held.** Amdahl on a 29.5 %
share caps the direct saving at 1.42x; measured 1.76x, for the same reason the FCI
sigma build over-delivered (4.8x against a profile implying ~2.1x): per-call churn
also costs cache pressure attributed to other frames.

**Build-hygiene trap, cost a wasted 25-minute build.** T1 was stashed to keep an
unverified change out of a commit *while its build was still running*; `make` then
compiled a file no longer in the tree and reported `MAKE_EXIT=0`. **A build in
flight is not pinned to the working tree** — kill it before stashing.

The original plan follows, unchanged.


Return fixed-capacity types instead of `std::vector`: `std::array<int, 32>` plus a
count for `occupied`/`virtuals`, a fixed array for the live-class list. The bound
is not a guess — `n_act <= kMaxPackedSpatialOrbitals` (31) is enforced by
`build_all_mo_ci_setup`, so assert it rather than truncating.

`enumerate_connections` still returns a `std::vector<Excitation>` and should keep
doing so: its size is genuinely variable, and it is **not** in the spawn path —
verified, `propagate_stochastic` does not call it. It appears in
`propagate_deterministic` (unused in production runs), `draw_uniform_excitation`
(the F2.2 reference generator, also unused in production), and `projected_energy`,
which runs once per *sampling step* rather than once per walker. That is ~30 000
calls against the spawn path's ~10⁹, so it is not where the allocator time is.

- **Verify:** energies **bitwise identical** to the current binary on
  `h2_fciqmc_sto3g` and `n2_fciqmc_sto3g` — this is a representation change with no
  arithmetic in it, so anything else is a defect.
- **Verify:** the malloc share falls, and record the wall-clock delta. The sigma
  build's 4.8x from the same fix *exceeded* what its profile implied, because
  per-element churn also costs cache pressure attributed to other frames — so
  **treat 29.5 % as a lower bound on the gain, not an estimate.**

### T2 — thread the spawn loop — **NEXT, ceiling 3.75x at 4 threads**

**Re-profiled after T1 (N2/STO-3G, 1 thread), which is what sizes this step.**
Self time, 12 572 samples:

| frame | self |
|---|---|
| `slater_condon_element` | **53.1 %** |
| `draw_excitation` | 11.4 % |
| walker `unordered_map` | 4.1 % |
| sort (`ordered_l1_norm`) | 3.8 % |
| malloc family | 1.4 % |

**`slater_condon_element` ROSE from 40.1 % to 53.1 % as a share, and that is the
expected result, not a surprise:** T1 removed ~30 % allocation, so the same
absolute Hamiltonian work is now a larger fraction of a smaller total
(`40.1 / (1 - 0.295) = 56.9 %` predicted). **It is the target for any future
kernel work and is untouched by threading** — T2 parallelizes it rather than
reducing it.

**A profile-parsing trap worth carrying.** `sample` output is a CALL TREE whose
lines carry `+ ! : |` prefixes before the count, and each count is INCLUSIVE. A
naive `^\s*(\d+)` regex matches almost nothing, and summing the counts it does
match double-counts parents. The first pass of this re-profile did exactly that
and reported `slater_condon_element` at **0.2 %** — an obviously impossible number
that was quoted before being sanity-checked. Strip the prefix, and compute self
time as a node's count minus its direct children's.

`propagate_stochastic` is **97.7 %** of `run_fciqmc` inclusive (8805 / 9008
samples). Amdahl on that share:

| threads | ceiling |
|---|---|
| 2 | 1.96x |
| 4 | **3.75x** |
| 8 | 6.91x |

The residual 2.3 % is the per-iteration serial tail — `ordered_l1_norm`'s sort,
`ctl.update`, and `projected_energy` (which the scope explicitly leaves serial).
**Do not thread those** to chase the last few percent; the determinism risk is not
worth it and T3 covers the merge if it shows up.

The design is already settled and verified on a model (§6 of the research scope):
**partition the parents** by `hash(parent) % kBins`, each thread accumulating into
its own bin, merged in fixed bin order at the end of the iteration.

#### The loop as it actually stands (read this before writing the pragma)

`propagate_stochastic` (`src/post_hf/ci/fciqmc.cpp`) iterates
`for (const auto &[det, weight] : population)` and does three things per parent:
a deterministic **death** write `next.add(det, ...)`, then `n_spawn_attempts`
**spawn** draws each writing `next.add(exc.det, ...)`, with the **initiator** rule
reading `population.weight_at(exc.det)`.

Three properties decide the shape of the change, and each was checked in the
source rather than assumed:

1. **`next` is the only mutable shared state.** Every write goes through
   `WalkerPopulation::add`, which is `_walkers[det] += w` on an
   `unordered_map` — not thread-safe, and the accumulation order is exactly what
   the bins exist to fix.
2. **The initiator rule reads `population`, not `next`** (there is already a
   comment saying why). `population` is `const&` throughout, so the rule is a
   pure read and needs nothing.
3. **`RandomSource &rng` is a single mutable object shared by every parent.**
   This is the sharpest hazard in the step: threading the loop as written is not
   merely non-deterministic, it is a data race on the generator state. Each bin
   must take its own stream from `RandomSource::derive(bin_index)` — which is
   already deterministic in the run seed and independent of how many shards are
   derived, so **the mechanism exists and must simply be used**. Deriving per
   *thread* instead of per *bin* would reintroduce thread-count dependence, the
   exact defect the sigma build hit twice.

- **`kBins` is a fixed constant, never tied to thread count.** The sigma build paid
  for this lesson twice: `schedule(dynamic)` gives an accumulator a different
  *subset* per run, and keying by `omp_get_thread_num()` makes contents depend on
  the thread *count*.
- **Do not bin by the child determinant.** It fixes which accumulator receives a
  spawn but not the order arrivals reach it, so two threads spawning onto the same
  determinant still race. The partition must be over the **work**.
- **The RNG needs no new mechanism.** `RandomSource::derive(index)` is already
  deterministic in the run seed and independent of how many shards are derived;
  derive one per bin.

- **Verify, in this order:**
  1. **CPU > 100 %** at 4 threads — one `ps` call, catches an inert pragma before
     any timing is read.
  2. **Bitwise identical** at `OMP_NUM_THREADS` = 1/2/4/8, against the T1 serial
     result. `h2_fciqmc_threads1/4` already asserts this at `atol = 0.0`; extend it
     to 8, and add an N2 pair since a 4-determinant space may not exercise the
     partition at all.
  3. All FCIQMC and FCI regression cases green.
  4. Speed. The post-T1 ceiling is **3.75x at 4 threads**; below ~2x, check bin
     count and imbalance before concluding the lever is absent.

- **One structural difference from the sigma build, in T2's favour.** That build
  binned by *index range* (`partials[j / bin_size]`), so a determinant could move
  between bins when the bin count changed. Here the key is the determinant itself
  (`hash(parent) % kBins`), so a parent maps to the same bin **regardless of the
  bin count** and its contributions accumulate in the same order. Invariance is
  therefore robust even to changing `kBins`, which the sigma build's scheme was
  not — recorded in `FCIQMC_RESEARCH_SCOPE.md` §6 and worth not rediscovering.

- **Sizing `kBins`.** Memory is `kBins` partial maps, not `nthreads` — independent
  of thread count by construction. Start at the sigma build's 64 and only revisit
  if step 4 shows imbalance; **more bins under static scheduling** is the move,
  never `schedule(dynamic)`, which breaks invariance.

### T3 — the merge, if it shows up

The fixed-order merge is serial. At `kBins = 64` and a few thousand occupied
determinants it should be negligible, but the sigma build's equivalent was
measured rather than assumed (0.1 % there).

- **Verify:** profile after T2 and only act if the merge is material. **Do not
  pre-optimize it.**

## What this must not do

- **Do not thread before T1.** The recorded reason from the sigma build is that
  threading an allocation-bound loop parallelizes `malloc` contention.
- **Do not use `omp atomic` on the walker map, or a completion-order reduction.**
  That is the DFT-grid jitter defect, and it would silently break the invariance
  the gate asserts.
- **Do not tie `kBins` to thread count**, and do not use `schedule(dynamic)` over
  the bins in a way that changes which bin gets which parent.
- **Do not accept "energies agree to 1e-10" for the threading.** Every other
  parallel path here is bitwise invariant and the gate is `atol = 0.0`. A tolerance
  would hide exactly the reduction-order defect the design exists to prevent.
- **Do not thread the estimators.** `projected_energy` runs once per sampling step
  over one determinant's connections; it is 2.5 % and not worth the determinism
  risk.

## Honest note on value

**Nothing currently needs this.** The N2 gate runs in 69 s serial, and the method
is validated but unused (`FCIQMC_RESEARCH_SCOPE.md` — Q1 remains unanswered in
practice). T1 is worth doing on its own merits as a serial speedup; **T2 is worth
doing when a target exists**, which today it does not.

## Key code locations

| what | where |
|---|---|
| the spawn loop to thread | `propagate_stochastic`, `src/post_hf/ci/fciqmc.cpp` |
| the allocation sites | `occupied` / `virtuals`, same file |
| the walker map | `WalkerPopulation`, `src/post_hf/ci/fciqmc.h` |
| RNG shard derivation | `RandomSource::derive` |
| the invariance gate | `h2_fciqmc_threads1` / `h2_fciqmc_threads4` |
| the decided policy | `FCIQMC_RESEARCH_SCOPE.md` §6 |
| the precedent, with its traps | `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
