# Scope: threading FCIQMC

**Scope for in-flight work. Not started.** FCIQMC is entirely serial today — zero
`#pragma omp` in `fciqmc.{h,cpp}` or `fciqmc_driver.cpp`. The determinism policy is
already decided (`FCIQMC_RESEARCH_SCOPE.md` §6): **no exception, bitwise
thread-count invariance is kept**, and the gate for it exists.

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

### T1 — remove the per-call allocations (no threading)

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

### T2 — thread the spawn loop

The design is already settled and verified on a model (§6 of the research scope):
**partition the parents** by `hash(parent) % kBins`, each thread accumulating into
its own bin, merged in fixed bin order at the end of the iteration.

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
  4. Speed. Amdahl on the post-T1 profile sets the ceiling; below ~2x at 4 threads,
     check bin count and imbalance before concluding the lever is absent.

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
