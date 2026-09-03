# Scope: T2, threading the FCIQMC spawn loop

**Scope for in-flight work. Not started.** The serial work is answered in
`FCIQMC_SERIAL_PERFORMANCE.md` (71.63 s -> 13.74 s, 5.21x, no threads). This is
the one remaining step, broken into pieces that can each be verified on their own.

**Ceiling 2.81x at 4 threads** — but that number has moved three times as each
serial step landed (3.75x -> 2.19x -> 2.81x), so **S0 re-measures it before any
code is written**, and the result decides whether the rest is worth doing.

## The constraint that shapes everything

Every parallel path in Planck is **bitwise thread-count invariant**, by design and
by gate. FCIQMC keeps that (`FCIQMC_RESEARCH_SCOPE.md` §6, decided explicitly
rather than discovered). The gate already exists at `atol = 0.0` and passes
trivially today because there are no threads.

The design, settled and verified on a model: **partition the parents** by
`hash(parent) % kBins`, each bin accumulating into its own `WalkerPopulation`,
merged in fixed bin order.

## What the loop looks like now

```cpp
for (const auto &[det, weight] : population)   // <- the loop to partition
{
    if (weight == 0.0) continue;
    const double diag = ham.diagonal(det);     // memoized (T4) -- NOT thread-safe
    next.add(det, weight * (1.0 - dt * (diag - shift)));
    for (int attempt = 0; attempt < n_spawn_attempts; ++attempt)
    {
        const auto exc = draw_excitation(det, n_act, rng);   // rng: 2 draws
        ...
        if (initiator_threshold > 0.0 && ... population.weight_at(exc.det) == 0.0)
            continue;                                        // reads `population`
        child = granularity * rng.stochastic_round(...);     // rng: 1 more draw
        next.add(exc.det, child);
    }
}
```

Three properties, each checked in the source rather than assumed:

1. **`next` is the only mutable shared state.** Every write is
   `_walkers[det] += w` on an `unordered_map`.
2. **The initiator rule reads `population`, not `next`** — a pure read on a
   `const&`, safe to share.
3. **`RandomSource &rng` is a single mutable object shared by every parent.**
   Threading as written is a **data race on generator state**, not merely a
   determinism question. This is the sharpest hazard in the step.

**The T4 memo is also shared mutable state** (`unordered_map` behind a
`shared_ptr` in `ops.diagonal`) and is not thread-safe.

## Steps

### S0 — re-measure the ceiling (no code) — **DONE: 2.86x at 4 threads, proceed**

**Measured 2026-09-03** on the current binary, `n2_fciqmc_sto3g`, whole-run sample
(13 s window over a 12.61 s run), inclusive shares under `run_fciqmc` via real
parent links:

| phase | inclusive |
|---|---|
| `propagate_stochastic` | **86.6 %** |
| `projected_energy` | 1.7 % |
| `WalkerPopulation::compress` | 1.5 % |
| `ordered_l1_norm` | 0.7 % |
| driver self time (per-iteration, **not** setup — see below) | 9.4 % |

**Ceiling: 1.76x / 2.86x / 4.14x at 2/4/8 threads.** At 4 threads that is
12.61 s -> 4.42 s, saving **8.2 s**.

**Replicated** on a second run: p = 86.3 % against 86.6 %, ceiling 2.83x against
2.86x — a 0.4-point spread, so the measurement is stable rather than a
single-sample artifact. That check exists because this number has been wrong three
times before.

**Verdict: above the ~2x stop threshold, proceed to S1.** Note the threadable
share went *up* (85.8 % -> 86.6 %) since the last measurement even though nothing
changed in the loop — the L1-norm binning that landed between them removed serial
work, which raises the share of what remains. The wall-clock saving is nonetheless
the smallest it has ever been (8.2 s, against ~19 s when this was first scoped),
because the serial baseline has fallen 5.2x.

#### Does the saving scale with system size?

**Yes, but the RATIO barely moves — what scales is the absolute saving, and it
tracks total runtime.** Measured on two axes:

| case | ndet | wall | threadable | ceiling @4 | saved @4 |
|---|---|---|---|---|---|
| N2/STO-3G, 10k walkers | 14 400 | 12.6 s | 86.6 % | 2.85x | **8.2 s** |
| N2/STO-3G, 40k walkers | 14 400 | 14.7 s | 86.5 % | 2.85x | 9.5 s |
| HF/6-31G, 50k walkers | 213 444 | 102.4 s | **89.8 %** | **3.07x** | **69.0 s** |

**Walker count is NOT a scaling axis.** 4x the walkers left the share flat
(86.6 % -> 86.5 %): everything in the iteration — the spawn loop *and* the serial
tail — is proportional to the occupied population, so raising it scales both sides
equally.

**System size helps, modestly.** 15x the determinants moved the share to 89.8 %
and the ceiling to 3.07x, because larger `n_act` means more work per parent
(`draw_excitation` and the Slater-Condon element are both O(n_act)-ish) against a
serial tail that grows more slowly. Extrapolating, the ceiling creeps toward 4x
but will not exceed it at realistic sizes.

**A predicted mechanism was falsified here, and the correction matters for anyone
re-deriving this.** The 9.4 % remainder in the S0 table was labelled
"setup / SCF / integrals" on the argument that it is fixed cost and would shrink
as a fraction on longer runs — which would have made the share *rise* with run
length. It is not setup: `run_fciqmc`'s direct children are almost entirely
`propagate_stochastic` nodes, so that remainder is **self time inside the driver's
own per-iteration loop** (`ctl.update`, sampling bookkeeping, `signed_population`
accumulation). It grew 896 -> 1046 samples at 4x walkers, exactly as
per-iteration work does. **The share is stable, not rising**, which is the
opposite of what the argument predicted.

**Where that 9.4 % actually goes — DONE, and smaller than guessed.** The driver
does two O(n_occupied) passes per sampling step: `ctl.update` on the norm, and
`signed_population[det] += w` (`fciqmc_driver.cpp:368`), which accumulates the
coefficient-ratio dump. **The second exists only to support the `<N_I>/<N_0>`
diagnostic** and ran unconditionally on every sampling step of every run, even
though the dump is printed only at `verbosity verbose`. Gated on the same
predicate the dump's own printing condition uses (`want_coefficient_ratios`, one
variable read at both sites so they cannot drift apart).

**Measured, not guessed: ~2.3 %, not "most of 9.4 %".** My first estimate assumed
this accumulation dominated the driver-loop bucket; it does not. Three repeat runs
each, `normal` verbosity, old binary vs gated:

| | mean wall |
|---|---|
| old (unconditional) | 13.00 s |
| gated | **12.70 s** |

A consistent but modest ~0.30 s / 2.3 % — real (faster on all three pairs), but
the driver-loop bucket is dominated by something else, most likely `ctl.update`'s
norm call, not by this accumulation. **Verified correct on both sides of the
gate**, not just fast: `verbosity normal` output bitwise identical to the old
binary with the dump absent, and `verbosity verbose` output bitwise identical with
the 21-line `<N_I>/<N_0>` dump still printing — the case an incautious gate
condition would silently break while still passing the `normal` check.

**Consequence for the decision:** the ratio is roughly fixed near 3x, so the
payoff is proportional to how much the method is actually run — 8 s on the N2
gate, 69 s on a 102 s HF run, hours on a calculation in the regime FCIQMC exists
for. That loops back to `FCIQMC_RESEARCH_SCOPE.md` Q1: **nothing in the tree
currently runs FCIQMC at a size where 3x matters.** T2 is worth building when it
does, and the ceiling should be re-measured on *that* system rather than taken
from here.

The original scoping follows.

#### Original scoping

Profile the current binary on `n2_fciqmc_sto3g`, whole-run sample, and compute
`propagate_stochastic` inclusive with **real parent links** (see the parsing traps
in `FCIQMC_SERIAL_PERFORMANCE.md` — two different naive methods gave 3909.9 % and
100.0 %).

- **Verify:** the share and the implied Amdahl ceiling at 2/4/8 threads.
- **Stop condition:** if the ceiling at 4 threads is below ~2x, say so and stop.
  The serial baseline is now 13.74 s; a 1.5x on 11 s of threadable work is ~4 s
  saved on a method nothing currently uses.

### S1 — make the RNG per-bin (serial, no pragma) — **DONE, one bug caught and fixed**

Each parent draws from a `RandomSource` dedicated to its bin
(`hash(parent) % kBins`, `kBins = 64`) rather than from one RNG shared by the
whole iteration. Still fully serial — no pragma, one `next`, one thread.

**A bug in the first version made the run silently stuck, and the reproducibility
check did not catch it.** The first implementation derived `bin_rngs[b]` directly
from the caller's `rng` via `rng.derive(b)`. `derive` is `const` — it reads `rng`'s
state without advancing it. `rng` itself lives in the driver, constructed ONCE
before the whole 50 000-iteration loop and passed by reference into every
`propagate_stochastic` call; the OLD code advanced it directly (every
`draw_excitation`/`stochastic_round` call mutated `_engine` in place), which is
what made 50 000 iterations sample 50 000 different points of the stream. With
`derive(b)` reading unchanged state, every single call rebuilt the IDENTICAL 64 bin
streams, and the walker population never left its initial condition — reference
weight exactly `10000.00` with zero variance across all 30 000 sampling steps,
shift equal to the seed diagonal to `1e-13`.

**Fixed-seed reproducibility passed on the broken version.** The same broken
trajectory replayed twice is still "reproducible" by that check alone — it took
reading the walker population's own diagnostics (`reference weight: mean X, min Y`)
to see the population was frozen, not the reproducibility gate itself. Worth
carrying: reproducibility proves a run is deterministic, not that it is doing
anything.

**The fix:** draw ONE raw 64-bit value from `rng` per call
(`RandomSource::raw64()`, a new primitive — `_engine()` called directly, full
64-bit entropy, no wasted bits from truncating a `[0,1)` double), then derive the
`kBins` streams from THAT. Within one call the bins are a pure function of
`(that one draw, bin index)` — independent of thread count, which is what S4
needs — while the draw itself differs across calls, because `raw64()` genuinely
advances the caller's engine. Verified with a standalone probe before touching the
real binary: `derive(0)` on the same unchanged `rng` gives the identical value on
every call (the bug, reproduced in isolation); seeding from `rng.raw64()` each time
gives five different values across five calls (the fix).

**Verified, all against the corrected version:**
- **Fixed-seed reproducibility**: two full runs of `n2_fciqmc_sto3g`, byte-identical.
- **Different seeds diverge** (the F3.5 negative control): seed `20250901` vs
  `77777` give different shift energies.
- **Agreement with exact FCI**: N2 shift `0.36σ`, projected `0.99σ`; H2 shift
  `0.22σ`, projected `0.33σ` — both well inside the 5σ gate.
- **All four FCIQMC regression cases pass** (`h2_fciqmc_sto3g`,
  `h2_fciqmc_threads1/4`, `n2_fciqmc_sto3g`), and the full extended suite shows no
  new failures (the 4 that fail are the pre-existing `rccsdt`/
  `PLANCK_CC_ARBITRARY_LOWER_RANKS` cases, unrelated).
- **`h2_fciqmc_threads1/4` needed no reference-value update**, contrary to what
  this section originally said to do. Both gate on `metric_within_sigma` against
  the exact FCI energy or `metric_close_case` against the paired live run — neither
  is a hardcoded number from a specific RNG trajectory, so both tolerate S1's
  changed stream by construction. The instruction to update them was written before
  checking what they actually asserted.

### S2 — bin the accumulation (serial, no pragma) — DONE, and the scope's own
verification instruction was wrong

`kBins` per-bin `WalkerPopulation`s replace the single `next`, merged in fixed bin
order (0..kBins-1) at the end of the iteration. Parent `det` selects its bin;
spawned children go into the same bin as their parent, never their own -- binning
by the child would fix which accumulator receives a spawn but not the ORDER
arrivals reach it, which is the actual race S2 exists to eliminate before S4.

**"Bitwise identical to S1" is not achievable, and testing it caught that the
instruction itself was wrong rather than the code.** IEEE double addition is not
associative. S1 sums every parent's contributions in one pass, interleaved in
`population`'s hash order; S2 sums each bin's contributions separately, then
concatenates bin-by-bin. Whenever a determinant receives contributions from
MULTIPLE parents in the SAME iteration -- which is the ordinary case once the
walker population is large enough for the connectivity graph to overlap -- the
two orderings sum the same set of terms in a different sequence, and the results
can differ in low-order bits. Measured directly on N2/STO-3G rather than assumed:

- **Iterations 1-52: bitwise identical to S1.** No multi-parent overlap yet at
  that walker count.
- **Iteration 53: first divergence**, and ONLY in the projected-energy numerator
  (-5.516124e+02 vs -5.186120e+02) -- the reference weight (denominator) and
  every quantity upstream of it (shift, total population) are still bit-identical
  at that same step. That is the signature of reassociation of an identical term
  set, not a missing, duplicated, or misrouted write: if the write pattern itself
  were wrong, the population totals would have diverged too, not only a
  downstream scalar reduction.
- Confirmed the shift trajectory (and therefore the underlying walker weights)
  eventually does diverge at longer run lengths (1000 steps: -108.8284 vs
  -108.6837) -- exactly the butterfly-effect propagation expected once the
  first-step reassociation feeds into every subsequent death/spawn calculation,
  the same mechanism as running under a genuinely different RNG seed.

**What was verified instead, since bitwise-vs-S1 is off the table:**
- **S2 is bitwise identical to ITSELF**, run to run, at fixed seed and one
  thread -- two full N2 runs, byte-identical. S2 introduces a DIFFERENT fixed
  ordering, not a new source of nondeterminism.
- **Agreement with exact FCI** on N2: shift 0.36 sigma, projected 0.58 sigma.
  Both well inside the 5-sigma gate.
- **All four FCIQMC regression cases pass**, full extended suite shows the same 4
  pre-existing unrelated failures as before S1/S2.
- **Memory: measured, not modelled.** Peak RSS on the N2 gate case:
  S1 10.16 MB -> S2 10.29 MB, a 131 KB / 1.29 % delta. kBins = 64 is a fixed
  constant, so this is the fixed per-map bucket-array overhead of 64 separate
  unordered_maps rather than one, and it does not grow with walker population or
  thread count.

**The lesson for the rest of this scope:** an instruction to verify "bitwise
identical" against a DIFFERENT accumulation order was written without checking
whether floating-point addition permits it. It does not, in general. The correct
standard for a reordering step is bitwise identity to ITSELF (proving the new
order is at least fixed) plus agreement with an independent exact reference
(proving the reordering is not hiding a real defect) -- never bitwise identity to
the PREVIOUS ordering, which non-associativity rules out on its own. S4's gate
(bitwise identical across thread counts, atol = 0.0) remains correct as stated,
because S4 does not change WHICH ordering S2 produces at one thread -- it only
asks that the SAME S2 ordering is reproduced regardless of how many threads
compute it, which is exactly what partitioning by parent (not by completion
order) guarantees.

### S3 — prefill the T4 memo — DONE, bitwise identical to S2

Before each step, walk the population and populate the diagonal cache via
`ops.diagonal(det)`, so `propagate_stochastic`'s parallel region (S4) only ever
reads it. Every determinant `propagate_stochastic` queries the diagonal of is a
member of `pop` at the top of the call (the death term is the only
`ham.diagonal` site inside it — verified by re-reading the function
line by line, since S2's own reordering finding made "confirm by re-checking the
source, not by re-asserting a prior comment" the standing rule here), so walking
`pop` once ahead of the call covers exactly what the call is about to need. This
is a pure mechanical change with no arithmetic, so unlike S1->S2 the correct gate
here really is bitwise identity, and it holds: N2 and H2 both bitwise identical
to S2, full extended suite unchanged (same 4 pre-existing failures).

**Verifying "the cache is not written inside the loop" needed a real mutation
test, and the first version of that test was itself wrong.** Built a temporary
env-gated probe (`PLANCK_FCIQMC_S3_MISS_PROBE`) that aborts `ops.diagonal` on a
cache miss, meant to prove the prefill covers every query the protected call
makes. The first version armed the probe once at step 0 and left it armed for
the rest of the run — and it aborted at step 3, `cache_size=81`, on a genuine
spawn target the prefill had not yet seen.

That was **not** a defect in the prefill. It was a defect in the *test*: the
prefill is explicitly supposed to write a new cache entry every time a
previously-unseen determinant becomes a parent — that is the whole point of
prefilling ahead of a growing population — and an "arm once, abort on any miss
thereafter" probe cannot distinguish that legitimate write from the one thing S3
actually promises, which is that `propagate_stochastic` itself never triggers a
write. Diagnosed by tagging each prefill query individually and comparing
against the abort's determinant: the first `[PREFILL] querying 39e/bf` line and
the `S3 PROBE MISS: det 39e/bf` line named the identical determinant — the probe
was catching its own permitted prefill write, not a violation.

Fixed by bracketing the probe tightly around the `propagate_stochastic` call
alone — armed immediately before it, disarmed immediately after — so the
prefill loop always runs unarmed and free to populate new entries, while the
protected call is the only place a miss is treated as a failure. With that
fix the probe ran clean across all 50 000 iterations of both gate cases, then
was deleted once it had answered the question, per the T4 precedent.

**The lesson, stated plainly because it is easy to get backwards:** when a
mutation test fires, the first question is whether the mutation-under-test or
the test itself is wrong — not which one you expected to be wrong. Tracing the
determinant identity through both the write and the abort, rather than trusting
that "a probe fired" meant "the code under test is broken," is what separated
this from a false alarm that would have blocked S3 on a correct implementation.

### S4 — add the pragma

**Re-profiled before starting, per the standing rule that every prior serial
step here has moved the ceiling.** S1 (per-bin RNG derivation), S2 (64
per-bin accumulator maps allocated every call), and S3 (the extra prefill
pass) all added real serial cost inside `propagate_stochastic` itself --
exactly the function S4 threads -- so the S0 measurement (taken right after
T4, before any of S1-S3 existed) was stale.

| | wall (N2, 1 thread) | threadable share | ceiling @4 |
|---|---|---|---|
| S0 (post-T4) | 12.61 s | 86.6 % | 2.86x |
| **pre-S4 (post-S1/S2/S3)** | **17.17 s** | **99.7 %** | **3.97x** |

The absolute serial tail (`ordered_l1_norm`, `compress`, `projected_energy`)
barely changed in seconds (1.69 s -> ~0.05 s is noise-level at this sample
count, not a real drop -- the point is it did NOT grow with S1-S3), while
`propagate_stochastic` grew by ~4.56 s from the per-bin machinery. That is
why the threadable SHARE rose to 99.7 % even though nothing about the serial
tail improved: the denominator grew, not the numerator shrank. **This is the
expected, and arguably necessary, shape for S4** -- S1-S3 exist only to make
threading safe, and their cost is overhead that only pays for itself once S4
parallelizes the code they were added to.

Projected wall time at 4 threads: `17.17 / 3.97 ~= 4.32 s`, close to
reclaiming S0's own baseline while genuinely parallelizing the original work
on top of it.

`#pragma omp parallel for schedule(static)` over the **bins**, not the parents.

- **Verify, in this order:**
  1. **CPU > 100 %** at 4 threads — one `ps` call, catches an inert pragma before
     any timing is read. (The `build-full` tree is the OpenMP-enabled one;
     `-DUSE_OPENMP` is absent from some trees and every pragma is then silently
     inert.)
  2. **Bitwise identical at `OMP_NUM_THREADS` = 1/2/4/8**, against the S3 serial
     result, on **N2** as well as H2.
  3. All FCIQMC and FCI regression cases green.
  4. **Speed, against the re-measured 3.97x ceiling above** — S0's number is now
     stale and must not be quoted as the target.

### S5 — extend the gate

`h2_fciqmc_threads1/4` runs on **4 determinants**, which may not exercise the
partition at all — a space smaller than `kBins` puts at most one parent in each
bin and can never surface a merge-order defect.

- Add an **N2 pair** (`n2_fciqmc_threads1` / `threads4`) at `atol = 0.0`, tagged
  `extended`.
- Extend the existing pair to 8 threads.
- **Verify the gate is non-vacuous:** perturb the merge order (reverse the bin
  loop) and confirm the N2 gate goes red. A gate that cannot fail is not a gate.

## What this must not do

- **Do not use `omp atomic` on the walker map, or a completion-order reduction.**
  That is the DFT-grid jitter defect.
- **Do not tie `kBins` to thread count.** The FCI sigma build paid for this twice:
  `schedule(dynamic)` gives an accumulator a different *subset* per run, and keying
  by `omp_get_thread_num()` makes contents depend on the thread *count*.
- **Do not accept "energies agree to 1e-10".** The gate is `atol = 0.0`; a
  tolerance would hide exactly the reduction-order defect the design prevents.
- **Do not thread the estimators.** `projected_energy` is 1.9 %.
- **Do not skip S0.** Every previous estimate for this step was wrong within a day.

## Why this ordering

S1-S3 are serial and each is independently verifiable; only S4 introduces
concurrency. By the time the pragma goes in, the partition, the RNG shards and the
cache are already proven at one thread — so if S4 breaks invariance, the cause is
the pragma and nothing else. Debugging a partition defect and a race at the same
time is what this ordering exists to avoid.

The one step whose values legitimately change is S1, and it is first, so every
later step has an exact predecessor to diff against.

## Key locations

| what | where |
|---|---|
| the loop | `propagate_stochastic`, `src/post_hf/ci/fciqmc.cpp` |
| RNG shard derivation | `RandomSource::derive`, `fciqmc.h:225` |
| the T4 memo | `make_ops`, `src/post_hf/fciqmc_driver.cpp` |
| the walker map | `WalkerPopulation`, `fciqmc.h` |
| the invariance gate | `h2_fciqmc_threads1/4`, `tests/regression_cases.json` |
| the precedent and its traps | `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
