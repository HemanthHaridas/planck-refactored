# Research scope: FCIQMC in Planck

**Status: both gating questions are answered. Implementation is under way — F1
(walker state) and F2 (excitation generator) are landed and gated; F3 (dynamics)
is scoped.**

FCIQMC (Booth/Thom/Alavi, *JCP* **131**, 054106) samples the FCI wavefunction with
a population of signed walkers evolving in imaginary time, instead of storing a CI
vector. It reaches determinant spaces deterministic FCI cannot, at the cost of a
stochastic error bar.

This scope opened with two questions that could each have killed the item. Both
are now answered, with evidence:

| | question | answer |
|---|---|---|
| **Q1** | Is there a calculation this codebase cannot do that FCIQMC could? | **Yes — Cr₂ CAS(12,18).** Two atoms; 344.6M determinants; breaks both the time and memory walls |
| **Q2** | Can a stochastic method be validated in a codebase gated on exactness? | **Yes — G1-G4 built and passing** as CTest `planck-statistical-gate` |

What remains is a decision and then a build. **The decision is whether anyone
wants the Cr₂ binding curve enough to justify the work** — that is not something
the codebase can answer.

---

## 1. The target: Cr₂

Surveying the standard multireference benchmarks against the *measured* FCI
boundary (§2), almost everything canonical is **already reachable**: N₂ and C₂
full-valence dissociation, benzene π, naphthalene π, [Fe₂S₂], and Cr₂ at the
textbook CAS(12,12). The first blocked case on a large molecule is anthracene π
CAS(14,14) — but that is 24 atoms.

**Cr₂ is the better target: the molecule is as small as a molecule gets while the
active space is not.**

| active space | ndet | FCI cost | one CI vector | status |
|---|---|---|---|---|
| CAS(12,12) | 853 776 | ~1.3 h | 7 MB | **reachable today** |
| CAS(12,14) | 9 018 009 | ~2 d | 72 MB | painful |
| CAS(12,16) | 64 128 064 | ~47 d | 513 MB | out of reach |
| **CAS(12,18)** | **344 622 096** | **~2 yr** | **2.76 GB** | **the target** |

The 12 active electrons are the 3d⁵4s¹ valence on each Cr. The sextuple Cr–Cr bond
is the textbook case where single-reference methods get the binding curve
qualitatively wrong, so the large active spaces are chemically motivated, not
invented to be big.

**CAS(12,18) is the sharpest statement of the gap, because it is the first case
where both walls fail at once.** Everywhere below it, time binds and memory does
not (at `n_act` 14 a CI vector is 0.09 GB against a 3-day solve). Here one vector
is 2.76 GB and a Davidson subspace ~22 GB. FCIQMC never stores a CI vector, so this
is exactly its regime.

**It needs no new determinant layer.** `kMaxPackedSpatialOrbitals` is 31
(`casscf_internal.h:27`), so 18 active orbitals fit the existing bitstring
representation. **Only the determinant count blocks** — the cleanest possible form
of the argument.

**Verified runnable, not merely arithmetic.** Cr is present in `sto-3g`, `6-31g`
and `cc-pVDZ/TZ/QZ`; Cr₂/STO-3G RHF converges in 44 iterations; the CAS(12,12) rung
starts and runs long, consistent with its estimate. Input:
`tests/inputs/exploratory/fciqmc/cr2_casscf_target.hfinp` (**not** a regression
case) — it runs the reachable rung so the boundary can be checked rather than
asserted, and raising `nactorb` to 14/16/18 walks off the cliff.

## 2. The deterministic boundary, measured

The FCI reference cost was remeasured after the sigma build got ~17x faster
(`FCI_SIGMA_BUILD_PERFORMANCE.md`). All timings at 4 threads.

| system | ndet | nα | FCI wall | s per 10³ det |
|---|---|---|---|---|
| Be/6-31g\* | 8 281 | 2 | 2.72 s | 0.328 |
| N2/STO-3G | 14 400 | 7 | 8.28 s | 0.575 |
| C2/STO-3G | 44 100 | 6 | 47.7 s | 1.081 |
| BeH2/6-31G | 81 796 | 3 | 36.7 s | 0.448 |

**Cost scales with electron count, not ndet alone**, and a single exponent cannot
express it — per-determinant cost varies **3.3x** at comparable ndet, ordered by
electron count. Fit each regime separately:

- **6-7 electrons: ndet^1.56**
- **2-3 electrons: ndet^1.14**

**Do not use a pooled exponent.** A three-point pooled fit gave ndet^1.69; BeH2 was
then run specifically to test it and came in at **36.7 s against a predicted
2.6 min — wrong by 4.3x**, in the optimistic direction. The falsifying input is
committed.

Half-filled CAS, at the 6-7 electron exponent:

| CAS(n,n) | ndet | FCI |
|---|---|---|
| (12,12) | 853 776 | ~1.3 h |
| (14,14) | 11 778 624 | ~3 d |
| (16,16) | 165 636 900 | ~208 d |
| (18,18) | 2.4 × 10⁹ | ~36 yr |

**The practical ceiling is `n_act` ≈ 12**, so the FCIQMC window opens around 13.
The ~17x speedup bought about **one** `n_act` step — a faster exponential is still
exponential.

**Consequence for validation strategy:** a deterministic reference exists only
below ndet ~10⁵, which is *not* the regime FCIQMC is for. So the **fixed-seed
reproducibility gate is the primary one** (it runs at any size) and the statistical
gate is restricted to small systems by construction.

### Validation fixtures

**Primary: `N2/STO-3G`** (10 orbitals, 7α/7β, ndet 14 400, FCI ~8 s). Smallest
system satisfying both constraints — the reference is cheap enough for a gate to
recompute, and the space is large enough that a few-thousand-walker population is a
genuine *sample*. Its 7α/7β also exercises the excitation generator properly.

**Second: `Be/6-31g*`** (ndet 8 281, 2α/2β). Comparable ndet, very different
electron count — a two-electron system cannot exercise doubles between different
occupied pairs, which is where `p_gen` bugs hide. Disagreement between N2 and Be
isolates the excitation generator.

**Third: `C2/STO-3G`** (ndet 44 100, 6α/6β, 47.7 s). At N2's orbital count but
between Be and N2 in electron count, so the three vary electron count against fixed
orbitals — three points to triangulate with rather than a confounded pair.

**`h2_fci_sto3g` and `water_fci_sto3g` are unsuitable** (4 and 441 determinants):
the walker population would exceed the space, and the gate would prove nothing
about sampling.

## 3. The validation machinery (G1-G4, built)

Built **before** any FCIQMC code, against the FCI that already ships. Passing in
2.0 s as CTest `planck-statistical-gate`.

| | what | where |
|---|---|---|
| **G1** | `metric_within_sigma` — the only runner assertion a mean-with-an-error-bar can satisfy | `tests/run_regressions.py` |
| **G2** | Flyvbjerg-Petersen blocking analysis | `tests/blocking.py` |
| **G3** | fixed-seed bitwise reproducibility harness | `tests/reproducibility.py` |
| **G4** | trivial stochastic estimator, end to end | `tests/mc_estimator.py` |

**G1** asserts a 2σ deviation passes at 3σ and **fails at 0.1σ**, and that σ = 0,
negative σ, NaN, and **infinite σ** all fail rather than passing everything.

**G2** recovers the analytic τ_int of an AR(1) series across a **39x range**
(τ 0.50 → 19.50), running 5-10 % high. The plateau is the **max** over the blocking
curve, deliberately: overestimating σ fails a gate loudly, underestimating passes
one silently. **The naive standard error understates σ by up to 6.6x** on
correlated data — the failure mode this step exists to prevent, which is why it is
gated on synthetic series with an analytic answer and never on real output.

**G3** digests raw IEEE bits (`struct '<d'`), not text: a print-precision
comparison would hide the last-ulp reduction-order defects this codebase has been
bitten by. Negative controls: an unseeded producer must fail, and one that ignores
its seed must fail seed-sensitivity.

**G4** asserts the mean within 3σ, **σ ~ N^-0.478** against a theoretical N^-0.5,
that the error bar is *calibrated* (`rms(deviation)/σ ≈ 1`, not merely large), and
— the anti-vacuity check — that the same trials **fail** against the naive σ.

All four are mutation-verified.

### Three findings from building it, each of which cost a wrong result first

1. **A fixture can be too structureless.** G4's first population was i.i.d.
   Gaussian, and a mutation restricting the sampler to half the space came back
   **green** — with i.i.d. values every sub-range has the same mean, so genuine
   sampler bias moved the answer by only **0.58σ**. A population that *trends with
   index* makes the same mutation **25.9σ** out and catches a subtler 90 %-coverage
   one. This is the inverse of the `CCGEN_MERGE_TRANSPOSES` trap where a fixture was
   too *general*. **A fixture must share the structure whose violation you intend
   to detect** — and real `H_ii` values are not exchangeable across the determinant
   space either, so the trending population is closer to the target, not merely
   more convenient.
2. **Python bytecode caching can invalidate a mutation test silently.** A `cp`
   restore preserved the mtime within the cache's resolution, so a stale `.pyc`
   kept the *mutated* module live while the file on disk was correct — misleading
   in either direction. Clear `__pycache__` between mutation runs.
3. **The naive standard error understates σ by up to 6.6x**, so every gate
   downstream of it would have passed.

## 4. What exists, and what must be built

**Reusable as-is:**

| piece | where | role in FCIQMC |
|---|---|---|
| bitstring determinants | `CIString`, `casscf_internal.h:21` | walker keys (`uint64_t`, 31 orbitals) |
| occupation/parity helpers | `count_occupied_below`, `parity_between`, `strings.h:42-43` | excitation generation |
| Slater-Condon elements | `slater_condon_element`, `ci.h:43` | the spawning step's `H_ij` |
| CI diagonal | `build_ci_diagonal`, `ci.h:101` | the death/cloning step's `H_ii` |
| active-space integrals | `h_eff` + `ga` | already transformed |
| deterministic reference | `src/post_hf/fci.cpp` | validation below ndet ~10⁵ |

`slater_condon_element` no longer heap-allocates per call (that fix landed with the
sigma-build work), so the spawn inherits it — **the allocation prerequisite this
scope originally named is satisfied.**

**Built (F1, F2):**

- **RNG policy** — `RandomSource`, seeded and per-shard, with `derive(index)`
  independent of how many shards exist.
- **Sparse walker container** — `WalkerPopulation`, a dynamic determinant-keyed map
  of signed weights with `add` / `compress`. (The existing `det_lookup` indexes a
  *fixed enumerated* space and cannot serve.)
- **Excitation generator with a consistent `p_gen`** — the O(1) `draw_excitation`,
  plus the brute-force oracle it is gated against and the in-space variant with a
  corrected `p_gen`.

**Still to build:**

- **The dynamics** — spawn, death, annihilation on a fixed shift (F3).
- **Population control.** Shift adjustment, walker targets, and the initiator
  approximation (i-FCIQMC), which is effectively mandatory beyond toy systems (F4).
- **Parallelism**, and the determinism decision in §6 (F5).

## 5. Implementation ladder

Ordered so the cheapest step can kill the expensive ones. Each is independently
verifiable against machinery that already exists.

**F1 — walker container + RNG policy. LANDED 2026-08-31.**
`src/post_hf/ci/fciqmc.{h,cpp}`. The sparse determinant-keyed map of signed
weights, plus `RandomSource`. The design point: **annihilation is not a separate
pass** — it is what accumulating signed weights into a map keyed by determinant
already does, which is also why the container is a map rather than a walker list.
`derive(index)` is deterministic in the run seed and independent of how many
shards were derived. Mutation-verified against round-to-nearest (biases the
energy), overwrite-instead-of-accumulate, and a call-order-dependent `derive()`.

**F2 — excitation generator with `p_gen`. LANDED 2026-08-31**, scoped separately
in `FCIQMC_F2_EXCITATION_SCOPE.md` because it is the one step that fails
*silently*. Five sub-steps: the brute-force oracle first (the measuring
instrument), a slow uniform reference generator, the O(1) production generator
with non-uniform `p_gen`, permanent broken-generator fixtures asserting the gate
can fail, and the spin/symmetry layer. Connection counts verified by independent
brute force: H2 3, water 140, N2 609.

Three findings from it are worth carrying into F3:

- **When a sampled quantity is used as a DIVISOR, unbiasedness of the estimator is
  the wrong property to check.** F2.5 nearly shipped a `p_gen` correction using
  the per-call attempt count: unbiased for `p_gen` (mean correct to 0.1 %), and
  **1.72x wrong** in the `1/p_gen` the spawn actually uses. F3 divides by `p_gen`
  on every spawn, so this is live there.
- **Support and frequency are different failure modes, but only once `p_gen` is
  non-uniform.** For a uniform generator a support hole also moves the frequencies
  (~54σ); for a weighted one a rare unreachable connection is ~0.6σ, invisible.
- **An equivalent mutant exposed a coverage hole.** Every fixture had been
  closed-shell, so an index bug that only manifests when α and β counts differ had
  zero coverage. F3's fixtures must include an open-shell case for the same reason.

**F3 — spawn / death / annihilation on a fixed shift.** The core loop, no
population control. Scoped in `FCIQMC_F3_DYNAMICS_SCOPE.md`. *Verify:* on
`h2_fci_sto3g` (4 determinants) the walker distribution converges to the known
ground state; the projected energy is within 3σ via G1+G2.

**F4 — shift control and the initiator approximation.** *Verify:* population
stabilises at the target; on N2/STO-3G the energy is within 3σ of the exact FCI, and
σ shrinks as √N — the **slope**, not just the value.

**F5 — parallelism, and the determinism decision.** See §6. Do not start until F1-F4
are green.

**Only then** Cr₂: CAS(12,12) against the deterministic answer first, then walk out
to CAS(12,18) where no reference exists.

## 6. The determinism tension — decide it explicitly

Every parallel path in Planck is **bitwise thread-count-invariant**, by design and
by gate: the DFT J/K builds, the CC kernels, the ERI transforms, and now the FCI
sigma build. FCIQMC's natural parallelisation is *not* — the annihilation step's
floating-point sum depends on arrival order.

**The sigma-build work makes this harder to waive, not easier.** It threaded a
scatter into a shared vector — the closest analogue in the tree to FCIQMC's spawn —
and **kept** bitwise invariance, for `kBins × dim × 8` bytes of fixed-partition
accumulators, at no measurable serial cost and 4.8 % idle. So "a costly fixed-order
reduction" is not hypothetical here; it has a worked precedent with a measured
price. The burden is to show why FCIQMC cannot do what the sigma build did.

**Two constraints inherited from that work:**

1. **A fixed-order reduction is necessary but not sufficient.** `schedule(dynamic)`
   gives an accumulator a different *subset* of terms per run, and keying
   accumulators by `omp_get_thread_num()` makes their contents depend on the thread
   *count*. **What must be deterministic is the partition of work into
   accumulators**, not merely the order they are summed. The working pattern is
   `partials[j / bin_size]` with a fixed bin count.
2. **A cheap-looking guard on an outer loop can carry asymptotic weight.** The
   recommended gather reformulation of the sigma build was built and was 2.2-2.4x
   *slower*, because moving a `|c| < 1e-15` test inward destroyed a sparsity
   exploitation. FCIQMC is built entirely on sparsity, so before restructuring any
   loop that skips negligible weight, measure what fraction of iterations the skip
   actually eliminates.

**Do not make the exception silently.** Either accept a fixed-order reduction, or
document FCIQMC as the one path where bitwise thread-invariance does not hold — as
a decision, not a discovery.

## 7. What this must not do

- **Do not skip `p_gen` validation.** An excitation generator whose returned
  probability disagrees with its actual sampling distribution produces a plausible,
  converged, wrong energy — the failure class this codebase has hit before (the
  spin-adapt default, the ERI symmetry table). Test against brute-force enumeration
  on a tiny space before trusting any energy.
- **Do not gate on `h2_fci_sto3g` or `water_fci_sto3g` alone.** Both are small
  enough that the walker population covers the space; they cannot detect a sampling
  defect. They are useful for F3's correctness, not for validating statistics.
- **Do not reimplement the determinant layer.** `src/post_hf/ci/` is shared by FCI
  and CASSCF and is the single representation. Extend it there.
- **Do not claim a speedup against FCI.** They compute different things: FCI gives
  the exact energy, FCIQMC an estimate with an error bar. The comparison is
  *reachable system size*, not wall-clock.
- **Do not let the ~17x FCI speedup be read as weakening the case.** It moved the
  constant, not the `ndet^1.56` scaling, and bought about one `n_act` step. What it
  genuinely changed is the validation budget.

## 8. Prior art

- Booth, Thom, Alavi, *JCP* **131**, 054106 (2009) — the original method.
- Cleland, Booth, Alavi, *JCP* **132**, 041103 (2010) — the initiator
  approximation, which is what makes it practical.
- NECI (Alavi group) — reference implementation; most useful for what the
  excitation generators and population control actually look like.

## Key code locations

| what | where |
|---|---|
| determinant layer (shared by FCI and CASSCF) | `src/post_hf/ci/` |
| Slater-Condon elements — the spawn | `slater_condon_element`, `ci.h:43` |
| CI diagonal — the death step | `build_ci_diagonal`, `ci.h:101` |
| deterministic reference | `src/post_hf/fci.cpp` |
| packed-orbital ceiling (31) | `casscf_internal.h:27` |
| statistical gate machinery | `tests/{statistical_gate,blocking,reproducibility,mc_estimator}.py` |
| the target case | `tests/inputs/exploratory/fciqmc/cr2_casscf_target.hfinp` |
| validation fixtures | `tests/inputs/exploratory/fciqmc/` |
| determinism precedent to imitate | `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` |
| determinism defect to avoid | `dft_xc_reduction_determinism` note |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
