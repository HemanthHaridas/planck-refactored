# HPC Campaign — What's Left

Companion to `docs/HPC_MPI_EXECUTABLE_SCOPE.md`, which scoped the whole track
before any of it landed. This file re-scopes the work **against the tree as it
stands on `devel` @ `59e5ff8`**, after PRs #146, #151, #152, #153.

Canonical status stays in `vault/Status/`.

---

## Rescope against measured data (`scale.json`, notch386, post-Gap-2)

The `scale.json` on disk (host notch386, 6-31g, engine os, 1 thread/rank, ranks
1→32) is a **post-Gap-2 run** — it was taken after the DFT grid rank-split
(6dde436, merged #156) landed on `origin/devel`. It settles the one question
the previous rescope left open and **overturns that rescope's central claim**.
All numbers below are recomputed from that file (`per_iter_s = wall_s /
n_iters`), not prediction.

> **This supersedes the earlier "DFT scaling wall did NOT close" section.** That
> analysis was correct for its data (notch460, *pre*-grid-split, DFT walling at
> 20–28%). Gap 2 then changed the code, and this newer file measures the result.
> The DFT-walls-forever conclusion is retracted — see below.

### HF — done, and provably so

| nb | x2 | x4 | x8 | x16 | x32 | eff@32 |
|---|---|---|---|---|---|---|
| 104 | 1.96 | 3.71 | 6.08 | 9.29 | 13.59 | 42% |
| 208 | 1.92 | 3.70 | 6.39 | 10.27 | 13.37 | 42% |
| 416 | 1.96 | 3.66 | 6.66 | 11.06 | 14.85 | 46% |

Near-linear to 16 ranks (9–11×), tapering to ~14× at 32 as the fixed serial
fraction (replicated diag + setup) starts to bite. Flat-to-rising across nb.
**Tier 1 for HF is closed; nothing here needs ScaLAPACK at these sizes.**

### DFT — BOTH walls closed (grid rank-split worked)

The correction to the correction. The pre-Gap-2 data showed DFT degrading
20→28% efficiency with size while HF held 60%+. **This file shows DFT now
scaling as well as HF, and *better* at large sizes:**

| nb | x2 | x4 | x8 | x16 | x32 | eff@32 |
|---|---|---|---|---|---|---|
| 104 | 1.95 | 3.71 | 6.31 | 9.85 | 14.62 | 46% |
| 208 | 1.94 | 3.76 | 6.66 | 11.32 | 15.09 | 47% |
| 312 | 1.99 | 3.91 | 7.32 | 13.02 | 20.90 | **65%** |

DFT efficiency now **rises** with size (46→65%), matching HF's Amdahl signature
instead of the old replicated-grid degradation. At nb=312 DFT scales *better*
than HF (65% vs ~44% at 32 ranks) — the grid is now genuinely distributed work,
so the larger the grid the better it amortizes. `grep -rn "Mpi::" src/dft/` is
no longer empty; `src/dft/driver.cpp` carries the rank-split.

**Gap 2's second acceptance criterion — the efficiency win at scale — is now
MET.** The Gap 2 section below is updated to CLOSED.

### The memory wall IS closed — clean nb², flat across ranks

| nb | HF MB | DFT MB | DFT/HF | RSS exponent |
|---|---|---|---|---|
| 104 | 15 | 332 | 22× | — |
| 208 | 35 | 1243 | 35× | ~1.93 |
| 312 | 68 | 2741 | 40× | ~1.95 |
| 416 | 115 | 4829 | 42× | ~1.97 |

The DFT/HF ratio looks alarming (42×) but the **exponent is the real test: DFT
RSS scales as nb^1.9 — clean nb², not nb⁴.** Per-rank RSS is also flat across
rank count (nb=208: 1243→1359 MB over 1→32 ranks — grows only ~9%, i.e. not
replicated per-rank). So the 42× is grid + AO-on-grid buffers (genuine nb² DFT
work HF lacks), **not** surviving nb⁴ tensor. Both DFT walls — memory (#151)
and scaling (Gap 2) — are closed.

**Consequence for the benchmark:** `scale_bench.py`'s Q2 `>50×` "fused path
regressed" cutoff is mis-set — 42× at nb=416 is the *healthy* number and it
rises with nb, so a larger case would false-positive. Key on the **exponent
(nb² vs nb⁴)**, not the raw ratio. See Gap 4.

### DFT nb=416 failed to CONVERGE, not OOM

`scale.json` DFT 32-water: `rc=1, "RKS did not converge"` at the serial
baseline, 4.8 GB — it ran, iterated, and hit max_cycles. The predicted failure
mode was *memory death*; the actual ceiling here is **SCF convergence**, not
memory and not scaling. The multi-rank DFT nb=416 rows are absent because the
serial baseline never converged to anchor a speedup. This is a
convergence-robustness item (guess/DIIS/damping at large water chains), wholly
separate from the HPC track — do not conflate it with a scaling defect.

---

## Where the original scope actually stands

The original doc sequenced `Tier 3 → Tier 1 → Tier 2`. Verified in-tree:

| | Original claim | Reality on `devel` |
|---|---|---|
| **Tier 3** | new target + rank-aware I/O + smoke test | **DONE.** `BUILD_MPI=ON` builds `planck-mpi` (`CMakeLists.txt:456`), `src/mpi/main.cpp` dispatches HF/DFT off the parsed `Calculator`, `Mpi::{rank,size,allreduce_inplace}` in `src/base/mpi_env.h`. Gated by `water_rhf_mpi_smoke` + `water_dft_mpi_smoke`. |
| **Tier 1** | distributed direct-SCF Fock + Allreduce, "covers RHF/UHF/ROHF **and the DFT J/K build**" | **DONE for the default engine, HF and DFT.** HF Fock in #146 (`fused_fock.h:257-267` stripe, `:403-411` reduce). DFT J/K in #151. DFT grid in #156 (Gap 2). All three go through the same stripe+`nb²`-reduce pattern; `scale.json` measures DFT at 46–65% efficiency to 32 ranks. |
| **Tier 2** | distributed RI post-HF | **NOT STARTED.** Correctly gated behind RI. |

So the "lazy path that captures most of the value" (Tiers 3 + 1) is **closed**,
with one carve-out and two gaps the original doc did not anticipate.

---

## Gap 1 — Two of four engines silently replicate (~S, low risk)

Tier 1 is complete for `IntegralMethod::ObaraSaika` (the default) and for every
engine going through `fused_fock_build`. It is **not** complete for the
one-shot `_compute_2e` tensor builds in HGP and Rys. Both carry an explicit
in-tree admission:

```cpp
// src/integrals/hgp.cpp:1392
// ponytail: HGP is NOT MPI-distributed — every rank builds the full tensor
// (correct, just replicated work); no allreduce here. Only OS (the default
// engine) is striped. Same upgrade path as the matching note in rys.cpp:
// `bra % nranks == rank` + one Mpi::allreduce_inplace on the tensor.
```

**Consequence:** `mpirun -n 8 planck-mpi` on an input with `engine hgp` or
`engine rys` in conventional mode is *correct but pointless* — 8 ranks each do
100% of the work. Exactly the failure mode #151 fixed for DFT, still live on
two engine paths.

**Why it is small:** OS already has the pattern, 6 lines at `os.cpp:2280-2287`
plus one `allreduce_inplace` at `:2380`. The comment above literally specifies
the change.

**Why it might be YAGNI:** these are the *conventional* (store-the-tensor)
paths. HPC runs use direct SCF, which is already striped for all engines. Worth
doing only if someone actually runs conventional mode on a cluster — otherwise
the honest fix is to **reject `-n > 1` with a clear message** rather than
silently replicate. Pick one; do not leave it as-is.

- **Verify:** `energy(-n 4) == energy(-n 1)` bitwise on `engine hgp` and
  `engine rys` inputs, extending `mpi_smoke_compare.py`.

## Gap 2 — DFT grid rank-split — CLOSED (6dde436, #156), proven at scale

**Status: DONE, both acceptance criteria met.** The grid was the last
replicated DFT cost after #151 distributed the J/K. The rank-split landed in
6dde436 (merged #156 on `origin/devel`): each rank evaluates and assembles its
contiguous grid-point slice, the nb² XC matrix is `MPI_SUM`-reduced in fixed
rank order (same pattern as the Fock build; done before the J/K combine so they
are not double-counted), and the partial energy/electron scalars are reduced
before feeding the SCF energy.

- **Correctness:** bitwise serial vs `-n` 2/3/4 on RKS and UKS. The DFT XC
  reduction is the historical jitter site, so the reduction sums in fixed rank
  order, never completion order — the same discipline the DFT-determinism work
  already established.
- **Scaling (the criterion the earlier draft left open):** `scale.json`
  (notch386, post-#156) measures DFT at **46% efficiency at nb=208 rising to
  65% at nb=312 on 32 ranks** — matching HF and exceeding it at the largest
  size, versus the pre-split 20–28%. The efficiency now *rises* with system
  size, the signature of genuinely distributed grid work. See the rescope
  section at the top.

Nothing remains on this gap. It was the whole DFT scaling story and it is
closed.

### Still out of scope on the DFT-grid front (separate, lower-priority items)

Gap 2 covered the **SCF-energy** grid only. These other grid consumers remain
whole-grid on every rank:

- **DFT analytic-gradient grid.** The slice seam defaults to whole-grid and only
  the two SCF-energy call sites opt in; the gradient path
  (`evaluate_current_density_and_xc`) passes no slice by design. Gradient MPI is
  a larger item — it also needs the AO-hessian-on-grid and the Pulay grid term
  distributed, not just density+XC.
- **TDDFT linear-response kernel builds** run whole-grid per rank. Out of scope
  until someone needs excited states on a cluster.

Neither is on the SCF critical path `scale.json` measures; both are future work
if the corresponding workflow goes to a cluster.

## Gap 3 — No scale-proving fixture in the regression suite (~S, no risk)

Original open risk #1, still fully open:

> **No scale-proving test exists.** Everything is validated at ≤6 atoms.

Confirmed: the largest input under `tests/inputs/regression/` is **4 atoms**
(ethylene). All 142 cases are small.

`tests/benchmarks/scale_bench.py` fills part of this — it generates water
chains up to 32 units and produced the committed Notchpeak sweep — but it is a
*benchmark harness*, run by hand, not a gate. Nothing in CI fails if
distribution silently regresses to replication.

**The cheap version:** commit one 16-water / 6-31g (nb=208) case that asserts
`energy(-n 2) == energy(-n 4) == serial` bitwise. That is a correctness gate at
a size where a partition bug can actually manifest, and it is the missing
tripwire — not a speed measurement, which CI cannot do reliably anyway.

---

## Gap 5 — post-HF is serial on BOTH axes (~M, medium risk)

The campaign's whole SCF/DFT story rides two parallel axes — OpenMP within a
rank (the `_compute_fock_rhf` pragmas, always on) and MPI across ranks (the
stripe + `nb²`-Allreduce of Tiers 1/2). **Post-HF has neither.** Confirmed
in-tree: every `src/post_hf/cc/*.cpp` backend and every ccgen-generated kernel
(`build/generated/cc/*.cpp`) has **zero** `#pragma omp`, and no post-HF path is
MPI-striped. Under the campaign's `threads_per_rank:1` model this compounds:
`mpirun -n 32 planck-mpi` on a CCSD input uses **one core** — no threads, no
ranks. This is why `scale.json` never surfaced it: it measures HF/DFT SCF at 1
thread/rank, where post-HF's missing OpenMP is invisible.

Two independent pieces, different gates:

### 5a — OpenMP within-rank — **PARTIALLY CLOSED (2026-08-30)**

`#pragma omp parallel for` on the outer contraction index of the hand-written
`tensor_backend.cpp` loops and in the `planck_tensor_cpp.py` term emitter,
mirroring the blessed `HartreeFock::ObaraSaika::_compute_fock_rhf` pattern.
Disjoint output slabs (each thread owns a slice of the residual tensor), so no
cross-thread summation for the tensor-result terms — the same shape as the
landed ERI/transform parallelization.

**Determinism constraint is load-bearing** (same discipline as Gap 2's DFT XC
reduction): the scalar-accumulator terms (`double acc; ... acc += ...; result
+= acc;` — roughly half the generated terms, all the energy-kernel and
fully-contracted terms) are **not** bitwise thread-count-invariant under a
naive `reduction(+:acc)`. Keep those serial or sum fixed-order partials; never
`omp critical`, never completion-order. Interacts with the `_partN` chunking
(the chunk sub-functions must each parallelize independently).

- **No RI dependency:** this is thread parallelism on the existing dense
  tensors, unrelated to the RI memory strategy. Lands now, independent of
  Tier 2.
- **Profile first:** which post-HF path is the actual wall-time sink at a
  realistic size (MP2 energy? CCSD iteration? CCSDTQ?) picks 5a's first target.
  Don't parallelize cold code.
- **Verify:** `energy(threads N) == energy(threads 1)` bitwise across
  N ∈ {1,2,4,8}, extending `mpi_smoke_compare.py`.

**Status.** The **generated** half is done: `planck_tensor_cpp.py` emits
`#pragma omp parallel for collapse(3) schedule(static)` on each residual nest
behind `CCGEN_OMP_COLLAPSE` (default off), measured **3.22x at 4 threads** on
HF/6-31G with energies bitwise identical at `OMP_NUM_THREADS` = 1/2/4/8 and
against the unthreaded baseline. Full record: `docs/CCGEN_CC_OPENMP.md`.

Three notes for whoever finishes this:

1. **The hand-written path is still serial.** `tensor_backend.cpp` has 0 pragmas.
2. **The determinism constraint above was real and was handled**, not dodged: the
   inner summed loop stays serial *within* a thread, so `acc` accumulation order
   is unchanged and no cross-thread reduction exists. Verified bitwise rather than
   argued.
3. **"Profile first" paid.** Doing so found that the builders — the obvious target
   at 45 % of runtime when this gap was written — had fallen to 13 % after an
   unrelated fix, and that a third of the remaining builder time was a **duplicate
   build the emitter was emitting twice**. Deleting that beat threading it. The
   advice in this section to profile before parallelizing is the reason that was
   found at all.

### 5b — MPI rank-split (= Tier 2 front half; reclassify the RI gate)

Stripe the dense contraction outer index across ranks + one fixed-order
`Mpi::allreduce_inplace` on the residual tensor, exactly the Fock/grid pattern.

**Reclassification:** the campaign parks "Tier 2 (distributed RI post-HF)"
entirely behind RI. But RI is a *memory* strategy (it shrinks the `nb⁴` tensor
to fit per-rank), not a *distribution* prerequisite — the **dense** MP2/CC
contractions can be rank-striped on the outer MO index today, with no RI. RI
gates only the *memory* ceiling, not the ability to distribute. So 5b is
dense-post-HF distribution; RI-post-HF distribution stays the separate,
memory-motivated Tier 2 tail.

- **Verify:** `energy(-n N) == energy(-n 1)` bitwise — the Gap 3 fixture
  pattern at a post-HF case.

**Dead code to leave dead:** `python/ccgen/emit/cpp_loops.py` carries an
unused OpenMP + GEMM emitter. The live Planck path is `planck_tensor_cpp.py`;
5a's pragmas go there. Do not resurrect the parallel emitter — that is a second
implementation, not a smaller diff.

---

## What the laptop cannot answer

The scaling numbers in this doc all come from Notchpeak (`scale.json`), never a
workstation, and for a concrete reason: an 8-core laptop cannot resolve MPI
scaling. At `-n 4` each rank spawns 8 OpenMP threads unless pinned → 4×
oversubscription and wall time *rises*; holding total threads constant leaves
the curve flat; single-node `mpirun` uses shared-memory transport, so no fabric
behavior is exercised. HF measures flat locally despite scaling ~10× on the
cluster — the control that proves the laptop is blind here.

**The division of labor stands:** the laptop proves bitwise correctness against
the serial oracle (which is where the risk lives — a wrong reduction yields
plausible-but-wrong numbers, not a crash); the cluster proves the payoff. Both
are now in hand for HF and DFT SCF.

---

## Recommended sequence — the DFT scaling path is closed

With Gap 2 landed **and measured** (#156, `scale.json`: DFT 46–65% eff to 32
ranks), the DFT SCF scaling story is finished. What remains is hygiene and the
post-HF track:

```
1. Gap 3 (scale fixture in CI)        ~S — commit a >6-atom multi-rank bitwise
                                          gate; protects Gaps 1+2 from regressing
2. Gap 1 (HGP/Rys stripe OR reject)   ~S — decide which; do not leave silent
3. DFT nb=416 convergence             ~S–M — separate from HPC; the largest
                                          case fails to converge, not to scale
4. Gap 5a tail (hand-written CC/MP2)  ~S–M — the GENERATED half is done (3.22x);
                                          tensor_backend.cpp is still serial,
                                          and MP2 is unprofiled. Profile first
5. Gap 5b (dense post-HF MPI stripe)  ~M — rank-split the dense contractions;
                                          NOT RI-gated (reclassified)
6. Tier 2 (distributed RI post-HF)    — the memory-motivated tail; the only
                                          item still atop RI, a second release

   DONE since this list was written:
     Gap 4 (scale_bench Q2 exponent, 2b764b21) and the Q1 Karp-Flatt verdict
     (ebf8ae5c); Gap 5a's generated half (CCGEN_OMP_COLLAPSE, 3.22x)
```

Gap 3 is now **#1** because it is the only thing standing between "DFT scales"
(measured, but by a hand-run harness) and "DFT scaling cannot silently
regress" (a CI gate). Everything above the line — Tiers 3 and 1, HF and DFT — is
proven; a fixture is what keeps it proven.

**Attribution caveat (write-up only):** `scale.json` reflects #151, #152, and
#156 together. Isolating any one needs a run at its parent commit. The combined
win is measured regardless.

## Gap 4 — **CLOSED (`2b764b21`).** `scale_bench.py` Q2 keyed on the nb exponent

New, surfaced by `scale.json`. The Q2 memory read-out flags "fused path
regressed" at `DFT/HF > 50×`. Measured healthy ratios are 22×→42× and **rising
with nb** (nb^1.9 grid growth over HF's near-flat), so a large enough case trips
the cutoff even though the memory wall is closed. The ratio is the wrong
discriminator. Fix: compute the DFT RSS **exponent** across two sizes and flag
only if it approaches nb⁴ (~4) rather than nb² (~2). One-line change to the
verdict logic; the exponent is already derivable from the rows Q2 collects.

**Done.** `scale_bench.py:533-546` fits the DFT RSS exponent across two sizes and
flags only above `nb^3.0`. The Q1 verdict was corrected in the same pass
(`ebf8ae5c`): it now keys on the **Karp-Flatt serial fraction** rather than
efficiency-at-max-ranks, which had been flagging WEAK -> "load imbalance" for
every case at 32 ranks and recommending a fix the measured data contradicts.

---

## What "done" looks like now

Tiers 3 and 1 are done and **measured**, for the default engine, HF and DFT,
across the Fock build (#146/#151) and the grid (#156). `scale.json` proves it:
HF 42–46% eff at 32 ranks (near-linear to 16), DFT 46–65% and rising with size.
Both DFT walls — memory (#151) and scaling (#156) — are closed.

Remaining, in value order:

- **Gap 3** — a committed multi-rank correctness gate at a size where partition
  bugs bite (>6 atoms, `-n 2 == -n 4 == serial` bitwise). The scaling is proven
  but only by a hand-run harness; this makes it un-regressable.
- **Gap 4** — retune the benchmark's memory verdict to key on the nb exponent,
  not the raw DFT/HF ratio, so it stops false-flagging healthy large cases.
- **Gap 1** — either distribute HGP/Rys conventional builds, or reject `-n > 1`
  on them. Not silent replication.
- **DFT nb=416 convergence** — a robustness item, not HPC: the largest case
  fails `RKS did not converge`, not memory and not scaling.
- **Gap 5** — post-HF parallelization, both axes. 5a (OpenMP, no RI gate) makes
  post-HF use more than one core per rank at all; 5b (dense MPI stripe,
  reclassified out from behind RI) distributes it. Do 5a first — higher rung,
  no RI, delivers single-node scaling immediately, and its determinism gate is
  the one 5b reuses.
- **Tier 2** — distributed **RI** post-HF: the memory-motivated tail after 5b,
  the one item genuinely still atop RI, still a second release.

## Risks, revised

1. ~~**No scale-proving test.**~~ Mostly retired: `scale_bench.py` + the
   committed `scale.json` now *prove* the scaling for HF and DFT SCF. Still open
   **as a CI gate** — nothing automated catches a regression from distribution
   back to replication. That is Gap 3, now the top remaining item.
2. **Bitwise fragility — unchanged and still the top *correctness* risk.** Every
   distributed path gates on `energy(-n k) == energy(serial)`. A wrong reduction
   yields plausible-but-wrong numbers, not a crash. The DFT XC grid reduction
   (#156) is the sharpest instance — the documented jitter site — and is why it
   sums in fixed rank order.
3. **Replicated diagonalization** — unchanged soft ceiling at nb ~2000. The 32×
   HF/DFT tapering at large rank counts (14× not 16× at 32 ranks) is the fixed
   serial fraction — mostly diag — starting to show, exactly as predicted. The
   32-water case (nb=416) is still nowhere near forcing ScaLAPACK.
4. **The two silent-replication paths (Gap 1)** mean "planck-mpi distributes the
   Fock build" is true for the default engine and false for HGP/Rys conventional
   mode, with no runtime signal. A user selecting `engine hgp` on a cluster gets
   no error and no speedup.
