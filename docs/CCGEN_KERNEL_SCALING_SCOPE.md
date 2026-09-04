# ccgen Generated-vs-Hand-Written Kernel Scaling

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Does the generated-vs-hand-written CC kernel gap grow with system size?**

**Answered: it grows. It is a scaling defect, not a constant tax.** H3 confirmed, H1 not
implicated (and not testable on this ladder). The ratio rises from 21.8× to 50.1× across the
measured range and both indices contribute.

**Revisited 2026-08-26** — H3's fix now has a second mechanism (derivation dressing, wired and
measured at 3.12×/3.61× end-to-end) which this ladder has not yet been re-run against. See the
"Revisited" material below before acting on the recommendation in "What was found"; the two
candidate fixes may overlap.

## Short answer

Six ladder points, rank-3 triples residual, `-O3`, single-threaded, repeats averaged, generated
and hand-written evaluated from identical amplitudes: the ratio rises from 21.8× (BH3/STO-3G) to
50.1× (C2H4/STO-3G) with no plateau. Per-index power-law fits give hand-written `o^3.94 v^4.18`
(4.5% max residual, textbook `o³v³` output times one contracted index) against generated
`o^4.87 v^4.52` (21.4% max residual, concentrated at high `v` — evidence of multiple contraction
regimes rather than one clean power law). The ratio itself fits `o^0.93 v^0.34`, sign-stable across
every leave-one-out variant, meaning both indices contribute with `o` roughly 3× the exponent of
`v`. This confirms H3 (no contraction-order optimization in the emitter) and leaves H1
(memory-bound by loop fission) untested, because the whole reachable ladder stays under 0.84 MiB
`t3`, inside L2.

## Where the logic lives

- `src/post_hf/cc/tensor_backend.cpp:2324` — generated-vs-hand branch + `T3_DIFF` probe
- `src/post_hf/cc/tensor_backend.cpp:1800` — hand-written triples (1-nest reference)
- `src/post_hf/cc/tensor_backend.cpp:241` — `choose_determinant_backstop`
- `python/ccgen/emit/planck_tensor_cpp.py:284`, `:443` — one-nest-per-term emission (H1's mechanism)
- `python/ccgen/tensor_ir.py:198,261,283` — unused contraction-order analysis (H3's fix,
  `_optimal_contraction_order`)
- `CMakeLists.txt:402` — `-O1` registry pin (rank 4+ only)
- `docs/CCGEN_KERNEL_PERFORMANCE.md` — the measurement record this continues

## What invariants matter

### 1. A power-law fit in k variables needs all k varied independently

Two earlier drafts of the per-index fit quoted wrong exponents, and both failures had the same
shape:

1. **Four points (before BH3/6-31G): `o^1.12 v^0.05`**, concluded as "the entire gap is in the
   occupied index — the generated kernels traverse one extra occupied loop." **Three of those four
   shared `o=5`**, so least squares had nothing to separate and loaded all divergence onto `o`.
   The 6.5% residual looked reassuring precisely *because* it was overfitting a nearly-fixed
   variable.
2. **Five points (adding BH3/6-31G, `o=4 v=11` — lowest `o`, second-highest ratio): `o^0.40
   v^0.32`**, quoted as the correction. But leave-one-out on those five swung the `o` exponent
   across **−0.65 .. +1.12** — it did not even hold its sign — with `log o` spanning only 0.223 and
   condition number 46.6. So the "correction" was no better established than what it replaced.

The sixth point (C2H4, `o=8`) tripled the `log o` spread and is what made the exponents stable.

Design rule:

- A power-law fit in k variables needs all k varied independently. Validate with leave-one-out or
  the design-matrix condition number — **never the residual**, which is what made both bad fits
  look good.

### 2. Never extrapolate a fitted ratio beyond the measured range

An earlier draft quoted ~69× at `o=10 v=40` and ~156× at `o=20 v=80` off the four-point fit; with a
21.4% residual and endpoint-sensitive exponents, projecting 2–4× beyond the measured range is not
supportable. By the same token the carried "~180× on `bh3`" folklore figure can be neither
explained nor dismissed from this data.

Design rule:

- Do not extrapolate to production sizes from a fit whose residual is large or whose exponents are
  endpoint-sensitive.

### 3. Time the residual evaluation, not a converged solve

A solve conflates kernel cost with convergence path. The `PLANCK_CC_T3_DIFF=1` probe evaluates the
generated and hand-written residuals once each from identical amplitudes and is the correct
harness for this class of measurement.

Design rule:

- Always time residual evaluation in isolation for a kernel-cost comparison, never a full solve.

### 4. The determinant backstop silently excludes small systems from the generated path

`choose_determinant_backstop` (`tensor_backend.cpp:241`) routes any case with `n_spin_orb <= 16`
**and** `ndet <= 10000` to the determinant-space teaching backstop, which never calls the generated
tensor triples kernel. A system below that threshold produces **no timing at all**, regardless of
what the backend override says. `water_rccsdt_sto3g` (`nso=14 ndet=1001`) is unusable for this
reason; `bh3`/STO-3G clears the gate only via `ndet=12870 > 10000` at `nso=16`.

Design rule:

- Any new ladder point for the **hand-written** arm must satisfy `nso > 16 || ndet > 10000`. The
  `optimized` route (through `rccgen.cpp` to the arbitrary-order harness) never consults the
  backstop, so this constraint binds the hand-written arm only and widens the usable points on the
  generated side.

### 5. Non-square test systems are mandatory, not just more informative

`no == nv` is actively hazardous, not merely uninformative: a wrongly-ordered read stays in bounds
and fails silently on a square system, which is exactly why the accessor gate (see
`CCGEN_KERNEL_PERFORMANCE.md`) uses distinct extents in every axis.

Design rule:

- Always use non-square `o`/`v` test systems for kernel-scaling or accessor-correctness work.

## What was measured

Six ladder points:

| case | o | v | o/v | generated | hand-written | ratio |
|---|---|---|---|---|---|---|
| BH3/STO-3G | 4 | 4 | 1.00 | 0.0309 s | 0.00142 s | 21.8× |
| CH4/STO-3G | 5 | 4 | 1.25 | 0.0930 s | 0.00347 s | 26.8× |
| HF/6-31G | 5 | 6 | 0.83 | 0.5681 s | 0.01779 s | 31.9× |
| H2O/6-31G | 5 | 8 | 0.62 | 1.7232 s | 0.06316 s | 27.3× |
| BH3/6-31G | 4 | 11 | 0.36 | 3.3287 s | 0.09741 s | 34.2× |
| C2H4/STO-3G | 8 | 6 | 1.33 | 5.9509 s | 0.11875 s | **50.1×** |

Per-index exponents. Six points, `log o` spread 0.693, `log v` spread 1.012, design-matrix
condition number 26.1:

| | fit | max residual |
|---|---|---|
| hand-written | **`o^3.94 v^4.18`** | 4.5% |
| generated | **`o^4.87 v^4.52`** | 21.4% |
| **ratio** | **`o^0.93 v^0.34`** | — |

Leave-one-out on the ratio exponents:

| dropped point | ratio fit |
|---|---|
| BH3/STO-3G | `o^+0.94 v^+0.35` |
| CH4/STO-3G | `o^+0.93 v^+0.35` |
| HF/6-31G | `o^+0.93 v^+0.34` |
| H2O/6-31G | `o^+0.94 v^+0.45` |
| BH3/6-31G | `o^+1.18 v^+0.04` |
| C2H4/STO-3G | `o^+0.40 v^+0.32` |

Four of six variants agree to two decimals, and the `o` exponent keeps its sign in all six
(+0.40 .. +1.18). That is a usable result — but note the two variants that move it are exactly the
two points that extend the ladder's reach (BH3/6-31G at max `v`, C2H4 at max `o`), i.e. the fit is
still leaning on its endpoints. Treat `o^0.9 v^0.3` as indicative, not settled.

**Both indices contribute, with `o` roughly 3× the exponent of `v`.** Consistent with the known
`t2·t3·v` case in `CCGEN_HIGHER_OPERATOR_REUSE.md` (`o⁵v⁵` n-ary vs `o³v⁴` factored), which is
superlinear in both.

**Where the generated fit is bad, and why that is informative.** The generated-side residual
(21.4%) is not scattered — it is concentrated at high `v`:

| case | actual | fit | error |
|---|---|---|---|
| BH3/STO-3G | 0.0309 s | 0.0307 s | −0.8% |
| CH4/STO-3G | 0.0930 s | 0.0910 s | −2.1% |
| HF/6-31G | 0.5681 s | 0.5696 s | +0.3% |
| C2H4/STO-3G | 5.9509 s | 5.6233 s | −5.5% |
| BH3/6-31G | 3.3287 s | 2.9781 s | −10.5% |
| H2O/6-31G | 1.7232 s | 2.0922 s | **+21.4%** |

The four lowest-`v` points fit to ≤5.5%; the two highest-`v` points (v=8, v=11) are where it breaks
down, in opposite directions. The hand-written kernel has no such pattern (4.5% max). So a single
`o^a v^b` term does not describe the generated cost — there is more than one contraction regime in
it, consistent with different terms in the residual having different optimal orders and the
emitter picking none of them.

**What the ladder supports:**

- The ratio grows, and steeply — 21.8× → 50.1× across the six points, with no plateau. Measured
  directly, not fitted. A scaling defect, not a constant tax. H3 confirmed.
- The hand-written kernel is textbook — `o^3.94 v^4.18` at 4.5% max residual.
- Both indices contribute, `o` about 3× more than `v` (`o^0.93 v^0.34`, sign-stable across all
  leave-one-out variants).
- The generated kernel does not obey a single power law (21.4% residual, concentrated at high `v`)
  while the hand-written one does — evidence of multiple contraction regimes in the generated code.

### Ladder point details (for extending this measurement)

Any new ladder point must satisfy `nso > 16 || ndet > 10000` (see Invariant 4). Candidates used
here, with the generated kernel actually reachable:

| case | occ | vir | o/v | nso | ndet | t3 (MiB) |
|---|---|---|---|---|---|---|
| BH3/STO-3G (baseline) | 4 | 4 | 1.00 | 16 | 12870 | 0.03 |
| CH4/STO-3G | 5 | 4 | 1.25 | 18 | 43758 | 0.06 |
| HF/6-31G | 5 | 6 | 0.83 | 22 | 646646 | 0.21 |
| BH3/6-31G | 4 | 11 | 0.36 | 30 | 5852925 | 0.65 |
| H2O/6-31G | 5 | 8 | 0.62 | 26 | 5311735 | 0.49 |
| C2H4/STO-3G | 8 | 6 | 1.33 | 28 | 30421755 | 0.84 |

That spans `o/v` from 0.36 to 1.33 and `t3` from 0.03 to 0.84 MiB — enough to vary shape at
near-fixed size (CH4 vs HF/6-31G) and size at near-fixed shape (BH3/STO-3G vs C2H4/STO-3G). None
of these leave L2 (0.84 MiB t3 at the top), so the H1 cache-transition prediction is **not**
testable on this ladder alone; reaching it needs cc-pVDZ-class cases (H2O/cc-pVDZ is 6.5 MiB t3)
whose runtime must be checked before committing to them.

Minimum useful set for any future extension: fixed shape with growing size (separates constant
from scaling), fixed size with varying `o/v` (separates H1's bytes hypothesis from H3's exponents),
and at least one case whose `t3` working set exceeds L2, or H1 stays untested by construction.

Rank 4 is the production target and is **not** covered by this rank-3 ladder — different tensor
types, different code path, and a lesson already paid for once: the fixed-rank-only accessor pass
fixed rank 3 by 76× and moved rank 4 by nothing. Rank 4 also still carries the `-O1` registry pin
(`CMakeLists.txt:402`) that rank 3 does not, so its ratio has a known extra term. `be_rccsdtq_sto3g`
at 11.4 s/iteration is the cheapest rank-4 handle, but no point has been measured there yet.

## What was found (H1 vs H3)

The two hypotheses that survived the earlier accessor-fix investigation (`CCGEN_KERNEL_PERFORMANCE.md`)
made opposite predictions here, and both were live before this ladder:

- **H1 (memory-bound by loop fission)** — falsified *at `no=nv=4`* because the residual is 32 KB,
  fully L1-resident, so 1063 separate sweeps cost nothing extra. That argument expires the moment
  the working set exceeds cache, which is a function of size alone. H1 predicts the ratio is ~flat
  while everything fits in cache, then **rises** once it does not.
- **H3 (no contraction-order optimization)** — the generated kernels evaluate each term n-arily in
  emission order; `CCGEN_HIGHER_OPERATOR_REUSE.md` records `t2·t3·v` as `o⁵v⁵` n-ary against
  `o³v⁴` if `v·t3` is factored first. H3 predicts the ratio **grows polynomially** in `o`/`v`, and
  does so regardless of cache.

They are separable by shape, not just by size: H1 keys on total working-set bytes, H3 on the
`o`/`v` exponents. The measurement above confirms H3 and leaves H1 untested because the whole
reachable ladder stays inside L2.

## Revisited 2026-08-26: H3's fix has a second, since-measured mechanism

This document names one fix for H3 — consume `_optimal_contraction_order` in the emitter. A
**second** mechanism addresses the same hypothesis and has since been wired and measured:
**derivation dressing** factorizes each term's n-ary contraction into a binary tree, which is
contraction-order optimization arrived at from the operator side rather than the loop side.

H3 is stated here as: *"the generated kernels evaluate each term n-arily in emission order;
`t2·t3·v` is `o⁵v⁵` n-ary against `o³v⁴` if `v·t3` is factored first."* Factoring `v·t3` first is
exactly what a derived operator does.

Measured (`docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`), generated-undressed vs
generated-derivation-dressed, same binary configuration apart from `--dressing`:

| system | o | v | undressed | dressed | speedup |
|---|---|---|---|---|---|
| LiH/STO-3G | 4 | 8 | 5.12 s | 1.64 s | **3.12x** |
| CH4/STO-3G | 5 | 4 | 104.56 s | 28.94 s | **3.61x** |

Energies identical to all printed digits; CH4 takes 15 iterations either way, so this is
per-iteration work.

**What that does and does not establish.**

- It is **consistent with H3** and with this document's answer: if the gap is a contraction-order
  defect, factoring the contractions should close part of it, and it does.
- It is **not** a measurement on this ladder. These are end-to-end solve times on two systems, not
  the isolated triples-residual timings the six points above use, and the two systems are not on
  the ladder (LiH is not; CH4 is, at `o=5 v=4`). **The two sets of numbers are not comparable and
  must not be combined into one ratio.**
- Two points cannot give exponents. Whether dressing reduces the *scaling* or only the *constant*
  is unmeasured — and that is precisely the distinction this document exists to make. The ratio
  grew slightly between the two (3.12 → 3.61) but with `o` and `v` both changing, which is exactly
  the degenerate-ladder trap in Invariant 1.

**A re-run of this ladder under `--dressing derived` was attempted and ABANDONED (2026-08-29).**
The reason is worth recording, because it bounds what this ladder can ever deliver.

`PLANCK_CC_T3_TIME` cannot fire in any build — it sits in the `use_generated_kernels` branch that
the rank-3 representation fix rerouted away from. A replacement three-arm probe was built and it
established something more basic: **the hand-written and generated arms have no residual-level
agreement gate.** They are distinct solvers with distinct amplitude representations, both
individually correct (each converges to `E_corr = -0.0791116825` on CH4, PySCF to 1.4e-08), with no
shared intermediate state where their residuals are elementwise comparable. Four framings were
tried and all failed; `restore` in particular belongs to a wedge-packed *amplitude* and annihilates
a raw residual (2.0e+05), which `CCGEN_RANK3_KERNEL_AND_SOLVER.md:21-24` had already established.

The only remaining comparison is whole-iteration timing validated by converged energy — which
measures *"solver iteration"*, not *"triples kernel"*, since each arm's own overhead is inside it.
That does not answer this document's question, and it cannot be quoted against the `o^4.87 v^4.52`
fitted here.

**So the actionable levers here are code-level comparison and FLOP estimates, and the measurement
route is closed.** The recommendation below stands on the fit already in this document; do not
reopen the dressed re-run expecting it to adjudicate.

**One constraint carries over unchanged.** `choose_determinant_backstop` still gates which systems
reach a generated kernel *by the hand-written route* — but not the generated one. `optimized`
routes through `rccgen.cpp` to the arbitrary-order harness, which never consults the backstop, so
the `nso > 16 || ndet > 10000` requirement recorded above applies to the hand-written arm of the
comparison only. That widens the set of usable ladder points on the generated side.

## What this makes worth doing

**SETTLED 2026-08-29 — read this before acting on the recommendation.** Both levers this document
points at were built and measured (`CCGEN_WHY_GENERATED_IS_SLOW.md`):

- **Contraction order is fixed by `--dressing derived`**, which eliminates the 391 four-deep
  `o⁵v⁵` terms `_optimal_contraction_order` would target — **measured 3.6x**. Consuming the IR
  hints is therefore **probably redundant**; re-check before building it.
- **Loop fusion is refuted at ~0 %**, twice: 806→15 nests changed runtime 0-3 %, and again after a
  later fix raised the residual's share of runtime from 32 % to 55 %.

What the profile found instead was **redundant operator construction** — 67.7 % of the kernel,
fixed for **1.76x** — and that **CC has no OpenMP at all** (modelled 3.86x). Neither was visible to
a cost model.

The originally recommended row was: **consume `_optimal_contraction_order` in the emitter**
(`python/ccgen/tensor_ir.py:283`, currently computed and discarded — `grep BLASHint
python/ccgen/emit/planck_tensor_cpp.py` returns nothing). That was the asymptotic fix and it
outranked loop fusion, which the earlier P2 measurement already showed buys nothing at small size.

The fit no longer localizes the defect to one index, so it does not hand you a single loop to go
delete. The term-level approach `CCGEN_HIGHER_OPERATOR_REUSE.md` frames is the productive next
step: find which generated terms are evaluated n-arily where a factored order is cheaper — the
recorded `t2·t3·v` case is `o⁵v⁵` n-ary against `o³v⁴` factored, which is superlinear in *both*
indices and therefore consistent with the measured `o^0.40 v^0.32`. Enumerate those terms and their
cost gap before writing any emitter change.

## Validation strategy that should remain in place

- Do not re-derive the accessor result — it is landed and gated (`CCGEN_TENSOR_ACCESSOR.md`); if a
  measurement here disagrees with the recorded before/after, suspect the build tree first.
- A/B in one configure, rebuilding both arms. Both misreads during the P1/P2 work came from
  comparing binaries built from different source states or different `CMAKE_BUILD_TYPE`s. Never
  compare against a stale `build/`.
- `make -j4` — the generated TUs are large enough that a full-width build is disruptive.
- Energies must stay bitwise-identical across anything this investigation motivates, same as the
  accessor fix — these are all evaluation-order-preserving changes until proven otherwise, and a
  fusion that reassociates floating-point accumulation is *not*, so that would need its own
  justification rather than being absorbed into a tolerance.

## Remaining architecture concern

- **H1 remains untested, by construction.** The whole reachable ladder tops out at 0.84 MiB `t3`,
  which never leaves L2, so the cache-transition prediction cannot fire here. Testing it needs
  cc-pVDZ-class systems (H2O/cc-pVDZ is 6.5 MiB `t3`); at ~50× generated-kernel slowdown that run
  should be time-boxed before committing to it. H1 and H3 are not exclusive — H1 could still add a
  term on top of the measured growth once the working set spills.
- **Rank 4 has no point on this ladder.** Different tensor types, different path, plus the `-O1`
  registry pin (`CMakeLists.txt:402`) that rank 3 lacks. The fixed-rank-only accessor pass already
  proved rank 3 is not a proxy for rank 4; do not assume `o^0.9 v^0.3` transfers.
- **The generated-side residual (21.4%) is itself a finding.** The hand-written kernel fits a clean
  power law (4.5%); the generated one does not. Something in it is not a simple `o^a v^b` — worth
  understanding before trusting any exponent from it too far.
- **Whether derivation dressing reduces the scaling or only the constant is unmeasured**, and the
  direct route to measure it (re-running this ladder under `--dressing derived`) is closed — see
  the Revisited section. Any future settlement of this needs a different instrument.

## Key code locations

| what | where |
|---|---|
| generated-vs-hand branch + `T3_DIFF` probe | `src/post_hf/cc/tensor_backend.cpp:2324` |
| hand-written triples (1-nest reference) | `src/post_hf/cc/tensor_backend.cpp:1800` |
| one-nest-per-term emission (H1's mechanism) | `python/ccgen/emit/planck_tensor_cpp.py:284`, `:443` |
| unused contraction-order analysis (H3's fix) | `python/ccgen/tensor_ir.py:198,261,283` |
| `-O1` registry pin (rank 4+ only) | `CMakeLists.txt:402` |
| measurement record this continues | `docs/CCGEN_KERNEL_PERFORMANCE.md` |
