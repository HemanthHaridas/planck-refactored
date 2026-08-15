# The generated rank-3 CCSDT triples residual is wrong — handoff

**Open defect, in-flight.** Rewrite as an architecture answer once fixed.

`compute_ccsdt_triples_residual` (ccgen-generated, rank 3, **undressed**) does not reproduce the
hand-written triples residual at identical amplitudes. It is independent of the dressing work — see
`CCGEN_DRESSING_AND_SPIN_ADAPTATION.md` for why that path was retired, which does **not** fix this.

## The measurement

One evaluation, identical amplitudes, `bh3_rccsdt_sto3g`:

```
raw (no restore): max|gen-hand|=2.033739e-02 rms=1.220552e-03 max|hand|=2.033739e-02
dims=4x4x4x4x4x4 n=4096 gen_zero=88.3% hand_live=26.2% missing=14.5% both_live=11.7%
intersection n=480 max|gen-hand|=1.067064e-02 rms=1.947137e-03
  ratio[min=-149.0149 mean=-0.0673 max=66.0037]
ratio buckets: neg=148 ~1/3=2 ~1/2=4 ~1=16 ~2=0 ~3=12 other=298
```

Two independent failures:

1. **It writes ~45 % of the elements it should.** Generated-live support is a strict *subset* of
   hand-written-live support — nothing lands outside the reference support.
2. **Values on the shared elements are largely unrelated.** `gen/hand` spans −149 to +66 with no
   dominant bucket.

Holds identically on both emit paths (plain rank-3 and arbitrary-order).

**One structured feature remains unexplained: 148/480 (31 %) sign flips.** Recorded deliberately
without a hypothesis — four structured-looking features have already died on the next measurement.

## Partially fixed

A missing physicist rebind (`eb1c611`): ccgen emits against physicist `<pq|rs>` while
`state.mo_blocks` holds chemists' `(pq|rs)`. The arbitrary-order path always rebound; the plain
rank-3 path did not. Worth 1.82e-4 → 2.7e-5 Eh — most of the error, not all.

## The comparison is sound

Both residuals feed the same `triples_residual`, the same `restore_restricted_t3_structure`, and the
same amplitude update (`tensor_backend.cpp:2586` vs `:2337`) — interchangeable producers of one
quantity, consumed identically. Audited statically.

## Ruled out by measurement — do not re-run

| hypothesis | verdict |
|---|---|
| Spin-adaptation config mismatch | **No.** A build matched on `SPIN_ADAPT=ON MAXORDER=4 ENGINE=diagram` gave the identical wrong energy. |
| `restore_restricted_t3_structure` is generated-only | **No.** The hand-written branch calls it too and converges correctly. |
| Its non-idempotent ×6 sum is a bug | **No.** Compensated by the repeated-index pre-scaling. |
| `_ERI_SYMMETRY_PERMUTATIONS` has invalid −1 perms | **No.** Already the corrected +1-only form. |
| The emit is unfaithful to the generator | **No.** A fresh `print_cpp_planck` reproduces the built TU's amplitude-read counts exactly. |
| T3 storage/slice layout (wrong extent/stride) | **No.** Per-axis histograms of missing elements are flat; a bad stride collapses them onto a sub-range. |
| Index permutation / transposition | **No.** Generated-live count *equals* `both_live` — nothing lands outside the reference support. |
| Dropped branch, intersection otherwise exact | **No.** Only 16/480 shared elements sit at ratio ≈ 1. |
| Constant mis-weight (½, 2, …) | **No.** No dominant ratio bucket; ~2 has **zero**. |
| The probe compares mismatched quantities | **No.** Audited statically. |
| The passing CCSDTQ==FCI gate proves the rank-3 equations sound | **No.** Rank 4 emits its **own** `compute_ccsdtq_triples_residual`; no emitted triples code is shared with rank 3. |

## Coupled: is the defect rank-parity dependent?

Rank 4 (CCSDTQ) reaches FCI while rank 3 fails. If even ranks are correct and odd ranks wrong,
the suspect list below narrows to the two odd-rank-specific entries. But the hypothesis currently
rests on two data points only — rank 2's generated TU has **no consumer** so it has never run, and
ranks 5/6 have no gate. Scoped as its own investigation, with the rank-4 exactness confound (Be has
4 electrons, so CCSDTQ is complete for it) controlled: `CCGEN_RANK_PARITY_INVESTIGATION.md`.

## Remaining suspects

1. **Lowering of the spatial (spin-adapted) triples terms at rank 3** — the `2·direct − exchange`
   structure. Leading candidate, but the failure is more than one missing branch.
2. **`restore_restricted_t3_structure` interaction** — the generated residual may not arrive in the
   pre-scaled form the convention assumes.

## Why this was not caught

`compute_ccsdt_triples_residual` **had no caller.** `choose_rccsdt_backend` returned only
`DeterminantPrototype` or `TensorProduction`; the single call site was guarded by
`use_generated_triples_kernel`, and both `run_tensor_rccsdt_impl` callers passed `false`. Generated,
compiled, linked, never executed. Wiring fixed in `64d0074`.

Every prior check — symbolic algebra, spec-metadata validity, flag-matrix byte identity,
does-it-compile, does-it-co-include — passes on an unreachable kernel. **Linkage is not execution.**
Now gated by `GeneratedKernelsAreReachableTests`
(`python/ccgen/tests/test_dressed_tu_coinclusion.py`), which asserts the chain link by link; each
link was verified to fail by re-injecting the original defect.

## How to reproduce

```bash
export BASIS_PATH=$PWD/basis-sets
PLANCK_RCCSDT_BACKEND=optimized PLANCK_CC_T3_DIFF=1 \
  /tmp/claude-501/rank3arb/hartree-fock \
  tests/inputs/regression/post_hf/bh3_rccsdt_sto3g.hfinp 2>&1 | grep -E "T3-DIFF|T3-R1"
```

**`PLANCK_RCCSDT_BACKEND=optimized` is required** — without it the run silently selects the
hand-written backend and prints no probe line at all. Confirm the backend marker (`RCCSDT[OPT]`,
`kernels=ccgen-generated`) before believing any number.

```bash
cmake -B /tmp/claude-501/rank3arb -S . -DCMAKE_BUILD_TYPE=Release \
  -DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_SPIN_ADAPT=ON -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON
make -C /tmp/claude-501/rank3arb hartree-fock -j4
```

## What NOT to do

- **Do not gate on converged energy or post-`restore` residuals.** `restore` masks the raw error
  ~11–29×. Compare the **raw** residual at fixed amplitudes.
- **Do not debug through the solve.** The probe is one evaluation; a solve is ~21 min and conflates
  kernel error with convergence path.
- **Do not fix at one rank.** The conventions live in the emitter; a per-call-site patch re-arms the
  trap elsewhere. The physicist-rebind fix was already this mistake once.
- **Do not trust a probe number without checking the backend marker.**
- **Do not assume the CCSDTQ==FCI gate covers rank 3.** Different function, different TU.
- **Do not revert the backend wiring to make things green.** Reaching this kernel is what exposed
  the defect.
- **Do not use a square test system for a new gate.** `nv == no` (Be/STO-3G is 4 and 4) lets a
  wrongly-ordered read stay in bounds and fail silently.

## Deferred

**The ~180× slowdown.** 7 s → ~1270 s on generated builds. Correctness first, but this is the
largest available performance lever in the CC path. Candidates: intermediates rebuilt inside loops,
absence of CSE. *Gate:* a profile naming the dominant cost, not a guess.

## Key code locations

| what | where |
|---|---|
| generated-vs-hand branch | `src/post_hf/cc/tensor_backend.cpp:2321` |
| T3 probe (`PLANCK_CC_T3_DIFF`) | same file, inside the generated branch |
| `rebind_physicist` | `generated_arbitrary_prepare.cpp`, declared in `generated_arbitrary_runtime.h` |
| backend selection | `choose_rccsdt_backend`, `tensor_backend.cpp:~2740` |
| `restore_restricted_t3_structure` | `tensor_backend.cpp:1976` |
| reachability gate | `python/ccgen/tests/test_dressed_tu_coinclusion.py` |

Commits: `64d0074` (wiring), `eb1c611` (physicist rebind), `38946ee` (probe).
