# Should `merge_transposes` be threaded into the production dressing path?

**Scope for in-flight work. Not started — but RE-COSTED 2026-08-29 against a
profile, and it is worth more than this document concluded.** Opened by
`CCGEN_WIRING_THE_DERIVATION_ROUTE.md`, which wired the derivation route,
measured it at 3.12x/3.61x, and left this as the one deferred lever.

## Re-costing (2026-08-29) — the original estimate was model-derived and low

This document deferred the merge on the grounds that the modelled FLOP saving is
**1.02x-1.20x** and "the likely win is compile time, not speed". That figure came
from an **operator-count model**. A profile of the post-H5 generated path
(`CCGEN_ARBITRARY_HARNESS_COST.md`) contradicts it.

**Measured operator counts**, spin-adapted `ccsdt`, `factorize_equations(...,
merge_transposes=True)` against `False`:

```
288 -> 91 operators overall   (3.2x)
```

**Weighted by measured runtime share** (HF/6-31G, 25 483 leaf samples; builders are
39.0 % of runtime in total):

| family | share of runtime | before | after | ratio | removable |
|---|---|---|---|---|---|
| `t2t2v_oooovv` | **20.4 %** | 38 | **4** | **9.5x** | 18.3 % |
| `t1t3v_oooovv` | 8.9 % | 19 | 11 | 1.7x | 3.7 % |
| `t1t1t2v_oooovv` | 5.6 % | 12 | 2 | 6.0x | 4.7 % |
| **top 3** | **34.9 %** | | | | **26.7 %** |

**Predicted speedup: 1.21x - 1.36x.**

- Upper bound **1.36x** assumes builder cost scales with call count.
- Lower bound **1.21x** applies H5's measured realization factor. H5 predicted 4x
  on call count and delivered **2.64x** (0.66), because the eliminated builds were
  cheaper than average. Assuming the same discount here is the conservative read.

**Even the discounted floor sits at the old model's ceiling.** The operator-count
model was not wrong about counts; it was wrong to treat operators as equal-cost.
`t2t2v_oooovv` writes a **rank-6 result over a 9-deep nest** — it is one of the
most expensive operator shapes emitted, and there are 38 of them that are **one
contraction** differing only in index slots.

**What this does not claim.** Merging cannot remove all of the 34.9 %: the merged
operator is still built once, and the call sites still read it (transposed, by
index, inside the loop nest — no copy). The 26.7 % is what the *duplicate builds*
cost. Nor is this a measurement — it is a profile-weighted estimate, which is
strictly better grounded than the operator-count model it replaces but still needs
M2/M3 below to confirm.

**Revised recommendation: do this next, ahead of the compile-time framing.** The
mechanism is built, symbolically exact, and value-gated 0/2536 at rank 4. That is
the position the derivation route was in before it proved worth 3.6x.



W3 deferred it deliberately — the merge is the end state, and absorbing the
factorizer's seven selection knobs before the route had proven correct was the
accumulation W3 set out to avoid. The route is now proven: energies match the
undressed baseline to 2e-10 (CH4) and exactly (LiH), and W5 gave it real
wall-clock numbers. The deferral has expired.

## The question, stated carefully

`merge_transposes=True` folds transpose-equivalent operators onto one shared
array. `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` measures it as:

| manifold | operators | **modelled savings** | value gate |
|---|---|---|---|
| `ccsd` doubles | 27 → 19 (1.4x) | **1.02x** | 0 / 45 |
| `ccsdt` triples | 80 → 39 (2.1x) | **1.03x** | 0 / 345 |
| `ccsdtq` quadruples | 254 → 69 (3.7x) | **1.20x** | 0 / 2536 |

**Read those two numeric columns as different quantities, because they are.**
The 1.4x-3.7x is an *operator count* reduction; the 1.02x-1.20x is the
*modelled FLOP saving*. Only the second is a performance claim, and it is
modest — 2-3 % at ranks 2-3.

So the honest framing is not "the merge is worth 3.7x" — an operator count is not
a speedup. **But it is also not "probably compile time, not speed", which is what
an earlier revision of this section concluded.** That reading treated all operators
as equal-cost. The profile above shows they are not: the 38-member
`t2t2v_oooovv` family is **20.4 % of runtime by itself**, because it writes a
rank-6 result over a 9-deep nest, and it merges 38 → 4.

**Current best estimate: 1.21x - 1.36x of runtime, plus the compile-time and
code-size win.** Still to be settled by measurement (M2/M3), but the prior is now
profile-weighted rather than count-weighted.

## Why this is worth doing anyway

Three reasons that do not depend on the FLOP model:

1. **Compile time is a real cost here.** `generated_kernel_registry.cpp` is
   pinned to `-O1` (`CMakeLists.txt:408-415`) because it is otherwise
   pathological, and the dressed CCSDTQ TU is 13 MB. 288 builders → fewer is a
   direct reduction in what the compiler must chew.
2. **Rank 4 is where the factorizer matters most** and where the merge ratio is
   largest — and `CCGEN_OPERATOR_IDENTITY_AND_REUSE` records the merge's rank-4
   value gate (0/2536) as the strongest single result in that document.
3. **It is already implemented and value-gated.** This is a threading question,
   not a research one.

## Steps

Ordered so the cheapest step can kill the expensive ones.

### M1 — measure before wiring (~S, no production change)

Emit the spatial `ccsdt` dressed TU both ways by calling
`factorize_equations(..., merge_transposes=True/False)` directly, and record:
builder count, TU bytes, and distinct operator arrays.

*Verify:* the numbers. The parent doc predicts 59 → 31 builders on spatial
`ccsd`; confirm the rank-3 figure rather than assuming it transfers.

### M2 — compile time and size, which is the likelier win (~M, one build each)

Build the dressed tree with and without the merge. Time
`generated_kernel_registry.cpp` specifically — it is the pathological TU — and
record binary size.

*Verify:* a wall-clock compile delta and a byte delta. **If this is the only
win, say so** and wire it on that basis rather than implying a speed claim the
FLOP model does not support.

### M3 — runtime, measured not modelled (~S, reuses M2's binaries)

Run `lih_rccsdt_generated_sto3g` and `ch4_rccsdt_sto3g` on both, three runs
each, against W5's baseline (LiH 1.64 s, CH4 28.94 s dressed).

*Verify:* medians and spread. **Expect 1.21x - 1.36x** from the profile-weighted
re-cost above — not the ~1.03x the original operator-count model predicted, which
this document now treats as superseded. Below ~1.05x means the profile weighting
is also wrong and the merge really is a compile-time change; at or above 1.2x it
is a speed lever and rank 4 deserves the same treatment immediately.

**Measure the `t2t2v_oooovv` family specifically**, via `PLANCK_CC_RANK_TIME` or a
`sample` on the merged build. It is 20.4 % of runtime and merges 9.5x, so it
carries most of the predicted effect; if the total moves but that family does not,
the causal story is wrong even though the number looks right.

*Energies must be bitwise-identical.* The merge shares arrays between call
sites — exactly the class of change that was wrong in D4 — so a wrong merge is a
correctness defect, not a slowdown.

### M4 — thread it, if M2/M3 justify it (~S)

`factorize_equations` already takes `merge_transposes`; `print_cpp_planck` does
not pass it. The wiring question is whether it becomes a fourth `--dressing`
value (`derived-merged`), a separate boolean, or simply the default for
`derived`.

**Prefer making it the default for `derived`** if M3 shows no regression: it
avoids a fifth axis on a function that already carries 16 branches, and the
un-merged form has no known advantage. A flag is only justified if M3 finds a
case where merging loses.

### M5 — extend the builder gate to the merged path (~S)

`test_emitted_builder_matches_spec.py` checks every `build_W_*` against its
spec. Under merging, several call sites read one array through a permutation, so
the gate must also check that each *call site* reads its operator correctly —
the property D3 verified by hand and D4's defect violated.

*Verify:* the extended gate is red if a permutation is dropped at a call site.
`CCGEN_OPERATOR_IDENTITY_AND_REUSE` warns that every `ccsd` merge permutation is
a **self-inverse two-element swap**, so applying it backwards is undetectable on
that manifold — use rank 3+, where 3-cycles exist.

## What this must not do

- **Do not absorb the factorizer's other six knobs.** `top_k`,
  `savings_fraction`, `memory_budget_bytes`, `max_operator_bytes`, `n_occ`,
  `n_vir` stay out of `print_cpp_planck`. W3's condition — one parameter, not
  seven — is what made the W3.3 deletion possible.
- **Do not quote 1.4x-3.7x as a speedup.** It is an operator count.
- **Do not use `random_tensors` in any new gate.** It antisymmetrizes `v`, under
  which invalid ERI relations are true; that fixture is why a 41/288 defect
  passed every symbolic check.

## Key code locations

| what | where |
|---|---|
| the merge, already implemented | `manifold_operators_with_plan`, `python/ccgen/optimization/factorize.py:703` |
| the seam it would thread through | `factorize_equations`, same file |
| the production caller that does not pass it | `print_cpp_planck`, `python/ccgen/generate.py` (the `dressing == "derived"` branch) |
| the merge's measurements | `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` |
| W5's baseline timings | `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
