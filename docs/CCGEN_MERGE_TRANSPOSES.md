# ccgen Transpose-Equivalent Operator Merging

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Should transpose-equivalent dressed operators share one array?**

## Short answer

**Yes, and it is worth 1.42x-1.52x — three times what the model that deferred it
predicted.** Landed 2026-08-29. `merge_transposes` is now unconditional for
`--dressing derived`; there is no flag.

The transform folds operators that differ only by a permutation of their result
slots onto one shared array, which the call sites then read through that
permutation instead of each building its own copy. It was already implemented,
symbolically exact, and value-gated 0/2536 at rank 4 — it simply had no
production caller. Wiring it was **9 lines in one file, 8 of them comment**.

## Where the logic lives

- `python/ccgen/optimization/factorize.py:703` — the merge (`manifold_operators_with_plan`)
- `python/ccgen/generate.py` — where it is switched on (`print_cpp_planck`, the
  `dressing == "derived"` branch)
- `python/ccgen/tests/test_merged_call_sites.py` — the call-site gate
- `python/ccgen/tests/test_emitted_builder_matches_spec.py` — the definition-only gate it
  complements
- `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` — the symbolic identity decision
- `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md` — the route this rides on

## What invariants matter

### 1. A wrong merge is a correctness defect, not just a missed optimization

One array now serves many readings. `W_t2t2v_oooovv_07fe` is read at 8 distinct
index orders and `..._16dd` at 12:

```
(i, j, k, l, a, b)  (i, j, k, l, b, a)  (i, k, j, l, a, c)
(j, i, k, l, a, b)  (k, j, i, l, c, b)  ...
```

A dropped or inverted permutation at any one of them silently computes a
different tensor while every builder still matches its own spec. That is exactly
the D4 failure shape — all symbolic objects exact, the emitted C++ computing
something else.

`test_emitted_builder_matches_spec` is **not vacuous** under merging (91 builders
checked, 0 bad) but it is **definition-only** and cannot see this. The gate that
can is `python/ccgen/tests/test_merged_call_sites.py`: it emits the residual
**both ways and requires the evaluated arrays to agree**. Merging is a pure
sharing transform, so any misapplied call-site permutation moves the residual
regardless of which operator or term carries it. It runs at ranks 3 and 4 and is
mutation-verified at both:

| mutation | rank | response |
|---|---|---|
| swap free indices at a `t2t2v_oooovv` read (3-cycle) | 3 | RED 4.5e-03 |
| invert a `t2v_ooov` permutation | 3 | RED 8.6e-03 |
| swap last two indices of a `t1t3v_oooovv` read | 3 | RED 1.5e-03 |
| swap two free indices of a `t2t3v_ooooovvv` read | 4 | RED 6.7e-03 |
| swap last two indices of a `t2t2v_oooovv` read | 4 | RED 3.7e-03 |

Rank 3+ is required: every `ccsd` merge permutation is a **self-inverse
two-element swap**, so applying one backwards is undetectable there. Rank 3 has
genuine 3-cycles.

Design rule:

- A definition-only gate (does each builder match its own spec) cannot catch a call-site
  permutation defect. Any merge-sharing transform needs a gate that evaluates the residual both
  ways and compares values.

### 2. A fixture withholding a symmetry the physical object actually has manufactures false failures

The call-site gate's first run went red at rel=6.4e-02 on 17 doubles terms, all `W_t2v_ooov`. The
merge plan maps `a049 -> 85b9` at the **identity** permutation while their definitions differ:

```
85b9:  t2(c,b,l,j) v(i,c,k,l)
a049:  t2(b,c,j,l) v(i,c,k,l)
```

They differ by a transpose of **`t2`**, not of the result — so the merge is valid
exactly when `t2(a,b,i,j) == t2(b,a,j,i)`, which RCC amplitudes really satisfy.
Measured on that pair: **9.3e-02** with a random `t2`, **2.8e-17** with a
symmetric one. The builder gate withholds antisymmetry from `v` deliberately (an
invalid ERI relation must not pass vacuously) and this gate inherited the fixture,
but **withholding a symmetry the physical object HAS manufactures failures**. The
end-to-end evidence agreed with that reading all along: no real permutation defect
survives 62 bitwise-identical iterations.

Design rule:

- Do not use `random_tensors` in any new gate here. It antisymmetrizes `v`, under which invalid
  ERI relations are true; that fixture is why a 41/288 defect passed every symbolic check
  elsewhere. Use fixtures that carry exactly the symmetries the real amplitude/ERI objects have,
  no more and no less.

### 3. A gate must cover every code path the mechanism touches, not just the convenient one

Covering only singles+doubles made the gate blind to the entire point. H5
splits the triples residual across `_partN` chunks accumulating into one shared
`result`, so it is not a single parseable function and the convenient gate stops
at doubles. But **all 56 `t2t2v_oooovv` reads live in those parts, none in
doubles** — the family that is the whole reason to do this. A perturbation moved
the singles+doubles-only gate by **2.2e-16**. The parts sum, so evaluating each
and adding recovers the residual; they carry no `// Term N` markers, so they need
their own splitter.

Design rule:

- Before trusting a gate's coverage, verify the specific mechanism under test actually appears in
  the code paths the gate exercises — not just in the paths that are easiest to parse.

### 4. Rank-4-only emitter conventions can fail silently or with a misleading message

Two rank-4-only emitter conventions, each failing with a message that names neither the rank nor
the cause: rank >= 7 targets use the **braced** runtime-rank accessor,
`result({i, j, k, l, a, b, c, d})`, so a paren capture keeps the braces and numpy rejects the
einsum with "subscripts must be letters". And rank >= 7 builders return **`TensorND`**, not
`Tensor<N>D`, so the shared `build_emitted_operators` cannot see them at all — a `ccsdtq` residual
reads many (`W_t3v_ooooovvv_*`, `W_t1t4v_ooooovvv_*`) and fails as an unrelated-looking
"unknown factor". Rank 4 also needs both `t4` Sz sectors in the fixture (`t4` and
`t4_aaabaaab` are independent; aaab does not reduce to aabb).

Design rule:

- Do not absorb the factorizer's other six knobs into `print_cpp_planck`. `top_k`,
  `savings_fraction`, `memory_budget_bytes`, `max_operator_bytes`, `n_occ`,
  `n_vir` stay out. W3's condition — one parameter, not seven — is what made the earlier W3.3
  deletion possible, and merging was added as a fixed behaviour rather than a seventh knob for the
  same reason.

## What was found

**The estimate was wrong twice, in the same direction, for the same reason:**

| stage | basis | predicted | verdict |
|---|---|---|---|
| original | operator-count FLOP model | 1.02x-1.20x, "compile time, not speed" | too low |
| re-cost | profile-weighted (runtime share x merge ratio) | 1.21x-1.36x | still low |
| **measured** | **two systems, 3 runs each** | **1.42x / 1.52x** | — |

Both models treated operators as **equal-cost**. They are not: the 38-member
`t2t2v_oooovv` family was **20.4 % of runtime by itself**, because it writes a
rank-6 result over a 9-deep nest — and it merges 38 -> 4.

The re-cost additionally applied H5's measured **0.66 realization factor** (H5
predicted 4x on call count and delivered 2.64x, because the builds it eliminated
were cheaper than average). **That discount does not transfer, and assuming it
was not the conservative choice it looked like.** The builds eliminated here are
the *expensive* ones — the opposite selection bias. **A realization factor encodes
which work a particular transform removed; it is not a general haircut to apply to
the next estimate.**

Two framings to avoid, both of which this document made at some point:

- **An operator count is not a speedup.** The 1.4x -> 2.1x -> 3.7x figures in
  `CCGEN_OPERATOR_IDENTITY_AND_REUSE` are counts. Quote the measured 1.42x/1.52x.
- **"Probably compile time, not speed"** was a conclusion drawn from a model, not
  a measurement, and it deferred a 1.5x lever for months.

**The causal story was verified, not just the total.** Leaf-sample attribution on CH4 (`sample`,
~6700 leaf samples per arm):

| family | merge ratio | before | after |
|---|---|---|---|
| `t2t2v_oooovv` | **9.5x** (38 -> 4) | **23.3 %** | **4.1 %** |
| `t1t1t2v_oooovv` | 6.0x (12 -> 2) | 6.6 % | 1.7 % |
| `t1t3v_oooovv` | 1.7x (19 -> 11) | 9.7 % | 9.4 % |

The two families that merge hard collapse; the one that barely merges barely
moves. **That negative control is what separates "the number moved" from "the
number moved for the stated reason"** — a total can improve for an unrelated
reason and look identical in a stopwatch.

At rank 4 the same structure is more extreme: `t2t2v_ooooovvv` 95 -> 1,
`t1t2t2v_ooooovvv` 68 -> 1, `t2t3v_ooooovvv` 313 -> 15.

## What was measured

| | rank 3 (`ccsdt`) | rank 4 (`ccsdtq`) |
|---|---|---|
| distinct builders | 288 -> **91** (3.2x) | 1615 -> **239** (6.8x) |
| TU bytes | 802 K -> 523 K (1.53x) | 11.0 M -> 6.6 M (1.68x) |

**Runtime, measured on the two generated-route cases** (3 runs each, spread
±0.03 s, `PLANCK_RCCSDT_BACKEND=optimized`):

| case | before | after | speedup |
|---|---|---|---|
| `lih_rccsdt_generated_sto3g` | 1.02 s | **0.72 s** | **1.42x** |
| `ch4_rccsdt_sto3g` | 16.65 s | **10.98 s** | **1.52x** |

Energies are **bitwise identical** — LiH agrees on `E_corr`, `dE`, `rms(res)` and
`rms(step)` at every one of 62 iterations; CH4's output diffs empty with timers
stripped — and iteration counts are unchanged (62, 15). So this is per-iteration
work removed, not faster convergence, which is how a builder-dedup should behave.

Compile time and size improve too, though that is the smaller half: the
`-O1`-pinned `generated_kernel_registry.cpp` 11.68 s -> 10.44 s with its object
1.50x smaller, `tensor_backend.cpp` 38.86 s -> 35.80 s, binary 1.11x smaller.

**Incidental, and worth knowing before costing generated-kernel compile time:**
`generated_kernel_registry.cpp` is the `-O1`-pinned TU but **`tensor_backend.cpp`
is the expensive one** — 38.9 s against 11.7 s, because it includes
`ccsdt_planck_generated.cpp` at full `-O3` while the registry is pinned down.
Timing only the pinned TU understates the cost 3x.

## Validation strategy that should remain in place

- `python/ccgen/tests/test_merged_call_sites.py` — evaluates the residual both ways (merged and
  unmerged) and requires agreement; mutation-verified at ranks 3 and 4
- `python/ccgen/tests/test_emitted_builder_matches_spec.py` — the complementary definition-only
  check (91 builders, 0 bad)
- End-to-end regression comparison on `lih_rccsdt_generated_sto3g` and `ch4_rccsdt_sto3g`
  (bitwise-identical energies, unchanged iteration counts)
- Never use `random_tensors` (antisymmetrized `v`) in a gate for this mechanism

## Remaining architecture concern

**A rank-4 end-to-end run.** The counts and the numeric gate are in hand; what has
not been done is building the 6.6 MB dressed TU against the `-O1`-pinned registry
and timing a solve. If one is built, check `be_rccsdtq_sto3g` against
`-14.4036550465` (its manifest value `-14.4036551081` is a known pre-existing
6.2e-08 discrepancy, independent of this work).

Beyond that the ranking in `CCGEN_ARBITRARY_HARNESS_COST.md` puts the remaining
weight on the two triples-residual parts, which no current lever addresses, and on
H6 (OpenMP) — CC is still the only hot path in Planck with no threading.
