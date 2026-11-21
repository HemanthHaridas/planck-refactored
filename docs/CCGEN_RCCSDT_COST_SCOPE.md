# What makes the generated RCCSDT (`ccsdt_gen`) slower than the hand-written one, and what removes it?

Scope, in progress. Status lives in `vault/Status/Open Work.md`. Related
answers: `docs/CCGEN_WHY_GENERATED_IS_SLOW.md` (causes 1–4),
`docs/CCGEN_ARBITRARY_HARNESS_COST.md`, `docs/CCGEN_CC_OPENMP.md`.

## Current answer

On CH4/STO-3G (o = 5, v = 4) the generated route costs **5.8× the
hand-written one at 4 threads and 4.6× on one.** Both give
E_corr = −0.0791116946.

| | 1 thread | 2 | 4 | 8 | iterations |
|---|---|---|---|---|---|
| generated (`rccsdt_backend optimized`) | 3.39 s | 1.96 s | 1.27–1.33 s | 1.34 s | 15 |
| hand-written (`rccsdt_backend tensor`) | 0.74 s | | 0.22 s | | 24 |

This is the end-to-end cost of two different solvers (dense against
wedge-packed amplitudes), not a codegen-quality ratio (WHY doc,
invariant 1). It went from 11.0 s / 4.10 s (1 / 4 threads) in two changes:
- **Binary terms.** Every residual term is now a binary contraction (next
  section): 3.01 s / 1.84 s.
- **Threaded builders.** The intermediate builders are now threaded:
  3.39 s / 1.27 s.

**Threaded builders.** The emitter gave the residual nests a `collapse(3)`
pragma and the builders none, so after the binary-terms fix the builders
took 48 % of a 1-thread run and four threads gained only 1.6×. Builder
nests now get the same pragma (`_emit_intermediate_builder`,
`ccgen/emit/planck_tensor_cpp.py`), 92 of them. It is safe for the reason
the residual pragma is: each nest writes disjoint elements of its output,
and each step of a factored builder is its own parallel region, so the
barrier orders a scratch write before the next step reads it.
- **Determinism.** E_corr, dE and rms(res) are identical at 1/2/4/8 threads.
- **Gates.** `ch4_rccsdt_generated_sto3g`, `lih_rccsdt_generated_sto3g`,
  `be_rccsdtq_sto3g`, `water_rccsd_t_{sto3g,631g}`, `b_ucc2_sto3g` and
  `oh_uccsd_t_631g` pass. The UCC builders got the pragma too.
- **Cost on one thread: 12 %.** 3.01 s became 3.39 s, all of it in the
  builders (about 1.45 s became 2.0 s). The residual did not change. Likely
  cause, unconfirmed: the outlined OpenMP bodies optimize worse. Kept,
  because production runs threaded. If it matters, put the pragma only on
  builder nests with enough work, and recheck on the ladder. CH4 is small
  enough that per-nest overhead is at its largest.

**Next:** symmetrize once (step 2), then the ladder (step 0).

## What was wrong, and why it stayed hidden

**112 residual terms were evaluated as three-tensor products.** That is 68
triples terms, 33 doubles and 11 singles of spin-adapted `ccsdt`. The worst
were 25 t2·t3·⟨oo‖vv⟩ terms at o⁵v⁵. The triples ones carried about 74 % of
the modelled FLOPs, and they were the scaling defect against the
hand-written path's o³v⁵. Removing them took the 1-thread time from 11.0 s
to 3.01 s (3.66×), and the 4-thread time from 4.10 s to 1.84 s. Before the
fix the triples residual parts 0 and 1 took 90 % of a 1-thread profile; after it,
50 %, with the builders at 48 %.

The derived route's emit-path helpers (`emittable_operators`,
`rewrite_term_factorized`, `manifold_operators`) defaulted to
`seeded_fingerprints()`, the six Stanton–Gauss operators of the retired
recognition route:
- **Match.** A sub-contraction that matched one of them (Wmbej, Wmnij, Fae,
  Fmi) was classified as `Reuse`.
- **Inline.** `Reuse` children are always inlined, from when recognition was
  expected to dress them.
- **No builder.** The derived route emits no `Reuse` builders
  (`include_reuse=False`), so nothing built those operators, and the term
  stayed n-ary.

Every sub-contraction is now derived from its own contraction tree:
`EMIT_FINGERPRINTS = ()` in `python/ccgen/optimization/factorize.py`. The
analysis helpers (`identify_tree`, `value_operators`, `reused_operators`)
keep the seeded default, since reporting reuse is their purpose.

This contradicts the WHY doc's record that cause 1 was fixed. It was fixed
for terms whose sub-contractions matched no seeded operator. The WHY doc now
carries the correction.

**Operator counts rose accordingly.** Each seeded operator is a sum of
sub-contractions, and the derived route builds one array per contraction
shape: a shared array across shapes computed the wrong tensor (D6). Spin
adaptation also splits each term into direct and exchange integral variants.

| Manifold | Before | After |
|---|---|---|
| ccsd doubles, GCC | 27 | 38 (+11: Wmbej 4, Fme 3, Wmnij 2, Fmi 1, Fae 1) |
| ccsd doubles, spatial | 59 | 81 |
| ccsdt triples | 80 | 94 |
| ccsdtq quadruples | 254 | 268 |
| spatial `ccsdt` emitted builders | 91 | 104 |

## Gates

- **No n-ary terms** (`ccgen/tests/test_factorize_all_binary.py`). No
  emitted residual term has three or more factors. The old default fails it
  with 112.
- **Values unchanged** (same file). Singles, doubles and triples equal the
  undressed spin-adapted emit, which shares no operators, to ≤ 6.5e-16
  relative. One flipped sign on an operator read fails it at 1.1e-2.
- **Operator-count pins.** Seven ccgen pins taken from the seeded default
  are restated with dated notes. The count assertion in
  `HigherRankMergeTests` comes before its value loop, so the value checks
  were re-run at the new counts. Rank 3 covers more than 300 rewritten
  terms and rank 4 more than 2000; every rewritten term reproduces its
  source to 1e-10.
- **Energy.** `ch4_rccsdt_generated_sto3g`, `lih_rccsdt_generated_sto3g`,
  `be_rccsdtq_sto3g`, `water_rccsd_t_sto3g` and `water_rccsd_t_631g` pass.
  `ch4_rccsdt_sto3g` fails as one of the four known routing failures
  (step 4).
- **Determinism.** The regenerated kernels are byte-identical whether the
  empty fingerprint set comes from the default or an explicit argument.

## Target

Generated end-to-end ≤ 2× hand-written on every ladder rung, with no rung
where the ratio grows with size. The scaling clause matters more than the
constant.

## Remaining steps

### 2. Symmetrize once

The census of the 806 triples nests finds only 450 distinct under the
closed-shell pair permutations (ia)(jb)(kc). That is a 1.62× cut in
modelled FLOPs, and a lower bound, since only external indices were
renamed. ccgen's `permutation_grouping` merges only terms with equal
values. Nothing computes one representative per orbit, which the
hand-written kernel's wedge storage gets for free.

The plan: emit one nest per orbit into a scratch X(ijk,abc), then
R += Σ_σ sign·σ(X), an o³v³ pass.
- **Premise first.** Measure that the spin-adapted generated residual has
  the pair-permutation symmetry. Its convention differed from the
  hand-written one before (`docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md`).
- **Gates.** The value and energy gates above, and bitwise-identical output
  at 1/2/4/8 threads.

### 0. Ladder

The rungs are LiH/STO-3G, CH4/STO-3G, H2O/6-31G, HF/6-31G, and one rung
above 0.85 MiB of t3 (every earlier ladder stayed inside L2). At 1 and 4
threads, record seconds per iteration, iteration count and peak RSS.
- Run `grep '^PLANCK_CC' build/CMakeCache.txt` first (WHY doc,
  invariant 4).
- Fit exponents only with leave-one-out validation.

The ladder decides whether the remaining gap is a constant or still grows
with size.

### 3. Only if a profile points there

Loop order (the summed index is innermost and strided), the braced rank-4
accessor `t2({..})` in place of a fixed-rank view, and fusion. Fusion was
refuted as a runtime lever once already (cause 2).

### 4. Default routing

A dressed build sends plain `correlation ccsdt` to the generated route by
default. The hand-written backend is still 4.6–5.8× cheaper on CH4, at the same
energy. Decide whether to keep that default; any change must be checked
against the four known rccsdt routing failures.

## Rank 2 and rank 4

The same fix covers the 44 n-ary singles and doubles terms that rank 2
shares. The generated RCCSD (`ccsd(t)_gen`) equals the hand-written RCCSD(T)
to ≤ 3e-10 Eh on all four `*_rccsd_t_*` inputs. Before the fix it took 55.4
against 32.1 s on C1 H2O2/6-31G, and 19.6 against 14.8 s on spherical
water/cc-pVDZ. Re-time both. Census the ccsdtq TU before claiming rank 4.

## Ruled out

- **Factorizing before spin adaptation.** `print_cpp_planck` already
  factorizes after `spin_adapt_equations`. The ternary terms came from
  seeded-fingerprint inlining, not from ordering.
- **The harness.** It takes ~1 % of the time
  (`docs/CCGEN_ARBITRARY_HARNESS_COST.md`).

## Not in this scope

BLAS/GEMM lowering of the generated nests, RI, MPI, and the UCC kernels.
The UCC kernels take the same steps once rank-3 RCC passes.
