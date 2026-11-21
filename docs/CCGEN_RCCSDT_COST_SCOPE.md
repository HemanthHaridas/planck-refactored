# What makes the generated RCCSDT (`ccsdt_gen`) slower than the hand-written one, and what removes it?

Scope, in progress. Status lives in `vault/Status/Open Work.md`. Related
answers: `docs/CCGEN_WHY_GENERATED_IS_SLOW.md` (causes 1–4),
`docs/CCGEN_ARBITRARY_HARNESS_COST.md`, `docs/CCGEN_CC_OPENMP.md`.

## Current answer

On CH4/STO-3G (o = 5, v = 4) the generated route costs **3.7× the
hand-written one at 4 threads and 3.1× on one.** Both give
E_corr = −0.0791116946.

| | 1 thread | 2 | 4 | 8 | iterations |
|---|---|---|---|---|---|
| generated (`rccsdt_backend optimized`) | 2.25–2.27 s | 1.26 s | 0.82 s | 0.81–0.82 s | 15 |
| hand-written (`rccsdt_backend tensor`) | 0.74 s | | 0.22 s | | 24 |

The latest change is symmetrize-once. The spin-adapted triples residual is
folded to one term per symmetry orbit (806 nests to 273), and the kernel
rebuilds the full residual in one o³v³ pass. Measured, it is 1.50× on one
thread and 1.55× on four, against a modelled 2.67×. E_corr, dE and
rms(res) are identical at 1/2/4/8 threads, and 8 threads gain nothing over
4 at this size.

This is the end-to-end cost of two different solvers (dense against
wedge-packed amplitudes), not a codegen-quality ratio (WHY doc,
invariant 1). It started at 11.0 s / 4.10 s (1 / 4 threads):

| Change | 1 thread | 4 threads |
|---|---|---|
| start | 11.0 s | 4.10 s |
| every residual term binary | 3.01 s | 1.84 s |
| builders threaded | 3.39 s | 1.27 s |
| symmetrize once | 2.26 s | 0.82 s |

Overall that is 4.9× on one thread and 5.0× on four. The target (≤ 2×
hand-written) is not met on CH4.

**Next:** the ladder (step 0). It decides whether the remaining 3–4× is a
constant or still grows with size, and a profile on it decides what step 3
takes on.

## Symmetrize once

**The symmetry the fold uses.** The spin-adapted t3 is the aabaab block:
(ia) and (jb) are alpha, (kc) is beta. Its residual is antisymmetric under
a↔b and under i↔j, and symmetric under both. That was measured to ≤ 4e-15
with an aab-correct t3. Nothing relates (kc) to the other two pairs. So
the group has order 4, `{e, (ab)⁻, (ij)⁻, (ab)(ij)⁺}`
(`AAB_TRIPLES_GROUP`). An earlier census assumed all six pair
permutations and gave 1.62×; that is the wrong group.

**The construction.** Write R = Σ_T c_T·T. Then |G|·R = Σ_g s_g·g(R). For
T = h(rep), Σ_g s_g·g(T) = s_h·Σ_g s_g·g(rep). So R = Σ_g s_g·g(X), with
X = (1/|G|) Σ_orbits κ_O·rep_O and κ_O = Σ_{T∈O} c_T·s_h. This is exact
whenever R has the symmetry. It does not need the terms to come in
complete orbits: orbits are matched on the term's text, which can only
find too few, never a wrong one. Because R is then exactly antisymmetric,
t3 stays antisymmetric through the iterations.
- **Fold.** `fold_external_symmetry`, `ccgen/optimization/permutation.py`.
- **When.** It runs in `print_cpp_planck` on spin-adapted `triples`, after
  spin adaptation and before factorization, so no operator is built for a
  folded-away term. The rank-4 TU's triples kernel is folded too.
- **Pass.** `_symmetrize_lines` in `ccgen/emit/planck_tensor_cpp.py` emits
  one threaded pass over a copy of X, before `return result;`, in both the
  inline and the chunked kernel. Its `// symmetrize ijkabc: ...` comment
  is what the numpy evaluator reads, so the value gates apply the same pass
  as the C++.

## Every residual term binary

**112 residual terms were evaluated as three-tensor products.** That is 68
triples terms, 33 doubles and 11 singles of spin-adapted `ccsdt`. The worst
were 25 t2·t3·⟨oo‖vv⟩ terms at o⁵v⁵. The triples ones carried about 74 % of
the modelled FLOPs, and they were the scaling defect against the
hand-written path's o³v⁵. Before the fix the triples residual took 90 % of a
1-thread profile; after it, 50 %, with the builders at 48 %.

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
keep the seeded default, since reporting reuse is their purpose. This
corrects the WHY doc's record that cause 1 was fixed: it was fixed only for
terms whose sub-contractions matched no seeded operator.

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

## Threaded builders

The emitter gave the residual nests a `collapse(3)` pragma and the builders
none. So after the binary-terms fix, the builders took 48 % of a 1-thread
run and four threads gained only 1.6×.

Builder nests now get the same pragma, 92 of them
(`_emit_intermediate_builder`). It is safe for the reason the residual
pragma is: each nest writes disjoint elements of its output, and each step
of a factored builder is its own parallel region, so the barrier orders a
scratch write before the next step reads it.

**It costs 12 % on one thread**, all in the builders (about 1.45 s became
2.0 s). The likely cause is that the outlined OpenMP bodies optimize worse;
that is unconfirmed. Kept because production runs threaded. If it matters,
put the pragma only on nests with enough work, and recheck on the ladder.

## Gates

All in `ccgen/tests/test_factorize_all_binary.py` unless named.
- **Value.** Singles, doubles and triples from the derived, folded emit
  equal the undressed, unfolded emit to ≤ 1e-12 relative. The two arms
  share no operators and no fold. The fixture is the physical one for this
  manifold: an aab-block t3 and full real-orbital ERI symmetry.
  - Mutations: a flipped operator-read sign fails at 1.1e-2, and a flipped
    group sign at 1.0 relative.
- **No n-ary terms.** No emitted residual term has three or more factors.
  The old default fails it with 112.
- **The fold is applied.** The folded kernel has under half the triples
  nests and carries the symmetrize pass. The unfolded arm has neither.
- **Operator-count pins.** Seven ccgen pins taken from the seeded default
  are restated with dated notes. `HigherRankMergeTests` asserts counts
  before its value loop, so its value checks were re-run at the new counts.
  Rank 3 covers more than 300 rewritten terms and rank 4 more than 2000,
  all reproducing their source to 1e-10.
- **Energy, with all three changes.**
  - Generated RCC: `ch4_rccsdt_generated_sto3g`, `lih_rccsdt_generated_sto3g`
    and `be_rccsdtq_sto3g` (its triples kernel is folded too).
  - CCSD(T): `water_rccsd_t_sto3g` and `water_rccsd_t_631g`.
  - UCC, which the fold does not touch: `b_ucc2_sto3g`, `b_ucc3_sto3g`,
    `b_ucc4_sto3g` and `oh_uccsd_t_631g`.

  `ch4_rccsdt_sto3g` fails as one of the four known routing failures
  (step 4).
- **Determinism.** E_corr, dE and rms(res) are identical at 1/2/4/8 threads
  on CH4.
- **Other ccgen gates, with the fold.** `test_merged_call_sites` (ccsdt and
  ccsdtq, merged against unmerged) and `test_emitted_builder_matches_spec`
  pass: 5 tests.

## Target

Generated end-to-end ≤ 2× hand-written on every ladder rung, with no rung
where the ratio grows with size. The scaling clause matters more than the
constant.

## Remaining steps

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
default. The hand-written backend is 3.1–3.7× cheaper on CH4, at the same
energy. Decide whether to keep that default; any change
must be checked against the four known rccsdt routing failures.

## Rank 2 and rank 4

- **Rank 2, binary terms.** The binary-terms fix also covers the 44 n-ary
  singles and doubles terms that rank 2 shares.
- **Rank 2, accuracy.** The generated RCCSD (`ccsd(t)_gen`) equals the
  hand-written RCCSD(T) to ≤ 3e-10 Eh on all four `*_rccsd_t_*` inputs.
- **Rank 2, cost.** Before either fix it took 55.4 s against 32.1 s on C1
  H2O2/6-31G, and 19.6 s against 14.8 s on spherical water/cc-pVDZ. Re-time
  both.
- **Rank 2, fold.** Its doubles residual is symmetric under (ia)(jb)
  (5.6e-16), an order-2 group the fold could use.
- **Rank 4.** Census the ccsdtq TU before claiming anything there.

## Ruled out

- **Factorizing before spin adaptation.** `print_cpp_planck` already
  factorizes after `spin_adapt_equations`. The ternary terms came from
  seeded-fingerprint inlining, not from ordering.
- **All six pair permutations as the triples symmetry.** The aabaab
  residual is not symmetric under any permutation that moves (kc), so
  folding over them would be wrong.
- **The harness.** It takes ~1 % of the time
  (`docs/CCGEN_ARBITRARY_HARNESS_COST.md`).

## Not in this scope

BLAS/GEMM lowering of the generated nests, RI, MPI, and the UCC kernels.
The UCC kernels take the same steps once rank-3 RCC passes.
