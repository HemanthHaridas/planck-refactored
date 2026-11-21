# What makes the generated RCCSDT (`ccsdt_gen`) slower than the hand-written one, and what removes it?

Scope, in progress. Status lives in `vault/Status/Open Work.md`. Related
answers: `docs/CCGEN_WHY_GENERATED_IS_SLOW.md` (causes 1–4),
`docs/CCGEN_ARBITRARY_HARNESS_COST.md`, `docs/CCGEN_CC_OPENMP.md`.

## Current answer

**Factored builder bodies are built and pass every value, energy and
determinism gate. They are not yet timed.** They target the one cause the
ladder found: the gap grows with the occupied space because the builders
were emitted as flat n-ary loop nests.
- **The change.** `print_cpp_planck` passes `factor_builder_bodies=True` on
  the derived route, so each builder is emitted as its contraction tree of
  binary scratch steps (M3.0). The rank-3 TU now carries 41 scratch-step
  tensors.
- **ccgen gates.** Derived and factored residuals equal unfolded, undressed
  ones to ≤ 1e-12, and every emitted builder matches its spec at rank 3 and
  rank 4. Merged call sites agree at both ranks.
  `test_no_nary_statements` finds no statement with three or more factors
  in any residual or builder of `ccsd` and `ccsdt`.
- **Energy gates.** All 9 pass:
  - `ch4_rccsdt_generated_sto3g`, `lih_rccsdt_generated_sto3g`,
    `be_rccsdtq_sto3g`;
  - `water_rccsd_t_{sto3g,631g}`;
  - `b_ucc{2,3,4}_sto3g`, `oh_uccsd_t_631g`.
- **Determinism.** Every iteration's E_corr, dE and rms(res) are identical
  at 1/2/4/8 threads on CH4 and C2H2/STO-3G.
- **Test evaluator.** The numpy evaluator found builder steps by
  `// Definition term` markers, which factored bodies lack, and ignored
  their scratch writes. So it read every factored builder as zero, and the
  first gate run failed at 1.35e-3 for that reason alone. `eval_body`
  (`ccgen/tests/emitted_cpp_eval.py`) now evaluates any body, flat or
  factored, step by step.
- **Pending.** The ladder re-run, to see whether the o^0.78 growth goes
  away. It waits for a full ccgen test run that is still competing for the
  CPU.

**The ladder result that motivated it.** A 9-rung ladder at 4 threads gives
per-iteration fits that hold under leave-one-out validation:

| Per-iteration cost | fit | LOO max / median error |
|---|---|---|
| generated | o^4.30 · v^3.55 | 22 % / 8 % |
| hand-written | o^3.52 · v^3.79 | 12 % / 6 % |
| ratio | o^0.78 · v^−0.24 | 15 % / 5 % |

So the generated route does not scale worse in v; the gap shrinks slightly
with it. It does scale worse in o, by about o^0.8.

**Cause.** On C2H6/STO-3G (o = 9, one thread) the builders take 82 % of the
run, and one family, `W_t1t3v_oooovv`, takes 55 %. Its builder is one flat
nest, t3·t1·⟨oo‖vv⟩, at o⁵v⁴. The factorizer's tree for it is cheaper
(t1·v first, then t3, o⁵v³), but production emits every builder flat.
Builder bodies are supposed to be emitted through `factored_builder_steps`
(M3.0, `docs/CCGEN_INTERMEDIATE_MEMORY_LOCALITY.md` design rule 3). The flag
that does it existed only on `emit_factorized_translation_unit`, which
production never calls. The earlier binary-terms census checked residual
terms only, so it could not see n-ary builders.

**Where it stands at 4 threads** (wall-clock generated / hand-written).
Both give the same E_corr to ≤ 1.5e-9 on every rung.

| Rung | o | v | per-iteration ratio | wall ratio | iterations, gen / hand |
|---|---|---|---|---|---|
| CH4/STO-3G | 5 | 4 | 7.3× | 3.7× | 15 / 24 |
| HF/6-31G | 5 | 6 | 5.8× | 3.6× | 15 / 23 |
| H2O/6-31G | 5 | 8 | 5.5× | 3.1× | 16 / 28 |
| NH3/6-31G | 5 | 10 | 5.7× | 3.2× | 18 / 32 |
| H2O/6-31G* | 5 | 14 | 5.2× | 3.1× | 16 / 27 |
| C2H2/STO-3G | 7 | 5 | 8.4× | 4.7× | 19 / 33 |
| C2H4/STO-3G | 8 | 6 | 8.8× | 3.2× | 15 / 41 |
| C2H6/STO-3G | 9 | 7 | 9.6× | 4.2× | 15 / 34 |

The wall ratio stays near 3–5× for two reasons. The generated route needs
about half as many iterations, and it is threaded while the hand-written
backend runs on one core. On one thread the per-iteration ratio is 18–21×
at o = 5 and 35× on C2H4. Peak RSS is lower for the generated route on every
rung, for example 121 MB against 412 MB on H2O/6-31G*.

**Correction.** An earlier version of this doc gave 3.1× on one thread for
CH4. The hand-written time it used, 0.74 s, came from a cold first run;
warm, it is 0.22 s. On one thread CH4 is 8.7× wall-clock, not 3.1×.

**How the CH4 number got here.** These are end-to-end costs of two
different solvers (dense against wedge-packed amplitudes), not a
codegen-quality ratio (WHY doc, invariant 1).

| Change | 1 thread | 4 threads |
|---|---|---|
| start | 11.0 s | 4.10 s |
| every residual term binary | 3.01 s | 1.84 s |
| builders threaded | 3.39 s | 1.27 s |
| symmetrize once | 2.26 s | 0.82 s |
| hand-written, for reference | 0.22 s | 0.22 s |

Symmetrize-once was measured at 1.50× on one thread and 1.55× on four,
against a modelled 2.67×. E_corr, dE and rms(res) are identical at
1/2/4/8 threads, and 8 threads gain nothing over 4 at CH4 size. The target
(≤ 2× hand-written on every rung, with no growth) is not met.

**Next:** build and time factored builder bodies, then re-run the ladder.
If the o^0.78 growth remains, profile the largest-o rung again.

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
- **No n-ary statements** (`test_no_nary_statements`). It reads the emitted
  text of the production route, residuals and builders, for `ccsd` and
  `ccsdt`. Seeded fingerprints fail it with 39 / 75 n-ary residual
  statements (ccsd / ccsdt). Flat builder bodies fail it with 14 / 38 n-ary
  builder statements.
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

### 0. Ladder (done; re-run after each fix)

The nine rungs are in the table at the top. Three of them are above
0.85 MiB of t3, and four vary o; the first version of the ladder had only
one rung with o ≠ 5, and its joint fit failed leave-one-out at about 97 %.
- Inputs and the runner live in the session scratchpad. Each rung is a
  copy of `ch4_rccsdt_generated_sto3g.hfinp` with `rccsdt_backend
  optimized` or `tensor` and the molecule swapped.
- Generated per-iteration time is the sum of the solver's own `t=`. The
  hand-written backend prints no per-iteration time, so its CC time is its
  wall time minus the generated run's non-CC overhead. Both runs share the
  same SCF and integrals.
- Peak RSS comes from `/usr/bin/time -l`.
- LiH and N2 are left out of the fits. On those, the hand-written run
  converged in its stage-1 warm start and never entered the tensor loop.
- Run `grep '^PLANCK_CC' build/CMakeCache.txt` first (WHY doc,
  invariant 4), and fit only with leave-one-out validation.

### 3. Only if a profile points there

Loop order (the summed index is innermost and strided), the braced rank-4
accessor `t2({..})` in place of a fixed-rank view, and fusion. Fusion was
refuted as a runtime lever once already (cause 2).

### 4. Default routing

A dressed build sends plain `correlation ccsdt` to the generated route by
default. The hand-written backend is 3–5× cheaper wall-clock at 4 threads
across the ladder, and about 9–13× on one thread, at the same energy. Decide whether to keep that default; any change
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
