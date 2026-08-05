# R3.1.2: the higher-rank SpinTerm→AlgebraTerm bridge — two-mechanism scope

Companion to `CCGEN_SPIN_ADAPTATION_SCOPE.md` (S3.0 landed the bridge;
`S4a2ArbitraryOrderTests` gates the arbitrary-order pipeline). This note scopes
the last blocker on the rank-≥6 spatial path: `spinterm_to_algebraterm` in
`python/ccgen/spin.py` is not spin-faithful, so the *solve* path
(bridge + `residual_einsum` against one spatial block per amplitude) disagrees
with the *oracle* (`_eval_spinterm`, which slices each factor per spin block)
on the CCSDT triples manifold. Whole-residual gap ~4.8e-3
(`test_rcc_bridge_solve_path_rank6`, xfail).

R3.1.0/R3.1.1 already isolated the blocker to the bridge (the rank-8 identity
`ucc_integrate_term_antisym == GCC slice` passes — the spin summation is right;
what's wrong is the SpinTerm→AlgebraTerm re-encoding). This note is the finished
diagnosis of *why the bridge is wrong*, proven exhaustive.

## The result: the failures partition into exactly two mechanisms

Harness: `S4a2ArbitraryOrderTests` in `python/ccgen/tests/test_spin.py`, rank-6
CCSDT triples, N2/sto-3g via `tests/pyscf/.venv/bin/python`, closed-shell
representative external block `aab` (`a↑ b↑ c↓ i↑ j↑ k↓`). Runs in seconds.

Of **859** merged rank-6 terms, **718** fail per-term bridge==oracle (P2.0).
Every one is explained by at least one of two mechanisms; **0 are unexplained**:

| mechanism | count | fixable by |
|---|---|---|
| both spin + layout | 595 | both halves |
| **layout only** | **116** | half (ii) alone |
| spin only | 7 | half (i) alone |
| unexplained | **0** | — |

### Mechanism 1 — SPIN (`_mech_spin`)

`spinterm_to_algebraterm` builds each factor as `Tensor(name, base_indices)` and
**discards the per-index spin** (`SpinIndex.spin`). `residual_einsum` then reads
each factor from ONE canonical spatial block:

```
t1 → aa    t2 → abab    t3 → aabaab    v → abab    f → aa
```

That single-block read is wrong on three surfaces — **all the same root cause**,
the dropped spin:

- **(a) factor block mismatch** — a factor whose actual per-index spins ≠ its
  canonical block (measured: `t3` surviving in `abbabb`, read as `aabaab`).
- **(b) summed-index cross-block contraction** — an internal index contracted
  across two slots whose canonical-block spins differ, so the spatial einsum
  sums the wrong spin channel. *This alone is the old
  `_has_mixed_spin_summed_index` — one surface, not the mechanism.*
- **(c) free-index slot conflict** — a free (external) index landing on a slot
  whose canonical-block spin ≠ its external spin.

The prior P2.1 tested only surface (b) and correctly disproved that it was the
whole story — but it then mislabeled the *layout* failures below as an
uncharacterized "second spin mechanism." They are not a spin problem at all.

### Mechanism 2 — LAYOUT (`_mech_layout`)

Even when every spin is consistent, the bridge sets `AlgebraTerm.free_indices`
in **first-appearance order** across factors. `residual_einsum` emits its output
as `[ext_vir…, ext_occ…]` in that first-appearance order — which **permutes the
canonical residual axes** `[a,b,c,i,j,k]`.

Witness (a real failing term, both factors cleanly `abab`, all spins consistent):

```
−1 · t2(a,c,i,l) · v(j,k,b,l)      free first-appearance order: a,c,i,j,k,b
                                    → residual_einsum output layout: [a,c,b,i,j,k]
                                    canonical oracle layout:         [a,b,c,i,j,k]
```

The two arrays have equal sums; they differ only by the virtual-block
permutation `(a,c,b) → (a,b,c)`. Transposing the bridge output back to canonical
order reproduces the oracle to **~9e-19** (P2.2). So LAYOUT carries **no numeric
error** — it is a bookkeeping bug in the bridge, entirely independent of spin.

## Why "exactly two, exhaustively" matters

It makes the fix decomposable and each half independently gateable:

- **(i) SPIN fix** — encode per-index spin in `spinterm_to_algebraterm` so the
  emitted `AlgebraTerm` reads/sums the correct spin channel (read the factor's
  own block; make a summed index carry a spin so the spatial contraction is per
  channel; place free indices on the right-spin slots). This is the substantive
  spin-adaptation half.
- **(ii) LAYOUT fix** — canonicalize the bridge's `free_indices` to the
  external-block order (all bra virtuals `a,b,c` then all ket occupieds `i,j,k`,
  matching `_closed_shell_representative_block`). Purely structural; clears the
  116 layout-only failures on its own, with no spin reasoning.

Half (ii) is the lazy first move: it is small, self-contained, provably a pure
transpose, and removes 116 of 718 failures — shrinking mechanism 1's search
space to spin-only terms before the harder spin work starts.

## Gates (all in `S4a2ArbitraryOrderTests`)

| gate | state | asserts |
|---|---|---|
| `test_p20_bridge_matches_eval_per_term_rank6` | xfail | per-term bridge==oracle; red until BOTH halves land |
| `test_p21_failures_partition_into_spin_and_layout_rank6` | **pass** | every failure is spin or layout; **0 unexplained** |
| `test_p22_layout_only_failures_are_a_pure_transpose_rank6` | **pass** | layout-only failures = canonical transpose to ~1e-9 |
| `test_rcc_bridge_solve_path_rank6` | xfail | whole-residual solve path == GCC slice (~4.8e-3); green when both halves land |

Classifier helpers on the test class: `_mech_spin`, `_mech_layout`,
`_REF_BLOCK`, `_bridge_output_layout`. P2 must stay exhaustive — if a future
change makes P2.1 report unexplained failures, a third mechanism has appeared
and the fix model is incomplete.

## What this reuses / builds on

The S3.0 bridge (`spinterm_to_algebraterm`), the S2 collapse+merge pipeline
(`canonicalize_spin_blocks` → `collapse_amplitudes` → `collapse_integrals` →
`merge_terms`), `ucc_integrate_term_antisym` (the S2 −K exchange fix), and
`residual_einsum` (`ccgen/tests/residual_eval.py`, output
`[ext_vir…, ext_occ…]` first-appearance order — the layout mechanism's origin).
See `CCGEN_SPIN_ADAPTATION_SCOPE.md` S3/S4 and the
`ccgen_r312_bridge_spin_layout` memory note.
