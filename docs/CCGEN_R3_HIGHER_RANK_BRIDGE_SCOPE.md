# R3.1.2: the higher-rank SpinTerm→AlgebraTerm bridge — two-mechanism scope

**Status: RESOLVED.** Both halves landed; the bridge now reproduces the oracle
per-term (P2.0 green) and whole-residual (`test_rcc_bridge_solve_path_rank6`
green) on the rank-6 CCSDT triples manifold. This note is the finished
diagnosis + fix record.

Companion to `CCGEN_SPIN_ADAPTATION_SCOPE.md` (S3.0 landed the bridge;
`S4a2ArbitraryOrderTests` gates the arbitrary-order pipeline). It scoped the last
blocker on the rank-≥6 spatial path: `spinterm_to_algebraterm` in
`python/ccgen/spin.py` was not spin-faithful, so the *solve* path
(bridge + `residual_einsum` against one spatial block per amplitude) disagreed
with the *oracle* (`_eval_spinterm`, which slices each factor per spin block)
on the CCSDT triples manifold. Whole-residual gap was ~4.8e-3; now ~0.

R3.1.0/R3.1.1 already isolated the blocker to the bridge (the rank-8 identity
`ucc_integrate_term_antisym == GCC slice` passes — the spin summation is right;
what's wrong is the SpinTerm→AlgebraTerm re-encoding). This note is the finished
diagnosis of *why the bridge is wrong*, proven exhaustive.

## The result: the failures partition into exactly two mechanisms

Harness: `S4a2ArbitraryOrderTests` in `python/ccgen/tests/test_spin.py`, rank-6
CCSDT triples, N2/sto-3g via `tests/pyscf/.venv/bin/python`, closed-shell
representative external block `aab` (`a↑ b↑ c↓ i↑ j↑ k↓`). Runs in seconds.

Of **859** merged rank-6 terms, **718** failed per-term bridge==oracle (P2.0)
*before the fix*. Every one was explained by at least one of two mechanisms;
**0 unexplained** — the precondition that made the fix decomposable:

| mechanism | count | fixed by |
|---|---|---|
| both spin + layout | 595 | both halves |
| **layout only** | **116** | half (ii) alone |
| spin only | 7 | half (i) alone |
| unexplained | **0** | — |

After half (ii) alone: 52 failures, all spin-only. After half (i): **0**.

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

**Root cause of the spin surfaces, and the fix (half (i), landed).** On the
rank-6 manifold every failing amplitude factor is `t3` in the `abbabb` block
(1α/2β per half) read as the stored reference `aabaab` (2α/1β). A β-majority
block is *not* a permutation of the α-majority reference — it is its **spin-flip
partner**. A closed-shell amplitude is spin-flip symmetric (`t[σ] = t[flip σ]`
index-for-index), so mapping the β-majority factor onto the stored block is a
two-step slot permutation: flip α↔β, then sort α-before-β. Both halves flip
together (a spin-balanced amplitude has `na_bra == na_ket`). The fix extends
`_canonicalize_amplitude_factor` to detect a β-majority bra half and flip the
sort key; the base (spatial) indices keep their identities and spins as seen by
the rest of the term, so shared/summed indices stay consistent across factors.
This is a pure slot reordering of the single stored block, verified exact
(~4e-18 per term). It clears all 52 spin-only failures.

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

It made the fix decomposable and each half independently gateable. Both landed:

- **(ii) LAYOUT fix (landed first).** Canonicalize the bridge's `free_indices`
  to the external-block order (all bra virtuals `a,b,c` then all ket occupieds
  `i,j,k`, matching `_closed_shell_representative_block`) —
  `free.sort(key=(0 if vir else 1, name))` in `spinterm_to_algebraterm`. Purely
  structural; cleared the 116 layout-only failures on its own (718 → 52, all
  now spin-only), shrinking the search for half (i) to the spin terms.
- **(i) SPIN fix (landed).** Map every amplitude factor onto its single stored
  reference block. `_canonicalize_amplitude_factor` already sorted α-before-β;
  the gap was β-majority blocks (t3 `abbabb`), which are the reference's
  spin-flip partner, not a permutation of it. The fix flips the sort key on a
  β-majority half, so the flip+sort lands on the reference layout. Cleared the
  remaining 52 (52 → 0).

Ordering was the lazy path: half (ii) is small, self-contained, and provably a
pure transpose, so landing it first isolated the residual spin work to a single
crisp mechanism (β-majority t3) rather than three tangled surfaces.

## Gates (all in `S4a2ArbitraryOrderTests`)

| gate | state | asserts |
|---|---|---|
| `test_p20_bridge_matches_eval_per_term_rank6` | **pass** | per-term bridge==oracle; 0 of 859 fail (both halves landed) |
| `test_p21_failures_partition_into_spin_and_layout_rank6` | **pass** | every failure is spin or layout; **0 unexplained** (vacuous now, guards the model) |
| `test_p22_layout_mechanism_is_fixed_rank6` | **pass** | no bridge output is non-canonical; any surviving failure is spin |
| `test_rcc_bridge_solve_path_rank6` | **pass** | whole-residual solve path == GCC slice (was ~4.8e-3, now ~0) |

Classifier helpers on the test class: `_mech_spin`, `_mech_layout`,
`_REF_BLOCK`, `_bridge_output_layout`. P2 must stay exhaustive — if a future
change (a new rank, a new external block, a different reference layout) makes
P2.0 fail again, P2.1 still partitions the new failures into spin/layout, or
reports a third mechanism if the model is incomplete.

## What this reuses / builds on

The S3.0 bridge (`spinterm_to_algebraterm`), the S2 collapse+merge pipeline
(`canonicalize_spin_blocks` → `collapse_amplitudes` → `collapse_integrals` →
`merge_terms`), `ucc_integrate_term_antisym` (the S2 −K exchange fix), and
`residual_einsum` (`ccgen/tests/residual_eval.py`, output
`[ext_vir…, ext_occ…]` first-appearance order — the layout mechanism's origin).
See `CCGEN_SPIN_ADAPTATION_SCOPE.md` S3/S4 and the
`ccgen_r312_bridge_spin_layout` memory note.
