# R3.1.2: the higher-rank SpinTerm→AlgebraTerm bridge — two-mechanism scope

**Status: rank-6 RESOLVED; rank-8 (t4) has a further gap — see R3.1.3 below.**
Both rank-6 halves landed; the bridge reproduces the oracle per-term (P2.0
green) and whole-residual (`test_rcc_bridge_solve_path_rank6` green) on the
rank-6 CCSDT triples manifold. At rank 8 a distinct, larger gap remains (t4's
independent Sz sectors), pinned by `test_rank8_bridge_solve_path` (xfail). This
note is the rank-6 diagnosis + fix record, then the rank-8 scope.

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
  with `free.sort(key=(space_rank, name))` in `spinterm_to_algebraterm`. Purely
  structural; cleared the 116 layout-only failures on its own (718 → 52, all
  now spin-only), shrinking the search for half (i) to the spin terms.

  **The load-bearing part is the name-sort WITHIN each space** — first-appearance
  order differs between terms with the same externals (`a,c,b` vs `a,b,c`),
  transposing their residual arrays so a manifold sum is wrong; name-sort makes
  every term agree. The occ-vs-vir **between-space** order is **occ-first**
  (`space_rank = 0 if occ else 1`), matching the C++ runtime's amplitude /
  denominator / residual layout (`rank_dims`: `t_r(i1..ir, a1..ar)`, occupied
  first) so the emitted spatial kernel's result buffer lines up with the runtime
  **without a transpose**. This between-space choice is provably invariant for the
  other consumers — `residual_einsum` re-splits by space internally (always
  `[vir, occ]` output) and the P2 gates' `_bridge_output_layout` re-derives
  `vir+occ` — so only the C++ path cares, and it needs occ-first. (An earlier
  draft sorted vir-first to cosmetically match the oracle's `[a,b,c,i,j,k]`
  printout; that emitted a transposed residual and broke the runtime with a
  rank-1 shape mismatch. Contract audit: occ-first is the one hard contract.)
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

---

## R3.1.3: the rank-8 (t4) gap — t4's independent Sz sectors

The rank-6 fix is exact, but the rank-8 CCSDTQ **quadruples** solve path is still
wrong (~5e-4, `test_rank8_bridge_solve_path`, xfail). This is the defect the Be
CCSDTQ run caught: the generated arbitrary-order kernel gives a spin-orbital
answer against spatial storage, so CCSDTQ misses FCI (Be `-0.0517714927` vs the
required `~1e-8` match). Rank 6 was necessary but not sufficient.

**Root cause.** After `canonicalize_spin_blocks`, the merged quadruples reference
**three** distinct t4 blocks: `aabbaabb` (the reference, 2α2β per half, ~358
occurrences), `aaabaaab` (3α1β, ~48), and `abbbabbb` (1α3β, ~48). `aaab` and
`abbb` differ from the reference in their **α-count per half** (Sz sector), so
they are **not** a permutation and **not** a spin-flip of `aabb` — the two
transforms the rank-6 fix (`_canonicalize_amplitude_factor`) applies. All 140
per-term rank-8 failures read a t4 factor in `aaab`/`abbb` from the single stored
`aabbaabb` block. This is real, not a fixture artifact: the bridge reads a
nonzero (`~1e-3`) block where the true `aaab`/`abbb` value is `~1e-18` (the
4-electron fixture zeros the higher-Sz sectors, but the *bridge injects nonzero
error regardless*), so the solve residual is wrong and the Be solve cannot reach
FCI.

Why rank 6 was clean: t3 has only two blocks per half — `aab` (2α1β, reference)
and `abb` (1α2β) — and `abb` **is** the reference's spin-flip partner (same |Sz|,
opposite sign), so the rank-6 flip closed it. t4 first exposes a third block with
a genuinely different Sz.

### The reduction does not exist — the sectors are independent (measured)

The tempting fix, "reduce `aaab`/`abbb` to the reference `aabb`", is **impossible**.
A least-squares fit of `t4[aaab]` from all 4!·4! signed bra/ket permutations of
`t4[aabb]` leaves residual `0.69 / 0.74` (norm), and even when both blocks are
built from one shared spatial `tau`, `0.196 / 0.253`. `aaab` (Sz=3 per half) and
`aabb` (Sz=2) are **genuinely independent Sz sectors** — `aabb` alone does not
carry the full t4 amplitude. This is not unfinished algebra to derive; it is a
storage-model fact, and it matches how production RCC codes hold t4 (multiple
spin cases, not one).

The full block census, per amplitude (Sz = α-count per half):

| amp | blocks (Sz) | independent set | maps to reference by |
|---|---|---|---|
| t2 | `abab` (1) | 1 | — |
| t3 | `aabaab` (2), `abbabb` (1) | 1 | `abbabb`→`aabaab` **spin-flip** (rank-6 fix) |
| t4 | `aabbaabb` (2), `aaabaaab` (3), `abbbabbb` (1) | **2** | `abbb`→`aaab` spin-flip; `aaab` **independent** of `aabb` |

So t4 needs **two** stored spatial blocks: `aabbaabb` (the balanced reference) and
`aaabaaab` (the Sz-max sector). `abbbabbb` folds onto `aaabaaab` by the existing
rank-6 spin-flip (they are flip partners, both Sz≠2). The rank-6 fix already
collapses 3 t4 blocks to these 2; what's missing is that the **second** block is
read from the reference instead of its own storage.

### The fix, in small verifiable steps (R3.1.3) — LANDED a–d

Everything gated on `test_rank8_bridge_solve_path` (~30s). The last step's Be
CCSDTQ confirmation (`GeneratedCcsdtqFciGate`, ~15min) is the only slow gate.

- **R3.1.3a — enumerate the independent block set per amplitude rank. LANDED.**
  `independent_spin_blocks(rank)` (the reference, α-before-β ⌈n/2⌉ α per half,
  plus each higher-Sz sector up to `n-1`; all-α `k=n` is split by
  `collapse_amplitudes`, so excluded) and `_amplitude_block_tag(block)` (folds any
  block to its sector via the spin-flip pairing `k↔n-k`). t3 both blocks → one
  component; t4 → two (`aabbaabb`, `aaabaaab`). *Gates:*
  `R313IndependentBlocksTests` — low-rank enumeration, flip-partner folding, and
  a completeness check (every merged rank-6/8 census block folds into the set).

- **R3.1.3b — the canonicalizer already carries the sector. LANDED (no change).**
  `_canonicalize_amplitude_factor`'s existing spin-flip+sort reorders the base
  indices + folds the sign so reading the *tagged* block's tensor is exact
  (verified ~7e-17 for `abbb` read from the stored `aaab` block). The only
  addition needed was to expose the tag (R3.1.3a's `_amplitude_block_tag`), not a
  new signature — the reordering itself was already correct.

- **R3.1.3c — the bridge names the sector. LANDED.** `spinterm_to_algebraterm`
  emits `Tensor(name, …)` for the reference sector and `Tensor(f"{name}_{tag}", …)`
  for a higher sector (guarded by `len(block) >= 8`, so t1/t2/t3 are byte-
  identical). *Gate:* `test_rank8_bridge_solve_path` now supplies `t4` **and**
  `t4_aaabaaab` and passes (xfail → pass; the whole-residual solve == GCC aabb
  slice to ~7e-17). The rank-6 bridge and S3.2 energy gates stay green.

- **R3.1.3d — `spin_adapt_equations` emits a residual per stored block. LANDED.**
  For each target it loops the independent Sz sectors (`_representative_block_for_sector`
  builds each sector's external) and keys the output `target` (reference) /
  `target_tag` (e.g. `quadruples_aaabaaab`). n≤3 targets have one sector → keys
  unchanged (byte-identical for ccsd/ccsdt). *Gates:*
  `test_representative_block_for_sector`, `test_ccsd_ccsdt_adapt_keys_unchanged`,
  and `test_ccsdtq_adapt_emits_both_t4_sectors` (both quadruples blocks non-empty,
  ~5s — the diagram engine caches, so this is fast, not the old ~10min estimate).
  *Final confirmation (deferred, slow):* the Be CCSDTQ solve
  (`GeneratedCcsdtqFciGate`) reaching FCI to 1e-8 — needs the solver wired to
  update both t4 blocks from their two residuals (the emit/solver-integration
  step), run once, not in the loop.

### Generalize to arbitrary order (R3.1.4) — LANDED structurally

Every R3.1.3 step is rank-agnostic; the only rank-specific input is the *choice*
of independent set, `independent_spin_blocks(rank)`, which already generalizes.
The rule: the independent blocks of a rank-2n amplitude are its distinct **Sz
sectors** `k = ⌈n/2⌉ … n−1` (α-count per half), one α-before-β representative
each — spin-flip pairs `k ↔ n−k` (folding the low-Sz half up), permutations
exhaust each sector, and all-α `k=n` is removed by `collapse_amplitudes`. So the
count is **⌊n/2⌋** blocks:

| amp | n | independent blocks |
|---|---|---|
| t2 | 2 | 1 — `abab` |
| t3 | 3 | 1 — `aabaab` |
| t4 | 4 | 2 — `aabbaabb`, `aaabaaab` |
| t5 | 5 | 2 — `aaabbaaabb`, `aaaabaaaab` |
| t6 | 6 | 3 — `aaabbb…`, `aaaabb…`, `aaaaab…` |

`spinterm_to_algebraterm` (R3.1.3c) and `spin_adapt_equations` (R3.1.3d) both key
off `independent_spin_blocks`/`_amplitude_block_tag` with no per-rank code, so a
rank-10 t5 residual would automatically emit `quintuples` + `quintuples_aaaabaaaab`.
*Evidence:* the rank-8 solve-path pass plus the pure-function enumeration
(`test_independent_blocks_low_ranks` asserts through t5) — the same structural
generalization `_split_same_spin_amplitude` relied on after its n=2,3 numeric
pins. A rank-10 numeric solve-path gate is deferred only for want of a tractable
t5 fixture, not for any code gap.

*Standing gates:* `test_rank8_bridge_solve_path` (green, the whole-residual
proof); `R313IndependentBlocksTests` (enumeration/fold/keys, seconds);
`test_rank8_full_collapse_pipeline` and `test_rank8_aabb_identity` stay green
throughout (per-block collapse was already correct — the gap was only that the
second independent block wasn't stored/read).

### What remains (out of this note's scope)

The Python spatial residual is now exact and multi-sector. The last mile to the
Be CCSDTQ number is **emit + solver integration**: emit the tagged residual set
to C++, allocate both t4 blocks in the runtime, and update each from its own
residual. That is the `spin_adapt`-into-codegen wiring tracked in
`generated_ccsdtq_energy_wrong` (the two-gap note) — the algebra it needs is now
complete.
