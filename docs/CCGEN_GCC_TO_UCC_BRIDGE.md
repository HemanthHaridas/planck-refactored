# GCC-to-UCC Spin-Block Bridge

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**How does a spin-orbital (GCC) CC manifold become a spin-block-resolved (UCC) one, and what makes that different from the spatial collapse the restricted path uses?**

## Short answer

**UCC is the collapse that isn't.** `spin_adapt_equations` folds spin blocks into one spatial tensor per rank; `ucc_adapt_equations` runs the same integration and then **stops**, keeping `t2_aaaa`, `t2_abab`, `t2_bbbb` as distinct arrays. The work is not in adding machinery — it is in finding every place the restricted path quietly assumed that folding was safe.

> **The name reads backwards, and that is inherited.** In this repo "adapt" means
> `spin_adapt_equations`, the **spatial collapse**. `ucc_adapt_equations` is the entry that *skips*
> it. Prefer "resolve" when writing new code; the existing names are kept for continuity.

Validated at rank 4 (~6e-16 vs PySCF UCCSD) and rank 6 (1.6e-17 vs GCC-sliced, with GCC itself
FCI-exact). **How that validation is constructed is a separate question, answered in
`CCGEN_UCC_NUMERIC_VALIDATION.md`** — read that before touching the numeric side.

## Where the logic lives

- `spin.py` — `ucc_adapt_equations`, `ucc_spinterm_to_algebraterm` (the bridge entry)
- `spin.py` — `ucc_independent_blocks`, `_ucc_block_tag`, `external_blocks(fold_spin_flip=…)` (block vocabulary)
- `ucc_integrate_term_antisym` → `_antisym_to_allowed` (shared integration)
- `_split_same_spin_amplitude` (the rank-6 closure relation)
- The gate that mattered: `U14c3UccIsGccSlicedAtRankSixTests` — asserts the UCC manifold *is* the GCC one sliced into spin blocks (1.6e-17 on a perturbed spin-orbital t3)

## What the bridge actually has to do

The GCC manifold carries per-index spin. The restricted bridge drops it, because after the spatial
collapse there is nothing to route. UCC needs it kept, which means exactly two things:

1. **Block-resolved factor names.** A factor becomes `t2_abab` / `v_aaaa` / `f_bb` rather than
   `t2` / `v` / `f`. This is the correctness half — everything downstream (ERI routing, denominator
   selection, loop bounds, result shapes) reads the block off the name.
2. **No spin-flip folding.** Every place the restricted path maps a β-majority block onto its
   α-majority partner has to be either skipped or given a UCC-correct replacement.

Nothing else changes. The integration itself (`ucc_integrate_term_antisym`) is shared, and the
orientation-invariance fix that normalizes each rank-4 `v` to one canonical member of its 8-fold
ERI orbit is inherited for free.

## What invariants matter

### 1. β-majority folding is safe at rank 4 and unsafe at rank 6

The restricted canonicalizer sorts amplitude slots toward one reference layout per rank. **At rank
4 that is block-local and harmless:**

```
aaaa -> +1  aa ba ia ja      abab -> +1  aa bb ia jb
baba -> +1  ba ab ja ib      bbbb -> +1  ab bb ib jb
```

`bbbb` stays `bbbb`; `baba` correctly reorders onto the `abab` layout. **At rank 6 it is not:**

| block | canonicalized slots | RCC tag | UCC tag |
|---|---|---|---|
| `aaaaaa` | `aaaaaa` | `aaaaaa` | `aaaaaa` |
| `aabaab` | `aabaab` | `aabaab` | `aabaab` |
| `abbabb` | **`bbabba`** | **`aabaab`** | **`abbabb`** |
| `bbbbbb` | `bbbbbb` | **`aaaaaa`** | `bbbbbb` |

Two distinct disagreements for β-majority blocks: the RCC tag folds them onto their spin-flip
partner (valid only when α ≡ β), *and* the canonicalized slot order (`bbabba`) is not the UCC tag's
own layout either. **A tag and a slot order that disagree is the worst failure mode in this
codebase** — a factor indexes the wrong slice and the residual comes out near-zero rather than
obviously broken.

**The resolution was to design the hazard out, not to retarget the canonicalizer.** The UCC bridge
simply never calls it, and `_ucc_block_tag` supplies the correct tag directly. That is why the
scope step written to "retarget the canonicalizer per block" turned out to be dead work: a route
the analysis had not considered removed the need for it entirely.

Design rule:

- Do not use `_amplitude_block_tag` on the UCC path. It folds β-majority onto α-majority
  (`abbabb` → `aabaab`, `bbbbbb` → `aaaaaa`), valid only when α ≡ β. `_ucc_block_tag` is the
  UCC-correct one.
- Do not assume the rank-4 canonicalizer behaviour generalizes. It is block-local at rank 4 and
  not at rank 6 — measured, and still true. It is harmless only because the UCC bridge never calls
  it; wiring it in would reintroduce the wrong-slice failure above.

### 2. The rank-6 closure relation already exists and must not be re-derived

`_split_same_spin_amplitude` exists and is pinned to 1e-12 against a real UCCSDT fixture. Three
hand-derivations were attempted and all three failed the bra-antisymmetry check — because they
built a closed-shell *spatial* identity where the pinned relation is a slice of a *spin-orbital*
tensor.

Design rule:

- Do not hand-derive the rank-6 closure relation. Use `_split_same_spin_amplitude`.
- A failed guess is evidence about the guess, not about the tree — a blocker was once asserted
  ("the t3 closure is underived") from three failed hand-derivations without checking whether the
  tree already had one. It did.

### 3. Numeric validation on symmetry-correct tensors, never symbolic term comparison

Two written forms of the same algebra differ freely at the term level; a multiset comparison
reports that as a defect even when both are correct. The bridge must be gated numerically, on
tensors that already carry the required symmetry.

Design rule:

- Do not gate the bridge on symbolic term comparison.
- A cheap invariant check is only as trustworthy as its fixture — the α↔β symmetry check was the
  right instrument and fired correctly, but the conclusion drawn from an early run was still wrong,
  because a bad fixture and a bad equation produce the same symptom. Four of the defects found here
  were in the fixture or the interface, none in the equations.

### 4. UCC factor names are never bare, and the UCC path must not touch RCC

No block is privileged on the UCC path, so an unsuffixed factor name is ambiguous rather than a
shorthand. Separately, `spin_adapt_equations` (the RCC path) must stay byte-identical to its
pre-UCC behavior.

Design rule:

- Do not emit a bare (unsuffixed) factor name on the UCC path.
- Do not let the UCC path touch the RCC one.

## What was found

Four lessons, and they rhyme — each cost a wrong conclusion that one earlier measurement would
have prevented:

1. **Do not trust a status header without running its gate.** The canonicalizer-retargeting step
   was added by this doc's own probe as "a step the original scope missed", with a correct
   measurement attached — and was then solved by a different route and never revisited. Its gate
   is four lines and passes.
2. **A failed guess is evidence about the guess, not about the tree.** A blocker was asserted
   ("the t3 closure is underived") from three failed hand-derivations, without checking whether
   the tree already had one. It did.
3. **A cheap invariant check is only as trustworthy as its fixture.** The α↔β symmetry check was
   the right instrument and fired correctly; the conclusion drawn from it was still wrong, because
   a bad fixture and a bad equation produce the same symptom. Four of the defects found here were
   in the fixture or the interface — none in the equations.
4. **The one that subsumes them:** the rank-6 question was rescoped four times, each rescope
   correctly dissolving its candidate blocker, and *none of them was the actual gap*. The gap was
   that the GCC→UCC adaptation had **22 call sites and no numeric gate at all** — a fact available
   from `grep` at any point. **When a question survives several rounds of narrowing, check what is
   UNVERIFIED before narrowing again.**

## Validation strategy that should remain in place

- `U14c3UccIsGccSlicedAtRankSixTests` — the UCC manifold *is* the GCC one sliced into spin blocks
  (1.6e-17 on a perturbed spin-orbital t3)
- Rank-4 validation against PySCF UCCSD (~6e-16)
- Rank-6 validation against GCC-sliced amplitudes, with GCC itself FCI-exact
- See `CCGEN_UCC_NUMERIC_VALIDATION.md` for how the numeric side is constructed and the four
  PySCF-interface defects that cost the most time

## Related but separate outcome: cross-references

See `CCGEN_UCC_NUMERIC_VALIDATION.md` (how a spin-block residual is validated, and the four
PySCF-interface defects that cost the most time), `CCGEN_UNRESTRICTED_CC.md` (the pipeline this
feeds, end to end), and `CCGEN_UCC_ERI_ANTISYMMETRY.md` (the ERI convention across the codegen
boundary).
