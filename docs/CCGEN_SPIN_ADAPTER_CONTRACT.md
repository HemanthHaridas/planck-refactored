# What does the spin adapter guarantee, and how do you check it?

Answers one question: **`ucc_integrate_term_antisym` turns a spin-orbital term into spatial terms —
what property does it promise, what breaks that property, and what kind of test can actually detect a
break?**

Short answer: it promises that its output depends on the *algebra* of its input, not on *how that
algebra was written*. That promise was false for `v` orientation and is now enforced by
canonicalizing every rank-4 `v` inside the adapter. And the only instrument that can check the
promise is a **numeric** comparison on symmetry-correct tensors — a symbolic term comparison cannot
distinguish "different algebra" from "same algebra, different writing", which cost five sub-steps and
one phantom defect to learn.

---

## The invariant

> For any two terms equal as algebra — i.e. equal under `_eri_canonical`, the 8-fold ERI symmetry
> fold — the adapter must produce spin-integrated results with equal total coefficient per (block,
> spatial content).

Note what it does *not* demand: identical intermediate `SpinTerm` lists. Two writings may
legitimately produce different representatives; what must match is the integrated contribution,
because that is what reaches `merge_terms` and the emitted kernel.

## How it broke: slot position is read as physics

`_line_pairs` pairs slot `k` with slot `k+n` — the physicist `<pq|rs>` convention, where `p–r` and
`q–s` are the interaction lines. That makes the *written arrangement* load-bearing. Two writings of
one integral, taken verbatim from the dressed-CCSD doubles manifold:

```
v(k,b,c,j)      lines:  k–c , b–j     (occ–vir, occ–vir)
v(j,c,k,b)      lines:  j–k , c–b     (occ–occ, vir–vir)
```

`v(j,c,k,b)` is `v(k,b,c,j)` bra↔ket-exchanged and then bra-swapped; for antisymmetric `v` that swap
costs −1, which is exactly the coefficient the second form carries. `_eri_canonical` agrees they are
one term. The adapter integrated them to **2 and 0**.

Per spin case the divergence was precise: the same 6 of 16 cases survived, into the same blocks, but
the four mixed-spin cases carried **opposite signs** — because `_antisym_to_allowed` re-derived its
sign from written slot order via `_permutation_parity`, treating one integral as two independent
inputs.

**The bra↔ket exchange alone is harmless** — 0 divergences across a 256-case sweep (all occ/vir slot
patterns × all spin labels), because exchange maps lines `p–r, q–s` to `r–p, s–q`: the same lines. The
defect requires exchange **composed with** a within-group swap, which is what re-pairs the lines. Any
fix that only tries the exchange misses it.

## The fix, and where it had to go

`_orientation_normalized` reorients every rank-4 `v` to one canonical member of its 8-fold orbit
before the lines are read, folding that reorientation's parity into the returned sign. It reuses
`_ERI_PERMUTATIONS` / `_perm_parity` from `dressing.py` rather than defining a second ERI group.

**Inside the adapter, not at the caller.** The alternative was normalizing `v` at the dress/adapt
boundary — cheaper, and it leaves validated code untouched. Rejected because the defect is *latent
and pre-existing*: it is a property of `ucc_integrate_term_antisym`, so any future caller writing its
`v` factors differently from the diagram generator hits it. A boundary patch fixes the one caller
that currently hurts and arms the trap for the next.

**The representative must be chosen by index name.** A name-independent `(space, spin)` key was tried
first, to avoid depending on names at all. It is degenerate — several orbit members tie on it — so the
tie-break picks the representative and two writings land on *different* ones. Measured, that made all
6 surviving spin cases disagree instead of 4: strictly worse than no fix. The landed version uses the
lexicographic `(space, name)` rule `_eri_normalize_factor` already uses, which is sound within a
single term because both writings of one integral carry the same dummy names, differently arranged.

Two consequences worth knowing:

- **Per-*factor* verdicts still differ between writings, correctly.** `v(j,c,k,b) = −v(k,b,c,j)` as a
  factor; the term's own coefficient compensates. The invariant is per-*term*. What changed is that
  the factor-level sign flip is now **uniform** across all surviving cases (the orbit parity) rather
  than an inconsistent 4-of-6 — uniformity is the signature of a clean global reorientation.
- **The spatial emit shrank**, 73260 → 65431 bytes, because normalization merges
  orientation-duplicate terms. Same algebra, fewer terms; verified by identical adapted-residual
  multisets before and after.

## Why the checking instrument matters more than the fix

After the fix, a symbolic term-by-term comparison of the dressed-vs-raw adapted doubles residual still
reported **14 mismatches**. That number was pursued as a defect. It was not one.

With a symmetry-correct `v`, the two manifolds agree to **~1e-14** on energy/singles/doubles across
three `(no, nv, seed)` triples. The algebra was already correct. The 14 were an artifact of comparing
*written forms*: a term multiset cannot see that two sides chose different, symmetry-equivalent
writings of the same thing.

**The real defect was in the test fixture.** `residual_eval.random_tensors` built `v` with intra-pair
antisymmetry only:

```
bra antisym  |v + v(1,0,2,3)|  = 0.0    ✓
ket antisym  |v + v(0,1,3,2)|  = 0.0    ✓
bra↔ket      |v − v(2,3,0,1)|  = 2.35   ✗   should be 0
```

Real antisymmetrized integrals satisfy all three — checked against pyscf on H2/STO-3G: 3.03e-16, 0.0,
2.22e-16. Since `_ERI_PERMUTATIONS` folds by that exchange, **any numeric comparison of two
exchange-related writings on that fixture reported a spurious difference** — concretely, it made a
GCC comparison look 170 % off on manifolds the symbolic fold correctly called identical.

Repairing it takes four projections, and the fourth is required rather than defensive: the two
intra-pair projections do not commute with the exchange projection, so a single pass leaves a residual.

```python
V = 0.5 * (V + V.transpose(2, 3, 0, 1))   # impose bra<->ket
V = 0.5 * (V - V.transpose(1, 0, 2, 3))   # restore bra antisym
V = 0.5 * (V - V.transpose(0, 1, 3, 2))   # restore ket antisym
V = 0.5 * (V + V.transpose(2, 3, 0, 1))   # re-impose
```

Two rules fall out, and they generalize past this adapter:

1. **Gate on numeric residual values, not symbolic term counts** — whenever both sides may choose
   among symmetry-equivalent forms, a multiset comparison reports differences that are not there.
2. **A numeric gate is only as good as its fixture's symmetry.** `random_tensors` was missing an
   identity real integrals have, for a long time, because nothing compared two exchange-related
   writings until dressing did. Assert the fixture's invariants, not only the result.

## Hypotheses that were eliminated

Recorded so they are not re-proposed:

- **Not `merge_terms`, not the closed-shell collapse.** Adaptation is additive over term partitions,
  and on the minimal reproducer both sides sum to the same coefficient through the full pipeline. The
  repeated-factor signature in the 14 (`t1t1v`, `t2t2v`, `t1t1t1t1v`) was incidental — those terms
  simply have the most written forms, hence the most opportunities to differ in choice.
- **Not the dressed operators.** GCC dressed assembly is exact on every manifold. Dressing merely
  *exposed* the adapter property, by being the first thing to feed it a differently-written form of an
  equation it already handled.

## What the adapter's own gates are

`test_spin` carries the S1/S2/S4 numeric gates and the rank-6/rank-8 FCI-limit fixtures, and the
Be CCSDTQ == FCI acceptance runs through this function — so a change here must re-run those, not just
the dressing tests.

**Run them through `tests/pyscf/.venv/bin/python`** (pyscf 2.13.0). In the default interpreter every
pyscf gate reports `skipped`, so a green run there is not evidence of anything — the numeric gates
were silently skipping during part of this investigation.

---

Status lives in `vault/Status/Completion.md`. Implementation: `python/ccgen/spin.py`
(`_orientation_normalized`, `_antisym_to_allowed`, `_line_pairs`); fixture:
`python/ccgen/tests/residual_eval.py`; gates: `test_spin_orientation.py`,
`test_residual_symmetry.py`, `test_spin.py`.
