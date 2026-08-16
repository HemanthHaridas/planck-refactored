# U1 — the UCC block-resolution entry, scoped as U1.0–U1.5

Decides U1's scope against the current tree. **Probed first**, and the probe moved the risk:
the half I flagged as dangerous is safe at rank 4 and dangerous at rank 6, for a reason the
original scope did not name.

U0 is landed (`ucc_independent_blocks`, `_ucc_block_tag`, `external_blocks(fold_spin_flip=…)`).

> **"Adapt" reads backwards here.** In this repo it means `spin_adapt_equations`, the **spatial
> collapse**; U1 is the entry that *skips* that collapse and keeps blocks resolved. The filename and
> the inherited `_adapt_on_block` naming say the opposite of what this step does. **UCC is
> spin-block resolved, never spatial** — see the three-assumption audit in
> `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md`, one of which (`_canonicalize_amplitude_factor`) fires at
> rank 2 and is not addressed by U1.1's naming fix alone.

---

## What the probe established

### U1.0's pipeline already works — no collapse needed, nothing to write

Running `ucc_integrate_target → merge_terms → spinterm_to_algebraterm` per U0 block, skipping
all three collapse steps, on `ccsd` doubles:

| block | integrated | merged | bridged |
|---|---|---|---|
| `aaaa` | 102 | 98 | 98 |
| `abab` | 103 | 82 | 82 |
| `bbbb` | 102 | 98 | 98 |

All three blocks integrate, merge, and bridge **today**, with `aaaa`/`bbbb` symmetric as they
must be. So U1.0 is assembling six existing lines per block, not new machinery — **~S, not part
of the ~M**.

### U1.1 is confirmed necessary: the bridge drops the block

Measured on one `abab` term — the `SpinTerm` carries `[('t1','bb'), ('v','abab')]`, and after
`spinterm_to_algebraterm` the factors are `['t1', 'v']`. The blocks are gone. Under RCC that is
correct (one stored tensor); under UCC `t1bb` and `t1aa` are different arrays, so this is the
correctness-critical half of U1, exactly as scoped.

The naming mechanism exists and is already used: `_factor_tensor_name` inside the bridge appends
a sector tag (`t4_aaabaaab`), and `block_keyed_intermediate_name` (V1.1c) does the same for
intermediates. **One naming shape, three consumers** — U1.1 generalizes the gate on
`_factor_tensor_name`, which today fires only for `len(block) >= 8` amplitudes.

### The `_canonicalize_amplitude_factor` risk is REAL but not where I said

The original scope warned this function "reorders slots to one reference layout per rank" and
that UCC must retarget it per block. Measured at **rank 4**, it is already block-local:

```
aaaa -> +1  aa ba ia ja      abab -> +1  aa bb ia jb
baba -> +1  ba ab ja ib      bbbb -> +1  ab bb ib jb
```

`bbbb` stays `bbbb`; `baba` correctly reorders onto the `abab` layout. **No cross-block folding
at rank 4**, so U1's headline risk does not bite at the rank U1's gate targets.

At **rank 6** it does, and this is the finding:

| block | canonicalized slots | RCC tag | UCC tag (U0) |
|---|---|---|---|
| `aaaaaa` | `aaaaaa` | `aaaaaa` | `aaaaaa` |
| `aabaab` | `aabaab` | `aabaab` | `aabaab` |
| `abbabb` | **`bbabba`** | **`aabaab`** | **`abbabb`** |
| `bbbbbb` | `bbbbbb` | **`aaaaaa`** | `bbbbbb` |

Two distinct disagreements for β-majority blocks: the RCC tag folds them onto their spin-flip
partner (valid only when α ≡ β), and the canonicalized *slot order* (`bbabba`) is not the UCC
tag's own layout either. So the tag and the slot order can disagree, which is precisely the
R3.1.2 failure mode — a factor indexing the wrong slice, residual ≈ 0.

**U0 already supplies the correct tag** (`_ucc_block_tag` returns `abbabb`/`bbbbbb`). The gap is
only that the canonicalizer sorts toward the RCC reference. That narrows the fix to one function
and makes the assertion writable.

---

## Scope decision

**U1 stays ~M, but the mass moves.** U1.0 drops to ~S (assembly only). U1.1 keeps the risk, and
it splits cleanly into a rank-4 path that already behaves and a rank-6 path that needs the
canonicalizer retargeted.

**Gate at rank 4 first, rank 6 second** — the original scope said "do this at rank 4 before
touching any C++", which is right, and the probe adds the reason: rank 4 exercises the naming
end-to-end while avoiding the β-majority hazard, so a rank-4 failure is unambiguously a *naming*
bug. Rank 6 then isolates the canonicalizer.

---

## Steps

### U1.0 — the no-collapse adapt entry (~S)

`ucc_adapt_equations(equations, blocks=None)` returning `{f"{target}_{tag}": [AlgebraTerm]}`,
driving the measured pipeline once per `ucc_independent_blocks` tag. No collapse steps.

Keep the adapter parameterized the way `adapt_intermediate_spec(adapter=…)` already is, so V5
substitutes rather than forks.

*Gate:* all U0 blocks produce non-empty term lists at ranks 2 and 4; the `aaaa`/`bbbb` counts are
equal (α↔β symmetry of the equations, a cheap structural check that catches a block-routing bug);
and RCC is untouched — `spin_adapt_equations` byte-identical.

### U1.1 — block-resolved factor names (~M, the correctness half)

Generalize `_factor_tensor_name`'s gate from `len(block) >= 8` to every factor whose tensor is
block-stored under UCC, using U0's `_ucc_block_tag` (**not** `_amplitude_block_tag`, which folds
the spin flip). Reuse the `f"{name}_{tag}"` shape.

**Reference-block naming decision, to make explicitly:** RCC's convention keeps the reference
sector bare (`t4`, not `t4_aabbaabb`) so the RCC emit stays byte-identical. UCC has no privileged
reference — `t2aa`, `t2ab`, `t2bb` are peers. Recommend **suffixing every UCC factor** and never
emitting a bare name on the UCC path: a bare name there would be ambiguous, and PySCF's own
`t2aa/t2ab/t2bb` naming has no unsuffixed member either.

*Gate:* every emitted factor name is `<tensor>_<tag>` with the tag drawn from U0's vocabulary; no
factor carries an unresolved or bare block; RCC unchanged.

### U1.2 — rank-4 numeric gate against PySCF UCCSD (~M, load-bearing)

Evaluate the `doubles_aaaa` / `doubles_abab` / `doubles_bbbb` residuals **at PySCF UCCSD's own
converged `t1a/t1b/t2aa/t2ab/t2bb`** on an open-shell case, and compare against
`pyscf.cc.uccsd.update_amps`.

This is UCC's one **direct** oracle — everything else is transitive through
`ucc_manifold == GCC-sliced`. Evaluate-at-PySCF-amps is the same convention-robust pattern the
RCC S3.2 gate used.

*Gate:* ≤1e-10 per block. **Not a symbolic term comparison** — V1.1e's e.2.0–e.2.5 established
that a term multiset cannot distinguish different algebra from a symmetry-equivalent rewriting.

**Do this before any C++.** The B5 precedent (a physicist-ERI convention bug found only by
injecting an FCI-correct oracle into live C++ state) is what this avoids.

### U1.3 — retarget the canonicalizer for β-majority blocks (~M, the measured hazard)

At rank ≥ 6, `_canonicalize_amplitude_factor` maps β-majority blocks toward the RCC reference:
`abbabb` → slots `bbabba`, RCC tag `aabaab`, UCC tag `abbabb`. Under UCC these are separate
tensors, so the factor must be canonicalized **within its own block**, and the tag must come from
`_ucc_block_tag`.

*Gate:* for every rank-6 block, the canonicalized slot order is the layout of the block's **own**
UCC tag — assert `tag == _ucc_block_tag(canonicalized_slots)`, which fails today for `abbabb`.
Plus rank-4 output unchanged (it is already correct, so this must not perturb it).

**Sequence after U1.2**, not before: a rank-4 gate that passes proves the naming path works, so a
rank-6 failure is then unambiguously this function.

### U1.4 — rank-6 numeric gate (~M)

Extend U1.2's oracle comparison to triples. PySCF has no UCCSDT `update_amps`, so the oracle is
weaker — likely a spin-orbital GCC slice comparison rather than a direct one.

*Gate:* whatever oracle is available, stated explicitly, plus the U1.3 structural assertion.
**Do not claim a direct oracle where only a transitive one exists** — that distinction is what
makes U1.2 valuable.

### U1.5 — closed-shell degeneracy check (~S)

For an α ≡ β input, `ucc_adapt_equations` composed with the three collapse relations must
reproduce `spin_adapt_equations`. **Compare numerically**, not as term multisets.

*Gate:* ≤1e-12 relative at rank 4. A useful independent check on U1.0+U1.1 that needs no PySCF.

---

## Sequencing

```
U1.0 (no-collapse entry)        ~S   ← pipeline already works; assembly only
   └→ U1.1 (block-resolved names) ~M ← the correctness half; rank-4-safe
        ├→ U1.2 (rank-4 vs PySCF UCCSD)  ~M  ← the one DIRECT oracle; before any C++
        │    └→ U1.3 (retarget canonicalizer)  ~M  ← the measured rank-6 hazard
        │         └→ U1.4 (rank-6 numeric)     ~M  ← weaker oracle, say so
        └→ U1.5 (closed-shell degeneracy) ~S  ← independent, no PySCF needed
              │
              ▼
         U2 (UHF reference, C++) → U3 → U4 → U5
```

---

## What this reuses

| Reused | From |
|---|---|
| `ucc_independent_blocks`, `_ucc_block_tag` | U0 (landed) |
| `ucc_integrate_target`, `merge_terms`, `spinterm_to_algebraterm` | S1/S2 (landed) |
| `f"{name}_{tag}"` naming | `_factor_tensor_name` (R3.1.3c), `block_keyed_intermediate_name` (V1.1c) |
| `v`-orientation invariance | e.2.1 — inherited free via `_antisym_to_allowed` |
| Numeric gating on symmetry-correct tensors | e.2.5 / `residual_eval` |
| Evaluate-at-PySCF-amps pattern | RCC S3.2 |

**Net new:** one adapt entry, one generalized naming gate, one canonicalizer retarget, and the
gates.

---

## What NOT to do

- **Do not use `_amplitude_block_tag` on the UCC path.** It folds β-majority onto α-majority
  (`abbabb` → `aabaab`, `bbbbbb` → `aaaaaa`), valid only when α ≡ β. U0's `_ucc_block_tag` is the
  UCC-correct one and already exists.
- **Do not assume the rank-4 canonicalizer behaviour generalizes.** It is block-local at rank 4
  and not at rank 6 — measured. This is why U1.3 exists and why the gate ladder is 4-then-6.
- **Do not gate U1 on symbolic term comparison.** Numeric, on symmetry-correct tensors.
- **Do not skip U1.2 to reach C++.** B5's convention bug was found only by an oracle injected
  into live C++ state; U1.2 is the cheap version of that.
- **Do not emit a bare (unsuffixed) factor name on the UCC path.** No block is privileged there;
  a bare name is ambiguous rather than a shorthand.
- **Do not let U1 touch the RCC path.** `spin_adapt_equations` must stay byte-identical — the
  same constraint U0 honoured via `fold_spin_flip=True`.

---

## Honest status

U1.0 is nearly free; the probe found its pipeline already working. U1.1 is the real work and is
well-understood. U1.3 is a **new** step this probe added — the original scope named
`_canonicalize_amplitude_factor` as U1.1's risk but had it backwards on where: safe at rank 4,
broken at rank 6, with the tag/slot-order disagreement as the specific mechanism.

The open question U1 cannot answer by itself is whether the rank-6 oracle (U1.4) is strong enough
to be worth the step, given PySCF has no UCCSDT `update_amps`. Decide that at U1.4, not now.

---

See `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (U0 landed, U2–U5 ahead, and the rescope that corrected
`external_blocks`), `CCGEN_SPIN_ADAPTER_CONTRACT.md` (the numeric-over-symbolic lesson), and
`spin.py:955` (`_factor_tensor_name`) / `spin.py:1088` (`block_keyed_intermediate_name`).
