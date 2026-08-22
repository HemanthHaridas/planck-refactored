# U1 — the UCC block-resolution entry, scoped as U1.0–U1.5

**U1.0, U1.1, U1.2 and U1.5 are LANDED. U1.3 is DEAD. Only U1.4 remains.**

Re-audited against the tree 2026-08-22, and three of the four steps that were open turned out not to
be work: U1.2's deliverable arrived through the F1/F2/F3 ladder (~6e-16 vs PySCF UCCSD), U1.5's is
the F2.3 closed-shell oracle (2.45e-15 relative), and **U1.3's hazard was designed out by U1.1**
rather than left open — its own gate now passes with 0 violations. Each is recorded in place below
with what was measured.

U1.4 is rescoped: its blocker is not the oracle *strength* this doc assumed, but a **t3 closure
relation** that does not generalize from rank 4. See U1.4.

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

> **Superseded 2026-08-22.** U1.1 landed by *disabling* slot canonicalization on the UCC path rather
> than retargeting it, so the "rank-6 path that needs the canonicalizer retargeted" does not exist —
> see U1.3 below, whose gate now passes with 0 violations. The rank-4-then-rank-6 ladder still holds,
> but rank 6 isolates the **t3 closure relation** (U1.4a), not the canonicalizer.

**Gate at rank 4 first, rank 6 second** — the original scope said "do this at rank 4 before
touching any C++", which is right, and the probe adds the reason: rank 4 exercises the naming
end-to-end while avoiding the β-majority hazard, so a rank-4 failure is unambiguously a *naming*
bug.

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

### U1.2 — rank-4 numeric gate against PySCF UCCSD (~M) — **LANDED**

Evaluate the doubles residuals at PySCF UCCSD's own amplitudes and compare against
`pyscf.cc.uccsd.update_amps`. Landed as `F3UccVsPyscfTests`
(`python/ccgen/tests/test_ucc_vs_pyscf.py`), reached through the F1/F2 fixture-and-evaluator ladder
this step was blocked on.

**Result: ~6e-16 in every block** — machine precision, gated at 1e-13 rather than the scoped 1e-10,
and covering the two *singles* blocks as well as the three doubles this step asked for.

Both vacuous-pass traps this scope named were real, and are asserted rather than commented:
amplitudes are perturbed off convergence (at PySCF's converged amplitudes the reference residual is
~1e-8, so a kernel returning zero passes), and CH3/STO-3G is used rather than OH/STO-3G. Three
evaluator mutations fail the gate.

Corrections this step produced, recorded in `CCGEN_UCC_NUMERIC_FIXTURE_SCOPE.md`: the PySCF
amplitude mapping is a **transpose**, not a pure rename; and `f_ov` must be zeroed on **both** sides
(one-sided zeroing is *worse* than neither).

### U1.3 — retarget the canonicalizer for β-majority blocks — **DEAD. U1.1 designed the hazard out.**

This step exists because `_canonicalize_amplitude_factor` folds β-majority blocks toward the RCC
reference. That measurement still reproduces exactly:

| block | canon slots | RCC tag | UCC tag |
|---|---|---|---|
| `aaaaaa` | `aaaaaa` | `aaaaaa` | `aaaaaa` |
| `aabaab` | `aabaab` | `aabaab` | `aabaab` |
| `abbabb` | **`bbabba`** | **`aabaab`** | **`abbabb`** |
| `bbbbbb` | `bbbbbb` | **`aaaaaa`** | `bbbbbb` |

**But the UCC path never calls that function.** `ucc_spinterm_to_algebraterm` (`spin.py:1021`)
disables slot canonicalization outright — "Slots are left exactly as the spin integration produced
them" — precisely because reordering onto another block's layout would read the wrong array. U1.1
solved this by *not calling* the canonicalizer, rather than by retargeting it, and this scope
predates that decision.

This step's own gate — `tag == _ucc_block_tag(canonicalized_slots)`, described here as failing today
for `abbabb` — **passes at rank 6 with 0 violations across all 2490 `ccsdt` UCC terms**, including
every `t3_abbabb` factor. Measured 2026-08-22.

Retained rather than deleted because the *measurement* is still true and still a trap for anyone who
later wires the RCC canonicalizer into a UCC path. The step is not work.

### U1.4 — rank-6 numeric gate (~M) — **the only remaining U1 work; rescoped**

This scope assumed the blocker was oracle *strength*: "PySCF has no UCCSDT `update_amps`, so the
oracle is weaker — likely a spin-orbital GCC slice comparison rather than a direct one."

**Probing says the blocker is somewhere else.** The closed-shell oracle F2.3 established *is*
available at rank 6, needs no PySCF, and pairs `triples_aabaab` against RCC `triples` — the same
per-target pairing, since RCC adapts on the closed-shell representative block. What it needs is a
**t3 closure relation**, and that does not fall out of the rank-4 one:

```
rank 4:  t2_aaaa = t2 - t2.transpose(1,0,2,3)      → bra/ket antisym defect 0.0   ✓
rank 6:  first-guess 3-term generalization          → bra antisym defect 3.5      ✗
```

A block that is not bra-antisymmetric is not a valid `aaaaaa` block, so the obvious generalization
is wrong. Deriving the right relation is the actual content of U1.4, and it is the one place in the
remaining ladder with real uncertainty.

Cheap facts confirmed while probing: `ccsdt` generation + UCC adaptation is **0.4 s**, and the
manifold carries exactly the four expected t3 blocks (`t3_aaaaaa`, `t3_aabaab`, `t3_abbabb`,
`t3_bbbbbb`) against RCC's single bare `t3`. So nothing here is gated on cost.

#### U1.4a — derive the t3 closure relation (~M, the real content)

Read it off `_split_same_spin_amplitude`'s own construction rather than guessing at it: for `n=3`
that function emits the `aabaab` block from three signed bra permutations, so what U1.4 needs is its
inverse. The first guess — mirroring the three permutations directly — is already falsified above,
which is why this gets its own step instead of being assumed inside the fixture.

*Gate:* the constructed `t3_aaaaaa` is bra- **and** ket-antisymmetric to 1e-14. That is exactly the
check the wrong guess fails, it needs no equations and no PySCF, and it localizes a closure error to
the relation rather than to the residual.

#### U1.4b — extend `ucc_closed_shell_tensors` to t3 (~S)

Add the four t3 blocks to the F2.3 fixture using U1.4a's relation.

*Gate:* shapes per block, and each same-spin block satisfies its own antisymmetry — the same shape
of check F1 uses, for the same reason.

#### U1.4c — the rank-6 closed-shell oracle (~S once a+b land)

`triples_aabaab` against RCC `triples`, by the per-target pairing F2.3 established.

*Gate:* ≤1e-11 elementwise on a **non-square** case, with a committed falsifiability check (a
corrupted block must break it by O(‖R‖)) — F2.3's pattern, for F2.3's reason.

Note `triples_aaaaaa` / `triples_bbbbbb` / `triples_abbabb` have **no RCC counterpart**, exactly as
the rank-4 same-spin blocks do not: `collapse_amplitudes` splits the all-α sector away rather than
storing it. They are covered structurally and by U1.4a's antisymmetry gate, not by this comparison.

#### U1.4d — decide whether a direct rank-6 oracle is worth building (~S, a decision)

This scope defers the decision to U1.4, which is right. **Recommendation after probing: no.** Write
down that U1.4c is a transitive oracle and stop there. A direct one needs a UCCSDT reference PySCF
does not have; U1.4c already exercises the rank-6 t3 naming and slot-order path end to end, which is
what rank 6 adds over rank 4. Cheap to reverse if U2+ surfaces a rank-6 defect.

**Do not claim a direct oracle where only a transitive one exists** — that distinction is what made
U1.2 valuable, and it is worth more than an extra gate here.

### U1.5 — closed-shell degeneracy check (~S) — **LANDED as F2.3**

For an α ≡ β input, `ucc_adapt_equations` must reproduce `spin_adapt_equations`, compared
numerically. This is exactly the F2.3 oracle, landed as `F23ClosedShellOracleTests` (`test_spin.py`).

*Scoped:* ≤1e-12 relative at rank 4. *Measured:* **2.45e-15** relative (3.9e-12 absolute against
‖R‖ ~1.6e3).

Two corrections F2.3 produced, recorded in `CCGEN_UCC_RESIDUAL_EVALUATOR_SCOPE.md`: the comparison is
a **per-target pairing, not a sum over blocks**; and it needed a **second fixture**
(`ucc_closed_shell_tensors`), because F1's independently-drawn blocks violate the closure relations
by construction.

## Sequencing

```
U1.0 (no-collapse entry)        ~S   ← pipeline already works; assembly only
   └→ U1.1 (block-resolved names)  LANDED  ← also designed out U1.3's hazard
        ├→ U1.2 (rank-4 vs PySCF UCCSD)  LANDED  ~6e-16
        │    └→ U1.3 (retarget canonicalizer)  DEAD — never called on the UCC path
        │         └→ U1.4 (rank-6 numeric)  ← THE ONLY REMAINING U1 WORK
        │              U1.4a derive t3 closure   ~M  ← the real content
        │              U1.4b extend the fixture  ~S
        │              U1.4c rank-6 oracle       ~S
        │              U1.4d direct-oracle call  ~S  ← recommend: no
        └→ U1.5 (closed-shell degeneracy)  LANDED as F2.3  2.45e-15 rel
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

| the closed-shell oracle + its fixture | F2.3 (`ucc_closed_shell_tensors`) — U1.4c extends both to t3 |
| the block-wise evaluator | F2.2 (`ucc_residual_einsum`) — rank-agnostic already |

**Net new, all of it inside U1.4:** one t3 closure relation (the only piece with real uncertainty),
four t3 blocks on an existing fixture, and one more per-target comparison. The canonicalizer
retarget this table used to list is **not** net new — it is not needed at all.

---

## What NOT to do

- **Do not use `_amplitude_block_tag` on the UCC path.** It folds β-majority onto α-majority
  (`abbabb` → `aabaab`, `bbbbbb` → `aaaaaa`), valid only when α ≡ β. U0's `_ucc_block_tag` is the
  UCC-correct one and already exists.
- **Do not assume the rank-4 canonicalizer behaviour generalizes.** It is block-local at rank 4 and
  not at rank 6 — measured, and still true. It is harmless only because the UCC bridge never calls
  it; wiring `_canonicalize_amplitude_factor` into a UCC path would reintroduce the R3.1.2 failure.
- **Do not assume the rank-4 *closure relation* generalizes either.** The obvious three-permutation
  generalization of `t2_aaaa = t2 - t2.transpose(1,0,2,3)` produces a t3 block that is **not**
  bra-antisymmetric (defect 3.5), so it is not a valid `aaaaaa` block. Derive it; do not pattern-match
  it. This is U1.4a.
- **Do not gate U1 on symbolic term comparison.** Numeric, on symmetry-correct tensors.
- **Do not skip U1.2 to reach C++.** B5's convention bug was found only by an oracle injected
  into live C++ state; U1.2 is the cheap version of that.
- **Do not emit a bare (unsuffixed) factor name on the UCC path.** No block is privileged there;
  a bare name is ambiguous rather than a shorthand.
- **Do not let U1 touch the RCC path.** `spin_adapt_equations` must stay byte-identical — the
  same constraint U0 honoured via `fold_spin_flip=True`.

---

## Honest status

**U1 is one step from done, and that step is smaller than this doc originally implied — but its
content is not where the doc looked.**

What the 2026-08-22 re-audit changed, and the lesson in it: this doc's own probe added U1.3 as a
"new step the original scope missed", with a correct measurement attached. U1.1 then solved that
hazard by a route the probe had not considered — *not calling* the canonicalizer instead of
retargeting it — and the step was never revisited. A correct measurement produced a step that was
already unnecessary by the time it was written down.

So the standing caution is not "the measurements were wrong". They were right, all three of them.
It is that **a measured hazard is not the same as remaining work**, and the gap between them is
exactly what a stale scope hides. Re-run a step's own gate before building it: U1.3's gate is four
lines and passes.

The one genuinely open question is now smaller and sharper than "is the rank-6 oracle strong
enough": it is **what the t3 closure relation actually is**. That is answerable without PySCF, in
Python, with an antisymmetry check that the first wrong guess already fails.

---

See `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (U0 landed, U2–U5 ahead, and the rescope that corrected
`external_blocks`), `CCGEN_SPIN_ADAPTER_CONTRACT.md` (the numeric-over-symbolic lesson), and
`spin.py:955` (`_factor_tensor_name`) / `spin.py:1088` (`block_keyed_intermediate_name`).
