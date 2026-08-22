# U1 — the UCC block-resolution entry, scoped as U1.0–U1.5

**U1 IS COMPLETE. U1.0–U1.2, U1.4 and U1.5 are LANDED; U1.3 is DEAD.** ccgen's UCC equations are
validated at rank 4 (~6e-16 vs PySCF UCCSD) and rank 6 (1.6e-17 vs GCC-sliced, with GCC reaching the
FCI limit exactly). **U2 is unblocked and in progress.**

One thread stays open and is **not** a ccgen defect: `test_ucc_rank6_vs_pyscf`'s triples target
differs from PySCF by rel ~2e-3 (`expectedFailure`). ccgen is cleared by two independent routes; the
undiagnosed side is PySCF's `r3aaa`. The physicist/chemist ERI convention — the obvious suspect,
being the class of the B5 defect — was checked and eliminated.

> **The full account now lives in `docs/CCGEN_UCC_RANK6_PYSCF_GAP_HANDOFF.md`** — how a spin-blocked
> residual is validated (three independent routes), the four PySCF-interface defects, the nine
> falsified hypotheses for the open gap, and the named next step. Read that before working on any of
> it; this doc is the step-by-step record behind it.

Re-audited against the tree 2026-08-22, and three of the four steps that were open turned out not to
be work: U1.2's deliverable arrived through the F1/F2/F3 ladder (~6e-16 vs PySCF UCCSD), U1.5's is
the F2.3 closed-shell oracle (2.45e-15 relative), and **U1.3's hazard was designed out by U1.1**
rather than left open — its own gate now passes with 0 violations. Each is recorded in place below
with what was measured.

U1.4 was rescoped four times, each time by measurement, and **every candidate blocker dissolved**:
not the oracle *strength* this doc assumed (PySCF ships a UCCSDT residual entry), not an underived
t3 closure relation (pinned green at `test_spin.py:2158`), not oracle plumbing (solved), and not a
rank-6 equation defect — that last one was a **fixture bug in the test that found it**, retracted
the same day. The rank-6 manifold is spin-flip symmetric to ~7e-16.

**What finally settled rank 6 was none of those: it was noticing that the GCC→UCC adaptation had 22
call sites and no numeric gate at all.** `U14c3UccIsGccSlicedAtRankSixTests` closes that — the UCC
manifold *is* the GCC one sliced into spin blocks (1.6e-17 on a perturbed spin-orbital t3) — and
combined with ccgen's GCC CCSDT reaching the FCI limit exactly, it makes rank-6 UCC correct. See
U1.4.

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

### U1.4 — rank-6 numeric gate — **LANDED**, after four rescopes that each dissolved a different blocker

Two premises were falsified here, in opposite directions, and both are recorded because the second
one was mine.

#### Falsified premise 1 (this doc's): "PySCF has no UCCSDT `update_amps`"

It has one.

```
pyscf.cc.uccsdt.UCCSDT.update_amps   ->  True
```

Further, `_uccsdt_so_tensors` (`python/ccgen/tests/test_spin.py:1772`) already builds a **converged
UCCSDT reference** in-tree and runs it in CI. So a **direct** rank-6 oracle — the same shape as
U1.2's — is available. The "likely a spin-orbital GCC slice comparison rather than a direct one"
hedge is unnecessary.

#### Falsified premise 2 (the first rescope's): "the t3 closure relation is underived"

An earlier pass at this section claimed U1.4's real content was deriving a t3 closure relation,
because the obvious generalization of the rank-4 one produced a block that was not bra-antisymmetric
(defect 3.5).

**The relation is already derived, pinned, and green.** `_split_same_spin_amplitude` implements it
for general rank-2n, and `S4bSplitterTests.test_rank6_splitter_reproduces_block`
(`test_spin.py:2158`) asserts it reproduces `t3[aaaaaa]` to **1e-12** — evaluated against the real
UCCSDT spin-orbital fixture, not a constructed one.

The three failed guesses that produced premise 2 shared one mistake: they tried to build a
closed-shell **spatial** identity, when the pinned relation is a slice of a genuine **spin-orbital**
tensor. The failures were real; they measured the construction, not the relation. Recorded because
"my generalization does not satisfy the symmetry" is weak evidence that a relation is underived, and
strong evidence only that the guess is wrong — the in-tree gate is the thing to check first.

#### Falsified premise 3 (mine, and retracted the same day): "rank 6 has a correctness defect"

Attempting U1.4b produced a strong signal that the landed rank-6 UCC equations were broken: with
α ≡ β inputs the manifold was not spin-flip symmetric — ~1e-1 relative, with singles, doubles **and**
triples all failing together, reproducible with no PySCF involved. It was committed as an
`expectedFailure`.

**That was wrong. The equations are correct; the defect was in the test fixture, in one line.**
Symmetry holds at **~7e-16** once fixed, and the rank-6 test now passes.

`abbabb` is the spin flip of `aabaab`, and a flip is **not** the identity: flipping `aab` → `bba`
leaves slots `(b,b,a | b,b,a)`, which must then be re-expressed in `abbabb`'s own `(a,b,b | a,b,b)`
order — a slot reversal within each half.

```
t3_abbabb = t3_aabaab.transpose(2,1,0,5,4,3)
```

What made the false signal credible, and what eventually killed it — all three said the manifold was
structurally sound, which left only the fixture:

- term counts are symmetric (`ccsdt`: 579/579 triples, 469/469 mixed, 25/25 singles);
- the factor vocabularies of every spin-flip pair are exact mirrors;
- every emitted factor's slot spins equal its own tag — **0 mismatches across all 2490 terms**.

**The check that would have found it first, and did not exist:** does each fixture block satisfy its
**own** antisymmetry? A block's antisymmetric slot pairs follow from its tag — `aabaab` is antisym in
vir (0,1) and occ (3,4), `abbabb` in vir (1,2) and occ (4,5), *different pairs*, which is exactly why
setting the two equal is wrong. The bad block fails by 1.5e-2 where the right one gives exactly 0.
It is a property of one array: no equations, no evaluator, no oracle. Now committed as
`test_every_fixture_block_satisfies_its_own_antisymmetry`.

**Corollary, and a correction to the block mapping recorded below:** PySCF's `bba` block does **not**
map to `abbabb` by the same axis permutation `aab` maps to `aabaab`. An exhaustive 720-permutation
search matched `bba` to `aab` and that match is real — but *matching `aabaab`* is not the same as
*being `abbabb`*, and the search never checked the target block's own symmetry.
`bba.transpose(2,3,5,0,1,4)` violates `abbabb`'s antisymmetry by 1.5e-2. The correct `abbabb` for the
PySCF fixture is `aabaab.transpose(2,1,0,5,4,3)`.

#### What settled rank 6 — U1.4c

**The GCC→UCC adaptation had 22 call sites and no numeric gate.** It was verified structurally at
rank 6 (names, counts, slot order) and never against a value. That was the actual gap; every other
step in this ladder was downstream of it, which is why four successive rescopes each dissolved their
candidate blocker and still left the question open.

`U14c3UccIsGccSlicedAtRankSixTests` (`test_spin.py`) closes it by the adaptation's defining
property: evaluate the GCC triples residual on spin-orbital tensors, slice the all-alpha block,
require UCC `triples_aaaaaa` to equal it. **1.6e-17** against ‖G‖ 2.9e-2, with singles and doubles
likewise exact. Combined with ccgen's GCC CCSDT reaching the FCI limit exactly (three existing gates
in `test_reference_vs_pyscf`, including the `engine="diagram"` path this manifold comes through),
**rank-6 UCC is correct**.

Two fixture requirements, both learned by getting them wrong first:

- the spin-orbital tensors must carry the real even=α/odd=β interleaving (`_uccsdt_so_tensors`).
  `random_tensors` is spin-**free**, and slicing it by that convention produces disagreements of
  order the residual itself — which look exactly like an adaptation defect (measured 3.1e2 against
  ‖G‖ 3.6e2);
- the comparison must be perturbed off convergence, or every residual is ~1e-13 and it passes
  vacuously. The perturbation goes on the spin-orbital t3 and is re-antisymmetrized there, so it is
  valid by construction and both sides see the same tensor.

| step | status |
|---|---|
| derive the t3 closure | **not work** — pinned at `test_spin.py:2158` |
| the rank-6 spin-flip symmetry | **not work** — equations correct, ~7e-16 |
| the rank-6 oracle plumbing | **not work** — solved while probing, kept below |
| U1.4a/b — fixture + direct PySCF oracle | **LANDED** — `test_ucc_rank6_vs_pyscf` |
| U1.4c — adjudicate the residual gap | **LANDED** — ccgen cleared; PySCF's `r3aaa` undiagnosed |

#### The four PySCF-interface defects fixed en route

Each silently handed the two sides different amplitudes, and each reads as an equation defect:

1. **The re-antisymmetrization was unnormalized.** PySCF's `t2aa`/`t3aaa` arrive *already*
   antisymmetric, so re-applying `a - a.transpose(...)` **multiplies** them — by 4× and 36×
   (measured). The tell: ‖ref‖ = 1.7e-1 at converged amplitudes where PySCF's own residual is
   ~1e-10. **That one number should have been checked before any bisecting.**
2. `t2aa` is determined by `t2ab` (`t2aa = t2ab - t2ab.transpose(0,1,3,2)`); they cannot be
   perturbed independently.
3. `t3aaa` is determined by `t3aab`, through the same-spin closure.
4. **A block carrying `aabaab`'s antisymmetries is not thereby a valid amplitude block.** The real
   block has exactly those two signed symmetries and no others — checked exhaustively over all 36
   signed occ×vir permutations — so **no permutation test separates it from antisymmetrized noise**.
   What separates them is that the two same-spin closure forms agree on a valid block. Fixed by
   building the perturbation as a slice of a genuine antisymmetric spin-orbital tensor.

The two closure forms — the 3-term one `_split_same_spin_amplitude` implies, and the 36-term
normalized double antisymmetrizer — are **equivalent on valid blocks** (both reproduce a genuine
`aaaaaa` to ~2e-15). They diverge only on inputs that are not amplitude blocks. An earlier claim in
this ladder that they were inequivalent and side-specific is **retracted**.

#### What remains, and it is not ccgen

`test_ucc_rank6_vs_pyscf`'s triples target differs from PySCF by rel **~2e-3** (down from 8.8e-2),
kept as an `expectedFailure`. Every fixture relation holds to ~1e-17, ccgen is cleared by two
independent routes, and the **physicist/chemist ERI convention is eliminated**: PySCF's UCCSDT
`eris` is a `_PhysicistsERIs` whose `pppp` is the non-antisymmetrized `<pq|rs>`, equal to this
gate's construction to 5.4e-15, and the gate's blocks carry the symmetries ccgen requires
(`v_aaaa` ket-antisymmetric, `v_abab` not). PySCF's `r3aaa` is undiagnosed. An unexpected PASS means
someone diagnosed it.

#### The oracle plumbing, established while probing

- PySCF's real CCSDT residual entry is **`update_amps_uccsdt_tri_(mycc, tamps, eris)`**, mutating
  `tamps` in place by `R/D`. `UCCSDT.update_amps` is the *inherited CCSD* one and silently omits t3
  — it exists and runs, which is the trap.
- t3 is stored **packed** (triangular); `tamps_tri2full_uhf` unpacks, `tamps_full2tri_uhf` repacks,
  and the round trip is exact.
- `aab` and `bba` are **one stored sector**. Perturbing them independently makes the two sides see
  different t3, and it surfaces as a **singles** error (5e-14 → 8.9e-3) far from its cause.


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
U1.0 (no-collapse entry)          LANDED
   └→ U1.1 (block-resolved names) LANDED  ← also designed out U1.3's hazard
        ├→ U1.2 (rank-4 vs PySCF UCCSD)   LANDED  ~6e-16
        │    └→ U1.3 (retarget canonicalizer)  DEAD — never called on the UCC path
        │         └→ U1.4 (rank-6 numeric)     LANDED
        │              U1.4a/b  fixture + direct PySCF oracle
        │              U1.4c    adjudication: UCC == GCC-sliced, 1.6e-17
        └→ U1.5 (closed-shell degeneracy) LANDED as F2.3  2.45e-15 rel
              │
              ▼
         U2 (UHF reference, C++) → U3 → U4 → U5
         U2.1 landed: build_ucc_block_denominator
```

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
- **Do not hand-derive the rank-6 closure relation.** It exists (`_split_same_spin_amplitude`) and is
  pinned to 1e-12 against a real UCCSDT fixture. Three hand-derivations were attempted while scoping
  U1.4 and all three failed the bra-antisymmetry check — because they built a closed-shell *spatial*
  identity where the pinned relation is a slice of a *spin-orbital* tensor. Check the in-tree gate
  before concluding a relation is missing.
- **Do not gate U1 on symbolic term comparison.** Numeric, on symmetry-correct tensors.
- **Do not skip U1.2 to reach C++.** B5's convention bug was found only by an oracle injected
  into live C++ state; U1.2 is the cheap version of that.
- **Do not emit a bare (unsuffixed) factor name on the UCC path.** No block is privileged there;
  a bare name is ambiguous rather than a shorthand.
- **Do not let U1 touch the RCC path.** `spin_adapt_equations` must stay byte-identical — the
  same constraint U0 honoured via `fold_spin_flip=True`.

---

## Honest status

**U1 is complete.** ccgen's UCC equations are validated at rank 4 against PySCF UCCSD (~6e-16) and
at rank 6 against GCC-sliced (1.6e-17), with GCC itself FCI-exact. U2 is unblocked and started.

Three lessons, and they rhyme — each cost a wrong conclusion that one earlier measurement would have
prevented:

**Do not trust a status header without running its gate.** U1.3 was added by this doc's own probe
as a "step the original scope missed", with a correct measurement attached. U1.1 then solved the
hazard by a route the probe had not considered — *not calling* the canonicalizer instead of
retargeting it — and the step was never revisited. Its gate is four lines and passes.

**A failed guess is evidence about the guess, not about the tree.** The first rescope of U1.4
asserted a blocker ("the t3 closure is underived") from three failed hand-derivations, without
checking whether the tree already had one. It did.

**A cheap invariant check is only as trustworthy as its fixture.** The α↔β symmetry check was the
right instrument and fired correctly; the conclusion drawn from it was still wrong, because a bad
fixture and a bad equation produce the same symptom. Four of the defects found in U1.4 were in the
fixture or the interface, none in the equations.

And the one that subsumes them: **U1.4 was rescoped four times, each rescope correctly dissolving
its candidate blocker, and none of them was the actual gap.** The gap was that the GCC→UCC
adaptation had 22 call sites and no numeric gate — a fact available from `grep` at any point. When a
question survives several rounds of narrowing, check what is *unverified* before narrowing again.
