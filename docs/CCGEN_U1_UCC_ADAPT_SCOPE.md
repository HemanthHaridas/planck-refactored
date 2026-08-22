# U1 — the UCC block-resolution entry, scoped as U1.0–U1.5

**U1.0, U1.1, U1.2 and U1.5 are LANDED. U1.3 is DEAD. Only U1.4 remains.**

Re-audited against the tree 2026-08-22, and three of the four steps that were open turned out not to
be work: U1.2's deliverable arrived through the F1/F2/F3 ladder (~6e-16 vs PySCF UCCSD), U1.5's is
the F2.3 closed-shell oracle (2.45e-15 relative), and **U1.3's hazard was designed out by U1.1**
rather than left open — its own gate now passes with 0 violations. Each is recorded in place below
with what was measured.

U1.4 was rescoped four times, each time by measurement, and **every candidate blocker dissolved**: not the
oracle *strength* this doc assumed (PySCF ships a UCCSDT residual entry), not an underived t3 closure
relation (pinned green at `test_spin.py:2158`), not oracle plumbing (solved), and not a rank-6 equation
defect — that last one was a **fixture bug in the test that found it**, retracted the same day. The
rank-6 manifold is spin-flip symmetric to **~7e-16**. What remains is a fixture extension and the
oracle itself. See U1.4.

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

### U1.4 — rank-6 numeric gate (~M) — **the only remaining U1 work; rescoped TWICE**

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

#### What is actually left

| step | status |
|---|---|
| derive the t3 closure | **not work** — pinned at `test_spin.py:2158` |
| the rank-6 spin-flip symmetry | **not work** — equations are correct, ~7e-16 |
| the rank-6 oracle plumbing | **not work** — solved while probing, see below |
| U1.4a — extend the fixture to t3 | **LANDED** |
| U1.4b — the direct rank-6 oracle vs PySCF UCCSDT | **LANDED for singles/doubles; triples OPEN at ~1.4e-2** |

#### U1.4 result

`U14RankSixVsPyscfTests` (`python/ccgen/tests/test_ucc_rank6_vs_pyscf.py`):

| target | vs PySCF UCCSDT |
|---|---|
| `singles_aa` / `singles_bb` | ~5e-14 ✓ |
| `doubles_aaaa` / `doubles_bbbb` | ~3e-15 ✓ |
| `triples_aaaaaa` | **~1.4e-2 — open**, `expectedFailure` |

**The split localizes the remaining defect.** The rank-6 singles and doubles
residuals *consume* t3, and they are exact — so the t3 blocks and ccgen's reading
of them are both right, and the discrepancy is confined to the **T3 equation**
(219 of 579 `triples_aaaaaa` terms carry a t3 factor).

Ruled out for the triples: not a layout or symmetry artifact (both residuals
bra-antisym to ~4e-16, and so is their difference); not a scale factor or
transpose (elementwise ratio median 0.9969, 5–95 spread 0.77–1.03 — a small
*additive* discrepancy); not the fixture t3 blocks (all four satisfy their tag's
antisymmetry, `aaa == bbb` to 2e-18); not the packing round-trip (bitwise exact).

Two rank-6 conventions found while building it, both traps worth keeping:

- **`update_amps_uccsdt_tri_` is the real CCSDT residual entry**, mutating `tamps`
  in place by `R/D`. `UCCSDT.update_amps` is the *inherited CCSD* one and silently
  omits t3 — it exists and runs.
- **`aab` and `bba` are ONE stored sector.** Perturbing them independently makes
  PySCF and ccgen see different t3, and it surfaces far from its cause — measured,
  it moves the *singles* residual from 5e-14 to 8.9e-3.

#### U1.4c — the rank-6 triples discrepancy (OPEN, scoped by investigation)

`triples_aaaaaa` disagrees with PySCF UCCSDT by ~1.2e-2 against ‖R‖~3.1, while rank-6 singles and
doubles are exact at ~1e-15. Bisected on a deterministic probe (amplitudes drawn wholly from the
seeded RNG — perturbing PySCF's converged ones drifts ~8% per process and makes any bisect
meaningless).

**Dead hypotheses, each measured:**

| hypothesis | killed by |
|---|---|
| T1-dressing convention difference | `t1 = 0` leaves the discrepancy unchanged |
| a spurious term family in ccgen | the 18 suspect `t2·v` terms are the standard `P(i/jk)P(a/bc)` expansions of two **textbook** T3 terms |
| a simple over/under-count | no integer combination of the two families fits (best-fit scales +0.23 / −0.10) |
| the t3 blocks or their reading | rank-6 singles/doubles *consume* t3 and are exact |

At `t1 = t3 = 0`, ccgen gives 4.93e-2 where PySCF gives 3.55e-3 (~14×), and every ccgen triples term
outside the pure-t2 class evaluates to exactly 0. The two surviving families cancel heavily — each
~4.5e-2, summing to 1.3e-2, with 38× cancellation across the full residual — so dropping either
makes agreement *worse*. Max-norm is a poor instrument on a quantity this cancellation-dominated.

**The blocking problem, and it is not what it first looked like.** The third source that would say
which side is wrong is ccgen's own RCC manifold — independent of both the UCC bridge and PySCF. It
reproduces `triples_aabaab` to 1.6e-17, but only at converged amplitudes where ‖R‖~1e-10, i.e.
vacuously. Making it non-vacuous requires perturbed amplitudes, and **that is where the closure
relations become load-bearing**:

```
RCC  reads a single spatial  t3
UCC  triples_aabaab needs    t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb
```

The four UCC blocks are not independent at closed shell — they are determined by the spatial one.
So a wrong closure relation is **indistinguishable from an equation defect** in this comparison, and
that is exactly the trap that produced the retracted "rank-6 spin-flip defect" earlier in this
ladder. This is not a new kind of problem: rank 4 has the identical structure, and F2.3 passes there
only because its closure (`t2_aaaa = t2 - t2.transpose(1,0,2,3)`) is known and simple.

**What is established about the rank-6 closure:**

```
t3_aaaaaa = (1/12) * A_bra A_ket t3_aabaab      # all 36 signed perms
```

It reproduces both real block pairs exactly (1.3e-18, 2.0e-18), and the 1/12 double antisymmetrizer
maps the mixed block's symmetry onto a fully antisymmetric one for a **generic** random block — so
that half is structural, not a fixture artifact.

**What is not established: that it is unique** — and the obvious way to settle it does not exist.
The real `aab` block's 36 permutation images span rank **5** where a generic block gives rank 9, so
the fit is underdetermined and the uniform-1/12 answer is just the min-norm solution. The
`bbb`←`bba` cross-check does **not** repair this: a deliberately different exact fit (`c0 + 3·n`,
`n` in the null space) passes it too, because both pairs share the null space. And refitting on an
open-shell reference — where α and β would be genuinely independent — is impossible: their spaces
have **different dimensions** there (CH3/STO-3G: `aaa` (5,5,5,3,3,3) vs `bbb` (4,4,4,4,4,4)), so
only 4 of 36 images are shape-legal and they span rank 1. **The permutation-fit approach is
closed-shell-only by construction.**

One wrong turn recorded: the extra degeneracy was guessed to be the joint `(0,1)(3,4)` spin-flip
symmetry, which the real block does satisfy exactly — but imposing it on a random block still gives
rank 9, so the source of the rank-5 collapse is unidentified.

**Steps, in order:**

- **U1.4c.1 — derive the closures from the spin integration, not by fitting (~M). LANDED.**
  `t3_aaaaaa` comes straight from `_split_same_spin_amplitude`, with the one reading that makes it
  work: **the splitter permutes BASE INDICES, not array axes**, and its output feeds an einsum keyed
  on index names against a fixed output order — so the base reordering applies the **inverse**
  permutation to the array. Forward it fails by the block's full magnitude; inverse is exact to
  4.8e-18. `t3_abbabb = bba.transpose(5,2,3,4,0,1)`, since PySCF's `bba` is 2-β-1-α in layout
  `[i,j,a,b,k,c]`; the check that picks it out is that it carries `abbabb`'s **own** antisymmetry
  (the β pairs, vir (1,2) / occ (4,5)) where every earlier candidate carried `aabaab`'s.

  **The previously committed fitted closure was wrong**, and this is the part worth remembering: the
  uniform-1/12 antisymmetrizer reproduced *both* real block pairs exactly and still is not the
  closure — on a generic block the two differ by ~80%. It agreed only because the fixture's
  permutation images span rank 5 against a generic block's 9. A relation that reproduces every case
  in a degenerate fixture is not thereby derived.
- **U1.4c.2 — extend F2.3's closed-shell oracle to rank 6 (~S).** With correct closures, feed both
  sides *perturbed* amplitudes and compare `triples_aabaab` against RCC `triples`. *Gate:* ≤1e-11 on
  a non-square case, plus the falsifiability check F2.3 carries.
- **U1.4c.3 — decide which side is wrong (~S, a decision, not code).** If U1.4c.2 passes, ccgen's
  T3 is self-consistent and the defect is in how this gate reads PySCF's `r3aaa`; if it fails, the
  defect is in ccgen's T3 and U1.4c.2 localizes it without PySCF at all. **Do not attempt this call
  before U1.4c.2** — it is the question four wrong answers in this ladder were guesses at.

**Do not** resume the max-norm bisect on the PySCF comparison. With 38× cancellation and an unproven
closure feeding it, that instrument cannot separate the two candidate causes; U1.4c.2 can.

#### The oracle plumbing, established while probing (reusable when U1.4a lands)

- PySCF's real CCSDT residual entry is **`update_amps_uccsdt_tri_(mycc, tamps, eris)`**, which
  mutates `tamps` in place adding `R/D`. `UCCSDT.update_amps` is the *inherited CCSD* one and does
  **not** include t3 — a trap, since it exists and runs.
- t3 is stored **packed** (triangular); `tamps_tri2full_uhf` unpacks it. Perturbing the packed array
  elementwise does **not** correspond to a valid antisymmetric t3, so perturb t1/t2 only.
- Block mapping to ccgen layout, verified exhaustively over all 720 axis permutations:
  `aaa` → `transpose(3,4,5,0,1,2)`; `aab` **and** `bba` both → `transpose(2,3,5,0,1,4)` (`bba` is
  stored spin-flipped in place, matching `aab` to 1.5e-17 at closed shell — the guess that its α line
  sat in different slots was wrong).

#### U1.4a (superseded) — extend the rank-6 tensor bundle (~S)

> Kept for its content; it is no longer the first step. The fixture work below is small and correct,
> but the **spin-flip asymmetry above must be fixed first** — a gate built on defective equations
> would pin the defect.


The UCC manifold needs `t3_aaaaaa`, `t3_aabaab`, `t3_abbabb`, `t3_bbbbbb` (measured: exactly these
four, against RCC's single bare `t3`). Reuse `_uccsdt_so_tensors`' spin-orbital read rather than
constructing spatial blocks — that read is already pinned by `map.3` and is what the splitter gate
evaluates against.

*Gate:* per-block shapes, and each same-spin block satisfies its own bra/ket antisymmetry to 1e-14.
Same shape of check F1 uses, for the same reason: a fixture that silently violates a block's own
symmetry makes every downstream comparison meaningless.

#### U1.4b — the direct rank-6 oracle against PySCF UCCSDT (~M, the deliverable)

Mirror U1.2 exactly: evaluate every rank-6 UCC target at PySCF's perturbed UCCSDT amplitudes and
compare against `UCCSDT.update_amps`, converting `t_new` back to a residual via `R = (t_new - t)·D`.

Carry U1.2's three hard-won conventions rather than rediscovering them:

- **`f_ov` zeroed on BOTH sides** — one-sided zeroing is *worse* than neither.
- **Layout is a transpose**, not a rename — PySCF is `(occ…, vir…)`, ccgen is `(vir…, occ…)`.
- **Perturb off convergence**, and re-impose each block's antisymmetry after perturbing.

*Gate:* the tolerance U1.2 actually reached (~1e-13), not the 1e-10 originally scoped — unless rank 6
demonstrably cannot hold it, in which case say so with the measured number rather than loosening
silently.

**Check `uccsdt`'s amplitude storage before trusting the mapping.** `_uccsdt_so_tensors`' own
docstring records that UCCSDT stores `t2ab` as `[i,a,j,b]` — unlike both `rccsd` and
`pyscf.cc.uccsd`'s `[i,j,a,b]`. That is exactly the class of convention defect U1.2's transpose
correction was, and it is already known to differ in this module.

#### On the closed-shell oracle at rank 6

Optional, and no longer load-bearing now that a direct oracle exists. It stays cheap (`triples_aabaab`
against RCC `triples`, F2.3's per-target pairing) and it localizes a defect to the evaluator rather
than the equations, which is why F2.3 was worth doing before F3. Add it if U1.4b fails and the cause
is unclear; skip it if U1.4b passes.

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
        │              [t3 closure: NOT work — pinned at test_spin.py:2158]
        │              U1.4a extend the fixture to t3   ~S
        │              U1.4b direct oracle vs UCCSDT    ~M  ← the deliverable
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

U1's gate now runs at rank 6. Singles and doubles reproduce PySCF UCCSDT exactly (~5e-16 / ~1.4e-15);
**the triples target is off by ~1.2e-2 and is the one genuinely open item in U1.** Because the
singles and doubles consume t3 and are exact, that discrepancy is confined to the T3 equation rather
than to the fixture, the evaluator or the block naming — which is the whole reason the gate reports
per target instead of in aggregate.

Second lesson, from the same session: the first rescope of U1.4 asserted a blocker ("the t3 closure
is underived") on the strength of three failed hand-derivations, without checking whether the tree
already had one. It did. **A failed guess is evidence about the guess, not about the tree** — the
same shape of error as trusting a status header, arrived at from the other direction.

Third lesson, and the sharpest of the three: **a cheap invariant check is only as trustworthy as its
fixture.** The α↔β symmetry check was the right instrument — free, no PySCF, decisive in one run —
and it fired correctly. The conclusion drawn from it was still wrong, because the fixture feeding it
had a bad block, and a bad fixture and a bad equation produce the *same* symptom.

What separated them was cheaper than any of the debugging that followed: check the **fixture's own
invariants** before believing what it says about the code. A block must satisfy the antisymmetry its
own tag implies; the bad `abbabb` failed that by 1.5e-2, needing no equations and no evaluator. That
check now exists, and it should be the first thing added whenever a fixture grows a new block.

The three lessons rhyme. Do not trust a status header without running its gate; do not trust a failed
guess as evidence about the tree; do not trust a passing instrument without checking its inputs. Each
one cost a wrong conclusion that measurement, one step earlier, would have prevented.

---

See `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (U0 landed, U2–U5 ahead, and the rescope that corrected
`external_blocks`), `CCGEN_SPIN_ADAPTER_CONTRACT.md` (the numeric-over-symbolic lesson), and
`spin.py:955` (`_factor_tensor_name`) / `spin.py:1088` (`block_keyed_intermediate_name`).
