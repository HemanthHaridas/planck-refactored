# Arbitrary-order UCC kernel generation and execution

Scopes one capability: **generate and run arbitrary-order *unrestricted* CC
kernels (UCC) alongside the existing arbitrary-order RCC path** — so an
open-shell reference can drive `ucc4`/`ucc5` the way a closed-shell reference
drives `cc4`/`cc5` today.

**Status, 2026-08-23.** **U0–U4 are landed and numerically validated.** The generator emits a
runnable UCC translation unit behind `--ucc`, and the runtime accepts, evaluates, updates and
allocates it. **U5 is all that remains, and it is bigger than this doc said** (`~S` → `~M`): the
three UCC C++ builders U2–U3 added have **no production callers at all** — `prepare_generated_
arbitrary_order_state` still builds a `CanonicalRHFCCReference` and the RCC block cache. Wiring
that, not the keyword, is the work.

| step | state |
|---|---|
| U0 | landed — `ucc_independent_blocks`, `_ucc_block_tag`, `external_blocks(fold_spin_flip=…)` |
| U1 | **landed** — `ucc_adapt_equations` + `ucc_spinterm_to_algebraterm`, validated against PySCF UCCSD at rank 4 (~6e-16) and against GCC-sliced at rank 6 (1.6e-17). U1.3 turned out to be **dead**, not work: U1.1 designed its hazard out. Detail in `CCGEN_U1_UCC_ADAPT_SCOPE.md` |
| U2 | **landed** — U2.1 `build_ucc_block_denominator`, U2.2 `build_ucc_denominator_cache` + `ArbitraryOrderDenominatorCache::{sectors,sector_tensor}`. RHF path bit-identical, measured. The one remaining item this doc used to name — a reference *variant* — is **withdrawn**; see U2 |
| U3 | **landed, U3.0–U3.4.** Spin-blocked ERIs and Fock, emitter routing, and the open-shell MP2 limit. The emitted UCC TU now has ZERO untagged reads of either kind, and the U2.0 pre-gate is green. Two things the scope did not predict: the per-tag block set had to be **derived** (6 same-spin, 10 mixed) because a mixed block's orbits reach only 11 of 16 patterns, and U3.4 turned out to need **no solver at all** |
| U4 | **landed, U4.0–U4.3.** The runtime accepts an ALL-SECTORS bundle (no per-rank reference residual), and `--ucc` reaches it from the build. U4.1 turned out **not to be work** — the update loop already handled it; U4.2 fixed a real out-of-bounds read (segfault, not a wrong number) that U4.0 had made reachable |
| U5 | **U5.0 + U5.1a landed** — distinct UCC symbols + filename (the two TUs co-link), and the 24-block canonical set, gated against ccgen on both sides. U5.1b–U5.5 open. Rescoped ~S → ~M: `build_ucc_{spin_block_cache,fock_blocks,denominator_cache}` are called **only from tests**, so U5 is prepare-path wiring plus a UHF reference, not just a keyword |

**Read `docs/CCGEN_UCC_NUMERIC_VALIDATION.md` first** if you are touching the UCC validation
story: it carries the three independent correctness routes, the interface conventions that cost the
most time, and the one open thread below.

All UCC gates are green, including the rank-6 PySCF comparison (triples 2.3e-15) and a direct UCC
FCI-limit check (3.7e-14).

> **Terminology, and it is a trap.** In this repo "**adapt**" means `spin_adapt_equations` — the
> **spatial collapse** that folds spin blocks into one tensor per rank. **UCC does the opposite**:
> it keeps blocks resolved (`t2aa`/`t2ab`/`t2bb` as separate arrays) and its defining property is
> *skipping* the three collapse steps. The names below (`ucc_adapt_equations`, "the UCC adapt
> entry", this doc's `_ADAPT_` sibling filename) are inherited from the RCC path and read backwards.
>
> **The constraint they obscure: UCC is spin-block resolved, never spatial.** Anything that folds
> α↔β, folds β-majority onto α-majority, or maps a block onto a single reference layout is a
> closed-shell assumption and is wrong here — see the audit of the three such mechanisms below.
> When implementing, prefer a name that says what it does (`ucc_resolve_equations` /
> `_resolve_on_block`) over the inherited "adapt".

---

## Rescope against the current tree (2026-08-12, post-V1.1e)

Re-verified after the V1.1e work (e.2.1 adapter orientation invariance, e.2.5 fixture
fix). **The plan holds; three corrections and one simplification.**

Verified still true — every claimed Python entry exists (`ucc_manifold`,
`ucc_integrate_term_antisym`, `block_exists`/`resolve_block`,
`spinterm_to_algebraterm`, `independent_spin_blocks`, `_amplitude_block_tag`, all three
collapse steps, `_adapt_on_block`), and so does every claimed C++ hook
(`ArbitraryOrderRCCAmplitudes::sectors` at `amplitudes.h:70`, `sector_tensor` at `:81`,
`sector_tags`/`sector_residuals` at `generated_arbitrary_runtime.h:55-56`,
`UHFReference`'s four spin counts at `common.h:145-148`, and `MOBlockCache`'s spin-free
single `oovv`/`ovov` at `mo_blocks.h:20-21`). The reuse surface is intact.

### Correction 1 — `external_blocks` folds a↔b, so U0 cannot build on it as written

Measured on the current tree:

```
singles: ['aa']                    ← no 'bb'
doubles: ['aaaa', 'abab']          ← no 'bbbb'
```

`external_blocks` (`spin.py:409`) dedups **up to a global a↔b flip**
(`key = min(combo, flip)`, line 425) — a closed-shell assumption, and precisely the leak
U0 exists to prevent. PySCF UCCSD stores `t1a/t1b` and `t2aa/t2ab/t2bb`, so UCC needs
`{aa, bb}` and `{aaaa, abab, bbbb}`.

**Consequence for U0's gate as written.** It says "every block returned by
`external_blocks` for that rank folds into exactly one returned tag" — which *passes
vacuously* while `bbbb` never appears, because `external_blocks` already folded it away.
U0 must therefore either add an unfolded sibling (`ucc_external_blocks`, no `min(combo,
flip)`) or take a `fold_spin_flip=False` flag, and the gate must assert **against PySCF's
block names directly** (`{aaaa, abab, bbbb}`), never against `external_blocks`'s output.

This raises U0 from ~S to **~S/M** — still small, but it is now a real code change plus a
gate rewrite rather than a pure addition. It also confirms U0's stated rationale
("the one place a closed-shell assumption could silently leak in") was right, and that
the leak is already there.

### Correction 2 — U1's numeric gate now has a working fixture (and needs its symmetry)

The `residual_eval` fixture violated `<pq||rs> = <rs||pq>` until e.2.5.0. Any UCC gate
comparing two written forms on that fixture would have reported spurious differences.
Fixed, with all symmetries asserted by `test_residual_symmetry.py`.

**For U1: the "evaluate at PySCF's converged amps" gate is the right instrument** (it
compares *values*, not written forms), and it is now safe to also use symmetry-correct
random tensors as a cheap pre-check before the PySCF solve. Do **not** gate UCC on a
symbolic term-multiset comparison — V1.1e spent e.2.0–e.2.5 learning that such a
comparison reports differences that are not there whenever both sides may choose among
symmetry-equivalent writings.

### Correction 3 — the orientation fix is inherited, so one U1 risk is already retired

`ucc_integrate_term_antisym` calls `_antisym_to_allowed` (`spin.py:382`), which since
e.2.1 normalizes every rank-4 `v` to one canonical member of its 8-fold ERI orbit. **The
whole UCC path inherits orientation invariance for free** — no UCC-specific work, and the
class of bug where two writings of one integral land in different blocks cannot recur
here.

This does not retire the `_canonicalize_amplitude_factor` risk U1.1 names, which is about
**amplitude** slot layout per block, a different mechanism. That one still needs its own
assertion.

### Simplification — `independent_spin_blocks` is closer to reusable than stated

The doc says the *machinery* is reusable but the *policy* (β-majority fold, exclude
all-α) is closed-shell-specific. Confirmed. Worth stating positively: U0's job is
literally "`independent_spin_blocks` minus the two policy lines", so the natural shape is
a shared enumerator with the fold as a parameter, not a second parallel function. Fewer
places for the closed-shell assumption to hide.

---

## The one-sentence framing

UCC is **not a parallel pipeline**. RCC is *UCC plus a collapse step*: the
spin-adaptation layer already integrates every GCC term into spin-resolved
blocks (`ucc_manifold`, landed and PySCF-transitively validated), and then RCC
throws the block resolution away via three closed-shell steps. UCC is what you
get by **not** running those three steps and keeping the block tag on the
tensor name.

So the Python work is mostly *subtraction*, and the C++ work is mostly
*reusing the multi-sector machinery that already exists for the t4 `aaabaaab`
Sz sector*.

---

## What already exists (the reuse surface)

### Python (`python/ccgen/spin.py`, 1269 lines)

- **`ucc_manifold(terms, residual_template)`** — the UCC block manifold. Returns
  `{block_tag: [SpinTerm]}` for **every** surviving external block, α/β
  distinct. This is the UCC equation set. It is validated: the scope doc records
  the decision that no separate PySCF `uccsd.update_amps` gate is needed because
  `ucc_manifold == GCC-sliced` and `GCC == PySCF-gccsd`, so
  `ucc_manifold == PySCF-uccsd` transitively.
- **`ucc_integrate_term_antisym(term, external_spins)`** — the per-term spin
  integration that carries the exchange (−K) correctly. Rank-general
  (`_antisym_to_allowed` gated numerically at rank-6 and rank-8).
- **`block_exists` / `resolve_block`** — spin conservation per line, one rule for
  every tensor kind, arbitrary rank. No per-tensor table.
- **`spinterm_to_algebraterm(spinterm, externals)`** — the bridge from spin-
  resolved `SpinTerm` to the flat `AlgebraTerm` the lowering/emit layers consume.
  Exact as of `ef42800` + `cfe302a` (718/859 → 0 failures). **The bridge drops
  the spin label** — that is precisely the thing UCC must change (see U1).
- **`independent_spin_blocks(rank)` / `_amplitude_block_tag(block)`** — the
  Sz-sector folding built for the closed-shell t4 case. The *machinery* is
  reusable for UCC block enumeration; the *policy* (fold β-majority to
  α-majority, exclude all-α) is closed-shell-specific.

### The three RCC-only collapse steps (what UCC drops)

| Step | Function | What it assumes |
|---|---|---|
| S2.2a | `canonicalize_spin_blocks` | flips a↔b freely — i.e. **α and β orbitals are identical** |
| S2.2b | `collapse_amplitudes` / `_split_same_spin_amplitude` | `t[aa..a]` is a signed combination of the mixed block — closed-shell only |
| S2.2c | `collapse_integrals` / `_split_vaaaa` | `v[aaaa] = v[abab] − v[abab](ket swap)` — closed-shell only |

`_adapt_on_block` runs all three in sequence. `spin_adapt_equations` calls
`_adapt_on_block` per sector. **UCC needs a sibling of `_adapt_on_block` that
runs neither collapse and keeps the block tag** — that is the core of U1.

### C++ (the part that is already multi-block)

The runtime is **already sector-keyed**, built for the closed-shell t4
`aaabaaab` case (commits `26041d7`…`dd45c1b`, gaps B1–B4):

- `ArbitraryOrderRCCAmplitudes::sectors` — `vector<pair<pair<int,string>, TensorND>>`,
  i.e. amplitude blocks keyed by **(rank, tag)** on top of the reference
  `by_rank` (`amplitudes.h:64-70`).
- `ArbitraryOrderRCCAmplitudes::sector_tensor(rank, tag)` — dense view of a
  tagged block, `expected`-returning.
- `GeneratedArbitraryOrderKernels::sector_tags` and `::sector_residuals`
  (`generated_arbitrary_runtime.h:36-56`) — the bundle declares which tagged
  blocks a method carries and supplies a residual kernel per tag.
- `ensure_amplitude_sectors(...)` — reconciles a prepared state's blocks against
  the bundle's declared tags, zero-init.
- `make_zero_rcc_amplitudes(..., sectors)` — sector-aware allocation.
- The solver evaluates each sector residual and Jacobi/DIIS-updates the matching
  block (B4, `993ca7d`).

**This is the exact shape UCC needs.** A UCC amplitude set is "one tagged block
per spin block per rank" — `t2aa`, `t2ab`, `t2bb` are three (rank=2, tag)
entries, structurally identical to `(4, "aaabaaab")`. **No new container, no new
solver loop, no new allocation path.** What is missing is dimensions (U3) and
the reference (U2).

### What is genuinely RHF-only (the real C++ work)

- `ArbitraryOrderTensorCCState::reference` is a `CanonicalRHFCCReference`;
  `generated_arbitrary_prepare.cpp:82` takes `RHFReference orbital_partition`
  with scalar `n_occ` / `n_virt`.
- `solver_arbitrary.{h,cpp}` takes `const RHFReference &reference` — the
  denominator cache is built from one spin-free `eps`.
- `MOBlockCache` (`mo_blocks.h:15-21`) holds **one** `oovv` / `ovov` — no spin
  blocks. UCC needs `oovv_aaaa/abab/bbbb` etc.
- `UHFReference` **exists** (`common.h:141-148`, with
  `n_occ_alpha/n_occ_beta/n_virt_alpha/n_virt_beta`) and is already used by the
  hand-written UCCSD/UCCSDT solvers — so the reference builder is not new work,
  only its wiring into the generated path is.

---

## Scope (small verifiable steps)

Ordered so each step is independently gated and the risky algebra lands before
the C++ plumbing.

### U0 — pin the UCC block vocabulary (~S/M, pure Python, no codegen)

> **Rescoped.** `external_blocks` already folds a↔b (measured: doubles →
> `['aaaa','abab']`, no `bbbb`), so this step must *also* provide an unfolded
> enumerator, and its gate must assert against PySCF's block names — **not** against
> `external_blocks`'s output, which would pass vacuously. See Correction 1. Prefer adding
> `fold_spin_flip=False` to the existing enumerator over writing a parallel one.


Write `ucc_independent_blocks(rank)`: the UCC blocks of a rank-2n amplitude that
must be **stored** (as opposed to derived). Unlike the closed-shell case there is
no a↔b flip available, so the α-count sectors `k = 0…n` are **all** independent
and the all-α / all-β blocks do **not** split away. For n=2 that is
`{aaaa, abab, bbbb}` (matching PySCF UCCSD's `t2aa/t2ab/t2bb`); for n=3,
`{aaaaaa, aabaab, abbabb, bbbbbb}`.

The within-half antisymmetry still folds slot permutations, so the tag is still
"α-before-β per half" — reuse `_amplitude_block_tag`'s *layout* convention but
**not** its β-majority flip.

*Gate:* `ucc_independent_blocks(4) == ["aaaa","abab","bbbb"]`, `(2) == ["aa","bb"]`, and
the rank-6/8 counts match `n+1` sectors. Assert the rank-4 set matches PySCF UCCSD's block
names (`t2aa/t2ab/t2bb`) **by name, directly** — this is the load-bearing assertion.
Seconds, no PySCF solve.

*Do NOT gate on* "every block from `external_blocks` folds into exactly one tag" (the
original wording): `external_blocks` folds a↔b itself, so that assertion passes while
`bbbb` is missing entirely. If a relationship to `external_blocks` is wanted, assert the
opposite direction — that the unfolded enumerator returns **strictly more** blocks than
`external_blocks`, and that folding its output by a↔b reproduces `external_blocks` exactly.

**Why first:** every later step keys off this vocabulary, and it is the one place
a closed-shell assumption could silently leak in.

### U1 — the UCC adapt entry (~M, the core algebra step)

Add `ucc_adapt_equations(equations)`, the UCC sibling of
`spin_adapt_equations`, returning `{target_tag: [AlgebraTerm]}` — e.g.
`doubles_aaaa`, `doubles_abab`, `doubles_bbbb`.

Two sub-parts:

- **U1.0 — a no-collapse `_adapt_on_block`.** A sibling that runs
  `ucc_integrate_target` → `merge_terms` → `spinterm_to_algebraterm` and
  **skips** `canonicalize_spin_blocks`, `collapse_amplitudes`,
  `collapse_integrals`. Driven once per block from `ucc_independent_blocks`
  rather than once per Sz sector.
### Audited 2026-08-16 — the three spatial assumptions U1 must neutralize

Probed against the tree rather than read. Two were already flagged; **the third was not**, and it
fires at rank 2 where the docs assumed the machinery was safe.

| # | mechanism | where | measured |
|---|---|---|---|
| 1 | `external_blocks` folds a↔b (`key = min(combo, flip)`) | `spin.py:409` | `doubles → ['aaaa','abab']`, no `bbbb` |
| 2 | `_amplitude_block_tag` folds β-majority onto α-majority | `spin.py` | `abbabb → aabaab` |
| 3 | **`_canonicalize_amplitude_factor` reorders *every* rank ≥ 2 amplitude onto one reference layout** | `spin.py:878` | `t2 baba` → slots reordered to `[j,i,b,a]`; `t2 aaaa` and `t2 bbbb` both emit as bare `t2` |

**Assumption 3 is the one to design against.** Its docstring is explicit that it exists because "the
spin→AlgebraTerm bridge drops the spin label, so unless every factor is first mapped to that one
reference layout, a factor read in a non-reference block indexes the wrong slice of **the single
spatial tensor**". That premise — one stored tensor per rank — is exactly what UCC removes. Two
consequences:

- The **α/β flip is only valid closed-shell.** The docstring says so itself: "A closed-shell
  amplitude is spin-flip symmetric (t[σ] = t[flip σ] index-for-index)". Under UCC `t2aa` and `t2bb`
  are different arrays and that identity is false.
- **`_factor_tensor_name`'s gate is `len(f.block) >= 8`**, so t1/t2/t3 can *never* receive a block
  tag. Measured: `t2 aaaa` and `t2 bbbb` both emit as bare `t2`. U1.1's naming fix must therefore
  **lower that gate to every rank**, not just generalize the tag — naming alone at rank ≥ 4 leaves
  rank 2 silently collapsed.

So U1.1 is not only "add the block to the name": it must **also disable the canonicalizer's
reordering for UCC**, or the name will be right while the slots have been permuted onto another
block's layout. These are one change — the canonicalizer and the naming both exist to service the
single-tensor assumption — and doing only the visible half is the failure mode.

Cheap gate, available before any C++: `t2 aaaa` and `t2 bbbb` must emit **distinct** tensor names
and **unpermuted** slots. Both are one-line assertions on `spinterm_to_algebraterm` output and
neither needs a solve.

- **U1.1 — spin-resolved factor names in the bridge.** `spinterm_to_algebraterm`
  currently drops the spin label, which is *correct* for RCC (everything lives
  in one spatial tensor) and *wrong* for UCC (`t2aa` and `t2ab` are different
  arrays). The bridge must fold each factor's block into its **name** —
  `t2` + block → `t2_aaaa`, `v` + block → `v_abab` — exactly as R3.1.3c already
  does for `t4_aaabaaab`. That precedent means the naming hook exists; UCC
  applies it to every factor rather than only to a higher sector.

  **The `_canonicalize_amplitude_factor` interaction is the risk — and it is now the
  *only* one of U1's two orientation-shaped risks left.** The `v`-orientation half is
  already handled: `ucc_integrate_term_antisym` → `_antisym_to_allowed` normalizes every
  rank-4 `v` to a canonical orbit member since e.2.1, so UCC inherits that invariance
  (Correction 3). What follows is a different mechanism — amplitude slot layout per
  block — and still needs its own assertion.

  It reorders
  slots to one reference layout per rank and returns a sign. For RCC that maps
  every block onto a single stored tensor. For UCC it must map each block onto
  **its own** stored tensor's canonical layout — same within-half sort, but the
  target tag is the factor's own block, not a global reference. Getting this
  wrong reproduces the R3.1.2 failure mode (a factor indexing the wrong slice,
  leaving a residual ≈0), so it needs its own assertion.

*Gate (structural):* for a closed-shell-degenerate input, U1 ∘ collapse ==
`spin_adapt_equations` at rank 4. Plus: every emitted `AlgebraTerm`'s factor names are
drawn from the U0 vocabulary, and no factor carries an unresolved block.

**Compare this NUMERICALLY, not as a term multiset** — evaluate both sides on
symmetry-correct random tensors (`residual_of` + `random_tensors`, both fixed in e.2.5.0)
and require agreement to ~1e-12 relative. V1.1e's entire e.2.0–e.2.5 arc was spent
learning that a symbolic term-by-term comparison reports differences that are not there
whenever two sides may choose among symmetry-equivalent written forms — `{"doubles": 14}`
was exactly that false alarm. The name/vocabulary half of this gate stays structural;
the equality half must be numeric.

*Gate (numeric, the load-bearing one):* evaluate the U1 `doubles_aaaa` /
`doubles_abab` / `doubles_bbbb` residuals **at PySCF UCCSD's own converged
`t1a/t1b/t2aa/t2ab/t2bb`** on an open-shell case, and compare against
`pyscf.cc.uccsd.update_amps`. This is the same "evaluate at PySCF amps"
convention-robust pattern the RCC S3.2 gate used, and it is the one place UCC
gets a *direct* oracle rather than a transitive one. Do this at rank 4 before
touching any C++.

### U2 — the UHF reference in the generated runtime (~M, C++) — **LANDED**

**U2.1 landed:** `build_ucc_block_denominator` (`src/post_hf/cc/amplitudes.{h,cpp}`), gated by
`planck-cc-ucc-denominator`. The spin-aware denominator this section calls for, keyed by block tag.

*The layout contract, which is the thing a caller gets wrong:* the tag is per-slot spin in the
**tensor's** index order, which is **occ-first then vir** (`rank_dims`). ccgen's UCC tags are
bra(vir)-half-then-ket(occ), so the caller converts and the tag reaching the builder is always
occ-half-first.

*Fixture note for whoever extends this:* the gate uses `noa=4 nva=3 nob=2 nvb=5` — all four extents
distinct — because with `n_occ == n_virt` a transposed slot still lands in bounds, and with
`noa == nob` a swapped spin does. Alpha and beta energies are ~100 apart so a spin mix-up is a
factor-100 error rather than a rounding one. **A tag whose two halves share a spin string
(`"abab"` is occ (a,b) *and* vir (a,b)) cannot catch a virtual slot reading the occupied slot's
spin** — that mutation passed until the `"abba"` case was added, where the halves differ.

**U2.2 landed:** `build_ucc_denominator_cache` plus
`ArbitraryOrderDenominatorCache::{sectors, sector_tensor}`
(`src/post_hf/cc/amplitudes.{h,cpp}`), same gate binary.

U2.1 built *one block's* denominator. The defect it could not reach was one layer up: the
Gap B4 sector update divided **every** sector by `denominators.tensor(rank)`, the rank's
*reference* denominator. The old comment stated the assumption plainly —

> with the SAME denominator as its rank's reference block (B2: for an RHF reference the
> orbital energies are spin-free, so a sector reuses `denominators.tensor(rank)`)

— which is true for RHF and false the moment the reference is unrestricted: `eps_alpha !=
eps_beta`, and with `noa != nob` an `abab` block does not even have the same *shape*
(`(noa,nob,nva,nvb)` against `(noa,noa,nva,nva)`).

Three call sites moved off `tensor(rank)`: the B4 Jacobi update, the sector residual shape
validation, and `ensure_amplitude_sectors`' block allocation — the last because a UCC
block's dims are known only from its own denominator.

**One code path, not two.** `sector_tensor` falls back to `tensor(rank)` when no per-block
entry is stored, which is exactly the RHF case (spin-free energies ⇒ every sector of a rank
really does share one denominator). So the call sites switched unconditionally rather than
branching on reference kind — a second sector-update loop would be a second thing to keep in
sync with `ensure_amplitude_sectors`. `by_rank` is deliberately left **empty** on a UCC
cache: UCC has no privileged reference sector, so an undeclared block must error rather than
quietly pick up another block's tensor.

*RHF bit-identity, measured not assumed.* `be_rccsdtq_sto3g` is the only landed method
carrying a sector (`{4, "aaabaaab"}`), so it is the case that could have moved. Built both
ways: `-14.4036550465` with the change and `-14.4036550465` without, every digit. Extended
suite 107/107. (The 6.2e-8 gap to that case's stored expectation is pre-existing, inside its
own 1e-7 atol, and unrelated.)

*The gate was verified falsifiable* against three mutations, each caught by the assertion
written for it: `sector_tensor` ignoring the tag (i.e. exactly the pre-U2.2 behavior, 3
checks); the cache building every entry from the first tag (6 checks); the RHF fallback
removed (3 checks).

#### The reference *variant* is withdrawn — do not build it

This section used to end by calling for `ArbitraryOrderTensorCCState::reference` and
`solver_arbitrary`'s `const RHFReference &` to become "a variant/parameterized partition".
**Do not do this.** Measured against the emitted TU, the generated kernels only ever touch
`reference.f_oo`, `f_ov`, `f_vv` and `orbital_partition` — never `RHFReference` *as a type*.
A variant would therefore change every kernel signature and force every generated TU to
regenerate, and buy nothing: the actual spin-resolution those three Fock blocks need is the
same split the ERIs need, driven by the same emitter tag. That is U3, and it is one change.

What the variant *was* meant to deliver — spin-resolved denominators — is what U2.2 landed,
without touching a kernel signature.

*Gate, still open and still worth doing:* the open-shell MP2-limit check — the rank-2 UCC
denominators reproduce the existing UMP2 correlation energy from a single Jacobi step. It
isolates the denominator from the residual algebra. It needs U3 first, because a Jacobi step
needs correct integrals as well as correct denominators.

**Sequencing note.** Do U2 *after* U1's numeric gate. If U1 is wrong, a UCC C++
run fails and you cannot tell whether the algebra or the reference is at fault.

### U3 — spin-blocked MO integrals **and the emitter half** (~M/L, C++ *and* Python)

> **Rescoped twice.** As originally written this step was C++-only, which is not enough (the
> emitter half is a silent wrong-answer path). Scoping it properly then turned up a *second*
> defect the emitter half was assumed not to have, and a structural fact that makes the C++
> half bigger than "three copies of the RCC cache". All three are measured below.

#### The three measurements this step is built on

**(1) The mixed block needs almost twice the space patterns.** Distinct space patterns
reaching each `v_<tag>` factor in the CCSD UCC manifold:

```
v_aaaa   7   oooo ooov oovv ovoo ovov ovvv vvvv
v_abab  13   + oovo ovvo vooo voov vovo vovv
v_bbbb   7   (same 7 as aaaa)
```

The extra six are the ones the 8-fold ERI symmetry folds away for same-spin and cannot fold
for mixed spin. Any sizing, memory estimate or block table that assumes "same 7 patterns,
three times over" is wrong.

**(2) Two of the emitter's four ERI permutations are INVALID for `abab`, and 37 of 142
`abab` reads use them today.** `_ERI_SYMMETRY_PERMUTATIONS`
(`python/ccgen/emit/planck_tensor_cpp.py:59`) is applied spin-blindly. Per block, a
permutation is valid iff it maps the spin tag to itself:

| permutation | `aaaa` | `abab` | `bbbb` | `abab` reads |
|---|---|---|---|---|
| identity `(0,1,2,3)` | ok | ok | ok | 92 |
| particle `<qp\|sr>` `(1,0,3,2)` | ok | **→ `baba`** | ok | **24** |
| bra↔ket `<rs\|pq>` `(2,3,0,1)` | ok | ok | ok | 13 |
| product `<sr\|qp>` `(3,2,1,0)` | ok | **→ `baba`** | ok | **13** |

So **37 of 142** mixed-block reads are currently routed through a symmetry that only holds
for a *different* spin block. This is a second defect, independent of the name-collapse the
pre-gate pins — **suffixing the array name without fixing this leaves all 37 wrong**, and
they would be wrong in the quiet way (right array, permuted indices).

Verified numerically on random real orbitals, not just by tag algebra:
`baba == abab.transpose(1,0,3,2)` true; `abab == abab.transpose(2,3,0,1)` true;
`abab == abab.transpose(1,0,3,2)` **false**.

**(3) The spin-blocked cache is not three copies of the RCC cache.** The spin lives on the
**chemists' charge-density pair**, not on the physicist block. Physicist `oovv_abab` is
chemists `(i_α a_α | j_β b_β)` — a *mixed* (α-pair | β-pair) transform, which is not
`chem.ovov` of either pure spin. `rebind_physicist`
(`generated_arbitrary_prepare.cpp:40`) already carries an `oovv`↔`ovov` cross-source, and
that is exactly where this bites: the source must be chosen by pair spin, not by physicist
tag.

*Memory:* ~3× the RCC block footprint — three pair-spin variants, since `(a|b) ≡ (b|a)` by
the chemists' bra↔ket swap. Exactly 3× only when `noa == nob`; it is not FLOPs but memory
that caps the reachable rank.

**Do not store `baba`.** It is `abab` under the particle swap (verified above), so storing
it costs ~33% more memory to avoid one explicit swap in the emitter. The emitter must apply
that swap **knowingly**, which is what U3.0 exists to make possible.

#### Steps — all landed

**U3.0 — the spin-aware ERI symmetry group** (`eri_permutation_preserves_block` /
`eri_permutations_for_block`, `test_ucc_eri_symmetry`). A permutation is usable on a block
iff it maps the tag to itself. Same-spin keeps all four; `abab` keeps identity and bra↔ket.
Grounded numerically on random real orbitals, not tag algebra alone.

**U3.1 — the spin-blocked ERI cache** (`TensorCCBlockCache::spin_blocks`,
`build_ucc_spin_block_cache`, `planck-cc-ucc-spin-blocks`). Its own TU
(`ucc_blocks.cpp`) so the block/transform logic links without `ensure_eri` dragging in the
Calculator, AO engine, RI, basis parsing and symmetry.

**U3.2 — the emitter routes ERIs by block** (`_map_eri_tensor` takes the tag;
`_eri_view_bindings` binds one view per (space, tag), mirroring the amplitude sector views).

> **The scope missed a third change here, and the emitter failing loudly is what surfaced
> it.** `_CANONICAL_ERI_BLOCKS` is spin-free and its 8-fold orbit reaches all 16 o/v
> patterns; a mixed block's orbits are smaller and reach only 11. The first UCC emit after
> restricting the permutations therefore raised `NotImplementedError` on `vovv` — exactly
> the coverage loss U3.0 had measured, now actually firing. `_canonical_eri_blocks_for`
> derives the stored set from each tag's own group: **6 same-spin, 10 mixed**, matching
> U3.1's cache. Three sides of the codegen boundary, one fact.

**U3.3 — spin-resolved Fock blocks** (`CanonicalRHFCCReference::spin_blocks`,
`build_ucc_fock_blocks`, `_fock_view_bindings`). Simpler by nature: the Fock is two-index,
so both slots carry the same spin — no mixed block, no permutation question. `vo` still
folds onto `ov` because the Fock is symmetric, and that reorder is spin-safe precisely
because a two-index tag cannot mix spins the way `<ab|ab>` does. This is where the spin
resolution **withdrawn from U2's reference-variant** belongs, and it costs no kernel-
signature change.

**U3.4 — the open-shell MP2 limit** (in `planck-cc-ucc-spin-blocks`).

> **Scoped as "reproduce UMP2 from a single Jacobi step"; that is not what it needed to be,
> and the difference matters for sequencing.** A Jacobi step needs the solver, a
> `UHFReference` threaded through the generated runtime, and an SCF — i.e. it needs U5, so
> as scoped U3.4 could not have closed before U5. But first-order MP2 amplitudes are
> `t2 = <ij||ab>/D` in closed form, so the energy can be assembled directly from U3.1's
> integrals and U2.1's denominators and compared against an independent transform. Stronger
> (it fails if either half is wrong **or** if both are right but misaligned) and cheaper
> (no solver, no SCF, no PySCF; runs in 0.01 s).
>
> The correspondence it rests on, verified against `mp2_internal.cpp` rather than assumed:
> UMP2's `ovOV = transform_eri(eri, nb, Ca_occ, Ca_virt, Cb_occ, Cb_virt)` is the same four
> matrices in the same order as U3.1's `oovv_abab`, so the mixed block must agree with the
> production UMP2 mixed-spin ERI exactly.
>
> A closed-form gate like this invites one specific failure — two zeros agreeing — so both
> channels are asserted non-trivially non-zero, and the spin orbital energies are
> deliberately non-degenerate so an aa/bb mix-up moves the answer instead of cancelling.

##### Two gate defects found by mutation, worth keeping

Both would have certified a broken fix, and neither was visible by reading:

- `test_ucc_eri_symmetry`'s original assertion **re-implemented the routing inline** against
  the module constants instead of calling the emitter. It measured a simulation of the old
  code and stayed red at exactly 37 even after `_map_eri_tensor` was fixed. *A gate that
  cannot observe the fix cannot certify it.*
- A mutation restoring the spin-blind permutation list **survived** the rewritten gate: it
  still binds only legitimate blocks, because the per-tag block set is wide enough to
  express its answer. What moves is which array each *read* names and in what *index order*
  (`v_abab_vovv(a,j,b,c)` vs `v_abab_ovvv(j,a,c,b)`) — invisible to anything inspecting the
  bound set. `test_every_read_is_the_one_the_valid_group_selects` replays the routing per
  block and compares every emitted read.

Also: once both `mo_blocks` and `reference` exposed `spin_block`, the ERI assertions began
counting Fock views as ERI blocks. Anchor such regexes on the receiver.

### U4 — the runtime accepts an all-sectors bundle (~M, C++ *and* Python) — **LANDED**

> **Scoped as "~S: add a `--ucc` switch, plus decide whether an empty `residuals_by_rank`
> needs a solver guard." It was not a guard question.** `validate_kernel_bundle`
> **required** `residuals_by_rank.size() == max_excitation_rank`, and a UCC bundle pushes
> **zero** per-rank residuals (every target is block-tagged: `doubles_aaaa`, never a bare
> `doubles`) against a `max_excitation_rank` of 2. So it was rejected before it could run —
> a structural mismatch, not a missing check.

**The cheaper fix does not work, and was ruled out first.** Promoting one block per rank into
the reference slot fails because that slot is sized by `rank_dims`, which yields **one shape
per rank**, while UCC blocks of a single rank have different shapes under UHF: `aaaa` is
`(noa,noa,nva,nva)`, `abab` is `(noa,nob,nva,nvb)`. Promoting one would silently mis-size the
other two.

So `residuals_by_rank` became **optional**: empty declares an all-sectors bundle
(`is_all_sectors()`) and every excitation is driven through `sector_residuals`. **Empty or
full, never partial** — a half-filled vector means a bundle lost a kernel, and that rank would
otherwise evaluate as a silent zero.

**U4.0 — validation and evaluation.** Two guards narrowed to RCC rather than deleted (the
residual count, and the "allocated through `max_excitation_rank`" check, which counts
*reference* blocks and so reads 0 for an all-sectors state by construction — its all-sectors
equivalent is the Gap B4 loop, which is stricter because it checks per (rank, tag) rather than
counting). One guard **added**: an all-sectors bundle with no sector residuals is now rejected,
since it would evaluate nothing and "converge" instantly at the reference energy.

**U4.1 — NOT WORK, and worth recording as such.** Scoped as "make pack/unpack and the update
loop tolerate an empty `by_rank`". Probing first showed they already do: the pack/unpack loops
iterate `by_rank` then `sectors`, so an empty `by_rank` simply contributes nothing and the
sector region starts at offset 0; the update's per-rank loop runs `rank <= max_rank()` = 0 and
is skipped; the rank-coverage guard compares 0 == 0 == 0. **U4.0's own gate had asserted the
update was unreachable**, inferred from `max_rank()` returning 0 — that inference was wrong and
the probe corrected it. U4.1 is therefore assertions, not a change. Pinned rather than left
implicit, because it holds only through two unrelated loops happening to be empty-tolerant.

**U4.2 — allocation, and a real out-of-bounds read.** `ensure_amplitude_sectors` already sized
UCC blocks correctly (U2.2), but its fallback indexes `by_rank[rank-1]`, which **does not exist**
on an all-sectors state — reachable the moment U4.0 let one through validation. Confirmed real,
not theoretical: removing the guard **segfaults** the gate (exit 139). Now skips the block, which
`validate_kernel_bundle` then rejects *by name* with its (rank, tag).

**U4.3 — the switch.** `ucc_adapt_equations` had **zero non-test callers** until this: everything
U0–U3 built was reachable only from the test suite. `--ucc` is mutually exclusive with
`--spin-adapt` and **raises** rather than picking a winner — the two resolve spin in opposite
directions (adaptation collapses blocks into one spatial tensor per rank, UCC keeps them
resolved), so running both would collapse and then attempt to re-resolve, which is not a
composition in either order. Intermediates are forced off, mirroring the `spin_adapt` precedent.
Default output verified **byte-identical** by regenerating with the change stashed and diffing —
not by checking for a marker, which would miss a change elsewhere in the file.

*Gates:* `planck-cc-all-sectors-bundle` (validation, evaluation, update, allocation) and
`test_ucc_emit_flag.py` (the switch). Both verified falsifiable; the mutations that mattered were
over-relaxing a guard, a silent skip in the sector update that still returns success, and making
`--ucc` a no-op.

> **Two fixture defects found by mutation**, both making an assertion pass for the wrong reason:
> `resize(1)` leaves a **null** `std::function`, so a "partial is rejected" case was caught by the
> missing-kernel guard rather than the count guard; and that fixture then used an all-sectors
> *state* against a non-all-sectors *bundle*, so rank-coverage fired first. Both assertions now
> **name the guard** they intend to exercise. Where two guards can reject the same input, asserting
> only "it was rejected" tests nothing in particular.

### U5 — driver routing + the end-to-end gate (~M, was ~S) — **U5.0 + U5.1a landed**

> **Rescoped `~S` → `~M`.** The original text said "the solver loop is unchanged — it already
> iterates tagged blocks", which is true and is not the work. Four measurements set the shape.

#### The four measurements

**(1) The three UCC C++ builders have no production callers.** `build_ucc_denominator_cache`,
`build_ucc_spin_block_cache` and `build_ucc_fock_blocks` are referenced only from `tests/`.
`prepare_generated_arbitrary_order_state` still calls `build_canonical_rhf_cc_reference` and
`build_tensor_cc_block_cache`, unconditionally. Wiring that is the work; the keyword is the
smallest part of it.

**(2) The RCC and UCC TUs COLLIDE, and would overwrite each other.** Measured per method:

```
ccsd    (RCC emits no bundle below the arbitrary floor)   1 colliding symbol
        compute_ccsd_energy
ccsdt   (bundles on both sides — the general case)        2 colliding symbols
        compute_ccsdt_energy,  make_generated_ccsdt_kernels
ccsdtq  (confirms the pattern, not a third case)          2 colliding symbols
        compute_ccsdtq_energy, make_generated_ccsdtq_kernels
```

Measured at three ranks rather than generalized from one: rank 2 is the outlier (one
collision, because RCC emits no bundle there), and ranks 3 and 4 agree exactly — the energy
kernel plus the bundle factory.

and the generator wrote **one filename per method** (`{method}_planck_generated.cpp`), so a UCC
build would have overwritten the RCC TU *before* the link could even fail. UCC therefore needed
distinct kernel names **and** a distinct filename — not just a distinct flag. **Fixed in U5.0**;
kept here because it is what ordered the steps.

**(3) The registry is rank-keyed only.** `make_generated_rcc_kernels(int rank)` switches on
rank behind `PLANCK_CC_MAXORDER` and has no notion of reference type, so UCC needs a sibling
entry point rather than an extra `case`.

**(4) The keyword path has a clean precedent.** `cc3`…`cc6` all map to one enum value with the
excitation rank carried separately on `OptionsSCF::_cc_generated_rank`. `ucc2` / `ucc4` follow
it exactly — no new enum member per rank and no new driver branch.

#### Steps

**U5.0 — distinct symbol names for UCC kernels — LANDED.** Emits
`compute_ucc_<method>_*` / `make_generated_ucc_<method>_kernels` into
`{method}_ucc_planck_generated.cpp`. Gated at ranks 2 **and** 3 (rank 2 alone would miss
`make_generated_*_kernels`); RCC emit byte-identical, SHA-256 unchanged. Rank 4 verified clean
out-of-band — zero collisions, factory `make_generated_ucc_ccsdtq_kernels` — but deliberately
**not** in the gate: `ccsdtq` generation takes ~15 min, and ranks 2/3 already exercise the same
naming path. If that path is ever made rank-dependent, add rank 4 to the gate.

> **`method` is overloaded, which is the trap.** The obvious fix — prefix `method` once in
> `emit_planck_translation_unit`, since `_kernel_name` and the factory both derive from it —
> fails, because `method` is *also* parsed for the excitation rank (`parse_cc_level`), which
> requires a `cc` prefix and rejects `ucc_ccsd`. The prefix belongs on the emitted **name**
> only.
>
> **And one measurement method to distrust:** `nm -g` on macOS reports weak/coalesced symbols
> as `T`, so a naive shared-symbol diff between the two objects lists ~50 inline and template
> instantiations the linker deduplicates by design. Grep the linker's own `duplicate symbol`
> diagnostic instead of inferring from `nm`. (A bare link harness will also report *undefined*
> `to_tensor_nd` — that is the runtime TU being absent, not a collision.)

**U5.1 — a UCC prepare path (~M, C++).** A `prepare_generated_ucc_state` **sibling** of
`prepare_generated_arbitrary_order_state`, not a branch inside it: all four steps differ (UHF
reference, spin-blocked ERIs, per-block denominators, sector-only amplitudes), so sharing one
function would mean a branch at every line.

> **The ERI block vocabulary is NOT passed in — and checking how RCC does it is what
> established that.** An earlier draft of this step said the vocabulary "comes from the
> bundle's `sector_tags`", on the "one vocabulary, defined in ccgen" reasoning U2.2 used for
> denominators. That was wrong twice over: `sector_tags` carry *amplitude* tags (`aa`,
> `abab`), from which the (space, spin) ERI set is not derivable at all; and RCC does not
> communicate a vocabulary in the first place.
>
> **`build_tensor_cc_block_cache` takes no block list.** The set *is* the struct's seven named
> members, built unconditionally — and it deliberately **over-builds**: measured, both `ccsd`
> and `ccsdt` read only 6 of the 7, with `ovvo` constructed and never touched. Nothing is
> negotiated with the emitter, so nothing can drift.

The UCC analogue is the same property one level up: for each spin tag, one stored array per
**orbit of the 16 o/v patterns under that tag's own symmetry group** (U3.0's predicate). That
is a closed set fixed by the *reference type*, not by the method:

```
aaaa   7 blocks
abab  10 blocks   (its orbits are smaller — two of the four permutations are not its symmetries)
bbbb   7 blocks
      ──
      24 stored arrays
```

Because the rule is the same one U3.0 already encodes, the C++ set and the emitter's set cannot
disagree by construction — which is what the discarded "pass the vocabulary" design was trying
to buy with coupling.

- **U5.1a — `ucc_canonical_blocks()` — LANDED.** The 24-block closed set in C++
  (`ucc_blocks.cpp`), derived from that orbit rule rather than listed, alongside a C++
  `eri_permutation_preserves_block` mirroring U3.0's predicate.

  *Gated on both sides, because two independent derivations of one rule are only cheaper than
  coupling if something checks they agree.* The C++ gate executes the rule (counts 7/10/7;
  `baba` absent; **every one of the sixteen patterns reachable per tag**; each canonical member
  the lexicographic minimum of its orbit; every block actually buildable from a reference). The
  Python gate parses the C++ spin tags and permutation table out of `ucc_blocks.cpp` and
  compares them against ccgen's — parsed rather than executed, since running it would need the
  whole CC link. Together they cover *the rule is right* and *both sides implement it*.

  Verified falsifiable in **three** directions: C++ storing an extra `baba` block (caught by
  both gates), C++ folding orbits spin-blindly (caught by the reachability assertion, which
  sees a wrong orbit rule directly rather than through a count), and the **Python** table
  drifting (caught by the pin).

  > **One assertion was wrong first, and the wrong version looks right.** It assumed canonical
  > members are occupied-first, and failed on `vovo` / `vovv`. Those *are* canonical — alone in
  > their orbit under `abab`'s reduced group, so there is no occupied-first member to prefer.
  > The assertion now states the actual rule (lexicographic minimum of the orbit).

- **U5.1b — `prepare_generated_ucc_state`** (next): builds all 24 unconditionally, plus the UHF
  reference (`build_uhf_reference` exists), the spin Fock blocks (`build_ucc_fock_blocks`, fed
  from `_info._scf.{alpha,beta}.fock` — both persisted by SCF, verified), and the per-block
  denominators. A **sibling** of `prepare_generated_arbitrary_order_state`, not a branch inside
  it, and with no `blocks` parameter. *Gate:* `by_rank` **empty** on amplitudes and
  denominators; one amplitude block and one denominator per sector tag, each with its own
  shape. Buildable from a fixture reference, so no SCF is required.

*Memory, now concrete:* 24 blocks is the ~3× footprint this doc flagged up front. If that
proves too heavy at rank 4, trim the **stored set by reference symmetry** — never by method —
so the closed-set property survives. Trimming per method reintroduces exactly the coupling this
step avoids.

**U5.2 — `rebind_physicist` for spin blocks (~S/M, C++).** The mid-axis transpose is the same,
but the **oovv↔ovov cross-source is spin-sensitive** (U3 measurement 3: physicist `oovv_abab`
is chemists `(i_α a_α | j_β b_β)`), so it must be re-derived per block rather than copied.

*Gate:* an RHF-degenerate reference ⇒ rebound UCC blocks equal the rebound RCC blocks. **Plus an
asymmetric companion**: degeneracy cannot see a swapped mixed pair, the same vacuity that made
U3.1's fixture load-bearing.

**U5.3 — registry + keyword (~S).** `make_generated_ucc_kernels(int rank)` beside the RCC one;
`ucc2` / `ucc4` in the keyword table; the driver's RHF-reference guard relaxed for them. This is
also where a `PLANCK_CC_UCC` CMake option belongs — **not earlier**: a build flag that emits a
TU nothing can reach is a flag that cannot be tested.

*Gate:* `correlation ucc2` reaches the solver and returns a number rather than an error.

**U5.4 — `ucc2` against hand-written UCCSD (~S, the first real number).** A radical cation,
using the in-tree oracle. **Land this before the FCI gate**: it exercises the whole stack at the
smallest rank, so a failure localizes to the wiring rather than the algebra — which U1–U4 have
already gated independently.

**U5.5 — open-shell UCCSDTQ == FCI (~M, the one that matters).** The closed-shell analog is the
strongest gate in the whole ccgen effort (`0970e21` / `ce03048`: Be CCSDTQ vs FCI, 6.4e-11).

> **Check the system is not vacuous before trusting a pass.** U1.5 found Li/STO-3G makes `t3`
> worth 0, so a broken T3 passes there; LiH⁺/6-31g was used instead because it makes the triples
> worth 8.1e-8. The same trap applies at rank 4 — verify the highest amplitude is worth something
> first.

#### What U5 does NOT need

Already done and gated independently: the spin algebra (U1, PySCF-validated to ~6e-16), the
denominators (U2), the integrals and their emitter routing (U3), and the runtime's ability to
accept, evaluate, update and allocate an all-sectors bundle (U4). **If the first `ucc2` run
disagrees with hand-written UCCSD, look at the wiring before the algebra.**

One coverage note: every UCC gate so far has run through the **diagram** engine
(`PLANCK_CC_ENGINE` defaults to `diagram`, as does the generator). `wick` is selectable and the
two are documented residual-equal, but no UCC gate pins that — worth one assertion rather than
an assumption.

---

## What this reuses (summary)

| Reused | From |
|---|---|
| UCC spin integration, block existence, exchange handling | `ucc_manifold`, `ucc_integrate_term_antisym`, `block_exists` (landed, PySCF-transitive) |
| Spin→AlgebraTerm bridge | `spinterm_to_algebraterm` (exact as of `cfe302a`) |
| Per-block tagged amplitude storage + solver update | `sectors` / `sector_tensor` / `sector_tags` / `sector_residuals` / `ensure_amplitude_sectors` (gaps B1–B4) |
| UHF reference struct + builder | `UHFReference` (`common.h:141`), already driving hand-written UCCSD |
| AO→MO transform | `Correlation::transform_eri` (parallel) |
| Codegen default-off switch pattern | `--spin-adapt` / `PLANCK_CC_SPIN_ADAPT` |
| FCI-equality gate pattern | Be CCSDTQ == FCI (`0970e21`) |

**Net new:** the no-collapse adapt entry + spin-resolved factor naming (U1), a
spin-resolved denominator (U2), spin-blocked ERI blocks (U3). Everything else is
wiring.

---

## What NOT to do

- **Do not fork the pipeline.** UCC is RCC-minus-collapse. A parallel
  `ucc/` module duplicates `ucc_integrate_term_antisym` and the bridge, and the
  two copies will drift — the R3.1.2 bridge bug took two commits to get exact and
  you do not want to fix it twice.
- **Do not add a new amplitude container or solver loop.** The `(rank, tag)`
  sector machinery is already the general case. A `UCCAmplitudes` type would be a
  second thing to keep in sync with `ensure_amplitude_sectors`.
- **Do not reuse `_amplitude_block_tag`'s β-majority flip.** It folds `abbabb`
  into `aabaab`, which is valid **only** when α and β orbitals coincide. In UCC
  those are different amplitudes. This is the single easiest way to introduce a
  silent wrong answer.
- **Do not enable `--include-intermediates` on the UCC path** until it is
  validated there — the RCC spin-adapt path disabled it for both correctness
  (CSE mislabels occ/vir on spatial spin-adapted terms) and compile time
  (`e0f3849`). UCC has strictly more terms.
- **Do not assume a rejected input proves which guard rejected it.** Two guards can reject
  the same fixture; asserting only "it was rejected" then tests nothing in particular. Both
  U4.0 fixture defects were of this shape. Assert on the guard's own message.
- **Do not read `Terminated: 15` from `cc1plus` as a compile error.** It is SIGTERM — the
  compiler was killed (an interrupt, or the OOM killer), not a defect in the code being
  compiled. It surfaces on `generated_kernel_registry.cpp`, the slow `-O1`-pinned TU, and
  reading it as a real error sends you debugging a change that is fine.
- **Do not treat a green `ctest` as evidence that `hartree-fock` builds.** The CC unit binaries
  do not link it, so they pass either way. A build failure and a 6/6 ctest run can appear side
  by side and the ctest result says nothing about the failure.
- **Do not trust a `make <target>` that prints nothing.** It can exit 0 having built nothing
  when the build directory is stale, and a mutation run against the previous binary reports
  whatever the previous binary did. This invalidated one reported mutation result and made a
  correct guard look broken. Use `make -B` before believing a mutation.
- **Do not wire U5 by adding a keyword alone.** The three UCC C++ builders have no production
  callers; `prepare_generated_arbitrary_order_state` still builds the RHF reference and RHF
  block cache unconditionally. A keyword that reaches an unwired prepare path produces a
  plausible RHF number under a UCC name.
- **Do not fix a spin-routing emitter by suffixing array names alone** (U3.2's lesson). Two of the four
  `_ERI_SYMMETRY_PERMUTATIONS` are invalid for `abab` (they map it to `baba`), and **37 of
  142** mixed-block reads currently use them. Name-only routing sends those 37 to the right
  array with permuted indices — wrong, and quieter than what is there now. Land U3.0's
  validity predicate first.
- **Do not store a `baba` ERI family.** It is `abab` under the particle swap (verified
  numerically on real orbitals), so storing it buys ~33% more memory in exchange for
  avoiding one explicit swap in the emitter.
- **Do not accept U3.1's RHF-degenerate gate on its own.** With `C_α == C_β` the `(a|b)` and
  `(b|a)` pair orderings coincide, so a swapped mixed pair passes it. It needs an
  asymmetric-reference companion (`noa != nob`), for the same reason U2.1's fixture uses
  four distinct extents.
- **Do not land a storage half without its emitter half** (U3's lesson, kept for the next such split). Spin-blocked arrays that
  no emitted kernel ever names change nothing, while making the step look finished and
  turning a loud absence into a silent wrong answer. They are one change.
- **Do not make the generated runtime's reference a variant.** The scope originally called
  for it; measured, the kernels never touch `RHFReference` as a type (only `f_oo`/`f_ov`/
  `f_vv` and `orbital_partition`), so it churns every kernel signature and every generated
  TU for nothing. U2.2 delivered the spin-resolved denominators it was meant to buy.
- **Do not skip the U1 numeric gate before writing C++.** PySCF
  `uccsd.update_amps` is a direct oracle at rank 4 and it costs minutes. Debugging
  a wrong UCC residual through the C++ runtime costs days — the B5 physicist-ERI
  convention bug is the precedent (found only by injecting an FCI-correct oracle
  into live C++ state).
- **Do not scope rank ≥ 5 UCC before measuring the rank-4 registry compile.**
  Compile time, not FLOPs, is the demonstrated wall.
- **Do not build U0 on `external_blocks`, and do not gate against its output.** It folds
  a↔b (`key = min(combo, flip)`, `spin.py:425`), so it returns `['aaaa','abab']` with no
  `bbbb` — measured on the current tree. Any gate phrased as "every `external_blocks`
  block maps to a tag" passes vacuously while the β-majority blocks are simply absent.
  Assert against PySCF's `t2aa/t2ab/t2bb` names instead.
- **Do not gate UCC equality on a symbolic term-multiset comparison.** Compare numeric
  residuals on symmetry-correct tensors. A multiset comparison cannot tell "different
  algebra" from "same algebra, different symmetry-equivalent writing" — that distinction
  cost V1.1e five sub-steps and one phantom defect (`{"doubles": 14}`).
- **Do not run the numeric gates in the default interpreter.** They skip silently (pyscf
  not importable) and a green run means nothing. Use `tests/pyscf/.venv/bin/python`
  (pyscf 2.13.0).

---

## Dressed UCC kernels (U6, scoped separately)

U0–U5 deliver **raw** (undressed) UCC kernels — the same validation altitude the
RCC generated path sits at today.

**Dressing is RETIRED as a production route** (dressing and spin adaptation do not
compose; measured 52 % short on Be — see `CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`),
so there is no dressed-UCC scope to defer to. The retirement note records that
*for UCC the mechanism predicts dressing would work*, because UCC keeps per-spin-block
tensors rather than folding to one spatial tensor — untested, and not on this path.

Two design constraints from that work still bind **inside** U1, so they are recorded
here rather than left in a retired doc:

- **Decision 5 is `GCC → dress → adapt`.** Dressing runs on the spin-orbital
  residual and is spin-adapted afterward, because recognition needs the diagram
  line-graph that post-adaptation `SpinTerm`s do not have. So `ucc_adapt_equations`
  (U1) is the thing a dressed manifold gets routed *through* — meaning **U1 must
  accept an already-dressed GCC manifold**, not assume a raw one. Concretely: do
  not hard-wire U1 to `generate_cc_equations` output; take the equation dict as a
  parameter. One-line difference now, a restructure later.
- **V1.1 requires block-keyed intermediate specs.** Under UCC one dressed
  operator becomes several spin-block variants (`Wmnij` yields an `oooo` variant
  per surviving block), each needing its own spec and builder. If U1.1's
  spin-resolved factor naming is built block-keyed — which it must be anyway for
  `t2_aaaa` vs `t2_abab` — dressed operators reuse the identical mechanism for
  free. **Use one naming path for amplitudes, ERIs, and intermediates.**

One further note, kept in case dressing is ever revisited: `Wmbej`'s
asymmetric-block (`ovvo`) binding sign is gated on `_block_is_asymmetric`, and under
UCC that predicate would have to key on the *spin-resolved* block rather than the
space pattern alone — the highest-risk single item in dressed UCC.

## Open question worth settling early

**Does UCC need the dressed-operator work first?** RCC's generated path carries
worse-than-optimal FLOP scaling because the generated residuals are CSE'd but not
factored (`ccgen_dressed_intermediates`). UCC multiplies the term count by the
block count, so it inherits that with a larger constant. If the target is
*production* open-shell CC, dressing is on the critical path — as it already is
for RCC (`ccgen_generated_kernels_to_production`). If the target is validation
(UCCSDTQ == FCI on a tiny open-shell system), scaling does not matter and UCC can
land first.

Recommendation: treat U0–U5 as the **validation** deliverable (small systems,
FCI-gated), and take dressed UCC via V5/V6 afterward — while honoring the two
U1 constraints above so V5 stays a switch rather than a rewrite.

---

See `CCGEN_SPIN_ADAPTATION_SCOPE.md` (the S0–S4 layer this extends; UCC is the
deferral noted at its S3 close), `CCGEN_CCSDTQ_MULTISECTOR.md` (the
bridge exactness and the multi-sector precedent), and
`CCGEN_CCSDTQ_MULTISECTOR.md` (gaps B1–B4, the sector runtime UCC
reuses wholesale).
