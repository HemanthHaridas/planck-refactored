# Arbitrary-order UCC kernel generation and execution

Scopes one capability: **generate and run arbitrary-order *unrestricted* CC
kernels (UCC) alongside the existing arbitrary-order RCC path** — so an
open-shell reference can drive `ucc4`/`ucc5` the way a closed-shell reference
drives `cc4`/`cc5` today.

**Status, 2026-08-22.** **U0 and U1 are landed and numerically validated; U2 is in progress; U3–U5
are ahead.**

| step | state |
|---|---|
| U0 | landed — `ucc_independent_blocks`, `_ucc_block_tag`, `external_blocks(fold_spin_flip=…)` |
| U1 | **landed** — `ucc_adapt_equations` + `ucc_spinterm_to_algebraterm`, validated against PySCF UCCSD at rank 4 (~6e-16) and against GCC-sliced at rank 6 (1.6e-17). U1.3 turned out to be **dead**, not work: U1.1 designed its hazard out. Detail in `CCGEN_U1_UCC_ADAPT_SCOPE.md` |
| U2 | **in progress** — U2.1 landed (`build_ucc_block_denominator` + `planck-cc-ucc-denominator`); threading `UHFReference` through the solver is next |
| U3–U5 | not started |

**Read `docs/CCGEN_UCC_RANK6_PYSCF_GAP_HANDOFF.md` first** if you are touching the UCC validation
story: it carries the three independent correctness routes, the interface conventions that cost the
most time, and the one open thread below.

One rank-6 thread stays open and is **not** a ccgen defect: `test_ucc_rank6_vs_pyscf`'s triples
target disagrees with PySCF by rel ~2e-3 (`expectedFailure`). ccgen is cleared by two independent
routes — its own closed-shell oracle, and UCC == GCC-sliced with GCC reaching the FCI limit exactly.
The undiagnosed side is PySCF's `r3aaa`.

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

### U2 — the UHF reference in the generated runtime (~M, C++) — **IN PROGRESS**

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

**Remaining in U2:** thread `UHFReference` through `ArbitraryOrderTensorCCState::reference` and
`solver_arbitrary`'s `const RHFReference &`, then the open-shell MP2-limit gate below.

#### Original scope, for the remaining work

Thread `UHFReference` through the generated arbitrary-order path.
`ArbitraryOrderTensorCCState::reference` and `solver_arbitrary`'s
`const RHFReference &` become a variant/parameterized partition carrying
`n_occ_alpha/n_occ_beta/n_virt_alpha/n_virt_beta`. The denominator cache becomes
spin-resolved: a rank-2n block with α-count `k` per half draws its
`eps` from the α set for the first `k` slots and the β set for the rest.

`UHFReference` and its builder already exist (used by hand-written UCCSD), so
this is wiring plus a spin-aware denominator, not new physics.

*Gate:* an open-shell MP2-limit check — the rank-2 UCC denominators reproduce the
existing UMP2 correlation energy from a single Jacobi step. That isolates the
denominator from the residual algebra.

**Sequencing note.** Do U2 *after* U1's numeric gate. If U1 is wrong, a UCC C++
run fails and you cannot tell whether the algebra or the reference is at fault.

### U3 — spin-blocked MO integrals (~M, C++)

`MOBlockCache` gains spin-blocked ERI blocks (`oovv_aaaa`, `oovv_abab`,
`oovv_bbbb`, and the `ovov` partners). The AO→MO transform is the existing
`Correlation::transform_eri` run per spin-block pair of coefficient matrices —
no new integral engine, and the transform is already OpenMP-parallel.

**Memory:** UCC roughly triples the ERI block footprint versus RCC. Worth
stating up front because it, not FLOPs, is what caps the reachable rank.

*Gate:* for an RHF-degenerate UHF reference (α coefficients == β), every spin
block equals the RCC block bytewise. That is a free, exact regression that
catches a transposed spin index immediately.

### U4 — emit + registry for UCC blocks (~S given U1/U3)

`emit_planck_translation_unit` already emits one kernel per residual key and the
registry already carries `sector_tags` / `sector_residuals`. UCC blocks map onto
that: each `doubles_aaaa`-style key becomes a `(rank, tag)` sector residual, and
`ensure_amplitude_sectors` allocates the blocks.

Add a `--ucc` CLI switch and a `PLANCK_CC_UCC` build gate, **default OFF**, so
the default build stays byte-identical (the pattern `--factorize-tau` /
`--spin-adapt` / `--dress-operators` all follow).

**Compile-time caution:** the spin-adapt path already forced
`--include-intermediates` OFF because ~1544 `build_W_*` functions took ~28 min at
`-O3` (`e0f3849`), and the registry is now compiled at `-O1` with 256-term
chunking (`a690014`, `c48a253`). UCC multiplies the residual count by the number
of blocks — **assume the registry compile time is the binding constraint on
reachable rank** and measure it at rank 4 before scoping rank 5+.

*Gate:* the emitted UCC TU compiles against the real CC headers (the `tau` A1
`test_generated_source_compiles` harness is the template).

### U5 — driver routing + the end-to-end gate (~S given U2–U4)

A `correlation ucc4` keyword routing to the same `solve_generated_rcc` call site
with a UHF reference. The solver loop is unchanged — it already iterates tagged
blocks.

*Gate (the one that matters):* **open-shell UCCSDTQ == FCI.** The closed-shell
analog is landed and is the strongest gate in the whole ccgen effort
(`0970e21` / `ce03048`: Be CCSDTQ vs FCI, gap 6.4e-11). Pick a small open-shell
system where UCC at rank = n_elec is the full CI limit — e.g. the Li atom or
BeH — and assert the same equality. If U0–U4 have a bug, this catches it; if it
passes, UCC is right.

Cheaper intermediate gate: UCC rank 2 (`ucc2`) against the existing hand-written
UCCSD energy on a radical cation. Land that first — it exercises the whole stack
at the smallest rank and reuses an in-tree oracle.

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
