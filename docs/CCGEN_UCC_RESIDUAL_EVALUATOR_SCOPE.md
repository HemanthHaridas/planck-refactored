# F2 — evaluating UCC residuals numerically

**F2 is COMPLETE — F2.0 through F2.4 all landed.** F1 landed the fixture (`ucc_random_tensors`); F2 is
the evaluator that consumes it, and it unblocks F3/U1.2 — the PySCF UCCSD gate that is the only
thing which will have checked the **values** of the landed UCC residuals rather than their
structure.

The landed surface: `ucc_resolve_factor` + `ucc_residual_einsum` + `ucc_closed_shell_tensors`
(`python/ccgen/tests/residual_eval.py`), `ucc_term_spins` (`python/ccgen/spin.py`), gated by
`F21FactorResolutionTests` / `F22aTermSpinMapTests` / `F22bcUccResidualEinsumTests` /
`F23ClosedShellOracleTests` in `python/ccgen/tests/test_spin.py`.

## The design problem, measured

`residual_einsum` picks each factor's array by **space** alone:

```python
occ, vir = slice(0, no), slice(no, n)
sl = tuple(occ if i.space == "occ" else vir for i in f.indices)
ops.append(tensors[f.name][sl])
```

Under UCC the array also depends on **spin**, and the spin is not in the term. Measured on the
emitted `ccsd` UCC residuals:

- Amplitude factors **are** block-resolved — `t1_aa`, `t2_abab`, … (U1.1 did that).
- `v` and `f` factors **are not**. They arrive as bare `v` / `f` with 13 distinct space patterns and
  no spin tag at all.
- `Index` has fields `name`, `space`, `is_dummy` — **no spin field**.

**And the spin cannot be recovered from term context.** The obvious repair is to infer each `v`
index's spin from an amplitude factor sharing that index. Measured: on `doubles_abab`, **51 of 82
terms** have at least one `v`/`f` index that appears on no amplitude factor. Inference is not
available for the majority of terms, so it is not a fallback — it is a dead end, and this is the
number that settles F2's shape.

The spin **does** exist one layer up: `SpinIndex` carries `.spin`, and `SpinFactor` carries
`.block`. `ucc_spinterm_to_algebraterm` is the point where it is dropped — deliberately, because
`AlgebraTerm` is the emit-layer type and the emitter names tensors, not spins.

So F2's real question is: **where should the spin live so the evaluator can slice, without pushing
spin into the emit-layer type that RCC also uses?**

## Steps

### F2.0 — decide where the spin is carried (~S, design, do first)

Three options, and the choice constrains everything after it:

| option | cost | risk |
|---|---|---|
| **A. Block-tag `v`/`f` in the bridge**, as U1.1 did for amplitudes (`v_abab`, `f_aa`) | small, symmetric with what already works | changes what the emitter will later see for UCC; must not touch RCC |
| **B. Return a spin sidecar** from `ucc_adapt_equations` — `{term_id: {index_name: spin}}` | leaves `AlgebraTerm` untouched | a parallel structure that can desynchronize from the terms; the exact defect class R3.1.2 was |
| **C. Evaluate from `SpinTerm`s** and never involve `AlgebraTerm` | no type changes at all | duplicates the einsum machinery; diverges from what the emitter will actually consume |

**Recommend A.** It is the mechanism already proven for amplitudes, it keeps one source of truth,
and the C++ runtime will need blocked ERI names regardless — `MOBlockCache` is spin-free today
(`mo_blocks.h:20-21`, noted in the UCC scope), so something must carry the block there too. B is the
option the project rule warns about: two structures that must agree, with nothing enforcing it.

#### Blast radius of A, measured — it is smaller than it looks

**A changes `ucc_spinterm_to_algebraterm`'s output, and that function has NO production consumers.**
Measured across `python/ccgen/`:

| consumer | kind |
|---|---|
| `ccgen/spin.py` (3) | the definition and `ucc_adapt_equations` itself |
| `ccgen/tests/test_spin.py` (8) | the U1.0/U1.1 gates written for it |
| `ccgen/optimization/dressed_equation.py` (1) | **docstring only** — names `ucc_adapt_equations` as a substitutable `adapter=`; `grep adapter=ucc_` returns nothing, so no code passes it |

So the reachable surface is the UCC gates themselves. Contrast the RCC side, which is why F1 was a
sibling and not a parameter: `residual_einsum`/`random_tensors` have **seven** consumers and 57
calls in `test_spin.py` alone.

**The constraint that must hold, and how to prove it rather than assert it:**

- `spin_adapt_equations` output **byte-identical** — the RCC bridge must not be touched. Pin it the
  way U1.0/U1.1 were: hash the adapted manifold, `git stash`, re-hash, compare. Baseline for `ccsd`
  is `sha256 e5a08b62b5dcfb932ac06801d7d1e299`.
- `random_tensors` output **byte-identical** — baseline `sha256 4047f5c9121fbb05d0c398a0acc8a616`.
- The full ccgen suite green, not just the UCC gates.

A that reaches beyond those three is not option A; it is a redesign, and should be re-scoped.

*Verify:* the decision written down with its consequence for the C++ side, before any evaluator code
exists. If A, the U1.1 gate extends to `v`/`f` and both baselines above still hold.

#### F2.0 — DECIDED: option A, block-tagged `v`/`f`, **and the emitter must be taught the tag**

Checked against the emitter before committing to it, which changed the answer's shape.

`_map_factor` (`python/ccgen/emit/planck_tensor_cpp.py:200-216`) dispatches on the **exact** strings
`"f"` and `"v"`:

```python
amplitude_match = re.fullmatch(r"t(\d+)(?:_([ab]+))?", tensor_obj.name)   # tolerates a tag
if tensor_obj.name == "f":  ...                                          # exact
if tensor_obj.name == "v":  return _map_eri_tensor(tensor)               # exact
```

So the amplitude branch **already** accepts a block suffix — that is why U1.1's `t2_abab` emits
today without touching the emitter. `v`/`f` do not: a factor named `v_abab` matches neither the `v`
branch nor the amplitude regex (verified: `re.fullmatch(pattern, "v_abab")` is `None`) and falls
through to `raise NotImplementedError`.

**Verdict: A, with the emitter change as an explicit part of it** — loosen the two exact-match
branches to the same `name(?:_([ab]+))?` shape the amplitude branch uses, and thread the tag to the
block lookup. Not a follow-on to discover later: A is incomplete without it, and discovering that
during F2.2 would look like an evaluator bug.

Why A survives anyway:

- **The C++ side needs it regardless.** `MOBlockCache` (`src/post_hf/cc/mo_blocks.h:15-25`) is
  spin-free — `oooo`, `ooov`, `oovv`, … over "the full spatial MO basis". UCC needs `v_aaaa`,
  `v_abab`, `v_bbbb` with *different shapes*, so a tag has to reach the C++ layer through some
  route. A makes that route the same one amplitudes already use.
- **One naming shape, now four consumers** — amplitudes (U1.1), higher Sz sectors (R3.1.3c),
  intermediates (`block_keyed_intermediate_name`, V1.1c), and now ERIs. B's sidecar would be a
  fifth mechanism carrying the same information.

**RCC safety is structural, not incidental:** the suffix is emitted only by
`ucc_spinterm_to_algebraterm`. `spinterm_to_algebraterm` keeps emitting bare `v`/`f`, so the
loosened regex is a superset that matches the old input identically. The two hash baselines above
are the proof obligation, not the argument.

*Extra gate this decision earns:* a `v_abab` factor must emit, and a bare `v` must emit **exactly as
before** — the second is what pins the superset claim.

### F2.0b — block-tag `v`/`f` in the UCC bridge + loosen the emitter (~S, falls out of the verdict)

Two edits, both small, both gated by the baselines above:

1. `ucc_spinterm_to_algebraterm` suffixes `v`/`f` with their `SpinFactor.block` the same way it
   already suffixes amplitudes. The block is on the factor; nothing needs deriving.
2. `_map_factor`'s two exact-match branches become `name(?:_([ab]+))?`, and the tag reaches the
   block lookup.

*Gate:* `v_abab` emits; bare `v` emits **byte-identically to today** (the superset claim); RCC
manifold hash `e5a08b62…` and RCC fixture hash `4047f5c9…` both unchanged; full ccgen suite green.

### F2.1 — resolve one factor to its array (~S)

A single function: `(factor, tensors, dims) -> ndarray`, choosing the block by (space, spin) per
index and slicing within it. No einsum, no terms.

*Verify:* `t2_abab` resolves to shape `(nva, nvb, noa, nob)`; `v_abab` with space pattern
`(occ,vir,occ,vir)` resolves to the `v_abab` block sliced `[occ_a, vir_b, occ_a, vir_b]`. Both are
assertable on F1's fixture alone, with no equations involved. After F2.0b the spin comes from the
factor's own name, so this step is a lookup and a slice — no inference.

**This is where a slice-assignment bug lives**, so it gets its own gate rather than being debugged
through a full residual.

### F2.2 — `ucc_residual_einsum` (~S) — **LANDED**

Same einsum construction as `residual_einsum`; only the operand lookup changes, via F2.1. Output
layout must match the RCC evaluator's convention (`[vir_ext…, occ_ext…]`) so the two are comparable.

Two properties were **measured** while scoping this, and both turn what looked like risk into an
assertion:

- **Spin is consistent per index within a term.** Across every emitted `ccsd` UCC term, **zero**
  have an index name carrying two different spins in different factors. So a single
  `{index_name: spin}` map per term is well-defined, and building it cannot silently pick a winner.
- **Free-index spins are recoverable from the factors, and match the target's own block.**
  Re-measured on the current tree: `singles_aa → (a,a)`, `doubles_aaaa → (a,a,a,a)`,
  `doubles_abab → (a,b,a,b)`, `doubles_bbbb → (b,b,b,b)` — one spin pattern per target, no
  exceptions. So the output shape is determined and does not need to be passed in.

**Term counts drifted since this doc was first written.** It claimed 328 `ccsd` UCC terms and
"51 of 82 on `doubles_abab`"; the tree now emits 352 (`singles_aa` 28, `singles_bb` 28,
`doubles_aaaa` 104, `doubles_abab` 88, `doubles_bbbb` 104, `energy` 8). The 51/82 argument for why
spin must be carried rather than inferred still holds directionally — do not re-cite the number
without remeasuring it.

#### F2.2a — the per-term spin map (~S) — **LANDED**

`ucc_term_spins(term) -> {index_name: "a"|"b"}` (`python/ccgen/spin.py:1335`), built by reading each
factor's tag positionally. Raises on a clash, an untagged factor, or a tag-length mismatch; all
three branches are unreachable on today's manifolds (measured 0 across ccsd/ccsdt/ccsdtq) and are
asserted anyway.

**Note for F2.2b: this is not the operand-lookup path.** F2.1 reads the block tag off each factor's
*own* name, so the einsum does not need the map to slice. `ucc_term_spins`'s role in the evaluator
is the per-term consistency **assertion** — call it for that, do not thread it through the operand
loop.

#### F2.2b+c — operand assembly and output layout (~S, one step) — **LANDED**

Originally split. They are merged because there is no state between them where a separate gate
localizes anything: F2.2c is one line of `ext` ordering *inside* the F2.2b function body, and the
whole function is ~15 lines. Swap the `if f.name in ("v", "f")` branch for `ucc_resolve_factor`
(F2.1); order free indices `[vir…, occ…]` as `residual_einsum` does, each axis sized from that
index's own spin.

*Verify:*

- `doubles_abab` evaluates to `(nva, nvb, noa, nob)` and `doubles_bbbb` to `(nvb, nvb, nob, nob)` —
  different shapes from the same code path, which is the whole point. Dims **non-square and
  asymmetric** (`noa=5 nva=4 nob=4 nvb=5`) so neither a transposed axis nor a swapped spin can hide.
- For one hand-picked term, the einsum subscript string is **identical** to what the RCC path would
  build for the same term — subscripts depend only on index identity, not on spin, so any difference
  means the swap disturbed something it should not have.

#### F2.2d — full-manifold smoke (~S) — **LANDED** (folded into the F2.2b+c class)

Evaluate **every** term of every UCC target on F1's fixture.

*Verify:* no raise, and every per-target sum has the shape F2.2b+c predicts. Explicitly **not** a
correctness gate — a wrong slice that keeps its shape passes here. F2.3 is what catches that, and
the split exists so that a failure in F2.2d localizes to assembly rather than to physics.

### F2.3 — the closed-shell oracle (~S, the load-bearing step, no PySCF needed) — **LANDED**

**On a closed-shell system** the UCC residual must reproduce the **existing RCC** `residual_einsum`
result for the same equations. Two things this doc previously got wrong about that, both found by
prototyping the comparison before scoping it:

**1. It is a per-target pairing, not a sum over blocks.** RCC adapts on the *closed-shell
representative external block* (`python/ccgen/spin.py:1088`), which for doubles is `abab`. So the
oracle is:

| UCC target | RCC target |
|---|---|
| `energy` | `energy` |
| `singles_aa` | `singles` |
| `doubles_abab` | `doubles` |

`doubles_aaaa` / `doubles_bbbb` have **no RCC counterpart** — `collapse_amplitudes` splits the
all-α sector away rather than storing it — so they are covered by F2.2d's shape smoke and by F3,
not by this gate.

**2. It is not free, and F1's fixture cannot serve it.** The two sides consume different bundles,
so the UCC blocks must be built **from** the spatial ones through the closure relations that
`collapse_amplitudes` / `collapse_integrals` invert:

```
t2_aaaa = t2_abab - t2_abab.transpose(1,0,2,3)      # same for bbbb
v_aaaa  = v_abab  - v_abab.transpose(0,1,3,2)       # same for bbbb
t1_aa = t1_bb = t1   ;   f_aa = f_bb = f
```

with the spatial `t2` carrying `t2[abij] = t2[baji]` and `v` carrying both `<pq|rs> = <qp|sr>` and
bra↔ket. `ucc_random_tensors` (F1) draws every block **independently**, so it violates these by
construction — feed it to both sides and the gate fails for a reason that has nothing to do with the
evaluator. F2.3 therefore needs a second fixture, `ucc_closed_shell_tensors(no, nv, seed)`,
returning the UCC bundle and the spatial dict that generated it. That is a step this doc did not
previously have.

Measured on the prototype at `no=5, nv=4`: `singles_aa` vs `singles` maxdiff **6.8e-13**,
`doubles_abab` vs `doubles` **1.8e-12** (against ‖R‖ ~1.1e3 / 1.6e3), energy exact.

*Verify:* ≤**1e-11** elementwise on a **non-square** `(no, nv)`. Not 1e-12 — the measured `doubles`
diff is 1.8e-12 and the tighter bound would flake. Square hides transposition errors, the trap
recorded in `CCGEN_RANK3_KERNEL_AND_SOLVER.md`.

*And commit the falsifiability check with it:* transposing one axis of `t2_abab` in the fixture must
break all three targets by O(‖R‖). Without this, the gate can rot into a vacuous pass and nothing
would say so.

#### What building it found

Measured at `no=5, nv=4`, agreement is at machine precision relative to the residual norm:

| pair | maxdiff | ‖R‖ | relative |
|---|---|---|---|
| `energy` | 2.3e-13 | 1.06e3 | 2.2e-16 |
| `singles_aa` vs `singles` | 4.5e-13 | 1.08e3 | 4.2e-16 |
| `doubles_abab` vs `doubles` | 3.9e-12 | 1.58e3 | 2.5e-15 |

The `doubles` figure confirms the 1e-11 bound: 1e-12 would flake.

**The spatial symmetries are NOT load-bearing for the oracle, contrary to what this doc assumed.**
Mutation-tested: removing `t2[abij] = t2[baji]` or `<pq|rs> = <qp|sr>` from the fixture leaves the
comparison holding to ~8e-13. The RCC/UCC identity is an *algebraic* property of the two term sets,
not of the tensors they contract. The symmetries are kept so the fixture describes a physically
reachable reference — do not cite them as why the gate works.

What *is* load-bearing is the **closure relation**: flipping the sign in `v_aaaa` breaks every
paired target. That is the mutation the gate exists to catch.

Also measured while mutating: `t2 - t2.transpose(1,0,2,3)` and `t2 - t2.transpose(0,1,3,2)` are
*identical* given the `t2[abij]=t2[baji]` symmetry, so a mutation swapping them is not a defect and
its survival is not a gate weakness.

**Do this before F3.** If F2.3 fails, the defect is in the evaluator; if F2.3 passes and F3 fails,
the defect is in the equations. Running them in the other order conflates the two, which is the
mistake the rank-3 investigation paid five falsified hypotheses for. That argument is *stronger*
now than when it was written: the oracle needs no PySCF and no converged amplitudes at all.

### F2.4 — hand F3 a usable entry (~S) — **LANDED**, and the mapping is *not* a pure rename

`ucc_residual_einsum` already accepts an externally supplied tensor dict, so the entry needed no
code. What F2.4 actually produced was a correction: **the PySCF→ccgen mapping is a TRANSPOSE, not a
rename.** The names correspond one-for-one (`t2ab` ↔ `t2_abab`) — that much was measured correctly —
but PySCF stores `(occ…, vir…)` while ccgen emits `(vir…, occ…)`, so every array needs its halves
swapped. The claim that it "was already measured to be true" checked the *names* and not the
*layout*.

Also corrected: CH3/STO-3G is `noa=5 nva=3, nob=4 nvb=4`, not the `nva=4`/`nvb=5` carried by
`CCGEN_UCC_NUMERIC_FIXTURE_SCOPE.md`.

The full F3 result, including the `f_ov` convention that took the investigation, lives in
`CCGEN_UCC_NUMERIC_FIXTURE_SCOPE.md` — F3 is landed at ~6e-16 in every block.

## What NOT to do

- **Do not infer spin from term context.** 51/82 terms on `doubles_abab` cannot be resolved that
  way. Any code that appears to work is silently guessing on the other 31.
- **Do not modify `residual_einsum` or `random_tensors`.** Seven consumers, 57 calls in
  `test_spin.py` alone. UCC gets siblings; that is why F1's RCC-fixture output is byte-identical
  (`sha256 4047f5c9…`).
- **Do not gate on a square system**, and do not gate at converged amplitudes or on OH/STO-3G — the
  three vacuous-pass traps recorded in `CCGEN_UCC_NUMERIC_FIXTURE_SCOPE.md`.
- **Do not skip F2.3 to reach the PySCF gate sooner.** It is the cheaper oracle and it localizes the
  defect; F3 alone cannot say whether the evaluator or the equations are wrong.

## Key code locations

| what | where |
|---|---|
| the RCC evaluator to sibling (do not modify) | `residual_einsum`, `python/ccgen/tests/residual_eval.py:63` |
| F1's spin-resolved bundle | `ucc_random_tensors`, same file |
| where spin still exists | `SpinIndex.spin` / `SpinFactor.block`, `python/ccgen/spin.py` |
| where it is dropped | `ucc_spinterm_to_algebraterm`, same file |
| the emit-layer type with no spin | `Index` (`name`, `space`, `is_dummy`), `python/ccgen/tensors.py` |
| what this unblocks | `CCGEN_UCC_NUMERIC_FIXTURE_SCOPE.md` F3, `CCGEN_U1_UCC_ADAPT_SCOPE.md` U1.2 |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`, which are canonical.
