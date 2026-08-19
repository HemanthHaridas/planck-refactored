# F2 — evaluating UCC residuals numerically

**Scope. Not started.** F1 landed (`ucc_random_tensors`). F2 is the evaluator that consumes it, and
it unblocks F3/U1.2 — the PySCF UCCSD gate that is the only thing which will have checked the
**values** of the landed UCC residuals rather than their structure.

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

### F2.2 — `ucc_residual_einsum` (~M)

Same einsum construction as `residual_einsum`; only the operand lookup changes, via F2.1. Output
layout must match the RCC evaluator's convention (`[vir_ext…, occ_ext…]`) so the two are comparable.

*Verify:* runs to completion on every emitted UCC term without raising. Not a correctness gate —
F2.3 is.

### F2.3 — the closed-shell oracle (~M, the load-bearing step, no PySCF needed)

**On a closed-shell system** (`noa == nob`, `nva == nvb`, α and β tensors equal), the UCC residual
summed over its blocks must reproduce the **existing RCC** `residual_einsum` result for the same
equations.

This is free — it needs no PySCF, no open-shell reference, no converged amplitudes — and it catches
a slice-assignment or block-routing error immediately, because both sides are computing the same
physical quantity by different routes.

*Verify:* ≤1e-12 elementwise, on a **non-square** `(no, nv)`. Square hides transposition errors —
the trap recorded in `CCGEN_RANK3_KERNEL_AND_SOLVER.md`.

**Do this before F3.** If F2.3 fails, the defect is in the evaluator; if F2.3 passes and F3 fails,
the defect is in the equations. Running them in the other order conflates the two, which is the
mistake the rank-3 investigation paid five falsified hypotheses for.

### F2.4 — hand F3 a usable entry (~S)

F3/U1.2 needs to evaluate at *PySCF's* amplitudes, not F1's random ones, so `ucc_residual_einsum`
must accept an externally supplied tensor dict keyed the same way. Confirm the PySCF→ccgen mapping
is a pure rename (`t2ab → t2_abab`), which was already measured to be true.

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
