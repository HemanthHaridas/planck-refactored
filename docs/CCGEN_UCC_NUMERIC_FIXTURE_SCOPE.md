# A spin-resolved evaluation fixture for the UCC numeric gate

**Scope. Not started.** Blocks U1.2 in `CCGEN_U1_UCC_ADAPT_SCOPE.md`. U1.0 and U1.1 have landed;
the UCC residuals exist and are structurally gated, but **nothing has evaluated them numerically**,
so their values are unverified.

## The blocker, measured

`residual_einsum` (`python/ccgen/tests/residual_eval.py:63`) assumes **one** orbital space pair and
**one** spin-free ERI tensor:

```python
occ, vir = slice(0, no), slice(no, n)          # a single (no, nv)
sl = tuple(occ if i.space == "occ" else vir for i in f.indices)
ops.append(tensors[f.name][sl])                 # one v / one f, sliced by space alone
```

UCC breaks both. On CH3/STO-3G the α and β spaces have **different dimensions** —
`noa=5 nva=4`, `nob=4 nvb=5` — and a mixed block like `t2_abab` indexes both at once. A factor's
`space` ("occ"/"vir") no longer determines its slice; its **spin** does too, and the fixture has
nowhere to carry that.

The ERI side is the same problem: UCC needs `v_aaaa`, `v_abab`, `v_bbbb` (different shapes), not one
`v[n,n,n,n]` sliced by space.

## What already works, and must keep working

The UCC name vocabulary is exactly PySCF's, verified against `pyscf.cc.uccsd`:

```
ccgen emits:  t1_aa  t1_bb  t2_aaaa  t2_abab  t2_bbbb
PySCF stores: t1a    t1b    t2aa     t2ab     t2bb
```

So the amplitude mapping is a rename, not a transform. That is the cheap half.

**Constraint that shapes the design:** `residual_einsum` / `random_tensors` have **seven** consumers
(`test_spin.py` alone calls them 57 times, plus `test_diagram`, `test_regressions`,
`test_residual_symmetry`, `test_reference_vs_pyscf`, `test_dress_per_operator`,
`test_emit_flag_matrix`). Changing their signature is a large blast radius on gates that currently
pass. **Add a spin-resolved sibling; do not retrofit the RCC one.** Same rule U1.0/U1.1 followed —
the RCC path stayed byte-identical because UCC got its own entry rather than a parameter on the
shared one.

## The work

### F1 — a spin-resolved tensor bundle (~S)

`ucc_random_tensors(noa, nva, nob, nvb, seed)` returning blocks keyed by the names the UCC bridge
emits (`t1_aa`, `t2_abab`, `v_aaaa`, …), with the antisymmetry each block actually has:
`aaaa`/`bbbb` antisymmetric in bra and ket independently, `abab` **not** (its two halves are
different spin spaces).

*Verify:* shapes match PySCF's `t1a/t1b/t2aa/t2ab/t2bb` for a real open-shell case, and each
same-spin block satisfies its antisymmetry to 1e-14. Getting `abab`'s (non-)symmetry wrong is the
easiest way to write a fixture that silently disagrees with PySCF.

### F2 — `ucc_residual_einsum` (~M)

Same einsum construction, but the slice for each factor index is chosen by **(space, spin)** rather
than space alone, and `v`/`f` are looked up per block instead of sliced from one tensor.

*Verify:* on a *closed-shell* system where `noa==nob` and `nva==nvb`, the UCC evaluator summed over
blocks must reproduce the existing RCC `residual_einsum` result for the same equations. That is a
free oracle — it needs no PySCF — and it catches a slice-assignment error immediately.

### F3 — U1.2, the PySCF UCCSD gate (~M) — the reason this exists

Evaluate `doubles_aaaa` / `doubles_abab` / `doubles_bbbb` at PySCF's amplitudes and compare against
`pyscf.cc.uccsd.update_amps`, converting `t_new` back to a residual via `R = (t_new − t)·D`.

Two traps already hit while probing, both of which would have produced a vacuous pass:

- **Do not evaluate at converged amplitudes.** Measured reference `|R|` there:
  `aaaa 0.0, abab 3.9e-09, bbbb 2.5e-20`. A kernel returning zero passes. Perturb the amplitudes
  first (and re-impose aa/bb antisymmetry after perturbing, or PySCF's own residual is meaningless).
- **Do not use OH/STO-3G.** `noa=5 nva=1` gives `C(1,2)=0` distinct αα doubles, so the `aaaa` block
  is trivially zero regardless of correctness. **CH3/STO-3G** (`noa=5 nva=4`, `nob=4 nvb=5`) makes
  all three blocks non-trivial *and* is non-square in both spins — measured reference residuals
  `aaaa 0.069, abab 0.320, bbbb 0.113`.

*Gate:* ≤1e-10 per block. **Not a symbolic term comparison** — V1.1e's e.2.0–e.2.5 established that
a term multiset cannot distinguish different algebra from a symmetry-equivalent rewriting.

## Why this is worth doing before any C++

U1.0/U1.1 are gated **structurally only** — names are distinct, blocks are non-empty, counts are
symmetric. None of that would catch a wrong coefficient or a transposed slot. The precedent is
explicit: the B5 physicist-ERI convention bug was found only by injecting an FCI-correct oracle into
live C++ state, after it had cost days. A Python-side value gate is hours.

The rank-3 investigation in this repo is the other precedent — five hypotheses formed by inspection,
all wrong; every correct result came from a direct numerical comparison.

## Key code locations

| what | where |
|---|---|
| the fixture to sibling (do not modify) | `python/ccgen/tests/residual_eval.py:63` |
| UCC bridge + driver (what needs evaluating) | `ucc_spinterm_to_algebraterm`, `ucc_adapt_equations`, `python/ccgen/spin.py` |
| structural gates already in place | `U10UccAdaptEntryTests`, `U11BlockResolvedFactorNamesTests`, `test_spin.py` |
| PySCF oracle | `pyscf.cc.uccsd.update_amps`; interpreter is `tests/pyscf/.venv/bin/python` |
| the U1 ladder this unblocks | `docs/CCGEN_U1_UCC_ADAPT_SCOPE.md` (U1.2) |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`, which are canonical.
