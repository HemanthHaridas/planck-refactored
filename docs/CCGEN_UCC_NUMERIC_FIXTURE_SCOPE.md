# A spin-resolved evaluation fixture for the UCC numeric gate

**F1, F2 and F3 are all LANDED. This ladder is complete.** The UCC residuals are validated against
PySCF UCCSD on CH3/STO-3G to **~6e-16** in every block — machine precision, not the scoped 1e-10.
U1.2 landed on the back of it, and **U1 is now complete** — see `CCGEN_U1_UCC_ADAPT_SCOPE.md`.

Landed surface: `ucc_random_tensors` + `ucc_closed_shell_tensors` + `ucc_residual_einsum` +
`ucc_resolve_factor` (`python/ccgen/tests/residual_eval.py`), `ucc_term_spins`
(`python/ccgen/spin.py`), gated by `F23ClosedShellOracleTests` (`test_spin.py`) and
`F3UccVsPyscfTests` (`test_ucc_vs_pyscf.py`). The F2 detail lives in
`CCGEN_UCC_RESIDUAL_EVALUATOR_SCOPE.md`.

## The blocker, measured

`residual_einsum` (`python/ccgen/tests/residual_eval.py:63`) assumes **one** orbital space pair and
**one** spin-free ERI tensor:

```python
occ, vir = slice(0, no), slice(no, n)          # a single (no, nv)
sl = tuple(occ if i.space == "occ" else vir for i in f.indices)
ops.append(tensors[f.name][sl])                 # one v / one f, sliced by space alone
```

UCC breaks both. On CH3/STO-3G the α and β spaces have **different dimensions** —
`noa=5 nva=3`, `nob=4 nvb=4` (measured; an earlier draft of this doc said `nva=4`/`nvb=5`) — and a
mixed block like `t2_abab` indexes both at once. A factor's
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

The *names* map one-for-one. **The arrays do not** — PySCF stores `(occ…, vir…)` and ccgen emits
`(vir…, occ…)`, so each one needs its halves transposed. An earlier draft of this doc called the
mapping "a rename, not a transform … the cheap half"; F2.4 measured otherwise. See F3 below.

**Constraint that shapes the design:** `residual_einsum` / `random_tensors` have **seven** consumers
(`test_spin.py` alone calls them 57 times, plus `test_diagram`, `test_regressions`,
`test_residual_symmetry`, `test_reference_vs_pyscf`, `test_dress_per_operator`,
`test_emit_flag_matrix`). Changing their signature is a large blast radius on gates that currently
pass. **Add a spin-resolved sibling; do not retrofit the RCC one.** Same rule U1.0/U1.1 followed —
the RCC path stayed byte-identical because UCC got its own entry rather than a parameter on the
shared one.

## The work

### F1 — a spin-resolved tensor bundle (~S) — **LANDED**

`ucc_random_tensors(noa, nva, nob, nvb, seed)` returning blocks keyed by the names the UCC bridge
emits (`t1_aa`, `t2_abab`, `v_aaaa`, …), with the antisymmetry each block actually has:
`aaaa`/`bbbb` antisymmetric in bra and ket independently, `abab` **not** (its two halves are
different spin spaces).

*Verify:* shapes match PySCF's `t1a/t1b/t2aa/t2ab/t2bb` for a real open-shell case, and each
same-spin block satisfies its antisymmetry to 1e-14. Getting `abab`'s (non-)symmetry wrong is the
easiest way to write a fixture that silently disagrees with PySCF.

### F2 — `ucc_residual_einsum` (~M) — **LANDED**, see `CCGEN_UCC_RESIDUAL_EVALUATOR_SCOPE.md`

Same einsum construction, but the slice for each factor index is chosen by **(space, spin)** rather
than space alone, and `v`/`f` are looked up per block instead of sliced from one tensor.

*Verify:* on a *closed-shell* system the UCC evaluator must reproduce the existing RCC
`residual_einsum` result for the same equations. **Two corrections found when building it:** it is a
per-target pairing (`doubles_abab` ↔ `doubles`), **not** a sum over blocks, because RCC adapts on the
closed-shell representative block; and it is **not** free of a new fixture — F1's independent draws
violate the closure relations, so `ucc_closed_shell_tensors` exists for it. Measured 3.9e-12 against
‖R‖~1.6e3. It does still need no PySCF, and it did catch defects the PySCF gate would have
conflated.

### F3 — U1.2, the PySCF UCCSD gate (~M) — **LANDED**, and the mapping was not what this scope said

Evaluate every UCC target at PySCF's amplitudes against `pyscf.cc.uccsd.update_amps`, converting
`t_new` back to a residual via `R = (t_new − t)·D`. **Result: ~6e-16 in all five blocks.** Gated at
1e-13 rather than the scoped 1e-10 — the agreement is at machine precision, and a 1e-10 bound would
pass a real defect of size 1e-11 unnoticed.

Both traps this scope recorded were real and are asserted in the gate:

- **Do not evaluate at converged amplitudes.** Confirmed: reference `|R|` there is ~1e-8, so a
  kernel returning zero passes. The gate perturbs first and re-imposes aa/bb antisymmetry after.
- **Do not use OH/STO-3G.** `noa=5 nva=1` gives `C(1,2)=0` distinct αα doubles. CH3/STO-3G is used,
  and the gate asserts every reference block exceeds 1e-2 (measured 0.34 … 1.04).

#### Two corrections to this scope, found by F2.4

**The amplitude mapping is a TRANSPOSE, not a rename.** This doc said "the amplitude mapping is a
rename, not a transform. That is the cheap half." The *names* do correspond one-for-one, but PySCF
stores `(occ…, vir…)` while ccgen emits `(vir…, occ…)`, so every array needs its halves swapped. On
`t2_bbbb` (4,4,4,4) that is invisible — which is exactly how a layout error hides — so the gate
asserts shapes on the **asymmetric** blocks instead.

**The dims are not the ones scoped here.** CH3/STO-3G is `noa=5 nva=3, nob=4 nvb=4`, not the
`noa=5 nva=4, nob=4 nvb=5` recorded above. Still non-trivial in every block and still non-square in
alpha, so the case selection stands — only the numbers were wrong.

#### The convention that took the investigation: `f_ov` must be zeroed on BOTH sides

Every Planck CC kernel gets a canonical Fock by construction (`f_ov = 0` identically), so the `f_ov`
terms ccgen carries are runtime-zero. PySCF's `f_ov` is *not* exactly zero — it is SCF convergence
noise, ~8e-9 even at `conv_tol=1e-14` — and `update_amps` uses it. Measured on `singles_aa`:

| | maxdiff |
|---|---|
| neither side zeroed | ~8e-9 (both carry the same noise) |
| ccgen side only | ~9e-9 — slightly **worse**, not better |
| both sides zeroed | **~6e-17** |

Zeroing one side is worse than zeroing neither: the two routes then disagree about whether the
`f_ov` terms are present at all. The first hypothesis — that ccgen's `f_ov` handling was the defect
— was falsified by exactly this. It is asserted as its own test so the convention cannot be quietly
dropped.

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
