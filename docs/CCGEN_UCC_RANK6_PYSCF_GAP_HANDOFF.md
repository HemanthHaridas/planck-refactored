# Validating a spin-blocked CC residual, and the rank-6 gap that is still open

Canonical status lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file is a **handoff**: it answers one architecture question and hands over
one unfinished investigation. It becomes an architecture doc when the
investigation closes; until then the open half is scoped in place at the end.

**The question:** how do you establish that a generated *unrestricted* (spin-blocked)
CC residual is correct — including "is it verified against FCI?" — and what does
the one remaining rank-6 disagreement with PySCF actually mean?

---

## The short answer

ccgen's UCC residuals are correct at ranks 4 and 6. This is established by four
mutually independent routes, none of which depends on the others being right:

| route | what it proves | measured |
|---|---|---|
| **FCI limit** | the equations, *solved*, give the exact energy | **3.7e-14** |
| **PySCF UCCSD**, rank 4 | direct external oracle | ~6e-16, all five blocks |
| **ccgen's own RCC**, closed shell | the blocks are self-consistent | triples 1.5e-12 vs ‖R‖~1e3 |
| **GCC-sliced**, rank 6 | the *adaptation* is the identity it claims | **1.6e-17** |

The first is the strongest and was the last to be built. Everything before it was
a *residual* check or a transitive argument — GCC is FCI-exact and UCC == GCC
sliced, therefore UCC must be right. Sound, but it never ran the UCC equations as
equations: 25 call sites of `ucc_adapt_equations` existed and none solved to an
energy. The FCI gate closes that; see *The FCI limit*.

A single disagreement remains: `test_ucc_rank6_vs_pyscf`'s `triples_aaaaaa`
differs from PySCF by **rel 1.9e-3**, kept as an `expectedFailure`. It is **not**
a defect in either code — both are independently FCI-exact — and the open half of
this doc scopes what to do about it.

---

## Why a spin-blocked residual needs its own validation story

Under RCC there is one spatial tensor per rank. Under UCC there are several
blocks per rank (`t2_aaaa`, `t2_abab`, `t2_bbbb`), and three things become true
at once that are not true for RCC:

1. **The blocks are not independent at closed shell.** `t2aa = t2ab -
   t2ab.transpose(0,1,3,2)`, and there is a rank-6 analogue. Any test that
   perturbs them separately is testing a state the equations never see.
2. **A block's validity is not a permutation property.** A real `aabaab` block
   has exactly two signed symmetries — antisymmetry in the α bra pair and the α
   ket pair — and antisymmetrized random noise has exactly the same two. **No
   permutation test distinguishes them.**
3. **Structural checks pass on wrong equations.** Names, term counts and slot
   order were all verified at rank 6 long before any value was, and all of them
   were correct while the adaptation itself was ungated.
4. **A residual check is not an equation check.** Comparing residuals at supplied
   amplitudes never exercises the equations as a *system*. Until the FCI gate,
   nothing solved the UCC manifold at all.

Consequence: for UCC, *a fixture that looks valid and a manifold that is valid
produce the same symptoms*, and a wrong fixture is indistinguishable from a wrong
equation. Every gate below is built to separate those two.

---

## The four routes, and what each one is for

### 0. The FCI limit — `U15UccReachesFciLimitTests`

For a 3-electron system CCSDT is exact, so solving the generated UCC manifold to
self-consistency must give `UHF + E_corr == FCI`. Jacobi over all nine spin
blocks, each with its own spin-resolved denominator.

```
LiH+/6-31g doublet   UCCSDT  -7.719839924447
                     FCI     -7.719839924447     diff 3.7e-14
```

**The system choice is load-bearing, and the obvious pick is a vacuous gate.**
Li/STO-3G also passes, at 3.8e-14 — but holding `t3` at zero gives a
*bit-identical* energy. Li is nearly a one-electron correlation problem and its
`t3` blocks converge to ~1e-19, so a completely broken T3 passes. On LiH+/6-31g
the triples are worth **8.1e-8**, 2000× the tolerance.
`test_triples_are_load_bearing` pins this, because without it the gate silently
degrades to a UCCSD test.

**What a 3-electron gate can and cannot reach.** With `noa=2, nob=1` the
same-spin `t3` blocks are structurally *empty* — three distinct same-spin
occupieds do not exist — so `aaaaaa` / `bbbbbb` are never exercised. `t3_aabaab`
needs 2α + 1β and *is* populated. So this route verifies the mixed-spin block
only, which is why routes 1–3 still carry weight: they reach the same-spin blocks
that FCI at 3 electrons cannot. (It is also, coincidentally, the block implicated
in the open gap below.)

### 1. Direct external oracle — `test_ucc_vs_pyscf` (rank 4)

Evaluate every UCC target at PySCF UCCSD's own perturbed amplitudes; recover
PySCF's residual as `R = (t_new - t) * D`. All five blocks agree to ~6e-16.

Three conventions this pinned, each of which cost a wrong conclusion first:

- **The amplitude mapping is a transpose, not a rename.** The names correspond
  one-for-one (`t2ab` ↔ `t2_abab`) but PySCF stores `(occ…, vir…)` and ccgen
  emits `(vir…, occ…)`. On `t2_bbbb`, which is square, this is invisible — so the
  shape assertions are on the asymmetric blocks.
- **`f_ov` must be zeroed on BOTH sides.** Planck CC kernels are canonical-Fock
  by construction, so ccgen's `f_ov` terms are runtime-zero; PySCF's `f_ov` is
  SCF convergence noise (~8e-9) that `update_amps` uses. Zeroing one side is
  *worse* than zeroing neither: 8e-9 → 9e-9 → 6e-17.
- **Do not evaluate at converged amplitudes** (reference is then ~1e-8, so a
  kernel returning zero passes) **and do not use OH/STO-3G** (`nva=1` makes the
  `aaaa` block identically zero).

### 2. Self-consistency — `U14c2RankSixClosedShellOracleTests`

At closed shell the UCC residual must reproduce ccgen's own RCC residual. Needs
no PySCF, so it localizes a defect to the evaluator rather than the physics —
which is why it is worth running before any external oracle.

This is the gate that needs the closure relations, and **both are load-bearing
and asserted to be**: corrupting `t3_abbabb` breaks it by 89.0 and `t3_aaaaaa` by
51.3, against ‖R‖~1e3. Without that assertion the gate would be measuring the
fixture.

### 3. The adaptation identity — `U14c3UccIsGccSlicedAtRankSixTests`

The UCC manifold is *by definition* the GCC manifold resolved into spin blocks.
This checks that directly: evaluate the GCC triples residual on spin-orbital
tensors, slice the all-α block, require UCC `triples_aaaaaa` to equal it.
**1.6e-17** against ‖G‖ 2.9e-2.

Combined with ccgen's GCC CCSDT reaching the FCI limit exactly (three gates in
`test_reference_vs_pyscf`, including the `engine="diagram"` path this manifold is
generated through), this makes rank-6 UCC correct:

```
GCC correct (FCI-exact)  +  UCC == GCC sliced  =>  UCC correct
```

Two fixture requirements, both learned by getting them wrong:

- the spin-orbital tensors must carry the real even=α/odd=β interleaving
  (`_uccsdt_so_tensors`). `random_tensors` is spin-**free**; slicing it by that
  convention gives disagreements of order the residual itself — 3.1e2 against ‖G‖
  3.6e2, which reads exactly like an adaptation defect;
- the comparison must be perturbed off convergence, or everything is ~1e-13 and
  it passes vacuously.

---

## What actually settled it

**The GCC→UCC adaptation had 22 call sites and no numeric gate.** It was verified
structurally at rank 6 and never against a value. That was the gap. Every other
step in the investigation was downstream of it, and the fact was available from
`grep` at any point.

The route there was four successive rescopes of the same step, each of which
correctly dissolved its candidate blocker and none of which was the actual gap:

| rescope | blocker claimed | why it was wrong |
|---|---|---|
| original | "PySCF has no UCCSDT `update_amps`" | it ships `update_amps_uccsdt_tri_`; `UCCSDT.update_amps` is the *inherited CCSD* one and silently omits t3 |
| second | "the t3 closure relation is underived" | it was pinned green in-tree; three hand-derivations had failed because they built a *spatial* identity where the relation is a spin-orbital slice |
| third | "rank 6 has a spin-flip defect" | a one-line fixture bug (`t3_abbabb` set equal to `t3_aabaab`); retracted the same day |
| fourth | "the two closures are inequivalent" | they are equivalent *on valid blocks*; they diverge only on inputs that are not amplitude blocks |

**The generalizable lesson: when a question survives several rounds of narrowing,
check what is *unverified* before narrowing again.** Each rescope was locally
correct reasoning about the thing it examined, and the answer was in what nobody
had examined.

---

## The four interface defects

These are worth their own section because each *reads* as an equation defect, and
three of the four were found only after the equations had been cleared.

1. **The re-antisymmetrization was unnormalized.** PySCF's `t2aa`/`t3aaa` arrive
   *already* antisymmetric, so re-applying `a - a.transpose(...)` does not
   project — it **multiplies**, by 4× and 36×. The tell was ‖ref‖ = 1.7e-1 at
   converged amplitudes where PySCF's own residual is ~1e-10. **That one number
   should have been checked before any bisecting**, and checking it is the
   cheapest sanity test in this whole document.
2. **`t2aa` is determined by `t2ab`** — they cannot be perturbed separately.
3. **`t3aaa` is determined by `t3aab`**, via the same-spin closure.
4. **A block with the right antisymmetries is still not a valid amplitude
   block.** See point 2 of *Why a spin-blocked residual needs its own validation
   story*. The operational test is that the two same-spin closure forms agree;
   the fix is to build the perturbation as a slice of a genuine antisymmetric
   spin-orbital tensor (`_valid_t3_blocks`).

Two rank-6 storage facts that cost time and are not documented in PySCF:

- **t3 is stored packed** (`i<j<k, a<b<c`); `tamps_tri2full_uhf` unpacks,
  `tamps_full2tri_uhf` repacks, round trip exact.
- **`aab` and `bba` are ONE stored sector.** Perturbing them independently makes
  the two sides see different t3 — and it surfaces as a **singles** error
  (5e-14 → 8.9e-3), far from its cause.

---

## OPEN: the residual rank-6 triples gap

`test_ucc_rank6_vs_pyscf::test_triples_reproduce_pyscf`, `expectedFailure`.

| target | rel difference |
|---|---|
| `singles_aa` / `singles_bb` | 5.5e-15 |
| `doubles_aaaa` / `doubles_bbbb` | 7.7e-15 |
| **`triples_aaaaaa`** | **1.9e-3** |

Down from 8.8e-2 after the four interface fixes.

### What is established about it

- **Neither code is wrong.** ccgen's GCC CCSDT is FCI-exact, and PySCF's UCCSDT
  is too — 3.7e-11 vs `fci.FCI` on H3/6-31g doublet. So this is a property of the
  comparison.
- **It vanishes without a t3 perturbation.** With PySCF's own untouched t3 on
  both sides, triples agree to 2.7e-11 against a 6.2e-11 reference — i.e. at the
  convergence floor, like everything else.
- **It survives with all dressing switched off.** At `t1 = t2 = 0`, where PySCF's
  `F`/`W` intermediates reduce to bare integrals, triples still differ by rel
  5.0e-3 with only **30 of 579 terms alive**.
- **The mixed-spin coupling is the best-aligned family.** Of those 30,
  `('t3_aabaab', 'v_abab')` — nine terms — has cos 0.74 with the difference,
  |S| 6.1e-3 against |D| 3.6e-3. No single term explains it (max per-term cosine
  0.22).

### Hypotheses falsified, each by measurement

Start past these:

| hypothesis | killed by |
|---|---|
| physicist/chemist ERI convention | PySCF's `pppp` is the non-antisymmetrized `<pq|rs>`, equal to this gate's construction to 5.4e-15; the gate's blocks carry the symmetries ccgen requires |
| T1 dressing | zeroing t1 on both sides changes nothing |
| F/W dressing | `F_oo`/`F_vv` diagonals differ from bare `eps` by 0.046/0.027, but reduce to `diag(eps)` to 1.1e-7 at `t1=t2=0` — the dressing is entirely `t2·v`, which ccgen carries explicitly |
| the packed wedge | the difference is the same size on and off the strict wedge |
| `t3_aabaab` / `t3_abbabb` layouts | every shape-legal signed permutation tried; the current ones are best |
| the denominator | `D3` matches PySCF's `eijkabc` to 2.8e-14; `focka.diagonal() == mo_energy`; `level_shift = 0` |
| a spurious or over-counted term family | the 18 `t2·v` terms are textbook `P(i/jk)P(a/bc)` expansions; no integer combination fits |

### The next step, and why it is different in kind

**Evaluate PySCF's `compute_r3aaa_tri_uhf` contraction by contraction against its
own intermediates**, rather than through `update_amps_uccsdt_tri_`.

Every hypothesis above shares one flaw: it was formed by reading one side and
testing it against the other. Nine such hypotheses have now been falsified. The
remaining move is to stop inferring and measure PySCF's individual contractions
directly — its `r3aaa` is ~30 named `einsum` calls against `F_oo`, `F_vv`,
`W_oooo`, `W_ovoo`, `W_vvvo`, `W_vvvv`, `W_voov`, `W_vOoV`, each of which can be
reproduced independently and compared to the matching ccgen term group.

**Do not** resume max-norm bisecting on the aggregate residual. The full residual
shows 38× cancellation, so max-norm cannot separate a small systematic difference
from a large one that mostly cancels.

### Whether it blocks anything

**No.** Rank-4 and rank-6 UCC are both validated by the three routes above. U2+
should not be held on this. An unexpected PASS on that test is the signal that
someone diagnosed PySCF's side.

---

## Where the code is

| what | where |
|---|---|
| rank-4 PySCF gate | `python/ccgen/tests/test_ucc_vs_pyscf.py` |
| rank-6 PySCF gate (the open one) | `python/ccgen/tests/test_ucc_rank6_vs_pyscf.py` |
| closed-shell oracle, rank 4 | `F23ClosedShellOracleTests`, `test_spin.py` |
| closed-shell oracle, rank 6 | `U14c2RankSixClosedShellOracleTests`, `test_spin.py` |
| the adaptation identity | `U14c3UccIsGccSlicedAtRankSixTests`, `test_spin.py` |
| **the UCC FCI limit** | `U15UccReachesFciLimitTests`, `test_spin.py` |
| GCC FCI-limit gates | `test_reference_vs_pyscf.py` (`*_reaches_fci_limit`) |
| the evaluator | `ucc_residual_einsum`, `python/ccgen/tests/residual_eval.py` |
| the adaptation | `ucc_adapt_equations`, `python/ccgen/spin.py` |

PySCF lives in `tests/pyscf/.venv`; the gates skip cleanly without it. Run them
with that interpreter from `python/`.
