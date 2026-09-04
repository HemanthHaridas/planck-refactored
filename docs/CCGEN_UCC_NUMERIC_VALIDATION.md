# UCC Spin-Blocked Residual Validation

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**How do you establish that a generated *unrestricted* (spin-blocked) CC residual is correct?**

## Short answer

ccgen's UCC residuals are correct at ranks 4 and 6. This is established by four mutually independent routes, none of which depends on the others being right:

| route | what it proves | measured |
|---|---|---|
| **FCI limit** | the equations, *solved*, give the exact energy | **3.7e-14** |
| **PySCF UCCSD**, rank 4 | direct external oracle | ~6e-16, all five blocks |
| **ccgen's own RCC**, closed shell | the blocks are self-consistent | triples 1.5e-12 vs ‖R‖~1e3 |
| **GCC-sliced**, rank 6 | the *adaptation* is the identity it claims | **1.6e-17** |

The first is the strongest and was the last to be built. Everything before it was a residual check or a transitive argument — GCC is FCI-exact and UCC == GCC sliced, therefore UCC must be right. Sound, but it never ran the UCC equations as equations: 25 call sites of `ucc_adapt_equations` existed and none solved to an energy. The FCI gate closes that.

Every gate is green. The one disagreement that stood open for a long stretch — `test_ucc_rank6_vs_pyscf`'s triples target at rel 1.9e-3 — turned out to be a defect in the comparison harness rather than in either code, and is now 2.3e-15. That investigation is kept in this document because what it cost is the most reusable part of the answer.

## Where the logic lives

- `python/ccgen/tests/test_ucc_vs_pyscf.py` — rank-4 PySCF gate
- `python/ccgen/tests/test_ucc_rank6_vs_pyscf.py` — rank-6 PySCF gate
- `F23ClosedShellOracleTests`, `test_spin.py` — closed-shell oracle, rank 4
- `U14c2RankSixClosedShellOracleTests`, `test_spin.py` — closed-shell oracle, rank 6
- `U14c3UccIsGccSlicedAtRankSixTests`, `test_spin.py` — the adaptation identity
- `U15UccReachesFciLimitTests`, `test_spin.py` — the UCC FCI limit
- `test_reference_vs_pyscf.py` (`*_reaches_fci_limit`) — GCC FCI-limit gates
- `ucc_residual_einsum`, `python/ccgen/tests/residual_eval.py` — the evaluator
- `ucc_random_tensors`, `python/ccgen/tests/residual_eval.py` — symmetry-correct random tensors, so a comparison cannot be fooled by a fixture that violates `<pq||rs> = <rs||pq>`
- `ucc_adapt_equations`, `python/ccgen/spin.py` — the adaptation

PySCF lives in `tests/pyscf/.venv`; the gates skip cleanly without it. Run them with that interpreter from `python/`.

## Why a spin-blocked residual needs its own validation story

Under RCC there is one spatial tensor per rank. Under UCC there are several blocks per rank (`t2_aaaa`, `t2_abab`, `t2_bbbb`), and three things become true at once that are not true for RCC:

1. The blocks are not independent at closed shell. `t2aa = t2ab - t2ab.transpose(0,1,3,2)`, and there is a rank-6 analogue. Any test that perturbs them separately is testing a state the equations never see.
2. A block's validity is not a permutation property. A real `aabaab` block has exactly two signed symmetries — antisymmetry in the α bra pair and the α ket pair — and antisymmetrized random noise has exactly the same two. No permutation test distinguishes them.
3. Structural checks pass on wrong equations. Names, term counts and slot order were all verified at rank 6 long before any value was, and all of them were correct while the adaptation itself was ungated.
4. A residual check is not an equation check. Comparing residuals at supplied amplitudes never exercises the equations as a *system*. Until the FCI gate, nothing solved the UCC manifold at all.

Consequence: for UCC, a fixture that looks valid and a manifold that is valid produce the same symptoms, and a wrong fixture is indistinguishable from a wrong equation.

## What invariants matter

### 1. A structural check can pass on wrong equations

Names, term counts and slot order were all verified at rank 6 long before any value was, and all of them were correct while the adaptation itself was ungated. See point 3 above.

Design rule:

- A gate on structure (names, shapes, term counts) is not evidence the underlying equations are correct. Follow it with a numeric gate before trusting the structure.

### 2. A residual check at supplied amplitudes does not exercise the equations as a system

25 call sites of `ucc_adapt_equations` existed with no gate that ever solved the manifold to an energy — every check was a residual evaluation at hand-supplied amplitudes. The FCI-limit gate (route 0 below) is the one that actually solves the system.

Design rule:

- Prefer a gate that solves the equations to self-consistency and compares the resulting energy against an independent exact reference (FCI, where affordable) over one that only evaluates the residual at a fixed point.

### 3. A gate's system choice can make it vacuous without the gate itself being wrong

For the FCI-limit gate, the obvious system choice is a vacuous gate. Li/STO-3G passes at 3.8e-14 — but holding `t3` at zero gives a *bit-identical* energy, because Li is nearly a one-electron correlation problem and its `t3` blocks converge to ~1e-19, so a completely broken T3 passes. On LiH+/6-31g the triples are worth 8.1e-8, 2000× the tolerance.

Design rule:

- Pin an explicit "the higher-rank term is load-bearing" assertion (`test_triples_are_load_bearing`) alongside any gate whose system could otherwise make a whole term family numerically inert. Without it the gate silently degrades to testing a lower-rank method.

### 4. When a discrepancy survives repeated narrowing, stop inferring across the interface and drive the other side directly

The rank-6 PySCF disagreement (below) was chased through nine hypotheses, each a reasonable inference about a convention difference, none of which could have been right — because the two sides were never evaluating at the same amplitudes in the first place. The decisive move was to stop treating PySCF as a black box that emits a residual and instead call its residual function (`compute_r3_tri_uhf`) directly with controlled intermediates.

Design rule:

- When a discrepancy survives several rounds of hypothesis-and-test across an interface, stop inferring from one side's behavior and instead drive the other side's own internal function directly, at the same explicit inputs.
- A reconstruction is part of the measuring instrument, not part of the measurement. `R = (t_new - t)·D` looks like an identity and is one only if `t_new` was actually computed at `t`. Prefer the quantity the other implementation actually computes over anything you have to invert.

### 5. When a question survives several rounds of narrowing, check what is unverified rather than narrowing again

The path to finding the real gap (see "What actually settled it" below) went through four rescopes, each locally correct about the thing it examined, none of which was the actual gap. The answer was in what nobody had examined: the adaptation itself had no numeric gate.

Design rule:

- If repeated rescoping keeps dissolving candidate blockers without closing the question, grep for what has never been checked at all, rather than narrowing the current candidate further.

### 6. Check the cheapest sanity number before bisecting

The re-antisymmetrization defect (see "The four interface defects" below) showed up as ‖ref‖ = 1.7e-1 at converged amplitudes, where PySCF's own residual is ~1e-10 — a number that, checked first, would have named the defect immediately.

Design rule:

- Before bisecting a discrepancy, check the cheapest available sanity number (e.g. the reference residual's own magnitude at the amplitudes being used).

## What was measured (the four validation routes)

### 0. The FCI limit — `U15UccReachesFciLimitTests`

For a 3-electron system CCSDT is exact, so solving the generated UCC manifold to self-consistency must give `UHF + E_corr == FCI`. Jacobi over all nine spin blocks, each with its own spin-resolved denominator.

```
LiH+/6-31g doublet   UCCSDT  -7.719839924447
                     FCI     -7.719839924447     diff 3.7e-14
```

What a 3-electron gate can and cannot reach: with `noa=2, nob=1` the same-spin `t3` blocks are structurally empty — three distinct same-spin occupieds do not exist — so `aaaaaa` / `bbbbbb` are never exercised. `t3_aabaab` needs 2α + 1β and *is* populated. So this route verifies the mixed-spin block only, which is why routes 1–3 still carry weight: they reach the same-spin blocks that FCI at 3 electrons cannot (it is also the block that the rank-6 investigation kept implicating — see below for why that signal was misleading).

### 1. Direct external oracle — `test_ucc_vs_pyscf` (rank 4)

Evaluate every UCC target at PySCF UCCSD's own perturbed amplitudes; recover PySCF's residual as `R = (t_new - t) * D`. All five blocks agree to ~6e-16.

Three conventions this pinned, each of which cost a wrong conclusion first:

- The closed-shell oracle is a per-target pairing, not a block sum. Collapsing a UCC manifold to its restricted limit does not mean adding the blocks up: each spatial target is reproduced by one specific combination of spin blocks, and which combination depends on the target. Summing instead produces a number that is wrong by a factor which varies per target — plausible, and hard to attribute.
- The amplitude mapping is a transpose, not a rename. The names correspond one-for-one (`t2ab` ↔ `t2_abab`) but PySCF stores `(occ…, vir…)` and ccgen emits `(vir…, occ…)`. On `t2_bbbb`, which is square, this is invisible — so the shape assertions are on the asymmetric blocks.
- `f_ov` must be zeroed on both sides. Planck CC kernels are canonical-Fock by construction, so ccgen's `f_ov` terms are runtime-zero; PySCF's `f_ov` is SCF convergence noise (~8e-9) that `update_amps` uses. Zeroing one side is worse than zeroing neither: 8e-9 → 9e-9 → 6e-17.
- Do not evaluate at converged amplitudes (reference is then ~1e-8, so a kernel returning zero passes) and do not use OH/STO-3G (`nva=1` makes the `aaaa` block identically zero).

### 2. Self-consistency — `U14c2RankSixClosedShellOracleTests`

At closed shell the UCC residual must reproduce ccgen's own RCC residual. Needs no PySCF, so it localizes a defect to the evaluator rather than the physics — which is why it is worth running before any external oracle.

This is the gate that needs the closure relations, and both are load-bearing and asserted to be: corrupting `t3_abbabb` breaks it by 89.0 and `t3_aaaaaa` by 51.3, against ‖R‖~1e3. Without that assertion the gate would be measuring the fixture.

### 3. The adaptation identity — `U14c3UccIsGccSlicedAtRankSixTests`

The UCC manifold is by definition the GCC manifold resolved into spin blocks. This checks that directly: evaluate the GCC triples residual on spin-orbital tensors, slice the all-α block, require UCC `triples_aaaaaa` to equal it. 1.6e-17 against ‖G‖ 2.9e-2.

Combined with ccgen's GCC CCSDT reaching the FCI limit exactly (three gates in `test_reference_vs_pyscf`, including the `engine="diagram"` path this manifold is generated through), this makes rank-6 UCC correct:

```
GCC correct (FCI-exact)  +  UCC == GCC sliced  =>  UCC correct
```

Two fixture requirements, both learned by getting them wrong:

- the spin-orbital tensors must carry the real even=α/odd=β interleaving (`_uccsdt_so_tensors`). `random_tensors` is spin-free; slicing it by that convention gives disagreements of order the residual itself — 3.1e2 against ‖G‖ 3.6e2, which reads exactly like an adaptation defect;
- the comparison must be perturbed off convergence, or everything is ~1e-13 and it passes vacuously.

## What was found

**What actually settled the investigation:** the GCC→UCC adaptation had 22 call sites and no numeric gate. It was verified structurally at rank 6 and never against a value. That was the gap. Every other step in the investigation was downstream of it, and the fact was available from `grep` at any point.

The route there was four successive rescopes of the same step, each of which correctly dissolved its candidate blocker and none of which was the actual gap:

| rescope | blocker claimed | why it was wrong |
|---|---|---|
| original | "PySCF has no UCCSDT `update_amps`" | it ships `update_amps_uccsdt_tri_`; `UCCSDT.update_amps` is the *inherited CCSD* one and silently omits t3 |
| second | "the t3 closure relation is underived" | it was pinned green in-tree; three hand-derivations had failed because they built a *spatial* identity where the relation is a spin-orbital slice |
| third | "rank 6 has a spin-flip defect" | a one-line fixture bug (`t3_abbabb` set equal to `t3_aabaab`); retracted the same day |
| fourth | "the two closures are inequivalent" | they are equivalent *on valid blocks*; they diverge only on inputs that are not amplitude blocks |

**The four interface defects found along the way**, each of which reads as an equation defect (three of the four were found only after the equations had been cleared):

1. The re-antisymmetrization was unnormalized. PySCF's `t2aa`/`t3aaa` arrive already antisymmetric, so re-applying `a - a.transpose(...)` does not project — it multiplies, by 4× and 36×. The tell was ‖ref‖ = 1.7e-1 at converged amplitudes where PySCF's own residual is ~1e-10.
2. `t2aa` is determined by `t2ab` — they cannot be perturbed separately.
3. `t3aaa` is determined by `t3aab`, via the same-spin closure.
4. A block with the right antisymmetries is still not a valid amplitude block (see invariant 1 above). The operational test is that the two same-spin closure forms agree; the fix is to build the perturbation as a slice of a genuine antisymmetric spin-orbital tensor (`_valid_t3_blocks`).

Two rank-6 storage facts that cost time and are not documented in PySCF:

- t3 is stored packed (`i<j<k, a<b<c`); `tamps_tri2full_uhf` unpacks, `tamps_full2tri_uhf` repacks, round trip exact.
- `aab` and `bba` are one stored sector. Perturbing them independently makes the two sides see different t3 — and it surfaces as a singles error (5e-14 → 8.9e-3), far from its cause.

**The rank-6 gap itself:** for a long stretch `test_ucc_rank6_vs_pyscf`'s triples target disagreed with PySCF by rel 1.9e-3 while singles and doubles were exact at ~1e-15. It is now 2.3e-15 — machine precision, like everything else.

The defect was in the comparison harness. PySCF's `update_amps_uccsdt_tri_` runs its T1/T2 half first and updates `t1`/`t2` in place, then builds the T3 intermediates from the already-updated amplitudes. Measured: `t1` moves by 9.4e-3 and `t2` by 4.8e-2 before the t3 half begins. So `R = (t3_new - t3)·D` recovered through that entry is the residual at *different amplitudes* than the ones handed to ccgen. The fix is to call `compute_r3_tri_uhf` directly. Singles and doubles keep using `update_amps` — their halves run before any in-place mutation, so their round trip is faithful and they agree at ~1e-16 either way.

That asymmetry is why the bug survived: it presented as a triples-only defect in a file whose singles and doubles were exact, which is precisely the signature of a T3-equation error. Nine hypotheses were formed and falsified against it:

| hypothesis | killed by |
|---|---|
| physicist/chemist ERI convention | PySCF's `pppp` equals this gate's construction to 5.4e-15 |
| T1 dressing | zeroing t1 on both sides changes nothing |
| F/W dressing | reduces to `diag(eps)` to 1.1e-7 at `t1=t2=0`; the gap survives with dressing off |
| the packed wedge | difference is the same size on and off the strict wedge |
| `t3_aabaab` / `t3_abbabb` layouts | every shape-legal signed permutation tried |
| the denominator | matches PySCF's `eijkabc` to 2.8e-14 |
| a spurious or over-counted term family | the 18 `t2·v` terms are textbook `P(i/jk)P(a/bc)` expansions |
| a rank-6 spin-flip defect | a one-line fixture bug, retracted the same day |
| two inequivalent closures | they are equivalent on valid blocks |

Every one of those was a hypothesis about a convention, and none could have been right, because the two sides were never evaluating at the same amplitudes. Bisecting produced real-looking signals that pointed nowhere: the "mixed-spin coupling is best-aligned at cos 0.74" observation was a property of the amplitude difference, not of any term family. Driving PySCF's own `compute_r3_tri_uhf` with controlled intermediates and comparing the raw residual, at `t1 = t2 = 0` it matched ccgen to 1.2e-15 immediately, while the harness's reconstruction differed from that same raw residual by 3.7e-3 — locating the defect in the reconstruction rather than in either equation, in one measurement.

## Validation strategy that should remain in place

- All four routes above (FCI limit, PySCF UCCSD rank 4, self-consistency against RCC, GCC-sliced identity at rank 6), each independent of the others
- `test_triples_are_load_bearing` alongside the FCI-limit gate, so it cannot silently degrade to a UCCSD-only test
- Running the PySCF-dependent gates through `tests/pyscf/.venv` specifically
