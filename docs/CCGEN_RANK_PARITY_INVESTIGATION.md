# Is the generated-kernel defect rank-parity dependent?

**Research scope, coupled to `CCGEN_RANK3_TRIPLES_DEFECT.md`.** Each step's verification is
part of the step — this investigation has already produced five fixes that passed a structural
gate and made the physics worse.

## The hypothesis

The rank-3 (CCSDT) generated triples residual is wrong. The rank-4 (CCSDTQ) generated kernel
reaches FCI. **If even ranks are correct and odd ranks are wrong, the bug surface collapses
dramatically** — it would point at something that distinguishes odd from even excitation
manifolds, rather than at rank-3-specific code.

That would be a large constraint, so it is worth testing directly rather than assuming.

## What the current evidence actually supports — and does not

| rank | generated kernel | numeric gate | verdict |
|---|---|---|---|
| 2 (CCSD) | emitted as `ccsd_planck_generated.cpp` | **none — the TU has NO consumer** | **unvalidated**, not validated |
| 3 (CCSDT) | `ccsdt_planck_generated.cpp`, wired `64d0074` | `bh3` raw-residual probe | **WRONG** (~45 % of elements written, unrelated values) |
| 4 (CCSDTQ) | `ccsdtq_planck_generated.cpp`, registry | `be_rccsdtq_sto3g` == FCI, atol 1e-7 | **passes** |
| 5, 6 | `cc5`/`cc6`, registry by `MAXORDER` | **none** (`grep` for cc5/cc6 in regression_cases.json → 0) | **unvalidated** |

**So the parity hypothesis currently rests on exactly two data points: rank 3 fails, rank 4
passes.** Rank 2 is not evidence for the "even is correct" half — its generated TU is compiled
into nothing, so it has never been executed. Ranks 5 and 6 have no gate at all.

Two of the four ranks being untested is why this is an investigation and not a conclusion.

### A confound that must be controlled

`be_rccsdtq_sto3g` uses **Be, which has 4 electrons**. At rank 4 the CC expansion is complete
for a 4-electron system, so CCSDTQ ≡ FCI *as a method*. That does not make the gate vacuous — a
wrong kernel still would not reach FCI — but it does mean the rank-4 pass is measured in the one
regime where the T4 manifold is maximally constrained. It is possible for a kernel to be right
at the exactness limit and wrong in general.

**Any parity claim needs at least one rank-4 test on a system with more than 4 electrons**,
where CCSDTQ is genuinely approximate. Otherwise "even ranks are correct" may be an artifact of
the only even-rank test being an exactness case.

---

## An independent oracle exists — use it

`tests/pyscf/.venv` has PySCF 2.13.0, which ships **`pyscf.cc.rccsdt` AND `pyscf.cc.rccsdtq`**
(plus `_highm` variants). Verified running on Be/STO-3G with `mol.cart = True`:

```
RHF               -14.3518804762
RCCSDT   e_corr = -0.0517702756   e_tot = -14.4036507518
RCCSDTQ  e_corr = -0.0517746320   e_tot = -14.4036551082
```

Two consequences that change this investigation:

**1. Planck's rank-4 gate agrees with an independent implementation to 1e-10** (Planck asserts
−14.4036551081; PySCF gives −14.4036551082). The even-rank data point is stronger than "it reaches
FCI" — it matches a second code. That does not remove the exactness confound (P2 still applies), but
it does mean the rank-4 kernel is not merely self-consistent.

**2. Rank 3 now has an oracle it never had.** Every rank-3 comparison so far has been
generated-vs-hand-written *within Planck*, which can establish that they differ but not which is
wrong. `pyscf.cc.rccsdt` resolves that.

Both modules expose residual-level entry points — `compute_r1r2`, `compute_r3` (rccsdt),
`compute_r4_tri` (rccsdtq) — so the comparison can be made **at fixed amplitudes on the residual**,
which is what this investigation requires: `restore_restricted_t3_structure` masks residual error
11–29×, so an energy-level comparison understates the defect by an order of magnitude.

**Caveat, and it is the same class of bug this investigation exists for.** These are PySCF's
reference implementations and use their own amplitude storage — the `_tri` suffixes indicate packed
triangular form. Mapping Planck's dense `Tensor6D` onto that layout is real work and a place where a
convention error can silently reappear. **Build any oracle comparison at `nv != no`**, so a
wrongly-ordered read raises instead of returning a plausible wrong number.

## The investigation

### P1 — ANSWERED: rank 2 is CORRECT

ccgen's spatial rank-2 residual vs PySCF `rccsd.update_amps`, **identical amplitudes**,
LiH/STO-3G cart (`no=2, nv=4` — asymmetric by design):

| manifold | ‖ccgen‖ | ‖pyscf‖ | ‖diff‖ |
|---|---|---|---|
| singles | 0.031767 | 0.031767 | **1.3e-08** |
| doubles | 0.323223 | 0.323223 | **1.5e-09** |

Against rank 3's failure — 45 % of elements unwritten, ratios spanning −149 to +66 — this is
agreement, not a milder version of the same defect.

**The residual ~1e-8 is explained.** `|f_ov| = 1.27e-08` on this system, matching the max absolute
deviation of 1.28e-08. PySCF uses the true Fock and carries that SCF-convergence residue; ccgen
generates with `canonical_fock=True`, which drops `f_ov` as Brillouin-zero. Both are correct; they
differ by exactly the term ccgen legitimately omits.

Two checks run before believing it: the residual conversion `r = (t_new − t_old)·D` round-trips to
1e-18, so it is not the source; and an elementwise dump shows the alarming-looking "3.554e-04 max
relative deviation" is entirely element `[1,0]`, whose magnitude is 3.6e-05 — a small denominator,
not a large error.

**Method note:** this evaluates the **equations**, not the C++ emitter. Rank 3's defect was shown to
be upstream of codegen, so equations are the right layer — but a clean result here does **not**
exonerate the rank-2 emit path, which still has no consumer and has never executed.

**This kills the leading P4 candidate.** Rank 2 and rank 3 both have exactly one Sz sector, yet rank
2 is correct. So "single-sector handling is broken" cannot explain the rank-3 defect.

Reproduce: `/tmp/claude-501/p1.py` (PySCF reference), `p1_ccgen.py` (ccgen evaluation),
`p1_cmp.py` (comparison; converts `update_amps` output back to a residual).

### P1 (original scope)

The cheapest new data point, and the one that most changes the picture. `ccsd_planck_generated.cpp`
is emitted but has no consumer; a generated RCCSD energy can be compared against the hand-written
RCCSD solver on the same input.

*Verify:* generated rank-2 `E_corr` vs hand-written, same molecule/basis, agreeing to solver
tolerance — or a named discrepancy.

**This is the hypothesis's pivot.** If rank 2 is *wrong*, parity is dead immediately and the bug is
"generated kernels are broken except where a gate happened to catch it", which is a different and
larger problem. If rank 2 is *right*, parity has two supporting points and is worth pursuing.

### P2 — the system selection criterion, and why BeH2 is not sufficient

**Selection must be on distinct excitations, not electron count.** Choosing BeH2 because it has 6
electrons (so CCSDTQ is non-exact) was wrong: with `no=3` there are **zero** permutationally-distinct
quadruple excitations, because `C(3,4) = 0` — you cannot pick 4 distinct occupied spatial orbitals
from 3. The T4 tensor still allocates 20,736 dense elements, which is why it is slow, but its
independent content is trivial.

So BeH2 confirms "CCSDTQ ≠ FCI" (measured: −4.4e-09) without exercising T4's freedom — the exact
blind spot P2 exists to close.

The criterion is **`C(no,4) × C(nv,4) > 0`**, i.e. `no ≥ 4` AND `nv ≥ 4`, plus `no ≠ nv` to avoid the
square-shape blind spot. Surveyed:

| system | ne | no | nv | T4 dense | **distinct Q** | usable? |
|---|---|---|---|---|---|---|
| Be/sto-3g | 4 | 2 | 3 | 1,296 | **0** | no — the existing gate's confound |
| BeH2/sto-3g | 6 | 3 | 4 | 20,736 | **0** | no — non-exact but no distinct Q |
| H2O/sto-3g | 10 | 5 | 2 | 10,000 | **0** | no — `nv < 4` |
| BH/6-31g | 6 | 3 | 8 | 331,776 | **0** | no — `no < 4` |
| **CH4/sto-3g** | 10 | 5 | 4 | 160,000 | **5** | **yes — smallest viable** |
| **H2O/6-31g** | 10 | 5 | 8 | 2,560,000 | **350** | **yes — strongest test** |

Note H2O/**sto-3g** does not qualify (`nv=2`); H2O/**6-31g** does.

### P2 — runs to perform (on a larger machine)

Two runs, both comparing Planck's generated rank-4 kernel against PySCF `rccsdtq` on the same
geometry and basis. **Run CH4 first** — it is 16× smaller in T4 and answers the same question; H2O
only adds confidence.

**Inputs** are the existing rank-4 regression input with the geometry swapped:

```bash
# CH4/sto-3g  (no=5 nv=4, 5 distinct quadruples)   -- SMALLEST VIABLE, run first
# H2O/6-31g   (no=5 nv=8, 350 distinct quadruples) -- strongest, ~16x the T4 elements
```

Take `tests/inputs/regression/post_hf/be_rccsdtq_sto3g.hfinp`, replace the `%begin_coords` block
(atom count on the first line), and for H2O set `basis 6-31g` in `%begin_control`.

```
CH4 (angstrom):            H2O (angstrom):
5                          3
0   1                      0   1
C   0.00  0.00  0.00       O   0.000  0.000  0.000
H   0.00  0.00  1.09       H   0.000  0.000  0.960
H   1.03  0.00 -0.36       H   0.930  0.000 -0.240
H  -0.51  0.89 -0.36
H  -0.51 -0.89 -0.36
```

**Binary:** stock `build/` is correct — verified `PLANCK_CC_DRESS_OPERATORS` unset (default OFF),
zero dressed builders in the emitted rank-4 TU, and the registry routing rank 4 to
`make_generated_ccsdtq_kernels()`. **Confirm the run's own marker says generated before believing any
number** — a build can silently select a hand-written backend.

```bash
export BASIS_PATH=$PWD/basis-sets
build/hartree-fock <input> 2>&1 | tee run.log | grep -E "Total RCCSDTQ Energy|RCCSDTQ\["
```

**Reference** — generate with the same geometry/basis, `mol.cart = True`, `mf.conv_tol = 1e-12`:

```python
from pyscf import gto, scf, fci
from pyscf.cc import rccsdtq
mol = gto.M(atom=..., basis=..., verbose=0); mol.cart = True
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
c = rccsdtq.RCCSDTQ(mf); c.verbose = 0; c.kernel()
print(mf.e_tot + c.e_corr)                       # <- the comparand
print(fci.FCI(mf).kernel()[0])                   # <- confirms CCSDTQ != FCI
```

*Verdict:* Planck matches PySCF `rccsdtq` (**not** FCI) to ~1e-7 → rank 4 is correct with T4 genuinely
exercised, and parity has two solid points. A material deviation → the defect is
manifold-freedom-dependent, which would reframe rank 3 (see below).

**Expected to pass.** Be already matches PySCF `rccsdtq` to 1e-10, and a structurally wrong kernel
does not land there by luck. This closes a specific blind spot rather than testing a live suspicion —
worth doing because it is cheap relative to what it rules out, not because rank 4 is in doubt.

**If it does deviate**, the implicated mechanism is amplitude permutational symmetry — something that
only manifests when the tensor has independent components to get wrong. That would move rank 3's
prime suspect from the spatial spin-adaptation lowering (a property of the *equations*, which do not
know the electron count) to the `restore_restricted_t3_structure` convention layer, and would offer
an explanation for both the ~45 % subset support and the unexplained 31 % sign flips.

### P2 — the BeH2 run (superseded as the confound test, still useful)

**System: BeH2/STO-3G cartesian** — 6 electrons (so CCSDTQ is genuinely approximate), `no=3, nv=4`
(so the square-shape blind spot is avoided), and small enough to run. Chosen over BH/STO-3G, which
has 6 electrons but `no = nv = 3`.

PySCF reference, SCF converged to 1e-12:

```
RHF       = -15.5612780323
RCCSD     = -15.5946728448   e_corr = -0.0333948125
RCCSDT    = -15.5950245980   e_corr = -0.0337465657
RCCSDTQ   = -15.5950470854   e_corr = -0.0337690531
FCI       = -15.5950470809

CCSDTQ - FCI = -4.442e-09    <- nonzero: the exactness confound is REMOVED
CCSDT  - FCI = +2.248e-05
```

The `CCSDTQ − FCI` gap being non-zero is the point: on Be it is zero by construction, so the
existing gate cannot distinguish "the kernel is right" from "the method is exact here". Here it can.

Planck's generated rank-4 kernel is being run on the same input
(`/tmp/claude-501/beh2_cc4.hfinp`, stock `build/` with `MAXORDER=4 SPIN_ADAPT=ON`). It is slow —
minutes, per the known ~180× generated-kernel slowdown.

*Verdict when it lands:* Planck ≈ −15.5950470854 (matching PySCF RCCSDTQ, **not** FCI) → rank 4 is
correct off the exactness limit, and parity has two solid supporting points. A material deviation →
rank 4's Be pass was the confound, and parity is dead.

### P2 (original scope)

Run the generated rank-4 kernel on a system with **more than 4 electrons**, where CCSDTQ is
approximate, and compare against the hand-written RCCSDTQ backend at identical amplitudes.

*Verify:* raw residual comparison (not converged energy — `restore_restricted_t3_structure` masks
residual error ~11–29×, per the rank-3 investigation). Use `nv != no`.

*Note:* cost grows steeply with rank; pick the smallest system that is not an exactness case.

### P3 — the parity verdict (~S)

With P1 and P2 in hand:

| P1 (rank 2) | P2 (rank 4, non-exact) | reading |
|---|---|---|
| correct | correct | **parity holds** on 4 points — proceed to P4 |
| correct | wrong | not parity; rank 4's pass was the exactness confound |
| wrong | correct | not parity; rank 3 and 2 wrong, rank 4 right — look for what rank 4 does differently |
| wrong | wrong | not parity; the generated path is broadly wrong and rank 4's gate is the outlier to explain |

*Verify:* a written verdict naming which row, with both numbers.

### P4 — only if parity holds: find what distinguishes odd from even (~M)

Candidate mechanisms, to be tested rather than assumed:

- **Sz sector count.** `independent_spin_blocks(rank)` gives ⌊n/2⌋ sectors. Rank 4 has two
  (`aabbaabb`, `aaabaaab`) and rank 3 has one. A defect in single-sector handling would look like
  odd-rank failure — but note rank 2 also has one sector, so P1 discriminates this directly.
- **The restricted-T3 convention.** `restore_restricted_t3_structure` and its repeated-index
  pre-scaling apply to odd-rank amplitude symmetry. Rank-4 has an analogous but distinct
  treatment.
- **Spatial spin-adaptation of an odd-rank manifold.** The `2·direct − exchange` structure at
  odd rank is exercised nowhere else.

*Verify:* a named mechanism plus a numeric before/after at fixed amplitudes.

---

## What NOT to do

- **Do not treat the rank-4 pass as proof that even ranks are correct.** One data point, measured
  at the exactness limit. P2 exists to control that.
- **Do not treat rank 2 as validated.** Its TU has no consumer; it has never run.
- **Do not gate on converged energy for a residual defect.** `restore` masks raw error ~11–29×.
  Compare raw residuals at fixed amplitudes.
- **Do not use a square test system.** `nv == no` (Be/STO-3G is 4 and 4) lets a wrongly-ordered
  read stay in bounds and fail silently. Every numeric gate here uses `nv != no`.
- **Do not conclude from a structural gate.** Five fixes in the rank-3 investigation passed one
  and degraded the energy. Numeric gate attached from the start, not added afterwards.
- **Do not assume rank 5/6 are correct because they compile.** They have no gate at all.

## Relationship to the rank-3 defect

If parity holds, `CCGEN_RANK3_TRIPLES_DEFECT.md`'s remaining suspects narrow to the two that are
odd-rank-specific: the spatial spin-adaptation lowering of an odd manifold, and the
`restore_restricted_t3_structure` interaction. Its unexplained **31 % sign flips** would then be
the concrete symptom to explain with whatever P4 names.

If parity does not hold, the rank-3 defect stands on its own and this document's value is having
bounded the surface — including, importantly, whatever P1/P2 reveal about ranks that currently
have no coverage at all.

## Key code locations

| what | where |
|---|---|
| rank-2 generated TU (no consumer) | `build/generated/cc/ccsd_planck_generated.cpp` |
| rank-3 wiring + probe | `src/post_hf/cc/tensor_backend.cpp:2321`, `PLANCK_CC_T3_DIFF` |
| rank ≥ 4 registry | `src/post_hf/cc/generated_kernel_registry.cpp` |
| rank-4 gate | `be_rccsdtq_sto3g` in `tests/regression_cases.json` |
| Sz sector count | `independent_spin_blocks`, `ccgen/spin.py` |
| restricted-T3 convention | `restore_restricted_t3_structure`, `tensor_backend.cpp:1976` |
