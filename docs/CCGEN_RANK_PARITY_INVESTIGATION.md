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

## The investigation

### P1 — validate rank 2 numerically (~S, do first)

The cheapest new data point, and the one that most changes the picture. `ccsd_planck_generated.cpp`
is emitted but has no consumer; a generated RCCSD energy can be compared against the hand-written
RCCSD solver on the same input.

*Verify:* generated rank-2 `E_corr` vs hand-written, same molecule/basis, agreeing to solver
tolerance — or a named discrepancy.

**This is the hypothesis's pivot.** If rank 2 is *wrong*, parity is dead immediately and the bug is
"generated kernels are broken except where a gate happened to catch it", which is a different and
larger problem. If rank 2 is *right*, parity has two supporting points and is worth pursuing.

### P2 — remove the exactness confound at rank 4 (~M)

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
