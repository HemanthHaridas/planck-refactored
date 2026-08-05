# Scope of the generated-CC spin-adaptation defect

The generated arbitrary-order CC kernels compute a **wrong energy** — for Be/STO-3G
`cc4` the correlation energy comes out ~5× too large and the total dives ~0.20 Eh
**below** the FCI floor, oscillating at tight tolerance. This document scopes how
far the defect reaches and where the fix must land. It is a correctness bug in the
codegen, independent of the warm-start / restart / Route-A/B plumbing (which is
mechanically correct — it just seeds and iterates a wrong kernel).

Everything below is grounded in the current tree.

---

## The defect in one line

The "restricted closed-shell lowering" (`ccgen.lowering.restricted_closed_shell`,
called by the emitter) **relabels index blocks and permutations but does NOT perform
the spin summation.** The emitted C++ therefore contracts **spin-orbital algebra**
(coefficients `±1, ±½, ±¼`, spin-orbital term counts) against **spatial** amplitudes
and **spatial chemists' integrals** — a category error that is only correct if every
summed index secretly ran over 2× the range, which it does not (`no = n_occ`, not
`2·n_occ`).

## Evidence (exhaustive, from the emitter itself)

Comparing each target's lowered terms to the emitted contraction
(`_emit_kernel(..., lowered_terms=lowered_equations.get(target))`, ccsd, canonical Fock):

- **Coefficients are pure spin-orbital.** Distinct source coefficients across all
  targets: `{-1, -1/2, -1/4, 1/2, 1/4}`. A spatial closed-shell RCC contraction is
  built from `2·(direct) − (exchange)` structure; **no factor of 2 and no 2×−1
  combination appears anywhere.**
- **Term counts are spin-orbital.** ccsd doubles = **64** lowered terms; a genuine
  spatial RHF-CCSD doubles residual is ~30–40. The lowering did not collapse the
  spin cases.
- **No spin-weight carrier.** `RestrictedClosedShellTerm` has `block_signature` and
  `spatial_permutation` but **no** field encoding a spin-summation multiplicity
  (`spin_weight` / `loop_factor` / `multiplicity` — all absent). So even a
  spin-aware emitter has nothing to read.
- **The energy kernel is the visible symptom.**
  `compute_ccsdtq_energy` emits `0.25 * t2(i,j,a,b) * oovv(i,j,a,b)` with `oovv` =
  chemists' spatial `(ij|ab)` (`mo_blocks.h:20`). Correct spatial RHF-CCSD energy is
  `Σ (2(ia|jb) − (ib|ja))(t2_ijab + t1_ia t1_jb)`.

## Blast radius — what is and isn't affected

| Path | Amplitude type | Uses generated residual/energy? | Affected |
|---|---|---|---|
| Generated arbitrary `cc4/cc5/cc6` (`run_rccsdtq`) | `ArbitraryOrderRCCAmplitudes` (spatial) | yes | **YES — wrong energy** |
| Generated `ccsdt` TU (`ccsdt_planck_generated.cpp`, `tensor_backend.cpp`) | `RCCSDTAmplitudes` (spatial) | only via the experimental `TensorOptimized` RCCSDT backend | **YES when that backend is selected** |
| Rank-3 arbitrary companion (`--arbitrary-lower-ranks`, Route A/B) | `ArbitraryOrderRCCAmplitudes` (spatial) | yes | **YES — same emit path** |
| Hand-written determinant / tensor RCCSD, RCCSDT | own spin-orbital solver (`no=2·n_occ`) | no (does not use these generated expressions) | **NO — PySCF-gated, correct** |
| Hand-written RMP2/UMP2, CASSCF, FCI, DFT | — | no | **NO** |

So the defect is confined to **the generated kernels when actually solved**: the
`cc4+` arbitrary path (default) and the experimental optimized-RCCSDT backend
(opt-in via `PLANCK_RCCSDT_BACKEND=optimized`). Every PySCF-gated production method
is a different code path and is unaffected.

The `0.25 * t2 * oovv` / spin-orbital-coefficient pattern is present in the residual
kernels of **all** generated methods (ccsd, ccsdt, ccsdtq), not just the energy — so
this is not a single-line energy typo; the whole emitted contraction is spin-orbital.

## Why nothing caught it

- **W3 (`be_rccsdtq_sto3g`, expected -14.4036550465) never numerically validated the
  kernel.** The compile chain was validated; the energy was not. Be cold `cc4` is
  intractably slow, so the case relied on `skip_if_contains` and almost certainly
  skipped or timed out rather than asserting the number.
- **The `planck-cc-arbitrary-solver` unit test uses a TOY energy kernel**
  (`tests/cc_arbitrary_solver.cpp:205` — it sums residual values), so the real
  generated `compute_*_energy` was never checked against a reference.
- The Python `test_*_tu_compiles` gates are **compile-only** (`-fsyntax-only`); they
  never run the kernel.

## Where the fix must land

The lowering, not the emitter. `ccgen.lowering.restricted_closed_shell` must actually
**spin-adapt**: sum over spin cases so the emitted terms carry spatial `2×−1`
structure and the reduced (spatial) term count, OR annotate each term with a
spin-summation weight the emitter multiplies in. Either way the emitter change is
mechanical once the lowered terms carry spatial semantics.

This is the S2/S4 spin-adaptation workstream (GCC→RCC) — the open question these
notes flagged was whether the pipeline *applies* the adaptation end-to-end. **It does
not:** the lowering is currently an index/block relabeler, and the numeric consequence
(Be `cc4` below FCI) is the proof.

## Validation to add alongside the fix

1. **A real numeric energy gate.** Replace the toy energy kernel in the arbitrary-
   solver unit test (or add a new `tests/generated_kernel_energy.cpp`, the W4 harness)
   that runs the actual `compute_*_energy` + residuals on a tiny reference and checks
   the CC energy against the hand-written solver / PySCF. This is the gate whose
   absence let the defect ship.
2. **Make `be_rccsdtq_sto3g` actually assert**, not skip — once tractable (warm-start
   + a corrected kernel), pin `rccsdtq_total_energy` to -14.4036550465 with the run
   required to run, not `skip_if_contains`-guarded.
3. A spatial-vs-spin-orbital **term-count assertion** in the lowering tests
   (doubles ≈ 30–40, not 64) so a relabel-only regression is caught symbolically.

## What NOT to do

Do not "fix" the energy kernel's `0.25` in isolation — the residuals carry the same
spin-orbital algebra, so patching only the energy would leave the amplitudes
converging to spin-orbital-scaled values. The fix is one change in the lowering that
propagates to energy + all residual targets together.
