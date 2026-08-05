# Remediation scope: generated-CC spin-adaptation defect

Companion to `GENERATED_CC_SPIN_ADAPTATION_DEFECT.md` (the diagnosis). This scopes
how to fix it. The defect: the generated arbitrary-order CC kernels emit
**spin-orbital** algebra (`0.25 t2 v`, coefficients `±1,±½,±¼`, spin-orbital term
counts) but run against **spatial** amplitudes + chemists' integrals, so the energy
is ~5× too large and dives below FCI.

## The two facts that frame every option

1. **The spin-summation machinery already exists and is tested — but is wired to
   nothing.** `ccgen/spin.py` (`ucc_integrate_term`, `ucc_integrate_term_antisym`
   with the S2 exchange −K fix, `ucc_manifold`, `canonicalize_spin_blocks`) is
   imported **only** by `ccgen/tests/test_spin.py`. No emitter calls it.
2. **The planck emit path uses a stopgap that is NOT a spin adapter.**
   `ccgen/lowering/restricted_closed_shell.py` says so in its own docstring: *"This
   module does not attempt a full symbolic spin summation yet."* It records
   block-signature + slot-permutation + ERI phase, and the emitter renders the raw
   spin-orbital coefficient unchanged. That relabel-only IR is what
   `_emit_kernel(..., lowered_terms=...)` emits.

So the fix is not "invent spin adaptation" — it is "connect the existing
`spin.py` adaptation to the planck emitter, replacing the relabel-only lowering."

## The reference for "correct": the running warm-start `.inc`

`ccgen/emit/planck_rccsd_warm_start.py` emits `compute_generated_spin_orbital_rccsd_*`
which also uses `0.25 * oovv * t2` — and is correct — because it runs against a
`ProductionSpinOrbitalReference` where `n_occ = 2·spatial` and `oovv` is the
antisymmetrized `⟨ij||ab⟩`. Same formula, genuinely spin-orbital storage. This tells
us the algebra is right; only the storage/summation the generated path targets is
wrong.

---

## Route 1 (recommended) — spin-adapt in the lowering, keep spatial storage

Make the emitted terms genuinely spatial: sum over spin cases so coefficients become
the `2·(direct) − (exchange)` structure and the term count drops to the spatial count
(doubles ~30–40, not 64). The runtime stays spatial (`n_occ`, chemists' `oovv`) — no
runtime change, no `2×` memory.

- **R1.0 — spin-adapt entry over the lowered manifold (~M).** Route each target's
  terms through `ccgen/spin.py` (`ucc_integrate_term_antisym` + `ucc_manifold` +
  `canonicalize_spin_blocks`) to produce spatial terms carrying the correct spatial
  coefficients, then feed *those* to `_emit_kernel` instead of the
  `restricted_closed_shell` output. The machinery exists; this is wiring +
  adapting its output shape to what `emit_planck_term` consumes.
- **R1.1 — emit the spatial contraction (~S given R1.0).** `emit_planck_term` renders
  spatial factors with chemists' `oovv` and the summed spatial coefficient. Likely a
  small change since the factor/block plumbing already exists; the coefficients just
  arrive already-summed.
- **R1.2 — the missing numeric gate (~S, load-bearing).** A real energy test:
  run the actual `compute_*_energy` + residuals on a tiny reference (Be or a 2e
  toy) and assert the CC energy vs the hand-written solver / PySCF. This is the
  guard whose absence let the defect ship — it must land WITH the fix, not after.
- **R1.3 — symbolic term-count assertion (~S).** In the lowering/spin tests, assert
  spatial doubles ≈ 30–40 (not the spin-orbital 64), so a future relabel-only
  regression is caught without a compile+solve.

**Effort:** ~M+ overall, concentrated in R1.0 (adapting `spin.py`'s manifold output
to the emitter's term IR). Everything downstream — runtime, restart, warm-start,
Route A/B — is already correct and needs no change once the kernels are right.

**Risk:** medium. `spin.py`'s output IR and the emitter's expected term IR differ;
R1.0 is the impedance match. The S2/S4 notes claim numeric agreement to ~1e-8 for
CCD/CCSD singles+doubles+energy, so the adaptation itself is validated — the risk is
in the wiring, gated by R1.2.

## Route 2 (not recommended) — feed spin-orbital storage to the generated kernels

Keep the spin-orbital algebra; give it spin-orbital storage. Build a
`2·n_occ` reference + antisymmetrized `⟨ij||ab⟩` blocks in the arbitrary runtime and
bind those. The emitted algebra is then correct as-is (it's the warm-start `.inc`
pattern generalized to higher rank).

**Why not:** it defeats the purpose of the *restricted* spatial path — `2×` orbital
range means `2^k×` cost/memory per rank (16× the ERI block at rank 2, far worse at
rank 4), exactly what spin adaptation exists to avoid. It also duplicates the
warm-start `.inc` storage model into the arbitrary runtime. Only sane as a **stopgap
correctness check**: bind SO blocks to confirm the emitted algebra is otherwise
correct, proving the defect is purely storage/summation before investing in R1.0.
Worth doing once as a ~S validation spike, not as the fix.

## R2 spike — DONE, hypothesis confirmed

The de-risking spike is already satisfied by existing (passing) tests in
`ccgen/tests/test_reference_vs_pyscf.py`, which evaluate the **generated** equations
under **spin-orbital** storage against PySCF/FCI:

- `test_ccgen_ccsd_energy_matches_pyscf` / `_solver_matches_pyscf`: generated CCSD
  energy at PySCF's converged GCC amplitudes = PySCF `e_corr` to **12 / 9 places**.
- `test_ccgen_ccsdt_reaches_fci_limit` (+ diagram variants): generated CCSDT energy
  reaches the **FCI limit to 8 places**.

Run: `tests/pyscf/.venv/bin/python -m pytest
ccgen/tests/test_reference_vs_pyscf.py -k "ccsd_energy or ccsd_solver or ccsdt_reaches_fci"`
→ **5 passed**.

**Conclusion:** the generated equations are numerically correct in spin-orbital
storage. The C++ runtime binds that same algebra to **spatial** storage
(`n_occ`, chemists' `oovv`), which is the entire defect — confirming R1 (spin-adapt
to spatial) is the right fix and there is no deeper equation bug to hunt. A quick
NumPy contrast on the CCSDTQ energy target (spin-orbital vs a spatial re-binding of
the same coefficients) gives materially different numbers, illustrating the same
mechanism at rank 4. No CCSDTQ-specific PySCF oracle exists (PySCF has no quad
amplitudes); the ccsd/ccsdt evidence plus the rank-agnostic emit path cover it.

## Recommended sequence

1. **R2 (validation spike) — DONE (above).** Algebra confirmed SO-correct; defect is
   storage/summation, as hypothesized.
2. **R1.2 (the numeric gate) — DONE.** `GeneratedSpatialEnergyGate.test_ccsd_spatial_energy_matches_rccsd`
   in `ccgen/tests/test_reference_vs_pyscf.py`: evaluates the generated CCSD energy
   the way the C++ runtime does — bound to SPATIAL storage (n_occ spatial, chemists'
   (pq|rs), spatial t1/t2 from restricted RCCSD) — and asserts it equals PySCF
   `RCCSD.e_corr`. Currently `@expectedFailure` (xfail): the generated spatial-bound
   energy comes out at **exactly 0.2500 × rccsd.e_corr** (H2/STO-3G: -0.00513 vs
   -0.02052) — the spin-orbital ¼ coefficient on spatial storage with no spin
   summation. R1 (spatial algebra) makes the ratio 1.0; **remove the
   `@expectedFailure` decorator when R1 lands** so it becomes a hard gate. This is
   the numeric guard whose absence let the defect ship.
3. **R1.0 → R1.1** (spin-adapt in the lowering, emit spatial), turning the gate green.
4. **R1.3** (symbolic term-count assertion) so it can't silently regress.
5. Re-enable `be_rccsdtq_sto3g` as an **asserting** case (not `skip_if_contains`),
   pinning -14.4036550465, once warm-start makes it tractable.

## R1 status (landed)

- **ccsd** — DONE, validated. `spin_adapt_equations` energy = PySCF RCCSD to 1e-9;
  singles+doubles residual vanishes at the converged RCCSD amps (fa0f088).
- **ccsdt** — DONE, validated. Adapted spatial CCSDT solves CCSD < CCSDT < FCI on
  Be, recovering ~28% of the CCSD→FCI gap (the T3 part) (ce33301).
- **ccsdtq / cc4** — **NOT fixed.** This is the actual broken binary path. See R3.

---

## R3 — finish the higher-rank (≥4) closed-shell reduction (the cc4 blocker)

**Symptom.** Adapted CCSDTQ on Be gives e_corr **identical to adapted CCSDT** (T4
contributes ~0); CCSDTQ − FCI = 3.14e-6, but Be (4e) requires CCSDTQ ≡ FCI to ~1e-8.

**Not structural.** The adapted quadruples manifold has 4033 T4-source terms + 524
t4-bearing terms — R4 is non-empty. The terms are **numerically wrong**, so the
solve converges to a T4 that nets ~0.

**Root cause (refined after experiment) — `spin_adapt_equations` integrates each
target on a SINGLE representative external block, but the higher-rank amplitude
residual needs the full set of independent spatial blocks summed.**

What the experiments ruled IN and OUT:

- **RULED OUT — the block convention.** Swapping `_closed_shell_representative_block`
  from alternating (`abababab`) to the splitter's single-beta block (`aaabaaab`)
  gives the *identical* wrong CCSDTQ (−0.0517714927, T4 ≈ 0, gap 3.14e-6). So it is
  not a one-line block swap.
- **RULED OUT — the amplitude splitter.** `_split_same_spin_amplitude` and the whole
  collapse+merge pipeline are numerically validated **per block** through rank-6
  (t3): `S4bSplitterTests` (rank-4 t2, rank-6 t3 same-spin relations, 13/13 pass)
  and `test_rcc_pipeline_generalizes_rank6` — the latter runs the FULL
  canonicalize→collapse→merge on the triples manifold and matches the GCC slice to
  1e-10, but **on one specific external block** `{a:a,b:a,c:b,i:a,j:a,k:b}` (=aabaab).
- **THE ACTUAL GAP.** `test_rcc_pipeline_generalizes_rank6` proves one block's
  residual is reproduced. `spin_adapt_equations` uses exactly one block per target —
  which is correct for the **energy** (a single scalar) and for ranks where one
  spatial block spans the residual, but the **rank-4 amplitude residual has multiple
  independent spatial components** (quadruples has 8 valid external blocks:
  aaaaaaaa, aaabaaab, aabaaaba, aabbaabb, abaaabaa, abababab, abbaabba, abbbabbb).
  Integrating on one representative and deriving the rest via the same-spin collapse
  is complete for doubles (2 blocks, 1 relation) but **not** for quadruples. T4 ends
  up under-driven → contributes ~0. Note: adapted CCSDT *passed* only a LOOSE gate
  (CCSD < CCSDT < FCI ordering), which one-block integration happens to satisfy; the
  exact Be CCSDTQ≡FCI oracle is what exposed the incompleteness.

**Fix, in order:**

- **R3.0 — a rank-4 numeric gate. DONE (xfail).** `GeneratedCcsdtqFciGate`
  (`test_reference_vs_pyscf.py`, `CCGEN_SLOW_TESTS`-gated): adapted spatial CCSDTQ on
  Be must reach FCI to 1e-8. Red now (T4≈0), green when R3.1 lands. Uses the shared
  `solve_spin_adapted_spatial` helper.
- **R3.1 — sum the full independent-block set per target (~L, the real fix).**
  `spin_adapt_equations` must, for each target, integrate the residual over ALL its
  independent closed-shell external blocks (not one representative) and assemble the
  spatial residual from them — the many-block generalization of the doubles
  "abab + derive aaaa" reduction. This is genuine S4 algebra: enumerate the
  independent blocks (`external_blocks` gives the 8), integrate each
  (`ucc_integrate_term_antisym`), collapse+merge, and combine into the spatial
  amplitude components. The per-block machinery already works (the rank-6 test proves
  it); what's missing is the multi-block assembly.
- **R3.2 — extend `test_rcc_pipeline_generalizes_rank6` to rank-8 (~M).** A rank-8
  analog of the per-block pipeline gate on the S4d t4 fixture (already exists,
  `_uccsdtq_so_tensors`), then the multi-block assembly gate.
- **R3.3 — re-run the ladder (~S).** ccsd/ccsdt stay green; R3.0 flips green; only
  then wire `spin_adapt` into the CMake default and confirm binary Be cc4 →
  -14.4036550465.

**Effort:** ~L, in the S4 multi-block assembly. This is real symbolic work — the
per-block reduction exists and is validated to t3; the missing piece is assembling
the full spatial residual from all independent blocks at rank 4. Until it lands, the
`spin_adapt` path is correct for the energy and ranks ≤3, and cc4+ stays wrong.

## What NOT to do

- Do not patch the energy kernel's `0.25` in isolation — the residuals carry the same
  spin-orbital algebra; the amplitudes would still converge to spin-orbital-scaled
  values. The fix is one change in the lowering that propagates to energy + all
  residual targets together.
- Do not build a second spin adapter — `ccgen/spin.py` is the one, already tested.
  R1.0 connects it; it does not reinvent it.
- Do not ship any generated-kernel numeric result until R1.2 exists — the toy energy
  kernel in `tests/cc_arbitrary_solver.cpp` is why this shipped wrong.
- Do not make `spin_adapt` the CMake codegen default until R3 lands — it would fix
  ccsd/ccsdt but silently keep cc4+ wrong, trading a loud defect for a quiet one.
- Do not assume the rank-generic collapse code is validated because it runs — only
  the doubles abab block has an end-to-end numeric gate; ranks ≥4 need R3.0/R3.2.
