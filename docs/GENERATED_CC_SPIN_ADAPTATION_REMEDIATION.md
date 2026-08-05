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
### R3.1 in small verifiable steps

The narrowed root cause (after all three block choices gave the identical wrong
answer): **the full collapse+merge pipeline is validated only through rank-6 (t3).
`test_rcc_pipeline_generalizes_rank6` runs canonicalize→collapse_amplitudes→
collapse_integrals→merge on the triples manifold and matches the GCC slice to
1e-10 — but there is NO rank-8 (quadruples) analog.** The rank-8 tests
(`test_rank8_aabb_identity`) run *integration only*, never the collapse. So the
untested/broken boundary is the **rank-8 collapse**, not the block choice or the
integration.

Iterate against a FAST per-block rank-8 gate (seconds, one block, no solve), NOT
the ~15-20 min Be CCSDTQ solve. The Be solve is the final confirmation only.

- **R3.1.0 — the fast rank-8 collapse gate. DONE, and it PASSES (overturned the
  hypothesis).** `test_rank8_full_collapse_pipeline` (`S4dRank8IdentityTests`) runs
  the full collapse+merge on the quadruples manifold, aabb block, vs the S4d t4
  fixture GCC slice — in ~27s, and it MATCHES to 1e-10. So the per-block collapse is
  NOT the bug. This gate is the R3.1 inner-loop harness (seconds, no Jacobi).
- **R3.1.1 — localize the bug. DONE — it's the SpinTerm→AlgebraTerm BRIDGE, not the
  collapse.** `spinterm_to_algebraterm` builds `Tensor(f.name, tuple(si.base for si
  in f.indices))` — it **drops the spin block label**, keeping only spatial indices.
  For doubles/triples this is lossless because every surviving amplitude factor is in
  a block where spatial-tensor == block (`t1[aa]`, `t2[abab]`). **At rank 4 the
  higher amplitudes survive in MIXED same-spin blocks the collapse does NOT reduce:**
  census of merged quadruples factors —
  `t3[aabaab]`, `t3[abbabb]`, `t4[aaabaaab]`, `t4[aabbaabb]`, `t4[abbbabbb]`.
  For these the spatial tensor is NOT the block, so dropping spin makes the solver
  contract the full spatial t3/t4 instead of the intended spin slice → T4
  mis-evaluated → ~0. The R3.1.0 test passes only because `_eval_spinterm` SLICES
  each factor by its spin block (`_slice_spinterm_factor`); the solve path (bridge +
  `residual_of`) discards it. Root cause: **`collapse_amplitudes` only splits the
  ALL-same-spin block** (`_is_same_spin_amplitude` requires `set(block)=={'a'}`), so
  mixed blocks like `aabaab`/`aabbaabb` pass through un-reduced and the bridge then
  loses them.
  **R3.1.1 refined — the bug is CROSS-TARGET spatial-block inconsistency.** A key
  experiment: even within the TRIPLES target, the merged t3 factors are in blocks
  `aabaab`/`abbabb` (not one canonical block), yet `_eval_spinterm` reproduces the
  GCC slice to 2e-17 and the *triples* residual output is on the `aab` block. So
  WITHIN one target, input-factor blocks and the output block are self-consistent,
  and the solve works (this is why ccsdt validated). The break is ACROSS targets:
  the quadruples residual output is on the `aabb` block but READS t3 in `aabaab` and
  t2 in `abab` — while the spatial t3 tensor the solver holds was DEFINED by the
  triples residual on ITS block. Same spatial `t3` name, different spin-block
  layout at definition vs use ⇒ the dropped-spin bridge reads the wrong slice ⇒ T4
  wrong. Doubles never hits this because t2 is `abab` at both definition and every
  use.
- **R3.1.2 — make the spatial amplitude layout consistent across targets (~L, the
  fix — genuine spin-algebra).** Every amplitude `t_n` must have ONE fixed
  spin-block → spatial-index layout, used identically whether it is a residual
  OUTPUT (its own target) or an input FACTOR (in higher targets). Two shapes:
  (a) pick one canonical block per rank and, in `spinterm_to_algebraterm` (and the
  residual-output layout), apply the block→canonical spatial permutation/sign so a
  factor in `aabaab` is reordered/signed into the canonical layout before the spin
  label is dropped — the `spatial_permutation` idea, but keyed on the SPIN block,
  which `restricted_closed_shell` does NOT compute (it canonicalizes by occ/vir
  space only); or (b) emit each amplitude in its own per-block spatial tensor and
  key the solver on (name, block). (a) is closer to standard RCC (one spatial t_n);
  (b) is mechanical but multiplies tensors. Iterate with a FAST gate:
  `spinterm_to_algebraterm`+`residual_of` on the quadruples aabb block must match
  the GCC slice (the solve path, not `_eval_spinterm`) — seconds, red now.
- **R3.1.3 — confirm end-to-end (~S, slow).** The R3.0 Be CCSDTQ≡FCI gate flips
  green. ccsd/ccsdt stay green.

**Status:** root cause FULLY diagnosed (cross-target spatial-block inconsistency in
the spin→algebra bridge); R3.1.0 fast gate landed and passing (per-block collapse is
correct); R3.1.2 (the fix) is genuine ~L spin-algebra, not yet started.

**The inconsistency, made symbolic (fast, no solve).** Per target, the residual
OUTPUT block vs the block each amplitude is USED in as a factor:

| target | output block | amplitude usage |
|---|---|---|
| ccsd/doubles | `abab` | t2 used in `abab` — CONSISTENT |
| ccsdt/triples | `abaaba` | t3 used in `aabaab`, `abbabb` — INCONSISTENT |

Doubles: output block == usage block (`abab`), so the dropped-spin bridge is exact —
ccsd is correct. Triples: the t3 residual is emitted on `abaaba` but t3 appears as a
factor on `aabaab`/`abbabb` — a DIFFERENT spatial-index layout for the same `t3`
tensor. **This means ccsdt is ALSO affected**; it passed only the LOOSE
CCSD<CCSDT<FCI ordering gate, which tolerates the resulting small error. At
quadruples the same class of inconsistency leaves T4≈0 and the exact Be CCSDTQ≡FCI
oracle exposes it. So the ccsd/ccsdt "validated" status is: ccsd exact, ccsdt
ordering-correct but NOT exact — re-verify ccsdt to tight tolerance after R3.1.2.

**The fix (R3.1.2) — two pieces; piece 1 landed, piece 2 is the remaining work.**

*Piece 1 (LANDED): permutation canonicalization within a β-count sector.*
`_canonicalize_amplitude_factor` sorts each amplitude half (bra, ket) to α-before-β,
folding the antisymmetry sign into the coefficient; `_closed_shell_representative_block`
now emits the SAME α-first-per-half reference (n=2 `abab`, n=3 `aabaab`, n=4
`aabbaabb`) so output and factors share a layout. Verified exact: `t3[aabaab]` =
signed-perm(`t3[abaaba]`) to 0.0 on the UCCSDT fixture. This is necessary and
correct but NOT sufficient.

*Piece 2 (REMAINING): same-spin-sector reduction (extend the collapse).* Sorting
only canonicalizes WITHIN a fixed β-count. At rank 3 the surviving t3 factors span
TWO β-count sectors — `aabaab` (1 β per half) AND `abbabb` (2 β per half) — which
are NOT permutations of each other (different β-counts) and NOT global-flip images.
A single spatial t3 cannot represent both by permutation alone; the higher-β-count
sector must be REDUCED to the reference sector via the closed-shell same-spin
relation. `collapse_amplitudes`/`_split_same_spin_amplitude` currently fire only on
the ALL-α block (`set(block)=={'a'}`); they must be extended to reduce any half
containing a same-spin PAIR (e.g. the `bb` in `abb`, the `aa` in `aab`) toward the
reference. For odd-per-half ranks (t3) one same-spin pair is unavoidable, so the
reference itself carries one — the reduction must target that specific convention.
This is the genuine ~M–L S4 algebra still open.

Fast red gate (LANDED): `test_rcc_bridge_solve_path_rank6` — the solve path
(`spinterm_to_algebraterm`+`residual_einsum` on one spatial tensor per amplitude,
`aabaab` reference) vs the GCC slice. ~7s, still RED (~7e-3 after piece 1, was
4.8e-3) until piece 2 reduces the `abbabb` sector. ccsd/ccsdt gates stay green;
piece 1 caused no regressions.

- **R3.2 — wire `spin_adapt` into codegen, ALL ranks at once (~S, HELD).** Only after
  R3.1 lands (user decision: no partial binary state). A `--spin-adapt` flag on
  `generate_planck_cc_kernels.py` + a CMake option, plumbed like `--arbitrary-lower-ranks`.
  Confirmed trivial: `print_cpp_planck(spin_adapt=True)` already emits the spatial
  `1.5/-0.5 oovv` + `2/-1 t1t1` structure for ccsd. Then rebuild and confirm the
  binary Be cc4 → -14.4036550465, and remove the raw xfail.

**Coupling note (Gap A vs Gap B).** Gap A (codegen never applies `spin_adapt`) and
Gap B (rank-4 adaptation incomplete) meet at `spin_adapt_equations`: the emitted C++
is exactly its output. Flipping A propagates B's per-rank quality into the binary —
so a correct cc4 binary needs BOTH, while a correct ccsd/ccsdt binary needs only A.
Per user decision, A is HELD until B lands, then wired for all ranks together (no
partial state). The Be cc4 binary run was wrong at every rank because A was never
flipped (the cc3 warm-start seed used a raw, un-adapted cc3 kernel) — do not read
that binary number as evidence about the Python adaptation.

**Effort:** R3.1 is ~M–L concentrated in the rank-8 amplitude collapse (R3.1.2), but
the fast red-first gate (R3.1.0) makes it iterable in seconds instead of 20-min
solves. Until it lands, `spin_adapt` is correct for the energy and ranks ≤3; the
binary stays uniformly wrong (A held).

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
