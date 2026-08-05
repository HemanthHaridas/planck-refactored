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

**Root cause — the higher-rank closed-shell reduction was never end-to-end
validated, and my `spin_adapt_equations` extended it past where it holds:**

1. **The whole S2 collapse pipeline is doubles-only-validated.** Every S2 test
   (`test_spin.py`) drives `ucc_manifold(...)["abab"]` on the **doubles** template
   `R2` alone. `collapse_amplitudes`/`collapse_integrals`/`merge_terms` are written
   rank-generically, but the only end-to-end numeric gate is the doubles abab block.
   `spin_adapt_equations` runs them on singles/doubles/triples/quadruples; ccsd and
   ccsdt happened to validate, quadruples does not.
2. **Representative-block convention mismatch (the concrete bug).**
   `_closed_shell_representative_block` integrates the manifold on the *alternating*
   external block (rank-4 → `abababab`), but the amplitude splitter
   `_split_same_spin_amplitude` reconstructs same-spin blocks against the
   *single-beta* mixed block ((a…a,b)×2, rank-4 → `aaabaaab`). They **coincide only
   at n≤2** (`abab`), so doubles/triples pass but rank-4 T4 coupling is fed the
   wrong block.
3. **The amplitude splitter itself is only numerically pinned at n≤3.**
   `_split_same_spin_amplitude`'s docstring: *"Numerically pinned at n=2,3"*. n=4
   (the rank-4 amplitude, 8 slots) is structural-only. The S4 workstream
   (memory: `ccgen_spin_adaptation_s4a_t3`) deliberately validated t3/t4 via a
   **different** FCI-limit iterate route, not via making this full-manifold
   closed-shell adaptation correct at rank 4.

**Fix, in order:**

- **R3.0 — a rank-4 numeric gate first (~S).** The Be CCSDTQ≡FCI solve is the
  oracle (adapted spatial CCSDTQ must reach FCI to ~1e-8). It is slow (~10 min:
  the 4557-term quadruples adaptation + t4 Jacobi), so gate it behind
  `CCGEN_SLOW_TESTS=1` like the AR1 ccsdtq gate. Red now, green when R3 lands.
- **R3.1 — reconcile the representative block with the splitter convention (~M).**
  Either (a) integrate the manifold on the single-beta block the splitter expects,
  or (b) drive the collapse off whatever block the integration used. Whichever, the
  external-block choice in `_closed_shell_representative_block` and the amplitude
  block in `_split_same_spin_amplitude` must be one convention. This is the operative
  bug and likely the bulk of the fix.
- **R3.2 — numerically validate `_split_same_spin_amplitude` at n=4 (~M).** Promote
  its n≤3 numeric pin to n=4 (rank-8 amplitude): the same-spin reconstruction must
  reproduce the sliced GCC block on real tensors. If it fails, the splitter relation
  itself (not just the block choice) needs the rank-4 generalization the S4 notes
  deferred.
- **R3.3 — re-run the ladder (~S).** ccsd/ccsdt gates stay green; the R3.0 Be
  CCSDTQ gate flips green. Only then wire `spin_adapt` into the CMake codegen
  default and confirm the real binary Be cc4 → -14.4036550465.

**Effort:** ~M–L, in the S4 spin-collapse machinery. This is real symbolic work, not
wiring — the opposite of R1.0 (which was mostly pre-built). Until it lands, the
`spin_adapt` path is correct for ranks ≤3 only, and cc4+ stays wrong.

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
