# Wiring the full generated kernel for CCSDTQ (cc4) and higher — scope

**The goal.** Make the `hartree-fock` binary compute a correct CCSDTQ (and
arbitrary-order cc≥4) energy from the **spin-adapted, multi-Sz-sector** generated
kernels — so `correlation rccsdtq` on Be/sto-3g reaches FCI (−14.4036550465), and
the raw-energy defect xfail can be deleted. This is the C++/runtime half of the
CCSDTQ=FCI verification; the Python half (`CCGEN_CCSDTQ_FCI_VERIFICATION_SCOPE.md`)
is its executable reference. **Land the Python solver first; port it here.**

**Status.** Two independent gaps, both open (see the two-gap note in
`generated_ccsdtq_energy_wrong`):

- **Gap A — codegen never spin-adapts.** `python/generate_planck_cc_kernels.py`
  calls `print_cpp_planck(...)` with **no** `spin_adapt=True`, so every emitted
  `*_planck_generated.cpp` still carries raw spin-orbital algebra
  (`0.25 t2 oovv`) bound to spatial storage — the defect. `grep spin_adapt
  generate_planck_cc_kernels.py` is empty.
- **Gap B — the runtime is single-sector.** `ArbitraryOrderRCCAmplitudes`,
  `ArbitraryOrderDenominatorCache`, the residual bundle
  (`GeneratedArbitraryOrderKernels.residuals_by_rank`, one per rank), and the
  Jacobi/DIIS update all assume **one** amplitude tensor per rank. t4 needs
  **two** (the reference `aabbaabb` + the second Sz sector `aaabaaab`;
  see `CCGEN_R3_HIGHER_RANK_BRIDGE_SCOPE.md`).

The emit itself is DONE: `print_cpp_planck("ccsdtq", spin_adapt=True)` emits the
spatial reference kernels (no raw `0.25`), the `quadruples_aaabaaab` sector kernel,
and the sector reads `amplitudes.sector_tensor(4, "aaabaaab")`; the whole TU
compiles against the real headers. The `ArbitraryOrderRCCAmplitudes::sector_tensor`
accessor + `sectors` storage vector already exist (empty by default). What remains
is populating/updating those sectors and flipping the codegen switch.

---

## Gap A — flip codegen to spin-adapted (small, do first)

- **A1 — add `--spin-adapt` to `generate_planck_cc_kernels.py`** and pass it into
  `print_cpp_planck(spin_adapt=...)`. Default it **on** for the production emit
  once A2/B land; keep an escape hatch for the historical raw path (the
  warm-start `.inc` uses a genuinely spin-orbital reference and is correct as-is,
  so it must NOT be spin-adapted — scope the switch to the tensor-backend TUs).
- **A2 — regenerate + wire the ccsdtq TU into a binary.** Today the generated
  ccsdtq TU is not `#include`d into any target (only compiled with
  `-fsyntax-only`). CMake (`CMakeLists.txt:440`, the
  `generate_planck_cc_kernels.py` custom command) must emit the spin-adapted TU
  and the driver's `PostHF::RCCSDTQ` path must call
  `make_generated_ccsdtq_kernels()`. *Gate:* the binary links; a Be
  `correlation rccsdtq` run produces a finite energy (correctness is B's gate).

Gap A alone fixes cc3 (CCSDT: one Sz sector, no Gap B) — so after A, a Be CCSDT
run through the generated path should already match PySCF CCSDT. That is the
cheap early win and the first end-to-end binary gate.

## Gap B — multi-Sz-sector runtime (the substantive work)

Mirror the Python solver (V1–V3 in the verification doc) block-for-block. Steps:

- **B1 — sector amplitude storage.** `ArbitraryOrderRCCAmplitudes.sectors`
  (already declared: `vector<pair<pair<int,string>, TensorND>>`) must be
  allocated + zero-init by `make_zero_rcc_amplitudes` for every independent
  sector of every rank ≥ 4. The sector set per rank comes from the generated
  metadata (emit a `sector_tags_by_rank` table alongside the kernel bundle, from
  `independent_spin_blocks`), so the runtime does not re-derive spin algebra.
  Its dims: the sector's own occ/vir shape from its external Sz block. *Gate:*
  `sector_tensor(4, "aaabaaab")` returns a correctly-shaped zero view after
  prepare on a CCSDTQ state.

- **B2 — sector denominators.** `ArbitraryOrderDenominatorCache` gains a matching
  `sectors` map (or the solve builds the sector denominator on demand). The
  denominator is `Σε_occ − Σε_vir` over the sector's slot layout — same formula,
  the sector's external block fixes which ε sits in which slot. *Gate:* the
  sector denominator equals the Python `den()` built on the sector template
  (cross-checked to 1e-12 on a toy).

- **B3 — sector residual kernels in the bundle.** Extend
  `GeneratedArbitraryOrderKernels` with a `sector_residuals` list keyed by
  `(rank, tag)` (the emit currently registers only `residuals_by_rank`; the
  sector kernel `compute_ccsdtq_quadruples_aaabaaab_residual` is emitted but not
  bundled — see the skip in `_emit_arbitrary_order_kernel_bundle`). The emitted
  `make_generated_<method>_kernels()` populates both. *Gate:* the bundle carries
  4 rank residuals + 1 sector residual for CCSDTQ; `validate_kernel_bundle`
  accepts the sector entries.

- **B4 — evaluate + update per block.** `evaluate_generated_arbitrary_order_residuals`
  evaluates each rank residual AND each sector residual;
  `update_amplitudes_with_jacobi_diis` updates each amplitude block (reference +
  sectors) from its own residual and denominator, with its own DIIS history.
  This is the exact port of Python V3. *Gate:* the multi-sector `ArbitraryOrderResiduals`
  carries the sector block and the update writes it (a single-iteration unit
  test on a perturbed Be state: the second sector moves off zero).

- **B5 — end-to-end Be CCSDTQ = FCI.** `correlation rccsdtq` on Be/sto-3g through
  the generated multi-sector runtime reaches −14.4036550465 (E_corr −0.05177…) to
  ~1e-8. This is the shared acceptance test with the Python doc's V4. Remove the
  raw-energy `@expectedFailure` (`GeneratedSpatialEnergyGate`) and delete the
  relabel-only `restricted_closed_shell` lowering once nothing depends on it.

## Arbitrary order (cc5, cc6, …)

Every step above keys off `(rank, sector_tag)` with the sector set coming from
`independent_spin_blocks(rank)` (⌊n/2⌋ sectors), so cc5/cc6 need **no new
per-rank code** — B1–B4 already loop the sector set. The only additions per new
rank are the generated kernels themselves (which the rank-agnostic emit already
produces). No numeric oracle exists above CCSDTQ (no small system makes cc5
exact; PySCF has no ≥4 amplitudes), so cc5+ is gated **structurally**: the TU
compiles, the bundle validates, and the solve runs to a self-consistent residual
— the same structural-generalization stance as R3.1.4.

## Coupling to the Python verification

The Python `solve_spin_adapted_spatial` (verification doc V1–V4) is the reference
implementation of B1–B4 in ~40 lines of NumPy. Build and gate it FIRST; it fixes
the algebra-side understanding and hands B a block-for-block spec plus a shared
acceptance number (Be = FCI). Do not develop B against the ~10 min Be binary run —
develop against the Python solver and the fast rank-8 bridge gate, reserve the
binary for B5.

## Acceptance (both gaps)

- Gap A: `--spin-adapt` codegen switch on for tensor-backend TUs; the ccsdtq TU
  linked into `hartree-fock`; Be CCSDT through the generated path matches PySCF.
- Gap B: Be CCSDTQ through the generated multi-sector runtime = FCI to 1e-8;
  raw-energy xfail removed; relabel-only lowering deleted.
- cc5/cc6 emit + link + run structurally (compile + bundle-validate + converge),
  no numeric oracle expected.

## Files

- `python/generate_planck_cc_kernels.py` — Gap A switch.
- `CMakeLists.txt` (~440) — regenerate + link the spin-adapted ccsdtq TU.
- `src/driver.cpp` — `PostHF::RCCSDTQ` routing to `make_generated_ccsdtq_kernels`.
- `src/post_hf/cc/amplitudes.{h,cpp}` — `sectors` storage (declared),
  `make_zero_rcc_amplitudes` sector alloc (B1), `sector_tensor` (done).
- `src/post_hf/cc/generated_arbitrary_runtime.{h,cpp}` — sector residual bundle
  (B3), per-block evaluate/update (B4).
- `src/post_hf/cc/solver_arbitrary.{h,cpp}` — per-block Jacobi/DIIS (B4).
- `python/ccgen/emit/planck_tensor_cpp.py` — emit `sector_tags_by_rank` +
  `sector_residuals` into the bundle (B1/B3); the term/read emit is done.
