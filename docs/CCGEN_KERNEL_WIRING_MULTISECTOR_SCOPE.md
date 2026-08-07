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

- **A1 — add `--spin-adapt` to `generate_planck_cc_kernels.py`. LANDED.** The
  flag threads into `print_cpp_planck(spin_adapt=...)` for both TUs this script
  emits (tensor-backend + `--arbitrary-lower-ranks` companion); it does NOT touch
  the warm-start `.inc` (emitted by `planck_rccsd_warm_start.py`, correct on a
  spin-orbital reference). Default **off** for byte-compatibility with the
  historical (defective) emit — A2/B flip production to on. *Gate:*
  `SpinAdaptedEmitTests::test_codegen_cli_spin_adapt_switch` (subprocess) — default
  CCSD energy keeps the raw `0.25`; `--spin-adapt` emits spatial 2J-K (no `0.25`).
- **A2 — build with the spin-adapted TUs. LANDED (switch + compile gate);
  binary-run acceptance pending a build.** The wiring the doc feared missing was
  already there: `generated_kernel_registry.cpp` `#include`s the generated
  ccsdtq TU (guarded `PLANCK_CC_MAXORDER >= 4`) and defines
  `make_generated_rccsdtq_kernels()`, which the driver's `run_rccsdtq`
  (`src/post_hf/cc/ccsdtq.cpp`) already calls through `make_generated_rcc_kernels(4)`
  → `run_generated_arbitrary_order_iterations`. So the only gap was that CMake
  emitted the **defective** TU. Added the `PLANCK_CC_SPIN_ADAPT` CMake option
  (default OFF for byte-compatibility until Gap B lands) that passes `--spin-adapt`
  to the generator. *Gate:*
  `SpinAdaptedEmitTests::test_registry_compiles_with_spin_adapted_ccsdtq` —
  generate spin-adapted ccsd/ccsdt/ccsdtq TUs and syntax-check
  `generated_kernel_registry.cpp` (the real link path, `MAXORDER=4`) against them,
  so the multi-sector `sector_tensor` reads compile in the binary context.
  *Remaining acceptance (needs an actual build):* configure
  `-DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_SPIN_ADAPT=ON`, build `hartree-fock`, and
  confirm a Be `correlation rccsdt` run matches PySCF CCSDT (CCSDT has one Sz
  sector → Gap-B-independent, the cheap early win); Be `rccsdtq` gives the
  CCSDT-level answer until Gap B drives the second t4 sector.

Gap A alone fixes cc3 (CCSDT: one Sz sector, no Gap B) — so after A, a Be CCSDT
run through the generated path should already match PySCF CCSDT. That is the
cheap early win and the first end-to-end binary gate.

## Gap B — multi-Sz-sector runtime (the substantive work)

Mirror the Python solver (V1–V3 in the verification doc) block-for-block. Steps:

- **B1 — sector amplitude storage. LANDED.** `make_zero_rcc_amplitudes` gains an
  overload taking a sector list `vector<pair<int,string>>`; it zero-inits each
  `ArbitraryOrderRCCAmplitudes.sectors` entry keyed `(rank, tag)` alongside the
  per-rank reference blocks. A sector block is `rank_dims(rank)`-shaped — the same
  occ/vir dims as its rank's reference, since the spin projection lives in the
  algebra, not the shape (confirmed against the Python V2 array shapes). The
  no-sector overload delegates with an empty list (unchanged for ≤ CCSDT). The
  sector list is supplied by the generated bundle (B3), so the allocator does not
  re-derive spin algebra. *Gate:*
  `cc_arbitrary_solver::test_make_zero_amplitudes_allocates_sectors` —
  `sector_tensor(4, "aaabaaab")` returns a correctly-shaped (`rank_dims(4)`) zero
  view, reference blocks intact, and an **unrequested** sector errors rather than
  silently zero-filling.

- **B2 — sector denominators. RESOLVED (no new storage needed).** The Python V2
  audit proved the sector denominator is **identical** to its rank's reference
  denominator: for an RHF reference the orbital energies are spin-free, so
  `Σε_occ − Σε_vir` over the spatial slots is the same for `aabb` and `aaab`
  (both rank-4). So the solve reuses the existing `denominators.tensor(rank)` for
  every sector of that rank — no `ArbitraryOrderDenominatorCache.sectors` map is
  required. (Kept as an explicit note so B4's update divides each sector residual
  by its rank denominator, not a per-sector one.)

- **B3 — sector residual kernels in the bundle. LANDED.**
  `GeneratedArbitraryOrderKernels` gained `sector_tags`
  (`vector<pair<int,string>>` — feeds B1's allocator) and `sector_residuals`
  (`vector<SectorResidual{rank, tag, kernel}>` — feeds B4). The emit
  (`_emit_arbitrary_order_kernel_bundle`) splits targets into reference residuals
  (`residuals_by_rank`) and sector residuals (`quadruples_aaabaaab` →
  `(4, "aaabaaab")`), registering the latter instead of skipping. Empty for
  ≤ CCSDT. *Gates:* `test_ccsdtq_bundle_registers_the_sector` /
  `test_ccsdt_bundle_has_no_sectors` (emit); `cc_arbitrary_solver::
  test_bundle_carries_sector_residual` (C++: struct holds the keyed sector
  residual, tag matches, kernel invokable); registry + standalone TU still
  compile with the sector-registering bundle.

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
