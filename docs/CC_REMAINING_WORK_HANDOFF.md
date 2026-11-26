# Coupled cluster: remaining work, ordered by dependency

Handoff, 2026-11-25. Collects the open steps from the CC scope docs and
orders them by what blocks what. The scope docs hold the measurements and
ruled-out routes; this file holds only the order. Delete it once the tiers
below are done or moved into their scopes. Status of record stays in
`vault/Status/Open Work.md`.

Sources: `docs/CCSD_T_SCOPE.md`, `docs/CCSD_T_SIZE_SCOPE.md`,
`docs/CCSDT_Q_SCOPE.md`, `docs/CC_MPI_SCOPE.md`,
`docs/ROHF_CORRELATION_SCOPE.md`.

## Where things stand

- `ccsd(t)` (generated spatial RCCSD) and `uccsd(t)` (generated `ucc2`,
  `-DPLANCK_CC_UCC=ON`) equal PySCF 2.15 in Cartesian and spherical bases.
- Water dimer / cc-pVTZ `ccsd(t)`: 36.4–38.6 s at 4 threads, 1.43 GB
  peak. CCSD 2.7× PySCF per iteration (flat in the basis); (T) 4.5–4.8 s,
  faster than PySCF's 6.0 s since 1a.
- Water⁺ / cc-pVTZ `uccsd(t)`: 3.67–3.95 s at 4 threads (was 9.4 s);
  UCC CCSD 0.339 → 0.099 s per iteration, (T) 1.33 → 0.19 s (the U steps,
  size scope "UCC").
- **Time, not memory, limits size now.**

## Tier 0: no blockers, cheap

| # | Step | State |
|---|---|---|
| 0a | `run_rccgen` refuses a `PLANCK_CC_SPIN_ADAPT=OFF` build by name | **Done**, `d2cbc11b`. Checked OFF (refuses) and back ON (gates pass) on `build-default`. |
| 0b | `planck-mpi` gets `hartree-fock`'s CC definitions (adds `PLANCK_CC_UCC`, `PLANCK_CC_DRESS_OPERATORS`) through one list, `_planck_posthf_cc_defs` in `CMakeLists.txt` | **Done.** `oh_uccsd_t_631g` through `mpirun -n 2 build/planck-mpi` equals serial `hartree-fock` to every printed digit (`Total UCCSD(T) Energy` −75.4625534240, E(T) −0.0005574953); compared by hand, since 0e is open. The CC gates on `build/` pass (`oh_uccsd_t_631g`, `b_ucc{2,4}_sto3g`, `water_rccsd_t_631g`, `h2o2_c1_rccsd_t_631g`, `lih_rccsdt_generated_sto3g`, `be_rccsdtq_sto3g`). |
| 0c | Profile what is left in (T) | **Done** (size scope, "(T) time"). Dimer cc-pVTZ (T) 8.1–8.4 s: W 89 %, V 6.7 %, energy loops 3.8 %. Inside W: gathers 22 %, GEMMs 27 %, `ooov` loops 25 %, assembly 25 %. No single hot spot; the GEMMs are a quarter. Input committed as `tests/benchmarks/ccsd_t/water_dimer_ccpvtz.hfinp`. |
| 0d | CCSD[T] diagnostic: (T) without V, `wx·wx/D` in `src/post_hf/cc/ccsd_t.cpp` | **Waiting for MRCC access** (user decision, 2026-11-26). A few lines; the gate needs a CFOUR or MRCC value, and neither is available. `docs/CCSDT_Q_SCOPE.md` step 0. |
| 0e | Teach `tests/mpi_smoke_compare.py` the `Total RCCSD(T) Energy` / `Total UCCSD(T) Energy` lines | **Done.** One pattern, `Total <label> Energy`, now reads every total printed by `Logger::correlation_energy` (MP2, RCCSD(T), UCCSD(T), ...). New cases `water_rccsd_t_631g_mpi_smoke` and `oh_uccsd_t_631g_mpi_smoke` (extended; `--atol 0`, 2 and 3 ranks) pass; a change confined to the UCCSD(T) line is now caught where before only the SCF was compared. These are M0/M1's baseline gates. |

## Tier 1: CCSD and (T) time (generator or build changes)

Independent of each other. Each needs an A/B timing against a binary from
the same tree and the gates in `docs/CCSD_T_SIZE_SCOPE.md` ("Gates for every
change here"), including re-pinning emit hashes only after showing the old
pins reproduce with the change off.

| # | Step | Depends on | Expected |
|---|---|---|---|
| 1a | (T) blocks as GEMMs on contiguous layouts made once per (T); contiguous assembly | 0c (done) | **Done**: (T) 8.1–8.4 → 4.5–4.8 s (1.8×), past PySCF's 6.0 s; W 29.6 → 14.6 thread-s. Left in (T): V and the energy loops (~20 %). Size scope, "(T) time". |
| 1b | R3b: CCSD o³v³ GEMMs on j ≥ i, one GEMM per i (staircase) in `cc_gemm_contract` | none | **Done.** CCSD per iteration 0.83–0.86 → 0.80 s on the dimer cc-pVTZ (1.04–1.08×). Helper-only change (the emitter already puts i/j first in every affected group). New unit test `planck-cc-gemm-contract` (4 staircase shapes + same-group + full, ≤ 1.1e-15; a wrong column range fails 600 elements); rccsd_t gates pass, CCSD/(T) output identical at 1/2/4/8 threads, core 178/178. |
| 1c | UCC: the restricted CCSD and (T) work, per spin block (U1–U4) | none | **Done** (size scope, "UCC"): GEMM lowering for every UCC CCSD term (U1, U2), each intermediate built once per residual evaluation (U3a, restricted too), the merges per spin block (U3b), (T) as shared elementary GEMMs on prepared layouts (U4). Water⁺ cc-pVTZ UCC CCSD 0.339 → 0.099 s per iteration, (T) 1.33 → 0.19 s, run 9.4 → 3.8 s. The half range does not apply to UCC. What is left: building intermediates (half the residual) and the spin-block ERI preparation (33 % of the run). |
| 1d | R2c: share linearly dependent builder combinations (four `W_t2v_oovv` → three contractions) | Re-measure its share on the lowered build | ≤ 1.06× |
| 1e | R3c: Accelerate / OpenBLAS behind `cc_gemm_contract` | **User decision**: the optional `USE_BLAS` build (B1, `docs/ERI_FOCK_BLAS_SCOPE.md`); hermeticity and bitwise reproducibility trade-offs | 2.5–4× on the GEMMs, the largest lever left |

## Tier 2: re-measure (gates the large-system work)

- **R4.** Re-run the size ladder against PySCF after whichever of tier 1
  lands; timing and footprint in separate runs (profiling slows the run).
  Ladder and PySCF script: `docs/CCSD_T_SIZE_SCOPE.md` ("Tools",
  "Evidence").
- Only if R4 shows memory binding again (ns⁴ AO tensor in the SCF, v⁴
  `vvvv` in the CCSD): build the MO blocks from a direct or RI transform
  without storing the AO tensor. RI-CCSD would be its own scope.

## Tier 3: MPI work-splitting (`docs/CC_MPI_SCOPE.md`)

| # | Step | Depends on |
|---|---|---|
| M0 | (T) across ranks: each rank fills the virtual-triple slots it `owns`, one `allreduce_inplace`, ordered serial sum. Bitwise equal to serial. | Nothing left (0e done). (T) is ~12 % of the dimer cc-pVTZ run after 1a (4.6 of ~37 s), so it pays less than M1. |
| M1 | Stripe the outer free index of each generated CCSD residual nest, one allreduce per residual tensor per iteration | Nothing left (0b, 0e, 1c done). |
| M2 | Split the memory (distributed t2 / `vvvv`) | R4 shows one node is saturated. Do not start without it. |

Gates for M0/M1: serial against 2 and 3 ranks, every printed digit, on
`ccsd(t)` and `uccsd(t)` cases, plus a drop-the-reduce mutation, all through
`tests/mpi_smoke_compare.py` with its `--timeout`.

## Tier 4: new methods (parallel to tiers 1–3)

**Quadruples** (`docs/CCSDT_Q_SCOPE.md`). **Waiting for MRCC access**
(user decision, 2026-11-26): step 1 needs a CFOUR/MRCC reference, and 2–3
follow it.
1. CCSDT(Q), restricted. Uses the generated rank-3 amplitudes (ready).
   Verify the restricted weights (2 S_aaaa + 32 S_aaab + 36 S_aabb)/576
   numerically; check any factored term against the unfactored oracle;
   needs a CFOUR/MRCC reference. Transcribe term lists from the papers.
2. UHF CCSDT(Q): after 1, needs `ucc3`.
3. CCSDTQ(P): only on request.

**ROHF** (`docs/ROHF_CORRELATION_SCOPE.md`, Open Work priority 6):
1. Semicanonical transform (per-spin C, ε, f_ov).
2. ROHF-MP2: UMP2 on semicanonical orbitals plus the singles term.
3. ROHF-CCSD: first check whether `solve_determinant_cc` already handles
   f_ov ≠ 0; the tensor route needs ccgen with `canonical_fock=False`.
4. ROHF-(T): same `canonical_fock=False` change; the UHF (T) it builds on
   exists.

**Λ** (`docs/CC_LAMBDA_SCOPE.md`). Generated by differentiating the CC
Lagrangian built from the T equations. Done in ccgen and gated against
PySCF: spin-orbital, RCC and UCC Λ equations, and the unrelaxed 1- and
2-RDMs (`python/ccgen/lagrangian.py`). Next: 3b, Λ kernels in Planck (t
and λ as kernel inputs) → 3c, a Λ solver with T frozen → 3d, density
kernels → 3e, a driver keyword (name to decide) with dipole and natural
occupations → 4, consumers: ΛCCSD(T), CCSDT(Q)_Λ, analytic CC gradients
(own scope). The foundation of gradients and properties.

**CC2, CC3, CCn, CCSDT-n** (`docs/CC_APPROXIMATE_MODELS_SCOPE.md`). Its
step 0 is independent and worth doing now: every `ccN`/`uccN` keyword means
rank-N CC, so `cc3` silently runs full CCSDT. Spelled rank names now exist
for every rank on both routes, from one table (`generated_cc_keyword`,
`PostHF::RCCGEN`/`UCCGEN`). Left: migrate the seven inputs and refuse `ccN`
until the real models exist (a user decision).
The models themselves are one per-term perturbation-order filter in ccgen;
build them with an excited-state CC scope or on request.

## Suggested order

1c is done. The user decides 1e → R4 → M0, M1; Λ 3b–3e alongside. Run the
ROHF chain alongside. 0d and the quadruples wait for MRCC access.
