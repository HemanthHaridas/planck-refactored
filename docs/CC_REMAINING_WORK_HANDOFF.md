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
- Water dimer / cc-pVTZ `ccsd(t)`: 39.6 s at 4 threads, 1.43 GB peak.
  CCSD 2.7× PySCF per iteration (flat in the basis), (T) 1.4×.
- **Time, not memory, limits size now.**

## Tier 0: no blockers, cheap

| # | Step | State |
|---|---|---|
| 0a | `run_rccgen` refuses a `PLANCK_CC_SPIN_ADAPT=OFF` build by name | **Done**, `d2cbc11b`. Checked OFF (refuses) and back ON (gates pass) on `build-default`. |
| 0b | `planck-mpi` gets `hartree-fock`'s CC definitions (adds `PLANCK_CC_UCC`, `PLANCK_CC_DRESS_OPERATORS`) through one list, `_planck_posthf_cc_defs` in `CMakeLists.txt` | **Done.** `oh_uccsd_t_631g` through `mpirun -n 2 build/planck-mpi` equals serial `hartree-fock` to every printed digit (`Total UCCSD(T) Energy` −75.4625534240, E(T) −0.0005574953); compared by hand, since 0e is open. The CC gates on `build/` pass (`oh_uccsd_t_631g`, `b_ucc{2,4}_sto3g`, `water_rccsd_t_631g`, `h2o2_c1_rccsd_t_631g`, `lih_rccsdt_generated_sto3g`, `be_rccsdtq_sto3g`). |
| 0c | Profile what is left in (T) | **Done** (size scope 10c). Dimer cc-pVTZ (T) 8.1–8.4 s: W 89 %, V 6.7 %, energy loops 3.8 %. Inside W: gathers 22 %, GEMMs 27 %, `ooov` loops 25 %, assembly 25 %. No single hot spot; the GEMMs are a quarter. Input committed as `tests/benchmarks/ccsd_t/water_dimer_ccpvtz.hfinp`. |
| 0d | CCSD[T] diagnostic: (T) without V, `wx·wx/D` in `src/post_hf/cc/ccsd_t.cpp` | A few lines. **External dependency:** the gate needs a CFOUR or MRCC value; PySCF does not print CCSD[T]. `docs/CCSDT_Q_SCOPE.md` step 0. |
| 0e | Teach `tests/mpi_smoke_compare.py` the `Total RCCSD(T) Energy` / `Total UCCSD(T) Energy` lines | **Done.** One pattern, `Total <label> Energy`, now reads every total printed by `Logger::correlation_energy` (MP2, RCCSD(T), UCCSD(T), ...). New cases `water_rccsd_t_631g_mpi_smoke` and `oh_uccsd_t_631g_mpi_smoke` (extended; `--atol 0`, 2 and 3 ranks) pass; a change confined to the UCCSD(T) line is now caught where before only the SCF was compared. These are M0/M1's baseline gates. |

## Tier 1: CCSD and (T) time (generator or build changes)

Independent of each other. Each needs an A/B timing against a binary from
the same tree and the gates in `docs/CCSD_T_SIZE_SCOPE.md` ("Gates for every
change here"), including re-pinning emit hashes only after showing the old
pins reproduce with the change off.

| # | Step | Depends on | Expected |
|---|---|---|---|
| 1a | (T) loop sections made contiguous (generator): gathers from once-transposed `ovvv`/t2 layouts, the paired `ooov` blocks as contiguous loops or small GEMMs, the scatter writing the index orders the assembly reads. Batch over one virtual only if that falls short. | 0c (done) | (T) ~2–2.5×, past PySCF's 6.0 s |
| 1b | R3b: CCSD o³v³ GEMMs on j ≥ i, one GEMM per i (staircase) in `cc_gemm_contract` | none | ≤ ~10 % of CCSD arithmetic |
| 1c | UCC residual: port the half range (R1), common-operand merges (R2, R2b) and GEMM lowering (R3) | Measure the UCC residual first | Possibly large. Every step-9 change is gated "not UCC" (`_pair_half_kernel`, `merge_common_operand_reads`, `_gemm_lowering`), so `uccsd(t)` has none of them. Not in any scope yet; open one. |
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
| M0 | (T) across ranks: each rank fills the virtual-triple slots it `owns`, one `allreduce_inplace`, ordered serial sum. Bitwise equal to serial. | Nothing left (0e done). (T) is ~21 % of the dimer cc-pVTZ run, so it pays now. |
| M1 | Stripe the outer free index of each generated CCSD residual nest, one allreduce per residual tensor per iteration | Nothing left (0b, 0e done). Better after 1c, so it distributes the faster UCC residual. |
| M2 | Split the memory (distributed t2 / `vvvv`) | R4 shows one node is saturated. Do not start without it. |

Gates for M0/M1: serial against 2 and 3 ranks, every printed digit, on
`ccsd(t)` and `uccsd(t)` cases, plus a drop-the-reduce mutation, all through
`tests/mpi_smoke_compare.py` with its `--timeout`.

## Tier 4: new methods (parallel to tiers 1–3)

**Quadruples** (`docs/CCSDT_Q_SCOPE.md`):
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

**Λ** (own scope, not opened): de-excitation operators, a left projector,
Λ residuals, a linear solver and the (n)_Λ energies. Largest item; also the
foundation of analytic CC gradients.

**CC3**: rename the existing `cc3` keyword first (it means generated CCSDT,
`src/io/io.cpp`), or a user asking for CC3 silently gets CCSDT. Low value
without EOM.

## Suggested order

1a, 1b and 1c, with the user
deciding 1e → R4 → M0, M1. Run the quadruples or ROHF chain alongside,
depending on whether accuracy or open-shell coverage comes first.
