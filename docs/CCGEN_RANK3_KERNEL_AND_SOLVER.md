# Why did the generated rank-3 CCSDT kernel give a wrong energy, when the kernel was correct?

A CC kernel and the solver that iterates it must agree on how amplitudes are **represented**. Planck
has two incompatible representations, and for a while one solver was iterating the other
representation's kernel. The energy came out self-consistent, converged, and wrong by 7.56e-05 Eh.

This answers that: what the two representations are, why mixing them fails silently, and how the
mismatch survived a green test suite.

## The two representations

| | `tensor_backend` (hand-written) | arbitrary-order harness (ccgen) |
|---|---|---|
| amplitude storage | **symmetry-packed**: DIIS packs only the unique wedge (`i<=j` for t2, `i<=j<=k` for t3) and rebuilds the rest via `restore_restricted_t{2,3}_from_unique` | **dense**: full tensors, every index stored |
| residual convention | one canonical representative per permutation orbit; `restore_restricted_t3_structure` re-imposes full permutational symmetry each iteration | every index permutation emitted explicitly (824 accumulations across 24 distinct factor skeletons at rank 3) |
| r1/r2 source | hand-written `build_dressed_sd_residuals` from dressed intermediates | generated rank-1/rank-2 kernels |

Both are internally correct. Each reproduces PySCF `rccsdt` on CH4/STO-3G to ~1.5e-08 when run
end-to-end in its own scheme.

**The wedge packing and `restore` are one coupled convention, not two independent choices.** Packing
only `i<=j<=k` is information-preserving *only* if the amplitudes carry full permutational symmetry
— which is exactly the property `restore` imposes. Delete either half and the other breaks: measured,
removing `restore` diverges with **both** hand-written and generated residual sources.

## The defect

`run_tensor_optimized_rccsdt` called `run_tensor_rccsdt_impl(..., true, true)` — the *generated*
triples kernel inside the *hand-written* solver. The generated kernel does not emit residuals in the
wedge representation, so the solver's packing silently discarded and reconstructed information every
iteration. The iteration stayed self-consistent, so it converged cleanly to the wrong fixed point.

Measured on CH4/STO-3G (`no=5 nv=4`), against PySCF `rccsdt` = −39.8058445240:

| r1/r2 | r3 | `restore` | result |
|---|---|---|---|
| hand | generated | yes | converges, **−7.56e-05** (the shipped defect) |
| hand | generated | no | **diverges** |
| generated | generated | yes | converges, +8.23e-05 |
| generated | generated | no | **diverges** |
| generated (arbitrary harness, dense) | generated | n/a | **+1.49e-08** ✓ |

No combination of residual sources inside `tensor_backend` is correct, because the representation
mismatch is structural rather than a missing or duplicated term.

**The fix**: route the generated rank-3 kernels to the arbitrary-order harness — the representation
they are emitted for. `optimized` now lands at +1.44e-08 (a 5247× error reduction) and agrees with
the hand-written path to 1.0e-10. Without `-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON` it fails with an
actionable message instead of returning a wrong number.

## Why the kernel was not the problem

Every component was verified correct *individually* before the composition was implicated:

- **The equations.** The plain and arbitrary rank-3 TUs share 811/811 normalized terms, zero
  differences either way. The arbitrary harness runs exactly them and matches PySCF.
- **The kernel call.** Both harnesses were made to evaluate the identical kernel from
  **bitwise-identical inputs** (a deterministic `0.01·sin(i+1)` pattern injected into t1/t2/t3, all
  seven ERI blocks already verified identical). `R3_out` agreed to all 16 digits in **sum and
  sum-of-squares** — the second rules out a permutation of the same values, which a single checksum
  would miss.
- **`restore` itself.** The hand-written backend applies the same `restore` in the same solver and
  reaches +1.45e-08.
- **Denominators.** `restricted_d3` and `build_arbitrary_order_denominator_cache` compute the same
  `Σε_occ − Σε_virt`, applied as the same Jacobi step with the same `1e-12` guard. (They are *inert*
  for the kernel, which never reads them — but the update does, so "inert" was only true of one
  consumer.)

## Ruled out — do not re-investigate

Each was killed by measurement, not inspection. Five of them were hypotheses formed by reading code,
and all five were wrong; the direct comparisons (bitwise fingerprints, converged energies against an
external oracle) produced every correct result.

| hypothesis | verdict |
|---|---|
| Double symmetrization — generated residual already permuted, `restore` permutes again | **No.** Removing (`−7.56e-05` → `+1.90e-04`) or halving `restore` made it worse and broke convergence. The supporting numerical model assumed `G == perm_sym(x)`, never measured on the real residual. |
| Stride mismatch — spin-orbital allocation walked with spatial extents | **No.** Arm A's amplitudes are spatial (`project_rccsd_warm_start_to_restricted`, `:2594`). The `T3~=3.91 MiB` figure cited as proof describes `state.triples.amplitudes` on the **staged** path, which this branch never executes. |
| Unique-triangle DIIS pack/unpack is lossy for generated amplitudes | **Partly** — it is half of the coupled convention, but removing it breaks the hand-written path too, so it is not the discriminator. |
| Pure double count of the T3→SD feedback | **No.** Removing it overshoots the other way (`+1.90e-04`) and is 2.5× worse. |
| Hybrid residual sources — wire the unused generated r1/r2 kernels in too | **No.** Converges cleanly (21 iters, `rms=7.9e-10`) to `+8.23e-05`, still wrong. |
| Block convention (`rebind_physicist`) | **No.** All seven ERI blocks bitwise identical between harnesses, including the `ovvo`/`oovv` coincidence — which appears in both, so it is a property of CH4's integrals. |
| Spatial spin-adaptation lowering of an odd-rank manifold | **No.** Shared by all ranks; rank 4 uses it and is correct. |
| Rank parity (odd ranks broken, even ranks fine) | **Dead — the premise was false.** Rank 3 is correct. The arbitrary harness is correct at ranks 2, 3 and 4. Ranks 5/6 still have no numeric gate. |

## Why a green suite missed it

**The hand-written tensor solver had no regression gate for its entire life**, and neither did the
generated kernel under it. `choose_determinant_backstop` routes anything with
`nso <= 16 && ndet <= 10000` to the determinant-space prototype, and *every* CC test system is under
that threshold:

| case | nso | ndet | path actually exercised |
|---|---|---|---|
| `h2_rccsdt_sto3g` | 4 | 6 | determinant prototype |
| `lih_rccsdt_sto3g` | 12 | 495 | determinant prototype |
| `water_rccsdt_sto3g` | 14 | 1001 | determinant backstop |
| `be_rccsdtq_sto3g` | 10 | 210 | backstop-eligible |

`water_rccsdt_sto3g` goes further and *asserts* the handoff string
`RCCSDT[TENSOR] : Using the determinant-space CCSDT backstop` — a gate pinning that the tensor path
**declines to run**. So the PySCF-validated CC suite validates the determinant prototype, and the
"hand-written is trusted" premise that framed this whole investigation rested on coverage that did
not exist.

This is the same defect class as the original rank-3 bug, where `compute_ccsdt_triples_residual` had
no caller for months. **Linkage is not execution — and a green suite is not coverage when every case
takes the other branch.**

`ch4_rccsdt_sto3g` closes it: `nso=18 ndet=43758`, `no=5 ≠ nv=4`, pinned to PySCF −39.8058445240.
It is the **only** in-tree rank-3 case that reaches the tensor path at all, and it was verified to
*fail* on the old path (by both energy and the `kernels=hand-optimized` marker) before being trusted
to pass.

Two generated kernels remain emitted-but-uncalled in the same TU:
`compute_ccsdt_{singles,doubles}_residual`. They were executed once, during this investigation, and
produce a converged-but-wrong energy in this solver — consistent with the representation story, and
untested anywhere else.

## Working constraints, learned the expensive way

- **Use a system that clears the backstop.** LiH (`nso=12 ndet=495`) converges correctly via a path
  that never calls the kernel — indistinguishable from a pass. CH4/STO-3G is the only in-tree rank-3
  system that is both backstop-clearing and non-square.
- **Never `no == nv`.** A square system lets a wrongly-ordered read stay in bounds and fail silently.
- **Build with an explicit `CMAKE_BUILD_TYPE`.** The repo's `build/` has it empty, which drops
  `-DNDEBUG`, re-enables the CC tensor bounds asserts, and makes every timing meaningless — the
  accessor fix is effectively reverted in that tree.
- **Compare totals, not `e_corr`, against PySCF.** Planck's RHF sits ~7.5e-08 above PySCF's; the
  offset cancels in the total and does not in the correlation energy.
- **A probe whose output is degenerate proves nothing.** The first fixed-amplitude comparison used
  zeroed amplitudes, so `R3_out` was trivially zero in both arms and matched vacuously.
- **Check the backend marker before believing any number.** `RCCSDT[OPT]` / `kernels=ccgen-generated`
  for the generated path; a build silently selects another backend otherwise.

## Cost

The correct path is **~500× slower** than the hand-written one (0.19 s vs ~100 s on CH4),
per-iteration rather than convergence — the arbitrary harness converges in *fewer* iterations
(14 vs 24). Two figures that circulated are wrong and should not be cited: "~3.1×" compared two
*generated* configurations and mislabelled one as the hand-written baseline; "~180×" predates the
tensor-accessor fix and never recorded its dimensions.

Open, with a blocking profile step: `docs/CCGEN_ARBITRARY_HARNESS_COST.md`.

## Key code locations

| what | where |
|---|---|
| the routing fix | `run_tensor_optimized_rccsdt`, `src/post_hf/cc/tensor_backend.cpp` |
| wedge-packed DIIS (half the coupled convention) | `pack_/unpack_restricted_unique_rccsdt_amplitudes`, same file |
| `restore_restricted_t3_structure` (the other half) | same file, `:~1977` |
| the correct harness | `src/post_hf/cc/generated_arbitrary_runtime.cpp`, `solver_arbitrary.cpp` |
| backstop gate that hid all of this | `choose_determinant_backstop`, same file |
| the gate that now covers it | `ch4_rccsdt_sto3g` in `tests/regression_cases.json` |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`, which are canonical.
