# Which environment overrides may reach a production binary?

Answered 2026-11-07 (opened as a scope 2026-10-12). Status lives in
`vault/Status/`.

## Answer

**Only ones that change what is printed, never what is computed.** Every override that changed a result, a solver, or the
set of allowed workflows has been deleted or turned into an input keyword.
E1 and E4 landed on 2026-10-12, E2 and E3 on 2026-11-07 (PR #181).

## The rule

An environment variable that a production binary reads may change output
only. Anything that changes a result, a solver, or which workflows are
allowed must be either an input keyword (a user-facing choice, visible in
the input file) or compiled out of the default build. The reason is the
failure mode: an exported variable is invisible in the input file and the
output, so a stale `export` in someone's shell silently changes numbers.

**Test-only solver choices** go through a programmatic `Options` field, not
an environment variable or a keyword. The precedents are
`DHZVectorOptions::backend` (the dense Z-vector backend, used only by
`tests/dh_zvector.cpp` and `tests/dft_coulomb_response.cpp`) and
`Options::udh_gradient_validation`.

## What a production binary still reads (checked 2026-11-07)

Every `getenv` in `src/` uses a literal name, and all of them are on this
list:

- configuration: `BASIS_PATH`;
- output only: `PLANCK_DFT_GRADIENT_DEBUG`, `PLANCK_DEBUG_GRID_ACC`,
  `PLANCK_DEBUG_RMP2_{TERMS,MATRICES,IMAT}`, `PLANCK_DEBUG_DM2BUF`,
  `PLANCK_DEBUG_RHF_RESPONSE`, `PLANCK_PHASE_TIMING`, `PLANCK_CC_RANK_TIME`,
  `PLANCK_DUMP_AO_INTEGRALS`.

A new `getenv` that changes a result breaks the rule. Re-run
`grep -rn 'getenv(' src` when adding one.

## What was removed, and where each went

| Override | What it did | Now |
|---|---|---|
| six unused debug overrides (`PLANCK_DEBUG_VHF1_KX`, `PLANCK_DEBUG_OVERLAP_SCALE`, `PLANCK_DEBUG_GAMMA_TRACE`, `PLANCK_DFT_DH_EQ40_DENSE_REFERENCE`, `PLANCK_CC_T3_DIFF`, `PLANCK_CC_FIXTURE_DIR`) | changed results; no test used them | deleted 2026-10-12 |
| `PLANCK_DFT_ALLOW_RS_WORKFLOWS` (E1) | bypassed the range-separated workflow guard (really only RS TDDFT, then unvalidated) | deleted 2026-10-12; RS TDDFT was later admitted through the normal guard |
| `PLANCK_CC_T3_TIME` (E4) | timing probe in an unreachable generated-kernel branch of `tensor_backend.cpp` | deleted with the branch, 2026-10-12 (~−230 lines, plus two orphaned generator steps) |
| `PLANCK_RCCSDT_BACKEND` (E2) | picked the RCCSDT backend | input keyword `rccsdt_backend auto\|determinant\|tensor\|optimized` in `[scf]`, validated at parse time; `auto` is the size-based choice |
| `PLANCK_DFT_DH_HESSIAN_PROBE_LOG` (E3) | switched the DH Z-vector to the dense backend **and returned a different gradient** | deleted with `dh_hessian_probe.{h,cpp}`, `dh_probe_gmres.h`, two CTests and two scripts; the production path is always GMRES |
| `PLANCK_CASSCF_CALL_PROBE` | per-macro call counters for `docs/CASSCF_PERFORMANCE.md` | deleted 2026-11-07 (it only printed, but its counters were temporary) |

Why E3 could be deleted rather than gated: the probe existed to justify
promoting matrix-free GMRES into the normal path, which landed in
`b4175570` with full-coordinate FD maxima 7.9e-9 to 1.7e-8 Ha/Bohr. The
comparison it made is covered at the operator level: `tests/dh_zvector.cpp`
requires `|z_gmres − z_dense| < 1e-11`. `tests/test_dh_o3_zvector.py` now
asserts that `compute_analytic_dh_gradient` contains no `getenv` and no
`DenseReference`.

## The shared-input trap (E2)

`ch4_rccsdt_sto3g` (hand-written path) and `ch4_rccsdt_generated_sto3g`
(generated path) used to share one input, and a setting changed for one
broke the other (`70a587d`, the `use_diis` flip). The generated case now has
its own input, `tests/inputs/regression/post_hf/ch4_rccsdt_generated_sto3g.hfinp`,
which carries `rccsdt_backend optimized`. Keep regression cases that pin
different code paths on separate inputs.

The regression runner keeps its generic per-case `env` support; the FCIQMC
thread-count cases use it for `OMP_NUM_THREADS`.
