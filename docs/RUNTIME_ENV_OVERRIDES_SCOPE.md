# Which environment overrides may reach a production binary?

**Status: SCOPE (in flight, opened 2026-10-12).** E1 and E4 landed 2026-10-12; E2 and E3 remain.
Rewrite this as an answer when they land.

## The rule this scope enforces

An environment variable that a production binary reads may change **what is printed**, never
**what is computed**. Anything that changes a result, a solver, or which workflows are allowed
must be either an input keyword (a user-facing choice, visible in the input file) or compiled
out of the default build. The reason is the failure mode: an exported variable is invisible in
the input file and the output, so a stale `export` in someone's shell silently changes numbers.

Six overrides that broke this rule were deleted on 2026-10-12, none of them used by any test or
script: `PLANCK_DEBUG_VHF1_KX`, `PLANCK_DEBUG_OVERLAP_SCALE`, `PLANCK_DEBUG_GAMMA_TRACE`,
`PLANCK_DFT_DH_EQ40_DENSE_REFERENCE`, `PLANCK_CC_T3_DIFF`, `PLANCK_CC_FIXTURE_DIR`.

## What is left (verified against the tree, 2026-10-12)

| Override | What it actually does | Users | Step |
|---|---|---|---|
| `PLANCK_DFT_ALLOW_RS_WORKFLOWS` | Bypasses the range-separated workflow guard. Since Gradient, Frequency, GeomOpt and GeomOptFrequency are already allowed, it only unlocks **LinearResponse** (unvalidated RS TDDFT) and **ImaginaryFollow**, which then fails anyway with "DFT imaginary-mode following is not implemented yet". | `tests/pyscf/hse06_gradient_components.py`, which sets it on a **gradient** run, where it does nothing | E1 |
| `PLANCK_RCCSDT_BACKEND` | Selects the RCCSDT backend (determinant / tensor / optimized). A real solver choice with no input keyword. | `ch4_rccsdt_generated_sto3g`, `lih_rccsdt_generated_sto3g` (via the runner's `env`), `tests/pyscf/compare_rccsdt_backends.py` | E2 |
| `PLANCK_DFT_DH_HESSIAN_PROBE_LOG` | **Only on `dft-dh-gradient-xc-kernel-rewrite`, not on `devel`.** Two things at once: (1) switches the double-hybrid Z-vector solve from GMRES to `DenseReference`, (2) runs the Hessian swap probe and **returns the swapped gradient instead of the normal one**. An empty value fails the gradient. The DH gradient is not build-gated, so this is live in every `planck-dft`. | `tests/dh_hessian_swap_audit.py`, `tests/test_dh_hessian_swap_audit.py`, `tests/dh_probe_gmres.cpp`, and `tests/test_dh_o3_zvector.py`, which asserts the `getenv` string is present in the source | E3 |
| `PLANCK_CC_T3_TIME` | Timing probe. **Unreachable:** it sits in the `use_generated_triples_kernel` branch of `run_tensor_rccsdt_impl`, whose only caller passes `false, false` (`tensor_backend.cpp:3067`). | none | E4 |

Output-only, and allowed under the rule (no change planned):
`PLANCK_DFT_GRADIENT_DEBUG`, `PLANCK_DEBUG_GRID_ACC`, `PLANCK_DEBUG_RMP2_{TERMS,MATRICES,IMAT}`,
`PLANCK_DEBUG_DM2BUF`, `PLANCK_DEBUG_RHF_RESPONSE`, `PLANCK_PHASE_TIMING`, `PLANCK_CC_RANK_TIME`,
`PLANCK_DUMP_AO_INTEGRALS`, and `BASIS_PATH` (configuration).

## Steps, cheapest first

### E1 — delete `PLANCK_DFT_ALLOW_RS_WORKFLOWS` (trivial) — DONE 2026-10-12

Delete `dft_allow_unvalidated_range_separated_workflows()` and its branch in
`validate_workflow_support` (`src/dft/driver.cpp`), and drop the `env[...]` line from
`hse06_gradient_components.py`. Nobody uses the one workflow it really unlocks (RS TDDFT). When
RS TDDFT gets validated, add `LinearResponse` to the allowed `switch`, the same way the gradient
workflows were promoted.

*Verify:* the script's gradient output is unchanged, and an RS `LinearResponse` input is rejected
with the existing message.

### E2 — `PLANCK_RCCSDT_BACKEND` becomes an input keyword (small)

Add `rccsdt_backend {auto,determinant,tensor,optimized}` next to `cc_warm_start` in
`src/io/io.cpp` (default `auto` = today's selection), read it in
`parse_rccsdt_backend_override` (`src/post_hf/cc/ccsdt.cpp`), and delete the `getenv`.

**Trap:** `ch4_rccsdt_generated_sto3g` and `ch4_rccsdt_sto3g` share
`tests/inputs/regression/post_hf/ch4_rccsdt_sto3g.hfinp`, and that sharing has already broken one
case once (the `use_diis` flip, `vault/Status/Open Work.md`). The generated case gets its own
input file carrying the keyword. `lih_rccsdt_generated_sto3g` already has its own input.

Then drop the two `env` entries from `tests/regression_cases.json` and switch
`compare_rccsdt_backends.py` to rewrite the keyword. The runner's generic `env` support stays,
since the FCIQMC thread-count cases use it for `OMP_NUM_THREADS`; only update its comment, which
cites `PLANCK_RCCSDT_BACKEND` as the example.

*Verify:* both generated cases still print the routing line and pass, and `ch4_rccsdt_sto3g`
(hand-written) is unchanged.

### E3 — delete the DH Hessian swap probe (medium, mostly deletion)

**Status of the path (checked 2026-10-12): not on `devel`.** On `origin/devel` double hybrids are
single-point only, and neither the DH analytic gradient nor the probe exists. Both live only on
`dft-dh-gradient-xc-kernel-rewrite` (77 commits ahead). There, RKS DH gradient, geomopt, freq and
geomopt+freq are admitted by `validate_workflow_support`, and UKS DH gradient is admitted on the
qualified grid. **So E3 is a merge blocker for that branch, not a fix to shipped code:** merging
as-is would put an override into `devel` that replaces the returned gradient.

**What the override does today** (`compute_analytic_dh_gradient`, `src/dft/driver.cpp`), when
`PLANCK_DFT_DH_HESSIAN_PROBE_LOG` is set:
1. It switches the Z-vector solve from `MatrixFreeGMRES` to `DenseReference`.
2. It runs `run_dh_hessian_swap_probe`, writes a ledger, and **returns the swapped gradient
   instead of the normal one**.

**Why it can go rather than be gated.** The probe existed to justify promoting the matrix-free
GMRES response into the normal path. That promotion landed in `b4175570` ("promote checked
matrix-free response"), with acceptance recorded in the commit message: water and C1 H2O2
full-coordinate FD maxima 7.9e-9 to 1.7e-8 Ha/Bohr, all below 5e-8. The probe's comparison is
now covered at the operator level without the binary:
- `tests/dh_zvector.cpp` runs both `MatrixFreeGMRES` and `DenseReference` on the same system and
  requires `|z_gmres - z_dense| < 1e-11`, plus restart, exhaustion, zero-RHS and failure cases.
- `tests/dft_coulomb_response.cpp` uses `DenseReference` as its O3 oracle.

So the dense backend does not need an input keyword either. It is a test-only reference, already
reachable programmatically through `DHZVectorOptions`, which is the precedent this file's driver
already sets with `Options::udh_gradient_validation` (no CLI or input exposure).

**Deletion list:**
- `src/dft/driver.cpp`: both `getenv("PLANCK_DFT_DH_HESSIAN_PROBE_LOG")` sites, the
  `z_options` dense switch, the 30-line probe block, and `#include "dh_hessian_probe.h"`. The
  normal path then always uses the default GMRES options.
- `src/dft/dh_hessian_probe.{h,cpp}` (294 lines) and `src/dft/dh_probe_gmres.h`, plus their three
  entries in the `CMakeLists.txt` source list.
- `tests/dh_probe_gmres.cpp` and its `planck-dh-probe-gmres` CTest. It only checks the `Probe::`
  alias onto `DFT::Response::gmres`, which `dh_zvector.cpp` exercises directly.
- `tests/dh_hessian_swap_audit.py`, `tests/test_dh_hessian_swap_audit.py`, and the
  `planck-dh-hessian-swap-runner` CTest.
- `tests/test_dh_o3_zvector.py`: drop the `dh_probe_gmres.h` bridge check, and change
  `test_production_defaults_iterative_and_probe_explicitly_dense` to assert the opposite: no
  `getenv` and no `DenseReference` inside `compute_analytic_dh_gradient`. The same no-`getenv`
  assertion already exists for the UDH path in `test_udh_molecular_audit.py`.

`KsOrbitalHessianInputs` survives: `driver.cpp:2129` uses it outside the probe.

**Correction to the earlier version of this scope:** `test_dh_hessian_swap_audit.py` does not
set the nonexistent `PLANCK_DFT_DH_LITERAL_EQ47_RHS`. It asserts that the audit script *scrubs*
it from the environment. That guard goes away with the file.

**If the probe must be kept instead:** move the block and both probe files behind a CMake option
`PLANCK_DFT_DH_PROBES` (default OFF), keep the log path as an env var inside that build, and
switch `dh_hessian_swap_audit.py` to fail with "rebuild with -DPLANCK_DFT_DH_PROBES=ON" when the
ledger marker is missing. That costs more code for a comparison the unit tests already make.

**Decision needed from the DH gradient owner:** delete (recommended) or gate.

*Verify:*
- Without the env var, a DH gradient on the retained fixtures
  (`tests/inputs/exploratory/dh_gradient/`) is byte-identical before and after, since the normal
  path was already GMRES.
- `planck-dh-pt2-amplitude-density` (which links `tests/dh_zvector.cpp`), `planck-dft-coulomb-response`, `planck-dh-o3-zvector-contract` and the remaining DH CTests pass.
- `grep getenv` inside `compute_analytic_dh_gradient` is empty.

### E4 — delete the dead generated-kernel branch in `tensor_backend.cpp` (small, larger diff) — DONE 2026-10-12

Landed as about -230 lines. It also removed `tensor_backend.cpp`'s includes of
`generated/cc/ccsdt_planck_generated.cpp`, `generated/cc/ccsd_spinorbital_warm_start.inc`
and `generated_arbitrary_runtime.h`, plus the per-iteration `rebind_physicist` transform the
hand-written path computed and never used. Two follow-ons this exposed:
- The `ccgen-spinorbital-warm-start` CMake target and
  `python/generate_spinorbital_ccsd_warm_start.py` generated a file nothing included. **Removed
  2026-10-12.** The RCC kernel step also no longer emits the plain rank-2/3
  `<method>_planck_generated.cpp`, which had no consumer. Rank-2/3 methods are passed to the
  generator only for their `_arbitrary_` companions under `PLANCK_CC_ARBITRARY_LOWER_RANKS`, so
  a default `PLANCK_CC_MAXORDER=3` build runs no RCC kernel generation at all.
- `tests/dressed_kernel_equivalence.py` requires the marker `kernels=ccgen-generated`, which this
  branch printed. The branch was already unreachable before E4, so that gate could not pass
  before this change either. It belongs to the retired dressed route.

`run_tensor_rccsdt_impl(..., use_generated_warm_start, use_generated_triples_kernel)` is only
ever called with `false, false`. The branch was retired deliberately (the representation mismatch
recorded at `tensor_backend.cpp:3074`). Remove both parameters, the generated branch, and the
`PLANCK_CC_T3_TIME` probe inside it.

*Verify:* `ch4_rccsdt_sto3g` passes with an unchanged energy and iteration count, and the
generated route (through `rccgen.cpp`, which this does not touch) still passes its two cases.

## What would make this an answer

When E2 and E3 land, rewrite this file under the same question, keeping the rule, the final list
of allowed output-only variables, the shared-CH4-input trap, and the precedent that test-only
solver choices go through a programmatic `Options` field rather than an env var or keyword.
Drop the step list.
