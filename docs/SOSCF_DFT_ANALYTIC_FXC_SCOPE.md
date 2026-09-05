# Scope: analytic XC second-derivative (fxc) for DFT SOSCF

**Scope for in-flight work. Not started.** Follow-on to
`docs/SOSCF_UHF_DFT_SCOPE.md` (D1), which decided DFT SOSCF's first path is
the finite-difference XC kernel builders (option (a): a correctness-only,
small-system reference, buildable now with no new libxc wiring). This doc
scopes option (b), the real production accelerator: an **analytic** XC
second derivative (`fxc`) that avoids the `O(n_occ · n_virt)` grid cost the
FD-kernel path pays every SOSCF iteration.

This is deliberately written after (a) exists rather than before it, for
the reason D1 gave: (a) is the numerical oracle (b) must be checked against.
**Do not derive or trust the analytic Hessian without it.**

## Why this is research, not wiring — confirmed by reading the actual libxc signatures

The vendored libxc header
(`src/external/libxc/install/include/xc.h`) exposes the full derivative
ladder:

```c
void xc_lda_fxc (const xc_func_type *p, size_t np, const double *rho,
                  double *v2rho2);
void xc_gga_fxc (const xc_func_type *p, size_t np, const double *rho,
                  const double *sigma,
                  double *v2rho2, double *v2rhosigma, double *v2sigma2);
```

Planck's wrapper (`src/dft/base/wrapper.h`) only ever calls
`xc_lda_exc_vxc` / `xc_gga_exc_vxc` (the `exc`/`vrho`/`vsigma` family — energy
and first derivative). No `_fxc` call exists anywhere in the tree
(`grep -rn "_fxc" src/` returns nothing outside libxc's own vendored
sources). This confirms the corrected framing in
`SOSCF_UHF_DFT_SCOPE.md`: wiring `fxc` in is not a thin additional call next
to the existing `evaluate_gga_exc_vxc`, because the **GGA case is
qualitatively harder than LDA**, for a structural reason visible directly in
the signature: `xc_gga_fxc` returns three second-derivative arrays
(`v2rho2`, `v2rhosigma`, `v2sigma2`), not one. `v2rhosigma` and `v2sigma2`
couple the density response to the **density-gradient response** at each
grid point — there is no analogue of this in the first-derivative KS matrix
assembly's `vrho`-only LDA term, and even GGA's own first-derivative
`vsigma` term (see `src/dft/ks_matrix.cpp:97-179`,
`assemble_xc_matrix`) is already a rank-2 (AO-times-gradient) contraction,
not the simple rank-1 (AO-times-AO) LDA update. The analytic Hessian-vector
product needs the **second**-derivative analogue of that same contraction —
a genuinely new piece of algebra, not a copy-paste of the existing
`vsigma` term with an extra factor.

## What already exists and can be reused unchanged

| Piece | Where | Reusable as-is? |
|---|---|---|
| Grid construction, AO/AO-gradient evaluation | `src/dft/base/grid.h`, `src/dft/ao_grid.h` | Yes — same grid, same AO values, no new evaluation needed |
| Density (and density-gradient, for GGA) on the grid from a trial perturbed density | `evaluate_density_on_grid` family, `src/dft/xc_grid.cpp` | Yes — this is exactly the machinery that turns a trial `δP` into `δρ`/`δ∇ρ` at each grid point, needed as the "input" side of the Hessian-vector product |
| `Functional` wrapper class, functional selection/combination (exchange + correlation, hybrids) | `src/dft/base/wrapper.h` | Structure yes, contents no — needs new `evaluate_lda_fxc` / `evaluate_gga_fxc` methods added alongside the existing `evaluate_*_exc_vxc` ones, same shape (chunked, `#pragma omp parallel for schedule(static)`, pointwise map so thread-count-invariant by the same argument the existing methods' comments already give) |
| The rank-1/rank-2 AO contraction pattern for the first-derivative `V_xc` | `assemble_xc_matrix`, `src/dft/ks_matrix.cpp` | Pattern yes, not the code — the Hessian-vector product's output contraction is structurally similar (AO products weighted by a per-point scalar/vector kernel) but the *input* is a trial density's response, not the ground-state density, and the kernel is the second derivative, not the first |
| The finite-difference oracle (D1's own deliverable) | `build_unrestricted_xc_kernel_blocks`, `src/dft/driver.cpp` | Yes, unmodified — this is what verifies the new analytic path, never touched by this work |
| `ResponseExcitationSpace` (arbitrary occ-virt subset, not TDDFT-specific) | `src/dft/driver.cpp:883` | Yes — same type the FD-kernel path already uses; the analytic path should build the same-shaped kernel block matrix so both paths are drop-in interchangeable behind one interface |

**Nothing here is a rewrite.** The grid, the AO evaluation, the functional
selection/combination logic, and the excitation-space bookkeeping are all
shared with the ground-state SCF/TDDFT machinery already in the tree. The
new work is entirely in one place: a second-derivative XC kernel evaluator
plus the contraction that turns it into a Hessian-vector product.

## Steps

Ordered so the cheapest, most isolated correctness check happens before any
SCF-loop wiring — mirroring how U1 (Track 1) built and verified
`build_uhf_cphf_matrix` in complete isolation from `run_uhf` before U2 ever
touched the SCF loop.

### F1 — wire `xc_lda_fxc` / `xc_gga_fxc` into the wrapper, unit-verify against `exc_vxc`'s own finite difference (~M) — DONE

Add `evaluate_lda_fxc` / `evaluate_gga_fxc` to `DFT::XC::Functional`
(`src/dft/base/wrapper.h`), mirroring the existing
`evaluate_lda_exc_vxc` / `evaluate_gga_exc_vxc` methods exactly: same
chunked/threaded shape, same `is_lda_like()`/`is_gga_like()` guards, same
error messages adapted to the new arrays (`v2rho2`, plus `v2rhosigma` /
`v2sigma2` for GGA).

**Do not trust libxc's own second derivative by construction.** Before this
touches any grid or density machinery, unit-test it the cheapest possible
way: for a single point (or a handful of points spanning a plausible ρ/σ
range for the functional under test), finite-difference libxc's own
first-derivative `vrho`/`vsigma` output with respect to `rho` (central
difference, small `h`, the same three-step-size pattern
(`1e-2`, `1e-3`, `1e-4`) RHF's and UHF's own FD probes used) and compare
against `v2rho2` from the new `_fxc` call. This isolates "does libxc's own
analytic second derivative agree with its own first derivative's finite
difference" from every other question this scope raises — if libxc's
`_fxc` output doesn't reproduce a bare finite difference of its own `vrho`,
nothing built on top of it can be trusted, and the bug is not in Planck's
code at all.

*Verify:* `v2rho2` (LDA) matches `d(vrho)/d(rho)` via FD to the precision
the step sizes allow, on at least Slater (LDA exchange) and one GGA
functional (e.g. PBE) actually used elsewhere in the tree, both spin-
unpolarized and spin-polarized. For GGA, additionally check `v2rhosigma`
against `d(vrho)/d(sigma)` (equivalently `d(vsigma)/d(rho)` — libxc's own
`v2rhosigma` should make these agree, which is itself a check worth
running) and `v2sigma2` against `d(vsigma)/d(sigma)`.

**If this disagrees, stop.** Nothing downstream can be verified against
D1's grid-level FD oracle if the per-point kernel itself is wrong.

**Landed as `evaluate_lda_fxc` / `evaluate_gga_fxc` in
`DFT::XC::Functional` (`src/dft/base/wrapper.h`)**, plus a new isolated
ctest, `planck-dft-fxc-selfcheck` (`tests/dft_fxc_selfcheck.cpp`).

**libxc's per-point second-derivative component counts are NOT
`spin_components()`/`sigma_components()`** — confirmed by reading libxc's
own `internal_counters_set_lda`/`_gga` (`src/external/libxc/src/libxc/src/
util.c`) rather than guessing from the first-derivative convention.
Unpolarized: `v2rho2 = v2rhosigma = v2sigma2 = 1`. Polarized:
`v2rho2 = 3` (the independent `aa`/`ab`/`bb` pairs, not `nspin = 2`),
`v2rhosigma = 6` (2 rho-channels × 3 sigma-channels), `v2sigma2 = 6` (the
independent sigma-sigma pairs). New `v2rho2_components()` /
`v2rhosigma_components()` / `v2sigma2_components()` accessors carry these,
mirroring `spin_components()`/`sigma_components()`'s existing shape.
Getting this wrong silently corrupts every downstream read (verified via
mutation: an off-by-one polarized `v2rho2` component count crashes rather
than passing quietly, since the test's own size assertion catches it
before any value comparison runs).

**Result: libxc's analytic second derivative agrees with a finite
difference of its own first derivative, on every functional and every
component tested, with no discrepancy at any step size.** Tested: Slater
exchange (`lda_x`, LDA, unpolarized and polarized) and PW92 correlation
(`lda_c_pw`, LDA, polarized) for `v2rho2`; PBE (GGA, unpolarized) for
`v2rho2`/`v2rhosigma`/`v2sigma2`, including the `v2rhosigma` mixed-partial
equivalence (`d(vrho)/d(sigma) == d(vsigma)/d(rho)`) checked as two
independent finite differences rather than assumed from libxc's naming.
**Nothing in this scope was actually wrong** — F1's job was to rule that
out before anything else gets built on top of it, and it did.

**A fixture choice mattered and is recorded because it nearly weakened the
polarized cross-spin check.** The first polarized test used `lda_x`
(Slater exchange), whose `v2rho2[ab]` (cross-spin) component is genuinely
~0 at every density (exchange has no cross-spin coupling by construction)
— a real physical fact, not a bug, but it meant an `aa`/`ab`/`bb` index
swap involving the `ab` slot specifically would have been hard to detect
against a near-zero expected value. Added `lda_c_pw` (LDA correlation),
whose cross-spin term is large and distinct from `aa`/`bb`
(measured at ρ_α=0.10, ρ_β=0.06: `aa=0.104, ab=-0.289, bb=0.327`), which
gives the same check real power. **Mutation-verified**: an `aa`↔`bb`
component swap in `evaluate_lda_fxc`'s output and a `v2rhosigma`↔`v2sigma2`
argument swap in `evaluate_gga_fxc` are both caught immediately (the latter
fails all three GGA sub-checks, at every step size); disabling the
`is_lda_like()` family guard is caught by the cross-family test. All
mutations reverted after verification.

**One genuinely separate finding, out of scope for F1 and left unfixed:**
`planck-dft` currently fails to LINK on a clean build (`make planck-dft`),
independent of anything in this scope. `CMakeLists.txt`'s `POSTHF_DFT_SRC`
deliberately excludes the CASSCF sources (`orbital.cpp`, `aug-hessian.cpp`,
`aug-hessian-orbital.cpp`), with an in-tree comment explaining that
exclusion — but `src/scf/scf.cpp` (built into both `hartree-fock` and
`planck-dft`) now calls `HartreeFock::Correlation::CASSCF::
apply_orbital_rotation` and `solve_augmented_hessian` unconditionally from
the RHF SOSCF branch, landed in `docs/SOSCF.md`'s work (already merged to
`devel`) before this scope's UHF SOSCF work (Track 1) added a second,
unrelated call site. **This blocks building `planck-dft` at all right now**
— D2 cannot be verified against a real KS loop until it is fixed. Verified
pre-existing (reproduced on a clean stash of this session's changes) and
NOT caused by F1's own wrapper-only edits (F1 touches nothing `planck-dft`
links). `F1`'s own verification used a header-only standalone ctest
(`planck-dft-fxc-selfcheck`) specifically because it does not need
`planck-dft` to link — this is why F1 could still be completed and
verified despite the gap. **Root cause confirmed, not assumed:** `grep -rn "run_rhf\|run_uhf" src/dft/`
returns nothing — the DFT driver never calls into RHF/UHF at all. `scf.cpp`
reaches `planck-dft` only because `CMakeLists.txt`'s `SCF_SRC` glob
(`file(GLOB SCF_SRC ... ${SRC_DIR}/scf/*.cpp)`, line ~292) is added to
`planck-dft`'s source list wholesale (line ~491), for the genuinely shared
helpers in that directory (SAD guess, DIIS, `working_state.cpp`) that the
KS loop does reuse — `run_rhf`/`run_uhf` themselves are simply dead code in
that binary, pulled in as an unused side effect of globbing the whole
directory rather than the specific files needed.

**FIXED.** Added `casscf/orbital.{h,cpp}`, `casscf/aug-hessian.{h,cpp}`, and
`casscf/aug-hessian-orbital.{h,cpp}` to `POSTHF_DFT_SRC`
(`CMakeLists.txt`). Checked their own `#include`s first to confirm this
pulls in nothing else: `orbital.cpp` only needs `integrals/os.h`,
`post_hf/integrals.h`, and `post_hf/ri/ri_eri.h` (all three already in
`POSTHF_DFT_SRC`); `aug-hessian.cpp` and `aug-hessian-orbital.cpp` have no
`post_hf`-internal dependencies at all. `planck-mpi` (built with
`BUILD_MPI=ON`, off by default) was never affected — it already links the
full `POSTHF_SRC` glob, CASSCF included.

Verified beyond "it links": `planck-dft` now runs a real calculation
end-to-end (H2/STO-3G B3LYP, `-1.1654185791` Eh, converged), the full
smoke suite (35/35) and extended suite (114/114, 5 pre-existing skips)
both pass, and `hartree-fock` still builds unaffected.

### F2 — density-response-on-grid: reuse, don't rebuild (~S)

Confirm (not assume) that the existing density-evaluation machinery
(`src/dft/xc_grid.cpp`) already produces exactly what a Hessian-vector
product needs when fed a **response** density `δP` (built from a trial
orbital rotation, the same `Ca_virt · x · Ca_occᵀ + h.c.` construction U1
used for UHF) instead of the ground-state `P`: `δρ` at each grid point
(trivially, since density-on-grid is linear in `P`), and for GGA, `δ∇ρ`
(also linear, same evaluation path). This should require zero new code —
if it does, that is itself a finding worth recording (it would mean the
existing density-on-grid evaluator has a non-linearity or an assumption
that breaks for a traceless/response density, which would be surprising and
worth stopping on).

*Verify:* `δρ` computed by feeding `δP` through the existing evaluator
equals `ρ(P + εδP) - ρ(P)` divided by `ε` in the limit `ε → 0`, for both
LDA and GGA density fields. This is a linearity check, not a new derivative
— cheap, and it rules out a whole class of subtle bugs before the real
Hessian contraction is built on top of it.

### F3 — the Hessian-vector product itself (~L, the actual research)

This is the step the doc's framing calls "closer in kind to deriving the
RHF Hessian than to writing the RHF SOSCF callbacks." Build the contraction
that takes a trial rotation `x` (packed the same `(a,i)` way U1/U2 do for
UHF, or the RHF single-channel way for RKS) and returns `H·x`:

1. Build `δP` from `x` (as in F2).
2. Evaluate `δρ` (and `δ∇ρ` for GGA) on the grid from `δP` (F2's linear
   map).
3. Contract the analytic second-derivative kernel from F1
   (`v2rho2`, `v2rhosigma`, `v2sigma2`) against `δρ`/`δ∇ρ` to produce the
   **induced XC potential** `δV_xc` at each grid point — the genuinely new
   algebra, structurally the second-order term in a Taylor expansion of
   `V_xc[ρ + δρ]` around `ρ`:
   ```
   δV_xc(r) = v2rho2(r)·δρ(r) + v2rhosigma(r)·[2∇ρ(r)·δ∇ρ(r)]   (LDA term + GGA cross term)
            + [GGA-only terms coupling δ∇ρ through v2rhosigma/v2sigma2 the same
               way the existing vsigma term in assemble_xc_matrix couples ∇ρ]
   ```
   The exact GGA form needs to be derived carefully by differentiating the
   existing first-derivative `assemble_xc_matrix` gradient term
   (`src/dft/ks_matrix.cpp:97-179`) with respect to the density one more
   time — do not guess it from a paper's notation without checking it
   reduces to the existing `vsigma` term's own structure at zeroth order.
4. Project `δV_xc` back into the `(a,i)` MO block the same way
   `assemble_xc_matrix`'s output is projected in the KS build:
   `H·x = C_occᵀ · δV_xc(AO basis) · C_virt`.

**Verify against D1's oracle, not against a hand-derivation alone.** On a
small closed-shell system, compute `H·x` for a handful of `x` directions
both via this analytic path and via the existing
`build_unrestricted_xc_kernel_blocks` FD path (feeding it a
`ResponseExcitationSpace` covering the same directions). They must agree to
the precision the FD path's own step size allows. **This is the load-
bearing check for the entire step — RHF SOSCF's own history is the direct
precedent for why**: a gradient/Hessian pairing that looks individually
correct on each side (right functional form, right units) was still
silently wrong by a factor of 2/4 until checked directly against the true
`E(κ)`. Here the FD-kernel oracle plays the role the finite-difference-of-
`E(κ)` probe played for RHF/UHF — do not skip straight to comparing energies
after a full SOSCF run; a wrong Hessian that happens to still converge (to
a linear rate, say) can hide for a long time, exactly as pure-unbounded RHF
SOSCF's own scale-mismatch bug did before it was checked directly.

**If this disagrees with the FD oracle, stop.** Do not wire a Hessian into
the SOSCF loop that has not been checked against the FD reference D1 built
for exactly this purpose.

### F4 — RKS wiring, mirroring D2 (~M, after F3's Hessian-vector product is verified)

Wire the verified `H·x` (and the matching analytic gradient, which already
exists as the ordinary KS-build `V_xc` projected the same way
`assemble_xc_matrix`'s output is used elsewhere — no new derivation needed
for the gradient side, only the Hessian side is new) into the RKS SOSCF
branch D2 built for the FD-kernel path. Same insertion point, same
`C_soscf_prev`/`eps_soscf_prev` state, same augmented-Hessian solver
(`solve_augmented_hessian`) — this step is now the mechanical wiring the
doc originally (wrongly) thought the whole DFT track would be, because the
hard part (F1-F3) is done by the time this step starts.

*Verify:* same-energy-as-DIIS check (D2's own verification), **and**
same-energy-as-the-FD-kernel-path's-own-SOSCF-run (D2 already built and
verified this path with the FD kernel; the analytic path replacing it
should converge to the identical energy on the same input, ideally in
comparable or fewer iterations at no worse wall-clock cost — the actual
speed claim this whole scope exists to deliver).

### F5 — UKS, mirroring D3 (~M, after F4)

Repeat the RKS→UKS generalization for the analytic path exactly as D3 does
for the FD-kernel path: separate α/β second-derivative kernel blocks
(`v2rho2` already comes out spin-resolved from libxc when the functional is
initialized `Polarized`, so this is substituting the polarized `_fxc` call
into the same per-spin structure U1/U2 already established for UHF's own
CPHF matrix — no new spin-coupling derivation needed beyond what libxc
already returns).

### F6 — cost measurement, the actual point of doing F1-F5 at all (~S)

Measure wall-clock per SOSCF iteration for the analytic path vs the
FD-kernel path (D1) vs plain DIIS, on at least one system large enough that
DIIS's own iteration count is the bottleneck (the same large-`nb` regime
`docs/SOSCF.md`'s unreproduced `scale.json` ladder was originally
motivated by). **This is the number the whole DFT SOSCF track was scoped
around** — if the analytic path is not meaningfully cheaper per iteration
than the FD-kernel path (it should be `O(1)` grid passes instead of
`O(n_occ · n_virt)`, but only measurement settles whether the `_fxc`
evaluation itself, or the new Hessian-vector contraction, has hidden costs
that erode that advantage), say so explicitly rather than assuming the
`O(n_occ · n_virt) → O(1)` asymptotic argument transfers to a real wall-
clock win at the sizes that matter.

## What this must not do

- **Do not skip F1's isolated libxc self-consistency check.** It is the
  cheapest possible falsification point in the whole scope and costs
  almost nothing to run before any grid or SCF-loop code exists.
- **Do not derive the GGA Hessian-vector contraction from a paper's
  notation without checking it reduces to the existing `vsigma` term's
  structure.** The existing first-derivative GGA contraction in
  `assemble_xc_matrix` is the one piece of ground truth in this codebase
  for what a correct gradient-coupling contraction looks like in Planck's
  own conventions (AO basis, grid weights, spin handling) — use it as the
  reference to differentiate, not an external formula sheet alone.
- **Do not trust F3's Hessian-vector product from algebra alone.** The
  FD-kernel oracle (D1) exists specifically so this step has an independent
  numerical check, the same role RHF/UHF's own `E(κ)` finite-difference
  probes played. Skipping straight to "does the full SOSCF run reach the
  same energy" risks the same silent-scale-mismatch failure mode RHF SOSCF
  already hit once.
- **Do not build a second augmented-Hessian solver, a second Cayley-
  rotation helper, or a second SCF loop.** F4/F5 route through the exact
  same `solve_augmented_hessian` / `apply_orbital_rotation` /
  `run_ks`-insertion-point machinery D2/D3 build for the FD-kernel path;
  only the Hessian-vector-product callback differs between the two.
- **Do not claim a wall-clock speedup without F6's measurement.** The
  `O(n_occ · n_virt) → O(1)` grid-pass argument is a plausible asymptotic
  claim, not a measured one, until F6 runs.

## Key code locations

| what | where |
|---|---|
| libxc's `_fxc` family (not yet called anywhere in Planck) | `src/external/libxc/install/include/xc.h`, `xc_lda_fxc` / `xc_gga_fxc` |
| Planck's libxc wrapper (first-derivative only today) | `src/dft/base/wrapper.h`, `evaluate_lda_exc_vxc` / `evaluate_gga_exc_vxc` — new `evaluate_lda_fxc` / `evaluate_gga_fxc` go here |
| The first-derivative KS-matrix contraction to differentiate for F3's GGA term | `assemble_xc_matrix`, `src/dft/ks_matrix.cpp:97-179` |
| Density-on-grid evaluation (reused unchanged for the response density) | `src/dft/xc_grid.cpp`, the `evaluate_density_on_grid` family |
| `XCGridEvaluation` (where new `v2rho2`/`v2rhosigma`/`v2sigma2` fields would live) | `src/dft/xc_grid.h:63` |
| The FD-kernel oracle this whole path must be checked against | `build_unrestricted_xc_kernel_blocks` / `build_closed_shell_xc_kernel_blocks`, `src/dft/driver.cpp` |
| `ResponseExcitationSpace` (shared occ-virt-subset type, reused unchanged) | `src/dft/driver.cpp:883` |
| The RKS/UKS SOSCF insertion points this wiring targets (D2/D3, once built) | `src/dft/driver.cpp`, the `!unrestricted` KS loop branch |
| The generic CIAH solver (reuse for F4/F5, do not rewrite) | `solve_augmented_hessian`, `src/post_hf/casscf/aug-hessian.h` |
| The FD-verification-against-truth pattern this scope's F1/F3 both reuse | the `PLANCK_SOSCF_FD_CHECK` probes in RHF's and UHF's `run_rhf`/`run_uhf` branches, `src/scf/scf.cpp` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`
once any of F1-F6 lands. Parent scope: `docs/SOSCF_UHF_DFT_SCOPE.md`.
