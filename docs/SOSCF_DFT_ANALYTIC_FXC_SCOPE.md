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

### F2 — density-response-on-grid: reuse, don't rebuild (~S) — DONE

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

**Confirmed: zero new code needed.** `evaluate_density_on_grid`
(`src/dft/xc_grid.cpp`) computes `rho(r) = phi(r)ᵀ P phi(r)` and
`grad_x(r) = 2·phi(r)ᵀ P ∇phi_x(r)` — both are exact quadratic forms in the
matrix `P` for fixed `phi(r)`, hence exactly linear in `P`, not merely
linear in the small-`ε` limit. New isolated ctest
(`planck-dft-density-response-linearity`,
`tests/dft_density_response_linearity.cpp`) verifies this against a
synthetic AO grid (random `values`/`grad_{x,y,z}`, no real basis/molecule
needed — F2 only tests the evaluator's linearity in `P`, not physical
correctness) for both the restricted (`evaluate_density_on_grid(P)`) and
unrestricted (`evaluate_density_on_grid(Pa, Pb)`) overloads, using the same
symmetric `δP` shape (`dm1a + dm1a.transpose()`) U1's real trial-orbital-
rotation density already produces. All checks pass.

**A real gap was found and fixed in the test design, not the production
code: a finite-difference self-consistency check cannot see a uniform
scale bug, because both sides of the comparison go through the same
(possibly wrong) formula.** Mutation-verified directly: scaling the
`grad_x` term's leading factor from `2.0` to `1.5` in
`evaluate_density_channel` passed every FD-based linearity check
unchanged (the `+h`/`-h` evaluations and the standalone `δP`-only
evaluation all use the same mutated formula, so the comparison stays
internally consistent even though the *absolute* value is wrong). Fixed
by adding an independent check
(`check_against_hand_computed_reference`) that hand-computes
`phi(r)ᵀ P phi(r)` and `2·phi(r)ᵀ P ∇phi_x(r)` directly from the raw AO
arrays in the test itself, bypassing `evaluate_density_channel`'s own
contraction order entirely — this caught the same mutation immediately.
A second mutation (a spurious `+0.01·rho²` term, a genuine non-linearity
rather than a uniform scale) was caught by nearly every check, including
the FD-based ones, confirming those checks are not vacuous in general —
they specifically cannot see a *uniform, self-consistent* scale error,
which is exactly why the independent reference check was needed as a
second, structurally different verification path rather than a
redundant one. Both mutations reverted after verification.

**This finding generalizes to F3.** The Hessian-vector product's own
verification (against D1's FD-kernel oracle) has the same shape as an
internal-consistency check only if the FD-kernel oracle and the analytic
path could ever share a common upstream bug; they do not (D1's oracle
perturbs the density and re-evaluates the full XC potential from scratch,
while F3's analytic path contracts a second-derivative kernel directly),
so F3's planned verification is already structurally independent in the
way this finding says is required — worth stating explicitly rather than
assuming the parallel is safe.

### F3 — the Hessian-vector product itself (~L, the actual research)

This is the step the doc's framing calls "closer in kind to deriving the
RHF Hessian than to writing the RHF SOSCF callbacks." Build the contraction
that takes a trial rotation `x` (packed the same `(a,i)` way U1/U2 do for
UHF, or the RHF single-channel way for RKS) and returns `H·x`.

**Broken into five sub-steps (F3.1–F3.5) rather than attempted as one
piece**, each independently buildable and independently verified against
D1's FD oracle before the next is attempted — the same discipline U1→U2
used (prove the isolated piece correct before composing it into anything
larger). The natural fault lines are the same ones F1 already found
matter: LDA is strictly simpler than GGA (no gradient coupling at all),
and unpolarized is simpler than polarized (no cross-spin terms). Each step
below produces a real, checkable intermediate — never "trust the algebra
and find out at the end whether the whole thing works."

| Step | Adds | New algebra | Verifies against |
|---|---|---|---|
| F3.1 | LDA, unpolarized `δV_xc` | `v2rho2·δρ` only | FD oracle, LDA unpolarized |
| F3.2 | LDA, polarized `δV_xc` | spin-resolved `v2rho2` (aa/ab/bb) | FD oracle, LDA polarized |
| F3.3 | GGA, unpolarized `δV_xc` | `v2rhosigma`/`v2sigma2` gradient coupling | FD oracle, GGA unpolarized |
| F3.4 | GGA, polarized `δV_xc` | cross-spin gradient coupling | FD oracle, GGA polarized |
| F3.5 | MO projection + `(a,i)` packing | none (pure plumbing) | FD oracle, full `H·x` in packed form |

Each step's own verification is a closed loop: build `δV_xc` (or, for
F3.5, the packed `H·x`) both the new way and via the FD oracle on the
same `x` direction(s), and require agreement before moving on. **If any
step disagrees with the oracle, stop there** — do not attempt the next,
harder case on top of an unverified simpler one.

#### What F3 needs from the D series — nothing (confirmed by reading the call site, not assumed)

**None of F3.1–F3.5 depends on D2, D3, or D4 landing.** The existing
FD-kernel oracle (`build_unrestricted_xc_kernel_blocks` /
`build_closed_shell_xc_kernel_blocks`, `src/dft/driver.cpp`) is TDDFT
machinery that already runs entirely **after** a converged SCF, using
whatever ground-state density the `Calculator` already holds
(`calculator._info._scf.alpha.density` / `.beta.density`) and a
`PreparedSystem` built once via the existing standalone
`DFT::Driver::prepare(calculator, options)` — confirmed by reading the
real TDDFT call site (`src/dft/driver.cpp`, the `kxc_blocks` construction
inside the linear-response block), which never touches `run_rhf`/`run_uhf`
or any SOSCF-window state. Both are usable today, before D2/D3/D4 write a
single line, because TDDFT is already a landed, working feature that calls
this exact function.

Concretely, every one of F3.1–F3.5 needs only:
- a converged DFT single-point calculation (any existing regression case
  will do — `h2_dft_pbe_sto3g` or `h2_dft_b3lyp_sto3g` are already in the
  suite),
- the `PreparedSystem` and `ResponseExcitationSpace` that calculation
  already builds for TDDFT-shaped work, spanning a handful of `(i,a)`
  directions rather than a full TDDFT root search,
- and F1's new `evaluate_lda_fxc`/`evaluate_gga_fxc` plus F2's confirmed-
  linear density evaluator.

None of that is gated on the SOSCF loop itself. **D2 (RKS SOSCF insertion
point) and D3 (UKS generalization) are F4/F5's prerequisites, not F3's** —
they wire the *verified* Hessian into the running SCF loop, which is a
separate, later concern from proving the Hessian is correct in the first
place. D1 is the only D-series item F3 (and this whole doc) already
depends on, and it is done.

#### F3.1 — LDA, unpolarized: `δV_xc = v2rho2·δρ` (~S) — DONE

The simplest possible case, with zero gradient-coupling algebra: for an
LDA functional the induced XC potential at each grid point is exactly
`δV_xc(r) = v2rho2(r)·δρ(r)` — no `∇ρ`, no `v2rhosigma`/`v2sigma2` at all,
since those only exist for GGA. This isolates "is the point-wise
second-derivative contraction itself correct" from every gradient-coupling
question F3.3/F3.4 raise.

Build:
1. `δP` from a trial `x` (F2's shape).
2. `δρ` on the grid via `evaluate_density_on_grid(δP)` (F2, unchanged).
3. `v2rho2` on the grid via `evaluate_lda_fxc` (F1, unchanged) evaluated
   at the GROUND-STATE density (not `δρ` — `v2rho2` is a property of the
   point where the Taylor expansion is centered, `ρ`, not of the
   perturbation).
4. `δV_xc(r) = v2rho2(r)·δρ(r)`, pointwise.
5. Project into AO basis the same way `assemble_xc_matrix`'s LDA-only
   term does (`accumulate_local_potential`'s `phi_μ φ_ν` rank-1 update,
   `src/dft/ks_matrix.cpp:79-93`) — reuse that pattern, do not invent a
   new AO contraction.

*Verify:* on an LDA-only functional (e.g. Slater exchange, `lda_x` — same
functional F1's own selfcheck used), compare `δV_xc` at every grid point
against the FD oracle's induced potential (`build_closed_shell_xc_kernel_blocks`
run with an `lda_x` exchange functional, LDA-only correlation set to a
matching zero/consistent choice) for a single `(i,a)` direction on a small
molecule (H2/STO-3G is enough — no gradient terms to exercise yet).
Agreement to the FD path's own step-size precision.

**If this disagrees, stop before attempting F3.2.** LDA unpolarized is the
floor — if the basic pointwise contraction is wrong here, every later step
inherits the same defect plus its own new algebra, compounding the search.

**Landed. Result: `δV_xc = v2rho2·δρ` matches the FD oracle to
`~2e-10`–`~3e-11` on two independent systems (H2/STO-3G, water/STO-3G) and
two different `(i,a)` directions**, once two real bugs — one in the plan,
one pre-existing and unrelated — were found and fixed.

**Prerequisite fix, not anticipated by the scope: `build_closed_shell_xc_kernel_blocks`
and `ResponseExcitationSpace` had internal linkage** (defined inside
`driver.cpp`'s file-spanning anonymous namespace), so the FD-kernel oracle
was not callable from outside `driver.cpp` at all — confirmed before
writing any verification code, not assumed from the doc's own plan. Fixed
by moving `ResponseExcitationSpace`, `ResponseEigenpair`,
`transition_density_matrix`, `evaluate_xc_matrix_from_spin_densities`,
`build_unrestricted_xc_kernel_blocks`, and `build_closed_shell_xc_kernel_blocks`
out of the anonymous namespace (declared in `driver.h`, bodies relocated to
just after the namespace closes, mirroring the existing
`evaluate_current_density_and_xc` placement) — a pure move, no logic
changed. Verified behavior-neutral: all 4 TDDFT regression cases and the
full core (71) + smoke (35) suites pass unchanged.

**A second, pre-existing and unrelated bug was found while building a
test fixture, before any comparison ran: `correlation vwn5` never
resolved to a real functional.** `driver.cpp` mapped `VWN5` to the libxc
name `"lda_c_vwn_5"`, which does not exist in libxc's functional table at
all (`src/external/libxc/install/include/xc_funcs.h` has `lda_c_vwn` for
VWN5 and separately-numbered `lda_c_vwn_{1,2,3,4}` for VWN1-4 — VWN5 alone
carries no numeric suffix). Zero regression coverage exercises VWN5, so
this had never been caught. Fixed the one-line mapping; verified the
functional now resolves and a real LDA (Slater + VWN5) single point
converges correctly.

**The real finding, once both of the above were out of the way: the
oracle's `.first` alone is not the correct RHF-orbital-rotation quantity —
`.first + .second` (same-spin plus cross-spin) is.** First measurement
disagreed by `1.989e-02` (~15% relative) using `oracle_blocks->first`
alone (the `aa` block only). Root-caused by reading the real TDDFT call
site rather than guessing: the singlet-response path there computes
`kxc = kxc_same + kxc_cross`, never `kxc_same` alone, because a real
orbital rotation in the closed-shell (RKS) formalism moves both spin
channels identically — exactly the singlet case, not the triplet (which
subtracts). Switching to `.first + .second` brought the two probed
directions to `2.6e-11` and `1.8e-10` agreement respectively. This is the
quantity F3.5's own MO-projection/packing step must carry forward — not
`.first` alone.

**One more thing verified along the way and worth recording as a negative
result: an apparent large second-direction disagreement (`7.9e-2`,
sign-flipped) was traced to a test-harness bug in the debug probe itself
(a stale, un-updated column index left over from probing a second
`(a,i)` pair), not a code defect** — re-checked by fixing the stale index
and re-running, which brought that direction to `1.8e-10` agreement too.
Recorded here because it is exactly the kind of false alarm this ladder's
"stop and investigate" discipline exists to filter correctly: a
disagreement is not evidence of a wrong formula until the comparison
harness itself has been checked for the more mundane failure mode first.

Landed as an env-gated debug probe (`PLANCK_FXC_F3_1_CHECK`, inert by
default, same shape as the `PLANCK_SOSCF_FD_CHECK` probes in
`src/scf/scf.cpp`) inside `run_ks_scf_scaffold`'s RKS convergence branch in
`src/dft/driver.cpp` — not yet a permanent ctest, since F3.1 is one step
in an ongoing ladder and the probe will need generalizing (spin, GGA) as
F3.2–F3.5 land; a committed regression gate is better added once the full
`H·x` (F3.5) exists to gate on, rather than one per sub-step.

#### F3.2 — LDA, polarized: spin-resolved `v2rho2` (~S, after F3.1) — DONE

Same contraction, but now `v2rho2` carries the `aa`/`ab`/`bb` packing F1
already measured and gated (3 components, not 2). The induced potential
per spin channel becomes
`δV_xc^α(r) = v2rho2_aa(r)·δρ_α(r) + v2rho2_ab(r)·δρ_β(r)` (and the
mirrored `β` form) — a real new step because it is the first place the
cross-spin coupling F1's own `lda_c_pw` mutation-check exercised actually
enters a Hessian, not just a diagnostic.

*Verify:* same shape as F3.1, on a functional with genuine cross-spin
coupling (`lda_c_pw`, matching F1's own choice for exactly this reason —
`lda_x`'s cross term is near-zero and would not meaningfully exercise
this step), spin-unrestricted trial `x` (both an alpha-only and a mixed
alpha/beta perturbation, since an alpha-only `x` cannot by itself catch a
bug in reading `v2rho2_ab`).

**Landed. Result: `δV_xc^α = v2rho2_aa·δρ_α + v2rho2_ab·δρ_β`
(and the mirrored β form) matches the FD oracle to `~4e-12`–`~8e-11`**
on two systems (triplet water/STO-3G with `lda_c_pw`, the water-cation
doublet/STO-3G) and both trial directions the doc's own spec asked for
(alpha-only and mixed), once three real issues — two genuine findings, one
of them non-obvious enough to be worth carrying forward, one test-harness
artifact — were found and resolved.

**Finding 1, real and non-obvious: the oracle's own `transition_density_matrix`
scales the symmetrized trial density by `0.5`
(`0.5·(occ·virtᵀ + virt·occᵀ)`), and F3.2's probe must build its own `δP`
with the SAME `0.5` factor to compare like with like.** First measurement
(unscaled `δP = C_virt·x·C_occᵀ + C_occ·x·C_virtᵀ`, matching F3.1's own
convention exactly) gave a clean, consistent factor of ~2 too high in
**both** probed directions (`alpha-only`: analytic `-0.0258175070` vs
oracle `-0.0129087535`; `mixed`: `-0.0239204728` vs `-0.0119602364` — the
ratio is `2.0000` to five figures in both, not noise). Scaling `δP` by
`0.5` to match `transition_density_matrix` brought both to `~1e-11`
agreement.

**This does not contradict F3.1 — it is a genuinely different oracle
composition, verified by re-running F3.1's own probe unchanged and
confirming it still agrees to `2.6e-11`.** F3.1's RKS oracle
(`build_closed_shell_xc_kernel_blocks`) duplicates one `space` into both
"source" slots and returns `.first + .second` — summing two `0.5`-scaled
blocks built from the identical construction restores the missing factor
of 2 by addition, which is why F3.1's own unscaled `δP` happened to match
without needing this scale fix. F3.2's UKS oracle
(`build_unrestricted_xc_kernel_blocks`) is read as a single block per spin
(`kxc_aa`, `kxc_ab`, never doubled by construction), so it has no
compensating sum and needs the `0.5` applied directly to `δP` instead.
**Both are correct; they are not the same convention, and F3.5 (packing
the general `H·x`, which must handle both the RKS-doubled and UKS-single-
block shapes) needs to get this right for each case rather than assuming
one scale rule covers both** — flagged explicitly so F3.5 does not
rediscover this by another factor-of-2 debugging pass.

**Finding 2, a real physical effect initially mistaken for a bug: the
first cross-spin probe direction `(i=0,a=0)` for beta was symmetry-
suppressed to `~8e-14`, not merely small.** Debugging the first
disagreement (before finding the `0.5` scale issue) included printing the
oracle's raw cross-spin kernel row directly; element `(0,0)` read
`7.87e-14` while adjacent elements in the same row read `9.5e-4` to
`2.9e-4` — a genuine near-exact cancellation for that specific
occupied-virtual pair on the triplet-water test geometry, not a
degenerate or unreachable direction chosen by mistake. Confirmed distinct
from a bug by checking neighboring elements were large: this is a
legitimate reason a *specific* direction can look wrong. Fixed by probing
`(i=0, a=2)` for beta instead (a nonzero column), landing the real,
non-degenerate cross-spin measurement.

**Finding 3, a pure test-harness bug, not a code defect: an intermediate
measurement with `alpha-only` and `mixed` giving bit-identical results was
traced to `Hx_oracle`'s cross-spin element being read from the
symmetry-suppressed `(0,0)` column (Finding 2, above) — the `scale_b`
weighting in the formula was always correct, but `kxc_ab` itself was
reading a near-zero value regardless of the trial direction, making
`scale_b`'s effect invisible.** Verified as resolved by the same
`(i=0,a=2)` fix: `alpha-only` and `mixed` now give genuinely different
`Hx_analytic`/`Hx_oracle` pairs, both self-consistent, confirming
`scale_b` was never the problem.

All 4 TDDFT regressions plus the full core (71) and smoke (35) suites
pass unchanged. Landed as `PLANCK_FXC_F3_2_CHECK`, same inert-by-default
shape as F3.1's probe, in the UKS convergence branch of
`run_ks_scf_scaffold`.

#### F3.3 — GGA, unpolarized: gradient coupling via `v2rhosigma`/`v2sigma2` (~M, after F3.2)

The genuinely new algebra. Derived symbolically (not guessed from a paper's
notation) by differentiating the existing first-derivative contraction in
`assemble_xc_matrix` (`src/dft/ks_matrix.cpp:171-188`, the unpolarized
branch) one more time with respect to the density. The existing
unpolarized potential term is, in AO-product form (`AA = φ_μφ_ν`,
`AG = φ_μ∇φ_ν + ∇φ_μφ_ν`, the vector the existing code builds via
`gradient_projection`):

```
V_xc = vrho·AA + 2·vsigma·(∇ρ·AG)
```

Taylor-expanding `vrho`/`vsigma` to first order via the mixed second
partials (`δ[vrho] = v2rho2·δρ + 2·v2rhosigma·(∇ρ·δ∇ρ)`,
`δ[vsigma] = v2rhosigma·δρ + 2·v2sigma2·(∇ρ·δ∇ρ)`, using
`δσ = 2∇ρ·δ∇ρ` since `σ = ∇ρ·∇ρ` is quadratic in `∇ρ`) and differentiating
the `∇ρ` argument inside `AG`'s own coupling gives **three structurally
distinct terms**, confirmed by symbolic differentiation rather than derived
by hand alone:

```
δV_xc = [v2rho2·δρ + 2·v2rhosigma·(∇ρ·δ∇ρ)] · AA                        (T1: coefficient x plain AO product)
      + 2·[v2rhosigma·δρ + 2·v2sigma2·(∇ρ·δ∇ρ)] · (∇ρ·AG)               (T2: coefficient x existing gradient-coupling vector)
      + 2·vsigma · (δ∇ρ·AG)                                             (T3: UNCHANGED ground-state vsigma x a NEW delta-gradient-coupling vector)
```

**T3 is easy to miss**: differentiating only the coefficients (`vrho`,
`vsigma`) and forgetting that `∇ρ` is itself an argument of the existing
`AG`-coupling factor drops this term entirely, silently truncating the
Taylor expansion. Broken into four sub-steps so a term-dropping bug shows
up as a specific, isolated disagreement rather than one combined mismatch
with no way to tell which piece is wrong — mirroring how F3 itself was
broken into F3.1–F3.5 for the same reason.

**This decomposition was checked independently before being written down,
not just re-derived and trusted.** Verified two ways: (1) a direct
multivariable chain-rule differentiation of
`V = vrho(ρ,σ)·AA + 2·vsigma(ρ,σ)·(∇ρ·AG)` with `σ=∇ρ·∇ρ` substituted
after differentiation (so the `∇ρ`-dependence of `σ` is captured
correctly rather than assumed), which reproduced the same three-term
structure; (2) a fully numeric check against a genuine, self-consistent
toy energy density `E(ρ,σ) = ρ³σ + ρ²σ² + ρσ` (chosen specifically so
`vrho=∂E/∂ρ` and `vsigma=∂E/∂σ` satisfy the Maxwell relation
`∂vrho/∂σ = ∂vsigma/∂ρ` a real XC functional's derivatives must satisfy —
an earlier attempt with two independently-picked, non-Maxwell-consistent
toy `vrho`/`vsigma` functions gave a **real, nonzero** disagreement
between the true chain-rule result and `T1+T2+T3`, which is not a flaw in
the T1/T2/T3 formula itself but a reminder that this decomposition
implicitly assumes `v2rhosigma` computed from `vrho`'s own `σ`-derivative
equals the one computed from `vsigma`'s `ρ`-derivative — exactly the
`v2rhosigma` equivalence F1's own selfcheck already verified holds for
real libxc functionals, and exactly what F3.3.4 re-exercises through the
full contraction). With a Maxwell-consistent toy functional, `T1+T2+T3`
matched the true numeric derivative to `~7e-18` (floating-point noise).

##### F3.3.1 — T1 alone: `[v2rho2·δρ + 2·v2rhosigma·(∇ρ·δ∇ρ)]·AA` (~S) — DONE

Build only T1, with T2 and T3 forced to zero. **Sanity check before any
FD-oracle comparison**: at a point where `∇ρ` (or the probed `δ∇ρ`
component along the trial direction) is negligible, T1 alone should
reduce toward F3.1's own LDA-only `v2rho2·δρ` term — not a formal proof
(the `2·v2rhosigma·(∇ρ·δ∇ρ)` piece is still genuinely part of T1, not an
LDA leftover), but a useful smoke check that the plain-AO-product half of
the contraction machinery (reused unchanged from F3.1) still works when
composed with a GGA functional's `v2rho2`/`v2rhosigma`.

*Verify:* compare `T1` alone against the FD oracle's induced potential
with `vsigma` and its derivatives artificially zeroed in the oracle's own
evaluation path (or, more simply, verify T1's contribution algebraically
matches the oracle's own decomposition by comparing at a probe direction
where T2 and T3 are independently confirmed negligible — do not just trust
a partial match against the full oracle, since T2/T3 could cancel T1's own
error there). **If this disagrees, stop before F3.3.2** — the plain-AO
term is the simplest of the three and any defect here recurs in T1's own
contribution to the combined sum.

**Implemented as a synthetic-point unit check
(`planck-dft-gga-hessian-selfcheck`, `tests/dft_gga_hessian_selfcheck.cpp`),
not a real-molecule grid search — decided explicitly rather than pursued
as originally sketched.** The original plan (search a real molecule's
grid for a point with negligible `∇ρ`, then compare the whole-molecule
projected `H·x` there) has a problem the doc's own text did not catch
until implementation was attempted: F3.1/F3.2's verification compares a
*projected, grid-integrated* MO matrix element, so making T2/T3
negligible *there* requires `∇ρ ≈ 0` at essentially every grid point along
the probed direction, not just one — true only for a trivial system (a
single spherically symmetric atom has zero *angular* gradient, but not
zero *radial* gradient in general, so even that does not give a clean
whole-grid cancellation). Switched to F1/F2's own successful pattern
instead: a point-level check against libxc's own finite difference,
exactly analogous to F1's `evaluate_lda_fxc` selfcheck extended to GGA.
This tests T1's coefficient formula (`δ[vrho] = v2rho2·δρ +
2·v2rhosigma·(∇ρ·δ∇ρ)`) directly against a raw central difference of
`vrho` under the correctly-substituted joint `(δρ, δσ=2∇ρ·δ∇ρ)`
perturbation — the AO-projection wiring itself is exercised later, in
F3.3.4/F3.5's whole-molecule comparisons, once the point-level formula is
trusted.

**Result: matches to the FD path's own step-size precision at every
tested point** (four points spanning nonzero and exactly-zero `∇ρ`,
`δ∇ρ` parallel and non-parallel to `∇ρ`), on PBE (matching F1's own GGA
choice). The exact-`∇ρ=0` point additionally confirmed T1's coefficient
reduces EXACTLY (not merely approximately) to F3.1's own `v2rho2·δρ` LDA
term there, since `∇ρ·δ∇ρ` vanishes identically when `∇ρ=0` regardless of
`δ∇ρ`. Mutation-verified: dropping the `2·` factor on the
`v2rhosigma·(∇ρ·δ∇ρ)` piece is caught cleanly at `h=1e-3`/`1e-4` (not
`h=1e-2`, whose own looser FD-truncation tolerance happened to absorb
that particular mutation's size at the tested points — expected, not a
gap, since `h=1e-2` is the least precise of the three step sizes by
design).

##### F3.3.2 — add T2: `2·[v2rhosigma·δρ + 2·v2sigma2·(∇ρ·δ∇ρ)]·(∇ρ·AG)` (~S, after F3.3.1)

Add the second coefficient-substitution term, still with T3 (the
`δ∇ρ`-argument term) forced to zero. This is the more delicate of the two
coefficient terms — it reuses the *existing* `∇ρ·AG` gradient-coupling
factor (unchanged from the ground-state first-derivative code) but weights
it by the *new*, second-derivative-sourced coefficient.

*Verify:* `T1 + T2` (still without T3) against the FD oracle **only where
T3 is independently confirmed small** (e.g. a probe direction where
`δ∇ρ·AG` measures near-zero on its own, checked directly rather than
assumed) — the same "isolate before combining" discipline F3.3.1 used, so
a T1+T2 agreement here is not accidentally validated by an uncompensated
T3 contribution hiding underneath it.

##### F3.3.3 — add T3: `2·vsigma·(δ∇ρ·AG)` (~S, after F3.3.2)

Add the term most likely to be silently dropped: the ground-state `vsigma`
(unchanged, already computed for the SCF's own converged KS potential)
contracted against a **new** gradient-coupling vector built from `δ∇ρ`
(F2's confirmed-linear density-gradient response) instead of `∇ρ`. This
needs no new kernel evaluation (`vsigma` is F1/production's existing
first-derivative output, not a second derivative) — only a second call to
`gradient_projection`-style AO coupling with `δ∇ρ` swapped in for `∇ρ`.

*Verify:* `T1 + T2 + T3`, the full sum, against the FD oracle — this is
the real F3.3 verification the original single-step scope asked for,
reached in a way that isolates which term is responsible if it fails.
Choose `x` directions with a genuinely non-uniform density gradient at the
test geometry (a bent triatomic like water rather than a homonuclear
diatomic, so `∇ρ` does not vanish by symmetry along the probed direction —
a diatomic's own axial symmetry would make `T3` trivially small along
many directions and weaken this as a check of T3 specifically, the same
class of trap F3.2's own symmetry-suppressed direction already
demonstrated). **If this disagrees, the earlier sub-steps already isolate
where to look** — do not re-derive from scratch; check whether F3.3.1's or
F3.3.2's own isolated agreement was itself compromised by an
uncompensated T3 leaking through (the risk F3.3.1/F3.3.2's own verify
notes flag), before assuming a new defect in T3 itself.

##### F3.3.4 — cross-check against F1/F3.1's own `v2rhosigma` equivalence finding (~S, after F3.3.3)

F1's own selfcheck verified `v2rhosigma` satisfies
`d(vrho)/d(sigma) == d(vsigma)/d(rho)` as two independently-measured
finite differences of libxc's own first derivatives. This sub-step is the
one place in F3.3 that specifically exercises whether that equivalence
also holds *through* the contraction (i.e. whether T1's
`2·v2rhosigma·(∇ρ·δ∇ρ)` piece and T2's `v2rhosigma·δρ` piece — which read
the SAME `v2rhosigma` array but multiply it against structurally different
factors — are both using it correctly, not just that the raw array value
is right). Not a new formula; a targeted regression-style re-check using
the already-verified full `T1+T2+T3` sum from F3.3.3, run once more on a
second, independent system to confirm the agreement wasn't specific to the
first test geometry.

*Verify:* `T1+T2+T3` against the FD oracle on a second system (different
molecule, same PBE functional) — agreement to the FD path's own step-size
precision, same as F3.3.3.

**If this disagrees, stop before attempting F3.4.** The polarized GGA case
adds cross-spin coupling on top of this gradient algebra; debugging both
new pieces at once from a single failure is exactly the compounding this
ladder exists to avoid.

#### F3.4 — GGA, polarized: cross-spin gradient coupling (~M, after F3.3)

**Work out this case explicitly before trusting F3.3's unpolarized form
generalizes.** The existing first-derivative polarized branch
(`src/dft/ks_matrix.cpp:194-206`) is not a simple duplication of the
unpolarized one — `coefficient_alpha` already mixes `vsigma(point,1)`
(the cross density-gradient term) into the alpha channel, and the induced
term inherits that same cross-coupling one derivative order higher,
through `v2rhosigma`'s own 6-component polarized packing (F1's own
finding: `2 rho-channels × 3 sigma-channels`, not a simple per-spin
duplication).

*Verify:* PBE, polarized, on a genuinely open-shell system (matching the
"do not test only closed-shell" lesson U1 already learned the hard way for
UHF) — a doublet or triplet small molecule, both a same-spin and a
cross-spin `x` direction, against the FD oracle.

#### F3.5 — MO projection and `(a,i)` packing (~S, after F3.4)

Pure plumbing, no new physics: project the verified `δV_xc` (whichever of
F3.1–F3.4's cases applies) into the `(a,i)` MO block the same way
`assemble_xc_matrix`'s output is projected in the KS build —
`H·x = C_occᵀ · δV_xc(AO basis) · C_virt` — and pack the result into the
same flat `(a,i)` vector layout U1/U2 already use for the UHF CPHF
matrix, so the eventual F4/F5 wiring can hand this directly to
`solve_augmented_hessian` without another translation layer.

*Verify:* the packed, projected `H·x` from this step matches a
*full end-to-end* FD-oracle comparison (build the FD oracle's own
`ResponseExcitationSpace`-packed kernel block and compare element-by-
element, not just the raw AO-basis `δV_xc` from the earlier steps) — this
is the first point where a packing-index bug (row/column transposition,
`(a,i)` vs `(i,a)` ordering) could hide independently of every earlier
step's own correctness, so it needs its own dedicated check.

#### Cross-cutting notes (apply to all of F3.1–F3.5)

**Verify against D1's oracle at every step, not against a hand-derivation
alone, and not only once at the end.** This is the load-bearing discipline
for all five sub-steps — RHF SOSCF's own history is the direct precedent
for why: a gradient/Hessian pairing that looked individually correct on
each side (right functional form, right units) was still silently wrong by
a factor of 2/4 until checked directly against the true `E(κ)`. Here the
FD-kernel oracle plays the role the finite-difference-of-`E(κ)` probe
played for RHF/UHF. Do not skip straight to comparing full-SOSCF-run
energies once F3.5 lands; a wrong Hessian that happens to still converge
(to a linear rate, say) can hide for a long time, exactly as pure-unbounded
RHF SOSCF's own scale-mismatch bug did before it was checked directly. The
five-step ladder exists specifically so a defect is caught at the smallest
sub-step that can exhibit it, rather than surfacing only once every piece
is assembled.

**Each sub-step's own oracle comparison is structurally independent, not
merely independently-run, per F2's own finding.** F2 discovered that a
finite-difference self-consistency check cannot see a bug shared by both
sides of the comparison (e.g. a uniform scale error, mutation-verified
there). F3's comparisons do not have this weakness: the FD-kernel oracle
perturbs the density and re-evaluates the FULL XC potential from scratch
through the ordinary first-derivative `exc_vxc` path, while the analytic
path contracts a second-derivative kernel through entirely different code
(`evaluate_lda_fxc`/`evaluate_gga_fxc`). The two share no common formula
that could be wrong in the same way on both sides — worth stating
explicitly rather than assuming the parallel to F1/F2's own verification
style is automatically safe, since F2 showed that assumption can fail in
general.

**Within each sub-step, test more than one `x` direction.** A single
diagonal-dominant direction can hide a bug an off-diagonal-heavy direction
would catch — F1's own full-index sweep and U1's UHF Hessian-diagonal
sweep both found exactly this shape of gap when they moved from one probed
direction to many.

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
