# Scope: extending SOSCF to UHF and DFT

**Scope for in-flight work. Not started.** Follow-on to `docs/SOSCF.md`
(RHF SOSCF, landed, PR #167), which investigated but did not build the UHF
and DFT extensions. This scope was written by actually reading the UHF
response code and the DFT KS loop, not by assuming the RHF pattern
transfers — it does, for UHF, in a modified form; it does not, cleanly, for
DFT.

These are two independent tracks. UHF can be built without touching DFT and
vice versa; do not block one on the other.

## Track 1 — UHF

### What exists already

`solve_uhf_cphf` (`src/post_hf/uhf_response.cpp`) already builds the full
dense coupled α/β orbital Hessian internally: a single matrix `A` of size
`(nova+novb) × (nova+novb)` where `nova = n_virt_α · n_occ_α`,
`novb = n_virt_β · n_occ_β`. Its diagonal is the unscaled
`ε_a - ε_i` per spin block — the same convention `build_rhf_cphf_matrix`
uses, not PySCF's `2×`-scaled one. This is a good sign but **must still be
verified by finite difference** (see Step U1) before being trusted; RHF's
own gradient/Hessian pairing looked individually plausible and was wrong by
a factor of 2 until checked directly against `E(κ)`.

### What is different from RHF, and why it is not a drop-in port

1. **The Hessian is built by column, via real integral-layer calls, not a
   closed-form ERI contraction.** `solve_uhf_cphf` constructs `A` one column
   at a time: for each trial rotation `x`, it forms the induced AO density
   response and calls `_compute_2e_fock_uhf` (or `RI::build_ri_fock_uhf`
   under RI) to get the induced Coulomb/exchange response, then projects
   that back into the `(a,i)` block. This is `(nova+novb)` separate Fock
   builds per Hessian construction, called once per SOSCF iteration. For
   RHF, `build_rhf_cphf_matrix` instead transforms the full ERI tensor once
   (`transform_eri`) and reads every matrix element out of it directly — a
   single `O(n_b^4)` transform, not `O(n_ov)` Fock builds. **The UHF
   Hessian build is a fundamentally more expensive operation per call than
   the RHF one**, not just a bigger matrix. Measure this cost before
   assuming UHF SOSCF is a net win the way RHF SOSCF was.
2. **`solve_uhf_cphf` currently couples matrix construction and the linear
   solve in one function** (`A.colPivHouseholderQr().solve(rhs)` at the
   end). SOSCF needs the matrix (or a Hessian-vector-product callback) on
   its own, the way `build_rhf_cphf_matrix` is already split from
   `solve_rhf_cphf`. This split does not exist yet for UHF and must be
   built — mechanical, but real work, not zero work.
3. **UHF's SCF loop has more state SOSCF must account for than RHF's does.**
   `run_uhf` builds `Fa_s`/`Fb_s` from the level shift *before* DIIS pushes
   (`src/scf/scf.cpp` around the `Fa_diag`/`Fb_diag` selection), and its
   DIIS restart is triggered by error growth (`restart_factor`), not just a
   handoff boundary. A SOSCF window must decide explicitly what happens to
   the level shift while active (RHF SOSCF has no separate level-shift
   staging to reason about, so this question never arose there) and must
   clear the *combined* `UHFDIISState` on handoff, not a single-spin one.
4. **The gradient is two coupled blocks, not one.** RHF's gradient is a
   single `F_mo(a,i)`; UHF's is `(g_α, g_β)` packed into one vector the same
   way `solve_uhf_cphf`'s `rhs` already is. The Cayley rotation and any
   post-step semicanonicalization (if kept — see Step U1's stop condition)
   need separate `κ_α`/`κ_β` and separate α/β occ-occ and virt-virt blocks.

### Steps

Ordered so a cheap, independent check happens before any SCF-loop wiring.

#### U1 — factor out `build_uhf_cphf_matrix`, verify by finite difference (~M)

Split `solve_uhf_cphf` into a `build_uhf_cphf_matrix` (returns `A`, or a
Hessian-vector-product callback if the per-column integral cost makes
materializing the full dense matrix too expensive at the sizes SOSCF will
actually run at — measure this before choosing) and a thin
`solve_uhf_cphf` that calls it and solves `A z = -rhs`, mirroring the
existing RHF split exactly. `solve_uhf_cphf`'s own behavior must be
unchanged (same energies on its one existing caller, the UHF MP2 gradient).

Build the same `PLANCK_SOSCF_FD_CHECK`-style probe RHF SOSCF used: perturb
`(α,i)` and `(β,i)` orbitals by a small `κ`, compute the actual UHF energy
`E(κ)` numerically, and check the analytic gradient and the extracted
Hessian diagonal reproduce the finite-difference values. **Do not skip this
because the RHF version already checked out at the same unscaled
convention** — the diagonal matching RHF's form is necessary, not
sufficient; the coupling terms between α and β blocks are new and were
never independently checked.

*Verify:* gradient and Hessian-diagonal FD agreement converges as `h → 0`,
the same three-step-size pattern (`1e-2`, `1e-3`, `1e-4`) RHF's check used.

**If the energies or FD checks disagree, stop.** Do not proceed to wiring a
wrong Hessian into the SCF loop — this was the exact failure mode that cost
the most debugging time in the RHF work.

#### U2 — wire the SOSCF branch into `run_uhf`, fixed iteration, no fallback (~M)

Mirror RHF SOSCF's branch shape exactly: persist `Ca_soscf_prev`,
`Cb_soscf_prev`, `epsa_soscf_prev`, `epsb_soscf_prev` across iterations;
build the gradient/Hessian from the *previous* iteration's basis against
the *current* Fock (both spins); solve with the same
`solve_augmented_hessian`, capped the same way (`kSoscfMaxRot`), with the
same `ah_start_tol` scaling investigation redone for UHF's gradient
magnitude (do not assume RHF's `0.1·‖g‖` factor transfers without
checking — UHF's gradient norm combines two spin blocks and may sit at a
different natural scale). Apply the joint α/β rotation via
`apply_orbital_rotation` on each spin channel separately.

Switch on a fixed iteration first (`scf_soscf_start`, reusing the same
keyword — UHF and RHF SOSCF are mutually exclusive per run, so one keyword
namespace is fine), not a criterion. The goal is proving the step correct,
exactly as RHF's own S2 was scoped.

*Verify:* on a UHF case that converges today, SOSCF from iteration N
reaches the same energy to all 10 printed digits as pure DIIS. Test on a
genuinely open-shell system (a doublet or triplet), not a UHF run on a
closed-shell molecule — the α/β coupling terms in the Hessian are exactly
what a closed-shell UHF run cannot exercise.

**If the energies differ, stop**, same rule as RHF's S2.

#### U3 — decide on semicanonicalization and the level-shift interaction (~S)

RHF SOSCF's semicanonicalization step (block-diagonalize occ-occ and
virt-virt separately after the Newton step) was built, measured to have no
effect on the actual bug it was meant to fix, and kept anyway because it is
correct and cheap. For UHF, re-derive this per spin channel and re-measure
whether it matters here specifically — do not assume RHF's "harmless, keep
it" verdict transfers.

Decide explicitly what SOSCF does with an active level shift: disable the
level shift during the SOSCF window (simplest, matches the reasoning that
level shift and second-order methods solve overlapping problems), or thread
it through the gradient/Hessian construction. **Do not leave this
undecided** — the level-shifted `Fa_s`/`Fb_s` currently feeds directly into
what would become the SOSCF gradient source, and silently ignoring the
shift's presence would double-count or drop it depending on which Fock the
new code reads.

#### U4 — the switch criterion and SAD composition (~S)

Repeat RHF's S3 and its own SAD-composition check for UHF: replace the
fixed iteration with `scf_soscf_diis_tol`/`scf_soscf_min_iter` (already
shared keywords, no new ones needed), and verify SOSCF composes with the
UHF SAD guess (`compute_sad_guess_open_shell`) the same way it was verified
for RHF's SAD guess — should be equally orthogonal (SAD only sets the
initial `Pa`/`Pb`, before the loop starts), but verify rather than assume,
matching the standing rule from `docs/SOSCF.md`.

*Verify:* same shape as RHF's S3 — iteration count falls on a system where
DIIS alone is slow, does not regress on a system where DIIS alone is
already fast.

### What this must not do

- **Do not assume the UHF Hessian's per-column integral cost is
  negligible.** Measure it on at least one system before claiming UHF
  SOSCF is a net iteration-count *and* wall-clock win; it may be a
  correctness-and-iteration-count win only, with the DIIS-only path still
  faster in wall-clock terms until the Hessian build is optimized.
- **Do not skip the finite-difference check because RHF's diagonal
  convention matched.** The off-diagonal α-β coupling in `A` is genuinely
  new territory.
- **Do not build a second augmented-Hessian solver or a second Cayley-
  rotation helper.** Both existing pieces (`solve_augmented_hessian`,
  `apply_orbital_rotation`) are reference-type-agnostic; call them per spin
  channel, do not fork them.

## Track 2 — DFT

### Corrected framing: this is a research question, not a wiring task

An earlier pass at this scoping said DFT SOSCF was "wiring, not
derivation," citing `build_unrestricted_xc_kernel_blocks` /
`build_closed_shell_xc_kernel_blocks` (`src/dft/driver.cpp`) as an existing
XC-kernel builder ready to reuse. **That framing was wrong, found by reading
the function body rather than trusting its name.** Those builders are a
*numerical finite-difference* construction: for every occ-virt pair in the
TDDFT excitation space, they perturb the density, re-evaluate the full
grid density/XC pass twice (`evaluate_xc_matrix_from_spin_densities` at
`+step` and `-step`), and finite-difference the result. This is acceptable
for TDDFT, where the excitation space is a deliberately small, user-chosen
subset of orbital pairs (`lr_nstates`, typically single digits to tens).
Reused directly as a SOSCF orbital Hessian, the same construction would
need one finite-difference pair **per every occ-virt pair in the full
orbital space** — `O(n_occ · n_virt)` full-grid XC evaluations, every SOSCF
iteration. For a molecule large enough that SOSCF's iteration-count
reduction would matter, this cost very likely dominates and could make DFT
SOSCF slower than plain DIIS, not faster.

The actual production-path answer is an **analytic** XC second derivative
(`fxc`), which does not exist anywhere in this codebase. libxc itself
exposes `xc_lda_fxc` / `xc_gga_fxc`; Planck's wrapper
(`src/dft/base/wrapper.h`) only ever calls the first-derivative
`exc_vxc` family. Wiring `fxc` through the wrapper and contracting it
against a trial density on the grid (the analytic analogue of what
`build_unrestricted_xc_kernel_blocks` computes by finite difference) is
real, unstarted work — closer in kind to deriving the RHF Hessian than to
writing the RHF SOSCF *callbacks* was.

### Steps

#### D1 — decide the target before writing any code (~S, but a real decision)

This is a genuine fork, and building the wrong one wastes the whole
remaining scope:

- **(a) Correctness-only / small-system path.** Reuse the existing FD
  kernel builders as-is, accept the `O(n_occ · n_virt)` grid cost, and
  scope DFT SOSCF explicitly as a small-system correctness reference or a
  research tool — not a production convergence accelerator. This is
  buildable now, with no new libxc wiring, but should not be marketed as a
  speedup.
- **(b) Analytic-fxc production path.** Wire `xc_lda_fxc`/`xc_gga_fxc`
  through the wrapper, build the analytic Hessian-vector product, and scope
  it as the real accelerator DFT needs at the sizes where iteration count
  actually matters (the same large-`nb` regime `docs/SOSCF.md`'s
  unreproduced `scale.json` ladder was originally about). This is
  materially more work: a new libxc call path, a new grid contraction, and
  its own finite-difference verification against the true DFT `E(κ)`
  before it can be trusted at all.

**Recommendation: (a) first, as a cheap correctness oracle for (b).** Building
the FD-kernel path first gives something to check the eventual analytic
`fxc` path against on a small system, the same role the FD-check probe
played for RHF — do not derive the analytic Hessian and trust it without an
independent numerical reference.

#### D2 — route the KS SCF loop through a shared insertion point (~M)

The KS loop (`src/dft/driver.cpp`, the `!unrestricted` branch around
`fock_for_diagonalization` → `diagonalize_in_ao_basis`) is architecturally
parallel to `run_rhf`'s DIIS-selection-then-diagonalize block, but it is a
separate implementation, not a call into `run_rhf`. A SOSCF branch here
needs its own persisted reference-basis state (mirroring
`C_soscf_prev`/`eps_soscf_prev`) and its own gate logic — do not attempt to
unify the HF and KS loops into one function as a prerequisite; that is a
much larger refactor with its own risk, out of scope here.

*Verify:* whichever kernel path D1 chose, the same-energy-as-DIIS check
RHF's S2 used, on a small closed-shell KS case first.

#### D3 — UKS (~M, after D2's RKS path is verified)

Repeat the RHF→UHF generalization (Track 1) for the KS analogue: separate
α/β kernel blocks, the DFT equivalent of `build_uhf_cphf_matrix`. Do this
after D2's restricted KS path works, not in parallel with it — DFT already
has more moving parts (grid, XC functional selection, hybrid exact-exchange
fraction) than either RHF or UHF SOSCF did, and stacking the UKS
generalization on an unverified RKS base compounds the debugging surface.

#### D4 — the switch criterion, and whether it should differ from HF's (~S)

Revisit whether `scf_soscf_diis_tol` should have a different default for
DFT. DFT's own DIIS error trajectory is shaped differently from HF's (the
`scale.json` data already on file shows DFT's iteration-count cliff arrives
at a smaller `nb` than HF's), so a single hardcoded default tuned against
HF cases may not be the right one for DFT. Sweep on whatever cases D2/D3
land, not on the HF cases already used to tune the HF default.

### What this must not do

- **Do not reuse the TDDFT finite-difference kernel builders as the
  production DFT SOSCF path without first measuring their actual cost at
  realistic system sizes.** If option (a) is chosen, say explicitly in the
  result that it is a correctness reference, not a speedup, unless
  measurement shows otherwise.
- **Do not derive or wire an analytic `fxc` path without a finite-difference
  reference to check it against.** This is the single lesson RHF SOSCF's
  own debugging most directly transfers: a gradient/Hessian pairing that
  looks individually correct on each side can still be silently
  inconsistent, and only a numerical check against the true energy catches
  that.
- **Do not attempt to unify the HF and KS SCF loops as a prerequisite for
  this work.** Route DFT SOSCF through its own insertion point in the
  existing KS loop, the same way RHF and UHF SOSCF are two separate
  branches in two separate loops, not one shared implementation.
- **Do not assume the `scf_soscf_*` defaults tuned on HF cases are correct
  for DFT.** Re-tune against DFT's own convergence trajectory.

## Key code locations

| what | where |
|---|---|
| UHF's existing coupled α/β CPHF matrix build (to be split out) | `solve_uhf_cphf`, `src/post_hf/uhf_response.cpp` |
| UHF's SCF loop insertion point | `run_uhf`, `src/scf/scf.cpp` (the `Fa_diag`/`Fb_diag` selection, immediately before `diagonalize_uhf_spin`) |
| UHF's combined-spin DIIS state (to be cleared on SOSCF handoff) | `UHFDIISState`, `src/base/types.h` |
| UHF's SAD guess (composition to verify) | `compute_sad_guess_open_shell`, `src/scf/sad.cpp` |
| the RHF pattern to mirror (landed reference) | the SOSCF branch in `run_rhf`, `src/scf/scf.cpp`; `docs/SOSCF.md` |
| the existing (finite-difference, TDDFT-scoped) XC kernel builders | `build_unrestricted_xc_kernel_blocks`, `build_closed_shell_xc_kernel_blocks`, `src/dft/driver.cpp` |
| DFT's KS SCF loop insertion point | `src/dft/driver.cpp`, the `!unrestricted` branch, `fock_for_diagonalization` → `diagonalize_in_ao_basis` |
| libxc's first-derivative-only wrapper surface (fxc not yet wired) | `src/dft/base/wrapper.h`, `evaluate_lda_exc_vxc` / `evaluate_gga_exc_vxc` |
| the generic CIAH solver (reuse for both tracks, do not rewrite) | `solve_augmented_hessian`, `src/post_hf/casscf/aug-hessian.h` |
| the finite-difference verification pattern to reuse | the `PLANCK_SOSCF_FD_CHECK` probe in RHF's `run_rhf` branch, `src/scf/scf.cpp` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
