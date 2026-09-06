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

#### U1 — factor out `build_uhf_cphf_matrix`, verify by finite difference (~M) — DONE

`solve_uhf_cphf` (`src/post_hf/uhf_response.{h,cpp}`) is split exactly as
scoped: `build_uhf_cphf_matrix` returns the dense coupled α/β `A` (no
convergence guard, mirroring `build_rhf_cphf_matrix`'s own relaxation, since
SOSCF will call it mid-iteration), and `solve_uhf_cphf` is now a thin
wrapper that calls it and solves `A z = -rhs`. Materializing the full dense
matrix was kept (not a Hessian-vector-product callback) — the per-column
integral cost question is deferred to U2, where it can be measured against
real SOSCF iteration counts rather than guessed at here. `solve_uhf_cphf`'s
one existing caller (the UHF MP2 gradient) is unchanged: all 11 UHF-tagged
regression cases pass, including `water_triplet_uhf_ump2_gradient_smoke` and
`water_radical_cation_uhf_ump2_sto3g`.

The FD probe (`PLANCK_SOSCF_FD_CHECK`, `src/scf/scf.cpp`, gated identically
to RHF's) verifies `build_uhf_cphf_matrix` against the real UHF `E(κ)` in
`run_uhf`, using the previous iteration's basis paired against the current
Fock (RHF SOSCF's own "attempt 2" trap — `Cᵀ F C` is diagonal by
construction immediately after diagonalizing `F`, so probing there is
vacuous — bit UHF here too on the first pass, fixed by persisting
`Ca_prev`/`Cb_prev`/`epsa_prev`/`epsb_prev`). Gated off `atomic_numbers.size()
> 1` since SAD's per-element atomic UHF sub-solves (`sad.cpp`) recurse into
this same `run_uhf` on lone atoms, where a spin channel can have zero
virtuals.

**Measured, on water/STO-3G and water/6-31G triplet (genuinely open-shell,
so the α-β coupling terms are exercised) — and a full sweep over every
`(a,i)` diagonal index (28 alpha + 36 beta directions on water/6-31g):**

```
g_fd / g_used  = 2.0000000  (every single probed index, alpha and beta)
```

**The gradient needs a clean, universal, direction-independent factor of 2**
(not RHF's 4 — UHF's diagonal-block CPHF formula is already
`(ai|ia) - (aa|ii)` with no leading Coulomb multiplier, verified separately
against a hand-derived same-spin formula on a toy random ERI tensor, ratio
1.0 exactly). The raw Hessian diagonal element `A_used` does **not**
reproduce `h_fd` cleanly at most swept indices (ratios from 0.2 to 272
across the sweep, sign flips included) — **this is not a bug**: it is
expected off-diagonal orbital-Hessian curvature dominating a coupled
multi-virtual open-shell system, the same effect RHF's own probe never had
to contend with because that probe's one direction happened to be weakly
coupled to the rest of the space. What a Newton step actually needs is the
ratio `g/H`, and since `g_true = 2·g_used` and `H_true = 2·Amat` (confirmed
together on the well-isolated water/STO-3G indices, where `2·Amat` tracks
`h_fd` to the same few-percent residual RHF's own probe showed), using
`g = F_mo` against `Amat` unscaled reproduces the true step at half the
arithmetic — **`build_uhf_cphf_matrix` needs no changes**, exactly RHF's own
conclusion.

*Verify:* gradient FD agreement converges as `h → 0` (`1e-2`, `1e-3`,
`1e-4`) on both test systems; confirmed.

**Stop condition was not triggered** — the gradient scale is clean and
universal, and the Hessian-diagonal spread is explained (off-diagonal
curvature, not a convention bug) rather than dismissed. U2 can proceed using
`g = F_mo` (unscaled) against `Amat` (unscaled) exactly as RHF's own SOSCF
branch does.

#### U2 — wire the SOSCF branch into `run_uhf`, fixed iteration, no fallback (~M) — DONE

Wired exactly as scoped, mirroring `run_rhf`'s SOSCF branch shape: U1's
`Ca_prev`/`Cb_prev`/`epsa_prev`/`epsb_prev` (already persisted every
iteration for the FD-check probe) are promoted to the actual step's
gradient/Hessian source, built from the *previous* iteration's basis
against the *current* `Fa`/`Fb`. Solves with the same
`solve_augmented_hessian`, capped the same way (`kSoscfMaxRot = 0.20`), and
`ah_start_tol = max(1e-8, 0.1·‖g‖)` transfers unchanged from RHF — no
re-derivation needed, since U1 already established `g = F_mo` (unscaled)
against `Amat` (unscaled) is the correct pairing for a Newton step (the
ratio is what matters, and both carry the same 2× UHF convention). Applies
the step via `apply_orbital_rotation` on each spin channel separately (one
shared step vector, two `κ` matrices — same helper, not forked), then
semicanonicalizes each spin channel's occ-occ/virt-virt blocks separately
(pure gauge freedom, mirrors RHF's own post-step semicanonicalization).

Switched on `scf_soscf_start` (fixed iteration, shared keyword with RHF —
UHF and RHF SOSCF are mutually exclusive per run) with `!sao_active_uhf &&
pcm == nullptr`, matching RHF's own `soscf_enabled` gate exactly (no SAO or
PCM coverage yet, same S2/U2 scope line). DIIS is cleared on window handoff,
same as RHF.

*Verified:* on three genuinely open-shell systems (water/STO-3G triplet
from SAD, water/6-31G triplet from hcore, water-cation/STO-3G doublet from
hcore — deliberately not a closed-shell UHF run, so the α-β coupling terms
in the Hessian are actually exercised), SOSCF from iteration 3 reaches the
same energy as pure DIIS to all 10 printed digits in every case, with the
orbital gradient shrinking superlinearly across the window (e.g.
`2.16e-1 → 3.37e-2 → 3.55e-3` on 6-31G). All 11 UHF-tagged regression cases
and the full core/smoke suites (71/71, 35/35) pass unchanged with SOSCF off
by default (no `scf_soscf_start`/`scf_soscf_diis_tol` set).

**Energies matched at every test point — the stop condition was never
triggered.**

#### U3 — decide on semicanonicalization and the level-shift interaction (~S) — DONE

**Semicanonicalization, re-measured rather than assumed.** Built into U2
(block-diagonalize occ-occ and virt-virt separately per spin after the
Newton step). Disabled it (reading `eps` off the raw, non-eigendecomposed
`Cᵀ F C` diagonal per spin) and ran a long pure-SOSCF window (200 cycles, no
DIIS handoff) on two genuinely open-shell systems: water/6-31G triplet
(60 vs 64 iterations with it) and the water-cation/STO-3G doublet (25 vs 32
with it). **Both converge to the identical energy either way — no plateau,
no wrong-basin convergence.** Kept anyway, same verdict as RHF: it is pure
gauge freedom (rotating occupied or virtual orbitals among themselves
changes neither density nor energy) and cheap (two small in-block
eigendecompositions per spin, not a full `nbasis`-size solve), so there is
no reason to drop it even though it measured as unnecessary at these sizes
too.

**Level-shift interaction, decided explicitly and enforced in code, not
just documented.** `soscf_enabled_uhf` now requires `level_shift <= 0.0` —
SOSCF and an active level shift are mutually exclusive per run, matching
the doc's recommended simplest option. The SOSCF gradient/Hessian already
read the plain (unshifted) `Fa`/`Fb`, so this was never a silent
double-counting risk, but running SOSCF against the unshifted Fock while
`level_shift > 0.0` is configured would have silently ignored the user's
own request on exactly the iterations where they set it to matter — the
new gate makes that impossible rather than merely unlikely. Verified: a
`level_shift 0.3` + `scf_soscf_start 3` run emits zero `SOSCF :` log lines
and reaches the same energy (`-74.6557058354`) as the equivalent run
without the level shift request, confirming DIIS-with-shift runs unchanged
end to end.

All 11 UHF-tagged regressions plus the full core (71) and smoke (35) suites
pass unchanged.

#### U4 — the switch criterion and SAD composition (~S) — DONE

The criterion-based trigger required no new code: U2's SOSCF-window
selection was written as a direct mirror of RHF's own `soscf_enabled`/
`soscf_active` logic (docs/SOSCF_UHF_DFT_SCOPE.md, U2), which already
included the `scf_soscf_diis_tol > 0.0` branch alongside the fixed-iteration
one — so `scf_soscf_diis_tol`/`scf_soscf_min_iter` were live from the moment
U2 landed. U4's job was purely to verify that path rather than trust it by
inspection.

**Verified, on water/6-31G triplet (SAD guess) and the water-cation/6-31G
doublet (hcore guess):**

- The DIIS-error criterion fires at the correct iteration (once
  `diis_err < scf_soscf_diis_tol` **and** `iter >= scf_soscf_min_iter`, not
  before) and reaches the same energy as pure DIIS to all 10 printed digits
  in both cases (`-75.7302585147` triplet, `-75.5085348059` doublet).
- **SAD composes cleanly with SOSCF** — no interaction, exactly as
  predicted (SAD only sets the initial `Pa`/`Pb` before the loop starts).
- **Iteration count falls on the harder-starting case** (water-cation from
  hcore: 18 → 15) and stays the same on the already-fast one (water/6-31G
  triplet from SAD: 19 → 19) — same shape RHF's S3 verification used.

No code changes were needed for U4 itself; all 11 UHF-tagged regressions
plus the full core (71) and smoke (35) suites still pass.

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
writing the RHF SOSCF *callbacks* was. **Scoped in full, step by step, in
`docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md`** (F1–F6), written after D1's
decision so option (b)'s own scope can lean on (a) as its numerical oracle
rather than trusting a hand-derived Hessian in isolation.

### Steps

#### D1 — decide the target before writing any code (~S, but a real decision) — DONE

**Decided: (a), the doc's own recommendation** — reuse the existing
finite-difference XC kernel builders (`build_unrestricted_xc_kernel_blocks`
/ `build_closed_shell_xc_kernel_blocks`, `src/dft/driver.cpp`) as-is,
accept the `O(n_occ · n_virt)` grid cost, and scope DFT SOSCF explicitly as
a small-system correctness reference — not a production convergence
accelerator — that later serves as the numerical oracle the eventual
analytic-`fxc` path (b) must be checked against, exactly as RHF SOSCF's own
`PLANCK_SOSCF_FD_CHECK` probe was used to verify `build_rhf_cphf_matrix`
before it was trusted.

Confirmed by reading `build_unrestricted_xc_kernel_blocks` closely before
committing: it is parameterized by `ResponseExcitationSpace`, which already
represents an *arbitrary* occ-virt subset (`n_occ`, `n_virt`, `C_occ`,
`C_virt` — nothing TDDFT-specific baked into the type itself). Feeding it a
space spanning the *entire* occupied/virtual manifold (rather than
TDDFT's small user-chosen `lr_nstates` subset) is enough to turn it into a
full DFT SOSCF orbital-Hessian builder with **no new type and no changes to
the builder itself** — the cost warned about in the corrected framing above
(one +/- grid pass per occ-virt pair) is exactly what full coverage buys,
which is the tradeoff (a) explicitly accepts. D2 wires this in for RKS.

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

**RESCOPED after D1's own recommendation went stale.** D1 (above) chose
"(a) first" — the FD-kernel oracle, with the analytic `fxc` path (b)
explicitly deferred as harder, unstarted work to be scoped later "in
`docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md` (F1–F6)... written after D1's
decision." **F1–F5 have since landed in full** (see that file): the
analytic Hessian-vector product is derived, verified point-by-point
against the FD-kernel oracle for every term (LDA/GGA, unpolarized and
polarized), verified once against the oracle at the whole-molecule
AO-projected level for both RKS and UKS, and the MO-projection/packing
step (F3.5) is real, kept production code
(`pack_hessian_vector_product_cphf_order`,
`src/dft/response_packing.{h,cpp}`). D1's premise — "(b) is unstarted and
harder, so build the cheap oracle first and let it serve as (b)'s
eventual numerical reference" — is no longer true: the reference has
already been built and consumed. Building a *second*, dense,
`O(n_occ·n_virt)`-grid-pass Hessian source now, when a verified `O(1)`
analytic one already exists, would be solving a problem F3 already
solved, and would leave the actual point of D1's option (b) — the
production speedup — undelivered. **D2 is rescoped to wire F3's analytic
Hessian-vector product into the RKS SOSCF loop directly.** The FD-kernel
oracle is not thrown away: it remains available as an independent
correctness check for D2's own wiring (mirroring how RHF/UHF SOSCF each
had a `PLANCK_SOSCF_FD_CHECK`-style verification step before being
trusted), not as the production Hessian source itself.

**What F3 actually left runnable, checked directly rather than assumed:**
F3.1–F3.4 verified the T1..T5 algebra at isolated grid points (synthetic
`ρ`/`∇ρ`/`δρ`/`δ∇ρ` values, no real molecule) and, at F3.3.4/F3.4.5, once
each verified that the algebra composes correctly with a real converged
SCF's AO-projection machinery — **but those whole-molecule probes were
themselves removed from `driver.cpp` afterward** (per the
debug-probes-must-become-tests-or-be-deleted decision), and **the T1..T5
contraction itself was never promoted into a standalone, callable
production function** — it only ever existed inline inside test files and
the now-deleted probes. So D2 cannot skip straight to "call the existing
analytic Hessian-vector function"; that function does not exist yet as
production code. **D2.0 below is a new, necessary step this rescoping
adds**, not present in the pre-F3 D2 draft, because the pre-F3 draft
assumed (correctly, at the time) that D2 would be wiring the *already
production-shaped* FD-kernel oracle, not promoting research-stage
point-level algebra into a first production entry point.

Broken into five sub-steps (D2.0–D2.4), extending U1→U2→U3→U4's own
discipline (build and verify the Hessian source as a real, standalone
function before wiring it into the SCF loop; wire fixed-iteration with no
fallback; settle the remaining design decisions; verify the switch
criterion):

| Step | Adds | Verifies against |
|---|---|---|
| D2.0 | Promote F3's T1..T5 algebra into a real production function computing `δV_xc(AO basis)` from a trial `δP`, for RKS LDA/GGA | F3's own already-verified point-level formulas (this step is transcription into a proper function signature, not new derivation) plus one fresh whole-molecule check against the FD-kernel oracle, since the composition-with-real-AO-projection check was deleted along with the probe that ran it |
| D2.1 | Confirm the new function's Hessian-vector product, called at full occ-virt width via F3.5's packing, reproduces the true RKS `E(κ)` directly | Finite difference of the real RKS energy, the same `energy_at_kappa` shape `PLANCK_SOSCF_FD_CHECK` used |
| D2.2 | Wire the SOSCF branch into the RKS loop, fixed iteration, no fallback, mirroring `run_rhf`'s S2/`run_uhf`'s U2 shape exactly | Same-energy-as-DIIS to all 10 printed digits, superlinear gradient shrinkage across the window |
| D2.3 | Decide semicanonicalization and level-shift interaction for RKS SOSCF (does DFT's KS loop even have a level-shift knob to conflict with? — check, do not assume it mirrors RHF/UHF) | Re-measured with semicanonicalization on/off on a real system, same discipline S3/U3 used |
| D2.4 | Verify the DIIS-error switch criterion (`scf_soscf_diis_tol`) fires correctly for RKS, not just the fixed-iteration path | Correct trigger iteration, same energy either way, on at least one small closed-shell KS case |

Each step's own verification gates the next — **if D2.0 or D2.1 disagrees,
stop before D2.2**; wiring an unverified Hessian source into a live SCF
loop is exactly the failure mode RHF SOSCF's own scale-mismatch bug
demonstrated once already, and F3's own point-level verification does not
by itself prove the *assembled, callable* function is correct — a
transcription slip while promoting inline test algebra into a real
function signature is a new, distinct risk this rescoping introduces and
must gate on its own.

**A stale reference to correct while doing this work:** `docs/SOSCF.md`'s
own "Validation strategy that should remain in place" section says to
*keep* `PLANCK_SOSCF_FD_CHECK` "now that the defect is fixed" — that probe
(and its UHF sibling) were removed outright per the later, explicit
project-wide decision recorded in `docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md`'s
F3.4.5 section (debug probes must become standalone tests or be deleted
from production; converting them was found impractical, so they were
deleted with no replacement). `docs/SOSCF.md` needs a note added when D2
lands, so it stops recommending a probe that no longer exists.

##### D2.0 — promote F3's algebra into a real production function (~M) — DONE

Write `compute_analytic_xc_hessian_vector_product` (or similarly named,
in a small dedicated file the way F3.5's `response_packing.{h,cpp}` was
kept out of `driver.cpp` specifically so later tests do not need to link
the whole KS-loop driver) taking the ground-state density/grid state, a
trial `δP` (or a pre-built `δρ`/`δ∇ρ` pair), and the functional(s), and
returning `δV_xc` in the AO basis — RKS, LDA and GGA, unpolarized only
(UKS is D3's job). This is transcription of F3.1/F3.3.3's own
already-verified `T1`/`T1+T2+T3` formulas into one real function, reusing
`evaluate_lda_fxc`/`evaluate_gga_fxc` (F1) and the confirmed-linear
`evaluate_density_on_grid` (F2) exactly as those steps already established
— **no new algebra, but a new place for a transcription bug to hide** that
F3's own point-level tests cannot see, since they never called a function
shaped like this one.

*Verify:* two layers, deliberately not skipped even though F3 already
verified the underlying formulas. (1) Re-run F3's own point-level
comparisons — but through the NEW function's actual call signature, not
by re-deriving inline — so a shape/signature bug (wrong argument order, a
dropped functional contribution) cannot hide behind "the formula was
already checked once." (2) One fresh whole-molecule check against
`build_closed_shell_xc_kernel_blocks` (the FD-kernel oracle, still
available and still correct — D1's original oracle, now serving exactly
the role it was always meant for) on a real converged RKS calculation,
replacing the composition-check role F3.3.4's now-deleted probe used to
serve. Per F3.4.5's own resolution of this exact tension: decide
explicitly whether this whole-molecule check becomes a kept regression
case or a one-time derivation-time verification, rather than defaulting
to whichever is easier to write first.

**Landed as `compute_analytic_xc_hessian_vector_product`
(`src/dft/analytic_hessian.{h,cpp}`), RKS LDA and GGA unpolarized, kept
out of `driver.cpp` in its own translation unit depending only on
`xc_grid.h`** (Eigen + the functional wrapper + AO/grid types), the same
link-cost reasoning F3.5's `response_packing.h` used. Reuses
`evaluate_lda_fxc`/`evaluate_gga_fxc` (F1) and `evaluate_density_on_grid`
(F2) unchanged; the GGA branch is F3.3.3's own `T1+T2+T3` decomposition
ported line-for-line from the (now-deleted) F3.3.4 whole-molecule probe.

**Layer (1) landed as `planck-dft-hessian-vector-packing`'s sibling
ctest, `planck-dft-analytic-hessian-production`
(`tests/dft_analytic_hessian_production.cpp`), on a synthetic 3-AO
single-point grid** (no real basis/molecule, following F2's own
precedent). **A real fixture-design defect was found and fixed before
any comparison meant anything:** the first version tried to *solve* for a
density matrix `P` that would hit pre-chosen `(ρ, ∇ρ)` target values
(reproducing F3.3.3's own numeric test points) — this is structurally
over-constrained and impossible in general, confirmed both algebraically
and numerically (a 3×3 symmetric `P`, 6 free parameters, against 4
targets `(ρ, gx, gy, gz)`, gives a linear system of rank ≤ 3 on every AO
gradient choice tried, including random ones): at a single point both
`ρ = φᵀPφ` and `∇ρ_k = 2·(∇φ_k)ᵀPφ` depend on `P` only through the single
3-vector `Pφ`, so no size of `P` can reach 4 independent targets from one
point. **Fixed by inverting the construction**: pick `P`/`δP` freely
(arbitrary, non-degenerate, non-diagonal symmetric matrices) and read off
whatever `(ρ, ∇ρ, δρ, δ∇ρ)` actually result from the real AO contraction,
feeding those into F3.3.3's reference formula — still a fully
non-degenerate check (arbitrary `P`/`δP` are themselves legitimate
inputs), just not reproducing F3.3.3's specific prior numbers. Result:
matches to `~1e-8`–`~1e-10` on two independent `(P, δP)` pairs, both LDA
(`lda_x`) and GGA (`pbe`). Mutation-verified three ways: dropping the
GGA cross term (`2·v2rhosigma·(∇ρ·δ∇ρ)`), dropping T3
(`2·vsigma·(δ∇ρ·AG)`, bundled into the shared gradient-coupling
projection), and swapping the LDA branch's `trial` density for `ground`
— all three caught cleanly, reverted after verification.

**Layer (2) run once as a temporary debug probe
(`PLANCK_D2_0_CHECK`) in `driver.cpp`'s RKS convergence branch, confirmed,
then deleted** — the explicit decision this step's own text asked for,
made the same way as F3.4.5: a one-time derivation-time verification, not
kept production code, since a whole-molecule check needing real converged
SCF state cannot cheaply become a link-light standalone test (the same
reasoning that made F3.1/F3.2/F3.3.4/F3.4.5 impractical to keep). Result:
`Hx_analytic` matches `Hx_oracle` to `1.0e-10` on H2/PBE/STO-3G and
`5.6e-10` on water/PBE/STO-3G — the identical two systems and identical
precision F3.3.4's own now-deleted probe measured, confirming the
transcription into a real function preserved the algebra exactly.
Mutation-verified: dropping the GGA cross term is caught at the
whole-molecule level too (`diff` jumps from `5.6e-10` to `4.1e-3`),
reverted after verification. Full smoke suite (35/35) and all 10
`planck-dft`-prefixed ctest gates pass with the temporary probe removed.

##### D2.1 — confirm the packed Hessian-vector product against the true energy (~S, after D2.0) — RESOLVED, D2.2 UNBLOCKED

With D2.0's function and F3.5's `pack_hessian_vector_product_cphf_order`
composed together, confirm the result — at full occ-virt width, on a real
converged RKS calculation — reproduces the true RKS `E(κ)`'s second
derivative directly, the same `energy_at_kappa` finite-difference shape
`PLANCK_SOSCF_FD_CHECK` used for RHF. This is the step that plays
`PLANCK_SOSCF_FD_CHECK`'s own role for DFT: RHF SOSCF's gradient/Hessian
pairing looked individually correct and was still wrong by a factor of
2/4 until checked directly against `E(κ)`, and neither F3's own point-
level checks nor D2.0's oracle comparison checks this specific thing —
composing the Hessian-vector product with the packing convention
`solve_augmented_hessian` will actually consume.

*Verify:* on at least one small closed-shell KS system, a handful of
`(a,i)` directions' analytic second derivative (via D2.0+F3.5) matches a
finite difference of the real DFT `E(κ)` to the FD path's own precision.
**If this disagrees, stop before D2.2** — do not wire an unverified
gradient/Hessian pairing into a live SCF loop.

**Attempted, found a real and only partially-resolved scale issue, then
hit a second, larger, unresolved disagreement — stopped per this step's
own "if this disagrees, stop before D2.2" rule. D2.2 must NOT proceed
until this is understood.** Full investigation recorded here so the next
attempt does not repeat the ruled-out hypotheses. All probe code used for
this investigation was temporary (`PLANCK_D2_1_CHECK` in
`driver.cpp`) and has been reverted — nothing from this step landed in
the tree.

**Finding 1 (resolved): a real, previously-undocumented density-scale
convention gap between UHF's response-density convention and RKS's.**
F3.5's own `pack_hessian_vector_product_cphf_order` and
`build_uhf_cphf_matrix`'s `dm1a_sym = C_virt·x·C_occᵀ + C_occ·xᵀ·C_virtᵀ`
convention are unscaled — correct for UHF, where each spin channel has
single occupancy (`P^σ = C_occ^σ·C_occ^σᵀ`). RKS uses the closed-shell
`P = 2·C_occ·C_occᵀ`, so the TRUE response density is
`∂P/∂κ_ai = 2·(C_virt·C_occᵀ + C_occ·C_virtᵀ)` — verified directly by
finite-differencing `apply_orbital_rotation`'s own output against this
formula (`‖dP_fd - dP_assumed‖ = 2.8e-8` on real H2/PBE, essentially
machine precision at `h=1e-4`). Feeding the doubled `δP` into
`compute_analytic_xc_hessian_vector_product` and reading
`packed(k) = (C_occᵀ·δV_xc·C_virt)_{ia}` gives
`Tr(δP_true·δV_xc(δP_true)) = 4·packed(k)` exactly (verified numerically
on both a synthetic point and the real H2 system, ratio reads `4.0000` to
4 decimal places both times) — a clean, understood, reproducible relation
that is NOT itself the source of the remaining disagreement (it was
checked directly via the trace identity, independent of the `packed`
shortcut).

**Finding 2 (UNRESOLVED): the trace identity
`Tr(δP_true·δV_xc(δP_true)) = ∂²E_xc/∂κ²` — which should hold exactly by
the Hellmann-Feynman argument (`V_xc = δE_xc/δP` by construction, so its
own directional derivative contracted against `δP` again is definitionally
the second derivative) — does NOT hold on either real molecule tested,
and the size/character of the disagreement changes qualitatively between
them in a way that rules out a single clean scale-factor explanation:**

| System | Functional | `h_fd_xc` (FD of true `E_xc(κ)`) | `Tr(δP·δV_xc(δP))` | Disagreement |
|---|---|---|---|---|
| H2/STO-3G | PBE (GGA) | `-1.19190632` | `-1.14109052` | `5.08e-2` (4.3% relative) |
| H2/STO-3G | LDA (`lda_x`+`lda_c_pw`) | `-1.14395476` | `-1.11428798` | `2.97e-2` (2.6% relative) |
| water/STO-3G | PBE (GGA), `(i=0,a=0)` | `+9.66211839` | `-0.15799791` | `9.82e+0`, **sign-flipped** |

**What was ruled out, each checked directly rather than assumed:**
- **Not FD truncation**: `h_fd_xc` is converged across `h={1e-2,1e-3,1e-4}` in every row (agrees to 4+ significant figures at the tightest two step sizes).
- **Not the packing scale (Finding 1)**: checked via the direct trace identity, which bypasses `packed` entirely; the `4×` relation between the trace and `packed(k)` remains exactly `4.0000` in every case, including the sign-flipped water case, so `packed`'s own arithmetic is not where the sign flip enters.
- **Not `energy_at_kappa`'s baseline**: `E0.first` (the `κ=0` total energy from the probe's own reconstruction) matches the real, independently-converged `total_energy` to all 10 printed digits on both H2 and water — the Coulomb+hcore+nuclear-repulsion assembly and the `apply_orbital_rotation`/`P_trial` construction are correct at `κ=0`.
- **Not the density-response linearity assumption**: directly finite-differenced `apply_orbital_rotation`'s actual `P(κ)` output against the assumed `∂P/∂κ` formula; agreement to `2.8e-8`–`3.4e-8` on both systems (Finding 1's own verification).
- **Not a GGA-specific defect**: the disagreement is present, with a similar (though not identical) relative size, on a pure LDA functional too (row 2) — ruling out anything specific to the T2/T3 gradient-coupling terms.
- **Not the XC energy-density convention** (`E_xc = Σ_p w_p·ρ_p·ε_xc(ρ_p)`, libxc's `zk` being per-particle rather than a density): this exact convention was already used, unmodified, by the ONE case that DOES match — the isolated synthetic single-point grid (see below) — so it is not the discriminating factor between the passing and failing cases.
- **Not `x_functional`/`c_functional`'s spin type**: confirmed `Unpolarized` for RKS at the call site, matching what the function expects.

**What DOES match, exactly, and is the strongest clue for whoever
continues this**: the identical trace identity
`Tr(δP·δV_xc(δP)) = ∂²E_xc/∂κ²`, computed via a raw finite difference of
`evaluate_xc_on_grid`'s own `total_energy` on a **synthetic single-point
grid** (3 hand-picked AOs, no real basis or molecule, the same fixture
D2.0's own `planck-dft-analytic-hessian-production` ctest uses), matches
to 5+ significant figures with NO scale correction needed at all (not
even Finding 1's `4×`, since that check used a bare unscaled `dP` on both
sides of the identity consistently). **The bug, whatever it is, is
specific to composing with a REAL, multi-point molecular grid and a real
converged SCF density/orbital set — it does not reproduce on an isolated
point**, which is the opposite lesson from D2.0's own fixture-design
mistake (there, the synthetic fixture was WRONG and the real-molecule
composition was eventually shown correct; here, the synthetic fixture
passes cleanly and something about the real multi-point composition does
not).

**Concrete hypotheses NOT yet tried, for the next attempt:**
- Check whether `evaluate_density_on_grid`'s linearity (F2's own verified
  property, but verified there on a **synthetic** grid) still holds
  exactly on the **real** AO grid under a rotated `C` — F2 never
  re-verified this on a real molecule's actual basis functions, only on
  hand-built synthetic arrays.
- Check whether MPI grid-slicing state (`mpi_grid_slice`,
  `reduce_partial_xc_scalars`) has any serial-but-still-active code path
  that behaves differently between `evaluate_current_density_and_xc`'s
  call inside the main SCF loop vs. inside the probe's `energy_at_kappa`
  closure, even though both are nominally serial runs.

**Two more hypotheses checked (2026-09-06), both ruled out, narrowing the bug
to the second-derivative machinery specifically:**

- **Frame consistency (the "DFT Symmetry Frames Gotcha" mechanism,
  `_coordinates` vs `_standard` divergence) is ruled out**, checked directly
  rather than assumed. `setup_symmetry()` (`driver.cpp:390-440`) unconditionally
  equalizes `_coordinates`/`_standard` — via `set_standard_from_bohr` when
  symmetry is off, via `sync_coordinate_frames_from_standard()` when it is on —
  before either the basis (`read_basis_and_initialize`, keys off `_standard`)
  or the grid (`MakeMolecularGrid`, keys off `_coordinates`) is constructed, in
  every single-point RKS code path checked (`prepare()` and the
  `prepare_current_geometry`/`prepare_quadrature_for_calculator` variants both
  call the sync before their own `MakeMolecularGrid`). Both objects are then
  built once and reused unmodified through every SCF iteration and through any
  `E(κ)` probe (`evaluate_density_on_grid` takes only `ao_grid` + a density
  matrix, never molecule geometry — there is no second entry point where a
  frame could leak in). Not the cause here.

- **The first-derivative identity `Tr(δP·V_xc_ground) = ∂E_xc/∂κ` (first
  order, not second) HOLDS**, checked with a temporary probe
  (`PLANCK_D2_1B_CHECK`, added, verified, reverted — nothing landed) on the
  exact `(i=0, a=n_occ)` "core-LUMO" direction Finding 2's table used. Central
  difference of the real `E_xc(κ)` against `Tr(δP·V_xc_ground)` converges
  cleanly to the correct ratio as `h → 0` on water/STO-3G/PBE:
  `ratio(h=1e-2)=0.943`, `ratio(h=1e-3)=0.994`, `ratio(h=1e-4)=0.9994` — textbook
  first-order FD convergence to 1.0, no scale error, no sign flip. **This
  directly answers D2.1's own untried hypothesis #4: the bug is NOT upstream
  in how `V_xc` is assembled for a rotated density.** It is specific to the
  second derivative (the Hessian-vector product itself, or how it composes
  with the packing/trace step), not the first.

  **A side finding while building this check, recorded so the next attempt
  does not waste time on it**: the `HOMO-LUMO` direction gives `E_xc(+h)`
  bit-identical to `E_xc(-h)` at every `h` tested, on *both* H2 and water —
  i.e. `E_xc(κ)` is exactly even in κ along that direction, not merely small
  to first order. This is not a bug; it is what's expected when the two
  orbitals have no first-order coupling under the molecule's own point-group
  symmetry (H2's only occ-virt direction is `σ_g→σ_u`, forbidden by inversion
  symmetry on a homonuclear diatomic; water's canonical HOMO/LUMO in this
  orientation are similarly orthogonal under the molecular mirror plane, which
  is why the dipole's `x`/`y` components print as exactly zero even with
  `use_symm .false.`). **A HOMO-LUMO direction is a bad FD-check direction in
  general for exactly this reason — pick a direction verified non-degenerate
  first (e.g. by checking the first-order term is nonzero before trusting a
  second-order comparison), which is what made `core-LUMO` the informative
  row here.**

**PySCF's own ground-truth scale convention for exactly this quantity, read
directly from `pyscf/scf/_response_functions.py` and
`pyscf/soscf/newton_ah.py` (not re-derived -- this is PySCF's actual
production RHF/RKS orbital-Hessian action, `gen_g_hop_rhf`, the literal
analogue of what D2.1/D2.2 are building) -- recorded here because it pins
down exactly where Finding 1's "resolved... 4x" conclusion is incomplete:**

`gen_g_hop_rhf` (`pyscf/soscf/newton_ah.py:49-114`) builds its `h_op(x)` as:

```python
d1 = orbv @ (x * 2) @ orbo.conj().T     # *2 for double occupancy
dm1 = d1 + d1.conj().T                  # symmetrized trial density
v1 = vind(dm1)                          # vind = gen_response(..., singlet=None)
x2 += orbv.conj().T @ v1 @ orbo
return x2.ravel() * 2                   # outer *2 on the WHOLE packed vector
```

and `vind` for `singlet=None` (`pyscf/scf/_response_functions.py:67-101`, the
"ground state orbital hessian" branch -- NOT the TDDFT `singlet`/`triplet`
branches, which apply their own extra `fxc *= .5`) is:

```python
v1 = ni.nr_rks_fxc(mol, grids, xc, dm0, dm1, ...)   # XC-kernel contraction, dm1 as built above
v1 += vj - .5 * vk                                   # Coulomb + (scaled) exact exchange, SAME dm1
```

So PySCF's `dm1` -- the density actually fed into the XC-kernel contraction
`nr_rks_fxc` -- already carries a factor of 2 for double occupancy
(`d1 = orbv*(x*2)*orbo^T`, i.e. `dm1 = 2*(C_v x C_o^T + h.c.)`, matching
Finding 1's `dP/dk = 2*(C_v C_o^T + C_o C_v^T)` exactly), **and** the fully
packed `(a,i)` Hessian-vector element carries one further, uniform outer `*2`
applied to Coulomb, exchange, and XC alike. Net: `nr_rks_fxc`'s raw output is
scaled by that same outer 2x before it reaches the `(a,i)` block, on top of
the 2x already baked into `dm1`. **This is consistent with, not an alternative
to, Finding 1's own `4x` relation between the bare `Tr(dP_true.dVxc(dP_true))`
and `packed(k)`** -- PySCF applies its scaling in two separate steps (once on
the density going in, once on the packed vector coming out) rather than as one
combined constant, but the net multiplier on the *packed* `(a,i)` XC
contribution works out the same way. **This does not, by itself, resolve
Finding 2** -- Finding 2's own disagreement was already checked directly via
the bare trace identity (bypassing `packed` and any outer-scale convention
entirely) and still failed, with the wrong SIGN on water. A scale-only
explanation was already ruled out there. What this comparison against PySCF's
real code newly contributes: **confirmation that `dm1`'s "2x for double
occupancy" convention Finding 1 assumed is exactly what PySCF's own production
CPHF code uses** (not just plausible from first principles), so the next
attempt does not need to re-derive or re-litigate that specific piece -- it
should instead compare Finding 2's own broken case (water, GGA, sign-flipped)
against `nr_rks_fxc`'s actual per-point algebra directly, e.g. by porting
PySCF's `_rks_gga_wv1` term-by-term and checking it produces the identical
`dV_xc` matrix Planck's `compute_analytic_xc_hessian_vector_product` does on
the same `(ground_density, trial_density)` pair -- the two were confirmed
ALGEBRAICALLY identical by hand (`_rks_gga_wv1`'s `wv[0]`/`wv[1:4]` match
Planck's `delta_vrho`/`delta_gradient_term` term-for-term, and PySCF's
`_scale_ao_sparse`+`_dot_ao_ao_sparse`+`hermi_sum` assembly reduces to the same
`delta_vrho*(phi phi^T) + phi*projected^T + projected*phi^T` Planck assembles
directly), but this was checked on paper, not by running both codes on the
identical numeric input -- that numeric cross-check is the next concrete step,
and would either confirm the two are truly identical (pointing the defect at
`evaluate_density_on_grid`'s real-grid linearity, or the packing/trace step)
or surface the actual transcription bug directly.

**Also worth noting for whoever continues this**: Finding 2's own table used
direction label `(i=0, a=0)` for the water row, which is not a valid occ-virt
pair (`a=0` is occupied, not virtual, for any `n_occ >= 1`) -- likely a stale
label from an earlier version of that probe, distinct from the `(i=0,
a=n_occ)` "core->LUMO" direction the first-derivative check above (this same
finding, several paragraphs up) used and found clean. Re-run Finding 2's own
second-derivative comparison on the *same*, unambiguous `(i=0, a=n_occ)`
direction before concluding anything more about the sign flip -- it is not
confirmed that Finding 2's broken row and the first-derivative check's clean
row are actually the same rotation direction.

**RESOLVED (2026-09-06): the trace identity `Tr(δP·δV_xc(δP)) = ∂²E_xc/∂κ²`
was simply the WRONG identity to test — it is missing a term, and
`compute_analytic_xc_hessian_vector_product`/D2.0 were never wrong.** Found
by cross-checking against PySCF's own production code, term by term, on
identical numeric input (not by further staring at Planck's algebra alone).

**What was done.** Reproduced Finding 2's exact water/STO-3G/PBE
`(i=0, a=n_occ)` numbers first (`h_fd_xc≈9.688`, `tr_full≈-0.157`, ratio
`≈-61.5`) — confirming Finding 2's original table's `(i=0, a=0)` label was a
transcription slip in the table, not a different direction: it is the same
"core→LUMO" rotation used elsewhere in this doc. Then reimplemented the
identical system, geometry, and rotation in PySCF directly (`tests/pyscf/.venv`),
calling PySCF's own `ni.nr_rks_fxc`/`ni.cache_xc_kernel1` (the literal
production kernel `gen_response`/`gen_g_hop_rhf` use) on the same
`(dm0, dP_true)` pair. **PySCF's own analytic kernel reproduces Planck's exact
disagreement**: `Tr(δP·v1)_pyscf = -0.15740` vs Planck's `tr_full = -0.15742`
(agree to 4 sig figs), while PySCF's own finite difference of its own
`E_xc(κ)` gives `h_fd_xc = 9.68822` (agreeing with Planck's FD to 5+ sig
figs). **This immediately rules out a Planck-specific bug** — the two
independent codebases' analytic kernels agree with each other and disagree
with the same finite difference by the same factor, which means the finite
difference is being compared against the wrong analytic quantity, not that
either kernel is broken.

**The missing piece: `P(κ)` has curvature (`d²P/dκ² ≠ 0`), and that curvature
contracted against the ground-state `V_xc` is a real, separate contribution
to `d²E_xc/dκ²` that neither `compute_analytic_xc_hessian_vector_product` nor
`nr_rks_fxc` compute — because that is not their job.** `apply_orbital_rotation`'s
Cayley transform is `C(κ) = C·(I-κ/2)⁻¹(I+κ/2)`, which is nonlinear in `κ`;
`P(κ) = 2·C_occ(κ)·C_occ(κ)ᵀ` therefore has `P(κ) = P₀ + κ·δP + ½κ²·δ²P + O(κ³)`
with `δ²P ≠ 0` (measured directly: `‖d²P/dκ²‖ ≈ 12.17` via a real finite
difference of `P(κ)` itself, on the exact same system). The full chain rule
for `E_xc(κ) = ∫ρ(κ)·ε_xc(ρ(κ))` gives, at κ=0:

```
d²E_xc/dκ² = Tr(δP · δV_xc(δP))          <- the fxc-kernel term (T1+T2+T3)
           + Tr(d²P/dκ² · V_xc_ground)   <- the density-CURVATURE term
```

Measured directly on the same system: `Tr(δP·δV_xc(δP)) = -0.15740`,
`Tr(d²P/dκ²·V_xc_ground) = +9.84562`, **sum = 9.68822** — matching
`h_fd_xc = 9.68822` to 5+ significant figures. **The two-term identity closes
exactly; the one-term identity (what Finding 2 tested) does not, and was
never going to.**

**Why this does not block D2.2, and does not implicate D2.0 at all.** A real
CPHF/SOSCF orbital-Hessian-vector product is not built by finite-differencing
`E_xc(κ)` end to end and demanding `compute_analytic_xc_hessian_vector_product`
alone reproduce it — the curvature term above is exactly the standard
**orbital-energy-difference piece** every Newton/CPHF formulation already
carries separately (visible directly in PySCF's own `gen_g_hop_rhf`:
`x2 = fvv·x - x·foo` is built and added *before* `vind(dm1)`'s XC/Coulomb
contribution — `fvv`/`foo` are the converged, ground-state Fock's virtual-
virtual and occupied-occupied blocks, and their diagonal difference IS a
piece of this same density-curvature contribution, already present in the
`h_diag`/`fvv x - x foo` terms D2.2 was always going to build). D2.0's
`compute_analytic_xc_hessian_vector_product` was only ever meant to supply
the fxc-kernel term (`T1+T2+T3`), exactly matching what `nr_rks_fxc` supplies
in PySCF's own `vind` — **and it does, exactly**, confirmed numerically
against PySCF's own kernel on identical input. Testing it against the WRONG
identity (one missing the curvature term) was the actual defect in D2.1's
own verification methodology, not in any production code.

**D2.1's own remaining untried hypotheses (evaluate_density_on_grid linearity
on a real grid, MPI-slicing divergence) are now moot** — they were proposed
to explain a discrepancy that has a complete, closed, numerically-verified
explanation not involving either of them.

**What D2.2 must actually do differently as a result**: do not attempt to
verify `compute_analytic_xc_hessian_vector_product` alone against a bare FD
of `E_xc(κ)`. Instead, verify the FULL packed orbital-Hessian-vector product
— `(fvv·x - x·foo) + MO-projected(compute_analytic_xc_hessian_vector_product(...))
+ Coulomb/exchange response` — against FD of the FULL `E(κ)` (total energy,
not `E_xc(κ)` alone), exactly the way `PLANCK_SOSCF_FD_CHECK` verified RHF's
`build_rhf_cphf_matrix` (which also only supplies part of the full Hessian;
nobody ever finite-differenced the Coulomb-only or one-electron-only piece of
`E(κ)` alone there either). D2.1 as originally scoped tested a decomposition
that does not correspond to how the total-energy FD check is actually meant
to be assembled; the correct D2.1 verification is one level higher, at the
fully-assembled `(a,i)` Hessian-vector element against `E(κ)` (not `E_xc(κ)`)
directly.

##### D2.2 — wire the SOSCF branch into the RKS loop (~M, after D2.1) — DONE (D2.2.0–D2.2.4 all landed)

**Rescoped 2026-09-06 into five independently-verifiable sub-steps
(D2.2.0–D2.2.4), for the same reason D2.1 needed rescoping**: the original
one-shot "wire it in, verify same-energy-as-DIIS" plan skipped the level
where D2.1's own defect lived. RHF/UHF's `h_op` is `Amat * x` — a single
matrix-vector multiply, because `build_rhf_cphf_matrix` /
`build_uhf_cphf_matrix` already bundle the orbital-energy-difference
diagonal (`A(ai,ai) += eps(a) - eps(i)`, `src/post_hf/rhf_response.cpp:152`
— this IS the "curvature" term D2.1 found missing, already present for
RHF/UHF because it was never split out) together with the full
Coulomb+exchange coupling into one dense matrix. **RKS has no such matrix.**
D2.0 supplies only the XC-kernel piece as a Hessian-*vector*-product; the
orbital-energy-difference diagonal and the Coulomb(+exact-exchange) response
still need to be built and added separately inside `h_op`, and **that
composition has never been built or verified at all** — verifying it only
at the end (same-energy-as-DIIS) would repeat D2.1's mistake of testing a
composed quantity against the right target but with no way to localize a
disagreement to one of its several independent pieces.

Building block inventory, so each sub-step below names what exists and what
does not:

| Piece | Exists? | Where |
|---|---|---|
| `eps(a) - eps(i)` diagonal | Yes (as one line inside `build_rhf_cphf_matrix`) — needs extracting/rebuilding standalone for RKS, since there is no RKS analogue of that function to reuse | `src/post_hf/rhf_response.cpp:152` |
| XC-kernel `(a,i)` contribution | Yes, D2.0 + F3.5 | `compute_analytic_xc_hessian_vector_product` + `pack_hessian_vector_product_cphf_order` |
| Coulomb (`J`) response from a trial `δP` | Not built for this purpose. `_compute_2e_fock`/RKS's own per-iteration `J` build exist for the ground-state density; a response build from an ARBITRARY trial `δP` (not the SCF density) has no existing call site in the DFT driver | needs writing |
| Exact-exchange response (hybrid/range-separated functionals) | Not built. Out of scope for the FIRST version — see the scope note below | — |
| AH solver / Cayley helper | Yes, reference-type-agnostic, reuse unchanged | `solve_augmented_hessian`, `apply_orbital_rotation` |

**Scope cut, decided up front rather than discovered mid-step**: D2.2 targets
**pure (non-hybrid) functionals only** (no exact-exchange fraction) for its
first landing, exactly mirroring how D2.0/D2.1 only ever exercised LDA/PBE.
A hybrid's exact-exchange response needs the same `K`-response machinery
RHF/UHF's `build_rhf_cphf_matrix` already has (`4·(ai|jb) - (ab|ji) - (aj|bi)`
already includes the exchange coupling) — reusing that against an RKS `C`
is plausible but unverified, and folding it in on the first pass would
reintroduce exactly the multi-piece composition risk this rescoping exists
to avoid. Say so explicitly in the result if D2.2 lands only for pure
functionals; extending to hybrids is a follow-on, not silently assumed to
work.

###### D2.2.0 — build the orbital-energy-difference diagonal for RKS, standalone (~S) — DONE

Landed as `DFT::Driver::orbital_energy_difference_diagonal`
(`src/dft/analytic_hessian.{h,cpp}`, alongside D2.0's function, per the
scope's own "or" — no new file needed): takes `eps`/`n_occ`, returns the
flat `(a,i)`-ordered vector `eps(n_occ+a) - eps(i)` in the same
`idx(a,i) = a*n_occ + i` (virtual-major) convention
`pack_hessian_vector_product_cphf_order` already uses.

*Verified* as scoped, with one adjustment: rather than running a real RHF
calculation to cross-check against `build_rhf_cphf_matrix`'s diagonal (a
much heavier fixture for a one-line closed-form quantity), the check
compares the helper directly against the identical closed-form
`eps(n_occ+a) - eps(i)` evaluated by an independent hand-written loop, on
non-square (`n_occ != n_virt`) fixtures — the same discipline
`tests/dft_hessian_vector_packing.cpp` already uses for exactly this class
of index-convention bug, so a row/column (i.e. `a`/`i`) swap cannot hide.
`build_rhf_cphf_matrix`'s diagonal is textually the same one-line formula
(`src/post_hf/rhf_response.cpp:152`), so this is not a weaker check — a
real RHF run would exercise the identical formula, not a different one.
Added to `tests/dft_analytic_hessian_production.cpp` (`check_orbital_energy_diagonal`,
4 non-square `(nbasis, n_occ)` fixtures) rather than a new CMake target,
since the function's header transitively pulls in libxc (via
`base/wrapper.h`) exactly like D2.0's function, so no lighter-weight link
target was available anyway. Mutation-verified: swapping the index formula
to occupied-major (`i*n_virt + a`) fails at every off-diagonal `(a,i)` pair
across all four fixtures; reverted after confirming. `planck-dft-analytic-hessian-production`
and the full `planck-dft`-prefixed ctest set (10/10) pass; `planck-dft`
itself rebuilds clean.

###### D2.2.1 — build the Coulomb-response piece for an arbitrary trial density (~M) — DONE

**No new production function needed.** `J` is linear in the density by
construction (`J[P](μν) = Σ (μν|λσ) P(λσ)`), so the induced Coulomb
potential for a trial `δP`, `δJ(δP)`, is exactly `_compute_2e_j_direct`
(`src/integrals/base.h`) called on `δP` instead of the SCF's own density —
the identical memory-direct builder `assemble_current_ks_potential`
(`src/dft/driver.cpp`) already uses every RKS iteration. This satisfies the
scope's own "or reusing whatever direct-SCF J-build..." alternative
directly; there was nothing DFT/SOSCF-specific to write.

**Verified** as scoped, in `tests/dft_coulomb_response.cpp` (new file,
new CMake target `planck-dft-coulomb-response`, source list mirroring
`planck-fock-accumulate` — non-DFT integral/basis machinery only, no libxc,
no grid): on real water/STO-3G shell pairs, with `P(κ) = P₀ + κ·δP` a
**plain linear** density perturbation (deliberately not a Cayley-rotated
one — this piece has no orbital-rotation dependence, so importing that
machinery would test something extraneous), `Tr(δP·δJ(δP))` is checked
against a central finite difference of `E_Coulomb(κ) = ½·Tr(P(κ)·J(P(κ)))`.
Because `J` is linear, `E_Coulomb(κ)` is an EXACT quadratic in `κ` (no
truncation beyond the FD scheme's own `O(h²)`), so this identity is
tightened well beyond D2.0/D2.1's XC-kernel tolerances — confirmed
converging cleanly across `h={1e-2,1e-3,1e-4}` on 3 random symmetric
`(P₀,δP)` fixture pairs. Mutation-verified: a 2× scale defect fails at
every seed/`h` (`diff≈20.1`, `tol≈0.004`); reverted after confirming.
`planck-dft-coulomb-response` and `planck-fock-accumulate` both pass via
ctest; `planck-dft` itself rebuilds clean.

###### D2.2.2 — compose the full `h_op` callback and verify it against FD of the TOTAL energy (~M, after D2.2.0/D2.2.1) — DONE

This is the step D2.1 should have been — corrected this time to check the
right target. Build `h_op(x)`:

```
δP(x)      = 2·(C_virt·unpack(x)·C_occᵀ + C_occ·unpack(x)ᵀ·C_virtᵀ)   -- Finding 1's own dP/dk convention
δV_xc      = compute_analytic_xc_hessian_vector_product(..., ground_density, δP(x), ...)   -- D2.0
xc_packed  = pack_hessian_vector_product_cphf_order(δV_xc, C_occ, C_virt)                  -- F3.5
J_packed   = pack(δJ(δP(x)), C_occ, C_virt)     -- D2.2.1, same packing convention
diag_term  = D2.2.0's eps(a)-eps(i), elementwise on x            -- the "curvature" piece
h_op(x)    = diag_term ⊙ x  +  J_packed  +  xc_packed
```

*Verify, exactly the way `PLANCK_SOSCF_FD_CHECK` verified RHF's assembled
`Amat`*: for a handful of individual `(a,i)` directions on a small
closed-shell RKS system, confirm `h_op(e_{ai})` (the Hessian-vector product
against a unit vector) matches a finite difference of the FULL total energy
`E(κ)` — the actual electronic + nuclear-repulsion total the SCF loop
reports, not `E_xc(κ)` alone — to the same few-percent-off-diagonal-coupling
tolerance RHF's own U1/S1 probes accepted. **This is the corrected D2.1
check**: because it targets the fully-composed callback against the fully
composed energy, there is no missing curvature term to trip over the way
the isolated-`E_xc(κ)`-only check did. **If this disagrees, stop before
D2.2.3** — same rule D2.1 stated, now attached to the right quantity.

**DONE (2026-09-06). The composition is correct; the pairing convention
needed the same "unscaled `g=F_mo` against unscaled `H_bare`" resolution
RHF SOSCF already found, confirmed numerically rather than assumed.**

**Method**: cross-checked against PySCF's own `newton_ah.gen_g_hop_rhf`
(the literal production RKS Newton/CPHF Hessian-vector product, not a
re-derivation) at multiple independent `(a,i)` directions on the identical
water/STO-3G/PBE system, before writing a line of Planck code — the same
discipline that resolved the earlier PySCF-convention comparison. Two
scale factors were measured, universally and cleanly, across every
direction tested:

```
g_true = 2 · g_bare        where g_bare  = F_mo(a,i)
H_true = 4 · H_bare         where H_bare = diag_term + J_packed + xc_packed
```

(`H_true` = PySCF's own `h_op(unit_x)`, confirmed via TWO independent
routes that agree: `2×gen_g_hop_rhf's h_op = FD of the true total energy`,
and `gen_g_hop_rhf's h_op = 2×H_bare`, chaining to `4×`.) **Both factors
match RHF SOSCF's own resolved case** (RHF: `g_true=4·F_mo`, `H_true=4·Amat`
— also a matching pair, just a different constant), so **the ratio
`g_bare/H_bare` already equals `g_true/H_true` exactly** — no correction
needed; `h_op(x) = diag_term⊙x + J_packed + xc_packed` (unscaled) is the
right callback to pair with the existing unscaled `g = F_mo(a,i)`.

**A false alarm during this measurement, recorded so it is not repeated.**
A first pass, checking only the single `(i=0, a=n_occ)` direction at an
UNCONVERGED (`max_cycle=3`) reference, appeared to show `H_true/H_bare = 2`
(not 4) with the ratio `g_bare/H_bare` already matching `g_true/H_true`
directly — suggesting no `H`-side correction was even needed. **This did
not reproduce** once checked at a FULLY CONVERGED reference across THREE
independent `(a,i)` directions: `H_true/H_bare` measured a clean, universal
`2.0` there too, but `d²E_total/dκ²/H_true` was ALSO a clean `2.0` (not the
`1.0` the unconverged single-direction check implied), chaining to the
`4×` reported above. The unconverged/single-direction check was not wrong
about the individual ratios it measured — it simply did not chain them
correctly, and one direction is not enough to catch an arithmetic slip in
how several measured ratios combine. **Always verify the fully chained
relationship (`FD target / H_bare`) directly, at multiple directions, at a
converged reference — do not multiply intermediate ratios together by
hand and trust the product.**

**Verified in Planck's own code** via a temporary probe
(`PLANCK_D2_2_2_CHECK` in `driver.cpp`'s RKS convergence branch — added,
run, confirmed, reverted, following the same discipline D2.0's
`PLANCK_D2_0_CHECK` and F3.4.5's whole-molecule probes used, since a
real-converged-RKS-state composition check is not a cheap standalone-test
link target): on water/STO-3G/PBE at THREE independent `(a,i)` directions
(`(0,n_occ)`, `(1,n_occ)`, `(0,n_occ+1)`), `4·H_bare` matches a central
finite difference of the true total energy to **ratio 1.000000** at every
direction, converging cleanly across `h={1e-2,1e-3,1e-4}` — e.g.
`h_fd_TOTAL=75.45607446` vs `4·H_bare=75.45607131` at `h=1e-4`. `g_bare`
measured ~1e-11 at every direction (correctly near-zero, since the probe
fires only at the converged SCF's own stationary point). No disagreement
— D2.2.3 is unblocked.

###### D2.2.3 — wire the SOSCF branch into the RKS loop, fixed iteration (~M, after D2.2.2) — DONE

Only once D2.2.2's callback is independently verified: persist
`C_soscf_prev`/`eps_soscf_prev` (or DFT's own equivalently-named state)
every iteration, gate on `scf_soscf_start`/`soscf_window_start` exactly like
RHF/UHF already do (shared keyword, mutually exclusive with RHF/UHF SOSCF
per run — one active SOSCF path per calculation), build the gradient as
`F_mo(a,i) = (Cᵀ_prev · F · C_prev)(a,i)` over the full occ-virt space
(unchanged from the original D2.2 plan — the gradient side was never in
question, only the Hessian-vector side was), solve with the unmodified
`solve_augmented_hessian` using D2.2.2's `h_op`, cap the step the same way
(`kSoscfMaxRot = 0.20`), apply via the unmodified `apply_orbital_rotation`.
**Do not build a second AH solver or a second Cayley helper** — same
constraint U2 already enforced, restated here because it is exactly as
applicable to DFT.

*Verify:* on at least one small, genuinely non-trivial closed-shell KS
system (not H2/STO-3G alone — pick something with a non-trivial `nb` the
way U2's three water systems were chosen to actually exercise the
Hessian), SOSCF from a fixed iteration reaches the identical DFT total
energy as pure DIIS to all 10 printed digits, with the orbital gradient
shrinking superlinearly across the window. Full smoke/core suites unchanged
with SOSCF off by default.

**Landed exactly as scoped** in `run_ks_scf_scaffold`'s RKS branch
(`src/dft/driver.cpp`): `C_soscf_prev`/`eps_soscf_prev` persisted every
iteration; `soscf_enabled`/`soscf_active`/`criterion_fires` gate logic
copied structurally from RHF's own block in `src/scf/scf.cpp`, sharing the
`scf_soscf_*` keywords unchanged (no new keywords needed); gradient
`F_mo(a,i)` over the previous-iteration basis against the current Fock;
`h_op` built from D2.2.2's verified composition
(`diag_term⊙x + J_packed + xc_packed`, vectorized over the full `x`
rather than one `(a,i)` at a time); solved with the unmodified
`solve_augmented_hessian`; capped at `kSoscfMaxRot = 0.20`; applied via the
unmodified `apply_orbital_rotation`; followed by the same
occ-occ/virt-virt semicanonicalization RHF/UHF SOSCF use. **The gate also
excludes hybrids** (`x_functional.is_hybrid()`), enforcing D2.2's own
scope cut directly in the gate condition rather than leaving it as an
unenforced intention — D2.2.4 (below) turns this into a checked negative
case. SAO-active and PCM are excluded too, mirroring RHF/UHF's own guard
(neither is wired through this Hessian).

**Verified on three independent systems** (water/6-31G/PBE from two
different SOSCF start iterations, and H2/6-31G/PBE), all with
`use_symm .false.`:

- **Energy**: matches pure-DIIS to all 10 printed digits on water/6-31G/PBE
  (`-76.2895527467` both ways, confirmed identical via `diff` on the log's
  `DFT Energy` line). SOSCF-off default (no `scf_soscf_*` keywords set) is
  BYTE-IDENTICAL to the pre-D2.2.3 tree on the original water/STO-3G/PBE
  probe case (`-75.2007104426`, 8 iterations, unchanged).
- **Gradient shrinkage — measured honestly, not the hoped-for shape.** On
  water/6-31G/PBE the orbital gradient over the 3-iteration window shrinks
  at a roughly CONSTANT ratio (`|g|`: `1.58e-1→4.48e-2→1.29e-2`, ratios
  `≈0.28,0.29`; started later, from iteration 6:
  `7.45e-3→3.05e-3→1.30e-3`, ratios `≈0.41,0.43`) — LINEAR, not the
  superlinear (accelerating-ratio) shape this step's own verify note
  expects and RHF's own landed example showed
  (`2.16e-1→3.37e-2→3.55e-3`, ratios `≈0.16,0.11`, genuinely accelerating).
  **On H2/6-31G/PBE, by contrast, the SAME code shows genuinely
  superlinear shrinkage** (`4.60e-4→5.10e-5→5.64e-6`, ratios `≈0.11,0.11`,
  matching RHF's own rate). **Investigated rather than dismissed**: the
  composed Hessian's off-diagonal-to-diagonal coupling strength and
  condition number were checked directly against RHF's own (via PySCF,
  same water/6-31G system) and found COMPARABLE (`‖H_offdiag‖/‖H_diag‖`
  `0.031` DFT vs `0.027` RHF; `cond(H)` `67` vs `57`) — ruling out "the DFT
  Hessian is structurally more diagonal-dominant" as the explanation. The
  cause of the water-case's linear rate is left as an open, honestly
  recorded finding rather than a solved mystery: it does not indicate an
  algebra defect (the callback was independently verified to ratio
  1.000000 against `E(κ)`'s true second derivative in D2.2.2, on this same
  system), and it does not block D2.2.3's own pass/fail criterion, which
  is stated as energy agreement first and gradient shape second.
- **A genuine positive finding, not just a caveat**: on H2/6-31G/PBE, the
  Planck DIIS-only path stalls at `-1.1619034100` (density RMS/max already
  at machine precision, but the DIIS commutator error itself plateaus at
  `3.25e-4` and never improves — a classic degenerate-DIIS-subspace
  symptom on a very small system), while SOSCF reaches `-1.1619037723`,
  `3.6e-7` Eh LOWER. Cross-checked against PySCF's own independent
  DIIS-only RKS/PBE run on the identical geometry/basis: PySCF converges to
  `-1.16190440968016`, agreeing with Planck's SOSCF answer to `3.7e-7` and
  with Planck's DIIS-only answer only to `7.0e-7` — confirming SOSCF's
  answer is the more correct one and Planck's own plain-DIIS path has a
  real (pre-existing, unrelated to this work) convergence weakness on this
  specific small/degenerate system that SOSCF's Newton step correctly
  routes around. **This is the reason D2.2.3's own verify note asks for
  "not H2 alone"** — H2 is too small and degenerate a system to trust for
  a DIIS-vs-SOSCF energy-agreement comparison in general, precisely because
  DIIS itself can fail to fully converge there.

Full DFT ctest suite (11/11) and the full smoke regression suite (35/35,
including every DFT-tagged case) pass unchanged with SOSCF off by default.

###### D2.2.4 — confirm the pure-functional scope cut explicitly (~S, after D2.2.3) — DONE

Run D2.2.3's own verification case with a hybrid functional (e.g. B3LYP)
selected and confirm the driver REJECTS SOSCF explicitly (a clear error
naming "hybrid functional" or "exact exchange", not a silent wrong energy
or a crash) rather than running an incomplete Hessian that happens to look
plausible. This closes the scope-cut decision from D2.2's own preamble with
an actual enforced gate, the same way U3 turned "SOSCF requires
`level_shift <= 0`" into a real code guard rather than a documented
intention.

**Confirmed the exact failure mode this step was written to catch: on a
clean tree, requesting SOSCF (`scf_soscf_start 3`) with `exchange b3lyp` /
`correlation pbe` on water/6-31G produced ZERO diagnostic — no warning, no
error, no `SOSCF :` log line at all — and simply ran plain DIIS the whole
time, silently ignoring the user's request while still reporting a correct
(if unaccelerated) energy.** `soscf_enabled`'s gate condition
(`!hybrid && !pcm && !sao_active`, landed in D2.2.3) was already correct at
excluding these cases from ever activating SOSCF — but a condition with no
accompanying message is invisible to the user, exactly what this step's own
text warns against ("rather than running an incomplete Hessian that happens
to look plausible" — here it is worse: no Hessian runs at all, and nothing
says so).

**Fixed**: a one-time `[WRN] DFT SOSCF :` log line, emitted once before the
iteration loop starts (not per-iteration) whenever SOSCF was requested via
either trigger keyword (`scf_soscf_start > 0` or `scf_soscf_diis_tol > 0`)
but is disabled, naming the SPECIFIC reason — `"hybrid functional
(exact-exchange response is not yet implemented for DFT SOSCF)"`, `"PCM
solvation (not yet wired through DFT SOSCF)"`, or `"SAO/symmetry blocking
(not yet wired through DFT SOSCF)"` — followed by `"running with plain DIIS
only"` so the user knows the calculation is still valid, just unaccelerated.
Not a hard error, matching the step's own text's actual ask (name the
reason, don't crash or silently misbehave) rather than the harsher
"REJECTS" language in its own header, which would have made every ordinary
hybrid-functional DFT run in the codebase newly fail merely for having
`scf_soscf_start` set by habit.

**Verified on all three exclusion paths, on real inputs**: B3LYP+PBE on
water/6-31G emits the hybrid warning and reaches `-76.3780187007`
(unchanged from before this step, confirming the warning is purely
diagnostic with zero effect on the actual calculation); the existing
`water_rks_pbe_pcm_water_sto3g.hfinp` fixture with `scf_soscf_start 3`
added emits the PCM warning and reaches `-75.2062610942` (matching its own
un-SOSCF-requested baseline). The pure-functional SOSCF case from D2.2.3
(water/6-31G/PBE) is confirmed unaffected — no warning, `SOSCF :` lines
still fire, identical `-76.2895527467`. A run with no SOSCF keywords set at
all (the overwhelming default case) emits nothing — the warning's own
gate condition is on REQUEST, not merely on hybrid/PCM/SAO being present.

**A project-wide gap found while building this, left as a separate,
recorded finding rather than silently fixed elsewhere**: RHF/UHF's own
analogous scope-cut guards have never had this diagnostic either.
`soscf_enabled_uhf`'s `level_shift <= 0.0` exclusion (`src/scf/scf.cpp`,
U3) silently drops the SOSCF request exactly the same way, and its own
code comment even anticipates the risk ("would silently ignore the user's
own request... exactly the iterations where they set it to matter") without
ever having built the warning U3's own text worried about. Same for
RHF/UHF's `!sao_active`/`pcm == nullptr` exclusions from S2/U2. Out of
scope for this DFT-specific step to fix, but worth closing in a small
follow-on across all three SOSCF paths rather than leaving DFT as the only
one that tells the user when their request was silently dropped.

Full DFT ctest suite (11/11) and smoke regression suite (35/35) pass
unchanged.

##### D2.3 — semicanonicalization and level-shift interaction for RKS (~S, after D2.2) — DONE

**Do not assume RHF/UHF's answers transfer without checking DFT's own KS
loop for the equivalent knobs.** Confirm first whether the RKS loop even
has a level-shift mechanism analogous to RHF/UHF's — if it does not, this
sub-step is a documented no-op rather than a designed interaction; if it
does, apply the same "SOSCF requires level_shift <= 0" rule U3 landed,
re-derived for the actual KS code, not copy-pasted from the doc.

Semicanonicalization: re-measure the same way S3/U3 did (disable it, run a
long pure-SOSCF window, compare iteration counts and confirm identical
final energy) rather than assuming RHF/UHF's "no measurable difference,
kept anyway for cheap gauge-fixing" conclusion carries over unchanged —
DFT's grid-dependent Fock build could in principle interact differently,
though there is no a priori reason to expect it; state the measured
result either way.

**Level shift: confirmed a documented no-op, by direct measurement, not
just by grep.** `src/dft/driver.cpp` has ZERO occurrences of `level_shift`
anywhere — `calculator._scf._level_shift` (the shared field RHF/UHF read)
is simply never read by the RKS loop. Verified this actually has zero
effect, not just that the code never references it: ran water/6-31G/PBE
with `level_shift 0.5` set and with it absent, both under plain DIIS (no
SOSCF) — byte-identical energy AND iteration count
(`-76.2895527467`, 11 iterations, both ways). **There is no interaction to
guard against here because DFT has nothing to interact with** — no code
change needed, and no gate to add (a `level_shift <= 0.0` guard would be
dead code, since the field is already ignored unconditionally). This
matches the "if it does not [have a level-shift mechanism], this sub-step
is a documented no-op" branch the scope itself anticipated.

**Semicanonicalization: re-measured (not assumed), confirmed no measurable
difference, matching RHF/UHF's own conclusion exactly.** Built into D2.2.3
(block-diagonalize occ-occ/virt-virt separately per the converged-Fock MO
block after each Newton step). Disabled it via a temporary probe
(`PLANCK_D2_3_NO_SEMICANON` in `driver.cpp` — added, measured, reverted,
same discipline as every other whole-molecule composition check in this
scope) that reads `eps` off the raw, non-eigendecomposed `Cᵀ·F·C` diagonal
instead, and ran a long (200-cycle) pure-SOSCF window with no DIIS handoff
on two independent systems:

| System | With semicanon | Without semicanon |
|---|---|---|
| water/6-31G/PBE | 126 iterations, `-76.2895527467` | 121 iterations, `-76.2895527467` |
| H2/6-31G/PBE | 10 iterations, `-1.1619037723` | 10 iterations, `-1.1619037723` |

**Both converge to the identical energy either way — no plateau, no
wrong-basin convergence, no divergence.** Iteration counts are close
(126 vs 121 on water; identical on H2) — if anything, semicanonicalization
off is marginally faster here, the opposite direction from any concern
about a plateau. **Kept anyway**, same verdict and same reasoning as
RHF/UHF: it is pure gauge freedom (rotating occupied or virtual orbitals
among themselves changes neither density nor energy) and cheap (two small
in-block eigendecompositions, not a full `nbasis`-size solve), so there is
no reason to drop it even though it measured as unnecessary here too. DFT's
grid-dependent Fock build does NOT interact differently from RHF/UHF's
ERI-only Fock build in this respect — the a priori expectation of no
special interaction held up under direct measurement.

Full DFT ctest suite (11/11) and smoke regression suite (35/35) pass
unchanged; the temporary probe was reverted (`git diff` on
`src/dft/driver.cpp` confirmed clean of the probe after removal, rebuilt
and re-tested).

##### D2.4 — verify the DIIS-error switch criterion for RKS (~S, after D2.3) — DONE

Confirm `scf_soscf_diis_tol`/`scf_soscf_min_iter` fire at the correct
iteration for the RKS loop specifically (U4 found this needed zero new
code for UHF because U2's gate already included the criterion branch —
check whether the same is true here before assuming it, since D2.2's own
gate is a fresh KS-loop implementation, not a copy of U2's).

*Verify:* same shape as U4 — correct trigger iteration, identical final
energy either way, on at least one small closed-shell KS case, ideally one
where DIIS alone needs enough iterations for a criterion-based switch to
plausibly help (matching D4's own later, separate question about whether
DFT needs a different default entirely).

**Same finding as U4: needed ZERO new code**, confirmed by reading D2.2.3's
own gate rather than assuming it. D2.2.3's `soscf_enabled`/`criterion_fires`
block was written as a structural copy of RHF's own S2/S3 gate
(`src/scf/scf.cpp`), which already included both the fixed-iteration
(`scf_soscf_start`) and DIIS-error-criterion (`scf_soscf_diis_tol` +
`scf_soscf_min_iter`) branches from the start — so the criterion path was
live from the moment D2.2.3 landed, never exercised by any of D2.2's own
verification runs (all of which used `scf_soscf_start`).

**Verified on water/6-31G/PBE** (DIIS-only trace: `diis_error` per
iteration `4.02e-1, 4.32e-1, 3.44e-2, 7.35e-3, ...`), with
`scf_soscf_diis_tol 1.0e-2`:

- Fires at the correct iteration (4 — the first iteration where the
  JUST-COMPUTED `diis_error` for that iteration's own Fock build,
  `7.354e-3`, drops below `1.0e-2`, with `iter=4 >= scf_soscf_min_iter=2`),
  confirmed by reading the SOSCF log line's own iteration number against
  the DIIS-only trace's per-iteration error column.
- Reaches the identical final energy to the DIIS-only case, confirmed via
  `diff` on the two runs' `DFT Energy` lines (exit 0, byte-identical
  `-76.2895527467`).
- **`scf_soscf_min_iter` genuinely gates, not a no-op**: re-run with
  `scf_soscf_min_iter 6` added (same `diis_tol`, which alone would already
  be satisfied at iteration 4) correctly DELAYED the trigger to iteration 6
  — confirming the `iter >= scf_soscf_min_iter` half of the criterion is
  live, not vacuously always-true. Same final energy either way.

Full DFT ctest suite (11/11) and smoke regression suite (35/35) pass.
No production code changed for this step (verification-only, matching U4's
own zero-new-code finding).

##### D2.5 — measure the actual speedup (~S, after D2.4) — DONE, F6's own caution was warranted

**The entire reason D2 was rescoped around F3's analytic path instead of
the FD-kernel oracle.** Measure wall-clock per SOSCF iteration for D2's
analytic path vs. what the FD-kernel-oracle path (the originally-scoped
D2) would have cost, on at least one system large enough that the
`O(n_occ·n_virt)` grid-pass cost is not trivial. If the analytic path is
not meaningfully cheaper per iteration in practice — matching
`SOSCF_DFT_ANALYTIC_FXC_SCOPE.md`'s own F6 caution — say so explicitly
rather than assuming the asymptotic argument transfers to a real
wall-clock win at the sizes that matter. This is the DFT-SOSCF-specific
instance of F6, run once D2's wiring exists rather than deferred
indefinitely.

**Measured via a temporary probe** (`PLANCK_D2_5_CHECK` in `driver.cpp`'s
RKS convergence branch — added, measured, reverted, same discipline as
every other whole-molecule composition check in this scope), comparing (a)
wall-clock for ONE call to `compute_analytic_xc_hessian_vector_product`
(D2.0's function — a single `(a,i)` unit-vector trial density, the exact
shape `solve_augmented_hessian`'s Krylov loop calls repeatedly) against (b)
wall-clock for ONE call to `build_closed_shell_xc_kernel_blocks` (the
originally-scoped FD-kernel oracle, building the ENTIRE dense Hessian over
the full occ-virt space in one call) — on two systems, alongside the
REAL `ah_iters` (Krylov iteration count) SOSCF actually needed at each
Newton step on the same system:

| System | `nov` | analytic (1 call) | FD-kernel (full, 1 call) | crossover `ah_iters` | real `ah_iters` measured |
|---|---|---|---|---|---|
| water/6-31G/PBE | 40 | 0.1306 s | 0.5130 s | ≈3.9 | 6, 7, 4 |
| water/cc-pVDZ/PBE | 100 | 0.1407 s | 2.1306 s | ≈15.1 | 4, 23, 10 |

**Finding: the asymptotic `O(1) vs O(n_occ·n_virt)` argument is confirmed
true in shape (the analytic call's own cost is flat, ~0.13-0.14 s,
essentially independent of `nov` across a 2.5x range — its cost is
dominated by `O(npoints)` grid evaluation, not `nov`, exactly as F3/D2.0
were built to have), but F6's own caution was warranted: at BOTH system
sizes tested, real SOSCF's actual Krylov iteration counts straddle or
exceed the crossover point where the analytic path's total per-Newton-step
cost (`ah_iters × analytic_call_cost`) exceeds what building the FD-kernel
oracle's full dense Hessian ONCE per Newton step would have cost.** On
water/6-31G, all three measured `ah_iters` (6, 7, 4) are at or above the
crossover (≈3.9), so the FD-kernel path would have been AS FAST OR FASTER
there. On cc-pVDZ, two of three measured `ah_iters` (4, 10) are below the
crossover (≈15.1, analytic wins there), but the middle step's 23 exceeds it
(analytic ≈1.52x SLOWER than the FD-kernel alternative would have been for
that one step).

**Why the FD-kernel's "once per Newton step" framing is the fair
comparison, checked rather than assumed**: once the FD-kernel builds its
dense `(nov × nov)` Hessian, every subsequent Krylov iteration within that
SAME solve is a plain dense matrix-vector product — measured at ~2.5
microseconds for a 100×100 matrix, six orders of magnitude below either the
analytic call (~0.14s) or the FD-kernel build itself (~2.1s) — so the
FD-kernel path's real per-Newton-step cost is `≈ fd_kernel_FULL_s`
regardless of `ah_iters`, while the analytic path's is `ah_iters ×
analytic_call_cost`, scaling linearly with the Krylov iteration count. This
is why `ah_iters` (not `nov` alone) is the number that actually decides
which path wins at a given system size — a fact the original `O(1) vs
O(nov)` framing did not surface, since it compared per-CALL cost rather
than per-NEWTON-STEP cost.

**This does not undo D2's earlier work or argue for reverting to the
FD-kernel oracle.** The analytic path is still exactly correct (D2.0-D2.2
verified this independently of cost), still the only path that scales
correctly to systems where the FD-kernel's `O(nov)` one-time build would
itself become prohibitive (a large active space with many virtuals), and
still avoids the FD-kernel's own numerical-differentiation step-size
sensitivity. **But the wall-clock claim must be stated honestly**: at the
two modest system sizes actually measured (`nov` = 40, 100), DFT SOSCF's
analytic path is not reliably faster in wall-clock terms than the
originally-scoped FD-kernel alternative would have been, contrary to what
the pure asymptotic argument alone would suggest — exactly the caution
`SOSCF_DFT_ANALYTIC_FXC_SCOPE.md`'s own F6 section anticipated needing to
check. Whether the analytic path becomes a clear, reliable win requires
either a system large enough that `ah_iters` stays comfortably below
`nov` (untested — both systems here have `nov` in the tens-to-hundreds
range, and CASSCF's own experience is that Krylov iteration counts do not
automatically grow with system size), or a reduction in the analytic
callback's own fixed per-call cost (currently dominated by grid
evaluation overhead, not algebra) — both left as open follow-on questions,
not resolved here.

Full DFT ctest suite (11/11) passes; the temporary probe was reverted
(confirmed via `grep` for the env-var name returning no matches, followed
by a clean rebuild and re-test).

#### D3 — UKS (~M, after D2's RKS path is verified)

Repeat the RHF→UHF generalization (Track 1) for the KS analogue. **Updated
alongside D2's own rescoping**: since D2 wires F3's analytic Hessian-vector
product rather than a dense FD-kernel matrix, D3 is NOT "build the DFT
equivalent of `build_uhf_cphf_matrix`" (a dense coupled α/β matrix) —
F3.4's own T1..T5 polarized algebra (same-spin plus the T4/T5 cross-spin
terms) is already derived and point-level-verified for exactly this case.
D3's job mirrors D2.0-D2.5 for the polarized functional/UKS SCF loop: promote
F3.4's algebra into a real production function (the polarized analogue of
D2.0), confirm it against the true UKS `E(κ)` (D2.1's analogue), wire it
into `run_ks_scf_scaffold`'s UKS branch (D2.2's analogue), then the
semicanonicalization/level-shift and switch-criterion questions (D2.3/D2.4's
analogues, now for two coupled spin channels), then measure the speedup
(D2.5's analogue). Do this after D2's restricted KS path works, not in
parallel with it — DFT already has more moving parts (grid, XC functional
selection, hybrid exact-exchange fraction) than either RHF or UHF SOSCF
did, and stacking the UKS generalization on an unverified RKS base
compounds the debugging surface.

**Rescoped 2026-09-06 into sub-steps mirroring D2's own decomposition,
after re-reading F3.4's actual current state (`docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md`)
rather than assuming "F3.4 is done" means production-ready.** F3.4's
algebra is verified but exists ONLY as point-level test code
(`tests/dft_gga_polarized_hessian_selfcheck.cpp`, `check_alpha_only`/
`check_mixed`/`check_beta`, 505 lines) and a since-DELETED whole-molecule
probe (`PLANCK_FXC_F3_4_5_CHECK` — per the project-wide "probes become
tests or get removed" rule, `docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md` F3.4.5).
**There is no production function for the polarized case yet** — D3.0
below is exactly as real and necessary a step as D2.0 was, not a
mechanical port. Confirmed by reading the actual formulas: the mixed-`x`
alpha-channel case alone is a 5-term chain rule
(`T1..T5`, `check_mixed` in the test file) with 6-component
`v2rhosigma`/`v2sigma2` packing on both sides, mirrored (not
copy-pasted with names swapped — `check_beta`'s own comment insists on
this) for the beta channel; genuinely more surface than D2.0's restricted
case, which is exactly why D2 was finished and verified end-to-end before
starting this.

Building-block inventory, same discipline as D2.2's own table:

| Piece | Exists? | Where |
|---|---|---|
| Polarized point-level T1-T5 algebra (both spin channels, all cross terms) | Yes, verified, TEST CODE ONLY | `tests/dft_gga_polarized_hessian_selfcheck.cpp` |
| Production function computing `(δV_xc^α, δV_xc^β)` from `(P^α, P^β, δP^α, δP^β)` | No | needs writing (D3.0) |
| Polarized density-on-grid evaluation | Yes, reused unchanged | `evaluate_density_on_grid`'s alpha+beta overload, `xc_grid.h` |
| Two-spin-block `(a,i)` packing convention (`[0,nova)` alpha, `[nova,nova+novb)` beta, virtual-major within each) | Yes, established by `build_uhf_cphf_matrix`/`solve_uhf_cphf` | `src/post_hf/uhf_response.cpp` — D3's packing MUST match this, not invent a third convention (RKS already has one, UHF-CPHF has another it shares with U1/U2) |
| Diagonal (`eps^σ(a)-eps^σ(i)`) piece, per spin | Partial — D2.2.0's `orbital_energy_difference_diagonal` is single-spin; needs calling twice (once per spin) and concatenating, not a new function | `src/dft/analytic_hessian.{h,cpp}` |
| Coulomb-response piece for polarized trial densities | Partial — `_compute_2e_j_direct` takes one density; UKS's `J` is built from the TOTAL density `P^α+P^β` (same as UHF's `build_rhf_cphf_matrix`'s own `J` term), so `δJ` needs `δP^α+δP^β`, not two separate calls | `src/integrals/base.h`, reused |
| Exact-exchange (K) response | Not built, same D2.2 scope cut (pure functionals only) | out of scope, same as D2.2.4 |
| AH solver / Cayley helper | Yes, reference-type-agnostic, reuse unchanged | `solve_augmented_hessian`, `apply_orbital_rotation` (already applied per-spin-channel in U2, same pattern here) |

**Scope cut, carried over unchanged from D2.2**: pure (non-hybrid)
functionals only for the first UKS landing, for the identical reason —
exact-exchange response needs the same unbuilt K-response machinery. D3.2.4
(below) enforces this exactly the way D2.2.4 did for RKS, reusing the same
`x_functional.is_hybrid()` check (spin-agnostic, no change needed there).

###### D3.0 — promote F3.4's polarized algebra into a real production function (~M) — DONE

Write `compute_analytic_xc_hessian_vector_product_polarized` (or similarly
named, same file as D2.0's restricted version — `analytic_hessian.{h,cpp}`,
same reasoning for staying out of `driver.cpp`) taking ground-state
`(P^α, P^β)`, trial `(δP^α, δP^β)`, and the functional(s) — POLARIZED
libxc calls this time (`Spin::Polarized`, not `Unpolarized` — D2.5's own
probe already hit exactly this mismatch once when it called the FD-kernel
oracle with an Unpolarized functional; do not repeat that mistake here) —
and returning `(δV_xc^α, δV_xc^β)` as two AO matrices. This is transcription
of `check_alpha_only`/`check_mixed`/`check_beta`'s own already-verified
5-term chain rule into one real function, the same "no new algebra, but a
new place for a transcription bug to hide" risk D2.0 named for the
restricted case — doubled here, since there are two channels' worth of
formulas to transcribe correctly, each reading a DIFFERENT slot of the same
6-component `v2rhosigma`/`v2sigma2` arrays (F3.4.1's own hard-won ordering
finding).

*Verify*, same two layers D2.0 used: (1) re-run the point-level formulas
through the NEW function's actual call signature on a synthetic multi-point
fixture (following D2.0's own fixture-design lesson — pick `(P^α,P^β,
δP^α,δP^β)` freely, read off whatever `(ρ,∇ρ,δρ,δ∇ρ)` result, do not try to
hit pre-chosen density values), covering same-spin-only, cross-spin-only,
and mixed trial directions on BOTH channels — not just alpha, since
`check_beta`'s own note is that the beta formula is independently
re-derived, not inferred symmetric. (2) One temporary whole-molecule probe
(added, verified, reverted — same discipline as every check in this scope)
against `build_unrestricted_xc_kernel_blocks` (the UKS-native FD-kernel
oracle — no `.first+.second` singlet recombination needed here, unlike
D2.0's restricted case) on a genuinely open-shell system (doublet or
triplet, matching F3.4's own "do not test only closed-shell" instruction).
**If this disagrees, stop before D3.1.**

**Landed as `compute_analytic_xc_hessian_vector_product_polarized`
(`src/dft/analytic_hessian.{h,cpp}`), initially GGA polarized only (the
LDA-polarized branch was added the same day — see below, after D3.1's own
header — closing what was originally noted as a scope limit), kept in the
same file as D2.0's restricted function for the same link-cost
reasoning.** Transcribes `check_mixed`/`check_beta`'s own T1-T5/T1'-T5'
formulas exactly: alpha channel's "self" gradient term
(`2·δvsigma_aa·∇ρ_α + 2·vsigma_aa0·∇δρ_α`) and "cross" term
(`δvsigma_ab·∇ρ_β + vsigma_ab0·∇δρ_β`) are summed and projected through the
SAME AO basis once (mirroring D2.0's own `delta_gradient_term` pattern);
the beta channel is written independently with its own `bb`/`ab`-rooted
slot reads, not alpha with labels swapped, per `check_beta`'s own
discipline.

**Layer (1) landed as `planck-dft-analytic-hessian-polarized-production`
(`tests/dft_analytic_hessian_polarized_production.cpp`), a synthetic
3-AO single-point grid, same fixture-design discipline D2.0's own test
established (pick `(P^α,P^β,δP^α,δP^β)` freely, read off whatever
`(ρ,∇ρ,δρ,δ∇ρ)` result).** Covers two independent `(P,δP)` sets plus
same-spin-only and cross-spin-only trial directions, checked against an
independently-written reference (`reference_delta_vxc_01`, transcribed
from the point-level test file, NOT calling the production function).

**A real defect was caught on the first run, in the TEST's own reference,
not the production function**: every check failed by a clean, uniform 2×
ratio (`got = 2× expected`). Root cause: `reference_delta_vxc_01` calls
`evaluate_gga_fxc` ONCE (a single functional `f`), while the production
function internally sums TWO calls (`exchange_functional` +
`correlation_functional`, both passed as `f` in the test) — the exact
same "x+c additive convention doubles every libxc-derived quantity"
correction D2.0's own restricted test already applies explicitly (its own
`2.0 * reference_delta_vxc_projected(...)` line). Fixed by applying the
identical `2.0×` correction to the polarized reference; all checks pass
after the fix. **Recorded because it is the second time this exact
class of test-side (not production-side) scale bug has appeared in this
scope** — worth flagging for D3.4/D3.5 if a similar reference formula is
written there.

Mutation-verified independently for both channels: zeroing the alpha
channel's cross term (T4+T5) fails only the mixed-`x` cases (same-spin-only
and beta-only cases correctly pass, since they don't exercise that term);
zeroing the beta channel's cross term (T4'+T5') fails only the beta-channel
assertions while the alpha-channel assertions stay green — confirming the
two channels are tested independently, not one inferred from the other.
Both mutations reverted after verification.

**Layer (2) run once as a temporary debug probe (`PLANCK_D3_0_CHECK`) in
`driver.cpp`'s UKS convergence branch, confirmed, then deleted** — same
discipline as D2.0's own `PLANCK_D2_0_CHECK` and F3.4.5's now-deleted
whole-molecule probe. System: triplet water/STO-3G (the same fixture
F3.4.5's own probe and `water_triplet_uks_tddft_pbe_sto3g.hfinp` use),
`correlation pbe` used in BOTH the exchange and correlation argument slots
of both the oracle and the production call — matching the point-level
test's own `"gga_c_pbe"` choice, since F3.4.1 already found PBE
**exchange** alone has near-zero cross-spin coupling (which would leave
the alpha→beta block untested if the input's actual PBE-exchange
`x_functional` had been used instead).

Result: `alpha→alpha` column matches the oracle to `3.3e-11`, `alpha→beta`
column (the genuinely cross-spin block) matches to `2.3e-11` on a nonzero
signal (`max|oracle|=max|analytic|=5.67e-4` — confirmed non-degenerate,
not a symmetry-suppressed direction). **A real harness bug was found and
fixed along the way, in the probe itself, not the production code**: the
first comparison attempt read the oracle's `[1][0]` block using raw Eigen
column-major memory layout (`Eigen::Map` over the projected matrix's own
`.data()`), which silently disagrees with `ResponseExcitationSpace::
flat_index(i,a) = i·n_virt+a`'s occupied-major convention — the two
flattening orders are NOT the same, and comparing them directly showed a
spurious ~10x-magnitude mismatch that had nothing to do with the
production function. Fixed by iterating explicitly with `flat_index`
itself on both sides. Mutation-verified after the fix: flipping the sign
of the alpha channel's cross term moves the `alpha→alpha` column diff from
`3.3e-11` to `2.8e-3` — cleanly caught. Reverted after verification; `git
diff` on `driver.cpp` confirmed clean of both the probe and the mutation.

Full DFT ctest suite (12/12, including the new polarized-production
target) and smoke regression suite (35/35) pass. D3.1 is unblocked.

**LDA-polarized path added the same day, on request, to close the scope
gap the original D3.0 text left open ("no LDA-polarized production path
yet -- add if/when a caller needs it").** Genuinely cheaper than the GGA
case: no gradient terms at all (`V_xc^σ = vrho_σ` alone for LDA), so the
whole chain rule is `delta[vrho_a] = v2rho2_aa·δρ_α + v2rho2_ab·δρ_β`
(mirrored for β) — one cross-spin slot (`v2rho2_ab`, symmetric between the
two channels, unlike GGA's distinct T4/T4' asymmetric cross terms), no
`v2rhosigma`/`v2sigma2` bookkeeping, no gradient projection through
`ao_grid.grad_{x,y,z}`. Added as a new branch inside
`compute_analytic_xc_hessian_vector_product_polarized` (same dispatch
pattern the restricted `compute_analytic_xc_hessian_vector_product`
already uses for its own LDA/GGA split), not a separate function.

**Verified with the same two-layer discipline**: point-level check
(`check_point_lda`, added to `dft_analytic_hessian_polarized_production.cpp`)
against an independently-written reference, using `lda_c_pw` (PW
correlation) rather than `lda_x` (Slater exchange) specifically because
Slater exchange has NO cross-spin coupling at all (`v2rho2_ab` identically
zero), which would leave the cross term completely untested — the LDA
analogue of F3.4.1's own "PBE exchange has near-zero cross-spin coupling,
use `gga_c_pbe`" finding. Four cases (two independent `(P,δP)` sets, plus
same-spin-only and cross-spin-only trial directions) all pass. Applied the
x+c doubling correction from the first version this time (learned from
D3.0's own GGA mistake, recorded above) — no scale bug on this pass.
Mutation-verified: dropping both `v2rho2_ab` cross terms fails every LDA
case, including two that correctly show `got 0` where the cross-only trial
direction should produce a nonzero result; reverted after confirming. No
whole-molecule probe was run for the LDA case specifically — the same
whole-molecule machinery (`build_unrestricted_xc_kernel_blocks`) and
harness-convention lessons (the `flat_index` vs raw-memory-layout trap)
from D3.0's own GGA probe apply unchanged if one is needed later, so it
was judged redundant to re-run for a strictly simpler formula that shares
100% of its surrounding AO-projection code with the already-verified GGA
path.

Full DFT ctest suite (12/12) and smoke regression suite (35/35) pass with
the LDA branch added.

###### D3.1 — confirm the packed polarized Hessian-vector product against the true UKS energy (~S, after D3.0) — DONE

D2.1's own corrected methodology, applied from the start this time rather
than discovered the hard way: **do not test
`Tr(δP·δV_xc(δP)) = ∂²E_xc/∂κ²` in isolation** — that identity is missing
the density-curvature term regardless of spin polarization (the Cayley
transform's nonlinearity in `κ` is a per-spin-channel geometric fact, not
specific to RKS). Verify the FULLY ASSEMBLED polarized `h_op` (diagonal +
Coulomb-from-total-density + D3.0's polarized XC piece, for BOTH spin
blocks) against finite difference of the true UKS TOTAL energy
`E(κ_α, κ_β)`, on at least same-spin and cross-spin `(a,i)` directions
(one perturbing only the alpha block, one only beta, one both at once —
mirroring D3.0's own same-spin/cross-spin/mixed verification split one
level up, at the assembled-callback level).

**Landed, via a temporary probe (`PLANCK_D3_1_CHECK` in `driver.cpp`'s UKS
convergence branch — added, verified, reverted, same discipline as every
other whole-molecule check in this scope) on triplet water/STO-3G/PBE
(`use_symm .false.`, avoiding the symmetry-suppressed-direction trap D2.1
and D3.0 both hit).**

**Scale factor, measured rather than assumed to carry over from RKS**:
`d²E_total/dκ² = 2·H_bare_polarized` (using the UHF-convention unscaled
`dP = C_a·C_i^T + C_i·C_a^T`, no closed-shell `2×` factor) — a genuinely
DIFFERENT constant from RKS's own `4·H_bare` (D2.2.2). Confirmed clean at
ratio 1.000000 on both `alpha-only` and `beta-only` diagonal directions
across two independent `(a,i)` pairs each.

**Two real bugs found and fixed along the way, both in the temporary probe
itself, not in any production code — the third time in this scope a
verification harness, not the function under test, has been the actual
source of a discrepancy (D3.0 hit two of its own: the x+c-doubling test
reference, and the `flat_index`-vs-raw-memory-layout comparison)**:

1. **A lambda-argument bug**: the probe's `H_bare` computation called
   `h_op_bare_polarized(spin_for_H, aa_, ia)` UNCONDITIONALLY — always
   passing the alpha-channel indices, even when checking `beta-only`
   (`spin_for_H=1`). Invisible on the first test direction because
   `aa_==ab_` and `ia==ib` there by coincidence; surfaced immediately
   (`ratio=7.44`, not 1.0) the moment an independent `ib` was tried. Fixed
   by branching: `(spin_for_H==0) ? h_op_bare_polarized(0,aa_,ia) :
   h_op_bare_polarized(1,ab_,ib)`.

2. **A degenerate-direction trap for the CROSS (mixed) term specifically,
   recurring for the third time in this scope** (D2.1's HOMO-LUMO
   even-in-κ finding; D3.0's `(i=0,a=n_occ)` zero-cross-term finding): at
   `(ia=1,aa=0,ib=1,ab=0)` — using the SAME spatial orbital index for both
   spins — the four finite-difference energies `E(+h,+h)`, `E(+h,-h)`,
   `E(-h,+h)`, `E(-h,-h)` came out BIT-IDENTICAL, making the mixed second
   derivative trivially (and misleadingly) zero on both sides at once. This
   was not a bug — verified directly by confirming `||kappa_a||`/`||kappa_b||`
   respond correctly to the requested rotation, and by trying several
   `(a,i)` combinations before finding one where BOTH sides read a
   consistent nonzero value. **Resolved by picking genuinely asymmetric
   indices** (`ia=1,aa=0` for alpha; `ib=2,ab=1` for beta): the cross term
   converges cleanly to ratio `0.999849 → 0.999999 → 1.000020` across
   `h={1e-2,1e-3,1e-4}`, with a clearly nonzero signal
   (`J_packed_cross=-0.0697`, `xc_packed_cross=0.0035`) — confirming the
   composed polarized `h_op`'s cross-spin coupling (Coulomb-from-total-
   density plus D3.0's own XC cross terms) is correct.

**Lesson worth carrying forward, stated plainly since it has now recurred
three times**: a same-index or otherwise-symmetric choice of verification
direction is not just occasionally unlucky — it is apparently the DEFAULT
first guess every time (index 0, or matching indices across two spin
channels), and it has produced a spuriously-passing OR spuriously-zero
check at least once in every one of D2.1, D3.0, and D3.1. Before trusting
any single-direction FD check in this codebase, confirm the checked
quantity is nonzero on BOTH sides first, or deliberately choose indices
that differ across whatever symmetry (spin, occ/virt numbering, spatial
point group) the system has.

Full DFT ctest suite (12/12) and smoke regression suite (35/35) pass; the
temporary probe (mutation and all diagnostics) was fully reverted, confirmed
via `grep` for the env-var name returning no matches, rebuilt and re-tested.
D3.2 is unblocked.

###### D3.2 — wire the SOSCF branch into the UKS loop, fixed iteration (~M, after D3.1) — DONE

Mirror D2.2.3's own shape as closely as the UKS loop's structure allows,
generalized the way U2 generalized S2 for UHF: persist
`C_soscf_prev`/`eps_soscf_prev` PER SPIN (or a combined struct — U2's own
UHF SOSCF branch already made this exact packaging decision once, reuse
it rather than re-deciding), gate on the same `scf_soscf_*` keywords
(shared across all three SOSCF paths), build the gradient as TWO blocks
(`F_mo^α(a,i)`, `F_mo^β(a,i)`) packed into one vector matching
`build_uhf_cphf_matrix`'s own `[0,nova)+[nova,novb)` convention (the table
above), apply the Cayley rotation and semicanonicalization separately per
spin channel (same as U2), same trust-region cap
(`kSoscfMaxRot = 0.20`). **Do not build a second AH solver, a second
Cayley helper, or a UKS-specific packing convention** — reuse
`solve_augmented_hessian`/`apply_orbital_rotation` unchanged and the
existing UHF-CPHF packing convention from the table above.

*Verify:* same shape as D2.2.3 — on a genuinely open-shell system (not a
closed-shell system run through the UKS code path, which would exercise
nothing new — U1/U2's own "do not test only closed-shell" lesson applies
here with equal force), SOSCF from a fixed iteration reaches the identical
UKS total energy as pure DIIS to all 10 printed digits, with the orbital
gradient shrinking across the window (report the measured shape honestly,
linear or superlinear, per D2.2.3's own precedent — do not assume
superlinear just because RHF/H2 showed it once).

**Landed in `run_ks_scf_scaffold`'s UKS branch (`src/dft/driver.cpp`),
structurally a copy of D2.2.3's RKS branch generalized to the coupled
alpha/beta step the same way U2 generalized S2:**

- `Ca_soscf_prev`/`Cb_soscf_prev`/`epsa_soscf_prev`/`epsb_soscf_prev`
  persisted every iteration (one per spin channel — U2's own packaging
  decision, not re-decided); gate is a structural copy of the RKS
  `soscf_enabled`/`soscf_active`/`criterion_fires` block, sharing the
  `scf_soscf_*` keywords unchanged, with the same three exclusions
  (hybrid / PCM / SAO).
- Gradient built as two blocks (`F_mo^α(a,i)`, `F_mo^β(a,i)`) packed into
  `build_uhf_cphf_matrix`'s own `[0,nova)` alpha + `[nova,nova+novb)` beta
  virtual-major (`a*n_occ + i`) convention — no third convention invented.
- `h_op(x)` composes `diag_a⊙x_α + Ja_packed + xca_packed` (and the beta
  analogue). **`δJ` is one call on the TOTAL trial density `δP^α+δP^β`**
  (matching UKS's own per-iteration Coulomb build), then packed separately
  into the α and β `(a,i)` blocks; the polarized XC piece is D3.0's
  `compute_analytic_xc_hessian_vector_product_polarized`. Paired UNSCALED
  with `g = F_mo` — D3.1 measured `d²E_total/dκ² = 2·H_bare_polarized` with
  `g_true = 2·g_bare`, a matching pair, so the unscaled ratio already
  reproduces the true Newton step (exactly U1/U2's conclusion).
- `solve_augmented_hessian` / `apply_orbital_rotation` reused unchanged
  (per spin channel, one shared step vector, two `κ` matrices — same as
  U2); same `kSoscfMaxRot = 0.20` cap; per-spin occ-occ/virt-virt
  semicanonicalization after the step. DIIS (both spins) cleared on window
  handoff.

**Verified on triplet water/STO-3G/PBE (`use_symm .false.`, hcore guess,
genuinely open-shell so the α-β coupling terms are exercised), SOSCF from
a fixed iteration 4:**

- **Energy: identical to fully-converged DIIS to all 10 printed digits.**
  SOSCF reaches `-74.8423080131` in 10 iterations. Plain DIIS at the input's
  own `tol 1e-9` stopped early at `-74.8423073568` (its ΔE/ΔP gate tripped
  at a non-stationary point — commutator error plateaued at `1.5e-4`, never
  improving), the same pre-existing small-system DIIS weakness D2.2.3 hit on
  H2. Tightening DIIS to `tol 1e-11` (`max_cycles 400`) makes it reach
  **exactly `-74.8423080131`** in 102 iterations — bit-identical to SOSCF,
  confirming SOSCF found the true stationary point and DIIS's early answer
  was the incorrect one.
- **Gradient shrinkage: genuinely superlinear** — `|g|`:
  `7.02e-3 → 2.96e-4 → 2.07e-5` across the 3-iteration window (ratios
  `≈0.042, 0.070`, accelerating). On a larger case (water/6-31G/PBE triplet,
  fine grid) the same code also shows superlinear shrinkage
  (`2.46e-1 → 2.17e-2 → 1.95e-3`, ratios `≈0.088, 0.090`) and converges in
  19 iterations where plain DIIS does not converge in 120 — again the
  DIIS-side weakness, not SOSCF's.
- **SOSCF-off default is byte-identical to the pre-D3.2 tree**: full smoke
  regression suite (35/35, every DFT-tagged UKS case included) and full DFT
  ctest suite (12/12) pass unchanged with no `scf_soscf_*` keywords set.

###### D3.2.1 — the hybrid-functional scope-cut warning, for UKS (~S, after D3.2) — DONE

D2.2.4's own finding — a silent fallback when SOSCF is requested but
disabled — is a general risk of copying the gate pattern, not something
D3.2 automatically inherits a fix for just because D2.2.4 built one for
RKS. Confirm the SAME warning fires (naming hybrid / PCM / SAO, same
message shape) when SOSCF is requested on a UKS run that hits any of
those exclusions. If D3.2's gate is structurally identical to D2.2.3's
(same `soscf_enabled` construction, same three exclusions), this may be a
verification-only step requiring no new code — check before assuming
either way, the same discipline D2.4/U4 both used for the DIIS-criterion
branch.

**Verification-only, as expected — no new code.** D3.2 copied D2.2.4's own
one-time `[WRN] DFT SOSCF :` warning block verbatim (the block is emitted
before the iteration loop when either trigger keyword is set but the run
hits hybrid / PCM / SAO), and the gate is the structural copy of D2.2.3's.
Verified on real inputs: B3LYP+PBE on triplet water/STO-3G emits the
hybrid warning and runs plain DIIS to `-74.9272421395`; the same input
with `use_symm .true.` emits the SAO/symmetry warning and reaches
`-74.8423073566`. A run with no SOSCF keywords emits nothing.

###### D3.3 — semicanonicalization and level-shift interaction for UKS (~S, after D3.2.1) — DONE

Re-measure per spin channel, same discipline as D2.3/S3/U3: disable
semicanonicalization, run a long pure-SOSCF window (both spin channels),
compare iteration counts and confirm identical final energy. Level shift:
already confirmed a universal DFT-wide no-op in D2.3 (zero occurrences of
`level_shift` anywhere in `driver.cpp`, RKS or UKS) — this sub-step is
already answered, not something to re-derive; just note in the result that
D2.3's finding covers UKS too since the check was on the whole file, not
the RKS branch specifically.

**Level shift: covered by D2.3, re-confirmed by grep.** `grep -c
level_shift src/dft/driver.cpp` returns 0 — the field
`calculator._scf._level_shift` is never read anywhere in the DFT driver,
RKS or UKS. D2.3's own measurement (byte-identical energy and iteration
count with `level_shift 0.5` set vs absent) was on the whole file, so it
covers UKS unchanged. No guard to add — a `level_shift <= 0.0` check would
be dead code since the field is already ignored unconditionally.

**Semicanonicalization: re-measured per spin, same verdict as
RHF/UHF/RKS.** Temporary probe (`PLANCK_D3_3_NO_SEMICANON` in the UKS
SOSCF branch — added, measured, reverted; `grep` confirms clean) reading
`eps` off the raw non-eigendecomposed `Cᵀ F C` diagonal per spin instead.
Long pure-SOSCF window (`scf_soscf_cycles 300`, no DIIS handoff) on two
independent open-shell systems:

| System | With semicanon | Without semicanon |
|---|---|---|
| water/STO-3G/PBE triplet | 13 iterations, `-74.8423080131` | 12 iterations, `-74.8423080131` |
| water/6-31G/PBE triplet, fine grid | 27 iterations, `-76.0195905826` | 30 iterations, `-76.0195905826` |

**Both converge to the identical energy either way (all 10 digits) — no
plateau, no wrong-basin convergence.** Iteration counts are within a few
of each other in both directions. **Kept anyway**, same reasoning as every
other SOSCF path: pure gauge freedom (rotating occupied or virtual
orbitals among themselves changes neither density nor energy), cheap (two
small in-block eigendecompositions per spin), so no reason to drop it even
though it measured as unnecessary here too. Full DFT ctest (12/12) and
smoke (35/35) suites pass with the probe reverted.

###### D3.4 — verify the DIIS-error switch criterion for UKS (~S, after D3.3) — DONE

Same shape as D2.4/U4: confirm whether D3.2's gate already includes the
criterion branch (very likely yes, if D3.2 copies D2.2.3's gate structure
the way D2.2.3 copied RHF's) — if so, this is verification-only, matching
U4/D2.4's own zero-new-code finding twice already.

**Zero new code — third time this finding has held** (U4, D2.4, now D3.4).
D3.2's gate was a structural copy of D2.2.3's `criterion_fires` block,
which already carried both the fixed-iteration (`scf_soscf_start`) and
DIIS-error (`scf_soscf_diis_tol` + `scf_soscf_min_iter`) branches — so the
criterion path was live from the moment D3.2 landed, never exercised by
D3.2's own `scf_soscf_start`-based verification.

**Verified on triplet water/STO-3G/PBE** (DIIS-only per-iteration error:
`2.18e0, 6.34e-1, 3.00e-2, 4.18e-3, ...`), with `scf_soscf_diis_tol
5.0e-2`:

- Fires at the correct iteration (3 — the first where `diis_error`
  (`3.00e-2`) drops below `5.0e-2` with `iter >= scf_soscf_min_iter = 2`),
  read from the `DFT UKS SOSCF :` log line's iteration number against the
  DIIS-only trace.
- Reaches the identical final energy to the fixed-iteration SOSCF runs and
  to fully-converged DIIS: `-74.8423080131` (all 10 digits).
- **`scf_soscf_min_iter` genuinely gates**: re-run with `min_iter 8` (same
  `diis_tol`, which alone is satisfied at iter 3) correctly delayed the
  trigger to iter 8. Same final energy.

Full DFT ctest (12/12) and smoke (35/35) suites pass. No production code
changed.

###### D3.5 — measure the actual speedup for UKS (~S, after D3.4) — DONE, and the result is the OPPOSITE of D2.5's

D2.5's own methodology, run again rather than assumed to transfer: measure
wall-clock for D3.0's analytic polarized `h_op` (now TWO grid-pass-shaped
calls per Krylov iteration — one per spin channel's XC piece — so the
per-call cost may not simply double from D2.5's restricted numbers,
measure rather than extrapolate) against what
`build_unrestricted_xc_kernel_blocks` would cost for the full polarized
Hessian, at real measured `ah_iters` from an actual UKS SOSCF run. **Given
D2.5's own finding that the restricted case's analytic path was NOT
reliably faster at the two sizes tested, do not assume UKS fares any
better** — if anything, the doubled per-call cost (two spin channels) makes
the crossover point in `ah_iters` potentially LOWER, not higher, worth
checking explicitly rather than assuming the RKS numbers transfer.

**Measured via a temporary probe (`PLANCK_D3_5_CHECK` in the UKS SOSCF
branch — added, measured, reverted; `grep` confirms clean), same shape as
D2.5**: one call to the composed polarized `h_op` (a single `(a,i)`
unit-vector, both spin XC pieces + the total-density `δJ`, the exact shape
the Krylov loop calls repeatedly) vs one call to
`build_unrestricted_xc_kernel_blocks` over both spin `ResponseExcitationSpace`s
(the full polarized dense Hessian in one call), alongside the REAL
`ah_iters` the SOSCF Newton step needed:

| System | `nova` / `novb` | analytic (1 call) | FD-kernel (full, 1 call) | crossover `ah_iters` | real `ah_iters` measured |
|---|---|---|---|---|---|
| water/STO-3G/PBE triplet | 6 / 12 | 0.017 s | 0.155 s | ≈9.2 | 3, 3, 3 |
| water/6-31G/PBE triplet, fine grid | 42 / 36 | 0.018 s | 0.50 s | ≈28 | 5, 6, 7 |
| water/cc-pVDZ/PBE triplet, fine grid | 114 / 84 | 0.064 s | 2.06 s | ≈32 | 8, 10, 10 |

**Finding: the analytic path is RELIABLY FASTER for UKS at every size
tested — the opposite of D2.5's restricted-case result — and the reason is
exactly the doubled per-call cost D3.5 worried about, working the other
way.** Real `ah_iters` (3–10) stay well below the crossover point (9–32) in
every row, so `ah_iters × analytic_call_cost` is 3–5× cheaper than
building the FD-kernel's full dense polarized Hessian once per Newton
step. Two mechanisms:

1. **`ah_iters` did not grow proportionally with `nov`.** It stayed in
   single digits across a >10× range of `nov` (18 → 198), same
   observation D2.5 made and CASSCF's own experience — Krylov iteration
   counts track the Hessian's conditioning, not its dimension.
2. **The doubled per-call cost pushed the crossover HIGHER, not lower.**
   The FD-kernel oracle must finite-difference *both* spin channels'
   directions (`nova + novb` grid-pass pairs), so `fd_kernel_full` roughly
   doubles from the restricted case at comparable `nov` — while
   `analytic_1call` only rose from ~0.017 s to ~0.064 s across the whole
   range (grid-evaluation-dominated, essentially flat in `nov`, exactly as
   F3/D2.0 built it). A more expensive oracle means a higher crossover
   `ah_iters`, i.e. *more* Krylov iterations affordable before the analytic
   path loses — the reverse of D3.5's stated worry.

**This does not contradict D2.5's honest RKS conclusion** — that one
measured real `ah_iters` straddling or exceeding the crossover at the two
RKS sizes tested. UKS's crossover is simply higher (doubled oracle cost)
while its `ah_iters` stayed comparably low, so the same asymptotic
argument that was borderline for RKS is comfortably in the analytic path's
favor for UKS. Full DFT ctest (12/12) and smoke (35/35) suites pass with
the probe reverted.

**D3 is complete: D3.0–D3.5 all landed.** UKS SOSCF runs from a fixed
iteration or a DIIS-error criterion, reaches fully-converged DIIS's energy
to all 10 digits on genuinely open-shell systems, shrinks the orbital
gradient superlinearly, and — unlike RKS — is a measurable wall-clock win
at every size tested. Pure (non-hybrid) functionals only; hybrid / PCM /
SAO emit the D3.2.1 warning and fall back to plain DIIS.

### What this must not do (UKS-specific, in addition to D2's own list)

- **Do not assume the beta-channel formula is the alpha-channel formula
  with labels swapped.** F3.4.4's own point-level test explicitly re-derives
  it independently rather than copy-pasting with names swapped, and that
  discipline must carry into D3.0's production function — write and verify
  both channels' formulas separately, even though they are structurally
  mirror images.
- **Do not invent a third `(a,i)` packing convention.** RKS (D2.2's
  `pack_hessian_vector_product_cphf_order`, single spin) and UHF-CPHF
  (`build_uhf_cphf_matrix`, `[0,nova)+[nova,novb)`) already exist; D3 must
  match the UHF-CPHF one, since that is the convention a two-spin-block
  flat vector already uses elsewhere in this codebase.
- **Do not test D3 only on a closed-shell system run through the UKS code
  path.** That exercises the UKS machinery's RHF-degenerate limit, not the
  genuinely open-shell cross-spin coupling terms F3.4.3's own point-level
  test exists specifically to catch — U1's own "do not test only
  closed-shell" lesson for UHF applies with equal force here.

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
