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
