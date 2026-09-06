# UHF SOSCF

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**How does second-order SCF work for UHF, and what makes it different from
the RHF case in `docs/SOSCF.md`?**

## Short answer

UHF SOSCF is the RHF pattern with the orbital step run per spin channel on
one shared step vector. It reuses `solve_augmented_hessian` and
`apply_orbital_rotation` unchanged (both reference-type-agnostic), and
reuses the coupled α/β orbital Hessian `solve_uhf_cphf` already builds
internally — split out as a standalone `build_uhf_cphf_matrix` so SOSCF can
call it mid-iteration.

The gradient and step are two blocks in one vector: `[0, nova)` alpha,
`[nova, nova+novb)` beta, virtual-major (`a·n_occ + i`) within each, with
`nova = n_virt_α·n_occ_α`. This is `solve_uhf_cphf`'s own `rhs` packing.

Off by default, gated on the same `scf_soscf_*` keywords RHF uses. Verified
against the true UHF `E(κ)` and against fully-converged DIIS on three
genuinely open-shell systems.

## Where the logic lives

- `src/scf/scf.cpp` — the SOSCF branch in `run_uhf` (the `Fa_diag`/`Fb_diag`
  selection, immediately before `diagonalize_uhf_spin`); `Ca_prev`/`Cb_prev`/
  `epsa_prev`/`epsb_prev` persisted every iteration; `soscf_enabled_uhf`
  gate (requires `level_shift <= 0.0`); per-spin `semicanonicalize` lambda
- `src/post_hf/uhf_response.{h,cpp}` — `build_uhf_cphf_matrix` (split out of
  `solve_uhf_cphf`, no convergence guard so SOSCF can call it mid-iteration);
  the full dense `(nova+novb)²` matrix `A` and its unscaled `ε_a - ε_i`
  diagonal
- `src/base/types.h` — `UHFDIISState`, the combined-spin DIIS state cleared
  on window handoff
- `src/scf/sad.cpp` — `compute_sad_guess_open_shell`, whose per-element
  atomic UHF sub-solves recurse into `run_uhf` (why the FD probe was gated
  off `atomic_numbers.size() > 1`)
- `src/post_hf/casscf/aug-hessian.h` — `solve_augmented_hessian`, reused
  per spin
- `docs/SOSCF.md` — the RHF precedent, and the three defects (double-step,
  scale mismatch, AH solver tolerance) it fixed

## What invariants matter

### 1. The UHF Hessian is a more expensive operation per call than the RHF one, not just a bigger matrix

`build_uhf_cphf_matrix` forms `A` one column at a time: per trial rotation
it builds the induced AO density response, calls `_compute_2e_fock_uhf`
(or `RI::build_ri_fock_uhf` under RI), and projects back into the `(a,i)`
block — `(nova+novb)` Fock builds per Hessian construction, once per SOSCF
iteration. RHF's `build_rhf_cphf_matrix` does one `O(n_b⁴)` `transform_eri`
and reads elements out directly. SOSCF materializes the full dense `A` and
passes `h_op(x) = A·x`, not a matrix-free product.

Design rule:

- Do not assume UHF SOSCF is a wall-clock win because RHF's was — it is a
  verified iteration-count win, and whether the Hessian build cost erodes
  that at a given size is unmeasured. Judge success on iteration count, not
  wall-clock alone.

### 2. The gradient/Hessian scale is a clean universal factor of 2 — different from RHF's 4

FD-probing `build_uhf_cphf_matrix` against the real UHF `E(κ)` over a full
sweep of every `(a,i)` diagonal index (28 alpha + 36 beta on water/6-31g):

```
g_fd / g_used = 2.0000000    (every probed index, alpha and beta)
```

UHF's diagonal-block CPHF formula is already `(ai|ia) - (aa|ii)` with no
leading Coulomb multiplier (verified against a hand-derived same-spin
formula on a random ERI tensor, ratio `1.0`). The raw Hessian diagonal does
NOT reproduce `h_fd` cleanly at most swept indices (ratios 0.2 to 272, sign
flips) — this is not a bug, it is off-diagonal orbital-Hessian curvature
dominating a coupled multi-virtual open-shell system, which RHF's own probe
never contended with (its one direction was weakly coupled). On the
well-isolated indices, `2·Amat` tracks `h_fd` to the same few-percent
residual RHF's probe showed.

Design rule:

- A Newton step depends only on the ratio `g/H`. Since `g_true = 2·g_used`
  and `H_true = 2·Amat`, use `g = F_mo(a,i)` unscaled against `Amat`
  unscaled — `build_uhf_cphf_matrix` needs no changes, exactly RHF's
  conclusion at a different constant.

### 3. Level shift and SOSCF are mutually exclusive, enforced in code

`soscf_enabled_uhf` requires `level_shift <= 0.0`. The SOSCF
gradient/Hessian read the plain unshifted `Fa`/`Fb`, so this was never a
silent double-counting risk — but running SOSCF against the unshifted Fock
while `level_shift > 0` is configured would silently ignore the user's
request on exactly the iterations they set it to matter. Verified:
`level_shift 0.3` + `scf_soscf_start 3` emits no `SOSCF :` lines and
reaches the same energy as the DIIS-with-shift run.

Design rule:

- When a SOSCF path deliberately ignores a configured knob, make it a code
  guard that disables SOSCF, not a documented intention — and (a gap not
  yet closed) emit a warning naming the reason, the way DFT SOSCF does.

### 4. Semicanonicalization measured unnecessary, kept anyway

Disabling it (raw `Cᵀ F C` diagonal per spin) over a long pure-SOSCF window
with no DIIS handoff: water/6-31G triplet 60 vs 64 iterations,
water-cation/STO-3G doublet 25 vs 32 — both converge to the identical
energy, no plateau, no wrong-basin.

Design rule:

- Keep it: it is pure gauge freedom (rotating occupied or virtual orbitals
  among themselves changes neither density nor energy) and cheap (two small
  in-block eigendecompositions per spin), so there is no reason to drop it
  even where it measures as unnecessary.

### 5. Do not test only closed-shell

A closed-shell system run through the UHF path exercises the RHF-degenerate
limit, not the α-β coupling terms in `A`. All verification used genuinely
open-shell references (triplet water, water-cation doublet).

Design rule:

- Every UHF SOSCF check runs on a genuinely open-shell reference — the same
  discipline the ROHF gradient and CASSCF-ROHF work each needed.

## What was fixed / built

1. **`build_uhf_cphf_matrix` split out of `solve_uhf_cphf`** — returns the
   dense coupled α/β `A` with no convergence guard (mirroring
   `build_rhf_cphf_matrix`'s own relaxation, since SOSCF calls it
   mid-iteration). `solve_uhf_cphf` is now a thin wrapper solving
   `A z = -rhs`; its one existing caller (the UMP2 gradient) is unchanged,
   all 11 UHF-tagged regressions pass.

2. **The SOSCF branch in `run_uhf`** — `Ca_prev`/`Cb_prev`/`epsa_prev`/
   `epsb_prev` persisted every iteration (built from the *previous*
   iteration's basis against the *current* Fock — probing right after
   diagonalizing `F` is vacuous, `Cᵀ F C` is diagonal there); joint α/β
   gradient `g`; `h_op(x) = A·x` unscaled; `solve_augmented_hessian` with
   `ah_start_tol = max(1e-8, 0.1·‖g‖)` (transfers from RHF unchanged);
   `kSoscfMaxRot = 0.20` trust-region cap; per-spin Cayley rotation and
   semicanonicalization; combined `UHFDIISState` cleared on handoff.

3. **Switch trigger — zero new code.** The window-selection logic mirrors
   RHF's `soscf_enabled`/`soscf_active` block, which already carried both
   the fixed-iteration (`scf_soscf_start`) and DIIS-error-criterion
   (`scf_soscf_diis_tol` + `scf_soscf_min_iter`) branches. Verified: the
   criterion fires once `diis_err < tol` **and** `iter >= min_iter`.

## Validation strategy that should remain in place

- Same-energy check to all 10 printed digits against pure DIIS, on three
  genuinely open-shell systems (triplet water from SAD, triplet water/6-31G
  from hcore, water-cation doublet from hcore), with the orbital gradient
  shrinking superlinearly across the window
  (`2.16e-1 → 3.37e-2 → 3.55e-3` on 6-31G)
- All 11 UHF-tagged regression cases plus full core (71) and smoke (35)
  suites, byte-identical with SOSCF off by default
- SAD + SOSCF composition: converges to the identical energy as SAD + DIIS;
  iteration count falls on the harder-starting case (water-cation from
  hcore 18 → 15), stays the same on the already-fast one

## What was measured but is not kept

`PLANCK_SOSCF_FD_CHECK` (RHF and UHF) — the finite-difference probe against
the true `E(κ)`, the only check that catches a gradient/Hessian pairing
individually plausible on each side but mutually inconsistent — was deleted
outright per the project-wide decision that debug probes must become
standalone tests or be removed. Each needs a converged SCF's internal state
the SCF loop does not expose. The ongoing gate is the SOSCF-vs-DIIS
energy-agreement regressions, which pin the fixed `g=F_mo` / `Amat`-unscaled
convention the probe found. `docs/SOSCF.md`'s "keep the probe" note is
stale.

## Remaining architecture concern

**ROHF SOSCF needs new theory, not a port.** There is no ROHF
orbital-response or CPHF machinery anywhere in this codebase — the same gap
behind ROHF-MP2, ROHF stability, and ROHF PCM all being unsupported. ROHF
orbitals diagonalize the effective Roothaan Fock, not separate per-spin
canonical Focks, the same subtlety that forced the ROHF analytic gradient
to use its own `W = P^α F^α P^α + P^β F^β P^β`. Nothing here transfers.

The original large-`nb` motivation (the `scale.json` iteration-count cliff)
is unreproduced — that ladder needs cluster access. Correctness was
verified against finite differences and exact DIIS-energy agreement
instead.
