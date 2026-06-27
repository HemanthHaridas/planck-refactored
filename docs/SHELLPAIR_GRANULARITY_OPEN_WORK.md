# Shell-pair Granularity (H-10) — Remaining & Deferred Work

Companion to `docs/SHELLPAIR_GRANULARITY_HANDOFF.md` (the architecture note).
Canonical status is `vault/Status/Completion.md` + `vault/Status/Open Work.md`;
this file holds the design-level detail for what is *not* yet done in the
shell-pair-granularity effort and the CASSCF speedups it surfaced.

None of these block the landed work.

## Rys hoist — B-4 (the one direct follow-on)

- **B-4 — explicit `engine rys` (`_compute_2e`).** The Rys hoist
  (`RysHoistedQuartet`, `_contracted_eri_block_hoisted_views`) is wired into the
  Auto path (B-3) but **not** into the non-Auto Rys tensor/Fock builds, so explicit
  `engine rys` is still per-component. Lower value (Auto is the default; explicit
  `engine rys` is rare). It adopts the same hoisted (reordered) order as B-3, so it
  is gated at the `1e-13` ERI bar, not bitwise. The *existing committed* SA-2
  `engine rys` uphill gate runs `_compute_2e`, so B-4 is the step that exercises
  that specific gate — kept robust by the already-landed B-2.5 plateau hardening
  and the B-2.5c reorder-robustness measurement (a ~1e-15 reorder left the uphill
  SA-2 case on its intended basin). Gate: `planck-hgp-engine-smoke`-style Rys
  self-consistency on g shells + the SA-2 uphill case under `engine rys`.

## Rys hoist — downstream phases (each independent)

- **Phase C — symmetry direct-SCF skeletons** (`os_symm` / `hgp_symm` /
  `rys_symm`): still per-AO. Convert to shell-quartet carrying the signed-AO orbit
  per component (same pattern as the `sym_ops` branch). Now unblocked — the block
  kernels exist for all three engines.
- **Phase D — gradient derivative ERIs** (`compute_eri_deriv_dispatch` + loop):
  still per-AO. Auto gradients fall back to OS by design (the dispatcher only
  recognizes explicit `HeadGordonPople`); Rys has **no derivative kernel at all**,
  so true per-quartet Auto gradients need one written from scratch first. Largest
  remaining piece; was an explicit non-goal.
- **Phase E — retire the per-AO adapter.** Change `build_shellpairs` to emit
  shell-level `ShellPair` objects and drop `expand_shell_groups_to_ao_pairs`. The
  only step that changes the shared `ShellPair` contract, so it must come **last**
  and resolve the per-component `_component_norm` folding (handoff invariant 2)
  across OS/HGP/Rys simultaneously.

## Cleanup

- **Shared `SpatialQuartetLayout`** (also in vault Open Work): OS's `EriScratch`,
  HGP's `EriScratch` / `g_hgp_scratch`, and Rys's `RysScratch` / `RysMaxBoxLayout`
  carry near-duplicate per-quartet 6-axis scratch. Extract the shared layout; the
  triangular-vs-rectangular distinction from the §4 design blocker should inform
  the interface. Pure refactor, no behavior change; gated by the existing
  cross-engine equality tests.

---

## CASSCF on a g-basis is slow (investigation)

Not a Rys task — surfaced when the B-3 `engine auto` cc-pVQZ SA-CASSCF gate proved
unrunnable (>15 min, no macroiteration completed, on both a laptop and a larger
machine). That gate was dropped: B-3 is an integral-path change, so its correctness
is the *integral* equality already proven by `ne_rhf_ccpvqz_highL_engine_os_vs_rys`
(OS==RYS==AUTO SCF energy on g-shells, ~14 s, now exercising the Auto-Rys hoist),
and its basin robustness is already measured by B-2.5c.

**Profiled root cause.** The MCSCF `evaluate` lambda (`casscf.cpp`, called *per
candidate orbital step*, not just per macro) rebuilds the inactive and active Fock
via `ObaraSaika::_compute_fock_rhf(eri, D, nbasis)` — a dense O(n_AO⁴) contraction
over the full materialized AO ERI tensor (`ensure_eri`, ~1.4 GB at cc-pVQZ water).
Cost is n_AO⁴ × (≈7 step scales + stagnation probes) × macros, so it explodes with
basis size even though the active space is tiny. The AO Fock build is *already*
OpenMP-parallel, so more cores help only linearly.

**Levers, ranked.**

- **A (big, the proper fix):** build the inactive/active Fock directly from
  shell-pair ERIs (direct-SCF style) or via the existing RI engine
  (`src/post_hf/ri/`), removing the per-call n_AO⁴ sweep (and eventually the 1.4 GB
  tensor). A genuine MCSCF-engine change — scoped below.
- **B (medium):** the inactive core Fock depends only on the *core* orbitals, which
  move far less than full `C`; build it once per accepted macro instead of per
  candidate.
- **C (low, already planned):** trim the macro-optimizer cascade (vault Open Work
  "CASSCF P2"). NOTE: this previously read "demote numeric-newton to debug-only" —
  **`numeric-newton` is load-bearing** (some cases only converge with it), so it
  must NOT be demoted. The trim is limited to removing redundant per-root
  candidates / pair probes, not the numeric-newton path.

### Phase A — direct/RI Fock in MCSCF

Small, independently-gated, reversible steps. Two full-tensor consumers, on
separate fronts:

1. **AO Fock builds** (the hot, basis-size-explosive path): `build_inactive_fock_mo`
   / `build_active_fock_mo` call `_compute_fock_rhf(eri, D, nbasis)` — dense
   O(n_AO⁴), `J − ½K`, per `evaluate`. **A0–A2.**
2. **AO→active-MO transforms** (`transform_eri_internal`,
   `transform_eri_active_cache`): take the dense `eri` as input. **A4/A5.**

A direct, engine-dispatched Fock entry already exists — `::_compute_2e_fock`
(global-namespace dispatch in `base.h`) — with the same `J − ½K` convention as
`_compute_fock_rhf`. Caveat: the *OS* entry still materializes the dense `nb⁴`
buffer internally, so the tensor-free path is the **HGP/Rys direct kernels**
(per-shell-quartet, screened). Each step gates against the **10 CASSCF regression
cases** (exact energy — direct is the same operator).

- **A0 — seam + equivalence harness. LANDED.** `build_inactive_fock_mo_direct` /
  `build_active_fock_mo_direct` in `orbital.{cpp,h}` take `shell_pairs`
  (+ engine=HGP, tol) instead of `eri`, building via `::_compute_2e_fock`. Not
  wired into production. Gate `planck-casscf-direct-fock`: water/6-31g\*, several
  MO bases × {core, active} densities, `‖F_direct − F_tensor‖_max < 1e-12`
  (measured ≤1.8e-15). Both sides screen at tol_eri=1e-14 to isolate algebra from
  screening.

- **A1 — inactive Fock direct. LANDED.** The hot `evaluate` inactive Fock build now
  calls `build_inactive_fock_mo_direct` when Cartesian. **Cartesian-only guard**
  (`calc._shells._spherical`): the direct kernel emits Cartesian ERIs, but a
  spherical SCF's cached `eri` + MO coefficients are spherical, so a direct build
  would be basis-inconsistent there (same reason `ensure_eri` refuses to rebuild a
  Cartesian tensor in spherical mode); spherical CASSCF keeps the tensor path. Gate:
  all 10 CASSCF cases unchanged (incl. ROHF, SA-2, uphill, and the spherical
  fallback case).

- **A2 — active Fock direct. ATTEMPTED, DROPPED (do not retry naively).** Swapping
  the per-root active Fock to direct flips **only** `water_casscf_sa2_sto3g_sad_guess_uphill`
  from its pinned basin (−74.7877864784) to the other valid SA stationary point
  (−74.7751377977); the other 9 cases pass. Fully diagnosed:
  - **Not a bug, not nondeterminism.** Matches the tensor to ≤8.9e-16 (A0 gate);
    the flipped energy is bit-identical across reruns and `OMP_NUM_THREADS`=1/2/4/8.
  - **Mechanism.** The uphill cascade leans on `numeric-newton` (load-bearing),
    whose FD Hessian `H = (g_orb(+ε) − g_orb(−ε)) / 2ε` divides the active-Fock
    reorder by `2ε`, amplifying ~1e-15 → ~1e-12 in `H`. Macro 1 is bit-identical
    between the tensor and direct trees (`step_norm=8.02e-02` both); they diverge
    deep in the cascade via deterministic chaos over 20+ numeric-newton steps — not
    localizable to one step.
  - **Saves rejected.** Option 1 (consistent `evaluate` path in the ±ε FD legs) is
    already satisfied — A2 makes every `evaluate` direct, so both legs match; the
    drift is base-trajectory, not leg inconsistency. Option 2 (match the direct
    build's `tol_eri` to `ensure_eri`'s 1e-10) was tested — **still landed −74.7751**,
    so it is the fundamental direct-vs-tensor contraction-order difference, not the
    screening set, and cannot be removed without bit-matching (which defeats
    "direct").
  - **Decision.** Drop A2, ship A1 only. A1 (inactive Fock, once per `evaluate`) is
    the dominant per-`evaluate` cost; the active Fock is per-root and a fraction of
    it. The `build_active_fock_mo_direct` builder + its A0 test are **kept** (valid,
    gated) for a future basis-size-gated route.
  - **Only viable revival (A3).** Use direct active Fock **only above an n_AO
    threshold**, so the STO-3G uphill canary (24 AOs) stays on the tensor path while
    large-basis CASSCF — the actual speed target — gets it.

- **A3 — screening tolerance + basis-size gate (also the A2 revival path).** Profile
  sto-3g and 6-31g\* CASSCF before/after; confirm `tol_eri` does not perturb the
  CASSCF energy below the 1e-5 gate. Direct-without-tensor can LOSE to tensor-reuse
  when n_AO is small and the evaluate-count is huge, so the direct path should be
  basis-size-gated regardless.

- **A4 — drop `ensure_eri` (blocked on A5).** Stop materializing the full tensor for
  large bases. Cannot land until A5 removes the transforms' dependence on it.

- **A5 (optional, separate sub-project) — transforms without the full tensor.**
  `transform_eri_internal` / `transform_eri_active_cache` take the dense `eri`.
  Removing it needs an RI/direct AO→active half-transform (the repo's
  `RI::compute_3c_eri` + metric factorization). **RI is a controlled approximation**
  (~1e-8 fitting error, not 1e-12), so it must be opt-in behind a `casscf_ri`
  keyword, never default.

**Landed cut.** A0 + A1 shipped (inactive Fock direct, Cartesian, zero
approximation, 10/10 CASSCF green). A2 dropped. A3+ deferred unless large-basis
CASSCF speed (or peak memory, for A4/A5) is explicitly pursued. Lever B composes
with A1 and is independent.

## How to verify the landed state

```
# from build/
./planck-compute-2e
./planck-os-block-kernel
./planck-hgp-engine-smoke
./planck-rys-box-invariance
./planck-casscf-direct-fock
# from repo root
python3 tests/run_regressions.py --suite all
python3 tests/engine_scf_energy_compare.py <input>   # OS == HGP == Rys == Auto
python3 tests/run_regressions.py --suite all --case ne_rhf_ccpvqz_highL_pyscf
python3 tests/run_regressions.py --suite all --case ne_rhf_ccpvqz_highL_engine_os_vs_rys
# Auto dispatch re-fit (0 median disagreements):
python3 scripts/fit_auto_dispatch.py
```
