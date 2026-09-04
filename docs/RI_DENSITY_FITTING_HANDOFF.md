# RI / Density-Fitting Architecture

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**What does `mp2_use_ri` fit, which workflows is it allowed in, what pins each piece of the RI gradient, and what is still held back?**

It consolidates eight step-scoped planning notes written while the feature was
built (`RI_GEOMETRY_CACHE_SCOPE.md`, `RI_MP2_GRADIENT_SCOPE.md`,
`RI_RG1A/1B/2/3/4/5_SCOPE.md`). Those were roadmaps; this is the record of
what landed.

## Short answer

RI is opt-in (`mp2_use_ri`) and fits only the MP2 correlation energy. The SCF
is always dense, so `E_RI-MP2 = E_dense_HF + E_RI_corr` and
`dE/dR = dense-HF-gradient + RI-correlation-gradient`. In `mp2_gradient.cpp`
the `vhf1` / `vhf1_rs·rq·pq·ps` HF-reference derivative terms stay dense; only
the correlation 2e-term and the response machinery (CPHF / Lagrangian / veff)
are RI. A reader who forgets this will look for an RI path that does not
exist. The fitted ERI is `(μν|λσ) = J V⁻¹ Jᵀ` with `J_{(μν),Q} = (μν|Q)` and
`V_{PQ} = (P|Q)`, packed over unique AO pairs (μ ≥ ν, off-diagonal pairs
carrying an explicit weight 2). The dense `nb⁴` ERI build is skipped entirely;
the working set drops to `nb²·naux` — 6.5x smaller at water/cc-pVDZ, growing
with system size.

## Where the logic lives

- `src/post_hf/ri/ri_eri.{h,cpp}` — every RI builder
- `src/post_hf/mp2_gradient.cpp` — RMP2/UMP2 gradient, RI branches
- `src/post_hf/{rhf,uhf}_response.cpp` — RI-consistent CPHF
- `src/hf_driver.cpp` — the workflow gate (`ri_workflow_ok`)
- `src/freq/hessian.cpp` — where correlated frequencies are rejected (not RI-specific)
- `tests/rmp2_gradient_fd.py` — the FD self-consistency harness

## What invariants matter

### 1. RI fits only the correlation energy; the SCF reference stays dense

The gradient decomposes as dense-HF-gradient plus RI-correlation-gradient.
Any code review or bug hunt that goes looking for an RI path in the
HF-reference derivative terms (`vhf1` and friends in `mp2_gradient.cpp`) is
looking in the wrong place — those terms are always dense.

Design rule:

- Keep this decomposition explicit at every RI call site; do not introduce a
  fitted SCF-reference term without renaming the design, since the entire
  gate and validation structure assumes the SCF stays dense.

### 2. Geometry-moving RI workflows depend entirely on cache invalidation, not on any geomopt-specific RI code

Four `Calculator` fields cache RI intermediates (`_ri_aux_basis`, `_ri_j2c`,
`_ri_metric_factor`, `_ri_j3c`) and all four depend on atom positions, because
the auxiliary shells sit on the atoms. The original `_ri_j3c` guard keyed off
the packed-pair count `npair × naux` — a function of atom count and basis,
not coordinates — so a geometry that moved the atoms returned the stale
tensor. `ri_invalidate_if_geometry_moved` fixes this with a
`_ri_cache_geometry` stamp keyed off `_molecule._standard`, called once at the
top of `ensure_ri_metric_ready`; `ensure_ri_3c_ready` funnels through the
metric path so one site covers every RI consumer (MP2 / FCI / CASSCF).
Consequently nothing in `geomopt.cpp` is RI-specific — it calls
`compute_{r,u}mp2_gradient` each step and the cache-invalidation layer handles
correctness underneath it.

This is load-bearing, verified by disabling it: RI geomopt does not merely
drift, it dies outright (`GeomOpt line search failed: unable to find an
acceptable step`), because stale first-geometry integrals make the energy and
gradient mutually inconsistent so no step ever decreases the energy. Pinned by
`planck-ri-cache-invalidation` (move an atom 0.3 Bohr on the same
`Calculator`, assert the 3-center tensor changed: `max|B−A|` = 0.882, was 0).

Design rule:

- Any new RI-cached quantity must be invalidated through the same
  `_molecule._standard`-keyed stamp, at the single choke point
  (`ensure_ri_metric_ready`), rather than each consumer inventing its own
  staleness check.

### 3. Every RI factor error found was a convention mismatch with the dense path being mirrored, never a derivation error

All three factor bugs found in this work (below) were caught by a gate, none
by inspection, and all three shared the same shape: the RI builder correctly
implemented the fitting algebra but used the wrong convention (angular
momentum reference point, coupling factor, or density normalization) relative
to the dense path it mirrors.

Design rule:

- When adding a new RI surface, treat "does this match the fitting algebra"
  and "does this match the dense path's convention at this specific point"
  as two separate questions — the algebra was never the bug.
- The low-angular-momentum or symmetric-density case can hide a convention
  bug entirely (see finding 1 below and the HGP screened `inv_2_delta`
  gotcha) — a validation sweep must deliberately exercise higher angular
  momentum and non-symmetric densities, not just the simplest case.

### 4. Bisect the RI surfaces before theorizing about a factor

When the UMP2 2e-term gradient disagreed with finite differences by a
non-uniform ratio across components (1.2956 / 1.2194 / 1.2956 — which reads
as structural, not scalar), the fix was found by temporary env switches
(`RG4_DENSE_{2E,IMAT,VEFF,CPHF}`), one per RI surface, each falling back to
dense, isolating the defect to a single surface in one build rather than
theorizing from the mixed ratios.

Design rule:

- When an assembled multi-surface gradient disagrees with a reference,
  isolate each surface with a dense/RI toggle before reasoning about the
  factor from the combined (and therefore misleading) ratios.

### 5. "Absence of evidence" from a truncated `head`/`grep` probe is not evidence of absence

A probe used `grep -iE 'Total MP2 Energy|Correlation Energy' | head -2` to
check whether UMP2 prints a combined total line. The case-insensitive match
let an unrelated log line (`[INF] UMP2 : Computing MP2 correlation energy`)
consume one of the two `head -2` slots, cutting off the actual `Total MP2
Energy` line that came third — leading to a false conclusion that UMP2 prints
no such line, and an unnecessary fallback path built on that false premise
(see "What was found" below).

Design rule:

- `head -n` on a grep is a truncation, not a search. When establishing that
  something is absent, print the whole match set rather than trusting a
  truncated view.

## What was built

1. Opt-in RI fitting (`mp2_use_ri`) of the MP2 correlation energy only, with
   the dense-SCF-plus-RI-correlation energy and gradient decomposition
   described above.
2. A gate contract (`ri_workflow_ok` in `src/hf_driver.cpp`) permitting RI for
   single-point energy always, `Gradient` and `GeomOpt` for RHF and UHF, and
   deliberately *not* rejecting `Frequency` / `GeomOptFrequency` /
   `ImaginaryFollow` inside the RI gate itself — they fall through to the
   correlated-frequency guard, which gives the accurate rejection reason
   (`hessian.cpp` never calls the MP2 gradient at all, dense or RI, so
   rejecting inside the RI gate would print a misleading "RI does not support
   frequency" message). ROHF is rejected for every gradient-consuming
   workflow, because there is no ROHF-MP2 gradient at all, dense or RI. The
   gate is basis-agnostic and a single predicate, so adding a workflow is a
   one-line change once validated. RI is also wired through FCI and
   CASSCF/RASSCF single-point energies.
3. The RI gradient itself, replacing five dense-ERI surfaces under `use_ri`,
   each independently pinned so a bug in one cannot hide behind a bug in
   another:

   | Surface | RI builder | Unit gate |
   |---|---|---|
   | CPHF orbital Hessian `A` | `build_rhf_cphf_matrix_ri` | `planck-ri-cphf-matrix` (7e-5 vs dense) |
   | veff `J − ½K` (RHF) | `build_ri_fock_rhf` | `planck-ri-jk-equivalence` |
   | veff `{J(Pa+Pb) − K(Pσ)}` (UHF) | `build_ri_fock_uhf` | `planck-ri-jk-equivalence` J4 block |
   | Lagrangian `imat` | `build_ri_imat` | `planck-ri-imat` |
   | 2e-gradient term | `build_ri_two_electron_gradient` | `planck-ri-two-electron-gradient` (8.7e-6) |

   plus the derivative integrals underneath them:

   | Tensor | Builder | Gate |
   |---|---|---|
   | `d/dR (μν\|Q)` element | `compute_3c_deriv_elem` | analytic-vs-FD ~1e-11; Σ_center = 0 |
   | `d/dR (μν\|Q)` packed | `compute_3c_eri_deriv` | whole-tensor vs FD ~1.1e-8 |
   | `d/dR (P\|Q)` metric | `compute_2c_eri_deriv` | whole-tensor vs FD ~7.9e-9 |

## What was found (three factor bugs)

1. **The aux Cartesian norm was fixed at the wrong momenta.**
   `compute_3c_deriv_elem` recomputed the aux normalization `normC` inside its
   contraction loop from the raised/lowered aux momenta the derivative
   recurrence walks — a d-aux raised to f used `cartesian_norm(3,0,0)` instead
   of `cartesian_norm(2,0,0)`. `normC` belongs to the contracted aux basis
   function, fixed at its original `lC`, not to the angular part the
   recurrence visits. This silently broke the translational identity
   `Σ_center = 0` for every p/d-aux element (Σ was O(0.7), not 0). An s-s-s
   test case hid it entirely, because a lone s raise is consistent with
   either interpretation. Found only because RG1a.3's sweep deliberately put
   p and d momentum on μ, so the lowering term `−l_Xq·I(l_X − ê_q)` actually
   fired — the same shape as the HGP screened `inv_2_delta` gotcha.
2. **Both gradient legs couple through `V⁻¹`, not `V^{-1/2}`.** Because the
   fitted ERI is `J V⁻¹ Jᵀ`, the 2e-gradient term is
   `E2(atom,q) = Σ_{(μν),P} w · gamma3 · dJ_{(μν),P} − ½ Σ_{PQ} γ_{PQ} · dV_{PQ}`
   with `gamma3 = Σ Γ·X` where `X = J V⁻¹`, `γ_{PQ} = Σ w·X·gamma3`, and `w`
   the packed off-diagonal doubling `(μ == ν ? 1 : 2)`. The `−½` is real, and
   the 3-center leg carries no factor of 2 — `dJ` already scatters all three
   legs (μ, ν, aux) to their atoms. An early `2·dJ − 1·dV` produced a clean
   4x / 2x / 1x ladder against the dense term until the gate pinned it.
3. **The 2e-term builder carried the RHF density convention.**
   `build_ri_two_electron_gradient` mirrors the RHF dense 2e term, which
   contracts `2.0 * dm2buf` (PySCF `grad/mp2.py`: `de -= ... * 2`). The UMP2
   dense term contracts `dm2a + dm2b` with no factor 2. Feeding `pair_dm2_ao`
   straight in therefore double-counted, and `mp2_gradient.cpp` now scales
   the UMP2 call by `0.5`. The FD gate caught this at 3.5e-2 against a 3e-4
   tolerance, and the isolating build (via the `RG4_DENSE_*` switches, per
   invariant 4) showed RI/dense = 2.0000 on every non-zero component of that
   one surface once bisected. The predicted risk going in was actually a
   different one (the scope had flagged the UHF-`K`-has-no-½ factor as the
   danger, and that one was right on the first try, pinned directly by the
   JK gate: `G_uhf(D/2, D/2) == G_rhf(D)` to `rel = 0` exactly).

## What was found: one non-bug worth recording

RG4 added a fallback to `tests/rmp2_gradient_fd.py` that reconstructed the
UMP2 correlated total as `Total Energy` + `Correlation Energy`, on the false
premise that UMP2 prints no combined `Total MP2 Energy` line (see invariant 5
for the truncated-probe cause). Nothing was ever wrong numerically:
`parse_mp2_energy` tries the primary regex first, it always matched, so the
fallback was unreachable and would have produced the same number anyway. But
it carried a latent trap — its `Total Energy` regex took the last match, which
in a geomopt run is the final optimized SCF energy rather than the one paired
with the correlation piece. Harmless while unreachable, wrong the moment the
primary regex ever failed. The fallback has been removed.

## Validation strategy that should remain in place

- **Self-consistency (the real gate): finite differences of the Planck RI
  energy itself.** The analytic RI gradient must equal the numerical
  derivative of the RI energy:

  | Case | max \|Δg\| |
  |---|---|
  | `water_ri_rmp2_gradient_fd` | 3.2e-7 Ha/Bohr |
  | `water_radical_cation_ri_ump2_gradient_fd` | 1.9e-7 Ha/Bohr |
  | `water_ri_rmp2_geomopt_stationary_fd` | 1.4e-7 Ha/Bohr |
  | `water_radical_cation_ri_ump2_geomopt_stationary_fd` | 1.4e-7 Ha/Bohr |

  The two `*_stationary_fd` cases are a single-point RI gradient at the
  geometry the geomopt converged to. They assert two things at once: the
  analytic gradient there is approximately 0 (a genuine stationary point)
  and it matches FD-of-the-RI-energy (the gradient really is that surface's
  derivative). Together they prove the optimizer landed on a stationary
  point of the RI surface, exactly what a geometry-moving RI run needs.
- **Cross-check against Planck dense MP2, which is itself PySCF-gated** —
  there is no external reference for a fitted MP2 gradient at all: PySCF
  2.13.0 raises `NotImplementedError` for the DF-MP2 analytic gradient by
  every route (`mp.MP2(mf).density_fit()`, DF-HF + `mp.MP2`, and
  `pyscf.mp.dfmp2_native.DFMP2`), so do not look for one.

  | | E_opt (Eh) | O-H (A) | angle |
  |---|---|---|---|
  | PySCF dense RMP2 geomopt | -75.0061363107 | 1.013460 | 97.2729 deg |
  | Planck dense RMP2 geomopt | -75.0061363361 | 1.013462 | 97.2703 deg |
  | Planck RI-RMP2 geomopt | -75.0061310245 | 1.013461 | 97.2704 deg |

  Planck dense matches PySCF to 2.5e-8 Eh; Planck RI sits 5.3e-6 Eh above
  dense (RI fitting error) at the same geometry to 1e-6 A. The UMP2 pair
  behaves identically (dense PySCF-gated to 1.0e-7 Eh; RI 4.0e-6 Eh above at
  the same minimum to 7.3e-7 A). RI energies themselves are separately
  PySCF-validated: DF-FCI to 1.5e-9, DF-CASSCF to 2.66e-5 (an
  optimizer-formulation spread, not fitting error).
- Reproduction commands:

  ```bash
  cmake --build build --target hartree-fock -j
  ctest --test-dir build -R planck-ri            # 9 RI unit gates

  # End-to-end FD-of-RI-energy self-consistency
  python3 tests/run_regressions.py --suite all --case water_ri_rmp2_gradient_fd
  python3 tests/run_regressions.py --suite all --case water_radical_cation_ri_ump2_gradient_fd

  # Geomopt landed on a stationary point of the RI surface
  python3 tests/run_regressions.py --suite all --case water_ri_rmp2_geomopt_stationary_fd
  python3 tests/run_regressions.py --suite all --case water_radical_cation_ri_ump2_geomopt_stationary_fd

  # Guard ordering: RI + frequency must hit the FREQUENCY guard, not the RI gate
  python3 tests/run_regressions.py --suite all --case water_ri_rmp2_freq_rejected
  ```

  To confirm the cache-invalidation fix (invariant 2) is load-bearing, make
  `ri_invalidate_if_geometry_moved` return early and re-run
  `water_ri_rmp2_geomopt_sto3g`: the line search fails.

## Conventions a new reader will trip on

- **UHF Fock has no ½ on K.** `G_σ = J(Pa + Pb) − K(P_σ)`: Coulomb from the
  total density, exchange per-spin. The closed-shell `J − ½K` carries that ½
  only because RHF's `P` is doubly occupied.
- **`bra_prefolded`.** `build_ri_two_electron_gradient` takes a flag. The
  RG2.2 synthetic gate used a bra-symmetric density, giving a packed single
  ordering with `w = (μ==ν?1:2)`. The real `dm2buf` is not bra-symmetric, so
  `gamma3` must sum both `(μν)` and `(νμ)` orderings and use `w = 1`
  (`bra_prefolded = true`). Same algebra, different packing.
- **`Total MP2 Energy` is printed for UMP2 too.** `hf_driver.cpp` maps both
  `PostHF::RMP2` and `PostHF::UMP2` to the same `"MP2"` `method_label`, which
  `Logger::correlation_energy` renders as `Total <label> Energy` for both
  restricted and unrestricted, dense and RI. `tests/rmp2_gradient_fd.py`
  parses that one line for every case, and always has.

## Remaining architecture concern

What is still held back, none of it RI-specific:

- **UMP2 / RMP2 frequencies** are blocked on a general gap, not RI —
  `hessian.cpp` never calls the MP2 gradient at all. Fixing that unlocks
  dense and RI frequencies together.
- **Spherical-basis RI gradients** — the MP2 gradient is rejected in the
  spherical basis for the dense path too, pending the response-machinery
  audit; RI inherits the rejection.
- **ROHF** — no ROHF-MP2 gradient exists, dense or RI.
- **RI-CASSCF geometry optimization** — unreachable for an unrelated reason:
  there is no analytic CASSCF gradient at all.
