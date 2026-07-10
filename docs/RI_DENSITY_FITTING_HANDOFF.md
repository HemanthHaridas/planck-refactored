# RI / Density-Fitting Architecture

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers one architecture question:

**What does `mp2_use_ri` fit, which workflows is it allowed in, what pins each
piece of the RI gradient, and what is still held back?**

It consolidates eight step-scoped planning notes written while the feature was
built (`RI_GEOMETRY_CACHE_SCOPE.md`, `RI_MP2_GRADIENT_SCOPE.md`, `RI_RG1A/1B/2/3/4/5_SCOPE.md`).
Those were roadmaps; this is the record of what landed.

## Core design choice

RI is **opt-in** (`mp2_use_ri`) and fits **only the MP2 correlation energy**.
The SCF is *always* dense. So

```
E_RI-MP2  = E_dense_HF + E_RI_corr
dE/dR     = dense-HF-gradient + RI-correlation-gradient
```

This one sentence explains most of the code's shape. In `mp2_gradient.cpp` the
`vhf1` / `vhf1_rs·rq·pq·ps` HF-reference derivative terms **stay dense**; only the
correlation 2e-term and the response machinery (CPHF / Lagrangian / veff) are RI.
A reader who forgets this will look for an RI path that does not exist.

The fitted ERI is

```
(μν|λσ) = J V⁻¹ Jᵀ ,   J_{(μν),Q} = (μν|Q) ,   V_{PQ} = (P|Q)
```

with `J` packed over unique AO pairs (μ ≥ ν, off-diagonal pairs carrying an
explicit weight 2). The dense `nb⁴` ERI build is skipped entirely; the working
set drops to `nb²·naux` — 6.5× smaller at water/cc-pVDZ, and the ratio grows with
system size.

Files:

- `src/post_hf/ri/ri_eri.{h,cpp}` — every RI builder
- `src/post_hf/mp2_gradient.cpp` — RMP2/UMP2 gradient, RI branches
- `src/post_hf/{rhf,uhf}_response.cpp` — RI-consistent CPHF
- `src/hf_driver.cpp` — the workflow gate

## Gate contract

`mp2_use_ri` permits:

- single-point energy — always,
- `Gradient` — RHF and UHF,
- `GeomOpt` — RHF and UHF,
- `Frequency` / `GeomOptFrequency` / `ImaginaryFollow` — **not rejected here**.
  They fall through to the correlated-frequency guard, which is the accurate
  reason (below).
- ROHF — rejected, for every gradient-consuming workflow.

Basis-agnostic. The gate is a single `ri_workflow_ok` predicate in
`src/hf_driver.cpp`; adding a workflow is a one-line change once it is validated.

RI is also wired through FCI and CASSCF/RASSCF energies (single point).

### Why frequency falls *through* the RI gate, not into it

`src/freq/hessian.cpp` finite-differences the **SCF** gradient, dispatching on
reference type only (`compute_{rhf,uhf,rohf}_gradient`). It never calls the MP2
gradient. So `correlation rmp2|ump2` + `frequency` would print an MP2 correlation
energy and then report frequencies built from an **HF** Hessian — a silently wrong
answer, and one that has nothing to do with RI: it is true of every MP2, dense or
fitted.

Rejecting it inside the RI gate would emit a misleading "RI does not support
frequency". So the RI gate deliberately lets freq workflows pass, and the
correlated-frequency guard in `hf_driver.cpp` rejects them with the real reason.
`water_ri_rmp2_freq_rejected` asserts the **frequency** guard's text, so a
regression that lets the RI gate swallow them is caught.

### Why ROHF is excluded

There is no ROHF-MP2 gradient at all — dense or RI. Nothing to fit.

## Geometry-moving workflows rest on cache invalidation (G1)

Four `Calculator` fields cache RI intermediates and **all four depend on atom
positions**, because the auxiliary shells sit on the atoms:

| Field | Holds |
|---|---|
| `_ri_aux_basis` | aux shells on atom centers |
| `_ri_j2c` | 2-center metric `(P\|Q)` |
| `_ri_metric_factor` | Cholesky/eigen factor of `_ri_j2c` |
| `_ri_j3c` | packed 3-center `(μν\|Q)` |

The original `_ri_j3c` guard keyed off the packed-pair count `npair × naux` —
functions of *atom count and basis*, **not coordinates**. A geometry that moved
the atoms returned the stale tensor.

`ri_invalidate_if_geometry_moved` (`ri_eri.cpp`) fixes this with a
`_ri_cache_geometry` stamp, called once at the top of `ensure_ri_metric_ready`;
`ensure_ri_3c_ready` funnels through the metric path, so that one site covers
every RI consumer (MP2 / FCI / CASSCF). It keys off `_molecule._standard` —
exactly the field the geomopt inner loop updates each step
(`sync_coordinate_frames_from_standard`).

**This is load-bearing, and verified so.** Disable the invalidation and RI geomopt
does not merely drift — it *dies*:

```
[ERR] Geometry Optimization : GeomOpt line search failed: unable to find an acceptable step
```

Stale first-geometry integrals make the energy and the gradient mutually
inconsistent, so no step ever decreases the energy. Pinned by
`planck-ri-cache-invalidation` (move an atom 0.3 Bohr on the same `Calculator`,
assert the 3-center tensor changed: `max|B−A|` = 0.882, was 0).

Consequently **nothing in `geomopt.cpp` is RI-specific**. It calls
`compute_{r,u}mp2_gradient` each step; G1 handles the caches.

## The RI gradient: five surfaces, each independently pinned

Under `use_ri` the RMP2/UMP2 gradient replaces five dense-ERI surfaces. Each has
its own fitting-accuracy unit gate, and the assembled gradient then has an
end-to-end FD gate. A bug in one cannot hide behind a bug in another.

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

### End-to-end validation

There is **no external reference** for a fitted MP2 gradient: PySCF 2.13.0 raises
`NotImplementedError` for the DF-MP2 analytic gradient by *every* route —
`mp.MP2(mf).density_fit()`, DF-HF + `mp.MP2`, and `pyscf.mp.dfmp2_native.DFMP2`.
Do not go looking for one.

So the RI gradient is validated two ways, and both matter:

**1. FD of the Planck RI energy** (self-consistency — the real gate). The analytic
RI gradient must equal the numerical derivative of the *RI* energy:

| Case | max \|Δg\| |
|---|---|
| `water_ri_rmp2_gradient_fd` | 3.2e-7 Ha/Bohr |
| `water_radical_cation_ri_ump2_gradient_fd` | 1.9e-7 Ha/Bohr |
| `water_ri_rmp2_geomopt_stationary_fd` | 1.4e-7 Ha/Bohr |
| `water_radical_cation_ri_ump2_geomopt_stationary_fd` | 1.4e-7 Ha/Bohr |

The two `*_stationary_fd` cases are a single-point RI gradient *at the geometry
the geomopt converged to*. They assert two things at once: the analytic gradient
there is ≈ 0 (a genuine stationary point) **and** it matches FD-of-the-RI-energy
(the gradient really is that surface's derivative). Together they prove the
optimizer landed on a stationary point **of the RI surface**, which is exactly
what a geometry-moving RI run needs.

**2. Cross-check against Planck dense MP2, which *is* PySCF-gated.**

| | E_opt (Eh) | O–H (Å) | angle |
|---|---|---|---|
| PySCF dense RMP2 geomopt | −75.0061363107 | 1.013460 | 97.2729° |
| Planck dense RMP2 geomopt | −75.0061363361 | 1.013462 | 97.2703° |
| Planck **RI**-RMP2 geomopt | −75.0061310245 | 1.013461 | 97.2704° |

Planck dense matches PySCF to **2.5e-8 Eh**; Planck RI sits **5.3e-6 Eh** above
dense — RI fitting error — at the same geometry to **1e-6 Å**. The UMP2 pair
behaves identically (dense PySCF-gated to 1.0e-7 Eh; RI 4.0e-6 Eh above at the
same minimum to 7.3e-7 Å).

RI energies themselves are PySCF-validated elsewhere: DF-FCI to 1.5e-9, DF-CASSCF
to 2.66e-5 (an optimizer-formulation spread, not fitting error).

## Three factor bugs, and what each one teaches

Every RI factor error found in this work was a **convention mismatch with the
dense path being mirrored**, never a derivation error in the fitting algebra. All
three were caught by a gate, none by inspection.

### 1. The aux Cartesian norm is fixed at the *original* momenta

`compute_3c_deriv_elem` recomputed the aux normalization `normC` inside its
contraction loop from the **raised/lowered** aux momenta the derivative recurrence
walks. A d-aux raised to f used `cartesian_norm(3,0,0)` instead of
`cartesian_norm(2,0,0)`.

`normC` belongs to the contracted aux basis *function* — fixed at its original
`lC` — not to the angular part the recurrence visits. This silently broke the
translational identity `Σ_center = 0` for every p/d-aux element (Σ was O(0.7),
not 0). **s-s-s hid it entirely**, because a lone s raise is consistent.

Found only because RG1a.3's sweep deliberately put p and d momentum on μ, so the
lowering term `−l_Xq·I(l_X − ê_q)` actually fired. Same shape as the HGP screened
`inv_2_delta` gotcha: *the low-angular-momentum case does not exercise the term
that is wrong.* RG1b pre-empted the same trap on both metric legs.

### 2. Both gradient legs couple through `V⁻¹`, not `V^{-1/2}`

Because the fitted ERI is `J V⁻¹ Jᵀ`, the 2e-gradient term is

```
E2(atom,q) = Σ_{(μν),P} w · gamma3 · dJ_{(μν),P}  −  ½ Σ_{PQ} γ_{PQ} · dV_{PQ}
```

with `gamma3 = Σ Γ·X` where `X = J V⁻¹`, `γ_{PQ} = Σ w·X·gamma3`, and `w` the
packed off-diagonal doubling `(μ == ν ? 1 : 2)`.

The **`−½` is real**, and the 3-center leg carries **no** factor of 2 — `dJ`
already scatters all three legs (μ, ν, aux) to their atoms. An early
`2·dJ − 1·dV` produced a clean 4× / 2× / 1× ladder against the dense term until
the gate pinned it.

### 3. The 2e-term builder carries the *RHF* density convention

`build_ri_two_electron_gradient` mirrors the RHF dense 2e term, which contracts
`2.0 * dm2buf` (PySCF `grad/mp2.py`: `de -= ... * 2`). The **UMP2** dense term
contracts `dm2a + dm2b` with **no** factor 2. Feeding `pair_dm2_ao` straight in
therefore double-counts, and `mp2_gradient.cpp` scales the UMP2 call by `0.5`.

This one is worth reading for the *method*, because the symptom lied. The FD gate
caught it (3.5e-2 vs a 3e-4 tolerance), but the per-component ratios were
1.2956 / 1.2194 / 1.2956 — non-uniform, which reads as a *structural* error, not a
scalar one. Guessing would have wasted a day.

Instead, four temporary env switches (`RG4_DENSE_{2E,IMAT,VEFF,CPHF}`), one per RI
surface, each falling back to dense. One build isolated it: only `RG4_DENSE_2E`
restored FD agreement. A direct probe of that term alone then showed
**RI/dense = 2.0000** on every non-zero component, uniform across atoms and axes.
The misleading ratios were just the *total* gradient mixing one doubled term with
several correct ones.

**Bisect the RI surfaces before theorising about a factor.** It is one build.

The predicted risk, incidentally, was the wrong one: the scope flagged the
UHF-`K`-has-no-½ factor as the danger. That was right first try. The JK gate pins
it directly — `G_uhf(D/2, D/2) == G_rhf(D)` to `rel = 0` exactly.

## Conventions a new reader will trip on

- **UHF Fock has no ½ on K.** `G_σ = J(Pa + Pb) − K(P_σ)`: Coulomb from the
  *total* density, exchange per-spin. The closed-shell `J − ½K` carries that ½
  only because RHF's `P` is doubly occupied.
- **`bra_prefolded`.** `build_ri_two_electron_gradient` takes a flag. The RG2.2
  synthetic gate used a bra-*symmetric* density → packed single ordering with
  `w = (μ==ν?1:2)`. The real `dm2buf` is **not** bra-symmetric → `gamma3` must sum
  both `(μν)` and `(νμ)` orderings and use `w = 1` (`bra_prefolded = true`). Same
  algebra, different packing.
- **UMP2 prints no `Total MP2 Energy` line**, only `Total Energy` +
  `Correlation Energy`. `tests/rmp2_gradient_fd.py` reconstructs the correlated
  total from the pair. That parser change was validated against the *dense* UMP2
  gradient (5e-8) before being trusted on RI — a broken parser would otherwise
  have silently "validated" a broken gradient.

## What is still held back

- **UMP2 / RMP2 frequencies** — blocked on a general gap, not RI: `hessian.cpp`
  never calls the MP2 gradient (above). Fixing that unlocks dense *and* RI
  frequencies together.
- **Spherical-basis RI gradients** — the MP2 gradient is rejected in the spherical
  basis for the dense path too, pending the response-machinery audit. RI inherits
  the rejection.
- **ROHF** — no ROHF-MP2 gradient exists.
- **RI-CASSCF geometry optimization** — unreachable for an unrelated reason: there
  is no analytic CASSCF gradient at all.

## How to reproduce the validation

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

To confirm G1 is load-bearing, make `ri_invalidate_if_geometry_moved` return
early and re-run `water_ri_rmp2_geomopt_sto3g`: the line search fails.
