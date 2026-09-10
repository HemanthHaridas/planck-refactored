# Planck's DFT grid does not converge: ~44 radial points where ~75 are needed

**Status:** scope. Nothing here is built. **This is a general DFT defect, not a
double-hybrid one** — it affects every DFT energy, gradient, geomopt and
frequency in the code.

---

## 1. The measurement

H2O2 (C1) / STO-3G / B2PLYP, Planck vs PySCF at comparable point counts:

| | points | energy step from previous level | gradient step |
|---|---|---|---|
| Planck coarse | 8772 | — | — |
| Planck normal | 17388 | 3.5e-3 | — |
| Planck fine | 30664 | 1.8e-4 | — |
| **Planck ultrafine** | **49824** | **2.9e-5** | **9.3e-5** |
| PySCF level 3 | 47784 | — | — |
| PySCF level 4 | 87936 | 1.3e-7 | 2.7e-6 |
| PySCF level 5 | 133040 | 2.9e-8 | 2.3e-7 |
| PySCF level 6 | 194408 | 9.4e-9 | **1.4e-7** |

**At its tightest setting Planck's gradient still moves by 9.3e-5 Ha/Bohr.
PySCF, at a *smaller* point count, is converged to 1.4e-7 — 680x tighter.**
Planck's ultrafine energy also sits **1.3e-6 Eh outside** PySCF's
grid-converged value, i.e. outside its own convergence band.

## 2. Root cause: the radial axis, isolated and confirmed

Planck's grid presets (`src/dft/base/grid.h`) scale the **angular** scheme with
level while the radial count barely moves. Computed for this molecule:

| level | radial pts (O) | radial pts (H) | max Lebedev (O) |
|---|---|---|---|
| coarse | 32 | 25 | 194 |
| normal | 36 | 27 | 302 |
| fine | 39 | 31 | 434 |
| **ultrafine** | **44** | **34** | **590** |

PySCF's Treutler radial count for Z=8 is **75 (level 3) to 99 (level 5+)**. So
the angular table is comparable (590 vs 590/770) and the **radial axis is
under-resolved by ~2.3x**.

**Confirmed by isolating it** — PySCF with the angular grid pinned at 590
Lebedev, pruning off, varying only the radial count:

```
  nr= 20  E=-149.2137469248
  nr= 30  E=-149.2137509342   dE=4.0e-06
  nr= 44  E=-149.2137896236   dE=3.9e-05   <== Planck ultrafine radial count
  nr= 60  E=-149.2137884810   dE=1.1e-06
  nr= 75  E=-149.2137884990   dE=1.8e-08
  nr= 99  E=-149.2137885011   dE=2.1e-09
```

At `nr = 44` the energy is still moving by **3.9e-5**, matching Planck's own
observed fine->ultrafine shift of 2.9e-5. **Convergence arrives at nr ~ 60-75.**

The count comes from one formula (`radial_point_count`):

```cpp
count = (15.0 * int_acc - 40.0) + radial_row_factor * row;
```

with `radial_row_factor = 5` **fixed at every level** and `int_acc` running
only 4.159 -> 4.959 across coarse->ultrafine. So the radial count can never
exceed ~44 for a second-row atom no matter what the user asks for.

## 3. What was checked and is NOT the cause

- **Pruning.** `pruning_region` keys on the radial **index fraction**
  (`ir / (nr-1)`), not on radius, so the point set per atom is
  geometry-independent and the quadrature stays differentiable. Pruning is a
  plausible-looking culprit for gradient error and it is not this one.
- **The Becke partition.** Its own contribution (XC_III) measures ~1e-6 on a
  real grid — verified independently in
  `tests/pyscf/dh_xc_realgrid_check.py`, where the full
  `d/dR{Phi_XC}` reproduces FD to 1.5e-9 on a 23896-point grid.
- **The angular (Lebedev) table.** Comparable to PySCF's at every level.

## 4. Why it matters beyond the DH work

The DH arc has been **scoring candidates at 1e-5 against a gradient carrying
9e-5 of its own quadrature noise**, and asserted "KS-only gradient accurate to
3.66e-5" at a tolerance *below* that noise floor. More broadly: every DFT
geometry optimization in this code converges forces against a threshold that
may sit under the grid's own error, and every frequency is a finite difference
of such gradients.

**It does NOT confound the `*corr` defect** — `*corr` moves 6.8e-8 between fine
and ultrafine against a 1.24e-3 defect (18000x apart), so
`docs/DH_CORR_GRADIENT_DEFECT_SCOPE.md` is independent and either can be done
first.

## 5. Steps

**G1. Make the radial count reachable.** The one-line version is to raise
`radial_row_factor` with level (it is `5` at all four today) or to widen the
`int_acc` range. **Decide deliberately whether ultrafine should mean nr ~ 75**
(PySCF level 5, converged to 1.8e-8) **or nr ~ 60** (1.1e-6, cheaper). Cost
scales linearly in nr, so nr 44 -> 75 is ~1.7x the points.
*Verify:* the energy step fine->ultrafine drops from 2.9e-5 to <1e-6, and the
gradient step from 9.3e-5 to <1e-6.

**G2. Add a reference-free accuracy gate.** `integrated_electrons` must equal
the electron count exactly, so its deviation measures the quadrature's own
error with no external reference and no FD. A `PLANCK_DEBUG_GRID_ACC` probe is
already committed but sits in a branch the DH path does not reach — move it to
where every DFT run passes.
*Verify:* assert the deviation per grid level on one committed case; it should
fall monotonically and reach ~1e-8 at ultrafine.

**G3. Gate the gradient's grid-independence.** The property that actually
matters is that the analytic gradient stops moving. Add a regression comparing
fine vs ultrafine on one small molecule with a tolerance the fixed grid can
meet (~1e-6), so a future preset change cannot silently regress it.

**G4. Re-baseline what the change moves.** Raising the radial count changes
every DFT energy in the suite at the 1e-5 level. That is a **correctness
improvement, not a regression**, but it means committed reference values must
be re-derived, and the PySCF-gated DFT cases should get *tighter* afterwards,
not looser — which is itself the check that G1 did the right thing.

## 6. Traps

- **Do not "fix" this by loosening a tolerance.** Several DFT cases will shift.
- **A larger point count is not the goal; a converged one is.** Planck's
  ultrafine already has 49824 points and PySCF is converged at 47784 — the
  points are in the wrong place, not too few.
- **Check both the energy AND the gradient.** They converge at different rates,
  and the gradient is the slower and the more consequential of the two.
