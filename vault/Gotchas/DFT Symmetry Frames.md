---
name: DFT Symmetry Frames
description: Symmetry-enabled DFT must keep grid, basis, and derivative code in the same coordinate frame
type: gotcha
priority: critical
include_in_claude: true
tags: [dft, symmetry, gradients, grid, frames, gotcha]
---

# DFT Symmetry Frames Gotcha

## Symptom

Symmetry-enabled DFT can produce a wildly wrong analytic gradient even when:

- the same input with `use_symm .false.` passes finite differences
- the same standardized geometry with `use_symm .false.` also passes finite differences
- SCF still converges and prints a plausible point group

The giveaway is a gradient component appearing along an axis that symmetry should forbid.
For water in the `xz` plane, atom 2 should have `y = 0`; the broken path produced a large
nonzero `y` component.

## Root Cause

The DFT codebase has two geometry holders that matter here:

| Field | Meaning |
|-------|---------|
| `molecule._coordinates` | Current working coordinates, often consumed by DFT grid generation |
| `molecule._standard` | Symmetry-standardized coordinates, used by basis construction and derivative code |

In the broken symmetry-enabled DFT path:

- symmetry detection updated `molecule._standard`
- but `molecule._coordinates` was left in the original input frame

That split the calculation across two frames:

- DFT grid construction used `_coordinates`
- basis centers, nuclear repulsion, and XC derivative terms used `_standard`

So the SCF/grid and the derivative machinery were no longer talking about the same molecular
orientation.

## What Fixed It

After `detectSymmetry(...)` succeeds in the DFT prepare path, immediately call:

```cpp
calculator.sync_coordinate_frames_from_standard();
```

This makes `_coordinates` and `_standard` consistent before:

- molecular grid generation
- AO-on-grid evaluation
- KS gradient assembly

## Rule

If a DFT path reorients the molecule with symmetry, every downstream subsystem must use the
same frame before grids or derivatives are built.

Do not assume that updating `_standard` is enough. In DFT, `_coordinates` is still a live
source of truth for the grid layer.

## Regression To Keep

Keep at least one symmetry-on DFT finite-difference regression that uses a rotationally stable
grid, for example:

- water / STO-3G / HSE06 / `use_symm .true.` / `grid ultrafine`

That exact case caught this bug cleanly.
