---
name: Coordinate Units
description: Angstrom vs Bohr pitfalls — which fields hold which units and when they are set
type: gotcha
priority: critical
include_in_claude: true
tags: [coordinates, units, angstrom, bohr, gotcha]
---

# Coordinate Units Gotcha

## The Three Fields

| Field | Units | When set |
|-------|-------|----------|
| `molecule.coordinates` | Angstrom | Input parser |
| `molecule._coordinates` | Bohr | `initialize()` via `_angstrom_to_bohr()` |
| `molecule._standard` | Bohr | `detectSymmetry()` × ANGSTROM_TO_BOHR, or `= _coordinates` when symmetry disabled |

## What Uses What

- **Basis centers**: use `_standard` (Bohr)
- **Nuclear repulsion** (`_compute_nuclear_repulsion()`): uses `_standard` (must be Bohr before calling)
- **Gradient output**: reported in Bohr
- **Geometry optimization displacements**: in Bohr

## The Trap

`_angstrom_to_bohr()` only converts `coordinates → _coordinates`. It does **NOT** set `_standard`.

`_standard` is set by `detectSymmetry()` (which multiplies the symmetry-adapted coordinates by ANGSTROM_TO_BOHR). When symmetry is disabled, `initialize()` sets `_standard = _coordinates`.

**If you add a new code path that bypasses `detectSymmetry()` and doesn't set `_standard`, basis centers and nuclear repulsion will use uninitialized or wrong-unit coordinates — results will be wildly wrong with no error message.**

## Rule

Always ensure `_standard` is in Bohr before calling `_compute_nuclear_repulsion()` or building basis centers. The safe path is: let `initialize()` → `detectSymmetry()` run in the normal flow.
