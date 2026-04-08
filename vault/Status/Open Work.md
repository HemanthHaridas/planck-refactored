---
name: Open Work
description: Known gaps, bugs, and polish items remaining
type: status
priority: high
include_in_claude: true
tags: [status, open-work, bugs, todo]
---

# Open Work

Last updated: 2026-04-08

## Known Bugs

### Per-root total energy display (casscf.cpp:1418)
- **Bug**: Line 1418 stores the CI eigenvalue, not the total energy
- **Fix needed**: total energy = E_CI + E_nuc + E_core (one-electron core energy from doubly occupied orbitals)
- **Impact**: display only — energies used in convergence are correct

## Missing Features

### Orbital Hessian action upgrade
- Current: `matrix_free_hessian_action` uses diagonal energy-denominator model (M3) as preconditioner only
- Needed: full matrix-free Hessian-vector product for better convergence near saddle points
- Files: `src/post_hf/casscf/orbital.cpp`

### SA stationarity assertion in run_regressions.py
- The regression runner parses `sa_g`, `root_screen_g`, `max_root_g` diagnostics
- Missing: assertion that the SA gradient norm is below threshold at convergence
- File: `scripts/run_regressions.py` (or similar)

## Potential Improvements

- Hybrid DFT (exact-exchange mixing) — not yet implemented in `planck-dft`
- Analytic Hessian (currently semi-numerical only)
- TDDFT / linear response
- Natural orbital analysis
- Mulliken / Löwdin population analysis output
