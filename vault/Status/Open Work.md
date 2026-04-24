---
name: Open Work
description: Known gaps, bugs, and polish items remaining
type: status
priority: high
include_in_claude: true
tags: [status, open-work, bugs, todo]
---

# Open Work

Last updated: 2026-04-24

## Missing Features

### SA stationarity assertion in regression runner
- `tests/run_regressions.py` parses `casscf_sa_gnorm` (`sa_g=...` in the log) but `tests/regression_cases.json` has no `lte` check on it for any SA case
- Missing: assertion that the SA gradient norm is below `1e-5` at convergence
- Fix: add `{ "metric": "casscf_sa_gnorm", "type": "lte", "value": 1e-5 }` to all SA entries in `tests/regression_cases.json`

## Potential Improvements

- Range-separated and double-hybrid DFT (only global hybrids are currently supported; B3LYP and PBE0 are available as of commit f208777)
- Analytic Hessian (currently semi-numerical only)
- TDDFT / linear response
- Löwdin population analysis (Mulliken is implemented)
- ccgen `TensorOptimized` solver path (Phase 4) — scaffolding exists in `src/post_hf/cc/tensor_optimized.{cpp,h}` and `generated_kernel_registry`
