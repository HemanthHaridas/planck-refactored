---
name: Open Work
description: Known gaps, bugs, and polish items remaining
type: status
priority: high
include_in_claude: true
tags: [status, open-work, bugs, todo]
---

# Open Work

Last updated: 2026-05-22

## Potential Improvements

- DFT gradients / geometry optimization / frequencies / TDDFT for range-separated and double-hybrid functionals are still not implemented; those functionals are currently single-point only
- Analytic Hessian (currently semi-numerical only)
- DFT imaginary-mode following is still not implemented (`src/dft/driver.cpp`)
- ROHF post-HF beyond FCI, analytic gradients, stability analysis, and PCM remain incomplete
- ccgen `TensorOptimized` RCCSDT backend exists, but it is still described in-tree as an experimental/Phase-4 path
