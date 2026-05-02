---
name: Open Work
description: Known gaps, bugs, and polish items remaining
type: status
priority: high
include_in_claude: true
tags: [status, open-work, bugs, todo]
---

# Open Work

Last updated: 2026-05-01

## Potential Improvements

- Range-separated and double-hybrid DFT (only global hybrids are currently supported; B3LYP and PBE0 are available as of commit f208777)
- Analytic Hessian (currently semi-numerical only)
- ccgen `TensorOptimized` solver path (Phase 4) — scaffolding exists in `src/post_hf/cc/tensor_optimized.{cpp,h}` and `generated_kernel_registry`
