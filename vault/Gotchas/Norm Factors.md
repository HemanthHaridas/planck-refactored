---
name: Norm Factors
description: Contracted Gaussian norm Nc is folded into Shell._coefficients at GBS read time
type: gotcha
priority: high
include_in_claude: true
tags: [normalization, basis, gotcha, integrals]
---

# Norm Factors Gotcha

## What Happens at GBS Read Time

In `src/basis/gaussian.cpp`, when a basis set file is read, the contracted Gaussian normalization constant Nc is computed and immediately multiplied into `Shell._coefficients`:

```cpp
shell._coefficients *= Nc;
```

This means the raw GBS contraction coefficients are replaced by pre-normalized coefficients.

## The Implication

In the integral engine (`os.cpp`, `rys.cpp`), you do **not** need to multiply by a separate Nc factor when computing ERIs. The contraction coefficients you read from `Shell._coefficients` already include normalization.

**If you add Nc multiplication inside the integral loop, you will double-count it and get wrong integrals.**

## What Norms Are NOT Folded

Primitive Gaussian norms (the angular-momentum-dependent factors N_prim) are handled separately within the integral recurrences themselves, not pre-folded. Only the contracted (overall) normalization Nc is pre-folded.

## Verification

If you suspect a normalization issue, check the overlap matrix S: the diagonal elements should all be 1.0 for a correctly normalized basis. Off-diagonal elements should be in [0, 1]. Large diagonal deviations (> 0.01) indicate a normalization bug.
