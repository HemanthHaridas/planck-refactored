---
name: HGP Screened inv_2_delta
description: HGP VRR inv_2_delta must scale by boys_scale for screened kernels — silently wrong on mixed bra/ket AM otherwise
type: gotcha
priority: critical
include_in_claude: true
tags: [hgp, integrals, screened, range-separated, gotcha, vrr]
---

# HGP Screened inv_2_delta Gotcha

## Symptom

The HGP `_compute_eri_deriv_elem` returns correct values for Coulomb but
wrong values for `LongRange` / `ShortRange` whenever the quartet has
**matching-axis angular momentum on both bra and ket**. Concrete trigger
patterns:

- (p_y, s | p_y, s): wrong on the LongRange y-component of any centre that
  raises the C / D shell.
- More generally: any (μν|λσ) where the C-VRR cross-coupling term
  `if (ax > 0) dst[m] += ax * inv_2_delta * cross[m+1]` fires.

Coulomb is unaffected because `screen.boys_scale = 1` makes the missing
factor a no-op.

## Root Cause

In `hgp_vrr` ([src/integrals/hgp.cpp](src/integrals/hgp.cpp)) the bra/ket
coupling factor was computed as a bare `0.5 / delta` for every kernel:

```cpp
const double inv_2_delta = 0.5 / delta;  // wrong for screened kernels
```

OS at [src/integrals/os.cpp](src/integrals/os.cpp) builds the same factor
with the screened scaling baked in:

```cpp
const double inv_2_delta =
    (0.5 / delta) *
    ((kernel == HartreeFock::ERIKernel::Coulomb) ? 1.0 : screen.boys_scale);
```

`inv_2_delta` is the prefactor for the cross-coupling term in the C-VRR
recurrence — it only fires when both an A-axis and the matching C-axis
carry angular momentum (lines around `if (ax > 0) ... cross[m+1]` in the
C-VRR x/y/z blocks). For Coulomb, `boys_scale = 1`, so the bare form
happens to be correct. For LongRange / ShortRange the factor is
`boys_scale = λ = ω² / (ω² + ρ)`, and skipping it shifts the cross-term
weight, corrupting any contracted ERI whose VRR path actually traverses
that term.

## Why the existing tests missed it for so long

The early screened tests exercised s-s-s-s (no AM anywhere, cross term
never fires) and p-s-s-s with raises only on the bra (`lCDx = lCDy = lCDz
= 0`, the entire C-VRR loop is skipped). The bug surfaces only on the
(p|s|p|s) mixed-bra/ket pattern, which the targeted derivative gate
introduced when it added p-s-p-s coverage.

## The Fix

Apply the same conditional scaling OS uses:

```cpp
const double inv_2_delta =
    (0.5 / delta) *
    ((kernel == HartreeFock::ERIKernel::Coulomb) ? 1.0 : screen.boys_scale);
```

Validated by the `test_hgp_unweighted_screened_eri_against_os` gate in
[tests/eri_derivative_kernels.cpp](tests/eri_derivative_kernels.cpp):
2352-quartet OS↔HGP sweep on water/STO-3G clean to ~4e-15, plus targeted
checks on s-s-s-s, p-s-s-s, and p-s-p-s for all three kernels.

## Rule

Any screened-kernel scaling that touches the OS recurrence variables
(`inv_2_zetaAB`, `inv_2_zetaCD`, `inv_2_delta`, `WP*`, `WQ*`, `T`,
`prefactor`) must be mirrored exactly in the HGP `hgp_vrr` — they share
the same recurrence algebra and any kernel-aware factor present in one
must be present in the other. When in doubt, diff the two engines'
seed/recurrence blocks side by side.
