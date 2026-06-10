---
name: Mayer Bond Order Density Convention
description: Open-shell Mayer bond order needs a factor of 2 on the per-spin (P S) contractions; closed-shell total-density form has no prefactor
type: gotcha
priority: medium
include_in_claude: true
tags: [populations, density, gotcha, mayer]
---

# Mayer Bond Order Density Convention Gotcha

## The Two Equivalent Forms

`src/populations/bond-order.cpp` (`mayer_bond_order_analysis`) has two
branches that must agree where they overlap (closed shell):

- **Closed shell** (only `total_density` supplied) — canonical form, **no
  prefactor**:

  ```
  B_AB = Σ_{μ∈A, ν∈B} (P_total S)_μν (P_total S)_νμ
  ```

  This gives `B(H–H) = 1` for H2, the textbook single bond.

- **Open shell** (`alpha_density` and `beta_density` supplied) — spin-resolved
  form, which carries an explicit **factor of 2**:

  ```
  B_AB = 2 · Σ_{μ∈A, ν∈B} [ (P^α S)_μν (P^α S)_νμ + (P^β S)_μν (P^β S)_νμ ]
  ```

## Why the factor of 2 (and why it bites)

For a closed shell `P^α = P^β = P_total / 2`, so

```
2 · [ 2 · (½ P_total S)² ] = 2 · 2 · ¼ (P_total S)² = (P_total S)²
```

i.e. the open-shell form reduces to the closed-shell form **only** with the
leading `2`. Drop it and every open-shell bond order comes out at **half** its
correct value (and the closed-shell H2 check would silently disagree with the
open-shell branch by 2×).

The bug originally lived in the **open-shell branch** (missing the `2`). It
went unnoticed because the only unit test passed explicit `&alpha`/`&beta`
densities and asserted the *halved* value, so the test agreed with the buggy
code. The closed-shell total-density branch was correct all along — do **not**
"fix" it by adding a ½; that breaks the correct H2 = 1 result.

## Rule

When a population/bond-order quantity is quadratic in the density, decide up
front whether you are contracting the **total** density `P_total` or the
**per-spin** densities `P^α`, `P^β`. A quadratic in `P_total` and the
spin-summed quadratic differ by a factor of 2 for a closed shell (and the
spin-summed form needs the explicit `2` to match). `total_density` handed in
by `src/driver.cpp` is the full `P_total = 2·C_occ C_occᵀ` (RHF), not a
per-spin density.

## Regression To Keep

- Unit (`tests/population_analysis.cpp`):
  - open-shell `B(0,1) = 0.4948` for the toy α/β densities (was asserting the
    buggy `0.2474`)
  - closed-shell `B(0,1) = 0.49` for `PS = [[1.1,0.7],[0.7,1.1]]` (no prefactor)
- End-to-end: `h2_rhf_mayer_bond_order_sto3g` asserts the printed H–H bond
  order is `1.00000000`.
