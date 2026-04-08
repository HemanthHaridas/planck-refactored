---
name: Integral Engine
description: Obara-Saika and Rys quadrature ERI engines, shell pairs, dispatch
type: implementation
priority: high
include_in_claude: true
tags: [integrals, obara-saika, rys, eri, shell-pairs]
---

# Integral Engine

## Overview

Two ERI engines, selected via `IntegralMethod`:
- `ObaraSaika` — Obara-Saika horizontal/vertical recurrences
- `RysQuadrature` — Rys quadrature (alternative)
- `Auto` — dispatch based on angular momenta (picks faster engine per shell quartet)

## Shell Pairs

`src/integrals/shellpair.cpp` + `shellpair.h`

`ShellPair` stores:
- Gaussian product center P = (α_a * A + α_b * B) / (α_a + α_b)
- Prefactor K_ab = exp(-α_a * α_b / (α_a + α_b) * |A-B|²)
- Sum of exponents ζ = α_a + α_b
- References to shell A and shell B

`build_shellpairs` ordering: **row-major** with outer loop over ia, inner loop over ib ≥ ia (upper triangle). This is an important invariant — see [[Shell Pair Indexing]].

## Norm Factors

Contracted Gaussian norm Nc is folded into `Shell._coefficients` during GBS reading:
```
shell._coefficients *= Nc
```
No separate Nc factor needed anywhere in the integral code. See [[Norm Factors]].

## Obara-Saika Engine (`src/integrals/os.cpp`)

Standard OS scheme:
1. Compute primitive integrals via vertical recurrences (VRR)
2. Transfer angular momentum via horizontal recurrences (HRR)
3. Contract over primitives
4. Place result into AO matrix at positions given by `ContractedView._index`

OpenMP parallelized over shell-pair loops when `USE_OPENMP` is defined.

## Index Placement

After computing a block of ERIs for shell quartet (μν|λσ), the result is placed at:
- Row: `sp_mn.A._index` to `sp_mn.A._index + n_a - 1`
- Col: `sp_mn.B._index` to `sp_mn.B._index + n_b - 1`

This requires `ContractedView._index` to be correctly set. **Never use `invert_pair_index`** to recover (ii, jj) — use `sp.A._index` / `sp.B._index` directly.

## Key Files

- `src/integrals/os.cpp` + `os.h` — Obara-Saika ERI
- `src/integrals/rys.cpp` — Rys quadrature ERI
- `src/integrals/shellpair.cpp` + `shellpair.h` — shell pair data and construction
