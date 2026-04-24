---
name: Coupled Cluster
description: RCCSD/UCCSD/RCCSDT/UCCSDT/RCCSDTQ and arbitrary-order CC solvers
type: implementation
priority: medium
include_in_claude: true
tags: [cc, ccsd, ccsdt, ccsdtq, ccgen, post-hf]
---

# Coupled Cluster

## Status

Teaching determinant-space prototypes plus a tensor production backend for RCCSD, UCCSD, RCCSDT, UCCSDT, and RCCSDTQ, plus an arbitrary-order RCC solver driven by `ccgen`-generated residuals.

## Namespace

All CC code is in `HartreeFock::Correlation::CC`. Lives under `src/post_hf/cc/`.

## Module Layout

| File | Responsibility |
|------|---------------|
| `common.{cpp,h}` | Reference structs (`RHFReference`, `UHFReference`, `CanonicalRHFCCReference`), small tensor wrappers (`Tensor2D/4D/6D/ND`) |
| `amplitudes.{cpp,h}` | `RCCSDAmplitudes`, `RCCSDTAmplitudes`, `ArbitraryOrderRCCAmplitudes`, zero-builders |
| `mo_blocks.{cpp,h}` | `MOBlockCache` — MO integral blocks cached for reuse |
| `diis.{cpp,h}` | CC-specific DIIS |
| `determinant_space.{cpp,h}` | Determinant-space teaching prototype infrastructure |
| `ccsd.{cpp,h}` | RCCSD and UCCSD teaching solvers |
| `ccsdt.{cpp,h}` | RCCSDT and UCCSDT teaching solvers |
| `ccsdtq.{cpp,h}` | RCCSDTQ solver |
| `tensor_backend.{cpp,h}` | Tensor production backend (selectable via `RCCSDTBackend` enum) |
| `tensor_backend_internal.h` + `tensor_backend_state.cpp` | Private helpers: tensor state prep, memory accounting, ERI block cache, canonical RHF reference construction, dense triples workspace allocation (extracted in commit ff56d66) |
| `solver_arbitrary.{cpp,h}` | General-rank RCC solver using `ArbitraryOrderResiduals`, Jacobi+DIIS via `update_amplitudes_with_jacobi_diis` |
| `tensor_optimized.{cpp,h}` | `TensorOptimized` solver path for ccgen-generated kernels (Phase 4) |
| `generated_kernel_registry.{cpp,h}` | Registry for ccgen-generated kernels |
| `generated_arbitrary_prepare.cpp` + `generated_arbitrary_runtime.{cpp,h}` | Runtime for arbitrary-order generated residuals |

## Tensor Types

All CC-internal tensors are contiguous storage with `operator(i,j,...)` indexing:
- `Tensor2D`, `Tensor4D`, `Tensor6D` — fixed-rank
- `TensorND` — arbitrary-rank owned
- `DenseTensorView` / `ConstDenseTensorView` — non-owning views

As of commit 1593541, `operator()` on all tensor types is bounds-checked and routes through `checked_fixed_rank_index`, returning a static `tensor_error_slot` on out-of-bounds instead of indexing into zero-length vectors.

## Error-Returning API (commit 1593541)

These accessors all return `std::expected<{Const,}DenseTensorView, std::string>`:
- `ArbitraryOrderResiduals::tensor`
- `ArbitraryOrderDenominatorCache::tensor`
- `ArbitraryOrderRCCAmplitudes::tensor`
- `DenominatorCache::tensor`, `RCCSDAmplitudes::tensor`, `RCCSDTAmplitudes::tensor`

Builders return `std::expected`:
- `make_zero_rcc_residuals` (errors if `max_excitation_rank < 1`)
- `unpack_amplitudes` (errors on size mismatch — previously silent no-op)

## Arbitrary-Order RCC Solver

`src/post_hf/cc/solver_arbitrary.{cpp,h}` — general-rank RCC solver:
- `ArbitraryOrderResiduals` holds one residual tensor per excitation rank
- Jacobi + DIIS update via `update_amplitudes_with_jacobi_diis`
- Driven by `ccgen`-generated residuals (Phase 3 integration)
- Tested in `tests/cc_arbitrary_solver.cpp`

## Driver Routing

`src/driver.cpp` lines ~692–787 handle `PostHF::RCCSD`, `UCCSD`, `RCCSDT`, `UCCSDT` (and extensions for RCCSDTQ).

## ccgen Integration

The Python `ccgen` tool (outside `src/`) generates CC equation residuals algebraically and emits C++ kernels. Phase 4 will route through `TensorOptimized` to apply algebraic optimizations.

## Third-Party Attribution

Portions derived from PySCF CC modules; see `src/post_hf/cc/LICENSE-Apache-2.0.txt` and `THIRD_PARTY_LICENSES.md`.
