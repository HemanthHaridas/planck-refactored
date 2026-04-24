---
name: Architecture Overview
description: High-level module layout, namespace, and error-handling pattern
type: architecture
priority: high
include_in_claude: true
tags: [architecture, modules, namespace]
---

# Architecture Overview

## Two-Binary Design

| Binary | Entry | Purpose |
|--------|-------|---------|
| `hartree-fock` | `src/driver.cpp` | RHF/UHF SCF, MP2, CASSCF/RASSCF, analytic gradients, geomopt, frequencies |
| `planck-dft` | `src/dft/main.cpp` | Kohn-Sham DFT (RKS/UKS), gradient, geomopt, frequencies |
| `chkdump` | `tools/` | Checkpoint file inspector (BUILD_TOOLS=ON) |

Both share: integrals, basis set handling, symmetry, I/O, geometry optimization.

## Namespace

All HF/MP2/CASSCF code lives in namespace `HartreeFock`. DFT code lives in namespace `DFT`. No mixing.

## Error Handling

Every public function returns `std::expected<T, std::string>`. Errors propagate explicitly with `?` (monadic `.and_then`, `.transform`). No exceptions in the hot path. This makes failure handling visible at every call site.

## Module Map

```
src/
  base/        — types.h (all structs/enums), tables.h (constants), basis.h (path)
  io/          — io.cpp (parser), checkpoint.cpp, logging.h
  lookup/      — elements.cpp (atomic data)
  basis/       — gaussian.cpp (GBS reader), basis.cpp (normalization)
  integrals/   — shellpair.cpp, os.cpp (Obara-Saika ERI), rys.cpp (Rys quadrature)
  symmetry/    — symmetry.cpp (libmsym wrapper), wrapper.h
  scf/         — scf.cpp (RHF/UHF loop), diis.cpp
  post_hf/
    mp2.cpp                — RMP2/UMP2 energies
    mp2_gradient.cpp/.h    — UMP2 gradient intermediates
    rhf_response.cpp/.h    — RHF Z-vector / CPHF machinery (shared with gradients)
    integrals.cpp/.h       — post-HF integral transforms
    casscf.h               — public CASSCF entry
    casscf_internal.h      — private CASSCF types
    casscf/                — casscf.cpp (run_mcscf_loop top-level),
                             casscf_driver_internal.cpp (root-tracking,
                             SA helpers, candidate-step assembly),
                             ci.cpp, orbital.cpp, rdm.cpp,
                             response.cpp, strings.cpp, casscf_utils.h
    cc/                    — RCCSD/UCCSD/RCCSDT/UCCSDT/RCCSDTQ:
                             ccsd.cpp, ccsdt.cpp, ccsdtq.cpp,
                             amplitudes.cpp, common.cpp, mo_blocks.cpp,
                             diis.cpp, determinant_space.cpp,
                             tensor_backend.cpp + tensor_backend_state.cpp,
                             tensor_optimized.cpp, solver_arbitrary.cpp,
                             generated_kernel_registry.cpp,
                             generated_arbitrary_prepare.cpp,
                             generated_arbitrary_runtime.cpp
  gradient/    — gradient.cpp (analytic ERI gradient, RHF/UHF/RMP2/UMP2)
  opt/         — geomopt.cpp (L-BFGS), intcoords.cpp (IC-BFGS)
  freq/        — hessian.cpp (semi-numerical Hessian + vibrational analysis)
  dft/
    base/      — grid.h, radial.h, angular.h, wrapper.h (libxc), ao_grid.h
    xc_grid.cpp, ks_matrix.cpp, driver.cpp, main.cpp
```
