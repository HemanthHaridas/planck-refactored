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
  io/          — io.cpp (parser), logging.h
  lookup/      — elements.cpp (atomic data)
  basis/       — gaussian.cpp (GBS reader), basis.cpp (normalization)
  integrals/   — shellpair.cpp, os.cpp (Obara-Saika ERI), rys.cpp (Rys quadrature)
  symmetry/    — symmetry.cpp (libmsym wrapper), wrapper.h
  scf/         — scf.cpp (RHF/UHF loop), diis.cpp
  post_hf/
    mp2/       — mp2.cpp (RMP2/UMP2)
    casscf/    — casscf.cpp (macro loop), ci.cpp (FCI/CI step),
                 orbital.cpp (orbital rotation), rdm.cpp (1/2-RDM),
                 response.cpp (SA coupled solve), strings.cpp (active space)
  gradient/    — gradient.cpp (analytic ERI gradient)
  opt/         — opt.cpp (L-BFGS/IC-BFGS geomopt)
  freq/        — freq.cpp (semi-numerical Hessian)
  dft/
    base/      — grid.h, radial.h, angular.h, wrapper.h (libxc), ao_grid.h
    xc_grid.cpp, ks_matrix.cpp, driver.cpp
```
