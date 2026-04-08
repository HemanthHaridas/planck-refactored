---
name: Build System
description: CMake configuration, dependencies, compile flags, and basis set installation
type: architecture
priority: medium
include_in_claude: true
tags: [build, cmake, dependencies]
---

# Build System

## CMake Setup

- CMake 3.5+, C++23 required, extensions disabled
- Source files collected via `file(GLOB ...)` per module dir — adding a new source file requires re-running `cmake ..`
- Two primary targets: `hartree-fock`, `planck-dft`

## Dependencies

| Dep | How acquired | Type |
|-----|-------------|------|
| Eigen 3.4.0 | `FetchContent` | Header-only, never compiled separately |
| libmsym | `ExternalProject_Add` | Static archive, point-group detection |
| libxc | `ExternalProject_Add` | Static archive, XC functionals; only linked to `planck-dft` |

All deps are hermetic — no system installs needed, fully reproducible.

## OpenMP

`USE_OPENMP` is ON by default. If `find_package(OpenMP)` succeeds, `OpenMP::OpenMP_CXX` is linked and `USE_OPENMP` is defined, activating `#pragma omp parallel` in ERI inner loops and finite-difference Hessian.

## CUDA

Opt-in via `-DUSE_CUDA=ON`. Routes through `gpu/` subdirectory (not in main codebase).

## Basis Path

Configure step generates `src/base/basis.h` from `src/base/basis.h.in`. The generated header provides `get_basis_path()` so runtime basis-set loading resolves without hard-coded paths. Basis sets installed to `share/basis-sets/`.

Available basis sets: `sto-3g`, `3-21g`, `6-31g`, `6-31g*` (in `basis-sets/`).

## Build Commands

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Binaries: build/hartree-fock, build/planck-dft
```
