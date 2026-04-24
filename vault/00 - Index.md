---
name: Project Index
description: Master index for the planck-refactored quantum chemistry codebase
type: reference
priority: critical
include_in_claude: true
tags: [index, overview]
---

# Planck — Knowledge Vault Index

Quantum chemistry engine in C++23. Two binaries: `hartree-fock` (HF/MP2/CASSCF) and `planck-dft` (KS-DFT).

## Navigation

### Architecture
- [[Overview]] — module layout, namespace, error handling
- [[Type System]] — `types.h` structs/enums reference
- [[Build System]] — CMake, deps, compile flags
- [[Data Flow]] — input → SCF → output pipeline

### Implementation
- [[SCF and DIIS]] — RHF/UHF loop, convergence, DIIS
- [[CASSCF and SA-CASSCF]] — CASSCF/SA-CASSCF/RASSCF details
- [[Coupled Cluster]] — RCCSD/UCCSD/RCCSDT/UCCSDT/RCCSDTQ + arbitrary-order
- [[DFT]] — KS-DFT, grid, libxc, global hybrids
- [[Integral Engine]] — Obara-Saika / Rys ERI, shell pairs
- [[Gradients and GeomOpt]] — analytic gradients (RHF/UHF/RMP2/UMP2), L-BFGS, IC-BFGS

### Gotchas
- [[Coordinate Units]] — Angstrom vs Bohr pitfalls
- [[Shell Pair Indexing]] — row-major ordering trap
- [[Norm Factors]] — contracted norm folded into coefficients
- [[Error Handling Pattern]] — std::expected throughout

### Status
- [[Completion]] — what is done and validated
- [[Open Work]] — known gaps and polish items

## Quick Reference

| What | Where |
|------|-------|
| All types/enums | `src/base/types.h` |
| Entry point | `src/driver.cpp` |
| Input parser | `src/io/io.cpp` |
| Checkpoint I/O | `src/io/checkpoint.cpp` |
| ERI engine | `src/integrals/os.cpp` |
| CASSCF main loop | `src/post_hf/casscf/casscf.cpp` (`run_mcscf_loop`) |
| CC tensor backend | `src/post_hf/cc/tensor_backend.cpp` |
| DFT driver | `src/dft/driver.cpp` |
