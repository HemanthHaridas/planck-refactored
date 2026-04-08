---
name: Data Flow
description: How data moves from input file through SCF/post-HF to output
type: architecture
priority: medium
include_in_claude: true
tags: [data-flow, pipeline, initialization]
---

# Data Flow

## Input → Initialization

```
.hfinp file
  └─ io::parse()                   → Calculator (options + Molecule)
       └─ initialize()             → builds Basis, sets _coordinates/_standard (Bohr)
            └─ detectSymmetry()    → sets _standard (symmetry-adapted Bohr coords)
                 └─ build_shellpairs() → vector<ShellPair>
```

**Critical ordering**: `_standard` must be Bohr before `_compute_nuclear_repulsion()` and before building basis centers. `detectSymmetry()` sets `_standard`; when symmetry is disabled, `initialize()` sets `_standard = _coordinates`.

## SCF Loop

```
Calculator
  └─ scf::run()
       ├─ build H_core (T + V_ne)
       ├─ diagonalize S → X (orthogonalizer)
       ├─ SCF iterations:
       │    ├─ build J, K from ERI (os::compute or rys::compute)
       │    ├─ build F = H_core + J - 0.5K
       │    ├─ DIIS extrapolate F
       │    ├─ F' = X†FX → diagonalize → C', ε
       │    ├─ C = XC', build P from occupied MOs
       │    └─ check ΔE + ‖ΔP‖ convergence
       └─ DataSCF{SpinChannel{P, F, C, ε}, ...}
```

## Post-HF Fork

```
PostHF::RMP2/UMP2  → mp2::compute(DataSCF)     → E_corr
PostHF::CASSCF     → casscf::run(DataSCF)       → SA-CASSCF loop
PostHF::RASSCF     → rasscf::run(DataSCF)       → RAS partitioning + CASSCF
```

## Gradient / GeomOpt Pipeline

```
CalculationType::Gradient    → gradient::compute() using DataSCF
CalculationType::GeomOpt     → opt::run(): L-BFGS or IC-BFGS loop
                                    ├─ each step: scf::run() + gradient::compute()
                                    └─ converge on max force + RMS displacement
CalculationType::Frequency   → freq::compute(): semi-numerical Hessian
                                    └─ finite-difference gradients at ±δ geometries
```

## DFT Data Flow

```
.hfinp (or .dftinp)
  └─ DFT::Driver::run()
       ├─ build grid (Treutler-Ahlrichs radial × Lebedev angular × Becke partition)
       ├─ evaluate AOs on grid (ao_grid.h)
       ├─ SCF loop: KS matrix = H_core + J + V_xc
       │    └─ V_xc from xc_grid.cpp via libxc wrapper
       └─ DataSCF (same struct as HF)
```
