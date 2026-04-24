---
name: Type System
description: Complete reference for types.h structs, enums, and key data layouts
type: architecture
priority: high
include_in_claude: true
tags: [types, structs, enums, data-layout]
---

# Type System (`src/base/types.h`)

Single header that every module includes. ~561 lines. Namespace `HartreeFock`. No business logic — data only.

## Key Structs

| Struct | Purpose |
|--------|---------|
| `Calculator` | Top-level container: all options + `DataSCF` |
| `DataSCF` | Holds two `SpinChannel` (alpha, beta) |
| `SpinChannel` | Density matrix, Fock matrix, MO energies, MO coefficients |
| `Molecule` | Atom list, coordinates (Angstrom + Bohr variants), charge, multiplicity |
| `Basis` | `vector<Shell>` + `vector<ContractedView>` |
| `Shell` | Center, `ShellType`, exponents, coefficients, norms |
| `ShellPair` | Gaussian product center, prefactors, exponent sums; refs to A/B shells |
| `ContractedView` | Span-based view into shell data; `_index` = position in basis functions |
| `CASOptions` | Active space spec: `n_active_electrons`, `n_active_orbitals`, state averaging weights |
| `CheckpointData` | MO coefficients, energies, basis tag — for warm-start / cross-basis projection |

## Enumerations

```cpp
enum class ShellType      { S=0, P=1, D=2, F=3, G=4, H=5 };
enum class SCFType        { RHF, UHF };
enum class SCFMode        { Conventional, Direct, Auto };
enum class IntegralMethod { ObaraSaika, RysQuadrature, Auto };
enum class CalculationType{ SinglePoint, Gradient, GeomOpt, Frequency,
                            GeomOptFrequency, ImaginaryFollow };
enum class PostHF         { None, RMP2, UMP2,
                            RCCSD, UCCSD, RCCSDT, UCCSDT, RCCSDTQ,
                            CASSCF, RASSCF };
enum class CoordType      { Cartesian, ZMatrix };
enum class OptCoords      { Cartesian, Internal };

// DFT-specific
enum class DFTGridQuality       { Coarse, Normal, Fine, UltraFine };
enum class XCExchangeFunctional { Custom, Slater, B88, PW91, PBE,
                                  B3LYP, PBE0 };
enum class XCCorrelationFunctional { Custom, VWN5, LYP, P86, PW91, PBE };
```

## ContractedView._index Invariant

`ContractedView._index` MUST be set to the position of the shell's first function in `Basis._basis_functions`. This index is used in `os.cpp` to place ERI contributions into the correct rows/columns of the AO matrix. Missing `_index` = silent wrong results.

## Coordinate Fields on Molecule

| Field | Units | Set by |
|-------|-------|--------|
| `coordinates` | Angstrom | Input parser |
| `_coordinates` | Bohr | `initialize()` via `_angstrom_to_bohr()` |
| `_standard` | Bohr | `detectSymmetry()` or `= _coordinates` when symmetry off |

`_standard` is what basis centers and nuclear repulsion calculations use. See [[Coordinate Units]].
