# PySCF Reference Calculations

External reference suite for validating Planck CASSCF, SA-CASSCF, CCSD,
CCSDT, and selected DFT energies against PySCF.

## Requirements

```
pip install pyscf
```

PySCF 2.x is required. These scripts have no other dependencies beyond PySCF and the standard library.

## Running

Run all equivalence cases and print a comparison table:

```bash
tests/pyscf/.venv/bin/python tests/pyscf/run_all.py
```

When configuring with CMake, the `planck-pyscf-equivalence` test will
automatically use `tests/pyscf/.venv/bin/python` when that repo-local virtualenv
exists. On other machines you can point CMake at any Python with PySCF installed:

```bash
cmake -S . -B build -DPLANCK_PYSCF_PYTHON=/path/to/python
```

Run specific cases:

```bash
tests/pyscf/.venv/bin/python tests/pyscf/run_all.py --case h2_cas22_sto3g --case water_cas44_sto3g
```

Run a single script directly:

```bash
tests/pyscf/.venv/bin/python tests/pyscf/h2_cas22_sto3g.py
```

List available cases:

```bash
tests/pyscf/.venv/bin/python tests/pyscf/run_all.py --list
```

## Cases

| Script | System | Basis | Active space | Type |
|---|---|---|---|---|
| `h2_cas22_sto3g.py` | H₂ | STO-3G | CAS(2e,2o) | SS-CASSCF |
| `lih_cas22_sto3g.py` | LiH | STO-3G | CAS(2e,2o) | SS-CASSCF |
| `water_cas44_sto3g.py` | H₂O | STO-3G | CAS(4e,4o) | SS-CASSCF |
| `water_cas44_631g.py` | H₂O | 6-31G | CAS(4e,4o) | SS-CASSCF |
| `water_cas44_ccpvdz.py` | H₂O | cc-pVDZ | CAS(4e,4o) | SS-CASSCF |
| `ethylene_casscf_321g.py` | C₂H₄ (planar) | 3-21G | CAS(2e,2o) | SS-CASSCF |
| `ethylene_casscf_ccpvdz.py` | C₂H₄ (planar) | cc-pVDZ | CAS(2e,2o) | SS-CASSCF |
| `water_cas44_sto3g_sa2.py` | H₂O | STO-3G | CAS(4e,4o) | SA-CASSCF (2 roots) |
| `ethylene_cas44_sto3g_sa2.py` | C₂H₄ (90° twisted) | STO-3G | CAS(4e,4o) | SA-CASSCF (2 roots) |
| `h2_dft_hse06_sto3g.py` | H₂ | STO-3G | n/a | RKS HSE06 |
| `h2_dft_b2plyp_sto3g.py` | H₂ | STO-3G | n/a | RKS B2PLYP + scaled MP2 |
| `h2_rccsd_sto3g.py` | H₂ | STO-3G | n/a | RCCSD |
| `lih_rccsdt_sto3g.py` | LiH | STO-3G | n/a | RCCSDT |
| `b_uccsdt_sto3g.py` | B | STO-3G | n/a | UCCSD/UCCSDT |
| `bh3_rccsdt_sto3g.py` | BH₃ | STO-3G | n/a | RCCSDT diagnostic |

The case manifest lives in `tests/pyscf/cases.json`. All geometries match the
Planck `.hfinp` inputs in `tests/benchmarks/casscf/pyscf_reference/` or
`tests/inputs/regression/`.

## DFT Notes

- `h2_dft_hse06_sto3g.py` uses PySCF's native `hse06` libxc mapping.
- PySCF 2.12.1 does not expose libxc's named `b2plyp` functional directly, so
  `h2_dft_b2plyp_sto3g.py` builds the equivalent hybrid reference with the
  custom XC string `0.53*HF + 0.47*B88, 0.73*LYP` and then adds the scaled
  MP2 correction (`0.27 * E_MP2`) before comparing to Planck.
- The DFT equivalence scripts force `fine` grids on both Planck and PySCF so
  they can use microhartree-level tolerances without being dominated by coarse
  quadrature differences.

## Tolerance

- Most CASSCF and CC cross-checks use `1e-7` to `1e-8` Eh tolerances inside the
  individual scripts.
- The BH₃ tensor RCCSDT case is intentionally diagnostic: it is accepted either
  as a full match or as the current known no-fallback non-convergence status.

## Notes on orbital selection

For most systems PySCF's default active orbital selection (ncas orbitals centered on
the HOMO/LUMO gap) matches the active space Planck uses. If a script produces an
energy far from the Planck reference, the likely cause is that PySCF selected
different active orbitals. Fix this with `mc.sort_mo(...)` before calling
`mc.kernel()`:

```python
# Example: select HOMO-1, HOMO, LUMO, LUMO+1 as the 4 active orbitals
nmo = mf.mo_coeff.shape[1]
nocc = mol.nelectron // 2
active_mo_indices = [nocc - 2, nocc - 1, nocc, nocc + 1]  # 1-indexed in sort_mo
mo = mc.sort_mo(active_mo_indices)
mc.kernel(mo)
```

For the twisted ethylene SA-CASSCF case the degenerate C 2p orbitals are nearly
degenerate at 90°, so the HOMO/LUMO gap is very small and PySCF may need explicit
orbital guidance for the CAS(4,4) selection.

## PySCF versions

Scripts were written for PySCF 2.x. The `mc.e_states` attribute for SA-CASSCF
requires PySCF >= 2.0. The `state_average_` API is stable across PySCF 2.x.
