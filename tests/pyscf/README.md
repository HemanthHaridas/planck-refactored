# PySCF Reference Calculations

External reference suite for validating Planck CASSCF/SA-CASSCF energies against PySCF.

## Requirements

```
pip install pyscf
```

PySCF 2.x is required. These scripts have no other dependencies beyond PySCF and the standard library.

## Running

Run all cases and print a comparison table:

```bash
python3 tests/pyscf/run_all.py
```

Run specific cases:

```bash
python3 tests/pyscf/run_all.py --case h2_cas22_sto3g water_cas44_sto3g
```

Run a single script directly:

```bash
python3 tests/pyscf/h2_cas22_sto3g.py
```

Adjust tolerance (default 1e-5 Eh):

```bash
python3 tests/pyscf/run_all.py --tolerance 1e-4
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

All geometries match the Planck `.hfinp` inputs in `tests/inputs/casscf_tests/`.

## Planck reference energies

| Case | Planck / Eh |
|---|---|
| h2_cas22_sto3g | −1.1372838351 |
| lih_cas22_sto3g | −7.8811184797 |
| water_cas44_sto3g | −74.4700757755 |
| water_cas44_631g | −75.5497490402 |
| water_cas44_ccpvdz | −75.6045806122 |
| ethylene_casscf_321g | −77.5145223872 |
| ethylene_casscf_ccpvdz | −77.9524855976 |
| water_cas44_sto3g_sa2 (SA-weighted) | −74.7751279351 |
| ethylene_cas44_sto3g_sa2 (SA-weighted) | −77.0034974301 |

## Tolerance

- SS-CASSCF: energies should agree within 1e-5 Eh
- SA-CASSCF: SA-weighted energies should agree within 1e-5 Eh

The tolerance covers differences in convergence thresholds, orbital reordering, and
minor numerical path differences. Larger disagreements indicate a genuine algorithmic
discrepancy.

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
