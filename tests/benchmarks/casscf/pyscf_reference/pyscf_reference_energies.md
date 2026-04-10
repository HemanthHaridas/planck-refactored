# CASSCF Reference Energies

PySCF 2.12.1 reference values. Scripts: `tests/pyscf/`. Runner: `tests/pyscf/.venv/bin/python tests/pyscf/run_all.py`.
All PySCF scripts use `mol.cart = True` to match Planck `basis_type cartesian`.
Tolerance: 1e-5 Eh.

Last updated: 2026-04-08. Current status: **11/11 passing**.

See `docs/CASSCF_STATUS.md` for full implementation status and remaining work.

---

## Single-root (SS-CASSCF) — 9/9 passing

| Input | Active space | Basis | Planck / Eh | PySCF / Eh | Delta / Eh | Status |
|---|---|---|---|---|---|---|
| h2_cas22_sto3g | CAS(2e,2o) | STO-3G | −1.1372838351 | −1.1372838345 | 6.0e-10 | **PASS** |
| lih_cas22_sto3g | CAS(2e,2o) | STO-3G | −7.8811184797 | −7.8811184639 | 1.6e-08 | **PASS** |
| water_cas44_sto3g | CAS(4e,4o) | STO-3G | −74.9760171760 | −74.9760171635 | 1.2e-08 | **PASS** |
| water_cas44_631g | CAS(4e,4o) | 6-31G | −75.9998609785 | −75.9998609866 | 8.1e-09 | **PASS** |
| water_cas44_ccpvdz | CAS(4e,4o) | cc-pVDZ | −76.0440109052 | −76.0440109036 | 1.6e-09 | **PASS** |
| water_cas44_b1 | CAS(4e,4o) | STO-3G | −74.5856163677 | −74.5856164513 | 8.4e-08 | **PASS** |
| ethylene_casscf_321g | CAS(2e,2o) | 3-21G | −77.5145223872 | −77.5145223959 | 8.7e-09 | **PASS** |
| ethylene_casscf_321g_nroot2 | CAS(2e,2o) | 3-21G | −77.5145223872 | −77.5145223959 | 8.7e-09 | **PASS** |
| ethylene_casscf_ccpvdz | CAS(2e,2o) | cc-pVDZ | −77.9524855977 | −77.9524856209 | 2.3e-08 | **PASS** |

## State-averaged (SA-CASSCF, 2 roots equal weights) — 2/2 passing

| Input | Active space | Basis | Roots | Planck SA / Eh | PySCF SA / Eh | Delta / Eh | Status |
|---|---|---|---|---|---|---|---|
| water_cas44_sto3g_sa2 | CAS(4e,4o) | STO-3G | 2 | −74.7751377977 | −74.7751378317 | 3.4e-08 | **PASS** |
| ethylene_cas44_sto3g_sa2 | CAS(4e,4o) | STO-3G | 2 | −77.0034974301 | −77.0034974774 | 4.7e-08 | **PASS** |

---

## Reference energy notes

### Water CAS(4,4) SS
PySCF references use the hcore-start converged values (both codes start from
the same RHF after the d2d branch-preservation fix, commit 46aa199).
The PySCF SAD-start minimum for water SA-2 (−74.7877865139 Eh, 13 mEh lower)
has not been validated in Planck — see `docs/CASSCF_STATUS.md` item P4.

### Water SA-2
Both codes converge from `guess hcore` to −74.7751378 Eh. The PySCF SAD-start
minimum is lower and has not been compared.

### Twisted ethylene SA-2
Both codes converge from `guess hcore` to −77.0034974 Eh. Planck and PySCF
reach different RHF stationary points from SAD guess (Planck's hcore landing
is 59 mEh lower than PySCF's hcore landing); the hcore-vs-hcore comparison is
the only well-defined reference.

### Ethylene SS (3-21G)
Planck's hcore guess finds a deeper RHF minimum than PySCF's SAD guess
(36 mEh lower). Despite different RHF starting points, both codes converge to
the same CASSCF energy to < 1e-8 Eh, indicating the CASSCF global minimum is
the same.

---

## RHF agreement summary

| Case | PySCF RHF / Eh | Planck RHF / Eh | Delta | Notes |
|---|---|---|---|---|
| h2_cas22_sto3g | −1.1167593074 | −1.1167593103 | 2.9e-9 | |
| lih_cas22_sto3g | −7.8620238601 | −7.8620238776 | 1.8e-8 | |
| water_cas44_sto3g | −74.9629329775 | −74.9629329919 | 1.4e-8 | Bug 1 fixed |
| water_cas44_631g | −75.9839985741 | −75.9839985679 | 6.2e-9 | Bug 1 fixed |
| water_cas44_ccpvdz | −76.0271370536 | −76.0271370568 | 3.2e-9 | Bug 1 fixed |
| water_cas44_b1 | −74.9629329775 | −74.9629329919 | 1.4e-8 | |
| water_cas44_sto3g_sa2 | −74.9629329775 | −74.9629329919 | 1.4e-8 | |
| ethylene_casscf_321g | −77.3837525458 | −77.4195705536 | 3.6e-2 | Planck hcore finds deeper RHF |
| ethylene_casscf_321g_nroot2 | −77.3837525458 | −77.4195705536 | 3.6e-2 | same |
| ethylene_casscf_ccpvdz | −77.8642124227 | −77.8642124071 | 1.6e-8 | |
| ethylene_cas44_sto3g_sa2 | −76.7930232593 | −76.8522465545 | 5.9e-2 | different stationary points |
