# How does Planck's TDDFT handle range-separated exchange?

It handles it the same way the SCF does. The response matrices carry one exact-exchange term per
kernel: full-range Coulomb exchange, plus erf short-range exchange when the functional has it.
Both coefficients come from the source the SCF reads. Validated against PySCF for HSE06,
CAM-B3LYP and wB97X (landed 2026-10-16).

## The response matrices

`run_linear_response` (`src/dft/driver.cpp`) builds dense Casida matrices:

```
A(ia,jb) = (e_a - e_i) d_ij d_ab + c_J (ia|jb) + f_xc(ia,jb) - sum_k c_k (ij|ab)_k
B(ia,jb) =                         c_J (ia|bj) + f_xc(ia,jb) - sum_k c_k (ib|ja)_k
```

`k` runs over at most two exchange terms:

| term | coefficient | AO integrals |
|---|---|---|
| full range | `c_fr = fock_exchange_coefficient()` (libxc `XC_HYB_FOCK`) | the dense Coulomb tensor (`ensure_eri`) |
| short range | `c_sr = short_range_exchange_coefficient()` (libxc `XC_HYB_ERF_SR`, the CAM beta) | `_compute_2e(..., ERIKernel::ShortRange, omega)`, built only when `c_sr != 0` |

The SCF (`xc_grid.cpp`) and the SOSCF Hessian build `c_fr K[Coulomb] + c_sr K[ShortRange, omega]`
from the same accessors. That one coefficient source is the design point: the response cannot
disagree with the ground state about which exchange it contains. For a global hybrid `c_sr = 0`
and `c_fr == exact_exchange_coefficient()`, so global-hybrid TDDFT computes exactly what it did
before range separation existed.

Each term goes through one helper, `add_same_spin_exact_exchange` (`src/dft/tddft_exchange.h`),
called by both the RKS and the UKS builder (UKS: same-spin blocks only). The B index lives in
that single place. The cost of range separation is a second dense `nb^4` AO tensor, which fits
the existing dense-response design.

What the matrices deliberately do not contain:

- **No extra XC kernel term.** libxc's second derivative of a range-separated functional already
  contains the screened DFT exchange. `f_xc` is a finite difference of `V_xc` through the same
  grid evaluator as the SCF, which zeroes the correlation slot for combined XC ids, so HSE06 and
  the other combined functionals do not double-count correlation. The PySCF agreement below
  confirms this.
- **No Yukawa or Gaussian screening.** libxc also defines `XC_HYB_YUKAWA_SR` and
  `XC_HYB_GAUSSIAN_SR`, and the ERI engines evaluate only erf.
  `Functional::has_non_erf_short_range_exchange` rejects them by name rather than treating them
  as erf. The SCF does not reject them yet (`vault/Status/Open Work.md`).
- **No range-separated double hybrids.** The PT2 path is untouched and stays rejected.

## The B-matrix defect that came first

Global-hybrid full TDDFT was wrong in both RKS and UKS. The B exchange term read `(ia|bj)`, the
same integral as the Coulomb part of B, where it should read `(ib|ja)`. In the stored
`(occ,virt,virt,occ)` layout the correct element is `k_b[i,b,a,j]`. Water/STO-3G B3LYP against
PySCF 2.13.0, in eV, measured before the fix:

| | root 1 | root 2 | root 3 | root 4 |
|---|---|---|---|---|
| Planck TDA | 11.71369 | 13.97601 | 14.64795 | 17.96275 |
| PySCF TDA | 11.71368 | 13.97602 | 14.64793 | 17.96276 |
| Planck full TDDFT | 11.65965 | 13.96749 | **14.51474** | **17.86475** |
| PySCF full TDDFT | 11.65965 | 13.96750 | **14.50727** | **17.86494** |
| PySCF matrices with Planck's B indexing | 11.65965 | 13.96750 | **14.51472** | **17.86476** |

How the cause was pinned:

- TDA, which has no B matrix, agreed to 1e-5 eV.
- Rebuilding PySCF's own `get_ab()` matrices with Planck's indexing reproduced Planck's numbers
  to 2e-5 eV.
- PBE (no exact exchange) full TDDFT agreed to 1e-5 eV.

**Why nothing caught it:** every TDDFT regression case was PBE, so the exact-exchange term was
never gated. It had to be fixed first, because range separation adds a second exchange term with
the same index pattern and would have inherited the bug.

## Validation against PySCF

`tests/pyscf/rs_tddft_references.py`: PySCF 2.13.0, water/STO-3G Cartesian, grid level 6
(Planck: ultrafine). Coefficients agree before any energy is compared. PySCF reports
`(alpha_LR, hyb_SR)` totals; Planck reports `c_fr = alpha` and `c_sr = hyb - alpha`:

| functional | omega | PySCF (LR, SR) | Planck (c_fr, c_sr) |
|---|---|---|---|
| HSE06 | 0.11 | (0.0, 0.25) | (0.0, 0.25) |
| CAM-B3LYP | 0.33 | (0.65, 0.19) | (0.65, -0.46) |
| wB97X | 0.30 | (1.0, 0.157706) | (1.0, -0.842294) |

Largest root deviation, eV, over 4 roots (RKS) or 3 roots (UKS triplet water):

| | RKS singlet Casida | RKS triplet Casida | RKS singlet TDA | RKS triplet TDA | UKS Casida | UKS TDA |
|---|---|---|---|---|---|---|
| HSE06 | 7.8e-5 | 4.6e-5 | 8.0e-5 | 4.5e-5 | 2.3e-4 | 2.2e-4 |
| CAM-B3LYP | 1.1e-5 | 1.4e-5 | 1.0e-5 | 1.3e-5 | 1.3e-4 | 1.3e-4 |
| wB97X | 1.4e-4 | 3.4e-4 | 1.4e-4 | 2.7e-4 | 3.9e-4 | 4.0e-4 |

The residual is grid and kernel, not exchange:

- **UKS has the same spread with no exact exchange.** UKS PBE is 1.9e-4 eV from PySCF and UKS
  B3LYP 1.7e-4.
- **wB97X is grid-sensitive in PySCF too.** PySCF's own roots move about 1e-4 eV between grid
  levels 7 and 8. Planck's radial grid is known to be coarse for second-row atoms.
- **Ground-state energies agree to 1–5e-6 Eh** for every functional and reference.
- **The gates can see a broken short-range term.** Dropping it moves HSE06 root 1 from 11.85 to
  15.03 eV.

The regression gates use `atol = 5e-4 eV`:

- `water_tddft_hse06_casida_sto3g`
- `water_tddft_hse06_triplet_casida_sto3g`
- `water_tddft_camb3lyp_casida_sto3g`
- `water_tddft_wb97x_tda_sto3g`
- `water_triplet_uks_tddft_camb3lyp_sto3g`
- `water_tddft_camy_b3lyp_rejected_sto3g` (negative case: Yukawa is rejected)

The global-hybrid gates (`water_tddft_b3lyp_{casida,tda}_sto3g`,
`water_triplet_uks_tddft_b3lyp_sto3g`) sit alongside them. `planck-tddft-exchange` pins the B
index on a rectangular occupied/virtual tensor: putting `(ia|bj)` back makes it fail.

## Out of scope

- Imaginary-mode following: the DFT driver does not implement it for any functional.
- Iterative (Davidson) response: the solver stays dense.
- TDDFT excited-state gradients.
