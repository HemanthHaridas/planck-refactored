# How does Planck's TDDFT handle range-separated exchange?

**Status: SCOPE (in flight, opened 2026-10-13).** Rewrite this as an answer when R0–R3 land.

Today it does not: `validate_workflow_support` rejects `LinearResponse` for range-separated
functionals. The temporary override that bypassed that check (`PLANCK_DFT_ALLOW_RS_WORKFLOWS`) was
deleted on 2026-10-12 (`docs/RUNTIME_ENV_OVERRIDES_SCOPE.md`, E1). This scope makes range-separated
TDDFT a supported workflow, validated against PySCF.

## What the code does now (read 2026-10-13)

`run_linear_response` in `src/dft/driver.cpp` builds dense Casida matrices:

```
A(ia,jb) = (e_a - e_i) d_ij d_ab + c_J (ia|jb) - c_x K_A + f_xc(ia,jb)
B(ia,jb) =                         c_J (ia|bj) - c_x K_B + f_xc(ia,jb)
```

It does this twice: once in the RKS branch (singlet/triplet) and once in the UKS branch
(spin-conserving). The integrals come from the dense Coulomb tensor (`ensure_eri`) via
`transform_eri`.

Three facts decide the work:

1. **Exact exchange has one coefficient and one kernel.** `c_x` is
   `exchange_functional.is_hybrid() ? exact_exchange_coefficient() : 0.0`, applied to full-range
   Coulomb integrals only. There is no short-range or long-range term.
2. **The ground-state SCF already does range separation**, with a different source for the
   coefficients. `xc_grid.cpp:460-461` gives `full_range_exchange_coefficient` (libxc `XC_HYB_FOCK`
   terms) and `short_range_exchange_coefficient` (libxc `XC_HYB_ERF_SR`, the CAM beta), plus
   `range_separation_omega`. The SCF and the SOSCF Hessian build `c_fr K[Coulomb] + c_sr
   K[ShortRange, omega]` (`driver.cpp:2715-2745`). That form covers every erf-based CAM functional:
   HSE06, CAM-B3LYP and wB97X. The integral side exists already, as
   `_compute_2e(..., ERIKernel::ShortRange, omega, ...)` in `src/integrals/base.h` for all three
   engines.
3. **The full-TDDFT B exchange term is indexed wrongly (an existing bug, found while scoping).**
   Both branches compute `B(ia,jb) -= c_x * k_b[idx_j_b(i,a,b,j)]`, which is `(ia|bj)`, the same
   integral as the Coulomb part of B. The correct exchange integral is `(ib|ja)`.

## The B-matrix bug, measured

Water/STO-3G (Cartesian), B3LYP, ultrafine grid, against PySCF 2.13.0 (grid level 6). Excitation
energies in eV:

| | root 1 | root 2 | root 3 | root 4 |
|---|---|---|---|---|
| Planck TDA | 11.71369 | 13.97601 | 14.64795 | 17.96275 |
| PySCF TDA | 11.71368 | 13.97602 | 14.64793 | 17.96276 |
| Planck full TDDFT | 11.65965 | 13.96749 | **14.51474** | **17.86475** |
| PySCF full TDDFT | 11.65965 | 13.96750 | **14.50727** | **17.86494** |
| PySCF matrices with Planck's B indexing | 11.65965 | 13.96750 | **14.51472** | **17.86476** |

- **TDA agrees** to about 1e-5 eV. TDA has no B matrix.
- **Full TDDFT is 7.5 meV off on root 3** and 0.2 meV off on root 4.
- **The cause is pinned exactly:** rebuilding PySCF's own `get_ab()` matrices with Planck's `(ia|bj)`
  B-exchange term reproduces Planck's numbers to 2e-5 eV.
- **Control:** PBE (no exact exchange) full TDDFT agrees with PySCF to 1e-5 eV on all four roots.
  The ground-state energy agrees to 1.4e-7 Eh.

**Why nothing caught it:** every TDDFT regression case uses PBE (`h2_dft_tddft_pbe_sto3g`,
`h2_dft_tddft_triplet_pbe_sto3g`, `water_triplet_uks_tddft_{,root2_}pbe_sto3g`). The exact-exchange
term has never been gated, so global-hybrid full TDDFT has been silently wrong in both RKS and UKS.
Range separation adds a second exchange term with the same index pattern, so this has to be fixed
first, or the new term inherits the bug.

## Steps

### R0 — fix the B exchange index and gate global-hybrid TDDFT (small)

- **One exchange helper.** Pull the exchange contribution into one helper that both branches call,
  so the index lives in one place. It takes the MO-transformed blocks and a coefficient, and adds
  `-c K_A` and `-c K_B` with `K_B(ia,jb) = (ib|ja)`. In the existing `(occ,virt,virt,occ)` layout
  of `k_b` that element is `k_b[idx_j_b(i, b, a, j)]`. In UKS the term only exists when target and
  source spin are the same.
- **New regression cases** (the first hybrid TDDFT gates):
  - `water_tddft_b3lyp_casida_sto3g`: RKS singlet, full TDDFT, 4 roots, against the PySCF row
    above. Root 3 is the sensitive one.
  - `water_tddft_b3lyp_tda_sto3g`: the same with TDA, to show the fix leaves TDA alone.
  - A UKS B3LYP full-TDDFT case on an open-shell molecule, since the UKS branch carries the same
    bug.
- **Mutation-verify:** putting `(ia|bj)` back must fail the Casida case at about 7.5e-3 eV.

### R1 — add the short-range exchange term to the response (small-medium)

- **Coefficients from the same source as the SCF.** Take `c_fr`, `c_sr` and `omega` from where the
  SCF gets them (`fock_exchange_coefficient`, `short_range_exchange_coefficient`,
  `range_separation_omega`) instead of `is_hybrid() ? exact_exchange_coefficient()`. For a global
  hybrid `c_sr = 0` and `c_fr` is unchanged, so global-hybrid results after R0 must stay
  byte-identical.
- **Short-range integrals.** When `c_sr != 0`, build the short-range AO tensor once with
  `_compute_2e(shell_pairs, nbasis, ERIKernel::ShortRange, omega, tol_eri, sym_ops)`, transform the
  same two exchange blocks, and call the R0 helper a second time with `c_sr`. The Coulomb `J` and
  `f_xc` terms are untouched.
- **Cost:** a second dense `nb^4` AO tensor. That fits the existing dense-response design, which
  already holds one; flag it rather than optimise it.
- **Reject non-erf kernels.** libxc also defines `XC_HYB_YUKAWA_SR` and `XC_HYB_GAUSSIAN_SR`, which
  the ERI engines cannot evaluate. Reject them by name with an error rather than treating them as
  erf.
- **Exchange-correlation kernel:** for a range-separated functional, libxc's second derivative
  already contains the screened DFT exchange, so no extra term is needed. One check: the RKS branch
  builds its response functionals from `_exchange_id` / `_correlation_id`. Confirm that a combined
  XC id (HSE06 is one) does not double-count correlation, the same defect found in the SOSCF work
  (`docs/DFT_ANALYTIC_FXC_HESSIAN.md`, invariant 3a). B3LYP is also combined XC and already matches
  PySCF on TDA, which is encouraging but not proof for a range-separated functional.

### R2 — validate against PySCF (small)

PySCF 2.13.0 in `tests/pyscf/.venv` runs range-separated TDA and full TDDFT for RKS and UKS
(checked 2026-10-13). Grid differences are not a concern at this size: PBE TDDFT already agrees to
1e-5 eV.

- **Matrix:** HSE06 (`omega = 0.11`, short-range only), CAM-B3LYP (`0.33`, full plus short-range)
  and wB97X (`0.30`, full-range coefficient 1). Each with RKS singlet and triplet, TDA and full,
  plus UKS spin-conserving on an open-shell molecule.
- **Check the coefficients first:** for each functional, the libxc coefficients Planck reads must
  match PySCF's `rsh_coeff`, so a convention difference shows up before any energy comparison.
- **Fixtures:** water/STO-3G closed shell, and one open-shell molecule (e.g. the water cation or
  triplet water) for UKS.
- **Tolerance:** 1e-4 eV per root, about 10x the PBE agreement.

### R3 — enable and gate (small)

- Add `LinearResponse` to the range-separated switch in `validate_workflow_support`, alongside the
  gradient workflows, with a comment recording the PySCF numbers, as the gradient entries do.
- Register the R2 comparisons as regression cases. Keep a negative case asserting that a Yukawa or
  Gaussian range-separated functional is still rejected.
- Range-separated double hybrids stay rejected; this scope does not touch the PT2 path.
- Update `vault/Status/{Completion,Open Work}.md`: drop "`LinearResponse` for range-separated
  functionals remains gated" and record the B-matrix fix.

## Out of scope

- Range-separated imaginary-mode following: the DFT driver does not implement it for any
  functional.
- Iterative (Davidson) response: the solver stays dense, as it is today.
- TDDFT excited-state gradients.

## What would make this an answer

When R0–R3 land, rewrite this file under the same question. Keep the B-matrix bug and how it was
pinned, the "one coefficient source shared with the SCF" design, the PySCF numbers, and why no
earlier gate saw any of it. Drop the step list.
