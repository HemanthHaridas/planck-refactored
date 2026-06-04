# Scope ROHF CASSCF / RASSCF in small verifiable steps

## Context

CASSCF and RASSCF currently require an RHF reference and reject ROHF outright,
both at the driver level (`src/driver.cpp:1192-1203`, a blanket "ROHF post-HF
references are not implemented" except FCI) and inside the MCSCF loop
(`src/post_hf/casscf/casscf.cpp:269-270`, "only RHF reference supported").
FCI already accepts ROHF (`src/post_hf/fci.cpp:40-42`) by reading the common
spatial-orbital set ROHF stores in the alpha channel. We want the same for
CASSCF/RASSCF so open-shell systems (triplets, doublet radicals) can be treated
at the MCSCF level.

### Why this is small

The existing CASSCF math already supports open-shell systems **as long as the
inactive core is a closed, doubly-occupied shell** (all unpaired electrons live
inside the active space). Specifically:

- C is already read from `calc._info._scf.alpha.mo_coefficients`
  (`casscf.cpp:342-345`), which for ROHF holds the shared spatial orbitals —
  identical to FCI's pattern.
- The active-space spin split is already multiplicity-driven:
  `n_alpha_act = (nactele + (mult-1))/2`, `n_beta_act = nactele - n_alpha_act`
  (`casscf.cpp:309-311`), feeding `build_spin_strings_unfiltered`
  (`casscf.cpp:440`). Same scheme FCI uses.
- The parity guard `(n_total_elec - nactele) % 2 != 0` (`casscf.cpp:288-289`)
  **is** the closed-core gate. When it passes, the core is genuinely paired, so
  the closed-shell `D = 2.0 * C.leftCols(n_core)...` in `build_inactive_fock_mo`
  (`orbital.cpp:62-67`) and `compute_core_energy` (`orbital.cpp:87-95`) are
  correct.
- RASSCF shares `run_mcscf_loop`; `ras1_holes` (`casscf_internal.h:57-62`)
  operates only on the active-space bitstrings, independent of the inactive
  core, so it needs nothing extra.

So the change is: accept ROHF, keep the parity guard as the closed-core gate,
add nothing else. A spin-polarized/open inactive core stays rejected.

## Steps (each independently buildable; only Step 2 changes behavior)

### Step 1 — Relax the casscf.cpp reference guard (first; dormant until Step 2)
File: `src/post_hf/casscf/casscf.cpp:267-270`
- Update the `_is_converged` message: "requires a converged RHF or ROHF reference."
- Replace `if (scf != RHF) reject "only RHF reference supported"` with the FCI
  form: `if (scf != RHF && scf != ROHF) reject "only RHF or ROHF references supported"`.

Ordering: do this before the driver change. Until Step 2, the driver still
blocks ROHF+CASSCF, so `run_mcscf_loop` is unreachable for ROHF and this edit is
dormant — no behavior change, existing RHF cases still pass.

Verify: build; run existing RHF CASSCF regressions
(`h2_cas22_sto3g`, `lih_cas22_sto3g`, `water_cas44_*`, `ethylene_casscf_*`). All
pass unchanged.

### Step 2 — Relax the driver blanket guard (activates ROHF dispatch)
File: `src/driver.cpp:1192-1203`
- Extend the exemption set to also skip CASSCF and RASSCF:
  `correlation != None && != FCI && != CASSCF && != RASSCF`.
- Update the comment to note CASSCF/RASSCF are now allowed under the closed-core
  constraint enforced downstream by the parity guard; other ROHF post-HF
  methods remain unimplemented.

Verify: build; the Step 4 ROHF CASSCF input now reaches `run_mcscf_loop`.
A deliberately bad input (open electron forced into core, odd parity) still
rejects cleanly with the parity message. RHF + ROHF-rejection regressions for
RMP2/CCSD unchanged.

### Step 3 — Update the parity-guard message + document the invariant
File: `src/post_hf/casscf/casscf.cpp:288-289` (and a comment near 309-313)
- The guard stays. Reword: "(n_elec - nactele) must be even: the non-active
  electrons must form a closed, doubly-occupied inactive core (a spin-polarized
  open inactive core is not supported)."
- Add a one-line comment at the active-split (309-313): the inactive core is
  closed-shell (Sz=0); all spin polarization is carried by the active space.

No new numeric guard needed — parity + the existing range checks on
`n_alpha_act`/`n_beta_act` already guarantee all unpaired electrons are active.

Verify: an odd-parity case (e.g. doublet with too-small active space) rejects
with the new message; common-path regressions unchanged.

### Step 4 — Add a CASSCF ROHF regression case (triplet O2)
System: triplet O2, CAS(8,6) in STO-3G — the CASSCF analogue of the existing
`o2_fci_rohf_sto3g` (mult 3). 16 electrons, `(16-8)=8` even → 4 doubly-occupied
core orbitals; active split `n_alpha_act=5, n_beta_act=3`.

- New input `tests/inputs/regression/post_hf/o2_casscf_rohf_sto3g.hfinp`,
  modeled on `o2_fci_rohf_sto3g.hfinp` (ROHF/geom) + a `water_cas44_sto3g`
  CASSCF input (MCSCF knobs). Must use `basis_type cartesian` (PySCF gate uses
  `mol.cart=True`), `scf_type rohf`, `correlation casscf`, `nactele 8`,
  `nactorb 6`, `nroots 1`, charge/mult `0 3`.
- PySCF reference: `mol.cart=True`, `mol.spin=2`, ROHF, `mcscf.CASSCF(mf, 6, (5,3))`.
  Record total energy.
- `tests/regression_cases.json` entry modeled on `water_casscf_sa2_sto3g` /
  `o2_fci_rohf_sto3g`: `expected_exit_code 0`, `contains` the
  "CASSCF : ... Converged." banner, checks
  `metric_close` on `casscf_total_energy` (atol 1e-8) and `metric_le` on
  `casscf_sa_gnorm` (1e-5). The runner already parses "CASSCF Total Energy" and
  `sa_g=` (`tests/run_regressions.py:33,39`).

Verify: `python tests/run_regressions.py` filtered to the new id; then the full
`extended` suite for no regressions.

### Step 5 — Second case for breadth (doublet radical)
A doublet radical (e.g. OH or CN, small active space, `mult 2`) to exercise the
odd-electron / `n_alpha_act - n_beta_act = 1` path and guard against an
O2-specific coincidence. Same input + regression-entry shape as Step 4, with a
separate PySCF reference (`mol.cart=True`, `mol.spin=1`, ROHF, CAS with the
matching active alpha/beta split). Both O2 and the doublet are required
verification.

## Explicitly out of scope (stays rejected)
- Spin-polarized / open inactive core (separate alpha/beta core orbitals). Would
  require alpha/beta core Fock, modified core energy, and modified orbital
  gradient/Hessian core blocks. Stays rejected by the parity guard.
- The known ROHF alpha/beta `mo_energies` bookkeeping asymmetry — CASSCF reads
  only `alpha.mo_coefficients`, never beta energies, so it is unaffected.
- ROHF for RMP2/UMP2/CCSD families — still rejected by the narrowed driver guard.

Note on spin states: CASSCF CI is a full CI in the active space, so multiplicity
only fixes the Sz sector (n_alpha_act - n_beta_act), not the spin eigenstate;
the lowest root in that sector is returned. This is existing behavior, unchanged.

## Files
- `src/post_hf/casscf/casscf.cpp` — guards 267-270, parity 288-289, split 309-313
- `src/driver.cpp` — blanket ROHF guard 1192-1203
- `src/post_hf/casscf/orbital.cpp` — closed-core assumptions 55-96 (read-only confirm; no change)
- `tests/inputs/regression/post_hf/o2_casscf_rohf_sto3g.hfinp` — new (Step 4)
- `tests/inputs/regression/post_hf/<doublet>_casscf_rohf_sto3g.hfinp` — new (Step 5)
- `tests/regression_cases.json` — two new entries
