# ROHF CASSCF / RASSCF Architecture

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers one architecture question:

**How do CASSCF and RASSCF consume an ROHF reference, and what would a
spin-polarized (open) inactive core need that is currently out of scope?**

## Core design choice

The MCSCF loop reads its starting orbitals straight from the converged SCF
reference and is agnostic to whether that reference was RHF or ROHF. This works
because ROHF stores a *single* common set of spatial orbitals shared by both
spin channels (`alpha.mo_coefficients == beta.mo_coefficients`), exactly like
RHF. The reference type only decides which orbitals seed the optimization, not
the structure of the determinant space.

So enabling ROHF is a guard change, not new MCSCF machinery: accept the
reference, keep the existing closed-inactive-core gate, add nothing else.

Files:

- `src/driver.cpp`
- `src/post_hf/casscf/casscf.cpp`

## ROHF-reference contract

A CASSCF/RASSCF run on an ROHF reference:

- reads the common spatial orbitals from `alpha.mo_coefficients`, identical to
  the FCI path (`src/post_hf/fci.cpp`),
- treats the inactive core as closed-shell and doubly occupied,
- carries all spin polarization in the active space,
- requires the unpaired electrons to live entirely inside the active space.

Any open-shell molecule that satisfies the last point (high-spin triplets,
doublet radicals) is supported; anything that would put net spin in the
inactive core is rejected, not approximated.

## How the closed-core condition is enforced

The inactive core enters the core Fock as `2 · C_core C_coreᵀ`
(`build_inactive_fock_mo`) and the core energy assumes paired occupation
(`compute_core_energy`). Both are exact only when the core is genuinely paired.

The existing parity guard is the gate: `(n_elec - nactele)` must be even. When
it passes, the non-active electrons pair up into a closed core and the
closed-shell core machinery is correct; when it fails, the run is rejected with
a message naming the constraint rather than silently using a wrong core. For an
open-shell ROHF reference this is exactly the requirement that the unpaired
electrons sit in the active space.

Files:

- `src/post_hf/casscf/orbital.cpp`
- `src/post_hf/casscf/casscf.cpp`

## How spin is carried by the active space

With the core forced to `Sz = 0`, the requested multiplicity fixes the
active-space `Sz` sector:

```
n_alpha_act = (nactele + (mult - 1)) / 2
n_beta_act  = nactele - n_alpha_act
```

so `n_alpha_act - n_beta_act = mult - 1`. Because the active CI is a *full* CI,
it spans every spin state reachable in that `Sz` sector and returns the
variationally lowest root there. This is the same multiplicity-driven split FCI
uses, feeding `build_spin_strings_unfiltered`.

## Why RASSCF needs nothing extra

RASSCF shares `run_mcscf_loop` and therefore inherits the same reference
handling. Its RAS1/RAS2/RAS3 constraints (`ras1_holes` and friends) act only on
the active-space determinant bitstrings and never touch the inactive core, so
the closed-core condition and the ROHF acceptance carry over unchanged.

## Why a spin-polarized open inactive core is out of scope

Allowing net spin in the inactive core is not an orchestration gap — it needs
new physical plumbing:

- distinct alpha/beta inactive orbitals and a genuinely unrestricted core Fock,
- a core energy that accounts for unequal alpha/beta core occupation,
- modified orbital gradient/Hessian core blocks in the response path.

That is a substantially larger change than the guard relaxation, so the narrow
safe scope is: closed, doubly-occupied inactive core only. The known ROHF
alpha/beta `mo_energies` bookkeeping asymmetry is unrelated here, since the
MCSCF loop reads only `alpha.mo_coefficients` and never the beta energies.

## Validation

PySCF-gated regressions exercise both the triplet and doublet paths:

- `o2_casscf_rohf_sto3g` — triplet O₂ CAS(8,6)/STO-3G
- `oh_casscf_rohf_sto3g` — doublet OH CAS(5,4)/STO-3G, the odd-electron
  (`n_alpha_act - n_beta_act = 1`) case

Both match the PySCF reference to ~1e-8 Eh. The existing RHF CASSCF gate suite
is unchanged, confirming the RHF path is untouched.

Files:

- `tests/inputs/regression/post_hf/{o2,oh}_casscf_rohf_sto3g.hfinp`
- `tests/pyscf/{o2,oh}_casscf_rohf_sto3g.py`
- `tests/regression_cases.json`
