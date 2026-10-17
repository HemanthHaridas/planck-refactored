# Effective Core Potential (ECP) support

**Status:** scope. Nothing here is built — there is no ECP code anywhere in
`src/`, no ECP block in any shipped basis set, and no `%begin_ecp`-style input
keyword. This doc exists to answer one question: what has to be added, and in
what order, for `hartree-fock`/`planck-dft` to run a calculation with a
pseudopotential replacing a heavy atom's core electrons.

---

## 1. Why this doesn't already work

Every basis set in `basis-sets/` (`grep -rl ECP basis-sets/` returns nothing)
is all-electron. The GBS reader (`read_gbs_basis`,
`src/basis/gaussian.cpp:37`) only ever parses primitive Gaussian shell blocks
— it has no concept of an ECP section, and none of the three integral engines
(`os.cpp`, `rys.cpp`, `hgp.cpp`) compute anything beyond the standard
kinetic/nuclear-attraction/ERI set. `H_core` is built as the one-liner
`calculator._hcore = T + V` (`src/hf_driver.cpp:1004`), where `V` is the
plain point-charge nuclear attraction (`_compute_1e`, `src/integrals/os.cpp`).
There is no third term anywhere in that sum.

This is a genuine gap, not a rejected/gated feature the way MP2-spherical-
gradients or ROHF-stability are — nothing here even parses an ECP-bearing
input, so there's no "reject cleanly" boundary to add first. The feature is
additive from zero.

## 2. What an ECP actually replaces

An ECP (pseudopotential) removes an atom's core electrons and substitutes a
one-electron operator `U_ECP` that reproduces the core's effect on valence
orbitals, added into `H_core`:

```
H_core = T + V_nuclear(remaining explicit electrons) + U_ECP
```

`U_ECP` has two pieces, both centered on the ECP atom:

- **Local part** `U_L(r)`: an effective potential that isn't
  angular-momentum-selective, same shape as a nuclear attraction term but
  with `Z_eff` reduced and extra `r^n e^{-ζr²}` radial factors.
- **Semi-local part** `Σ_l U_l(r) P_l`: one term per angular momentum channel
  up to some `l_max`, each `P_l` an angular-momentum projector built from
  real spherical harmonics `Σ_m |lm⟩⟨lm|`. This is what makes ECP integrals
  structurally different from anything OS/Rys/HGP compute — a nuclear
  attraction integral is a single 3-center 1-electron integral; an ECP
  semi-local integral is a **double** radial integral (bra side, ket side)
  coupled through the angular projector, with no closed-form Obara-Saika
  recurrence.

`ecp.py`/`parse_nwchem_ecp.py`, already vendored under
`tests/pyscf/.venv/lib/python3.14/site-packages/pyscf/gto/`, is a working
reference implementation of exactly this split (`type1_by_shell` = local,
`type2_by_shell` = semi-local) and of the NWChem ECP text format most basis
set libraries (Basis Set Exchange, etc.) ship pseudopotentials in. Treat it as
the algorithm and format reference, not something to link against — Planck's
engines are hand-written for a reason (see `docs/ERI_PERFORMANCE_SCOPE.md`
and the libint rejection in memory).

## 3. Machinery required, in dependency order

### 3.1 Data model (`src/base/types.h`, `src/basis/`)

- New `ECPShell`/`ECPBlock` struct: per-element `l_max`, one local channel and
  `l_max` semi-local channels, each a set of `(power n, exponent ζ,
  coefficient c)` triples — structurally parallel to `Shell` but with an
  extra integer power per primitive that `Shell` doesn't carry.
- `Molecule`/`Basis` need a place to hold these per-atom (keyed by
  `atomic_numbers[a]`, same lookup `_shell` construction already uses).
- **Core electron count.** This is the part most likely to bite silently: an
  ECP replaces `n_core` electrons, so `total_nuclear_charge()` and every
  `n_electrons = total_nuclear_charge() - charge` call site
  (`src/scf/scf.cpp:463,1141,1942` — three separate copies, not one) must
  subtract the ECP-replaced core count too, or occupation/Aufbau comes out
  wrong while everything still "runs." SAD guess (`src/scf/sad.cpp`) also
  needs to know an atom's true valence electron count to build a sane
  starting density.

### 3.2 Basis-file format (`src/basis/gaussian.cpp`)

- Extend the GBS reader (or add a sibling reader) for the NWChem ECP block
  format. This is a parsing addition, not a rewrite of `read_gbs_basis` —
  ECP sections are self-delimited (`<element> nelec <n>` header, then
  per-channel primitive lists) and can be read into the new struct
  independently of the existing shell loop.
- At least one ECP-bearing basis set needs to exist under `basis-sets/`
  before anything downstream is testable — current sets top out around
  `aug-cc-pV7Z`/`cc-pV9Z`, all-electron, nothing 4th-row+. `LANL2DZ` or a
  `def2-*`/`ccECP` set (PySCF's venv already ships `ccecp-basis` — see
  `tests/pyscf/.venv/lib/python3.14/site-packages/pyscf/gto/basis/`) is the
  natural first import, both for having a real pseudopotential and for
  giving PySCF a same-format reference to validate against.

### 3.3 Integral engine — the actual new subsystem

This is not an extension of OS/Rys/HGP; it's a fourth, structurally distinct
engine, because ECP integrals have no VRR/HRR recurrence of that shape:

- **Local part**: closest to existing code — a modified nuclear-attraction
  integral with extra radial powers, reachable by generalizing the existing
  Boys-function machinery in `os.cpp` rather than writing new quadrature.
- **Semi-local part**: needs the angular-momentum projection via the real
  spherical harmonic addition theorem, and a genuinely new radial quadrature
  (Gauss-Rys-type over the coupled bra/ket radial integral — see
  `type2_by_shell` in the vendored `ecp.py` for the reference recurrence).
  This is the piece with no existing analogue anywhere in `src/integrals/`.
- Lives in a new `src/integrals/ecp.{h,cpp}`. Public entry point should mirror
  `_compute_1e`'s shape (shell-pair-like loop, but bra/ket driven by the ECP
  atom rather than a shell pair) so `H_core` assembly stays close to today's
  `H = T + V` one-liner: `H = T + V + V_ecp`.
- **Symmetry**: full-point-group direct-SCF (`src/symmetry/{os,rys,hgp}_symm.cpp`)
  would need its own ECP hookup to stay correct under `use_symm true` — out
  of scope for a first landing; gate ECP+symmetry off explicitly rather than
  let it silently ignore the operator.

### 3.4 SCF and driver wiring

- `H_core = T + V + V_ecp` in `src/hf_driver.cpp` (and the DFT driver's
  equivalent site) — mechanically trivial once §3.3 exists.
- Reduced electron counts flow into `run_rhf`/`run_uhf`/`run_rohf` via the
  `n_electrons` computation already centralized (mostly) through
  `total_nuclear_charge()`; audit the three call sites in `src/scf/scf.cpp`
  rather than assume one fix covers all.
- Checkpoint (`src/io/checkpoint.cpp`) doesn't need new fields for MOs/
  energies, but cross-basis checkpoint restart needs the ECP core-electron
  count to agree between source and target basis, or Löwdin projection will
  silently project onto the wrong occupied space.

### 3.5 Gradients (separate landing, not required for single points)

- ECP integrals have their own analytic derivative formulas (radial +
  angular pieces), needed in `src/gradient/gradient.cpp` for
  Gradient/GeomOpt/Frequency with an ECP present. Until built, those
  calculation types must explicitly reject ECP+gradient input — same
  boundary-marker pattern as `water_rmp2_spherical_gradient_rejected` — not
  silently return an incomplete gradient.

### 3.6 What needs nothing (downstream consumers)

SCF, MP2, CC, CASSCF, RASSCF, DFT, TDDFT, PCM all consume `H_core` and an
electron/orbital count; none of them care *why* `H_core` has a third term.
Provided §3.1–§3.4 land correctly (reduced core count flows everywhere it's
currently derived from `atomic_numbers`), these need no code changes at all.
Mulliken/Löwdin population analysis will just show fewer "core" electrons —
cosmetic, not a defect, but worth a one-line doc note so a user doesn't
mistake a foreshortened core population for a bug.

## 4. Suggested landing order

1. §3.1 data model + §3.2 parser, gated so a file with an ECP block parses
   but the driver rejects `calculation` if any atom carries one (clean
   boundary marker, mirrors existing rejection patterns).
2. §3.3 local-part integral only, validated against PySCF's `type1_by_shell`
   on a same-basis single atom (simplest possible ECP: local-only pseudo-
   potentials exist, e.g. some alkali ECPs, so this is independently
   testable before semi-local exists).
3. §3.3 semi-local part, validated against `type2_by_shell` the same way,
   then an end-to-end PySCF RHF/UHF energy gate on a real ECP basis
   (LANL2DZ or ccECP) for a heavier atom.
4. §3.4 SCF/driver wiring + reduced-electron-count audit, with a
   `metric_close` regression case the moment step 3 produces a number.
5. §3.5 gradients, as a separate follow-on; everything up to step 4 already
   delivers single-point RHF/UHF/DFT with ECPs, which is most of the value.

## 5. Out of scope for a first landing

- Spin-orbit ECPs (`so_by_shell` in the vendored reference) — needed for
  relativistic heavy-element work, not for a first correctness landing.
- ECP analytic gradients/geomopt/frequencies (§3.5, deferred).
- ECP under full-point-group symmetry direct-SCF (§3.3 note).
- ECP + PCM, ECP + CASSCF active-space selection interactions — should fall
  out for free once `H_core`/electron-count plumbing is right, but not
  validated until someone runs that combination.
