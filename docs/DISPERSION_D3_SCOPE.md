# How should Planck add Grimme's D3 dispersion correction?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Why it matters

Semilocal and hybrid functionals miss long-range dispersion. Nearly every
routine DFT geometry, conformer or interaction energy is run with D3(BJ) (or
D4). Planck has no dispersion correction anywhere in `src/`.

## What it is

D3(BJ) is a function of the nuclear coordinates only:

$$
E_\text{disp} = -\sum_{A<B} \sum_{n=6,8} s_n
\frac{C_n^{AB}}{r_{AB}^n + (a_1 R_0^{AB} + a_2)^n},
\qquad R_0^{AB} = \sqrt{C_8^{AB} / C_6^{AB}}
$$

- $C_6^{AB}$ is interpolated from a reference table (5 reference
  coordination numbers per element, about 32k pairwise values) by each atom's
  fractional coordination number, itself a smooth function of geometry.
- $C_8^{AB} = 3 C_6^{AB} \sqrt{Q_A Q_B}$, from tabulated $\langle r^4\rangle/\langle r^2\rangle$.
- $s_6, s_8, a_1, a_2$ depend on the functional (published per functional,
  e.g. B3LYP, PBE, PBE0, B2PLYP, HSE06).
- An optional three-body (ATM) term is controlled by $s_9$.

Because it depends only on geometry, it plugs in exactly where nuclear
repulsion does: one energy scalar and one gradient. Frequencies follow from
finite differences of gradients with no extra work.

## Integration points

- **Energy:** the DFT driver adds `calculator._nuclear_repulsion` to the
  electronic energy (`src/dft/driver.cpp`, the RKS and UKS scaffolds). D3
  adds a second geometry-only term next to it, reported on its own line.
- **Gradient:** next to the nuclear-repulsion and XC nuclear gradients in the
  DFT gradient assembly.
- **Input:** a `dispersion` keyword (`none`, `d3bj`, `d3zero`, `d3bj-atm`)
  in the DFT block, with parameters looked up from the functional name and
  overridable. The same term could serve HF (HF-D3 parameters exist), but
  DFT comes first.

## The decision: library or native

| option | for | against |
|---|---|---|
| **simple-dftd3** (Fortran, has a C API) via `ExternalProject`, like libxc | the reference implementation; parameters, CN and gradients done; maintained | needs a Fortran compiler in the build (gfortran is present on this machine; CI must add it); **LGPL-3.0**, linked into a PolyForm-Noncommercial program |
| **native C++** | no new toolchain; small (a few hundred lines) | the reference C6 table (~32k values) and parameters must still come from Grimme's data, so provenance and licence remain; we own correctness of CN and its gradient |

Both need the reference data, so the licensing question does not go away
with a native port. The steps below are written so either option fits
behind one interface.

### Licence findings (checked 2026-10-26; not legal advice)

- **simple-dftd3 and dftd4 are LGPL-3.0-or-later** (GitHub licence API;
  README "either version 3 of the License, or (at your option) any later
  version"). They ship `COPYING` (the GPL-3.0 text LGPL-3.0 builds on) and
  `COPYING.LESSER`, which is why some package indexes also list GPL-3.0.
  The reference C6 table and functional parameters live inside that
  LGPL-licensed source; no separate data licence is stated.
- **Planck is distributed as source only** (GitHub; no built binaries are
  shipped). Fetched at build time by `ExternalProject`, as libxc (MPL-2.0)
  is, simple-dftd3 never enters the repository, and a user who builds
  Planck for their own use conveys nothing. The LGPL's binary-distribution
  conditions (§4: licence texts, relinking, permitting reverse engineering
  for debugging) therefore do not arise for the project. They would apply to
  anyone who later redistributes a built binary, which is worth one line in
  the README next to the build instructions.
- **A native port does not escape the LGPL.** Code or the C6 table
  translated from simple-dftd3 would put LGPL-3.0 files in Planck's tree,
  where they cannot be relicensed under PolyForm. That is allowed (they stay
  LGPL) but mixes licences in the source, which the build-time fetch avoids.
  The licence of Grimme's original `dftd3` program and its data could not be
  confirmed, so it is not an alternative source.
- **Recommendation:** simple-dftd3 via `ExternalProject`, static or shared
  as suits the build (libxc is static). Make it optional at configure time
  (`-DPLANCK_WITH_DFTD3`, default ON once landed) because it needs a Fortran
  compiler, which the build does not otherwise require; CI must install one.

## Steps

1. **Decide library vs native** (the licence question above).
2. **Energy.** `dispersion d3bj` for the functionals whose parameters are
   tabulated; an unknown functional is rejected by name, never silently
   zero.
   - Gate: energies against the reference `dftd3` implementation (Python
     `simple-dftd3`, or PySCF's dispersion interface) to 1e-10 Eh on water,
     the water dimer, benzene and a C1 organic molecule, for B3LYP, PBE,
     PBE0, B2PLYP.
   - Gate: total DFT-D3 energy against PySCF with the same functional and
     D3 variant, at the existing DFT tolerance.
3. **Gradient.** Analytic, including the coordination-number terms.
   Gate: FD of the D3 energy to 1e-9 Eh/Bohr, and against the reference
   implementation's gradient.
4. **Workflows.** Geometry optimization and frequencies pick it up through
   the gradient. Gate: a water-dimer geomopt against PySCF-D3.
5. **D3(0) and ATM** variants, if wanted, behind the same keyword.

## Not in this scope

D4 (charge-dependent C6; needs atomic partial charges) and VV10-type
nonlocal correlation, which belongs with meta-GGA (`docs/META_GGA_SCOPE.md`).
