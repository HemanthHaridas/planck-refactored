# What do MP2 and coupled cluster need to accept an ROHF reference?

Scope, opened 2026-11-10. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 7).

## The state

`src/hf_driver.cpp` rejects every ROHF post-HF method except FCI, CASSCF and
RASSCF ("ROHF post-HF references are not implemented"). UHF-based MP2 and CC
exist. ROHF-based correlation is what most codes default to for open-shell
CCSD(T), because it has far less spin contamination.

## The one constraint that shapes everything

Every CC kernel in Planck assumes a canonical reference, f_ov = 0
(`cc_canonical_fock_only` memory). ROHF orbitals are not canonical for either
spin Fock. The standard route is **semicanonicalization**: diagonalize the
alpha Fock in the alpha occupied and alpha virtual blocks separately (and the
same for beta). Afterwards f_oo and f_vv are diagonal per spin, but **f_ov is
not zero** (the ROHF condition makes only the effective Fock's ov block
vanish). So:

- **ROHF-MP2 (RMP2 in PySCF's sense, "ROHF-MBPT2")**: UMP2 on semicanonical
  orbitals plus the singles term Σ |f_ia|² / (ε_i − ε_a). Small, closed form.
- **ROHF-CCSD**: UCCSD on semicanonical orbitals with f_ov ≠ 0. The hand-
  written and generated kernels drop every f_ov term. They must gain them, or
  a solver that tolerates them must be used.

## Steps

### 1. Semicanonical transform

One function next to the ROHF SCF: given ROHF C and the stored spin Focks
(already persisted for the ROHF gradient), return per-spin semicanonical
`C_α`, `C_β`, orbital energies and the `f_ov` blocks, packaged as a
`UHFReference`-shaped object plus f_ov. Unit gate: f_oo, f_vv diagonal per
spin; ROHF energy unchanged from the transformed densities.

### 2. ROHF-MP2 energy

UMP2 kernel on the semicanonical reference plus the singles term. Gate
against PySCF `mp.UMP2(mf)` on an ROHF `mf` (PySCF semicanonicalizes and
adds the singles) for OH and water⁺, 6-31G*, Cartesian and spherical.

### 3. ROHF-CCSD

The determinant-space engine (`solve_determinant_cc`) works from a
Hamiltonian, not from Fock blocks, so it may already handle a non-canonical
reference: check before writing kernels. If it does, ROHF-CCSD (small
systems) is a reference-building change plus a gate against PySCF
`cc.UCCSD(rohf_mf)`. A tensor-backend ROHF-CCSD needs the f_ov terms in the
residuals; for the generated route that is a ccgen change (the generator
currently assumes canonical), and it is the larger task. Scope it after the
determinant-space result.

### 4. ROHF-CCSD(T)

Open-shell (T) exists: the UHF (T) (`docs/CCSD_T_SCOPE.md`) is generated
per spin block. Semicanonical (T) needs the f_ov-dependent terms too:
generate with `canonical_fock=False`. That is the same generator change as
the tensor ROHF-CCSD in step 3.

## Not in this scope

ROHF-MP2 gradients (need a three-block ROHF CPHF solver), ROHF stability,
SOSCF and PCM (nice-to-have in Open Work).
