# What does Planck need to compute Rydberg states?

Research scope, opened 2026-11-11. Nothing here is built. It expires when the
work lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md`.

## Why it matters

The lowest UV excitations of most small molecules (water, ammonia, methanol,
ethylene, formaldehyde) are Rydberg or Rydberg-valence mixed: an electron
promoted into a diffuse 3s/3p/3d-like orbital several bohr across. Getting
them needs three things together:

1. a basis with diffuse functions (aug-, d-aug-, or Rydberg functions on a
   centre);
2. a method whose potential decays as −1/r far out. In Planck that means a
   range-separated hybrid in TDDFT (CAM-B3LYP, ωB97X). Global hybrids and
   GGAs put Rydberg states too low;
3. numerics that hold up with diffuse functions: a DFT grid that integrates
   them, an orthogonalizer that tolerates near-linear dependence, and a
   response solver that fits in memory once the basis grows.

TDDFT is Planck's only excited-state method for anything beyond a CAS. There
is no EOM-CC. So this scope is about TDDFT with diffuse bases.

## What already works (measured 2026-11-11)

**Basis files.** `basis-sets/` ships aug-cc-pVXZ (D–7Z), aug-cc-pCVXZ,
6-31(1)+G / ++G variants and def2-SVPD. It ships no d-aug/t-aug sets and no
Rydberg (Kaufmann) sets.

**Integrals.** Water / Planck's own aug-cc-pVDZ GBS, Cartesian, RHF:
Planck −76.0419833681, PySCF −76.0419833681 (basis loaded from the same file
with `parse_gaussian`). Every printed digit agrees.

**Functionals and TDDFT.** CAM-B3LYP, ωB97X and HSE06 TDDFT are gated against
PySCF on water/STO-3G (`docs/RS_TDDFT.md`). Water/aug-cc-pVDZ CAM-B3LYP
Casida runs: 4 roots in a 190-dimensional space, 43 s, 0.41 GB peak.

**The fixture is genuinely Rydberg.** In PySCF, each of the four lowest roots
is ≥ 0.91 one configuration, and its target virtual has ⟨r²⟩ = 33–52 bohr².
The occupied orbitals have ⟨r²⟩ ≤ 2.8 bohr². Water/aug-cc-pVDZ is a usable
first gate.

## Finding 1: the default grid under-integrates diffuse density

Water/aug-cc-pVDZ CAM-B3LYP ground-state energy. PySCF is converged here:
levels 6, 7 and 8 agree to 7e-9 Eh, and the roots to 5e-6 eV.

| Planck grid | E (Eh) | Δ vs PySCF level 8 |
|---|---|---|
| `ultrafine` (legacy pruning) | −76.4177786199 | 7.2e-7 |
| `ultrafine` + `grid_radial_points 200` | −76.4177785771 | 6.8e-7 |
| `ultrafine` + `grid_angular_points 590` (unpruned) | −76.4177778858 | 1.0e-8 |
| `grid_policy g4-3-v4`, 135 radial | −76.4177778918 | 4.4e-9 |
| radial 200, angular 974 | −76.4177778921 | 4.1e-9 |

The excitation energies on `ultrafine` miss PySCF by 0.7–2.2 meV:

| root | Planck `ultrafine` (eV) | PySCF level 7 (eV) | Δ (meV) |
|---|---|---|---|
| 1 | 7.0987028 | 7.1000918 | −1.4 |
| 2 | 8.6861814 | 8.6840342 | +2.1 |
| 3 | 9.2749913 | 9.2767430 | −1.8 |
| 4 | 10.4452494 | 10.4445472 | +0.7 |

On STO-3G the same functional is within 5e-4 eV. On the 200 × 974 grid,
Planck's roots are 7.1000920, 8.6840333, 9.2767455 and 10.4445487 eV, which
are within 1.5 µeV of PySCF level 8. The whole meV gap is the grid; the
response code is right.

**The cause is angular pruning, not radial count.** More radial shells do not
help. Removing pruning, or the `g4-3-v4` schedule, fixes it.
`pruning_region` (`src/dft/base/grid.h`) assigns angular size by
radial-index fraction, so the outer shells get the fewest points. That is
where Rydberg density lives.

This is the same decision as `docs/DFT_DEFAULT_GRID_SCOPE.md`: `legacy` is
not qualified, and `g4-3-v4` is. Rydberg states are a second, independent
reason to migrate the default. No separate fix is needed. Until the default
moves, Rydberg inputs should set `grid_policy g4-3-v4` with
`grid_radial_points 135`.

## Finding 2: linear dependence, which is not a problem yet

`SCF::build_orthogonalizer` (`src/scf/scf.cpp`) is symmetric (S^−½), and
refuses a run when λ_min(S) < 1e-8. There is no canonical orthogonalization.
Overlap λ_min, from PySCF (d-aug built by even-tempered extension of each l):

| system / basis | Cartesian nb, λ_min | spherical nb, λ_min |
|---|---|---|
| water aug-cc-pVDZ | 43, 2.4e-3 | 41, 2.7e-3 |
| water aug-cc-pVTZ | 105, 2.0e-4 | 92, 3.9e-4 |
| water d-aug-cc-pVDZ | 61, 1.2e-4 | 58, 1.2e-4 |
| water d-aug-cc-pVTZ | 145, 1.3e-5 | 126, 2.0e-5 |
| water aug-cc-pVDZ + 5s5p5d at the centre | 93, 2.1e-5 | 86, 4.7e-4 |
| water dimer aug-cc-pVTZ | 210, 6.5e-5 | 184, 2.3e-4 |
| benzene 6-311++G** | 180, 1.1e-6 | 174, 1.1e-6 |
| benzene aug-cc-pVDZ | 204, 2.1e-6 | 192, 2.3e-6 |
| benzene aug-cc-pVTZ | 480, 7.7e-8 | 414, 3.2e-7 |

Nothing a Rydberg study on a small molecule would use crosses the wall.
Benzene aug-cc-pVTZ Cartesian is within one decade of it. Larger aromatics
with aug-TZ, or t-aug sets, will cross it.

Canonical orthogonalization drops the small modes, so n_MO < n_AO. Every
consumer that assumes a square C then breaks: 55 `working_nbasis()` sites and
182 MO-count sites, across SCF, post-HF, TDDFT, gradients and checkpoint.
That is a cross-cutting refactor. **Do not start it until a real target
crosses 1e-8**, and then do it as one mechanism (an `n_mo` on the SCF state
that every consumer reads), not by patching callers.

## Finding 3: TDDFT memory is the actual wall

`run_linear_response` (`src/dft/driver.cpp`) builds dense A and B, and the
full AO ERI tensor. For a range-separated functional it builds a second
nb⁴ tensor for K_SR. Range-separated functionals are the ones Rydberg states
need, so they pay double.

| case | nb (Cartesian) | one nb⁴ tensor | RS total |
|---|---|---|---|
| water aug-cc-pVDZ | 43 | 27 MB | 55 MB |
| water aug-cc-pVTZ | 105 | 0.97 GB | 1.9 GB |
| water d-aug-cc-pVTZ | 145 | 3.5 GB | 7.1 GB |
| benzene aug-cc-pVDZ | 204 | 13.9 GB | 27.7 GB |

Diffuse bases add mostly virtuals, so nb grows fastest exactly where Rydberg
work goes. On an 8 GiB machine, Rydberg TDDFT stops at triatomics with
aug-cc-pVTZ. Benzene aug-cc-pVDZ, the textbook Rydberg-valence case, cannot
run. This is a bigger limit than the grid or the orthogonalizer.

**Most of the machinery already exists.** `build_ks_orbital_hessian_op` and
`build_uks_orbital_hessian_op` (`src/dft/ks_orbital_hessian.h`) apply (A + B)
matrix-free: orbital-energy differences, direct J, the prepared analytic fxc,
and −(c_fr K + c_sr K_SR). `davidson_lowest_eigenpairs` (`src/dft/driver.cpp`)
already drives them for KS stability. A matrix-free TDDFT needs:

- an (A − B) action. The trial density is antisymmetric
  (C_v x C_oᵀ − transpose), so J and fxc vanish and only the exchange terms
  remain. That is the existing K builds on a non-symmetric density;
- TDA, which is A = ½[(A + B) + (A − B)], with Davidson on A;
- Casida, which is Hermitian Davidson on (A − B)^½ (A + B) (A − B)^½, or
  the symplectic form PySCF uses;
- RKS triplet, which reuses the RKS stability triplet route;
- transition dipoles and oscillator strengths from the converged vectors,
  which the dense path already computes from X and Y.

The direct J/K builds need no nb⁴ storage, so memory drops to O(nb²·n_roots).
The dense solver stays as the test oracle, as the DH Z-vector did.

## Finding 4: input gaps

- **No d-aug / t-aug files.** These are data, not code: add the BSE files for
  d-aug-cc-pVXZ (X = D, T) next to the aug sets.
- **No basis-only centre.** Ghost atoms (`Gh(O)`, `@O`, `O:`) carry their
  element's basis. Rydberg functions at the charge centroid would need a
  centre with its own basis and no charge. A workaround exists today: a custom
  GBS through `basis_path` that defines a ghost element with only the Rydberg
  shells. It is awkward, and nobody has asked for it. Skip it until someone
  does.
- **No per-element basis.** The single `basis` keyword applies to every atom.
  Same workaround (one custom GBS), same verdict.

## Finding 5: no Rydberg diagnostic

The TDDFT printout gives the dominant configurations but not their spatial
extent. ⟨r²⟩ of each MO about the charge centre, together with each root's
weight-averaged ⟨r²⟩ of its target virtuals, classifies roots as valence or
Rydberg in one column. The r² integrals already exist for the quadrupole
report (`src/populations/multipole.cpp`). It is cheap and optional.

## Not in scope

- EOM-CCSD or any CC excited states. None exist.
- Rydberg states from SA-CASSCF beyond what a CAS already reaches.
- Asymptotic corrections (LB94, AC shifts). The range-separated hybrids
  cover this.

## Steps, in order

1. **Gate.** Water/aug-cc-pVDZ CAM-B3LYP Casida, singlets, with
   `grid_policy g4-3-v4` and 135 radial shells, against PySCF (Planck's GBS
   via `parse_gaussian`, `cart=True`). Add a TDA case and a ωB97X case. On a
   converged grid the roots agree to 1.5 µeV, so an atol near 1e-5 eV is
   realistic. Measure it on `g4-3-v4` when the gate is built.
   This is the first regression case with a diffuse basis; there is none
   today.
2. **Grid.** No new work; covered by the default-grid decision
   (`docs/DFT_DEFAULT_GRID_SCOPE.md`). Add this fixture to that decision's
   evidence.
3. **Matrix-free TDDFT** (Finding 3). Do TDA first, then Casida, then UKS.
   Gate it equal to the dense oracle on every existing TDDFT case. Then add
   benzene aug-cc-pVDZ CAM-B3LYP against PySCF, which is the case the dense
   path cannot run.
4. **d-aug-cc-pVDZ/TZ basis files**, with a water d-aug gate.
5. **⟨r²⟩ diagnostic** in the TDDFT root printout (optional).
6. **Canonical orthogonalization**, only when a target crosses λ_min < 1e-8
   (Finding 2).

The fixtures and reference script used here are not committed. They are in
the session scratchpad: `ref.py` (PySCF, with an ⟨r²⟩ classification) and
`smin.py` (the overlap survey). Step 1 should commit a cleaned version under
`tests/pyscf/`.
