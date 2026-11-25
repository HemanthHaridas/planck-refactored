# How should ccgen generate the Λ (left-hand) CC equations, and what do they unlock?

Scope, opened 2026-11-25. Nothing here is built. This doc expires when the
first step lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md`. It is step 4 of `docs/CCSDT_Q_SCOPE.md`,
split out because it is the largest item there and the base of several
others.

## Why it matters

Coupled cluster is not variational, so the energy alone gives no densities
and no cheap derivatives. The Λ equations make the CC energy stationary by
adding a left-hand state ⟨0|(1 + Λ):

    L(T, Λ) = ⟨0| (1 + Λ) H̄ |0⟩,   H̄ = e^{−T} H e^{T}

Stationarity in Λ gives back the T equations; stationarity in T gives the
Λ equations, one linear set per rank:

    0 = ⟨0| (1 + Λ) [H̄, τ_ν] |0⟩      for every excitation ν

With T and Λ converged, every first-order property is one expectation value:

    D_pq   = ⟨0| (1 + Λ) e^{−T} {p† q} e^{T} |0⟩       (1-RDM)
    Γ_pqrs = ⟨0| (1 + Λ) e^{−T} {p† q† s r} e^{T} |0⟩  (2-RDM)

What that unlocks, in order of distance:

| Consumer | Needs beyond Λ | Elsewhere |
|---|---|---|
| Unrelaxed CC 1- and 2-RDMs, dipoles, natural orbitals | the density expressions | here, step 3 |
| ΛCCSD(T), CCSDT(Q)_Λ, CR-CC(2,3) | a Λ-weighted left vector in the perturbative energy | `docs/CCSDT_Q_SCOPE.md` |
| Analytic CC gradients | orbital response: the CC orbital Lagrangian from the RDMs and a Z-vector | own scope; the RMP2/UMP2 gradients already have the Z-vector machinery (`rhf_response`, `uhf_response`) |
| EOM-CC left states and transition properties | EOM itself | own scope; same left-hand operator algebra |

One generator gives all of these at any rank, which is the reason to do it
in ccgen rather than by hand: hand-written Λ-CCSD exists in every code,
Λ-CCSDT and Λ-CCSDTQ almost nowhere.

## What ccgen has, and what is missing (checked 2026-11-25)

| Piece | Status |
|---|---|
| T_n as a weighted operator string (`build_tn`, `python/ccgen/cluster.py`) | present |
| H̄ by BCH commutators (`bch_levels`, `python/ccgen/algebra.py`) | present |
| Bra projectors ⟨Φ_μ\| with free indices (`nfold_projector`, `python/ccgen/project.py`) | present |
| Full contraction of an arbitrary operator string (`python/ccgen/wick.py`) | present; it already contracts projector + H̄ strings, and nothing in it assumes the excitation direction |
| Λ_n, the de-excitation operator | **missing**: `build_tn` with the strings swapped (creators on occupied, annihilators on virtual), tensor `l<n>` |
| A ket excitation τ_ν\|0⟩ with free indices | **missing**: the mirror of `nfold_projector` |
| The Λ connectivity rule | **missing**: H̄ must connect to τ_ν (the commutator); Λ may connect to H̄, τ_ν or both. `iter_projected_terms_from_terms(require_connected=...)` is all-or-nothing, and `bucket_terms_by_manifold` drops disconnected products because it assumes BCH commutators (`docs/CCSD_T_SCOPE.md`), so Λ needs its own projection path |
| Spin adaptation of λ (`spin_adapt_equations`, `ucc_adapt_equations`) | **missing**: λ has t's permutational symmetry with occupied and virtual roles swapped; the adapter knows only `t<n>` names |
| Emitted Λ kernels and runtime | **missing**: kernels that read both t and λ, a second amplitude set, a solver with T frozen |
| Density expressions | **missing**: same machinery with {p† q} or {p† q† s r} in place of H |

## Steps

### 1. Spin-orbital Λ equations (generator only)

- `build_ln(n)`: Λ_n = (1/n!)² Σ λ_{a…}^{i…} i† … a … (de-excitation), tensor
  `l<n>`, as `build_tn` with the operator strings swapped.
- `nfold_excitation(n)`: τ_ν|0⟩ with free indices, the mirror of
  `nfold_projector`.
- A projection over ⟨0| (1 + Λ) H̄ τ_ν |0⟩ − ⟨0| (1 + Λ) τ_ν H̄ |0⟩ that
  keeps exactly the terms where H̄ connects to τ_ν. The second product is
  what the commutator subtracts; with the vacuum on the right it reduces to
  E_CC times ⟨0|Λ τ_ν|0⟩, so keeping only H̄–τ_ν-connected terms is the
  same thing. **Check that reduction by evaluating both forms numerically**
  before relying on it.
- Truncation: for CC with ranks up to n, Λ has ranks up to n, and H̄ up to
  the BCH order that can still reach a full contraction.

**Gate.** PySCF `gccsd_lambda.update_lambda` at PySCF's converged t and λ:
the generated spin-orbital Λ residual, evaluated in numpy at those
amplitudes, is zero to ~1e-12, and nonzero at a perturbed λ. Do this on
water STO-3G and C1 H2O2 6-31G, the fixtures the (T) gates use. Term counts
for Λ-CCSD against a published spin-orbital list (Gauss–Stanton or
Crawford–Schaefer); transcribe, do not rebuild from memory.

### 2. Closed-shell and UHF lowering

- Teach `spin_adapt_equations` and `ucc_adapt_equations` the λ tensors:
  the same symmetry as t, with the occupied and virtual slots swapped.
- **Gates.** Spatial Λ residual against PySCF `ccsd_lambda.update_lambda`
  (RCCSD) and `uccsd_lambda` (UCCSD), same form as step 1. PySCF's λ
  convention (`l1[i,a]`, `l2[i,j,a,b]`, its normalization) must be
  matched explicitly; write the mapping down where it is first used.

### 3. Λ solver and densities in Planck

- **Runtime.** A λ amplitude set of the same type as t
  (`ArbitraryOrderRCCAmplitudes`; the UCC sector form for UHF), emitted Λ
  residual kernels that read t and λ, and the existing Jacobi/DIIS harness
  (`run_generated_arbitrary_order_iterations`) run with T frozen. The Λ
  equations are linear in λ with the same orbital-energy denominators as T,
  so the Jacobi step and DIIS carry over. Start λ from t (the usual guess).
- **Emitter.** Λ residuals are binary contractions after dressing like the
  T residuals, so the GEMM lowering and the factorizer apply unchanged.
  Measure the Λ iteration cost against the T iteration; it should be about
  the same.
- **Densities.** Generate D and Γ, emit them as kernels, and assemble the
  unrelaxed 1-RDM and 2-RDM. Output: unrelaxed dipole and natural
  occupations.
- **Gates.**
  - 1-RDM and 2-RDM against PySCF `ccsd_rdm` / `uccsd_rdm` (RCCSD/UCCSD),
    elementwise.
  - **Rank-independent gate, needed for CCSDT and up**, where no external
    code is at hand: perturb the MO two-electron integrals by εX with the
    Fock diagonal held fixed (the kernels read the orbital energies
    separately), solve CC, and compare dE/dε by central differences
    against Σ Γ_pqrs X_pqrs. A canonical-Fock perturbation of the
    one-electron part checks D the same way on its oo and vv diagonals.
    That gate tests the whole Λ chain without any reference code, at every
    rank the build generates.
  - Energy check: E_CC = Σ h D + ¼ Σ g Γ (convention to be fixed) to
    roundoff.

### 4. Consumers (each its own decision)

- **ΛCCSD(T)**: (T) with a Λ-dependent left vector. Gate against PySCF
  `ccsd_t_lambda_slow` / its (T) gradient machinery where it applies.
- **CCSDT(Q)_Λ, CR-CC(2,3)**: `docs/CCSDT_Q_SCOPE.md`.
- **Analytic CC gradients**: their own scope. The pieces Planck has: the
  RMP2/UMP2 gradient's relaxed density, Z-vector and derivative-integral
  contractions. The new piece: the CC orbital Lagrangian from D and Γ.
  PySCF `grad.ccsd` and `grad.ccsd_t` are the references.

## Risks

- **Connectivity.** The existing projection helpers assume BCH-shaped
  terms and drop disconnected products. Λ terms are "disconnected but
  linked" (Λ need not touch H̄), so reusing those helpers silently drops
  terms. Step 1's numeric gate is what catches it.
- **Conventions.** λ normalization and index order differ between codes
  (PySCF stores `l2[i,j,a,b]`, a de-excitation amplitude written with
  occupied first). A wrong factor or transpose can still give a converged,
  wrong λ; the residual-at-PySCF-amplitudes gate catches it, a converged
  energy does not (the CC energy does not depend on λ).
- **Spin adaptation.** The retired dressed spin-adapted route
  (`docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`) shows that two correct
  transforms can compose wrongly. Gate the Λ residuals after spin
  adaptation and after dressing, separately.
- **Canonical Fock only.** Every Planck CC reference has f_ov = 0 and the
  generator drops f_ov terms (`_drops_under_canonical_fock`). Λ for ROHF
  references inherits the `canonical_fock=False` work of
  `docs/ROHF_CORRELATION_SCOPE.md`.
- **Memory.** λ doubles the amplitude storage: at CCSDT, t3 and λ3 are
  both o³v³.

## Recommended order

1 → 2 → 3, restricted first, then UHF through the same generator. Step 4
items are separate decisions: unrelaxed properties come free with step 3;
ΛCCSD(T) is small once step 3 exists; gradients are a larger scope that
should start only after the densities pass the rank-independent gate.

## Not in this scope

EOM-CC and linear-response CC (their own scope; they share the left-hand
operator algebra), orbital-relaxed properties and gradients (step 4's own
scope), ROHF references, frozen core, and RI.
