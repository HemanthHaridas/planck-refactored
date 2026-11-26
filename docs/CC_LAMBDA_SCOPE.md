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

## Route: differentiate the Lagrangian, do not project again

The T equations are already generated, and in production by the diagram
engine (`PLANCK_CC_ENGINE` defaults to `diagram`; it is 3.0 s against
Wick's 615 s at CCSDT, `docs/CCGEN_DIAGRAM_REPRESENTATION.md`). The
Lagrangian is built from them:

    L(T, Λ) = E(T) + Σ_μ λ_μ R_μ(T)

so the Λ residual is a derivative of terms that already exist:

    0 = ∂L/∂t_ν = ∂E/∂t_ν + Σ_μ λ_μ ∂R_μ/∂t_ν

and ∂R_μ/∂t_ν = ⟨μ| [H̄, τ_ν] |0⟩, the commutator form above. Each Λ term
comes from one E or R term by removing one t factor (its indices become
the free indices ν) and, for R, contracting the residual's free indices
with λ_μ. No new operator, projector or connectivity rule is needed: the
terms where Λ touches only τ_ν are those where the removed t carried the
residual's external lines, and they come out of the algebra.

Two other routes were considered:
- **Wick projection of ⟨0|(1 + Λ)[H̄, τ_ν]|0⟩**: needs a de-excitation
  operator, a ket excitation with free indices, and a linked-but-not-connected
  projection (the existing helpers drop disconnected products). And it
  costs Wick's time at the ranks the scope is for. Kept as a cross-check at
  Λ-CCSD only (step 1 gate).
- **Λ vertices in the diagram encoding**: a new vertex type in D1, its
  well-formedness and orbit weights. More work than differentiation for
  the same terms.

## What ccgen has, and what is missing (checked 2026-11-26)

| Piece | Status |
|---|---|
| E and R_μ as `AlgebraTerm` lists from the diagram engine (`generate_cc_equations(..., engine="diagram")`, manifolds `energy`, `singles`, …) | present; note the Python default is still `engine="wick"`, so pass it |
| Canonicalize and merge a term list (`canonicalize_term_to_fixed_point`, `merge_term_into_buckets`) | present; the diagram engine already runs them |
| Wick projection (`python/ccgen/wick.py`, `bch_levels`, `nfold_projector`) | present; used only for the Λ-CCSD cross-check |
| ∂/∂t_ν over an `AlgebraTerm` (`python/ccgen/project.py`: `coeff`, `factors`, `free_indices`, `summed_indices`) | **missing**: the core of step 1 |
| λ contracted onto R's free indices, tensor `l<n>` | **missing**: one factor added per R term; λ_n has t_n's index slots with the occupied and virtual roles swapped |
| Emitted Λ kernels and runtime | **missing**: kernels that read both t and λ, a second amplitude set, a solver with T frozen |
| Density expressions | **missing**: the same derivative, taken in a perturbation of H (step 3) |

## Steps

### 1. Spin-orbital Λ equations by differentiation (generator only)

- **Input.** `generate_cc_equations(method, engine="diagram")`, spin-orbital
  (GCC), at the same `canonical_fock` setting as production. The f_ov terms
  it drops multiply f_ov = 0, so their derivatives vanish too.
- **`differentiate(term, rank)`.** For each factor `t<rank>` in the term
  (a term with two t2 factors gives two derivative terms): remove it, rename
  its indices to the free ν indices, move them from `summed_indices` to
  `free_indices`. Terms without a `t<rank>` factor contribute nothing.
- **Amplitude symmetry.** t_n is antisymmetric within its occupied and its
  virtual slots, so ∂/∂ of a unique element is a sum over the (n!)² orderings
  with signs. Equivalently: differentiate as if every element were
  independent, then antisymmetrize the result over ν. Write down the
  convention once (L sums λ over all index tuples with (1/n!)², λ
  antisymmetric, as for t) and fix every factor from it. This is where a
  wrong 2 or a transpose comes from; the gate below catches it.
- **The λ contraction.** For an R_μ term, add a factor `l<rank(μ)>` over
  R's free indices (now summed) and multiply by R's (1/n!)² weight. The
  energy terms give the inhomogeneous ∂E/∂t_ν.
- **Then** canonicalize and merge as the diagram engine does. Differentiate
  the undressed terms; dressing and the factorizer run afterwards, on the Λ
  terms, as they do on the T terms.
- **Rank.** For CC up to rank n, ν and μ run over ranks 1..n.

**Gates.**
1. **Against PySCF.** `gccsd_lambda.update_lambda` at PySCF's converged t
   and λ: the generated spin-orbital Λ residual, evaluated in numpy at those
   amplitudes, is zero to ~1e-12, and nonzero at a perturbed λ. Water
   STO-3G and C1 H2O2 6-31G, the fixtures the (T) gates use.
2. **Against Wick at Λ-CCSD.** The Wick projection of
   ⟨0|(1 + Λ)[H̄, τ_ν]|0⟩ and the differentiated terms give the same
   residual at random antisymmetric t and λ (residual equality, not term
   equality; the two paths split terms differently, as the diagram and Wick
   T equations do).
3. **Rank-independent.** ∂L/∂t_ν by central differences on the T residual
   code (perturb one antisymmetrized t element, evaluate E + λ·R) against
   the generated Λ residual, at random t and λ. This needs only the T
   equations, so it runs at every rank, including where PySCF has nothing.
   Mutation-check it: drop one derivative term and see it fail.

Term counts for Λ-CCSD against a published spin-orbital list (Gauss–Stanton
or Crawford–Schaefer) are a further check; transcribe, do not rebuild from
memory.

### 2. Closed-shell and UHF lowering

Two ways, to be decided by the step-1 gates:
- **Differentiate after spin adaptation** (preferred if it holds): apply
  step 1's derivative to the spin-adapted RCC (or UCC) E and R_μ. Then λ is
  the multiplier of the spatial parametrization and never enters the spin
  adapter. Its λ is not PySCF's (different metric on the spatial residuals),
  so gate it by the parametrization-independent quantities: the
  rank-independent FD gate, and the densities in step 3.
- **Differentiate the spin-orbital terms, then adapt**: teach
  `spin_adapt_equations` and `ucc_adapt_equations` the λ tensors (t's
  symmetry with the occupied and virtual slots swapped). The retired
  dressed route shows adaptation composing wrongly with another transform,
  so gate it after adaptation.
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
- **Densities.** D and Γ are the same derivative taken in another variable:
  with H → H + εX, D_pq or Γ_pqrs is ∂L/∂X_pq or ∂L/∂X_pqrs at fixed T and
  Λ. So differentiate E and R_μ with respect to the Fock and ERI factors,
  as step 1 does with respect to t. Emit them as kernels, and assemble the
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

- **Symmetry weights in the derivative.** The differentiation route has no
  connectivity problem, but each term's derivative needs the right weight
  over equivalent amplitude orderings and repeated t factors. A wrong weight
  gives a residual that still converges to a wrong λ. Gates 1 and 3 of
  step 1 catch it; the energy does not.
- **Connectivity, Wick cross-check only.** The existing projection helpers
  drop disconnected products, and Λ terms are linked but not connected (Λ
  need not touch H̄). The Wick cross-check needs its own projection path;
  reusing the helpers silently drops terms.
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
