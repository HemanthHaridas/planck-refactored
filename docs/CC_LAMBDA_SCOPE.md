# How should ccgen generate the Λ (left-hand) CC equations, and what do they unlock?

Status lives in `vault/Status/Open Work.md`. Split out of
`docs/CCSDT_Q_SCOPE.md` (its step 4).

## Answer

Differentiate the CC Lagrangian built from the T equations the diagram
engine already produces. No new projection machinery is needed, and one
generator gives Λ and the unrelaxed densities at any rank, restricted (RCC),
unrestricted (UCC) and spin-orbital (GCC). The Python side is
`python/ccgen/lagrangian.py`. In Planck, the Λ residual kernels go through
the same emit path as the T kernels, and a Jacobi/DIIS solver runs them with
T frozen.

| Piece | State | Gate |
|---|---|---|
| Spin-orbital Λ (1) | done | residual at PySCF GCCSD's t, λ ≤ 3e-11 |
| UCC Λ (2) | done | residual at PySCF UCCSD's t, λ ≤ 1.0e-10 (UCC λ is PySCF's, unscaled) |
| RCC Λ (2) | done | unrelaxed energy derivative against an FD of PySCF RCCSD's E_corr ≤ 5.6e-9 |
| Derivatives, every block | done | complex step exact (≤ 5.4e-15) through CCSDT, GCC/RCC/UCC |
| Unrelaxed 1- and 2-RDMs (3a) | done, Python | PySCF `make_rdm1`/`make_rdm2` ≤ 2.5e-10 (RCC), ≤ 1.1e-16 (UCC); E from the RDMs ≤ 8e-12 |
| Λ kernels in Planck (3b) | done, RCC and UCC, any rank | emitted C++ text against the Python Λ residual ≤ 2e-15 relative, every block: RCC ranks 2–4, UCC ranks 2–3 |
| Λ solver, T frozen (3c) | done, RCC and UCC | `water_rccsd_lambda_631g`, `oh_uccsd_lambda_631g`: every λ block's rms against PySCF, atol 1e-9 |
| λ in the `.ccamp` file (3c) | open | |
| Density kernels and assembly in C++ (3d) | open | |
| Dipole, natural occupations and orbitals (3e) | open | |
| ΛCCSD(T), CCSDT(Q)_Λ, analytic CC gradients (4) | separate decisions | |

Python gates: `python/ccgen/tests/test_lambda_{equations,spatial,ucc,density,emit}.py`.
PySCF-based tests skip without PySCF. `test_lambda_emit.py` takes ~150 s
because of the rank-4 emit.

Not gated numerically:
- the rank ≥ 4 sector blocks (`t4_aaabaaab`) in the Python equations (the
  emit gate covers them at rank 4);
- ranks 5–6, which take the same path;
- a rank-4 Λ solve: the rank-4 Λ kernels are built (`planck_cc_generated`)
  but no regression case runs them.

## Why it matters

Coupled cluster is not variational, so the energy alone gives no densities
and no cheap derivatives. The Λ equations make the CC energy stationary by
adding a left-hand state ⟨0|(1 + Λ):

    L(T, Λ) = ⟨0| (1 + Λ) H̄ |0⟩,   H̄ = e^{−T} H e^{T}

Stationarity in Λ gives back the T equations; stationarity in T gives the
Λ equations, one linear set per rank. With T and Λ converged, every
first-order property is an expectation value of (1 + Λ) and e^{−T} … e^{T}.

| Consumer | Needs beyond Λ | Elsewhere |
|---|---|---|
| Unrelaxed CC 1- and 2-RDMs, dipoles, natural orbitals | the density kernels in Planck | 3d–3e |
| ΛCCSD(T), CCSDT(Q)_Λ, CR-CC(2,3) | a Λ-weighted left vector in the perturbative energy | `docs/CCSDT_Q_SCOPE.md` |
| Analytic CC gradients | the CC orbital Lagrangian from the RDMs and a Z-vector | own scope; `rhf_response`/`uhf_response` have the Z-vector machinery |
| EOM-CC left states and transition properties | EOM itself | own scope |

Hand-written Λ-CCSD exists in every code; Λ-CCSDT and Λ-CCSDTQ almost
nowhere.

## The route: differentiate the Lagrangian

    L(T, Λ) = E(T) + Σ_μ λ_μ R_μ(T)

The Λ residual is ∂L/∂t_ν = ∂E/∂t_ν + Σ_μ λ_μ ∂R_μ/∂t_ν, with
∂R_μ/∂t_ν = ⟨μ|[H̄, τ_ν]|0⟩. Each Λ term is an L term with one t factor
removed and its indices freed. The terms where Λ touches only τ_ν (linked
but not connected) come out of the algebra, so no connectivity rule is
needed. The densities are the same derivative in another variable: ∂L/∂f
and ∂L/∂v.

The T equations come from the diagram engine, the default of
`generate_cc_equations`, `--engine` and `PLANCK_CC_ENGINE` (3.0 s against
Wick's 615 s at CCSDT, `docs/CCGEN_DIAGRAM_REPRESENTATION.md`).

Routes not taken:
- Wick projection of ⟨0|(1 + Λ)[H̄, τ_ν]|0⟩. It needs a de-excitation
  operator, a ket excitation with free indices and a linked-but-not-connected
  projection, at Wick's cost.
- Λ vertices in the diagram encoding: more work than differentiating, for
  the same terms.

### Conventions

- λ_n is `l<n>` with indices (occ..., vir...), e.g. `l2(i,j,a,b)`, which
  is PySCF's `l1[i,a]`, `l2[i,j,a,b]`. Each residual key gets its own λ:
  `l2`, `l4_aaabaaab`, `l2_abab`.
- Weight of λ·R:
  - GCC: 1/(n!)², PySCF's normalization (1/n! fails the gate).
  - UCC: 1/(n_α! n_β!)² per block. This is the GCC Lagrangian restricted to
    the stored blocks, so UCC λ is PySCF's λ.
  - RCC: 1/(n!)². Any weight is valid; it only rescales λ.
- `differentiate` returns G = ∂L/∂T with every element independent, in t's
  (vir..., occ...) layout. The Λ residual is G symmetrized over t's own
  symmetry, R = Σ_g s_g g(G): antisymmetrized for GCC and the same-spin UCC
  blocks, the pair symmetry for RCC.
- **Canonical Fock is enough for Λ; the densities need the f_ov terms.**
  - Production generates with `canonical_fock=True`, which drops every term
    with an f_ov factor. ∂/∂t removes a t factor, not f_ov, so the
    derivative of a dropped term is still zero. The Λ kernels therefore come
    from the same equations as the T kernels.
  - The densities differentiate with respect to f itself. ∂L/∂f_ov removes
    the f_ov factor and leaves a nonzero term: 2 f_ia t_ia in the spatial
    energy gives 2 t_ia, the leading piece of D_ov. Density terms are
    generated with `canonical_fock=False` and evaluated with f_ov = 0.
  - The PySCF gates use `canonical_fock=False` for a different reason: the
    reference. PySCF's SCF stops with |f_ov| = 1.1e-8 on H2O2 6-31G, and
    with the f_ov terms dropped the rank-1 residual sits at exactly that
    value. Planck's reference is exactly canonical.

## The pieces

### Spin-orbital Λ (step 1)

`lagrangian_terms(eqs, "gcc")` contracts λ onto each residual term.
`differentiate(lag, name)` then removes each `name` factor, renames dummies
so the freed indices take the pool names, canonicalizes and merges.

Gates:
- **Complex step** (ε = 1e-20i): Σ G·D equals dL/dε to ≤ 3.7e-15 at every
  rank of CCSD and CCSDT. Dropping one G term fails it.
- **PySCF:** at GCCSD's t and λ, the antisymmetrized G is ≤ 3e-11 (water
  STO-3G, C1 H2O2 6-31G). 1 % on λ2 makes it nonzero.

Term counts with canonical Fock, after merging:

| | λ1 | λ2 | λ3 | λ4 |
|---|---|---|---|---|
| Λ-CCSD | 54 | 32 | | |
| Λ-CCSDT | 193 | 91 | 32 | |
| Λ-CCSDTQ | 752 | 262 | 91 | 32 |

Λ-CCSDTQ generates in 1.3 s.

### RCC and UCC (step 2)

`lagrangian_terms(eqs, "rcc" | "ucc")` differentiates the adapted
equations directly, so λ never enters the spin adapters.

- **UCC:** at PySCF UCCSD's t and λ, every block's residual is
  ≤ 1.0e-10 (CH3 STO-3G, water⁺ 6-31G). The GCC weight 1/(n!)² on the
  αβαβ block misses by 8e-2.
- **RCC:** the spatial λ is not PySCF's, so the gate is the unrelaxed
  energy derivative. Λ is solved by GMRES at PySCF RCCSD's t, and dL/dε for
  a random Fock perturbation X (which has occupied–virtual elements) is
  compared with a four-point FD of PySCF's E_corr. Water STO-3G 4.8e-10,
  C1 H2O2 6-31G 5.6e-9, which is the FD's noise. Dropping one λ2 term misses
  by 9e-2.
- **Complex step** is exact for every RCC and UCC block through CCSDT.
  RCC Λ-CCSDT has 795 / 1141 / 345 terms and UCC up to 1133 per block,
  each generated in under 3 s.

### Unrelaxed densities (3a)

`density_terms(lag, spin)` returns ∂L/∂(f or v) per space block
(`differentiate_blocks`), from the f_ov-keeping equations. Total RDMs follow
by the chain rule through E = E_ref(h, g) + L(f(h, g), v(g)), in PySCF's
convention E = Σ h D + ½ Σ (pq|rs) Γ (UCC: ½ αα, 1 αβ, ½ ββ).

| | 1-RDM | 2-RDM | E from RDMs − E_CC |
|---|---|---|---|
| RCC water STO-3G (our spatial λ) | 7.8e-11 | 1.6e-10 | −2.6e-13 |
| RCC C1 H2O2 6-31G | 1.2e-10 | 2.5e-10 | 1.7e-13 |
| UCC CH3 STO-3G (PySCF's λ) | 1.7e-18 | 1.1e-16 | −7.2e-12 |
| UCC water⁺ 6-31G | 1.7e-18 | 1.1e-16 | −7.7e-12 |

The RCC rows compare our λ's densities with PySCF's from its own λ, so
they also show that the densities do not depend on how λ is parametrized.
Dropping one density term misses by 1e-4 to 8e-2.

### Λ kernels (3b)

`print_cpp_planck(..., lambda_=True)` (`--lambda` in
`generate_planck_cc_kernels.py`) differentiates the adapted T equations
(`lambda_residual_terms`, free indices in λ's occ-first layout). The terms
then take the T kernels' path: factorizer, merges and GEMM lowering,
unchanged. It covers RCC (`spin_adapt`) and UCC at any rank, undressed or
`derived`; the retired `recognized` route is refused. Differences from the
T emit:

- Every emitted function takes `const ArbitraryOrderRCCAmplitudes &lambdas`
  after `amplitudes`. `l<rank>[_<tag>]` binds from it the way
  `t<rank>[_<tag>]` binds from the amplitudes.
- Symbols carry `lambda_` (`compute_[ucc_]lambda_ccsd_doubles_residual`,
  `build_W_..._lambda_ccsd`), so the Λ files link beside the T ones.
- Each block is R = Σ_g s_g g(G). `block_symmetry_group`
  (`optimization/permutation.py`) derives the group from the block's spins
  at any rank: the signed permutations within each (space, spin) class,
  plus, for a balanced RCC sector, the closed-shell spin flip (α pairs ↔
  β pairs, sign +). That gives the t2 pair symmetry at rank 2,
  `AAB_TRIPLES_GROUP` at rank 3 (checked equal) and each UCC same-spin
  antisymmetrizer. G is folded by the group (`fold_external_symmetry`) and
  scaled by |G|, and the kernel's symmetrize pass forms R. The T kernels'
  j ≥ i half range is off.
- The entry point is `make_generated_[ucc_]lambda_<method>_kernels()`, a
  `GeneratedLambdaKernels`. It returns every block from one set of
  intermediates: the per-rank residuals by rank, then the tagged blocks in
  `sector_tags` order. The registry entry is
  `make_generated_lambda_kernels(rank, ucc)`.

CMake builds Λ kernels for every rank it builds T kernels for (rank 2
always, rank 3 with `PLANCK_CC_ARBITRARY_LOWER_RANKS`, rank ≥ 4 by
`PLANCK_CC_MAXORDER`, UCC with `PLANCK_CC_UCC`), always dressed. The T
kernels are unchanged by it.

Sizes:

| | Builders | File | Emit time |
|---|---|---|---|
| RCC Λ-CCSD | 107 | | |
| RCC Λ-CCSDT | 338 | | |
| RCC Λ-CCSDTQ | 2,558 | 18.7 MB | 138 s (factorizer 120 s) |
| UCC Λ-CCSD | 292 | 0.57 MB | |
| UCC Λ-CCSDT | 1119 | 3.3 MB | |
| UCC Λ-CCSDTQ | | 42 MB | |

Λ-CCSDTQ G has 7,835 / 15,717 / 6,489 / 522 + 661 terms, against 13.7k
T terms in total: the high-rank λ contracts down into the low-rank
residuals. The fold finds few images (doubles 15,717 → 15,545) because G is
not written pair-symmetrically. Canonical-form fold keys gave no gain on Λ
and stalled the emit (`docs/CCGEN_UCC_ORBIT_FOLD.md`).

**Gate** (`test_lambda_emit.py`): the emitted C++ text is evaluated with
numpy and every block compared with the Python Λ residual, RCC ranks 2–4
(both rank-4 Sz sectors) and UCC ranks 2–3, ≤ 2e-15 relative. Fixtures
carry production symmetry: each t and λ block is projected onto its group,
and UCC integrals are stored plain per spin block, as Planck stores them.
It fails by 1e-3 to 1.4 on any of these mutations:
- a permuted `l2` read;
- a flipped `l1` term;
- a dropped symmetrization.

Constraints on this path:
- **Operator names must not collide.** Derived operators are named
  `W_<factors>v_<block>_<tag>` and deduplicated by name, with the tag a
  hash of the contraction shape. A 16-bit tag collided at rank 4 (one
  T-CCSDTQ and four Λ-CCSDTQ names each covered two contractions), so one
  builder served two operators. λ1–λ3 were then off by up to 1.4e-4 while
  the undressed emit was exact. The tag is 8 bytes, and
  `_manifold_operators_impl` raises if a name covers two shapes.
- **UCC builder symbols carry `ucc_`** (`build_W_..._ucc_lambda_ccsdtq`),
  as the entry points do (`compute_ucc_...`). An RCC sector tag
  (`aaabaaab`) equals a UCC spin tag, and a builder with no `v` factor has
  nothing else spin-specific in its name. Without the marker, the RCC and
  UCC Λ-CCSDTQ files both defined
  `build_W_l4_aaabaaabt4_aaabaaabv_oovv_<tag>_lambda_ccsdtq` in the registry
  TU. `_builder_symbol` adds it at every definition and call site, and
  `test_ucc_emit_flag` pins it.
- **Factorial sweeps are restricted above six summed indices.** λ4·t4
  sums up to eight, and the full sweeps stalled the Λ-CCSDTQ emit.
  `_contraction_shape` and `_canonicalize` now sweep only within groups of
  indices with equal signatures. The merge classes are unchanged
  (`test_operator_identity` checks them against the pairwise predicate).
- **The text evaluator reads every `cc_gemm_contract` of a term.** A UCC
  same-spin difference is lowered to two calls; reading only the first
  drops the exchange half.
- **Generated TUs compile in shards.** The registry used to `#include`
  every generated file: ~76 MB of source in one `-O1` compile at rank 4
  with UCC, which swapped past an hour on 8 GiB without finishing. Each file
  is now `<stem>.h` (the residual-ops struct and every declaration) plus
  `<stem>_s<k>.cpp` (`ccgen/emit/shard.py`), compiled once into the
  `planck_cc_generated` library that `hartree-fock` and `planck-mpi` link.
  The registry includes only the headers (2.8 s, 0.48 GB). Shard counts are
  one table in `CMakeLists.txt` (`_planck_cc_shards`), sized for about
  1.4 MB per shard. Peak compile memory is 1.31 GB at 1.4 MB and 1.74 GB at
  2.8 MB; headers alone take 0.35 GB. GCC's `ggc-min-*` parameters barely
  lower it and can make a compile far slower.
- **The build-time generator runs on plain Python** (CMake's
  `Python3_EXECUTABLE`, which need not have numpy). Generation code must
  not import numpy; tests may.

ponytail: t-only builders are rebuilt on every Λ evaluation, though T is
frozen; hoist them out of the Λ ops struct if a Λ iteration costs more than
a T one.

### Λ solver (3c)

`cc_lambda .true.` in `%begin_scf`, after a generated RCC (`run_rccgen`)
or UCC (`run_uccgen`) T solve, calls `solve_lambda_after_t`, which calls
`run_generated_lambda_iterations`. The Λ equations are linear in λ.
- **Kernels:** the caller passes them in (`make_generated_lambda_kernels`),
  as it does the T kernels. The runtime TU does not include the registry,
  because the test binaries that link the runtime
  (`planck-cc-all-sectors-bundle`, `-amplitude-sector-seed`,
  `-arbitrary-solver`) do not link the registry.
- **Update:** the T solver's Jacobi/DIIS step, with T frozen.
- **Denominators:** the T denominators D scaled per block by c, where the
  Λ residual's diagonal is −c·D. ccgen derives c at generation time
  (`lagrangian.lambda_diagonal_scale`: RCC 1, ½, 1/9 at ranks 1–3; UCC 1),
  and the kernels carry it as `GeneratedLambdaKernels::diagonal_scale`.
- **Start:** λ = 0, so the first step is the perturbative λ.
- **Convergence:** residual rms below the T solver's tolerance. The run
  fails if Λ does not converge, and logs the rms of each λ block.

Gates, against the PySCF references in `tests/pyscf/cc_lambda_references.py`
(both pass; water 2.8 s, OH 0.4 s):
- `water_rccsd_lambda_631g` (core): the rms of λ1 and λ2 against the
  spatial Λ equations solved by GMRES at PySCF RCCSD's t;
- `oh_uccsd_lambda_631g` (extended, needs `PLANCK_CC_UCC`): the rms of
  every block against PySCF UCCSD `solve_lambda`.

Both atol 1e-9.

## Open

- **A rank-4 Λ case.** The kernels build; a gate needs a reference, e.g.
  the unrelaxed energy derivative against an FD of the CCSDTQ energy, as
  the RCC Λ-CCSD gate does.
- **Λ cost.** Measure a Λ iteration against a T iteration; they should cost
  about the same.
- **λ in the `.ccamp` file (3c).** The checkpoint stores only t
  (`docs/CC_AMPLITUDE_CHECKPOINT.md`). λ is valid only for the t it was
  solved at, so it goes in the same file:
  - `CCAMP_VERSION` 4 appends an optional λ block: a count, then `by_rank`
    and `sectors` through the existing `write_tensor` codec. A count of 0
    means no λ, which is also how version 1–3 files read.
  - Whoever rewrites t drops λ, so a stale λ cannot survive a new T solve.
  - A restart seeds λ from the file when t also matched; otherwise λ starts
    at zero.

  Gates: a round trip with λ; a version-3 file reads with no λ; a T-only
  rewrite of a file that had λ carries none; a `guess density` restart
  converges Λ in about one iteration.
- **Densities (3d).** Kernels from `density_terms`, generated with
  `canonical_fock=False` and evaluated with Planck's f_ov = 0, plus the
  chain-rule assembly above in C++. The gates must check D_ov: generating
  the densities from canonical-Fock equations drops it with no error.
- **Properties (3e).** Unrelaxed dipole and natural occupations. Gates
  through the regression runner: PySCF's RDMs, the energy identity, and,
  from CCSDT up where no reference code is at hand, dE/dε by central
  differences of the CC energy under an integral perturbation against
  Σ Γ X.
- **Natural orbitals in the `.hfchk` (3e, user request).** This is so that
  `chkdump` can write them as cube files, like MOs.
  - Diagonalize the total unrelaxed 1-RDM in the MO basis, back-transform
    (C·U), sort by occupation.
  - Store them as a general natural-orbital block (coefficients,
    occupations, a method label) in checkpoint version 9, not as a
    CC-specific block. CASSCF keeps its own v3/v5/v6 blocks today, and a
    general block is what it would move to.
  - `chkdump --natural N` writes orbital N as a cube; the occupations go in
    the fchk export.

  Gates: occupations against the eigenvalues of PySCF's CCSD `make_rdm1`;
  Σ occupations = N electrons; the stored orbitals are S-orthonormal; a
  cube of NO 1 matches one built from the same coefficients through `--mo`.

## Step 4: consumers (each its own decision)

- **ΛCCSD(T):** (T) with a Λ-dependent left vector, gated against PySCF's
  (T) gradient machinery where it applies.
- **CCSDT(Q)_Λ, CR-CC(2,3):** `docs/CCSDT_Q_SCOPE.md`.
- **Analytic CC gradients:** their own scope, after the densities pass in
  Planck. The new piece is the CC orbital Lagrangian from D and Γ.
  References: PySCF `grad.ccsd`, `grad.ccsd_t`.

## Risks

- **Conventions in C++.** A wrong factor or transpose still converges, to
  a wrong λ. The CC energy does not depend on λ, so only the λ and density
  gates catch it.
- **Dressing.** The retired dressed spin-adapted route
  (`docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`) shows two correct
  transforms composing wrongly. The emit gate covers dressing after
  differentiation through rank 4 (RCC) and rank 3 (UCC).
- **ROHF.** Λ for ROHF references, where f_ov ≠ 0, inherits
  `docs/ROHF_CORRELATION_SCOPE.md`.
- **Memory.** λ doubles amplitude storage: at CCSDT, t3 and λ3 are both
  o³v³.

## Not in this scope

EOM-CC and linear-response CC (they share the left-hand operator algebra),
orbital-relaxed properties and gradients, ROHF references, frozen core, RI.
