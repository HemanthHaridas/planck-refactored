# Double-hybrid analytic gradients: implementation architecture

This document is the maintainer reference for Planck's double-hybrid (DH)
analytic-gradient implementation. It records the architectural contracts,
equation-to-code mapping, validation model, and remaining scalability work.
It intentionally excludes the superseded debugging diary and historical
finite-difference investigations.

For the user-facing description of the implemented energy and supported
workflows, see [DOUBLE_HYBRID_GRADIENTS.md](DOUBLE_HYBRID_GRADIENTS.md).
Equation numbers refer to Neese, Schwabe and Grimme, *J. Chem. Phys.* **126**,
124115 (2007), DOI [10.1063/1.2712433](https://doi.org/10.1063/1.2712433).

## 1. Scope and status

The landed production path is the restricted, global double-hybrid Cartesian
gradient. It supports the hybrid-KS reference plus a conventional PT2
correction, and returns

```math
g_\mathrm{DH}=g_\mathrm{KS}+\Delta g_\mathrm{PT2}.
```

RKS DH geometry optimization and semi-numerical frequencies reuse this
gradient. A frequency calculation therefore differentiates the analytic
gradient numerically; it is not an analytic DH Hessian.

The implementation currently excludes solvent response, range-separated DH,
general SCS/DOS variants, and a general meta-GGA DH derivative. UKS DH has a
separate validation and grid-qualification track; it must not be inferred
from the RKS contracts in this document.

## 2. Conventions

Planck uses real spatial orbitals and a total closed-shell AO density:

```math
P=2C_oC_o^T,
\qquad
F_\mathrm{KS}=h+J[P]-\frac{a_x}{2}K[P]+V_\mathrm{XC}[P].
```

`J` and `K` are unscaled AO builders. The explicit one-half on exchange is
therefore part of the contract, not an implementation detail. AO indices are
`mu,nu,kappa,tau`; occupied and virtual spatial MO indices are `i,j,k` and
`a,b,c`, respectively. MO coefficients are ordered `C=[Co Cv]`.

The DH energy is

```math
E_\mathrm{DH}=E_\mathrm{KS}+c_\mathrm{PT2}E_\mathrm{MP2}.
```

Only the correlation correction is scaled. No reference KS term may enter
the PT2 correction a second time.

The canonical PT2 amplitude has flat `[i,j,a,b]` storage:

```math
t_{ij}^{ab}=\frac{(ia|jb)}{\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b},
\qquad
\widetilde t_{ij}^{ab}=
\frac{2c_\mathrm{PT2}}{1+\delta_{ij}}
\left(2t_{ij}^{ab}-t_{ij}^{ba}\right).
```

`t` is unscaled. The `tilde` accessor owns the sole PT2 scale in the
gradient algebra.

## 3. Stationary functional and response

The KS orbitals are stationary for the hybrid KS reference, not for the
double-hybrid energy. Orbital response is eliminated through the closed-shell
Z-vector equation

```math
H_\mathrm{KS} Z=-L.
```

Planck represents this in the `v x o` convention (`a,i` ordering):

| Paper role | Code contract | Responsibility |
| --- | --- | --- |
| Eq. 37--39 | `DHPT2AmplitudeDensity` | canonical/tilde amplitudes and `D'` |
| Eq. 40 | `DHEq40AmplitudeRHS` | literal four-coefficient orbital RHS |
| Eq. 41 | `DHEq41ResponseDensity`, `DHEq41ResponseOperator` | `R(D')`, direct J/K and XC-I response |
| Eq. 27 | `apply_dh_eq27_hessian`, `solve_dh_zvector` | Hessian action and `AZ=-L` solve |

The Eq. 40 production convention is `LiteralEq47`. Its four coefficient
slots are the authoritative amplitude orbital derivative. The legacy
compressed internal bracket is retained only for reference checks and must
not be added to `LiteralEq47`, which would double count stationary transfer.

The ordinary solver is GMRES over the typed Hessian action. A dense Hessian
backend exists only as an explicit reference/probe. It is not a production
fallback.

## 4. Gradient assembly

After stationarity, the correction follows Eq. 33:

```math
\Delta g_\mathrm{PT2}=
D:h^x+W:S^x+
\Gamma^{46}:(\mathrm{ERI})^x+
\Gamma^{47}:(\mathrm{ERI})^x+g_\mathrm{XC-II}.
```

Every term is correction-only. The KS gradient already contains all reference
one-electron, Coulomb, exchange, semilocal-XC, overlap, and nuclear terms.

| Channel | Object | Contract |
| --- | --- | --- |
| Relaxed density | `DHRelaxedDifferenceDensity` | Eq. 28, raw and symmetric MO forms |
| Overlap density | `DHOverlapDensity` | adapter over Eqs. 42--45 raw blocks |
| Eq. 46 | `DHEq46_47TwoParticleDensity` | separable `D P-a_x D P/2` correction only |
| Eq. 47 | `DHEq46_47TwoParticleDensity` + tile execution | literal tilde-amplitude pair density |
| Non-XC total | `DHEq33NonXCGradient` | individual h/S/separable/nonseparable ledgers |
| XC-II total | `DHEq33CompleteXCIIGradient` | fixed-coefficient XC geometry contribution |

The overlap-density builders accept no dense MO ERI input on their production
literal-pair path. The Eq. 47-derived pair-metric builder supplies their
amplitude blocks. Dense MO ERIs are required only by explicit legacy or
reference adapters.

### XC-II partition

XC-I belongs in the KS response operator and therefore the Z-vector. XC-II
is the explicit fixed-coefficient nuclear derivative in Eq. 33. The LDA/GGA
implementation contains all of:

- P-side fixed-coefficient contribution;
- D-side AO-density contribution;
- Becke partition movement; and
- atom-centred quadrature-point translation, including the GGA
  Hessian-bearing channel.

Moving-grid terms are mandatory: omitting them gives a physically plausible
but finite-difference-inconsistent gradient.

## 5. Driver ownership and lifetime contracts

The DH driver has intentionally narrow ownership boundaries.

1. `RMP2Result` owns the canonical unscaled `t2` amplitudes.
2. `DHPT2AmplitudeDensity::SourceView` refers to that canonical storage and
   evaluates Eq. 39 on demand. It is the production representation.
3. `Materialized` tilde storage remains an opt-in dense-reference/test
   adapter.
4. `DHStationaryProducts` is built once, moved into
   `DHGradientDriverContract`, and binds its source PT2 result and MO
   coefficients to reject stale handoffs.
5. `DHGradientGeometryWorkspace` shares geometry derivative inputs. The
   dense MO-ERI member is empty in ordinary production and populated only by
   reference callers.
6. `DHEq47TiledExecution` is call-scoped. It may borrow amplitudes and MO
   coefficients only during the synchronous direct Eq. 33 contraction; it
   must never be stored in a returned result.

For an analytic RKS DH gradient, `apply_rmp2_result` publishes energy and
metadata to `Calculator` without copying T2. The MP2 kernel result is then
moved into the local DH snapshot. Other MP2 callers retain the default
calculator amplitude cache.

## 6. Integral and memory strategy

### Production data path

`build_dh_eq41_response_and_solve_zvector` builds only five MO-ERI sectors
needed by the literal Eq. 40 RHS:

| Sector | MO pattern |
| --- | --- |
| `ovov` | `(ia|jb)` |
| `vvov` | `(ab|ic)` |
| `ooov` | `(ij|ka)` |
| `ovvv` | `(ia|bc)` |
| `ovoo` | `(ia|jk)` |

`DHMolecularERIBlocks` makes those layouts explicit. Ordinary RKS DH gradients do not allocate a global MO-ERI tensor.

For Eq. 46, production retains only quadratic `D`, `P`, and `a_x` data and
evaluates the separable coefficient during ERI-derivative traversal. For
Eq. 47, `for_each_dh_eq47_nonseparable_raw_ao_block` produces bounded AO
blocks and the direct derivative consumer contracts each block immediately.
No final raw or symmetric AO four-index pair tensor survives in the production
contract. `ReferenceAll` remains available for literal dense comparisons.

### Current working-set status

| Workspace | Production status |
| --- | --- |
| Full MO ERI | removed; reference-only |
| Final AO Eq. 46/Eq. 47 tensor | removed; streamed/direct contraction |
| Eq. 47 penultimate transform | blocked with an explicit workspace ledger |
| Scaled tilde tensor | removed; source-backed view |
| Calculator MP2 T2 copy during gradient | removed; metadata-only handoff |
| Canonical T2 | pair-packed scratch-backed tile source; no resident amplitude buffer |

The canonical amplitude owner is a recoverable external source. Its remaining
O7 evidence is not a storage redesign, but a measured whole-driver live-memory
and I/O qualification across increasing AO dimensions and tile widths.

## 7. Validation model

Validation proceeds from algebraic primitives to total molecular energies.
All exact reorganizations must pass the relevant lower-level equality before
they are accepted from cancellation in a total gradient.

### Required invariants

- Explicit-index Eq. 37--40 amplitudes, `D'`, Eq. 40 slots and RHS.
- Eq. 41 direct J, K, and XC response checks.
- Dense and matrix-free Eq. 27 actions, RHS sign convention, and Z residual.
- Eq. 42--45 raw/symmetric overlap construction and contraction equivalence.
- Direct-sector versus dense-MO Eq. 40 equality.
- Eq. 46/47 raw, symmetric, tiled, and direct-contraction equality.
- Source-view versus materialized tilde values and D-prime matrices.
- Rigid translation and finite response residual checks.

### Molecular acceptance

The production molecular gate is central total-energy finite differences on
water and nonsymmetric C1 H2O2 B2PLYP/STO-3G in every Cartesian direction,
at two stencil scales. The accepted threshold is

```math
\max |g_\mathrm{analytic}-g_\mathrm{FD}|\le5\times10^{-8}\ \mathrm{Ha/Bohr}.
```

The same fixtures are rerun after any driver, response, density, integral, or
memory-ownership change. The direct/dense `planck-dft-coulomb-response` and
literal-index `planck-dh-pt2-amplitude-density` suites are the companion
component gates.

## 8. How to extend safely

1. State the paper quantity and Planck density/index convention first.
2. Add a typed boundary rather than passing anonymous dense arrays.
3. Add an independently derived reference calculation or finite-difference
   oracle before routing production through the new path.
4. Keep correction-only and KS-reference terms separate.
5. Preserve the `LiteralEq47` RHS convention and `AZ=-L` sign.
6. Run component gates, then both molecular fixtures.
7. Document any retained reference allocation and the exact condition under
   which it is enabled.

## 9. Remaining scalability roadmap

### O7.6.2 — accepted single-owner handoff

The metadata-only `apply_rmp2_result` path and move into the local gradient
snapshot passed the component gates and water/C1 H2O2 molecular-FD pair on
2026-10-11. Analytic RKS DH gradients now have one live canonical T2 owner
after the MP2 kernel completes. Ordinary MP2 callers continue to retain their
calculator-level amplitude cache.

### O7.6.3a — pair-packed canonical representation

Canonical RHF amplitudes obey the exact pair-interchange identity
\(t_{ij}^{ab}=t_{ji}^{ba}\). The first provider migration adds a lossless
upper-triangle adapter over composite `(i,a)` indices. It must reject
noncanonical/nonfinite input and reconstruct every original amplitude before
any production storage owner is switched. This approximately halves canonical
T2 storage but does not, by itself, provide a bounded working set. The packed
identity gate and downstream DH checks passed on 2026-10-11, accepting this
representation as the reference-verified input to the production handoff.

### O7.6.3b — canonical amplitude provider

Replace `RMP2Result::t2` as an always-resident vector with a provider that
can expose bounded `[i,j,a,b]` tiles. The provider must have a recoverable
source, deterministic layout, explicit tile budget, and a materialized
reference implementation. Eq. 37--40 and Eq. 47 consumers must request
tiles rather than retain an unbounded copy. Validate tile-size invariance and
equality to the current materialized reference before switching production.

The first production substep must write the pair-packed owner directly from
the canonical RMP2 kernel. Packing an already materialized full T2 is only an
oracle: it preserves the full-kernel peak and is therefore not a scalability
claim. The DH source view must then retrieve amplitudes through a single
full-or-packed accessor. Generic RMP2 RDM/natural-orbital callers retain full
T2 by default until they have their own packed-accessor migration. A later
tile or recoverable external backend can replace the pair-packed owner without
changing the DH equations.

The direct packed-kernel handoff is accepted: the RKS DH driver requests a
private compact-T2 MP2 option, the canonical kernel writes composite-pair
upper-triangle storage directly, and `RMP2Result::t2_at` abstracts full versus
packed access for DH source views. Generic callers retain the default full
owner. Packed/full DH amplitude and response checks, plus the molecular gates,
passed on 2026-10-11. The compact owner approximately halves the retained
canonical T2 storage, but remains asymptotically quartic and therefore does
not close the bounded-provider objective.

The first consumer seam is also landed: `for_each_rmp2_t2_pair_tile` streams
lexicographically ordered rectangular tiles over composite `(i,a)` / `(j,b)`
pairs. It allocates at most `pair_tile_width^2` values for a callback and is
identical for full and packed owners. Its contract is intentionally independent
of the current backing vector, so Eq. 37--40 and Eq. 47 can migrate one
consumer at a time before the owner becomes recoverable. The tile API itself
does **not** make the current packed result bounded; its full/packed and
width-invariance gate is only a migration prerequisite. This seam and the
full amplitude-density component suite passed on 2026-10-12. Storage mode is
explicit: a `Materialized` object requires its exact Eq. 39 extent, while a
`SourceView` requires an empty local buffer and a valid PT2 source. This
prevents a corrupted one-amplitude materialization from being mistaken for a
valid source view.

### O7.6.4 — transform-and-consume provider

The next closure item is not another T2 packing variant. In the conventional
RMP2 path, `make_eris_rmp2` first materializes the complete chemists-order
`ovov[(ia),(jb)]` transform, and the canonical kernel reads that matrix before
it can publish packed T2. Thus the peak still includes an
`(o v)^2` transformed-integral owner (and, for the conventional backend, the
existing dense AO ERI cache). Replacing only `RMP2Result::t2_pair_packed`
cannot make this path bounded.

The required next contract is a direct transform-and-consume callback: the
AO/RI transform emits a deterministic `(ia),(jb)` tile; the canonical MP2
consumer computes energy and the canonical amplitude tile while it is live;
and the DH gradient either consumes that tile immediately or uses a
recoverable tile source. It must be introduced beside the current materialized
`ChemistsERIs` reference path, with full-tensor equality, tile-width
invariance, pair-symmetry, and peak-live-memory measurements before production
is switched. This is necessarily a transform-backend change, not a safe local
replacement of `t2_at`.

The remaining O7.6 work is deliberately split into these independently
verifiable steps:

1. **O7.6.4a — tile contract and dense oracle.** Define the private
   `(ia),(jb)` tile shape, lexicographic traversal, maximum live-tile budget,
   and callback cancellation/error semantics. Stream the existing materialized
   `ovov` tensor through it and prove exact tile reconstruction at widths 1,
   awkward widths, and a width exceeding the pair dimension. This does not
   change production storage.
2. **O7.6.4b — conventional direct transformer.** Implement the same callback
   at the AO-to-OVOV transform boundary without constructing its full output.
   Compare every tile and total MP2 energy to the dense-transform oracle.
   The AO ERI cache remains an explicitly documented separate owner in this
   step.
3. **O7.6.4c — RI direct transformer.** Emit the equivalent tiles from the
   RI pair factors, using bounded panels of the auxiliary contraction rather
   than a full pair-pair Gram matrix. Check conventional/RI rules separately;
   neither backend may be silently substituted for the other.
4. **O7.6.4d — canonical kernel consumer.** Make RMP2 energy evaluation and
   packed-T2 production consume tiles. Validate energies, full/packed T2,
   pair symmetry, and tile-width invariance against the current canonical
   kernel. Remove the production `ChemistsERIs::ovov` owner only after those
   checks pass. **Implemented:** the compact canonical branch now requests
   metadata-only `ChemistsERIs` and invokes the direct producer at its kernel
   seam. Conventional calculations stream AO-ERI transformed tiles; RI
   calculations stream bounded pair-factor Gram panels. The dense `ovov`
   builder and consumer remain the non-compact reference path. The AO-ERI
   cache and RI pair-factor panel are separate source owners and are not
   claimed to be removed by this step.

   `planck-dh-pt2-amplitude-density` exercises that same canonical tile
   consumer with dense, conventional-direct, and RI-direct producers at
   widths 1, 2, and oversized widths. It compares pair-packed amplitudes and
   both spin energy components; this is a test fixture only, with no runtime
   environment override or dense-reference allocation in production.

5. **O7.6.4e — DH recoverable amplitude source.** Migrate Eq. 37--40 and
   Eq. 47 consumers from scalar `t2_at` calls to their natural amplitude-tile
   access patterns. The source may be an external/recomputable provider, but
   it must have a geometry-scoped lifetime, deterministic reload semantics,
   and no hidden materialized fallback. Keep the current full/packed owner as
   a test oracle until all stationary and molecular gates pass.

   The first consumer migrations are now present: `DHPT2AmplitudeDensity`
   exposes ordered bounded Eq. 39 composite-pair tiles, Eq. 47 fills its
   initial backtransform tensor through that stream, and Eqs. 37--38 read the
   two occupied-pair panels required by each (D') contraction. The component
   gate checks widths 1, 2, and an oversized width against the materialized
   Eq. 39 oracle. The typed Eq. 40 production builder now also scatters the
   literal four coefficient slots directly from those tiles; its dense scalar
   builder remains an equality oracle only. This establishes the DH-facing
   contract. The compact RMP2 owner is now a geometry-scoped, self-cleaning
   scratch-backed packed-amplitude source: after the canonical energy pass it
   clears the in-memory packed vector, and DH reloads only the requested
   composite-pair tile. The component fixture verifies scalar/tile equality
   and proves that both in-memory amplitude buffers are empty for this source.
   The remaining O7.6.4e work is performance qualification of its tile I/O;
   it no longer has an in-memory full/packed fallback.
6. **O7.6.5 — ownership audit.** Verify that normal RKS-DH gradient,
   optimization, and frequency calls retain neither a duplicate `t_tilde`, a
   full canonical T2, nor a full transformed `ovov` buffer after its consumer
   finishes. The production driver enforces this at its stationary-product
   handoff. Generic stationary builders remain permissive because their
   algebraic unit tests intentionally use noncanonical raw tensors; generic
   MP2/RDM callers may also retain their independent full owner.

   The stationary-product handoff now enforces the amplitude portion of this
   contract: a production `SourceView` must retain zero resident bytes for
   raw T2, packed T2, and tilde amplitudes, and must reference a valid
   nonresident source. The component fixture checks this record directly.
   The remaining O7.6.5 closure evidence is a whole-driver peak-live audit,
   including the Eq. 40 sector and derivative-workspace owners.

O7.7 follows O7.6: it measures the resulting peak-live memory across AO size
and tile widths, rather than inferring a memory claim from these contracts.

### O7.7 — measured memory closure

Measure peak live memory over increasing AO dimensions with fixed block
widths. Report direct-versus-reference equality, block-width invariance, and
the water/C1 H2O2 molecular gates. Only then may O7 claim bounded production
working sets.

### O8 — response and XC throughput

Batch grid AO values/derivatives, fuse compatible J/K traversal, and add
reproducible parallel reductions. These are throughput improvements and must
not drop an XC-II moving-grid channel or change closed-shell factors.

### O9 — RI-DH backend

RI-DH is a separate theory backend, not an optimization switch for the
conventional implementation. It requires its own energy-consistent amplitude
objects, auxiliary-metric derivative, and analytic-versus-same-energy FD
validation.
