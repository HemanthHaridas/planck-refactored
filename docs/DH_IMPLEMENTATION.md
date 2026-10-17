# Double-hybrid analytic gradients: architecture

This is the maintainer architecture for Planck's global double-hybrid (DH)
analytic-gradient implementation. It describes the live restricted-KS path,
its mathematical contracts, data ownership, and validation boundaries. It is
not a development diary.

For input-level capabilities and user-facing examples, see
[DOUBLE_HYBRID_GRADIENTS.md](DOUBLE_HYBRID_GRADIENTS.md). Equation numbers are
from Neese, Schwabe, and Grimme, *J. Chem. Phys.* **126**, 124115 (2007),
DOI [10.1063/1.2712433](https://doi.org/10.1063/1.2712433).

---

## 1. What is computed

For a restricted global double hybrid, Planck evaluates

```math
E_\mathrm{DH}=E_\mathrm{KS}+c_\mathrm{PT2}E_\mathrm{MP2}.
```

The hybrid KS calculation is self-consistent. The MP2 correlation correction
is evaluated on those converged orbitals and is not made self-consistent. Its
Cartesian derivative is consequently

```math
g_\mathrm{DH}=g_\mathrm{KS}+\Delta g_\mathrm{PT2}.
```

`g_KS` is the ordinary analytic KS gradient. `Δg_PT2` contains only the
correction induced by the scaled PT2 energy. It must never repeat a reference
one-electron, Coulomb, exchange, semilocal-XC, overlap, or nuclear term that
is already in `g_KS`.

RKS DH gradients support geometry optimization and semi-numerical frequencies:
the latter finite-difference the analytic gradient, rather than using an
analytic DH Hessian. A separate, deliberately narrower UKS DH path has its
own spin response and grid contracts; it is not an unrestricted version of
the closed-shell equations documented here.

---

## 2. Conventions that govern every interface

Planck uses real spatial orbitals and the closed-shell total AO density

```math
P=2C_oC_o^T,
\qquad
F_\mathrm{KS}=h+J[P]-\frac{a_x}{2}K[P]+V_\mathrm{XC}[P].
```

`J` and `K` are raw, unscaled AO contractions. The factor `1/2` on exact
exchange belongs to the closed-shell KS convention above. AO indices are
`μ,ν,κ,τ`; occupied and virtual spatial MO indices are `i,j,k` and `a,b,c`;
and `C=[C_o C_v]`.

The canonical amplitude is unscaled:

```math
t_{ij}^{ab}=\frac{(ia|jb)}{\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b}.
```

The gradient's sole PT2 scale is attached to the paper tilde amplitude:

```math
\widetilde t_{ij}^{ab}=
\frac{2c_\mathrm{PT2}}{1+\delta_{ij}}
\left(2t_{ij}^{ab}-t_{ij}^{ba}\right).
```

This rule prevents the most damaging class of implementation error: scaling
the KS reference again, or applying `c_PT2` twice in a PT2 derivative.

---

## 3. The stationary-functional architecture

The hybrid KS orbitals are stationary for `E_KS`, not for `E_DH`. Orbital
relaxation is removed with the KS Z-vector equation

```math
H_\mathrm{KS}Z=-L.
```

Planck stores orbital-rotation quantities in `v × o` (`a,i`) order. The
following types are the paper-to-code boundary; callers should use these
objects rather than passing anonymous arrays.

| Paper quantity | Planck type or function | Meaning |
|---|---|---|
| Eqs. 37–39 | `DHPT2AmplitudeDensity` | canonical/tilde amplitudes and `D'` |
| Eq. 40 | `DHEq40AmplitudeRHS` | literal four-coefficient amplitude orbital RHS |
| Eq. 41 | `DHEq41ResponseDensity`, `DHEq41ResponseOperator` | `R(D')` and its J/K/XC-I channels |
| Eq. 27 | `apply_dh_eq27_hessian`, `solve_dh_zvector` | Hessian action and `AZ=-L` solution |
| Eq. 28 | `DHRelaxedDifferenceDensity` | raw and symmetric relaxed-density forms |
| Eqs. 42–45 | `DHOverlapDensity` | energy-weighted overlap density |
| Eqs. 46–47 | `DHEq46_47TwoParticleDensity` | separable and amplitude-pair ERI densities |
| Eq. 33 | `DHGradientDriverContract` | checked correction assembly |

The production Eq. 40 convention is `LiteralEq47`. Its four coefficient
slots are the authoritative amplitude orbital derivative. A legacy compressed
internal expression exists only as a reference oracle; combining it with
`LiteralEq47` double-counts the stationary transfer.

The ordinary solver is restarted matrix-free GMRES over the typed Eq. 27
action. A dense Hessian is an explicit test/reference backend only. GMRES has
one right-hand side and one trial direction per action, so there is no
production batch of independent orbital-response vectors to exploit.

---

## 4. Driver data flow and ownership

The production pipeline is intentionally a one-way handoff:

```text
converged hybrid KS state
        │
        ├── analytic KS gradient ──────────────────────────────┐
        │                                                       │
        └── RMP2 result → stationary products → Z vector        │
                              │                                 │
                              ▼                                 ▼
                       DH gradient contract ── Eq. 33 PT2 correction
                                                         │
                                                         ▼
                                           requested-frame total gradient
```

The ownership rules make the data path auditable and keep quartic objects out
of the normal gradient lifetime.

1. `RMP2Result` owns the canonical, unscaled T2 source.
2. `DHPT2AmplitudeDensity::SourceView` reads that source and evaluates tilde
   amplitudes on demand. This is the production representation.
3. `Materialized` tilde storage is an explicit dense-reference/test adapter,
   never a requirement of the driver.
4. `DHStationaryProducts` is built once, then moved into
   `DHGradientDriverContract`. It records its source PT2 object, `c_PT2`, exact
   exchange fraction, and MO coefficients so stale handoffs are rejected.
5. `DHGradientGeometryWorkspace` shares geometry-dependent derivative inputs.
   Its dense MO-ERI storage is empty in normal production and populated only by
   reference callers.
6. `DHEq47TiledExecution` is call-scoped. It may borrow amplitudes and MO
   coefficients only during a synchronous direct Eq. 33 contraction; no
   returned object may retain those pointers.

`apply_rmp2_result` publishes energy and metadata to `Calculator` without a
second amplitude copy for the DH-gradient workflow. Other MP2 workflows retain
their normal amplitude cache.

---

## 5. Response operator: J, K, and XC-I

`DHEq41ResponseOperator` is the closed-shell KS response primitive. For an AO
trial density `Q`, it composes raw physical channels at the response boundary:

```math
R(Q)=4J[Q]-a_x\left(K[Q]+K[Q]^T\right)+4V_\mathrm{XC}^{(1)}[Q].
```

The external factors belong here, not in the integral engines. The direct
integral engines return raw `J` and raw `K`.

For hybrid response, the OS, HGP, Rys, and Auto engines use one paired raw
J/K shell-quartet traversal. It shares screening, quartet generation,
symmetry-orbit enumeration, and deterministic reduction while preserving the
same raw output as separate J and K calls. Generic or reference operators may
still provide separate callbacks.

`RKSXCKernel` is an immutable per-geometry XC-I snapshot. Construction caches
the ground density and libxc first/second derivative arrays. Each action only
evaluates the trial density and projects the cached kernel. LDA and GGA
projection use deterministic ascending 128-point blocks, with bounded
`O(B N_AO)` temporary panels rather than grid-sized weighted-AO matrices.

XC-I is response physics and therefore belongs in Eq. 41 and Eq. 27. It is
not an explicit nuclear-gradient channel.

---

## 6. From Z to the correction gradient

After the Z-vector solve, Planck constructs the correction-only Eq. 33
derivative:

```math
\Delta g_\mathrm{PT2}=
D:h^x+W:S^x+
\Gamma^{46}:(\mathrm{ERI})^x+
\Gamma^{47}:(\mathrm{ERI})^x+
g_\mathrm{XC-II}.
```

| Channel | Production object | Important contract |
|---|---|---|
| `D:h^x` | `DHEq33NonXCGradient::one_electron` | relaxed correction density only |
| `W:S^x` | `DHEq33NonXCGradient::overlap` | raw Eqs. 42–45 pass through one raw-to-symmetric adapter |
| Eq. 46 ERI | `two_electron_separable` | closed-shell separable `D,P,a_x` correction |
| Eq. 47 ERI | `two_electron_nonseparable` | literal tilde-amplitude pair density |
| XC-II | `DHEq33CompleteXCIIGradient` | explicit fixed-coefficient XC geometry derivative |

The Eq. 47 literal four-coefficient builder also supplies the amplitude-bearing
pair-metric pieces of Eqs. 42–45. The production overlap builders deliberately
accept no global dense MO-ERI array on this path.

### XC-II is not XC-I

XC-II is the explicit nuclear derivative of the semilocal XC contribution in
Eq. 33. It contains:

- P-side fixed-coefficient AO contribution;
- D-side AO-density contribution;
- Becke partition movement; and
- atom-centred quadrature-point translation, including the GGA
  Hessian-bearing contribution.

These moving-grid terms are required for a finite-difference-consistent
gradient. Treating the grid as fixed gives an apparently reasonable result
that is not the derivative of the implemented energy.

---

## 7. Integral and memory architecture

The production code avoids all global four-index intermediates that are not
mathematically necessary for the final scalar contractions.

### MO-side inputs

Only the five Eq. 40 sectors are constructed:

| `DHMolecularERIBlocks` sector | MO pattern |
|---|---|
| `ovov` | `(ia|jb)` |
| `vvov` | `(ab|ic)` |
| `ooov` | `(ij|ka)` |
| `ovvv` | `(ia|bc)` |
| `ovoo` | `(ia|jk)` |

There is no normal-production full MO-ERI tensor.

### AO-side derivative contractions

Eq. 46 is evaluated from quadratic `D`, `P`, and `a_x` data during the
ERI-derivative traversal. Eq. 47 uses
`for_each_dh_eq47_nonseparable_raw_ao_block`: it emits bounded AO blocks that
the direct derivative consumer contracts immediately. Neither a final raw nor
a symmetrized AO four-index pair tensor survives in the production result.

Dense `ReferenceAll` objects remain valid diagnostic/reference adapters. They
are intentionally named and separated so a test allocation cannot silently
become a production retention requirement.

---

## 8. Validation architecture

The implementation is accepted from independent equalities outward, never
only from cancellation in a final molecular gradient.

### Component contracts

- Literal-index Eqs. 37–40 amplitudes, `D'`, coefficient slots, and RHS.
- Direct J, K, paired J/K, and XC-I response equivalence.
- Dense and matrix-free Eq. 27 actions, Z solutions, sign convention, and
  refreshed residual.
- Eq. 42–45 raw blocks, orientations, symmetric adapter, and contractions.
- Eq. 46/47 raw, symmetric, tiled, and direct derivative-contraction equality.
- Source-view versus materialized tilde values and D-prime matrices.
- Fixed-grid and moving-grid XC-II channels, including rigid translation.
- Ownership and storage contracts: no copied T2, no retained legacy MO-ERI,
  and bounded Eq. 47 workspace.

### Molecular acceptance

The end-to-end gate uses total-energy central differences on water and
nonsymmetric C1 H2O2, B2PLYP/STO-3G, in every Cartesian direction and at two
displacement sizes:

```math
\max|g_\mathrm{analytic}-g_\mathrm{FD}|≤5\times10^{-8}\ \mathrm{Ha/Bohr}.
```

These are production executable checks: no DH debug flags, reference Hessian
override, or internal scalar override is allowed. They must be repeated after
changing the driver, response, density/overlap construction, derivative
contraction, or memory ownership.

---

## 9. Extension rules

Before extending DH gradients:

1. State the paper expression and translate it into Planck's total-density,
   index, and raw-versus-symmetric conventions.
2. Introduce a typed object at the physical boundary; do not pass an unnamed
   dense array between stages.
3. Build an independent literal-index or finite-difference oracle before
   routing the new path into production.
4. Preserve correction-only separation from the KS reference.
5. Preserve the `LiteralEq47` Eq. 40 convention and the `AZ=-L` Z-vector sign.
6. Run the component contract first, then both molecular Cartesian-FD gates.
7. Label every retained dense object as a reference adapter and state its
   lifetime explicitly.

This architecture does not by itself extend the theory to range-separated DH,
solvent response, arbitrary SCS/DOS coefficients, or general meta-GGA DH
derivatives. Each requires a new stationary functional and validation chain;
it is not a parameter change to the current implementation.

---

## 10. Scalability status

The conventional RKS DH gradient now has the following live scalable choices:

| Hot operation | Production design |
|---|---|
| Hybrid Eq. 41 J/K | one paired direct shell-quartet traversal |
| XC-I projection | cached libxc data and bounded fixed-size AO blocks |
| Z vector | single-RHS matrix-free GMRES; no dense production Hessian |
| Eq. 40 data | five explicit MO sectors, not a global MO-ERI |
| Tilde amplitudes | source-backed pair tiles, not a scaled resident tensor |
| Eq. 46/47 derivative | streamed direct contraction, not a retained AO4 tensor |

A block-Krylov interface is intentionally absent: the present single-RHS GMRES
solver supplies one trial direction at a time, so batching cannot remove a
current J/K or XC action. It should be reconsidered only when a real caller
provides multiple independent response right-hand sides at the same geometry.

Hardware-specific timing and peak-RSS studies are useful characterization, but
they are not substitutes for the component and molecular correctness contracts
above.
