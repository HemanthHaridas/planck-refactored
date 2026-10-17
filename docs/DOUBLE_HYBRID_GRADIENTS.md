# Double-hybrid analytic gradients: what Planck computes

This is the production-facing reference for Planck's **restricted global
double-hybrid (RKS DH)** analytic gradient. It describes the implemented
energy, derivative contracts, supported scope, and validation evidence. It is
not a chronological derivation log. The ongoing design and scalability ledger
is [DH_IMPLEMENTATION.md](DH_IMPLEMENTATION.md), which records the
maintainer-facing architecture and scalability roadmap.

The implementation follows Neese, Schwabe and Grimme, *J. Chem. Phys.*
**126**, 124115 (2007), in Planck's closed-shell total-density convention.
Equation numbers below refer to that paper.

---

## 1. What is implemented

Planck evaluates a conventional global double hybrid as a hybrid Kohn–Sham
reference plus a non-self-consistent PT2 correction:

```math
E_\mathrm{DH}=E_\mathrm{KS}+c_\mathrm{PT2}E_\mathrm{MP2}.
```

For B2PLYP, the KS reference has 53% exact exchange and the PT2 coefficient
is 0.27. The MP2 **correlation** energy alone is scaled. One-electron,
Coulomb, exchange, semilocal-XC, and nuclear-repulsion terms remain part of
the ordinary KS energy and gradient; they must never be scaled a second time.

The landed analytic-gradient path is restricted, Cartesian-basis, global DH.
It adds a correction gradient to the existing analytic KS gradient:

```math
g_\mathrm{DH}=g_\mathrm{KS}+\Delta g_\mathrm{PT2}.
```

RKS geometry optimization and semi-numerical frequency workflows reuse this
same analytic-gradient path. Frequencies are not analytic DH Hessians.

A narrower UKS global-DH analytic-gradient path is also production-enabled.
It is limited to conventional all-active UMP2 with stored T2, no solvent or
range separation, and `grid_policy g4-3-v4` with
`grid_radial_points 135` and no angular-grid override. UKS DH geometry
optimization and frequencies remain explicitly rejected.

## 2. The convention that controls all factors

Planck uses the total closed-shell AO density

```math
P=2C_oC_o^T,
```

and the hybrid KS operator

```math
F_\mathrm{KS}=h+J[P]-\frac{a_x}{2}K[P]+V_\mathrm{XC}[P].
```

Here $J$ and $K$ are unscaled direct AO builders. Therefore the factor
of one half belongs explicitly on exact exchange. The PT2 amplitudes use
chemists' MO integrals $(ia|jb)$, with occupied/virtual orbital energies
from the converged hybrid reference:

```math
t_{ij}^{ab}=\frac{(ia|jb)}{\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b}.
```

The stored tilde amplitude is $\tilde t=c_\mathrm{PT2}t$. Its scale is
introduced once and is carried through every PT2 density and multiplier.

## 3. Why a Z-vector is necessary

The PT2 correction is evaluated on orbitals that are stationary for the hybrid
KS energy, not for the total DH energy. Directly differentiating the PT2 energy
therefore produces orbital-response terms. Planck removes that response by a
closed-shell KS Z-vector solve:

```math
H_{ai,bj}^\mathrm{KS}Z_{bj}=-L_{ai}.
```

`L_ai` combines the Eq. 41 response-density contribution and the literal
four-coefficient Eq. 40/Eq. 47 amplitude contribution. The production path
uses the typed matrix-free Hessian action; the dense Hessian remains a
reference backend and is checked against the shared action.

The crucial implementation rule is that the Eq. 40 four-coefficient orbital
derivative is the authoritative source of the amplitude RHS and pair-metric
overlap terms. It is not supplemented by an extra compressed “internal” term
in the production `LiteralEq47` convention, because that would double count
the same stationary transfer.

## 4. The correction gradient

After stationarity is imposed, the correction is assembled as paper Eq. 33:

```math
\Delta g_\mathrm{PT2}=
D:h^x+W:S^x+\Gamma^{46}:(\mathrm{ERI})^x+
\Gamma^{47}:(\mathrm{ERI})^x+g_\mathrm{XC-II}.
```

The implementation keeps these channels separate until the final sum:

| Channel | Implemented object | Meaning |
| --- | --- | --- |
| One electron | relaxed density `D` | $D:h^x$ |
| Pulay/overlap | energy-weighted density `W` | $W:S^x$ |
| Separable ERI | Eq. 46 | the correction-only $D P-a_xDP/2$ part |
| Nonseparable ERI | Eq. 47 | the tilde-amplitude pair density |
| XC-II | complete fixed-coefficient derivative | fixed grid, partition, point translation, and AO-density channels |

Eq. 46 contains no reference $PP$ terms: those already belong to
$g_\mathrm{KS}$. Eq. 47 is the literal four-coefficient MO-to-AO transform
of the tilde amplitudes. Raw pair tensors may be contracted against a
symmetry-equivalent ERI derivative; the reference adapters retain explicit
eightfold symmetrizations for independent checking.

## 5. XC-II is a complete geometry derivative, not only a matrix derivative

The explicit XC contribution contains more than the derivative of a fixed AO
matrix. The landed LDA/GGA implementation tracks and sums:

- the ground-density/P-side fixed-coefficient term;
- the relaxed-density/D-side AO term;
- atom-centred Becke partition motion; and
- quadrature-point translation, including the GGA Hessian-bearing term.

The companion XC-I response is represented in the KS response operator and
Z-vector construction, not added again to Eq. 33. Moving-grid terms are
essential for agreement with molecular total-energy finite differences.

## 6. Implementation boundaries

The DH implementation is intentionally typed around the paper objects rather
than a single “MP2 gradient” routine:

| Paper role | Primary code object |
| --- | --- |
| Eq. 37–39 amplitudes and $D'$ | `DHPT2AmplitudeDensity` |
| Eq. 40 orbital RHS | `DHEq40AmplitudeRHS` |
| Eq. 41 response density/operator | `DHEq41ResponseDensity`, `DHEq41ResponseOperator` |
| Eq. 42–45 overlap density | `DHEq42_43EnergyWeightedDensity`, `DHEq44_45EnergyWeightedDensity` |
| Eq. 46–47 pair density | `DHEq46_47TwoParticleDensity` |
| Eq. 33 non-XC channels | `DHEq33NonXCGradient` |
| Eq. 33 XC-II channels | `DHEq33CompleteXCIIGradient` |
| Driver result | `DHGradientDriverContract` |

The implementation lives primarily in
`src/dft/dh_pt2_gradient.{h,cpp}`. `src/dft/driver.cpp` binds its typed
contracts to Planck's AO derivative primitives, direct J/K builders, XC
kernel callback, and converged RKS state.

The direct derivative interfaces are deliberate: h/S and ERI primitives are
contracted as visited instead of retaining one dense AO derivative tensor per
nuclear coordinate. Eq. 46 is evaluated from quadratic D/P/$a_x$ factors.
Eq. 47 streams exact AO tiles directly into the ERI derivative contraction.

## 7. Validation that matters

The implementation was not accepted from a single energy comparison. Its
gates include:

- literal-index amplitude, density, RHS, overlap, and raw/symmetric
  contraction identities on rectangular test systems;
- independent J, K, XC-kernel, Eq. 41, and Eq. 27 response checks;
- dense versus matrix-free Z-vector action and solution comparisons;
- direct-versus-dense Eq. 33 channel equality;
- water and nonsymmetric C1 H2O2 B2PLYP/STO-3G total-energy central
  differences at two displacement sizes; and
- analytic rigid-translation checks.

The accepted production FD threshold is $5\times10^{-8}$ Ha/Bohr. Recent
direct-derivative checks were below that threshold for both water and C1
H2O2, without DH debug flags.

## 8. Current limitations

This document makes no claim of universal DH-gradient support. In particular:

- the broad production contract is RKS global DH. The narrower UKS gradient
  contract above does not extend to UKS geometry optimization or frequencies;
  range-separated, SCS, frozen-core/active-space, and near-degenerate cases
  need their own response and validation contracts;
- the XC-II implementation is LDA/GGA-like, not a general meta-GGA gradient;
- solvated DH gradients are rejected because solvent response is absent; and
- the performance work is still active: O7 retains whole-driver memory and
  scratch-I/O scaling qualification, despite production streaming the compact
  PT2 amplitude and direct MO-ERI tile paths.

## 9. Where to look next

Use [DH_IMPLEMENTATION.md](DH_IMPLEMENTATION.md) for the typed implementation
contracts, ownership rules, validation model, and open O7/O8 scalability work.
