# Double-hybrid analytic gradients: formulas and implementation contracts

This is the single current DH implementation reference. It supersedes the
27 earlier scope, derivation, handoff and audit notes, whose original contents
are preserved in the local historical archive
`docs/archive/dh-notes-2026-09-12.tar.gz`. That archive is intentionally
git-ignored and is not distributed with this change; previously tracked
notes are also recoverable from Git history.
Historical conjectures and obsolete debug commands are not implementation
requirements.

Quick access: [energy](#3-energy-and-scfpt2-boundary),
[amplitudes and Dprime](#4-amplitudes-and-unrelaxed-density-eqs-3739),
[response and Z](#5-ks-response-rhs-and-z-vector-eqs-40-41-and-27),
[XC-II](#9-complete-fixed-coefficient-xc-ii-geometry-derivative),
[typed contracts](#10-eq-33-and-driver-contracts),
[validation](#13-validation-evidence-and-remaining-acceptance-work),
[UKS extension scope](#15-uks-extension-scope-and-acceptance-plan).

Equation numbers refer to Neese, Schwabe and Grimme, *J. Chem. Phys.* **126**,
124115 (2007), DOI `10.1063/1.2712433`; the supplied reference is
`/Users/hemanthharidas/Downloads/124115_1_online.pdf`. The equations below
state the actual closed-shell Planck realization, not a verbatim transcription
of the paper. In particular, the literal four-coefficient Eq. 47 derivative
and the KS hybrid exchange coefficient are essential convention mappings.

## 1. Status, supported scope and governing rules

Production enablement and debug cleanup have been rebuilt and validated
(2026-09-12). Both C++ invariant suites pass, and the clean normal-return
production path passes all-coordinate water and nonsymmetric H2O2 total-energy
finite differences at two step sizes without DH debug flags. The production
analytic vectors and FD values exactly reproduce the pre-cleanup results.

- Production is the restricted, global-double-hybrid Cartesian gradient path.
- The validated configurations are B2PLYP/STO-3G, Cartesian basis functions,
  ultrafine quadrature, and symmetry disabled; they are not a validation of
  every supported energy functional or basis.
- Unrestricted and range-separated DH gradients remain excluded. Solvated DH
  gradients are rejected because solvent response is not implemented.
- DH optimization, frequency and response workflows are not enabled by this
  gradient change. The DFT driver rejects spherical basis functions.
- The current XC-II implementation supports LDA-like and GGA-like functionals,
  not a general meta-GGA derivative. No general frozen-core/active-space
  response, spin-component-scaled DH, or near-degeneracy validation is claimed.
- There is no HF-MP2 full-minus-reference subtraction and no
  `RMP2Lagrangian` bridge in this production DH assembly.
- There are no `PLANCK_DFT_DH_*` production switches. The literal RHS and
  literal pair-overlap convention are unconditional.

The fundamental rule is

\[
g_{\mathrm{DH}}=g_{\mathrm{KS}}+\Delta g_{\mathrm{PT2}},
\]

with each perturbative contribution scaled exactly once. The ordinary KS
gradient owns the reference one-electron, Coulomb, exchange, semilocal XC,
Pulay and nuclear-repulsion terms. Every DH contract object below is a
correction object; it must not reintroduce those reference terms.

## 2. Conventions and storage

| Symbol/object | Contract |
|---|---|
| \(\mu,\nu,\kappa,\tau\) | AO indices; \(N\) AOs |
| \(i,j,k,l\) | Occupied spatial orbitals, count \(o\) |
| \(a,b,c\) | Virtual spatial orbitals, count \(v\); code column is \(o+a\) |
| \(p,q\) | General MO indices; \(m=o+v\) |
| \(C=(C_o,C_v)\) | Real \(N\times m\) coefficients, occupied columns first; \(C^TSC=I\) |
| \(P\) | Ground-state **total** RKS AO density, \(P=2C_oC_o^T\); not one-spin density |
| \(h\) | Kinetic plus electron–nuclear attraction matrix |
| \((pq\vert rs)\) | Chemists' ERI ordering; MP2 uses \((ia\vert jb)\) |
| \(t_{ij}^{ab}\) | Unscaled spatial amplitudes, flat `[i,j,a,b]` |
| \(Z_{ai},L_{ai}\) | Matrices with shape \(v\times o\) |
| \(x=(A,q)\) | Nuclear Cartesian component; flat index \(3A+q\), q=x,y,z |
| Units | Energies Ha; nuclear derivatives Ha/Bohr, not forces |

Flat offsets are

\[
\operatorname{idx}_t(i,j,a,b)=(((i\,o+j)v+a)v+b),\qquad
\operatorname{idx}_{\mathrm{ERI}}(p,q,r,s)=(((p\,n+q)n+r)n+s).
\]

Use \(n=N\) for AO ERIs and \(n=m\) for MO ERIs. Matrix contractions below
mean elementwise/Frobenius pairing, \(A:B=\sum_{pq}A_{pq}B_{pq}\).
Do not silently turn a raw nonsymmetric block into a transposed trace.

For symmetric AO matrices, define the unscaled direct builders by

\[
J[Q]_{\mu\nu}=\sum_{\kappa\tau}(\mu\nu|\kappa\tau)Q_{\kappa\tau},
\quad
K[Q]_{\mu\nu}=\sum_{\kappa\tau}(\mu\kappa|\nu\tau)Q_{\kappa\tau}.
\]

The XC callback \(f_{\mathrm{XC}}[Q]\) is the fixed-geometry derivative
\(\left.dV_{\mathrm{XC}}[P+\eta Q]/d\eta\right|_0\). It carries no
extra closed-shell factor supplied by the DH caller.

## 3. Energy and SCF/PT2 boundary

The SCF reference uses

\[
F_{\mathrm{KS}}=h+J[P]-\frac{a_x}{2}K[P]+V_{\mathrm{XC}}[P].
\]

In total-density notation its electronic energy is

\[
E_{\mathrm{KS,elec}}=P:h+\frac12P:J[P]
-\frac{a_x}{4}P:K[P]+E_{\mathrm{XC,semilocal}}[P].
\]

For B2PLYP, \(a_x=0.53\), \(c= c_{\mathrm{PT2}}=0.27\), and

\[
E_{\mathrm{DH,elec}}=T_s+V_{ne}+J
+0.53E_x^{\mathrm{HF}}+0.47E_x^{\mathrm{B88}}
+0.73E_c^{\mathrm{LYP}}+0.27E_c^{\mathrm{MP2}}.
\]

The reported molecular total adds \(E_{\mathrm{nuc}}\) once. In operational
form, \(E_{\mathrm{DH,total}}=E_{\mathrm{KS,total}}+cE_c^{\mathrm{MP2}}\).
Only the MP2 **correlation** energy is scaled; the MP2 total energy is never
added or scaled. MP2 is evaluated after the hybrid SCF, using its orbitals
and orbital energies, without an MP2 self-consistent orbital optimization.

With \(\Delta_{ij}^{ab}=\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b\),

\[
t_{ij}^{ab}=\frac{(ia|jb)}{\Delta_{ij}^{ab}},\qquad
E_c^{\mathrm{MP2}}=\sum_{ijab}(ia|jb)(2t_{ij}^{ab}-t_{ij}^{ba}).
\]

The kernel also reports
\(E_{\mathrm{OS}}=\sum t_{ij}^{ab}(ia|jb)\) and
\(E_{\mathrm{SS}}=\sum t_{ij}^{ab}[(ia|jb)-(ib|ja)]\).
The current DH gradient contract accepts one scalar \(c\), not independent
OS/SS derivative coefficients.

The combined B2PLYP functional supplies its semilocal mixture. A separately
configured correlation functional is ignored as an additional contribution
when exchange identifies a combined XC functional. XC contractions zero the
separate correlation contribution in that case; do not add PBE or LYP again.

Implementation: `src/post_hf/mp2_rmp2.cpp` and
`apply_post_ks_double_hybrid_correction` in `src/dft/driver.cpp`.

## 4. Amplitudes and unrelaxed density: Eqs. 37–39

`build_dh_pt2_amplitude_density` consumes `RMP2Result` and \(c\):

\[
\widetilde t_{ij}^{ab}=\frac{2c}{1+\delta_{ij}}
 (2t_{ij}^{ab}-t_{ij}^{ba}),\qquad
\Theta_{ij}^{ab}=(1+\delta_{ij})\widetilde t_{ij}^{ab}.
\]

\[
D'_{ij}=-\sum_{kab}(1+\delta_{ik})
 \widetilde t_{ik}^{ab}t_{kj}^{ba},
\]

\[
D'_{ab}=\sum_{i\le j,c}
 \left(\widetilde t_{ij}^{ac}t_{ij}^{bc}
       +\widetilde t_{ij}^{ca}t_{ij}^{cb}\right),\qquad
D'_{ia}=D'_{ai}=0.
\]

The occupied contraction's second amplitude is **\(t_{kj}^{ba}\)**.
The virtual contraction is over **\(i\le j\)**; it must not be changed to
an unrestricted pair sum without changing its multiplicities.

`DHPT2AmplitudeDensity` owns `t_tilde`, `dprime_oo`, `dprime_vv`, and
`dprime_mo`. Products contain one scaled and one unscaled amplitude, making
\(D'\) linear in \(c\), not quadratic. Physical canonical amplitudes obey
\(t_{ij}^{ab}=t_{ji}^{ba}\); do not assume symmetry under just one swap.
Particle-number conservation gives \(\operatorname{Tr}D'=0\).

The stationary Eq. 11 scalar provides a useful normalization check:

\[
\mathcal H_{11}=\mathcal H_{\mathrm{pair}}+D':F_{\mathrm{KS,MO}},
\qquad \mathcal H_{\mathrm{pair}}=\sum_{ijab}\Theta_{ij}^{ab}(ia|jb).
\]

At the canonical stationary solution,
\(\mathcal H_{\mathrm{pair}}=2cE_c^{\mathrm{MP2}}\),
\(D':\epsilon=-cE_c^{\mathrm{MP2}}\), so their sum is the production
correction energy. The pair scalar alone is not that energy.

## 5. KS response, RHS and Z-vector: Eqs. 40, 41 and 27

### 5.1 Two different response operators

The physical fixed-geometry Fock response is

\[
\mathscr K[Q]=J[Q]-\frac{a_x}{2}K[Q]+f_{\mathrm{XC}}[Q].
\]

Eq. 41's closed-shell adjoint operator is implemented as

\[
R(Q)=4J[Q]-a_x\{K[Q]+K[Q]^T\}+4f_{\mathrm{XC}}[Q].
\]

For symmetric physical \(Q\), \(R(Q)=4\mathscr K[Q]\). These are not
interchangeable callbacks: using \(R\) for a physical density response
adds a factor of four. `DHEq41ResponseOperator::apply_channels` retains J,
K and XC separately and applies these factors itself.

`build_dh_eq41_response_density` forms
\(D'_{\mathrm{AO}}=CD'C^T\) and `response_ao = R(D'_AO)`.
J/K callbacks bind Planck's direct integral builders. The XC callback binds
`compute_analytic_xc_hessian_vector_product`, evaluated at the converged
ground density on the unchanged reference grid.

### 5.2 Literal amplitude RHS

Define a four-coefficient derivative tensor \(G\) by independently varying
MO coefficients, with fixed AO integrals and fixed \(\Theta\):

\[
\delta\mathcal H_{\mathrm{pair}}=\sum_{pq}G_{pq}U_{pq},\qquad \delta C=CU.
\]

In the following expression, virtual labels inside Kronecker deltas mean
their full MO column indices:

\[
G_{pq}=\sum_{klcb}\Theta_{kl}^{cb}\left[
\delta_{qk}(pc|lb)+\delta_{qc}(kp|lb)
+\delta_{ql}(kc|pb)+\delta_{qb}(kc|lp)\right].
\]

An occupied–virtual rotation has \(U_{ai}=X_{ai}\),
\(U_{ia}=-X_{ai}\), hence

\[
L^{\mathrm{pair}}_{ai}=G_{ai}-G_{ia},\qquad
L_{ai}=[C_v^TR(D'_{\mathrm{AO}})C_o]_{ai}+L^{\mathrm{pair}}_{ai}.
\]

This is what `build_dh_eq40_amplitude_rhs` places in the historically named
`three_external` field. That name does **not** mean the current value is
only one compressed external term: it is the literal derivative of all
four coefficients. `build_dh_lagrangian_rhs` uses `LiteralEq47` by default
and the production solver/contract select it unconditionally.

The old additional bracket

\[
I_{ai}^{\mathrm{SS}}=-c\sum_{klb}t_{kl}^{ab}(ki|lb),\qquad
I_{ai}^{\mathrm{OS}}=+c\sum_{klb}t_{kl}^{ab}(kb|li)
\]

is retained only for explicit reference tests (`include_legacy_internal=true`).
Production does not evaluate it and `included_internal_ai` is zero. Adding
that bracket to the literal \(G_{ai}-G_{ia}\) double-counts the occupied-
coefficient contribution. No extra outer factor of three is to be inferred
from the old field name.

### 5.3 Physical orbital Hessian and sign

For a trial \(X\) of shape \(v\times o\),

\[
\delta P_X=2(C_vXC_o^T+C_oX^TC_v^T),
\]

\[
(AX)_{ai}=(\epsilon_a-\epsilon_i)X_{ai}
+[C_v^T\mathscr K[\delta P_X]C_o]_{ai},\qquad AZ=-L.
\]

`apply_dh_eq27_hessian` returns separate orbital-energy, J, K and XC
channels. Although it stores `raw_z_ao = C_v X C_o^T`, it applies the
physical response to **\(2(\mathrm{raw}+\mathrm{raw}^T)\)**, not to raw Z
or the relaxed-density half-block adapter.

The driver constructs a dense \((ov)\times(ov)\) Hessian by unit-vector
actions, flattens \((a,i)\) as \(ao+i\), and solves by column-pivoted QR.
It checks finite Z and
\(\|AZ+L\|_\infty\le10^{-9}\max(1,\|L\|_\infty)\).
The independently assembled contract must agree with the solver RHS within
1e-12 and satisfy its absolute Z residual within 1e-9.

## 6. Relaxed correction density: Eq. 28

`build_dh_relaxed_difference_density` preserves two distinct objects:

\[
D_{\mathrm{raw}}=\begin{pmatrix}D'_{oo}&0\\Z&D'_{vv}\end{pmatrix},\qquad
D=\begin{pmatrix}D'_{oo}&Z^T/2\\Z/2&D'_{vv}\end{pmatrix},\qquad
D_{\mathrm{AO}}=CDC^T.
\]

The Dprime diagonal blocks are preserved as constructed; the adapter splits
only the raw Z block. For physical symmetric Dprime, D is symmetric, and
\(D_{\mathrm{raw}}:M=D:M\) for any symmetric operator M.
Placing a full Z in both off-diagonal blocks doubles its contribution.

The one-electron, separable ERI and complete XC-II contractions use this D.
The response term in Eq. 42 also uses this symmetric relaxed density.
This is not the total SCF density P, nor P+D.

## 7. Energy-weighted density and overlap adapter: Eqs. 42–45

Using the same literal G of Section 5, define the pair metric blocks

\[
B_{ij}=-\frac12G_{ij},\quad B_{ab}=-\frac12G_{ab},\quad
B_{ia}=-G_{ia},\quad B_{ai}=0.
\]

`build_dh_eq47_pair_metric_overlap_density` evaluates these directly from
\(\Theta\) and the MO ERIs. It uses the paper metric connection
\(U_{ia}=-s_{ia}^{x}\), \(U_{ai}=0\), with
\(s^x=C^TS^xC\), rather than symmetric \(-s^x/2\) in the off-diagonal blocks.

The complete raw W blocks used by production are

\[
W_{ij}=-\frac12[C^TR(D_{\mathrm{AO}})C]_{ij}
-\frac12D'_{ij}(\epsilon_i+\epsilon_j)+B_{ij},
\]
\[
W_{ab}=-\frac12D'_{ab}(\epsilon_a+\epsilon_b)+B_{ab},\qquad
W_{ia}=B_{ia},\qquad W_{ai}=-\epsilon_iZ_{ai}.
\]

The production contract calls the Eq. 42–45 primitives with
`include_legacy_pair=false`: it retains their response and orbital-energy
pieces, skips the superseded compressed pair contractions, and supplies the
literal B blocks. Eq. 45's vo block is not replaced by the pair builder.

Only `build_dh_overlap_density_adapter` converts these raw blocks:

\[
W_s=\tfrac12(W_{\mathrm{raw}}+W_{\mathrm{raw}}^T),\qquad
W_{\mathrm{AO}}=CW_sC^T,\qquad
g_S^x=W_{\mathrm{AO}}:S^x.
\]

The negative metric signs are already inside W. There is no further minus
sign or factor of two in the overlap contraction. The invariant is
\(W_{\mathrm{raw}}:s^x=W_s:s^x\), not equality of raw ov and vo blocks.

## 8. Correction-only two-electron densities: Eqs. 46–47

In AO chemists' ordering, `build_dh_eq46_47_two_particle_density` forms

\[
\Gamma^{\mathrm{sep}}_{\mu\nu\kappa\tau}
=D_{\mu\nu}P_{\kappa\tau}
-\frac{a_x}{2}D_{\mu\kappa}P_{\nu\tau},
\]
\[
\Gamma^{\mathrm{NS}}_{\mu\nu\kappa\tau}
=\sum_{ijab} C_{\mu i}C_{\nu a}C_{\kappa j}C_{\tau b}\Theta_{ij}^{ab}.
\]

Here D means the symmetric AO correction density. The separable term is
the derivative partner of \(D:F_{\mathrm{KS}}\), so its exchange coefficient
must be the **same \(a_x\)** as the KS response and energy. Its contraction
with ordinary ERIs is \(D:J[P]-a_xD:K[P]/2\), not full HF exchange.
\(a_x\) does not scale the nonseparable tensor. The latter uses the
\((ia|jb)\) pairing, not \((ij|ab)\), and already carries c through Theta.

The SCF reference \(\tfrac12PP-\tfrac{a_x}{4}PP\) tensor is excluded.
There is no extra 1/2 multiplying the correction's two-electron contraction.

Raw and eightfold-symmetric tensors are stored separately. The adapter
averages within each AO pair and under pair exchange. Thus

\[
\Gamma_{\mathrm{raw}}:\operatorname{sym}_8(V)
=\operatorname{sym}_8(\Gamma_{\mathrm{raw}}):V.
\]

Production contracts the symmetric tensors with symmetric AO ERI derivatives.
AO and MO tensor ordering, multiplicities and the symmetry adapter must all
agree before an integral engine or contraction is substituted.

## 9. Complete fixed-coefficient XC-II geometry derivative

### 9.1 The scalar being differentiated

For the unpolarized energy density \(f(\rho,\sigma)\), let
\(\rho=\rho_P\), \(p=\nabla\rho_P\), \(d=\nabla\rho_D\),
\(\sigma=p\cdot p\), and \(s=p\cdot d\). Then

\[
\Phi[P,D]=D:V_{\mathrm{XC}}[P]
\approx\sum_g w_g\ell_g,\qquad
\ell=f_\rho\rho_D+2f_\sigma s.
\]

f is an energy density per volume, not the per-electron `exc` value.
`vrho`, `vsigma`, `v2rho2`, `v2rhosigma`, `v2sigma2` supply its indicated
derivatives. The restricted spin reduction uses total density; there is
no outer spin multiplicity in Phi or its nuclear derivative.

Equivalently, its spin-resolved definition is
\(f(\rho,\sigma)=f_{\mathrm{spin}}(\rho/2,\rho/2,
\sigma/4,\sigma/4,\sigma/4)\), with the last three arguments the alpha–alpha,
alpha–beta and beta–beta gradient invariants. All f derivatives above and
below are derivatives of this restricted function.

Both AO coefficient matrices P and D are held numerically fixed in this
partial derivative. Their AO functions and atom-centered quadrature depend
on geometry. Differentiating those AO functions is not the eliminated
orbital/density-matrix response called XC-I.

### 9.2 P-side and D-side AO-center terms

At a fixed laboratory grid point, define
\(r_A=\rho_P^{[Aq]}\), \(g_A=\nabla\rho_P^{[Aq]}\),
\(d_A=\rho_D^{[Aq]}\), \(h_A=\nabla\rho_D^{[Aq]}\).
Square brackets mean AO-center derivatives at fixed AO matrices.

\[
G_P^{Aq}=\sum_gw_g\{\rho_D[f_{\rho\rho}r_A+2f_{\rho\sigma}p\cdot g_A]
+2s[f_{\rho\sigma}r_A+2f_{\sigma\sigma}p\cdot g_A]
+2f_\sigma g_A\cdot d\},
\]
\[
G_D^{Aq}=\sum_gw_g\{f_\rho d_A+2f_\sigma p\cdot h_A\}.
\]

`build_dh_eq33_xc_fixed_density_gradient` supplies only G_P despite its
broad historical name. The LDA/GGA `difference_ao_gradient` builders supply
G_D. Omitting G_D does not produce the complete operator derivative.

For an AO centered on atom A,
\(\partial_{R_{Aq}}\chi_\mu=-\partial_q\chi_\mu\).
For symmetric fixed Q, the density-center derivative is
\(-2\sum_{\mu\in A,\nu}Q_{\mu\nu}(\partial_q\chi_\mu)\chi_\nu\).
Its spatial gradient uses AO Hessians as well as AO first derivatives.

### 9.3 Becke partition and moving points

Write \(w_g=w_g^{\mathrm{atom}}b_g\), with owner atom B. The point moves by
\(d\mathbf r_g/dR_{Aq}=\delta_{AB}\mathbf e_q\).
`becke_partition_owner_derivatives` returns \(\dot b_g^{Aq}\), the total
partition derivative along that moving point path. Atomic radial/angular
weights are unchanged by translation.

\[
G_{\mathrm{Becke}}^{Aq}=\sum_gw_g^{\mathrm{atom}}\dot b_g^{Aq}\ell_g,
\qquad
G_{\mathrm{point}}^{Aq}=\sum_gw_g\delta_{A,B(g)}\partial_q\ell_g.
\]

Do not add another spatial Becke-weight derivative: it is already in
\(\dot b\). Point translation differentiates the integrand only.
With \(H_P(:,q)=\partial_qp\) and \(H_D(:,q)=\partial_qd\),

\[
\begin{aligned}
\partial_q\ell={}&[f_{\rho\rho}p_q+2f_{\rho\sigma}p\cdot H_P(:,q)]\rho_D+f_\rho d_q\\
&+2[f_{\rho\sigma}p_q+2f_{\sigma\sigma}p\cdot H_P(:,q)]s\\
&+2f_\sigma[H_P(:,q)\cdot d+p\cdot H_D(:,q)].
\end{aligned}
\]

For fixed symmetric Q, density Hessians are
\((H_Q)_{sq}=2[(\partial_s\partial_q\chi)^TQ\chi+
(\partial_s\chi)^TQ(\partial_q\chi)]\). The GGA implementation needs these
AO/density Hessians, but not third functional derivatives for XC-II.

For LDA, set all sigma derivatives to zero:
\(G_P=\sum w f_{\rho\rho}r_A\rho_D\),
\(G_D=\sum w f_\rho d_A\),
\(\partial_q\ell=f_{\rho\rho}p_q\rho_D+f_\rho d_q\).

### 9.4 Assembly and exclusions

`DHEq33CompleteXCIIGradient` owns four \(N_{\mathrm{atom}}\times3\) fields:
`p_side_fixed`, `d_side_ao`, `becke_partition`, `point_translation`, plus

\[
G_{\mathrm{XC-II}}=G_P+G_D+G_{\mathrm{Becke}}+G_{\mathrm{point}}.
\]

This is the complete explicit fixed-coefficient geometry contribution.
Do not append \(D^x:V_{\mathrm{XC}}\), a separate XC-I correction, another
Z-multiplier derivative, or an empirical XC scale. Amplitude/orbital response
has already been eliminated into the relaxed stationary objects.

## 10. Eq. 33 and driver contracts

The correction for coordinate x is

\[
\boxed{\Delta g^x=D_{\mathrm{AO}}:h^x+W_{\mathrm{AO}}:S^x
+\Gamma^{\mathrm{sep}}:V^x+\Gamma^{\mathrm{NS}}:V^x
+G_{\mathrm{XC-II}}^x.}
\]

No global 1/2, spin factor, additional PT2 scale, or extra nuclear-repulsion
derivative is applied. The nuclear term is already in g_KS.

| Boundary/type | Inputs and ownership | Output/obligation |
|---|---|---|
| `RMP2Result` | Converged KS MOs/energies and unscaled t2 | One geometry and consistent occupied/virtual indexing; populated t2 required |
| `DHEq41ResponseOperator` | Direct J/K callbacks, XC HVP, a_x | Raw callbacks; operator applies closed-shell adjoint factors |
| `DHEq41ZVectorProducts` | Converged restricted driver geometry | Amplitudes, MO ERIs, response, L, Z and checked residual |
| `DHEq33NonXCDerivatives` | Same converged geometry | 3N_atom h/S/ERI arrays in common atom-major order |
| `DHEq33XCFixedDensityInputs` | Molecule, basis, grid, AO values/Hessians, P, functionals | Fixed-coefficient reference state and live pointer lifetimes |
| `DHGradientDriverInputs` | PT2 result, c, C, epsilon, MO ERIs, response, solved Z, derivative/XC inputs | All belong to the same geometry and MO convention; no internal SCF or Z solve |
| `DHGradientDriverContract` | Inputs above | Owns all assembled Dprime, L, D, W, Gamma and non-XC/XC-II matrices |
| `DHEq33PT2CorrectionGradient` | Completed contract | Non-XC, XC-II and total correction, each natoms by 3 |
| Driver finalization | Analytic KS gradient plus correction | Sum in common frame, rotate once, store/return normal analytic gradient |

The direct-response callback borrows shell-pair/symmetry data. The XC callback
copies the ground density but borrows grid, AO data and functionals. All
borrowed objects must outlive every solve/contract evaluation. An endpoint
must not mutate them while a reference-geometry oracle holds them fixed.

`response_operator.exact_exchange` is the contract's single a_x source for
both KS response and Eq. 46. c enters through scaled amplitudes once;
consistent c=0 inputs and Z=0 produce a zero correction.

The derivative builder constructs overlap, kinetic/electron–nuclear and
four-center ERI derivatives analytically in the Cartesian AO basis, including
derivatives with respect to the nuclear-attraction centers. Derivative arrays
are not derivatives of already-transformed live MO tensors. The current
implementation materializes dense AO/MO tensors and a dense orbital Hessian:
storage includes O(3N_atom N^4) ERI derivatives and O((ov)^2) response storage.
It is not a density-fitted, sparse or large-system scalability implementation.

Numerical dimension, finite-value and Z-residual checks remain production
safety checks. Debug ledgers, selfcheck FD loops, snapshots and environment
gates have been removed from the production driver.

## 11. Stationary transport and what may be compared

Amplitude stationarity requires
\(\mathcal H_{\mathrm{pair},t}[t^x]+(D':F)_{,t}[t^x]=0\).
Neither summand vanishes independently. Orbital stationarity requires
\(L:X+Z:AX=0\). Do not differentiate live D, W and Gamma separately and
interpret those live-object derivatives as missing terms of Eq. 33.

A valid frozen metric frame is
\(C_\parallel(R)=C_0[C_0^TS(R)C_0]^{-1/2}\), retaining t0 in those labels.
Copying raw t0 into an independently optimized endpoint MO basis does not
hold the same amplitude tensor fixed. Occupied-only/virtual-only alignment
cannot remove a changing occupied–virtual subspace.

For a symmetric-metric oracle,

\[
P_{\mathrm{metric}}^x=-C(s^xO+Os^x)C^T,\quad O=\mathrm{diag}(I_o,0_v),
\]
\[
F_{\mathrm{op}}^x=h^x+J^x[P]-a_xK^x[P]/2+(V_{\mathrm{XC}}[P])_{\mathrm{geom}}^x,
\]
\[
F_{ai,\mathrm{comoving}}^x=
[C^T\{F_{\mathrm{op}}^x+\mathscr K[P_{\mathrm{metric}}^x]\}C]_{ai}
-\tfrac12(\epsilon_a+\epsilon_i)s_{ai}^x.
\]

`build_dh_ks_fixed_density_fock_derivative` is a retained standalone oracle
primitive, not an additional production gradient channel. The expression
above and the paper ov/vo metric split use different connections. Converting
the non-Z sector to symmetric transport adds \(-L:s_{ai}^x/2\), while the
Z-sector connection changes oppositely. Compare their **combined** functional,
not mismatched individual blocks.

There are three distinct validation levels:

1. Explicit-index/unit tests check algebra, dimensions, multiplicities and
   adapters without proving molecular stationarity.
2. Fixed-contract/fixed-geometry oracles check a specified scalar or operator
   with stated held objects. They are not arbitrary decompositions of the
   live total-energy derivative.
3. Molecular FD reruns SCF/PT2 at each displaced geometry and compares the
   assembled analytic gradient to the same live energy expression.

## 12. Resolved defects and superseded interpretations

| Superseded implementation or claim | Resolution used by production |
|---|---|
| HF-MP2 full-minus-reference gradient objects are an adequate KS bridge | Direct correction-only KS/PT2 contract |
| Literal external pair derivative needs the old internal bracket added | Literal four-leg derivative already includes it; do not add it twice |
| Solver and contract may select different RHS conventions | Both use the same literal convention with residual checks |
| Compressed Eq. 42–44 amplitude blocks are adequate in stored-t2 orientation | Literal Eq. 47 pair metric blocks replace them |
| Eq. 46 correction always uses full HF exchange | Use -a_x/2 for Dprime and Z alike |
| P-side fixed-grid XC alone is complete XC-II | Include D-side AO, partition and Hessian-bearing point-motion terms |
| Nonzero mixed-metric Z ledger residual is a missing overlap term | Convert non-Z and Z connections consistently; their shifts cancel |
| A small algebraic AZ+L residual proves the full nuclear derivative | Require independent orbital/function FD and molecular energy FD |

The decisive water H1-x exchange attribution was an analytic increase of
`+1.021419311292372e-3` from Dprime and `+2.318901107702650e-4` from Z,
total `+1.253309422062637e-3` Ha/Bohr. The corrected analytic PT2 derivative
is `-0.005201722911447675`; two-step FD errors were 3.73e-10 and 3.40e-10.
The older roughly 1.6178e-4 mixed-metric remainder was not an additional
net overlap defect. Historical claims of an unresolved 31% remainder,
necessary empirical XC scales, or a disabled mathematical path are obsolete.

## 13. Validation evidence and remaining acceptance work

Independent tests include tilde/Dprime invariants, literal four-leg RHS and
metric contractions, raw/symmetric equivalence, separate J/K/XC responses,
Z residual perturbations, Eq. 46 direct density-energy variations at
a_x=0,0.53,1, physical Eq. 47 contractions, and complete LDA/GGA XC geometry
FD. Important files are:

- `tests/dh_pt2_amplitude_density.cpp`
- `tests/dft_coulomb_response.cpp`
- `tests/dh_eq41_xc_response.cpp`
- `tests/dh_cartesian_fd_audit.py`
- `tests/test_dh_cartesian_fd_audit.py`

The following **pre-cleanup, rebuilt diagnostic-path** molecular tests passed
the declared 5e-8 Ha/Bohr tolerance. Steps are in Bohr; errors are maximum
absolute analytic-versus-FD differences in Ha/Bohr:

| Molecule | Step | KS | PT2 | Total |
|---|---:|---:|---:|---:|
| Water, all 9 coordinates | 1e-4 | 1.3805e-8 | 6.8841e-10 | 1.4466e-8 |
| Water, all 9 coordinates | 2e-4 | 1.6363e-8 | 6.0925e-10 | 1.6953e-8 |
| C1 H2O2, all 12 coordinates | 1e-4 | 6.2719e-9 | 4.0342e-9 | 9.0345e-9 |
| C1 H2O2, all 12 coordinates | 2e-4 | 1.3418e-8 | 1.9309e-9 | 1.3427e-8 |

This comprises 126 scalar KS/PT2/total comparisons, 84 displaced energy
calculations and two centers. Every endpoint converged. H2O2 is genuinely
nonplanar C1: the displacement triple product is 1.039303744 angstrom^3 and
only the identity atom-type-preserving permutation preserves its distances.
No DH analytic-derivative comparison against PySCF is used as acceptance.

Full-precision analytic total vectors for these geometries are:

```text
Water (O, H1, H2), Ha/Bohr
 0.000000000000   0.000000000000   0.189476207965
-0.090098382592   0.000000000000  -0.094738103982
 0.090098382592   0.000000000000  -0.094738103982

C1 H2O2 (O1, O2, H1, H2), Ha/Bohr
 0.028317258016  -0.004080151648   0.010478148676
-0.136935823421  -0.008869204937   0.060366332901
-0.023984900274  -0.007883489466  -0.013521100305
 0.132603465680   0.020832846050  -0.057323381273
```

The display is rounded; machine-readable evidence retains full precision.
Largest analytic total translation components were 1.77e-14 (water) and
1.42e-13 (H2O2). Corresponding center AZ+L residuals were 9.541e-18 and
1.561e-17, with zero solver/contract RHS differences.

Logs and individual inputs/results, retained from those runs:

- `/private/tmp/dh-water-full-cartesian-fd.log`
- `/private/tmp/dh-h2o2-full-cartesian-fd.log`
- `/private/var/folders/b6/8bkpkc5j2hjgttz3q3yv16gr0000gn/T/dh-cartesian-fd-t2p_atgu/`
- `/private/var/folders/b6/8bkpkc5j2hjgttz3q3yv16gr0000gn/T/dh-cartesian-fd-zvj_x24k/`

Temporary log paths may not survive system cleanup. The archive preserves
the prior detailed reports and coordinate-by-coordinate error tables.

### Post-cleanup production acceptance

After the user-managed rebuild, `planck-dh-pt2-amplitude-density` and
`planck-dft-coulomb-response` both passed with exit code 0, including the
new production-default, skipped-legacy-work and contract tests. No build
was started by the agent. The following normal-output molecular checks
were then run and passed:

```sh
python3 tests/dh_cartesian_fd_audit.py \
  tests/inputs/exploratory/dh_gradient/water_b2plyp_gradient_fd.hfinp \
  --jobs 2 > /private/tmp/dh-water-production-fd.log 2>&1
python3 tests/dh_cartesian_fd_audit.py \
  tests/inputs/exploratory/dh_gradient/h2o2_c1_b2plyp_gradient_fd.hfinp \
  --jobs 2 > /private/tmp/dh-h2o2-production-fd.log 2>&1
```

The current runner consumes ordinary `--json` energies and gradients,
requires successful exit and convergence, sets no DH flags, and checks every
coordinate at both steps. JSON now emits 17 significant digits; 12-digit
total energies were insufficient for this FD tolerance. The current runner
checks total production gradients, not the removed separate debug ledgers.
Its two fast Python tests also pass. Post-cleanup production results are:

| Molecule | Step (Bohr) | Max total-gradient error (Ha/Bohr) | RMS error (Ha/Bohr) |
|---|---:|---:|---:|
| Water | 1e-4 | 1.44657175e-8 | 6.30309594e-9 |
| Water | 2e-4 | 1.69526181e-8 | 6.89204095e-9 |
| C1 H2O2 | 1e-4 | 9.03449667e-9 | 3.67931789e-9 |
| C1 H2O2 | 2e-4 | 1.34265307e-8 | 6.07051681e-9 |

All 42 total-gradient comparisons pass the unchanged 5e-8 tolerance.
All 86 production calculations (two gradient centers and 84 energy endpoints)
converged and exited normally. The full analytic vectors and each total-energy
FD value agree exactly, at stored precision, with the pre-cleanup diagnostic
results. Thus cleanup and the normal-return wiring introduced no measured
change for these two systems. The normal output contains no DH audit ledgers
or disabled-gradient error; the remaining ERI-construction progress message
is ordinary production logging.

Production evidence:

- `/private/tmp/dh-production-amplitude-test.log` (successful suite is silent).
- `/private/tmp/dh-production-coulomb-test.log`.
- `/private/tmp/dh-water-production-fd.log`.
- `/private/tmp/dh-h2o2-production-fd.log`.
- Water inputs, outputs and `results.json`:
  `/private/var/folders/b6/8bkpkc5j2hjgttz3q3yv16gr0000gn/T/dh-production-cartesian-fd-ek6g9t3g/`.
- H2O2 inputs, outputs and `results.json`:
  `/private/var/folders/b6/8bkpkc5j2hjgttz3q3yv16gr0000gn/T/dh-production-cartesian-fd-hyjg_5da/`.

The broader scope limitations in Section 1 remain; production acceptance
on these fixtures is not a claim of universal functional/basis coverage.

## 14. Source map and historical archive

| File | Responsibility |
|---|---|
| `src/dft/driver.cpp` | Workflow restrictions, converged geometry, SCF/PT2 energy, Z solve, derivative arrays, final KS+PT2 sum and frame handling |
| `src/dft/dh_pt2_gradient.h` | Typed correction-only interfaces, dimensions and callback lifetimes |
| `src/dft/dh_pt2_gradient.cpp` | Eqs. 27–47 primitives, literal adapters, complete XC-II and contract assembly |
| `src/dft/analytic_hessian.*` | Analytic XC Hessian-vector primitive used by the DH response |
| `src/dft/dft_gradient.*` | AO-center/density derivative helpers and ordinary KS XC nuclear gradient |
| `src/post_hf/mp2_rmp2.cpp` | Unscaled restricted amplitudes and MP2 correlation energy |
| `src/io/results_json.h` | Normal machine-readable output with round-trip double precision |

Some low-level field comments retain historical names such as
`three_external` or `fixed_density`; Sections 5 and 9 specify their actual
production meaning. The legacy internal and compressed-pair test options
are not runtime production gates.

The local, git-ignored archive contains all original `DH_*.md` and `DOUBLE_HYBRID_*.md` notes
that this document replaces, including both tracked and previously untracked
files. It is historical evidence, not a second current specification.
Every archived file was verified byte-for-byte against its original before
the originals were removed. Archive SHA-256:
`be01a6caf16cc14dd1aba5a4a217860312f86132c554e252d0cf3fc037626945`.
To inspect or restore it without overwriting the consolidated document:

```sh
tar -tzf docs/archive/dh-notes-2026-09-12.tar.gz
mkdir -p /private/tmp/planck-dh-notes-restored
tar -xzf docs/archive/dh-notes-2026-09-12.tar.gz \
  -C /private/tmp/planck-dh-notes-restored
```

The archived `DH_FULL_CARTESIAN_VALIDATION.md` preserves full molecular
tables; `DH_ERROR_ATTRIBUTION_EXCHANGE.md` preserves the final termwise
defect isolation. Earlier proposals are retained for provenance only and
must be read in light of the resolved distinctions in Section 12.

## 15. UKS extension scope and acceptance plan

Scoped 2026-09-12 following the completed RKS production validation.
**This section is a proposed implementation contract, not UKS validation
evidence or production enablement.** No source code, build, or runtime gate
is changed by this scope. Sections 1–14 continue to describe production RKS.
Keep this extension in the same authoritative document.

### 15.1 Deliverable and boundaries

Implement a correction-only analytic Cartesian gradient for a converged,
real, collinear UKS global double hybrid, initially B2PLYP with one PT2
coefficient, all electrons correlated, integer occupations, Cartesian basis
functions, direct conventional four-center PT2, and LDA/GGA semilocal XC.
The first molecular acceptance set uses STO-3G, ultrafine quadrature and
symmetry disabled, as in the restricted validation. LDA is a primitive
validation case; B2PLYP supplies the physical GGA acceptance case.

The output remains

\[
g_{\rm UDH}=g_{\rm UKS}+\Delta g_{\rm UPT2}.
\]

Do not enable ROHF/ROKS, noncollinear or spin-flip response, range separation,
meta-GGA, SCS/SOS coefficients, RI/DF gradients, frozen core/truncated active
spaces, solvent, external-field extensions, optimization or frequencies in
this step. Do not add spin projection or claim to remove UKS spin
contamination. Near-zero denominators and singular orbital-response systems
require explicit failures, not regularization that changes the energy.
Memory scaling remains that of the existing dense DH derivative bundle;
large-system streaming is a separate project.

### 15.2 Baseline architecture: reuse versus new work

| Existing code | Verified architectural fact | Extension decision |
|---|---|---|
| `driver.cpp::apply_post_ks_double_hybrid_correction` | UKS single-point DH already calls `ump2_kernel`, scales only correlation energy, and adds it to KS energy; only the RMP2 result can currently be returned to a gradient caller | Preserve energy semantics; add a typed UMP2 result handoff, with amplitudes retained from the same converged UKS snapshot |
| `mp2.h::UMP2Result`, `mp2_ump2.cpp` | Separate coefficients, energies, dimensions and `t2_aa`, `t2_ab`, `t2_bb` | Reuse the unscaled energy/amplitude contract after explicit normalization tests |
| `mp2_ump2.cpp::ump2_gamma1_intermediates` | Correction-only per-spin oo/vv contractions exist; `ump2_make_rdm1` adds occupied identity | Audit gamma1 against the stationary scalar; never use the reference-containing RDM as the DH difference density |
| `driver.cpp` UKS SOSCF action | Joint alpha/beta action uses total-density J, same-spin K and polarized analytic XC response | Extract or bind a checked physical UKS response primitive; do not copy the whole SCF lambda or change RKS SOSCF |
| `analytic_hessian.*::compute_analytic_xc_hessian_vector_product_polarized` | Physical LDA/GGA spin-coupled AO XC action already exists | Bind to one immutable reference; retain all four target/source spin channels in tests |
| `driver.cpp::build_unrestricted_xc_kernel_blocks` | Uses density finite differences, not the analytic response implementation | Independent test oracle only; not a production DH Hessian |
| `post_hf/uhf_response.*` | UHF CPHF matrix/solver infrastructure exists | Packing/solver patterns only; the HF operator is not the UKS operator with fractional exchange and XC |
| `driver.cpp::compute_analytic_ks_gradient` | Already dispatches HF-like and XC nuclear terms to UKS implementations | Reuse the existing reference gradient; do not reconstruct or subtract a UMP2 total gradient |
| `driver.cpp::build_dh_eq33_derivative_integrals` | AO h, S and ERI derivatives depend on geometry/basis, not spin | Reuse a single derivative bundle for both spins |
| `dh_pt2_gradient.*` | Public DH response, density, overlap, XC-II and driver types carry a single restricted space/density | Add dedicated unrestricted contracts; share only genuinely spin-independent contraction/grid mechanics |
| `tests/dh_cartesian_fd_audit.py` | Normal JSON all-coordinate total-energy FD runner preserves input charge/multiplicity | Reuse after UKS acceptance fixtures and state-continuity checks; do not restore production debug dumps |

Two concrete edge cases must not be inherited silently. The UMP2 dimension
resolver permits zero beta occupancy, but `ump2_gamma1_intermediates` rejects
any empty amplitude vector, including a mathematically empty spin block.
The new contract must distinguish expected zero extent from missing stored
amplitudes. Also, the inline UKS SOSCF action returns a zero vector on XC
callback failure; a new DH response wrapper must propagate the error and
must not imitate that fallback.

Existing polarized regression targets worth rerunning after user-managed
builds are `planck-dft-gga-polarized-fxc-ordering`,
`planck-dft-gga-polarized-hessian-selfcheck` and
`planck-dft-analytic-hessian-polarized-production`. Their existence does not
establish a validated unrestricted DH RHS, W, XC-II or final gradient.

### 15.3 Paper map and normalization boundary

Use the paper's unrestricted starting equations, **Eqs. 7–35**, rather than
adding spin labels to Eqs. 37–47:

| Paper equations | Proposed Planck object / proof |
|---|---|
| 1, 7–10 | UKS energy, canonical aa/ab/bb amplitudes and physical Fock |
| 11–13 | One stationary PT2 scalar and correction-only per-spin Dprime |
| 18–21 | Separate alpha/beta coefficient connections constrained by the same AO overlap |
| 22–26 | Literal four-coefficient pair RHS plus the polarized Dprime response adjoint |
| 27 | One coupled alpha/beta KS Z solve |
| 28 | Raw vo multipliers and separate symmetric AO density adapters |
| 29–32 | Per-spin oo/vv/ov/vo W blocks, then one AO overlap contraction |
| 33 | Correction-only non-XC terms plus complete polarized XC-II |
| 34–35 | Hybrid separable density and aa/ab/bb nonseparable pair density |
| 36–47 | Closed-shell reduction checks, not unrestricted storage definitions |

Important reading/convention cautions from the supplied paper:

- Eqs. 31/32 label the pair and Z off-diagonal blocks `ai`/`ia`, whereas
  Eqs. 44/45 label the analogous closed-shell blocks `ia`/`ai`. Planck must
  select orientation from the coefficient differential and contraction,
  not silently transpose based on an equation label.
- Eq. 9 explicitly contains the hybrid coefficient \(a_x\); later compact
  exchange expressions do not display it consistently. Derive response and
  separable ERI exchange from Eq. 9, retaining \(a_x\) exactly once.
- The response operator printed around Eq. 23 is not a specification for
  multiplying Planck's physical polarized XC callback by an inferred
  closed-shell factor. Fix its adjoint normalization with the scalar
  identity below, including XC, then check the paper mapping.
- The printed stationary functional must be reconciled with Planck's full
  ordered-pair amplitude storage by stationarity and energy recovery. A
  trace or occupied-pair restriction cannot be copied without its weights.

These are explicit mapping obligations, not grounds to introduce fitted
factors. The proposed formulas below follow one operational scalar and
remain subject to the U0–U11 acceptance gates.

### 15.4 Spin, energy and amplitude contracts

For \(\sigma\in\{\alpha,\beta\}\), maintain distinct occupied/virtual
dimensions and coefficient matrices, even when their AO row counts agree:

\[
P^\sigma=C_o^\sigma C_o^{\sigma T},\quad
P=P^\alpha+P^\beta,\quad C^{\sigma T}SC^\sigma=I,
\]
\[
F^\sigma=h+J[P]-a_xK[P^\sigma]+V_{\rm XC}^\sigma[P^\alpha,P^\beta].
\]

Occupied spin orbitals have occupation **one**. The KS electronic energy is

\[
E_{\rm UKS}=P:h+\tfrac12P:J[P]
-\tfrac{a_x}{2}\sum_\sigma P^\sigma:K[P^\sigma]
+E_{\rm XC}[P^\alpha,P^\beta].
\]

Define \(g_{ij}^{ab,\sigma\tau}=(i_\sigma a_\sigma|j_\tau b_\tau)\),
\(v^{\sigma\sigma}_{ijab}=g^{\sigma\sigma}_{ijab}
-g^{\sigma\sigma}_{ijba}\), and the appropriate spin-resolved denominator
\(\Delta^{\sigma\tau}_{ijab}=\epsilon_i^\sigma+\epsilon_j^\tau
-\epsilon_a^\sigma-\epsilon_b^\tau\). The stored amplitudes are

\[
T^{\sigma\sigma}_{ijab}=v^{\sigma\sigma}_{ijab}/\Delta^{\sigma\sigma}_{ijab},
\qquad T^{\alpha\beta}_{ijab}=g^{\alpha\beta}_{ijab}/\Delta^{\alpha\beta}_{ijab}.
\]

Same-spin tensors are antisymmetric under each occupied or virtual swap.
The ab tensor is not antisymmetric; if needed, reconstruct
\(T^{\beta\alpha}_{ji ba}=T^{\alpha\beta}_{ijab}\), without adding a
second independent OS energy. The exact energy contract is

\[
E_{\rm UPT2}=\tfrac14\sum_{ijab}T^{\alpha\alpha}_{ijab}v^{\alpha\alpha}_{ijab}
+\sum_{ijab}T^{\alpha\beta}_{ijab}g^{\alpha\beta}_{ijab}
+\tfrac14\sum_{ijab}T^{\beta\beta}_{ijab}v^{\beta\beta}_{ijab}.
\]

All sums here are full ordered sums, not \(i<j\) sums. The same-spin
kernel's `0.5 * (g / Delta) * (g - g_exchange)` sum is algebraically the
same energy after index summation; it is not the stored same-spin amplitude.
Scale the final energy by \(c=c_{\rm PT2}\) once. For gradient intermediates,
absorb this same \(c\) linearly into each correction object, as in RKS.

Storage is `[i_sigma,j_tau,a_sigma,b_tau]`, offset
`(((i * nocc_tau + j) * nvirt_sigma + a) * nvirt_tau + b)`.
No equal-spin-dimension assumption, no automatic occupied identity, and no
RKS `t_tilde` conversion belong at this boundary.

### 15.5 One common stationary scalar, Dprime and literal pair derivative

Define \(\mathcal P=\Gamma_{\rm NS}:I_{\rm AO}\) using

\[
\Gamma_{\mu\nu\kappa\tau}^{\rm NS}
=c\sum_{\sigma,ijab}
 C^\sigma_{\mu i}C^\sigma_{\nu a}C^\sigma_{\kappa j}C^\sigma_{\tau b}
 T^{\sigma\sigma}_{ijab}
+2c\sum_{ijab}
 C^\alpha_{\mu i}C^\alpha_{\nu a}C^\beta_{\kappa j}C^\beta_{\tau b}
 T^{\alpha\beta}_{ijab}.
\]

This is the Eq. 35 counterpart of the validated literal Eq. 47 builder.
Neither part carries \(a_x\). The coefficient 2 on ab belongs here, not
in the OS correlation-energy sum. Symmetrize the AO tensor by the existing
eightfold averaging convention only at its ERI-contraction boundary.

Using unscaled T, the alpha density blocks are

\[
D_{ij}^{\prime\alpha}=-c\left[
 \tfrac12\sum_{kab}T^{\alpha\alpha}_{ikab}T^{\alpha\alpha}_{jkab}
+\sum_{kab}T^{\alpha\beta}_{ikab}T^{\alpha\beta}_{jkab}\right],
\]
\[
D_{ab}^{\prime\alpha}=c\left[
 \tfrac12\sum_{ij d}T^{\alpha\alpha}_{ijad}T^{\alpha\alpha}_{ijbd}
+\sum_{ij d}T^{\alpha\beta}_{ijad}T^{\alpha\beta}_{ijbd}\right].
\]

For beta, use bb in the same-spin terms and place the beta indices on
the second occupied/virtual ab slots:

\[
D_{ij}^{\prime\beta}=-c\left[
 \tfrac12\sum_{kab}T^{\beta\beta}_{ikab}T^{\beta\beta}_{jkab}
+\sum_{kab}T^{\alpha\beta}_{kiab}T^{\alpha\beta}_{kjab}\right],
\]
\[
D_{ab}^{\prime\beta}=c\left[
 \tfrac12\sum_{ij d}T^{\beta\beta}_{ijad}T^{\beta\beta}_{ijbd}
+\sum_{ij d}T^{\alpha\beta}_{ijda}T^{\alpha\beta}_{ijdb}\right].
\]

Both Dprime off-diagonal occupied/virtual blocks vanish. Transform each
spin with its own C. Require separate spin number conservation,
\(\operatorname{Tr}D^{\prime\sigma}=0\), and the stationary scalar

\[
\mathcal H(C,T;R)=\mathcal P(C,T;R)
+\sum_\sigma D^{\prime\sigma}(T):F^\sigma_{\rm MO}(C;R).
\]

At canonical amplitudes, test separately
\(\mathcal P=2cE_{\rm UPT2}\),
\(\sum_\sigma D^{\prime\sigma}:\epsilon^\sigma=-cE_{\rm UPT2}\),
\(\mathcal H=cE_{\rm UPT2}\), and amplitude stationarity.
Away from canonical orbitals, use the full oo/vv Fock blocks in this scalar
and its amplitude residual. Replacing them by freshly chosen diagonal
denominators is not an off-shell amplitude-response oracle.

Define G by differentiating all four coefficient factors of \(\mathcal P\)
at fixed T and fixed AO ERIs:

\[
\delta\mathcal P=\sum_{\sigma,pq}G^\sigma_{pq}U^\sigma_{pq},
\qquad \delta C^\sigma=C^\sigma U^\sigma.
\]

Retain aa/ab/bb contributions and all four differentiated slots in tests.
For a spin-preserving orbital rotation \(U_{ai}=X_{ai}\),
\(U_{ia}=-X_{ai}\), the pair RHS is
\(\ell^{\sigma,\rm pair}_{ai}=G^\sigma_{ai}-G^\sigma_{ia}\).
Do not append an extra compressed internal-amplitude bracket: this literal
derivative already contains occupied-coefficient variations. Reconcile it
with each same/opposite-spin term of Eq. 22 in an independent index oracle.

### 15.6 Physical response, adjoint RHS and coupled Z

Use a physical two-spin Fock derivative callback, with no implicit response
or occupancy factor:

\[
\mathscr K^\sigma[Q^\alpha,Q^\beta]
=J[Q^\alpha+Q^\beta]-a_xK[Q^\sigma]
+\sum_\tau f_{\rm XC}^{\sigma\tau}[Q^\tau].
\]

Self-adjointness is a **joint-spin** condition:
\(\sum_\sigma A^\sigma:\mathscr K^\sigma[B]
=\sum_\sigma B^\sigma:\mathscr K^\sigma[A]\).
Test alpha-only, beta-only and mixed trials; cross-spin exchange is zero,
but cross-spin Coulomb and XC generally are not.

For a physical UKS rotation,

\[
\delta P_X^\sigma=C_v^\sigma X^\sigma C_o^{\sigma T}
+C_o^\sigma X^{\sigma T}C_v^{\sigma T}.
\]

There is no closed-shell factor 2 on this trial density. By adjointness,

\[
\sum_\sigma D_{\rm AO}^{\prime\sigma}:\mathscr K^\sigma[\delta P_X]
=\sum_\sigma X^\sigma:
 2C_v^{\sigma T}\mathscr K^\sigma[D'_{\rm AO}]C_o^\sigma.
\]

Thus the proposed solver-normalized response RHS is
\(\ell^{\sigma,\rm response}=2C_v^{\sigma T}\mathscr K^\sigma[D']C_o^\sigma\),
including a factor 2 on the entire adjoint action, not only J/K. This is
a derivation from the chosen scalar/rotation convention, not a claim that
every printed R expression uses this normalization. The independent
stationary orbital FD must fix this mapping before driver work.

With \(\ell=\ell_{\rm pair}+\ell_{\rm response}\), define the residual
Jacobian by differentiating the co-moving \(F^\sigma_{ai}\):

\[
(AX)^\sigma_{ai}=(\epsilon_a^\sigma-\epsilon_i^\sigma)X^\sigma_{ai}
+[C_v^{\sigma T}\mathscr K^\sigma[\delta P_X]C_o^\sigma]_{ai}.
\]

The constrained scalar is
\(\mathcal L=\mathcal H+\sum_\sigma z^\sigma:F^\sigma_{vo}\),
so solve \(A^Tz=-\ell\). Use \(Az=-\ell\) only after testing the
joint-spin self-adjoint convention. Pack alpha then beta, each virtual-major
`a*nocc+i`; total dimension is \(v_\alpha o_\alpha+v_\beta o_\beta\).
Keep gap, J, same-spin K and all XC coupling blocks independently inspectable
in test products. Check both the scaled residual and rank/conditioning;
never silently return zero Z after a failed solve.

### 15.7 Relaxed density, overlap and separable ERIs

Use raw \(D^{\sigma}_{vo}=z^\sigma\), raw \(D^{\sigma}_{ov}=0\), with
Dprime in oo/vv. The symmetric adapter replaces vo/ov by
\(z^\sigma/2\), \(z^{\sigma T}/2\). For a symmetric AO Fock matrix this
reproduces exactly \(z^\sigma:F^\sigma_{vo}\); copying z into both raw
blocks would double it. Define
\(D_{\rm AO}=\sum_\sigma C^\sigma D^\sigma_{\rm sym}C^{\sigma T}\).

Use the paper's occupied/virtual metric gauge separately for each spin:
\(U_{oo}=-S_{oo}/2\), \(U_{vv}=-S_{vv}/2\),
\(U_{ov}=-S_{ov}-U_{vo}^T\). The literal pair blocks in Planck raw storage
are proposed as

\[
B^\sigma_{ij}=-\tfrac12G^\sigma_{ij},\quad
B^\sigma_{ab}=-\tfrac12G^\sigma_{ab},\quad
B^\sigma_{ia}=-G^\sigma_{ia},\quad B^\sigma_{ai}=0.
\]

Derive and test the accompanying nonpair blocks from the same scalar:

\[
W^\sigma_{ij}=B^\sigma_{ij}
-\tfrac12D^{\prime\sigma}_{ij}(\epsilon_i^\sigma+\epsilon_j^\sigma)
-[C_o^{\sigma T}\mathscr K^\sigma[D_{\rm AO}^\alpha,D_{\rm AO}^\beta]C_o^\sigma]_{ij},
\]
\[
W^\sigma_{ab}=B^\sigma_{ab}
-\tfrac12D^{\prime\sigma}_{ab}(\epsilon_a^\sigma+\epsilon_b^\sigma),
\quad W^\sigma_{ia}=B^\sigma_{ia},\quad
W^\sigma_{ai}=-\epsilon_i^\sigma z^\sigma_{ai}.
\]

The spin-specific response term is one physical K action in this notation,
equivalently half of the adjoint RHS operator defined above. These W
expressions are **acceptance targets**, not validated new routines. Test
oo, vv and ov metric directions independently, then symmetrize each W
before AO transformation and sum:
\(W_{\rm AO}=\sum_\sigma C^\sigma\operatorname{sym}(W^\sigma)C^{\sigma T}\).
Raw MO matrices cannot be added across different spin orbital bases.

The separable correction is unambiguously fixed by \(\sum_\sigma D^\sigma:F^\sigma\):

\[
\Gamma^{\rm sep}_{\mu\nu\kappa\tau}
=D_{\mu\nu}P_{\kappa\tau}
-a_x\sum_\sigma D^\sigma_{\mu\kappa}P^\sigma_{\nu\tau}.
\]

There is no extra 1/2 here and no opposite-spin exchange. Test Coulomb,
alpha exchange and beta exchange separately; apply eightfold averaging
only for the symmetric derivative-ERI consumer. In the closed-shell limit,
\(D^\alpha=D^\beta=D/2\) and \(P^\alpha=P^\beta=P/2\), this becomes
\(DP-a_xDP_{\rm exchange}/2\), precisely the corrected RKS contract.

### 15.8 Complete polarized XC-II: one scalar, four geometry channels

The target is

\[
Q_{\rm XC}(R)=\sum_\sigma D^\sigma_{\rm AO}:
 V^\sigma_{\rm XC}(P^\alpha_{\rm AO},P^\beta_{\rm AO};R),
\]

with both P and both D **AO coefficient matrices held fixed**. Their basis
functions and the atom-centered quadrature still move with R. This is not
the ordinary UKS XC energy gradient with P replaced by D.

A compact specification avoids missing mixed-spin GGA terms. Use the
implementation ordering
\(y=(\rho_\alpha,\rho_\beta,\gamma_{\alpha\alpha},
\gamma_{\alpha\beta},\gamma_{\beta\beta})\), explicitly different from
the ordering written in Eq. 6. Write \(u_\sigma=\nabla\rho_{P^\sigma}\),
\(v_\sigma=\nabla\rho_{D^\sigma}\), and

\[
s_D=\delta_Dy=(\rho_{D^\alpha},\rho_{D^\beta},
 2u_\alpha\cdot v_\alpha,
 u_\alpha\cdot v_\beta+u_\beta\cdot v_\alpha,
 2u_\beta\cdot v_\beta).
\]

For the **energy-per-volume** integrand \(f\), the scalar and its complete
geometry derivative are

\[
Q_{\rm XC}=\sum_g w_g f_y(y_g)\cdot s_{D,g},
\]
\[
Q_{\rm XC}^{(x)}=\sum_g w_g^{(x)} f_y\cdot s_D
+\sum_g w_g\left[s_D^T f_{yy}y^{(x)}+f_y\cdot s_D^{(x)}\right].
\]

Do not substitute libxc's per-particle `exc` for f. Use its correctly packed
potential/kernel outputs, including the single shared ab gradient invariant.
The derivative of the ab component of s_D has four terms, differentiating
both u and both v factors. AO spatial Hessians are needed to move the GGA
gradients. No third XC functional derivative is needed for this first
geometry derivative of Q.

Package the result as the same four non-overlapping channels used in RKS:

1. **P-side AO-center:** changes to y and the P factors of s_D, at fixed
   quadrature points and fixed D-side AO fields.
2. **D-side AO-center:** changes to the D factors of s_D, at fixed points
   and fixed P-side fields.
3. **Becke partition:** the weight-derivative term at fixed owner radial/
   angular quadrature, using the same partition implementation as UKS.
4. **Owner-point translation:** spatial changes of both P and D fields
   from motion of a point's owner atom, counted separately from AO centers.

For LDA, retain only the two density components of y and s_D. A polarized
LDA full-geometry oracle must pass before adding GGA. For GGA, check each
mixed-spin Hessian slot and each geometry channel before the complete scalar.
Do not append \(D^{(x)}:V_{\rm XC}\) (XC-I): live relaxed-matrix response
belongs to stationary cancellation, not this fixed-coefficient derivative.
Combined functionals still suppress duplicate correlation contributions.
Use identical density cutoffs, grid ownership and partition conventions in
the scalar and its analytic/FD evaluations; avoid oracle steps that cross
non-smooth screening thresholds without reporting it.

### 15.9 Typed boundaries and file-level scope

Prefer an unrestricted sibling `src/dft/udh_pt2_gradient.{h,cpp}` over adding
optional beta members to every restricted type. Proposed names, not existing
APIs, are:

- `UDHPT2AmplitudeDensity`: spin-space descriptors, unscaled aa/ab/bb views,
  c, and owned correction-only Dprime blocks.
- `UDHKSResponseOperator`: checked two-spin physical K action; single source
  of \(a_x\), immutable P/grid/functionals, explicit callback lifetimes.
- `UDHPairCoefficientGradient` and `UDHLagrangianRHS`: spin-resolved G,
  pair/response RHS and total in declared vo orientation.
- `UDHZVectorProducts`: coupled solution, residuals, and the exact snapshot
  and RHS metadata used to solve it.
- `UDHRelaxedDifferenceDensity`, `UDHOverlapDensity`: distinct raw/symmetric
  spin-MO objects and correctly summed AO consumers.
- `UDHTwoParticleDensity`: separable J/alpha-K/beta-K and aa/ab/bb pair
  channels, with explicit tensor symmetry state.
- `UDHCompleteXCIIGradient`: P-side, D-side, partition, translation, total.
- `UDHGradientDriverInputs/Contract`: all correction objects, one derivative
  bundle, spin-aware XC inputs and immutable converged UKS/PT2 snapshot.

Reuse the spin-independent AO derivative bundle and final atom-by-three
addition/frame mechanics. The current non-XC builder accepts restricted
types, so extract a small AO-only contraction consumer or provide a checked
unrestricted overload; do not manufacture fake restricted MO spaces to call
it. Do not force UKS through `RMP2Result`, `RMP2Lagrangian`, or an HF full-
minus-reference bridge. Existing UMP2 gradient intermediates can supply
separate algebra checks, but do not define a DH stationary functional.

The driver should create polarized functionals, run UKS once, retain the
UMP2 result with `with_t2=true`, and pass matching alpha/beta coefficients,
energies, occupations and active maps forward. Require the all-active maps
for this first release. No re-SCF, hidden orbital recanonicalization, stale
grid capture, or overwritten KS density is permitted between these steps.
Resolve physical AO/standard/requested coordinate frames once and rotate
the summed UKS+PT2 gradient once. Errors propagate through checked returns.

### 15.10 Small implementation steps and acceptance gates

U0 code and fixtures are implemented; its C++ acceptance is pending the
user-managed rebuild (see Section 15.12). U1–U11 remain pending.
The user continues to own builds. Each step
should add a bounded primitive and independent tests; wait for the user's
rebuild before running the relevant executable. Do not start background
builds. New numerical ledgers belong in standalone tests/logs, not production
environment switches or diagnostic driver branches.

| Step | Bounded implementation | Required evidence before advancing |
|---|---|---|
| U0 | Freeze the UKS spin/energy/storage contract and test fixtures; keep workflow rejection intact | Reconstruct aa/ab/bb energies from `UMP2Result`; validate shapes, antisymmetry, spin swap, scaling and closed-shell Eq. 36 reduction; reject unsupported options explicitly |
| U1 | Dprime and common stationary scalar | Literal-index density checks, separate spin traces, pair=2cE and Dprime:F=-cE, independent amplitude-direction stationarity; unequal alpha/beta dimensions and valid zero-size blocks |
| U2 | Checked physical two-spin KS response primitive | Separate J, alpha-K, beta-K and all four XC blocks against density FD; joint adjointness; reuse existing polarized XC tests; callback failures remain errors |
| U3 | Literal Eq. 35 four-coefficient G and Eq. 22 RHS | All four coefficient slots and aa/ab/bb channels checked by basis-direction FD; independent same/opposite-spin external/internal index sums; include Dprime response exactly once |
| U4 | Coupled Eq. 27 action and Z solve | Every response matrix column from an independent co-moving F_ai orbital FD, transpose/sign proof, alpha-only/beta-only/mixed trials, rank/error handling and residual; dense small-system solve as oracle |
| U5 | Eq. 28 raw/symmetric relaxed spin densities | Raw multiplier equals symmetric AO contraction for independent symmetric operators; no factor-two duplication; spin traces and closed-shell reduction |
| U6 | Eq. 29–32 literal pair and nonpair W blocks | Fixed-AO/fixed-amplitude oo, vv, ov metric FDs with both spin connections generated by the same AO S perturbation; separate Z constrained-Fock mapping; raw-to-symmetric equivalence |
| U7 | Eq. 34/35 ERI tensors and non-XC Eq. 33 consumer | Fixed-contract h/S/ERI scalar FDs, J and each spin-K isolation, all pair spin channels, hybrid coefficient sweep, symmetry averaging and rigid translation |
| U8a | Polarized LDA complete XC-II | P-side/D-side/weight/point tests and full fixed-AO-coefficient geometry FD, including one-spin and mixed-spin D directions |
| U8b | Polarized GGA complete XC-II | All rho/sigma kernel slots and cross-gradient factors; Hessian-bearing point motion; same four-channel ledger and complete geometry FD; closed-shell reduction |
| U9 | Common stationary-transfer integration test | Noncanonical/off-shell amplitude-response solve and multiplier cancellation; basis-direction orbital stationarity; analytic fixed-AO and Eq. 33 forms agree channel-by-channel in a declared common metric gauge; Z and non-Z residuals shown separately |
| U10 | Driver handoff and assembly behind test-only access | Same UKS/PT2 snapshot through all builders; UKS+correction composition, c=0 limit, ordinary result shape/frame/error handling; all RKS tests remain passing |
| U11 | Molecular acceptance and narrow production enablement | Full-coordinate total-energy FD on open-shell nonsymmetric fixtures at two steps, state continuity, closed-shell-limit RKS comparison, rejection tests; only then relax the UKS global-DH Gradient guard |

U2 and U8 do not depend on completed pair algebra, but this is a dependency
observation, not permission to run competing builds. Keep milestone evidence
in this section when each gate passes; do not spawn another series of scope
documents.

### 15.11 Validation matrix and quantitative policy

Use independent Planck energy/scalar finite differences, explicit-index
reference loops and spin-orbital algebra. **Do not use PySCF double-hybrid
analytic gradients as an oracle.** Agreement between a producer and a
consumer sharing the same contraction loops is not independent evidence.

Required fixtures and limits:

- Synthetic unequal spin dimensions: aa, ab and bb sectors individually,
  sign/permutation changes, independent alpha/beta rotations, c=0 and linear
  c scaling. Spin-label swap must preserve the energy and final gradient.
- Closed-shell UKS snapshot: set identical alpha/beta orbitals, energies and
  half densities from a restricted reference; Eq. 36 amplitudes must recover
  the existing RKS energy, AO D/W/Gamma contractions and gradient. Compare
  mapped contractions, not differently normalized intermediate arrays.
  For the scalar/packing convention above, expect equal-spin
  \(\ell^\alpha=\ell^\beta=\ell^{\rm RKS}/2\) and
  \(z^\alpha=z^\beta=z^{\rm RKS}/2\) on a nonsingular spin-symmetric branch;
  explicitly prove the charge-sector Hessian reduction.
- A genuine unequal-occupation radical, initially distorted water cation
  (doublet), exercises the unequal alpha/beta spaces cheaply. Rotate and
  distort it so symmetry-zero Cartesian components cannot hide errors.
- A genuinely nonsymmetric open-shell geometry, initially distorted HO2
  (doublet), exercises non-equivalent atomic environments. Fix and record
  actual coordinates, stable occupations and convergence settings in U0;
  these are proposed fixtures, not computations already performed.
- A higher-spin fixture, such as distorted triplet water, exercises another
  spin population. A one-electron/zero-beta test requires identically zero
  PT2 correction and valid empty-sector handling; this does not imply that
  the underlying semilocal UKS energy is free of self-interaction.
- Retain the current all-coordinate RKS water/H2O2 production checks without
  relaxing their 5e-8 Ha/Bohr tolerance. Add one modest larger Cartesian
  basis after the minimal-basis UKS cases pass, before claiming basis breadth.

For small well-scaled algebra tests, target roughly 1e-12 absolute/relative
agreement; for independently differenced response/geometry primitives,
target 1e-8 or better with a step-convergence study. Require a documented
scaled Z residual at most 1e-10 on the well-conditioned acceptance fixtures.
These are proposed test targets, not assertions of measured UKS accuracy.

For the first molecular acceptance use the existing steps 1e-4 and 2e-4 Bohr
and a max-component target of 5e-8 Ha/Bohr on every Cartesian component.
If open-shell convergence or grid noise prevents that target, first tighten
SCF/integral settings and study the step/grid dependence; do not silently
raise tolerance. Record convergence, spin populations, available spin
contamination diagnostics and occupied-subspace overlaps across displacements
so root switching is not mistaken for a gradient defect. Track orbital
phases/subspaces only in diagnostic comparisons; total-energy FD itself
requires the same physical SCF branch, not a forced coefficient gauge.
Preserve logs, actual rendered displacements, energies and gradients.

Finally require translation and rotation covariance checks (allowing for
finite quadrature orientation error), independent UKS reference-gradient
FD, correction scaling, and full total-energy FD. A passing sum does not
waive the U9 stationary-transfer identity, and passing U9 does not waive
live molecular FD. Production remains restricted until both pass.

### 15.12 U0 implementation and validation record

Implemented a detached `build_udh_pt2_energy_contract` in
`src/dft/udh_pt2_gradient.{h,cpp}`. The three new types are
`UDHPT2Scope`, `UDHPT2DirectIntegrals` and `UDHPT2EnergyContract`.
The contract accepts the existing `UMP2Result` directly; it returns an
owned, correction-only scalar ledger, without retaining borrowed inputs.
Direct MO integrals are explicitly **repacked into amplitude order**, not
passed in the MP2 backend's ovov order. It checks:

- All-active spin spaces, occupied-one populations, coefficient/overlap
  normalization, convergence and canonical-result metadata.
- Expected aa/ab/bb extents, finite arrays, same-spin amplitude
  antisymmetry and direct-integral pair symmetry, and canonical amplitude
  residuals against the supplied direct integrals and orbital energies.
- Nonnegative or near-zero denominators are rejected (cutoff -1e-12 Ha).
  Algebra/energy consistency uses 1e-10 absolute-plus-relative tolerance;
  the independent small-fixture test comparisons use 1e-12.
- Independent aa/ab/bb reconstruction, cached SS/OS/total energy agreement,
  and exactly one finite scalar coefficient on the output correction.
- Frozen-space requests, RI, shifted PT2, omitted amplitudes, non-UKS,
  non-Cartesian, range-separated, solvent, meta-GGA and independent
  spin-scaling scope declarations are rejected.
- Zero-extent spin channels are valid, including zero alpha occupancy after
  spin swap; missing nonzero-extent storage is rejected even when c=0.

This validator does not compute AO-to-MO integrals, prove that caller-supplied
integrals came from the supplied C, or verify the live KS F_ai residual.
Those integration/response checks remain explicit later gates. U0 neither
changes the UMP2 backend nor calls the new code from the production driver.
Only the standalone test target links this new translation unit at present.

`tests/udh_pt2_energy_contract.cpp` builds synthetic canonical `UMP2Result`
objects. Its independent oracle sums **all occupied and virtual spin
orbitals**, including spin selection and antisymmetrization, rather than
sharing the production aa/ab/bb prefactor loops. Tests cover unequal spin
dimensions, each channel, alpha/beta swap with ab-slot transport, orbital
phase changes, c=0/linear scaling, one-electron zero sectors, nonidentity
AO overlap, malformed inputs and unsupported requests. Eq. 36 amplitudes
and the full closed-shell spatial MP2 energy are checked independently.
These synthetic integral fixtures are algebra tests, not molecular AO-to-MO
validation.

The new CMake/CTest target is `planck-udh-pt2-energy-contract`. No compiler
or build has been started. **The new C++ suite has not yet been executed;
U0 is not marked fully validated.** After the user's rebuild, run:

```sh
build/planck-udh-pt2-energy-contract > /private/tmp/udh-u0-contract.log 2>&1
```

Five normal energy inputs are fixed under
`tests/inputs/exploratory/dh_gradient/uks_u0/`:

| Fixture stem | Charge / multiplicity | n_alpha / n_beta | N_AO | v_alpha / v_beta |
|---|---|---|---|---|
| water_cation_asymmetric | +1 / 2 | 5 / 4 | 7 | 2 / 3 |
| ho2_asymmetric | 0 / 2 | 9 / 8 | 11 | 2 / 3 |
| water_triplet_asymmetric | 0 / 3 | 6 / 4 | 7 | 1 / 3 |
| h2plus_zero_beta | +1 / 2 | 1 / 0 | 2 | 1 / 2 |
| h2o2_cation_c1 | +1 / 2 | 9 / 8 | 12 | 3 / 4 |

The three-atom fixtures have inequivalent bond environments and all-coordinate
displacements, but are not claimed to have C1 point-group symmetry: any
three atoms lie in a plane. The additional H2O2-cation fixture has nonzero
scalar triple product and no nonidentity species-preserving distance
automorphism, giving a genuinely C1 oracle. It is not a claim of a stable
electronic branch across geometry displacements; U11 must test that.

All inputs use B2PLYP/STO-3G, Cartesian AOs, ultrafine grid, symmetry off,
and 1e-11 energy/density SCF thresholds. Ordinary DIIS initially failed to
converge HO2 and H2O2+ within 200 iterations. HO2 now uses a three-cycle
UKS SOSCF window starting at iteration 5; H2O2+ uses a longer window.
These are fixture settings, not modifications to the SCF implementation.

`tests/udh_u0_molecular_audit.py` runs the existing energy executable and
then generates gradient inputs that must fail with the existing UKS DH
workflow rejection. It checks normal-output single PT2 scaling and the
one-electron zero correction; this is **not** an independent molecular
aa/ab/bb audit or a gradient FD. All inputs, JSON and logs are retained.
The driver guards, public gradient path and normal output remain unchanged.

The three fast `tests/test_udh_u0_fixtures.py` tests pass: fixture populations/
settings/rendering, actual C1 geometry, and success/failure audit parsing.
They are also registered as `planck-udh-u0-fixtures` in CTest and can be run
without a rebuild:

```sh
python3 -m unittest discover -s tests -p 'test_udh_u0_fixtures.py' -v
python3 tests/udh_u0_molecular_audit.py > /private/tmp/udh-u0-molecular-audit.log 2>&1
```

The existing (unchanged) production executable was used for fixture smoke
checks during U0 implementation. Four fixtures converge and their gradient
requests hit the expected scope rejection:

| Fixture | Total energy (Ha) | Bare PT2 (Ha, normal-log precision) | Result |
|---|---:|---:|---|
| water_cation_asymmetric | -74.83611796895414 | -0.0304538294 | energy/scaling/rejection pass |
| ho2_asymmetric | -148.64947167660466 | -0.0915123065 | energy/scaling/rejection pass with SOSCF |
| water_triplet_asymmetric | -74.77385491577844 | -0.0211474313 | energy/scaling/rejection pass |
| h2plus_zero_beta | -0.5872125980237026 | 0.0 | exact zero correction; rejection pass |
| h2o2_cation_c1 | not converged | not evaluated | candidate only; not accepted |

H2O2+ still fails the 1e-11 SCF convergence request within 200 iterations,
including the longer SOSCF trial. The final reported density change is
approximately 1.38e-10 and the DIIS error approximately 9.91e-9; neither a
converged energy nor a DH gradient is claimed. Its geometry passes the C1
test, but this candidate needs SCF/fixture stabilization before U11. No
tolerance was relaxed and no SCF implementation change was made. The overall
molecular smoke command correctly exits 1 until all five fixtures pass.

Latest smoke log: `/private/tmp/udh-u0-molecular-final-audit.log`.
Detailed inputs/results/logs:
`/var/folders/b6/8bkpkc5j2hjgttz3q3yv16gr0000gn/T/udh-u0-molecular-_sjkzzi_/`.
The initial DIIS and short-SOSCF attempts remain in
`/private/tmp/udh-u0-molecular-audit.log` and
`/private/tmp/udh-u0-molecular-soscf-audit.log`.
These smoke results do not execute the new C++ contract; that acceptance
remains pending the rebuild independently of the fixture convergence issue.
