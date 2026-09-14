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
[UKS extension scope](#15-uks-extension-scope-and-acceptance-plan),
[Hessian swap evidence](#16-hessian-free-z-vector-swap-probe),
[scalability plan](#17-recommended-optimization-sequence).

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
- RKS global-DH geometry optimization, frequency and combined opt/frequency
  workflows share the full gradient and pass the water/C1 H2O2 molecular
  workflow audits (2026-09-13; see Section 18). Frequencies use
  a semi-numerical Hessian, not an analytic second derivative. DH response
  and imaginary-mode following remain excluded. The DFT driver rejects
  spherical basis functions.
- The current XC-II implementation supports LDA-like and GGA-like functionals,
  not a general meta-GGA derivative. No general frozen-core/active-space
  response, spin-component-scaled DH, or near-degeneracy validation is claimed.
- There is no HF-MP2 full-minus-reference subtraction and no
  `RMP2Lagrangian` bridge in this production DH assembly.
- The literal RHS and literal pair-overlap convention are unconditional.
  O3 now routes normal production through the typed matrix-free Eq. 27
  GMRES backend, with no DH flags required. The user rebuild passes the
  water/C1 H2O2 all-coordinate molecular FD and typed contraction checks
  (2026-09-14). The corrected standalone failure-path test also passes
  after its test-only rebuild (Section 17).
  Dense QR remains an explicit reference backend for the Section 16 probe.

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

`solve_dh_zvector` flattens \((a,i)\) as \(a n_{\rm occ}+i\) and solves
\(AZ=-L\) with checked restarted, right-preconditioned GMRES (O3).
The defaults are tolerance \(10^{-12}\), restart 24, and 256 iterations.
Only the preconditioner uses \(1/\max(|\epsilon_a-\epsilon_i|,10^{-8})\);
neither the physical Hessian nor PT2 denominators are shifted. A fresh
unpreconditioned action must satisfy
\(\|AZ+L\|_\infty\le10^{-12}\max(1,\|L\|_\infty)\).
An explicit `DenseReference` backend builds unit columns and uses
column-pivoted QR for tests/probes; it is never a convergence fallback.
The assembled contract must still agree with the solver RHS within 1e-12
and satisfy its additional absolute Z residual check within 1e-9.
Post-O3 molecular, typed-contract and standalone solver checks pass;
the acceptance evidence is recorded in Section 17.

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
implementation materializes dense AO/MO tensors:
storage includes O(3N_atom N^4) ERI derivatives. O3 removes the normal-path
O((ov)^2) orbital Hessian, replacing it with O(k ov + k^2) Krylov matrix
storage at bounded restart k, plus the action and O2 XC-cache workspace.
It is not a density-fitted, sparse or large-system scalability implementation.

Numerical dimension, finite-value and Z-residual checks remain production
safety checks. Historical debug ledgers, selfcheck FD loops, snapshots and
mathematical-convention gates have been removed. The separately requested
Hessian-swap diagnostic in Section 16 is inactive unless explicitly selected.

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

U0 code, invariant tests and all five molecular smoke fixtures pass the latest
user-managed rebuild (2026-09-13, `/private/tmp/udh-u0-molecular-retest-sep13.log`).
The failed SCF runs in Sections 15.12–15.14 are historical, not the latest
U0 status. U1 and U2 passed their rebuilt primitive acceptance tests on
2026-09-13 (Sections 15.16 and 15.15; logs `/private/tmp/udh-u1-stationary.log`
and `/private/tmp/udh-u2-response.log`). U3 passed its rebuilt primitive
acceptance on 2026-09-14 (Section 15.17).
U4 passes its rebuilt detached primitive suite on 2026-09-14 after correcting
the test-only rectangular/all-active fixture boundary (Section 15.18).
U5–U11 remain pending;
U2 does not depend on U1.
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

The new CMake/CTest target is `planck-udh-pt2-energy-contract`. The user
rebuilt it, and it now passes with exit code 0. No agent-initiated build
was started. U0 is not marked fully validated because the C1 molecular
fixture remains unconverged. Reproduce the successful C++ run with:

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
UKS SOSCF window starting at iteration 5; the failed H2O2+ trials used a
longer window. H2O2+ is now configured for the new 0.3-Ha UKS SCF level-shift
path instead (Section 15.13); its post-rebuild retest still fails to converge.

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
Those initial smoke results did not execute the new C++ contract. Its
subsequent successful run is independent of the remaining fixture
convergence issue; see the post-rebuild results below.

### 15.13 UKS SCF level-shift support and retest

The input parser already accepted `%begin_scf` `level_shift`, but the UKS
driver did not consume it. The UHF implementation did. UKS support has now
been rebuilt by the user and tested. The projector/handoff unit suite passes,
but the 0.3-Ha shift does not resolve H2O2+ convergence at the original
settings. Detailed results are in Section 15.14.

`src/dft/uks_level_shift.{h,cpp}` provides the occupancy-one AO-metric
projector and an unshifted eigenpair validator. For each spin the iteration
matrix is

\[
F^\sigma_{\rm iter}=F^\sigma_{\rm physical}
+\lambda(S-SP^\sigma S),\qquad \lambda\ge0.
\]

The physical Fock, KS energy expression, XC evaluation and PT2 Hamiltonian
are unchanged. The projector is added before DIIS; errors use the physical
commutator, which the projector leaves unchanged. Negative/nonfinite shifts
are rejected. `level_shift 0` retains the previous UKS numerical path.
This SCF keyword is distinct from `mp2_level_shift`; shifted PT2 denominators
remain excluded by the U0 contract.

With a positive shift, SOSCF is disabled for the entire run and an explicit
warning is emitted if it was also requested. This matches the UHF policy
of not combining a shifted iteration with an unshifted SOSCF Hessian. It
does not disable hybrid exchange or polarized XC response.

Convergence of the shifted iteration is only an intermediate state. The
driver removes the shift, clears both DIIS histories, rebuilds physical
Fock matrices, and performs unshifted, non-extrapolated diagonalizations
until the requested convergence thresholds are met again. For this new
positive-shift path it checks both spin-density changes independently and
the physical commutator residual; cancellation in the total spin density
cannot produce a false handoff. Polishing uses the same iteration budget;
exhaustion is a failure, never a return of shifted orbital energies.

Before post-KS work, both spin channels must satisfy the unshifted equations
\(FC=SC\epsilon\) and \(C^TSC=I\) (scaled eigenpair and metric check at
1e-9). Simply subtracting lambda from shifted virtual eigenvalues is not
used: away from an exact fixed point the coefficients also differ. The
normal log records shift application, removal and the verified unshifted
handoff. No production debug switch is added.

The new standalone `planck-uks-level-shift` target tests the projector on a
nonidentity AO overlap, unequal spin occupations, occupied/virtual shifts,
unchanged commutator, zero shift, empty/full spin densities, malformed
inputs, rejection of shifted final spectra, and unshifted PT2 denominators.
An off-shell test explicitly rejects eigenvalue subtraction with unchanged
shifted coefficients. These C++ tests now pass with exit code 0.

H2O2+ now has `level_shift 0.3` and no SOSCF window, with the original
geometry, 200-iteration cap and 1e-11 thresholds unchanged. The molecular
audit checks that unshifted-handoff messages occur before the PT2 energy
summary, so a binary that silently ignores the keyword cannot pass.
Four fast U0 Python tests pass, including missing/reordered/wrong-shift
handoff rejection. Molecular success is established for the shifted
one-electron H2+ handoff, not for the correlated open-shell fixtures.

Commands used after rebuilding `planck-dft`, `planck-uks-level-shift` and
`planck-udh-pt2-energy-contract`:

```sh
build/planck-uks-level-shift > /private/tmp/uks-level-shift-unit.log 2>&1
build/planck-udh-pt2-energy-contract > /private/tmp/udh-u0-contract.log 2>&1
python3 tests/udh_u0_molecular_audit.py \
  --fixture h2o2_cation_c1_b2plyp_sto3g.hfinp \
  > /private/tmp/uks-level-shift-h2o2-retest.log 2>&1
```

The full U0 fixture set, shifted/unshifted comparisons and existing RKS
validation checks were also rerun as recorded below. UKS DH gradients remain
disabled throughout.

### 15.14 Post-rebuild validation: passes and unresolved failures

Completed 2026-09-12 with the user-rebuilt binaries; no agent build was
started. The following tests pass:

- `planck-uks-level-shift` (projector, physical eigenpair and denominator
  invariants): `/private/tmp/uks-level-shift-unit.log`.
- `planck-udh-pt2-energy-contract` (U0 spin-orbital, storage and scope
  invariants): `/private/tmp/udh-u0-contract.log`.
- `planck-dh-pt2-amplitude-density`:
  `/private/tmp/dh-amplitude-post-uks-shift.log`.
- `planck-dft-coulomb-response`:
  `/private/tmp/dh-coulomb-post-uks-shift.log`.
- Six fast Python tests (two existing runner tests plus four U0 fixture/
  audit-parser tests): `/private/tmp/dh-python-post-uks-shift.log`.

Both all-coordinate RKS B2PLYP/STO-3G production FD checks pass unchanged
at steps 1e-4 and 2e-4 Bohr, with the same stored results as before:

| Molecule | max error at 1e-4 | max error at 2e-4 | units |
|---|---:|---:|---|
| Water | 1.44657175e-8 | 1.69526181e-8 | Ha/Bohr |
| C1 H2O2 (neutral RKS) | 9.03449667e-9 | 1.34265307e-8 | Ha/Bohr |

Logs: `/private/tmp/dh-water-post-uks-shift-fd.log` and
`/private/tmp/dh-h2o2-post-uks-shift-fd.log`. Detailed outputs are under
`/var/folders/b6/8bkpkc5j2hjgttz3q3yv16gr0000gn/T/` in
`dh-production-cartesian-fd-2qdc0pfp/` and
`dh-production-cartesian-fd-ccdqipoi/`, respectively.

UKS molecular results are deliberately reported separately:

- H2+ with shift 0.3 converges in three iterations, explicitly removes the
  shift and verifies the physical canonical handoff before PT2. Its total
  energy is -0.5872125980237026 Ha, identical at stored precision to the
  two-iteration unshifted run, and its PT2 correction is exactly zero.
- H2O2+ with shift 0.3 still fails within 200 iterations at the unchanged
  1e-11 thresholds. At iteration 200 its per-spin density RMS/max changes
  are 9.108e-7 / 5.997e-6 and physical commutator RMS is 5.816e-5. It has
  not reached shift removal, so this is a shifted-iteration convergence
  failure, not a PT2 or unshifted-spectrum failure.
- Additional shifted water-cation and triplet-water checks also fail by
  iteration 200, although both unshifted fixtures converge. Their final
  physical commutator RMS values are 2.022e-10 and 4.123e-10, respectively,
  above the requested 1e-11 threshold. No unconverged PT2 energies are
  reported. A correlated same-branch shift-invariance check is consequently
  still outstanding; the one-electron success does not establish it.
- The complete U0 smoke run remains four passes out of five; the other four
  inputs retain their original unshifted settings (including HO2 SOSCF).
  UKS DH gradient rejection passes for these converged fixtures.

Targeted log: `/private/tmp/uks-level-shift-h2o2-retest.log`, with details in
`/var/folders/b6/8bkpkc5j2hjgttz3q3yv16gr0000gn/T/udh-u0-molecular-7z1obnw9/`.
Full smoke log: `/private/tmp/udh-u0-post-level-shift-full.log`, with details
in `.../udh-u0-molecular-kbkwrto0/`. Shift/no-shift comparison inputs and
outputs: `/private/tmp/uks-level-shift-energy.6Elye8/`.

The formerly unlinked `planck-dh-eq41-xc-response` now links and runs, but
exits 1. Its test contained a definite stale composition assertion:
it equated the full Eq. 41 XC channel with the physical callback, whereas
the documented closed-shell adjoint is four times that callback. The test
source now checks that factor and also compares against four times an
independent potential FD, with separate error logging. No production XC
factor was changed and no primitive FD tolerance was relaxed. The corrected
test needs a target-only rebuild and rerun before claiming it passes.
Original failure log: `/private/tmp/dh-eq41-xc-response-rebuilt.log`.

Remaining work: establish reliable correlated UKS SCF convergence/handoff,
complete the C1 fixture acceptance, and execute the corrected Eq. 41 test.
These failures do not invalidate the unchanged measured RKS molecular FD
results, nor do the RKS passes establish unrestricted derivative support.

### 15.15 U2: checked physical two-spin KS response

Implemented in `src/dft/udh_ks_response.h/.cpp` as a detached AO primitive,
not wired into the UKS DH gradient driver. `UDHSpinMatrices` contains two
symmetric nao×nao matrices, one per spin; a zero spin trial is a zero matrix
of that shape, not a missing/zero-dimensional AO matrix. Indefinite trial
densities are legal. Occupation counts, amplitudes, PT2 scaling, orbital
energy differences and Z-vector signs do not enter this primitive.

`make_udh_ks_response_operator` owns checked raw-J, raw-K and physical
polarized-XC callbacks. `apply_channels` returns the following ledger:

| Field | Definition |
|---|---|
| `coulomb` | J[Qalpha+Qbeta], shared by both output spins |
| `exchange.alpha`, `.beta` | -a_x K[Qalpha], -a_x K[Qbeta]; no cross-spin exchange |
| `xc_from_alpha.alpha`, `.beta` | f_aa[Qalpha], f_ba[Qalpha] |
| `xc_from_beta.alpha`, `.beta` | f_ab[Qbeta], f_bb[Qbeta] |
| `total.alpha`, `.beta` | Each spin's J + weighted K + both XC inputs |

There is no RKS occupancy factor, PT2 coefficient, or adjoint factor 2/4 in
this physical response. `apply` returns the two totals from the same checked
assembly. The four-XC-block ledger uses two spin-only calls to the existing
polarized Hessian-vector primitive; it does not assume that the two cross
output matrices are equal. Self-adjointness is tested in the joint-spin
Frobenius pairing of Section 15.6.

`make_udh_direct_ks_response_operator` binds raw J/K to Planck's memory-direct
Coulomb-kernel builders and XC to
`compute_analytic_xc_hessian_vector_product_polarized`. It requires matching
polarized global LDA/GGA functionals, finite AO/grid/ground-density data,
consistent matrix dimensions, and available VXC/FXC support. Ground spin
densities are copied. Basis/shell-pair, AO/grid and functional objects are
borrowed: they must remain alive and unchanged until the operator is discarded.
Rebuild the operator when geometry, grid, density or functional changes.
No integral-symmetry reduction is imposed on arbitrary response densities.
No solvent, range-separated, orbital-Hessian or UKS gradient support is added.

All callback errors, exceptions, nonfinite/asymmetric matrices and wrong
shapes remain errors with a channel label. No failed action is replaced by
zero. Exact-exchange coefficient zero permits an absent K callback and skips
K evaluation, but does not suppress J/XC error checks.

The new standalone target `planck-udh-ks-response` contains:

- Explicit independent channel maps for signs, spin orientation and absence
  of closed-shell prefactors; missing/invalid callbacks, exceptions, NaNs,
  wrong shapes, zero-spin trials and callback-lifetime checks.
- A nonsymmetric water-cation/STO-3G fixed-geometry fixture with core-orbital
  spin densities Tr(Palpha S)=5 and Tr(Pbeta S)=4. No converged-SCF or DH
  amplitude stationarity is assumed or needed for this density derivative.
- Independent full-index AO-ERI J/K contractions and density FDs of ordinary
  first-derivative XC potentials, with alpha-only, beta-only and mixed trials.
  It logs J, alpha-K, beta-K and all four XC block errors separately.
- LDA (X+VWN), GGA (PBE X+C), and combined B2PLYP coverage; three density FD
  steps (1e-3, 3e-4, 1e-4), joint adjointness, linearity, spin swapping,
  ground-density snapshot immutability and physical closed-shell RKS reduction.

Acceptance thresholds are 1e-8 for J/K density FDs, 2e-7 for XC/total FDs,
and 1e-10 for joint adjointness and the RKS reduction. They are prospective
thresholds, not measured results: **the new C++ target has not been rebuilt
or run yet**. Five fast U2 API/routing checks and the four existing U0 fixture
checks pass. The existing polarized production-HVP, GGA FXC-ordering and
GGA Hessian-selfcheck executables also exit successfully; these are baseline
checks of reused code, not validation of the new binding.

After the user rebuilds `planck-udh-ks-response`, run:

```sh
build/planck-udh-ks-response > /private/tmp/udh-u2-response.log 2>&1
```

Baseline logs: `/private/tmp/udh-u2-polarized-baseline.log`,
`/private/tmp/udh-u2-fxc-ordering-baseline.log`,
`/private/tmp/udh-u2-hessian-selfcheck-baseline.log`; fast checks:
`/private/tmp/udh-u2-source-tests.log`. The UKS DH derivative workflow guard
remains unchanged. U4 will separately establish orbital packing, occupation
factors, residual conventions and the coupled Z Hessian.

### 15.16 U1: Dprime and the common stationary scalar

Implemented in `src/dft/udh_pt2_gradient.h/.cpp`, alongside the U0 boundary.
This is a detached algebraic primitive, not UKS gradient enablement. It uses
the unscaled aa/ab/bb amplitudes and the four Dprime blocks of Section 15.5
literally: same-spin contractions have weight c/2 and opposite-spin
contractions weight c. Beta ab contributions use the second occupied and
virtual slots. No reference density, exact-exchange coefficient, closed-shell
occupation factor or extra PT2 scale is included.

The public interfaces separate off-shell evaluation from canonical validation:

| Interface | Contract |
|---|---|
| `UDHPT2Amplitudes` | Owned unscaled aa/ab/bb arrays in [i,j,a,b] order with explicit spin dimensions; no cached energy or denominators |
| `build_udh_pt2_dprime` | Finite, shape-checked, antisymmetry-checked amplitudes to owned alpha/beta MO matrices; only oo/vv are nonzero |
| `transform_udh_pt2_dprime_to_ao` | Separate C-alpha and C-beta transforms, with checked all-active C^T S C=I; preserves each spin's trace and operator contractions |
| `evaluate_udh_pt2_stationary_scalar` | Off-shell H=P+Dprime-alpha:F-alpha+Dprime-beta:F-beta, using the supplied full symmetric MO Fock matrices |
| `build_udh_pt2_stationary_contract` | Canonical U0 validation followed by the same scalar at F=diag(epsilon); checks each pair spin channel and the summed stationary identities |

The scalar result retains `pair_aa=c*Taa:gaa`, `pair_ab=2c*Tab:gab`,
`pair_bb=c*Tbb:gbb`, their sum, the two Dprime:F spin contributions, their
sum, the total, and the Dprime matrices from that evaluation. Same-spin g
is direct, not antisymmetrized. It is checked for pair symmetry. Nothing is
rediagonalized and no amplitude is recomputed inside the off-shell evaluator.
It accepts nonstationary T and noncanonical F: H=cE and amplitude stationarity
are asserted only at a canonical validated snapshot, not for arbitrary input.

Raw algebra permits unequal spin dimensions and genuinely zero-extent spin
sectors, including an empty occupied or virtual space. Nonzero-extent arrays
may not be omitted, even at c=0. This does **not** broaden U0's canonical
all-active scope or its current virtual-space restrictions. The AO adapter
requires full square spin MO spaces over the same nonempty AO basis.
Malformed dimensions, nonfinite data, invalid same-spin antisymmetry,
asymmetric Fock matrices and nonfinite contractions are returned as errors.

The independent numerical target is `planck-udh-pt2-stationary`. It expands
the three storage sectors into the full antisymmetric spin-orbital tensor,
including all four mixed-spin occupied/virtual placements. Besides the
spin-orbital -1/2 TT and +1/2 TT density oracle, it evaluates

\[
\mathcal H=\frac c2\sum_{ijab}T_{ijab}V_{ijab}
-\frac c4\sum_{ijab}T_{ijab}\mathcal A_F(T)_{ijab},
\]
\[
\mathcal A_F(T)_{ijab}=\sum_k(F_{ik}T_{kjab}+F_{jk}T_{ikab})
-\sum_d(F_{ad}T_{ijdb}+F_{bd}T_{ijad}).
\]

Thus the independent amplitude-direction derivative is
`c/2 deltaT:(V-A_F(T))`. This oracle never constructs Dprime. Central
differences at 1e-3 and 1e-4 test every independent same-spin antisymmetric
and direct ab amplitude basis direction, both at canonical stationarity and
with off-shell amplitudes and noncanonical Fock matrices. Every symmetric
spin MO Fock direction is also tested: the derivative is Dprime_pp on the
diagonal, 2 Dprime_pq off diagonal, and zero in ov/vo. This exposes missing
off-diagonal Fock contractions without using the implementation as its own
reference.

Other checks cover isolated aa/ab/bb sectors; separate spin traces; spin
swapping; zero and unequal spaces; scales 0, 0.27, 0.54, 1 and -0.27;
pair=2cE, Dprime:F=-cE and H=cE; the spatial closed-shell total-density
reduction; and literal AO transforms with a nonorthogonal metric and distinct
spin coefficients. Algebraic tolerance is 2e-12 (scaled), and amplitude/Fock
FD tolerance is 2e-10 (scaled). These are acceptance thresholds, not measured
errors until the user rebuilds and the executable passes.

The four U1 boundary guards, five U2 boundary guards and four U0 fixture
checks pass (13 fast checks). Logs are `/private/tmp/udh-u1-source-tests.log`
and `/private/tmp/udh-u1-fixture-tests.log`. The existing U0 executable also
exits successfully (`/private/tmp/udh-u1-u0-baseline.log`), but that binary
predates these source edits and is only a baseline, not a rebuilt regression.

No C++ rebuild has been started for U1. After rebuilding the new target
and the existing U0 target, run with output preserved:

```sh
build/planck-udh-pt2-stationary > /private/tmp/udh-u1-stationary.log 2>&1
build/planck-udh-pt2-energy-contract > /private/tmp/udh-u1-u0-regression.log 2>&1
```

Fast boundary guards (not numerical acceptance) can run without a build:

```sh
python3 -m unittest discover -s tests -p test_udh_pt2_stationary_contract.py -v
```

U3 will differentiate the pair's four coefficient factors and add the Dprime
response once. No G/RHS, Z solve, W, geometry derivative or production driver
change is part of U1.

### 15.17 U3: literal pair orbital gradient and stationary RHS

Source: `src/dft/udh_pt2_orbital.h/.cpp`. New standalone acceptance target:
`planck-udh-pt2-orbital`. U3 composes the validated U1 amplitudes/Dprime and
U2 physical two-spin response. It does not solve Z or enable a UKS derivative
workflow. The supplied paper's page 124115-4, particularly Eq. 22 and the
response expression preceding Eq. 23, was visually checked for this mapping.

**Coefficient convention.** With delta C=C U, G_pq multiplies U_pq, not
U_qp. For a pair amplitude T_ijab and the chemist integral g_(ia|jb), the
four contributions with weight w are

\[
G^{(i)}_{pi}\mathrel{+}=wT_{ijab}(pa|jb),\qquad
G^{(a)}_{pa}\mathrel{+}=wT_{ijab}(ip|jb),
\]
\[
G^{(j)}_{pj}\mathrel{+}=wT_{ijab}(ia|pb),\qquad
G^{(b)}_{pb}\mathrel{+}=wT_{ijab}(ia|jp).
\]

Here w=c for aa/bb and w=2c for ab. In ab, the first two slots belong
to alpha and the last two to beta. Same-spin pair symmetry combines its
two occupied and two virtual legs. For U_ai=X_ai and U_ia=-X_ai,
ell_pair_ai=G_ai-G_ia. The positive occupied-coefficient contribution is
stored as `external_ai`; the negative virtual-coefficient contribution is
`internal_ai`. These are the two parts of the literal derivative, not an
additional legacy internal bracket.

In Planck's ordered T storage, the alpha Eq. 22 pair terms reduce to

\[
\ell^{\alpha,\mathrm{SS,ext}}_{ai}
=2c\sum_{jbc}(a_\alpha c_\alpha|j_\alpha b_\alpha)T^{aa}_{ijcb},
\quad
\ell^{\alpha,\mathrm{SS,int}}_{ai}
=-2c\sum_{kjb}(k_\alpha i_\alpha|j_\alpha b_\alpha)T^{aa}_{kjab},
\]
\[
\ell^{\alpha,\mathrm{OS,ext}}_{ai}
=2c\sum_{jbc}(a_\alpha c_\alpha|j_\beta b_\beta)T^{ab}_{ijcb},
\quad
\ell^{\alpha,\mathrm{OS,int}}_{ai}
=-2c\sum_{kjb}(k_\alpha i_\alpha|j_\beta b_\beta)T^{ab}_{kjab}.
\]

For beta, interchange spins; ab is repacked with beta on the second
occupied/virtual slots: T^ba_ijab=T^ab_jiba. This double interchange
has positive sign. The tests implement these reduced expressions
independently of the four-slot builder, separately for each spin sector.
No exact-exchange coefficient multiplies the pair sector.

**Response and coefficient connection.** U1's fixed-T common scalar also
has the derivative of Dprime:F. The physical density variation is
delta P^sigma=Cv X Co^T+Co X^T Cv^T (no closed-shell factor two).
Joint-spin adjointness gives

\[
\ell^{\sigma,\mathrm{response}}_{ai}
=2[C_v^{\sigma T}\mathscr K^\sigma[D'_\mathrm{AO}]C_o^\sigma]_{ai}.
\]

The two belongs to the adjoint projection of **every** physical channel,
including both XC inputs. It is outside U2 and does not rescale T or Dprime.
The Fock coefficient connection is explicitly retained:

\[
\ell^{\sigma,\mathrm{connection}}_{ai}
=2[(F^\sigma D'^{\sigma})_{ai}-(F^\sigma D'^{\sigma})_{ia}].
\]

It vanishes when F has no ov/vo block, in particular at the canonical
reference. Keeping it explicit permits off-shell/noncanonical tests of the
same functional without silently applying a canonical-only simplification.
The returned total is pair + response + connection. At the canonical
reference this is exactly the pair + response RHS of Section 15.6. The
printed compact XC response is mapped through the physical U2 derivative
and the scalar identity; an independent nonlinear potential FD, not a
factor inferred from a printed R label, is the acceptance test. U4 still
owns the residual Jacobian, its transpose convention and the Z solve.

**Typed boundaries.** `UDHFullMOIntegrals` stores aa/ab/bb *full direct MO*
integrals in chemist [p,q,r,s] order, with separate alpha/beta dimensions.
This is deliberately different from U1's [i,j,a,b] amplitude-order ovov
contract. The builder checks extents, finite values, within-pair symmetry,
same-spin pair interchange and U1 amplitude antisymmetry, even at c=0.
This dense boundary is a correctness implementation, not an optimized
integral transformation. The result retains all four square G matrices for
each sector/spin and their rectangular vo projections.

`build_udh_orbital_rhs` additionally checks C^T S C=I through the U1 AO
adapter and full symmetric spin Fock shapes. It returns Dprime in both
bases, U2's physical AO response ledger, separate projected J, K,
XC-from-alpha and XC-from-beta channels, the coefficient connection and the
total. Errors are propagated; nonfinite projections fail. U2's callbacks
must represent the derivative of the same Fock model used in the scalar
and be jointly self-adjoint; the callback type alone cannot prove this.
No amplitude-response or independent legacy bracket is added to the RHS.

**Independent numerical oracles.** The C++ test uses
common symmetric AO ERI factors, distinct spin MO coefficients, a
nonorthogonal AO metric and unequal occupied/virtual dimensions. For each
aa/ab/bb sector, it perturbs each of the four coefficient matrices separately
in every MO basis direction and evaluates the pair scalar from AO factors,
without using G or the stored MO integrals. Slot FDs use 1e-3 and 1e-4.

The complete stationary oracle uses a fixed one-body spin potential plus
physical AO J/K and a nonlinear two-spin XC model with nonzero cross
couplings. Fixed one-body matrices make its reference Fock canonical; it is
a synthetic orbital-functional oracle, **not** a molecular UKS calculation
or an independent Libxc validation. U2 supplies the separate physical
Libxc/molecular-grid evidence. The oracle evaluates the changing Fock
potential and all coefficient factors directly at displaced orbitals. It
checks each alpha/beta vo direction and a mixed-spin direction, the pair,
fixed-Dprime AO response, coefficient connection and full stationary sum,
then repeats with off-shell amplitudes and nonzero Fock ov blocks. It also
checks the separate J/K/XC contractions, spin reversal, scales 0/0.54/-0.27,
empty spin sectors, zero virtual blocks, canonical vanishing connections,
callback errors and invalid inputs. A spatial 2c(2t-t_exchange):g scalar
independently checks the closed-shell summed G reduction.

Pair/index/connection tolerance is 2e-11 (scaled). The full stationary and
response FDs use 1e-4 and 5e-5 with 3e-8 tolerance; the closed-shell
four-factor spatial FD tolerance is 3e-9. These are acceptance bounds;
measured results are recorded below. U1's amplitude validation was extracted into
a shared checked function without changing its density contractions; rerun
U1 after rebuilding along with U3. No build was started by the agent.

```sh
build/planck-udh-pt2-orbital > /private/tmp/udh-u3-orbital.log 2>&1
build/planck-udh-pt2-stationary > /private/tmp/udh-u3-u1-regression.log 2>&1
OMP_NUM_THREADS=2 build/planck-udh-ks-response > /private/tmp/udh-u3-u2-regression.log 2>&1
```

The 17 fast UKS boundary/fixture checks and 24 fast RKS DH checks pass;
logs are `/private/tmp/udh-u3-source-tests.log` and
`/private/tmp/udh-u3-rks-boundary-regression.log`. U3 boundary checks are in
`tests/test_udh_pt2_orbital_contract.py`. They do not establish numerical
acceptance. UKS production guards remain intact.

**Rebuilt acceptance (2026-09-14).** The user rebuilt U3 and U1 at 07:42
local time. `planck-udh-pt2-orbital` and the U1 regression both exit 0;
the unchanged U2 executable also passes again. Maximum logged U3 errors:

| Check | Maximum absolute error |
|---|---:|
| All four pair-coefficient slots, aa/ab/bb and both spin orientations | 2.82481083e-15 |
| Canonical full stationary orbital FD | 3.57231063e-14 |
| Off-shell/noncanonical full stationary orbital FD | 4.93354272e-14 |
| Fixed-Dprime physical response FD | 4.81773070e-15 |
| Dprime:F coefficient-connection FD | 5.67664541e-16 |

The independent Eq. 22 external/internal sums, isolated aa/ab/bb stationary
identities, alpha/beta/mixed rotations, J/K/XC projections, spin reversal,
zero sectors, scaling, spatial closed-shell reduction and rejection checks
all pass. U1's largest amplitude-direction FD error remains 1.73472348e-14;
U2's largest total physical-response FD error remains 1.14205e-10. Logs:
`/private/tmp/udh-u3-orbital.log`, `/private/tmp/udh-u3-u1-regression.log`,
`/private/tmp/udh-u3-u2-regression.log`.

U3 is accepted at the detached orbital-functional primitive level. This is
not a molecular UKS DH gradient test or a validation of the coupled Z solve;
U4's subsequent source implementation is recorded below. U5–U11 and the
UKS production guard remain unchanged.

### 15.18 U4 — coupled canonical KS Jacobian and transpose Z solve

**Rebuilt detached primitive acceptance passes (2026-09-14).**
This is a detached correctness primitive, not UKS gradient enablement or a
promotion of the RKS iterative solver into UKS. Sources are
`src/dft/udh_zvector.h/.cpp`; the standalone target is
`planck-udh-zvector`. No new driver branch, environment switch, SCF iteration,
level shift, orbital recanonicalization, or gradient composition is added.

**Derivation in Planck conventions.** The supplied paper's Eq. 27 has the
orbital gap plus response on the left, and minus its Lagrangian RHS on the
right. To fix its implementation normalization, start with spin-specific
orthonormal coefficients and the physical UKS density
\(P^\sigma=C_o^\sigma C_o^{\sigma T}\). For the non-metric variation
\(\delta C^\sigma=C^\sigma U^\sigma\), set
\(U_{vo}^\sigma=X^\sigma\), \(U_{ov}^\sigma=-X^{\sigma T}\), and
\(U_{oo}^\sigma=U_{vv}^\sigma=0\). Then

\[
\delta P^\sigma=C_v^\sigma X^\sigma C_o^{\sigma T}
                 +C_o^\sigma X^{\sigma T}C_v^{\sigma T}.
\]

There is no RKS occupation factor 2. Differentiating the *co-moving* Fock
matrix, rather than its fixed-coefficient AO operator alone, gives

\[
\delta F_{\rm MO}^\sigma
 =F_{\rm MO}^\sigma U^\sigma-U^\sigma F_{\rm MO}^\sigma
 +C^{\sigma T}\mathscr K^\sigma[\delta P^\alpha,\delta P^\beta]C^\sigma.
\]

At a canonical reference the vo block of the coefficient commutator is
\((\epsilon_a^\sigma-\epsilon_i^\sigma)X_{ai}^\sigma\). Thus

\[
(AX)^\sigma_{ai}=(\epsilon_a^\sigma-\epsilon_i^\sigma)X^\sigma_{ai}
 +(C_v^{\sigma T}\mathscr K^\sigma[\delta P]C_o^\sigma)_{ai},
\quad
\mathscr K^\sigma[Q]=J[Q^\alpha+Q^\beta]-a_xK[Q^\sigma]
 +\sum_\tau f_{\rm XC}^{\sigma\tau}[Q^\tau].
\]

The U3 derivative \(\ell\) is used unchanged. In particular, its adjoint
R(Dprime) contribution already contains the factor 2 appropriate to
\(\mathcal H=P_{\rm pair}+\sum_\sigma D^{\prime\sigma}:F^\sigma\).
That RHS factor must **not** be copied into this residual Jacobian, or
applied to its XC channel a second time. Likewise, U4 does not multiply
the already-scaled RHS by \(c_{\rm PT2}\) again.

For \(\mathcal L=\mathcal H+\sum_\sigma z^\sigma:F_{vo}^\sigma\),

\[
\delta\mathcal L=\ell:X+z:AX=(\ell+A^Tz):X,
\qquad A^Tz=-\ell.
\]

Joint-spin adjointness follows from
\[
Y:AX=\sum_{\sigma ai}(\epsilon_a^\sigma-\epsilon_i^\sigma)
Y^\sigma_{ai}X^\sigma_{ai}
+\tfrac12\sum_\sigma\delta P_Y^\sigma:
\mathscr K^\sigma[\delta P_X].
\]
The second term is symmetric in X and Y for U2's jointly self-adjoint
physical response. This is not a symmetry assertion about each cross-spin
matrix separately. The implementation nevertheless solves **the transpose
explicitly**; it does not use a presumed equality to conceal the convention.

**Typed boundaries and controls.**

- `UDHZVectorInputs` owns alpha/beta C, canonical full MO Fock matrices, S,
  occupation counts, and a U2 response object. It requires positive-definite
  S, \(C^{\sigma T}SC^\sigma=I\), finite dimensions and diagonal Fock
  matrices. The gaps come from these physical Fock diagonals, not separately
  passed potentially shifted eigenvalues. The caller must supply the same
  physical unshifted reference to F and U2; the type cannot prove consistency
  of arbitrary callback captures. A diagonal level-shift contamination
  cannot be recognized from the Fock matrix alone.
- `apply_udh_eq27_hessian` returns the two AO trial densities and separately
  inspectable gap, J, same-spin K, XC-from-alpha and XC-from-beta vo pairs.
  Each XC pair exposes both target spins, retaining all four spin blocks.
  U2 validates callback outputs and propagates errors/exceptions.
- Alpha and beta may have different MO and occupied/virtual dimensions.
  Packing is alpha first, then beta, each `a*nocc+i`; an empty ov sector
  retains its `(nvirt,nocc)` shape and contributes a zero AO density.
- `solve_udh_zvector` builds the dense Jacobian from these checked actions,
  verifies zero-trial homogeneity and joint adjointness, checks singular
  values and column-pivoted QR rank, then solves the transposed system.
  Defaults are residual tolerance 1e-12, scaled adjoint tolerance 1e-10 and
  minimum singular-value ratio 1e-12. Negative eigenvalues are not rejected
  merely for their sign. Singular/ill-conditioned systems return errors;
  no level shift, regularization or zero-solution fallback is applied.
- The final residual is **fresh**: new unit-column actions are paired with
  z to obtain \((A^Tz)_q=z:(A e_q)\), not multiplication by the cached
  matrix or substitution of Az. Require
  \(\|A^Tz+\ell\|_\infty\le10^{-12}\max(1,\|\ell\|_\infty)\).
  A successful nonempty solve uses `1+2*n` actions, where
  \(n=o_\alpha v_\alpha+o_\beta v_\beta\); an empty total space validates
  its zero action and returns correctly shaped zero blocks.
- `UDHZVectorProducts` owns the C/F/S and RHS snapshot, Jacobian, Z,
  per-spin residual matrices, maximum residual, adjoint defect, rank,
  reciprocal condition and action count. U2 callbacks retain their existing
  borrowed geometry/functionals lifetimes; copying the products does not
  deep-copy these external resources. They must remain unchanged and alive.
  This O(n²)-storage/O(n³)-factorization reference is intentionally not a
  large-system algorithm; UKS matrix-free promotion is a separate change.

**Independent test construction and acceptance gates.**
`tests/udh_zvector.cpp` builds a nonidentity AO metric, distinct spin
coefficients and unequal spin spaces, common symmetric AO ERI factors,
and a nonlinear coupled two-spin XC scalar/potential. Fixed spin-specific
one-body matrices make the chosen center exactly canonical. This is a
synthetic orbital-functional model, **not a converged molecular UKS state**.
Its Fock potential uses explicit AO four-index contractions while the U2
response callbacks use the factorized actions. Cayley rotations move the
occupied density and both coefficient sides of F. The FD reference never
differentiates the new Jacobian builder.

Rectangular MO spaces test the detached U4 response/solve only, using a
fixed nonzero RHS and the independently FD-built transpose Jacobian. They
also assert that the current U3 handoff rejects those spaces. U1's AO Dprime
adapter requires both spin coefficient matrices and Dprime matrices to be
square/all-active in the common AO dimension. Consequently the U3-to-U4
stationary cancellation tests use full square coefficient matrices, with
unequal alpha/beta occupations and virtual counts. Missing occupied or
virtual spin blocks are exercised within both appropriate boundaries. U4's
rectangular response capability does not enable a truncated-space DH gradient.

| Added check | Acceptance condition / purpose | Status |
| --- | --- | --- |
| Every alpha/beta Jacobian column | Co-moving F_ai FD at 1e-4 and 3e-5; gap/J/K/XC channels isolated; maximum error 3e-8 | PASS; maximum full-column error 1.67507577e-8, isolated-channel error 1.65749281e-8 |
| Mixed-spin direction and spin swap | Combined FD and linear action; permuted solutions agree to 2e-12 | PASS |
| Dense transpose solve | Solve independently FD-built A^T against the same U3 RHS; Z agreement 2e-9; rectangular spaces use a separate fixed RHS | PASS; maximum logged rectangular Z difference 8.12023643e-11 |
| Common stationary cancellation | Independently FD U1 H(C) plus z:F_vo using fixed stationary amplitudes, regenerated MO integrals and co-moving F; each basis and mixed direction below 2e-9 | PASS; maximum error 3.02673106e-14 |
| Closed-shell reduction | Equal-spin action equals existing RKS Eq. 27 action; each spin Z is half the RKS Z for half-spin RHS | PASS |
| Empty/unequal spaces | Missing occupied/virtual sectors, unequal MO counts, and empty total pair space retain valid shapes | PASS; U3 rejection of rectangular spaces also confirmed |
| Failure/conditioning controls | Bad metric/C/F/RHS, noncanonical F, callback failures, nonadjoint kernel, singular/poor condition rejected; indefinite nonsingular problem solved | PASS |
| Transpose and fresh-action controls | Deliberately relaxed adjoint gate only in a nonphysical test distinguishes A^Tz from Az; callback error injected after matrix construction must propagate | PASS |

The stationary test holds amplitudes fixed at their canonical stationary
value; U1 independently established amplitude-direction stationarity.
It does not claim a live geometry-dependent amplitude-response solve or
the U9 transport identity. The U2 physical J/K/polarized-XC regressions
remain separately required; these synthetic U4 tests do not replace a
future physical molecular orbital-Jacobian or full-gradient validation.

**First rebuilt run and correction (2026-09-14).** The user-supplied
`planck-udh-zvector` binary (08:35) exits 1 at the first U3 RHS handoff:
`UDH U1 AO adapter: invalid all-active matrix shape or values`.
The initial fixture used four alpha but only three beta MOs in a four-AO
basis. That is valid for the detached U4 Jacobian, but not for U3's existing
all-active AO Dprime adapter. Its preceding canonical, per-column/channel
FD and adjoint assertions reported no failures; the later solve/cancellation
and control suites were not reached. That first run was not a complete U4 pass.

Only `tests/udh_zvector.cpp` was corrected: separate the rectangular-space
solver oracle from the all-active stationary handoff, explicitly retain the
U3 rejection, and add full-square fixtures for the U3-to-U4 tests. Neither
the U1/U3 contract nor the U4 solver was changed. The original failure log
is retained at `/private/tmp/udh-u4-zvector.log`.

**Corrected rebuilt acceptance.** The user rebuilt `planck-udh-zvector`
(binary stamped 2026-09-14 08:41). Running with `OMP_NUM_THREADS=2` exits
zero and reports `PASS U4 coupled canonical KS Jacobian and transpose Z solve`.
All seven fixture audits, transpose/error controls and the closed-shell
reduction pass. The rectangular audits have 6, 4 and 2 response directions;
the all-active stationary audits have 7, 7, 4 and 3 directions. The largest
logged fresh residual is 1.38777878e-16 for the fixed-RHS rectangular tests
and 2.71050543e-20 for the U3-RHS stationary tests. The latter fixtures have
singular-value ratios between 0.83097165 and 0.86458992. The separate
ill-conditioned and singular rejection controls also pass. Retest log:
`/private/tmp/udh-u4-zvector-retest.log`.

The 22 UKS no-build Python checks pass, including five U4 architecture/oracle
guards (`/private/tmp/udh-u4-source-tests-retest.log`). The extra guard fixes
the rectangular/all-active boundary in the test layout. All 28 RKS DH
Python checks pass (`/private/tmp/udh-u4-rks-source-tests.log`), and
`git diff --check` passes. These source/runner checks do not establish
numerical acceptance of the corrected C++ suite.

The rebuilt U1, U2 and U3 numerical regressions pass, as do the four RKS
amplitude/density, physical contraction, XC callback and GMRES suites.
Logs (all `/private/tmp/`): `udh-u4-u1-regression.log`,
`udh-u4-u2-regression.log`, `udh-u4-u3-regression.log`,
`udh-u4-rks-invariants.log`, `udh-u4-rks-contract.log`,
`udh-u4-rks-xc.log`, and `udh-u4-rks-gmres.log`. No agent build or
configuration was started. U4 is accepted at the detached canonical
orbital-response/Z-vector primitive level, not as a molecular UKS gradient.
U5–U11 and the UKS production guard remain unchanged.

## 16. Hessian-free Z-vector swap probe

This section records the diagnostic experiment that preceded O3. It is
not the production iterative implementation; Section 17 describes that
separate promotion and its acceptance evidence. Select the probe by setting
`PLANCK_DFT_DH_HESSIAN_PROBE_LOG` to a writable ledger path for a restricted
global-DH gradient request. With the variable absent, the normal backend
is now O3 GMRES; with it present, the baseline is explicitly dense QR.
An empty path is rejected; MPI runs and more than 128
occupied–virtual pairs are excluded from the probe.

### Question and controls

The diagnostic dense Hessian is built column-by-column from
`apply_dh_eq27_hessian`. A separate implementation,
`build_ks_orbital_hessian_op`, retains the shared KS/SOSCF action and its
`kernel_scale=2` convention. Its input object is kept alive throughout the
probe because that callback borrows it by reference. The probe distinguishes:

1. Dense QR production Z versus GMRES using the same dense matrix.
2. Dense QR production Z versus GMRES calling Eq. 27 directly.
3. Dense QR production Z versus GMRES calling the shared KS action.

All iterative solves start at zero. They use right-preconditioned restarted
GMRES (24-vector restart, maximum 256 iterations, two-pass orthogonalization),
with absolute orbital gaps floored at 1e-8 **only in the preconditioner**.
The operator and PT2 denominators are not shifted. The solver tolerance is
`1e-12 * max(1, max_abs(rhs))`, checked with a fresh unpreconditioned residual.
There is no assumption of positive definiteness and no dense-solution fallback.

The ledger prints full-precision orbital-energy/J/K/XC matrices, differences,
total-matrix asymmetries, mixed-direction actions at several scales, Z vectors,
iteration residuals, and residuals against both the candidate and reference
operators. Shared-action channels use its one-spin-sized trial density and
ov projection, whereas Eq. 27 uses the total-density trial and vo projection.
Their comparison explicitly exposes factor and orientation differences. The
shared action is also checked against its channel reconstruction; callback
errors cannot be accepted as the legacy callback's silent zero response.

The dense matrix here is **not an independent physical Hessian oracle**.
Basis-direction agreement alone is also insufficient: the mixed-direction
checks test superposition and screening effects. Passing this experiment
establishes equivalence for these inputs, not universal operator correctness.

### Actual gradient substitution

One explicitly selected diagnostic run constructs the converged SCF/PT2 state, dense Z,
derivative arrays, dense contract, and analytic KS gradient. The probe then
rebuilds the same typed contract with **only Z changed** to the shared-action
GMRES solution. There is no intervening SCF, amplitude, geometry, quadrature,
or orbital update. The output gradient in normal JSON is the swapped result.

The ledger records raw/symmetric D and W matrices and the separate h, overlap,
separable ERI, nonseparable ERI, P-side XC, D-side XC, Becke partition, and
point-translation gradient contributions. Dense and candidate PT2 and total
gradients are printed in the same driver-standard frame. Normal JSON retains
the usual requested output frame. The RHS must remain unchanged.

`STATUS PASS` requires action agreement within 1e-10, both control and candidate
Z agreement within 1e-9, candidate dense residual within 1e-9, unchanged RHS
within 1e-12, and total-gradient agreement within 1e-9 Ha/Bohr. A finite,
converged but discrepant candidate is deliberately returned with ledger status
`DIFFERENCE`, so its energy-FD discrepancy can be measured. It must not be
mistaken for an accepted production result. Solver/callback failure emits
`FAILURE` and aborts the diagnostic run rather than substituting dense Z.

### Rebuild and run

The user owns rebuilding. Rebuild `planck-dft` and the new Eigen-only test
target `planck-dh-probe-gmres`; no build was started when adding this probe.
After rebuilding:

```sh
build/planck-dh-probe-gmres > /private/tmp/dh-probe-gmres.log 2>&1
python3 tests/dh_hessian_swap_audit.py \
  > /private/tmp/dh-hessian-swap.log 2>&1
```

The runner defaults to the validated water and nonplanar C1 H2O2 fixtures.
It first runs a clean normal gradient, then the opt-in swap, requires an
explicit probe marker, and preserves every input/log/result in a unique
temporary directory. It reports an error rather than silently accepting a
stale binary. It also checks that energy and normal-output gradients agree.
For the stronger end-to-end comparison:

```sh
python3 tests/dh_hessian_swap_audit.py --fd \
  > /private/tmp/dh-hessian-swap-fd.log 2>&1
```

`--fd` compares both gradients against the same unmodified total-energy
central differences on every coordinate at 1e-4 and 2e-4 Bohr, with the
existing 5e-8 Ha/Bohr tolerance. Energy endpoints run with all DH probe flags
removed. A probe mismatch is a failed audit even if its molecular error falls
below the looser FD tolerance. The runner serializes full matrix ledgers into
`report.json`, with a top-level `summary.json` listing fixture outcomes.

Source: `src/dft/dh_hessian_probe.*`, diagnostic solver
`src/dft/dh_probe_gmres.h`, and the single opt-in branch in `driver.cpp`.
Tests: `tests/dh_probe_gmres.cpp`, `tests/test_dh_hessian_swap_audit.py`.

### Measured swap results (2026-09-12)

The user rebuilt and ran the full `--fd` audit. Both fixtures pass, including
all 42 coordinate/step pairs for each of the dense and swapped gradients.
The total energies are identical at stored precision. Maximum absolute
differences are:

| Quantity | Water | Nonplanar C1 H2O2 |
|---|---:|---:|
| Dense/shared Hessian matrix | 2.22044605e-16 | 3.55271368e-15 |
| Orbital-energy channel | 0 | 0 |
| J channel | 2.22044605e-16 | 1.24900090e-16 |
| K channel | 1.11022302e-16 | 1.66533454e-16 |
| XC channel | 1.38777878e-17 | 2.08166817e-17 |
| Largest action/reconstruction audit difference | 3.55271368e-15 | 3.55271368e-15 |
| Shared-action Z minus dense QR Z | 4.66206934e-18 | 1.06897650e-12 |
| Shared-action Z minus dense-matvec GMRES Z | 8.67361738e-18 | 5.20417043e-17 |
| Candidate residual against the dense matrix | 6.50521303e-18 | 6.64818894e-13 |
| Total gradient difference (Ha/Bohr) | 1.38777878e-17 | 1.39017270e-13 |
| Shared-gradient/energy-FD error, h=1e-4 Bohr (Ha/Bohr) | 1.44657175e-8 | 6.60811300e-9 |
| Shared-gradient/energy-FD error, h=2e-4 Bohr (Ha/Bohr) | 1.69526181e-8 | 1.26284304e-8 |

All three GMRES variants converge in 4 iterations on water and 12 on H2O2.
The H2O2 Z difference from dense QR is attributable to iterative stopping,
not a detected operator difference: GMRES using the dense matrix reproduces
the shared-action solution to 5.20e-17. The contract RHS and nonseparable ERI
gradient term are exactly unchanged. Mixed-direction tests also pass.

Evidence: `/private/tmp/dh-hessian-swap-fd.log`; full matrix ledgers and JSON
reports are in
`/var/folders/b6/8bkpkc5j2hjgttz3q3yv16gr0000gn/T/dh-hessian-swap-f5akb2ul/`.
These temporary files may not survive cleanup. The standalone
`/private/tmp/dh-probe-gmres.log` was not present when reviewing the run;
the separate solver failure-path suite is not claimed to have run merely
because the molecular audit succeeded.

The previously reported action defect is **not reproduced by the current
implementation on these fixtures**. This does not explain the historical
failure or establish large-system convergence. At the time of those results,
the normal production solve remained dense. The probe still constructs dense matrices for comparison and
duplicates channel evaluations: it demonstrates correctness of the swap,
not a production memory or timing improvement.

## 17. Recommended optimization sequence

This sequence supersedes the earlier recommendation to block matrix-free
work pending the swap experiment. Section 16 now supports staging an
iterative Z backend early, after inexpensive grid and ownership improvements.
O1 is implemented and its rebuilt invariant, XC-channel and molecular FD
checks pass (2026-09-13). O2 is implemented; its rebuilt cache, contraction,
and invariant checks pass, with molecular revalidation recorded below.
O3 is rebuilt and passes typed/physical, molecular FD and standalone solver
checks, including the corrected failure-path assertion. O4–O9 remain plans.
UKS derivative development remains a separate scope.

Let N denote AO count, o/v occupied/virtual counts, G grid points, and M
atoms. The goal is a memory-bounded conventional gradient with fifth-order
PT2 contractions, not a claim of linear scaling. Iterative response costs
also depend on iteration count. Exact contraction reordering comes before
RI or locality approximations, which require their own energy contracts.

### O0. Freeze correctness evidence and establish performance baselines

- Retain the dense QR, literal four-coefficient, and full-array contractions
  as small-system test oracles. Preserve the Section 16 input configurations,
  thresholds, and matrix/gradient evidence.
- Run the standalone GMRES solver tests, including indefinite/nonsymmetric
  matrices, restart, exhaustion, singular breakdown, and callback failures.
- Measure wall time and peak resident memory by stage: PT2 transform,
  D-prime/RHS, Z solve, derivative construction, pair backtransform, XC-II.
  Record dimensions, grid size, threads, and response-action count.
- Separate diagnostic cost from normal production cost; the swap probe's
  extra dense/channel calculations must not be used to estimate speedup.

Acceptance: reproducible numerical baselines and a performance ledger. The
source-level cost analysis below is not a substitute for those measurements.

### O1. Remove accidental quadratic-grid work

Implemented in `analytic_hessian.cpp` and the three GGA paths in
`dh_pt2_gradient.cpp`: each loop now computes the whole-grid
`gradient_squared()` vector once, then copies its entries. The old schedule
recomputed all G values for each of G points. Sigma preparation now performs
O(G) work rather than O(G²), with no change to the arithmetic inside
`gradient_squared()`. Each temporary is block-scoped and released before
Libxc evaluation; no cross-call cache or new density/grid lifetime is added.

The affected consumers are the analytic XC Hessian-vector action, the
fixed-grid P-side XC-II term, GGA moving-grid partition/point-translation,
and the GGA D-side AO term. Grid points, weights, cutoffs, coefficients,
response conventions, and Libxc calls are unchanged.

Verification status: the no-build source/schedule checks in
`tests/test_dh_grid_sigma_hoist.py` pass. The new C++ micro-oracle in
`tests/dh_grid_sigma.cpp` is part of `planck-dh-pt2-amplitude-density`: it
compares the actual Eigen whole-grid method under both schedules for
G = 0, 1, 7, 64, 256, 1024, 4096, requires exact output equality, counts
G² versus G evaluated grid values, and logs timings without a flaky wall-time
ratio threshold. The rebuilt suite passes with exactly zero sigma differences
at every tested size. At G = 4096, the whole-grid evaluations visit 16,777,216
values under the old schedule and 4,096 under the new schedule. The diagnostic
single-sample timings were 3,089 microseconds and 1 microsecond, respectively;
the small timings are not a reliable throughput benchmark or an end-to-end
speedup claim. This removes one quadratic operation, not every scaling
bottleneck.

Rebuilt validation (2026-09-13): `planck-dh-pt2-amplitude-density` and
`planck-dft-coulomb-response` both exit successfully, including the physical
XC-II channel/geometry checks. Both molecular fixtures pass every Cartesian
coordinate at both steps, without DH diagnostic flags:

| Fixture | FD step (Bohr) | Maximum gradient error (Ha/Bohr) |
|---|---:|---:|
| Water | 1e-4 | 1.43673238e-8 |
| Water | 2e-4 | 1.70318601e-8 |
| C1 H2O2 | 1e-4 | 7.33748209e-9 |
| C1 H2O2 | 2e-4 | 1.17966995e-8 |

The acceptance threshold remains 5e-8 Ha/Bohr. The two molecular result
directories are `dh-production-cartesian-fd-8wbix1bn` and
`dh-production-cartesian-fd-m0a4k813` under the host temporary directory.
Reproduce the checks and their logs with:

```sh
build/planck-dh-pt2-amplitude-density > /private/tmp/dh-o1-invariants.log 2>&1
build/planck-dft-coulomb-response > /private/tmp/dh-o1-xc-channels.log 2>&1
python3 tests/dh_cartesian_fd_audit.py \
  tests/inputs/exploratory/dh_gradient/water_b2plyp_gradient_fd.hfinp \
  > /private/tmp/dh-o1-water-fd.log 2>&1
python3 tests/dh_cartesian_fd_audit.py \
  tests/inputs/exploratory/dh_gradient/h2o2_c1_b2plyp_gradient_fd.hfinp \
  > /private/tmp/dh-o1-h2o2-fd.log 2>&1
```

Acceptance: unchanged XC channel actions and all four XC-II contributions;
both molecular FD fixtures pass. Show linear growth of this operation with
grid size without changing grid points, weights, or numerical cutoffs.

### O2. Separate ownership, reuse stationary inputs, and cache fixed XC data

Implemented in the restricted DH production path. At O2 acceptance, the dense QR Z solve,
orbital packing, response factors, PT2 scaling and all contraction formulas
were unchanged. O3's subsequent solver change is documented below; neither
step enables UKS derivatives.

**Per-geometry ownership.** `DHGradientGeometryWorkspace` owns the MO ERIs
and h/S/ERI derivative bundle. `DHGradientDriverInputs` holds a
`shared_ptr<const DHGradientGeometryWorkspace>` instead of copying those
arrays. The driver moves the finished MO transform and derivative bundle
into that owner. Input/probe copies share the same immutable owner and
quartic buffers; no unowned span escapes. The small C/epsilon/Z input
matrices remain owned copies. The PT2 snapshot and XC-II pointer inputs are
still synchronous borrowed inputs: they must remain unchanged and alive for
the complete geometry evaluation, as before. Direct J/K callbacks continue
to borrow the prepared shell-pair/basis data.

**One stationary construction.** `build_dh_stationary_products` constructs
the tilde amplitudes, Dprime, Eq. 41 response and literal Eq. 40 RHS once.
The Z solver uses these products; the two-argument
`build_dh_gradient_driver_contract(inputs, std::move(stationary))` then
consumes exactly those objects. The handoff checks snapshot/ERI identity,
C and scale/exchange bindings, dimensions, finite entries and the literal
RHS convention. The geometry's PT2/ERI data must not be mutated in place
between preparation and consumption; pointer identity is not a content
checksum. A fresh final physical Z action/residual remains mandatory in the
driver. The one-argument contract overload is the independent reconstruction
adapter used by small-system tests and the explicit Hessian-swap probe.

**Gamma storage.** `DHGammaStorage::ContractionOnly` computes the unchanged
eightfold-symmetric separable and nonseparable tensors. It releases each
raw temporary after its symmetric adapter is complete and never constructs
the unused raw/symmetric totals. Eq. 33 consumes only the two separate
symmetric tensors. `ReferenceAll` remains the default for low-level and
one-argument reference calls and returns all six arrays. Finished tensors
and stationary products are moved, not copied, into the contract.

| Storage/work item | Previous production schedule | O2 production schedule |
|---|---|---|
| Driver MO ERIs and derivative bundle | Value copies at the input boundary | Shared immutable owner; buffers moved into it |
| Dprime/Eq. 41/Eq. 40 preparation | Solver construction plus contract reconstruction | One construction, then move handoff |
| Retained Gamma arrays per contract | Six N^4 arrays, plus transient result copies | Two N^4 arrays, moved; at most three during their construction |
| Fixed XC ground-density evaluation | Once per response action | Once per kernel preparation |
| Libxc evaluations | Two for LDA / four for GGA per action | Two / four at preparation, none during actions |

**Fixed XC cache.** `prepare_rks_xc_kernel` creates an immutable owning
`RKSXCKernel`. It snapshots AO values/gradients, weights, ground rho/gradient
fields and the summed functional derivatives. The GGA cache retains
vsigma, f-rho-rho, f-rho-sigma and f-sigma-sigma, including the combined-XC
correlation-suppression rule. Each action evaluates only the new trial
density fields and projects the original LDA/GGA formula. The original
uncached HVP is retained as an independent comparison path. No global cache,
geometry-key inference, mutable functional pointer or trial-dependent cache
is introduced. Each geometry/SCF call prepares a new kernel; an old kernel
continues to represent its original snapshot if caller objects change or
are destroyed. Kernel copies share that snapshot.

The owned AO snapshot adds four G-by-N arrays to the cache; that deliberate
lifetime-safety cost must be included in memory measurements. Reported
`storage_bytes()` counts retained numerical arrays, not allocator overhead
or whole-process RSS. A future shared geometry-wide AO owner could remove
this extra AO copy, but no lifetime-unsafe borrowing is used to claim that
saving here. No end-to-end peak-RSS or timing speedup is claimed from the
local storage counts.

**Rebuilt primitive acceptance (2026-09-13).** The user-managed rebuild
completed at 12:54 local time; the following executables exit successfully:

- `planck-dh-eq41-xc-response`: cached/uncached actions for LDA, GGA and
  combined B2PLYP at G=2,17,64, four trial scales, two potential-FD steps,
  copied/moved kernels, invalid trials, snapshot lifetime and new-snapshot
  changes. Maximum cached/reference norm difference is 8.881784e-16. The
  largest potential-FD norm error is 1.535470e-7, within the 2e-7 bound.
- `planck-dft-coulomb-response`: physical water compact/reference Gamma
  arrays agree exactly; Dprime, response, RHS, W, separate h/S/ERI and all
  four XC-II channels agree. Pointer checks confirm stationary products are
  moved and workspace copies share buffers. Stale-scale/nonfinite products
  fail. Eq. 41 preparation makes one response call; downstream assembly
  makes one further call for Eq. 42 rather than rebuilding Eq. 41.
  Retained Gamma storage is 115,248 bytes in the six-array reference and
  38,416 bytes in production.
- `planck-dh-pt2-amplitude-density`: all existing algebraic invariants,
  including O1's exact sigma schedule comparison, pass.

The 24 fast DH source/workflow tests pass. Primitive logs:
`/private/tmp/dh-o2-xc-cache.log`, `/private/tmp/dh-o2-contract.log`,
`/private/tmp/dh-o2-invariants.log`, `/private/tmp/dh-o2-source-tests.log`.
Molecular all-coordinate revalidation uses the same water/C1 H2O2 inputs,
1e-4 and 2e-4 Bohr steps, 5e-8 Ha/Bohr tolerance, and unchanged user SCF
limits. Logs: `/private/tmp/dh-o2-water-fd.log` and
`/private/tmp/dh-o2-h2o2-fd.log`.

Both all-coordinate molecular runs pass after the same rebuild:

| Fixture | FD step (Bohr) | Maximum error (Ha/Bohr) |
|---|---:|---:|
| Water | 1e-4 | 1.45093823e-8 |
| Water | 2e-4 | 1.69607557e-8 |
| C1 H2O2 | 1e-4 | 7.90939006e-9 |
| C1 H2O2 | 2e-4 | 1.17256475e-8 |

All calculations use the ordinary production gradient with no DH diagnostic
flags. Analytic translation-sum components are below 6.6e-14 Ha/Bohr.
Artifacts are `dh-production-cartesian-fd-prji24gp` (water) and
`dh-production-cartesian-fd-mlmxasae` (C1 H2O2) under the host temporary
directory. This establishes numerical acceptance and the measured local
storage/construction reductions. Whole-process peak-RSS and timing scaling
remain O0 performance-ledger work; they have not been inferred from these
small-system tests.

Acceptance: invariant objects agree with the reference; no stale-data or
borrowed-lifetime failures; peak memory and construction/action counts drop.
Retain finite-value, dimension, convention, and final Z-residual checks.

### O3. Promote the validated matrix-free Z solve behind a typed backend

**Rebuilt molecular, typed-contract and standalone solver validation pass
(2026-09-14), including the corrected singular-failure assertion.**
Unit-column construction and dense QR are replaced in the
normal restricted production branch by a checked action-based iterative
solve of the same AZ=-L equation. The following requirements govern it:

- First retain the successful probe's GMRES settings and preconditioner.
  Change solver tuning only after numerical equivalence is established.
- Use one explicitly specified physical KS action, preserving density
  normalization, a*nocc+i packing, and all J/K/XC coefficients. The probe
  validates both the Eq. 27 and shared KS conventions on these systems;
  do not mix pieces of the two conventions implicitly.
- Propagate callback errors. The shared callback's historical zero-vector
  return on XC failure is not an acceptable production error contract.
  Resolve its borrowed-input lifetime explicitly.
- Compute a fresh final residual with the unmodified action. A preconditioner
  floor is not a shift of A or of PT2 denominators. No silent dense fallback
  and no requirement that A be positive definite.
- In the iterative production backend, do not allocate the dense Hessian,
  construct its columns, run the three diagnostic solves, or reconstruct
  duplicate channels. Keep those operations in explicit tests/probes only.

Acceptance: reproduce the channel, Z, and gradient bounds in Section 16;
exercise larger ov spaces, restart, tighter tolerances, failure handling,
and more than one nonsymmetric geometry. Preserve the existing molecular
FD tolerance. Measure memory reduction from O((ov)^2) Hessian storage to
O(k*ov) Krylov storage plus the action workspace, with bounded restart k.
Report actual action counts and time; two small-fixture passes do not
establish general convergence or an asymptotic iteration bound.

#### O3 implementation contract and acceptance ledger

- `src/dft/dh_zvector.h` provides `DHZVectorOptions`, the backend enum,
  `DHZVectorResult` and `solve_dh_zvector`. The default is
  `MatrixFreeGMRES`; `DenseReference` is an explicit test/probe choice.
- `src/dft/response_gmres.h` contains the same restarted right-preconditioned
  algorithm used by the validated probe: tolerance 1e-12, restart 24,
  maximum 256 iterations, and two-pass modified Gram–Schmidt. The old
  `dh_probe_gmres.h` aliases this implementation, avoiding a second solver.
- Each action invokes `apply_dh_eq27_hessian` with the unchanged physical
  total-density trial and separate gap/J/K/XC factors. This does not use
  the historical shared action's zero-on-XC-error behavior. Detailed
  callback errors, exceptions, nonfinite values and invalid dimensions
  propagate; nonconvergence aborts the gradient, with no dense fallback.
- The solver borrows its arguments synchronously only. No action escapes.
  Direct J/K borrow the current prepared geometry; XC owns the immutable
  O2 kernel snapshot. The same stationary RHS is moved into the gradient
  contract after solving. No SCF, PT2, grid or coefficient convention changes.
- `DHEq41ZVectorProducts::solver` and the normal `DH Z-vector` log line
  report backend, pair count, iterations, actions, restarts, final residual,
  matrix-storage bytes and solver seconds. Action count includes the initial
  zero trial, Arnoldi and per-iterate residual actions, and the final fresh
  residual. The downstream contract residual is one additional action
  outside this count. There is no timing-based correctness assertion.
- For cycle width k and n=ov, `krylov_matrix_bytes` reports the largest
  allocated basis/direction/Hessenberg matrix sum:
  `sizeof(double) * [n*(2*k+1) + k*(k+1)]`. It excludes vectors, small QR
  temporaries, the action/cache workspace and allocator overhead; it is
  **not peak process RSS**. Dense-reference bytes report only its n-by-n
  Hessian. For tiny n the Krylov matrices can exceed a dense Hessian; the
  benefit is bounded-restart scaling for growing n, not guaranteed savings
  on every molecule. No whole-process speedup/RSS reduction is claimed.
- Ordinary gradient, optimization and frequency calls use GMRES. The
  existing `PLANCK_DFT_DH_HESSIAN_PROBE_LOG` selects the dense reference
  baseline plus the existing diagnostic comparisons; no new input flag is
  added. The historical audit runner calls its normal-run files `dense`:
  after O3 those files hold the production Eq. 27 GMRES result. Its explicit
  probe still compares dense QR, dense-matvec GMRES, Eq. 27 GMRES and shared
  KS GMRES on one fixed state, and tests their gradients.

The user supplied the rebuild (test targets stamped 2026-09-14 08:01,
`planck-dft` 08:02); no agent build or configuration was started. The
rebuilt executables were then run with `OMP_NUM_THREADS=2`, writing outputs
to logs. The **28 fast DH Python routing/runner checks
pass**, including four O3 guards, and `git diff --check` passes. Output:
`/private/tmp/dh-o3-source-tests.log`. These checks do not establish numerical
equivalence or C++ compilation. All 17 UKS Python contract/fixture checks also
pass (`/private/tmp/dh-o3-uks-source-tests.log`); UKS production remains
disabled. Added numerical acceptance tests are:

| Target / audit | Added or retained O3 coverage | Status |
| --- | --- | --- |
| `planck-dh-probe-gmres` | Existing indefinite/nonsymmetric, restart, exhaustion and error tests; exceptions; matrix-free 128/256-direction banded operators with restart 4, tolerance 1e-13, exact solution, action counts and bounded matrix storage | PASS after test-only rebuild at 08:12; singular case explicitly reports `GMRES: nonfinite iterate` |
| `planck-dh-pt2-amplitude-density` (`tests/dh_zvector.cpp`) | Independent Eq. 27 gap/J/K/XC expressions; typed dense/GMRES Z and packing; restart/tighter tolerance; zero RHS; gap floor without shift; singular/exhausted solves; J/K/XC failures; changed-callback fresh residual | PASS; typed restart fixture: 8 iterations, 3 restarts, 18 actions, residual 4.55191e-15 |
| `planck-dft-coulomb-response` | Water/STO-3G physical direct J/K + analytic GGA action; same-RHS typed dense/GMRES; D/W, h/S, separable/pair ERI, all four XC-II channels and total correction (1e-9 bounds) | PASS; Z difference 3.35051e-15, W 2.49865e-14, total correction 8.09763e-15 |
| `planck-dh-eq41-xc-response` | O2 cached physical XC regression | PASS |
| `tests/dh_hessian_swap_audit.py` (without `--fd`) | Retained dense/shared/explicit Eq. 27 channel ledger and molecular gradient comparison; FD run separately below | PASS on water and C1 H2O2 |
| `tests/dh_cartesian_fd_audit.py` | Normal production B2PLYP water and C1 H2O2, every Cartesian coordinate at 1e-4 and 2e-4 Bohr; max error 5e-8 Ha/Bohr; user SCF limits preserved | PASS: all 42 coordinate/step comparisons |

**Molecular acceptance and solver telemetry.** Both center logs explicitly
show `backend=GMRES` and `dense_matrix_bytes=0`; no DH debug flags entered
the production FD runs. All energies are Planck total-energy calculations.

| Quantity | Water | C1 H2O2 |
| --- | ---: | ---: |
| Maximum analytic–FD error, h=1e-4 Bohr (Ha/Bohr) | 1.45093822e-8 | 7.90936419e-9 |
| Maximum analytic–FD error, h=2e-4 Bohr (Ha/Bohr) | 1.69607556e-8 | 1.17256574e-8 |
| Production ov pairs | 10 | 27 |
| Production GMRES iterations / actions | 4 / 10 | 12 / 26 |
| Fresh final residual | 3.686e-17 | 6.648e-13 |
| Reported Krylov matrix bytes | 2560 | 15384 |
| Solver elapsed seconds in this run | 0.185351 | 1.457091 |
| Probe dense–shared gradient difference (Ha/Bohr) | 2.775558e-17 | 1.360786e-13 |
| Normal Eq. 27 GMRES–shared gradient difference (Ha/Bohr) | 1.387779e-17 | 1.353084e-16 |

The 128/256-direction synthetic restart tests both took 17 iterations,
35 actions and 4 restarts. Reported matrix bytes were 9376 and 18592,
versus dense Hessian sizes of 131072 and 524288 bytes. Their true residuals
were 1.55431e-13 and 1.54765e-13, within the specified scaled tolerance
`1e-13 * max(1, norm_inf(b))`. The exact-solution checks passed. These are
synthetic linear systems, not evidence of larger-molecule convergence.
The solver timings above were collected during validation, not isolated
benchmark runs; no speedup or whole-process memory claim follows from them.

**Resolved test-only correction.** The standalone suite originally required
the zero-operator/inconsistent-RHS case to return a `Solve` object with
`converged=false`. The API also permits an error: the zero projected Arnoldi
matrix can produce a nonfinite QR iterate that the solver rejects before
its explicit breakdown return. A zero action cannot satisfy the nonzero RHS.
The assertion now accepts either an error or explicit nonconvergence, but
never convergence, and prints the outcome. No solver or production source
was changed after the molecular tests. The user rebuilt `planck-dh-probe-gmres`
(binary stamped 2026-09-14 08:12), and the rerun exits zero with all controls
and failure paths passing. The singular outcome is explicitly logged as
`GMRES: nonfinite iterate`, confirming error propagation rather than false
convergence. The 128/256-direction restart/storage checks also pass unchanged.
The typed backend's singular rejection tests already pass. Retest output:
`/private/tmp/dh-o3-gmres-retest.log`; the original failed run is retained
separately in `/private/tmp/dh-o3-gmres.log`.

Logs: `/private/tmp/dh-o3-gmres.log`, `dh-o3-invariants.log`,
`dh-o3-contract.log`, `dh-o3-xc-cache.log`, `dh-o3-water-fd.log`,
`dh-o3-h2o2-fd.log`, and `dh-o3-hessian-swap.log`, all under `/private/tmp`.
Each molecular log identifies its retained per-calculation artifacts and
full-precision JSON report. U1, U2 and U3 rebuilt regression executables
also pass; logs are `/private/tmp/dh-o3-u1-regression.log`,
`/private/tmp/dh-o3-u2-regression.log`, and `/private/tmp/dh-o3-u3-regression.log`.

The new water contraction fixture uses core-H orbitals and physical AO
integrals/GGA quadrature, not a self-consistent B2PLYP state. It isolates
solver/contract equivalence; only the separate live molecular FD audit tests
the full energy-to-gradient route. More nonsymmetric geometries and larger
physical ov spaces remain necessary before claiming general convergence.

### O4. Reduce the literal Eq. 40 RHS to four fifth-order contractions

Substitute each Kronecker condition before entering the loops instead of
scanning every klcb for every ai and testing equality inside. This reduces
O(o^3 v^3) loop work to O(o^2 v^3 + o^3 v^2). Block the contractions into
matrix products and reuse compatible pair-derivative intermediates in W.
Optimize D-prime's occupied and virtual contractions similarly, preserving
the t(k,j,b,a) orientation and unordered occupied-pair multiplicities.

Acceptance: each of the four coefficient derivatives, every D-prime and
raw W block, and their symmetric contractions agree with explicit-index
oracles. Never restore the excluded legacy internal bracket.

### O5. Replace the Eq. 47 eighth-order backtransformation

Replace the nested AO-quartet/MO-quartet sum, O(N^4 o^2 v^2), with successive
one-index transformations and blocked matrix products. The conventional
all-dimensions-growing cost becomes O(N^5). Initially retain dense output
so that the contraction reordering can be validated independently of a
new derivative-streaming interface.

Acceptance: raw and eightfold-symmetric nonseparable tensors and their
derivative contractions agree on small systems; demonstrate the changed
time scaling. This step alone still has quartic storage and is not the
finished memory-bounded algorithm.

### O6. Stream derivative contractions instead of storing 3M tensors

Evaluate a derivative quartet once, contract immediately, and scatter its
center contributions into the M-by-3 result. Eliminate the O(3M N^4)
ERI-derivative bundle. Stream h/S contractions as well. Generate the
separable coefficient directly from D and P, without storing its AO tensor.
Use the O5 pair density as an intermediate reference before removing it.

Reuse derivative dispatch and permutation infrastructure from `gradient.cpp`
only through a verified convention adapter: the existing accumulator's
1/4 coefficient is not the DH correction's coefficient. Start unscreened;
introduce derivative-aware screening with an explicit error budget and
checks across thresholds, rather than assuming an energy-integral bound
controls gradient error. Retain the same a_x in separable exchange.

Acceptance: separate h/S/separable/nonseparable channel equality, atom-center
and permutation multiplicities, rigid-translation invariance, and molecular
FD; demonstrate absence of all-coordinate quartic derivative storage.

### O7. Bound pair-density, integral, and amplitude working sets

Connect blocked Eq. 47 backtransforms to the derivative consumer so that
global AO Gamma tensors disappear. Reuse partial transforms across tiles;
recomputing the full amplitude sum independently for every quartet would
reintroduce the excessive work. Share the PT2 integral workspace and
transform only the needed ovov and three-occupied/three-virtual blocks,
not the full MO tensor. Introduce blocked/out-of-core amplitudes separately.

Acceptance: dense/tiled contraction equivalence and block-size invariance;
measured memory bound under increasing AO count. Account explicitly for
the remaining O(o^2 v^2) amplitudes until their storage is also blocked;
removing AO tensors alone does not make the algorithm quadratic-memory.

### O8. Batch and fuse the XC and direct-response work

Evaluate AO values, gradients, and Hessians in grid batches. Reuse P*phi,
D*phi, and gradient projections; accumulate P-side, D-side, partition, and
point-translation outputs together while preserving their separate ledgers.
Replace forward all-atom Becke derivative propagation, currently O(G M^3),
with a separately derived reverse accumulation targeting O(G M^2), handling
zero switching factors without division by zero.

Fuse J/K integral traversal for response actions and introduce controlled
threading/distribution of independent tiles, with reproducible reductions
and explicit memory limits. This need not change physical exchange factors.

Acceptance: batch-size and thread-count consistency; all four XC-II geometry
oracles and translation checks; unchanged operator action and molecular FD.
Retain GGA AO Hessians and the partition's moving-point contribution. Do not
change quadrature or drop a geometry channel as a performance shortcut.

### O9. Add RI-DH as a separately validated backend

Reuse Planck's three-center integrals, auxiliary metric, fitted pair factors,
and derivative primitives, but derive all PT2 objects from the chosen RI
energy consistently. Include auxiliary-metric derivatives. An exact KS
reference may remain exact if that is the defined energy; its response must
then remain consistent with that choice. Do not transplant the HF-MP2
full-minus-reference gradient or assume the current RI helpers are already
fully streamed.

Acceptance: analytic derivatives match finite differences of the same
RI-DH energy, independently of the conventional-DH comparison. Quantify
fitting error and memory/time separately. Local-pair or low-rank amplitude
approximations require further scope and are not implied by RI.

### Rules common to every step

Make one independently testable change at a time and retain the current
water/C1 H2O2 two-step, all-coordinate 5e-8 Ha/Bohr FD acceptance gate.
For exact reorganizations, compare channel matrices/contractions before
relying on their total cancellation. Keep c scaled once, the raw/symmetric
D and W adapters, AZ=-L, hybrid-scaled separable exchange, and complete
XC-II intact. Add larger-basis and memory-scaling tests as the bottlenecks
are removed. Rebuilds remain user-managed; changing this plan does not
authorize production promotion or a new scientific approximation.

## 18. RKS DH geometry optimization and frequencies

Status (2026-09-13): rebuilt water Cartesian optimization, frequency and
combined opt/frequency audits pass; water internal-coordinate optimization
also passes. C1 H2O2 optimization, input-geometry frequency and combined
opt/frequency audits now all pass. Its first combined audit exposed an
SCF-limit override during a Hessian displacement:
the geometry-preparation helper replaced the requested 100 cycles with 50.
The helper now matches `Calculator::initialize()`: it applies the automatic
limit only when `_max_cycles == 0`. An explicit limit is preserved on every
trial/displaced geometry; tolerances and early convergence are unchanged.
Ten fast workflow/oracle tests pass, including explicit-limit preservation.
The rebuilt fix passes the combined retest with the original 100-cycle input:
the formerly failing displacement converges in 93 iterations, and another
takes 88. No iteration-limit increase or tolerance relaxation was needed.

Measured maximum errors before the SCF-limit fix: water energy-FD gradients
1.16201e-9 Ha/Bohr (Cartesian) and 1.12171e-9 (internal); water Hessians
1.91120e-9 Ha/Bohr² at input geometry and 1.37320e-7 after optimization;
C1 H2O2 energy-FD gradient 7.93391e-9 Ha/Bohr and input Hessian 3.25837e-8
Ha/Bohr². Water optimizations took 5 Cartesian / 4 internal steps; C1 H2O2
took 7 Cartesian steps. Logs are `/private/tmp/dh-workflow-cartesian-validation.log`,
`/private/tmp/dh-workflow-internal-validation.log`, and
`/private/tmp/dh-workflow-h2o2-validation.log`. The last log records the
combined-path failure, not a full pass. No tolerance was relaxed.

The successful post-fix C1 H2O2 combined audit is recorded separately in
`/private/tmp/dh-workflow-h2o2-scf-limit-retest.log` (artifacts:
`dh-workflows-bwu_jzzh` under the host temporary directory). Optimization
converges in 7 steps. Maximum errors are 7.93391e-9 Ha/Bohr for the optimized
energy-FD gradient and 2.96052e-8 Ha/Bohr² for the full 12×12 Hessian against
independently differenced standalone gradients. Restored energy and gradient
agree with a fresh reference calculation within 3.12639e-12 Ha and
5.60629e-10 Ha/Bohr. UKS workflow rejection remains intact. The six optimized
frequencies are 125.55, 1241.95, 1416.90, 1586.14, 3683.81 and 3697.54 cm⁻¹;
none is imaginary. These are B2PLYP/STO-3G test results, not a general accuracy
claim for all DH functionals, bases or geometries.

Use `calculation geomopt`, `calculation freq`, or `calculation geomoptfreq`
with the same RKS/global-DH/Cartesian-basis input as a supported DH gradient.
Both `opt_coords cartesian` (L-BFGS) and `opt_coords internal` (IC-BFGS) retain
the full derivative callback, including the IC-to-Cartesian fallback. UKS,
range-separated DH, and solvent-response derivative paths remain rejected.
Neither DH linear response nor imaginary-mode following is enabled.

### Shared energy/gradient contract

At every trial or displaced geometry, `run_analytic_gradient_current_geometry`
rebuilds basis/integrals, atom-centered grid and AO values, converges the KS
reference, computes a fresh RMP2 amplitude snapshot, and applies the PT2
correlation coefficient once. It supplies that same snapshot to
`compute_analytic_dh_gradient`, also used by the standalone gradient branch.
The Eq. 41/27 response, derivative integrals, Eq. 33 correction and complete
XC-II are rebuilt; only the previous density is reused as an SCF guess.

The callback sets both `current_total_energy()` and `_gradient` consistently:

\[
E(R)=E_{\mathrm{KS}}(R)+c E_{\mathrm{PT2}}(R),\qquad
g(R)=g_{\mathrm{KS}}(R)+\Delta g_{\mathrm{PT2}}(R).
\]

Optimizer gradients and Hessian columns remain in the current working frame,
flattened atom-major. The standalone gradient retains its existing requested-
frame rotation. Merely opening the workflow guard without replacing the old
KS-only callbacks would have paired a DH energy with the wrong gradient.

### Semi-numerical frequencies and state restoration

The existing frequency engine constructs

\[
H_{pq}=\frac{g_p(R+h e_q)-g_p(R-h e_q)}{2h},\qquad
H\leftarrow\tfrac12(H+H^{\mathsf T}),
\]

using the **full analytic DH gradient** at both endpoints. The step is
`calculator._hessian_step` (currently 0.005 Bohr), not the separate numerical-
energy-gradient step. Mass weighting, translation/rotation projection, normal
modes and ZPE use the existing frequency implementation. This is not an
analytic DH Hessian. At a nonstationary input geometry, interpret frequencies
accordingly; `geomoptfreq` runs the frequency stage only if optimization
converged.

After all displacements, the DFT wrapper recomputes the complete undisplaced
KS/PT2 energy and gradient. Restoring coordinates alone leaves the last
displaced wavefunction and correlated energy in the calculator. The JSON
result includes `hessian` (Ha/Bohr²), `hessian_step_bohr`, `frequencies_cm1`,
`normal_modes`, and `zpe_hartree` when a Hessian exists, alongside the restored
reference energy, geometry and gradient.

### Small, verifiable acceptance steps

1. Run the fast checks (no build or molecular calculation):

   ```sh
   python3 -m unittest discover -s tests -p test_dh_workflow_audit.py -v
   ```

2. After rebuilding `planck-dft`, run the water full-matrix frequency audit:

   ```sh
   python3 tests/dh_workflow_audit.py --workflows freq \
     > /private/tmp/dh-frequency-audit.log 2>&1
   ```

3. Run Cartesian optimization and combined optimization/frequency:

   ```sh
   python3 tests/dh_workflow_audit.py --workflows geomopt geomoptfreq \
     > /private/tmp/dh-optimization-audit.log 2>&1
   ```

4. Exercise the internal-coordinate callback separately:

   ```sh
   python3 tests/dh_workflow_audit.py --workflows geomopt --opt-coords internal \
     > /private/tmp/dh-internal-optimization-audit.log 2>&1
   ```

5. Repeat frequency/optimization with `--input` pointing to the existing C1
   H2O2 B2PLYP fixture, then rerun the standalone water/C1 H2O2 all-coordinate
   total-energy FD regression to guard the shared-assembly extraction.

The workflow runner preserves each input, log and JSON in a unique directory.
It compares restored energy (1e-8 Ha) and gradient (1e-7 Ha/Bohr) with a fresh
standalone gradient at the returned geometry; compares every Hessian matrix
entry with independent fresh-gradient differences (2e-6 Ha/Bohr²); checks
optimizer convergence and full-coordinate energy FD gradients (5e-8 Ha/Bohr);
and verifies UKS optimization/frequency rejection. These are acceptance
thresholds, not measured results until the rebuilt runner passes. No PySCF DH
derivative reference is used.
