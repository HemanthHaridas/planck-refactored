### Planck Teaching Guide

A complete theory-to-code walkthrough of the Planck quantum chemistry program.
Intended for students learning Hartree-Fock and post-HF methods, contributors
reading the source, and researchers auditing the implementation.

### 1. What Planck Is

Planck is a compact electronic structure program built around Gaussian-basis
self-consistent field theory. It implements:

- Restricted and unrestricted Hartree-Fock (RHF/UHF) with DIIS acceleration
- Kohn-Sham DFT (RKS/UKS) with LDA and GGA exchange-correlation functionals via libxc
- Obara-Saika and Rys-quadrature two-electron integral engines
- Conventional (stored ERI tensor) and direct (on-the-fly Fock build) SCF
- Point-group detection, symmetry-adapted orbitals, and MO irrep labeling
- RMP2 and UMP2 correlation energies with RMP2 natural orbital analysis
- Analytic RHF and UHF nuclear gradients
- Analytic RMP2 nuclear gradients (Z-vector / CPHF)
- Geometry optimization in Cartesian and internal coordinates
- Semi-numerical Hessians and harmonic vibrational analysis
- CASSCF and RASSCF active-space multiconfigurational SCF
- Binary checkpoint save/restart with cross-basis Löwdin projection

The HF/post-HF calculation is coordinated by `src/driver.cpp`. The Kohn-Sham
DFT calculation uses the separate entry point `src/dft/main.cpp` and the
`DFT::Driver` pipeline in `src/dft/driver.cpp`. The central data object is
`HartreeFock::Calculator` in `src/base/types.h`, which is shared by both
pipelines and carries all options, molecular data, basis data, SCF state, and
results.

### 2. Architecture Overview

### Data Flow

```
Input file (.hfinp)
  → parse (src/io/io.cpp)
  → Molecule + options in Calculator
  → coordinate conversion + symmetry detection (src/symmetry/)
  → GBS basis reading + normalization (src/basis/)
  → shell-pair construction (src/integrals/shellpair.cpp)
  → one-electron integrals S, T, V (src/integrals/os.cpp)
  → optional SAO basis construction (src/symmetry/mo_symmetry.cpp)
  → SCF loop (src/scf/scf.cpp)
       ├── conventional:  build ERI tensor once, reuse
       └── direct:        rebuild G(P) from scratch each iteration
  → post-HF: MP2 or CASSCF (src/post_hf/)
  → gradient (src/gradient/)
  → geometry optimization (src/opt/)
  → frequency analysis (src/freq/)
  → checkpoint write (src/io/checkpoint.cpp)
```

### Directory Summary

| Directory | Contents |
|---|---|
| `src/base` | `types.h` (all structs/enums/Calculator), `tables.h`, `basis.h` |
| `src/io` | input parsing, checkpoint I/O, logging |
| `src/basis` | GBS file reading, primitive normalization, contraction |
| `src/integrals` | shell pairs, Obara-Saika OS engine, Rys quadrature engine |
| `src/scf` | orthogonalizer, initial guess, RHF/UHF SCF loops |
| `src/symmetry` | libmsym wrapper, SAO basis, MO labeling, integral sym ops |
| `src/post_hf` | MP2 energy/gradient, CASSCF/RASSCF, AO→MO transforms, CPHF |
| `src/gradient` | analytic RHF, UHF, RMP2 gradients |
| `src/opt` | L-BFGS/BFGS optimizer, internal coordinates, constraints |
| `src/freq` | finite-difference Hessian, vibrational analysis |
| `src/dft` | Kohn-Sham DFT pipeline: molecular grid, AO evaluation, XC matrix, KS driver |
| `src/dft/base` | grid construction headers: radial (Treutler-Ahlrichs), angular (Lebedev), Becke partition, libxc wrapper |

### 3. Core Data Structures

### `Molecule`

Holds atomic numbers, charge, multiplicity, and three coordinate
representations that are easy to confuse:

- `coordinates` — user-input geometry in Angstrom
- `_coordinates` — same in Bohr, set by `prepare_coordinates()`
- `standard` — symmetry-standard orientation in Angstrom (set by `detectSymmetry()`)
- `_standard` — symmetry-standard orientation in Bohr (used by all integrals)

Basis centers and the nuclear repulsion sum both use `_standard`. Moving
a geometry (e.g. in a geometry optimization step) must update `_standard`.

### `Shell` and `ContractedView`

`Shell` stores the angular momentum type (`ShellType::S/P/D/F/G/H`), center
position in Bohr, atom index, primitive exponents, contracted coefficients (with
the contracted norm \(N_c\) pre-folded in), and per-primitive normalizations.

`ContractedView` is a lightweight reference into one Cartesian component of one
shell: it holds a pointer to its parent `Shell`, the Cartesian exponent triple
\((l_x, l_y, l_z)\), its global AO index `_index`, and a component norm factor.
The `shell_pairs` array is a flat list of all unique `(ContractedView_i,
ContractedView_j)` pairs with \(i \le j\), one entry per unique AO pair.

### `ShellPair`

Precomputed data for one pair of contracted AOs:

- `R = A - B`, `R2 = |R|^2`
- a `primitive_pairs` vector, one entry per \((\alpha_i, \beta_j)\) combination
- each `PrimitivePair` stores combined exponent \(\zeta = \alpha + \beta\),
  the Gaussian product center \(\mathbf P\), displacements \(\mathbf{PA}\) and
  \(\mathbf{PB}\), prefactor, and contracted coefficient product

The Gaussian product theorem guarantees that the product of two Gaussians on
different centers is a Gaussian on their weighted center, so precomputing these
quantities once is a large speedup.

### `Calculator`

The top-level object. Owns everything: options structs
(`OptionsSCF`, `OptionsBasis`, `OptionsGeometry`, `OptionsIntegral`,
`OptionsOutput`, `OptionsDFT`), `Molecule`, `Basis`, `DataSCF`, all integral
matrices (`_overlap`, `_hcore`, `_eri`), energies, gradient, Hessian, SAO data,
integral symmetry ops, and active-space results.

`OptionsDFT` holds the DFT-specific settings: `_grid` (grid quality enum),
`_exchange` and `_correlation` (XC functional enums), optional raw libxc integer
IDs (`_exchange_id`, `_correlation_id`), and boolean flags for SAO blocking,
grid printing, and checkpoint saving.

---

## 4. Gaussian Basis Functions

### Primitive Gaussians

A primitive Cartesian Gaussian centered at \(\mathbf A\) is:

\[
g(\mathbf r; \alpha, \mathbf A, l_x, l_y, l_z)
= (x - A_x)^{l_x}(y - A_y)^{l_y}(z - A_z)^{l_z}
  e^{-\alpha |\mathbf r - \mathbf A|^2}
\]

The total angular momentum is \(L = l_x + l_y + l_z\). For \(L=0\) there is
one s-type function; for \(L=1\) there are three p-type functions
(\(l_x l_y l_z = 100, 010, 001\)); for \(L=2\) there are six Cartesian
d-type functions, and so on. Planck uses only Cartesian Gaussians; spherical
harmonics are not supported.

### Contracted Gaussians

Real basis sets contract primitives into shells. A contracted basis function is:

\[
\chi_\mu(\mathbf r) = N_c \sum_{p=1}^{K} d_p \, N_p \,
  (x-A_x)^{l_x}(y-A_y)^{l_y}(z-A_z)^{l_z}
  e^{-\alpha_p |\mathbf r - \mathbf A|^2}
\]

where \(d_p\) are contraction coefficients, \(N_p\) is the primitive
normalization, and \(N_c\) is the contracted normalization that ensures
\(\langle \chi_\mu | \chi_\mu \rangle = 1\) for the \(s\)-type component.

**Implementation note**: Planck folds \(N_c\) into `Shell._coefficients` during
GBS reading (`shell._coefficients *= N_c`). Integral code therefore does not
apply a separate \(N_c\) factor. The per-primitive normalization \(N_p\) is
stored in `Shell._normalizations` and multiplied into `PrimitivePair.coeff_product`.

### Normalization of a Primitive

For a Cartesian Gaussian with angular momentum \((l_x, l_y, l_z)\) and exponent \(\alpha\):

\[
N_p = \left(\frac{2\alpha}{\pi}\right)^{3/4}
\left(\frac{(4\alpha)^L}{(2l_x-1)!!(2l_y-1)!!(2l_z-1)!!}\right)^{1/2}
\]

The contracted norm \(N_c\) is determined so that the \(s\)-component of the shell
integrates to 1. Each `ContractedView` stores a `_component_norm` factor equal
to \(1/\sqrt{(2l_x-1)!!(2l_y-1)!!(2l_z-1)!!}\) which handles the
\((l_x,l_y,l_z)\)-dependent part of \(N_p\) at the AO level.

---

## 5. The Obara-Saika Integral Engine

All one- and two-electron integrals in Planck are evaluated using the
Obara-Saika (OS) recursion. The key idea is to express integrals over
high-angular-momentum Gaussians in terms of simpler integrals via
horizontal and vertical recursion relations.

### Overlap Integral

The one-dimensional OS overlap table \(S(l_A, l_B)\) is seeded at
\(S(0,0) = 1\) (the Gaussian prefactor is applied by the caller), where
\(\zeta = \alpha + \beta\) and \(\mu = \alpha\beta/\zeta\). The full
\(l_A \times l_B\) table is built in three phases.

*Phase 1 — A-column* \((l_B = 0)\): increment angular momentum on center A
with \(l_B\) fixed at zero:

\[
S(l_A+1,\,0) = (P_x - A_x)\,S(l_A,\,0) + \frac{l_A}{2\zeta}\,S(l_A-1,\,0)
\]

*Phase 2 — B-row* \((l_A = 0)\): increment angular momentum on center B
with \(l_A\) fixed at zero:

\[
S(0,\,l_B+1) = (P_x - B_x)\,S(0,\,l_B) + \frac{l_B}{2\zeta}\,S(0,\,l_B-1)
\]

*Phase 3 — full table* \((l_A > 0,\; l_B > 0)\): fill remaining entries using
the general A-increment, which now has a non-zero \(l_B\) coupling term:

\[
S(l_A+1,\,l_B) = (P_x - A_x)\,S(l_A,\,l_B)
               + \frac{l_A}{2\zeta}\,S(l_A-1,\,l_B)
               + \frac{l_B}{2\zeta}\,S(l_A,\,l_B-1)
\]

The symmetric B-increment (not used in the main table fill but structurally
identical with \(A \leftrightarrow B\)) is:

\[
S(l_A,\,l_B+1) = (P_x - B_x)\,S(l_A,\,l_B)
               + \frac{l_B}{2\zeta}\,S(l_A,\,l_B-1)
               + \frac{l_A}{2\zeta}\,S(l_A-1,\,l_B)
\]

The two recursions differ only in the first term (\(P_x - A_x\) vs
\(P_x - B_x\)); the remainder terms are identical. The 3D overlap is a product
of three independent 1D tables.

In `os.cpp`, `_os_1d` evaluates the three-phase table for one Cartesian
direction.
`_compute_3d_overlap_kinetic` builds the overlap and kinetic energy for one
`ShellPair` using this recursion and the kinetic energy relation:

\[
T(l_A, l_B) = \frac{\beta(2l_B+3)}{1}\,S(l_A,l_B)
            - 2\beta^2 \,S(l_A, l_B+2)
            - \frac{l_B(l_B-1)}{2}\,S(l_A, l_B-2)
\]

The final overlap and kinetic integrals are assembled by `_compute_1e`, which
loops over all `ShellPair` entries and places results into the \(n_b \times n_b\)
matrices \(S\) and \(T\) using `sp.A._index` and `sp.B._index`.

### Nuclear Attraction Integral

The nuclear attraction integral requires the Boys function:

\[
V_{\mu\nu} = -\sum_C Z_C \langle \chi_\mu | |\mathbf r - \mathbf C|^{-1} | \chi_\nu \rangle
\]

Evaluation uses the Obara-Saika vertical recursion involving auxiliary integrals
\([0|0]^{(m)}\):

\[
[0|0]^{(m)} = \frac{2\pi}{\zeta}\, e^{-\mu R_{AB}^2}\, F_m(\zeta R_{PC}^2)
\]

where \(F_m\) is the \(m\)-th order Boys function:

\[
F_m(t) = \int_0^1 u^{2m} e^{-t u^2}\, du
\]

For large \(t\), the Boys function is computed via the asymptotic expansion
\(F_m(t) \approx (2m-1)!!/(2t)^{m+1}\sqrt{\pi/t}\). For small \(t\), a
Taylor series or Horner-scheme polynomial is used. Planck stores precomputed
Boys function tables in `src/lookup/`.

The vertical recursion for nuclear attraction auxiliary integrals:

\[
(a+1_i|0)^{(m)} = (P_i - A_i)(a|0)^{(m)} - (P_i - C_i)(a|0)^{(m+1)}
+ \frac{a_i}{2\zeta}\left[(a-1_i|0)^{(m)} - (a-1_i|0)^{(m+1)}\right]
\]

seeds at \([0|0]^{(m)}\) and builds up angular momentum.

### Two-Electron Repulsion Integrals (ERIs)

The electron repulsion integral over contracted Gaussians:

\[
(\mu\nu|\lambda\sigma) =
\iint \chi_\mu(\mathbf r_1)\chi_\nu(\mathbf r_1)
\frac{1}{r_{12}}
\chi_\lambda(\mathbf r_2)\chi_\sigma(\mathbf r_2)\,
d\mathbf r_1\, d\mathbf r_2
\]

The OS scheme splits ERI evaluation into two stages.

**Vertical Recursion Relation (VRR)**. Starting from the primitive auxiliary
integral:

\[
(ss|ss)^{(m)} = \frac{2\pi^{5/2}}{\zeta\eta\sqrt{\zeta+\eta}}\,
e^{-\mu_{AB} R_{AB}^2 - \mu_{CD} R_{CD}^2}\,
F_m\!\left(\frac{\zeta\eta}{\zeta+\eta} R_{PQ}^2\right)
\]

the VRR first builds angular momentum on bra center A (with the ket still at
zero), then on ket center C. Let δ = ζ + η, ρ = ζη/δ, and
W = (ζP + ηQ)/δ be the weighted Gaussian product center.

*A-side VRR* — increments angular momentum on center A while the ket center C
is held at angular momentum **c** (initially zero):

\[
(a+1_i\,0\,|\,c\,0)^{(m)} =
  (P_i - A_i)\,(a\,0\,|\,c\,0)^{(m)}
+ (W_i - P_i)\,(a\,0\,|\,c\,0)^{(m+1)}
+ \frac{a_i}{2\zeta}\!\left[
    (a{-}1_i\,0\,|\,c\,0)^{(m)}
  - \frac{\rho}{\zeta}(a{-}1_i\,0\,|\,c\,0)^{(m+1)}
  \right]
+ \frac{c_i}{2\delta}\,(a{-}1_i\,0\,|\,c{-}1_i\,0)^{(m+1)}
\]

*C-side VRR* — after the A-side VRR has produced \((a\,0\,|\,0\,0)^{(m)}\),
angular momentum is built on ket center C:

\[
(a\,0\,|\,c+1_i\,0)^{(m)} =
  (Q_i - C_i)\,(a\,0\,|\,c\,0)^{(m)}
+ (W_i - Q_i)\,(a\,0\,|\,c\,0)^{(m+1)}
+ \frac{c_i}{2\eta}\!\left[
    (a\,0\,|\,c{-}1_i\,0)^{(m)}
  - \frac{\rho}{\eta}(a\,0\,|\,c{-}1_i\,0)^{(m+1)}
  \right]
+ \frac{a_i}{2\delta}\,(a{-}1_i\,0\,|\,c{-}1_i\,0)^{(m+1)}
\]

The A-side and C-side recurrences are structurally symmetric: P↔Q, A↔C, ζ↔η.
The cross-coupling term \(a_i/2\delta\) in the C-side VRR is non-zero because
the A-side VRR has already built up nonzero bra angular momentum a by that
point; the analogous \(c_i/2\delta\) term in the A-side VRR is zero when the
A-side is applied first (c = 0 then).

**Horizontal Recursion Relation (HRR)**. After the VRR produces
\((a\,0\,|\,c\,0)\) integrals, angular momentum is transferred to the second
center of each shell-pair without re-running the VRR. The bra transfer (A→B):

\[
(a\,b\,|\,cd) = (a+1_i\,b-1_i\,|\,cd) + (A_i - B_i)\,(a\,b-1_i\,|\,cd)
\]

and the symmetric ket transfer (C→D):

\[
(ab\,|\,c\,d) = (ab\,|\,c+1_i\,d-1_i) + (C_i - D_i)\,(ab\,|\,c\,d-1_i)
\]

In the implementation the same three-phase sweeping routine (`_nuclear_hrr`) is
reused for both transfers; the C→D pass operates on the fixed A-side slice
extracted after the A→B pass completes.

**Thread-local scratch buffers**. The VRR and HRR accumulators are large
temporary arrays. Planck uses `thread_local` static arrays (`_vrr_buf`,
`_hrr_buf`) so that each OpenMP thread has its own workspace without heap
allocation per quartet.

### Schwarz Screening

Before evaluating a quartet, the Schwarz inequality provides an upper bound:

\[
|(\mu\nu|\lambda\sigma)| \le \sqrt{(\mu\nu|\mu\nu)}\,\sqrt{(\lambda\sigma|\lambda\sigma)}
\]

Planck precomputes the Schwarz table \(Q(i,j) = \sqrt{|(ij|ij)|}\) for all
unique diagonal pairs and skips any quartet where:

\[
Q(i,j) \cdot Q(k,l) < \epsilon_{ERI}
\]

This screening is applied in `_compute_2e` and `_compute_2e_fock` in `os.cpp`
and analogously in `rys.cpp`.

### Permutation Symmetry of the ERI Tensor

The ERI tensor has 8-fold permutation symmetry:

\[
(\mu\nu|\lambda\sigma) = (\nu\mu|\lambda\sigma) = (\mu\nu|\sigma\lambda)
= (\nu\mu|\sigma\lambda) = (\lambda\sigma|\mu\nu) = \cdots
\]

Planck iterates only over pairs \(p \le q\) in the AO-pair index and fills all
8 equivalent slots after each computation, reducing work by a factor of 8.

---

## 6. Rys Quadrature

### The Basic Idea

The Obara-Saika VRR builds an \((L+1)\)-deep stack of auxiliary integrals at
each auxiliary order \(m\). For high-angular-momentum quartets (d+d, f+p, …)
the stack grows large, and intermediate storage dominates the cost. The Rys
quadrature method avoids auxiliary-order recursion entirely by converting the
Boys function integral into a discrete sum:

\[
F_m(T) = \int_0^1 t^{2m}\, e^{-T t^2}\, dt
       = \sum_{r=1}^{n} w_r(T)\, \bigl[t_r^2(T)\bigr]^m
\]

where \(\{t_r^2, w_r\}\) are the Rys roots (squared) and weights, which depend
on the Boys argument \(T = \rho\,|\mathbf{PQ}|^2\). When this representation
is substituted into the ERI expression, the integral factorizes into independent
1D integrals in \(x\), \(y\), and \(z\) for each quadrature point \(r\).

### Number of Roots

For a quartet with total angular momentum \(L = l_A + l_B + l_C + l_D\), the
exact number of Rys roots required is:

\[
n = \left\lfloor \frac{L}{2} \right\rfloor + 1
\]

Planck supports up to \(n = 11\) roots (`RYS_MAX_ROOTS`), corresponding to
\(L \le 20\) (two H-shells). For S, P, D, F, G shells the root counts are:

| Shell quartet | L | Roots |
|---|---|---|
| (ss∣ss) | 0 | 1 |
| (sp∣ss) | 1 | 1 |
| (pp∣ss) | 2 | 2 |
| (pp∣pp) | 4 | 3 |
| (dd∣ss) | 4 | 3 |
| (dd∣pp) | 6 | 4 |
| (dd∣dd) | 8 | 5 |
| (ff∣dd) | 10 | 6 |

### Root Finding

Computing the roots and weights for a given \(T\) is the central numerical
challenge. Planck uses two strategies, selected by `rys_roots_weights` in
`src/integrals/rys_roots.cpp`:

**T ≈ 0 (Gauss-Legendre limit)**: As \(T \to 0\) the weight function
\(e^{-Tt^2} \to 1\), so the Rys quadrature degenerates to the standard
Gauss-Legendre rule on \([0,1]\). Pre-tabulated GL roots and weights for
\(n = 1, \ldots, 11\) are stored in `gl_roots` and `gl_weights` and used
directly when \(T < 10^{-14}\).

**T > 0 (Stieltjes–Jacobi procedure)**: For non-zero \(T\) the Rys
measure is \(e^{-Tt^2} dt\) on \([0,1]\). The roots and weights are obtained
by building the three-term recurrence (Jacobi) matrix of the orthogonal
polynomial family with respect to this measure. The algorithm:

1. Compute \(2n+1\) Boys moments
   \(F_m(T) = \int_0^1 t^{2m} e^{-Tt^2} dt\) in long double precision.
2. Construct orthonormal polynomials via the Gram-Schmidt Stieltjes procedure,
   recording the diagonal (\(\alpha_k\)) and sub-diagonal (\(\beta_k\)) entries
   of the symmetric \(n \times n\) Jacobi matrix \(\mathbf J\).
3. Diagonalize \(\mathbf J\) using Eigen's `SelfAdjointEigenSolver`. The
   eigenvalues are the Rys roots \(t_r^2\); the weight for root \(r\) is
   \(w_r = F_0(T) \cdot V_{0r}^2\) where \(V_{0r}\) is the first component of
   the \(r\)-th eigenvector. This is the Golub–Welsch formula.
4. If the Gram-Schmidt procedure encounters a degenerate norm, the algorithm
   falls back to the Gauss-Legendre table.

### The Rys 1D VRR

For each Rys root \(u = t_r^2\), the ERI factorizes into three independent 1D
integrals. Each 1D table \(I[a][c]\) is filled by its own three-term recursion:

\[
I[0][0] = 1 \qquad\text{(seed)}
\]

*Bra increment* (\(c = 0\)):

\[
I[a+1][0] = C_{00}\, I[a][0] + a\, B_{10}\, I[a-1][0]
\]

*Ket increment* (general \(a\)):

\[
I[a][c+1] = D_{00}\, I[a][c] + c\, B_{01}\, I[a][c-1] + a\, B_{00}\, I[a-1][c-1]
\]

The root-dependent coefficients are:

\[
B_{00} = \frac{u}{2\delta}, \qquad
B_{10} = \frac{1}{2\zeta} - B_{00}, \qquad
B_{01} = \frac{1}{2\eta}  - B_{00}
\]

\[
C_{00} = (P_q - A_q) + u\,(W_q - P_q), \qquad
D_{00} = (Q_q - C_q) + u\,(W_q - Q_q)
\]

where \(\delta = \zeta + \eta\), \(\mathbf W = (\zeta\mathbf P + \eta\mathbf Q)/\delta\) is the
weighted Gaussian product center, and \(q\) is the Cartesian direction. As
\(u \to 0\) the root sits at the A-center (\(C_{00} \to P_q - A_q\)); as
\(u \to 1\) the root sits at the W-center. These recurrences are implemented
in `_rys_vrr_1d` in `src/integrals/rys.cpp`.

### 6D Accumulation and HRR

After running `_rys_vrr_1d` for all three Cartesian directions, the 3D outer
product is accumulated into a six-index buffer:

\[
W[a_x][a_y][a_z][c_x][c_y][c_z]
  \mathrel{+}= w_r \cdot I_x[a_x][c_x] \cdot I_y[a_y][c_y] \cdot I_z[a_z][c_z]
\]

This sum runs over all \(n\) roots. After the root loop the buffer holds
\((a\,0\,|\,c\,0)\) intermediates analogous to those produced by the OS VRR.
The thread-local six-index array `_rys_sum_buf[13][13][13][13][13][13]`
is the Rys counterpart of `_vrr_buf` in `os.cpp`.

Angular momentum is then transferred to the second center of each shell pair
using the same HRR as the OS path:

- **AB-HRR** (`_rys_hrr_ab`): \((a+1_i\,b-1_i\,|\,c\,d) + (A_i - B_i)(a\,b-1_i\,|\,c\,d)\)
- **CD-HRR** (`_rys_hrr_cd`): operates on the 3D slice extracted at \((l_A, l_A, l_A)\)
  after the AB sweep.

The contracted ERI is obtained by summing the primitive results over all
\((\alpha, \beta)\) and \((\gamma, \delta)\) primitive pairs inside
`_rys_contracted_eri`.

### Auto-Dispatch: OS vs. Rys Cost Model

Planck's `auto` engine mode selects OS or Rys per contracted shell quartet using
an analytic operation-count estimate (`_auto_prefers_rys` in `rys.cpp`).

Define:

\[
\text{six\_d} = (l_{AB,x}+1)(l_{AB,y}+1)(l_{AB,z}+1)
                (l_{CD,x}+1)(l_{CD,y}+1)(l_{CD,z}+1)
\]

This counts the number of entries in the 6D accumulation buffer. The estimated
flop counts are:

\[
W_{\text{OS}}  = \text{six\_d}\cdot(L+1)
               + (l_B + l_D + 1)\cdot\text{six\_d}\cdot 0.25
\]

\[
W_{\text{Rys}} = \text{six\_d}\cdot n
               + (l_B + l_D + 1)\cdot\text{six\_d}\cdot 0.20
               + 24\cdot n
\]

where \(n = \lfloor L/2 \rfloor + 1\) is the number of Rys roots and the
constant 24 accounts for the per-root overhead of root finding and 1D
coefficient computation. Rys is preferred when \(W_{\text{Rys}} < W_{\text{OS}}\).

For a (dd|dd) quartet: \(L = 8\), \(n = 5\),
\(\text{six\_d} = 3^4 \cdot 3^4 = \ldots\), and the Rys path wins because the
OS stack grows as \(L+1 = 9\) deep while Rys only needs 5 root evaluations.
For (ss|ss) through (sp|sp) the OS path is cheaper. The empirical crossover is
around \(L = 4\) (constant `RYS_CROSSOVER_L`). The `_auto_contracted_eri`
wrapper dispatches to `_rys_contracted_eri` or `ObaraSaika::_contracted_eri_elem`
at this level.

### Implementation Files

| File | Role |
|---|---|
| `src/integrals/rys.h` | Public API: `_compute_2e`, `_compute_2e_fock`, `_compute_2e_fock_uhf`, and `_auto` variants |
| `src/integrals/rys.cpp` | VRR (`_rys_vrr_1d`), HRR (`_rys_hrr_ab`, `_rys_hrr_cd`), primitive and contracted ERI, Schwarz table, Fock builders, auto-dispatch |
| `src/integrals/rys_roots.h` | `rys_roots_weights` declaration; exact 1-point formula `rys_1pt` |
| `src/integrals/rys_roots.cpp` | Pre-tabulated GL rules; Boys moment recursion; Stieltjes–Jacobi Gram-Schmidt + Eigen eigendecomposition |

---

## 7. Hartree-Fock Theory

### The Variational Principle

HF approximates the ground state \(|\Psi\rangle\) as a single Slater
determinant built from \(N\) molecular spin-orbitals:

\[
|\Psi_{HF}\rangle = |\phi_1 \phi_2 \cdots \phi_N\rangle
\]

The HF energy is:

\[
E_{HF} = \langle \Psi_{HF} | \hat H | \Psi_{HF} \rangle
= \sum_i h_{ii} + \frac{1}{2}\sum_{ij}(J_{ij} - K_{ij})
\]

where \(h_{ii}\) are core one-electron energies, \(J_{ij}\) are Coulomb
integrals, and \(K_{ij}\) are exchange integrals.

### Roothaan Equations (RHF)

Expanding spatial MOs in the AO basis \(\chi_\mu\):

\[
\phi_i(\mathbf r) = \sum_\mu C_{\mu i}\, \chi_\mu(\mathbf r)
\]

and applying the variational condition yields the Roothaan matrix eigenvalue problem:

\[
\mathbf F \mathbf C = \mathbf S \mathbf C \boldsymbol\varepsilon
\]

where \(\mathbf S\) is the AO overlap matrix, \(\mathbf C\) contains MO
coefficients (columns = MOs), and \(\boldsymbol\varepsilon\) contains orbital
energies.

The **Fock matrix** is:

\[
F_{\mu\nu} = H_{\mu\nu}^{core} + G_{\mu\nu}
\]

The core Hamiltonian is:

\[
H_{\mu\nu}^{core} = T_{\mu\nu} + V_{\mu\nu}
\]

The two-electron contribution for closed-shell RHF is:

\[
G_{\mu\nu} = \sum_{\lambda\sigma} P_{\lambda\sigma}
\left[(\mu\nu|\lambda\sigma) - \frac{1}{2}(\mu\lambda|\nu\sigma)\right]
\]

The **density matrix** for \(n_{occ}\) doubly-occupied orbitals is:

\[
P_{\mu\nu} = 2\sum_{i=1}^{n_{occ}} C_{\mu i}\, C_{\nu i}
\]

The total RHF energy is:

\[
E_{RHF} = \frac{1}{2}\sum_{\mu\nu} P_{\mu\nu}\left(H_{\mu\nu}^{core} + F_{\mu\nu}\right)
+ E_{nuc}
\]

### Pople-Nesbet Equations (UHF)

For open-shell systems, separate spin-orbital sets are maintained. Defining
\(P^\alpha_{\mu\nu} = \sum_{i \in \alpha} C^\alpha_{\mu i} C^\alpha_{\nu i}\)
and similarly for \(\beta\), the total density is
\(P^T = P^\alpha + P^\beta\). The UHF Fock matrices are:

\[
F^\alpha_{\mu\nu} = H^{core}_{\mu\nu}
+ \sum_{\lambda\sigma}\left[P^T_{\lambda\sigma}(\mu\nu|\lambda\sigma)
- P^\alpha_{\lambda\sigma}(\mu\lambda|\nu\sigma)\right]
\]

\[
F^\beta_{\mu\nu} = H^{core}_{\mu\nu}
+ \sum_{\lambda\sigma}\left[P^T_{\lambda\sigma}(\mu\nu|\lambda\sigma)
- P^\beta_{\lambda\sigma}(\mu\lambda|\nu\sigma)\right]
\]

The UHF energy is:

\[
E_{UHF} = \frac{1}{2}\sum_{\mu\nu}
\left[P^T_{\mu\nu} H^{core}_{\mu\nu}
+ P^\alpha_{\mu\nu} F^\alpha_{\mu\nu}
+ P^\beta_{\mu\nu} F^\beta_{\mu\nu}\right]
+ E_{nuc}
\]

---

## 8. SCF Algorithm

### Symmetric Orthogonalization

The AO basis is non-orthogonal (\(\mathbf S \ne \mathbf I\)). To diagonalize
the Fock matrix, it is transformed to an orthonormal basis using:

\[
\mathbf X = \mathbf S^{-1/2}
\]

computed via the eigendecomposition \(\mathbf S = \mathbf U \boldsymbol\sigma \mathbf U^T\):

\[
\mathbf X = \mathbf U\,\mathrm{diag}(\sigma_i^{-1/2})\,\mathbf U^T
\]

The transformed Fock matrix is then:

\[
\mathbf F' = \mathbf X^T \mathbf F \mathbf X
\]

which is a standard symmetric eigenvalue problem \(\mathbf F' \mathbf C' = \mathbf C' \boldsymbol\varepsilon\).
The AO-basis MO coefficients are recovered by:

\[
\mathbf C = \mathbf X \mathbf C'
\]

Implemented in `build_orthogonalizer` in `src/scf/scf.cpp`.

### Initial Density Guess

The default initial guess (`SCFGuess::HCore`) diagonalizes the core
Hamiltonian \(\mathbf H^{core} = \mathbf T + \mathbf V\) to produce an initial
set of MO coefficients and a starting density matrix. This corresponds to
completely neglecting electron-electron repulsion in the initial guess.

### SCF Iteration Loop

Each RHF iteration in `run_rhf`:

1. Compute \(G_{\mu\nu}[P]\) from the current density (either from the stored ERI
   tensor via `_compute_fock_rhf`, or on-the-fly via `_compute_2e_fock`)
2. Form \(\mathbf F = \mathbf H^{core} + \mathbf G\)
3. Compute the current energy
4. Form the DIIS error vector and call `diis.push(F, e)`; if DIIS is ready,
   replace \(\mathbf F\) with the extrapolated Fock
5. Transform to orthonormal basis: \(\mathbf F' = \mathbf X^T \mathbf F \mathbf X\)
6. Diagonalize \(\mathbf F' \mathbf C' = \mathbf C' \boldsymbol\varepsilon\)
7. Back-transform: \(\mathbf C = \mathbf X \mathbf C'\)
8. Build new density \(P_{\mu\nu} = 2\sum_i^{occ} C_{\mu i} C_{\nu i}\)
9. Test convergence: \(|\Delta E| < \epsilon_E\) and \(\|\Delta P\|_{max} < \epsilon_P\)

Convergence is declared when both criteria are simultaneously satisfied.

### Level Shifting

If `_level_shift > 0`, the virtual orbital energies are shifted upward by
\(\Delta\) before each diagonalization:

\[
F'_{ab} \leftarrow F'_{ab} + \Delta
\quad \text{(virtual-virtual block in the MO basis)}
\]

This increases the HOMO-LUMO gap and prevents the SCF from alternating between
states with different orbital occupations, at the cost of slower convergence
near the solution. Level shift is applied in `run_rhf` and `run_uhf`.

---

## 9. DIIS Convergence Acceleration

Pulay's Direct Inversion in the Iterative Subspace (DIIS) accelerates SCF
convergence by extrapolating a Fock matrix from a stored subspace of recent
Fock matrices that minimizes the residual.

### Error Metric

The Pulay error vector at iteration \(k\) is the FPS-SPF commutator in the
orthonormal basis:

\[
\mathbf e_k = \mathbf X^T(\mathbf F_k \mathbf P_k \mathbf S - \mathbf S \mathbf P_k \mathbf F_k)\mathbf X
\]

When the SCF is converged, \(\mathbf F\) and \(\mathbf P\) commute and
\(\mathbf e = \mathbf 0\). The norm \(\|\mathbf e\|_{RMS}\) is the primary
convergence diagnostic.

### DIIS Linear System

Given \(m\) stored pairs \(\{(\mathbf F_k, \mathbf e_k)\}\), find coefficients
\(\{c_k\}\) such that:

\[
\mathbf F^{extrap} = \sum_{k=1}^m c_k \mathbf F_k
\quad \text{subject to} \quad \sum_k c_k = 1
\]

minimizes \(\|\sum_k c_k \mathbf e_k\|^2\). Using a Lagrange multiplier
\(\lambda\) for the constraint, this becomes the augmented linear system:

\[
\begin{pmatrix} \mathbf B & -\mathbf 1 \\ -\mathbf 1^T & 0 \end{pmatrix}
\begin{pmatrix} \mathbf c \\ \lambda \end{pmatrix}
=
\begin{pmatrix} \mathbf 0 \\ -1 \end{pmatrix}
\]

where \(B_{ij} = \mathrm{Tr}(\mathbf e_i^T \mathbf e_j)\).

Solved via column-pivoted QR decomposition in Eigen. Implemented in the
`DIISState::extrapolate()` method in `src/base/types.h`. Subspace is capped at
`_DIIS_dim` (default 8) vectors, evicting the oldest on overflow.

---

## 10. Symmetry

### Point Group Detection

The `detectSymmetry` function in `src/symmetry/symmetry.cpp` wraps the
`libmsym` library to:

1. Identify the molecular point group
2. Reorient the molecule into the standard frame (principal axis along \(z\), etc.)
3. Store the standard-orientation geometry in `molecule._standard` (Angstrom)
   and `molecule._standard * ANGSTROM_TO_BOHR` (Bohr)

### Symmetry-Adapted Orbitals (SAO Basis)

For non-trivial point groups, the Fock matrix is block-diagonal in the
symmetry-adapted orbital (SAO) basis. `build_sao_basis` in
`src/symmetry/mo_symmetry.cpp`:

1. Re-enters libmsym to obtain the character table and group operations
2. For groups with multi-dimensional irreps, selects the **largest Abelian
   subgroup** with all-1D irreducible representations (at most D\(_{2h}\))
3. Builds projection operators for each irrep \(\Gamma_g\):
   \[
   \hat P^{(\Gamma)} = \frac{d_\Gamma}{h} \sum_{R} \chi^{(\Gamma)}(R)^* \hat R
   \]
4. Applies these to each AO to generate SAO trial vectors; orthonormalizes via
   modified Gram-Schmidt

The resulting unitary transformation \(\mathbf U\) (columns = SAOs) is stored
in `calculator._sao_transform`. In the SAO basis, the Fock and overlap matrices
block-diagonalize, and each block is diagonalized independently. This reduces
the \(O(n_b^3)\) diagonalization cost to \(\sum_g O(n_g^3)\) where \(n_g\) is
the number of SAOs in irrep \(g\).

### MO Irrep Assignment

After convergence, each MO is labeled by its irreducible representation.
`assign_mo_symmetry` in `mo_symmetry.cpp` builds the AO representation matrix
\(D_R\) for each group operation \(R\) — for the all-1D Abelian subgroups used,
each Cartesian Gaussian transforms with a sign \(\pm 1\) under each operation,
so \(D_R\) is diagonal. The irrep label of MO \(i\) is determined by finding
the character pattern \(\chi_i(R) = \sum_\mu |C_{\mu i}|^2 D_R(\mu,\mu)\) and
matching it against the character table.

### Integral Symmetry Reduction

The `update_integral_symmetry` function in `src/symmetry/integral_symmetry.cpp`
finds which of the seven axis-sign-flip candidates
\(\{(-1,1,1),(1,-1,1),(1,1,-1),(-1,-1,1),(-1,1,-1),(1,-1,-1),(-1,-1,-1)\}\)
are true symmetry operations of the molecule. Each valid operation is stored as
a `SignedAOSymOp` — a permutation `ao_map[mu] = nu` and sign `ao_sign[mu] = ±1`
that maps each AO to its symmetry-equivalent partner.

Since the Abelian subgroups used are subgroups of D\(_{2h}\), all operations
are products of coordinate-axis reflections. Under any such reflection, a
Cartesian Gaussian \(x^{l_x} y^{l_y} z^{l_z} e^{-\alpha r^2}\) maps to
\(\pm 1\) times a Gaussian on the equivalent atom — no mixing of Cartesian
components occurs. This means `ao_sign ∈ {+1, -1}` is always exact for all
angular momenta.

These operations are used to reduce integral work in the ERI loops (described
further in the implementation plan).

---

## 11. MP2 Correlation Energy

### Second-Order Perturbation Theory

Møller-Plesset perturbation theory partitions the Hamiltonian as
\(\hat H = \hat F + \hat V'\), where \(\hat F = \sum_i \hat f_i\) is the Fock
operator and \(\hat V'\) is the fluctuation potential. The second-order energy
correction is:

\[
E^{(2)} = \sum_{ijab} \frac{|\langle ij || ab \rangle|^2}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
\]

where \(i, j\) label occupied and \(a, b\) label virtual orbitals, and
\(\langle ij || ab \rangle\) are antisymmetrized two-electron integrals in the
MO basis.

### RMP2 (Closed-Shell)

For RHF reference, using spatial orbitals and factoring out spin:

\[
E_{RMP2} = \sum_{i \le j}^{occ} \sum_{a \le b}^{virt}
\frac{(ia|jb)\left[2(ia|jb) - (ib|ja)\right]}
{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
\]

where \((ia|jb)\) are MO-basis ERIs obtained by the AO→MO four-index
transformation:

\[
(ia|jb) = \sum_{\mu\nu\lambda\sigma} C_{\mu i} C_{\nu a} (\mu\nu|\lambda\sigma) C_{\lambda j} C_{\sigma b}
\]

Planck performs this transformation via a sequence of half-transformations
(AO→MO in bra, then AO→MO in ket) to reduce the \(O(n^8)\) naive cost to
\(O(n^5)\). Implemented in `src/post_hf/integrals.cpp`.

### UMP2 (Open-Shell)

For UHF reference, same-spin (SS) and opposite-spin (OS) channels are computed
separately:

\[
E_{UMP2}^{SS} = -\frac{1}{4}\sum_{ijab}
\frac{|\langle ij||ab\rangle_{\alpha\alpha}|^2 + |\langle ij||ab\rangle_{\beta\beta}|^2}
{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
\]

\[
E_{UMP2}^{OS} = -\sum_{i^\alpha j^\beta a^\alpha b^\beta}
\frac{|\langle i^\alpha j^\beta|a^\alpha b^\beta\rangle|^2}
{\varepsilon_{i^\alpha} + \varepsilon_{j^\beta} - \varepsilon_{a^\alpha} - \varepsilon_{b^\beta}}
\]

Implemented in `run_ump2` in `src/post_hf/mp2.cpp`.

### RMP2 Natural Orbitals

After the correlation energy is computed, Planck diagonalizes the unrelaxed RMP2 one-particle density matrix to produce **natural orbitals** (NOs) and their occupation numbers. The unrelaxed density is block-diagonal in the canonical MO basis:

\[
\gamma^{MP2}_{pq} = \begin{cases}
2\delta_{ij} + P^{occ}_{ij} + P^{occ}_{ji} & p,q \in \text{occupied} \\
P^{virt}_{ab} + P^{virt}_{ba} & p,q \in \text{virtual} \\
0 & \text{otherwise (unrelaxed)}
\end{cases}
\]

where \(P^{occ}\) and \(P^{virt}\) are the occupied-occupied and virtual-virtual MP2 density corrections assembled in `build_rmp2_unrelaxed_density`. The symmetrized matrix is diagonalized by `compute_rmp2_natural_orbitals` in `src/post_hf/mp2.cpp` using Eigen's `SelfAdjointEigenSolver`. Eigenvalues are sorted in descending order; eigenvectors give the canonical-MO → natural-orbital rotation. The AO-basis coefficients are obtained by left-multiplying with the HF MO coefficient matrix:

\[
\mathbf C^{NO}_{AO} = \mathbf C^{HF}_{AO} \cdot \mathbf U^{MO \to NO}
\]

The result struct `RMP2NaturalOrbitals` carries three fields: `occupations` (descending eigenvalues), `coefficients_mo` (the rotation \(\mathbf U\)), and `coefficients_ao`. Occupation numbers near 2 indicate strongly occupied HF-like NOs; values of 0.01–0.1 indicate correlation-driven virtual occupation. These can guide active-space selection for a subsequent CASSCF calculation.

---

## 12. Analytic Nuclear Gradients

### Hellmann-Feynman Theorem and Pulay Forces

For a variational wavefunction, the nuclear gradient has the
Hellmann-Feynman form only when the basis is complete. For finite atom-centered
Gaussian basis sets, the basis functions move with the nuclei, introducing
**Pulay forces** — additional terms arising from the nuclear-coordinate
dependence of the basis.

The full RHF energy gradient with respect to nuclear coordinate \(X_A\) is:

\[
\frac{dE}{dX_A} =
\sum_{\mu\nu} P_{\mu\nu} \frac{\partial H^{core}_{\mu\nu}}{\partial X_A}
+ \frac{1}{2}\sum_{\mu\nu\lambda\sigma} \Gamma_{\mu\nu\lambda\sigma}
\frac{\partial(\mu\nu|\lambda\sigma)}{\partial X_A}
- \sum_{\mu\nu} W_{\mu\nu} \frac{\partial S_{\mu\nu}}{\partial X_A}
+ \frac{\partial E_{nuc}}{\partial X_A}
\]

where:

- \(P_{\mu\nu}\) is the one-particle density matrix
- \(\Gamma_{\mu\nu\lambda\sigma} = 2P_{\mu\nu}P_{\lambda\sigma} - P_{\mu\lambda}P_{\nu\sigma}\) is the two-particle density for RHF
- \(W_{\mu\nu} = \sum_{i}^{occ} \varepsilon_i C_{\mu i} C_{\nu i}\) is the
  **energy-weighted density matrix** (the Pulay term coefficient)

### Derivative Integrals

The derivative of the overlap integral with respect to center \(\mathbf A\):

\[
\frac{\partial S_{\mu\nu}}{\partial A_x}
= l_{Ax}\, S(l_{Ax}-1, l_{Bx}; \ldots) - 2\alpha\, S(l_{Ax}+1, l_{Bx}; \ldots)
\]

by the Gaussian angular-momentum shift rule. Similarly for kinetic, nuclear
attraction, and ERI derivative integrals. These are computed in
`_compute_1e_deriv_A`, `_compute_nuclear_deriv_A_elem`,
`_compute_nuclear_deriv_C_elem`, and `_compute_eri_deriv_elem` in `os.cpp`.

The gradient assembly loops over all contributing shell pairs/quartets,
contracts the derivative integrals against the appropriate density matrices,
and accumulates into the \(N_{atoms} \times 3\) gradient array.
Implemented in `compute_rhf_gradient` in `src/gradient/gradient.cpp`.

### UHF Gradient

The UHF gradient has the same structure but uses the total density
\(P^T = P^\alpha + P^\beta\) for the Coulomb part and separate \(P^\alpha\),
\(P^\beta\) for the spin-specific exchange. The energy-weighted density is:

\[
W_{\mu\nu} = \sum_{i}^{\alpha,occ} \varepsilon^\alpha_i C^\alpha_{\mu i} C^\alpha_{\nu i}
           + \sum_{i}^{\beta,occ}  \varepsilon^\beta_i  C^\beta_{\mu i}  C^\beta_{\nu i}
\]

---

## 13. Coupled-Perturbed HF and the MP2 Gradient

### The Z-Vector Method

The RMP2 energy gradient requires the response of the HF density to a nuclear
perturbation. Rather than solving the full CPHF equations for every nuclear
displacement (which would scale as \(O(3N \cdot n_b^3)\)), the Z-vector method
(Handy and Schaefer, 1984) collapses the response into a single solve.

The unrelaxed MP2 one-particle density is:

\[
\tilde P_{\mu\nu} = P^{HF}_{\mu\nu} + D^{MP2}_{\mu\nu}
\]

where \(D^{MP2}\) contains the orbital-response correction from the second-order
amplitudes:

\[
t_{ij}^{ab} = \frac{(ia|jb)}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
\]

The relaxed density is obtained from the Z-vector equation:

\[
\sum_{bj} A_{ai,bj} Z_{bj} = L_{ai}
\]

where \(\mathbf A\) is the orbital Hessian (also the CPHF matrix) and
\(\mathbf L\) is the MP2 Lagrangian source term. `build_rhf_cphf_matrix` in
`src/post_hf/rhf_response.cpp` builds \(\mathbf A\). The final gradient is then
assembled from the relaxed density and the appropriate derivative integrals.

---

## 14. CASSCF and RASSCF

### Motivation

Hartree-Fock fails near bond breaking, in transition metal chemistry, and
wherever a single Slater determinant is qualitatively wrong. CASSCF (Complete
Active Space SCF) partitions orbitals into:

- **inactive** — doubly occupied, excluded from CI
- **active** — partially occupied, included in CI
- **virtual** — unoccupied, excluded from CI

The wavefunction is a full CI expansion within the active space:

\[
|\Psi_{CASSCF}\rangle = \sum_I c_I |D_I\rangle
\]

where \(|D_I\rangle\) ranges over all Slater determinants formed by distributing
the active electrons among the active orbitals.

### Determinant Representation

Each determinant is stored as a pair of 64-bit integers (one per spin), where
bit \(k\) indicates occupation of active orbital \(k\). These `CIString = uint64_t`
bitmasks allow efficient generation of all \(\binom{n_{act}}{n_\alpha}\) alpha
strings and \(\binom{n_{act}}{n_\beta}\) beta strings via Gosper's algorithm
(which enumerates all integers with exactly \(k\) set bits in ascending order).

### CI Hamiltonian Matrix-Vector Product

The CI energy and gradient require the Hamiltonian acting on a CI vector,
\(\mathbf H \mathbf c\). Matrix elements between determinants \(|D_I\rangle\)
and \(|D_J\rangle\) are evaluated using Slater-Condon rules:

- **Zero excitation** (\(|D_I\rangle = |D_J\rangle\)):
  \(H_{II} = \sum_i h_{ii} + \frac{1}{2}\sum_{ij}(2J_{ij} - K_{ij})\) over occupied active MOs

- **Single excitation** (\(|D_I\rangle\) and \(|D_J\rangle\) differ by one orbital):
  \(H_{IJ} = \langle I|\hat h|J\rangle \pm \text{exchange terms}\)

- **Double excitation**: pure two-electron term involving \((ij|kl)\)

For large active spaces, the CI problem is solved iteratively using the
**Davidson algorithm**: build a small Krylov subspace, diagonalize the projected
Hamiltonian, and extend until convergence. For smaller spaces, full
diagonalization via Eigen is used.

### Reduced Density Matrices

The orbital gradient requires the one- and two-particle reduced density matrices
(RDMs) of the CI wavefunction.

**1-RDM**:
\[
D_{pq} = \langle \Psi | \hat a^\dagger_p \hat a_q | \Psi \rangle
= \sum_{IJ} c_I c_J \langle D_I | \hat a^\dagger_p \hat a_q | D_J \rangle
\]

**2-RDM**:
\[
d_{pqrs} = \langle \Psi | \hat a^\dagger_p \hat a^\dagger_r \hat a_s \hat a_q | \Psi \rangle
\]

These are assembled in `compute_1rdm` and `compute_2rdm` by looping over string
pairs and applying creation/annihilation operators via bitmask arithmetic.

### Orbital Gradient and Generalized Fock Matrix

The CASSCF orbital gradient \(\mathbf g = \partial E / \partial \boldsymbol\kappa\)
with respect to orbital rotation parameters \(\kappa_{pq}\) is:

\[
g_{pq} = 2(F^{gen}_{pq} - F^{gen}_{qp})
\]

where the generalized Fock matrix is:

\[
F^{gen}_{pq} = \sum_r h_{pr} D_{rq} + \sum_{rst} (pr|st) d_{qrst}
\]

This is computed via two AO→MO half-transformations of the four-index integral
tensor contracted with the 2-RDM.

### Orbital Update: Augmented-Hessian Step and Cayley Transform

Orbital rotations are parameterized by an antisymmetric matrix \(\mathbf \kappa\)
and applied as a unitary transformation via the Cayley map:

\[
\mathbf C_{\text{new}} = \mathbf C_{\text{old}}
\left(\mathbf I - \frac{\boldsymbol\kappa}{2}\right)^{-1}
\left(\mathbf I + \frac{\boldsymbol\kappa}{2}\right)
\]

This approximates \(\mathbf C_{\text{old}}\,e^{\boldsymbol\kappa}\) to second order without
computing a matrix exponential. After the Cayley step, a Löwdin symmetric
re-orthogonalization restores exact S-orthonormality in the AO metric.

The rotation matrix \(\boldsymbol\kappa\) comes from an **augmented-Hessian step**
(`augmented_hessian_step`): the gradient is preconditioned by the diagonal orbital
Hessian with a level shift, symmetry-forbidden blocks are zeroed, and the step
length is capped at \(|\boldsymbol\kappa|_{\max} \le 0.20\) to stay in the
Cayley-transform validity regime.

### Macro-Iteration Structure

The full CASSCF macro-iteration (one pass of `run_casscf`):

1. Form one-electron integrals in the current MO basis (transform \(h_{\mu\nu}\))
2. Form active-active two-electron integrals from AO ERIs; cache the mixed-basis
   puvw tensor (`build_active_integral_cache`)
3. Solve CI eigenproblem to get \(\{c_I^{(r)}\}\) and \(\{E_{CI}^{(r)}\}\) for all roots;
   reorder roots by maximum CI-vector overlap to prevent state flipping
4. Compute per-root 1-RDM and 2-RDM; form state-averaged \(\bar{\gamma}\) and
   \(\bar{\Gamma}\) weighted by the SA weights
5. Compute inactive Fock \(F^I\), active Fock \(F^A\), Q matrix, and orbital
   gradient \(\mathbf g\)
6. Run micro-iterations: for each micro-step,
   a. Form an augmented-Hessian orbital step \(\boldsymbol\kappa\)
   b. Compute the first-order CI response per root (`solve_ci_response_davidson`)
      to account for the change in CI coefficients driven by \(\delta h_{\text{eff}} = [{\boldsymbol\kappa}, F^I]_{\text{act}}\)
   c. Update the gradient with the response correction (`fep1_gradient_update` + CI contribution)
   d. Accumulate the total rotation \(\boldsymbol\kappa_{\text{total}}\)
7. Select the best orbital step from a set of candidates (augmented-Hessian
   accumulated step, first micro-step only, gradient fallback, and pairwise
   averages) using a merit function
   \(m = E_{\text{CAS}} + w\,\|\mathbf g\|^2\).
   Numeric Newton is available only via the `mcscf_debug_numeric_newton` flag
   and is not part of the normal production path.
8. Apply the accepted \(\boldsymbol\kappa\) via the Cayley transform followed by
   Löwdin re-orthogonalization: \(\mathbf C \leftarrow \mathbf C\,\mathbf U\)
9. Check convergence: \(\|\mathbf g\| < \epsilon_{\text{grad}}\) and
   \(|\Delta E| < \epsilon_E\)

### State-Averaged CASSCF

When `nroots > 1`, the driver performs **state-averaged CASSCF** (SA-CASSCF).
Per-root CI vectors \(\{c^{(r)}_I\}\), energies \(E^{(r)}\), 1-RDMs \(\gamma^{(r)}\),
and 2-RDMs \(\Gamma^{(r)}\) are computed independently, then combined as weighted
averages before building the orbital gradient:

\[
\bar{\gamma}_{pq} = \sum_r w_r \gamma^{(r)}_{pq},\quad
\bar{\Gamma}_{pqrs} = \sum_r w_r \Gamma^{(r)}_{pqrs}
\]

with user-specified weights \(w_r\) (equal weights by default). The reported
total energy is the SA energy \(E_{\text{SA}} = \sum_r w_r E^{(r)}\). Root
identities are tracked across macro-iterations using maximum CI-vector overlap
with a Hungarian maximum-weight assignment so that SA weights remain attached to
the same physical states even when roots cross in energy.

Each root carries a `StateSpecificData` record through the macro loop that holds
that root's CI vector, energy, 1-RDM, 2-RDM, active Fock contribution, Q
contribution, orbital gradient, CI-response data, first-order 2-RDM, Q1
contribution, and CI-driven orbital correction. The state-averaged quantities
\(\bar{\gamma}\), \(\bar{\Gamma}\), \(F^A\), and \(\mathbf g_{\text{orb}}\) are
rebuilt as explicit weighted sums of those per-root records rather than being
formed from early-averaged inputs. The CI-response RHS is built analytically
from active-space one- and two-electron Hamiltonian derivatives
(`ResponseRHSMode::ExactOrbitalDerivative`); the older commutator-only shortcut
is available only via the `mcscf_debug_commutator_rhs` debug flag.

### Convergence and Robustness

The orbital macro-step uses **merit-function-based step selection**: multiple
candidate orbital steps (augmented-Hessian result, first micro-step, gradient
fallback, and their pairwise averages) are each evaluated by a full CASSCF
energy computation, and the step that minimizes
\(m = E_{\text{CAS}} + w\,\|\mathbf g\|^2\) is accepted. This avoids the sign
ambiguity that plagued earlier Cayley-map implementations and removes any
dependence on DIIS extrapolation.

When repeated macro-iterations accept only negligibly small steps without
reducing the true orbital gradient (stagnation), the driver switches to direct
orbital-gradient probe steps and single-pair directional probes, letting the
exact CASSCF energy screen pick the productive rotations.

The CI density matrices (1-RDM and 2-RDM) are built using exact
creation/annihilation operators in the spin-orbital determinant basis with a
determinant lookup table, ensuring the CI eigenvalue, density matrices, and
reconstructed energy are mutually consistent for all active-space sizes.

At convergence, the active-space 1-RDM is diagonalized to yield **natural
orbitals** with occupation numbers reported in descending order
(`_cas_nat_occ`).

Validation energies (RHF/STO-3G geometry unless noted):

| System | Active space | Basis | E(CASSCF) / Eh |
|---|---|---|---|
| H₂ | CAS(2,2) | STO-3G | −1.1372744062 |
| H₂O | CAS(2,2) | STO-3G | −74.9641865744 |
| H₂O | CAS(4,4) | STO-3G | −75.9851092026 |
| H₂O | CAS(4,4) | 6-31G | −75.5497490402 |
| H₂O | CAS(4,4) | cc-pVDZ | −75.6045806122 |
| C₂H₄ (90° twist) | CAS(2,2) | 3-21G | −77.5145223871 |
| C₂H₄ (90° twist) | CAS(2,2) | cc-pVDZ | −77.9524855977 |

### Twisted Ethylene: A Canonical CASSCF Example

Twisted ethylene at 90° C–C torsion is the prototypical system for which a
single Slater determinant is qualitatively wrong.

**Physical picture.** In planar ethylene the π system is described well by a
single HF configuration. When the two CH₂ groups are twisted 90° relative to
each other, the p-orbitals on the two carbons become orthogonal, breaking the
π overlap entirely. The result is a **biradical**: two electrons that once
formed a π bond now occupy one orbital on each carbon with nearly equal
probability. Neither a closed-shell configuration (both on one center) nor an
open-shell singlet (one on each, wrong spin pairing) captures this correctly
alone. The true ground state is a 50/50 mixture:

\[
|\Psi_0\rangle \approx \frac{1}{\sqrt{2}}\bigl(|\pi^2\rangle - |\pi^{*2}\rangle\bigr)
\]

The first excited singlet \(S_1\) is the complementary combination:

\[
|\Psi_1\rangle \approx \frac{1}{\sqrt{2}}\bigl(|\pi^2\rangle + |\pi^{*2}\rangle\bigr)
\]

At exactly 90° twist these two states are nearly degenerate (the splitting is
small and purely two-electron in origin), making this a strong-correlation
problem where the HF reference energy is far from the true energy and the
perturbation-theory expansion is unreliable.

**Active space selection.** The minimum correct active space is CAS(2,2): the
two electrons that formed the π bond, in the two orbitals that span the π/π\*
manifold. After optimization the two active natural orbitals have occupation
numbers near 1.0 each, confirming the biradical character. For a more complete
treatment one can include the σ/σ\* C–C bond (CAS(4,4)) or add the CH σ
manifold, but CAS(2,2) already recovers the qualitative physics.

**Geometry.** The test inputs use a C–C bond length of 1.339 Å (near the
experimental double-bond length) with the left CH₂ plane in the \(xy\) plane
and the right CH₂ plane in the \(xz\) plane, giving exactly 90° twist:

```
C    -0.669500    0.000000    0.000000
C     0.669500    0.000000    0.000000
H    -1.233698    0.927942    0.000000   ← left CH₂ in xy plane
H    -1.233698   -0.927942    0.000000
H     1.233698    0.000000    0.927942   ← right CH₂ in xz plane
H     1.233698    0.000000   -0.927942
```

The point-group symmetry is D₂d. At 90° twist the molecule gains an S₄
improper rotation axis along C–C and two σd mirror planes that bisect the
H–C–H angles, in addition to the three C₂ axes. The two active orbitals
transform as different irreps of D₂d, which is why the biradical wavefunctions
are even and odd combinations rather than simple MO products.

**Input example** (`tests/inputs/casscf_tests/ethylene_casscf_321g.hfinp`):

```
%begin_scf
    scf_type    rhf
    correlation casscf
    nactele     2
    nactorb     2
    nroots      1
%end_scf
```

A two-root SA-CASSCF run (`nroots 2`) optimizes orbitals for an equal-weight
average of S₀ and S₁. Because the two roots are nearly degenerate at 90° twist,
the SA orbital optimization is the recommended approach when studying the
S₀/S₁ gap or the conical intersection seam.

**Significance as a test case.** Twisted ethylene serves two validation roles:

1. *Single-root correctness*: the CAS(2,2) single-point energy should match
   external codes (PySCF, ORCA) at the same geometry and basis.
2. *SA robustness*: a two-root run near degeneracy exercises root tracking,
   overlap-based Hungarian assignment, and the merit-function step selector
   under conditions where the state ordering can change between macro-iterations.

### RASSCF Extensions

RASSCF (Restricted Active Space SCF) partitions the active space into three
subspaces:

- **RAS1**: orbitals from which at most `max_holes` electrons may be removed
- **RAS2**: full CAS subspace (no restrictions)
- **RAS3**: orbitals into which at most `max_elec` electrons may be added

The same CI machinery is used, but the string generation enforces the
occupation restrictions via bitcount masks on the RAS1 and RAS3 blocks.

---

## 15. Geometry Optimization

### L-BFGS (Cartesian Coordinates)

Cartesian L-BFGS minimizes \(E(\mathbf x)\) where \(\mathbf x \in \mathbb{R}^{3N}\) is
the flattened nuclear coordinate vector. The quasi-Newton update direction is:

\[
\mathbf p_k = -\mathbf H_k^{-1} \mathbf g_k
\]

L-BFGS avoids forming the approximate inverse Hessian \(\mathbf H_k^{-1}\)
explicitly. Instead it stores \(m\) recent displacement-gradient pairs
\(\{(\mathbf s_j, \mathbf y_j)\}\) where:

\[
\mathbf s_j = \mathbf x_{j+1} - \mathbf x_j,\quad
\mathbf y_j = \mathbf g_{j+1} - \mathbf g_j
\]

The matrix-vector product \(\mathbf H_k^{-1} \mathbf g_k\) is computed via the
two-loop recursion (Nocedal, 1980) in \(O(m \cdot 3N)\) operations, where
\(m\) (default 10 in `_geomopt_lbfgs_m`) is the history size.

A **Wolfe line search** (both sufficient-decrease and curvature conditions)
ensures the step satisfies:

\[
E(\mathbf x_k + \alpha_k \mathbf p_k) \le E(\mathbf x_k) + c_1 \alpha_k \mathbf g_k^T \mathbf p_k
\quad \text{and} \quad
|\mathbf g(\mathbf x_k + \alpha_k \mathbf p_k)^T \mathbf p_k| \le c_2 |\mathbf g_k^T \mathbf p_k|
\]

### Internal Coordinate Optimization (BFGS)

Internal coordinates (bond distances, valence angles, dihedral angles) are more
natural for describing molecular geometry changes. The Wilson **B matrix**
relates infinitesimal changes in internal coordinates \(\mathbf q\) to
Cartesian displacements \(\mathbf x\):

\[
d\mathbf q = \mathbf B\, d\mathbf x, \quad B_{kl} = \frac{\partial q_k}{\partial x_l}
\]

The gradient in internal coordinates is:

\[
\mathbf g^{int} = (\mathbf B \mathbf B^T)^{-1} \mathbf B\, \mathbf g^{Cart}
\]

A BFGS Hessian update is performed in internal coordinate space. The
back-transformation from internal to Cartesian steps uses iterative
microiterations that converge \(\Delta \mathbf q\) for a given \(\Delta \mathbf x\).

Geometry constraints (fixed bonds, angles, dihedrals, frozen atoms) are
enforced by projecting out the constrained internal coordinate contributions
from the gradient before the BFGS step.

Convergence criterion: maximum absolute gradient element
\(\max_i |\partial E / \partial X_i| < \epsilon_{grad}\) (default \(3 \times 10^{-4}\)
Ha/Bohr).

---

## 16. Vibrational Analysis

### Semi-Numerical Hessian

The Hessian matrix is computed by central finite differences of analytic
gradients:

\[
H_{ij} = \frac{\partial^2 E}{\partial X_i \partial X_j}
\approx \frac{\mathbf g_i(\mathbf x + h\hat e_j) - \mathbf g_i(\mathbf x - h\hat e_j)}{2h}
\]

with step size \(h\) (default \(5 \times 10^{-3}\) Bohr, stored in
`_hessian_step`). This requires \(2 \times 3N\) SCF+gradient calculations.
The symmetry of the Hessian is enforced by averaging \((H_{ij} + H_{ji})/2\).

### Mass-Weighting and Eckart Projection

The mass-weighted Hessian is:

\[
\tilde H_{ij} = \frac{H_{ij}}{\sqrt{m_i m_j}}
\]

where \(m_i\) is the mass of the atom to which Cartesian coordinate \(i\)
belongs.

Six vibrational modes correspond to rigid-body translation and rotation and
have zero frequency. These are projected out using the **Eckart conditions**:
six orthonormal vectors in \(\mathbb{R}^{3N}\) are constructed that span the
translational and rotational subspace, and the \(3N \times 3N\) projector onto
the vibrational subspace is applied to \(\tilde{\mathbf H}\) before diagonalization:

\[
\tilde{\mathbf H}^{vib} = \mathbf P \tilde{\mathbf H} \mathbf P,\quad
\mathbf P = \mathbf I - \sum_{k=1}^{6} \mathbf d_k \mathbf d_k^T
\]

### Normal Mode Frequencies

The \(3N - 6\) non-zero eigenvalues \(\lambda_k\) of \(\tilde{\mathbf H}^{vib}\)
give vibrational frequencies:

\[
\tilde\nu_k = \frac{1}{2\pi c}\sqrt{\lambda_k}
\]

converted to cm\(^{-1}\) by multiplying by appropriate unit factors. Imaginary
frequencies (negative \(\lambda_k\)) correspond to transition states or saddle
points on the potential energy surface.

The zero-point energy is:

\[
E_{ZPE} = \frac{1}{2}\sum_{k=1}^{3N-6} h\nu_k
\]

Vibrational symmetry labels are assigned in `src/symmetry/vibrational_symmetry.cpp`
by projecting each normal mode onto the SAO blocks and determining its irrep.

---

## 17. Checkpoint and Restart

### Binary Checkpoint Format

The checkpoint file (`*.hfchk`) stores:

- A 4-byte magic number and format version (v2)
- Molecular geometry (standard-orientation, Bohr)
- Basis set name
- Density matrices (alpha and optionally beta)
- Total SCF energy
- Optional geometry optimization metadata

### Cross-Basis Löwdin Projection

When restarting from a checkpoint computed with a smaller basis
(e.g., STO-3G) to a larger basis (e.g., 6-31G*), the stored density matrix
cannot be used directly. Planck projects the old density into the new basis
using the cross-overlap matrix:

\[
S^{cross}_{\mu\nu} = \langle \chi^{large}_\mu | \chi^{small}_\nu \rangle
\]

computed by `_compute_cross_overlap` in `os.cpp`. The projection is then:

\[
P^{large}_{\mu\nu} = \sum_{\lambda\sigma}
(S^{LL})^{-1}_{\mu\lambda}\, S^{cross}_{\lambda\lambda'}\,
P^{small}_{\lambda'\sigma'}\, (S^{cross})^T_{\sigma'\mu}\,
(S^{LL})^{-1}_{\mu\nu}
\]

implemented via a singular value decomposition (Löwdin SVD) of the
cross-overlap in `src/io/checkpoint.cpp`. This provides a physically motivated
initial density for the new basis, significantly reducing the number of SCF
iterations required.

---

## 18. Execution Flow of a Typical Run

```
driver.cpp
  parse_input()                → Calculator._scf, _basis, _geometry, etc.
  prepare_coordinates()        → molecule._coordinates (Bohr)
  checkpoint restore (if any)  → geometry / density
  detectSymmetry()             → molecule._standard (Bohr), _point_group
  read_gbs_basis()             → shells, basis functions, normalization
  build_shellpairs()           → shell_pairs[0..nb*(nb+1)/2-1]
  _compute_1e()                → S, T  (os.cpp)
  _compute_nuclear_attraction()→ V     (os.cpp)
  H_core = T + V
  build_sao_basis()            → U, block sizes, irrep names
  update_integral_symmetry()   → _integral_symmetry_ops
  build_canonical_pairs()      → _canonical_ao_pair[]

  if Conventional SCF:
      _compute_2e()            → _eri[nb^4]  (os.cpp)

  run_rhf() or run_uhf()       → C, ε, P, E_SCF  (scf.cpp)
      each iteration:
          G = _compute_fock_rhf(eri, P) or _compute_2e_fock(shell_pairs, P)
          F = H_core + G
          DIIS.push(F, e)
          F' = X^T F X
          diagonalize F' → C', ε
          C = X C'
          rebuild P
          check convergence

  if post_hf == RMP2:
      AO→MO transform → (ia|jb) MO integrals
      run_rmp2()               → E_MP2
  elif post_hf == CASSCF:
      run_casscf()             → E_CASSCF, natural orbitals

  if gradient or geomopt or frequency:
      compute_rhf_gradient()   → _gradient (gradient.cpp)

  if geomopt:
      run_geomopt()            → optimized geometry (geomopt.cpp)
          each step: SCF → gradient → L-BFGS or BFGS update

  if frequency:
      compute_hessian()        → _hessian (hessian.cpp)
          for each displacement: SCF → gradient (2×3N calculations)
      vibrational_analysis()   → _frequencies, _normal_modes, _zpe

  save_checkpoint()
```

---

## 19. Theory-to-Code Map

| Theory concept | Primary file(s) | Key function(s) |
|---|---|---|
| Data structures | `src/base/types.h` | `Calculator`, `Shell`, `Basis`, `ShellPair` |
| Input parsing | `src/io/io.cpp` | `parse_input` |
| Basis reading | `src/basis/gaussian.cpp` | `read_gbs_basis` |
| Shell pairs | `src/integrals/shellpair.cpp` | `build_shellpairs` |
| Overlap and kinetic | `src/integrals/os.cpp` | `_compute_1e`, `_compute_3d_overlap_kinetic` |
| Boys function | `src/lookup/` | table lookup and asymptotic expansion |
| Nuclear attraction | `src/integrals/os.cpp` | `_compute_nuclear_attraction` |
| ERI tensor | `src/integrals/os.cpp` | `_compute_2e`, `_contracted_eri` |
| Direct Fock build | `src/integrals/os.cpp` | `_compute_2e_fock`, `_compute_2e_fock_uhf` |
| Rys quadrature | `src/integrals/rys.cpp` | `_rys_eri_primitive`, `_rys_contracted_eri` |
| Orthogonalizer | `src/scf/scf.cpp` | `build_orthogonalizer` |
| RHF SCF | `src/scf/scf.cpp` | `run_rhf` |
| UHF SCF | `src/scf/scf.cpp` | `run_uhf` |
| DIIS | `src/base/types.h` | `DIISState::push`, `DIISState::extrapolate` |
| Symmetry detection | `src/symmetry/symmetry.cpp` | `detectSymmetry` |
| SAO basis | `src/symmetry/mo_symmetry.cpp` | `build_sao_basis` |
| MO irrep labels | `src/symmetry/mo_symmetry.cpp` | `assign_mo_symmetry` |
| Integral symmetry ops | `src/symmetry/integral_symmetry.cpp` | `update_integral_symmetry` |
| AO→MO transform | `src/post_hf/integrals.cpp` | half-transformation functions |
| RMP2 energy | `src/post_hf/mp2.cpp` | `run_rmp2` |
| UMP2 energy | `src/post_hf/mp2.cpp` | `run_ump2` |
| MP2 amplitudes | `src/post_hf/mp2.cpp` | `build_rmp2_amplitudes` |
| CPHF Z-vector | `src/post_hf/rhf_response.cpp` | `build_rhf_cphf_matrix` |
| RMP2 gradient | `src/post_hf/mp2_gradient.cpp` | `compute_rmp2_gradient` |
| CI string generation | `src/post_hf/casscf.cpp` | Gosper enumeration |
| CI solve (Davidson) | `src/post_hf/casscf.cpp` | Davidson solver |
| 1-RDM, 2-RDM | `src/post_hf/casscf.cpp` | `compute_1rdm`, `compute_2rdm` |
| Orbital gradient | `src/post_hf/casscf.cpp` | generalized Fock matrix |
| Orbital update | `src/post_hf/casscf.cpp` | Cayley transform |
| RHF gradient | `src/gradient/gradient.cpp` | `compute_rhf_gradient` |
| UHF gradient | `src/gradient/gradient.cpp` | `compute_uhf_gradient` |
| Derivative integrals | `src/integrals/os.cpp` | `_compute_1e_deriv_A`, `_compute_eri_deriv_elem` |
| L-BFGS optimizer | `src/opt/geomopt.cpp` | `run_geomopt` |
| Internal coordinates | `src/opt/intcoords.cpp` | Wilson B matrix |
| Semi-numerical Hessian | `src/freq/hessian.cpp` | `compute_hessian` |
| Vibrational analysis | `src/freq/hessian.cpp` | `vibrational_analysis` |
| Vibrational symmetry | `src/symmetry/vibrational_symmetry.cpp` | mode irrep assignment |
| Checkpoint I/O | `src/io/checkpoint.cpp` | `save_checkpoint`, `load_checkpoint` |
| Cross-basis projection | `src/io/checkpoint.cpp` | Löwdin SVD projection |
| Molecular grid | `src/dft/base/grid.h` | `MakeMolecularGrid`, `MakeAtomicGrid` |
| AO evaluation on grid | `src/dft/ao_grid.h` | `AOGridEvaluation` |
| Density on grid | `src/dft/xc_grid.cpp` | `evaluate_density_on_grid` |
| XC evaluation (libxc) | `src/dft/xc_grid.cpp` | `evaluate_xc_on_grid` |
| XC matrix assembly | `src/dft/ks_matrix.cpp` | `assemble_xc_matrix` |
| KS potential matrices | `src/dft/ks_matrix.cpp` | `combine_ks_potential` |
| KS-DFT driver | `src/dft/driver.cpp` | `DFT::Driver::run` |

---

## 20. Kohn-Sham Density Functional Theory

### 20.1 The Kohn-Sham Equations

Kohn-Sham DFT maps the interacting many-electron problem onto a fictitious system of non-interacting electrons moving in an effective potential \(v_s(\mathbf r)\) that yields the same ground-state density as the real system. The total electronic energy is:

\[
E[P] = T_s[P] + V_{ne}[P] + J[P] + E_{xc}[P] + V_{nn}
\]

where \(T_s\) is the non-interacting kinetic energy, \(V_{ne}\) is the electron-nuclear attraction, \(J\) is the Coulomb (Hartree) energy, \(E_{xc}\) is the exchange-correlation energy, and \(V_{nn}\) is the nuclear repulsion. Minimising \(E[P]\) under the constraint that the KS orbitals are orthonormal leads to the KS secular equations:

\[
F^{KS}_{\mu\nu} = h_{\mu\nu} + J_{\mu\nu} + V^{xc}_{\mu\nu}
\]

This is identical in structure to the HF Fock matrix, with the HF exchange matrix \(K\) replaced by the XC potential matrix \(V^{xc}\). Planck reuses the HF SCF loop for KS-DFT: the only structural difference is how the two-electron contribution to the Fock matrix is assembled (Coulomb only, no exchange, plus \(V^{xc}\) from numerical integration).

### 20.2 Exchange-Correlation Functional Families

#### LDA (Local Density Approximation)

The XC energy depends only on the local electron density \(\rho(\mathbf r)\):

\[
E_{xc}^{LDA}[\rho] = \int \rho(\mathbf r)\, \varepsilon_{xc}^{LDA}(\rho(\mathbf r))\, d\mathbf r
\]

Planck's LDA components:
- **Slater exchange** (`lda_x` in libxc): the Dirac expression \(\varepsilon_x = -\tfrac{3}{4}\left(\tfrac{3}{\pi}\right)^{1/3}\rho^{1/3}\)
- **VWN5 correlation** (`lda_c_vwn_5`): Vosko-Wilk-Nusair parametrisation of the uniform electron gas correlation energy (the most common LDA correlation functional)

The combination Slater + VWN5 is referred to as SVWN.

#### GGA (Generalized Gradient Approximation)

The XC energy also depends on the density gradient:

\[
E_{xc}^{GGA}[\rho] = \int f(\rho(\mathbf r),\, |\nabla\rho(\mathbf r)|^2)\, d\mathbf r
\]

GGA functionals satisfy more exact constraints than LDA and generally give better geometries and energies. Planck's GGA components and their common pairings:

| Exchange | Correlation | Combination name |
|---|---|---|
| B88 (`gga_x_b88`) | LYP (`gga_c_lyp`) | BLYP |
| B88 | P86 (`gga_c_p86`) | BP86 |
| B88 | PW91 (`gga_c_pw91`) | BPW91 |
| PW91 (`gga_x_pw91`) | PW91 | PW91 |
| PBE (`gga_x_pbe`) | PBE (`gga_c_pbe`) | PBE (default) |

### 20.3 Numerical Integration: Molecular Grid

Because \(V^{xc}_{\mu\nu}\) has no analytic closed form, it is evaluated numerically:

\[
V^{xc}_{\mu\nu} = \int \phi_\mu(\mathbf r)\, v_{xc}(\mathbf r)\, \phi_\nu(\mathbf r)\, d\mathbf r
\approx \sum_g w_g\, \phi_\mu(\mathbf r_g)\, v_{xc}(\mathbf r_g)\, \phi_\nu(\mathbf r_g)
\]

The sum runs over quadrature grid points \(\{\mathbf r_g, w_g\}\). Planck builds the molecular grid from three layers:

#### Radial grid — Treutler-Ahlrichs M4

Each atom's radial shells are placed according to the Treutler-Ahlrichs M4 mapping, which concentrates points near the nucleus (where \(\rho\) varies rapidly) and uses element-specific radii (`treutler_radius(Z)` in `src/dft/base/radial.h`). The number of radial shells is an increasing function of both the grid quality preset and the row of the periodic table.

#### Angular grid — Lebedev quadrature

At each radial shell, angular integration is performed using a Lebedev grid of order \(N_\Omega\) (`MakeLebedevGrid(N)` in `src/dft/base/angular.h`). Lebedev grids integrate polynomials in \((x, y, z)\) exactly up to a maximum degree that grows with \(N_\Omega\). Planck uses five angular shell sizes arranged in five radial regions (pruning).

#### Five-region pruning

To reduce cost without sacrificing accuracy, the molecular grid is pruned: regions far from and very close to the nucleus use coarser angular grids, while the valence shell region uses the finest grid. The five regions and their shell sizes are controlled by `angular_shells_for_scheme` in `src/dft/base/grid.h`.

#### Becke partitioning

A single-centre quadrature cannot integrate the full molecular density accurately. The Becke scheme partitions space into atom-centred cells using smooth step functions \(s_{ij}(\mathbf r)\) derived from a confocal elliptic coordinate:

\[
\mu_{ij} = \frac{|\mathbf r - \mathbf R_i| - |\mathbf r - \mathbf R_j|}{|\mathbf R_i - \mathbf R_j|}
\]

Three applications of the Hermite switch \(f_k(\mu) = \tfrac{3}{2}\mu - \tfrac{1}{2}\mu^3\) smooth the partition. Planck uses Treutler-Becke size-adjusted partitioning (`treutler_becke_adjustment` in `src/dft/base/grid.h`), which accounts for the different atomic radii of unlike atom pairs. The effective grid weight at point \(\mathbf r\) on atom \(i\) is:

\[
w_i(\mathbf r) = \frac{P_i(\mathbf r)}{\sum_k P_k(\mathbf r)} \cdot w_i^{radial-angular}
\]

#### Grid quality presets

| Preset | Angular scheme | Pruned regions (5) |
|---|---|---|
| `Coarse` | 3 | 14 / 26 / 50 / 110 / 50 |
| `Normal` | 4 | 26 / 110 / 194 / 302 / 194 |
| `Fine` | 5 | 26 / 194 / 302 / 434 / 302 |
| `UltraFine` | 6 | 50 / 302 / 434 / 590 / 434 |

### 20.4 AO Evaluation on the Grid

`AOGridEvaluation` (declared in `src/dft/ao_grid.h`) stores the AO values and
gradients at every grid point in a matrix of shape `(N_grid, N_AO)`. These are
computed once before the KS iteration begins. For each grid point and each AO
\(\phi_\mu\), the value and Cartesian gradient components are evaluated from the
contracted shell data in the `Basis` object.

### 20.5 Density and XC Evaluation

Given the density matrix \(P_{\mu\nu}\), the electron density at grid point \(g\) is:

\[
\rho(\mathbf r_g) = \sum_{\mu\nu} P_{\mu\nu}\, \phi_\mu(\mathbf r_g)\, \phi_\nu(\mathbf r_g)
\]

For GGA functionals, the gradient \(\nabla\rho\) and the reduced gradient invariant \(\sigma = |\nabla\rho|^2\) are also needed. Both are assembled in `evaluate_density_on_grid` (`src/dft/xc_grid.cpp`).

The libxc library (`src/dft/base/wrapper.h`) is then called with the density (and gradient for GGA) arrays to return the XC energy density \(\varepsilon_{xc}(\rho)\) and the potential derivatives \(v_\rho = \partial(\rho\varepsilon_{xc})/\partial\rho\) and \(v_\sigma = \partial(\rho\varepsilon_{xc})/\partial\sigma\). The XC energy is:

\[
E_{xc} = \sum_g w_g\, \rho(\mathbf r_g)\, \varepsilon_{xc}(\mathbf r_g)
\]

### 20.6 XC Matrix Assembly

The XC potential matrix element is:

\[
V^{xc}_{\mu\nu} = \sum_g w_g\, v_{\rho,g}\, \phi_\mu(\mathbf r_g)\, \phi_\nu(\mathbf r_g)
+ 2\sum_g w_g\, \mathbf v_{\sigma,g} \cdot \nabla\rho_g \cdot \left[\phi_\mu \nabla\phi_\nu + \phi_\nu \nabla\phi_\mu\right]_g
\]

The second term (present only for GGA) involves the density gradient and the AO gradients on the grid. Both terms are assembled in `assemble_xc_matrix` (`src/dft/ks_matrix.cpp`).

### 20.7 KS-DFT SCF Loop

The KS SCF loop in `DFT::Driver::run` follows the same outer structure as the HF loop:

1. Compute 1e integrals (\(S\), \(T\), \(V_{ne}\)), build orthogonalizer, form initial guess
2. At each iteration:
   a. Build the Coulomb matrix \(J[P]\) using the standard ERI or direct path
   b. Evaluate the density \(\rho\) and gradient \(\nabla\rho\) on the molecular grid
   c. Call libxc to get \(\varepsilon_{xc}\), \(v_\rho\), \(v_\sigma\) on the grid
   d. Assemble \(V^{xc}_{\mu\nu}\) by numerical quadrature
   e. Form the KS Fock matrix \(F^{KS} = h + J + V^{xc}\)
   f. Solve the KS secular equation, update \(P\), check convergence

The `KSPotentialMatrices` struct holds the Coulomb, XC-alpha, and XC-beta matrices and their sum (the full KS two-electron+XC potential). For RKS the alpha and beta components are identical; for UKS they differ because \(\rho_\alpha \neq \rho_\beta\).

### 20.8 DFT Code Map

| Task | File | Function/struct |
|---|---|---|
| Treutler-Ahlrichs radial grid | `src/dft/base/radial.h` | `MakeTreutlerAhlrichsGrid` |
| Lebedev angular grid | `src/dft/base/angular.h` | `MakeLebedevGrid` |
| Atomic and molecular grid | `src/dft/base/grid.h` | `MakeAtomicGrid`, `MakeMolecularGrid` |
| Becke partitioning | `src/dft/base/grid.h` | `becke_partition_weight` |
| libxc functional wrapper | `src/dft/base/wrapper.h` | `DFT::XC::Functional` |
| AO evaluation on grid | `src/dft/ao_grid.h` | `AOGridEvaluation` |
| Density on grid | `src/dft/xc_grid.cpp` | `evaluate_density_on_grid` |
| XC energy and potential on grid | `src/dft/xc_grid.cpp` | `evaluate_xc_on_grid` |
| XC matrix \(V^{xc}_{\mu\nu}\) | `src/dft/ks_matrix.cpp` | `assemble_xc_matrix` |
| Full KS potential | `src/dft/ks_matrix.cpp` | `combine_ks_potential` |
| KS-DFT main loop | `src/dft/driver.cpp` | `DFT::Driver::run` |
| DFT entry point | `src/dft/main.cpp` | `main` |

---

## 21. Current Implementation Status

| Feature | Status |
|---|---|
| RHF and UHF SCF | Complete |
| Obara-Saika 1e and 2e integrals | Complete |
| Rys quadrature ERIs | Complete |
| Conventional and direct SCF | Complete |
| Schwarz screening | Complete |
| DIIS acceleration | Complete |
| Level shifting | Complete |
| Point group detection and SAO blocking | Complete |
| MO irrep labeling | Complete |
| RMP2 and UMP2 energy | Complete |
| Analytic RHF gradient | Complete |
| Analytic UHF gradient | Complete |
| Analytic RMP2 gradient (Z-vector) | Complete |
| UMP2 gradient | Not implemented (throws at runtime) |
| CASSCF / RASSCF | Complete |
| Geometry optimization (RHF/UHF/RMP2) | Complete |
| UMP2 geometry optimization | Blocked by missing UMP2 gradient |
| Semi-numerical Hessian | Complete |
| Harmonic vibrational analysis | Complete |
| Checkpoint save/restart | Complete |
| Cross-basis density projection | Complete |
| Kohn-Sham DFT (RKS/UKS) | Complete |
| LDA XC functionals (Slater, VWN5) | Complete |
| GGA XC functionals (B88, PBE, PW91 exchange; LYP, P86, PBE, PW91 correlation) | Complete |
| Arbitrary libxc functionals via integer ID | Complete |
| Molecular grid (Treutler-Ahlrichs + Lebedev + Becke) | Complete |
| DFT geometry optimization / gradients | Not implemented |
| Hybrid XC functionals (e.g. B3LYP) | Not supported (requires HF exchange) |
| Spherical harmonic basis | Not supported (Cartesian only) |

---

## 22. How to Study This Codebase

Recommended reading order for the HF/post-HF pipeline:

1. `src/base/types.h` — understand every struct before reading any algorithm
2. `src/driver.cpp` — the control flow map for one complete calculation
3. `src/io/io.cpp` — how input files become a Calculator
4. `src/integrals/os.cpp` — the Obara-Saika integral engine top to bottom
5. `src/scf/scf.cpp` — the SCF iteration in detail
6. `src/gradient/gradient.cpp` — how analytic gradients are assembled
7. `src/post_hf/mp2.cpp` and `src/post_hf/integrals.cpp` — MP2 energy
8. `src/post_hf/casscf.cpp` — the most complex module: CI, RDMs, orbital update
9. `src/opt/geomopt.cpp` — L-BFGS and internal coordinate optimization
10. `src/freq/hessian.cpp` — finite-difference Hessian and normal modes

Recommended reading order for the KS-DFT pipeline (read after items 1–5 above):

11. `src/dft/base/radial.h` — Treutler-Ahlrichs M4 radial grid
12. `src/dft/base/angular.h` — Lebedev angular quadrature
13. `src/dft/base/grid.h` — Becke partitioning, pruning, molecular grid assembly
14. `src/dft/ao_grid.h` — AO and gradient evaluation at grid points
15. `src/dft/xc_grid.cpp` — density, XC energy and potentials on the grid
16. `src/dft/ks_matrix.cpp` — \(V^{xc}_{\mu\nu}\) and full KS potential matrices
17. `src/dft/driver.cpp` — the KS-DFT SCF loop end to end

This order follows the dependency graph: basic state and types first, then
integral machinery, then the SCF loop that uses those integrals, then the
higher-level methods that build on SCF.
