# Deriving the Complete XC Contribution to the Double-Hybrid PT2 Gradient

In-flight scope. Fold into `docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md` when
this lands. Canonical status lives in `vault/Status/`.

**Question this scopes:** derive, from first principles rather than by matching
candidates against a target, the complete XC contribution to `E_PT2^x` for a
truncated double hybrid. Planck currently implements XC_II (Eq. 33's
basis-derivative piece) and is **3.887e-4 Ha/Bohr** from finite difference on
C1 H2O2. Every empirical avenue is exhausted: H2 (Planck's KS-side assembly) is
exonerated end to end, and every DH-specific object in the paper's Eqs. 40-47
has been verified or excluded. **The remaining work is algebra, not probing.**

## The constraint the derivation must satisfy (this is the whole problem)

Let `T` = the full missing quantity = `FD - Planck(no XC term)`. Measured on
C1 H2O2:

| object | norm | `sum_A` (net force) |
|---|---|---|
| **`T`** (true missing term) | `1.786e-3` | **`1.1e-8`** -- invariant |
| `XC_II` (what Planck adds) | `1.590e-3` | **`[-1.6e-4, 5.5e-5, -5.2e-4]`** -- NOT invariant |
| `R = T - XC_II` (companion) | `6.503e-4` | `[+1.6e-4, -5.5e-5, +5.2e-4]` -- exactly `-sum_A(XC_II)` |

**`T` is a true force and is translationally invariant. `XC_II` alone is not.**
So the missing companion `R` is doubly constrained: it must supply

- **(a)** real physics in the invariant subspace -- **91%** of `|R|`, and
- **(b)** exactly `-sum_A(XC_II)` -- **42%** of `|R|`

*simultaneously*. Every failed candidate failed one of these. `kXcIIt` satisfied
(b) and not (a) (it is `-XC_II` redistributed, cos -0.997). `kXcIIw` satisfied
neither (cos -0.019). The `vhf_s1occ` family is bounded away from (a) by
`sqrt(1-cos^2) >= 72%`.

### Why XC_II is non-invariant, and why that is correct

`rho_P^(x)` (Eq. 15) is the **basis-function derivative at a fixed spatial
point**. `drho_channel` (`dft_gradient.cpp:140`) sums only over `mu in A`, so
`sum_A drho_channel = -d(rho)/dr != 0`. **This is not a bug** -- Eq. 33 is
written in exactly this convention. It means Eq. 33 as printed is an incomplete
force expression, and the paper never says otherwise because it never
decomposes the XC term the way this project has had to.

### The obvious companion is ruled out by measurement

`XC_III` (the moving-grid term on `Phi_XC`'s own integrand `I_p`) cannot be it:

```
|XC_III| = 4.57e-07     (needed |R| = 6.50e-04, 1400x larger)
sum_A(XC_III) ~ 1e-07   (needed 5.2e-04, 5000x larger)
```

D3 shows by linearity that the `rx`/`gx` channel of a moving-frame correction
on `XC_II` is also closed (it reproduces `kXcIIt` byte-for-byte). **But the
coefficient channel is NOT closed and has never been probed** -- see D3.
Identifying which of the two remaining candidates is right is the derivation's
job.

## What is already established (do not re-derive)

| fact | where |
|---|---|
| XC_II == Eq. 33's XC term, term-matched against Eq. 23 | N3.5.7.14 |
| the `f^{rho_s rho_s zeta}` notation is spin labelling, not `f^(3)` | N3.5.7.14 |
| relaxed `D` is the right density (cos 0.93 vs unrelaxed 0.75) | N3.5.7.10 |
| no closed-shell spin-factor error (vs polarized `R^XC`, 1e-17) | Sec. II cross-check |
| Eq. 42's `R^XC(D)` in isolation: 7% WORSE | N3.5.7.14 |
| Eq. 39's `1/(1+delta_ij)`: Planck's unrestricted sum is equivalent | N3.5.7.14 |
| amplitudes elementwise to 8e-9; `Gamma^NS` exonerated | H2.2 |
| Z-vector: exact QR, symmetric 5e-17, SPD -- but **3.6x leverage** | H2.3 |
| residual is **97% linear in `c_pt2`** -- it IS a term in `E_PT2^x` | H2.1 |
| the KS-only gradient is accurate to 3.66e-5 (11x below the residual) | scope msmt 2 |
| residual is not grid noise (ultrafine converged to 9.3e-5) | scope msmt 1 |

## The derivation

### D1 -- restate `E_PT2` as an explicit functional of geometry (~0.5 day)

Write `E_PT2` with **every** `R`-dependence explicit, before any derivative is
taken. The energy is a Hylleraas functional (Eq. 11) stationary in `t`, so

```
E_PT2(R) = E_PT2[ t(R), C(R), eps(R), grid(R), basis(R) ]
```

and the total derivative is

```
dE_PT2/dR = (partial/partial R)|_explicit          <- basis + grid move
          + (dE/dC)(dC/dR)                          <- handled by the Z-vector
          + (dE/dt)(dt/dR)                          <- ZERO (Hylleraas stationary)
```

**Deliverable:** an explicit list of every place `R` enters, with the
Hylleraas-stationarity cancellation shown rather than asserted. **Do not skip
this** -- the arc's four reverted attempts all came from differentiating a
scalar (`Phi_XC`) that was not the right functional of `R`.

**Check:** the list must contain the grid, because the XC energy is a
quadrature; if the derivation produces no moving-grid term, D1 is wrong.

### D2 -- derive the XC part of `dE_PT2/dR` from the SCF-operator route (~1 day)

The paper's own prose (p.6) says the XC contribution "arises from the
contraction of the relaxed PT2 difference density with the **derivative of the
SCF operator**". Make that literal:

```
E_PT2 |_XC-dependence  =  sum_munu D_munu (V_xc[rho_P])_munu   ==  Phi_XC
```

so the XC contribution to the gradient is `d/dR{Phi_XC}` -- **which this project
already derived and FD-validated to rel 3e-9** (S5), and which decomposes as
`XC_I + XC_II + XC_III`.

**But wiring all three was measured at 3.5e-3, 9x WORSE than the baseline.**
That is the central puzzle and D2 must resolve it, not sidestep it:

- **`XC_I` (85% of `d/dR{Phi_XC}`) is provably not in the answer** -- adding it
  blows the gradient up, and Eq. 33 contains no `rho_D^(x)` factor anywhere.
- **The reason must be that `D` is not an independent constant.** `D` is the
  *relaxed* difference density; it depends on `R` through `t`, `C`, and the
  Z-vector. `XC_I` differentiates `rho_D` as if `D` were fixed, but that
  variation is **already counted** -- via the Z-vector (which is exactly the
  `dE/dC dC/dR` channel) and via Hylleraas stationarity in `t`.

**Deliverable:** show explicitly that `XC_I` double-counts the `D`-response, by
identifying which term in the Lagrangian/Z-vector chain already carries it.

**Falsifiable:** if `XC_I` is genuinely absent from the Lagrangian chain, this
explanation is wrong and D2 must find another -- but then the 3.5e-3 measurement
needs a different account, and there is no candidate for one.

### D3 -- the companion term (~1-2 days) -- THE CORE STEP

D2 leaves `XC_II` (keep) and `XC_III` (measured negligible, 4.6e-7). Neither
supplies `R`.

**The constraint says what to look for. An earlier draft of this scope claimed
the whole moving-frame family was closed by a linearity argument; that claim was
too broad and is corrected here.**

`XC_II` is a **quadrature**, `XC_II(A,q) = sum_p w_p * F_p(A,q)` with

```
F_p(A,q) = c1 * rx(A,q)  +  c2 . gx(A,q)          [LINEAR in rx, gx]
c1 = v2rho2*rho_D + 2*v2rhosigma*(g.grad_rho_D)
c2 = 2*v2rhosigma*g*rho_D + 4*v2sigma2*(g.grad_rho_D)*g + 2*vsigma*grad_rho_D
```

If the grid point translates with its owner atom, **three** things move, not
one:

| channel | what moves | probed? |
|---|---|---|
| (i) | `rx`, `gx` -- the density derivatives | **yes** -- `kXcIIt` |
| (ii) | `c1`, `c2` -- i.e. `v2*`, `rho_D`, `grad_rho_D`, `g` | **NO** |
| (iii) | `w` -- the Becke weight | **yes** -- `kXcIIw` |

**The linearity argument covers channel (i) ONLY, and there it is confirmed:**
completing the density derivative inside `rx`/`gx` was implemented as a fresh
probe and came out **byte-identical** to `kXcIIt` (cos -0.997), exactly as
`XC_II[rx + c] = XC_II[rx] + XC_II[c]` predicts. So that channel really is
closed.

**Channel (ii) was never probed by anything in this arc**, and it is where the
missing invariant physics would naturally live, because `c1`/`c2` contain
`rho_D` and `grad_rho_D`. A scratch probe of it produces a non-zero,
direction-distinct contribution. **It was deliberately NOT scored**, because the
probe as written carried zeroed and dead terms (`f^(3)` pieces are unavailable);
scoring a half-built term is precisely the "tune until it fits" failure this arc
has repeated. **Deriving channel (ii) properly is D3's first task.**

### Does channel (ii) even belong? The bookkeeping question D3 must settle first

`d/dR{Phi_XC} = XC_I + XC_II + XC_III` was FD-verified to **rel 3e-9** (S5), so
that decomposition is complete -- there is no unaccounted fourth channel *in
`Phi_XC`*. The three movement channels above are already distributed across it:
(i) and (ii) sit inside what S5 called XC_II and XC_I respectively, and (iii) is
XC_III(a).

**So the real question is not "is a term missing from `d/dR{Phi_XC}`" -- it is
"which channels does Eq. 33's convention keep".** Eq. 33's `rho_P^(x)` (Eq. 15)
is basis-only at a **fixed** point, which is what `drho_channel` computes and
what Planck implements. The moving-point version differs by exactly the
point-translation piece. **Planck's XC_II is faithful to Eq. 33; the open
question is whether Eq. 33's fixed-point convention is the right object for a
force, given that `T` is invariant and `XC_II` is not.**

That is a question about the paper, not about Planck, and it is the same
question H1 asks. **D3 must resolve it before writing any term**, because the
two readings prescribe different code:
- if the fixed-point convention is right, the companion is a *separate* term
  (D3's `d eps/dR` candidate below), and
- if the moving-point convention is right, the companion is channel (ii) of a
  correctly-completed `XC_II`, and Eq. 33 as printed is incomplete.

**What `R` must therefore be.** Measured in the invariant subspace (where the
frame question does not arise):

```
cos(T, XC_II) = 0.9462   best scale 1.0782
XC_II leaves 32% of |T_free| unexplained  (3.71e-4)
```

So `R` carries **real invariant physics at 32% of the total**, plus the frame
constraint as a by-product of whatever object supplies it. **D3's task is to
find a scalar whose `d/dR` produces both at once** -- not to patch XC_II.

**D3's two candidates.** The bookkeeping question above decides between them;
do not build either before settling it.

**Candidate 1 -- channel (ii): `XC_II`'s coefficients move with the point.**
Applies if the moving-point convention is the right one for a force. Never
probed, direction-distinct in a scratch test, and structurally where the missing
invariant physics would live (`c1`/`c2` carry `rho_D` and `grad_rho_D`). Needs
`f^(3)` for the `v2*`-moves piece -- **the `evaluate_*_kxc` wrappers built at S1
would finally have a consumer** -- verified: `grep` shows their only callers are
in `tests/dft_kxc_selfcheck.cpp`, i.e. they are orphaned production code built
on the later-overruled `f^(3)` reading. A channel that genuinely needs `f^(3)`
would give them their first real use. Weak corroboration, but it points the
same way.

**Candidate 2 -- the XC part of `d eps/dR` in the MP2 denominators.**
Applies if Eq. 33's fixed-point convention is right and the companion is a
separate term.

`Phi_XC` is **not the only place XC enters `E_PT2`**. The amplitudes carry
`D_ijab = eps_i + eps_j - eps_a - eps_b`, and for a double hybrid the `eps` are
**KS** eigenvalues:

```
eps_p = <p| h + J - a_x K + V_xc |p>
```

so `d eps_p/dR` has a `V_xc` contribution with no HF counterpart.

**Why this is not already counted.** In MP2 gradient theory the `eps`-dependence
is not a separate term -- it is absorbed into the energy-weighted density `W`
via Eqs. 42-43's `-1/2 D_ij(eps_i + eps_j)` and `-1/2 D_ab(eps_a + eps_b)`,
contracted against `S^(x)`. **Planck has exactly that** (the `s_zeta`
accumulator). But `W` is built in the HF form: it carries `d eps/dR` through the
overlap-derivative channel only, which is correct when `eps` are Fock
eigenvalues. **A KS `eps` moves additionally because `V_xc` moves**, and no HF-
derived `W` can know about it.

**Magnitude is plausible, not proven.** `sum t^2 ~ |E_corr|/D ~ 0.1`, so an XC
contribution to `d eps/dR` of only `~4e-3` Ha/Bohr yields the needed `4e-4`.
That is a small, entirely ordinary size for a `V_xc` derivative -- unlike the
frame-correction candidates, which had to be implausibly large or implausibly
structured.

**Why it fits the constraints.** It is a genuine energy derivative, so it is
translationally invariant *as a whole* -- satisfying (a) and (b) together rather
than one at a time, which is exactly what every failed candidate could not do.
And it is linear in `c_pt2` (it lives inside `E_PT2`), matching H2.1.

**This channel has never been examined in this arc.** Deriving it is D3's
content: write `d eps_p/dR` for a KS reference, isolate the `V_xc` piece, and
contract it against `-sum t^2` (equivalently, against the `D_ij`/`D_ab` blocks
already in `W`). **That first-line refutation test was run while scoping, and the candidate
SURVIVES it.** `zeta_weights` (`mp2_gradient.cpp:412-421`) is built from
`result.mo_energy`, which on the DH path **is** the KS `eps` -- so the *values*
are already KS. But the contraction channel is `<W S^(x)>`, i.e. the
**overlap-derivative** channel only. That absorbs the part of `d eps/dR` coming
from basis-function movement; it cannot absorb `V_xc`'s own geometry response,
which is not an overlap effect. For the SCF energy this is harmless because
`E_KS` is variational and the `V_xc` response is captured by
`compute_xc_nuclear_gradient_rks`. **`E_PT2` depends on `eps`
non-variationally, through the denominators, so for it the channel is genuinely
missing.**

**D3's concrete first task**, therefore: write `d eps_p/dR` for a KS reference,
isolate the `V_xc` piece (the part with no HF analogue), and contract it against
the amplitude-squared weights `-sum t^2` -- equivalently against the `D_ij` /
`D_ab` blocks already present in `W`. Then symbolic `sum_A = 0`, toy FD, and the
two-fixture score, per D4.

**Deliverable:** the complete `d/dR` of the XC contribution, written as one
expression with one gradient index, translationally invariant by construction.

**Check before implementing:** verify `sum_A = 0` **symbolically**. The arc has
burned four attempts on terms that were only checked numerically after the fact.

### D4 -- validate against the FD target, then implement (~1 day)

The instruments exist. In order:
1. **Symbolic check** -- `sum_A grad_A = 0` for the D3 expression.
2. **Toy-model FD** -- two Gaussians, two atoms, one grid point; FD the derived
   scalar and compare. Catches factor errors before any C++ is written. This is
   the step N3.5.7.1/.2 showed is worth its cost (both found real factor bugs).
3. **Score against both fixtures** -- `tests/pyscf/dh_gradient_score.py`, and
   **require the coefficient to agree across geometries** (a single-fixture cos
   of 0.5 is reachable by a random vector; see the harness's negative control).
4. **End-to-end FD** on both fixtures. Success = residual falls from 3.89e-4 to
   the grid-convergence floor (~9e-5).

**Stop condition:** if D3's expression scores `cos < 0.9` on either fixture,
**do not tune it**. Return to D1 and find which `R`-dependence was missed. The
arc's failures all came from adjusting a term rather than re-deriving it.

## Risks

1. **The paper may not contain the answer.** Eq. 33 is written in the
   fixed-point convention and is not a complete force expression. If the
   derivation needs a term the paper omits, that is a genuine finding, not a
   transcription error -- and it means Planck will be *more* correct than the
   published formalism. H1.2 (cross-check against ORCA) is the way to confirm
   that, and it needs a licence.
2. **`XC_I`'s double-count explanation (D2) may be wrong.** It is the only
   account of the 3.5e-3 measurement currently available, but it is not yet
   demonstrated. If D2 fails, the whole XC decomposition is suspect and D3 has
   no foundation.
3. **The Z-vector's 3.6x leverage (H2.3)** means a coefficient error there would
   look exactly like a missing XC term. H2.3 verified the operator is
   well-formed, not that every coefficient is right. If D1-D4 produce a clean
   derivation that still does not close the residual, re-examine
   `build_ks_orbital_hessian_op`'s coefficients against Eq. 41 term by term.
4. **UKS is out of scope.** All of the above is the closed-shell (RKS) path.
   Eq. 33's spin-resolved form is the starting point for N3.7 and should not be
   collapsed until the RKS case closes.

## Effort and order

| step | effort | gate |
|---|---|---|
| D1 explicit `R`-dependence | 0.5 d | list contains the grid |
| D2 `XC_I` double-count | 1 d | identifies the Lagrangian term carrying it |
| **D3 companion derivation** | **1-2 d** | **`sum_A = 0` symbolically** |
| D4 validate + implement | 1 d | cos > 0.9 both fixtures, then FD to ~9e-5 |

**Total ~4 days.** D3 is the core; D1 and D2 exist to make D3 correct rather
than another guess. **Do not start at D3** -- it is the step every previous
attempt jumped to.

**Method rule carried from N3.5.7.8-.14:** no probe without a derivation behind
it. Every productive step in this arc came from reading a source; every
speculative probe was negative.

---

## D1-D3 RUN. The derivation closes, and it REPRODUCES what Planck ships.

### D1 -- the decisive structural result: `E_PT2` has NO explicit grid dependence

```
E_PT2 = sum_ijab t_ijab Kbar_ijab ,  t = Kbar/D ,  D = e_i+e_j-e_a-e_b
Kbar  = (ia|jb) - (ib|ja)  -- pure Coulomb ERIs, NO XC, NO grid sum
```

`R` enters through (1) basis functions, (2) `C(R)`, (3) `eps(R)`. **The XC
quadrature enters ONLY through `C` and `eps`** -- there is no `V_xc` matrix
element and no grid sum anywhere inside `E_PT2`.

**This kills candidate 1 outright.** Channel (ii) ("XC_II's coefficients move
with the grid point") is the derivative of a quadrature that `E_PT2` does not
contain. The three-channel decomposition was a correct description of
`d/dR{Phi_XC}`, but `Phi_XC` is a *construction* used to express the XC term,
not something `E_PT2` contains -- so asking which of its channels "move" is the
wrong question. **The fixed-point convention is right, for a reason the paper
never states: there is no quadrature in `E_PT2` to move.**

### D2/D3 -- the XC term derived from scratch equals XC_II + XC_III

`dE_PT2/d eps` contracts with `<p| dF_KS/dR |p>`, and `dF_KS/dR` contains
`dV_xc/dR`. That contraction is

```
sum_munu D_munu * d/dR { <mu|V_xc[rho_P]|nu> }      at FIXED D
```

which is exactly `XC_II + XC_III` -- and explicitly **not** `XC_I`, because
`XC_I` moves `rho_D`, i.e. moves `D`, which is held fixed here. **This
independently explains the 3.5e-3 blow-up** that wiring `XC_I` produced, from
the derivation rather than from measurement.

Since `XC_III` measures ~1e-7, **the derived term IS what Planck already
ships.** The derivation produces no new term.

### Two concrete predictions, both made and both FALSIFIED

**(1) Unrelaxed `D`. WRONG.** I argued `dE_PT2/d eps` uses the unrelaxed `D'`
because the `Z` part is "the C channel, already handled by the Lagrangian".
Measured: residual **1.03e-3 / 8.13e-4**, i.e. **165% / 202% WORSE**.

**The error:** the Lagrangian *determines* `Z` (Eq. 27); it does not *contract*
`Z` against `dF/dR`. That contraction happens once, in Eq. 33, and the paper
uses the **relaxed** `D` there and in Eq. 46 consistently. Conflating "determines"
with "already counted" was the mistake. **Planck's relaxed `D` is correct**, and
N3.5.7.10's empirical choice stands.

**(2) `a_x` on `Gamma`'s exchange. WRONG.** Eq. 17's SCF `Gamma` carries `a_x`
explicitly while Eq. 46's SCF+PT2 `Gamma` does not, and the paper does say it
"suppresses explicit reference to `a_x`". Tested at `a_x = 0.53`: residual
**6.54e-3 / 5.85e-3**, **16x / 21x WORSE**. **N3.5.6 was right** -- full HF
exchange weight in `Gamma` is correct and Eq. 46's coefficients mean what they
say.

### What the derivation actually establishes

The complete `dE_PT2/dR` is four terms, and **Planck has all four**:

```
<D h^x>  +  sum Gamma^PT2 (munu|kt)^(x)  +  [XC term]  +  <W^PT2 S^(x)>
```

The `eps` channel splits into (a) "the operator moves" -- carried by the first
three -- and (b) "the orbitals re-orthonormalise" -- carried by `<W S^(x)>`.
With the XC term in, **the assembly is complete**.

**Therefore the residual is NOT a missing term. It is an error inside one of the
four.** That is a different search, and it inverts the arc's entire premise:
every candidate since N3.5.7.4 has been a hunt for something absent.

**Where to look, in order of remaining suspicion:**
1. **`<W^PT2 S^(x)>`** -- the only term with a known KS-vs-HF gap
   (`vhf_s1occ` is HF `J - K/2` where Eq. 42 wants `-1/2 R(D)`). The
   `s_vhf` family is bounded away from the target by >=72%, but that bound
   assumed the rest of the assembly was right. **With the assembly now proven
   complete, a `W` error is the leading candidate** -- and Eqs. 43/44's `W_ab`
   and `W_ia` blocks have never been checked against Planck term by term.
2. **The Z-vector's coefficients** -- H2.3 proved the operator well-formed
   (symmetric, SPD, exact solve) but not that every coefficient matches Eq. 41.
   It has **3.6x leverage** on the final gradient.
3. **`Gamma^NS`** -- amplitudes are exonerated elementwise (H2.2), but the
   backtransformation `Eq. 47` with its `(1+delta_ij)` weight was checked only
   for convention-equivalence, not numerically.

**Method note:** both falsified predictions were derivation-driven, stated
before measuring, and cost ~20 minutes each to kill. That is the intended cost
of a wrong hypothesis; the failure mode this arc kept hitting was *tuning*
after a partial match instead of predicting first.
