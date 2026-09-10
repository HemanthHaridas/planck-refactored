# The `*corr` gradient defect: 6.6% against Planck's own finite difference

**Status:** scope. Nothing here is built.

**The measurement that defines it** (h2o2 C1, B2PLYP/STO-3G, ultrafine,
`PLANCK_DEBUG_DH_CHANNELS=1` against Planck's own FD of its own energy):

| half | `max|analytic - FD|` | magnitude | share of the defect |
|---|---|---|---|
| KS | **1.02e-6** | 1.27e-1 | 0% |
| `*corr` | **1.24e-3** | 1.88e-2 | **100%** |

`*corr` is **6.63%** wrong against the finite difference of the very energy it
claims to differentiate. Planck's KS gradient is correct to 1e-6.

---

## 1. Why this is a new problem, not the old one restated

The whole arc measured against the **total** gradient, where the number is
3.89e-4. That understates the defect by **3.2x**, because the KS and `*corr`
halves partially cancel in the total. Every candidate score, every
"bounded away by >=72%", every leverage ratio in
`docs/DH_GRADIENT_XC_DERIVATION_SCOPE.md` and
`docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md` was computed against the wrong
scale.

**Re-score every surviving candidate against `corr` vs `corr_FD` before
concluding anything from the old numbers.** They are not wrong in direction —
the pairing was total-vs-total, which is self-consistent — but they are wrong
in magnitude, and several were rejected on "removes only X% of the target".

## 1b. The angular-momentum hypothesis: RAISED, TESTED, FALSIFIED

**Hypothesis (worth recording because the pattern really does suggest it):** in
STO-3G, H is pure `S` (`l = 0`) while O carries `S` + `SP` (`l = 1`). The
defect is concentrated on the two oxygens along the O-O axis with H components
~6x smaller. So it looks like a defect in how the derivative raises/lowers
angular momentum -- `_compute_eri_deriv_elem`'s
`+2*alpha*ERI(l+1) - l*ERI(l-1)`, whose lowering term **only fires when
`l > 0`**, i.e. on O and never on H.

**Two independent measurements kill it.**

1. **The same derivative-ERI path is FD-gated and passes.**
   `water_rmp2_gradient_fd` (water/STO-3G, so O p-shells, the same
   `_compute_eri_deriv_elem`, the same MP2 amplitudes) agrees with finite
   difference to **2.2e-07** against a 3e-4 tolerance. A broken raise/lower
   would fail this too.

2. **The defect survives with NO p-shells anywhere.** A pure-`s` fixture --
   H4 as two H2 units, STO-3G, so every function is `l = 0` and the lowering
   term can never fire -- shows a **7.58%** relative defect, slightly WORSE
   than H2O2's 6.63%:

   ```
     H1: +1.740e-06  +6.982e-06  -3.484e-04
     H2: +1.328e-06  +3.740e-06  +3.479e-04
     H3: +9.185e-05  -2.903e-04  +7.894e-05
     H4: -9.489e-05  +2.796e-04  -7.847e-05
   ```

**What the pattern actually tracks is BONDED PAIRS, not angular momentum.**
On H4 the defect is antisymmetric along each H-H bond axis
(-3.484e-4 / +3.479e-4 on the bonded pair), exactly as it is along O-O in
H2O2. In H2O2 the heavy-atom pair simply *is* the bonded pair, so the p-shells
were **correlated with the signature, not causing it**.

Fixture committed at `tests/inputs/exploratory/dh_gradient/h4_s_only_b2plyp.hfinp`
-- it is the cheapest discriminator in the set (4 s-functions) and it
**excludes every angular-momentum-dependent explanation in one run**. Use it
first on any future candidate.

## 2. What the defect looks like (measured, use it to discriminate)

```
  max|d|           1.2431e-03
  |d| / |corr_FD|  6.63%
  net force        1.000e-08     -> 0.0% of max|d|
  translation-free 100.0%
```

**Reproduced on three fixtures**, which is what makes the structural
properties below trustworthy rather than a single-geometry artifact:

| fixture | `max|d|` | `|corr_FD|` | relative | net force |
|---|---|---|---|---|
| h2o2 C1 | 1.243e-3 | 1.876e-2 | **6.63%** | 1.0e-8 |
| h2o2b C1 | 9.827e-4 | 1.636e-2 | **6.01%** | 2.7e-8 |
| H4 (s-only) | 3.484e-4 | 4.594e-3 | **7.58%** | 2.7e-8 |

Per-atom, on the C1 H2O2 fixture (O at y = +-0.717, so **y is the O-O axis**):

```
   atom 1 (O): +2.660e-04  -1.197e-03  +8.856e-05
   atom 2 (O): -2.084e-04  +1.243e-03  -7.698e-05
   atom 3 (H): -2.005e-05  +1.879e-04  -3.086e-05
   atom 4 (H): -3.750e-05  -2.344e-04  +1.929e-05
```

Three properties worth more than any single number:

1. **Perfectly translation-free** (net force 1e-8 against a 1.2e-3 defect).
   Whatever is wrong conserves momentum, which **excludes** a missing
   one-centre term, a moving-frame/grid term, and any term that does not sum
   to zero over atoms by construction.
2. **Bond-directed and antisymmetric between the bonded pair**
   (-1.197e-3 / +1.243e-3 along O-O, equal to 4%). **NOT a heavy-atom or
   angular-momentum signature** -- see section 1b: the identical pattern
   appears along each H-H bond in a pure-`s` fixture. It tracks bonds.
3. **NOT a uniform scale error.** The elementwise ratio `d / corr_FD` spans
   +0.002 to -15.3 (the large value is where `corr_FD` itself is ~1e-5).
   A wrong prefactor on the whole term is therefore excluded; on the two large
   y-components the ratio is +0.064 / +0.068, consistent, but it does not hold
   elsewhere.

## 3. What is already eliminated, and what that now means

All of these were measured against the **total**, so their *magnitudes* need
re-deriving — but the ones resting on an exact identity or a cos/consistency
argument survive unchanged:

| eliminated | basis | survives re-scaling? |
|---|---|---|
| Z-vector operator (diag/J/K) | exact vs HF-CPHF, **3.7e-15** | **yes** — identity, not a fit |
| `Gamma^NS` / Eq. 47 | contraction invariant **4.000000000000** | **yes** — identity |
| `h_op`'s XC channel | bounded, 0.36x/0.44x of the total | **NO** — re-measure vs 1.24e-3 |
| `vhf1` HF exchange weight | implied `kx` 0.984/0.990, 2 geometries | **yes** — a ratio, scale-free |
| KS-veff at all 3 sites | 1.4x-83x **worse** | **yes** — worse is worse |
| combination error | per-term translation invariance 1e-14 | **yes** |
| amplitudes `t2` | elementwise vs PySCF **8e-9** | **yes** |

**The XC-channel bound is the one that flips.** It was called safe because its
whole contribution (1.41e-4) was 0.36x the total residual. Against the real
1.24e-3 defect it is 0.11x — still short, but it was never the leading
candidate on magnitude alone, and the argument that retired it was partly the
scale.

## 4. Steps

**A1. Re-score every candidate against the right target.** Change
`dh_gradient_score.py` to take `corr_FD` (Planck's own, per-fixture) as the
target and Planck's `*corr` as the prediction, instead of total-vs-total. This
is bookkeeping, not new physics, and it must come first — otherwise every
number in the next step is off by 3.2x.
*Verify:* the XC_II positive control still scores cos ~0.95 at scale ~1.

**A2. DONE -- `corr_FD` measured on three fixtures** (table in section 2), all
committed as reference blocks so nobody re-runs 24 SCFs each. Two geometries
is the standing requirement; the third (s-only) additionally serves as the
angular-momentum discriminator.

**A3. Use the structural properties as a FILTER, not a fit.** Any candidate
must be translation-free by construction, **bond-directed and antisymmetric
between bonded pairs**, non-uniform in its ratio to `corr_FD`, and
**independent of angular momentum** (it must survive on the s-only fixture).
That last one is cheap and rules out a whole family in a single run.

**A4. The first specific candidate: the `s_zeta` / `s_im1` overlap terms.**
They are the only `*corr` pieces contracted against `S^(x)`, which is
intrinsically bond-directed and antisymmetric between a bonded pair — matching
property 2. `s_zeta` is 5.1e-1 in magnitude, so a 0.2% error there is the whole
defect. D7 verified `W`'s oo/vv blocks and **D9 now provides the ov/vo oracle
D7 lacked** (`tests/pyscf/dh_zov_derivation_check.py`) — so this is testable
now in a way it was not when D7 stopped.
*Verify:* score against `corr_FD` on both fixtures; require a consistent
coefficient.

**A5. If A4 is negative, FD the `*corr` accumulators individually.**
`PLANCK_DEBUG_RMP2_TERMS` already dumps 14 of them. Each is a contraction of a
known quantity against a derivative integral, so each has its own FD. This is
brute force and slow (14 x 24 SCF runs) but it is **guaranteed to localise**,
which nothing tried so far has been.

## 5. Traps carried forward

- **Score against `corr_FD`, never the total.** The halves cancel.
- **Two geometries, always** — and check the rank of any span-fit against the
  dimension of the translation-free space (9 for a 4-atom fixture) before
  believing a small residual. A rank-9 fit returns 1e-16 for anything.
- **Do not localise by elimination.** Measuring the halves directly took one
  probe and overturned a conclusion reached by subtracting totals.
- **A different functional is a different test.** A pure-B3LYP self-FD does not
  isolate B2PLYP's KS half.
- The grid defect (`docs/DFT_GRID_CONVERGENCE_SCOPE.md`) does **not** confound
  this: `*corr` moves only **6.8e-8** between fine and ultrafine, against a
  1.24e-3 defect — 18000x apart. **The two are independent; either can be done
  first.**
