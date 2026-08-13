# Planck-vs-PySCF Integral Precision Floor — Investigation Scope

## Status: ROOT CAUSE FOUND — basis-set coefficient precision, NOT an integral bug

**The entire ~1e-8 "integral floor" is a basis-set data-precision difference.** Planck's GBS files store STO-3G coefficients/exponents to **10 significant figures** (e.g. Be 1s `0.1543289673`, exp `30.16787069`); PySCF's built-in library uses the **8-figure** basis-set-exchange standard (`0.15432897`, `30.167871`). The coeffs differ ~2.5e-9, which propagates to ~6e-8 in contracted integrals and ~1e-8 in the SCF energy.

PROOF (R3): a hand-computed S(1s,2s) from exact analytic overlap formulas matches **PySCF exactly (0.0)** and **Planck by 6.25e-8** — a pure normalization/data difference, no angular-momentum recurrence involved (both s-shells). Then feeding PySCF **Planck's own 10-digit coefficients**: S(1s,2s) matches Planck to **0.0**, and RHF matches to **1.78e-14** (machine precision). So both integral ENGINES are essentially identical; the difference was entirely the input basis data.

CONCLUSION: there is **no integral-engine bug** and nothing to fix in the recurrences, normalization code, or Boys function (all verified correct — R1/R2/R3). The ~1e-8 disagreement with any *PySCF* reference is expected and benign: the two codes use different-precision truncations of the same basis. Planck's 10-digit values are arguably *more* precise.

### Fix scope (optional, cosmetic)

There is nothing to fix for correctness. The only actionable items, both optional:

1. **Regression references** — Planck's PySCF-anchored gates (CASSCF 1e-9, CCSDTQ 1e-7, RHF 1e-9) are pinned to Planck's *own* converged values precisely because they differ from a fresh PySCF run at ~1e-8. That is already the correct design (the gates encode Planck's numbers, not PySCF's). No change needed — but this explains WHY those references are Planck-specific and why the `be_rccsdtq` FCI gate sits at 6e-8, not 1e-11.

2. **If bit-agreement with PySCF is ever wanted** (e.g. for a direct cross-code validation), the only lever is to make the *basis coefficients* identical — either regenerate Planck's GBS files from the 8-digit basis-set-exchange source PySCF uses, or load the basis into PySCF from Planck's GBS. This is a *data* change, not a code change, and is not worth doing: Planck's 10-digit coefficients are more precise, and the physics already agrees to 8 figures.

The everything-below (R1/R2 findings) is retained as the diagnostic trail; note R1's "difference grows with angular momentum" read was a red herring — it was the coefficient difference propagating through progressively more terms, not a recurrence bug. R3's hand-computation cut through it.

---

## (Superseded) R1+R2 diagnostic trail

**R1 result (2026-08-09): even the Boys-FREE integrals differ at ~1e-7**, so the
floor is in the fundamental OS integral evaluation, not Boys and not the ERI path.
Be/STO-3G AO integrals, Planck vs PySCF (`mol.intor`, cart=True), element max|diff|:

| Integral | Boys? | max\|diff\| | structure |
|---|---|---|---|
| **S** (overlap) | no | **6.25e-8** | ENTIRELY in the 1s-2s cross element; every diagonal (=1) and the whole p-block are bit-identical |
| **T** (kinetic) | no | **1.05e-7** | s-block (1s,2s) AND the p-shell DIAGONALS (2px/py/pz ~1.05e-7 each); p cross-terms 0 |
| **V** (nuc attr) | yes | **5.0e-7** | s-block and p-diagonals (~1.8e-7) |

Diagnosis: S's diagonals are exact (self-normalization is right) but the 1s-2s
CROSS overlap is off 2.4e-7 relative → a primitive/contraction subtlety in the
cross-s-shell contraction. T and V additionally differ on the p-shell DIAGONALS
while S does not → the difference GROWS with operator/angular-momentum complexity
(overlap < kinetic < nuclear), the signature of the OS angular-momentum recurrence
/ primitive normalization diverging from libcint at ~1e-7, NOT a Boys or
accumulation issue. So the ~1e-8 energy floor traces to Planck's base 1e integral
recurrences (and, by extension, the same primitives feed the 2e ERIs).

**R2 result: the Boys function is NOT the cause** — Planck's `Lookup::boys` table +
6-term Taylor is accurate to ~1e-12 absolute (verified against a scipy
incomplete-gamma reference over n=0..8, x=0..60). Refuted the prime hypothesis.

**Remaining (R3/R4):** pin whether it's (a) primitive Gaussian normalization
`N_prim` / contraction `Nc` (Norm Factors gotcha), or (b) the OS vertical/horizontal
recurrence accumulation, by comparing a single contracted primitive-overlap
`Σ c_p c_q S_pq^prim` against a hand/mpmath computation. Then R4 decides
cost/benefit. Scripts: `/tmp/r2_boys.py`, `/tmp/r1_compare.py`; AO dump env-gated
`PLANCK_DUMP_AO_INTEGRALS` at `src/hf_driver.cpp` (~line 931, uncommitted debug).

**R2 result (2026-08-09): the Boys function is NOT the cause.** Planck's
`Lookup::boys` table + 6-term Taylor interpolation is accurate to **~1e-12
absolute** worst-case (n=0..8, worst at x≈0.1, the interpolation midpoint) —
verified by extracting the actual `boysTable` and running the identical
interpolation against a scipy incomplete-gamma reference. Absolute error is what
enters the integral sum; ~1e-12 is 10,000× below the ~1e-8 floor. (The large
*relative* errors, up to 7e-5 at x≈59, are where F_n(x) itself is ~1e-12, so the
5e-16 absolute error there is physically negligible.) The prime hypothesis is
refuted. **R1 (AO-level S/T/V/ERI attribution) is now the primary step** — with
Boys exonerated, the ~1e-8 must be contraction normalization (Nc), fp
accumulation order, screening thresholds, or the orthogonalizer. Script:
`/tmp/r2_boys.py` (scratchpad).

This is a **diagnostic** investigation: locate *where* the ~1e-8 integral
difference between Planck and PySCF originates, and decide *whether* it is worth
improving. It is not a commitment to fix anything — the physics already agrees to
8 significant figures.

## Origin of this investigation

While chasing the ~6e-8 gap between Planck's generated CCSDTQ energy and the
PySCF-FCI reference for Be/STO-3G, the residual was traced (see
`generated_ccsdtq_energy_wrong` memory, R0–R8) not to any CC bug but to a
**uniform ~1e-8 difference in the underlying integrals**. The generated CC
kernels are correct; they simply run on Planck's integrals, which differ from
PySCF's at ~1e-8.

## The established facts (do not re-derive)

Planck agrees with PySCF at a **uniform ~1e-8 floor across every method**, at
byte-identical basis and geometry, fully converged, spherical/tolerance ruled out:

| Method | System | Planck vs PySCF |
|---|---|---|
| RHF | Be/STO-3G (single atom, no geometry) | **7.6e-8** |
| RHF | water/STO-3G (identical geometry) | **1.5e-8** |
| CASSCF | h2_cas22 | 6.0e-10 |
| CASSCF | water_cas44/STO-3G | 1.2e-8 |
| CASSCF | ethylene_casscf/cc-pVDZ | 2.3e-8 |
| CCSDTQ | Be/STO-3G | 6.16e-8 |

The consistency of the floor across independent methods is itself evidence for a
**shared integral-evaluation difference** rather than a method-specific bug. The
CASSCF cases match PySCF at the *same* ~1e-8 scale as RHF — they are **not**
tighter (an earlier "CASSCF matches to 1e-11" belief was wrong); CASSCF's
variational relaxation suppresses the sensitivity somewhat (h2 = 6e-10) but the
floor is the same integral difference.

Ruled out as causes:
- **SCF convergence** — Planck RHF at `tol 1e-12` is bit-identical to default
  (converges in 2 iterations); PySCF is tight-converged.
- **Spherical vs Cartesian** — no-op for the tested cases (STO-3G has only s,p
  shells, so Cartesian ≡ spherical; PySCF gives identical `cart=True/False`).
- **Basis** — Planck and PySCF STO-3G exponents/coefficients are byte-identical.
- **Geometry** — Be is a single atom; water was verified at the identical
  geometry (an earlier 1.5e-4 "discrepancy" was a geometry-entry mistake).

FCIDUMP attribution (MO basis, so conflated with orbital rotation): **both** the
one-electron (occ trace 2·Σh_ii differs 2.7e-7) and two-electron MO integrals
differ at ~1e-7, **partially cancelling** to ~1e-8 in the totals. For calibration,
PySCF's own density-fitting approximation moves the energy by ~1e-4, so Planck
agreeing with PySCF's *exact* integrals to ~1e-8 is very tight — the two
independent engines essentially agree, differing only in the last ~1–2 digits.

## Prime hypothesis: the Boys function table

`src/integrals/boys.h` → `src/lookup/boys.cpp`: the Boys function `F_n(x)` is a
**precomputed table (step 0.1, n = 0..65) with 6-term Taylor interpolation**, plus
an asymptotic formula beyond the table range. This is the classic ~1e-8-accuracy
Boys implementation. It feeds **every** ERI and nuclear-attraction integral.
PySCF/libcint evaluates Boys to ~1e-14.

Supporting evidence:
- The OS↔HGP cross-engine ERI gate agrees to **1e-13**
  (`tests/eri_derivative_kernels.cpp`), i.e. both engines share the *same* Boys
  function. A Boys inaccuracy would therefore be **common to all Planck engines**
  and appear as a *uniform* PySCF disagreement — exactly what is observed.
- The Be run used `engine auto`; the floor is identical regardless of engine
  dispatch, consistent with a shared upstream (Boys) source.

## Investigation steps

Each step is small and has a runnable check that fails loudly if that layer is
not the cause. Stop at the first rung that gives a definitive attribution.

### R2 — Isolate the Boys function directly (~S, do FIRST)

Cheapest and most likely decisive: a pure unit comparison, no SCF.

Compare Planck's `HartreeFock::Lookup::boys(n, x)` against a high-precision
reference (mpmath, or `scipy.special` via the lower incomplete gamma
`F_n(x) = γ(n+½, x) / (2 x^{n+½})`) over the `(n, x)` grid ERIs actually hit:
n = 0..~8 for STO-3G/cc-pVDZ, x = 0..40, dense near the table-interpolation
midpoints and the asymptotic crossover.

- **Check:** max relative error of the table + 6-term Taylor interpolation.
- **Expected:** ~1e-8 somewhere in the mid-`x` interpolation region or at the
  asymptotic-crossover; ~1e-14 would refute the hypothesis.
- **Deliverable:** the `(n, x)` region and magnitude of Boys error.

### R1 — AO-level attribution (~S)

Runs regardless of R2 outcome; guards against tunnel-vision on Boys. Compares at
the **AO level** (no MO-rotation confound, unlike FCIDUMP).

Planck already writes `_overlap` to checkpoint (`src/io/checkpoint.cpp:421`).
Extend a debug dump to emit AO **S, T, V, and a few ERIs**, and diff element-wise
against PySCF `mol.intor('int1e_ovlp')`, `int1e_kin`, `int1e_nuc`, `int2e` at the
identical basis/geometry.

- **Discriminator:** S and T are **Boys-free** (Gaussian products only); V and
  ERI use Boys.
  - S, T agree ~1e-13 but V, ERI differ ~1e-8 → isolates it to the
    **Boys-dependent** path → confirms the hypothesis.
  - S, T also differ → contraction-normalization (`Nc`, see the Norm Factors
    gotcha) or a deeper primitive issue, not (only) Boys.

### R3 — Attribute quantitatively: does Boys error explain the RHF gap? (~M)

Only if R2 shows a Boys error ~1e-8. Verify it propagates to the observed RHF
gap:
- (a) Temporarily swap Planck's Boys for a high-precision evaluation and re-run
  Be/water RHF. Gap collapses to ~1e-12 → **Boys is the sole cause**.
- (b) Or bound analytically: integral error ≈ Boys error × contraction magnitude.

- **Check:** high-precision-Boys RHF matches PySCF to ~1e-11 → Boys is the whole
  story. Residual gap remains → a second source (normalization, screening
  threshold, fp accumulation order) — measure and name its contribution.

### R4 — Decide and record (~S)

State the conclusion and the cost/benefit; this closes the "can we tighten the
floor" question.

- **If Boys is it (likely):** the fix would be a higher-precision Boys — a denser
  table + more Taylor terms, or a proper series / `erf`-based evaluation for low
  `n`. This is a real accuracy improvement that would tighten **every** integral
  (RHF/CASSCF/CC alike) against PySCF simultaneously. Scope it as a **separate**
  change with its measured cost (the table is fast; a series evaluation is
  slower — quantify the ERI-build slowdown).
- **If not just Boys:** document each residual source with its measured
  contribution.
- Either way: record whether the ~1e-8 floor is improvable and at what cost.

## Sequencing

```
R2 (Boys unit vs high-precision)   ~S — pure Python vs one C++ function, no SCF.
                                        Most likely decisive; do first.
R1 (AO S/T/V/ERI attribution)      ~S — guards against Boys tunnel-vision;
                                        run regardless.
R3 (swap Boys, re-run RHF)         ~M — only if R2 shows ~1e-8 Boys error.
R4 (decide + cost/benefit)         ~S — the actionable conclusion.
```

## Non-goals

- **Fixing it.** This is diagnosis. A Boys upgrade (or any integral-precision
  work) is a real but optional investment; R4 states the tradeoff rather than
  committing.
- **Bit-matching libcint.** Making Planck's integrals byte-identical to PySCF is
  out of scope — the target is understanding and, at most, deciding whether to
  narrow the floor.

## Risks / caveats

- The ~1e-8 could be a **sum of small sources** (Boys + normalization +
  accumulation order) that individually are smaller than the total. R1's AO
  attribution is the guard: if the Boys-free integrals (S, T) also differ, Boys
  is not the whole answer.
- MO-basis comparisons (FCIDUMP) conflate integral difference with orbital
  rotation — always attribute at the **AO level** (R1), never MO.
- A fixed-point / integral probe must compare against the **same engine's**
  integrals. The CCSDTQ investigation burned a long chain (R0–R8) partly because
  a probe injected PySCF-derived amplitudes into a Planck-integral kernel; the
  ~1e-8 residual *was* the integral difference, misread as a CC bug. Do not
  repeat that: compare like against like.

## Loose thread surfaced during scoping

`CLAUDE.md`'s CASSCF gate table lists `water_cas44` Δ = 1.2e-8 while its
regression gate atol is 1e-9 — a 1.2e-8 delta would **fail** a 1e-9 gate. Worth a
quick check of whether the gate actually passes at its committed tolerance (stale
doc delta vs real tolerance mismatch), since it bears on whether the ~1e-8 floor
is already quietly at the edge of the CASSCF gates. Independent of R1–R4 but
related: if the floor is at the gate edge, narrowing it (R4) has real regression
value, not just cosmetic precision.
