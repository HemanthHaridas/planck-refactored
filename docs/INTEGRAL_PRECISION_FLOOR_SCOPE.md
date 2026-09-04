# Planck-vs-PySCF Integral Precision Floor

This file answers a narrower architecture question:

**Why does Planck agree with PySCF at only a ~1e-8 floor across every method, and is that a bug?**

> **RESOLVED.** Root cause found: basis-set coefficient precision, not an
> integral bug. No fix is required for correctness; see "What was found"
> below. The "Superseded diagnostic trail" section at the end is kept as
> history — it documents the investigation steps taken before the root cause
> was isolated, including a red herring, and is retained rather than deleted.

## Short answer

The entire ~1e-8 "integral floor" is a basis-set data-precision difference,
not an integral-engine bug. Planck's GBS files store STO-3G coefficients and
exponents to 10 significant figures (e.g. Be 1s `0.1543289673`, exponent
`30.16787069`); PySCF's built-in library uses the 8-figure basis-set-exchange
standard (`0.15432897`, `30.167871`). The coefficient difference is ~2.5e-9,
which propagates to ~6e-8 in contracted integrals and ~1e-8 in the SCF energy.
There is no integral-engine bug and nothing to fix in the recurrences,
normalization code, or Boys function — all verified correct. The ~1e-8
disagreement with any PySCF reference is expected and benign: the two codes
use different-precision truncations of the same basis. Planck's 10-digit
values are arguably more precise.

## Where the logic lives

- `src/integrals/boys.h` / `src/lookup/boys.cpp` — the Boys function table
  and interpolation (verified correct, not the cause)
- Planck's GBS basis-set files (10-significant-figure coefficients) versus
  PySCF's built-in 8-figure basis-set-exchange library data
- `tests/eri_derivative_kernels.cpp` — the OS/HGP cross-engine ERI gate
- `generated_ccsdtq_energy_wrong` investigation (R0-R8) — where this floor
  was first surfaced while chasing a different, unrelated CC discrepancy

## What invariants matter

### 1. A uniform floor across independent methods points to a shared upstream cause, not a method-specific bug

Planck agreed with PySCF at a uniform ~1e-8 floor across RHF, CASSCF, and
CCSDTQ, at byte-identical basis and geometry, fully converged, with
spherical/tolerance ruled out. That consistency across independent methods is
itself evidence for a shared integral-evaluation (or, as it turned out,
input-data) difference rather than a method-specific defect.

Design rule:

- When several independently-implemented methods disagree with an external
  reference by the same magnitude, suspect a shared upstream input or
  primitive before auditing each method's own algebra.

### 2. MO-basis comparisons conflate the thing being measured with orbital rotation

FCIDUMP-level (MO-basis) attribution mixes the integral difference with
orbital rotation, since both the one-electron and two-electron MO integrals
differ at ~1e-7 and partially cancel to ~1e-8 in the totals — a confusing
signal. Always attribute at the AO level instead.

Design rule:

- Never diagnose an integral-precision question from MO-basis (FCIDUMP-level)
  numbers alone; compare AO-level S/T/V/ERI directly against the same-basis
  external reference.

### 3. A cross-implementation probe must compare against the same engine's own integrals

The CCSDTQ investigation that surfaced this floor burned a long chain (R0-R8)
partly because a probe injected PySCF-derived amplitudes into a Planck-
integral kernel; the ~1e-8 residual was actually the integral difference,
misread as a CC bug.

Design rule:

- When cross-checking one implementation against another, keep every input
  (including basis-derived amplitudes or integrals) sourced from the same
  engine as the kernel under test — do not mix a reference engine's derived
  quantities into the engine being validated.

## What was found

1. **The root cause is basis-set coefficient precision, not an integral
   bug.** Proof: a hand-computed `S(1s,2s)` from exact analytic overlap
   formulas matches PySCF exactly (0.0 difference) and Planck by 6.25e-8 — a
   pure normalization/data difference, with no angular-momentum recurrence
   involved (both are s-shells). Feeding PySCF Planck's own 10-digit
   coefficients then makes `S(1s,2s)` match Planck to 0.0, and RHF match to
   1.78e-14 (machine precision). Both integral engines are essentially
   identical; the difference was entirely in the input basis data.
2. **The Boys function is not the cause.** Planck's `Lookup::boys` table plus
   6-term Taylor interpolation is accurate to ~1e-12 absolute worst-case
   (verified against a scipy incomplete-gamma reference over n = 0..8,
   x = 0..60) — 10,000x below the ~1e-8 floor. This was the prime hypothesis
   going in and it is refuted.
3. **The uniform floor is measured across methods:**

   | Method | System | Planck vs PySCF |
   |---|---|---|
   | RHF | Be/STO-3G (single atom, no geometry) | 7.6e-8 |
   | RHF | water/STO-3G (identical geometry) | 1.5e-8 |
   | CASSCF | h2_cas22 | 6.0e-10 |
   | CASSCF | water_cas44/STO-3G | 1.2e-8 |
   | CASSCF | ethylene_casscf/cc-pVDZ | 2.3e-8 |
   | CCSDTQ | Be/STO-3G | 6.16e-8 |

   CASSCF matches PySCF at the same ~1e-8 scale as RHF, not tighter (an
   earlier belief that "CASSCF matches to 1e-11" was wrong); CASSCF's
   variational relaxation suppresses the sensitivity somewhat (h2 = 6e-10)
   but the floor is the same underlying difference.
4. **Ruled out as causes:** SCF convergence (Planck RHF at `tol 1e-12` is
   bit-identical to default, converging in 2 iterations; PySCF is
   tight-converged); spherical vs Cartesian (a no-op for the tested cases,
   since STO-3G has only s, p shells); basis identity (Planck and PySCF
   STO-3G exponents/coefficients were confirmed byte-identical at the level
   this investigation initially checked, before the deeper 8-vs-10-digit
   discrepancy was found); geometry (Be is a single atom; water was verified
   at the identical geometry — an earlier apparent 1.5e-4 discrepancy was a
   geometry-entry mistake).
5. **Calibration point:** PySCF's own density-fitting approximation moves the
   energy by ~1e-4, so Planck agreeing with PySCF's exact integrals to ~1e-8
   is very tight — the two independent engines essentially agree, differing
   only in the last 1-2 digits.

## Validation strategy that should remain in place

- Planck's PySCF-anchored regression gates (CASSCF 1e-9, CCSDTQ 1e-7, RHF
  1e-9) are pinned to Planck's own converged values, precisely because a
  fresh PySCF run differs at ~1e-8. That is already the correct design — the
  gates encode Planck's own numbers, not a live PySCF comparison — and
  explains why those references are Planck-specific and why the
  `be_rccsdtq` FCI gate sits at 6e-8, not 1e-11.
- The OS/HGP cross-engine ERI gate (`tests/eri_derivative_kernels.cpp`,
  agreement to 1e-13) remains the check that a Boys or recurrence defect
  would be common to all Planck engines and would show as a uniform PySCF
  disagreement — exactly the signature that was chased and ruled out here.

## What was not done, deliberately

There is nothing to fix for correctness, so no code change was made. Two
optional items were identified and explicitly not pursued:

- If bit-agreement with PySCF is ever wanted for a direct cross-code
  validation, the only lever is making the basis coefficients identical —
  either regenerating Planck's GBS files from the 8-digit basis-set-exchange
  source PySCF uses, or loading Planck's GBS into PySCF. This is a data
  change, not a code change, and was judged not worth doing: Planck's
  10-digit coefficients are more precise, and the physics already agrees to
  8 figures.
- A loose thread surfaced during scoping, not yet independently verified:
  `CLAUDE.md`'s CASSCF gate table lists `water_cas44` at Delta = 1.2e-8 while
  its regression gate tolerance is 1e-9 — a 1.2e-8 delta would fail a 1e-9
  gate on the face of it. Worth a quick check of whether the gate actually
  passes at its committed tolerance (a possible stale-doc-vs-real-tolerance
  mismatch), since it bears on whether the ~1e-8 floor is already quietly at
  the edge of the CASSCF gates. If the floor is at the gate edge, narrowing
  it would have real regression value, not just cosmetic precision.

## Superseded diagnostic trail (R1/R2, kept as history)

This section documents the investigation as it was actually conducted, before
the coefficient-precision root cause (above) was isolated by the hand-computed
overlap check (R3). It is retained because it records a genuine red herring
worth remembering, not because it is still an open question.

**R1 (2026-08-09): even the Boys-free integrals differed at ~1e-7**, which
initially looked like it placed the floor in the fundamental OS integral
evaluation rather than in Boys or the ERI path. Be/STO-3G AO integrals,
Planck vs PySCF (`mol.intor`, `cart=True`), element max|diff|:

| Integral | Boys? | max\|diff\| | structure |
|---|---|---|---|
| S (overlap) | no | 6.25e-8 | entirely in the 1s-2s cross element; every diagonal (=1) and the whole p-block bit-identical |
| T (kinetic) | no | 1.05e-7 | s-block (1s,2s) and the p-shell diagonals (2px/py/pz ~1.05e-7 each); p cross-terms 0 |
| V (nuc attr) | yes | 5.0e-7 | s-block and p-diagonals (~1.8e-7) |

At the time this read as: S's diagonals are exact (self-normalization is
right) but the 1s-2s cross overlap is off by 2.4e-7 relative, suggesting a
primitive/contraction subtlety; T and V additionally differed on the p-shell
diagonals while S did not, suggesting the difference grows with
operator/angular-momentum complexity — the signature of an OS
angular-momentum recurrence or primitive-normalization defect. **This read
was a red herring**: it was actually the same coefficient difference
propagating through progressively more terms as angular momentum increased,
not a recurrence bug. R3's hand computation (see "What was found" above) cut
through it directly.

**R2: the Boys function was ruled out as the cause**, as already stated
above — Planck's `Lookup::boys` table plus 6-term Taylor is accurate to
~1e-12 absolute, verified against a scipy incomplete-gamma reference over
n = 0..8, x = 0..60.

Investigation scripts referenced during this trail: `/tmp/r2_boys.py`,
`/tmp/r1_compare.py`; an AO-dump debug hook was env-gated as
`PLANCK_DUMP_AO_INTEGRALS` in `src/hf_driver.cpp` (uncommitted debug tooling,
not part of the shipped source).
