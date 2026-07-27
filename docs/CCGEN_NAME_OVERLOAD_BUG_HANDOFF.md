# ccgen name-overload bug — handoff

**One-line:** ccgen's generated CCSD/CCSDT **doubles** residual is wrong because
`canonicalize.py` keys index identity on `(space, name)` and drops the
free-vs-summed distinction, so a summed dummy that reuses a projector external's
name is confused with that external. This corrupts the coefficients of every
`t1·t2`-mixing doubles term. **Dominant half fixed** (`canonicalize_tensor`
false-zero: maxdiff 4.76 → 1.53, antisymmetry restored). A **separate ~3%
error remains in ccgen's raw generation** (`project.py`/`wick.py`) — proven NOT
the merge key, and the reference is PySCF-validated (so the error is ccgen's,
not the gate's). See "What remains".

**Scope of impact:** the *algebraic* generation path
(`generate_cc_equations`). The default build does **not** compile this — the
shipping CCSD warm-start emits a hardcoded call to the hand-written
`build_residuals` (`src/post_hf/cc/ccsd.cpp`). So this is a generator-correctness
bug, not a wrong-energy-in-production bug. It blocks trusting the algebraic path
(arbitrary-order solver, the diagram work) and any future move to replace the
hand-written residual with generated code.

---

## The bug, precisely

After Wick projection, a doubles term can contain **two distinct index objects
with the same name** — a free external and a summed dummy both called `i`. They
are genuinely different indices (`is_dummy` differs) that a correct evaluator
sums independently. But `canonicalize.py` keys on `(idx.space, idx.name)`
throughout, which ties them, causing two failures:

1. **False zero (the dominant effect — FIXED).** `canonicalize_tensor`'s
   degeneracy check reads an antisym pair holding `(occ,i)` and `(occ,i)` as a
   repeated slot and returns sign **0**, zeroing a legitimate term. Measured:
   **536 of 1312 raw `t1·t2·v` terms were falsely zeroed** — all 536 confirmed
   false (same-name, distinct object), 0 genuine. This under-counts the residual
   by a seed-dependent factor (norm ratio ~0.45–0.67).

2. **Merge conflation (the remaining effect — NOT fixed).** The merge signature
   `_term_signature(term) = (term.factors, term.free_indices)` and the dummy
   relabel `relabel_term_dummies` also collapse free-vs-summed identity, so two
   **genuinely different contractions** (differing only in which same-named
   index is the external) map to one canonical form and are summed into one
   bucket. Measured after fix 1: **57 buckets merge distinct-by-object
   contractions**; one 16-raw-term bucket carries **2 distinct contraction
   values** collapsed together. This over-counts.

The two failures are the same root cause (dropped `is_dummy` bit) in two places
(the tensor canonicalizer vs. the term merge key).

---

## Why it went unnoticed

- The **energy** manifold is all-summed (no externals), so no name can overload
  an external — energies are correct and unaffected. Every energy test passes.
- The **default binary** routes around the algebraic path (hand-written
  residual), so no shipped calculation is wrong.
- The generated equations *look* clean after canonicalization (fresh dummy
  names), so a term-count or repr inspection does not reveal it. Only a
  **numerical residual comparison** on random amplitudes exposes it — which is
  what the gate below does.

---

## What is already fixed (T1.2b)

`python/ccgen/canonicalize.py`:

- New `_antisym_slot_key(idx) = (space_rank, name, is_dummy)` — the trailing
  `is_dummy` breaks same-name free/summed ties and never reorders otherwise.
- Applied at all four sites in `canonicalize_tensor`: the accel `slot_order` /
  `codes` construction, the pure-Python degeneracy check, and
  `_canonical_ordering_for_group`'s sort key.

Effect, on the whole canonical CCSD doubles residual vs. the reference:

| metric | before | after T1.2b |
|---|---|---|
| maxdiff vs reference | 4.76 | **1.53** |
| residual antisymmetric | no | **yes** |
| t1·t2·v norm ratio (ccgen/target) | 0.45–0.67 | **1.03** |
| false-zeroed raw terms | 536 | **0** |
| energy manifold | correct | **unchanged** |

Guardrail held: genuine degeneracy (the **same object** in both slots) still
returns 0 — the fix does not over-correct.

---

## What remains (residual maxdiff 1.53) — and what it is NOT

**The merge is NOT the problem — T1.2b-2 as originally scoped is disproven.**
Measured directly: the by-object-faithful sum of all raw `t1·t2·v` terms (each
with its own coefficient) equals the merged form (canonical term × summed
coefficient) to **maxdiff 0.0**, across every bucket. So `canonicalize_term` +
`merge_term_into_buckets` correctly sum the raw terms; the 57 "conflated"
buckets are conflated in their *object-graph* but still sum correctly (the
coefficients account for it). The final merged terms have **zero** free/summed
name collisions, so the gate's by-name evaluator is faithful on them.

**Where the 1.53 actually is.** The error is:
- **antisymmetric** (so it is a real coefficient error, not a symmetry artifact),
- **small and structural** — the `t1·t2·v` norm ratio is **1.03** (3% over), not
  a gross miscount,
- traceable to **uneven term multiplicities** in the raw generation: grouping
  the final `t1·t2·v` terms by structural signature shows P-partners with
  *unequal* counts (e.g. `t1(a),t1(i) → 14` vs `t1(a),t1(j) → 6`;
  `t1() ,t1(a) → 7` vs `t1(),t1(b) → 4`). A symmetric residual needs equal
  partner multiplicities. The recovered (formerly-false-zeroed) terms are each
  structurally valid (every external is a single open line by object), so they
  are not artifacts — but they enter with a multiplicity/sign pattern that does
  not fully restore P(ij)P(ab) symmetry of the *magnitude*.

**So the remaining fix is NOT in canonicalize's merge key.** It is upstream in
the **raw projection / Wick layer** — the reference is ruled out (see next).

**RESOLVED: the reference is correct; the 3% is in ccgen.** Cross-checked
`gccsd_reference.py` against **PySCF 2.13.0 `gccsd.update_amps`** on identical
GHF integrals and random amplitudes (`test_reference_vs_pyscf.py`):
- H2/STO-3G: reference == PySCF r2 to **maxdiff 0.0** (exact).
- LiH/STO-3G (nocc=4, nvir=8): agree to **2.4e-5** (the precision of the
  `(t2n − t2)·D2` residual reconstruction, not a reference error).

So the gate is authoritative and the remaining 3% t1·t2·v error is entirely in
**ccgen's raw generation** — the false-zero recovery exposed terms whose
relative weights/signs are set during Wick contraction + projection, and those
weights do not fully restore P(ij)P(ab) magnitude symmetry (the uneven
partner multiplicities above). The T1.2b `is_dummy` fix stands (it removed the
dominant error, 4.76 → 1.53, restored antisymmetry). The remaining defect is a
*separate, smaller* one in `project.py` / `wick.py`, now with a trustworthy
gate and a validated reference to drive it.

**Next step:** find which raw t1·t2·v terms are mis-weighted by diffing ccgen's
per-structure multiplicities against the reference's, structure by structure —
the reference can now be treated as ground truth (PySCF-validated).

**T1.2c LANDED — fixed-point canonicalization (a real second bug, but NOT the
3%).** `canonicalize_term` is not idempotent: its dummy relabel can reorder an
antisym factor's indices (`v(c,d,k,l)` → `v(c,d,l,k)`) *after* the sign was
normalized, so one pass gives two canonical forms for one term (the two summed
dummies of an antisym pair distributed across the two equivalent `t1` factors in
`t1·t1·t2·v` swap). Those forms failed to merge. Fix:
`canonicalize_term_to_fixed_point` (new in `canonicalize.py`, mirrors the
existing `optimization/tau._canonical_fixed_point`) used in the generation merge
(`generate.py`).

Effect: **massive term-count reduction** (CCSD doubles 210 → 66, CCD 75 → 18)
because the split forms now merge; **CCD stays exact (0.0)**; energy unchanged.
**But the whole-residual CCSD maxdiff is UNCHANGED at 1.53** — the merged forms
were already summing coefficient-faithfully (proven earlier), so collapsing them
compacts the output without changing the physics. The 3% is elsewhere.

**Ruled out so far for the 3%:** the merge key (sums faithfully), the reference
(PySCF-exact), non-idempotence (fixed, no residual change), and CCD entirely
(exact). The error is **CCSD-t1-specific** and survives every canonicalization
fix — pointing at the **raw projection/BCH coefficients** for specific
t1-containing doubles diagrams. The per-structure multiplicity table (see the
repro) shows P(ab)/P(ij) partners with unequal coefficient-sums (e.g.
`t1(a)t1(i)t2(b,j) = −1/3` vs its partner `= +1/6`, a factor of 2), which is a
generation-time weight error, not a canonicalization one. **This is the open
frontier.** Note the by-object faithful sum of *raw* terms is ~9.8× the
reference (expected — raw is pre-merge and massively redundant), so that diff is
not itself the signal; the signal is the *merged* residual's 2% and the uneven
partner weights.

**CCD IS NOW FULLY CORRECT — the remaining error is t1-specific.** ccgen's CCD
doubles residual (no t1 terms) matches the reference to **maxdiff 0.0** after
T1.2b (it was *wrong before* the fix — CCD also had name-overload in its `f·t2`
and `t2·t2·v` terms and was silently under-counted at 35 terms; it is now
correct at 75). The reference CCD is itself PySCF-validated to 0.0 (t1=0 GCCSD).
So:
- T1.2b **completely fixes CCD** — a clean, PySCF-anchored win.
- The residual 2–3% CCSD error is **confined to the t1·t2-mixing terms** — the
  raw-generation defect involves t1 specifically, narrowing the search further.

**Test fallout (must handle):** T1.2b changes generated term counts (CCD doubles
35→75, CCSD doubles 123→210, singles 21→26) because the false-zeroed terms are
recovered. This breaks **14 count-pinning tests** (3 in `test_optimizations`, 2
in `test_tau`, 9 in `test_diagram`). The CCD counts are now PySCF-*correct* and
their pins should be updated to the new values. The CCSD counts are recovered
but the residual is still 2–3% off, so those pins should NOT be frozen to the
current (not-yet-final) values — update them only once the t1·t2 raw-generation
fix lands and the whole-residual gate reaches 0. Until then the count-pin
failures are expected and documize the intermediate state.

---

## The gate — how to know when it is done

`python/ccgen/tests/test_gccsd_gate.py` + `tests/gccsd_reference.py`.

The reference is the hand-written GCCSD doubles residual (`ccsd.cpp`,
PySCF-validated) transcribed to numpy with the tau intermediates expanded. The
gate evaluates both ccgen's and the reference residual on the **same** random
amplitudes/integrals and diffs the arrays — stronger than antisymmetry alone
(an antisymmetric-but-wrong residual, the failure mode of the naive attempts,
fails this).

Key tests:

- `test_conventions_align_on_the_bare_eri` — calibration: the `⟨ij||ab⟩` term
  matches to 0.0, so a later mismatch is a real coefficient error, not a
  convention artifact. **Load-bearing; if this breaks, the gate is untrustworthy.**
- `test_t1t2v_terms_hit_their_target_T1_GATE` (`@expectedFailure`) — the
  **class-local** gate: the ERI `t1·t2` terms alone must equal `Rref − Rgood`.
  Sharpest signal; flip to a hard assertion when T1.2b-2 lands.
- `test_ccgen_matches_reference_KNOWN_BUG` (`@expectedFailure`) — the
  whole-residual gate. Flip when maxdiff reaches 0.

**Definition of done:** both `@expectedFailure` decorators removed and passing;
CCD residual unchanged; energy manifolds unchanged; CCSDT doubles antisymmetric
and matching (same bug, more instances — re-check after the fix). Then update the
count-pinning regressions in `test_optimizations.py` (the term counts legitimately
change: the recovered terms appear and the conflated ones split).

Use `canonical_fock=True` + a **diagonal** Fock in the gate — the reference is
canonical HF, so this makes it the exact target. (`generate_cc_equations(...,
canonical_fock=True)` is T2: it drops the `f_ov`/`f_vo` terms that are zero for a
canonical reference, which also removed the Fock-driven half of this same bug.)

---

## Reproduce / verify

```bash
cd python
# unit-level (the fixed piece):
CCGEN_NO_ACCEL=1 python -m unittest ccgen.tests.test_canon_sign_bug   # green
# the gate (still expectedFailure until T1.2b-2):
CCGEN_NO_ACCEL=1 python -m unittest ccgen.tests.test_gccsd_gate
# current whole-residual maxdiff (expect ~1.53):
```

The 57-conflated-buckets measurement (drives T1.2b-2):

```python
from collections import defaultdict
from ccgen.hamiltonian import build_hamiltonian
from ccgen.cluster import build_cluster
from ccgen.algebra import bch_expand
from ccgen.project import project
from ccgen.canonicalize import canonicalize_term
raw = project(bch_expand(build_hamiltonian(), build_cluster("ccsd"), 4), "doubles")
grp = [t for t in raw if {f.name for f in t.factors} == {"t1", "t2", "v"}
       and canonicalize_term(t).coeff != 0]
def objgraph(t):
    m = defaultdict(list)
    for f in t.factors:
        for si, x in enumerate(f.indices):
            if x not in set(t.free_indices):
                m[id(x)].append((f.name, si))
    return frozenset(tuple(sorted(v)) for v in m.values())
by = defaultdict(list)
for t in grp:
    by[repr(canonicalize_term(t))].append(t)
conflicts = sum(1 for ts in by.values() if len({objgraph(t) for t in ts}) > 1)
print(conflicts, "buckets merge distinct-by-object contractions")  # expect 57
```

---

## Files

- `python/ccgen/canonicalize.py` — `_antisym_slot_key` (fixed), `canonicalize_tensor`
  (fixed), `relabel_term_dummies` + `_term_signature` (T1.2b-2 target).
- `python/ccgen/tests/test_canon_sign_bug.py` — unit gate (green).
- `python/ccgen/tests/test_gccsd_gate.py`, `tests/gccsd_reference.py` — numerical gate.
- `python/ccgen/tests/test_canonical_fock.py` — T2 (`canonical_fock` mode).
- The C extension `_wickaccel.cpp` (`canonicalize_tensor_layout`) is **not**
  touched — T1.2b feeds it distinct integer codes for same-name free/summed
  indices, so the kernel is correct unchanged. Verify the same holds for any
  accel entry T1.2b-2 uses (`assign_dummy_ordinals`).
- Full context and the history of failed approaches:
  `docs/CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md`.
