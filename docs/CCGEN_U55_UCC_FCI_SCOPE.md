# U5.5 — validating the generated UCC stack against exact diagonalization

Scopes ONE question: **what is the strongest end-to-end correctness gate the generated UCC
path can actually carry, and on what system?**

**Status, 2026-08-24. Not started. The scope as previously written is IMPOSSIBLE and is
rewritten here.** The measurements below are the whole basis for the plan; all but the rank-4
generation cost (~10 min) are reproducible in seconds with the committed `build-ucc` tree.

---

## The premise that does not survive contact

The parent scope said: *"U5.5 — open-shell UCCSDTQ == FCI, the closed-shell analog is the
strongest gate in the whole ccgen effort (Be CCSDTQ vs FCI, 6.4e-11)"*, with a warning to check
the system is not vacuous.

**Three requirements fight, and no system satisfies all three:**

| requirement | needs |
|---|---|
| CCSDTQ is *exact* (so `== FCI` is the assertion) | `n_elec ≤ 4` |
| T4 is worth something (so a broken T4 fails) | ≥ 2 electrons of **each** spin |
| open shell (so it tests the UCC path at all) | `n_alpha ≠ n_beta` |

Four electrons with two of each spin is a **closed** shell. Open-shell with four electrons is
3α/1β — and with one beta electron, beta can be excited at most once, so the `aabb` T4 sector
is identically zero.

**Measured, not argued.** Triplet Be/STO-3G (4 electrons, 3α/1β, `<S²> = 2.000000` exactly, no
spin contamination):

```
in-tree FCI (ROHF reference)   -14.2866221716
hand-written UCCSD             -14.2866221716      <- identical to 1e-10
```

**UCCSD already equals FCI there**, so T3 *and* T4 contribute nothing and a broken implementation
of either passes. It is Li/STO-3G's vacuity from U1.5, one rank up — the trap the old scope
warned about, in the only system its own constraints permitted.

---

## What to gate instead

**Boron/STO-3G doublet — 5 electrons, 3α/2β.** Two beta electrons, so the `aabb` T4 sector is
live. CCSDTQ is *not* formally exact there (T5 would be), but that is the point: the assertion
becomes an *interval*, which is falsifiable in both directions.

```
in-tree FCI (ROHF reference)   -24.1892649766
hand-written UCCSD             -24.1892581442
gap T3+T4 must recover                6.832e-06
```

**The gate:** `ucc4` must land strictly between UCCSD and FCI, and much nearer FCI — everything
but quintuples. Both bounds have teeth: a T3/T4 that contributes nothing fails the lower bound
(it would sit at UCCSD), and one that over-corrects fails the upper (below FCI is variationally
impossible for a converged CC energy in this basis, and was the exact signature of the B5 defect
— *"a total this far below FCI means the generated kernels are grossly wrong"*).

`ucc3` should be gated the same way on the same system, and is much cheaper — **land it first.**

---

## Constraints measured on the tree

**1. In-tree FCI rejects a UHF reference.** `fci.cpp:39` — "requires a converged RHF or ROHF
reference". `ucc4` requires UHF. So the two cannot run from one input, and the reference must
come from a *separate* ROHF-reference FCI run.

That is sound, and worth stating because it looks wrong at first glance: **FCI is
reference-independent** — it is the exact diagonalization in the basis, so ROHF-derived and
UHF-derived FCI agree by construction, provided the geometry and basis match and UHF has not
broken symmetry into a different state. Report `<S²>` alongside; triplet Be gives exactly
2.000000 and boron's doublet should be checked the same way.

**2. `ucc4` reaches the registry only at `PLANCK_CC_MAXORDER >= 4`**
(`generated_kernel_registry.cpp:50`), and `PLANCK_CC_UCC` is default OFF. The committed
`build-ucc` tree is configured at maxorder 3, so U5.5 needs a reconfigure.

**3. Rank-4 generation costs ~10 minutes, and NONE of it is the UCC step.** Measured:

```
generate_cc_equations('ccsdtq')   579.1 s      <- the GCC manifold; the RCC path pays this too
ucc_adapt_equations(...)            1.8 s      <- the UCC step itself
total 18533 terms across 14 sectors (5 quadruples sectors: aaaaaaaa, aaabaaab,
aabbaabb, abbbabbb, bbbbbbbb)
```

**So rank-4 UCC is not meaningfully more expensive than rank-4 RCC** — the adaptation is 0.3% of
the cost. An earlier draft of this scope recorded it as "had not completed after ~10 minutes"
and named it the practical risk; that was a `tail`-buffered probe misread as a hang, and the
inference from it was wrong. The real cost is the GCC generation the build already does at
`PLANCK_CC_MAXORDER=4`, and it is a one-time build cost, not a per-run one.

The remaining rank-4 risk is therefore **compile time, not generation** — 18533 terms across 14
kernels, against the `-O1` registry pin the RCC path already needs. Measure that before
committing, not the generation.

**4. `ucc4` prints "Total Correlated Energy", not a UCC-specific label.** `PostHF::UCCGEN` is
absent from the `method_label` chain (`hf_driver.cpp:1463`). The closed-shell harness
(`ccsdtq_fci_acceptance.py`) greps `Total RCCSDTQ Energy`, so a UCC harness cannot reuse that
anchor. **This intersects U5.3c**, which makes labels rank-derived; doing U5.3c first would give
U5.5 a stable string to grep. Otherwise anchor on `Total Correlated Energy` and accept that it
is not UCC-specific.

---

## Steps

**U5.5a — the `ucc3` interval gate on boron (~S).** Cheapest real assertion, and it exercises
the whole stack at a rank that already generates. `uccsd < ucc3 <= fci`, with `ucc3` materially
closer to FCI than UCCSD is.

**U5.5b — the `ucc4` interval gate on boron (~M).** Same shape, one rank up. The generation
cost is measured and acceptable (~10 min, almost all of it the GCC step the RCC path already
pays); **the open question is compile time** for 18533 terms across 14 kernels. Measure that
first, not the generation.

**U5.5c — register both as regression cases (~S/M).** Blocked on the same missing plumbing U5.4
hit: the runner has **no build-option gating**, so any `ucc*` case fails in a default build
(`PLANCK_CC_UCC` OFF). That plumbing is shared with U5.4's unregistered case and should be done
once, for both.

---

## Traps

- **Do not use a 4-electron system.** It is the only class where `== FCI` is exact, and it is
  exactly the class where T4 is worth nothing. Measured above on triplet Be.
- **Do not assert `== FCI` at 5 electrons.** CCSDTQ is not exact there; the interval is the
  correct assertion and is strictly stronger than a loose tolerance around FCI.
- **Below FCI is not "close enough".** It is the B5 signature and means the kernels are wrong.
- **Check `<S²>` on the UHF reference.** A contaminated reference does not invalidate the gate
  (CC converges to the same fixed point) but it does invalidate the "same state" assumption
  behind using an ROHF-reference FCI number.
- `BASIS_PATH=$PWD/basis-sets` is required to run any input from a build tree; a failed `make`
  can still report exit code 0, so check for the binary.

## Reference numbers (all in-tree, B/STO-3G doublet and Be/STO-3G triplet)

```
B  doublet   UHF        -24.1489886649    FCI  -24.1892649766    UCCSD -24.1892581442
Be triplet   ROHF/UHF   -14.2863733088    FCI  -14.2866221716    UCCSD -14.2866221716
```
