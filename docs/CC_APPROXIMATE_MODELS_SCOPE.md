# How should ccgen generate the approximate iterative CC models (CC2, CC3, CCn, CCSDT-n)?

Scope, opened 2026-11-25. Nothing here is built. This doc expires when the
first model lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md`. Siblings: `docs/CCSDT_Q_SCOPE.md` (the
non-iterative corrections: [T], (T), (Q)) and `docs/CCSD_T_SCOPE.md`.

## What these models are

Each one keeps the singles and doubles equations of the parent method and
truncates the equation for the highest excitation rank by perturbation
order. The highest rank is then never a fully coupled unknown:

| Model | Parent | Truncated equation | Highest-rank equation kept | Cost / iteration |
|---|---|---|---|---|
| CC2 | CCSD | doubles | ⟨μ₂\| H̃ + [F, T₂] \|0⟩ | o²v³ (o v⁴ with RI) |
| CC3 | CCSDT | triples | ⟨μ₃\| [F, T₃] + [H̃, T₂] \|0⟩ | o³v⁴ |
| CCn (n ≥ 4) | CC with Tₙ | rank n | ⟨μₙ\| [F, Tₙ] + [H̃, Tₙ₋₁] \|0⟩ | one rank below the parent |
| CCSDT-1a, -1b, -2, -3 | CCSDT | triples, and T₃'s feedback into singles/doubles | by variant | o³v⁴ |

Here H̃ = e^{−T₁} H e^{T₁} is the T₁-dressed Hamiltonian, and F is the Fock
operator.

- **CC2, CC3:** T₁ counts as zeroth order. That is what separates them from
  the older CCSDT-1 family, where T₁ is first order.
- **CC3 against CCSDT-1:** CC3 keeps T₁ to all orders in the T₃ equation;
  CCSDT-1a drops those terms.
- **Value:** CC2 and CC3 are built for response theory. Their main use is
  excitation energies and properties (LR-CC2, LR-CC3, EOM-CC3), where they
  form a hierarchy with consistent accuracy and CCSD(T) has no response
  formulation. For ground-state energies, CC3 is close to CCSD(T) at
  per-iteration (T)-like cost, and CC2 is close to MP2.

References (transcribe term lists from them, do not rebuild from memory):
- **CC2:** Christiansen, Koch and Jørgensen, CPL 243, 409 (1995).
- **CC3:** Koch, Christiansen, Jørgensen, Sánchez de Merás and Helgaker,
  JCP 106, 1808 (1997).
- **CCn, CC(n)-1a/1b:** Kállay and Gauss, JCP 123, 214105 (2005), the
  paper `docs/CCSDT_Q_SCOPE.md` already cites.
- **CCSDT-1/-n:** Lee and Bartlett, JCP 81, 5906 (1984); Urban, Noga, Cole
  and Bartlett, JCP 83, 4041 (1985).

## Why this matters only conditionally

Planck has no EOM-CC or linear-response CC. Without them, CC2 and CC3 add
ground-state energies that MP2 and CCSD(T) already cover at lower or equal
cost. **Build the models when an excited-state CC scope is opened, or when a
user asks for a ground-state CC3 number.** Step 0 is the exception: it
fixes a naming defect that exists today.

## Step 0. The keyword collision (needed regardless)

Every `ccN` and `uccN` keyword means "generated CC truncated at rank N"
(`src/io/io.cpp`, `generated_cc_keyword`, one table giving route, rank and
(T) per keyword):

| Keyword | Means in Planck | Means in the literature |
|---|---|---|
| `cc2` | CCSD | CC2 |
| `cc3` | CCSDT | CC3 |
| `cc4`, `cc5` | CCSDTQ, CCSDTQP | CC4, CC5 (Kállay and Gauss) |
| `ucc2`…`ucc4` | UCCSD…UCCSDTQ | UHF CC2…CC4 |

A user asking for CC3 today silently gets full CCSDT, at higher cost and a
different energy. Seven committed inputs use these keywords (`cc4` ×3,
`ucc2` ×2, `ucc3`, `ucc4`), and thirteen docs name them.

Recommendation:
1. **Done 2026-11-25.** Every rank 2–6 has a spelled name on both routes
   (`ccsd_gen` … `ccsdtqph_gen`, `uccsd_gen` … `uccsdtqph_gen`, plus the
   plain `ccsdtq`/`uccsdtq` … forms for ranks 4–6), and the restricted
   route's enum is `PostHF::RCCGEN` (it was `RCCSDTQ`, which also covered
   ranks 2, 3, 5 and 6), matching `PostHF::UCCGEN`. Checked by running
   water STO-3G through every restricted spelling (each reaches rank 2–6
   as listed, (T) only on `ccsd(t)`), and every UCC spelling on a UCC
   build: OH 6-31G for ranks 2–3 (`uccsd(t)` adds (T)), B STO-3G for
   ranks 4–6 (the three rank-4 spellings all give −24.1892649766, the
   `b_ucc4_sto3g` value; ranks 5–6 reach the route at their rank and are
   refused because that build stops at rank 4). Input validation now refuses
   non-single-point runs for both routes, and the CCSD-reference summary
   takes its label from the rank. Core suite 178/178.
2. Migrate the seven inputs and the docs to the spelled names.
3. Make `cc2`…`cc6` and `ucc2`…`ucc6` refuse by name, giving the spelled
   alternative, until the real models land. A keyword must not silently
   change meaning between versions.

**User decision:** whether to free the `ccN` names for the literature
models (recommended), or keep them as ranks and spell the models
differently (e.g. `cc3_model`). Either way the refusal in item 3 is the
safe interim state.

## Step 1. The filter (the one new mechanism)

These are not new equations: they are subsets of terms ccgen already
generates. The mechanism is a per-term order filter applied to the
generated residual. It is a sibling of `_drops_under_canonical_fock`
(`python/ccgen/generate.py`), which already drops terms one at a time.

Order of a term: the sum of its factors' orders, with
- f (Fock): 0;
- v (two-electron): 1;
- t₁: 0 for CC2, CC3 and CCn, 1 for the CCSDT-1 family;
- t_k for k ≥ 2: k − 1.

For CCn, the rank-n residual keeps only the terms of order ≤ n − 1, and the
lower-rank residuals are kept whole. This rule must be checked against the
papers' term lists before use. It reproduces both hand cases:

| Equation | Term | Order | CC2/CC3 keeps? |
|---|---|---|---|
| CC2 doubles | f·T₂ | 1 | yes |
| CC2 doubles | v·T₁ᵐ | 1 | yes |
| CC2 doubles | v·T₂ | 2 | no |
| CC3 triples | f·T₃ | 2 | yes |
| CC3 triples | v·T₂·T₁ᵐ | 2 | yes |
| CC3 triples | v·T₃ | 3 | no |
| CC3 triples | v·T₂² | 3 | no |

The CCSDT-1/-n variants also restrict which T₃ terms enter the singles and
doubles equations. Encode each one as a per-target maximum order taken from
the papers, not as a separate code path.

**Where the filter runs.** On the raw spin-orbital terms, before spin
adaptation (`spin_adapt_equations`, `ucc_adapt_equations`), and before
dressing or factorization. Perturbation order does not depend on spin, so
one filter serves RCC and UCC. The factorizer then sees only the kept
terms. Applying it after dressing would hit derived operators whose order
is mixed.

**Fock off-diagonal terms.** With canonical references f_ov = 0, so the
[F, Tₙ] term is the orbital-energy denominator. Non-canonical references
(ROHF, `docs/ROHF_CORRELATION_SCOPE.md`) would need f_oo/f_vv off-diagonal
terms; out of scope here.

## Step 2. Solve: stored first, on the fly later

**First version: reuse the arbitrary-order harness unchanged.** The highest
rank is stored as ordinary amplitudes (o³v³ for CC3) and updated by the same
Jacobi/DIIS step as rank-N CC. Its truncated residual is linear in Tₙ with
only the diagonal Fock term, so the Jacobi step solves it exactly each
iteration. That is correct, but carries CCSDT's storage.

**Later, for size:** never store Tₙ. Each iteration, build it per virtual
triple as (H̃ T₂)/D and contract it straight into the singles and doubles
residuals, the way the batched (T) builds W per virtual triple
(`docs/CCSD_T_SIZE_SCOPE.md`, steps 8 and 1a). This is what makes CC3
o²v² in memory, and it is a driver change. Do it only after the stored
version is validated, using it as the oracle.

## Gates

Each one has to fail when the code is broken:

- **CC2 with T₁ frozen at zero equals MP2** (energy and amplitudes, to
  1e-12). The doubles equation then reduces to f·T₂ + v = 0. A filter that
  keeps v·T₂ breaks this.
- **CC3's first T₃ from converged CCSD amplitudes with T₁ = 0 equals −W/D**
  from the existing (T) kernel, to roundoff. That ties the CC3 triples
  equation to the already PySCF-gated (T) W.
- **Order filter unit test:** on the spin-orbital terms, the kept/dropped
  sets match the papers' term lists for CC2 doubles and CC3 triples, term by
  term. Mutation: shift t₁'s order to 1 and the CC3 set must change.
- **Stored against on-the-fly** (step 2, later): equal to roundoff, and
  bitwise across thread counts.
- **External reference:** PySCF has no CC2 or CC3. Use Psi4 (RHF/UHF/ROHF
  CC2 and CC3 energies), Dalton or CFOUR, on water STO-3G and 6-31G with
  Planck's GBS files, to every printed digit. Tolerance as for the (T)
  gates.

## Recommended order

1. Step 0 now: it is a correctness issue in naming, and independent of
   everything else.
2. Steps 1–2 for CC3 (restricted), then CC2, when an excited-state CC
   scope opens or a user asks.
3. UHF versions come from the same filter through `ucc_adapt_equations`.
4. CCn (n ≥ 4) and CCSDT-n only on request.

## Not in this scope

EOM-CC and LR-CC (the reason these models exist, and their own scope),
RI-CC2, analytic gradients, ROHF references, and the non-iterative
corrections (`docs/CCSDT_Q_SCOPE.md`).
