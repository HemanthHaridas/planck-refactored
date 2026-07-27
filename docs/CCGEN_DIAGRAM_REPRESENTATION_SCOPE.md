# ccgen diagram representation — ground truth + next steps

Replacing ccgen's Wick **term** enumeration with a **diagram** (Kállay–Surján
integer-string) front end. Two payoffs: kill the term-explosion waste (78× at
CCSDT), and make dressed-intermediate recognition topological instead of the
combinatorial exact-cover search that failed on the term algebra.

**Companion:** the ccgen dressed-intermediates section of `vault/Status/Open
Work.md` (why the term-algebra path stalled). **Last rewrite:** 2026-07-27 —
condensed from a 1387-line probe log; the ruled-out attempts are preserved in
§Appendix so their negative results aren't re-run.

---

## 1. Ground truth (what is true and landed now)

Everything below is in `python/ccgen/diagram.py`, gated in
`python/ccgen/tests/test_diagram.py` (fast, no PySCF) and
`python/ccgen/tests/test_reference_vs_pyscf.py` (the PySCF venv at repo-root
`tests/pyscf/.venv`). **Oracle rule (load-bearing): no diagram gate is pinned to
`generate_cc_equations` — the term-path generator is the code under suspicion
and is never its own oracle. Gates pin to PySCF or to pure topology.**

### 1.1 Enumeration — DONE

`enumerate_diagrams(ranks, bra_level)` returns the distinct connected diagrams,
canonical-by-construction (no term explosion, no post-hoc dedup). Reproduces the
term path's diagram set exactly and scales where the term path does not:

| method / manifold | diagrams | | method / manifold | diagrams |
|---|---|---|---|---|
| ccd / doubles | 9 | | ccsdt / singles | 14 |
| ccsd / singles | 13 | | ccsdt / doubles | 36 |
| ccsd / doubles | 30 | | ccsdt / triples | 47 |
| ccsd energy | 3 | | ccsdtq / quadruples | 74 |

Enumeration cost: ccsdtq in **0.2 s** vs the term path's **546 s / 2732 quad
terms**. The closure filter (`closes_internally`, `matches_manifold` with the
per-species internal cap, `admissible_hamiltonian_ranks`) was fitted to CCD
doubles and reproduced CCSD **and CCSDT untouched** (out-of-sample, the
evidence — pinned by `test_ccsdt_was_not_fitted_against`).

### 1.2 Assembly — DONE, PySCF-validated

`diagram_representative(ds, h_rank)` builds one unit-coefficient `AlgebraTerm`
(the contraction) for any diagram — all shapes (ladder, single-op mixed,
multi-op mixed), guard fully lifted. The oracle is a **full-rank least-squares
solve against the PySCF doubles residual** (`solve_diagram_weights_vs_pyscf`):
on LiH/STO-3G the 31-diagram basis is rank 31, so the weight solve is **exact
and unique** (span residual ~9e-15). The committed weight table
`ccsd_diagram_weights.json` is that solve's output; its diagram-id set comes from
`enumerate_diagrams` (PySCF-free), so the table is **PySCF-provenance-only**.

### 1.3 The diagram weight — DONE, structural, solve-free (AR2)

`diagram_signed_weight(ds, h_rank)` = `structural_sign · diagram_magnitude`,
derived from topology alone — **no PySCF solve, no stored table, no generator** —
and reproduces all 30 PySCF-solved CCSD-doubles signed weights.

**Magnitude** `diagram_magnitude` (AR2.2):
```
|w| = equivalent_vertex_factor / 2^(equivalent_line_pairs + external_pairs)
```
- `equivalent_line_pairs` (AR2.2a): summed lines sharing both endpoints + species.
- `equivalent_vertex_factor` (AR2.2b): `∏ 1/n_v!` over identical operators with
  the same internal connection signature (not a naive `1/n!` over all same-rank).
- `external_pair_factor` (AR2.2c): `2^p`, `p` = same-species **external**-line
  pairs, counted per amplitude AND on the Hamiltonian vertex (the
  bare-antisymmetric storage convention).

**Sign** `structural_sign` (AR2.1 + AR2.3(i)):
```
sign = crossing_parity · (-1)^l · (-1 if the Fock line contracts a hole)
```
- `(-1)^l` from `directed_loops` (Crawford oriented-loop count, AR2.1).
- `crossing_parity` (AR2.3(i).1b.1): +1 if the open directed loops pair the
  doubles externals identity `(i↔a, j↔b)`, −1 if crossed. Alone: 23/30.
- The `(-1)^l` factor lifts the ERI-vertex diagrams to **26/26** (AR2.3(i).1b.2a).
- The Fock factor is genuinely **species-dependent** (−1 iff the one-body line is
  a hole): `-1 if fock-occ` = 4/4, `-1 always` = 3/4, `-1 if fock-vir` = 0/4
  (AR2.3(i).1b.2b). Total: **30/30**.

`sign_correction` (the .1a stored ±1) is retained only as the diagnostic
`structural_sign` is regression-checked against.

**Why "solve-free" matters:** the weight now generalizes past any rank with a
PySCF solve to build a table against — the precondition for true arbitrary-rank
generation.

### 1.4 The seam (what a full diagram front-end replaces)

Everything downstream consumes `dict[str, list[AlgebraTerm]]`. If the diagram
layer *produces* `AlgebraTerm`s, the back half is untouched.

| layer | modules | fate |
|---|---|---|
| enumeration | `hamiltonian`, `cluster`, `wick`, `project`, `algebra` | **replaced** by `diagram.py` |
| dedup / canonicalize | `canonicalize.py` | mostly retired (canonical by construction) |
| optimization | `optimization/*` | retained, re-pointed at diagram terms |
| lowering / IR / emit | `lowering/*`, `tensor_ir`, `emit/*` (4 emitters) | **unchanged** |

The 5 enumeration producers total ~2600 lines; the consume/transform half
(~7000 lines) is enumeration-agnostic and stays regardless.

---

## 2. Next steps

### 2.1 D4 is NOT blocked by a generator bug — corrected 2026-07-27

**Prior belief (overturned):** that ccgen's CCSD doubles residual was wrong on
the `t1·t2·v` / `t1·t1·t2·v` / `f·t1·t2` terms, blocking D4.

**Measured ground truth:** the **full ccgen doubles residual matches PySCF
`gccsd.update_amps` to 3.9e-16** (LiH/STO-3G, scale=0.05, non-canonical-fock,
all 200+ terms; evaluated by `residual_of` with the `test_reference_vs_pyscf`
integrals). ccgen is **correct**. This is consistent with the already-passing
`test_diagram_basis_spans_the_pyscf_doubles_residual` (the diagram basis, which
groups the same ccgen terms, spans the PySCF residual to ~1e-15) and with §1.3
(the buggy-class diagrams each match the PySCF-solved `diagram_signed_weight` to
1e-15).

**What the `test_gccsd_gate.py` "~3% / maxdiff 1.1" actually is — a gate
artifact, not a ccgen defect:** the gate compares ccgen against the
hand-transcribed Stanton–Gauss dressed reference (`gccsd_reference.py`) on
`_shared_inputs` = **random off-shell `t1`/`t2` + a random-diagonal Fock**. On
*real* CC-consistent amplitudes both agree with PySCF (reference vs PySCF =
4e-14 at tiny amps, 9e-6 at scale=0.05; ccgen vs PySCF = 4e-16). The raw
projection residual `⟨Φ|H̄|0⟩` and the dressed (τ/W-intermediate) reference are
algebraically distinct expressions that coincide **on-shell and in their
antisymmetric projection**, but need not agree term-by-term on arbitrary random
tensors — which is exactly the off-shell regime `_shared_inputs` probes.

**Consequence:** the `@expectedFailure` gates in `test_gccsd_gate.py` pin an
artifact of their own off-shell random-input construction, not a real bug. The
canonical-Fock work and T1.2b (the `is_dummy` false-zero fix) were genuine
correctness improvements and stand; there is **no T1.2b-2 merge-key bug on the
critical path**. D4 (§2.3) is gated only on the diagram-emit wiring itself.

**Open follow-up (small, not a blocker):** re-point the GCCSD gate at PySCF
directly (compare ccgen and `gccsd_reference` to `gccsd.update_amps` on the same
inputs) so it stops pinning the off-shell dressed-form gap; the `@expectedFailure`
end-to-end check should become a PASS against PySCF. Verify the ccgen==PySCF
4e-16 match under `canonical_fock=True` as well (the raw-residual match already
holds; canonical-fock only drops the identically-zero `f_ov` terms).

### 2.2 AR3 — extend the weight to CCSDT / CCSDTQ

**How to validate arbitrary-rank CC (the strategy shift, 2026-07-27).** There is
**no per-term CCSDT residual oracle** — PySCF ships no spin-orbital `gccsdt`
(only spin-adapted, T1-dressed `rccsdt`/`uccsdt` and the perturbative `(T)`
`gccsd_t`, energy only). Trying to build one (the old "B0") is L-effort and
error-prone. Production codes (MRCC/CFOUR) don't have a per-term oracle either;
they validate arbitrary order by **reduction + limit + converged energy**. AR3
adopts the same, all reachable without bridging PySCF's dressed `rccsdt`:

- **AR3.1 — cross-rank reduction (LANDED, ~S, no oracle build).**
  ccgen's `generate_cc_equations("ccsdt")["singles"/"doubles"]` with the
  T3-containing terms dropped canonicalizes to **exactly** the CCSD
  singles/doubles residual (verified: doubles 70 terms, identical coefficient
  multiset). Since CCSD is PySCF-validated to 4e-16 (§2.1), this chains that
  validation up to CCSDT for free. Gated by
  `test_regressions.py::test_ccsdt_reduces_to_ccsd_when_t3_dropped`.
- **AR3.3 — converged-energy check (LANDED, the decisive integration test).**
  `ccgen_energy_at_pyscf_amps` plugs PySCF's own converged CCSD amplitudes into
  the GENERATED equations; the generated E_corr equals PySCF's to ~1e-15 (H₂/
  STO-3G). Evaluating at PySCF's amplitudes (not iterating our own Jacobi)
  isolates the equations from solver-convergence and amplitude-layout confounds
  (a plain-Jacobi self-iterate settles ~1.6e-7 off due to a residual_of doubles
  layout wrinkle that the fully-contracted energy is robust to). Gated by
  `test_reference_vs_pyscf.py::test_ccgen_ccsd_energy_matches_pyscf`. The harness
  extends to CCSDT once its energy manifold is iterated — it is the gate the
  magnitude extension is validated against (no per-diagram CCSDT weight oracle
  exists; PySCF ships no spin-orbital gccsdt).
- **AR3.2 — FCI limit (~M, remaining).** Iterate the generated CCSDT residual to
- **AR3.2 — FCI limit (LANDED). The decisive triples-correctness gate.** For an
  N-electron system CCSDT = FCI exactly, so on a 3-electron doublet the generated
  CCSDT (singles+doubles+**triples**) must recover the exact FCI total energy.
  `ccgen_iterate_amps` (AR3.2.0, Jacobi solver, rank-general denominators
  `_amp_denominators`, `residual_einsum` for speed) solves the generated residual;
  `fci_total_energy` gives the reference. **Measured on H₃/6-31g doublet (nvir=9,
  so T3 is non-trivial — CCSD alone misses FCI by 1.4e-4): GHF+CCSDT reaches FCI
  to 5.7e-13 in ~11s.** Gated by `test_ccgen_ccsdt_reaches_fci_limit` (+ AR3.2.0's
  `test_ccgen_ccsd_solver_matches_pyscf`). The earlier runtime worry is resolved:
  `residual_of` (per-index-tuple Python loop) is unusable at triples (>120s/eval);
  `residual_einsum` (per-term np.einsum) does the full 417-term triples residual in
  0.04s — the whole solve in ~11s. Note the "AR3.3 layout wrinkle" was a red
  herring (just PySCF's loose default CC conv_tol); the plain Jacobi iterate
  converges correctly.

Together AR3.1 (reduction, LANDED) + AR3.2 (FCI limit, LANDED) + AR3.3 (energy,
LANDED) validate the generated CCSDT equations to the standard production codes
actually meet — **without a per-term CCSDT oracle**. This replaced the old B0
residual-tensor build. Note: these validate the *equations as generated by the
term path*; when the diagram-path weights are wired into generation (D4), the
same three gates re-validate the diagram path.

**The weight extension itself — M1, LANDED and FCI-validated at triples.** The
diagram weight now reproduces the ccgen residual per-diagram across ccsd/ccsdt
singles+doubles+triples (140 diagrams, ~1e-13), and a CCSDT residual built
**entirely from `diagram_signed_weight`** reaches the FCI energy (H₃/6-31g,
`test_diagram_weighted_ccsdt_reaches_fci_limit`). Two pieces:
- **Magnitude (M1.0/M1.1):** the amplitude normalization is `∏_amp ∏_species
  (1/n_ext!)` (`_amplitude_norm_factor`), replacing the old floor-div pair count
  which saturated at n=2. Identical on doubles (`(1/2)^(k//2) == 1/k!` for
  k≤2 — verified 30/30), non-dyadic at T3 (`1/3!` appears; 18/47 triples
  diagrams). The vertex part stays a pair count (`_vertex_pair_factor`).
- **Sign (M1.2):** a per-**manifold** factor `(-1)^bra_level` was found missing —
  invisible on doubles (`(-1)^2=+1`) but the whole singles (bra=1) and triples
  (bra=3) manifolds were sign-flipped. Now folded into `diagram_signed_weight`.
  Discovered exactly because M1.3 validated the diagram-built residual against
  ccgen manifold-by-manifold rather than only on doubles.
- **AR1 (LANDED):** all 74 ccsdtq diagrams assemble, orbit residuals
  antisymmetric, rank 74/74. Well-formedness; correctness rides AR3.1/AR3.2 up.

So the **solve-free diagram weight (sign + magnitude) is validated through
CCSDT** — AR3 is essentially complete bar CCSDTQ (rides AR1 + the CCSDT formula).

**B1 (LANDED):** `crossing_parity` generalized to `external_pairing_parity` (sign
of the occ→vir external permutation); identical to `crossing_parity` on all 30
doubles diagrams, defined for triples/higher. `structural_sign` uses it. So the
sign machinery is rank-ready; only the magnitude's `(1/n!)²` factor and the
AR3.1–3.3 validation remain.

### 2.3 D4 / AR4 — wire the weighted diagram into generation

Once §2.1 clears: emit `diagram_signed_weight · orbit(rep)` as `AlgebraTerm`s
through the unmodified emitter, behind a flag
(`generate_cc_equations(..., engine="wick"|"diagram")`, default `wick`).
*Gate:* default output byte-identical; diagram path reproduces the (bug-fixed)
term path after canonicalization. AR4 is the same pipeline generalized to any
`(ranks, manifold)`.

### 2.4 D5 — retire the term-path enumeration

Once D4 holds, `wick.py` + `project.py` + the BCH path in `algebra.py` are dead
for generation. Pure deletion behind the passing D4 gate (~2600 lines).

### 2.5 Later, separate decisions (not on the critical path)

- **D6 — string-driven contraction.** MRCC's runtime half (drive contractions
  over excitation strings, never materialize high-rank equations). Changes the
  runtime, not codegen. Only if generation time stops being the constraint.
- **D7 — dressing on diagrams.** With canonical topologies, `Wmnij`/`Wabef` are
  recognizable subgraphs. But optimal factorization is **NP-hard** — production
  codes use staged heuristics — so D7 buys tractable *recognition*, not an
  optimal factorizer. Curated templates (Open Work, Option B) stay the pragmatic
  route for shipping dressed CCSD/CCSDT.

---

## 3. Rewrite-from-scratch verdict — DO NOT

Rewriting discards all the verified, correct work (§1.1–§1.3 enumeration,
assembly, the structural weight rule) and re-derives the same encoding against
the same oracles — pure re-work. The one genuinely hard piece (the structural
weight rule) is a property of the physics, not the code; a rewrite inherits it
unchanged. The only legitimate rewrite-adjacent action is **D5** (targeted
deletion behind a passing gate). Rewrite rejected.

---

## Appendix — ruled-out approaches (don't re-run these)

Negative results, each measured. Kept so the dead ends aren't re-explored.

- **Route 2 (uniform-orbit invariance).** The idea that a diagram could emit one
  representative × weight and lean on a per-diagram *sum* gate. Dead: ccgen's
  actual doubles residual is the **direct projection `⟨Φ|H̄|0⟩`**, NOT a
  materialized antisymmetric `P(ij)P(ab)` orbit. A diagram maps to a **set of
  distinct arrangements**, each with its own coefficient (13/15 term pairs in
  `t1·t2·v` are non-proportional). The earlier "Probe A invariance holds" was
  measured against a fabricated antisymmetric target and is retracted.

- **D4-via-table.** Emitting the committed weight table's per-diagram weight ×
  orbit as `AlgebraTerm`s. Dead: hits the same ragged-split/convention wall
  (orbit expansion gives maxdiff 15.5–17.3 vs the term path; per-diagram rescale
  is a no-op because the `t1·t1·t2·v` over-count is *sub-diagram*). The table is
  a correct *tensor* oracle but not a drop-in term weight.

- **Sign as a scalar count.** The AR2.3 convention delta (11/30) is NOT any
  loop/hole count: `(-1)^h`, `(-1)^l`, `(-1)^(h+l)`, `(-1)^open`, `(-1)^closed`,
  free-index-inversion parity all score ≤ 21/30. It is also NOT fixable by
  external relabel (h and l are invariant under a↔b / i↔j). The resolution was
  the *crossing parity of the open loops' external endpoints* + the Fock species
  factor (§1.3), not a count.

- **t1·t2 merge fixes (attempts 1–3).** Post-projection rename (corrupts correct
  terms — coupled to the merge bucketing), `>1 collision` exclusion (local-only,
  breaks other term types), rename-before-canonicalize (splits correct buckets).
  All fail the whole-residual GCCSD gate. The fix must co-design the merge key
  and projection-relabel; not a post-hoc pass. (This is §2.1's open work.)

- **Embedded-τ residue pairing** and the **naive global-sum operator check** (the
  dressed-intermediate recognition path) — both dead ends recorded in Open Work;
  the exact-cover model reaches only 20/70 residual fragments.

- **ccgen parallel generation** (`parallel_workers>1`) is not equivalence-safe
  (spawn-unsafe `_wickaccel`, partition-local raw merge). Serial is
  deterministic + correct; parallel stays opt-in. Unrelated to the diagram work.

---

## References

**Enumeration (topology generation — §1.1):**
- Kállay & Surján, *Computing coupled-cluster wave functions with arbitrary
  excitations*, JCP **113**, 1359 (2000).
- Kállay & Surján, *Higher excitations in coupled-cluster theory*, JCP **115**,
  2945 (2001) — the string representation is attributed here.
- *Generating coupled cluster code for modern distributed memory tensor
  software*, arXiv:2409.06759 — diagram strings (reproduces Kállay–Surján
  explicitly), and the NP-hardness / heuristic-staging discussion for D7.
- *Overview of Developments in the MRCC Program System*, PMC11874011.

**Diagram weight/sign rule (§1.3) — NOT in the enumeration papers:**
- **Crawford & Schaefer III, "An Introduction to Coupled Cluster Theory for
  Computational Chemists", Rev. Comput. Chem. 14, 33–136 (2000)** — the source
  used: `sign = (-1)^(h+l)`, oriented loops (incl. open/residual diagrams),
  equivalent-line/vertex magnitude factors, worked examples. `directed_loops`
  was validated against its p.84/87/91 values.
- Shavitt & Bartlett, *Many-Body Methods in Chemistry and Physics* (Cambridge,
  2009) — formal `(-1)^(h+l)`.
- Kucharski & Bartlett, Adv. Quantum Chem. 18, 281 (1986) — loop-counting for
  arbitrary diagrams.
