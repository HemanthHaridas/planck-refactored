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
  antisymmetric, rank 74/74. Well-formedness.

**CCSDTQ is now VALIDATED, not just well-formed (2026-07-27).** The T4 `1/4!`
amplitude factor generalizes from M1's `∏(1/n_ext!)` with no rank-4-specific
work, and the whole diagram-engine CCSDTQ residual solved to convergence reaches
**FCI to 1.12e-12** (H4/STO-3G, 4 electrons, where CCSDTQ = FCI). So the
solve-free diagram weight (sign + magnitude) is validated **end-to-end through
CCSDTQ**, and the diagram engine emits the full CCSDTQ kernel (einsum 3289 lines,
C++ 57925 lines, all 5 manifolds) — where the wick engine takes **615 s** to
generate CCSDTQ, the diagram engine takes **3.0 s** (~205×). The generated
kernel is not yet compiled into any binary (§5).

**B1 (LANDED):** `crossing_parity` generalized to `external_pairing_parity` (sign
of the occ→vir external permutation); identical to `crossing_parity` on all 30
doubles diagrams, defined for triples/higher. `structural_sign` uses it. So the
sign machinery is rank-ready; only the magnitude's `(1/n!)²` factor and the
AR3.1–3.3 validation remain.

### 2.3 D4 — wire the weighted diagram into generation. LANDED.

`generate_cc_equations(method, engine="wick"|"diagram")`, default `"wick"`
(byte-identical, guarded by `test_default_engine_is_wick`). The `"diagram"`
engine (`_generate_diagram_equations`) builds each manifold from the solve-free
diagram weights — `diagram_manifold_terms` = the signed `AlgebraTerm` orbit of
every enumerated diagram (`diagram_orbit_terms`, D4.0) + the bare Hamiltonian
term (`_bare_manifold_term`: `f(a,i)` singles, `⟨ij||ab⟩` doubles, none higher)
— then runs the SAME `canonicalize_term_to_fixed_point` + `merge_term_into_buckets`
finalization as the wick path. No BCH/Wick.

**The gate is RESIDUAL equality, not canonical-multiset equality** — the crucial
finding that vindicates the Route-2 decision. The two engines' term multisets
differ, but **only** by how repeated-factor terms are split (measured: 0
non-repeated-factor differences across every ccsd/ccsdt manifold — the
`t1·t1·v` / `t1·t1·t2·v` exchange pairs the wick path keeps as two `±½` terms,
the diagram path merges to one). Both lower/emit to the same runtime
accumulation, so the tensors are identical: the diagram engine's per-manifold
residual equals the wick engine's to ≤1e-13 across ccsd + ccsdt (`residual_einsum`,
`test_diagram_engine_matches_wick_residual`). The diagram path even emits *fewer*
terms (ccsdt triples 417→414). End-to-end: `engine="diagram"` CCSDT solved to the
FCI energy on H₃/6-31g (`test_diagram_engine_ccsdt_reaches_fci_limit`, D4.3).

Ladder: D4.0 orbit-terms == array orbit (reuses M1.3) → D4.1 manifold == full
ccgen residual → D4.2 engine flag, residual-equal to wick → D4.3 FCI energy.
AR4 (arbitrary `(ranks, manifold)`) is the same `_generate_diagram_equations`
generalized — already rank-parameterized; CCSDTQ rides AR1 + the same weights.

### 2.4 D5 — retire the term-path enumeration. SCOPE CORRECTED; kernel-equivalence LANDED.

**D5 is NOT pure deletion** — the earlier "~2600-line deletion behind the D4
gate" was wrong. `project.py` holds the shared types (`AlgebraTerm`,
`MANIFOLD_NAMES`/`_NAME_TO_RANK`/`manifold_name`) that the diagram engine and all
downstream (`canonicalize`, `tensor_ir`, the diagram engine itself) consume, so
it stays. Only `wick.py`, the BCH path in `algebra.py`, and the
projection-of-Hbar helpers are genuinely term-path-only — and they can only be
deleted **after flipping the default to `engine="diagram"`**, which changes what
the real codegen consumers (`cli.py`, `bench.py`,
`generate_planck_cc_kernels.py`, `generate_cc_equations_lowered`) emit. So D5 is
gated on a production-codegen default flip, not a mechanical deletion.

**Kernel-equivalence prerequisite — LANDED (the safe-to-flip evidence).**
`engine` already threads through `generate_cc_equations_lowered` /
`print_einsum` (via `**kwargs`). Emitting both engines as numpy einsum, exec'ing,
and comparing arrays surfaced a real convention gap: the diagram path emitted the
residual in `[vir, occ]` order vs the term path's `[occ, vir]` — same tensor
(energy bit-identical, residual ~1e-14) but a **layout** downstream solvers
depend on. Fixed by ordering the diagram terms' `free_indices` occ-first
(`diagram_orbit_terms` / `_bare_manifold_term`). Now the einsum-emitted kernels
match in value AND layout (`test_diagram_engine_emits_equivalent_kernels`:
R1/R2 same shape, ≤1e-11). `residual_einsum` normalizes to `[vir,occ]`
internally, so `ccgen_iterate_amps` / the FCI gates are unaffected by the
reorder (verified).

**Still deferred (deliberately):** the default flip + deletion. The diagram
engine stays opt-in; the term path remains the default and fallback. The flip is
a separate, higher-risk decision now backed by kernel-equivalence evidence.

### 2.5 Later, separate decisions (not on the critical path)

- **D6 — string-driven contraction.** MRCC's runtime half (drive contractions
  over excitation strings, never materialize high-rank equations). Changes the
  runtime, not codegen. Only if generation time stops being the constraint.

#### D7 — dressing on diagrams (scoped 2026-07-27)

**What D7 is — and is NOT.** D7 is **not** about generation speed: the diagram
enumeration already banks that (CCSDTQ 3.0 s vs wick 615 s). D7 is about the
**FLOP scaling of the generated kernel** — factoring the residual into dressed
intermediates (`Wmnij·τ`, `Wabef·τ`, `Fae·t2`, …) so contraction cost drops from
`O(n⁶)` to `O(n⁵)` in the places dressing covers. It is the same reason
PySCF/CFOUR ship dressed CC.

**Why it is tractable now (the exact-cover retirement, made concrete).** The
term-algebra route tried to *recognize* dressed operators by index-binding +
exact cover over the flat post-Wick term list (`optimization/dressing.py`'s
A2/A3, the embedded-τ firewall in `optimization/tau.py`). Dead end — 20/70
fragments; two `@expectedFailure` tests in `test_tau.py` pin it, now re-marked
OBSOLETE. **The diagram representation makes each dressed operator an
identifiable subgraph of `diagram_representative`'s assembled contraction**, so
recognition is a topological match, not a combinatorial search. That supersession
is the whole reason exact-cover was retired: D7 replaces it. (Dead-code deletion
of the A2/A3 stack is a separate deferred decision — doc-only retirement now.)

**Carries over — already built, no rework:**
- `dressing.py::seeded_operators()` — the six Stanton–Gauss operator definitions
  (`Fae/Fmi/Fme`, `Wmnij/Wabef/Wmbej`), each itself a small diagram/term sum.
- `dressed_equation.py` — the **rank-agnostic verifier** (`expand_dressed_term`,
  `verify_dressed_equation`): expand every operator + pseudo-amplitude and check
  the dressed equation equals the raw residual exactly. Works at any rank.
- `diagram_representative` / `build_line_graph` — the assembled diagram + its
  line graph, the substrate a subgraph match runs on.

**What D7 must add (recognition — the new piece):**
- **D7.1 — operator-diagram encoding (~M).** Express each seeded operator as a
  diagram fragment (partial line-graph / index-wiring pattern) rather than a term
  list. Reuses the definitions; the work is casting them in the diagram encoding.
  Scoped into small verifiable steps:
  - **D7.1.0 — fragment line-graph data model. LANDED.** `FragmentLineGraph` +
    `OperatorFragments` in `optimization/dressing.py`. A `FragmentLineGraph` is an
    OPEN line graph: the same `(species, endpoint_a, endpoint_b)` edge format as
    `diagram.LineGraph` (so D7.2 matches one homogeneous representation), but its
    block indices are dangling `("port", slot)` endpoints instead of `"bra"`.
    Endpoints are `("factor", k)` (a definition-term factor: v/t1/tau/f) or
    `("port", s)` (a block index wiring outward). A line between two factor nodes
    is internal (a summed index); a line touching a port is dangling (a block
    index). occ→hole "h", vir→particle "p" (same species convention as the
    engine). Read-backs: `internal_lines` / `dangling_lines` / `port_species`.
    `OperatorFragments` is the D7.1.3 container (one `(coeff, FragmentLineGraph)`
    per defining term + name/block/uses). *Gate*
    (`FragmentLineGraphModelTests`, PySCF-free): a hand-built Wmnij `¼ τ v` term
    fragment splits into 2 internal particle lines + 4 dangling hole ports, ports
    read back `{0:h,1:h,2:h,3:h}`, and the line format constructs a `LineGraph`
    unchanged (shape-compatible). Data model only; encoders are D7.1.1/1.2.
  - **D7.1.1 — single-factor fragment encoder. LANDED.** `factor_to_fragment`:
    one tensor factor → its p/h lines, each index a dangling `("port", slot)`
    (block index) or a `("stub", name)` half-line (summed index) awaiting the
    D7.1.2 join; occ→hole, vir→particle. *Gate* (`FactorToFragmentTests`): the
    concrete Wmnij factors (`v`, `t1`, `τ`, interaction `v`) emit the right
    port/stub/species pattern.
  - **D7.1.2 — definition-term assembler. LANDED.** `term_to_fragment`: compose
    the single-factor fragments, fusing the two `("stub", name)` half-lines of
    each summed index into one internal factor↔factor line; raises on a dummy
    that is not exactly 2-contracted. *Gate* (`TermToFragmentTests`): the Wmnij
    `τ·v` term reproduces the D7.1.0 hand-built oracle (2 internal particle lines
    + 4 dangling hole ports), and a malformed uncontracted-dummy term is
    rejected.
  - **D7.1.3 — operator fragment set. LANDED.** `operator_fragments`: map
    `term_to_fragment` over the definition terms → `OperatorFragments`. *Gate*
    (`OperatorFragmentsTests`): all six seeded operators (incl. Wabef's all-virtual
    block and the F-operators / `tau_tilde` / `f` factors) encode, one fragment
    per term, ports matching the block's occ/vir pattern.
  - **D7.1.4 — encoding fidelity (injectivity). LANDED — and it drove a
    data-model fix.** `fragment_signature` + the injectivity gate
    (`FragmentFidelityTests`): distinct definition terms across the family must
    have distinct signatures. The gate CAUGHT that line topology alone collides
    `t2·v` with `t1·t1·v` (Wmbej — they wire identically), so `FragmentLineGraph`
    now carries `factor_names` and the signature keys on factor SPECIES, not just
    wiring. Without this a D7.2 match could mis-recognize one operator term as
    another. (Chosen over a full `fragment_to_term` inverse: injectivity across
    the seeded family is the property D7.2 actually needs, and cheaper.)
  - **D7.1 is COMPLETE.** `operator_fragments(op)` is the D7.2 input.
  - YAGNI notes: treat `tau` as an ATOMIC factor node (don't expand — that's
    D7.3); use a direct `factor→lines` encoder, NOT the whole-diagram
    Kallay-Surjan `DiagramString` machinery (which is built for closed diagrams,
    not open fragments).
- **D7.2 — subgraph recognition (~L, the core). LANDED for Wmnij end-to-end;
  family generalization deferred (D7.2.5, 4 diagnosed gaps).**
  `find_operator_occurrences(op, terms)` automatically recognizes `½ Wmnij·τ`
  in the CCSD doubles residual, matching the hand-transcribed reference — the
  machinery (encode → match → hypothesize → verify → dedup) is proven and
  operator-agnostic. Extending to the other five operators is NOT free: it hits
  4 concrete convention gaps (a real `v`-sign bug in `_eri_canonical`, antisym
  dedup, Wmbej's combined term, Fmi), scoped under D7.2.5. Find occurrences of
  each
  operator fragment as a subgraph of a diagram's assembled contraction
  (topological + species-consistent match on the line graph). Subgraph iso is
  NP-hard in general, but the graphs are tiny (≤4 operators, bounded lines), so
  this is a bounded search — the term-algebra intractability does not carry over.
  Scoped into small verifiable steps (the residual and the operator are BOTH
  AlgebraTerms, so both encode with the D7.1 machinery):
  - **D7.2.0 — residual-term fragment encoding. LANDED.**
    `residual_term_to_fragment(term)` = `term_to_fragment(term, term.free_indices)`
    — the residual's FREE indices play the operator-block role (become ports),
    its summed indices become internal lines. A summed index in a CC residual
    always appears on exactly two factors (a contraction is one edge — verified
    across the whole CCSD singles+doubles manifold), so the 2-endpoint invariant
    holds with no special-casing. *Gate* (`ResidualTermToFragmentTests`): the
    `½ t2 v` term encodes (4 ports i,j,a,b; 2 internal k,l lines), EVERY residual
    term encodes without raising, and the `v` factor's port-species match Wmnij's
    bare-`v` fragment (a D7.2.2 preview).
  - **D7.2.1 — τ-expanded operator fragments. LANDED.**
    `tau_expanded_operator_fragments(op)` = `operator_fragments(
    tau_expanded_operator(op))` — the raw residual carries NO literal `tau`
    (measured: doubles shapes are `t2·v` / `t1·t1·v`, never `tau·v`), so the
    operator patterns must be matched in raw tensors. τ expands to `t2 + t1t1`
    (fixed point), so e.g. Wmnij's 4 defining terms become 5 (the `¼ τ v` splits
    into `¼ t2·v` + `½ t1·t1·v`). *Gate* (`TauExpandedFragmentsTests`): all six
    operators expand to raw tensors (`f`/`t1`/`t2`/`v` only, no `tau`/`tau_tilde`,
    `uses` empty), Wmnij grew 4→5, and the expanded `t2·v` fragment has the same
    signature shape as the residual `t2·v` term (a D7.2.2 preview).
  - **D7.2.2 — single-fragment containment match (~L, the core). LANDED (a–d).**
    `match_fragment(op_frag, residual_term)` finds every occurrence of an operator
    fragment as an EXACT induced sub-fragment. Sub-steps:
    - **D7.2.2a — `candidate_factor_subsets`:** the factor-name prefilter — only
      residual subsets whose name multiset equals the operator's survive (a `t2·v`
      op → only `{t2,v}` residual pairs). Bounds the search.
    - **D7.2.2b — `induced_subfragment`:** the sub-fragment a subset induces —
      within-subset shared indices → internal lines, outward/external indices →
      ports, factor nodes renumbered 0..n-1 (comparable to an operator fragment).
      Makes the "extra shared line" explicit: `t2(c,d,j,l) v(c,d,k,l)` shares
      `c,d` AND `l` → 3 internal lines, one more than the operator's 2.
    - **D7.2.2c — `fragments_match` (the core):** an EXACT isomorphism — a
      species-consistent node bijection carrying the op's internal lines onto the
      induced ones as equal multisets, and ports species-matched. The extra
      `l`-line makes the internal-line multisets unequal → correctly NO match.
      This exactness is what prevents the false positives that killed the retired
      exact-cover route. Bounded backtracking (≤4 nodes → ≤24 perms, prefiltered).
    - **D7.2.2d — `match_fragment` driver:** compose a→c; each occurrence records
      `subset` + `nodes` + `port_index` (op port slot → the RESIDUAL index name it
      bound to, so D7.2.3 can check consistent block binding across an operator's
      defining terms).
    *Gates* (`CandidateSubsets` / `InducedSubfragment` / `FragmentsMatch` /
    `MatchFragment`, PySCF-free): bare-`v` matches `v(i,j,k,l)` with the index
    binding; the `t2·v` operator finds NO occurrence in `t2·t2·v` (extra-line
    rejection, the load-bearing correctness case); species mismatch blocks a
    match; the driver runs clean over the whole doubles manifold.
  - **D7.2.3 — full-operator occurrence. RE-SCOPED to hypothesize-and-verify
    (a structural finding forced it).**
    - **D7.2.3a — `collect_fragment_occurrences`. LANDED.** Fan `match_fragment`
      out over every residual term for each τ-expanded operator fragment; each
      hit carries `frag_id` / `op_coeff` / `term_id` / `term_coeff` / `subset` /
      `port_index`. *Gate* (`CollectFragmentOccurrencesTests`): Wmnij's 5
      fragments all collect occurrences (2/4/4/2/4 = 16), records carry the
      coefficients + full port binding.
    - **THE FINDING (why 3b/3c changed).** The original plan — group the fragment
      matches by a shared `(block-binding, rest-signature)` anchor and require all
      5 present — DOES NOT WORK: measured on CCSD doubles, **no single group
      covers all 5 Wmnij fragments**. Wmnij enters `R2` as `½ Wmnij(kl,ij)
      t2(ab,kl)`; expanding its definition, the outer `t2` attaches to DIFFERENT
      block slots per piece (bare `v(k,l,i,j)` carries `k,l` on the `v`; the
      `t2·v` piece splits them), so the pieces scatter across bindings/rests.
      Structural grouping cannot cleanly re-assemble one operator instance — this
      is exactly the wall the retired exact-cover route hit.
    - **The corrected approach: hypothesize-and-verify** (the same direction
      `verify_dressed_equation` / `expand_dressed_term` already work in). From one
      ANCHOR fragment match (e.g. the bare `v`) + its rest, HYPOTHESIZE a
      `W·rest` dressed term; `expand_dressed_term` regenerates all its raw pieces;
      require every expanded piece to be present in the residual with matching
      coefficient. No need to group all 5 structurally — expansion does the
      re-assembly and the coefficient check is the firewall. Reuses existing
      `expand_dressed_term`.
      - **D7.2.3b — hypothesize `W·rest` from an anchor match. LANDED.**
        `hypothesize_operator_term(op, occurrence, term)` builds the dressed
        `AlgebraTerm` `c · W(block) · rest`: `c = term.coeff / op_coeff` (the
        anchor fragment's operator-internal coefficient divided out), the `W`
        factor carries the residual indices its block bound to (`port_index`),
        `rest` = residual factors outside the anchor subset. *Gate*
        (`HypothesizeOperatorTermTests`): from the bare-`v` anchor in `½ t2 v`,
        the hypothesis is `½ Wmnij(i,j,k,l) t2(a,b,k,l)`, and `expand_dressed_term`
        regenerates all 5 raw Wmnij pieces (only `t1/t2/v/f` survive).
      - **D7.2.3c-0 — hypothesis enumeration. LANDED (a scoping finding forced
        it).** A single anchor UNDERDETERMINES the hypothesis, in two ways found
        by tracing the correct target `½ Wmnij(m,n,i,j) τ(a,b,m,n)`:
        (1) **block orientation** — a symmetric fragment (bare `v`, all-hole
        ports) admits several port bindings; `match_fragment` returned only ONE,
        and it was the wrong one (`Wmnij(i,j,k,l)`, 1/4 present) — the correct
        orientation is `Wmnij(k,l,i,j)` with `(m,n)→summed`, `(i,j)→external`.
        (2) **rest interpretation** — the true rest is a DRESSED `τ` (`Wmnij·τ`),
        not a raw `t2`; the raw residual carries `τ` only as `t2 + t1t1`.
        `enumerate_hypotheses(op, occurrence, term)` yields `W·rest` over
        {all valid port orientations} × {rest=t2, rest=τ} (`_all_port_bindings`
        gives every orientation; `_rest_variants` adds the τ variant of a single
        `t2` rest). *Gate* (`EnumerateHypothesesTests`): the enumeration CONTAINS
        the correct hypothesis (`Wmnij(k,l,i,j) τ`, expands to 10 keys ALL present
        in the residual), both orientations+rests appear, and the wrong
        `(i,j,k,l)+t2` orientation is confirmed NOT fully present (why the
        enumeration was necessary).

        **This also corrected the "canonicalization convention" framing.** The
        earlier "3 missing keys" was NOT a canonicalization gap — `_eri_canonical`
        (ERI 8-fold + free-order) is adequate: the CORRECT `Wmnij·τ` expansion's
        10 keys all match residual keys. The blocker was hypothesis
        underdetermination, not incomplete canonical form. (The 2-of-10 keys where
        the Wmnij coefficient is HALF the residual's are correct — those
        primitives get contributions from multiple operator instances; the
        whole-equation `verify_dressed_equation` at D7.3 is the exact arbiter, so
        D7.2.3c-1 only needs a SOUND containment filter.)
      - **D7.2.3c-1 — sound containment verify. LANDED.**
        `hypothesis_is_consistent(hyp, residual)`: expand `hyp` to primitives;
        every ERI-canonical key must be present in the residual with the same
        sign and `|hyp_coeff| ≤ |raw_coeff|`. A SOUND NECESSARY filter, not an
        exactness check — a primitive shared by several operator instances carries
        only PART of the residual coefficient in one hypothesis (2 of Wmnij·τ's 10
        keys are half the residual's), so equality would wrongly reject the
        correct one; the whole-equation `verify_dressed_equation` at D7.3 is the
        exact arbiter. *Gate* (`HypothesisConsistencyTests`): of the 48
        enumerated candidates only 4 survive — the antisym-equivalent correct
        orientations `Wmnij(k,l,i,j)`/`(l,k,i,j)` × {t2, τ rests} — the wrong
        `(i,j,k,l)` orientation is rejected (a primitive absent), and the filter
        is selective (<¼ survive). The τ-rest correct hypothesis is among the
        survivors; the partial t2-rest also passes (sound, not a false accept —
        D7.2.3d/D7.3 prefer the complete τ form).
      - **D7.2.3d — `find_operator_occurrences` driver. LANDED — D7.2 COMPLETE.**
        Enumerate every anchor's hypotheses (c-0), keep the consistent ones
        (c-1), and dedup to MAXIMAL primitive covers: a partial hypothesis
        (`W·t2` or `W·t1t1`, cover 5) is dropped because its cover is contained in
        the complete `W·τ` (cover 10 = t2 ∪ t1t1 covers, measured) — no arbitrary
        "prefer τ" rule, just "keep covers not strictly contained in another."
        Each occurrence is `{"term": W·rest, "cover": frozenset}`. *Gate*
        (`FindOperatorOccurrencesTests`): Wmnij is recognized as EXACTLY ONE
        occurrence `½ Wmnij(k,l,i,j) τ(a,b,k,l)` (cover 10), the partial rests are
        deduped away, and it **matches the hand-transcribed `ccsd_dressed_r2`
        reference** (same coeff `½`, same `Wmnij`+`τ` factors) — the automatic
        recognition D7 exists to provide. This validates the whole
        recognize-then-rewrite premise end-to-end for one operator.
  - **D7.2.4 — coefficient consistency.** Folded into D7.2.3c's verify step (the
    hypothesize-and-verify path checks coefficients as it expands), rather than a
    separate pass on a structural group. `verify_dressed_equation` is the exact
    backstop.
  - De-risk: gate on `Wmnij` first (cleanest; exercises τ expansion), then the
    family. The hypothesize-and-verify path is inherently sound — a false anchor
    fails the expansion/coefficient check — so over-proposing anchors is safe.

  - **D7.2.5 — family generalization. NOT "just validation" — 4 distinct gaps,
    diagnosed, deferred.** Running `find_operator_occurrences` on all six seeded
    operators: only **Wmnij** (rank-4 `oooo`) fully recognizes; **Fme** (rank-2)
    partially. The other four each hit a concrete, bounded gap:
    1. **`v` antisymmetry SIGN in `_eri_canonical` (the shared root, blocks
       Fae + Wabef). LANDED (D7.2.5.1).** `_eri_normalize_factor` reordered a
       `v` factor to its canonical arrangement but **discarded the reordering's
       parity**. So a `v` reachable only by an ODD intra-pair swap compared with
       the WRONG sign: the correct Fae/Wabef hypothesis had all keys present but
       ~4 sign-flipped, and the sound filter rejected them. Fix: `_perm_parity` +
       `_eri_normalize_factor` now returns `(factor, sign)`, and
       `_eri_normalize_term` folds the accumulated parity into the coefficient.
       The dressed `W` operators get intra-pair antisym ONLY (bra↔ket
       `<pq|rs>=<rs|pq>` is a symmetry of the integral, NOT of a dressed
       operator): both hypothesis-construction sites (`enumerate_hypotheses`,
       `hypothesize_operator_term`) now stamp `((0,1),(2,3))`, not the earlier
       ERI-style `((0,2),(1,3))`.
    2. **Antisymmetry-aware dedup (exposed by fix 1). LANDED (D7.2.5.1, same
       unit).** With the sign fold, `Wmnij` returned TWO occurrences
       (`Wmnij(k,l,i,j)τ` and the even double-swap `Wmnij(l,k,j,i)τ`) — the same
       instance written two ways. `find_operator_occurrences` now dedups on
       `_dressed_canonical_key`: `_antisym_sort_factor` sorts each antisym
       factor's indices within its groups (folding the sign) before the
       free-order-normalize + fixed-point, so antisym-equivalent orientations map
       to one key. `Wmnij` is back to exactly ONE occurrence; **Fae (0→2) and
       Wabef (0→3) now recognize.** *Gate:* `test_dressing.py` 88/88, incl. the
       `test_correct_hypothesis_passes` assertion relaxed to the block-index SETS
       (both orientations sound at the hypothesis level, deduped at occurrence
       level) and the `test_wrong_orientation_rejected` guard on `(i,j,k,l)` still
       holding.
    3. **Wmbej — LANDED (D7.2.5.3).** Recognizes as `+Wmbej·t1` (singles) and the
       `P(ij)P(ab) Wmbej·t2` quartet (doubles, 4 branches, alternating sign), via
       the asymmetric-block binding sign (S1 design below, implemented in
       `enumerate_hypotheses` + `_block_is_asymmetric` + `_binding_sign`). Gated by
       `test_wmbej_recognized`; the whole family is now guarded by
       `test_full_operator_family_recognized` (Fme 2, Fae 2, Fmi 2, Wmnij 1,
       Wabef 1, Wmbej 4 in doubles). **All six seeded operators now recognize.**
       Full derivation retained below.

       ~~DIAGNOSED, deferred~~ **The original "combined term"
       framing was WRONG** (like Fmi's turned out to be): Wmbej's definition
       already writes `−½ t2·v` and `−t1t1·v` as SEPARATE defining terms, so
       nothing bundles two amplitudes. The real cause is an **`ovvo`-block
       fragment/port-binding sign issue**. Wmbej is the ONLY seeded operator with
       an asymmetric ket — its bare `v(m,b,e,j)` is `⟨mb|ej⟩`, ket `(e,j)` =
       `(vir,occ)` — while the diagram engine writes every such integral in the
       `ovov` `(occ,vir)`-ket orientation (residual terms 9–12 are
       `±t2(a,c,i,k) v(j,c,k,b)`, an `ovov` pattern; NO residual term has Wmbej's
       `ovvo` pattern). The hand transcription `ccsd_dressed_r2` binds the
       PHYSICAL contraction `+t2(a,e,i,m) Wmbej(m,b,e,j)` and its bare `t2·v`
       expands to the correct sign (matches raw). But `find_operator_occurrences`
       enumerates block orientations via `_all_port_bindings` on the `ovvo`
       fragment and every orientation it produces binds the summed contraction
       indices to the wrong slots, so the bare `t2·v` comes out `+1` where the
       residual is `−1` — no orientation matches in any of the 81 anchors.
       Verified this is NOT an `_eri_canonical` sign bug: `⟨kb|cj⟩ = −⟨jc|kb⟩` is
       genuinely a sign difference, correctly folded; and a global definition
       negation is a self-consistency-preserving no-op for recognition (the
       hypothesis coeff `term.coeff/op_coeff` cancels it), so it does NOT fix the
       binding. The fix is in the **`ovvo` fragment port-binding**: enumerate the
       block orientation that reproduces the physical `t2(a,e,i,m)·Wmbej(m,b,e,j)`
       contraction (summed `(e,m)` into the operator, `(b,j)` external), with the
       ket-antisymmetry sign carried into the hypothesis coefficient.

       **S1 SCOPED (design validated in simulation, ready to implement).** The
       sign fix is smaller and safer than "structural port-binding rewrite":
       - Naive "multiply `coeff` by the bound bare-v canonical sign" COLLIDES:
         Wmbej 0→4 ✓ but Wmnij 6→12, Wabef 8→16 ✗ — it rescues spurious
         `−1`-bare-v orientations the current unsigned filter correctly rejects
         (verified they don't dedup away: 3 distinct dressed keys for Wmnij).
       - The distinguisher is **structural, from `space_sig` alone**: every
         ACCEPTED Wmnij/Wabef orientation has bare-v sign `+1`; only an operator
         whose block has an ASYMMETRIC (mixed-space) bra or ket pair
         (`ss[0]!=ss[1] or ss[2]!=ss[3]`) is forced to a non-`+1` genuine
         orientation. `oooo`/`vvvv` → never; `ovvo` → yes.
       - **Validated design:** apply the bare-v binding sign to `coeff` GATED on
         block asymmetry (`block_is_asymmetric(op)`), at the single site
         [dressing.py] where `coeff = term.coeff / op_coeff` is set. Full-pipeline
         simulation: Wmnij→1 (`tau`, unchanged), Wabef→1 (`tau_c`, unchanged),
         **Wmbej→4** (`t2` P(ij)P(ab) branches) — the fix, no collision.
       Remaining: S3 implement the gated sign, S4 add `test_wmbej_recognized` +
       run the full suite (invariant: the other five operators unchanged).
    4. **Fmi** — **LANDED (D7.2.5.2 Fmi).** Root cause was NOT an Fmi-specific
       sign case but an ORDERING bug in the shared `_eri_canonical`: it folded a
       `v`'s bra↔ket exchange (`_eri_normalize_factor`, which picks the
       lexicographically smallest (space,name) arrangement) BEFORE dummy
       relabeling, so Fmi's `t1(e,n)v(m,n,i,e)` piece and the residual's
       `t1(b,k)v(i,b,j,k)` — the same integral with differently-named dummies —
       normalized their `v` to DIFFERENT orientations and never folded, leaving
       a MISSING key that failed the sound filter. Fix: canonicalize (relabel
       dummies) FIRST, then fold bra↔ket, then settle. Fmi now recognizes as
       `−Fmi·t1` (singles) and the `−P(ij) Fmi·t2` pair (doubles) — the same
       legitimate P-antisymmetrizer multiplicity as Fae. Gated by
       `test_fmi_recognized`. **Bonus:** the same fix cut the whole-equation
       oracle's structural mismatches 19→7 (bra↔ket folding was inflating the
       count), pinned by the updated `test_r2_mismatch_decomposition_against_diagram`
       (now 14 = 7 τ-weight + 7 structural). Only **Wmbej (gap 3)** remains.
    Also surfaced (not in the original 4, follow-up): **Wabef assembles only
    cover-5 PIECES** (a `t2` rest + two un-completed `t1t1` rests), never the full
    cover-10 `Wabef·τ` the way `Wmnij` does — this is **D7.2.5.2**.
    - **D7.2.5.2 — Wabef τ-completion. LANDED (V0.4).** Wabef now recognizes as
      exactly ONE cover-complete occurrence `½ Wabef·tau_c` (gated by
      `test_wabef_assembles_single_tau_c_occurrence`), matching how Wmnij
      assembles `½ Wmnij·τ`. Two pieces, both validated against the RAW
      per-operator oracle (not `ccsd_dressed_r2`):
      - **W2 — `tau_c` half-weight, threaded by factor name.** A new
        `TAU_CONTRACTED_NAME = "tau_c"` (`tau.py`) expands its written t1t1 half
        at weight 1 instead of 2. `_rest_variants` emits `tau_c` (not `tau`) when
        the rest t2's bra (virtual) pair is BOTH summed AND inside the operator
        block — the Wabef case where the pair contracts antisymmetrically into the
        operator's own v, which supplies the P(t1t1) partner the doubled
        representative would double-count. The name is the carrier because a
        rest-`tau_c` and an operator-definition `tau` (weight 2) COEXIST in one
        term after operator expansion, so no local term inspection can separate
        them — the earlier heuristic attempts all mis-fired (see W1/W2 note
        below, now superseded). `tau_c` never appears in an operator DEFINITION,
        so the tau-recognition (A1) path and `ccsd_dressed_r2` are untouched.
      - **W3 — cover closed under external-pair antisymmetry.** The single
        written t1t1 representative covers Wabef's residual term 27 but not its
        i↔j antisym partner term 28, which resurfaced as a spurious standalone
        `Wabef·t1t1`. `_hypothesis_cover` now unions, for each expanded primitive,
        the keys reached by swapping the hypothesis's free (external) pairs
        ((i,j) occ, (a,b) vir) throughout — so the occurrence's cover includes
        both partners and the standalone t1t1 is strictly contained → dropped.
        Grows Wmnij's cover 10→12 (correct; the `== 10` assertion was updated).
      Wmnij stays exactly one occurrence; the full ccgen suite stays green. The W1
      diagnosis below is retained as the derivation.
    - **D7.2.5.2 (W1 diagnosis, retained).** The correct
      `½ Wabef(a,b,c,d)·τ(c,d,i,j)` hypothesis expands to
      the right 10 canonical keys but **3 are 2× OVER** (coeff `±1` vs the
      residual's `±½`), so the sound-filter over-cover guard (`|coeff| ≤ |raw|`)
      rejects the τ rest — only the cover-5 pieces survive. Root cause is
      `TAU_SPEC.written_t1t1_weight = 2` (`tau.py`): the pipeline writes the
      SINGLE ordered t1t1 representative at 2× the bare ½ P-weight, correct ONLY
      when the two P-permutations land on DISTINCT keys — i.e. when τ's amplitude
      pair is external (Wmnij's `τ(a,b,k,l)`, a,b external doubles). For Wabef the
      τ's bra pair `(c,d)` is SUMMED and antisymmetrically contracted into
      `v(c,d,a,b)`, whose OWN antisymmetry already folds the two permutations, so
      the written weight 2 double-counts against the residual's ½-per-key
      convention. **Empirically confirmed:** re-expanding Wabef's τ with the t1t1
      representative at weight **1** makes all 10 keys land `≤ raw` (0 bad); Wmnij
      still needs **2** (external τ, keys distinct). So the fix is NOT a global
      constant — the weight is context-dependent.

      **[SUPERSEDED by V0.4 above — the following records why the naive
      approaches failed, kept as derivation history. The shipped fix is the
      `tau_c` factor-name carrier + cover closure described under "LANDED
      (V0.4)".]**

      **W2 discriminator — the hard part, DIAGNOSED, not a local term-inspection.**
      The naive predicate "halve when τ's amplitude pair is summed AND both
      indices sit in an antisym `v`" is WRONG: the *identical* structure (summed
      virtual pair inside an antisym `v`) appears in **both** Wabef's rest-τ
      (`tau(c,d,i,j)` into the operator's own `v(c,d,a,b)`, needs weight 1) **and**
      Wmnij's OWN definition term (`¼ tau(e,f,i,j) v(m,n,e,f)`, needs weight 2).
      Refining to "complementary pair of the contracting `v` is free" also fails:
      it fires only on Wabef's bare-`v` sub-term and not the `t1`/`t2`-dressed
      ones, so the halving becomes non-uniform across a single operator's
      expansion pieces (Wabef went to 3, worse). The expansion is the SHARED
      `_expand_pseudo_amplitude_in_term`, used identically for a hypothesis rest-τ
      and for operator-definition τ, so any term-inspection heuristic mis-fires
      one way or the other. **The correct fix threads the halve decision from the
      rest-τ substitution site (`_rest_variants`, where "this τ replaced a t2 rest
      of a hypothesis" is known) through expansion as an explicit flag/weight,
      leaving operator-definition expansion at the canonical weight 2.** Attempted
      term-inspection heuristics broke `test_verifies_synthetic_complete_occurrence`
      (Wmnij's def-τ halved to ⅛ instead of ¼); reverted. A `NOTE` at the halving
      site records this; no behavior change (default weight preserved, 91/91).

      **W3 — cover-closure under antisymmetry, DESIGNED + PROTOTYPED (also
      reverted, depends on W2).** Even once W2 makes `½ Wabef·τ` cover-10, a
      SECOND spurious occurrence survives: the single written t1t1 representative
      covers residual term 27 (`t1(c,i)t1(d,j)`) but NOT its i↔j antisym partner
      term 28 (`−t1(c,j)t1(d,i)`), which resurfaces as a standalone `Wabef·t1t1`.
      The dressed occurrence physically replaces BOTH, so `_hypothesis_cover` must
      be closed under the residual's external-pair antisymmetry (swap the free
      (i,j) / (a,b) pairs throughout each expanded primitive, re-canonicalize,
      union the keys). Prototyped and CONFIRMED to collapse Wabef to exactly ONE
      occurrence — but it also grows Wmnij's cover 10→12 (correct, but breaks the
      `== 10` assertion pinned pre-closure), and is meaningless until W2 lands, so
      reverted with W2. Land W3 *with* W2, updating the
      `test_recognizes_single_wmnij_tau` cover assertion to the closed size.

      **W4** = gate Wabef = exactly ONE cover-complete `Wabef·τ`. Adjacent to gap
      3's combined-term handling but independent. **Net: W1 diagnosed and
      empirically pinned; W2/W3 designed and prototyped-green but reverted — W2's
      clean implementation is a flag-thread through the τ-expansion API, not the
      local heuristic first attempted, and that API change is the remaining work.**
    The `Wmnij` end-to-end result (D7.2.3d) stands as the validated proof of the
    machinery; family generalization is a deliberate multi-part follow-up — the
    sign fold + antisym dedup (gaps 1+2) landed together as **D7.2.5.1**; Wabef
    completion, Wmbej, and Fmi remain.
  - **D7.2.5.V0 — the whole-equation oracle is RED, and that reframes the Wabef
    fix (V0.1/V0.2 LANDED).** Before more per-operator recognition work,
    `verify_dressed_equation(ccsd_dressed_r2(), diagram_doubles)` must be the
    exact arbiter — and it currently **fails on 26 keys**. Classifying those (new
    tripwire `test_r2_mismatch_decomposition_against_diagram` in
    `tests/test_dressed_equation.py`) shows the Wabef τ-weight issue is the
    *minority* cause:
    - **19 keys — STALE TRANSCRIPTION (dominant).** `ccsd_dressed_r2` is an
      incomplete hand transcription: **8 dressed-only** keys (terms in the
      reference with no raw counterpart) + **11 raw-only** (raw terms the
      reference omits). No t1t1 / ratio-2 signature — just wrong or missing terms.
    - **7 keys — τ WRITTEN-WEIGHT (secondary).** t1t1 pair contracted into a
      same-space antisym `v`, coming out 2× (or ½×) — exactly the W1 diagnosis.
    This means chasing the per-operator W2/W3 weight fix against `ccsd_dressed_r2`
    was building on sand: 19 of 26 mismatches are the reference being stale, not
    the τ model. **Authority verdict:** the diagram engine is FCI-validated
    through CCSDT (see the spin-adaptation scope + memory), the hand transcription
    is not — so the diagram raw is authoritative and `ccsd_dressed_r2` must be
    re-derived (ideally auto-generated from the diagram residual), not the raw
    residual bent to fit it. Recommended order: **V0.3** re-derive/auto-generate
    `ccsd_dressed_r2` from the diagram engine → the 19 structural mismatches go to
    0; **V0.4** then land the τ-written-weight flag-thread (W2) against the now-
    trustworthy whole-equation oracle (7 keys → 0), where a mis-fire is caught by
    all 7 at once instead of Wabef's 3; **V0.5** re-run the family sweep — Wabef
    (and likely Fae) should assemble for free once the shared expansion is
    calibrated. The old W2/W3/W4 collapse into V0.4/V0.5 — right mechanism, wrong
    scope; the whole-equation oracle fixes the class, not the instance.
    - **V0.3 SCOPED (this is a diff-driven reconciliation, NOT a one-liner).**
      Investigated the 19 structural mismatches:
      - All six seeded operator DEFINITIONS are individually self-consistent
        (`operator_definition_is_consistent` True for all) — so no operator is
        broken; the defect is in how `ccsd_dressed_r2` COMBINES them.
      - The 8 dressed-only + 6-of-11 raw-only `(t1,t2,v)` keys trace to the
        `Fmi` and `Wmbej` transcription terms (rows 5/6/11/13/15/17): their
        definition `t1·v` correction pieces, times the outer `t2`, emit a `v`
        orientation/sign the diagram residual distributes differently. This is
        exactly the "sign conventions on the Fock/ERI terms and the P(pq)
        antisymmetrizer expansions" the `ccsd_dressed_r2` docstring already flags
        as unfinished.
      - The remaining raw-only keys (`t1t1t1v`, `t1t1v`, `t1t1t1t1v`) are the 7
        τ-weight-class keys (V0.4).
      - Removing the explicit Fme-correction terms makes it WORSE (26→30), so the
        transcription is not simply double-counting; each term is load-bearing.
      **Verdict:** V0.3 (whole-equation reconciliation) IS a genuine multi-term
      spin-orbital reconciliation — but the critical follow-up finding is that
      **it is NOT on the critical path for family recognition, so it should be
      DE-SCOPED from the Wabef effort.** `grep ccsd_dressed_r2` shows it is
      consumed ONLY by tests, never by production recognition; the recognition
      machinery (`find_operator_occurrences` / `hypothesis_is_consistent` /
      `_hypothesis_cover`) validates every hypothesis directly against the RAW
      residual and never references `ccsd_dressed_r2`. Its one load-bearing test
      use (`test_matches_hand_transcribed_reference`) pulls only the **Wmnij**
      term — which is one of the transcription's CORRECT terms (the 26 mismatches
      are all in Fmi / Wmbej / the τ-weight tail, not Wmnij). So V0's earlier
      premise ("the oracle must be green before recognition means anything") was
      half wrong: recognition already has its oracle — the raw residual.

      **Reframed plan.** The whole-equation `verify_dressed_equation` green-ness is
      a DOWNSTREAM D7.3-era goal (needed only when the *final emitted* dressed
      equation must be exact-verified), not a Wabef prerequisite. `ccsd_dressed_r2`
      as a hand transcription is deprecated in favor of eventual auto-generation
      (which needs D7.3 → this oracle is circular to bootstrap by hand anyway).
      **The real next step is V0.4 done against the RAW per-operator oracle:** fix
      the τ-written-weight so `find_operator_occurrences(Wabef, raw)` returns one
      cover-complete occurrence — validated by the family sweep, not by
      `ccsd_dressed_r2`. The V0.3 diagnostic detail above (Fmi surplus
      `t1(e,n)v(m,n,i,e)` piece, factor-2 τ_tilde on Fmi's `f·t1·t2` and
      `t1t1·t2·v`, the `test_r2_mismatch_decomposition_against_diagram` tripwire)
      is retained for whoever eventually does the D7.3 whole-equation verify, but
      it is explicitly NOT blocking Wabef.
- **D7.3 — factorization + emit (~M–L, rescoped after D7.3.0 investigation).**
  Rewrite the residual to reference the recognized operators, order the
  intermediate DAG (`Wmnij` needs `τ`), emit through the existing builder path.
  *Gate:* the dressed residual expanded via `verify_dressed_equation` equals the
  undressed residual exactly, AND the emitted kernel reaches the same energy.
  - **D7.3.0 — occurrence assembly / coefficient reconciliation (INVESTIGATED —
    this is the real linchpin, larger than the original "just rewrite" framing).**
    D7.2 recognition is sound (each occurrence's primitives are all present in raw
    with `|coeff| ≤ raw`), but the occurrence set is **NOT a partition**: naive
    summation of all 12 occurrences' expansions gives **24 mismatches** vs raw
    (worse than the hand `ccsd_dressed_r2`'s 14). Measured cause — dressed
    operators' definitions OVERLAP, so a shared primitive is over-counted:
    - `{Fae, Fme}` (6 keys, recon 3/2× raw): Fae's Stanton-Gauss `−½ t1·Fme`
      correction shares `t1·t2·v` primitives with the independently-recognized
      Fme occurrence.
    - `{Wabef, Wmnij}` (2 keys, 3/2×): shared `t2t2v`/`t1t1t2v` from both τ pieces.
    - `Fmi` (4 keys, 1/2× — under-counted): its `½ f·t1` and `½ τ̃·v` corrections.
    - 9 keys covered by NO occurrence — the genuine un-dressed remainder (stay
      bare). 44 of 59 covered keys are shared across up to 4 occurrences.
    **Key finding:** on a Fae/Fme-shared key the HAND `ccsd_dressed_r2` gives the
    correct raw coeff (1) while recognition-recon gives 3/2 — so the correct
    dressed form resolves the overlap through JOINTLY-tuned coefficients, which
    recognition (choosing each occurrence's coeff locally) breaks. So D7.3.0 must
    select/reconcile occurrence coefficients so their summed expansion equals raw
    EXACTLY — driven by the `verify_dressed_equation` diff (24→0), the same
    tripwire pattern. This is the exact-cover-with-coefficients problem the flat
    term-algebra route hit before; here it is bounded (12 occurrences, 6 operators)
    and gated by the exact oracle, so it is tractable but genuinely ~M, not the
    ~S "apply all rewrites" the original scope implied.
    - **D7.3.0a + 0b — LANDED.** The overlap has two halves; 0b removes the
      same-operator half. *0a (DAG):* the τ/τ̃ deps come straight from `op.uses`
      (Fme≺{Fae,Fmi} via τ̃, τ≺{Wmnij,Wabef}); the Fme-nesting (Fae/Fmi's `−½ f·t1`
      correction) is structural, needed by 0c not 0b. *0b (P-branch consolidation,
      `consolidate_p_branches`):* an operator's multiple occurrences are the
      branches of ONE antisymmetrized dressed term — Fae `P(ab)`, Fmi `P(ij)`,
      Fme `P(ab)`, Wmbej `P(ij)P(ab)` (verified: each branch is a signed
      external-pair image of the base on the dressed-canonical key). Consolidation
      folds the 12 occurrences into 6 per-operator groups `{base, antisym_pairs,
      branches, cover}`, **lossless** (partitions the occurrences exactly, no
      loss/dup; covers preserved — gated by `test_p_branch_consolidation`). NOTE:
      consolidation is STRUCTURAL — on its own it does not change the 24-mismatch
      count (a group re-expands to the same branches); its value is giving 0c ONE
      clean handle per operator with the P-structure exposed, and removing the
      same-operator shared-primitive bookkeeping (Fae&Fae=10, Fmi&Fmi=10,
      Fme&Fme=4 pairwise overlaps) from 0c's cross-operator problem.
    - **D7.3.0c-1 — Fme nesting scale. LANDED.** `reconcile_operator_scales`
      DERIVES (not hardcodes) each operator's coefficient scale by dependency-
      order subtraction: roots (Fae/Fmi/Wmnij/Wabef/Wmbej) at 1, then the nested
      Fme solved as `(raw − already_accounted)/own` over its keys — a unique,
      consistent **½** (the complement of the `−½ f·t1` Fme-correction Fae/Fmi
      carry; `_is_nesting_root` detects the nesting structurally from the
      definitions). Fme has NO own-only keys — every Fme key is shared — so its
      scale is entirely determined by the overlap, and ½ is the unique value
      across all of them. Applying it drops the over-count **24 → 20** (the 4
      Fme/Fae keys close). Gate `test_nesting_scale_reconciliation`.
    - **0c-2 — the `{Wabef,Wmnij}` τ-overlap. DIAGNOSED with a PRINCIPLED rule
      (ready to implement).** Wabef & Wmnij share **4** primitive keys; **2 are
      genuinely additive** (`Wa=Wn`, the τ-t2 pieces — `raw = Wa+Wn`, correct)
      and **2 over-count** (`t1t1·t2·v` raw ¼ vs ⅛+¼=3/8; `t1t1t1t1·v` raw ½ vs
      ¼+½=¾). Ruled out: NOT a per-operator scale (Wmnij's shared keys need
      inconsistent ½/1 and its own keys need 1); NOT re-weighting either operator
      (both are individually CORRECT — Wmnij's τ is `τ(a,b,k,l)`, a,b EXTERNAL →
      correctly weight 2; Wabef's is `τ_c`, correctly weight 1). The over-count is
      purely two correct operators landing t1t1 on ONE shared primitive that the
      raw residual writes ONCE. **Principled rule (derived, not raw-peeking):** on
      the 2 over keys the excess equals EXACTLY Wabef's contribution, and
      `raw = Wmnij` — i.e. the external-τ operator (Wmnij, weight-2 τ) OWNS the
      shared t1t1 primitive, and the contracted-τ operator (Wabef, τ_c) has it
      already folded via its lower weight and must NOT re-add it. So 0c-2 =
      cross-operator per-primitive dependency subtraction on the t1t1-half shared
      keys, keyed on τ(weight-2, external) vs τ_c(weight-1, contracted). The 2
      genuinely-additive keys are left untouched (the rule fires only where one
      operator's τ is weight-2-external and the other's is τ_c). Verified:
      subtracting Wabef's contribution on exactly the 2 over keys → 20→18. This is
      the same dependency-order-subtraction principle as 0c-1, applied at the
      primitive level. Implementation is the remaining work (intricate but the
      rule is exact); these 2 keys are also 2 of the hand `ccsd_dressed_r2`'s own
      τ-weight mismatches, so fixing them here is strictly ahead of the hand form.
    - **0d — Fmi correction tail (4 keys, ratio ½). DIAGNOSED, NOT a simple
      tau_c reuse.** Both Fmi occurrences summed still under-count 4 keys by ½ —
      genuinely a weight under-count in Fmi's CORRECTION terms, not a missing
      antisym partner (W3-closure makes it worse: 10/10 bad), not a per-operator
      scale (Fmi keys are mixed ½/1, so a global scale breaks the ok ones). It
      splits into TWO under-weighted correction terms:
      (i) Fmi's `½ f·t1` Fock-correction → the 2 `f·t1·t2` keys, fixed by ×2 on
          that term (found by grid, needs a PRINCIPLED derivation — likely the
          P(ij) antisymmetrizer doubling, not a hardcode);
      (ii) Fmi's τ̃ t1t1 half → the 2 `t1t1·t2·v` keys, at ½ vs raw 1 — the
          tau_c-analog for τ̃ (summed-antisym-contracted t1t1 pair, exactly the
          Wabef condition but on the τ̃ path), NOT the term's outer coeff (it's
          inside `expand_dressed_term`'s τ̃ `written_t1t1_weight/2`).
      So 0d = (i) derive the Fock-correction P-doubling + (ii) extend the tau_c
      contracted-weight to τ̃. Deferred pending the principled rule for (i) rather
      than a grid-searched ×2.
    - **0e** exact-partition gate: after 0c/0d, 20 → the 14 uncovered remainder
      wired as bare terms, tripwire → 0.
  - **D7.3.1** occurrence→`IntermediateSpec` bridge (~S; `_build_tau_spec` template).
  - **D7.3.2** multi-term rewrite (~M; drop `_try_substitute`'s single-term guard),
    driven by the D7.3.0-reconciled occurrence set.
  - **D7.3.3** dependency-ordered emit (~S; topo-sort `uses`, the `factorize_tau` slot).
  - **D7.3.4** exact algebra gate: `verify_dressed_equation(rewrite(raw), raw)` == 0.
  - **D7.3.5** numeric energy gate vs `gccsd_reference.py` (PySCF-validated numpy
    dressed reference) — no C++ compile needed.
  - **Oracles (important — `ccsd_dressed_r2` is NOT one).** Both D7.3 gates check
    against authoritative references that are INDEPENDENT of the stale hand
    transcription: (1) the raw diagram residual (`generate_cc_equations(
    engine="diagram")`, FCI-validated through CCSDT) is the algebraic target of
    `verify_dressed_equation(candidate, raw)`; (2) `gccsd_reference.py` (validated
    directly against PySCF `gccsd.update_amps` in `test_reference_vs_pyscf.py`) is
    the numeric target. `ccsd_dressed_r2` is only ever a CANDIDATE passed as the
    first arg to `verify_dressed_equation` — being stale just makes it a bad
    candidate (14 mismatches); D7.3 emits from recognition and is never checked
    against it, so there is no circularity.
  Best first step: **D7.3.0** — it is the one open question (can the sound-but-
  overlapping occurrence set be reconciled to an exact partition?); the rest is
  mechanical once it is.
- **D7.4 — scaling assertion (~S, the honest check).** The dressed residual must
  actually drop the FLOP exponent, not merely rename subexpressions — else the
  pass was cosmetic. Assert the dressed leading cost < undressed.

**Honest ceiling.** Optimal factorization is **NP-hard**; production codes use
staged heuristics (contraction-path search + CSE + memory-aware rollback). D7
buys tractable *recognition of a curated operator set*, not an optimal
factorizer. The operator set stays curated per method (the six seeded ops cover
CCSD; CCSDT/CCSDTQ add their own W-intermediates, human-derived — as PySCF/CFOUR
do). `verify_dressed_equation` gates every step against the exact undressed
residual, so D7 can never silently emit wrong algebra.

**Priority — D7 is the production gate (intent confirmed).** The generated
kernels are intended to replace the hand-written `src/post_hf/cc/` solvers in
production, **once D7 is done**: an undressed generated kernel has the wrong FLOP
scaling to ship, so the dressing D7 provides is the load-bearing prerequisite for
the swap — not optional optimization of an un-shipped path. Today the default
build still compiles `ccsd.cpp` (§5), but that is the current state, not the end
state. So D7 is critical-path, alongside the spin-adaptation layer
(`CCGEN_SPIN_ADAPTATION_SCOPE.md`) that production RHF/UHF CC needs. D7 does
**not** need the default engine flipped: dressing operates on the
diagram-produced equations regardless.

#### D7 vs the spin-adaptation layer — dress in GCC, NOT in RCC/UCC (decided)

**Question:** the production kernel is spatial RCC/UCC, so is it better to search
for dressed intermediates in the RCC/UCC residual instead of GCC?

**Answer: dress GCC first, then spin-adapt the dressed equation.** The pipeline is
`GCC → dress (D7) → adapt (spin layer) → dressed spatial RCC/UCC`, not "dress the
RCC directly." Three load-bearing reasons, all measured:

1. **The recognition substrate only exists in GCC.** D7.2 (subgraph recognition,
   the hard core) runs on `diagram_representative` / `build_line_graph` — the
   assembled spin-orbital diagram. RCC/UCC terms are post-adaptation `SpinTerm`s
   with **no diagram and no line graph**, so dressing RCC directly would first
   require building the whole diagram substrate for spatial terms — rebuilding
   D4's machinery on the wrong side. The seeded operators (`Wmnij/Wabef/Fae/…`)
   are also *defined* in spin-orbital form (`dressing.py::seeded_operators`), so
   they match the GCC substrate as-is; their RCC spatial forms (with `2J−K`
   coefficients) do not exist as clean definitions.

2. **The RCC surface is strictly harder to search.** Measured on CCSD doubles
   (diagram engine): GCC = **68 terms / 11 distinct contraction shapes**; the
   merged RCC doubles = **124 terms / 11 shapes**. Spin adaptation *splits* each
   GCC term across spin blocks (the `t2[aaaa]→t2ab−P` splits, `abab`/`aaaa`
   proliferation) — ~1.8× more terms — and stamps every factor with a spin block,
   so a single `Wmnij` becomes several block-variants (`Wmnij[abab]`,
   `Wmnij[aaaa]`, …). Same operator, more matches to find, on a substrate that
   would have to be built. The distinct-shape count is UNCHANGED (11→11), which is
   the next point.

3. **Adaptation preserves contraction topology, so dressing survives it for
   free.** Spin adaptation is a linear, per-term rewrite that preserves each
   term's contraction *shape* (11 GCC shapes → 11 RCC shapes; only coefficients
   and block tags change). And the adapter is **name-agnostic** —
   `_line_pairs`/`_antisym_to_allowed`/`block_exists` key on `len(indices)` and
   slot structure, not the factor name — so a dressed `W` intermediate flows
   through spin adaptation exactly like a bare `v` (verified: `_line_pairs` reads
   only rank). Therefore `adapt(dress(GCC))` yields the dressed spatial RCC kernel
   directly: the recognized operators carry through, becoming `Wmnij·τ` etc. with
   `2J−K` coefficients, and the FLOP win transfers because the shape is preserved.

**The one caveat (per-operator, not a blocker).** A dressed operator may carry
DIFFERENT symmetry than a bare ERI (e.g. `Wabef`'s exchange structure differs from
`⟨ab||ef⟩`), so the adapter's block treatment of each dressed factor needs a
per-operator check when it first flows through — the same `ucc_integrate_term_antisym`
== GCC-slice identity gate used for `t2`/`v`, applied to the dressed factor. That is
validation, not new machinery.

**Consequence for sequencing.** D7 and the spin layer compose cleanly in the order
`D7 ∘ adapt`, and the spin layer is already proven arbitrary-order
(`S4a2ArbitraryOrderTests`). So D7 can be built and gated entirely on the GCC
diagram surface (where its substrate lives), and the existing spin-adaptation
pipeline turns its output into the production spatial kernel with no D7-side spatial
work. Do NOT open a parallel "dress RCC" track.

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
