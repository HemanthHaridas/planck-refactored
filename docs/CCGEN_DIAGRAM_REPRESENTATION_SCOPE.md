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
- **D7.2 — subgraph recognition (~L, the core).** Find occurrences of each
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
      - **D7.2.3d — `find_operator_occurrences` driver.** Enumerate anchors →
        enumerate_hypotheses → sound-verify, return verified occurrences for D7.3
        to rewrite. NOT yet done.
  - **D7.2.4 — coefficient consistency.** Folded into D7.2.3c's verify step (the
    hypothesize-and-verify path checks coefficients as it expands), rather than a
    separate pass on a structural group. `verify_dressed_equation` is the exact
    backstop.
  - De-risk: gate on `Wmnij` first (cleanest; exercises τ expansion), then the
    family. The hypothesize-and-verify path is inherently sound — a false anchor
    fails the expansion/coefficient check — so over-proposing anchors is safe.
- **D7.3 — factorization + emit (~M).** Rewrite the residual to reference the
  recognized operators, order the intermediate DAG (`Wmnij` needs `τ`), emit
  through the existing builder path. *Gate:* the dressed residual expanded via
  `verify_dressed_equation` equals the undressed residual exactly, AND the
  emitted kernel reaches the same energy (reuse the AR3.3 / FCI harness).
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
