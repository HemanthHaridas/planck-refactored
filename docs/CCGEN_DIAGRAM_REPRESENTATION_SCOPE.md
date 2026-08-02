# How does ccgen generate a production coupled-cluster kernel for Planck?

This document answers one architecture question: **given a CC method name
(`ccsd`, `ccsdt`, …), how does ccgen turn it into a correctly-scaled, dressed
C++ kernel that Planck can run — and why is the pipeline shaped the way it is?**

Everything below is either landed or has its rationale pinned to a measured
result. Section headings name the architectural decision; the prose is *why*.

---

## The pipeline

```
method name
  │
  ▼  enumerate_diagrams(ranks, bra_level)          [diagram.py]
diagrams  ── canonical by construction, no term explosion
  │
  ▼  diagram_representative + diagram_signed_weight [diagram.py]
raw residual  ── dict[str, list[AlgebraTerm]], weight from topology alone
  │
  ▼  canonical_fock=True                            [generate.py]
canonical raw  ── f_ov terms dropped (Planck feeds only canonical Fock)
  │
  ▼  dress_operators (D7)                           [optimization/dressing.py]
dressed equation  ── W/F intermediates + tau/tau_c + bare remainder
  │
  ▼  spin-adapt (RCC/UCC)                           [spin.py] (separate layer)
  │
  ▼  emit                                           [emit/planck_tensor_cpp.py]
C++ translation unit  ── build_<op> functions + kernels, dependency-ordered
```

The **seam** is `dict[str, list[AlgebraTerm]]`: the enumeration front-end
produces it, everything downstream consumes it. That is what lets the front-end
be swapped (Wick → diagram) without touching the ~7000-line consume/transform
half.

| layer | modules | status under the diagram front-end |
|---|---|---|
| enumeration | `hamiltonian`, `cluster`, `wick`, `project`, `algebra` (~2600 ln) | replaced by `diagram.py` |
| dedup / canonicalize | `canonicalize.py` | mostly retired (canonical by construction) |
| optimization / dressing | `optimization/*` | retained, re-pointed at diagram terms |
| lowering / IR / emit | `lowering/*`, `tensor_ir`, `emit/*` | unchanged |

---

## Decision 1 — a diagram front-end, not Wick term enumeration

`enumerate_diagrams(ranks, bra_level)` returns the distinct connected diagrams
(Kállay–Surján integer strings), **canonical by construction** — no term
explosion, no post-hoc dedup. It reproduces the Wick term path's diagram set
exactly and scales where the term path does not (CCSDTQ generation: **3.0 s vs
615 s**; 78× fewer intermediate terms at CCSDT).

Two payoffs drove this:
1. Kill the term-explosion waste.
2. Make dressed-intermediate recognition a **topological subgraph match** instead
   of the combinatorial exact-cover search that failed on the flat term algebra
   (see Decision 4).

**Oracle rule (load-bearing).** No diagram gate is pinned to
`generate_cc_equations` — the Wick term-path generator is the code under
suspicion and must never be its own oracle. Every gate pins to **PySCF** (via the
repo-root `tests/pyscf/.venv`) or to **pure topology**. Fast gates:
`tests/test_diagram.py`; numeric: `tests/test_reference_vs_pyscf.py`.

---

## Decision 2 — the diagram weight is derived from topology, solve-free

`diagram_signed_weight(ds, h_rank) = structural_sign · diagram_magnitude`,
computed from the diagram's topology alone — **no PySCF solve, no stored table,
no generator** — and reproduces all 30 PySCF-solved CCSD-doubles signed weights.

**Magnitude:** `|w| = equivalent_vertex_factor / 2^(equivalent_line_pairs + external_pairs)`
- `equivalent_line_pairs`: summed lines sharing both endpoints + species.
- `equivalent_vertex_factor`: `∏ 1/n_v!` over identical operators with the same
  internal connection signature (NOT a naive `1/n!` over all same-rank vertices).
- `external_pair_factor`: `2^p`, `p` = same-species external-line pairs, counted
  per amplitude and on the Hamiltonian vertex (the bare-antisymmetric storage
  convention).

**Sign:** `crossing_parity · (-1)^l · (-1 if the Fock line contracts a hole)`
- `(-1)^l` from `directed_loops` (Crawford oriented-loop count).
- `crossing_parity`: +1 if the open directed loops pair the doubles-externals
  identity `(i↔a, j↔b)`, −1 if crossed.
- The Fock factor is genuinely species-dependent (−1 iff the one-body line is a
  hole).

**Why solve-free is the load-bearing property:** the weight generalizes past any
rank *without* a PySCF solve to build a table against — the precondition for true
arbitrary-rank generation. A weight that needed a per-rank solve would cap the
generator at whatever rank PySCF could solve.

---

## Decision 3 — dress against the CANONICAL-Fock residual (the CC-reference invariant)

**Invariant, verified in the C++ tree:** every Planck CC kernel receives a
**canonical** Fock reference — `f_ov = 0` identically, `f_oo`/`f_vv` diagonal — by
construction, not convention.

- All CC references route through `build_rhf_reference` / `build_uhf_reference`
  (`src/post_hf/cc/common.cpp`) — RHF/UHF only; ROHF is FCI/CASSCF-only and is
  rejected from the CC path. No Brueckner / semicanonical / external-Fock entry
  point exists.
- `build_canonical_rhf_cc_reference` sets `f_ov(i,a) = (Cᵀ F C)(i, n_occ+a)` where
  `C` diagonalizes `F` (the converged SCF), so `f_ov = 0` to convergence.

**Consequence for the architecture:** all `f_ov`-bearing terms in the CC algebra
are runtime-inert in Planck. `generate_cc_equations(..., canonical_fock=True)`
drops them at generation time (impl `_drops_under_canonical_fock`; the hand-written
`src/post_hf/cc` backends already use only `eps_occ`/`eps_virt`, never `f_ov`).
This is not an optimization footnote — it *dissolves* an otherwise-hard question:
any dispute over the coefficient of an `f_ov` term (e.g. the general-Fock Fmi
`f·t1` coefficient) is moot, because Planck never evaluates it. The canonical
builder is the validation boundary; a general-Fock oracle is not needed.

Emit honors it too: `generate_ccsdt_cpp.py --canonical-fock` strips `f_ov` from
the emitted kernel.

---

## Decision 4 — recognize dressed operators as diagram subgraphs, not by exact-cover

Dressing factors the residual into intermediates (`½ Wmnij·τ`, `½ Wabef·τ`,
`t2·Fae`, …) so contraction cost drops from `O(n⁶)` toward `O(n⁵)` where the
dressing covers. This is why PySCF/CFOUR ship dressed CC, and — since an
undressed generated kernel has the wrong FLOP scaling to ship — it is the
load-bearing prerequisite for the generated kernels to replace the hand-written
`src/post_hf/cc` solvers in production.

**Why the diagram representation makes it tractable.** The retired term-algebra
route tried to *recognize* dressed operators by index-binding + exact cover over
the flat post-Wick term list — a dead end (reached only 20/70 residual
fragments). The diagram representation makes each dressed operator an
identifiable **subgraph of the assembled contraction** (`diagram_representative` /
`build_line_graph`), so recognition is a topological match, not a combinatorial
search. That supersession is the entire reason exact-cover was retired.

**The recognition + assembly stack (all landed):**

- **Operator encoding.** The six Stanton–Gauss operators
  (`Fae/Fmi/Fme`, `Wmnij/Wabef/Wmbej`, `seeded_operators()`) are encoded as
  `FragmentLineGraph`s — open line graphs whose block indices are dangling
  `("port", slot)` endpoints, in the same edge format as `diagram.LineGraph` so
  one homogeneous match works.

- **Recognition** (`find_operator_occurrences`). All six operators now recognize
  in the CCSD residual. Four hard-won sign/weight fixes made the family work:
  1. **v-parity sign fold** — `_eri_normalize_factor` must fold the parity of the
     `v`-reordering into the coefficient (an odd intra-pair swap carries −1);
     dropping it silently rejected correct Fae/Wabef hypotheses. Operator
     antisymmetry groups are derived from the block (`(0,1),(2,3)`), not
     hardcoded ERI-style `(0,2),(1,3)` — correct for `oooo`/`vvvv`, correctly
     empty for the mixed-space `ovvo` (Wmbej).
  2. **`tau_c` contracted-weight (Wabef)** — a τ whose bra pair is summed and
     antisymmetrically contracted into the operator's own `v` needs its written
     t1t1 half at weight 1, not 2 (the `v`'s antisymmetry supplies the partner).
     Carried on a distinct factor name (`tau_c`) because a rest-τ and a
     definition-τ coexist in one term after expansion — no local term inspection
     can separate them.
  3. **`_eri_canonical` ordering (Fmi)** — fold bra↔ket exchange AFTER dummy
     relabeling, not before; otherwise the same integral with differently-named
     dummies normalizes its `v` to different orientations and never folds. This
     one fix also cut the whole-equation oracle mismatch 19→7.
  4. **asymmetric-block binding sign (Wmbej)** — an `ovvo` block's genuine
     orientations carry the bare-`v` antisymmetry sign; apply it GATED on block
     asymmetry (`_block_is_asymmetric`), never to `oooo`/`vvvv` (where it would
     rescue spurious orientations and double the accepted set).

- **Assembly** (`assemble_dressed_equation`). The dressed manifold =
  **bare + dressed + corrections**:
  - *bare* = raw terms whose canonical key is NOT in any occurrence's
    **expansion footprint**. Load-bearing subtlety: partition on the expansion
    footprint, NOT the occurrence `cover` — the cover was antisym-closed for
    dedup and over-claims partner keys the single written `W·rest` form does not
    emit, so partitioning on `cover` silently drops bare terms.
  - *dressed* = each `W·rest` occurrence term × its per-operator nesting scale
    (`reconcile_operator_scales`, dependency-ordered: e.g. Fme's scale derived as
    the complement of the `−½ f·t1` correction Fae/Fmi carry).
  - *corrections* = the τ/τ_c cross-operator overlap deltas
    (`tau_overlap_corrections`): where an external-τ operator (Wmnij, weight-2 τ)
    and a contracted-τ operator (Wabef, τ_c) share a t1t1 primitive, the τ_c
    duplicate is subtracted (the external-τ operator owns it).

  Against the canonical raw, this re-expands **exactly** (0 mismatches). The
  earlier 4 "real" mismatches all dissolved in canonical mode — 2 were `f_ov`
  terms, 2 were `f_ov`-entangled τ̃ artifacts — so no `tau_tilde_contracted`
  machinery was needed and the Fmi coefficient stays at the textbook ½.

- **Bridge + emit** (`operator_to_intermediate_spec`, `dress_operators`).
  `print_cpp_planck(dress_operators=True)` (CLI `--dress-operators`) assembles the
  dressed equation, builds the operator `IntermediateSpec`s + `tau`/`tau_c`
  specs, and emits `build_<op>` functions **dependency-ordered** (τ/τ_c before the
  W/F that reference them). Default off ⇒ byte-identical to the undressed emit.
  One emit-layer change: `_map_factor`/`emit_planck_term`/
  `_emit_intermediate_builder` take an `intermediate_names` set so any declared
  intermediate resolves as a local reference (previously only `W_*`/`tau` were
  recognized).

**Gates.** Equation-level correctness is fully gated in Python
(`test_dressing.py`, `test_dressed_equation.py`): recognition (all six operators),
the exact canonical partition, and the emit path (default-identical, builders
present, dependency order). The one remaining step is the **numeric energy gate**
(D7.3.5): compile the emitted dressed kernel into Planck and check the CC energy
against the hand-written solver / PySCF. That is a C++ build-integration step —
the emitted TU is not `#include`d into a binary yet — not a Python-side one; the
algebraic + canonical-Fock exactness already establish equation-level
correctness.

---

## Decision 5 — dress in GCC, then spin-adapt (not dress RCC/UCC directly)

The production kernel is spatial RCC/UCC, but dressing runs on the **spin-orbital
(GCC)** residual, and the dressed equation is spin-adapted afterward:
`GCC → dress → adapt → dressed spatial RCC/UCC`. Three measured reasons:

1. **The recognition substrate only exists in GCC.** Subgraph recognition runs on
   `diagram_representative` / `build_line_graph` — the assembled spin-orbital
   diagram. RCC/UCC terms are post-adaptation `SpinTerm`s with no diagram/line
   graph; dressing them directly would mean rebuilding the diagram substrate on
   the wrong side. The seeded operators are also *defined* in spin-orbital form.
2. **The RCC surface is strictly harder.** Adaptation splits each GCC term across
   spin blocks (CCSD doubles: 68 GCC terms → 124 RCC terms, ~1.8×) and stamps
   every factor with a block, so one `Wmnij` becomes several block-variants — more
   matches to find, on a substrate that would have to be built.
3. **Adaptation preserves contraction topology.** It is a linear per-term rewrite
   that keeps each term's contraction shape (11 GCC shapes → 11 RCC shapes; only
   coefficients and block tags change), and the adapter is name-agnostic
   (`_line_pairs`/`block_exists` key on rank + slot structure, not factor name).
   So a dressed `W` intermediate flows through spin adaptation exactly like a bare
   `v`, and the FLOP win transfers for free. One per-operator caveat: a dressed
   operator may carry different symmetry than a bare ERI, so each dressed factor's
   block treatment needs the same `ucc_integrate_term_antisym`-vs-GCC-slice check
   used for `t2`/`v` when it first flows through — validation, not new machinery.

---

## Do not rewrite the enumeration/weight from scratch

The verified, correct work — enumeration, assembly, and especially the structural
weight rule — is not something a rewrite improves. The one genuinely hard piece
(the weight rule) is a property of the physics, not the code; a rewrite inherits
it unchanged and re-derives the same encoding against the same oracles. The only
legitimate rewrite-adjacent action is targeted dead-code deletion behind a passing
gate (retiring the Wick enumeration producers once kernel-equivalence is pinned).

---

## Appendix — ruled-out approaches (measured; do not re-run)

- **Route 2 (uniform-orbit invariance).** ccgen's doubles residual is the direct
  projection `⟨Φ|H̄|0⟩`, NOT a materialized antisymmetric `P(ij)P(ab)` orbit; a
  diagram maps to a set of distinct arrangements each with its own coefficient
  (13/15 `t1·t2·v` pairs are non-proportional). "Probe A invariance holds" was
  measured against a fabricated antisymmetric target — retracted.

- **D4-via-table.** Emitting a stored per-diagram weight × orbit as terms hits the
  ragged-split/convention wall (the `t1·t1·t2·v` over-count is sub-diagram; a
  per-diagram rescale is a no-op). The table is a correct *tensor* oracle, not a
  drop-in term weight.

- **Sign as a scalar count.** The sign convention delta is NOT any loop/hole count
  (`(-1)^h`, `(-1)^l`, `(-1)^(h+l)`, `(-1)^open/closed`, inversion parity all
  ≤ 21/30) and NOT fixable by external relabel. The resolution was the crossing
  parity of the open loops' external endpoints + the Fock species factor.

- **t1·t2 merge fixes (post-projection rename, `>1` collision exclusion,
  rename-before-canonicalize).** All fail the whole-residual GCCSD gate — the fix
  must co-design the merge key and projection relabel, not a post-hoc pass.

- **Exact-cover / embedded-τ recognition of dressed operators.** Index-binding +
  exact cover over the flat term list reached only 20/70 fragments. Superseded by
  diagram subgraph recognition (Decision 4).

- **Fmi `f·t1` coefficient ½→1.** Chasing the diagram raw's general-Fock coeff
  contradicts the textbook (Fmi `f·t1` = ½) and rests on an `f_ov` term that is
  runtime-zero in Planck (Decision 3). Not a fix — the term never evaluates.
  Fmi coefficient left at ½.

- **Global `written_t1t1_weight` bump for the τ̃ tail.** Breaks
  `reconcile_operator_scales` (changes all τ/τ̃ weights inconsistently); a
  contracted variant must be a distinct name, not a global change. Moot in
  canonical mode anyway (Decision 4, assembly).

- **ccgen parallel generation** (`parallel_workers>1`) is not equivalence-safe
  (spawn-unsafe `_wickaccel`, partition-local raw merge). Serial is deterministic
  and correct; parallel stays opt-in. Unrelated to the diagram work.

---

## References

- Kállay & Surján, *J. Chem. Phys.* **115**, 2945 (2001) — diagram integer strings.
- Crawford & Schaefer, *Rev. Comput. Chem.* **14**, 33 (2000) — oriented-loop sign.
- Stanton & Gauss, *J. Chem. Phys.* **94**, 4334 (1991) — the dressed CCSD
  intermediates (Fae/Fmi/Fme, Wmnij/Wabef/Wmbej).
