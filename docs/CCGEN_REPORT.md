# ccgen: An Automatic Coupled-Cluster Kernel Generator with Diagrammatic Derivation, Contraction-Path Factorization, and Memory-Aware Emission

**A cohesive technical report synthesizing the ccgen design documents, intended
as the basis for a research publication.**

---

## Abstract

`ccgen` is a Python system that derives coupled-cluster (CC) residual equations
at arbitrary truncation order directly from the second-quantized Hamiltonian and
emits C++ tensor kernels for the Planck quantum-chemistry code. This report
synthesizes its architecture and five investigations, each answering one
question. (1) **Generation:** two engines — a textbook Wick/BCH path and a
diagrammatic path enumerating canonical Kállay–Surján diagrams with a *solve-free*
topological weight — produce identical residuals; the diagrammatic path generates
CCSDTQ in **3.0 s versus 615 s** and makes dressed-operator recognition a
topological subgraph match rather than a combinatorial search. (2) **Validation:**
because no per-term spin-orbital CCSDT oracle exists, correctness is established by
reduction to PySCF-validated lower ranks and by convergence to full configuration
interaction (FCI) through CCSDTQ (agreement to ~1e-12), with the discipline that
the generator is never its own oracle. (3) **Spin adaptation:** an isolated layer
maps spin-orbital equations to restricted/unrestricted spatial form, validated
against PySCF `rccsd`/`uccsd` residuals, numerically complete through rank 8.
(4) **Factorization and rank locality:** re-associating each residual contraction
into the minimum-FLOP-exponent binary tree simultaneously reduces cost and
*derives* the reused intermediates; the derived-operator set obeys a precise
rank-locality theorem and is cumulative across the CC hierarchy (CCSDT → CCSDTQ →
CCSDTQP). (5) **Memory-aware emission:** the naive emitter is FLOP-greedy and
memory-blind; a joint selection and locality-shaping pass beats the FLOP-only
baseline on FLOP savings (+5.68%), memory (−19%), and loop-stride penalty (−98%)
simultaneously. All optimizations are behind default-off flags and exactness-gated
by construction; the cost model is symbolic, and compiled-binary runtime
measurement is the identified boundary.

---

## 1. Introduction and motivation

Coupled cluster is the standard for high-accuracy molecular electronic structure,
but its working equations grow combinatorially with truncation order: hundreds of
tensor-contraction terms at CCSDT, thousands at CCSDTQ. Hand-deriving and
hand-coding these equations is error-prone and does not scale past the ranks a
human can maintain. Automatic derivation is therefore a prerequisite for
arbitrary-order CC, and the quality of the *generated* code — its FLOP scaling,
its memory footprint, its cache behavior — determines whether generation is a
convenience or a production path.

`ccgen` addresses both halves. It derives the equations from first principles and
emits Planck-native C++ tensor kernels. This report treats the system as five
connected questions, each with a measured answer:

| # | Question | Answer (one line) |
|---|---|---|
| 1 | How are the equations generated? | Two equivalent engines; the diagrammatic one is ~200× faster and enables topological operator recognition. |
| 2 | How do we know they are correct? | Reduction to PySCF-validated ranks + FCI-limit convergence through CCSDTQ; the generator is never its own oracle. |
| 3 | How do we get spatial RCC/UCC kernels? | An isolated spin-adaptation layer, validated against PySCF residuals. |
| 4 | Can the intermediates be *derived*, not hand-seeded? | Yes — contraction-path factorization derives them and obeys a rank-locality theorem. |
| 5 | Does the emitter optimize memory and cache locality? | The baseline did not; a joint pass wins on FLOPs, memory, and stride at once. |

A design principle runs through all five: **every optimization is default-off and
exactness-gated**, so the default build is byte-identical to a naive emit, and
every transformation is proved to preserve the algebra before it is trusted.

---

## 2. Architecture

`ccgen` is a pipeline whose central data structure — a dictionary mapping each
excitation manifold to a list of `AlgebraTerm`s — is the seam that decouples the
front end (equation derivation) from the ~7000-line back end (canonicalization,
lowering, emission). This seam is what allows the derivation engine to be swapped
without touching the consumer half.

```
  method name (ccsd, ccsdt, …)
        │
        ▼   ENGINE: wick (BCH+Wick)  OR  diagram (Kállay–Surján)
  raw residual   dict[manifold → list[AlgebraTerm]]
        │
        ▼   canonical_fock=True   (drop identically-zero f_ov terms)
  canonical residual
        │
        ▼   dress / factorize      (derive reused intermediates)
        │
        ▼   spin-adapt             (GCC → spatial RCC/UCC)
        │
        ▼   lower → emit           (Planck Tensor2D/4D/6D C++)
  C++ translation unit   build_<op> functions + residual kernels
```

The stages are independently testable. The generation front end is validated to
FCI; each optimization pass carries its own algebra-preservation gate; the
emitter is validated by compilation against the real CC headers. Nothing in the
default build depends on any optimization flag.

---

## 3. Equation generation: two engines, one residual

### 3.1 The problem

The textbook route to CC equations — Baker–Campbell–Hausdorff (BCH) expansion of
the similarity-transformed Hamiltonian, projection onto each excitation manifold,
Wick contraction — enumerates every algebraic term and deduplicates afterward.
The work grows far faster than the surviving equation: at CCSDT, ~78× more
intermediate terms are produced than survive; CCSDTQ generation by this route
takes ~615 s. The term explosion also makes downstream structure (which
sub-contractions recur) invisible until after a combinatorial dedup.

### 3.2 The solution: a diagrammatic front end

The diagram engine enumerates the distinct connected diagrams (Kállay–Surján
integer strings), **canonical by construction** — no term explosion, no post-hoc
dedup. Each diagram is assigned a signed weight computed *from its topology
alone*:

- **Magnitude** = equivalent-vertex factor / 2^(equivalent-line-pairs +
  external-pairs), from the diagram's automorphisms and its bare-antisymmetric
  storage convention.
- **Sign** = crossing parity of the open oriented loops × (−1)^(loop count) ×
  a species-dependent Fock factor.

The weight is **solve-free**: it requires no reference solve and no stored table,
so it generalizes past any rank a reference program could reach. This is the
load-bearing property — a weight that needed a per-rank numerical solve would cap
the generator at the highest rank a reference could handle. The topological
weight reproduces all 30 PySCF-solved CCSD-doubles signed weights exactly.

### 3.3 Result: identical residual, ~200× faster generation

Both engines feed the unchanged downstream and produce the **same residual
tensor**. (Their term *multisets* differ — the Wick path keeps `t1·t1·v` as two
`±½` terms, the diagram path merges them — but this is an exchange-symmetry
representational choice that lowers and emits to the same runtime accumulation, so
equivalence is checked at the residual and emitted-kernel level, never by
comparing term lists.)

| method | Wick generation | diagram generation | speedup |
|---|---|---|---|
| CCSDTQ | ~615 s | ~3.0 s | ~205× |

The diagram engine is validated end-to-end through CCSDTQ and is the default for
the kernel-generation path; the Wick path is retained as the reference and can be
retired once kernel-equivalence is pinned across all consumers.

The second, deeper payoff of the diagrammatic representation is that dressed
intermediates become **identifiable subgraphs** of the assembled contraction,
which turns operator recognition from a combinatorial exact-cover search (which
reached only 20 of 70 residual fragments and was abandoned) into a topological
match (§5, §7).

---

## 4. Validation: correctness without a per-term oracle

### 4.1 The problem

There is no reference implementation to diff generated CCSDT equations against
term-by-term: PySCF ships no spin-orbital `gccsdt`, only spin-adapted
`rccsdt`/`uccsdt` and the perturbative `(T)`. A generator validated only against
itself proves nothing.

### 4.2 The solution: reduction + FCI, with a strict oracle rule

`ccgen` is validated the way MRCC and CFOUR validate arbitrary order — by a ladder
of independent anchors:

- the full CCSD residual matches PySCF `gccsd.update_amps` to machine precision;
- CCSDT singles/doubles with `T3 = 0` reduce **exactly** to the validated CCSD;
- generated CCSDT solved to convergence reaches FCI on a 3-electron system;
- generated CCSDTQ solved to convergence reaches FCI on a 4-electron system, at
  **~1.1e-12**;
- the diagram engine reproduces both the Wick residual and the FCI energy.

The load-bearing discipline: **no generation gate is pinned to
`generate_cc_equations` itself.** The generator is never its own oracle — every
gate pins to PySCF, to FCI, or to a lower rank already pinned to PySCF.

A canonical-Fock mode (§5) and an `is_dummy` fix to the canonicalization key were
the two genuine corrections from this validation effort; an earlier reported
"~2–3% raw-generation weight bug" was retracted as an off-shell dressed-reference
artifact once the residual was compared to PySCF on real amplitudes.

---

## 5. The canonical-Fock invariant

A structural invariant, verified in the C++ tree, simplifies the entire pipeline.
Every Planck CC kernel receives a **canonical** Fock reference — `f_ov = 0`
identically, `f_oo`/`f_vv` diagonal — because all CC references route through
`build_rhf_reference` / `build_uhf_reference`, which construct `f_ov = (Cᵀ F C)_ov`
with `C` the converged SCF eigenvectors, so `f_ov = 0` to convergence. No
Brueckner, semicanonical, or external-Fock entry point exists.

The consequence is not a footnote: every `f_ov`-bearing term in the CC algebra is
runtime-inert in Planck. Generation drops them at derivation time
(`canonical_fock=True`). This *dissolves* otherwise-hard questions — any dispute
over the coefficient of an `f_ov` term is moot because Planck never evaluates it —
and it makes the canonical builder, not a hypothetical general-Fock oracle, the
validation boundary. Several coefficient controversies from the dressing work
(§7) resolved simply because the terms in dispute were `f_ov`-entangled and
vanish under this invariant.

---

## 6. Spin adaptation: spin-orbital to spatial RCC/UCC

### 6.1 The problem

`ccgen` derives everything in spin-orbital (generalized, GCC) form, but production
kernels for RHF/UHF references are spatial restricted (RCC) or unrestricted (UCC)
coupled cluster. The two are related by performing the spin summation, but adding
a spin field to the index type would perturb every canonicalization/Wick/diagram
hash in the validated GCC path.

### 6.2 The solution: an isolated adaptation layer

Spin adaptation is a separate stage consuming and producing `AlgebraTerm`s, so
generation, canonicalization, lowering, and emission are untouched. It wraps a
spatial index in a lightweight `SpinIndex` and performs the spin summation on the
terms generation already produces. It is engine-agnostic by construction.

- **UCC** keeps α/β distinct: each GCC term expands into the spin blocks its
  external indices allow, filtered by a single rule — *spin conservation per
  line* (a rank-2n tensor's line pairs slot k with slot k+n; a block is nonzero
  iff the paired spins match). This proved **mechanical**: the raw GCC coefficient
  is already correct, no block-combinatoric factor needed. Validated by a
  spin-orbital identity (the chosen external block equals the sum of surviving
  integrated terms on matching slices) to ~1e-14 on the complete CCD and CCSD
  residuals, and against PySCF's UCC block set.

- **RCC** = UCC + the closed-shell constraint α ≡ β. This is the genuinely hard
  step: imposing `t2aa = t2ab − P(t2ab)` substitutes one block's amplitude by a
  combination of another's, which *changes coefficients* — terms merge and the
  characteristic `2J − K` structure appears. The derivation is de-risked
  numerically before any symbolic collapse: the "abab + substitution" model is
  proved to reproduce the sliced GCC residual on closed-shell tensors (~1e-11)
  *first*, so the symbolic coefficient collapse is written against a validated
  target.

### 6.3 The validation advantage

Unlike the GCC case, the adapted targets **have** per-residual oracles
(`pyscf.cc.rccsd.update_amps`, `uccsd.update_amps`, and at higher rank
`rccsdt`/`uccsdt`), so adapted equations are validated directly against PySCF
residuals — a stronger gate than the FCI limit alone. The symbolic RCC collapse is
complete end-to-end for CCSD, and the numeric spin-adaptation mechanism is
complete through **rank 8** (the CCSDTQ `t4` amplitude lift is validated via an
FCI-limit route that sidesteps the missing spin-orbital oracle). The remaining
work is C++ emission wiring, not new derivation.

---

## 7. Factorization and the rank-locality theorem

### 7.1 The problem

Efficient CC implementations dress the residual into named intermediates
(`Wmnij`, `Wabef`, `Fae`, …) so that contraction cost drops from O(n⁶) toward
O(n⁵). The conventional route hand-seeds each rank's intermediates. Can the
intermediates instead be *derived* mechanically — and does their structure repeat
across the CC hierarchy?

### 7.2 The solution: minimum-FLOP-exponent contraction trees

The key observation is that **the FLOP win and the intermediate are the same act
of factoring.** A residual term written as one n-ary contraction has a peak cost
equal to the number of distinct occupied/virtual indices it touches. Re-associating
it into a binary contraction tree can lower that peak — and the sub-contraction the
tree materializes to achieve it *is* a candidate intermediate. For example,
`t2·t3·v` drops from o⁵v⁵ (n-ary) to o⁴v³ when `(t3·v)` is contracted first.

`ccgen` searches all binary associations of each residual term (≤5 factors, so
exhaustive), selects the minimum-peak-exponent tree, and identifies each internal
node against a canonical key: a match to a known CCSD operator is *reuse*, a
non-match is a *newly-derived* operator. The tree search is made deterministic by
a total-order selection key (peak exponent, then a canonical tree fingerprint), so
the derived operator set is a function of the equations, not of factor input order
— a concern that affected 41% of terms before the tie-break was added.

### 7.3 The rank-locality theorem

Within this optimization model (exhaustive per-term trees, diagram-generated
canonical-Fock residuals), the derived operators obey a precise structure. Writing
`Rₙ` for the rank-n residual manifold, `Tₙ` for its highest-rank amplitude, and
`V·Tₘ` for a derived operator whose definition contracts the integral `V` with
amplitude `Tₘ`:

1. **Rank-local generation** (structural). Every operator whose definition
   contains `Tₙ` is generated only in `Tₙ`-bearing terms — because a definition
   containing `Tₙ` requires a `Tₙ` leaf, which the term must supply.
2. **Compositional separation** (structural). No operator whose definition
   contains `Tₙ` appears in a `Tₙ`-free term.
3. **Lower operators are not confined** (the main, non-obvious result). A
   `Tₙ`-bearing *term* can reuse a *lower*-rank operator, because association order
   can route the term through a low-rank intermediate before touching `Tₙ`.
   Measured: 36 such reuses in CCSDT triples, 64 in CCSDTQ quadruples. This
   refutes the natural conjecture that low-rank operators live only in `Tₙ`-free
   terms, and it establishes that **operator composition, operator reuse, and
   excitation rank are three distinct concepts.**
4. **Cumulative across rank** (observed, CCSDT → CCSDTQ). Every operator derived
   at the lower rank is reused verbatim at the higher one; each rank adds only its
   own `V·Tₙ` family. The 35 CCSDT-triples operators are fully contained in the
   CCSDTQ-triples set.

The implication for implementation is a **recursive intermediate library**: the
same builder kernels serve CCSD, CCSDT, and CCSDTQ, and each rank extends the
library by only its new `V·Tₙ` family.

### 7.4 Data: the optimizer across the CC hierarchy

Running the factorizer over the CCSDT / CCSDTQ / CCSDTQP hierarchy quantifies the
theorem. The operator count grows modestly while the per-operator FLOP savings and
memory footprint explode by orders of magnitude per rank, and a handful of
operators always carry nearly all the savings.

**Figure 1** (`python/optimizer_hierarchy.svg`, generated by
`python/plot_optimizer_hierarchy.py`): five panels over CCSDT/CCSDTQ/CCSDTQP —
operators vs rank, maximum savings vs rank, largest footprint vs rank, reuse per
operator vs rank, and coverage of the top-k operators.

| metric (O=30, V=100) | CCSDT | CCSDTQ | CCSDTQP |
|---|---|---|---|
| distinct emittable operators | 24 | 43 | 59 |
| maximum single-operator FLOP savings | 4.1×10¹⁶ | 8.7×10²⁰ | 6.5×10²⁴ |
| largest operator footprint | 64.8 GB | 1.9×10⁵ GB | 5.8×10⁸ GB |
| maximum reuse count (one operator) | 77 | 479 | 2808 |
| operators for 99% of savings (top-k knee) | 4 | 5 | 6 |

Two facts stand out. First, operators grow roughly linearly with rank while
savings and footprint grow by ~4 orders of magnitude per rank — the high-rank
`o^a v^b` blocks dominate both FLOPs and memory. Second, the savings are extremely
concentrated: at every rank, fewer than 7 operators carry 99% of the total FLOP
savings. This concentration is what makes a memory budget nearly free — the long
tail can be inlined at negligible FLOP cost (§8).

### 7.5 Emission bridge

The derived operators are emitted as C++ `build_W` functions referenced by the
residual kernels. The factorized CCSDT translation unit compiles against the real
Planck CC headers. The factorizer emits only its *newly-derived* operators;
recognition of the CCSD-standard operators (which require τ/τ̃ pseudo-amplitude
builders) is handled by the separate diagram-subgraph dressing path (§3.3), so the
two are complementary.

---

## 8. Memory-aware and cache-local emission

### 8.1 The problem

The factorized emitter of §7 selects and materializes intermediates by **FLOP
savings alone**. It never reads an operator's memory footprint or the cache
behavior of its build loop. This is measurably suboptimal on three axes:

- **B1 — selection ignores memory.** The FLOP-savings ranking and the
  savings-per-byte ranking pick *different* top operators: the FLOP winner is a
  64.8 GB tensor, the density winner is 0.02 GB (3000× smaller) for a higher
  flops/byte.
- **B2 — no feasibility guard.** The highest-savings operators are unmaterializable
  at scale — the rank-8 CCSDTQ intermediates are 194,400 GB each — yet the
  FLOP-only ranking would still select them.
- **B3 — unshaped builder loops.** Each builder was emitted as one flat n-ary loop
  nest (so an operator meant to *save* FLOPs was itself computed above its factored
  cost), with loops ordered alphabetically rather than for memory stride.

### 8.2 The solution

Each defect is addressed by a pass, all default-off and exactness-gated:

- **B1/B2 — joint budgeted selection.** A per-operator footprint guard inlines
  over-budget operators (never materializing an un-storable one), and a
  total-memory budget selects the operator *set* by running both the savings- and
  density-greedy fills and taking the higher-savings one. This "best-of-both-greedy"
  was validated against an exact 0/1 knapsack (branch-and-bound with a
  fractional-relaxation bound — not an integer-weight dynamic program, which zeros
  the small high-density operators): it is within **0.002%** of optimal on CCSDTQ
  across a dense budget sweep, so no exact solver is warranted. (At CCSDTQP the
  operator footprints span 11 orders of magnitude; the exact solver does not even
  terminate, reinforcing the greedy choice.)

- **B3 — builder-body factorization and stride shaping.** The tree search of §7 is
  applied one level down — to each operator's own definition — so the builder is
  emitted as a sequence of pairwise scratch-step contractions rather than a flat
  nest. 10 of 24 CCSDT builders drop to their factored cost, at scratch memory ~0.3×
  the operator's own footprint (a FLOP win at no peak-memory cost). A static stride
  metric (the distance of the innermost loop index from each factor's unit-stride
  axis, volume-weighted) then drives a summed-loop reordering that cuts the
  aggregate stride penalty by 55% — a pure reorder that provably preserves the sum.

### 8.3 Result

At a fixed CCSDTQ memory budget, the fully optimized emit beats the FLOP-only
baseline on all three axes **simultaneously**:

| at an 850 GB budget (CCSDTQ, O=30/V=100) | FLOP-only baseline | optimized |
|---|---|---|
| operators materialized | 15 | 26 (smaller) |
| FLOP savings retained | 1.40×10¹⁸ | **1.48×10¹⁸ (+5.68%)** |
| total memory used | 850 GB | **691 GB (−19%)** |
| builder loop stride penalty | 1.5×10¹⁶ | **2.3×10¹⁴ (−98%)** |

The three objectives were never in tension; the baseline simply ignored two of
them. The optimized selection retains more FLOP savings while using less memory
(26 small operators instead of 15 large ones), and its builders are both factored
and stride-shaped.

---

## 9. Limitations and future work

- **The cost models are symbolic.** FLOP degree, memory footprint, and the stride
  metric are computed from index-space sizes, not measured on hardware. The gates
  are model improvements; a compiled-binary runtime, cache-miss rate, or numeric
  CC energy from the emitted kernels is the identified boundary (the emitted
  translation unit is not yet compiled into a Planck binary).
- **Cross-operator sharing is out of scope.** Each operator's footprint and loops
  are shaped independently; sharing scratch or tiles across operators — where the
  largest wins compound — is the harder follow-on. Optimal cross-term contraction
  scheduling is NP-hard in general.
- **Rank-4 cumulativity is two-rank evidence.** The rank-locality theorem's parts
  1–3 are structural; part 4 (cumulativity) is measured for CCSDT → CCSDTQ and
  supported at CCSDTQP, but a proof for arbitrary rank is open.
- **Production wiring remains.** The generated kernels are intended to replace the
  hand-written `src/post_hf/cc` solvers, gated on completing the dressing and
  spin-adaptation emission paths. Today the default build compiles neither
  generated path into a binary, so generator correctness gates trusting the
  algebraic path rather than any production energy — which is precisely why the
  validation is built to the FCI standard it is.

---

## 10. Summary

`ccgen` demonstrates that arbitrary-order coupled-cluster kernels can be derived
and emitted automatically, with three results beyond mere generation. First, a
diagrammatic derivation with a solve-free topological weight generates at a rank
no reference program could tabulate, ~200× faster than the term-algebra route.
Second, contraction-path factorization *derives* the reused intermediates instead
of requiring them to be hand-seeded, and those intermediates obey a rank-locality
structure that makes the intermediate library cumulative across the CC hierarchy.
Third, the emitter can jointly optimize FLOP savings, memory footprint, and cache
locality, beating a FLOP-only baseline on all three at once. Throughout, every
optimization is default-off and exactness-gated, and correctness is anchored to
external oracles (PySCF, FCI) rather than to the generator itself.

---

## Appendix: reproducibility

- Generation: `generate_cc_equations(method, engine="diagram"|"wick", canonical_fock=True)`.
- Factorization and memory passes: `python/ccgen/optimization/factorize.py`
  (`emit_factorized_translation_unit(memory_budget_bytes=…, factor_builder_bodies=…)`).
- Figure 1: `python/plot_optimizer_hierarchy.py` → `optimizer_hierarchy.svg`.
- Tests: `python/ccgen/tests/` — `test_factorize.py` (factorization + memory, 70
  gates), `test_diagram.py`, `test_dressing.py`, `test_dressed_equation.py`,
  `test_spin.py`, `test_reference_vs_pyscf.py` (PySCF/FCI, run under a PySCF
  virtualenv).

### Source documents synthesized

- `CCGEN_GENERATION_AND_VALIDATION.md` — generation engines and validation (§3, §4).
- `CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md` — diagram front end, canonical Fock,
  dressed-operator recognition (§3, §5).
- `CCGEN_SPIN_ADAPTATION.md` — GCC → RCC/UCC (§6).
- `CCGEN_HIGHER_OPERATOR_REUSE.md` — factorization and the rank-locality theorem (§7).
- `CCGEN_INTERMEDIATE_MEMORY_LOCALITY_SCOPE.md` — memory-aware emission (§8).
- `CCGEN_TEACHING_GUIDE.md` — module-level architecture (§2).

### Key references

- Kállay & Surján, *J. Chem. Phys.* **115**, 2945 (2001) — diagram integer strings.
- Crawford & Schaefer, *Rev. Comput. Chem.* **14**, 33 (2000) — oriented-loop sign.
- Stanton & Gauss, *J. Chem. Phys.* **94**, 4334 (1991) — dressed CCSD intermediates.
