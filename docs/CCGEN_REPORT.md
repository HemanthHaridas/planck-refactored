# ccgen: automatic coupled-cluster kernel generation, from spin-orbital algebra to a running production kernel

**A synthesis of the ccgen design documents, intended as the basis for a research
publication. Rewritten 2026-08-28 against the current document set.**

Status — what is landed and what is open — lives in `vault/Status/Completion.md`
and `vault/Status/Open Work.md`, which are canonical. This report is the
architecture narrative, not the status board.

---

## Abstract

`ccgen` derives coupled-cluster (CC) residual equations at arbitrary truncation
order from the second-quantized Hamiltonian and emits C++ tensor kernels for the
Planck quantum-chemistry code. This report synthesizes its architecture and seven
investigations, each answering one question. (1) **Generation:** two engines — a
textbook Wick/BCH path and a diagrammatic path enumerating canonical
Kállay–Surján diagrams with a *solve-free* topological weight — produce identical
residuals; the diagrammatic path generates CCSDTQ in **3.0 s against 615 s**.
(2) **Validation:** with no per-term spin-orbital CCSDT oracle available,
correctness is anchored by reduction to PySCF-validated lower ranks and by
convergence to full configuration interaction (FCI) through CCSDTQ (~1e-12), under
the discipline that the generator is never its own oracle. (3) **Spin
adaptation:** an isolated layer maps spin-orbital equations to restricted (RCC)
and unrestricted (UCC) spatial form; it exists for **cost, not correctness**, and
is validated against PySCF `rccsd`/`uccsd` residuals to ~6e-16. (4) **Rank-4
multi-sector structure:** at rank 4 a spatial amplitude is no longer one tensor —
`t4` carries two independent S_z sectors, and storing one silently collapses
CCSDTQ onto CCSDT. (5) **Open-shell CC:** arbitrary-order UCC kernels reach FCI
to ten digits on an *open-shell* system. (6) **Factorization:** re-associating
each residual contraction into a minimum-FLOP-exponent binary tree both cuts cost
and *derives* the reused intermediates, which obey a rank-locality theorem and
are cumulative across the CC hierarchy. (7) **Production:** the derived-operator
route is wired end to end and measured at **3.12×/3.61× wall-clock** against the
undressed generated kernel, with energies matching to 2e-10 and exactly.
(8) **Where the production cost actually is:** profiling the generated path against
itself put **98.8 %** of an iteration in a single kernel call and **67.7 %** of that
in operator builders being rebuilt once per emitted chunk — fixed for a further
**1.76×** with bitwise-identical energies — and identified the largest remaining
lever as the fact that the coupled-cluster code carries **no OpenMP at all**.

Two cost models built from a census of the emitted code preceded that profile and
went **one for two**; the profile found in twenty minutes a defect neither could
see, because neither modelled work that should not happen at all. *Profile before
modelling* is the transferable result.

A second thread runs through the whole record and is reported here as a result in
its own right: **five separate defects in this system converged, self-consistently,
to a plausible wrong number**, and in every case the gate that should have caught
it passed because its fixture or its assertion carried more symmetry than the real
object. That failure mode, and the discipline that now guards against it, is §10.

---

## 1. Introduction

Coupled cluster is the standard for high-accuracy molecular electronic structure,
but its working equations grow combinatorially with truncation order: hundreds of
tensor-contraction terms at CCSDT, thousands at CCSDTQ. Hand-derivation does not
scale past the ranks a human can maintain, so automatic derivation is a
prerequisite for arbitrary-order CC. The quality of the *generated* code — its
FLOP scaling, its memory footprint, whether it can be trusted — determines whether
generation is a convenience or a production path.

`ccgen` addresses both halves. This report treats the system as seven connected
questions, each with a measured answer:

| # | Question | Answer (one line) | §|
|---|---|---|---|
| 1 | How are the equations generated? | Two equivalent engines; the diagrammatic one is ~200× faster and canonical by construction. | 3 |
| 2 | How do we know they are correct? | Reduction to PySCF-validated ranks + FCI-limit convergence through CCSDTQ; the generator is never its own oracle. | 4 |
| 3 | How do we get spatial RCC/UCC kernels? | An isolated adaptation layer, validated against PySCF residuals. It exists for cost, not correctness. | 6 |
| 4 | Does one spatial tensor per rank suffice? | No — at rank 4, `t4` has two independent S_z sectors. | 7 |
| 5 | Does this work open-shell? | Yes; UCC reaches FCI to ten digits on a doublet. | 8 |
| 6 | Can intermediates be *derived* rather than hand-seeded? | Yes — contraction-path factorization derives them, and they obey a rank-locality theorem. | 9 |
| 7 | Does any of it run in production? | Yes, as of 2026-08-26: the derivation route is wired and measured at 3.12×/3.61×. | 5, 9.4 |

A design principle runs through all of it: **every optimization is flag-gated and
exactness-gated**, so a default build is byte-identical to a naive emit and every
transformation is proved to preserve the algebra before it is trusted. §10 is the
record of where that discipline was insufficient, and why.

---

## 2. Architecture

`ccgen` is a pipeline whose central data structure — a dictionary mapping each
excitation manifold to a list of `AlgebraTerm`s — is the seam that decouples the
front end (equation derivation) from the back end (canonicalization, lowering,
emission). That seam is what lets the derivation engine be swapped without
touching the consumer half, and what lets spin adaptation and factorization be
inserted as stages rather than forks.

```
  method name (ccsd, ccsdt, ccsdtq, …)
        │
        ▼   ENGINE: wick (BCH+Wick)  OR  diagram (Kállay–Surján)
  raw residual   dict[manifold → list[AlgebraTerm]]
        │
        ▼   canonical_fock=True        drop identically-zero f_ov terms  (§5)
  canonical residual
        │
        ▼   [recognized dressing]      hand-seeded operators — RETIRED  (§9.3)
        │
        ▼   spin-adapt                 GCC → spatial RCC, or → spin-blocked UCC  (§6)
        │
        ▼   [derived dressing]         factorize into binary trees      (§9)
        │
        ▼   lower → emit               Planck Tensor2D/4D/6D/ND C++
  C++ translation unit   build_W_* builders + residual kernels
```

**The two dressing stages sit at different points, and that is structural, not an
accident of implementation.** `recognized` must run *before* spin adaptation
because its hand-seeded specs declare GCC layouts that the adapter then transforms.
`derived` must run *after*, because it derives operators from whatever manifold
reaches it, so its specs are already in the adapted layout. Placing `derived`
early would declare one layout and build another.

Each stage is independently testable, and each carries its own algebra-preservation
gate. Reaching a running binary is a further problem, treated in §5.

---

## 3. Equation generation: two engines, one residual

### 3.1 The problem

The textbook route — Baker–Campbell–Hausdorff expansion of the
similarity-transformed Hamiltonian, projection onto each excitation manifold, Wick
contraction — enumerates every algebraic term and deduplicates afterwards. The
work grows far faster than the surviving equation: at CCSDT roughly 78× more
intermediate terms are produced than survive, and CCSDTQ generation by this route
takes ~615 s. The term explosion also hides downstream structure — which
sub-contractions recur — until after a combinatorial dedup.

### 3.2 The diagrammatic front end

The diagram engine enumerates the distinct connected diagrams (Kállay–Surján
integer strings), **canonical by construction** — no term explosion, no post-hoc
dedup. Each diagram is assigned a signed weight computed from its topology alone:

- **Magnitude** = equivalent-vertex factor / 2^(equivalent-line-pairs +
  external-pairs), from the diagram's automorphisms and its bare-antisymmetric
  storage convention.
- **Sign** = crossing parity of the open oriented loops × (−1)^(loop count) × a
  species-dependent Fock factor.

The weight is **solve-free**: no reference solve, no stored table. This is the
load-bearing property — a weight requiring a per-rank numerical solve would cap
the generator at the highest rank a reference program could reach. The topological
weight reproduces all 30 PySCF-solved CCSD-doubles signed weights exactly.

### 3.3 Result

Both engines feed the unchanged downstream and produce the **same residual
tensor**. Their term *multisets* differ — the Wick path keeps `t1·t1·v` as two
`±½` terms, the diagram path merges them — but that is an exchange-symmetry
representational choice which lowers and emits to the same runtime accumulation.
Equivalence is therefore checked at the residual and emitted-kernel level, **never
by comparing term lists**.

| method | Wick generation | diagram generation | speedup |
|---|---|---|---|
| CCSDTQ | ~615 s | ~3.0 s | ~205× |

The diagram engine is validated end to end through CCSDTQ and drives the
kernel-generation path; `wick` remains the library default and the reference, and
can be retired once kernel equivalence is pinned across all consumers. `project.py`
holds the shared types and stays either way.

---

## 4. Validation: correctness without a per-term oracle

### 4.1 The problem

There is no reference implementation to diff generated CCSDT equations against
term by term: PySCF ships no spin-orbital `gccsdt`, only spin-adapted
`rccsdt`/`uccsdt` and the perturbative `(T)`. A generator validated against itself
proves nothing.

### 4.2 A ladder of independent anchors

`ccgen` is validated the way MRCC and CFOUR validate arbitrary order:

- the full CCSD residual matches PySCF `gccsd.update_amps` to machine precision;
- CCSDT singles/doubles with `T3 = 0` reduce **exactly** to the validated CCSD;
- generated CCSDT solved to convergence reaches FCI on a 3-electron system;
- generated CCSDTQ solved to convergence reaches FCI on a 4-electron system, at
  **~1.1e-12** (Python solver) and `-14.4036551081` against PySCF `rccsdtq`
  `-14.4036551082` (C++ runtime);
- arbitrary-order **UCC** reaches FCI to ten digits on an *open-shell* system (§8);
- the diagram engine reproduces both the Wick residual and the FCI energy.

The load-bearing discipline: **no generation gate is pinned to
`generate_cc_equations` itself.** Every gate pins to PySCF, to FCI, or to a lower
rank already pinned to PySCF.

A four-electron closed shell is always an oracle, because CCSDTQ ≡ FCI there — so
a cheaper system than Be is a legitimate substitute. Rank ≥ 10 has **no** numeric
oracle: no small system makes CCSDTQP exact and PySCF has no quadruple/pentuple
amplitudes, so the arbitrary-order algebra above rank 4 is gated structurally.

### 4.3 A retracted defect, worth recording

Earlier notes documented a ~2–3 % weight error in the CCSD `t1·t2`-mixing doubles
terms. It was an artifact: the gate compared ccgen to a hand-transcribed *dressed*
(Stanton–Gauss) reference on random **off-shell** amplitudes, where the raw
projection and the dressed form need not agree term by term. On real amplitudes
both match PySCF. Two genuine fixes survive that investigation — canonical-Fock
mode (§5) and adding `is_dummy` to the canonicalization key.

---

## 5. Two structural invariants: the canonical Fock, and the route to a binary

### 5.1 The canonical-Fock invariant

Every Planck CC kernel receives a **canonical** Fock reference — `f_ov = 0`
identically, `f_oo`/`f_vv` diagonal — because all CC references route through
`build_rhf_reference` / `build_uhf_reference`, which construct
`f_ov = (Cᵀ F C)_ov` with `C` the converged SCF eigenvectors. No Brueckner,
semicanonical, or external-Fock entry point exists. (ROHF references reach FCI and
CASSCF, never CC.)

The consequence is not a footnote: every `f_ov`-bearing term in the CC algebra is
runtime-inert in Planck, and generation drops them at derivation time
(`canonical_fock=True`). This *dissolves* otherwise-hard questions — a dispute
over an `f_ov` term's coefficient is moot because Planck never evaluates it — and
it makes the canonical builder, not a hypothetical general-Fock oracle, the
validation boundary. Several coefficient controversies from the dressing work
resolved for exactly this reason.

### 5.2 Emission is not execution

A ccgen-emitted kernel is a `.cpp` file. Getting it *executed* is a separate
problem from getting it *emitted*, and for a long time the two were confused —
kernels existed, compiled, and were never called. Three generated artifacts enter
a binary three different ways:

| artifact | how it enters | reached by |
|---|---|---|
| `ccsd_spinorbital_warm_start.inc` | `#include`d unconditionally | always — the RCCSD warm start |
| `<method>_planck_generated.cpp` | `#include`d in `tensor_backend.cpp` | only under the tensor-optimized RCCSDT backend |
| `<method>_arbitrary_planck_generated.cpp` | registry, behind `PLANCK_CC_MAXORDER` / `PLANCK_CC_ARBITRARY_LOWER_RANKS` | `rccgen.cpp` → the arbitrary-order harness |

**The third row is the production route.** The build flags that gate each stage:

| flag | gates | default |
|---|---|---|
| `PLANCK_CC_MAXORDER` | which ranks are emitted at all | 3 |
| `PLANCK_CC_ARBITRARY_LOWER_RANKS` | lowers `generated_floor` 4→3 | OFF |
| `PLANCK_CC_SPIN_ADAPT` | spatial RCC vs the historical spin-orbital emit | **ON** since 2026-08-26 |
| `PLANCK_CC_UCC` | a second emit pass for spin-resolved kernels | OFF |
| `PLANCK_CC_DRESS_OPERATORS` + `PLANCK_CC_DRESSING` | dressed operators, and which route derives them | OFF / `recognized` |
| `PLANCK_RCCSDT_BACKEND` (env) | `determinant` / `tensor` / `optimized` at run time | — |

**Two of these have silently produced wrong answers**, both because a default
preserved historical rather than correct behaviour. `PLANCK_CC_SPIN_ADAPT=OFF`
emitted algebra whose correlation energy is ~4× wrong (measured factor 3.63 on Be
CCSDTQ), and cost an entire investigation that read convincingly as a kernel
defect before anyone diffed the build cache. `PLANCK_CC_DRESSING` did not exist,
so CMake hard-coded the retired recognition route and the derivation route was
unreachable from a build. The lesson both times is the same: **check the build
cache before the code**, and diff the whole cache rather than the flag you changed.

A third trap hid all of this. `choose_determinant_backstop` routes any case with
`nso ≤ 16 && ndet ≤ 10000` to a determinant-space teaching backstop that **calls
no generated code**, and most CC regression cases land there. For a long time the
one case above the threshold *asserted* `kernels=hand-optimized` — green for its
entire life while never executing the kernel it was added to protect. The backstop
binds the **hand-written** path only; `PLANCK_RCCSDT_BACKEND=optimized` routes
through `rccgen.cpp` and never consults it, so small systems *can* exercise the
generated route (LiH/STO-3G does, in 5 s). Several ccgen documents still record
the threshold as universal; it is not.

---

## 6. Spin adaptation: spin-orbital to spatial

### 6.1 Why the layer exists — cost, not correctness

This is the fact most worth keeping straight, because it inverts the intuition.
**The GCC equations already give the right closed-shell answer**: evaluated on
spin-orbitals from an RHF reference, ccgen's GCC energy matches PySCF RCCSD
`e_corr` to 1e-8 and PySCF GCCSD to 7e-9, since GHF-CC and RCC are
energy-equivalent for a closed shell.

RCC/UCC are therefore **not needed for the right number**. They exist to exploit
spin symmetry for efficiency — the spin-orbital representation costs ~16× the `t2`
storage and ~64× the doubles-contraction FLOPs of the spatial one, a ratio of
`2^(2·rank)` that grows with system size. That makes this layer a *performance*
prerequisite for replacing the hand-written solvers, not a correctness one.

### 6.2 Where it sits, and the two directions

An insertable stage consuming and producing `AlgebraTerm`s, so generation,
canonicalization, lowering and emission are untouched, and it is engine-agnostic
by construction. It does **not** add a spin field to `Index` — that identity
`(name, space, is_dummy)` is baked into every canonicalize/Wick/diagram hash, and
perturbing it would invalidate the validated GCC path. A lightweight `SpinIndex`
wraps a spatial index instead.

- **UCC keeps the blocks resolved.** One residual per stored block
  (`doubles_aaaa`, `doubles_abab`, `doubles_bbbb`). The raw GCC coefficients come
  through unchanged — this half is bookkeeping.
- **RCC collapses them under α ≡ β.** This is where the real derivation lives and
  the only place coefficients genuinely change: the familiar `2J − K` structure
  *emerges from the merge*. The exchange term is not an extra input; it is what
  the collapse produces.

### 6.3 Four traps, each of which passed a gate first

1. **A synthetic `v` hid the exchange entirely.** A `block_exists` filter looked
   like a harmless optimization. Every gate to that point used a
   spin-conserving-per-line synthetic `v` whose forbidden blocks are *zero*, so
   the filter only ever dropped zeros. On a real antisymmetric `<pq||rs>` those
   blocks are nonzero — **they are the exchange** — and the filter discarded them
   (residual off by ~0.06). The fix, `ucc_integrate_term_antisym`, re-expresses
   each forbidden factor into its allowed block via bra/ket swaps carrying `−1`;
   that `−1` *is* the `−K`.
2. **Rank-4 hardcoding dropped valid higher-rank terms.** Generalized to rank-2n:
   a factor maps to an allowed block iff `sorted(bra_spins) == sorted(ket_spins)`,
   with the within-group permutation parity giving the sign.
3. **Line-swap antisymmetry is not axis antisymmetry.** The `aab`/`abb` `t3`
   blocks are antisymmetric under the occ-pair swap alone and under the vir-pair
   swap alone, but **symmetric under the joint swap** — and a physical line swap
   *is* the joint swap. The value must be `sign(P) × blk[canonical-line-order]`
   where `P` permutes **lines**, not axes.
4. **Rank 4 has two independent S_z sectors** — §7.

**The pattern across all four:** a fixture with more symmetry than the real object
cannot see a defect that abuses symmetry. That recurs throughout this system (§10).

### 6.4 Validation

Unlike the GCC case the adapted targets *have* per-residual oracles, so adapted
equations are checked directly against PySCF rather than only at the FCI limit.
RCC reproduces the GCC residual on real integrals to ~1e-16 and vanishes at
PySCF's converged RCCSD amplitudes; UCC matches PySCF UCCSD to **~6e-16** across
all five blocks. Three interface corrections cost the most time and are worth
carrying: the closed-shell oracle is a **per-target pairing**, not a block sum;
the PySCF amplitude mapping is a **transpose**, not a rename; and `f_ov` must be
zeroed on **both** sides — one-sided is worse than neither.

---

## 7. Rank 4 breaks the one-block assumption

Spin-adapting to a spatial equation is exact at ranks 1–3 with a single amplitude
block per rank. **At rank 4 it is not.**

| block | S_z sector | independent? |
|---|---|---|
| `t4` (reference) | `aabbaabb` | yes |
| `t4_aaabaaab` | `aaabaaab` | **yes — not reducible to the reference** |
| `abbb…` | — | folds onto `aaab` by spin flip |

`aaab` is not derivable from `aabb` — proven, not assumed, and not even through a
shared `tau`. The generalization is `⌊n/2⌋` sectors for rank `2n`, so rank 4 is
where the single-block assumption first breaks rather than a special case.

Every layer downstream must carry the *set*: the bridge names the sector, the
generator emits a residual per stored block, and the solver allocates and drives
each block from its own residual. Before that, the algebra referenced a block the
amplitude dict did not supply, and the solver iterated a fixed `targets` list, so
the second sector stayed zero. **The failure was invisible because CCSDTQ-with-no-T4
is a converged, self-consistent CCSDT answer** — nothing crashes, the energy is
simply 3e-6 short, which reads as a tolerance question. The recovered T4
contribution is −4.4e-6.

One correction worth keeping: the sector denominator is **identical to the
reference rank denominator**, not built from the sector's S_z layout. For an RHF
reference the orbital energies are spin-free, so `Σε_occ − Σε_vir` over the spatial
slots is the same for `aabb` and `aaab`. Denominators are keyed by rank alone.

Two independent implementations (a Python damped-Jacobi solver and the C++
block-keyed runtime) reach the same energy, and PySCF agrees to 1e-10.

---

## 8. Open-shell: arbitrary-order UCC

**UCC is RCC minus the spatial collapse.** The same generator, bridge, runtime,
solver and sector machinery serve both; UCC's defining property is *skipping* the
steps that fold spin blocks into one tensor per rank. Everything hard about it
follows from one consequence — **a quantity that is one thing under RHF is several
under UHF** — and every layer that assumed "one" had to learn "several":

| layer | RHF | UHF |
|---|---|---|
| amplitudes | one tensor per rank | one per `(rank, tag)` sector |
| denominators | one per rank | one per block — `abab` differs in **shape** |
| ERIs | 7 named members | **24** arrays (7 `aaaa` / 10 `abab` / 7 `bbbb`) |
| Fock | `f_oo`/`f_ov`/`f_vv` | per-spin blocks |
| orbital counts | one `(n_occ, n_virt)` | **four** counts |

The block vocabulary is **derived, never negotiated**: one array per orbit of the
16 o/v patterns under each tag's own symmetry group. A mixed block's orbits are
smaller — two of the four permutations map `abab` to `baba`, so they are not its
symmetries — which is why it needs 10 arrays where a same-spin tag needs 7. The
C++ and the emitter derive that independently from the same rule and are gated
against each other.

Validated end to end on B/STO-3G (doublet, 3α/2β):

```
ucc2  -24.1892581442   == hand-written UCCSD, exactly
ucc3  -24.1892636163   T3 recovers 80.1 % of the UCCSD→FCI gap
ucc4  -24.1892649766   == in-tree FCI, all ten digits
```

**`ucc4 == FCI` on an open-shell system is the strongest single UCC gate**, and
the reason it exists at all overturned the plan's own conclusion. That plan
reasoned exactness needs `n_elec ≤ 4`, a worthwhile T4 needs ≥2 electrons of each
spin, and open shell needs `n_α ≠ n_β` — and 4 electrons with 2 of each spin is
closed. That is **true at four electrons** (triplet Be is 3α/1β, its `aabb` T4
sector is identically zero, and UCCSD already *is* FCI there — measured, so a
broken T3 or T4 would pass). But it is wrong as a generalization: B/STO-3G has
only **2 alpha virtuals for 3 alpha electrons**, so T5 is unreachable and CCSDTQ
is exact. **The orbital count enforced what the electron count could not.**

Two lessons generalize past CC. First: **an exact rational ratio is evidence of a
constant, and a constant is as likely to be a configuration default as a
coefficient bug** — `cc_damping` defaults to 0.8 and the Jacobi update is
`δ = damping·R/D`, so iteration 1 sits at exactly 80 % of MP2 on every channel,
which masqueraded as a structural defect. *Grep the knobs before theorizing about
the equations.* Second: **exactness is set by what the basis can reach, not by the
electron count alone.**

---

## 9. Factorization: deriving the intermediates

### 9.1 The FLOP win and the intermediate are the same act

Efficient CC implementations dress the residual into named intermediates
(`Wmnij`, `Wabef`, `Fae`, …) so contraction cost drops from O(n⁶) toward O(n⁵).
The conventional route hand-seeds each rank's set. The key observation is that
**the FLOP win and the intermediate are the same act of factoring**: a term
written as one n-ary contraction has a peak cost equal to the number of distinct
occupied/virtual indices it touches, re-associating it into a binary tree can
lower that peak, and the sub-contraction the tree materializes to achieve it *is*
a candidate intermediate. `t2·t3·v` drops from `o⁵v⁵` n-ary to `o³v⁴` when
`(t3·v)` is contracted first.

`ccgen` searches all binary associations of each term (≤5 factors, so exhaustive),
selects the minimum-peak-exponent tree, and identifies each internal node against
a canonical key — a match to a known operator is *reuse*, a non-match is *newly
derived*. Tree selection is made deterministic by a total-order key (peak
exponent, then a canonical fingerprint), so the derived set is a function of the
equations rather than of factor input order.

### 9.2 The rank-locality theorem

Within this optimization model — exhaustive per-term trees over diagram-generated,
canonical-Fock residuals — the derived operators obey a precise structure. Writing
`Rₙ` for the rank-n residual manifold, `Tₙ` for its highest-rank amplitude, and
`V·Tₘ` for an operator whose definition contracts the integral `V` with `Tₘ`:

1. **Rank-local generation** (structural). Every operator whose definition
   contains `Tₙ` is generated only in `Tₙ`-bearing terms — a definition containing
   `Tₙ` requires a `Tₙ` leaf, which the term must supply.
2. **Compositional separation** (structural). No operator whose definition
   contains `Tₙ` appears in a `Tₙ`-free term.
3. **Lower operators are not confined** (the main, non-obvious result). A
   `Tₙ`-bearing *term* can reuse a *lower*-rank operator, because association order
   can route it through a low-rank intermediate before touching `Tₙ`. Measured: 36
   such reuses in CCSDT triples, 64 in CCSDTQ quadruples. This refutes the natural
   conjecture that low-rank operators live only in `Tₙ`-free terms, and establishes
   that **operator composition, operator reuse, and excitation rank are three
   distinct concepts.**
4. **Cumulative across rank** (observed, CCSDT → CCSDTQ). Every operator derived
   at the lower rank is reused verbatim at the higher one; each rank adds only its
   own `V·Tₙ` family.

The implication is a **recursive intermediate library**: the same builder kernels
serve CCSD, CCSDT and CCSDTQ, and each rank extends the library by one family.

Across the hierarchy the operator count grows modestly while savings and footprint
explode, and a handful of operators always carry nearly all the benefit:

| metric (O=30, V=100) | CCSDT | CCSDTQ | CCSDTQP |
|---|---|---|---|
| distinct emittable operators | 24 | 43 | 59 |
| maximum single-operator FLOP savings | 4.1×10¹⁶ | 8.7×10²⁰ | 6.5×10²⁴ |
| largest operator footprint | 64.8 GB | 1.9×10⁵ GB | 5.8×10⁸ GB |
| maximum reuse count (one operator) | 77 | 479 | 2808 |
| operators for 99 % of savings (top-k knee) | 4 | 5 | 6 |

That concentration — fewer than 7 operators carrying 99 % of savings at every rank
— is what makes a memory budget nearly free: the long tail inlines at negligible
FLOP cost (§9.5).

### 9.3 Two dressing routes, and production was wired to the weaker one

ccgen has **two** routes producing dressed operators, and the distinction was
missed for months because they share no machinery:

| route | operators from | status |
|---|---|---|
| **recognition** | 6 hand-seeded spin-orbital fingerprints | **retired** — 52 % short on Be/STO-3G, five failed fix attempts |
| **derivation** | each term's own contraction tree | **wired to production, 2026-08-26** |

Recognition works by matching a sub-expression and **subtracting what the operator
absorbs from the remaining terms** — a subtraction only valid against the term set
it was computed for. Spin adaptation changes that term set, so the composition is
wrong in *either* order: dressing first computes the subtraction against GCC terms
that no longer exist after adaptation, and adapting first matches against spatial
terms whose operator *definitions* were derived in a basis where they were never
valid. Measured dressed Be/STO-3G CCSDTQ `E_corr` = −0.0247182895 against an exact
−0.0517746319, **52 % short**. Retired rather than fixed: five attempts each passed
their gate and made the energy worse.

**The retirement's decision stands; its stated reason does not generalize.** The
derivation route recognizes five of the same six Stanton–Gauss operators **on
spatial terms**, deriving them from contraction topology, so it never needed a
spatial fingerprint set — and it composes. The retirement's argument that a spatial
`Wmbej` "is several different operators, so deriving it is research, not porting"
is right about the antisymmetry (spatial has 13 `v` index-space patterns against 9
in GCC) but wrong in its conclusion: deduped, the two bases give 18 and 19
operators, and the derivation never needed the antisymmetry.

The derivation route was built eight days after recognition, shipped an emit
bridge that compiled, was deferred in its own commit ("CCSD dressing stays D7.3's
job"), and had **no production caller** for months. It also failed value
preservation when first measured — 23 of 66 GCC `ccsd` doubles terms, on the basis
where there is no spin adaptation to blame — via two defects, both now fixed:

- **Incomplete summed lists.** `node_to_term` recorded only the indices consumed at
  *that* tree step, while its factors are the whole subtree's leaves, so inner
  contraction indices bound to nothing and the emitted builder had no loop over
  them. Fixed by completing the summation (`used − free`) at the single upstream
  source. 20/52 → 0/50 malformed specs.
- **One name, several contractions.** `_derived_name` built names from sorted
  factor names plus a block signature, discarding slot order, so one `build_W`
  served two different contractions. Fixed by folding the contraction shape into
  the name; three properties of that shape key were each isolated by a failing
  case (slot *position* not free/summed: 21→13; positions not index *names*:
  13→6; same-tensor copies kept distinct: 6→**0**).

The route is now value-gated at ranks 2–4: **0 disagreements** on GCC and spatial
`ccsd` singles+doubles, on `ccsdt` triples (345 terms), and on `ccsdtq` quadruples
(**2536 terms**). Rank 4 is asserted separately because this codebase has twice
shown rank 3 does not predict it.

### 9.4 Operator identity: how finely should operators be distinguished?

The shape key that fixed correctness also **over-split** operators that are one
contraction up to a transpose (GCC 12→27, rank 3 26→83), collapsing the sharing
the factorizer exists to create. `symbolic_transpose` decides transpose-equivalence
**symbolically**, on the shape key rather than by comparing arrays, exact against a
numeric oracle at two fixtures × three seeds. Merging reaches the emitted C++:
**27→19 builders on `ccsd`, 254→69 at rank 4**, value-gated at 0/2536.

Two lessons carry. **Only sign-preserving symmetries may be folded** — using all
eight ERI permutations produced two false merges, because four members are odd and
hold only up to `<qp|rs> = -<pq|rs>`; this is the *same blind spot* that let the
52 % defect pass every symbolic check. And **the oracle's fixture must match the
basis**: ~30 of 48 apparent spatial misses were oracle false positives, pairs equal
only because the spin-orbital fixture antisymmetrizes `t2`.

A caution the parent measurement invites: the 1.4× → 2.1× → 3.7× merge figures are
an **operator-count** reduction, while the modelled FLOP saving is only
1.02×–1.20×. The likely win is compile time — the registry TU is `-O1`-pinned and
the dressed CCSDTQ TU is 13 MB — not speed. Threading `merge_transposes` into the
production path is scoped, measure-first, and not yet done.

### 9.5 Memory-aware and cache-local emission

The factorized emitter selects and materializes intermediates by **FLOP savings
alone**, reading neither footprint nor build-loop cache behaviour. That is
measurably suboptimal on three axes, each now addressed by a flag-gated pass:

- **Selection ignored memory.** The FLOP-savings and savings-per-byte rankings pick
  *different* top operators — the FLOP winner is 64.8 GB, the density winner
  0.02 GB (3000× smaller) at a higher flops/byte. Fixed by a per-operator footprint
  guard plus a total-budget joint selection that runs both greedy fills and takes
  the higher-savings set. Validated against an exact 0/1 knapsack (branch-and-bound
  with a fractional-relaxation bound — *not* an integer-weight DP, which zeros the
  small high-density operators): within **0.002 %** of optimal across a dense
  CCSDTQ budget sweep, so no exact solver is warranted. At CCSDTQP the footprints
  span 11 orders of magnitude and the exact solver does not terminate.
- **No feasibility guard.** The highest-savings operators are unmaterializable at
  scale — the rank-8 CCSDTQ intermediates are 194,400 GB each — yet a FLOP-only
  ranking would still select them.
- **Unshaped builder loops.** Each builder was one flat n-ary nest, so an operator
  meant to *save* FLOPs was itself computed above its factored cost, with loops
  ordered alphabetically rather than for stride. Applying the tree search one level
  down drops 10 of 24 CCSDT builders to their factored cost at scratch ~0.3× the
  operator's own footprint; a static stride metric then drives a summed-loop
  reordering that provably preserves the sum.

At a fixed budget the optimized emit beats the FLOP-only baseline on all three axes
**simultaneously**:

| at an 850 GB budget (CCSDTQ, O=30/V=100) | FLOP-only baseline | optimized |
|---|---|---|
| operators materialized | 15 | 26 (smaller) |
| FLOP savings retained | 1.40×10¹⁸ | **1.48×10¹⁸ (+5.68 %)** |
| total memory used | 850 GB | **691 GB (−19 %)** |
| builder loop stride penalty | 1.5×10¹⁶ | **2.3×10¹⁴ (−98 %)** |

The three objectives were never in tension; the baseline simply ignored two.

---

## 10. Performance, measured

### 10.1 The accessor: a 206× constant factor

Every CC kernel reads tensor elements through `operator()`. Those accessors were
defined **out-of-line in `common.cpp`** with no LTO configured, so each access was
a cross-TU call that heap-allocated one or two `std::vector<int>` and built a
`std::expected` before indexing — **two malloc/free pairs per element read, on the
innermost loop of an `o³v³` kernel**. Inlining them as flat row-major index
computations, energies bitwise-identical throughout:

| | before | after | speedup |
|---|---|---|---|
| rank-3 generated T3 residual (`bh3`) | 6.40 s | 0.031 s | **206×** |
| rank-3 hand-written T3 residual | 0.170 s | 0.0014 s | **121×** |
| rank-4 CCSDTQ per iteration (Be/STO-3G) | 38.5 s | 11.4 s | 3.4× |
| `water_rccsdt_sto3g` regression | 44.6 s | 0.39 s | **114×** |

The cost is per *access*, so it scales with call sites: **3416** in the generated
rank-3 triples residual against 186 hand-written. Counterintuitively the
generated-vs-hand ratio *widened* before it narrowed — the hand-written kernel is
*more* accessor-dominated (fewer FLOPs per access) and gained more.

**Rank 3 is not a proxy for rank 4.** Fixing only the fixed-rank accessors sped
rank 3 by 76× and left rank-4 CCSDTQ *completely unchanged*, because rank ≥ 4
kernels index exclusively through braced lists on `TensorND`/`DenseTensorView` —
23,338 accesses per residual evaluation, each copying into a fresh vector *before*
the out-of-line call. Their exclusion rested on a reading of signatures, not a
measurement. Two carried folklore figures were also retired: "~180× slower,
attributed to intermediates rebuilt inside loops" reproduced as ~37.6× with a
different cause, and loop fission — the originally hypothesized culprit — measured
at **0.62×**, i.e. faster.

### 10.2 The residual gap is a scaling defect

With the accessor fixed, a six-point ladder isolates what is genuinely structural:

| case | o | v | generated | hand-written | ratio |
|---|---|---|---|---|---|
| BH3/STO-3G | 4 | 4 | 0.0309 s | 0.00142 s | 21.8× |
| CH4/STO-3G | 5 | 4 | 0.0930 s | 0.00347 s | 26.8× |
| HF/6-31G | 5 | 6 | 0.5681 s | 0.01779 s | 31.9× |
| H2O/6-31G | 5 | 8 | 1.7232 s | 0.06316 s | 27.3× |
| BH3/6-31G | 4 | 11 | 3.3287 s | 0.09741 s | 34.2× |
| C2H4/STO-3G | 8 | 6 | 5.9509 s | 0.11875 s | **50.1×** |

**The ratio grows, with no plateau — a scaling defect, not a constant tax.** The
hand-written kernel is textbook, fitting `o^3.94 v^4.18` at 4.5 % residual. The
generated one fits `o^4.87 v^4.52` but at 21.4 %, and that residual is
**concentrated at high `v`** (the four lowest-`v` points fit to ≤5.5 %), so a
single power law does not describe it — evidence of multiple contraction regimes,
consistent with different terms wanting different orders and the emitter picking
none of them.

**The ladder is itself a methodological result.** Two earlier drafts quoted wrong
exponents. The four-point version concluded "the entire gap is in the occupied
index" — but three of those four shared `o=5`, so least squares had nothing to
separate and loaded all divergence onto `o`; its 6.5 % residual looked reassuring
*precisely because* it was overfitting a nearly-fixed variable. The five-point
correction was no better established: leave-one-out swung the `o` exponent across
−0.65..+1.12, not even holding its sign. **A power-law fit in k variables needs all
k varied independently, and the diagnostic is leave-one-out or the design-matrix
condition number — never the residual.**

### 10.3 The derivation route, end to end

The first wall-clock numbers for any dressed route, same input and same binary
configuration apart from `--dressing`:

| system | o/v | undressed | derivation-dressed | speedup |
|---|---|---|---|---|
| LiH/STO-3G | 4/8 | 5.12 s | **1.64 s** | **3.12×** |
| CH4/STO-3G | 5/4 | 104.56 s | **28.94 s** | **3.61×** |

Energies identical to all printed digits, and CH4 takes 15 steps either way — so
this is per-iteration work, not fewer iterations. Both land inside the modelled
2.0×–7.1× range, which is worth stating plainly because §10.2 gave good reason to
expect otherwise.

**Two caveats bind.** These are end-to-end solve times on two systems, one of them
off the §10.2 ladder; the two sets of numbers are **not comparable and must not be
combined into one ratio**. And two points cannot give exponents, so whether
dressing reduces the *scaling* or only the *constant* is unmeasured — precisely the
distinction §10.2 exists to make. Dressing addresses the same contraction-order
hypothesis as consuming the emitter's unused `_optimal_contraction_order`, by a
different mechanism, so the ladder should be re-run under `--dressing derived`
before that emitter change is attempted; the two fixes may overlap rather than add.

### 10.4 What the production path still costs

**Profiled 2026-08-29, and the four standing hypotheses resolved to one defect.**

First, a framing correction the profiling forced: the carried "~500× slower per
iteration" is **a ratio across a solver boundary, not a defect size.** The two
paths are different algorithms — wedge-packed against dense amplitudes, cheap
dressed intermediates against a full generated kernel per rank, **40 against 16
iterations on CH4.** It states which production path is cheaper; it does not
bound what is recoverable.

Measured, generated-against-generated:

| | share | outcome |
|---|---|---|
| the harness itself (all-rank loop, DIIS, energy) | **~1 %** | three of the four hypotheses **dead** |
| one call to the rank-3 kernel | **98.8 %** | the fourth confirmed |
| …of which `build_W_*` operator builders | **67.7 %** | **fixed — 1.76×** |

The builders were being rebuilt **once per emitted chunk**: 1112 calls for 278
distinct operators at rank 3, and **16 092 for 894 at rank 4**, the duplication
factor equalling the part count — so the waste scaled with kernel size, worst at
the production target. The emitter had recorded the rebuilds as "cheap, local, and
keeps each part self-contained", which was true for an *undressed* emit with no
operators to rebuild and was never re-examined once dressing populated the list.
Hoisting them into a single per-kernel struct gives **CH4 29.59 s → 16.81 s**, with
`E_corr` bitwise identical and the rank-4 translation unit down 12.8 → 10.5 MB.

**The largest remaining lever is that CC has no OpenMP at all** — none in the
hand-written solvers, none in the generated kernels, none emitted. A CH4 solve with
`OMP_NUM_THREADS=8` runs at 98.8 % CPU: one core busy, seven idle, while every
other hot path in the code is threaded. Amdahl on the measured split gives
**3.86× at four cores**, and both sites are reduction-free — builders write private
tensors, residual nests write disjoint slices — which is the shape that made the
DFT J/K builds bitwise thread-invariant rather than the grid reduction that caused
the historical jitter.

**Method, stated because it is the transferable part.** Two cost models built from
a census of the emitted C++ went **one for two**: contraction order was a real 3.6×,
loop fusion was a confident 32× prediction that measured at ~0 % — twice, including
after a later fix nearly doubled its Amdahl leverage. The profile cost about twenty
minutes and found a defect neither model could see, because **neither modelled work
that should not happen at all**: both priced the residual's arithmetic while two
thirds of the time was redundant operator construction outside it. The rank-3
investigation's rule — *a profile decides, not a reading* — held again.

Compile time is a real cost of its own. `generated_kernel_registry.cpp` is pinned
to `-O1` because a ~230k-line TU is super-linear to optimize; the dressed CCSDTQ TU
is 13 MB, and a full-width build on these TUs is disruptive.

---

## 11. The recurring failure mode

Five separate defects in this system converged, self-consistently, to a plausible
wrong number. None crashed; none failed to converge; each passed the gates in place
at the time. Reported together because the pattern is the transferable result:

| defect | symptom | why the gate passed |
|---|---|---|
| recognition dressing | Be `E_corr` 52 % short | structural gate on the rewrite, never a value gate |
| `SPIN_ADAPT=OFF` | `E_corr` ~4× wrong, `rms(res)`=1.9e-12 | no case pinned the flag; the one case above the backstop asserted the *hand-written* path |
| rank-4 single sector | CCSDTQ collapsed to CCSDT, 3e-6 short | the missing sector is a *converged* CCSDT answer |
| amplitude representation | −7.56e-05, converged cleanly | the tensor solver had no regression gate for its entire life |
| ERI symmetry table | 41/288 builders read the wrong block with a bogus sign | the value gate never emits C++, covers 27/142 doubles, and its fixture **antisymmetrizes `v`**, under which the invalid relation is *true* |

Three rules follow, each earned rather than assumed.

**A fixture with more symmetry than the real object cannot see a defect that abuses
symmetry.** This is the single most repeated cause here — it hid the exchange in
§6.3, produced 30 oracle false positives in §9.4, and let the 41/288 ERI defect
pass every symbolic check. Measured directly: 0/288 builders disagree on an
antisymmetrized fixture, 41/288 on a spatial one. A fixture must match the basis
whenever a check asserts a property *of the tensors* or compares against an
independent oracle.

**A structural gate cannot stand in for a value gate.** `tree_preserves_term`
checks leaf and index bookkeeping; `test_budgeted_rewrite_is_exact` compares a
factor `Counter`, blind to index order by construction. Both return `True` on a
malformed node, because they ask the question at term level while the defect lives
at node level. 47 green structural tests coexisted with 23/66 terms numerically
wrong.

**Comments document an invariant; only a test enforces one.** Two warning comments
already recorded that folding all eight ERI permutations caused a 52 % energy
defect that "passed every symbolic check". Neither prevented a *third* module from
carrying the bad set. The guard written afterwards found a fourth copy.

Three practical corollaries, each of which cost real time: verify a gate is
**falsifiable** before trusting a pass (mutation-test it — two vacuous gates in the
UCC effort were caught this way and not by review); **check the case runs the path
you think** (`kernels=hand-optimized` in a log means the hand-written path ran);
and when one component looks broken while a sibling sharing its code looks fine,
**compare them directly before investigating either** — a byte-level `cmp` of the
CCSDTQ and CCSDT bundles' shared manifolds excluded "the rank-3 kernel is wrong" in
one step, after an entire investigation had assumed it.

---

## 12. Limitations and open work

- **Symbolic cost models.** FLOP degree, memory footprint and the stride metric are
  computed from index-space sizes. §10.3 is the first wall-clock validation of any
  of it, on two points.
- **The scaling ladder was never re-run under dressing, and is now unlikely to be
  worth it.** The question it would settle — does dressing flatten the exponents or
  shift them — was overtaken: profiling found the dominant cost was neither
  contraction order nor traffic but redundant operator construction (§10.4). The
  emitter's discarded `_optimal_contraction_order` targets terms `--dressing
  derived` already eliminates, so consuming it is **probably redundant**; re-check
  against a profile rather than the ladder.
- **The memory-bound hypothesis is refuted at reachable sizes, not merely
  untested.** Reducing the residual from 806 loop nests to 15 — a 54× cut in
  traversals of the `o³v³` result — changed runtime by 0–3 % at three sizes
  spanning 7× in `t3`, and again after a later fix nearly doubled the residual's
  share of runtime. Consecutive nests over the same result hit the same cache
  lines, so the traffic being modelled was already served from cache. It remains
  untested only *above* L2, which needs cc-pVDZ-class systems.
- **Rank 4 has no point on the ladder**, and carries an `-O1` registry pin rank 3
  does not. §10.1 already demonstrated rank 3 is not a proxy for rank 4.
- **Dressed UCC is unbuilt.** The emitter rejects spin-blocked manifold names and
  recognition finds *zero* operators on spin-tagged factors. The obvious tag-blind
  fix recovers 152 reuse sites and is **measured and unsound** — it collapses 12
  distinct spin-tagged contractions onto one `Wmbej`, which is the 52 % defect one
  level down.
- **`merge_transposes` is not threaded** into production, so `derived` emits 59
  un-merged builders on spatial `ccsd` rather than 31.
- **Rank ≥ 5 has no numeric oracle**, and rank-4 cumulativity (§9.2 part 4) is
  two-rank evidence; a proof for arbitrary rank is open.
- **`wick`-engine coverage for UCC.** Every UCC gate ran through `diagram`; the two
  are documented residual-equal but unpinned there.
- **Cross-operator sharing is out of scope.** Each operator's footprint and loops
  are shaped independently; optimal cross-term contraction scheduling is NP-hard in
  general.
- **Parallel generation is not equivalence-safe.** `parallel_workers > 1` produces a
  genuinely different equation set via two order-dependent defects (a
  non-spawn-safe C extension, and a partition-local pre-canonical merge). The
  default serial path is deterministic and correct; the parallel regression carries
  its root cause inline as an `expectedFailure`.

---

## 13. Summary

`ccgen` demonstrates that arbitrary-order coupled-cluster kernels can be derived
and emitted automatically, and — as of 2026-08-26 — run in production. Four results
stand beyond the generation itself. A diagrammatic derivation with a *solve-free*
topological weight generates at ranks no reference program could tabulate, ~200×
faster than term algebra. Contraction-path factorization **derives** the reused
intermediates rather than requiring them to be hand-seeded, and those intermediates
obey a rank-locality structure making the library cumulative across the CC
hierarchy. The emitter can jointly optimize FLOP savings, memory footprint and
cache locality, beating a FLOP-only baseline on all three at once. And the derived
route, once wired, is worth a measured 3.12×/3.61× at energies identical to the
undressed baseline.

The fifth result is methodological, and this system produced it five times over:
**a gate whose fixture carries more symmetry than the real object cannot see a
defect that abuses symmetry**, and a structural check that passes is not evidence
that a value is preserved. Every correctness claim in this report is anchored to an
external oracle — PySCF, FCI, or a lower rank already pinned to one — because the
alternatives were tried and each of them held while the answer was wrong.

---

## Appendix: reproducibility

- **Generation:** `generate_cc_equations(method, engine="diagram"|"wick", canonical_fock=True)`.
- **Spin adaptation:** `--spin-adapt` → `print_cpp_planck(spin_adapt=...)`;
  `PLANCK_CC_SPIN_ADAPT` defaults ON.
- **Dressing:** `--dressing {none,recognized,derived}`; `PLANCK_CC_DRESSING` on the
  CMake side, active only with `PLANCK_CC_DRESS_OPERATORS=ON`.
- **Factorization and memory passes:** `python/ccgen/optimization/factorize.py`
  (`factorize_equations(..., merge_transposes=, memory_budget_bytes=, factor_builder_bodies=)`).
- **Diagnostics:** `PLANCK_CC_T3_TIME=N` (isolated triples-residual timer, inert
  when unset); `PLANCK_CC_FIXTURE_DIR` (inject amplitudes, evaluate residuals once,
  dump per-rank tensors, exit — no solver, no DIIS).
- **Build:** `make -j4`; the generated TUs are large enough that a full-width build
  is disruptive. Always set an explicit `CMAKE_BUILD_TYPE` — an empty one drops
  `-DNDEBUG`, re-enables the CC bounds asserts, and effectively reverts §10.1.
- **Tests:** `python/ccgen/tests/` — `test_factorize.py`,
  `test_factorize_value_preservation.py` (the referee), `test_operator_identity.py`,
  `test_emitted_builder_matches_spec.py`, `test_eri_symmetry_tables.py`,
  `test_diagram.py`, `test_spin.py`, `test_reference_vs_pyscf.py` (PySCF lives in
  `tests/pyscf/.venv`, not the system Python).

### Source documents synthesized

| doc | contributes |
|---|---|
| `CCGEN_GENERATION_AND_VALIDATION.md` | §3, §4 |
| `CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md` | §3 |
| `CCGEN_KERNEL_WIRING.md` | §5.2 |
| `CCGEN_SPIN_ADAPTATION.md`, `CCGEN_SPIN_ADAPTER_CONTRACT.md` | §6 |
| `CCGEN_SPIN_ADAPT_DEFAULT.md` | §5.2, §11 |
| `CCGEN_CCSDTQ_MULTISECTOR.md` | §7 |
| `CCGEN_UNRESTRICTED_CC.md`, `CCGEN_UCC_ERI_ANTISYMMETRY.md`, `CCGEN_UCC_NUMERIC_VALIDATION.md`, `CCGEN_GCC_TO_UCC_BRIDGE.md` | §6.4, §8 |
| `CCGEN_HIGHER_OPERATOR_REUSE.md` | §9.1, §9.2 |
| `CCGEN_TWO_DRESSING_ROUTES.md`, `CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`, `CCGEN_DRESSING_COST.md`, `CCGEN_DRESSED_KERNEL_PIPELINE.md` | §9.3 |
| `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`, `CCGEN_MERGE_TRANSPOSES.md` | §9.4 |
| `CCGEN_INTERMEDIATE_MEMORY_LOCALITY_SCOPE.md` | §9.5 |
| `CCGEN_TENSOR_ACCESSOR.md`, `CCGEN_KERNEL_PERFORMANCE.md` | §10.1 |
| `CCGEN_KERNEL_SCALING_SCOPE.md` | §10.2 |
| `CCGEN_WIRING_THE_DERIVATION_ROUTE.md` | §2, §9.3, §10.3, §11 |
| `CCGEN_RANK3_KERNEL_AND_SOLVER.md`, `CCGEN_ARBITRARY_HARNESS_COST.md` | §10.4, §11 |
| `CCGEN_TEACHING_GUIDE.md` | §2 |

### Key references

- Kállay & Surján, *J. Chem. Phys.* **113**, 1359 (2000); **115**, 2945 (2001) — diagram integer strings.
- Crawford & Schaefer, *Rev. Comput. Chem.* **14**, 33 (2000) — oriented-loop sign.
- Stanton & Gauss, *J. Chem. Phys.* **94**, 4334 (1991) — dressed CCSD intermediates.
