# How does a dressed CC residual become a running C++ kernel?

Answers one question: **what are the stages between a `ccgen` residual and a dressed kernel executing
inside Planck, what does each stage have to get right, and how is each one checked?**

Short answer: five stages — recognize, adapt, emit, build, run — and the interesting content is that
each stage has a *different failure mode*, invisible to the stage before it. Four real defects were
found here, one per new kind of check: a declared-vs-built layout mismatch, a flag interaction, a
translation unit that had never compiled, and a kernel that had no caller at all.

> **The route this doc describes has been retired.** Stages 1-4 hold and are worth reading -- the
> defects they found are real and the mechanics still describe how the emitter works. But stage 5
> never passes: dressing and spin adaptation **do not compose**, in either order, and the
> spin-adapted dressed kernels are wrong by ~52 % of the correlation energy. The decision, the
> measurements behind it, and what was kept are in `CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`.
>
> Read this document for *how the dressed emit pipeline works*, not as a live plan.

---

## The pipeline

```
ccgen residual            diagram engine, canonical Fock
   │
   ├─ recognize           dressed operators: Wmnij / Wabef / Wmbej / tau / tau_c
   │                      + an IntermediateSpec per operator
   ├─ adapt               spin-adapt the residual AND the specs (RCC spatial)
   ├─ emit                build_<op>_<method>() + the residual kernel
   ├─ build               -DPLANCK_CC_DRESS_OPERATORS=ON → generator → registry / tensor_backend
   └─ run                 == undressed energy and iteration count
```

Default **OFF**, so the default build is byte-identical to an undressed one.

## Stage 1 — recognize

`_dress_operator_equations` matches the seeded operator family against the residual and returns
`(dressed_equations, ordered_specs)`. The order is dependency-first: the `tau`/`tau_c`
pseudo-amplitude specs precede the operators that reference them.

Two facts about what recognition actually yields:

**Only three of six seeded operators appear.** Under `canonical_fock=True` — the only mode Planck
feeds CC — the `f_ov`-bearing definition terms of `Fme`/`Fae`/`Fmi` are Brillouin-zero and drop, so
`Fme` collapses to its `t1*oovv` piece and `Fae`/`Fmi` lose their corrections. The dressed equation
references exactly `{Wmnij, Wabef, Wmbej, tau, tau_c}`. Three seeded operators are inert, which is
intended, not a gap — and is asserted, because three silently-unused operators otherwise look like a
bug.

**Usage grows sharply with rank**, which is what makes dressing worth doing at higher rank:
rank 2 gives `Wmnij(1) Wabef(1) Wmbej(5)`; rank 3 gives `(13) (13) (32)`; rank 4 gives `(79) (79) (240)`.

Recognition is also where the time goes — see `CCGEN_DRESSING_COST.md`.

## Stage 2 — adapt, and the layout that must not drift

Spin adaptation converts the spin-orbital residual to spatial terms. The subtlety is that **the specs
must be adapted too**, because adaptation changes their declared layout:

| spec | GCC sig | adapted sig |
|---|---|---|
| `tau` | `vvoo` | **`oovv`** |
| `tau_c` | `vvoo` | **`oovv`** |
| `Wmbej` | `ovvo` | **`oovv`** |
| `Wmnij` | `oooo` | `oooo` (slots reordered) |
| `Wabef` | `vvvv` | `vvvv` (slots reordered) |

Emitting GCC specs beside a spin-adapted residual is a **live miscompile**: the residual references
spatially-adapted `Wmbej` while `build_Wmbej` builds the GCC slot order. Three of five layouts
disagree, so this is not a corner case.

`adapt_intermediate_spec` handles it, and `validate_intermediate_specs` is wired in as an *assertion*
at that point rather than only as a test — this is the one place such a mismatch is introduced, so the
check is load-bearing there. It verifies the sig matches the slot spaces **positionally** (`oovv` vs
`vvoo` has the same characters in the wrong order and would emit transposed dimensions), that no slot
repeats, and that every definition term carries the spec's own free indices **in order** (a permuted
term writes a transpose into the buffer).

The adapter itself has a separate contract that this stage depends on — `CCGEN_SPIN_ADAPTER_CONTRACT.md`.

## Stage 3 — emit, and two symbol-scope problems

The emitter writes one `build_<name>_<method>` per spec plus the residual kernel. Two things it has to
get right, both of which it originally did not:

**A builder that references a sibling must bind it.** `build_Wmnij` and `build_Wabef` reference
`tau(...)`. The `sibling_names` set only made such a factor *render* as a bare identifier — nothing
declared it, so the emitted C++ used `tau` with no `tau` in scope and **the dressed TU had never been
valid C++**. Fixed by binding referenced siblings the way the residual kernel already did
(`const auto tau = build_tau_ccsd(...)`). Correct by the existing dependency order, since a sibling's
builder is emitted above its consumer.

**Builder names must be method-qualified.** The kernel registry `#include`s several generated TUs into
*one* translation unit, so builder symbols from different methods share a scope. Whether unsuffixed
names collided depended on configuration:

- non-arbitrary: `RCCSDAmplitudes` vs `RCCSDTAmplitudes` made them **overloads** — co-inclusion compiled
- `force_arbitrary`: both take `ArbitraryOrderRCCAmplitudes`, identical signatures, so they were
  **redefinitions** — 5 errors, one per builder

The failing case is the mode the registry uses, so the hazard was conditional on exactly the target
configuration and invisible in the other mode. `_builder_symbol(method, name)` resolves it, with all
four emission sites (the definition and three call sites) routed through the one helper so a
definition and its calls cannot drift apart.

Chosen over restricting dressing to a single rank: the collision is a property of the **naming
scheme**, not of how many ranks are enabled, so a restriction would leave the trap armed for whoever
enabled a second dressed rank.

**Emitted builder order is a valid topological sort of the TU** — `tau`, `tau_c`, `Wmnij`, `Wabef`,
`Wmbej`, zero forward references — asserted on the emitted text rather than the spec list, so an
emit-layer reordering is caught too.

## Stage 4 — build, and the flag interactions

`print_cpp_planck` has four interacting flags. Dressing feeds the same single exit path as every other
flag; a second `emit_planck_translation_unit` call site would fork the composition and force UCC to be
wired twice.

Two exclusions, and the difference between them is deliberate:

- `dress_operators` × `factorize_tau` **raises**. Both materialize `tau`. This one is instructive: it
  was documented as "already mutually exclusive", but the exclusion was *unreachability* — an early
  return meant `factorize_tau` was silently ignored under dressing. Removing that early return
  **activated** it. Silent precedence is what disguised the hazard, so the fix raises rather than
  picking a winner.
- `dress_operators` × `include_intermediates` **forces CSE off**, mirroring the `spin_adapt`
  precedent, so a caller passing it does not get a failed build. Both materialize through the same
  `intermediates` channel.

Note on CSE: index-space validity is *not* the live blocker for re-enabling it. `detect_intermediates`
yields 7 specs on raw GCC and 16 on spin-adapted equations, and all 23 pass
`validate_intermediate_specs`. What stands is compile time (~1544 `build_W_*`, ~28 min at `-O3`) and
the absence of a numeric gate — not a known index defect.

Which TU the build reaches matters:

| TU | consumer | amplitude type | co-included? |
|---|---|---|---|
| `ccsd_planck_generated.cpp` | **nothing** | — | — |
| `ccsdt_planck_generated.cpp` | `tensor_backend.cpp`, unconditional | `RCCSDTAmplitudes` | no |
| `ccsdt_arbitrary_…` | registry, opt-in | `ArbitraryOrderRCCAmplitudes` | yes |
| `ccsdtq` / `cc5` / `cc6` | registry, by `MAXORDER` | `ArbitraryOrderRCCAmplitudes` | yes |

Rank 2 is generated but compiled into nothing — which is why the validated anchor is rank 3, and why
earlier tau work could never add an energy gate.

## Stage 5 — run

Rank-3 RCCSDT/STO-3G, dressed build vs undressed reference:

| case | E_corr (both) | iterations |
|---|---|---|
| `h2` | −0.0205682660 | 12 vs 12 |
| `lih` | −0.0203779358 | 16 vs 16 |
| `bh3` | −0.0533629199 | 26 vs 26 |

**Iteration count is part of the gate, not decoration.** Equal energy shows the two kernels share a
fixed point; equal iteration count shows they take the same *trajectory* there — which is what
"dressing is a pure refactorization of the residual" actually claims.

`water_rccsdt_sto3g` is excluded, explicitly: it reaches the determinant backstop
(`RCCSDT[DET-BACKSTOP]`) in *both* builds, so it never exercises the generated kernel. It converges to
the same energy in 26 iterations dressed vs 54 undressed — a property of the backstop path, not of
dressing. The exclusion is self-verifying: the gate fails if water stops showing the backstop marker,
since it would then belong in the tested set.

**Why this needs two builds.** A dressed build's output is indistinguishable from an undressed one —
same energy, same iteration count, no backend marker — so no single-binary regression case can detect
whether dressing was enabled. `dressed_kernel_equivalence_rccsdt` therefore compares two build
directories, skips when the opt-in build is absent, and fails if either build is misconfigured (two
undressed builds would otherwise agree vacuously).

## The stage-by-stage lesson

Each new *kind* of check found a defect the previous kinds could not see:

| check | found |
|---|---|
| symbolic algebra | nothing here (it was exact) |
| spec-metadata validity | the GCC-vs-adapted layout mismatch |
| flag-matrix byte identity | `factorize_tau` activating under dressing |
| **does it compile** | the TU had never been valid C++ |
| does it run | (clean — but only reachable once the above were fixed) |

The compile check was listed as a gate and skipped; it found the oldest defect of the five. Run the
gate you wrote down.

---

Status lives in `vault/Status/Completion.md`. Gates: `test_emit_flag_matrix.py`,
`test_dressed_tu_coinclusion.py`, `test_intermediate_validity.py`, `test_dress_per_operator.py`,
`test_residual_symmetry.py`, `tests/dressed_kernel_equivalence.py`.
