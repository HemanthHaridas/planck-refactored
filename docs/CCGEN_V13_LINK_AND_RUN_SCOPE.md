# V1.3 remainder — link and run a dressed generated kernel

> **PARTLY LANDED — still live.** V1.3.0/.1/.2/.3/.4 are done: the dressed kernel generates from
> the build, compiles, links, runs matching the undressed energy and iteration count, and its
> builders are method-suffixed so two dressed TUs can share the registry's translation unit.
> **V1.3.5** (regression-pin the dressed config) is the only step left, and its section below is
> the plan of record.
>
> For overall state see [`CCGEN_DRESSED_KERNEL_COMPLETION.md`](CCGEN_DRESSED_KERNEL_COMPLETION.md).

Scopes the last open piece of V1. V1.1 validated the algebra and metadata; V1.2 made the
dressed+adapted composition reachable from `print_cpp_planck`; V1.3's emit half fixed the
sibling binding so the TU compiles. **Nothing has linked or executed it.**

**Probed before scoping.** The probe changed the shape of this step twice — one hypothesis of
mine was wrong, and the real blocker is narrower and more specific than "wire it up".

---

## What the probe established

### Finding 1 — dressing is not reachable from the build at all

`grep -n "dress" python/generate_planck_cc_kernels.py` returns **nothing**. The build-time
generator has no `--dress-operators` flag; the only CLI exposure is in
`python/generate_ccsdt_cpp.py:232`, a different script that does not feed the registry.

So dressed kernels have never entered a Planck build. `CMakeLists.txt` has the option pattern
ready (`PLANCK_CC_SPIN_ADAPT`, `PLANCK_CC_INCLUDE_INTERMEDIATES`, … each appending a CLI
flag), but no `PLANCK_CC_DRESS_OPERATORS`. That plumbing is the bulk of the work, and it is
mechanical.

### Finding 2 — the symbol collision is real, but only under `force_arbitrary`

`generated_kernel_registry.cpp` `#include`s each generated TU into **one** translation unit,
and already documents a collision hazard: the shape-named CSE builders (`build_W_oo_3`, …)
carry no method suffix, which is why the rank-3 arbitrary companion is emitted *without*
intermediates.

Dressed builders have the same unsuffixed names (`build_tau`, `build_tau_c`, `build_Wmnij`,
`build_Wabef`, `build_Wmbej`) in every method. **I expected co-inclusion to fail immediately.
It does not** — measured, `ccsd` + `ccsdt` co-include and object-compile cleanly (`rc=0`,
0 redefinitions), because the builders differ in their amplitude parameter
(`RCCSDAmplitudes` vs `RCCSDTAmplitudes`) and are therefore **overloads, not redefinitions**.

But that reprieve vanishes in the mode the registry actually uses. Under `force_arbitrary`
every method's builders take `ArbitraryOrderRCCAmplitudes` / `ArbitraryOrderDenominatorCache`
— **identical signatures**. Measured on the same co-include:

```
rc=1, 5 redefinitions:  build_tau, build_tau_c, build_Wmnij, build_Wabef, build_Wmbej
```

**So the collision is conditional on exactly the configuration this work targets**, and it is
invisible in the non-arbitrary mode. A scope written from the non-arbitrary probe alone would
have missed it and produced a build that breaks the moment a second dressed rank is enabled.

---

## The shape of the step

Two independent pieces, and only the second is interesting:

1. **Plumbing** (~S): `--dress-operators` on the build generator, `PLANCK_CC_DRESS_OPERATORS`
   in CMake, registry inclusion. Follows the existing option pattern exactly.
2. **The single-dressed-rank constraint** (~S–M): the registry cannot co-include two dressed
   arbitrary-order TUs. Either suffix the builder names per method, or restrict dressing to
   one rank and enforce it. **Decide deliberately** — see V1.3.2.

---

## Steps

### V1.3.0 — pin the collision as a test before touching the build (~S, do first)

Encode both measured facts as unit tests: the non-arbitrary co-include compiles, and the
arbitrary-order co-include fails with 5 redefinitions on exactly those five names.

*Gate:* both pass on the current tree, i.e. they record reality.

**Why first:** V1.3.2's choice hinges on this asymmetry, and it is the kind of fact that is
easy to misremember later ("didn't we check that co-including works?" — yes, in the mode that
doesn't matter). Pinning it makes the constraint durable and makes V1.3.2's fix verifiable.

### V1.3.1 — expose `--dress-operators` to the build generator (~S, mechanical)

Add the flag to `python/generate_planck_cc_kernels.py`, threading it into its
`print_cpp_planck` call, and `PLANCK_CC_DRESS_OPERATORS` (default **OFF**) to `CMakeLists.txt`
appending it — mirroring `PLANCK_CC_SPIN_ADAPT` line-for-line.

Keep the mutual exclusions V1.2.4 established: `dress_operators` × `factorize_tau` raises, and
dressing forces CSE off. The CMake layer should fail configuration on a contradictory
combination rather than let Python raise mid-build, since a CMake-time error names the option
the user set.

*Gate:* `-DPLANCK_CC_DRESS_OPERATORS=OFF` regenerates byte-identical kernels (default build
must not move); `=ON` emits TUs containing the five `build_<op>` functions.

### V1.3.2 — **DECIDED: route (b), method-suffixed builders — LANDED**

`_builder_symbol(method, name)` → `build_<name>_<method>`, with all four emission sites (the
definition plus three call sites) routed through the one helper.

**Measured on the configuration that failed:** two dressed TUs co-included with
`force_arbitrary=True` object-compile at `rc=0`, 0 redefinitions — where they produced 5.

**Route (b) over (a), overruling this document's own recommendation.** It recommended (a)
(restrict dressing to one rank and enforce it) as "cheapest and honest". Rejected for the same
reason V1.1e.2 chose route (b): the collision is a property of the **naming scheme**, not of how
many ranks are enabled, so a scope restriction leaves the trap armed for whoever enables a second
dressed rank. Fix the mechanism, not the callers.

Two consequences worth noting:

- **`factorize_tau`'s baseline moved too** (37413 → 37433), deliberately. Tau's builder is
  suffixed like every other, so there is **one naming rule with no exceptions** rather than a
  special case for the tau-only path.
- **The rank-4 registry path is no longer blocked.** With the cost objection gone post-D1 and the
  naming collision now gone, dressing rank 4 is available — the anchor stays rank 3 only because
  that is where the validated end-to-end run is.

### V1.3.2 — original scoping (retained for the rejected alternative)

Two options; they are not equivalent and the choice should be explicit rather than discovered
by a build failure.

- **(a) Restrict dressing to one rank, and enforce it.** Cheapest and honest. `MAXORDER` picks
  the dressed rank; every other rank generates undressed. A CMake-time error if the
  configuration would dress two ranks. Risk: silently limits the feature, so the error message
  must say *why*.
- **(b) Suffix the dressed builder names per method** (`build_tau_ccsd`, …). Removes the
  constraint permanently and matches how `make_generated_<method>_kernels` already
  disambiguates. Costs an emitter change plus a new naming convention, and the residual's
  factor resolution must follow the same suffix — the `_map_factor` / `intermediate_names`
  path already resolves by name, so this is where a mismatch would silently reference the
  wrong method's builder.

**Recommendation: (a) now, (b) when a second dressed rank is actually wanted.** V1's goal is
one dressed kernel that runs; (b) is speculative generality until then, and it touches the
naming path that V1.1c/U1.1 also key off. But **record the choice** — (a) is a real
limitation, not a non-issue, and the enforcement is what keeps it from becoming a silent
miscompile.

*Gate:* the configuration that would collide is rejected at CMake time with a message naming
`PLANCK_CC_DRESS_OPERATORS`; V1.3.0's collision test still documents why.

### V1.3.3 — link (~S)

Build the target with dressing on and confirm it **links**, not just compiles. This is the
first step beyond `-fsyntax-only`, and the first that can catch an undefined symbol — a
builder declared and called but never emitted would pass every check so far.

*Gate:* `hartree-fock` links with `PLANCK_CC_DRESS_OPERATORS=ON`; no undefined `build_*`.

### V1.3.4 — run, and compare against the undressed kernel (~M, the actual point)

Execute a CC calculation with the dressed kernel and require the **same energy** as the
undressed build.

**Anchor: rank 3 (`ccsdt` via `tensor_backend.cpp`), settled by measurement** — see the two
subsections below. Not rank 2 (`ccsd_planck_generated.cpp` has no consumer, so there is nothing
to run), and not `be_rccsdtq_sto3g` — originally because rank-4 dressing cost hours, now
because the rank-4 path is **co-included in the registry** and would hit the 5-redefinition
collision that V1.3.2 has yet to resolve. Post-D1 the cost objection is gone (rank 4 = 61.6 s);
the structural one stands.

Existing rank-3 regression cases to anchor against: `h2_rccsdt_sto3g`, `lih_rccsdt_sto3g`,
`water_rccsdt_sto3g`. Pick the cheapest that exercises the dressed operators.

**Rank-3 recognition confirmed** — the check that dressing is not vacuous there. Measured
(294 s total, generation + dressing):

```
specs: tau(1), tau_c(1), Wmnij(13), Wabef(13), Wmbej(32)
manifolds: energy 2, singles 9, doubles 47, triples 330
```

Usage is *higher* than rank 2 (`Wmnij` 13 vs 1, `Wabef` 13 vs 1, `Wmbej` 32 vs 5), so the
dressed rank-3 kernel exercises all five builders substantially — V1.3.4 cannot pass vacuously
on an unrecognized operator set.

*Post-D1 the same run takes **9.1 s**, with identical specs and manifold sizes* — so the
recognition figures above are unchanged and V1.3.1's help string no longer needs a cost warning
at rank 3.

> **SUPERSEDED by D1 (`b25b896`).** The cost below was a single loop-invariant rebuild inside
> the hypothesis search, now hoisted: rank-4 dressing is **61.6 s** (was >25 min, abandoned) and
> rank-3 is **9.1 s** (was 293.7 s). See `CCGEN_DRESSING_SUPERLINEAR_SCOPE.md`.
>
> **What this changes for V1.3:** the cost argument against a rank-4 anchor is gone. The
> *structural* arguments still stand and still point to rank 3 — `ccsdt_planck_generated.cpp` is
> a single non-co-included TU with a method-specific amplitude type, so the 5-redefinition
> collision cannot arise for it, while the rank-4 registry path is co-included and would hit it.
> So the anchor stays rank 3, but now **by choice rather than by necessity**, and dressing at
> rank 4 becomes a real option once V1.3.2 is decided.
>
> Also gone: the "dressing lands in the build's critical path" concern. At 61.6 s for rank 4 it
> is a normal build step, not an apparently-hung one, which softens what V1.3.1's help string
> needs to warn about.

**Measured before D1 — retained for the record: rank 4 was not viable as the anchor, and the
cost was in *recognition*, not generation.** Profiled per manifold:

| step | terms | time |
|---|---|---|
| generation (diagram engine, whole `ccsdtq`) | 3172 | **3.5 s** |
| `assemble_dressed_equation` energy | 2 | 0.0 s |
| … singles | 12 | 0.2 s |
| … doubles | 74 | **16.5 s** |
| … triples | 412 | **307.6 s** |
| … quadruples | 2672 | abandoned (>25 min, killed) |

The diagram engine is **not** the problem — 3.5 s for all four manifolds. All the cost is
`_dress_operator_equations`, whose per-manifold time scales **super-linearly in term count**
(5.6× the terms from doubles→triples costs 19× the time), so quadruples' 2672 terms extrapolate
to hours. The run was killed rather than waited out.

**Consequences as assessed at the time — both since resolved by D1:**

- V1.3.4 must use a rank-2/3 anchor. *Still rank 3, but now for the structural reason (single
  non-co-included TU), not the cost one.*
- V1.3.1's option must stay off the unconditional build path. *Default OFF is still right, but
  61.6 s at rank 4 is an ordinary build cost, not a hazard.*

The "recognition-performance fix is out of V1.3's scope" note was correct to defer it and wrong
about its size: it turned out to be **one hoisted loop invariant**, not a subgraph-matching
rewrite. Investigated and fixed separately in `CCGEN_DRESSING_SUPERLINEAR_SCOPE.md` (D0–D2)
rather than inside V1.3.

### Which TU to target — measured, and it resolves the collision too

`ccsd_planck_generated.cpp` **has no consumer**: nothing in `src/` includes it. Only rank 3 and
rank ≥ 4 are wired, by two different mechanisms:

| TU | included by | amplitude type | co-included with others? |
|---|---|---|---|
| `ccsd_planck_generated.cpp` | **nothing** | — | — |
| `ccsdt_planck_generated.cpp` | `tensor_backend.cpp:17`, unconditional | `RCCSDTAmplitudes` | **no** — its own TU |
| `ccsdt_arbitrary_planck_generated.cpp` | registry, `-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON` | `ArbitraryOrderRCCAmplitudes` | yes |
| `ccsdtq` / `cc5` / `cc6` | registry, `#if PLANCK_CC_MAXORDER >= N` | `ArbitraryOrderRCCAmplitudes` | yes |

**So `ccsdt_planck_generated.cpp` via `tensor_backend.cpp` is the V1.3 target.** It is a single
non-co-included TU with a method-specific amplitude type, which means:

- **The V1.3.2 collision does not arise for it.** The 5-redefinition failure needs two dressed
  TUs sharing `ArbitraryOrderRCCAmplitudes` in one translation unit; this path has neither.
  V1.3.2 still needs deciding before the *registry* path is dressed, but it stops blocking
  V1.3.3–V1.3.5.
- Dressing cost is the rank-3 figure (~5 min, dominated by triples), not rank 4's hours.

That also explains why the earlier tau work could not add an energy gate: `ccsd_...generated.cpp`
is generated but never compiled into a binary, so there was nothing to run. Same constraint
applies here, and it is why the anchor is rank 3 rather than the rank-2 case every V1
measurement used.

*Gate:* dressed vs undressed energy agreement at the anchor's own tolerance (1e-7 for
`be_rccsdtq_sto3g`; tighter if a smaller case is used), plus matching iteration count — a
kernel that reaches the same fixed point in a different number of iterations has a different
residual, even though the energy agrees.

**This is what the whole V1 chain has been building toward.** Every gate so far is symbolic,
metadata, or numeric-in-Python; none proves the emitted C++ computes the intended residual.
Note also the honest limit of the Python numeric gates: they validated
`adapt(expand(dressed)) == adapt(raw)` on random tensors, which shares no code with the
emitted kernel.

### V1.3.5 — regression-pin it (~S)

Add the dressed configuration to `tests/regression_cases.json` so it does not rot, mirroring
`be_rccsdtq_sto3g`: same `extended` + `generated-kernel` tags, a `skip_if_contains` guard so
builds without the option skip rather than fail, and the `metric_lt_metric` sanity check that
the CC energy is below the RHF reference.

**Gate on the CC total energy, matching the existing case.** An earlier draft of this scope
said to prefer correlation energy on the grounds that dressing cannot move the SCF and total
energy dilutes the signal. That reasoning is sound in isolation but wrong here: the committed
runner metric is `rccsdtq_total_energy`, and inventing a parallel correlation-energy metric
just for this case adds a second convention for no gain. The existing case already pins
`rhf_total_energy` separately at 1e-9, which is what isolates the SCF part.

*Gate:* the case runs in the standard suite and fails if the dressed energy drifts.

---

## Sequencing

```
V1.3.0 (pin the collision asymmetry)   ~S    ← the fact V1.3.2 hinges on
   └→ V1.3.1 (--dress-operators + CMake option)  ~S   mechanical
        └→ V1.3.2 (single-dressed-rank decision)  ~S-M ← decide, don't discover
             └→ V1.3.3 (links)                    ~S   ← first undefined-symbol check
                  └→ V1.3.4 (runs; E_corr matches undressed)  ~M  ← the point
                       └→ V1.3.5 (regression-pinned)          ~S
```

---

## What this reuses

| Reused | From |
|---|---|
| CMake option → CLI flag pattern | `PLANCK_CC_SPIN_ADAPT`, `PLANCK_CC_INCLUDE_INTERMEDIATES` |
| Per-rank guarded registry inclusion | `generated_kernel_registry.cpp`'s `#if PLANCK_CC_MAXORDER` |
| Mutual exclusions | V1.2.4 (raise on dress×tau; force CSE off) |
| Compile harness (`-fsyntax-only`, Eigen path) | `test_emit_flag_matrix`'s `_syntax_check` |
| Be CC energy anchor | the existing CC regression cases |
| Unsuffixed-builder collision precedent | the registry's own `--arbitrary-lower-ranks` comment |

**Net new:** one CLI flag, one CMake option, one enforcement check, and the link/run gates.

---

## What NOT to do

- **Do not assume co-inclusion is safe because the non-arbitrary probe passed.** It passes only
  because differing amplitude types make the builders overloads. Under `force_arbitrary` —
  the mode that matters — the same co-include produces 5 redefinitions.
- **Do not enable dressing by default.** Default OFF, like `PLANCK_CC_SPIN_ADAPT`, until
  V1.3.4 passes. The default build must stay byte-identical.
- **Do not treat "it compiles" as "it works".** Compiling is `-fsyntax-only`; linking catches
  undefined symbols; only running catches a wrong residual. The V1 chain has three defects on
  record that each passed every prior gate.
- **Do not skip V1.3.3 to get to the run.** An undefined-symbol failure is far cheaper to
  diagnose at link than as a mysterious runtime result.
- **Do not gate the run on total energy.** Dressing cannot change the SCF; gate the
  correlation energy so the signal is not diluted.
- **Do not silently limit dressing to one rank.** If V1.3.2 takes option (a), the restriction
  must be an explicit CMake-time error naming the option — an unenforced restriction is how the
  registry's existing builder-collision hazard stayed latent.

---

## Honest status and risk

The plumbing is mechanical and low-risk. The two real risks:

1. **The collision (V1.3.2)** — measured, understood, and conditional on the target
   configuration. Mitigated by pinning it first.
2. **The run (V1.3.4)** — genuinely unvalidated. Nothing to date proves the emitted C++
   computes the intended residual, and the V1 chain's track record is that each new *kind* of
   gate found a defect the previous kinds could not see: V1.2.2 (layout), V1.2.4 (flags),
   V1.3-emit (never compiled). A run gate is a new kind. Budget for it finding something.

---

See `CCGEN_DRESS_ADAPT_COMPOSITION_SCOPE.md` (V1.3's original framing, corrected in place),
`CCGEN_V12_EMIT_WIRING_SCOPE.md` (V1.2.0–V1.2.5 and the emit-half fix), and
`generated_kernel_registry.cpp` (the co-inclusion model and its existing collision note).
