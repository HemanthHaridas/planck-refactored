# V1.3 remainder — link and run a dressed generated kernel

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

### V1.3.2 — resolve the single-dressed-rank constraint, deliberately (~S–M, the real decision)

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

**Rank mismatch to resolve first.** The established generated-kernel anchor is
`be_rccsdtq_sto3g` — rank **4**, gated at `rccsdtq_total_energy = -14.4036551081` (atol 1e-7),
tagged `extended`/`generated-kernel`, with a 600 s timeout. But the dressed operators
recognized today are the **CCSD** family (`Wmnij`/`Wabef`/`Wmbej` + `tau`/`tau_c`), and every
V1 measurement has been on `ccsd`. So either:

- dress at rank 4 and confirm the CCSD-family operators still recognize in the CCSDTQ
  residual (they should — `Wmbej` had usage 5 in the ccsd probe, but rank-4 recognition is
  unmeasured), or
- add a rank-2/3 generated-kernel case, which means checking whether `PLANCK_CC_MAXORDER=2`
  produces a registry-included TU at all (the registry only includes rank ≥ 4 plus the
  optional rank-3 arbitrary companion).

**Settle this before V1.3.1**, because it determines which `MAXORDER` the option must work at
and therefore whether V1.3.2's single-rank restriction bites immediately.

*Measurement in progress:* rank-4 dressed recognition (`_dress_operator_equations` on
`ccsdtq`) did **not** finish within 10 minutes, versus seconds at rank 2. That is itself a
finding for V1.3.1 — if dressing at rank 4 is minutes-slow, it lands in the build's critical
path (the `ccgen-planck-kernels` custom target), and the CMake option's cost needs stating.
Confirm the runtime before committing to rank 4 as the anchor.

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
