# Does the generated-vs-hand-written CC gap grow with system size?

**Scope for P3, the open item left by `CCGEN_KERNEL_PERFORMANCE_SCOPE.md`.** Not started.

The accessor fix (`CCGEN_TENSOR_ACCESSOR_FIX_SCOPE.md`) removed the per-access allocation that
dominated every earlier measurement, leaving a residual **22×** generated-vs-hand-written ratio at
rank 3. That residual is the first number in this investigation that is plausibly *structural* —
loop shape and contraction order — rather than an artifact of how an element is fetched.

But every ratio measured so far comes from **one point**: `bh3`/STO-3G, `nocc=8 nvirt=8`,
`no == nv`. One point cannot distinguish a constant factor from a scaling defect, and those have
opposite implications: a constant 22× is a fixed tax worth paying down opportunistically, while a
ratio that *grows* with size means the generated kernels have a worse asymptotic contraction order
than the hand-written ones and will diverge arbitrarily on production systems.

**The one question: is 22× a constant, or does it grow with `o` and `v`?**

## Why this is not answerable from the existing numbers

The two hypotheses that survived P2 make opposite predictions here, and both are still live:

- **H1 (memory-bound by loop fission)** — falsified *at `no=nv=4`* because the residual is 32 KB,
  fully L1-resident, so 1063 separate sweeps cost nothing extra. That argument expires the moment
  the working set exceeds cache, which is a function of size alone. H1 predicts the ratio is ~flat
  while everything fits in cache, then **rises** once it does not.
- **H3 (no contraction-order optimization)** — the generated kernels evaluate each term n-arily in
  emission order; `CCGEN_HIGHER_OPERATOR_REUSE.md` records `t2·t3·v` as `o⁵v⁵` n-ary against
  `o³v⁴` if `v·t3` is factored first. H3 predicts the ratio **grows polynomially** in `o`/`v`, and
  does so regardless of cache.

They are separable by shape, not just by size: H1 keys on total working-set bytes, H3 on the
`o`/`v` exponents. A ladder that varies `o` and `v` *independently* distinguishes them; one that
only scales the molecule does not.

## What to measure

Time the **residual evaluation**, not a converged solve — a solve conflates kernel cost with
convergence path, and the rank-3 defect work already established that rule. The
`PLANCK_CC_T3_DIFF=1` probe evaluates the generated and hand-written residuals once each from
identical amplitudes and is the natural harness; the temporary timing instrumentation hung off it
for P1/P2 was removed after landing, so step one is re-adding it (see that doc's Reproducing
section).

Record `o`, `v`, `o/v`, both residual times, the ratio, **and** the working-set size in bytes, so
an H1 cache transition is visible rather than inferred.

### The ladder

In-tree cases give a non-square contrast immediately — `bh3` is `nocc=8 nvirt=8` (ratio 1.0),
`water_rccsdt_sto3g` is `nocc=10 nvirt=4` (ratio 0.4, and *inverted* vs `bh3`). That pair alone
tests shape at nearly fixed cost. Extending `o` and `v` upward needs larger bases (6-31G on the
same molecules) rather than new molecules, so the comparison stays like-for-like.

Minimum useful set:

- **fixed shape, growing size** — separates constant from scaling.
- **fixed size, varying `o/v`** — separates H1 (bytes) from H3 (exponents).
- **at least one case whose `t3` working set exceeds L2**, or H1 is untested by construction.

Non-square throughout. `no == nv` is not just uninformative here, it is actively hazardous: the
rank-3 defect work flagged that a wrongly-ordered read stays in bounds and fails silently on a
square system, which is exactly why the accessor gate uses distinct extents in every axis.

### Rank 4

Rank 4 is the production target and is **not** covered by a rank-3 ladder — different tensor types,
different code path, and a lesson already paid for once: the fixed-rank-only accessor pass fixed
rank 3 by 76× and moved rank 4 by nothing. Rank 4 also still carries the `-O1` registry pin
(`CMakeLists.txt:402`) that rank 3 does not, so its ratio has a known extra term. Measure at least
one rank-4 point; `be_rccsdtq_sto3g` at 11.4 s/iteration is the cheapest handle.

## What each outcome implies

| result | reading | what it makes worth doing |
|---|---|---|
| ratio ~flat in size and shape | 22× is a constant tax | emitter fusion / IR hints, opportunistically — no urgency |
| ratio rises past a working-set threshold | H1 returns at scale | fuse the 1063 sweeps; the fix is loop structure and the payoff is size-dependent |
| ratio grows polynomially, cache-independent | H3 — worse contraction order | consume `_optimal_contraction_order`; this is the asymptotic fix and outranks everything else |
| rank-4 ratio ≫ rank-3 ratio | rank-specific defect (possibly the `-O1` pin) | chunk the giant residual kernels in the emit so any optimization level stays cheap |

The last row has a standing follow-on already recorded at `CMakeLists.txt:402`.

## Constraints

- **Do not re-derive the accessor result.** It is landed and gated; if a measurement here disagrees
  with the recorded before/after, suspect the build tree first.
- **A/B in one configure, rebuilding both arms.** Both misreads during the P1/P2 work came from
  comparing binaries built from different source states or different `CMAKE_BUILD_TYPE`s. Never
  compare against a stale `build/`.
- **`make -j4`.** The generated TUs are large enough that a full-width build is disruptive.
- **Energies must stay bitwise-identical** across anything this investigation motivates, same as
  the accessor fix — these are all evaluation-order-preserving changes until proven otherwise, and
  a fusion that reassociates floating-point accumulation is *not*, so that would need its own
  justification rather than being absorbed into a tolerance.

## Key code locations

| what | where |
|---|---|
| generated-vs-hand branch + `T3_DIFF` probe | `src/post_hf/cc/tensor_backend.cpp:2324` |
| hand-written triples (1-nest reference) | `src/post_hf/cc/tensor_backend.cpp:1800` |
| one-nest-per-term emission (H1's mechanism) | `python/ccgen/emit/planck_tensor_cpp.py:284`, `:443` |
| unused contraction-order analysis (H3's fix) | `python/ccgen/tensor_ir.py:198,261,283` |
| `-O1` registry pin (rank 4+ only) | `CMakeLists.txt:402` |
| measurement record this continues | `docs/CCGEN_KERNEL_PERFORMANCE_SCOPE.md` |
