# UCC Antisymmetrized ERIs Across the Codegen Boundary

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**The UCC equations are written for antisymmetrized `<pq||rs>` and the C++ cache stores plain `<pq|rs>` — where does the exchange get added, and why is that the hardest thing in the UCC pipeline to get right?**

## Short answer

The exchange is added in the emitter, per read, routed through the same block search as the direct read. Every other placement is either undefinable or silently wrong, and the two defects this cost — one in the choice of side, one in the implementation — were both invisible to structural gates and both reached the runtime as plausible wrong numbers. Landed in `fe744e6` (the convention) and `e33c09b` (the routing). `ucc2` reproduces hand-written UCCSD exactly: B/STO-3G `-0.0402694793` both, H2O⁺/STO-3G to 1e-10.

## Where the logic lives

- `ucc_integrate_term_antisym` → `_antisym_to_allowed` — maps factors into allowed blocks and folds swap signs into coefficients
- `_block_needs_explicit_exchange` — states the convention: every array in the UCC block cache stores the plain `<pq|rs>`; the emitter writes any antisymmetrization it needs into the emitted text
- `_resolve_eri_read` and `_resolve_eri_block_name` — the shared block-search routing used by both the direct read and its ket-swapped partner
- `_eri_blocks_used` — bound-view derivation, now calling the shared resolver
- `build_ucc_spin_block_cache_from_eri` — stores the plain `transform_eri` result
- `ucc_blocks.cpp:32` — the rule that the four antisymmetric single-swap relations hold only for the antisymmetrized `<pq||rs>`
- `amplitudes.cpp:315` — `ucc_amplitude_blocks`, the unenforced amplitude-antisymmetry convention
- `python/ccgen/tests/test_ucc_eri_convention.py` — the gate, rewritten to assert the contract rather than the implementation

## What invariants matter

### 1. The stored cache and the algebra's convention must be checked against each other, not just self-consistent

ccgen's UCC manifold means the antisymmetrized integral when it writes `v_aaaa`. Two independent pieces of evidence, either sufficient: `ucc_integrate_term_antisym` documents itself as integrating "for REAL ANTISYMMETRIC tensors"; and the energy manifold carries exactly one same-spin doubles term, at coefficient `1/4`, with no exchange partner — which only balances for `<ij||ab>`:

```
1/4 t2_aaaa * v_aaaa        1 t2_abab * v_abab        1/4 t2_bbbb * v_bbbb
```

`build_ucc_spin_block_cache_from_eri` stores a plain `transform_eri` result. Both sides are correct in isolation; nothing checked they agreed. Unfixed, `ucc2` returned `-0.0705299626` against a true `-0.0402694793` — 75% off, converging, plausible.

Design rule:

- When one layer's convention (antisymmetrized vs plain integrals) is implicit in its equations and another layer's storage convention is implicit in its transform, add an explicit check or contract between them — do not rely on both being independently correct.

### 2. Antisymmetrization must be applied at the emitter, not by transforming the stored cache

The alternative — antisymmetrize the stored arrays — fails on three independent counts, the first fatal on its own:

1. It is not uniformly definable. `<pq||rs> = <pq|rs> - <pq|sr>` requires slots 2 and 3 to be interchangeable, i.e. to carry the same spin. For a mixed block they do not, and the exchange partner is a different tensor shape: `oovv_abab` is `(noa, nob, nva, nvb)` and its partner would be `(noa, nob, nvb, nva)`. The array cannot hold it. So "antisymmetrize the cache" is not one transform over a vocabulary but a conditional applied to a subset — 12 of the 28 blocks the rank-2 TU reads — leaving one accessor with two meanings and no marker saying which.
2. It contradicts a rule the C++ already states: `ucc_blocks.cpp:32` says the four antisymmetric single-swap relations "hold only for the ANTISYMMETRIZED `<pq||rs>` … do not add them here either". `kEriSymmetries` is built on that premise and would be invalid for exactly the blocks that changed.
3. It silently redefines what three landed gates assert on — U3.4's MP2 limit, U5.2c's UMP2 energy, the structural rebind gate. They would keep passing, having lost their meaning.

Design rule:

- Do the antisymmetrization in the emitter, extending `_antisym_to_allowed`'s existing mechanism, rather than adding a third place that knows about spin-block shape. `_block_needs_explicit_exchange` records the resulting convention: same-spin tags need the explicit exchange, mixed tags do not, and two-index Fock blocks are plain on both sides already.

### 3. A partner block must be found through the same routing as the direct read, not by permuting argument positions

`<pq||rs> = <pq|rs> - <pq|sr>` swaps the two ket slots. The first implementation swapped the last two argument positions of the emitted read. Those coincide only when the swap stays inside one space:

```
oooo   4     oovv  80     vvvv   6      last-two swap == ket swap: OK
ooov  32     ovov  20     ovvv  38      last-two swap crosses occ/vir: WRONG
```

Half the emitted exchange pairs — 90 of 180 — were wrong. Concretely, `doubles_abab`'s `-1 * t2_abab * v_aaaa` binds `v_aaaa` as `(i:occ, c:vir, k:occ, a:vir)`, an `ovov` read. The ket-swapped partner `<ic|ak>` has pattern `ovvo` — a different stored block — but the code read it out of `ovov` with permuted arguments. Wrong values, and in-bounds only by luck when the occupied and virtual extents happen to be compatible.

Design rule:

- Re-resolve the partner through the same block search as the direct read, on the ket-swapped *abstract* pattern (`_resolve_eri_read` / `_resolve_eri_block_name`), rather than manipulating argument positions of the already-resolved read. Two copies of one search is one copy too many — `_eri_blocks_used` had re-derived the block search independently and drifted the instant the routing changed, despite its own docstring warning against exactly that. The partner's permutation also carries its own sign, which must be folded against the direct read's rather than assumed to be a plain minus.

### 4. A gate written from the implementation cannot falsify the implementation

`test_ucc_eri_convention.py` originally asserted the partner swapped "the LAST TWO slots" of the direct read — a description of what the code did, not of the contract it should satisfy. It passed with the bug and could never have failed with it.

Design rule:

- Write a gate against the contract (a ket-swapped *pattern* resolves to its own stored block) rather than against a description of the current implementation. The replacement here is pinned by name on `ovov → ovvo`, checked against the routing rather than the emitted text, and is mutation-verified — restoring the position-only swap fails it.

### 5. An exact rational ratio in a diverging comparison may be a configuration default, not an equation bug

`cc_damping` defaults to `0.8`, and that will masquerade as a defect. The Jacobi update is `delta = damping * R/D`, so iteration 1 lands at exactly 80% of the MP2 amplitude — an exact `0.800000` ratio, on every channel and every system, which reads as a structural coefficient bug. It cost this investigation two full steps.

Design rule:

- An exact rational ratio is evidence of a constant, and a constant is as likely to be a configuration default as a coefficient bug — grep the knobs before theorising about the equations.

## What was fixed

1. Established and documented the convention that the UCC block cache stores plain `<pq|rs>` everywhere, with the emitter responsible for any antisymmetrization (`fe744e6`).
2. Fixed the ket-swap routing so the exchange partner is resolved through the same block search as the direct read, on the abstract pattern rather than by permuting argument positions (`e33c09b`).
3. Routed `_eri_blocks_used` through the shared resolver instead of an independent re-derivation.
4. Rewrote `test_ucc_eri_convention.py` to assert the contract (pattern-based block resolution) rather than the old implementation description.

## Why the gates did not catch either defect

The convention defect was invisible to every structural gate because both sides were internally consistent. Only a number could see it, and no numeric UCC gate existed — everything through U3, U4 and U5.0–U5.3b asserted on emitted *text*.

The routing defect was worse: the gate encoded it, as described in invariant 4 above.

## Validation strategy that should remain in place

Two instruments, in order, both decisive here:

1. **First order, with `cc_damping 1.0`.** At a zero start the residual collapses to a single constant term per block, so `t = R(0)/D` is closed-form and iteration 1 must equal UMP2 exactly. It does, to ten digits, on both fixtures — which clears in one measurement the stored ERI blocks, the per-block denominators including `abab`'s spin assignment, the physicist rebind, the amplitude write-back and the energy coefficients. Anything still wrong is in the higher-order terms. This is the same lever U3.4 used.
2. **Iterate-by-iterate against the hand-written solver, DIIS off.** The first divergence names the order the defect lives at:

   ```
   iter    generated        hand-written       diff
      1  -0.0190946435    -0.0190946435    +0.000e+00      <- first order clean
      2  -0.0251026330    -0.0252106915    +1.081e-04      <- linear in t2: here
      3  -0.0304876742    -0.0295950340    -8.926e-04
   ```

Fixtures to keep:

- `h2o_cation_ucc2_sto3g.hfinp` (H2O⁺ doublet, C1) is the one to reason on. `noa=5, nob=4, nva=2, nvb=3`; all three channels non-zero; no degeneracy in the same-spin block (`max|v(ijab) - v(ijba)| = 3.9e-2`). UMP2 `-0.0272204807`, UCCSD `-0.0384280769`.
- `b_ucc2_sto3g.hfinp` (B/STO-3G doublet) is simpler but degenerate for same-spin questions. Its `oovv_aaaa` satisfies `v(i,j,a,b) == v(i,j,b,a)` identically — a real property of that high-symmetry atom with two degenerate 2p virtuals — so both same-spin channels are exactly zero at first order and any same-spin assertion on it passes vacuously. Keep it: that same zero cleanly isolates `abab`. Never conclude from it alone. UMP2 `-0.0190946435`, UCCSD `-0.0402694793`. It passed its own non-vacuity check (four different orbital counts, non-trivial `E_corr`) — both true, and both insufficient, because neither says anything about degeneracy *within* a channel. A fixture is not general because its shapes differ; any assertion about same-spin behaviour must run on a C1 system.

## Build and run notes

- `-DPLANCK_CC_MAXORDER=2` does not build: `tensor_backend.cpp` hard-includes `ccsdt_planck_generated.cpp`, so rank 3 is the floor.
- A failed `make` can still report exit code 0. Check for the binary, not the code.
- `BASIS_PATH=$PWD/basis-sets` is required to run any input from a build tree.

## Remaining architecture concern

- **No regression case is registered.** `PLANCK_CC_UCC` is default OFF and the runner has no build-option gating, so a `ucc2` case would fail in a default build. That plumbing is the remaining U5.4 work; a case asserting the right number in a build that cannot produce it is worse than none.
- **The amplitude-side antisymmetry convention is unenforced.** `ucc_amplitude_blocks` (`amplitudes.cpp:315`) states that "the within-half antisymmetry folds slot permutations, so only the count matters", and nothing checks it. Measured, the amplitudes *are* antisymmetric to ~1e-16 — so this is not a live defect, but it is the same shape as the ERI convention defect above: a convention asserted on one side and assumed on the other. Worth a gate before it becomes one.
