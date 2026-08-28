# Antisymmetrized ERIs across the codegen boundary

Answers one question: **the UCC equations are written for antisymmetrized `<pq||rs>` and the
C++ cache stores plain `<pq|rs>` — where does the exchange get added, and why is that the
hardest thing in the UCC pipeline to get right?**

The short answer: **in the emitter, per read, routed through the same block search as the
direct read.** Every other placement is either undefinable or silently wrong, and the two
defects this cost — one in the choice of side, one in the implementation — were both invisible
to structural gates and both reached the runtime as *plausible wrong numbers*.

Landed in `fe744e6` (the convention) and `e33c09b` (the routing). `ucc2` reproduces
hand-written UCCSD exactly: B/STO-3G `-0.0402694793` both, H2O⁺/STO-3G to 1e-10.

---

## The mismatch

ccgen's UCC manifold means the **antisymmetrized** integral when it writes `v_aaaa`. Two
independent pieces of evidence, either sufficient:

- `ucc_integrate_term_antisym` documents itself as integrating "for REAL ANTISYMMETRIC
  tensors";
- the energy manifold carries exactly **one** same-spin doubles term, at coefficient `1/4`,
  with no exchange partner — which only balances for `<ij||ab>`.

```
1/4 t2_aaaa * v_aaaa        1 t2_abab * v_abab        1/4 t2_bbbb * v_bbbb
```

`build_ucc_spin_block_cache_from_eri` stores a plain `transform_eri` result. Both sides are
correct in isolation; nothing checked they agreed. Unfixed, `ucc2` returned `-0.0705299626`
against a true `-0.0402694793` — 75% off, converging, plausible.

---

## Why the exchange belongs in the emitter

The alternative — antisymmetrize the stored arrays — fails on three independent counts, and
the first is fatal on its own.

**1. It is not uniformly definable.** `<pq||rs> = <pq|rs> - <pq|sr>` requires slots 2 and 3 to
be interchangeable, i.e. to carry the same spin. For a mixed block they do not, and the
exchange partner is a **different tensor shape**: `oovv_abab` is `(noa, nob, nva, nvb)` and its
partner would be `(noa, nob, nvb, nva)`. The array cannot hold it. So "antisymmetrize the
cache" is not one transform over a vocabulary but a conditional applied to a subset — 12 of the
28 blocks the rank-2 TU reads — leaving one accessor with two meanings and no marker saying
which.

**2. It contradicts a rule the C++ already states.** `ucc_blocks.cpp:32`: the four antisymmetric
single-swap relations "hold only for the ANTISYMMETRIZED `<pq||rs>` … do not add them here
either". `kEriSymmetries` is built on that premise and would be invalid for exactly the blocks
that changed.

**3. It silently redefines what three landed gates assert on** — U3.4's MP2 limit, U5.2c's UMP2
energy, the structural rebind gate. They would keep passing, having lost their meaning.

The emitter, by contrast, already owns this knowledge: `ucc_integrate_term_antisym` →
`_antisym_to_allowed` maps factors into allowed blocks and folds swap signs into coefficients.
Emitting the exchange extends that mechanism instead of adding a third.

`_block_needs_explicit_exchange` states the convention in one place: **every array in the UCC
block cache stores the plain `<pq|rs>`; the emitter writes any antisymmetrization it needs into
the emitted text.** Same-spin tags need it; mixed tags do not (no exchange partner in the
algebra, coefficient `1` not `1/4`); two-index Fock blocks are plain on both sides already.

---

## The part that is easy to get wrong, and was

`<pq||rs> = <pq|rs> - <pq|sr>` swaps the two **ket slots**. The first implementation swapped the
**last two argument positions of the emitted read**. Those coincide only when the swap stays
inside one space:

```
oooo   4     oovv  80     vvvv   6      last-two swap == ket swap: OK
ooov  32     ovov  20     ovvv  38      last-two swap crosses occ/vir: WRONG
```

Half the emitted exchange pairs — 90 of 180 — were wrong. Concretely, `doubles_abab`'s
`-1 * t2_abab * v_aaaa` binds `v_aaaa` as `(i:occ, c:vir, k:occ, a:vir)`, an `ovov` read. The
ket-swapped partner `<ic|ak>` has pattern `ovvo` — **a different stored block** — but the code
read it out of `ovov` with permuted arguments. Wrong values, and in-bounds only by luck when
the occupied and virtual extents happen to be compatible.

**The fix: re-resolve the partner through the same block search as the direct read**, on the
ket-swapped *abstract* pattern. `_resolve_eri_read` and `_resolve_eri_block_name` factor that
search out so the emitted reads and the bound views share one routing:

```
ovov -> ovvo          the case that was wrong
ooov -> ooov(j,i,k,a) reached by a bra permutation, not identity
oovv -> oovv(i,j,b,a) where the old and new agree
```

Two details the compiler caught that reading would not have:

- **The partner needs its own bound view.** `_eri_blocks_used` re-derived the block search
  independently, so it went out of step the instant the routing changed — despite its own
  docstring saying it must resolve "exactly the way `_map_eri_tensor` will … rather than
  re-deriving it and risking the two disagreeing". It now calls the shared resolver. This is
  the general lesson: *two copies of one search is one copy too many.*
- **The partner's permutation carries its own sign**, which must be folded against the direct
  read's rather than assumed to be a plain minus.

---

## Why the gates did not catch either defect

**The convention defect was invisible to every structural gate** because both sides were
internally consistent. Only a number could see it, and no numeric UCC gate existed —
everything through U3, U4 and U5.0–U5.3b asserted on emitted *text*.

**The routing defect was worse: the gate encoded it.** `test_ucc_eri_convention.py` asserted
the partner swapped "the LAST TWO slots" of the direct read — a description of what the code
did, not of the contract it should satisfy. It passed with the bug and could never have failed
with it. Written from the implementation, a gate cannot falsify the implementation.

The replacement asserts the **contract**: a ket-swapped *pattern* resolves to its own stored
block, pinned by name on `ovov → ovvo`, checked against the routing rather than the emitted
text so it cannot drift back into re-describing the code. Mutation-verified — restoring the
position-only swap fails it.

---

## How to localize a defect in this pipeline

Two instruments, in order. Both were decisive here.

**1. First order, with `cc_damping 1.0`.** At a zero start the residual collapses to a single
constant term per block, so `t = R(0)/D` is closed-form and iteration 1 must equal UMP2 exactly.
It does, to ten digits, on both fixtures — which clears in one measurement the stored ERI
blocks, the per-block denominators including `abab`'s spin assignment, the physicist rebind, the
amplitude write-back and the energy coefficients. **Anything still wrong is in the higher-order
terms.** This is the same lever U3.4 used.

**2. Iterate-by-iterate against the hand-written solver, DIIS off.** The first divergence names
the order the defect lives at:

```
iter    generated        hand-written       diff
   1  -0.0190946435    -0.0190946435    +0.000e+00      <- first order clean
   2  -0.0251026330    -0.0252106915    +1.081e-04      <- linear in t2: here
   3  -0.0304876742    -0.0295950340    -8.926e-04
```

**`cc_damping` defaults to `0.8`, and that will masquerade as a defect.** The Jacobi update is
`delta = damping * R/D`, so iteration 1 lands at exactly 80% of the MP2 amplitude — an exact
`0.800000` ratio, on every channel and every system, which reads as a structural coefficient
bug. It cost this investigation two full steps. *An exact rational ratio is evidence of a
constant, and a constant is as likely to be a configuration default as a coefficient bug: grep
the knobs before theorising about the equations.*

---

## Fixtures

**`h2o_cation_ucc2_sto3g.hfinp` (H2O⁺ doublet, C1) is the one to reason on.** `noa=5, nob=4,
nva=2, nvb=3`; all three channels non-zero; no degeneracy in the same-spin block
(`max|v(ijab) - v(ijba)| = 3.9e-2`). UMP2 `-0.0272204807`, UCCSD `-0.0384280769`.

**`b_ucc2_sto3g.hfinp` (B/STO-3G doublet) is simpler but DEGENERATE for same-spin questions.**
Its `oovv_aaaa` satisfies `v(i,j,a,b) == v(i,j,b,a)` identically — a real property of that
high-symmetry atom with two degenerate 2p virtuals — so both same-spin channels are exactly zero
at first order and any same-spin assertion on it passes vacuously. Keep it: that same zero
cleanly isolates `abab`. Never conclude from it alone. UMP2 `-0.0190946435`, UCCSD
`-0.0402694793`.

> **It passed its own non-vacuity check.** `b_ucc2_sto3g` was verified to have four different
> orbital counts and a non-trivial `E_corr` — both true, and both insufficient, because neither
> says anything about degeneracy *within* a channel. A fixture is not general because its shapes
> differ. **Any assertion about same-spin behaviour must run on a C1 system.**

---

## Still open

- **No regression case is registered.** `PLANCK_CC_UCC` is default OFF and the runner has no
  build-option gating, so a `ucc2` case would fail in a default build. That plumbing is the
  remaining U5.4 work; a case asserting the right number in a build that cannot produce it is
  worse than none.
- **The amplitude-side antisymmetry convention is unenforced.** `ucc_amplitude_blocks`
  (`amplitudes.cpp:315`) states that "the within-half antisymmetry folds slot permutations, so
  only the count matters", and nothing checks it. Measured, the amplitudes *are* antisymmetric
  to ~1e-16 — so this is not a live defect, but it is the same shape as the one above: a
  convention asserted on one side and assumed on the other. Worth a gate before it becomes one.

## Build and run notes

- `-DPLANCK_CC_MAXORDER=2` does not build: `tensor_backend.cpp` hard-includes
  `ccsdt_planck_generated.cpp`, so rank 3 is the floor.
- A failed `make` can still report exit code 0. Check for the binary, not the code.
- `BASIS_PATH=$PWD/basis-sets` is required to run any input from a build tree.
