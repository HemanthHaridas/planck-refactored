# Generating and running unrestricted CC kernels

Answers one question: **how does a spin-orbital CC manifold become a runnable open-shell
(UHF-reference) kernel set, and what has to stay spin-resolved all the way down?**

The short answer: **UCC is RCC minus the spatial collapse.** The same generator, bridge,
runtime, solver and sector machinery serve both; UCC's defining property is *skipping* the three
steps that fold spin blocks into one tensor per rank. Everything hard about it follows from one
consequence — **a quantity that is one thing under RHF is several under UHF**, and every layer
that assumed "one" had to learn "several."

**Landed and validated end to end** (2026-08-25):

```
ucc2  -24.1892581442   == hand-written UCCSD, exactly
ucc3  -24.1892636163   strictly between UCCSD and FCI; T3 recovers 80.1% of the gap
ucc4  -24.1892649766   == FCI, all ten digits          (B/STO-3G doublet, 3a/2b)
```

Registered as `b_ucc{2,3,4}_sto3g`, behind `-DPLANCK_CC_UCC=ON` (default OFF).

---

## The shape of the pipeline

| layer | RHF | UHF | where |
|---|---|---|---|
| equations | one manifold | one manifold, **spin-resolved per block** | `ucc_adapt_equations` |
| amplitudes | one tensor per rank | one per `(rank, tag)` sector | `sectors` / `sector_tensor` |
| denominators | one per rank | one per block — `abab` differs in **shape** | `build_ucc_denominator_cache` |
| ERIs | 7 named members | **24** arrays (7 `aaaa` / 10 `abab` / 7 `bbbb`) | `ucc_canonical_blocks` |
| Fock | `f_oo`/`f_ov`/`f_vv` | per-spin blocks | `build_ucc_fock_blocks` |
| orbital counts | one `(n_occ, n_virt)` | **four** counts | `CanonicalRHFCCReference` |
| kernels | `compute_<method>_*` | `compute_ucc_<method>_*`, own TU | U5.0 naming |

The block vocabulary is **derived, never negotiated**: one array per orbit of the 16 o/v patterns
under each tag's own symmetry group. A mixed block's orbits are smaller (two of the four
permutations map `abab` to `baba`, so they are not its symmetries), which is why it needs 10
arrays where a same-spin tag needs 7. The C++ and the emitter derive that independently from the
same rule and are gated against each other, so they cannot drift.

---

## The four defects worth knowing about

Each was invisible to the gates in place at the time, and each returned a *plausible wrong
number* rather than an error.

**1. Spin-blind loop bounds and shapes.** Every emitted kernel opened
`const int no = reference.orbital_partition.n_occ` — a scalar that is deliberately left default
on a UCC reference, because one pair cannot describe an unrestricted partition. Both counts were
0. Fixed by deriving each index's spin from the *factors* (slot *k* of `t2_abab` carries
`tag[k]`, read **positionally**, not grouped by space) and each result's extents from the
target's tag. Full answer: `CCGEN_UCC_ERI_ANTISYMMETRY.md` covers the sibling ERI case.

**2. The ERI antisymmetry convention.** ccgen means `<pq||rs>` when it writes `v_aaaa`; the cache
stores plain `<pq|rs>`. Both sides correct in isolation, never checked against each other.
Returned −0.0705 against a true −0.0403.

**3. The exchange routing.** The fix for (2) swapped the last two *argument positions* rather
than the two *ket slots*. Those coincide only for `oooo`/`oovv`/`vvvv`; on `ooov`/`ovov`/`ovvv`
the swap crosses occ/vir and reads the wrong array. Half the emitted exchange pairs were
affected. **The gate asserted the buggy behaviour** and so could never have failed with it.

(2) and (3) in full: **`CCGEN_UCC_ERI_ANTISYMMETRY.md`**.

**4. A regression into the neighbouring path.** Keying the spin map on "the map came back empty"
rather than on `ucc` also fires on the **spin-adapted** RCC path, whose rank-4 sector amplitudes
are named `t4_aaabaaab`. Neither RCC gate could see it — the SHA-256 pin emits with
`spin_adapt=False`, and the flag matrix pins `spin_adapt` only at `METHOD = "ccsd"`, below the
rank where those names exist. Two comprehensive-looking gates, blind for two different reasons.

---

## What NOT to do

Most of these cost a real investigation.

- **Do not fork the pipeline.** UCC is RCC-minus-collapse. A parallel `ucc/` module duplicates
  the bridge and the two copies drift.
- **Do not add a new amplitude container or solver loop.** The `(rank, tag)` sector machinery is
  already the general case.
- **Do not reuse `_amplitude_block_tag`'s β-majority flip.** It folds `abbabb` into `aabaab`,
  valid only when α and β orbitals coincide. The easiest way to a silent wrong answer.
- **Do not enable `--include-intermediates` on the UCC path** until it is validated there — CSE
  mislabels occ/vir on spatial spin-adapted terms, and UCC has strictly more terms.
- **Do not validate a spin-routing claim against a same-spin block.** `aaaa` is invariant under
  the spin-tag permutation and symmetric under the mid-axis swap, so it agrees with *both* of two
  contradictory hypotheses. **This scope hit that vacuity eight times**, including once while
  writing the warning about it, and once on a fixture that had passed its own non-vacuity check
  (four different orbital counts and a non-trivial `E_corr` say nothing about degeneracy *within*
  a channel). Assertions about spin routing need a **mixed block with asymmetric extents**, where
  a wrong hypothesis changes the SHAPE.
- **Do not fix a spin-routing emitter by suffixing array names alone.** Two of the four ERI
  symmetry permutations are invalid for `abab`; name-only routing sends those reads to the right
  array with permuted indices — quieter than the collapse it replaces.
- **Do not store a `baba` ERI family.** It is `abab` under the particle swap; storing it buys
  ~33% more memory to avoid one explicit swap.
- **Do not write a gate from the implementation.** (3) above passed because it asserted what the
  code did rather than the contract it should satisfy. **Mutation-test every gate here before
  trusting a pass** — two more vacuous gates in this effort were caught that way and not by
  review.
- **Do not assume a rejected input proves which guard rejected it.** Two guards can reject the
  same fixture; assert on the guard's own message.
- **Do not read `Terminated: 15` from `cc1plus` as a compile error.** It is SIGTERM — the
  compiler was killed, on the slow `-O1`-pinned registry TU.
- **Do not treat a green `ctest` as evidence that `hartree-fock` builds.** The CC unit binaries
  do not link it.
- **Do not trust a `make` that prints nothing, or its exit code.** A failed build has reported
  exit 0 in this tree more than once. Check for the binary.

---

## Two lessons that generalize past CC

**An exact rational ratio is evidence of a CONSTANT, and a constant is as likely to be a
configuration default as a coefficient bug.** `cc_damping` defaults to 0.8 and the Jacobi update
is `delta = damping·R/D`, so iteration 1 sits at exactly 80% of MP2 on every channel and every
system. That masqueraded as a structural defect and cost two investigation steps. *Grep the knobs
before theorising about the equations.*

**Exactness is set by what the basis can reach, not by the electron count alone.** The plan for
the final gate concluded that open-shell `CCSDTQ == FCI` was impossible: exactness seemed to need
`n_elec ≤ 4`, a worthwhile T4 needs ≥2 electrons of each spin, and open shell needs
`n_alpha ≠ n_beta` — and 4 electrons with 2 of each spin is closed.

That is **true at four electrons**, and it is worth keeping the measurement that shows it.
Triplet Be/STO-3G (4 e⁻, 3α/1β, `<S²>` exactly 2.000000, no contamination):

```
in-tree FCI (ROHF reference)   -14.2866221716
hand-written UCCSD             -14.2866221716    identical to 1e-10
```

UCCSD already *is* FCI there — T3 and T4 both contribute nothing, so a broken implementation of
either passes. Any 4-electron open-shell system is 3α/1β, and one beta electron can be excited at
most once, so the `aabb` T4 sector is identically zero.

**But it is wrong as a generalization**, which is what made the "impossible" conclusion wrong.
B/STO-3G has 5 electrons and only **2 alpha virtuals for 3 alpha electrons**, so T5 needs ≥3
alpha excitations and cannot exist — CCSDTQ *is* exact there. The orbital count enforced what the
electron count could not.

*The interval framing was still the right instrument*: it is what made `ucc3` a real gate (T3
recovers 80.1% of the UCCSD→FCI gap, so a T3 doing nothing fails the lower bound), and it is what
let `ucc4` **demonstrate** exactness rather than assume it. Both `ucc3` and `ucc4` keep both
bounds, mutation-verified in each direction — below FCI is the B5 signature, not a rounding
issue.

---

## How to localize a defect in this pipeline

**1. First order, at `cc_damping 1.0`.** At a zero start every residual collapses to one constant
term per block, so `t = R(0)/D` is closed-form and iteration 1 must equal UMP2 exactly. One
measurement clears the stored ERI blocks, the denominators, the rebind, the write-back and the
energy coefficients.

**2. Iterate-by-iterate against the hand-written solver, DIIS off.** The first divergence names
the order the defect lives at — clean at iteration 1 and wrong at 2 means it is linear in `t2`.

**3. A probe in the runtime loop.** Per-channel energies, residual norms, amplitude symmetry.
Inserted, run, reverted — never committed. This is what proved the ERI exchange fix works, and
what killed the amplitude-antisymmetry suspect by measurement rather than argument.

---

## Still open

- **`wick` engine coverage.** Every UCC gate ran through the `diagram` engine. The two are
  documented residual-equal but unpinned for UCC — worth one assertion rather than an assumption.
- **The amplitude-antisymmetry convention.** `ucc_amplitude_blocks` states that "the within-half
  antisymmetry folds slot permutations, so only the count matters", and nothing enforces it.
  Measured satisfied to ~1e-16, so not a live defect — but it is the same implicit-convention
  shape as defects (2) and (4).
- **Dressed UCC.** Dressing is retired as a production route for RCC (it does not compose with
  spin adaptation; 52% short on Be — `CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`). For UCC the
  mechanism *predicts* it would work, because UCC keeps per-spin-block tensors rather than
  folding to one spatial tensor. Untested. Two constraints bind if it is ever tried: the adapt
  entry must accept an already-dressed manifold (it takes the equation dict as a parameter, so it
  does), and intermediate specs must be block-keyed — the same naming path amplitudes and ERIs
  already use. `Wmbej`'s `ovvo` binding sign is the highest-risk single item.

## Costs, measured

```
rank-4 UCC generation      579 s, of which ucc_adapt_equations is 1.8 s
                           (the rest is the GCC step the RCC path already pays)
rank-4 configure+build     ~10 min end to end, -DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_UCC=ON
rank-4 manifold            18533 terms across 14 sectors
b_ucc4_sto3g runtime       ~44 s
```

`-DPLANCK_CC_MAXORDER=2` does not build: `tensor_backend.cpp` hard-includes the rank-3 TU, so
rank 3 is the floor. A UCC tree built with `make hartree-fock` alone cannot run the full
`extended` suite — the DFT cases need `planck-dft`, and the runner dies with a bare
`FileNotFoundError`.

---

See `CCGEN_UCC_ERI_ANTISYMMETRY.md` (the ERI convention and its routing, in full),
`CCGEN_UCC_NUMERIC_VALIDATION.md` (how a spin-block residual is checked against PySCF),
`CCGEN_U1_UCC_ADAPT_SCOPE.md` (the GCC→UCC adaptation), and `CCGEN_CCSDTQ_MULTISECTOR.md`
(the sector runtime UCC reuses wholesale).
