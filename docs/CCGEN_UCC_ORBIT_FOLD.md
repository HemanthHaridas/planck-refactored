# Folding the UCC Residuals by Their External Symmetry

Canonical status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.

**Question.** Can each generated UCC residual block be emitted as one term per
orbit of its external index symmetry, followed by one symmetrize pass, as the
spin-adapted RCC `aab` triples block already is? What does it save, and what
must hold for it to be exact?

## Short answer (landed 2026-11-22)

Yes. Every UCC block whose free indices include two of the same space and spin
now folds: the same-spin doubles, all four triples blocks and all five
quadruples blocks. The block's group is derived from its indices' spins.
`fold_external_symmetry` keeps one term per orbit, and the kernel ends with
`R = Σ_g s_g g(X)`. Measured against an unfolded build of the same commit (both
with derived operators, 4 threads, whole-run wall time, three alternating
repeats each):

| case | unfolded | folded | speedup | energy |
|---|---|---|---|---|
| `b_ucc4_sto3g` (rank 4) | 5.04 s | 1.13 s | 4.5× | identical; still equals FCI |
| `b_ucc3_sto3g` (rank 3) | 0.70 s | 0.36 s | 1.9× | differs by 1e-10 (reassociated sums) |
| `oh_uccsd_t_631g` | 0.30 s | 0.24 s | 1.25× | identical |
| `water_cation_uccsd_t_spherical_ccpvdz` | 0.82 s | 0.64 s | 1.28× | identical |

The `uccsd(t)` cases gain least: rank 2 folds only the same-spin doubles, and
these runs are mostly SCF and (T). The binary went from 52.8 MB to 30.7 MB, and
the rank-4 UCC file from 12.6 MB to 2.9 MB.

## What the fold removes

Measured on the equations before any C++ was written. Modelled FLOPs at
o = 5, v = 20; symmetry and error are max-abs relative on `ucc_random_tensors`
(noa/nva/nob/nvb = 3/5/4/6).

| block | group order | terms | modelled FLOPs | symmetry of R | fold + symmetrize vs R |
|---|---|---|---|---|---|
| singles, `doubles_abab` | 1 | unchanged | 1.00× | — | — |
| `doubles_aaaa` / `bbbb` | 4 | 118 → 56 | 1.85× | ≤ 4.9e-16 | ≤ 6.2e-16 |
| `triples_aabaab` / `abbabb` | 4 | 469 → 202 / 192 | 2.24× / 2.36× | ≤ 1.1e-15 | ≤ 1.5e-15 |
| `triples_aaaaaa` / `bbbbbb` | 36 | 579 → 72 | 7.47× | ≤ 9.7e-16 | 1.8e-15 |
| `quadruples_aabbaabb` | 16 | 2634 → 451 | 5.5× | | |
| `quadruples_aaabaaab` / `abbbabbb` | 36 | 2816 → 358 / 324 | 7.2× / 8.3× | | |
| `quadruples_aaaaaaaa` / `bbbbbbbb` | 576 | 3630 → 111 | 28.8× | | |

Whole residual: CCSDT 2,484 → 802 terms (derived route 3.5× fewer modelled
FLOPs); rank 4 18,131 → 2,230 (10.7× undressed). The derived operator count does
not change (333 at CCSDT): the orbit representatives use the same contraction
shapes, so no operator is lost and none is added. Generation costs 14 s per
order-576 block; regenerating all three UCC files takes 52 s.

**Canonical orbit keys: tried and reverted (2026-11-26).** The fold keys
each image by its text with dummies as named, so images that differ only
in dummy names are separate orbits. Keying on the canonical form instead
(factors sorted, then `canonicalize_term_to_fixed_point`, joining an
existing representative when any image is one) is exact (2e-15 against the
unfolded residual) but buys little: UCC CCSDT 802 → 790 terms, RCC T
triples 331 → 306, Λ residuals no change (they hold few images). Its cost
is the canonicalizer's, which grows steeply with summed indices: the
Λ-CCSDTQ fold went from under a second to many minutes (single large
terms took minutes), and a bound at six summed indices did not rescue it.
Reverted. A cheaper complete canonical form is the route if the ~8 % on
the T triples is wanted.

## Where the logic lives

- `ccgen/optimization/permutation.py`: `spin_class_group(term)` (the group) and
  `fold_external_symmetry` (generic over any closed signed group).
- `ccgen/generate.py`, `print_cpp_planck`: folds every UCC target with a
  non-trivial group, before factorization, and passes the groups to the emitter
  as `symmetrizers`. RCC keeps its one hand-written `AAB_TRIPLES_GROUP`.
- `ccgen/emit/planck_tensor_cpp.py`: `_symmetrize_lines` (spin-resolved bounds
  through `result_spins`) and `_accessor`, the one rule for element access that
  both term emission and the symmetrize pass use.
- `ccgen/tests/test_factorize_value_preservation.py`: `UCCOrbitFoldTests`.

## What invariants matter

1. **The group comes from spins, never from a table.** It is the product, over
   each (space, spin) class of free indices, of that class's symmetric group,
   with each element signed by its parity. It never swaps two indices of
   different spin, because `abab` → `baba` is a different block (the tag-blind
   error, `CCGEN_TWO_DRESSING_ROUTES.md` invariant 3). The group orders are
   pinned by `test_group_orders_follow_spin_classes`.
2. **An element need not be its own inverse.** `fold_external_symmetry` used to
   state that requirement, and S3's 3-cycles break it. The requirement is not
   needed. The fold and the symmetrize pass both run over the whole group, and
   an element and its inverse have the same sign, so whether a rename applies
   g or g⁻¹ does not matter. The docstring now says this.
3. **Exactness rests on antisymmetric amplitudes within each same-spin class.**
   The symmetrize pass projects R onto its symmetric part. That equals the
   unfolded R only when T has the symmetry. UCC starts from zero amplitudes or
   from a `.ccamp` file written by this solver; it has no MP2 guess and no
   cross-rank seed (`uccgen.cpp`). Jacobi and DIIS take linear combinations, so
   T keeps the property. A new seed path would have to keep it too. The durable
   amplitude-antisymmetry gate in Open Work is where such a check belongs.
4. **Fold before factorizing.** Otherwise operators would be derived for terms
   that fold away.
5. **Element access has one rule.** Fixed-rank types take `x(i, ...)`;
   `TensorND` (rank ≥ 7) takes `x({i, ...})`. The symmetrize pass used to write
   the fixed-rank form by hand. RCC never noticed, because it folds only a
   rank-6 block. The first rank-4 fold failed to compile with 1,245 errors. Both
   writers now call `_accessor`.

## Gates

- **Value (Python), `UCCOrbitFoldTests`.** Composes adapt → fold → factorize →
  symmetrize, exactly as `print_cpp_planck` does, and compares each folded
  block with the unfolded residual. CCSD and CCSDT are exact to 1e-10.
  Mutation: flipping the sign of one odd element of the order-36 group fails
  exactly `triples_aaaaaa`.
- **End to end (C++).** `b_ucc{2,3,4}_sto3g`, `b_uccsd_sto3g`,
  `b_uccsdt_sto3g`, `oh_uccsd_t_631g`, `water_cation_uccsd_t_631g` and
  `water_cation_uccsd_t_spherical_ccpvdz` pass on the folded build.

- **Emit-shape tests restated, not loosened.** Two tests in
  `test_ucc_loop_bounds.py` assumed the unfolded term set. One pinned the
  pre-fix loop-bound totals (1182/1192). It now asserts what those totals stood
  for: every emitted loop carries exactly one spin-resolved bound. The other
  relied on the CCSDT triples exceeding the 256-term chunk threshold, and after
  the fold none do. It now lowers the threshold so the `_partN` emission site is
  still exercised. Its own guard (`test_chunking_actually_fires`) is what caught
  that the class had gone vacuous. The full ccgen sweep before these fixes:
  804 passed, 2 failed (these two).

A process note: an earlier run of these gates passed against a stale binary.
The build had failed on the accessor defect, `make` left the previous binary in
place, and the runner ran that one. Check `make`'s exit status and the binary's
timestamp before trusting a gate run after a rebuild.

## Not done

- **Cost of the order-576 symmetrize pass at size.** It does 576 reads per
  element of an o⁴v⁴ array. That is not the bottleneck on B/STO-3G, where rank 4
  sped up most, but it has not been profiled on a larger system.
- **RCC doubles** (joint (ia)(jb) swap, order 2). Same mechanism, different
  group source; not measured.
- **Spin-resolved `tensor_symmetries`** (more operator merges) and **packed
  same-spin amplitude storage**. Both are independent of the fold.
