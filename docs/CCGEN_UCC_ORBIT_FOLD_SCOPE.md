# Folding the UCC Residuals by Their External Symmetry (scope)

Canonical status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.

**Question.** Can each generated UCC residual block be emitted as one term per
orbit of its external index symmetry, followed by one symmetrize pass, as the
spin-adapted RCC `aab` triples block already is? What does it save, and what
must hold for it to be exact?

## Short answer, measured before any code (2026-11-22)

Yes, for every block whose free indices include two of the same space and spin.
On CCSDT this cuts the residual's terms from 2,484 to 802. Modelled residual
FLOPs fall 3.5× on the derived route (o = 5, v = 20). The fold needs one new
helper (the group comes from the indices' spins), one emitter fix, and one
convention it makes load-bearing.

| block | group | terms | modelled FLOPs | symmetry of R | fold + symmetrize vs R |
|---|---|---|---|---|---|
| singles, `doubles_abab` | trivial | unchanged | 1.00× | — | — |
| `doubles_aaaa` / `bbbb` | (ij)(ab), order 4 | 118 → 56 | 1.85× | 3.1e-16 / 4.9e-16 | 3.9e-16 / 6.2e-16 |
| `triples_aabaab` | (ij)(ab), order 4 | 469 → 202 | 2.24× | 1.1e-15 | 1.3e-15 |
| `triples_abbabb` | (jk)(bc), order 4 | 469 → 192 | 2.36× | 8.2e-16 | 1.5e-15 |
| `triples_aaaaaa` / `bbbbbb` | S3(ijk) × S3(abc), order 36 | 579 → 72 | 7.47× | 6.2e-16 / 9.7e-16 | 1.8e-15 |

CCSDT, whole residual: undressed 1.05e13 → 2.81e12 (3.74×); derived
4.69e10 → 1.35e10 (3.49×). CCSD: 322 → 214 terms, derived 1.63×. Symmetry and
error columns are max-abs relative, on `ucc_random_tensors` with
noa/nva/nob/nvb = 3/5/4/6.

The operator count does not change (333 → 333 at CCSDT): the surviving orbit
representatives use the same contraction shapes. So after the fold the
**builders** are the larger cost: 3.3e10 flat-definition FLOPs against the
residual's 1.35e10. That figure overstates them, because builder bodies are
emitted factored (M3.0); measure before acting on it.

Probe: `fold_probe.py` (scratch, not committed). It derives each block's group
from its free indices' (space, spin) classes, checks R's symmetry, folds with
the existing `fold_external_symmetry`, and checks the fold plus symmetrize
against R.

## Where the logic lives

- `ccgen/optimization/permutation.py`: `fold_external_symmetry` (generic, takes
  any closed signed group) and `AAB_TRIPLES_GROUP` (RCC's one hand-written group).
- `ccgen/generate.py`, `print_cpp_planck`: folds only `if spin_adapt and
  "triples" in eqs`, before factorization, and passes `symmetrizers` to the
  emitter.
- `ccgen/emit/planck_tensor_cpp.py`: `_symmetrize_lines` (the C++ pass, also
  read back by the numpy evaluator through its `// symmetrize` comment), applied
  on both the inline and the chunked kernel paths.

## What invariants matter

1. **The group is derived from spins, never written by hand.** A block's group
   is the product, over each (space, spin) class of its free indices, of that
   class's symmetric group, signed by parity. That gives order 1, 4 or 36 at
   ranks ≤ 3 and order 576 for `aaaaaaaa` at rank 4. The fold is only exact for
   a symmetry R actually has. A hand-written table per block is how a wrong
   entry gets in (`AAB_TRIPLES_GROUP`'s comment records measuring its own).
   The group must never swap a mixed-spin pair: `abab` → `baba` is a different
   block, the same error as a tag-blind merge (`CCGEN_TWO_DRESSING_ROUTES.md`,
   invariant 3).
2. **Exactness rests on the amplitudes being antisymmetric within each
   same-spin class.** R has the symmetry only if T does. Today nothing enforces
   that: `CCGEN_UNRESTRICTED_CC.md` lists it as an unpinned convention, measured
   to ~1e-16. Folding makes it load-bearing. Without the fold, a T that drifts
   from antisymmetry gives an R that drifts with it. With the fold, the
   symmetrize pass projects R onto the symmetric part, so the two can differ.
   Jacobi keeps T antisymmetric if R and the denominators are (both are). The
   risky entry points are seeds: the MP2 guess, `.ccamp` restart and the W6
   cross-rank seed. Step 1 checks them, and the durable amplitude gate already
   in Open Work is the right home for the check.
3. **Fold before factorizing.** As on RCC, so no operator is derived for a term
   that folds away.
4. **The symmetrize pass is O(|G| · n_block), negligible next to the residual.**
   Even the order-36 pass is 36 reads per element of an o³v³ array. It is not
   free at rank 4 (order 576 over o⁴v⁴). Measure it there before assuming.

## Steps

0. **Emitter: spin-resolved bounds in the symmetrize pass.** `_symmetrize_lines`
   uses `_loop_bound(idx, None)`, which gives spin-free `no`/`nv`, so a UCC
   pass would not compile. Give it the result's spins (the kernel already has
   `result_spins`). No change for RCC, which passes None as today.
1. **Group helper + fold.** `spin_class_group(term)` in `permutation.py`.
   `print_cpp_planck` folds every UCC target whose group has order > 1 and
   passes the groups as `symmetrizers`. Before wiring, check that the MP2 guess,
   `.ccamp` seeds and W6 seeds are antisymmetric per same-spin class (invariant 2).
2. **Value gate.** Extend `UCCRewritePreservesValueTests` (or a sibling class) to
   evaluate fold + symmetrize + factorize against the unfolded residual, per
   block, at CCSD and CCSDT. Mutation: flip the sign of one group element. The
   numpy evaluator already applies the `// symmetrize` line, so the emitted-text
   gates (`test_merged_call_sites`, `test_factorize_all_binary`) cover the C++
   form once they run on UCC.
3. **Rank 4, measured not assumed.** Fold ratio, symmetrize cost and generation
   time for `ccsdtq` UCC (generation is ~10 min). Rank 3 has failed to predict
   rank 4 before in this codebase.
4. **End to end.** `b_ucc{2,3,4}_sto3g` (`ucc4 == FCI` is the strongest),
   `oh_uccsd_t_631g`, `water_cation_uccsd_t_631g`,
   `water_cation_uccsd_t_spherical_ccpvdz`. Energies to the gates' tolerances,
   not bitwise: the fold reassociates sums. Then time against the dressed,
   unfolded build.

## Out of scope

- **RCC doubles.** The spatial doubles block is symmetric under the joint swap
  (ia)(jb), order 2. It is the same mechanism but a separate measurement, and
  RCC's group comes from spatial pairs, not spins.
- **Spin-resolved `tensor_symmetries`** (more operator merges). Independent of
  this; it changes builders, not residual terms.
- **Packed storage of same-spin amplitudes** (store only i<j, a<b). A larger
  memory win, but it changes the runtime's tensor layout; the fold does not
  need it.
