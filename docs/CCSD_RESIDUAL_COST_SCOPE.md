# Why does the generated RCCSD lose to PySCF by a factor that grows with size, and what in the generator closes it?

Scope, opened 2026-11-23. Status lives in `vault/Status/Open Work.md`
(priority 1, CCSD(T) size). Parent: `docs/CCSD_T_SIZE_SCOPE.md` step 9.
Background on the generated route: `docs/CCGEN_GENERATED_CC_COST.md`.

## The question

`correlation ccsd(t)` runs the generated spatial RCCSD. Against PySCF 2.15
RCCSD (4 threads, same geometries and basis files, energies equal):

| Rung | o / v | Planck CCSD | PySCF CCSD | ratio |
|---|---|---|---|---|
| water cc-pVDZ | 5 / 19 | 0.19 s | 0.1 s | ~2× |
| water cc-pVTZ | 5 / 53 | 3.1 s | 0.2 s | ~16× |
| water dimer cc-pVDZ | 10 / 38 | 9.1 s | 0.4 s | ~23× |
| water dimer cc-pVTZ | 10 / 106 | 213 s | 7.4 s | 29× |

The CCSD is now 77 % of the dimer cc-pVTZ `ccsd(t)` run (278 s).

**Constraint: the fix goes in the generator.** A hand-written spatial
RCCSD (PySCF `rccsd` form) is the last resort, taken only if the
conditions in "Last resort" hold after every generator step below.

## What the profile shows

`sample` over a whole water dimer / cc-pVDZ `ccsd(t)` run (4 threads,
before the (T) fix of size step 8, so (T) is in the totals):

| Share of all samples | What |
|---|---|
| 50.0 % | `compute_ccsd_doubles_residual` |
| 29.5 % | (T), since cut 5× |
| ~12 % | intermediate builders (`build_W_*`), none above 1.5 % |
| 0.7 % | singles residual |

Inside the doubles residual (104 parallel nests; 24,102 samples):

| Nest(s) | Shape | Samples |
|---|---|---|
| 31 nests (9–18, 40–48, 73–84) | o³v³: `t2 × X` summed over one occupied and one virtual | ~640 each, ~20,000 together |
| 19 | o²v⁴: `t2(i,j,c,d) · vvvv(a,b,c,d)`, the particle–particle ladder | 2,689 |
| 72 others | o³v², o²v³, o⁴v² | small |

So the residual is 31 separate o³v³ contractions plus the ladder, each a
scalar loop nest. PySCF's RCCSD does its o³v³ work in a few contractions,
each a BLAS GEMM.

Two counts on those 31 nests (from the generated text, not estimates):

- **6 are the same contraction with output indices swapped**: 25 distinct
  up to renaming. That is the (ia)(jb) symmetry of the doubles residual
  (measured 5.6e-16).
- **31 → 18 when terms that read t2 the same way are merged**, i.e. terms
  `t2(·) × X₁ + t2(·) × X₂ + …` with the same t2 access and the same index
  set on the other factor. Summing X₁ + X₂ + … first is o²v², then one
  o³v³ contraction. The 18 are literal t2 accesses; t2's own symmetry
  t2(i,k,a,c) = t2(k,i,c,a) should merge more (`t2(k,j,c,b)` and
  `t2(j,k,b,c)` are the same read in two groups).

## Steps

Each step is a generator change, gated before the next. Measure first,
as size step 8 taught: there, contiguous operand gathering bought 0–4 %
because the kernel was compute-bound, and counting the arithmetic found
the 6× redundancy that was the real lever.

| # | Step | Expected | Needs |
|---|---|---|---|
| R0 | Census and measurement | decides the order of R1–R3 | — |
| R1 | (ia)(jb) fold of the doubles residual | up to 2× on non-symmetric nests | R0 |
| R2 | Merge terms sharing a t2 read | 31 → ≤ 18 o³v³ contractions, fewer with t2 symmetry | R0 |
| R3 | GEMM lowering of the contractions | the constant factor PySCF gets from BLAS | R0 |
| R4 | Re-measure the ladder against PySCF | decides on the last resort | R1–R3 |

### R0. Census and measurement

- Per-iteration time on both sides (PySCF prints its iteration count at
  `verbose ≥ 4`); the ratios above are totals and include iteration count.
- Arithmetic per iteration from the generated nests' loop bounds, and the
  same from PySCF `rccsd.update_amps`' einsums, at the dimer cc-pVTZ sizes.
  That separates "more arithmetic" (R1, R2) from "slower arithmetic" (R3).
- A profile at the dimer in cc-pVTZ (v = 106), not only cc-pVDZ: the ladder
  is o²v⁴ against o³v³ and gains share with v.
- The R2 count with t2's (ia)(jb) symmetry applied.

### R1. Fold the doubles residual over (ia)(jb)

- The mechanism exists: `fold_external_symmetry`
  (`python/ccgen/optimization/permutation.py`) folds the triples residual
  over its order-4 group and `_symmetrize_lines` emits one pass at the end
  (`docs/CCGEN_GENERATED_CC_COST.md`, defect #9). The doubles group is
  order 2, `{e, (ia)(jb)}`.
- The cost doc judged it "not worth doing at these sizes"; that was
  measured on small rungs, before the gap was seen to grow.
- Gate: rank-2 energies to ≤ 1e-12 relative (folding reorders sums), the
  four `*_rccsd_t_*` cases, `planck-cc-*` tests, and the fold's numpy
  value test extended to doubles.

### R2. Merge terms that share a t2 read

- Distributivity: `Σ_n t2(P) · X_n(Q_n)`, with each `X_n` read over the
  same index set, becomes `t2(P) · Y` with `Y = Σ_n ±X_n` permuted to one
  layout. Y is o²v² work per iteration, against the o³v³ contraction it
  saves.
- Canonicalize t2 accesses modulo t2's (ia)(jb) symmetry before grouping.
- Where it lives: the factorization layer (`python/ccgen/optimization/
  factorize.py`) already builds derived operators from contraction
  structure; a "common operand" pass belongs there, not in the emitter.
- Gate: as R1, plus a value test that the merged residual equals the
  unmerged one term by term (the merged-call-site test pattern of
  `test_merged_call_sites.py`).

### R3. GEMM lowering

- Each remaining contraction (the o³v³ group and the o²v⁴ ladder) is a
  matrix product after reshaping: gather operands to contiguous matrices,
  multiply, scatter. Eigen is already a dependency; no BLAS is linked
  (`USE_BLAS` is the open B1 item in `docs/ERI_FOCK_BLAS_SCOPE.md`).
- `python/ccgen/tensor_ir.py` has an unused contraction-order / GEMM
  analysis; start there.
- Order relative to R1 and R2 is R0's call: if R0 shows the arithmetic
  count is close to PySCF's, R3 comes first.
- Gate: as R1; the summation order changes, so ≤ 1e-12 relative, not
  bitwise.

### R4. Re-measure

The size ladder and the PySCF comparison
(`tests/pyscf/ccsd_t_ladder_reference.py`), timing and footprint in
separate runs.

## Last resort

A hand-written spatial RCCSD is justified only if, after R1–R3:

1. Planck's CCSD is still more than 3× slower than PySCF's per iteration
   on the dimer in cc-pVTZ, **and**
2. the ratio still grows from the dimer in cc-pVDZ to the dimer in
   cc-pVTZ, **and**
3. R0's census shows the remaining gap is not something the generator can
   express (a contraction order or intermediate the factorizer cannot
   reach).

If it comes to that, the hand-written code replaces only the rank-2 RCC
residual; (T), UCC and higher ranks stay generated.

## Not in this scope

The (T) kernels (size scope; 8× behind PySCF after step 8), RI-CCSD,
frozen core, MPI, the UCC residual (same levers likely apply; measure it
separately), and iteration-count differences from DIIS.
