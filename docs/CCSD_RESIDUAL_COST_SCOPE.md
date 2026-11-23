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
| R0 | Census and measurement | done: 4.5× arithmetic × 8.1× rate; R1, R2, then R3 | — |
| R1 | (ia)(jb) fold of the doubles residual | up to 2× on non-symmetric nests | R0 |
| R2 | Merge terms sharing a t2 read | 31 → 15 o³v³ contractions (≈ 8 with R1) | R0 |
| R3 | GEMM lowering of the contractions | the 8.1× rate gap | R1, R2 |
| R4 | Re-measure the ladder against PySCF | decides on the last resort | R1–R3 |

### R0. Census and measurement — done

**The 36× is two factors of similar size: 4.5× more arithmetic and 8.1×
slower arithmetic.** Iteration counts are not the cause.

| Water dimer, per iteration, 4 threads | Planck | PySCF 2.15 | ratio |
|---|---|---|---|
| cc-pVTZ iterations × time | 18 × 11.8 s | 23 × 0.32 s | 36× per iteration |
| cc-pVTZ multiply-adds | 6.24 × 10¹⁰ | 1.38 × 10¹⁰ | 4.5× |
| cc-pVTZ rate | 5.3 G/s | 43 G/s | 8.1× |
| cc-pVDZ iterations × time | 20 × 0.45 s | 21 × 0.019 s | 24× |
| cc-pVDZ multiply-adds | 2.69 × 10⁹ | 4.6 × 10⁸ | 5.9× |
| cc-pVDZ rate | 6.0 G/s | 24 G/s | 4.0× |

Water cc-pVTZ: 16 × 0.19 s against 14 × 0.021 s, 9× per iteration.

**How counted.** Planck: `tests/benchmarks/ccsd_residual/
count_generated_fma.py` multiplies the loop bounds around every
accumulation statement in the generated rank-2 TU (doubles and singles
residuals and every `build_W_*` builder). PySCF (`cc.RCCSD` is the
optimized `ccsd.CCSD`): `tests/pyscf/ccsd_update_amps_count.py` wraps
`lib.einsum`, `lib.ddot`, `lib.dot`, `numpy.dot` and `numpy.einsum` over
one `update_amps` and adds the C `vvvv` ladder from its formula
o(o+1)/2·v⁴.

**The count is a good model of the time.** Shares of the Planck CCSD:

| | dimer cc-pVTZ count | profile | dimer cc-pVDZ count | profile |
|---|---|---|---|---|
| 31 o³v³ nests | 59 % | 60.4 % | 63 % | ~67 % |
| `vvvv` ladder (o²v⁴) | 20 % | 20.1 % | 7.7 % | ~9 % |
| builders | 14 % | 9.6 % | 17 % | ~12 % |
| other residual terms | 5.5 % | 4.2 % | | |

**Where the extra arithmetic is** (dimer cc-pVTZ):
- Ladder: 1.26 × 10¹⁰ against PySCF's 6.9 × 10⁹. PySCF contracts i ≥ j
  only, which is the (ia)(jb) symmetry: R1 closes this 1.8×.
- o³v³: 3.7 × 10¹⁰ in 31 statements, and builders 8.8 × 10⁹, against
  6.8 × 10⁹ for everything else in PySCF.
- The 31 o³v³ statements fall into **15 merge groups** once t2 reads are
  compared modulo t2(p,q,r,s) = t2(q,p,s,r) (18 on literal reads). Many
  groups pair up under (ia)(jb) on the output, so R1 + R2 should leave
  about 8 o³v³ contractions.

**Projection after R1 + R2:** o³v³ ~0.95 × 10¹⁰, ladder 0.63 × 10¹⁰,
other residual ~0.18 × 10¹⁰, builders 0.88 × 10¹⁰ (untouched): about
2.6 × 10¹⁰, 1.9× PySCF. The builders are then the largest remaining excess
(four `build_W_t2v_oovv_*` at 1.19 × 10⁹ each).

**Order decided:** R1 and R2 first, since they remove arithmetic R3 would
otherwise lower; then R3 for the 8.1× rate gap, which grows with v (4.0×
at cc-pVDZ, 8.1× at cc-pVTZ), as expected of GEMM against scalar loops.

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
- After R1 and R2 (R0).
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
