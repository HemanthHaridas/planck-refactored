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
| R1 | Doubles residual on j ≥ i only, then mirror | done: CCSD 1.4–1.56× per iteration | R0 |
| R2 | Merge terms sharing a t2 read | 31 → 15 o³v³ contractions | R0 |
| R2b | Cut builder arithmetic | builders 0.88 × 10¹⁰ toward PySCF's share | R2 |
| R3 | GEMM lowering of the contractions | the 8.1× rate gap | R1, R2, R2b |
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

**Projection after R1 + R2** (written for the fold; the half range that
landed for R1 gives the same factor on every residual nest): o³v³
~0.95 × 10¹⁰, ladder 0.63 × 10¹⁰, other residual ~0.18 × 10¹⁰, builders
0.88 × 10¹⁰ (untouched): about 2.6 × 10¹⁰, 1.9× PySCF. The builders are then the largest remaining excess
(four `build_W_t2v_oovv_*` at 1.19 × 10⁹ each).

**Order decided:** R1 and R2 first, since they remove arithmetic R3 would
otherwise lower; then R3 for the 8.1× rate gap, which grows with v (4.0×
at cc-pVDZ, 8.1× at cc-pVTZ), as expected of GEMM against scalar loops.

### R1. Use the (ia)(jb) symmetry — done, as an output half range

**What landed.** The spatial RCC doubles residual is computed on j ≥ i
only and mirrored, R(j,i,b,a) = R(i,j,a,b), which is PySCF's i ≥ j.
`_pair_half_kernel` (`python/ccgen/emit/planck_tensor_cpp.py`) starts every
nest's free `j` loop at `i` and adds one `// mirror ijab` pass before the
return; `emit_planck_translation_unit` applies it to the `doubles` target
when spin-adapted and not UCC (under UCC the swap maps `abab` to `baba`,
another block). It also reaches the doubles residual of the generated
CCSDT/CCSDTQ files. GCC refuses `schedule` on a non-rectangular collapsed
loop, so those nests drop it; each element is written by one iteration, so
the schedule does not matter.

- Exact because t2 keeps t2(i,j,a,b) = t2(j,i,b,a): the MP2 start has it,
  Jacobi divides by symmetric denominators, DIIS takes linear
  combinations.
- Work: (o+1)/(2o) of every residual nest, the `vvvv` ladder included
  (55 % at o = 10). Builders are untouched.
- The numpy evaluator in the ccgen tests ignores loop bounds, so its value
  gates see the full residual; the half range is gated by the C++ energies.

**Result** (4 threads, energies identical to every printed digit):

| Rung | CCSD per iteration, before | after |
|---|---|---|
| water cc-pVTZ | 16 × 0.195 s | 16 × 0.136 s (1.43×) |
| water dimer cc-pVDZ | 20 × 0.454 s | 20 × 0.301 s (1.51×) |
| water dimer cc-pVTZ | 18 × 11.8 s | 18 × 7.55 s (1.56×); whole run 278 → 201 s |

Gates: core suite 178/178; `water_rccsd_t_{sto3g,631g,spherical_ccpvdz}`,
`h2o2_c1_rccsd_t_631g`, `water_spherical_ccsd_sto3g`,
`be_ccsd_spherical_631gd`, `be_rccsdtq_sto3g` on `build-default`;
`{ch4,lih}_rccsdt_generated_sto3g`, `be_rccsdtq_sto3g`, `b_ucc{2,3,4}_sto3g`,
`oh_uccsd_t_631g` and two `*_rccsd_t_*` cases on the fully featured
`build/`. Against PySCF the dimer cc-pVTZ CCSD is now 23× per iteration
(was 36×).

**The scoped term fold does not work here** (ruled out). Folding the
doubles over `{e, (ia)(jb)}` with `fold_external_symmetry` cut the
arithmetic by 2 % (6.24 → 6.12 × 10¹⁰ on the cc-pVTZ dimer: 1 of 31 o³v³
and 2 of 56 o³v² statements). It matches orbits on the terms' text with
dummies as named, and a term and its image mostly differ in dummy names or
in which of t2's two equivalent spellings they read; the ladder is its own
image and has nothing to fold. The half range needs no orbit matching and
covers every term.

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

### R2b. Cut builder arithmetic

After R1 + R2 the intermediate builders are the largest arithmetic excess
over PySCF: 0.88 × 10¹⁰ multiply-adds on the dimer in cc-pVTZ (R0), a
third of the projected 2.6 × 10¹⁰, with the four `build_W_t2v_oovv_*` at
1.19 × 10⁹ each. Neither R1 nor R2 touches them.

- Census first: per builder, its multiply-adds, its readers, and whether
  it is a transpose or (ia)(jb) image of another builder (the
  `merge_transposes` pass already folds transposes; check what it misses
  on the rank-2 TU).
- Candidates, decided by the census: build the `t2v_oovv` family once
  and read it through a permutation; use the (ia)(jb) half range on
  builders whose output has that symmetry; inline a builder into its single
  reader when building it costs more than the contraction it feeds.
- Gate: as R2 (value test of the residual against the unmodified builders),
  plus the R1 gates.

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
