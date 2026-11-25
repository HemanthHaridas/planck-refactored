# What limits the size of a CCSD(T) calculation, and what removes it?

Status lives in `vault/Status/Open Work.md` (priority 1). Parent:
`docs/CCSD_T_SCOPE.md`. Background on the generated CC route:
`docs/CCGEN_GENERATED_CC_COST.md`.

**Constraint: fixes go in the generator.** A hand-written spatial RCCSD is
the last resort, under the conditions in "Last resort".

## Answer (2026-11-24)

**Memory limited the size first, and no longer binds at the water dimer in
cc-pVTZ.** What filled memory was the AO integral handling and avoidable
copies around it. Each phase now holds one large object:

| Phase | Largest object | Peak footprint |
|---|---|---|
| SCF | spherical AO ERI tensor, 1.38 GB (ns⁴) | 1.31 GB |
| CCSD | `vvvv`, 0.96 GB (v⁴), with the other blocks and amplitudes | 1.43 GB |
| (T) | `vvvv` and blocks, kept by the state | 1.14 GB |

The next memory wall is the ns⁴ AO tensor in the SCF and v⁴ `vvvv` in the
CCSD. Removing either is a different method (direct or RI integrals, an
AO-driven or RI `vvvv` contraction), not a fix.

**Time is the limit now.** Against PySCF 2.15 the gap used to grow with size
(CCSD ~2× → 29×, (T) ~10× → 47× from water cc-pVDZ to the dimer in
cc-pVTZ).
- **CCSD: closed to 2.7× per iteration, flat in the basis** (3.1× on the
  dimer in cc-pVDZ, 2.7× in cc-pVTZ). The gap was more arithmetic times
  slower arithmetic. Generator changes brought the arithmetic to 1.4×
  PySCF's and moved 99.6 % of it into Eigen GEMMs.
- **(T): 1.4× behind** after step 10: the t2 symmetry halved its blocks,
  and the `t2 · ovvv` blocks run as small GEMMs per triple, as PySCF's
  do.

| Water dimer / cc-pVTZ `ccsd(t)` (nb 116 sph / 130 cart, o 10, v 106) | |
|---|---|
| Total energy | −152.7012785674 Eh, equal to PySCF 2.15 to every printed digit |
| E_corr(CCSD), E(T) | −0.5642828768, −0.0160404999 |
| Peak footprint | 6.7 GB (did not finish) → 5.5 GB → 3.93 GB → 2.83 GB → **1.43 GB** |
| CCSD / (T) / whole run | 18 × 0.85 s / 8.25 s / **39.6 s**, 4 threads (512 s at step 7) |
| Against PySCF 2.15 (4 threads) | CCSD 0.85 vs 0.32 s per iteration (2.7×), (T) 8.25 vs 6.0 s (1.4×) |

## What changed

| # | Change | Where | Effect |
|---|---|---|---|
| 1 | One SCF copy of the ERI tensor; the spherical transform frees its input | `build_conventional_eri` (`src/scf/scf.cpp`), `transform_eri_cart_to_sph` | `885109ce`. Water cc-pVTZ Cartesian SCF 466 → 330 MiB |
| 2 | `transform_eri` batched over its leading index | `src/post_hf/integrals.cpp` | `25714026`. Removes two n1·nb³ / n1·n2·nb² intermediates (1.32 + 1.21 GB on the dimer); bitwise identical, no time cost |
| 3 | (T) reuses the CCSD's converged integral blocks | `compute_rccsd_t_energy(calculator, state)` | `94fb61b5`. E(T) bitwise unchanged; dimer (T) 379 → 354 s |
| 3b | Re-exec with libmalloc's large-block cache off | `src/base/malloc_env.h`, both `main`s | `22589e59`. Water cc-pVTZ peak footprint 624 → 262 MiB |
| 4 | Free the engines' thread-local ERI scratch after a one-shot tensor build | `_compute_2e` in `src/integrals/base.h` | `22589e59`. ~130 MiB on water cc-pVTZ |
| 6 | Spherical tensor built per shell quartet, never the Cartesian one | `RysQuad::_compute_2e_spherical` (`src/integrals/rys.cpp`) | `538430e9`. SCF peak nc⁴ + ns·nc³ (4.1 GB on the dimer) → ns⁴ (1.38 GB) |
| — | CC solver moves its state instead of copying it | `run_generated_arbitrary_order_iterations` | `5be4a29e`. One copy of every integral block fewer (963 + 91 MiB on the dimer). Water cc-pVTZ peak live 245 → 186 MiB |
| 5 | Free the AO tensor before the CC iterations | `Correlation::release_eri`, called in `rccgen.cpp` and `uccgen.cpp` | `d9f5fb69`. Dimer peak footprint 2.83 → 1.43 GB |
| 7 | Re-measure the size ladder against PySCF | "Evidence" | Time, not memory, is the limit |
| 8 | (T) W computed once per virtual triple for all six orderings | `_emit_triple_kernel` (`python/ccgen/methods/ccsd_t.py`), `compute_ccsd_t_W_triple` | (T) 5.0–5.7×: dimer cc-pVTZ 283 → 49.9 s |
| 9 | Generated RCCSD residual: half range (R1), common-operand merges (R2, R2b), GEMM lowering (R3) | "CCSD time" | CCSD 13.9× per iteration: dimer cc-pVTZ 11.8 → 0.85 s; whole run 278 → 81 s |
| 10a | (T) blocks keyed up to the t2 pair swap: 24 → 12 per triple | `_emit_triple_kernel`, `_swap_t2_pair` (`python/ccgen/methods/ccsd_t.py`) | (T) 1.7–1.9×: dimer cc-pVTZ 49.8 → 26.8 s; whole run 81 → 58.5 s |
| 10b | (T) `t2 · ovvv` blocks as small GEMMs, two per shared t2 slice | `_emit_triple_kernel` | (T) 1.7–3.2×: dimer cc-pVTZ 26.5 → 8.25 s; whole run 57.8 → 39.6 s |

Energies are identical to every printed digit after every change. The core
suite passes (178/178) with all of them.

## Memory

**Step 5** frees `calculator._eri` right before a generated CC route's
iterations. The iterations and (T) read only the CC blocks, and every
rank's `prepare`, the warm-start recursion's included, runs before any rank
iterates. A later `ensure_eri` rebuilds the tensor, so a missed reader
would cost time, not correctness. Both routes are single-point only.
Gated by the RCC cases on `build-default` and the UCC and generated CCSDT
cases (`b_ucc{2,3,4}`, `*_uccsd_t_*`, `*_rccsdt_generated_*`) on the fully
featured `build/`.

**Step 6.** Each canonical shell quartet's Cartesian block comes from the
same kernels `_compute_2e` uses (Auto per quartet, HGP or Rys hoisted blocks,
per-component OS). It gets the same per-component Schwarz screen, then four
one-index transforms with the shells' blocks of `C`. Only canonical
spherical elements are written, with their 8 permutations, so the tensor is
exactly symmetric. A canonical element can have its ket shell pair before
its bra pair when both start on the same shell, (P1 P2 | P1 P1) holding
(3,6|4,4), so the ket loop starts at (sa, sa). The builder checks that `C` is
block-diagonal by shell instead of assuming it. With integral symmetry
operations, which act on Cartesian AOs, it falls back to
build-then-transform. Gated in `planck-compute-2e`: equal to
build-then-transform to ≤ 6.7e-16 (cc-pVDZ, all four engines) and 2.1e-15
(cc-pVTZ, Auto), against integrals up to 4.9, and exactly 8-fold symmetric.
The gate's first run caught the missing (sa, sa) kets (4592 elements, all
unwritten). The SCF's spherical build no longer has the transform's hidden
150-Cartesian-AO limit, matching `ensure_eri` and the Cartesian build.

**Where the memory was** (dimer cc-pVTZ, live blocks ≥ 1 MiB during the
CCSD, from the profiler in "Tools"):

| MiB | Object | After 3b, 4, 6 | + move fix | + step 5 |
|---|---|---|---|---|
| 2179 | spherical AO tensor in the recycled Cartesian block (1448 requested) | 1381 | 1381 | gone |
| 963 | `vvvv`, `rebind_physicist` | 963 | 963 | 963 |
| 963 | `vvvv` copy, `TensorCCBlockCache` copy in the solver | 963 | gone | gone |
| 91 + 91 | `ovvv` and its copy | 91 + 91 | 91 | 91 |
| ~140 | thread-local ERI scratch | gone | gone | gone |
| | **peak live / footprint** | 3.89 / 3.93 GB | 2.80 / 2.83 GB | 1.41 / 1.43 GB |
| | **through (T)** | 2.49 GB live | 2.49 GB live | 1.11 GB live |

Before 3b–6 the SCF also held the Cartesian tensor (2179 MiB) with the
transform's first quarter (1944 MiB), 4.26 GB at ~23 s.

Water cc-pVTZ `ccsd(t)` through the changes:

| | peak live | peak footprint |
|---|---|---|
| after step 3 | 424 MiB (`malloc_size`) | 624 MiB |
| + 3b, 4 | 245 MiB | 262 MiB |
| + 6 | 245 MiB (SCF phase 235 → 185) | 260 MiB |
| + move fix | 186 MiB | 213 MiB |

**Findings.**
- **macOS libmalloc held up to half the footprint.** It keeps a freed large
  block dirty in its large-block cache and returns it whole for a later,
  smaller request. The freed 2.18 GB Cartesian tensor became the 1.45 GB
  spherical tensor's block, pinning a dirty 0.73 GB tail for the whole run.
  A standalone C program reproduces it: freeing a touched 1000 MiB block
  leaves the footprint at 1001 MiB, and a 632 MiB request gets the same
  pointer back.
  - Ruled out: `malloc_zone_pressure_relief(NULL, 0)` (releases 0 bytes),
    `madvise(MADV_FREE_REUSABLE)` or `MADV_FREE` before `free` (footprint
    unchanged), `MallocXzoneEnabled=0` (no change).
  - Works: `MallocLargeCache=0`, read only at process start, hence the
    re-exec. A user-set value is kept, a failed exec continues unchanged,
    `planck-mpi` does not re-exec (unsafe under an MPI launcher). No cost:
    water cc-pVTZ 5.82–5.86 s with the cache, 5.87–5.92 s without.
  - Linux: a no-op. glibc serves blocks above its mmap threshold (at most
    32 MiB) with `mmap` and returns them on `free`. From glibc's documented
    defaults, not measured.
- **The live-byte total read off a profile depends on `malloc_size`.** An
  earlier profile concluded the CCSD's ≥ 1 MiB allocations summed to less
  than the SCF peak (4.26 GB) and looked for small allocations. They summed
  to 5.34 GB, because the spherical tensor occupied a 2179 MiB block. Small
  allocations were under 20 MiB of 5.36 GB live. Track requested size next
  to `malloc_size` and the reuse shows as slack.
- **An earlier attribution of the dimer's peak to (T) was wrong.** (T)
  rebuilt every integral block (step 3 removed that), but the footprint
  peak had already been reached in the CCSD, at 73 s.
- **Measure footprint, not RSS, near the RAM limit.** On a paging run RSS
  read 0.1–1.3 GiB while the footprint was 5–6.7 GB; an RSS watchdog never
  fired, and `/usr/bin/time` max-RSS understates the same way.
- **`vmmap` pauses the process it inspects** (water cc-pVTZ 6.3 → 7.9 s at
  a 2 s poll). The profiler reads `task_info` in-process instead.
- **The spin-orbital RCCSD was the first limit.** `run_rccsd` works in 2o
  and 2v spin orbitals: 16 v⁴ `vvvv`, 16 o²v² t2, a 64 o²v⁴ leading
  contraction. Moving `ccsd(t)` to the generated spatial RCCSD removed it
  ("Evidence").

## CCSD time (step 9)

### Where the gap came from (R0)

Measured per iteration, 4 threads, against PySCF 2.15 RCCSD (`cc.RCCSD`, the
optimized `ccsd.CCSD`), same geometry and basis files, energies equal,
before any of step 9:

| Water dimer | Planck | PySCF | ratio |
|---|---|---|---|
| cc-pVTZ (o 10, v 106): iterations × time | 18 × 11.8 s | 23 × 0.32 s | 36× per iteration |
| cc-pVTZ multiply-adds | 6.24 × 10¹⁰ | 1.38 × 10¹⁰ | 4.5× |
| cc-pVTZ rate | 5.3 G/s | 43 G/s | 8.1× |
| cc-pVDZ (o 10, v 38): iterations × time | 20 × 0.45 s | 21 × 0.019 s | 24× |
| cc-pVDZ multiply-adds | 2.69 × 10⁹ | 4.6 × 10⁸ | 5.9× |
| cc-pVDZ rate | 6.0 G/s | 24 G/s | 4.0× |

Two factors of similar size, then: more arithmetic and slower arithmetic.
Iteration counts are not the cause.

**How counted.**
- **Planck:** `tests/benchmarks/ccsd_residual/count_generated_fma.py`
  multiplies the loop bounds around every accumulation in the generated
  rank-2 TU (residuals and every `build_W_*`). It counts a `j = i` loop as
  half. Lowered contractions are not loops, so it counts only what stays in
  loops after R3.
- **PySCF:** `tests/pyscf/ccsd_update_amps_count.py` wraps `lib.einsum`,
  `lib.ddot`, `lib.dot`, `numpy.dot` and `numpy.einsum` over one
  `update_amps`, plus the C `vvvv` ladder, o(o+1)/2·v⁴.

**The count models the time.** Its shares of the CCSD matched a `sample`
profile within a few percent:

| Dimer cc-pVTZ | count | profile |
|---|---|---|
| 31 o³v³ nests (`t2 × X`, one occupied and one virtual summed) | 59 % | 60.4 % |
| `vvvv` ladder (o²v⁴) | 20 % | 20.1 % |
| builders | 14 % | 9.6 % |
| other residual terms | 5.5 % | 4.2 % |

The extra arithmetic was the ladder over all (i, j) where PySCF takes i ≥ j
(1.8×), and 31 separate o³v³ contractions where PySCF has a few.

### What closes the arithmetic (R1, R2, R2b)

Multiply-adds per iteration on the dimer in cc-pVTZ, × 10¹⁰:

| | o³v³ | ladder | other residual | builders | total | vs PySCF |
|---|---|---|---|---|---|---|
| start | 3.70 (31 nests) | 1.26 | 0.40 | 0.88 | 6.24 | 4.5× |
| + half range (R1) | 2.03 | 0.69 | 0.24 | 0.88 | 3.85 | 2.8× |
| + t2 merge (R2) | 0.39 (6 nests) | 0.69 | 0.08 | 0.89 | 2.06 | 1.5× |
| + v merge, τ ladder (R2b) | 0.39 | 0.69 | 0.08 | 0.75 | 1.92 | 1.4× |

CCSD per iteration, each row measured back to back against the row above
with binaries built from the same tree (4 threads; CCSD and (T) energies
identical to every printed digit):

| | water cc-pVTZ | dimer cc-pVDZ | dimer cc-pVTZ |
|---|---|---|---|
| start | 0.195 s | 0.454 s | 11.8 s |
| + half range | 0.136 s (1.43×) | 0.301 s (1.51×) | 7.55 s (1.56×) |
| + t2 merge | 0.108 s (1.28×) | 0.216 s (1.38×) | 5.45 s (1.41×) |
| + v merge | 0.093 s (1.16×) | 0.209 s (1.02×) | 5.23 s (1.06×) |

Time fell less than the count after the first step: with every contraction
still a scalar loop, what remained was rate-bound, so cutting statements no
longer cut time proportionally. GEMM lowering addresses the rate.

**Half range on the doubles residual (R1).** The spatial RCC doubles
residual is computed for j ≥ i only and then mirrored,
R(j,i,b,a) = R(i,j,a,b), which is PySCF's i ≥ j.
- **Where.** `_pair_half_kernel` (`python/ccgen/emit/planck_tensor_cpp.py`)
  starts every nest's free `j` loop at `i` and adds one `// mirror ijab`
  pass. `emit_planck_translation_unit` applies it to the `doubles` target
  when spin-adapted and not UCC; under UCC the swap maps `abab` to `baba`,
  another block. It reaches the doubles of the generated CCSDT/CCSDTQ too.
- **Why exact.** t2 keeps t2(i,j,a,b) = t2(j,i,b,a): the MP2 start has it,
  Jacobi divides by symmetric denominators, and DIIS takes linear
  combinations.
- **Work.** (o+1)/(2o) of every residual loop nest (55 % at o = 10).
- **OpenMP.** GCC refuses `schedule` on a non-rectangular collapsed loop, so
  those nests have none; each element is written by one iteration, so it
  does not matter.
- **Gate.** The numpy evaluator in the ccgen tests ignores loop bounds and
  so sees the full residual; the half range is gated by the C++ energies.

**Common-operand merge (R2, R2b).** `merge_common_operand_reads`
(`python/ccgen/optimization/factorize.py`) rewrites Σ_n c_n A(P) R_n as
A(P) Y, where Y = Σ_n c_n R_n is an intermediate whose builder sums the
rests. `print_cpp_planck` runs it twice after the factorizer, when
spin-adapted and not UCC, on every non-energy target.
- **Grouping.** Two terms share a group when their A read is the same once
  summed indices are renamed in order of appearance (smallest of A's
  pair-swap spellings), and their rests carry the same index *set*. Each
  rest is permuted into Y's slots, so its own index order does not split a
  group.
- **Guards.** A rest may not sum an index of its own, and Y may not be
  larger than A. Without the second guard a t1·t2 rest becomes an o⁴v⁴ Y,
  and the total went to 3.4 × 10¹³.
- **A = t2 (R2)**, with the factorizer's W's as rests. On the rank-2
  doubles, 69 binary `t2 × X` terms become 12 contractions, and the 31
  o³v³ become 6. ccsdt: singles 36 → 23, doubles 142 → 85, triples
  331 → 189 terms.
- **A = v, after the t2 merge (R2b).** By then every term the t2 merge took
  reads t2·Y and no v, so the two passes cannot compete for a term. The
  pass sees through an intermediate read once and defined by one term
  (`single_use_specs`); if that reader merges, the intermediate is dead and
  `drop_unread_specs` removes it. On the rank-2 doubles exactly one group
  forms: the ladder t2(c,d,i,j)·v(a,b,c,d) and t1(d,j)·(t1(c,i)·v(a,b,c,d))
  become one contraction with τ = t2 + t1 t1. That removes the
  `W_t1v_ovvv` builder (o v⁴, 1.27 × 10⁹), the largest builder. In ccsdt
  that builder is also read by the triples, so it is not single-use and the
  ladder stays unmerged.
- **Gate.** `test_merged_amplitude_reads.py` emits ccsd and ccsdt with and
  without both passes, requires every residual to agree to 1e-10 relative
  on a fixture whose t2 has exactly the (ia)(jb) symmetry, and asserts a v
  merge on ccsd. Mutation-checked: one member's sign flipped (0.44
  relative), a wrong Y slot order (shape error), an inlined term's sign
  flipped (4.3e-3).

### What closes the rate (R3, GEMM lowering)

**The rates available**, G multiply-adds/s on the M3 (4 performance cores),
for the CCSD shapes on the water dimer, against about 5 for the scalar loop
nests and 43 for PySCF's iteration average:

| Shape | Eigen, 1 thread | Eigen, 4 threads | Eigen, caller splits rows (4) | Accelerate |
|---|---|---|---|---|
| o³v³, cc-pVTZ (1060³) | 20 | 63 | 61 | 154 |
| ladder, cc-pVTZ (55 × 11236²) | 16 | 31 | 30 | 114 |
| o³v³, cc-pVDZ (380³) | 21 | 57 | 49 | 201 |
| ladder, cc-pVDZ (55 × 1444²) | 17 | 37 | 37 | 134 |

Eigen and Accelerate agree to 2e-15 relative. Planck builds Eigen with
`EIGEN_DONT_PARALLELIZE`, but a caller-side split matches Eigen's own
threading, so that define is no obstacle. The benchmark source is not in
the repo; it is a 90-line program around `C.noalias() = A * B` and
`cblas_dgemm`.

**The helper.** `cc_gemm_contract` (`src/post_hf/cc/gemm_contract.h`)
computes out(...) += coeff · Σ_k A(m..., k...) B(k..., n...), given the
index groups m, k, n as label strings.
- **Operands in place or copied.** Each operand is described by per-group
  offset tables. If they show the storage already is the (rows, cols)
  matrix or its transpose, the operand is used in place; otherwise it is
  copied into one.
- **Fixed blocks.** The product runs in 64-wide blocks of the larger output
  dimension. Every element's summation order is therefore independent of
  the thread count, and results are bitwise the same at 1–8 threads.
- **Scatter.** The product is added into the output through the same
  offset tables.
- **`pair_half`.** A group holding both i and j keeps only j ≥ i. That is a
  rectangle when i and j share a group, as in the ladder (55 of 100 ij
  rows).

**The emitter.** `_gemm_lowering` (`python/ccgen/emit/planck_tensor_cpp.py`),
called from `emit_planck_term` for every non-UCC term, lowers a term when:
- it has two factors, at least one summed index, and five or more distinct
  indices (o²v³, o³v² and up; smaller nests stay loops);
- every summed index is in both factors and every free index in exactly one;
- each operand is one the helper uses in place, or has at most three
  virtual indices. That keeps any copy at o·v³ (the size of `ovvv`) or
  smaller, and `mo_blocks.vvvv` is never copied.

Group orders: k as stored in the operand with more virtual indices, m and n
as stored in their factors. `_pair_half_kernel` passes `pair_half` to the
lowered calls of the spatial doubles kernel. A TU includes the header only
when it uses it (`with_gemm_include`, shared with the (T) TU, whose
full-tensor reference kernel is lowered too). On the rank-2 TU 98
contractions are lowered; 7.3 × 10⁷ multiply-adds stay in loops, mostly the
merge-sum builders.

**Why one-index contractions are included.** Lowering only the two-index
sums left two builders, `W_t1v_oovv_{1336,390c}` (1.2 × 10⁸ multiply-adds
each), as strided loops that took about a third as many samples as all the
GEMMs together. They sum over one index, and their `ovvv` operand needs a
copy because the summed index sits in the middle of its layout.

**Result** (4 threads, back to back against the v-merge binary; CCSD and (T)
energies identical to every printed digit):

| | water cc-pVTZ | dimer cc-pVDZ | dimer cc-pVTZ |
|---|---|---|---|
| v merge, loops | 0.097 s | 0.211 s | 5.37 s |
| + GEMM lowering | 0.037 s (2.6×) | 0.060 s (3.5×) | 1.10 s (4.9×) |
| + ladder on i ≤ j | 0.038 s (1.0×) | 0.060 s (1.0×) | 1.089 s (1.01×) |
| + one-index contractions | 0.039 s (1.0×) | 0.059 s (1.0×) | 0.850 s (1.23×) |
| PySCF | 0.021 s | 0.019 s | 0.32 s |

The ladder on i ≤ j halves its arithmetic but hardly shows. As a GEMM the
ladder was no longer a large share of the time; the strided one-index
builders and serial stretches were (a `sample` profile of the dimer in
cc-pVTZ put about 27 % of thread time idle).

**Gate.** The value tests read the lowered calls too: the numpy evaluator
(`python/ccgen/tests/emitted_cpp_eval.py`) turns each `cc_gemm_contract`
call into the einsum it replaced. With `_gemm_lowering` returning None, every
emit-hash pin reproduces its previous value; the pins were restated with
that note.

## (T) time

**Step 8.** A profile of the dimer in cc-pVDZ put 99 % of (T) in the W
block kernel. The batched (T) evaluated W for each of the six orderings of
every virtual triple a ≥ b ≥ c, 24 terms each. Every term is an elementary
o³ block, one contraction of two factors, up to a renaming of i, j, k, and
over the six orderings the 144 evaluations hit only 24 distinct blocks (12
`t2 · ovvv`, o³v; 12 `t2 · ooov`, o⁴). The emitter canonicalizes each
(ordering, term) to a block key (factor texts with occupied externals
renamed by first appearance and virtuals by triple position), computes each
block once, and adds it to every ordering with its occupied indices
transposed. Each block is the same sum in the same order as the old
per-term accumulator and the terms are added in the same order, so E(T) is
unchanged (gates `ccsd_t_oracle_diff` and the four `*_rccsd_t_*` cases, 8
ccgen PySCF (T) tests). V stays per ordering (1 % of (T)). The UCC (T)
does not have this redundancy: its work items are already one ordering per
spin block.

**Step 10a.** The 24 blocks pair up under t2(i,j,a,b) = t2(j,i,b,a). The
six ooov pairs are the same block (t2(u0,l,v1,v2) and t2(l,u0,v2,v1) read
one value); the six ovvv pairs are the same block with u0 and u1 swapped.
`_emit_triple_kernel` keys each term by the smaller key over the two
spellings of its t2 read (`_swap_t2_pair`), and the occupied renaming by
first appearance then carries the transposition. 12 blocks per triple,
PySCF's count. E(T) identical to every printed digit; gates
`ccsd_t_oracle_diff` through `water_rccsd_t_631g`, the four `*_rccsd_t_*`
cases, the 8 ccgen PySCF (T) tests.

| (T), 4 threads | 24 blocks | 12 blocks |
|---|---|---|
| water cc-pVTZ | 0.370 s | 0.219 s (1.69×) |
| dimer cc-pVDZ | 0.790 s | 0.461 s (1.71×) |
| dimer cc-pVTZ | 49.8 s | 26.8 s (1.86×) |

Less than 2× because V and the energy loops per triple do not shrink.

**Step 10b.** PySCF's `get_wv` calls a small `dgemm` per triple; Planck now
does the same for the six `t2 · ovvv` blocks. Per triple a block is
C(u0 u1, u2) = Σ_d t2(u0, u1, d, vX) ovvv(u2, d, vY, vZ), a
(o² × v) · (v × o) product. Blocks reading the same t2 slice share one GEMM
with their ovvv columns side by side, so the six are three GEMMs of
o² × v × 2o (100 × 106 × 20 on the dimer in cc-pVTZ). The strided operands
are copied into thread-local contiguous buffers first (about 10 % of the
multiply), and the result is copied into the block slots. Sizes are fixed,
so E(T) does not depend on the thread count. The six `t2 · ooov` blocks
(o⁴, about 10 % of W) stay loops.

| (T), 4 threads | loops | ovvv blocks as GEMMs |
|---|---|---|
| water cc-pVTZ | 0.216 s | 0.115 s (1.9×) |
| dimer cc-pVDZ | 0.459 s | 0.266 s (1.7×) |
| dimer cc-pVTZ | 26.5 s | 8.25 s (3.2×) |

E(T) identical to every printed digit; the same gates as 10a.

**(T) was compute-bound, not memory-bound.** Gathering each W term's
operands into contiguous buffers (both were strided along the summed index,
by nv and nv² for `t2 · ovvv`) changed (T) by 0–4 % (water cc-pVTZ 1.92 s,
dimer cc-pVDZ 4.04 → 3.88 s). Counting work explained it: ~34 G
multiply-adds on the dimer in cc-pVDZ, run at ~9 G/s, so the excess was
arithmetic. Reverted; step 8 removed 5/6 of it.

## Gates for every change here

- **Core suite** on `build/`.
- **CC cases on `build-default`:** `water_rccsd_t_{sto3g,631g,spherical_ccpvdz}`,
  `h2o2_c1_rccsd_t_631g`, `water_spherical_ccsd_sto3g`,
  `be_ccsd_spherical_631gd`, `be_rccsdtq_sto3g`.
- **CC cases on `build/`:** `{ch4,lih}_rccsdt_generated_sto3g`,
  `be_rccsdtq_sto3g`, `b_ucc{2,3,4}_sto3g`, `oh_uccsd_t_631g`,
  `water_rccsd_t_631g`, `h2o2_c1_rccsd_t_631g`.
- **The full ccgen suite.** Byte and hash pins of the emitted code
  (`test_emit_flag_matrix`, `test_ucc_emit_flag`) move with any emit
  change. Re-pin them only after showing that the old pins reproduce with
  the change switched off.

## Ruled out

- **Folding the doubles over {e, (ia)(jb)}** with `fold_external_symmetry`
  cut 2 % (1 of 31 o³v³ statements). It matches orbits on the terms' text
  with dummies as named, and a term and its image mostly differ in dummy
  names or in which t2 spelling they read. The half range needs no matching.
- **`factorize_tau`** collapses 0 spin-adapted doubles pairs: its validated
  pairs are spin-orbital shaped.
- **The v merge before the factorizer.** It forms PySCF-style groups (τ for
  both ladders, 2 t2 − t2ᵀ − t1 t1 for the rings), but the ring groups pull
  t2·v terms out of the t2 merge: o³v³ nests went 6 → 10 and the total rose
  to 2.19 × 10¹⁰. Running it after the t2 merge keeps only the ladder.
- **Contiguous operands for the (T) W kernel**: 0–4 % ("(T) time").

## Open, in order

| # | Step | Expected |
|---|---|---|
| 10c | (T): profiled (see 10c). Make the gathers, `ooov` blocks and assembly contiguous; batch over a virtual only if that falls short | (T) ~2–2.5×, past PySCF |
| R3b | CCSD o³v³ contractions on j ≥ i (staircase) | ~45 % of the six o³v³ GEMMs |
| R3c | Accelerate `dgemm` behind the GEMM helper | 2.5–4× on the GEMMs; a build decision |
| R2c | Share linearly dependent combinations of builders | 4 o³v³ builders → 3 contractions (≤ 1.06×) |
| R4 | Re-measure the size ladder against PySCF | timing and footprint in separate runs |

Beyond these: build the MO blocks from a direct or RI transform without
storing the AO tensor at all. RI-CCSD would be its own scope.

### 10c. What is left in (T)

**Profiled 2026-11-25** (water dimer cc-pVTZ,
`tests/benchmarks/ccsd_t/water_dimer_ccpvtz.hfinp`, the ladder's S22 rung,
4 threads, (T) 8.1–8.4 s; PySCF 6.0 s). A `sample` profile of the whole
run, aggregated over the four threads' (T) subtrees, gives W 89 %, V
blocks 6.7 %, the driver's energy loops 3.8 %. Inside W, thread-local
`steady_clock` marks inserted into the generated TU (by hand, then
removed; the build has no debug info for line attribution) give, out of
29.6 thread-seconds:

| W section | thread time | share | per triple |
|---|---|---|---|
| gathers into `ga`/`gb` (t2 slice o²v; six `ovvv(u2, d, vX, vY)` slices o·v, strided v² in d and v³ in u2) | 6.42 s | 22 % | 31 µs |
| GEMMs (three o² × v × 2o) | 7.91 s | 27 % | 39 µs |
| scatter into the block slots | 0.30 s | 1 % | |
| six `t2 · ooov` loop blocks (o⁴ each, `t2(l, u0, ..)` strided nv² in l) | 7.34 s | 25 % | 36 µs |
| assembly (per ordering, 24 reads of the blocks in transposed index orders) | 7.48 s | 25 % | 37 µs |

So the GEMMs are only a quarter of W: there is no single hot spot, and
batching the GEMMs alone would not pay. The `ooov` loops do a tenth of the
GEMMs' multiply-adds (6·10⁴ against 6.4·10⁵ per triple) in the same time,
and the assembly's 1.4·10⁵ adds take as long again. All three loop sections
are access-bound: strided accessors and transposed reads that do not
vectorize. Levers, each a generator change gated as in step 8:

- **Gathers:** read from layouts made once per (T): `ovvv` as
  [vX][vY][d][u2] (o·v³, 91 MB on the dimer) and t2 as [b][i][j][a]
  (9 MB), so each slice is contiguous.
- **`ooov` blocks:** blocks 4/5, 8/9 and 10/11 share one `ooov` slice each;
  gather the slice (o³) and the two t2 slices (o² each) contiguously and
  run (2o × o)·(o × o²), or a contiguous loop the compiler vectorizes.
- **Assembly:** have the scatter write each block in the index orders the
  orderings read (o³ copies, already 1 % for one order), so every
  ordering's sum is 24 contiguous adds.

If the loop sections ran at the GEMMs' rate, W would be about a third of
today and (T) roughly 2–2.5× faster, below PySCF's 6.0 s. The batch-over-c
restructure below attacks all three at once, at o³v² memory and a driver
change; take it only if the per-section levers fall short.

Original plan, kept for that case. If the small GEMMs themselves limit, batch over one virtual: for fixed c,
Σ_d t2(i,j,d,c) ovvv(k,d,a,b) over all (a, b) is a (o² × v) · (v × o v²)
GEMM, o³v³ per c and o³v² memory (90 MB on the dimer in cc-pVTZ). That
needs the (T) driver (`batched_triples_energy` in
`src/post_hf/cc/ccsd_t.cpp`) to loop over c outermost, compute W for every
(a, b) at that c, then accumulate the energy over a ≥ b ≥ c. A driver
restructure, not an emitter change. Gates as in step 8, plus thread-count
invariance of E(T).

### R3b. The CCSD o³v³ contractions on j ≥ i

In an o³v³ contraction the output matrix has rows (i, a) and columns
(j, b). The condition j ≥ i couples a row coordinate to a column coordinate,
so the kept region is a staircase (block upper-triangular in i, j), not a
rectangle. One GEMM call cannot express it, so today these contractions are
computed in full and the mirror overwrites j < i.

One GEMM per value of i does express it. Each takes rows (i, ·) (v rows),
columns (j ≥ i, ·) ((o − i)·v columns) and the full K = o·v. Summed over i
that is o(o+1)/2 · v² · K, the same half as the ladder: about 45 % less at
o = 10. The pieces stay large GEMMs (106 × up to 1060 × 1060 on the dimer
in cc-pVTZ). The emitter chooses the order within each group, so it can put
i first in its group and make the rows of one i contiguous. In the helper
this is a loop over i in place of the single product, when i and j are in
different groups. Expected gain: up to about 10 % of the CCSD arithmetic,
less in time.

### R3c. Accelerate behind the GEMM helper

Every lowered contraction goes through one product call in
`cc_gemm_contract`, so swapping Eigen for `cblas_dgemm` is a change in one
place: the optional `USE_BLAS` build, B1 in `docs/ERI_FOCK_BLAS_SCOPE.md`.
Its constraints:
- **Hermeticity.** Accelerate is always present on macOS; Linux needs a
  system or fetched OpenBLAS.
- **Bitwise reproducibility.** BLAS results are not bitwise equal to
  Eigen's. Accelerate is bitwise reproducible across thread counts;
  OpenBLAS/MKL are unmeasured.

A user decision, not a generator step.

### R2c. Share linearly dependent combinations of builders

**The case.** The four `W_t2v_oovv_{c3cc,68d5,6ae9,7b99}` are t2·v. With
the ERI pair swap v(l,k,d,c) = v(k,l,c,d) they form a 2 × 2 table:

| | v(k,l,c,d) | v(k,l,d,c) |
|---|---|---|
| t2(a,c,i,k) | c3cc | 68d5 |
| t2(c,a,i,k) | 6ae9 | 7b99 (j renamed to i) |

Only three of the t2 merge's sums read them, all on slots (i,k,a,c):

| sum | c3cc | 68d5 | 6ae9 | 7b99 | plus |
|---|---|---|---|---|---|
| `ab52bf` | 4 | −2 | −2 | 1 | 2 v reads, 2 cheap t1·v builders |
| `dedafd` | −2 | 1 | 1 | 0 | 1 v read, 1 cheap builder |
| `1e1c0d` | 0 | 0 | 0 | 1 | 1 v read, 1 cheap builder |

On those columns `ab52bf` = −2·`dedafd` + `1e1c0d`: rank 2. So two
combinations suffice:
- Z₁ = −2·c3cc + 68d5 + 6ae9, which is
  Σ_kc (−2 t2(a,c,i,k) + t2(c,a,i,k)) v(k,l,c,d) + t2(a,c,i,k) v(k,l,d,c):
  two contractions after o²v² sums;
- Z₂ = 7b99: one contraction.

Three contractions instead of four is the floor for this span, and it is
the structure of PySCF's hand-written `Wovvo`/`Wovov`. The common-operand
merge cannot find it: the saving is in the dependence *between* sums, and
inlining the W's into each sum separately would take 6 contractions.

**Mechanism.** A pass over the specs after the merges:
1. **Write the sums as vectors.** Every spec whose definition terms are all
   single-factor reads on one slot set becomes a coefficient vector over
   basis reads (factor name, permutation into the slots).
2. **Pick the expensive columns.** These are builders with a single binary
   definition costing more than |slots| × o.
3. **Row-reduce** those columns exactly (`Fraction`). Choose the Z's from
   the sums' own expensive parts, smallest support first, and express every
   sum through them.
4. **Build each Z** from its inlined t2·v terms, running the merge on v and
   on t2 and keeping the variant with fewer contractions.
5. **Rewrite the sums** to read the Z's; `drop_unread_specs` removes the
   W's.

**Risks.**
- `factored_builder_steps` reads only `definition_terms[0]`. A multi-term
  spec is emitted correctly only because ≤ 2-factor definitions take the
  flat path, which emits every term. A Z whose first term has 3 factors
  would lose its other terms silently, so assert against it.
- Smallest-support Z's are a heuristic; confirm the contraction count with
  the cost model.
- Rank 3 is unchecked. Take the census on ccsdt before claiming it there.

**Gate.** The value test, extended to assert that the four W's are gone;
the count (four o³v³ builders → three contractions); the gates above; A/B
timing.

**Order.** At most about 1.06× before GEMM lowering; re-measure its share
of the lowered build before taking it.

## Last resort

A hand-written spatial RCCSD is justified only if, after the GEMM lowering:

1. Planck's CCSD is still more than 3× slower than PySCF's per iteration on
   the dimer in cc-pVTZ, **and**
2. the ratio still grows from the dimer in cc-pVDZ to cc-pVTZ, **and**
3. the remaining gap is not something the generator can express (a
   contraction order or intermediate the factorizer cannot reach).

It would replace only the rank-2 RCC residual; (T), UCC and higher ranks
stay generated.

**Where it stands:** neither condition 1 (2.7×) nor condition 2 (3.1× in
cc-pVDZ, 2.7× in cc-pVTZ) holds, and the levers in "Open" are all generator
or build changes. Not justified.

## Tools

- **Allocation profiler.** A `DYLD_INSERT_LIBRARIES` interposer over
  malloc/calloc/realloc/free/posix_memalign/aligned_alloc. Every N seconds
  it logs live bytes by log2 size class, slack (`malloc_size` minus
  requested) for blocks ≥ 1 MiB, the malloc zone's in-use and allocated
  bytes, and the physical footprint and compressed bytes from `task_info`.
  At exit it writes the backtraces, requested and actual sizes of the
  ≥ 1 MiB blocks alive at the live peak after an arm time. Build it with
  gcc-15 (the system clang's linker cannot read the macOS 27 SDK) and
  launch the binary directly: SIP strips `DYLD_*` through `/usr/bin/time`.
- **Arithmetic count.** `tests/benchmarks/ccsd_residual/count_generated_fma.py`
  and `tests/pyscf/ccsd_update_amps_count.py` ("Where the gap came from").
- **PySCF reference and timing.** `tests/pyscf/ccsd_t_ladder_reference.py
  <input> <gbs>`.

## Evidence

### Hand-written against generated RCCSD

HEAD `a1692526`, 4 threads, 3 repeats each, whole-run wall and peak RSS.

| | hand-written RCCSD + (T) | generated RCCSD + (T) |
|---|---|---|
| C1 H2O2 6-31G | 32.0–32.4 s, 83–86 MiB, 22 iter | 0.54–0.58 s, 21 MiB, 26 iter |
| water cc-pVDZ spherical | 14.5–14.8 s, 100–106 MiB, 15 iter | 0.30–0.31 s, 37–40 MiB, 19 iter |

E(T) agrees to every printed digit, and both totals equal the PySCF-pinned
gates. Plain `ccsd` still runs the hand-written RCCSD. The generated route's
speed came from the dressing: undressed, rank 2 ran the same 26 iterations
in 45.5 s on C1 H2O2, slower than hand-written; dressed (`--dressing
derived`, every residual term a binary contraction) 0.54 s. So rank 2 is
emitted dressed in every build.

### The ladder at step 7

HEAD `d9f5fb69`. `build-default`, spherical, conventional SCF, 4 threads.
Timing from plain runs (3 repeats; one for the dimer in cc-pVTZ, whose
profiled run took the same 512 s); footprint from separate profiled runs.
Energies identical across repeats and equal to the gates.

| Rung | nb sph / cart | o / v | CCSD | (T) | Whole run | Peak footprint |
|---|---|---|---|---|---|---|
| water cc-pVDZ | 24 / 25 | 5 / 19 | 19 × 0.010 s | 0.03 s | 0.26 s | ~33 MiB |
| water cc-pVTZ | 58 / 65 | 5 / 53 | 16 × 0.19 s | 1.8–1.9 s | 5.8–6.1 s | 212 MiB |
| water dimer cc-pVDZ | 48 / 50 | 10 / 38 | 20 × 0.45 s | 4.0–4.1 s | 13.4 s | 84 MiB |
| water dimer cc-pVTZ | 116 / 130 | 10 / 106 | 18 × 11.8 s | 283 s | 512 s | 1.43 GB |

Per-iteration times were unchanged by the memory work except on the dimer
in cc-pVTZ, which no longer paged (CCSD 15.3 → 11.8 s per iteration, (T)
354 → 283 s). Steps 8 and 9 since: (T) water cc-pVTZ 0.37–0.41 s, dimer
cc-pVDZ 0.80 s, dimer cc-pVTZ 49.9 s; CCSD per iteration in "CCSD time".

### Against PySCF at step 7

PySCF 2.15.0, RHF + RCCSD + `ccsd_t.kernel`, Planck's own GBS files through
`parse_gaussian`, `cart=False`, `conv_tol` 1e-11, CCSD `conv_tol` 1e-10, 4
threads (`OMP_NUM_THREADS=4`), same geometries. PySCF times are wall clock
to 0.1 s.

| Rung | E total, PySCF | Δ Planck | CCSD Planck / PySCF | (T) Planck / PySCF |
|---|---|---|---|---|
| water cc-pVDZ | −76.2431380866 | 5e-10 | 0.19 / 0.1 s (~2×) | 0.03 / < 0.05 s |
| water cc-pVTZ | −76.3457669118 | 7e-10 | 3.1 / 0.2 s (~16×) | 1.9 / 0.2 s (~10×) |
| water dimer cc-pVDZ | −152.4977225012 | 2e-10 | 9.1 / 0.4 s (~23×) | 4.0 / 0.2 s (~20×) |
| water dimer cc-pVTZ | −152.7012785674 | 0 | 213 / 7.4 s (29×) | 283 / 6.0 s (47×) |

PySCF's peak footprint on the dimer in cc-pVTZ is 1.94 GB, against
Planck's 1.43 GB. The Δ on the small rungs is CCSD convergence (Planck
stops at `tol_energy`), not a defect.

## Not in this scope

RI-CCSD and RI-(T), frozen core, MPI distribution of the CC solve, the UCC
residual (the same levers likely apply; measure it separately),
iteration-count differences from DIIS, and the generated rank-3 CCSDT's
cost (`docs/CCGEN_GENERATED_CC_COST.md`).
