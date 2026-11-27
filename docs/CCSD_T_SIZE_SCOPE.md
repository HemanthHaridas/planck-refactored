# What limits the size of a CCSD(T) calculation, and what removes it?

Status lives in `vault/Status/Open Work.md` (priority 1). Parent:
`docs/CCSD_T_SCOPE.md`. Background on the generated CC route:
`docs/CCGEN_GENERATED_CC_COST.md`. Labels in parentheses (R1, R3b, 1a, ...)
are the step names other docs cite.

**Constraint: fixes go in the generator.** A hand-written spatial RCCSD is
the last resort, under the conditions in "Last resort".

## Answer

**Time is the limit, not memory.** On the water dimer in cc-pVTZ every phase
holds one large object and the run peaks at 1.43 GB. The remaining gap to
PySCF is the CCSD's per-iteration time; (T) is faster than PySCF.

| Water dimer / cc-pVTZ `ccsd(t)`, 4 threads (nb 116 sph / 130 cart, o 10, v 106) | Planck | PySCF 2.15 |
|---|---|---|
| Total energy | −152.7012785674 Eh | equal to every printed digit |
| CCSD per iteration | 0.79–0.80 s × 18 | 0.32 s × 23 (2.5× per iteration) |
| (T) | 4.5–4.8 s | 6.0 s |
| whole run | 35.8–37.1 s | |
| peak footprint | 1.43 GB | 1.94 GB |

Where it started (2026-11-21): the same run did not fit in 8 GiB (6.7 GB
footprint, paging), and once it did it took 512 s, CCSD 29× and (T) 47×
behind PySCF, with the gap growing with the basis. Now the CCSD ratio is
flat in the basis (3.1× on the dimer in cc-pVDZ, 2.5× in cc-pVTZ).

**UCC (`uccsd(t)`) now has the restricted work that applies to it** (the
U steps, "UCC"):

| Water⁺ / cc-pVTZ spherical `uccsd(t)`, 4 threads (o 5/4, v 53/54) | start | now (U4) |
|---|---|---|
| UCC CCSD per iteration | 0.339 s | 0.099–0.0995 s (3.4×) |
| UCC (T) | | 0.19–0.20 s (was 1.33–1.35 s before U4) |
| whole run | 9.4 s | 3.67–3.95 s |
| RCCSD on neutral water, same basis, for scale | | 0.039 s |

The run is now 46 % CCSD iterations (half of the residual is building
intermediates), 33 % spin-block ERI preparation, 15 % SCF, 5 % (T).

The next walls:
- **Memory:** the ns⁴ AO tensor in the SCF and v⁴ `vvvv` in the CCSD.
  Removing either is a different method (direct or RI integrals, an
  AO-driven or RI `vvvv` contraction), not a fix.
- **Time:** the restricted CCSD's remaining 2.5×; for UCC, the residual's
  builders and the spin-block ERI preparation.

## Memory: one large object per phase

| Phase | Largest object | Peak footprint (dimer cc-pVTZ) |
|---|---|---|
| SCF | spherical AO ERI tensor, ns⁴ (1.38 GB) | 1.31 GB |
| CCSD | `vvvv`, v⁴ (0.96 GB), with the other blocks and amplitudes | 1.43 GB |
| (T) | `vvvv` and blocks, kept by the state | 1.14 GB |

What keeps it there, each with the measurement that justified it:

| Mechanism | Where | Effect |
|---|---|---|
| The spherical tensor is built per shell quartet; the Cartesian one never exists | `RysQuad::_compute_2e_spherical` (`src/integrals/rys.cpp`) | SCF peak nc⁴ + ns·nc³ (4.1 GB on the dimer) → ns⁴ (1.38 GB) |
| One SCF copy of the tensor; the Cartesian-to-spherical transform frees its input | `build_conventional_eri` (`src/scf/scf.cpp`), `transform_eri_cart_to_sph` | water cc-pVTZ Cartesian SCF 466 → 330 MiB |
| `transform_eri` batched over its leading index | `src/post_hf/integrals.cpp` | removes n1·nb³ and n1·n2·nb² intermediates (1.32 + 1.21 GB on the dimer); bitwise identical, no time cost |
| The AO tensor is freed before the CC iterations | `Correlation::release_eri`, called in `rccgen.cpp` and `uccgen.cpp` | dimer peak 2.83 → 1.43 GB |
| The CC solver moves its state instead of copying it | `run_generated_arbitrary_order_iterations` | one copy of every integral block fewer (963 + 91 MiB on the dimer) |
| (T) reuses the CCSD's converged integral blocks | `compute_rccsd_t_energy(calculator, state)` | E(T) bitwise unchanged |
| The engines' thread-local ERI scratch is freed after a one-shot build | `_compute_2e` (`src/integrals/base.h`) | ~130 MiB on water cc-pVTZ |
| Binaries re-exec with libmalloc's large-block cache off | `src/base/malloc_env.h`, both `main`s | water cc-pVTZ footprint 624 → 262 MiB |

**Freeing the AO tensor.** The iterations and (T) read only the CC blocks,
and every rank's `prepare` (the warm-start recursion's included) runs
before any rank iterates. A later `ensure_eri` rebuilds the tensor, so a
missed reader costs time, not correctness. Both generated routes are
single-point only.

**The per-quartet spherical build.** Each canonical shell quartet's
Cartesian block comes from the same kernels `_compute_2e` uses, gets the
same Schwarz screen, then four one-index transforms with the shells' blocks
of `C`. Only canonical spherical elements are written, with their 8
permutations, so the tensor is exactly symmetric. A canonical element can
have its ket pair before its bra pair when both start on the same shell
((P1 P2 | P1 P1) holds (3,6|4,4)), so the ket loop starts at (sa, sa); the
gate's first run caught the missing kets (4592 elements). The builder
checks that `C` is block-diagonal by shell, and falls back to
build-then-transform under integral symmetry operations, which act on
Cartesian AOs. Gated in `planck-compute-2e`: equal to build-then-transform
to ≤ 6.7e-16 (cc-pVDZ, all four engines) and 2.1e-15 (cc-pVTZ, Auto), and
exactly 8-fold symmetric.

**Where the memory was** (dimer cc-pVTZ, live blocks ≥ 1 MiB during the
CCSD):

| MiB | Object | Now |
|---|---|---|
| 2179 | spherical AO tensor in a recycled Cartesian block (1448 requested) | freed before the CC |
| 963 | `vvvv` | kept |
| 963 | `vvvv` copy in the solver | gone (move) |
| 91 + 91 | `ovvv` and its copy | 91 |
| ~140 | thread-local ERI scratch | gone |

**Findings.**
- **macOS libmalloc held up to half the footprint.** It keeps a freed large
  block dirty and returns it whole for a later, smaller request: the freed
  2.18 GB Cartesian tensor became the 1.45 GB spherical tensor's block,
  pinning a dirty 0.73 GB tail. A standalone C program reproduces it.
  `MallocLargeCache=0` fixes it; it is read only at process start, hence
  the re-exec. A user-set value is kept, a failed exec continues unchanged,
  and `planck-mpi` does not re-exec (unsafe under an MPI launcher). No
  cost (water cc-pVTZ 5.82–5.86 s with the cache, 5.87–5.92 s without).
  On Linux it is a no-op: glibc returns blocks above its mmap threshold on
  `free` (from glibc's documented defaults, not measured).
- **Read live bytes with the requested size next to `malloc_size`.** A
  profile that summed `malloc_size` concluded the CCSD's large allocations
  fell short of the SCF peak and went looking for small ones; they summed
  to 5.34 GB because the spherical tensor sat in a 2179 MiB block.
- **Measure footprint, not RSS, near the RAM limit.** On a paging run RSS
  read 0.1–1.3 GiB while the footprint was 5–6.7 GB; `/usr/bin/time`
  max-RSS understates the same way.
- **The spin-orbital RCCSD was the first limit.** `run_rccsd` works in 2o
  and 2v spin orbitals (16× `vvvv`, 16× t2, a 64× o²v⁴ leading
  contraction). `ccsd(t)` runs the generated spatial RCCSD instead
  ("Evidence").

## CCSD time

### Why it was slow: more arithmetic, and slower arithmetic

Per iteration, 4 threads, against PySCF 2.15 RCCSD (`cc.RCCSD`), same
geometry and basis files, before any time work (R0):

| Water dimer | Planck | PySCF | ratio |
|---|---|---|---|
| cc-pVTZ (o 10, v 106): iterations × time | 18 × 11.8 s | 23 × 0.32 s | 36× per iteration |
| cc-pVTZ multiply-adds | 6.24 × 10¹⁰ | 1.38 × 10¹⁰ | 4.5× |
| cc-pVTZ rate | 5.3 G/s | 43 G/s | 8.1× |
| cc-pVDZ (o 10, v 38): iterations × time | 20 × 0.45 s | 21 × 0.019 s | 24× |
| cc-pVDZ multiply-adds | 2.69 × 10⁹ | 4.6 × 10⁸ | 5.9× |
| cc-pVDZ rate | 6.0 G/s | 24 G/s | 4.0× |

Iteration counts are not the cause. The extra arithmetic was the `vvvv`
ladder over all (i, j) where PySCF takes i ≥ j (1.8×), and 31 separate
o³v³ contractions where PySCF has a few. The count models the time: its
shares matched a `sample` profile within a few percent (o³v³ nests 59 % vs
60.4 %, ladder 20 % vs 20.1 %, builders 14 % vs 9.6 %).

### Matching PySCF's arithmetic

Multiply-adds per iteration on the dimer in cc-pVTZ, × 10¹⁰:

| | o³v³ | ladder | other | builders | total | vs PySCF |
|---|---|---|---|---|---|---|
| start | 3.70 (31 nests) | 1.26 | 0.40 | 0.88 | 6.24 | 4.5× |
| + half range (R1) | 2.03 | 0.69 | 0.24 | 0.88 | 3.85 | 2.8× |
| + t2 merge (R2) | 0.39 (6 nests) | 0.69 | 0.08 | 0.89 | 2.06 | 1.5× |
| + v merge, τ ladder (R2b) | 0.39 | 0.69 | 0.08 | 0.75 | 1.92 | 1.4× |

**Half range (R1).** The spatial RCC doubles residual is computed for
j ≥ i only and mirrored, R(j,i,b,a) = R(i,j,a,b), which is PySCF's i ≥ j.
`_pair_half_kernel` (`python/ccgen/emit/planck_tensor_cpp.py`) starts every
nest's free `j` loop at `i` and adds one `// mirror ijab` pass; it applies
to the spatial doubles of every generated rank, not to UCC. Exact because
t2 keeps t2(i,j,a,b) = t2(j,i,b,a): the MP2 start has it, Jacobi divides by
symmetric denominators, and DIIS takes linear combinations. The numpy
evaluator in the ccgen tests ignores loop bounds, so the half range is gated
by the C++ energies.

**Common-operand merge (R2, R2b).** `merge_common_operand_reads`
(`python/ccgen/optimization/factorize.py`) rewrites Σ_n c_n A(P) R_n as
A(P) Y, with Y = Σ_n c_n R_n an intermediate. `print_cpp_planck` runs it
after the factorizer on every non-energy target, spin-adapted and not UCC.
- **Grouping.** Terms share a group when their A read is the same once
  summed indices are renamed in order of appearance, and their rests carry
  the same index set.
- **Guards.** A rest may not sum an index of its own, and Y may not be
  larger than A. Without the second guard a t1·t2 rest becomes an o⁴v⁴ Y
  (total 3.4 × 10¹³).
- **A = t2** (R2): 69 binary `t2 × X` doubles terms become 12
  contractions, the 31 o³v³ become 6. ccsdt: singles 36 → 23, doubles
  142 → 85, triples 331 → 189 terms.
- **A = v, after the t2 merge** (R2b): on the rank-2 doubles exactly one
  group forms, the τ = t2 + t1 t1 ladder, which removes the largest builder
  (`W_t1v_ovvv`, o v⁴). In ccsdt that builder is also read by the triples,
  so the ladder stays unmerged there.
- **Gate.** `test_merged_amplitude_reads.py`: every residual with and
  without both passes agrees to 1e-10 relative on a fixture with exactly
  the (ia)(jb) symmetry; mutation-checked.

### Reaching GEMM rates

**The rates available**, G multiply-adds/s on the M3 (4 performance cores),
against about 5 for the scalar loop nests and 43 for PySCF's average:

| Shape | Eigen, 1 thread | Eigen, 4 threads | Eigen, caller splits rows (4) | Accelerate |
|---|---|---|---|---|
| o³v³, cc-pVTZ (1060³) | 20 | 63 | 61 | 154 |
| ladder, cc-pVTZ (55 × 11236²) | 16 | 31 | 30 | 114 |
| o³v³, cc-pVDZ (380³) | 21 | 57 | 49 | 201 |
| ladder, cc-pVDZ (55 × 1444²) | 17 | 37 | 37 | 134 |

Eigen and Accelerate agree to 2e-15 relative. Planck builds Eigen with
`EIGEN_DONT_PARALLELIZE`, and a caller-side split matches Eigen's own
threading. (The benchmark is a 90-line program around `C.noalias() = A * B`
and `cblas_dgemm`, not in the repo.)

**The helper** (R3). `cc_gemm_contract` (`src/post_hf/cc/gemm_contract.h`)
computes out(...) += coeff · Σ_k A(m..., k...) B(k..., n...), given the
index groups m, k, n as label strings.
- **Operands in place or copied.** Per-group offset tables show whether the
  storage already is the (rows, cols) matrix or its transpose; if not, the
  operand is copied into one.
- **Fixed partition.** The product runs in 64-wide blocks, so every
  element's summation order, and the result, are independent of the thread
  count.
- **Half range.** With `pair_half`, a group holding both i and j keeps only
  j ≥ i (the ladder: 55 of 100 ij rows). Where i and j lead the row and
  column groups, the kept part is a staircase: one GEMM per row-index value
  x on its row slab against the kept column range (j ≥ x, or i ≤ x), and
  the scatter writes only that region (R3b). In the rank-2 kernel that
  covers `ia|kc|jb`, `jb|kc|ia`, `ib|kc|ja`, `ja|kc|ib` and the o³v² calls
  `i|c|jba`, `j|c|iab`, `iab|k|j`, `jab|k|i`, with no emitter change: i or
  j already leads each group.
- **Small calls run serially.** Each copy, product and scatter loop runs on
  one thread below 2¹⁵ elements or multiply-adds. Without that, tiny
  systems paid for thousands of parallel regions on matrices a few elements
  wide (see "UCC"). The partition into tasks is unchanged, so results still
  do not depend on the thread count.

**The emitter.** `_gemm_lowering` (`python/ccgen/emit/planck_tensor_cpp.py`)
lowers a term when:
- it has two factors, at least one summed index, and five or more distinct
  indices (o²v³, o³v² and up);
- every summed index is in both factors and every free index in exactly one;
- each operand is used in place, or has at most three virtual indices, so
  any copy is o·v³ (the size of `ovvv`) or smaller and `vvvv` is never
  copied.

Group orders: k as stored in the operand with more virtual indices, m and n
as stored in their factors. One-index contractions are lowered too: without
them two strided `W_t1v_oovv` builders took a third as many samples as all
the GEMMs together. On the rank-2 TU 98 contractions are lowered and
7.3 × 10⁷ multiply-adds stay in loops, mostly the merge-sum builders. The
value tests read the lowered calls: `python/ccgen/tests/emitted_cpp_eval.py`
turns each `cc_gemm_contract` back into its einsum.

**Per-iteration time**, each row back to back against the one above (4
threads; energies identical to every printed digit):

| | water cc-pVTZ | dimer cc-pVDZ | dimer cc-pVTZ |
|---|---|---|---|
| start (loops) | 0.195 s | 0.454 s | 11.8 s |
| + half range | 0.136 s | 0.301 s | 7.55 s |
| + t2 merge | 0.108 s | 0.216 s | 5.45 s |
| + v merge | 0.093 s | 0.209 s | 5.23 s |
| + GEMM lowering, ladder on i ≤ j, one-index contractions | 0.039 s | 0.059 s | 0.850 s |
| + staircase (R3b) | | | 0.80 s |
| + serial small calls | | | 0.787 s |
| PySCF | 0.021 s | 0.019 s | 0.32 s |

Time fell less than the count once every contraction was still a scalar
loop: the rest was rate-bound, which the GEMM lowering addressed. The
staircase halves only the o³v³ and o³v² GEMMs, so it gains 1.04–1.08× per
iteration (slowest iteration 0.92–1.08 → 0.86–0.89 s).

## (T) time

**How the restricted (T) kernel works.** For each virtual triple
a ≥ b ≥ c, W is needed for all six orderings. Every W term is an
elementary o³ block (one contraction of two factors) up to a renaming of
i, j, k, so:
- **24 → 12 blocks per triple.** Over the six orderings the 144 term
  evaluations hit 24 distinct blocks (8), and those pair up under
  t2(i,j,a,b) = t2(j,i,b,a) (10a), leaving 12: six `t2 · ovvv` (o³v) and
  six `t2 · ooov` (o⁴), PySCF's count. `_emit_triple_kernel`
  (`python/ccgen/methods/ccsd_t.py`) keys each term by its canonical block
  and adds the block to every ordering with its occupied indices
  transposed.
- **Contiguous layouts, made once (1a).** The emitted
  `prepare_ccsd_t_triple_layouts`, called once by `batched_triples_energy`
  (`src/post_hf/cc/ccsd_t.cpp`), copies t2 as [x][p][q][r] and
  [x][y][p][q], `ovvv` as [x][y][d][u] and `ooov` as [z][l][u][v] (about
  110 MB on the dimer, 17 ms). Every slice a triple reads is then
  contiguous.
- **Every block is a GEMM** on views of those layouts, written straight into
  its slot: (o² × v)(v × o) for `ovvv`, (o × o)ᵀ(o × o²) for `ooov`. No
  per-triple gathers or scatter.
- **Assembly.** Per ordering, the terms that read their blocks in the same
  occupied order are summed in one contiguous loop and added once through
  that transposition: 30 transposed passes per triple instead of 144
  strided adds.
- The kernel asserts the two block shapes, so a change in the (T) terms
  fails at generation instead of emitting a wrong layout. Results do not
  depend on the thread count. V stays per ordering.

**How it got there** (water dimer cc-pVTZ, 4 threads, E(T) equal to every
printed digit at each step):

| | (T) |
|---|---|
| one W evaluation per ordering | 283 s |
| W once per triple for all orderings, 24 blocks (8) | 49.9 s |
| 12 blocks under the t2 pair swap (10a) | 26.8 s |
| `t2 · ovvv` blocks as small GEMMs with per-triple gathers (10b) | 8.25 s |
| all blocks as GEMMs on once-made layouts, contiguous assembly (1a) | 4.5–4.8 s |
| PySCF | 6.0 s |

**What 1a removed** (profiled on the 10b kernel, 2026-11-25): W was 89 % of
(T). Inside W, thread-local timers put the gathers at 22 %, the GEMMs at
27 %, the `ooov` loops at 25 % and the assembly at 25 %, out of 29.6
thread-seconds. The three loop sections were access-bound (strided reads
that did not vectorize), not arithmetic: the `ooov` loops did a tenth of
the GEMMs' multiply-adds in the same time. After 1a, W is 14.6
thread-seconds.

**What is left in (T)**: W about 80 % (12 small GEMMs and the assembly), V
2.3 and the energy loops 1.4 thread-seconds. The batch-over-a-virtual
restructure (below, "Open") is not needed to pass PySCF.

## UCC

`uccsd(t)` runs the generated `ucc2`, then (T) over the four t3 spin
blocks. Every change below is in the generator or the shared runtime; the
restricted output is byte-identical or, where noted, equal to every printed
digit.

| water⁺ / cc-pVTZ spherical `uccsd(t)`, 4 threads (o 5/4, v 53/54) | UCC CCSD per iteration | (T) | whole run |
|---|---|---|---|
| start | 0.339 s | | 9.4 s |
| GEMM lowering (1c) | 0.226–0.231 s | | |
| U1: no loop nests left in the same-spin doubles | 0.145–0.146 s | | 5.9–6.0 s |
| U2: batched `t1 · vvvv` | 0.127–0.129 s | | 5.7–6.1 s |
| U3a: each intermediate built once per evaluation | 0.115–0.116 s | | 5.26–5.34 s |
| U3b: the merges per spin block | 0.099–0.0995 s | 1.33–1.35 s | 4.96–5.01 s |
| U4: (T) as shared elementary GEMMs | 0.099 s | 0.19–0.20 s | 3.67–3.95 s |

U1–U4 are each measured against the same commit without the change, three
alternating runs; the peak footprint (511–519 MiB) never moved, because the
run peaks in the ERI preparation. For scale, RCCSD on neutral water in the
same basis is 0.039 s per iteration, and UCC costs ~3× inherently.

### What removed the time

- **GEMM lowering (1c).** `_gemm_lowering` takes the UCC operands: ERI and
  Fock spin blocks (`v_abab_oovv`, `f_aa_vv`), `t2_abab`-style views and
  builder scratch. A same-spin ERI read is antisymmetrized in the emitted
  text, `(v(..cd) − v(..dc))`, and each side is its own call with its sign.
- **U1, the last same-spin loops.** The vvvv ladder
  `t2(i,j,c,d) · (v(a,b,c,d) − v(a,b,d,c))` needed a forbidden copy of
  `vvvv`; renaming a pair of summed indices in both operands moves the swap
  onto t2 and keeps `vvvv` in place. A difference of two different arrays
  (`v_ovov − v_ovvo`) is accepted as two calls.
- **U2, batched calls.** `t1(i,c) · vvvv(a,b,c,d)` has no in-place matrix
  view and `vvvv` cannot be copied, so the call batches over `vvvv`'s
  leading free labels: each slice `vvvv(a,b,·,·)` is a contiguous v × v
  matrix (`cc_gemm_batched`, threaded over batches, each writing its own
  part of the output). It also lowers 19 t4 · `oooo`/`vvvv` terms at
  restricted rank 4.
- **U3a, intermediates built once.** Each residual kernel built the
  intermediates it read, so one read by k kernels was built k times (10 %
  of the UCC residual). A file with a kernel bundle now builds them once per
  evaluation into `<method>_residual_ops`, passed to every kernel by
  `const&`; the bundle's `all_residuals` evaluates every residual from it.
  Output bitwise identical. The restricted gain is within noise: two
  kernels, and its `t2merge` builders build their own inputs inside.
- **U3b, the merges per spin block.** `merge_operand_reads`
  (`optimization/factorize.py`), used by `print_cpp_planck` and the value
  gate alike, runs R2/R2b once per spin block: the pair swap (b,a,j,i) on
  same-spin blocks (sign +1), identity only on αβαβ, never a single swap
  (sign −1, which the pass cannot carry). GEMM calls in the UCC CCSD /
  CCSDT / CCSDTQ files: 371 / 1309 / 2964 → 268 / 1039 / 2767.
- **U4, (T) as shared elementary GEMMs.** Each UCC W block was one strided
  loop nest per term per virtual triple, 16–18 nests over 6–10 distinct
  contractions. `_emit_ucc_triple_kernels` (`methods/ccsd_t.py`) derives,
  from the terms, one elementary block e(u0, u1, u2) = Σ_s A B per class of
  terms equal up to an occupied permutation, computes it once per item as
  an Eigen GEMM, and adds it through each term's occupied order. Every
  factor read (the antisymmetrized difference included) gets a layout made
  once per (T), 16 in all; V stays per term.

### What did not apply, and why

- **The half range (R1).** αβαβ has no i↔j symmetry, and the same-spin
  doubles already use the orbit fold (one representative per orbit of
  {e, (ij), (ab), (ij)(ab)}, then R = A(X)), which removes more (4× against
  2×); X itself has no i↔j symmetry for a j ≥ i rewrite.
- **(T)'s six-ordering sharing.** Within a UCC spin block the orderings of
  a triple are related by antisymmetry, which the per-block multiplicity
  already uses; the gain is within one item.

### Findings

- **A flop model over kernel terms overstates merges.** U3b's dry run
  modelled 1.82× on the kernels (322 → 194 terms); measured 1.37×, and the
  merged sums cost the builders 58 samples, so 1.16× per iteration. The
  folded same-spin kernels gained less than αβαβ.
- **GEMMs beat their arithmetic count.** U4 cut (T) arithmetic 1.6–3× and
  ran 6.9× faster: the strided nests, not the flops, were the cost.
- **Untagged operators need a slot-spin table.** A merged sum is defined
  through the factorizer's W's, whose names carry no spin, so both its
  builder and a kernel whose first term read only it fell back to the
  restricted `no`/`nv`, and the file did not compile. Spins now resolve
  through one table per file, built in dependency order
  (`_operator_slot_spins`), and a disagreement is refused.
- **Tiny GEMMs need to run serially.** The lowering first made B STO-3G
  `ucc4` 1.82 → 5.82 s (thousands of calls, each opening parallel regions);
  small calls now run serially, which also sped up the small restricted
  cases (`be_rccsdtq_sto3g` 6.99 → 1.16 s, `lih_rccsdt_generated_sto3g`
  3.18 → 0.51 s).

### Gates

`UCCMergeTests` (adapt → fold → factorize → merge, every block of UCC CCSD
and CCSDT against the plain residual to 1e-10; a dropped merged term caught
in its block); `planck-cc-gemm-contract` (including batched calls); the
full-tensor (T) oracle to ~1e-18; the PySCF-pinned `*_uccsd_t_*` cases,
`b_ucc{2,3,4}_sto3g` (rank 4 = FCI) and the MPI smoke; identical output at
1/2/4/8 threads; core suite 178/178 after each step.

### Where UCC time goes now

3505 main-thread samples of the water⁺ cc-pVTZ run: CCSD iterations 46 %
(half of the residual is building intermediates; the t1·v and multi-t1
builders are ~64 % of that), spin-block ERI preparation 33 % (nearly all
`build_ucc_spin_block_cache`), SCF ~15 %, (T) 5 %. Neither remaining lever
is scoped: a t1-dressed (T1-transformed) Hamiltonian would remove the
t1-dependent builders at one integral transform per iteration (measure the
transform first), and the ERI preparation is a transform, not a CC kernel.

## Open, in order

| # | Step | Expected |
|---|---|---|
| R3c | Accelerate `dgemm` behind the GEMM helper | 2.5–4× on the GEMMs; a build decision |
| R2c | Share linearly dependent combinations of builders | 4 o³v³ builders → 3 contractions (≤ 1.06×) |
| T1 | (T): V through layouts, energy loops | ~20 % of (T) |
| R4 | Re-measure the size ladder against PySCF | timing and footprint in separate runs |

Beyond these: build the MO blocks from a direct or RI transform without
storing the AO tensor. RI-CCSD would be its own scope.

**R3c, Accelerate behind the GEMM helper.** Every lowered contraction goes
through one product call, so swapping Eigen for `cblas_dgemm` is a change in
one place: the optional `USE_BLAS` build (B1 in
`docs/ERI_FOCK_BLAS_SCOPE.md`). Accelerate is always present on macOS;
Linux needs a system or fetched OpenBLAS. BLAS results are not bitwise equal
to Eigen's; Accelerate is bitwise reproducible across thread counts,
OpenBLAS/MKL unmeasured. A user decision, not a generator step.

**R2c, shared builder combinations.** The four `W_t2v_oovv_{c3cc,68d5,6ae9,
7b99}` are t2·v. With the ERI pair swap v(l,k,d,c) = v(k,l,c,d) they form a
2 × 2 table:

| | v(k,l,c,d) | v(k,l,d,c) |
|---|---|---|
| t2(a,c,i,k) | c3cc | 68d5 |
| t2(c,a,i,k) | 6ae9 | 7b99 |

Only three of the t2 merge's sums read them, all on slots (i,k,a,c):

| sum | c3cc | 68d5 | 6ae9 | 7b99 | plus |
|---|---|---|---|---|---|
| `ab52bf` | 4 | −2 | −2 | 1 | 2 v reads, 2 cheap t1·v builders |
| `dedafd` | −2 | 1 | 1 | 0 | 1 v read, 1 cheap builder |
| `1e1c0d` | 0 | 0 | 0 | 1 | 1 v read, 1 cheap builder |

On those columns `ab52bf` = −2·`dedafd` + `1e1c0d`: rank 2, so two
combinations suffice, Z₁ = −2·c3cc + 68d5 + 6ae9 (two contractions after
o²v² sums) and Z₂ = 7b99. Three contractions instead of four is the floor,
and it is PySCF's `Wovvo`/`Wovov`. The common-operand merge cannot find it:
the saving is in the dependence between sums. Mechanism: write the sums as
coefficient vectors over basis reads, row-reduce the expensive columns
exactly (`Fraction`), build each Z from its inlined terms, rewrite the sums
to read the Z's. Risks: `factored_builder_steps` reads only
`definition_terms[0]`, so assert that a multi-term Z never has a 3-factor
first term; smallest-support Z's are a heuristic; rank 3 is unchecked.
At most about 1.06×; re-measure its share before taking it.

**T1 and batching (T) over a virtual.** If (T) becomes a large share again:
for fixed c, Σ_d t2(i,j,d,c) ovvv(k,d,a,b) over all (a, b) is one
(o² × v) · (v × o v²) GEMM, o³v³ per c and o³v² memory (90 MB on the dimer
in cc-pVTZ). That is a driver restructure (`batched_triples_energy` loops
over c outermost), gated like the (T) changes plus thread-count invariance
of E(T).

## Ruled out

- **Folding the doubles over {e, (ia)(jb)}** with `fold_external_symmetry`:
  2 % (1 of 31 o³v³ statements). It matches orbits on the terms' text with
  dummies as named, and a term and its image mostly differ in dummy names or
  in which t2 spelling they read. The half range needs no matching.
- **`factorize_tau`** collapses 0 spin-adapted doubles pairs: its validated
  pairs are spin-orbital shaped.
- **The v merge before the factorizer.** It forms PySCF-style groups, but
  the ring groups pull t2·v terms out of the t2 merge: o³v³ nests 6 → 10,
  total 2.19 × 10¹⁰. After the t2 merge only the ladder forms.
- **Contiguous operands for the old per-term (T) W kernel**: 0–4 %. That
  kernel was compute-bound (~34 G multiply-adds on the dimer in cc-pVDZ at
  ~9 G/s); removing the redundant work (8) was the fix. Contiguity paid
  later, once the work was GEMMs (1a).
- **Batching only the (T) GEMMs**: they were a quarter of W (10c profile);
  the loop sections were the cost.

## Last resort

A hand-written spatial RCCSD is justified only if, after the GEMM lowering:

1. Planck's CCSD is still more than 3× slower than PySCF's per iteration on
   the dimer in cc-pVTZ, **and**
2. the ratio still grows from the dimer in cc-pVDZ to cc-pVTZ, **and**
3. the remaining gap is not something the generator can express.

It would replace only the rank-2 RCC residual; (T), UCC and higher ranks
stay generated. **Not justified:** 2.5× in cc-pVTZ against 3.1× in cc-pVDZ,
so neither condition 1 nor 2 holds, and every open lever is a generator or
build change.

## Gates for every change here

- **Core suite** on `build/`.
- **CC cases on `build-default`:** `water_rccsd_t_{sto3g,631g,spherical_ccpvdz}`,
  `h2o2_c1_rccsd_t_631g`, `water_spherical_ccsd_sto3g`,
  `be_ccsd_spherical_631gd`, `be_rccsdtq_sto3g`.
- **CC cases on `build/`:** `{ch4,lih}_rccsdt_generated_sto3g`,
  `be_rccsdtq_sto3g`, `b_ucc{2,3,4}_sto3g` (rank 4 = FCI),
  `oh_uccsd_t_631g`, `water_cation_uccsd_t_{631g,spherical_ccpvdz}`,
  `oh_uccsd_t_631g_mpi_smoke`, `water_rccsd_t_631g`,
  `h2o2_c1_rccsd_t_631g`.
- **Thread-count invariance:** CCSD and (T) output identical at 1/2/4/8
  threads (`h2o2_c1_rccsd_t_631g`, `oh_uccsd_t_631g`); the batched (T)
  against the full-tensor kernel (`ccsd_t_oracle_diff` ≤ 1e-14).
- **`planck-cc-gemm-contract`:** the GEMM helper against a label-driven
  reference loop (staircase shapes, same-group filter, full product);
  a wrong column range fails 600 elements.
- **ccgen tests, targeted per change, full per batch.** For an emitter
  change, run the tests that emit or evaluate emitted code:
  `test_emit_flag_matrix`, `test_emitted_builder_matches_spec`,
  `test_factorize_value_preservation`, `test_merged_amplitude_reads`,
  `test_merged_call_sites`, `test_ucc_*`, and every test file that imports
  `emitted_cpp_eval` (`grep -l emitted_cpp_eval ccgen/tests/test_*.py`;
  `residual_eval` evaluates the algebra, not the emitted code). Run the full
  suite (about 26 minutes) once per batch of changes, before pushing. Emit
  pins move with any emit change; re-pin only after showing the old pins
  reproduce with the change switched off. Tests that read the emitted text
  must read lowered calls too:
  `test_every_read_is_the_one_the_valid_group_selects` spells each
  `cc_operand(name), "labels"` as a read.

## Tools

- **Benchmark input:** `tests/benchmarks/ccsd_t/water_dimer_ccpvtz.hfinp`
  (S22 water dimer, the ladder's top rung).
- **Allocation profiler.** A `DYLD_INSERT_LIBRARIES` interposer over the
  malloc family that logs live bytes by size class, slack (`malloc_size`
  minus requested) for blocks ≥ 1 MiB, and the footprint from `task_info`,
  and at exit writes the backtraces of the blocks alive at the live peak.
  Build it with gcc-15 and launch the binary directly (SIP strips `DYLD_*`
  through `/usr/bin/time`). `vmmap` is no substitute: it pauses the process
  (water cc-pVTZ 6.3 → 7.9 s at a 2 s poll).
- **Profiling a phase:** `sample` the whole run and aggregate the phase's
  subtree over all threads; stdout to a file is buffered, so a log line is
  no trigger. Set `MallocLargeCache=0` first, or the re-exec changes the
  PID. The release build has no debug info; for line-level splits, insert
  thread-local `steady_clock` marks into the generated TU, then restore it
  (and `touch` it: `cp` keeps the old mtime).
- **Arithmetic count.** `tests/benchmarks/ccsd_residual/count_generated_fma.py`
  (loop bounds around every accumulation in the generated rank-2 TU, a
  `j = i` loop counted as half; lowered calls are not counted) and
  `tests/pyscf/ccsd_update_amps_count.py` (PySCF's einsum/dot calls over one
  `update_amps`, plus the C ladder).
- **PySCF reference and timing.** `tests/pyscf/ccsd_t_ladder_reference.py
  <input> <gbs>`.

## Evidence

### Hand-written against generated RCCSD

HEAD `a1692526`, 4 threads, 3 repeats each, whole-run wall and peak RSS.

| | hand-written RCCSD + (T) | generated RCCSD + (T) |
|---|---|---|
| C1 H2O2 6-31G | 32.0–32.4 s, 83–86 MiB, 22 iter | 0.54–0.58 s, 21 MiB, 26 iter |
| water cc-pVDZ spherical | 14.5–14.8 s, 100–106 MiB, 15 iter | 0.30–0.31 s, 37–40 MiB, 19 iter |

E(T) agrees to every printed digit. The speed came from the dressing:
undressed, rank 2 took 45.5 s on C1 H2O2, slower than hand-written. So rank
2 is emitted dressed in every build. Plain `ccsd` still runs the
hand-written RCCSD.

### The size ladder once memory was fixed, before the time work

HEAD `d9f5fb69`, `build-default`, spherical, conventional SCF, 4 threads.
Timing and footprint from separate runs.

| Rung | nb sph / cart | o / v | CCSD | (T) | Whole run | Peak footprint |
|---|---|---|---|---|---|---|
| water cc-pVDZ | 24 / 25 | 5 / 19 | 19 × 0.010 s | 0.03 s | 0.26 s | ~33 MiB |
| water cc-pVTZ | 58 / 65 | 5 / 53 | 16 × 0.19 s | 1.8–1.9 s | 5.8–6.1 s | 212 MiB |
| water dimer cc-pVDZ | 48 / 50 | 10 / 38 | 20 × 0.45 s | 4.0–4.1 s | 13.4 s | 84 MiB |
| water dimer cc-pVTZ | 116 / 130 | 10 / 106 | 18 × 11.8 s | 283 s | 512 s | 1.43 GB |

Against PySCF 2.15 at that point (same inputs, Planck's GBS files,
`cart=False`): total energies within 7e-10 (Planck stops at `tol_energy`),
CCSD ~2× → 29× and (T) ~10× → 47× from water cc-pVDZ to the dimer in
cc-pVTZ. R4 re-measures this ladder.
