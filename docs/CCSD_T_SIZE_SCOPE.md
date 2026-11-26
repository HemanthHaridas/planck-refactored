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

**UCC (`uccsd(t)`) is the active work (the U steps).** It had only part of
the restricted work; U1 and U2 brought the GEMM lowering to all of its
CCSD terms.

| Water⁺ / cc-pVTZ spherical `uccsd(t)`, 4 threads (o 5/4, v 53/54) | start | now (U4) |
|---|---|---|
| UCC CCSD per iteration | 0.339 s | 0.099–0.0995 s (3.4×) |
| UCC (T) | | 0.19–0.20 s (was 1.33–1.35 s before U4) |
| whole run | 9.4 s | 3.67–3.95 s |
| RCCSD on neutral water, same basis, for scale | | 0.039 s |

U3a built each intermediate once per residual evaluation instead of once
per kernel (1.12× per iteration); U3b ran the restricted merges on each
UCC spin block (1.16×). Half of the residual is now building
intermediates, which no merge removes. U4 gave the UCC (T) the
restricted kernel's shared elementary GEMMs on prepared layouts: (T) 6.9×.
The run is now 46 % CCSD iterations, 33 % spin-block ERI preparation, 15 %
SCF, 5 % (T). The half range does not apply to UCC (see "UCC").

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

`uccsd(t)` runs the generated `ucc2`. The restricted work above reached UCC
only partly.

| | reaches UCC? |
|---|---|
| GEMM lowering (R3) | yes (1c, U1, U2) |
| small calls serial | yes |
| half range (R1) | not applicable: αβαβ has no i↔j symmetry, and the same-spin blocks already use the orbit fold (below) |
| intermediates built once per evaluation | yes (U3a); restricted too |
| merges (R2, R2b) | yes (U3b), per spin block |
| (T) layouts and GEMM blocks (8–1a) | yes (U4): shared elementary GEMMs per spin block; the six-ordering sharing has no UCC counterpart |

**The GEMM lowering for UCC** (1c). `_gemm_lowering` accepts the UCC
operands, which `cc_operand` already took: ERI and Fock spin blocks
(`v_abab_oovv`, `f_aa_vv`), `t2_abab`-style views and builder scratch
(`X1`). A same-spin ERI read is stored plain and antisymmetrized in the
emitted text (`_block_needs_explicit_exchange`), as `(v(..cd) − v(..dc))`;
each side becomes its own call with its sign. Lowered calls: UCC CCSD 350,
CCSDT 1309, CCSDTQ 2964. The restricted rank-2 TU is byte-identical.

| water⁺ cc-pVTZ spherical (o 5/4, v 53/54), 4 threads | before | 1c + U1 | now (+ U2) |
|---|---|---|---|
| UCC CCSD per iteration (median of 18) | 0.339 s | 0.145–0.146 s | 0.127–0.129 s (2.65×) |
| whole `uccsd(t)` run | 9.4 s | 5.9–6.0 s | 5.7–6.1 s |
| RCCSD on neutral water, same basis, for scale | 0.039 s | | |

**Small systems.** The lowering first made B STO-3G `ucc4` slower, 1.82 →
5.82 s: thousands of calls on tiny matrices, each opening up to four
parallel regions. Running small calls serially fixed it and sped up the
small restricted cases, which had paid the same cost since R3:

| case | before | now |
|---|---|---|
| `b_ucc4_sto3g` | 1.82 s | 0.63 s |
| `b_ucc3_sto3g` | 2.48 s (lowered, unfixed) | 0.17 s |
| `be_rccsdtq_sto3g` | 6.99 s | 1.16 s |
| `lih_rccsdt_generated_sto3g` | 3.18 s | 0.51 s |

**Where UCC time goes now (after U2).** `sample` of the water⁺ cc-pVTZ
`uccsd(t)` run, 4 threads, main thread only (it runs every parallel region
as thread 0 and waits at the barriers, so its samples track wall time; 4684
samples):

| phase | share |
|---|---|
| UCC CCSD residuals | 43 % (doubles αβαβ 16, αααα 14, ββββ 10, singles 4) |
| UCC (T) | 25 % |
| spin-block ERI preparation | 23 % |
| UHF SCF | 7 % |

Inside the CCSD residual, building intermediates (`build_W_*`) is 48 % and
the kernels' GEMM calls about half; operand reshaping (`cc_matrix`,
`cc_group_offsets`) is ~12 % of all GEMM-call samples. Each kernel built
the intermediates it reads, every iteration: 18 are built in two of the
five kernels and 13 in three, and the repeat builds are 10 % of the CCSD
residual. The restricted rank-2 file has the same pattern (19 of 57 built
in both of its kernels). Hence U3a, then U3b.

**The half range does not apply to UCC.** The same-spin doubles kernels
use the orbit fold: they compute one representative per orbit of
{e, (ij), (ab), (ij)(ab)} into X over all (i, j), then R = A(X). X has no
i↔j symmetry, so R1's j ≥ i rewrite cannot run on it, and the fold already
removes more than a half range would (4× against 2×).

**Before U1** (same profile method, whole run): loop code in the same-spin
doubles kernels was 19 % (aaaa 12 %, bbbb 7 %), the `t1 · vvvv` builders
~7 %. The census of the UCC CCSD residual before 1c (modelled at o 10,
v 50): two-read contractions 53 %, antisymmetrized differences 31 %, three
or more factors 15 %.

**The same-spin loop code (U1, done).** Two kinds of term were still
loops, the 19 % above:
- **The vvvv ladder**, `t2(i,j,c,d) · (v(a,b,c,d) − v(a,b,d,c))`: the
  second read would copy `vvvv`, which is forbidden. When a call needs a
  forbidden copy, `_gemm_lowering` now tries renaming a pair of summed
  (dummy) indices in both operands, which leaves the sum unchanged and here
  moves the swap onto t2: `t2(i,j,d,c) · v(a,b,c,d)`, `vvvv` in place.
- **A difference of two different arrays**, `(v_ovov(..) − v_ovvo(..))`:
  `_DIFF_READ` accepts two names; the sides were independent calls anyway.

No loop nests are left in the same-spin doubles kernels; UCC CCSD has 366
calls and 83 nests (was 350 and 91). The RCC rank-2 TU is byte-identical,
`oh_uccsd_t_631g` prints exactly what it printed before, and UCC output is
identical at 1/2/4/8 threads. UCC CCSD 0.226–0.231 → 0.145–0.146 s per
iteration (1.56×), now 3.7× the RCCSD in the same basis against the ~3×
UCC inherently costs.

**The `t1 · vvvv` builders (U2, done).** `t1(i,c) · vvvv(a,b,c,d) →
W(i,a,b,d)` was a loop: no grouping of the labels makes `vvvv` one matrix
in place, and copying it is forbidden. When neither a rename nor a copy
works, `_gemm_lowering` now batches over the leading free labels of the
operand that cannot be copied (`ab` here), so each slice `vvvv(a,b,·,·)` is
a contiguous v × v matrix used in place, and the call carries the batch as
a trailing `false, "ab"`. `cc_gemm_contract` runs one product per batch
(`cc_gemm_batched`), threaded over batches; each batch writes its own part
of the output, so the result does not depend on the thread count. Batched
calls keep the full range under the pair half; the mirror covers them. UCC
CCSD has three (the `aaaa`, `abab`, `bbbb` builders). UCC CCSD
0.145–0.146 → 0.127–0.129 s per iteration (1.14×, 3 runs; the 0.145 s is
the U1 measurement, not a same-session control), 18 iterations, E_corr
−0.2253381491 and E(T) −0.0048503368 in every run.
`planck-cc-gemm-contract` covers the batch on either operand with the slice
in place, transposed and copied.

U2 also reaches restricted rank 4: t4 · `oooo` and t4 · `vvvv` terms lower
as batched calls (19 in the plain spin-adapted `ccsdtq` emit, 178 in the
production build's dressed one). The restricted rank-2 and rank-3 emits are
byte-identical; the `ccsdtq` pin moved and reproduces with the emitter at the
pre-U2 commit. Gates: `be_rccsdtq_sto3g`, `b_ucc{2,3,4}_sto3g`, the UCC and
RCC (T) cases and `oh_uccsd_t_631g_mpi_smoke` pass; `be_rccsdtq_sto3g`,
`b_ucc4_sto3g` and `oh_uccsd_t_631g` print identical iterations at 1/2/4/8
threads.

**Intermediates built once per residual evaluation (U3a, done).** A
generated file that emits a kernel bundle now defines one
`<method>_residual_ops` struct holding every intermediate its residual
kernels read, built once per residual evaluation by
`build_<method>_residual_ops`; each residual kernel takes it as a last
`const&` and binds its intermediates from it (chunked kernels pass it to
their parts). The bundle gains `all_residuals`, which builds the struct
once and evaluates every residual; the runtime uses it when set, and the
per-kernel entries (each building the struct itself) otherwise. The energy
kernel, and files without a bundle, are unchanged.

| | pre-U3a, same commit | U3a |
|---|---|---|
| water⁺ cc-pVTZ UCC CCSD per iteration (3 alternating runs) | 0.129–0.131 s | 0.115–0.116 s (1.12×) |
| whole `uccsd(t)` run | 5.51–5.59 s | 5.26–5.34 s |
| peak footprint | 512–519 MiB | 511–512 MiB |
| water cc-pVTZ RCCSD per iteration | 0.038–0.039 s | 0.037–0.038 s (noise) |

The footprint does not move: the run peaks in the ERI preparation, not in
the iterations. Output is bitwise identical to the pre-U3a binary, and
across 1/2/4/8 threads, on `be_rccsdtq_sto3g`, `lih_rccsdt_generated_sto3g`,
`b_ucc4_sto3g`, `oh_uccsd_t_631g` and `h2o2_c1_rccsd_t_631g`; the 12 CC
regression cases pass; no emit pin moved (the pinned emits have no
intermediates). The restricted gain is small: two kernels, and its merged
`t2merge` builders build the intermediates they read inside themselves,
which can still repeat a struct member (passing the struct into the
builders would remove that).

**Where UCC time goes after U3a** (same profile method, 4660 main-thread
samples; the run's phase shares are as before U3a). The UCC CCSD residual,
1809 samples:

| part | samples | share |
|---|---|---|
| building the intermediates once (`build_ucc_ccsd_residual_ops`) | 753 | 42 % |
| doubles kernels (αβαβ 360, αααα 332, ββββ 234) | 926 | 51 % |
| singles kernels | 123 | 7 % |

Of the builder time, t1·v builders are 46 % (the three t1·ovvv builders
alone 193 samples), t2·v 37 %, builders with two or more amplitudes 18 %.

**The merges on UCC (U3b, done).** `merge_common_operand_reads` (R2,
R2b) had run only for the restricted emit. Its loop moved into one
function, `merge_operand_reads(eqs, specs, ucc)` (`optimization/factorize.py`),
which `print_cpp_planck` and the value gate both call. Under UCC it runs
once per spin block (`t2_aaaa`, `t2_abab`, `t2_bbbb`, then the three `v`
blocks). Same-spin blocks get the pair swap (b,a,j,i), a sign +1 symmetry;
αβαβ gets identity only (the swap lands on βαβα, another block), and no
single swap is offered (sign −1, which the pass cannot carry). The
restricted emit is byte-identical.

Before building it, a dry run of the pass on the factorized UCC CCSD
equations (cost modelled at o 5, v 54) took kernel terms 322 → 194 and
modelled kernel cost 2.19e9 → 1.20e9 (1.82×), almost all t2 with the
identity spelling.

**Two emitter defects, both found by the first build failing to compile.**
A merged sum Y is defined through the factorizer's W's, whose names carry
no spin, so a term reading only W's has no spin of its own:
- Y's builder took loop bounds and extents from that term and fell back to
  the restricted `no`/`nv`. Fixed: the builder emitter takes a slot-spin
  table built once per file (`_operator_slot_spins`, in dependency order,
  each operator resolved through the slots already known), and a builder's
  own slots set its extents.
- The triples kernels took their result spins from their first term alone,
  which after merging can read only a Y. Fixed: the free indices' spins
  come from every term through the same table, and a disagreement is
  refused.

| water⁺ cc-pVTZ `uccsd(t)`, 4 threads, 3 alternating runs | pre-U3b, same commit | U3b |
|---|---|---|
| UCC CCSD per iteration | 0.114–0.115 s | 0.099–0.0995 s (1.16×) |
| whole run | 5.23–5.34 s | 4.96–5.01 s |
| peak footprint | 511–512 MiB | 511–512 MiB |
| GEMM calls in the UCC CCSD / CCSDT / CCSDTQ files | 371 / 1309 / 2964 | 268 / 1039 / 2767 |

Short of the modelled 1.35× on the residual. The profile (main thread):

| UCC CCSD residual | post-U3a | post-U3b |
|---|---|---|
| building intermediates | 753 | 811 (+58, the merged sums) |
| doubles αβαβ | 360 | 186 |
| doubles αααα | 332 | 269 |
| doubles ββββ | 234 | 196 |
| singles | 123 | 114 |
| total | 1809 | 1581 (1.14×) |

The kernels fell 1.37× (1049 → 765), not 1.82×: αβαβ halved, the folded
same-spin kernels gained less, and building the merged sums costs 58
samples. A flop model over kernel terms overstates the gain.

Gates: `UCCMergeTests` (adapt → fold → factorize → merge, every block
against the plain residual, UCC CCSD and CCSDT exact to 1e-10; dropping one
merged term fails exactly the block it came from); the 12 CC regression
cases including the MPI smoke; identical output at 1/2/4/8 threads on
`oh_uccsd_t_631g`, `water_cation_uccsd_t_631g`, `b_ucc3_sto3g`,
`b_ucc4_sto3g`, `h2o2_c1_rccsd_t_631g`; final energies equal to the
pre-U3b binary's to every printed digit; targeted ccgen tests pass (no pin
moved: the pinned emits are undressed).

**The builders are now the larger half.** 811 of 1581 samples (51 %) is
building intermediates, and merges remove none. A different lever, not
scoped: absorb t1 into the integrals once per iteration (t1-dressed /
T1-transformed Hamiltonian), which removes the t1·v and multi-t1 builders
(64 % of builder time after U3a) at the cost of one integral transform per
iteration. Measure that transform first.

**UCC (T) as shared elementary GEMMs (U4, done).** After U3b, (T) was 27 %
of the water⁺ cc-pVTZ run and its W blocks ~97 % of it (all-thread samples:
αab 1771, αbb 1478, ααα 829, βββ 456; V 103 in all). Each W block was one
strided loop nest per term, per virtual triple, with no GEMM:

| UCC W block | loop nests per item | distinct elementary contractions (up to an occupied permutation) |
|---|---|---|
| ααα, βββ | 18 | 6 |
| αab, αbb | 16 | 10 |

`_emit_ucc_triple_kernels` (`python/ccgen/methods/ccsd_t.py`) brings the
restricted triple kernel's mechanism to each spin block, derived from the
terms rather than matched by shape:
- every W term is one contraction of two factors over one summed index;
  occupied externals are renamed u0, u1, u2 by first appearance, so terms
  equal up to an occupied permutation share one elementary block
  e(u0, u1, u2) = Σ_s A[u's of A, s] B[s, u's of B], computed once per item
  as an Eigen GEMM and added through each term's occupied order;
- each factor read (a same-spin ERI read is the antisymmetrized difference)
  gets a layout made once per (T), [fixed virtuals][rows][cols] as its GEMM
  reads it, shared across blocks by a canonical text: 16 layouts in all;
- `compute_ucc_ccsd_t_W_<tag>_triple(a, b, c, reference, lay, out)` writes
  the same (i, j, k) block the per-term kernel did, so the driver changed
  only in building `UCCSDTripleLayouts` once and passing it. V stays per
  term.

The restricted kernel's sharing across the six orderings of a triple has
no UCC counterpart: within a spin block the orderings are related by
antisymmetry, which the per-block multiplicity already uses.

| water⁺ cc-pVTZ `uccsd(t)`, 4 threads, 3 alternating runs | pre-U4, same commit | U4 |
|---|---|---|
| (T) | 1.33–1.35 s | 0.19–0.20 s (6.9×) |
| whole run | 4.86–5.16 s | 3.67–3.95 s |
| peak footprint | 511–519 MiB | 511–517 MiB |

Above the 3–5× expected: the GEMMs gain more over the strided nests than
the 1.6–3× cut in arithmetic alone.

Gates: the full-tensor oracle (verbose output) agrees to 1.2e-18 (OH
6-31G) and 1.7e-18 (water⁺ 6-31G); the `*_uccsd_t_*` cases (PySCF-pinned),
the MPI smoke, UCC ranks 2 and 4 and the restricted (T) cases pass; E(T)
and every printed line identical at 1/2/4/8 threads on the three UCC (T)
cases; targeted ccgen tests 157 passed; core suite 178/178.

**Where UCC time goes after U4** (3505 main-thread samples): CCSD
iterations 46 % (half of the residual is building intermediates),
spin-block ERI preparation 33 % (nearly all `build_ucc_spin_block_cache`),
SCF ~15 %, (T) 5 %. The next UCC levers are the ERI preparation and the
builders (a t1-dressed Hamiltonian, not scoped).

## Open, in order

| # | Step | Expected |
|---|---|---|
| R3c | Accelerate `dgemm` behind the GEMM helper | 2.5–4× on the GEMMs; a build decision |
| R2c | Share linearly dependent combinations of builders | 4 o³v³ builders → 3 contractions (≤ 1.06×) |
| T1 | (T): V through layouts, energy loops | ~20 % of (T) |
| R4 | Re-measure the size ladder against PySCF | timing and footprint in separate runs |

Beyond these: build the MO blocks from a direct or RI transform without
storing the AO tensor. RI-CCSD would be its own scope.

The U3 profile and why the half range is out are under "UCC".

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
