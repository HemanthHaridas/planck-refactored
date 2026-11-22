# Why was the generated CC route slower than the hand-written one, and what fixed it?

Answered 2026-11-21 (opened as scopes 2026-08 to 2026-11-21; merged from
eight docs on 2026-11-21). Status lives in `vault/Status/`.

## Answer

**Eleven defects in the generated route, all in the emitter or the tensor
accessors and none in the algorithm, made it slower. All are fixed. On
CH4/STO-3G one thread went from 103.86 s to 0.30 s against the hand-written
0.19–0.24 s. At four threads the generated route now beats the hand-written
one wall-clock on every ladder rung but CH4, and on one thread it costs
1.4–2.3× per iteration.**

| # | Defect | Fix | Gain | Found by | Landed |
|---|---|---|---|---|---|
| 1 | Accessors out of line, two heap allocations per element read | inlined in `common.h` | 206× rank-3 triples kernel; 3.4× rank 4 | microbenchmark | 2026-08 |
| 2 | 391 four-deep o⁵v⁵ terms evaluated n-arily | derivation dressing (`--dressing derived`) | 3.6× (CH4) | census + FLOP model | 2026-08-29 |
| 3 | Every operator rebuilt in every `_partN` chunk | build once into a `<kernel>_ops` struct | 1.76× | `sample` profile | 2026-08-29 |
| 4 | Transpose-equivalent builders each built their own array | `merge_transposes` | 1.42–1.52× | `sample` profile | 2026-08-29 |
| 5 | No OpenMP anywhere in CC | `collapse(3)` on residual nests | 3.22× at 4 threads | asking whether it was threaded | 2026-08-30; default 2026-11-20 |
| 6 | Chunked kernels built every operator twice | emit builds only into `ops` | 2.6 % | reading the emitted text | 2026-08-30 |
| 7 | 112 residual terms left n-ary by seeded-operator inlining | `EMIT_FINGERPRINTS = ()` | 11.0 → 3.01 s (CH4, 1 thread) | profile + census of the production TU | 2026-11-21 |
| 8 | Builder nests unthreaded | same pragma on builders | 1.84 → 1.27 s (CH4, 4 threads) | profile | 2026-11-21 |
| 9 | Triples residual computed for every symmetry-equivalent element | fold over the order-4 group, symmetrize once | 1.5× (CH4) | symmetry census | 2026-11-21 |
| 10 | Builders emitted as one flat n-ary nest | `factor_builder_bodies` on the production route | 1.4–2.8× | profile | 2026-11-21 |
| 11 | Amplitudes read through a runtime-rank accessor | `ConstFixedRankView<N>` | 5.1–6.1× (1 thread) | disassembly + one-nest benchmark | 2026-11-21 |

Every fix left the energies bitwise unchanged, except dressing (#2), which
reorders the arithmetic and agrees to every printed digit. Generated and
hand-written E_corr agree to ≤ 4e-10 on seven of eight ladder rungs and
1.7e-9 on C2H4, within the tolerance both converge to. A 1.0e-8 gap on C2H2
was the hand-written solver stopping early (The hand-written solver's
convergence test).

**What the remaining ratio means.** The two routes are different solvers,
not two implementations of one algorithm: dense against wedge-packed
amplitudes, a full generated kernel per rank against cheap hand-written
dressed intermediates, 15–19 against 23–41 iterations. A ratio between them
prices each route end to end, never codegen quality alone (Measuring rule
1).

**Refuted levers:** loop fusion (~0 %), the `-O1` registry pin (1.15–1.23×
for a 21-minute compile), the arbitrary-order harness (~1 %), memory traffic
from loop fission at reachable sizes, and the add-chain latency in the hot
nests (Ruled out).

## Where the two routes stand

Eight rungs, both routes in one session on one build (2026-11-21), best of
two runs each. s per iteration:

| Rung | o | v | gen 1 thread | gen 4 threads | hand-written | gen1/hw | gen4/hw | wall s, hw / gen1 / gen4 |
|---|---|---|---|---|---|---|---|---|
| CH4/STO-3G | 5 | 4 | 0.0155 | 0.0150 | 0.0073 | 2.13 | 2.06 | 0.24 / 0.30 / 0.28 |
| HF/6-31G | 5 | 6 | 0.0551 | 0.0277 | 0.0317 | 1.74 | 0.87 | 0.80 / 0.89 / 0.48 |
| H2O/6-31G | 5 | 8 | 0.146 | 0.053 | 0.091 | 1.60 | 0.58 | 2.62 / 2.41 / 0.91 |
| NH3/6-31G | 5 | 10 | 0.313 | 0.098 | 0.208 | 1.51 | 0.47 | 6.09 / 5.70 / 1.84 |
| H2O/6-31G* | 5 | 14 | 1.127 | 0.325 | 0.812 | 1.39 | 0.40 | 22.0 / 18.1 / 5.28 |
| C2H2/STO-3G | 7 | 5 | 0.114 | 0.048 | 0.057 | 2.00 | 0.83 | 1.44 / 2.24 / 0.99 |
| C2H4/STO-3G | 8 | 6 | 0.351 | 0.111 | 0.155 | 2.26 | 0.71 | 6.44 / 5.34 / 1.73 |
| C2H6/STO-3G | 9 | 7 | 0.971 | 0.282 | 0.460 | 2.11 | 0.61 | 15.8 / 16.6 / 4.88 |

- **Wall-clock, four threads:** generated is faster on every rung but CH4:
  1.5–1.7× on HF/6-31G and C2H2, 2.9–4.2× from H2O/6-31G up, and 1.17×
  slower on CH4. The hand-written backend runs on one core.
- **The 2026-11-21 target** (end-to-end ≤ 2× hand-written on every rung,
  with no growth) is met at four threads on every rung (CH4 1.17×).

**Fits, one thread, per iteration** (t = C·o^a·v^b, leave-one-out):

| Per-iteration cost | fit | LOO max / median |
|---|---|---|
| generated | o^3.89 · v^3.42 | 17 % / 6 % |
| hand-written | o^3.50 · v^3.73 | 14 % / 6 % |
| ratio | o^0.38 · v^−0.32 | 11 % / 4 % |

- **The ratio grows slowly in o and falls in v:** a factor of 1.25 over
  o = 5 → 9. No scaling defect remains.
- **No four-thread fit holds,** because thread efficiency itself grows with
  size: 1.03× on CH4, 2.0× on HF/6-31G, 3.2–3.5× on the largest rungs. The
  4-thread fits fail leave-one-out (48 % / 22 %, or 27 % / 13 % without
  CH4). Read the four-thread results as measured ratios, not as a law.

## How the CH4 number moved

`ch4_rccsdt_generated_sto3g` (or the same system through the generated
route), one thread unless marked, s. The rows come from different sessions
and builds, so read each step's ratio, not differences between distant
rows.

| After | generated | hand-written |
|---|---|---|
| accessor fix (#1), undressed | 103.86 | 0.19 |
| derivation dressing (#2) | 28.67 (another session: 28.94) | 0.19 |
| operators built once (#3) | 29.59 → 16.81 (best of 2) | |
| transpose merge (#4) | 16.65 → 10.98 | |
| 2026-11-21 start | 11.0 (4 threads: 4.10) | |
| every residual term binary (#7) | 3.01 (1.84) | |
| builders threaded (#8) | 3.39 (1.27) | |
| symmetrize once (#9) | 2.26 (0.82) | |
| factored builder bodies (#10) | 1.27 (0.56–0.68) | 0.22 |
| fixed-rank amplitude view (#11) | 0.30 (0.28) | 0.24 |

- Before #1 the comparison was on the isolated rank-3 triples kernel (BH3:
  6.40 s generated against 0.170 s hand-written), not a whole run.
- Threading (#5) does not show in this column: CH4 barely threads (1.03× at
  four threads per iteration). On HF/6-31G it measured 3.22×.
- An earlier record gave 3.1× on one thread for CH4 at row #10; the
  hand-written 0.74 s it used came from a cold first run.

## 1. Out-of-line accessors that allocated

**Cause.** Every CC kernel reads elements through `operator()`. Those
accessors were defined in `common.cpp` with no LTO or IPO configured, so
each element read was a cross-TU call that:
1. could not be inlined;
2. constructed `std::vector<int>{dims}` and `std::vector<int>{indices}`, two
   heap allocations;
3. built a `std::expected<std::size_t, std::string>`;
4. then indexed.

The runtime-rank path was worse: `TensorND::operator()(initializer_list)`
copied the list into a fresh vector (`to_vector`) before the out-of-line
`flatten_index`. The rank ≥ 4 generated kernels use that path exclusively.

**Why it dominated the generated kernels.** Both routes called the same
accessor; the generated one called it far more often: 3416 call sites in the
rank-3 triples residual against 186 hand-written, and 23,338 braced-index
accesses per rank-4 quadruples evaluation.

**Measured** (P1, BH3/STO-3G, o = v = 4, Release `-O3`, one thread, both
residuals from identical amplitudes via a since-removed
`PLANCK_CC_T3_DIFF=1` probe, six iterations): generated 6.40 s, hand-written
0.170 s (+0.014 s for its intermediates), ratio 37.6× ± 0.4. The rank-3
triples TU is `#include`d into `tensor_backend.cpp`, so both were `-O3`.

Microbenchmarks at the real size (P2, 1063 sweeps):

| Variant | time |
|---|---|
| checked + vector-allocating accessor (as shipped) | 0.0446 s |
| inlined flat index | 0.0033 s (13.5×) |
| fissed loops, 1063 sweeps (generated shape) | 0.0018 s |
| fused, 1 sweep (hand-written shape) | 0.0029 s (fissed 0.62×, faster) |

13.5× (accessor) × ~2.8× (structural) ≈ the observed 37.6×.

**Fix.** Fixed-rank (`Tensor2D/4D/6D`) and runtime-rank (`TensorND`,
`DenseTensorView`, `ConstDenseTensorView`) accessors inlined in `common.h`
as flat row-major index computations. `to_vector` and
`checked_fixed_rank_index` deleted. Energies bitwise identical.

| | before | after | gain |
|---|---|---|---|
| rank-3 generated T3 residual (BH3) | 6.40 s | 0.031 s | 206× |
| rank-3 hand-written T3 residual | 0.170 s | 0.0014 s | 121× |
| generated / hand-written | 37.6× | 22× | |
| rank-4 CCSDTQ per iteration (Be/STO-3G) | 38.5 s | 11.4 s | 3.4× |
| `water_rccsdt_sto3g` | 44.6 s | 0.39 s | 114× |

- **The runtime-rank half was load-bearing at rank 4.** A first pass fixed
  only the fixed-rank accessors: rank 3 gained 76×, rank 4 nothing. The
  runtime-rank accessors had been excluded as "not on the hot path" and as
  needing an interface change; neither was true or measured.
- **The ratio widened before it narrowed** (37.6× → 59× after the fixed-rank
  pass → 22×), because the hand-written kernel is more accessor-dominated.
  An early draft predicted ~2.8× by assuming equal accessor cost on both
  sides.
- **The bounds check moved into a debug assert,** keeping both conditions:
  per-index range and `offset < data.size()`. The storage half is not
  redundant: `data` is a public member that call sites assign after
  construction (`tensor_backend.cpp`), so the size invariant is breakable.
  In release an out-of-bounds read is now UB instead of returning the shared
  `tensor_error_slot`, which had no consumers and already fired
  `assert(false)`.
- **The carried "~180× on bh3" never reproduced.** It had no recorded o, v,
  compiler or flags. Do not cite it.

## 2. Contraction order (derivation dressing)

**Cause.** 391 of 824 rank-3 triples nests carried a four-index inner sum
(o²v²) costing o⁵v⁵, in four families: t1·t2·t2·⟨oo‖vv⟩ (172),
t2·t3·⟨oo‖vv⟩ (151), t1·t1·t3·⟨oo‖vv⟩ (44), t1·t1·t1·t2·⟨oo‖vv⟩ (24); 83–90 %
of generated FLOPs. `docs/CCGEN_HIGHER_OPERATOR_REUSE.md` had predicted the
t2·t3·v case: o⁵v⁵ n-ary against o³v⁴ with v·t3 factored first.

**Fix.** `--dressing derived` factorizes each term into a binary tree of
derived operators: 824 nests → 414. Modelled 10–18× (11.2× on CH4); measured,
generated-undressed against generated-dressed, same configuration otherwise:

| system | o | v | undressed | dressed | gain |
|---|---|---|---|---|---|
| LiH/STO-3G | 4 | 8 | 5.12 s | 1.64 s | 3.12× |
| BH3/STO-3G | 4 | 4 | 33.70 s | 9.34 s | 3.6× |
| CH4/STO-3G | 5 | 4 | 103.86 s | 28.67 s | 3.62× |

Energies identical to every printed digit; CH4 takes 15 iterations either
way. The emitter's own contraction-order analysis
(`_optimal_contraction_order`, `_detect_gemm`, `BLASHint` in
`python/ccgen/tensor_ir.py`) is computed and discarded by the emitter;
dressing removes the terms it targets, so consuming it is probably
redundant.

**The isolated-kernel ladder that confirmed this cause** (before dressing,
rank-3 triples residual, `-O3`, one thread, generated and hand-written from
identical amplitudes):

| case | o | v | t3 MiB | generated | hand-written | ratio |
|---|---|---|---|---|---|---|
| BH3/STO-3G | 4 | 4 | 0.03 | 0.0309 s | 0.00142 s | 21.8× |
| CH4/STO-3G | 5 | 4 | 0.06 | 0.0930 s | 0.00347 s | 26.8× |
| HF/6-31G | 5 | 6 | 0.21 | 0.5681 s | 0.01779 s | 31.9× |
| H2O/6-31G | 5 | 8 | 0.49 | 1.7232 s | 0.06316 s | 27.3× |
| BH3/6-31G | 4 | 11 | 0.65 | 3.3287 s | 0.09741 s | 34.2× |
| C2H4/STO-3G | 8 | 6 | 0.84 | 5.9509 s | 0.11875 s | 50.1× |

- **Fits** (log o spread 0.693, log v 1.012, condition number 26.1):
  hand-written o^3.94 · v^4.18 (4.5 % max residual), generated
  o^4.87 · v^4.52 (21.4 %), ratio o^0.93 · v^0.34. Leave-one-out on the
  ratio: o exponent +0.40 to +1.18, sign-stable; the two variants that move it
  drop the endpoints (BH3/6-31G at max v, C2H4 at max o).
- **The generated misfit is concentrated at high v** (H2O/6-31G +21.4 %,
  BH3/6-31G −10.5 %, the four lowest-v points ≤ 5.5 %): more than one
  contraction regime, as expected of n-ary terms with different optimal
  orders.
- **Two earlier fits were wrong** for one reason. Four points, three at
  o = 5, gave o^1.12 · v^0.05 ("the gap is all in o"). Five points gave
  o^0.40 · v^0.32, but leave-one-out swung the o exponent from −0.65 to
  +1.12 (condition number 46.6). Measuring rule 4.
- **Undressed and dressed isolated-kernel exponents were never compared.**
  Re-running this ladder under dressing was abandoned: its timing probe sat
  on a branch the rank-3 representation fix rerouted away from, and the two
  routes have no residual-level agreement gate (wedge-packed against dense;
  `restore` annihilates a raw residual to 2.0e+05). The 2026-11-21 per-
  iteration fits above measure a different quantity and cannot be combined
  with these.

## 3. Operators rebuilt in every chunk (H5)

**Cause.** Large kernels are split into `_partN` functions by term count
(`_emit_chunked_kernel`, `_KERNEL_CHUNK_TERMS`), and every part built every
dressed operator. The duplication factor equalled the part count:

| kernel | parts | distinct ops | builder calls | after | reduction |
|---|---|---|---|---|---|
| `ccsdt` triples | 4 | 278 | 1112 | 278 | 4.0× |
| `ccsdtq` quadruples | 18 | 894 | 16,092 | 894 | 18.0× |
| ccsdtq TU total | | | 50,601 | 2431 | 20.8× |

The emitter's comment called the per-part builds "cheap, local". That was
true when chunking was introduced for the undressed emit, where
`required_intermediates` is empty; dressing populated it and nothing
re-examined the claim.

**Fix.** Each operator is built once into a generated `<kernel>_ops` struct
passed by `const&`; each part binds `const auto &W_x = ops.W_x;`. Members use
`_tensor_type(spec.rank)`, the builder's own return-type expression.

| CH4, best of 2 | before | after | |
|---|---|---|---|
| wall | 29.59 s | 16.81 s | 1.76× |
| builder share | 67.7 % | 45.1 % | |
| builder time | 20.03 s | 7.58 s | 2.64× |
| residual time | 9.56 s | 9.23 s | 1.04× (untouched, as it must be) |

BH3 9.52 → 5.57 s (1.71×). E_corr bitwise identical. Rank-4 TU 12.8 → 10.5
MB, 48,170 fewer call sites.

**The prediction missed.** Builders 67.7 % → ~17 % predicted, 45.1 %
measured; builder time fell 2.64×, not the 4× the call count implied. The
eliminated builds were cheaper than average (a part's 270 builds span every
rank while its 59–118 used operators skew to what its terms need). That 0.66
realization factor belongs to this fix only (Measuring rule 2).

## 4. Transpose-equivalent operators (merge)

**Cause.** After #3, `build_W_t2t2v_oooovv` was 20.4 % of HF/6-31G: 38
emitted builders, each writing a rank-6 result over a 9-deep nest, and with
index names normalized away all 38 are the same contraction
t2·t2·⟨oo‖vv⟩ with slots permuted. `merge_transposes`
(`manifold_operators_with_plan`, `python/ccgen/optimization/factorize.py`)
already decided that equivalence symbolically, exact against a numeric
oracle and value-gated 0/2536 at rank 4, but had no production caller.

**Fix.** Unconditional for `--dressing derived` (no flag): transpose-
equivalent operators share one array, and call sites read it through the
permutation. Wiring it was 9 lines, 8 of them comment.

| | rank 3 (`ccsdt`) | rank 4 (`ccsdtq`) |
|---|---|---|
| distinct builders | 288 → 91 (3.2×) | 1615 → 239 (6.8×) |
| TU bytes | 802 K → 523 K | 11.0 M → 6.6 M |

| case, 3 runs each, ±0.03 s | before | after | gain |
|---|---|---|---|
| `lih_rccsdt_generated_sto3g` | 1.02 s | 0.72 s | 1.42× |
| `ch4_rccsdt_sto3g` (generated route then) | 16.65 s | 10.98 s | 1.52× |

LiH agrees on E_corr, dE, rms(res), rms(step) at all 62 iterations; CH4's
output diffs empty without timers; iteration counts unchanged.

**The causal story was checked per family** (CH4, ~6700 leaf samples per
arm): `t2t2v_oooovv` merges 9.5× (38 → 4), 23.3 % → 4.1 %; `t1t1t2v_oooovv`
6.0× (12 → 2), 6.6 % → 1.7 %; the negative control `t1t3v_oooovv` merges
only 1.7× (19 → 11), 9.7 % → 9.4 %. At rank 4: `t2t2v_ooooovvv` 95 → 1,
`t1t2t2v_ooooovvv` 68 → 1, `t2t3v_ooooovvv` 313 → 15.

**Estimated wrong twice, low, for one reason.** An operator-count model gave
1.02–1.20× ("compile time, not speed") and deferred it for months; a
profile-weighted re-cost with #3's 0.66 factor gave 1.21–1.36×. Both priced
operators as equal-cost, and the 0.66 factor came from removing cheap
builds, where this removes expensive ones.

**Compile time also improved:** registry 11.68 → 10.44 s (object 1.50×
smaller), `tensor_backend.cpp` 38.86 → 35.80 s, binary 1.11× smaller.
`tensor_backend.cpp` is the expensive TU (it includes
`ccsdt_planck_generated.cpp` at `-O3`); timing only the `-O1` registry
understates compile cost 3×.

## 5. No OpenMP in CC (H6)

**Cause.** Zero `#pragma omp` in `src/post_hf/cc/*.{cpp,h}` or the emitted
kernels, while ERI, Fock, 4-index transforms and DFT J/K were threaded.
Proven with a positive control in the same binary: CC flat 78.03 → 78.49 s
from 1 to 4 threads, direct-SCF HF/cc-pVTZ 2.70 → 0.81 s (3.3×). The first
evidence ("98.8 % CPU on 8 cores") came from a build where
`OpenMP_CXX_FLAGS` was `NOTFOUND`, so every pragma in Planck was inert
there.

**Fix.** `#pragma omp parallel for collapse(3) schedule(static)` on every
residual nest (`emit_planck_term`'s `omp_collapse`). A guard
(`len(free) >= omp_collapse`) leaves too-shallow nests alone; none of 948
pragmas collapses more levels than it has. HF/6-31G through the generated
rank-3 route (g++-15, Release, MAXORDER=4, dressed + merged):

| step | change | 1t | 4t | speedup |
|---|---|---|---|---|
| baseline | none | 78.67 s | 78.49 s | 1.00× |
| `part1` only | 256 nests | 78.67 s | 40.85 s | 1.93× |
| all four parts | 806 nests | 78.67 s | 27.98 s | 2.81× |
| + dead builds removed (#6) | | 73.79 s | 23.70 s | 3.11× |
| from the emitter | + singles/doubles | 74.37 s | 23.11 s | 3.22× |

- `part1` alone hit its Amdahl ceiling (1.93× against 1.94× for a 64.6 %
  part). The 3.22× is against the threaded binary at one thread (74.37 s);
  against the 80.92 s pre-work binary it would read 3.5× and price a rebuild
  as threading. 8 threads: 18.31 s, on a machine with 4 performance cores.
  CPU 99.1 % → 359.9 %.
- **`collapse(3)`, not `collapse(2)` or a bare `parallel for`.** The outer
  index has o = 5 trips on every test system (125 after collapsing three
  levels). `collapse(3)` beat `collapse(2)` by ~4 % (40.85 against 42.63 s).
  The collapsed indices are output indices, so writes stay disjoint at any
  depth.
- **Default since 2026-11-20.** CMake option `PLANCK_CC_OMP_COLLAPSE`
  (default 3; 0 emits no pragma) is passed to the RCC, UCC and CCSD(T)
  generator runs as `CCGEN_OMP_COLLAPSE`. Custom commands cannot see cache
  variables, so the value is also written to
  `build/generated/cc/omp_collapse.stamp`, rewritten only when it changes,
  which every generator command depends on. A direct run of
  `generate_planck_cc_kernels.py` without the variable emits no pragma.
  Pragma counts: `ccsdt` 948, `ccsdtq` 11,995, `ccsdtq_ucc` 18,425.

| default switch, 2026-11-20 | single-core baseline | 4 threads | 8 threads |
|---|---|---|---|
| `oh_uccsd_t_631g` (generated UCC + (T)) | 12.25 s | 3.59 s (3.4×) | 2.90 s |
| `be_rccsdtq_sto3g` (generated rank 4) | 1.50 s | 1.75 s | 2.70 s |
| `water_rccsd_t_631g` (hand-written CCSD) | 1.07 s | 0.97 s | 0.97 s |

Be/STO-3G rank 4 gets slower: about 12,000 short nests per iteration, each
paying a parallel-region start.

## 6. Every operator built twice

While threading the triples builders: `compute_ccsdt_triples_residual`
emitted its 88 operator builds twice, once as unused `const auto` locals and
again into the `ops` struct the parts read. `_emit_kernel` emitted the
builds unconditionally and then delegated to `_emit_chunked_kernel`; #3 added
the struct without removing the superseded emission. Worth 4.9 s serial /
4.3 s at 4 threads on HF/6-31G (~2.6 %), scaling with operator count (rank 4
has 894). Fixed by a `chunked` guard in `_emit_kernel`. A provably unused
computation is invisible to every value gate (Measuring rule 9).

## 7. Residual terms left n-ary by seeded-operator inlining

**Cause.** After #2, 112 residual terms of spin-adapted `ccsdt` were still
three-tensor products: 68 triples, 33 doubles, 11 singles. The worst were 25
t2·t3·⟨oo‖vv⟩ at o⁵v⁵, about 74 % of modelled FLOPs. The derived route's emit
helpers (`emittable_operators`, `rewrite_term_factorized`,
`manifold_operators`) defaulted to `seeded_fingerprints()`, the six
Stanton–Gauss operators of the retired recognition route:
- a sub-contraction matching one of them (Wmbej, Wmnij, Fae, Fmi) was
  classified `Reuse`;
- `Reuse` children are always inlined, from when recognition was expected to
  dress them;
- the derived route emits no `Reuse` builders (`include_reuse=False`), so
  nothing built them and the term stayed n-ary.

**Fix.** `EMIT_FINGERPRINTS = ()` in `python/ccgen/optimization/factorize.py`:
every sub-contraction is derived from its own tree. The analysis helpers
(`identify_tree`, `value_operators`, `reused_operators`) keep the seeded
default, since reporting reuse is their job. CH4 11.0 → 3.01 s on one thread.
A 1-thread profile went from 90 % triples residual to 50 %, with builders at
48 %.

**Operator counts rose:** the derived route builds one array per
contraction shape (a shared array across shapes computed the wrong tensor,
D6), and spin adaptation splits terms into direct and exchange variants.

| Manifold | Before | After |
|---|---|---|
| ccsd doubles, GCC | 27 | 38 (+11: Wmbej 4, Fme 3, Wmnij 2, Fmi 1, Fae 1) |
| ccsd doubles, spatial | 59 | 81 |
| ccsdt triples | 80 | 94 |
| ccsdtq quadruples | 254 | 268 |
| spatial `ccsdt` emitted builders | 91 | 104 |

**Why the August census missed it.** It reported "factorized: 414 nests, 0
four-deep". Censused the same way at `ac80d93f`: GCC plain TU 9 four-deep
triples nests, spin-adapted plain TU 25, spin-adapted arbitrary TU (the one
that runs) 25. The seeded inlining dates from 2026-08-04, before the census,
so no configuration of that code gives 0. The census was taken on a
`build-profile/` tree that no longer exists, with unrecorded flags.

## 8. Unthreaded builders

#5 put the pragma on residual nests only. After #7 the builders were 48 % of
a 1-thread run, and four threads gained 1.6×. Builder nests now get the same
pragma, 92 of them (`_emit_intermediate_builder`): each nest writes disjoint
output elements, and each step of a factored builder is its own parallel
region, so the barrier orders a scratch write before the next step reads it.
It cost 12 % on one thread at the time, all in the builders (probably
outlined OpenMP bodies optimizing worse; unconfirmed). Kept because
production runs threaded.

An earlier recommendation in this record said not to thread the builders (in
August they were 12–13 % and mostly dead work, #6). That held then; after #7
they were half the run.

## 9. The triples residual over every equivalent element

**The symmetry.** The spin-adapted t3 is the aabaab block: (ia) and (jb)
alpha, (kc) beta. Its residual is antisymmetric under a↔b and under i↔j and
symmetric under both (measured to ≤ 4e-15 with an aab-correct t3). Nothing
relates (kc) to the other pairs, so the group has order 4,
`{e, (ab)⁻, (ij)⁻, (ab)(ij)⁺}` (`AAB_TRIPLES_GROUP`). An earlier census
assumed all six pair permutations and gave 1.62×, from the wrong group.

**The construction.** Write R = Σ_T c_T·T. Then |G|·R = Σ_g s_g·g(R). For
T = h(rep), Σ_g s_g·g(T) = s_h·Σ_g s_g·g(rep). So R = Σ_g s_g·g(X), with
X = (1/|G|) Σ_orbits κ_O·rep_O and κ_O = Σ_{T∈O} c_T·s_h. Exact whenever R
has the symmetry, with no need for complete orbits: orbits are matched on
the term's text, which can find too few, never a wrong one. R is then
exactly antisymmetric, so t3 stays antisymmetric.
- **Fold:** `fold_external_symmetry` (`python/ccgen/optimization/permutation.py`),
  in `print_cpp_planck` on spin-adapted `triples`, after spin adaptation and
  before factorization, so no operator is built for a folded-away term. The
  rank-4 TU's triples kernel is folded too.
- **Pass:** `_symmetrize_lines` (`python/ccgen/emit/planck_tensor_cpp.py`)
  emits one threaded pass over a copy of X before `return result;`. Its
  `// symmetrize ijkabc: ...` comment is what the numpy evaluator reads.

## 10. Flat builder bodies

**Cause.** On C2H6 (one thread) the builders then took 82 %, and
`W_t1t3v_oooovv` alone 55 %: one flat t3·t1·⟨oo‖vv⟩ nest at o⁵v⁴, where the
factorizer's tree (t1·v first, then t3) is o⁵v³. M3.0 (2026-08-04) had built
factored builder bodies (`factored_builder_steps`,
`docs/CCGEN_INTERMEDIATE_MEMORY_LOCALITY.md` design rule 3), but only on
`emit_factorized_translation_unit`, which production never calls. W3.2
(2026-08-27) wired the derived route into `print_cpp_planck` with
`factorize_equations` and without `factor_builder_bodies`.

**Fix.** `print_cpp_planck` passes `factor_builder_bodies=True` on the
derived route; the rank-3 TU carries 41 scratch-step tensors.

**Measured** (earlier ladder geometries, not comparable rung-for-rung with
Where the two routes stand), s per generated iteration at four threads:

| Rung | o | v | before → after | gain |
|---|---|---|---|---|
| CH4/STO-3G | 5 | 4 | 0.050 → 0.035 | 1.44× |
| HF/6-31G | 5 | 6 | 0.181 → 0.098 | 1.85× |
| H2O/6-31G | 5 | 8 | 0.497 → 0.248 | 2.01× |
| NH3/6-31G | 5 | 10 | 1.139 → 0.521 | 2.19× |
| H2O/6-31G* | 5 | 14 | 4.182 → 1.757 | 2.38× |
| C2H2/STO-3G | 7 | 5 | 0.403 → 0.192 | 2.10× |
| C2H4/STO-3G | 8 | 6 | 1.419 → 0.587 | 2.42× |
| C2H6/STO-3G | 9 | 7 | 4.329 → 1.570 | 2.76× |

The 4-thread per-iteration fit moved from o^4.30 · v^3.55 to o^3.63 · v^3.17,
and the ratio to hand-written from o^0.78 · v^−0.24 to o^0.11 · v^−0.63 (LOO
max 31 % and 24 %; see the four-thread caveat above). This is when the gap
stopped growing with size. The August census missed n-ary builders for two
reasons: it counted builder calls (#3), never what one builder computes, and
every measurement had o = 5, where o⁵v⁴ hides among o⁴v⁴ work.

## 11. Runtime-rank amplitude reads

**Cause.** After #2–#10, C2H6 on one thread ran at 0.37 G multiply-adds/s,
about 8 cycles each. The loop bounds give 2.1e9 multiply-adds per iteration:
75 % triples residual (331 nests), 22 % builders (145 nests), 2 % singles
and doubles; largest shapes triples o⁴v⁴ 29 %, o⁴v³ 21 %, o⁵v³ 18 %, builder
o⁵v³ 15 %. A profile put 63 % in triples part 0 and 23 % in builders. It said
where, not why.

**Disassembly of the hottest nest** (triples part 0, nest 0,
`acc += -t2({i,l,a,b}) * ooov(j,k,l,c)`, `l` innermost). Even at `-O3`, one
`l` step is about 40 instructions around a single scalar `fmsub`:
- `nd_flat_index` runs as a real loop over `dims.size()`, a runtime value:
  4 trips × 6 instructions;
- the index array (the `initializer_list`) is written to the stack every
  step;
- the fixed-rank `ooov` read beside it reloads its dims and data pointer
  every step, because those `int` stores might alias its `int` dims;
- the add chain is not the limit: one `fmsub` per ~40 instructions hides
  its latency.

**One-nest benchmark.** The nest extracted with the real tensor types and
pragma, runtime sizes, `noinline`, one thread; every variant bitwise
identical. ns per multiply-add at the C2H6 shape (o = 9, v = 7):

| Header, level | emitted | reordered | fixed-rank view |
|---|---|---|---|
| current, `-O1` (production) | 2.41 | 0.80 | **0.42** |
| current, `-O2` | 2.11 | 0.75 | 0.38 |
| current, `-O3` | 2.02 | 0.58 | 0.37 |
| `nd_flat_index` bounded by `indices.size()`, `-O1` | 1.81 | 0.82 | – |
| same, `-O2` | 1.75 | 0.69 | – |
| same, `-O3` | 0.37 | 0.29 | 0.37 |

- **A fixed-rank view gives 5.7× at `-O1`,** within 12–18 % of the best
  setting; C2H4 (8, 6) and H2O/6-31G* (5, 14) gain 5.6× and 4.9×.
- **Bounding the runtime loop needs `-O3`** to unroll completely (≤ 1.35× at
  `-O1`/`-O2`). An `initializer_list` read stays slow at `-O1` even with a
  constant bound, so the emitted reads take plain arguments.
- **Reordering** (`l` outside `c`, one accumulator per `c`, same sum order)
  gives 2.8–3.5× without the fix by moving the accessor out of the inner
  loop; on top of the view at `-O3` it adds nothing at v = 6–7 and up to
  1.7× at v = 14.

**Fix.** `ConstFixedRankView<N>` (`src/post_hf/cc/common.h`): a data pointer
and `int dims[N]`, offset expanded by a fold over the rank;
`fixed_rank_view<N>(view)` binds it from a `ConstDenseTensorView`.
`_amplitude_view_bindings` binds every amplitude and sector view through it
once per kernel, and `_map_tensor` emits `t2(i, l, a, b)`. RCC and UCC both
change. The production class matches a hand-written view exactly in the
benchmark (0.42 / 0.44 / 0.52 ns).

| Rung | 1 thread, s/iteration: before → after | gain | 4 threads: before → after | gain |
|---|---|---|---|---|
| CH4/STO-3G | 0.0788 → 0.0154 | 5.1× | 0.0345 → 0.0145 | 2.4× |
| H2O/6-31G | 0.837 → 0.142 | 5.9× | 0.245 → 0.051 | 4.8× |
| H2O/6-31G* | – | – | 1.757 → 0.331 | 5.3× |
| C2H4/STO-3G | 2.110 → 0.352 | 6.0× | 0.579 → 0.111 | 5.2× |
| C2H6/STO-3G | 5.870 → 0.958 | 6.1× | 1.607 → 0.299 | 5.4× |

The gain grows with size and exceeds the one-nest 5.7×. CH4 at four threads
gains least because its time is per-call overhead.

## The hand-written solver's convergence test

**Symptom.** On C2H2/STO-3G the routes differed by 1.0e-8 Eh (−0.1700991464
generated, −0.1700991567 hand-written), ≤ 1.5e-9 elsewhere.

**Cause.** `run_restricted_tensor_rccsdt_no_fallback`
(`src/post_hf/cc/tensor_backend.cpp`), the hand-written solver these systems
use, accepted when `|dE| < 1e-10` and the amplitude step `< 1e-6`, a fixed
value (PySCF's default `conv_tol_normt`). The step test was always met, so
|dE| alone decided. On C2H2 one step had dE = 9.5e-11 while the previous one
moved 1.5e-8; the run stopped with rms(R3) = 2.9e-8. The generated route
converges its residual to `tol_density` (1e-10 by default).

**Fix.** The step tolerance is `tol_density`, as in the hand-written RCCSD
warm start and the generated route. Its iteration cap,
`min(64, max(24, 2·max_cycles))`, became `max(max_cycles, 100)`, the
generated route's: under the stricter test CH4 needed all 64 steps and
C2H4's cap was 44, and this path has no fallback. The staged triples loop
(`run_staged_tensor_triples_iterations`) had the same kind of looseness, a
1e-7 floor under its residual tolerance (`std::max(1e-7, tol_density)`), and
now uses `tol_density` too.

| Rung | hand-written, before → after | generated | gap after |
|---|---|---|---|
| CH4/STO-3G | −0.0791116946 → −0.0791116949 | −0.0791116946 | 3e-10 |
| HF/6-31G | −0.1319388447 → −0.1319388445 | −0.1319388447 | 2e-10 |
| H2O/6-31G | −0.1364577851 → −0.1364577853 | −0.1364577853 | 0 |
| NH3/6-31G | −0.1312497552 → −0.1312497564 | −0.1312497563 | 1e-10 |
| H2O/6-31G* | −0.1993312224 → −0.1993312220 | −0.1993312220 | 0 |
| C2H2/STO-3G | −0.1700991567 → −0.1700991465 | −0.1700991464 | 1e-10 |
| C2H4/STO-3G | −0.1620505321 → −0.1620505323 | −0.1620505306 | 1.7e-9 |
| C2H6/STO-3G | −0.1469878956 → −0.1469878953 | −0.1469878949 | 4e-10 |

C2H4's 1.7e-9 is convergence, not disagreement: at `tol_density 1e-12` the
generated route reaches −0.1620505312, between the two default-tolerance
values.

## Measuring rules

These came from wrong conclusions this record made and later corrected.

1. **Compare generated against generated.** A generated/hand-written ratio
   prices two solvers. Attribute a runtime effect to a codegen change only by
   comparing configurations of the same route one setting apart. Never set a
   target from a cross-route ratio. Do not mix hand-written and generated
   residual sources: that hybrid gave a self-consistent wrong answer
   (−7.56e-05).
2. **Profile before modelling, and treat a profile share as a lower bound.**
   The census-and-FLOP model went 1 for 2 (contraction order hit at 3.6×,
   predicted 10–18×; fusion predicted large, measured ~0 %). A twenty-minute
   `sample` profile found #3, which no model priced. Every model here
   underestimated, always by treating units of work (operators, ranks) as
   equal-cost. A realization factor belongs to the work one fix removed; it
   is not a haircut for the next estimate. An operator count is not a
   speedup.
3. **A model predicts direction, not magnitude.** Trust a FLOP census for
   exponents (it reproduced a measured o exponent to 0.05), never for a
   ratio's size. State a falsification criterion first, and test it at a
   size that can falsify it (BH3 is L1-resident and cannot test a
   cache-traffic claim).
4. **Fit only what the ladder varies.** A power law in k variables needs all
   k varied independently. Validate by leave-one-out or the design-matrix
   condition number, never by the residual. Do not extrapolate beyond the
   measured range. Several rungs need o ≠ 5.
5. **Check the build before trusting a number.**
   - `grep '^PLANCK_CC' <build>/CMakeCache.txt`: a dropped flag redefines the
     manifold under test (the first fusion build lacked
     `PLANCK_CC_DRESS_OPERATORS=ON`).
   - Set `CMAKE_BUILD_TYPE` explicitly: without it there is no `-DNDEBUG`,
     the tensor bounds asserts return, and every timing is meaningless (one
     wrong diagnosis already).
   - A/B in one configure, both arms rebuilt from the same source. Never
     compare against a stale `build/`.
   - Do not copy a build tree: `CMAKE_CACHEFILE_DIR` is absolute, so the copy
     rebuilds into the original.
   - `make` regenerates the kernels (`ccgen-planck-kernels` is an
     unconditional dependency), silently wiping hand edits to generated
     files. Make a toggle a build option, or compile and link the object
     directly, and check a structural count (`grep -c "pragma omp"`,
     `nm | grep GOMP_parallel`, symbols, TU bytes), not only a timing.
   - Do not use `git stash` for a historical emitter; use
     `git archive <commit> python` into a scratch tree.
   - `make -j4`: the generated TUs make a full-width build disruptive.
6. **Check the working set against the cache** before attributing a cost to
   memory traffic. At o = v = 4 the residual is 32 KB, L1-resident.
7. **A CPU percentage alone does not prove a path unthreaded.** Pair it with
   a positive control in the same binary, measured warm, and confirm
   `-fopenmp` reaches the compile line. Pick a control genuinely bound by a
   threaded path (a small DFT case is too small; water/cc-pVDZ ultrafine is
   grid-bound and reaches 1.09×).
8. **Threads must not change bits.** No cross-thread reduction; a future
   term shape that needs one sums in fixed thread order, never completion
   order (the DFT grid jitter). The structural argument is a reason to
   expect determinism, not evidence: verify E_corr, dE, rms(res) and
   rms(step) at every iteration at 1/2/4/8 threads.
9. **Read the emitted text of the production route.** A property verified
   on Python objects, or on one artifact, was twice assumed for what
   production emits, and both times was false (#7, #10). An n-ary term or a
   duplicate build computes the right value slowly, so no value gate sees
   it. Fix the emitter, never the generated files.
10. **Gate fixtures carry exactly the physical symmetries.** Withholding one
    the object has manufactures failures (the merge gate's first run: 6.4e-2
    on a random t2, 2.8e-17 with `t2(a,b,i,j) == t2(b,a,j,i)`); granting one
    it lacks hides defects (`random_tensors` antisymmetrizes v, under which
    invalid ERI relations hold). Use non-square o/v: on a square system a
    wrongly ordered read stays in bounds.
11. **A gate must reach the mechanism.** The merge gate first covered
    singles and doubles only; all 56 `t2t2v_oooovv` reads live in the triples
    `_partN` chunks, and a perturbation moved it by 2.2e-16.
12. **Rank 3 is not a proxy for rank 4.** Different tensor types and code
    paths (the fixed-rank-only accessor pass, #1). Rank ≥ 7 targets use the
    braced accessor `result({i, j, ...})` and rank ≥ 7 builders return
    `TensorND`; both break tools written for rank 3 with messages naming
    neither cause. Rank 4 needs both t4 Sz sectors in fixtures (`t4` and
    `t4_aaabaaab` are independent).
13. **Time the code you mean.** An isolated residual evaluation from
    identical amplitudes measures a kernel; a converged solve measures a
    solver. Do not compare absolute `sample` counts across runs of different
    length; use wall-clock and within-run shares.

## Measuring the ladder

- Each rung is a copy of `tests/inputs/regression/post_hf/ch4_rccsdt_generated_sto3g.hfinp`
  with the molecule swapped, run with `rccsdt_backend optimized`
  (generated) or `tensor` (hand-written). The 2026-11-21 inputs and driver
  lived in a session scratchpad; the geometries are not those of the earlier
  ladders.
- Generated per-iteration time is the sum of the solver's `t=` over rank-3
  iterations. The hand-written backend prints none, so its CC time is its
  wall time minus the generated run's non-CC time, over its `TENSOR-R`
  iterations.
- The hand-written arm needs `nso > 16 || ndet > 10000`, or
  `choose_determinant_backstop` routes it to the determinant prototype and
  no tensor timing exists (`water_rccsdt_sto3g`, nso 14, is unusable). The
  generated route never consults the backstop.
- LiH and N2 stay out of fits: the hand-written run converges in its stage-1
  warm start and never enters the tensor loop.
- Rank 4: `-DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_SPIN_ADAPT=ON`,
  `tests/inputs/regression/post_hf/be_rccsdtq_sto3g.hfinp`, per-iteration
  `t=` on the `RCCSDTQ[TENSOR]` lines.
- Set `BASIS_PATH` if the compiled-in basis path points at a stale install.

## The `-O1` registry pin (measured, kept)

Every generated kernel compiled into `generated_kernel_registry.cpp` is
pinned to `-O1` by `CMakeLists.txt`, set when the rank-4 quadruples were
single ~5000-statement functions and `-O3` took 40+ minutes; its claim that
the loops "gain almost nothing from -O3" had no measurement. Measured
2026-11-21 before #11, registry object compiled outside the build tree and
linked into scratch binaries, s per iteration:

| Rung | 1 thread: `-O1` | `-O2` | `-O3` | gain | 4 threads: `-O1` | `-O3` | gain |
|---|---|---|---|---|---|---|---|
| CH4/STO-3G | 0.0788 | 0.0657 | 0.0639 | 1.23× | 0.0345 | 0.0300 | 1.15× |
| H2O/6-31G | 0.837 | 0.740 | 0.727 | 1.15× | 0.245 | 0.224 | 1.09× |
| H2O/6-31G* | – | – | – | – | 1.757 | 1.569 | 1.12× |
| C2H4/STO-3G | 2.110 | 1.849 | 1.803 | 1.17× | 0.579 | 0.547 | 1.06× |
| C2H6/STO-3G | 5.870 | 5.116 | 4.992 | 1.18× | 1.607 | 1.415 | 1.14× |

- E_corr identical at all three levels. C2H6 takes 17 iterations at `-O1`
  and 15 at `-O2`/`-O3` (1.33× end to end there); probably different
  rounding shifting the DIIS path, unverified.
- Registry compile: 1067 s at `-O2`, 1270 s at `-O3`, a few minutes at
  `-O1`. The chunked kernels did not make higher levels cheap.
- The rank-3 triples TU included by `tensor_backend.cpp` is `-O3`
  regardless; the pin applies to the registry.

## Gates

- **Energy:** `ch4_rccsdt_generated_sto3g`, `lih_rccsdt_generated_sto3g`,
  `be_rccsdtq_sto3g`, `water_rccsd_t_sto3g`, `water_rccsd_t_631g`,
  `b_ucc{2,3,4}_sto3g`, `oh_uccsd_t_631g`. Cross-check: CH4/STO-3G against
  PySCF `rccsdt`, +1.49e-08. `ch4_rccsdt_sto3g` fails as one of the four
  known routing failures.
- **Bitwise, after any change to builders or residual emission:** E_corr,
  dE, rms(res), rms(step) at every iteration, against the previous binary and
  at 1/2/4/8 threads (checked on CH4 and H2O/6-31G for #11).
- **Value** (`python/ccgen/tests/test_factorize_all_binary.py`): singles,
  doubles and triples from the derived, folded emit equal the undressed,
  unfolded emit to ≤ 1e-12 relative, with an aab-block t3 and full
  real-orbital ERI symmetry. A flipped operator-read sign fails at 1.1e-2, a
  flipped group sign at 1.0 relative.
- **No n-ary statements** (`test_no_nary_statements`) in any residual or
  builder of `ccsd` and `ccsdt`. Seeded fingerprints fail it with 39 / 75
  n-ary residual statements (ccsd / ccsdt), flat builder bodies with 14 / 38
  builder statements.
- **Merged call sites** (`test_merged_call_sites.py`): residual emitted
  merged and unmerged must agree, at ranks 3 and 4; mutations caught at
  1.5e-3 to 8.6e-3. Needs rank ≥ 3: every `ccsd` merge permutation is a
  self-inverse swap. Complemented by the definition-only
  `test_emitted_builder_matches_spec.py` (91 builders).
- **Builds once** (`test_chunked_kernel_builds_once.py`): no operator built
  twice and no `const auto` build before the `ops` struct; mutation-verified.
- **Accessor layout** (`planck-cc-tensor-index`, `tests/cc_tensor_index.cpp`):
  flat index against an independent row-major reference on non-square shapes
  at each rank, the `swap_mid_axes` pattern of `rebind_physicist`, and the
  braced overload against the out-of-line `vector<int>` one.
- **No braced amplitude reads** (`test_optimizations.py`): asserts the
  `fixed_rank_view<8>` binding and that no `t4({` read remains.
- **Loop fusion** (`test_loop_fusion_grouping.py`) pins the grouping against
  emitted text; the default emit is byte-identical without it.

## Ruled out

- **Loop fusion.** 806 nests share 15 `(free, summed)` signatures;
  `CCGEN_FUSE_LOOPS=N` fuses them (806 → 15, half the TU, 845 KB less
  binary) and changes runtime by 0–3 %, non-monotonic, at BH3 (0.031 MiB
  t3), CH4 (0.061) and HF/6-31G (0.21), before and after #3. Its traffic
  model (32× fewer o³v³ passes) fails its own falsification test. Kept as an
  option, default 0. Traps found building it: the triples kernel goes
  through `_emit_chunked_kernel`, which had its own term loop; chunks are
  contiguous, so groups must be ordered before chunking; routing the chunked
  path through the shared helper broke byte-identity.
- **Memory traffic from loop fission (H1)** at reachable sizes. At o = v = 4
  the residual is 32 KB and fissed loops vectorize better (0.62×). The
  isolated-kernel ladder topped out at 0.84 MiB t3, inside L2, so a
  cache-spill effect was never testable (see Not done).
- **Loop-invariant work.** None of 824 nests accumulates while ignoring any
  of i, j, k, a, b, c.
- **The harness.** Per-iteration phases on CH4: `t_res` is all of `t=`
  (DIIS and energy unmeasurable). Per rank: rank 1 0.04 %, rank 2 1.1 %,
  rank 3 98.8 %. Evaluating every rank each iteration costs ≤ ~1 %.
- **The `-O1` pin** as a main cause (above).
- **The add chain** as the limit of the hot nests (#11).
- **Factorizing before spin adaptation.** `print_cpp_planck` already
  factorizes after `spin_adapt_equations`.
- **All six pair permutations as the triples symmetry** (#9).
- **Iteration count:** the generated route takes fewer.
- **Chunk imbalance as a defect.** Chunks split by term count: at o = 5,
  v = 6 the four triples parts model at 2.46e8, 6.68e8, 3.72e7 and 5.13e6
  (deepest inner sums o²v¹, o²v², o²v⁰, o¹v⁰). Same total work; it only
  matters for per-part parallelism.

## Key code locations

| What | Where |
|---|---|
| tensor accessors, `ConstFixedRankView` | `src/post_hf/cc/common.h` |
| hand-written triples (one nest) | `build_dressed_triples_residual`, `src/post_hf/cc/tensor_backend.cpp` |
| determinant backstop | `choose_determinant_backstop`, same file |
| generated harness, per-phase and per-rank timing | `run_generated_arbitrary_order_iterations`, `evaluate_generated_arbitrary_order_residuals` (`PLANCK_CC_RANK_TIME`), `generated_arbitrary_runtime.cpp` |
| term, kernel and chunk emission; `ops` struct; pragmas; symmetrize pass; amplitude bindings | `python/ccgen/emit/planck_tensor_cpp.py` |
| derived-route switches (factored builders, merge, fold) | `print_cpp_planck`, `python/ccgen/generate.py` |
| fingerprints, merge, factored builder steps | `python/ccgen/optimization/factorize.py` |
| triples fold | `fold_external_symmetry`, `python/ccgen/optimization/permutation.py` |
| unused contraction-order / GEMM analysis | `python/ccgen/tensor_ir.py` |
| `-O1` registry pin | `CMakeLists.txt` |

Related answers: `docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md` (why the generated
rank-3 path runs in the arbitrary-order harness, and the storage
convention), `docs/CCGEN_HIGHER_OPERATOR_REUSE.md` (factorizing to cut FLOP
scaling), `docs/CCGEN_INTERMEDIATE_MEMORY_LOCALITY.md` (intermediate
selection under a memory budget; symbolic, not timed),
`docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`,
`docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`.

## Not done

- **Default routing.** A dressed build sends plain `correlation ccsdt` to the
  generated route, now the faster route at four threads on every ladder rung
  but CH4. Whether to keep that default is open; any change must be checked
  against the four known rccsdt routing failures.
- **Hand-written timings after its convergence fix.** The hand-written
  columns of Where the two routes stand were measured with the old, loose
  test. Fixed, it takes more iterations (H2O/6-31G* 27 → 56, about 22 → 41 s
  wall); per-iteration cost is unchanged. Re-run the ladder before quoting
  wall ratios.
- **Hand-written plateau.** Its residual stalls near rms(R3) ≈ 1.7e-10, so
  with `tol_density` below that it reports non-convergence (C2H4 at 1e-12).
- **Loop order.** The summed index is innermost and strided; reordering adds
  up to 1.7× on top of #11 at v = 14 in the one-nest benchmark. An emitter
  rule, not measured on a whole run.
- **Rank 4 timing.** #3, #4, #5 and #11 all apply at rank 4 and
  `be_rccsdtq_sto3g` passes, but no rank-4 system large enough to time has
  been measured: not merged against unmerged, not threaded, not with the
  fixed-rank view. Be/STO-3G is slower threaded (per-nest overhead); a
  minimum-work threshold per nest would fix that if a small case matters.
  Rank-4 `result` and builders at rank ≥ 7 still use the runtime-rank
  `TensorND` accessor. `be_rccsdtq_sto3g`'s manifest value −14.4036551081
  differs by 6.2e-08 from the dressed merged run (−14.4036550465), a
  discrepancy older than this work.
- **Rank 2.** The generated RCCSD equals the hand-written RCCSD(T) to
  ≤ 3e-10 Eh on all four `*_rccsd_t_*` inputs. Re-timed 2026-11-22 after
  #7–#11: C1 H2O2/6-31G 0.54 s against 32.0 s hand-written (was 55.4 s
  against 32.1 s), spherical water/cc-pVDZ 0.30 s against 14.5 s (was
  19.6 s against 14.8 s), 4 threads; full table in
  `docs/CCSD_T_SIZE_SCOPE.md`. Its doubles residual is symmetric under
  (ia)(jb) (5.6e-16), an order-2 group the fold could use; at these sizes
  that is not worth doing.
- **Cache spill (H1).** No measured system has t3 beyond L2; cc-pVDZ-class
  cases (H2O/cc-pVDZ, 6.5 MiB t3) would test it.
- **Beyond four threads.** The measuring machine has four performance cores.
- **The hand-written backend is unthreaded.**
- **Small rungs** (CH4, C2H2) are dominated by per-call overhead and barely
  thread.

## Not in this scope

BLAS/GEMM lowering of the generated nests, RI, MPI, intermediate selection
under a memory budget (`docs/CCGEN_INTERMEDIATE_MEMORY_LOCALITY.md`), and
the correctness, spin-adaptation, UCC, dressing-route and CCSD(T)/(Q)
questions, which have their own docs.
