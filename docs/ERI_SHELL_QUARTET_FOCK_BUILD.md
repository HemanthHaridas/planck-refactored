# Why was the memory-direct Fock build slow, and what made it fast?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`.

This file answers one architecture question:

**The fused (memory-direct) Fock build took 1.50 s for 8 waters in 6-31G at
one thread. Where did that time go, and which changes to the integral kernels,
the digestion loop and the threading actually paid?**

## Short answer

The build evaluated integrals one Cartesian component at a time, and every
component call reran the whole primitive loop, 27–78× per shell quartet
wherever HGP ran. Evaluating whole shell quartets removed that repetition:
closed forms at total L ≤ 3 and an HGP block kernel at 4–12. That was most of
the gain. Then the cost moved twice. Inside the kernels it became per
primitive quartet overhead, which only fewer or batched primitive passes cut
(lanes, batched Boys). Outside them it became digesting one component at a
time, which dense per-block digestion cut. Last came the threads: a
triangular loop needs a cyclic schedule, and partials keyed by a fixed
partition of the work keep G identical at any thread count while still being
scheduled dynamically.

| fixture (one thread, `engine auto`) | before | after | speedup | after, four threads |
|---|---|---|---|---|
| 8 H2O / 6-31G (nb = 104) | 1.50 s | 0.105 s | **14×** | 0.0284 s |
| 4 H2O / 6-31G\* (nb = 76) | ~1.12 s† | 0.0625 s | **~18×** | 0.0169 s |
| 4 H2O / cc-pVDZ | 2.573 s§ | 0.358 s | **7.2×**§ | 0.0959 s |
| 1 H2O / cc-pVTZ | 1.631 s§ | 0.0698 s | **23×**§ | 0.0193 s |
| naphthalene / 6-31G, D2h (nb = 106) | 1.454 s | 0.188 s‡ | **7.7×** | 0.0514 s‡ |

`planck-eri-class-share --k1` production arm, Apple M3, g++-15 `-O3`, Boys
core v1.2.0, best of three or more interleaved, 2026-10-24. "Before" is the
build before C1, after the Boys fix C0 (itself 1.37× / 1.13×). † The product
of per-step ratios, never timed. § Against the D0 build (after P1, before S1
and D1). ‡ With `sym_ops`, on a group-averaged density. Fock-build numbers,
not SCF wall time. Explicit engines get only the engine-independent steps:
Auto is ~7× and ~11× faster than `engine hgp` on 8 H2O / 6-31G and
4 H2O / 6-31G\*. The ~200× libcint gap in `docs/ERI_PERFORMANCE_SCOPE.md`
predates this work and has not been re-measured.

Work stopped on 2026-10-24 with the kernel ideas measured and either landed
or declined. What was left undone is under "Not done".

## Where the logic lives

- `src/integrals/fused_fock.h`: `fused_fock_build`, the one shell-quartet
  loop for every engine. Picks the dense or per-component traversal per
  build, runs the bins, merges and symmetrizes G.
- `src/integrals/block_eri.h`: the block-kernel contract (`prepare`,
  `value`, `covers`), `ChainBlockKernel`, `grow_to`.
- `src/integrals/low_l_eri.h`: `low_l_shell_eri` and `ClosedFormBlockKernel`
  (total L ≤ 3).
- `src/integrals/hgp_block_eri.h`: `HgpBlockKernel` and `hgp_block::vrr_lanes`
  (total L 4–12).
- `src/integrals/quartet_orbit.h`: orbit representatives, S1 weights,
  `symmetrize_over_ops`.
- `src/integrals/rys.cpp`: Auto's block-kernel chain.
- `src/lookup/loader/orchestrator.cpp`: `boys_batch`, the batched Boys entry
  (core v1.2.0 required).
- `tests/eri_class_share_benchmark.cpp` (`planck-eri-class-share`, not a
  ctest): the measuring instrument.

## How the build works

`fused_fock_build` chooses one of two traversals **per build**, because they
visit different block sets and mixing them double-counts or drops quartets.

**Dense** is taken when the block kernel `covers()` every total L the basis
can produce (Auto, Coulomb, shells up to f), every density is symmetric to
1e-10, and there is no symmetry or the skeleton scheme applies. Per bra shell
pair p and every ket q ≥ p: block Schwarz; the block prepared lazily at its
first surviving component; then each component above the component Schwarz
bound digested as v′ = deg · weight · value, with
deg = (A=B?1:2)(C=D?1:2)(p=q?1:2), into six writes (two Coulomb, four
exchange) from a body instantiated per active write group, and (G + Gᵀ)/2 at
the end. **Per-component** is everything else: explicit engines, screened
kernels, total L > 12, non-symmetric densities, symmetry by replication. It
filters by block and by component and accumulates each value over its
distinct 8-fold permutations.

**Symmetry (S1).** When every density has the molecule's symmetry, each
non-forbidden orbit representative is accumulated once, weighted by its orbit
size, and G is symmetrized at the end:

```math
G_{\mu\nu} = \frac{1}{|G|}\sum_R s_R(\mu)\,s_R(\nu)\,G^{\mathrm{skel}}_{R(\mu)R(\nu)}.
```

Under the dense traversal, a per-build table of each operation's image of
each shell pair decides most operations per block with two comparisons; only
operations that map the block onto itself are applied per component (K1.4).

**Kernels.** Auto chains `ClosedFormBlockKernel{3}` with
`HgpBlockKernel{4, 12}`, one copy per thread. The closed forms treat a quartet
as up to three *raises* (center, axis). Per primitive quartet, with
$`\theta_m = c\,K_{AB}K_{CD}\,2\sqrt{\rho/\pi}\,F_m(T)`$, $d$ the raised
center's $P-A$, $P-B$, $Q-C$ or $Q-D$, and $w$ its $W-P$ or $W-Q$:

```math
\begin{aligned}
A_i(m) &= d_{1,i}\,\theta_m + w_{1,i}\,\theta_{m+1} \\
B_{ij}(m) &= d_{2,j}\,A_i(m) + w_{2,j}\,A_i(m+1) + [i=j]\,\kappa_{12}(\theta;\,m) \\
C_{ijk} &= d_{3,k}\,B_{ij}(0) + w_{3,k}\,B_{ij}(1) + [k=i]\,\kappa_{13}(A^{(2)}_j) + [k=j]\,\kappa_{23}(A_i)
\end{aligned}
```

$\kappa_{yx}$ couples a raise to an earlier one it lowers: on the bra
$\bigl(X(m) - \tfrac{\rho}{\zeta}X(m+1)\bigr)/2\zeta$, on the ket the same with
$\eta$, across $X(m+1)/2(\zeta+\eta)$. The HGP block kernel runs the
Obara–Saika VRR into an **L-major** table [e0|f0]^(m), four primitive quartets
per pass in blocks with K ≥ 3 (lane-minor, padding lanes at weight 0), then
the HRR once per shell quartet. Both kernels compute T for a block's primitive
quartets and make one batched Boys call (per block, or per lane group). Their
scratch is grow-only: a buffer keeps its largest size.

**Threads and determinism.** OpenMP hands out **bins** dynamically; bin b
holds bras b, b + n_bins, … in increasing order (round-robin, because bra p
visits every ket q ≥ p). Partial Fock matrices are keyed by bin and summed per
element in bin order, so G is identical at any thread count. n_bins depends on
nb only: 64, fewer once 64 nb² passes 256 MB per output, at least 8. MPI
stripes bras by rank on top; ranks reduce only the nb² Fock matrix.

## What was found

### One component per call was the floor (C1–C3.5, D1)

The (ss|ss) closed form has no recurrence, yet per component it was only
1.66–1.75× faster than HGP, because every call reran the primitive loop. Once
per shell quartet it was 4.9×, and 10–16× at total L = 4. Removing a
`ShellPair` rebuilt per component quartet (C1) gave 1.16–1.26×. The closed
forms through total L ≤ 3 (C2–C3.5) gave 1.22–1.28×, 1.61× and 1.87× on 8 H2O.
The HGP block kernel at total L 4–12 (D1) gave 1.5× on sp, 4.3–4.9× on d and
12.9× on f bases.

**Layout decided whether it won.** An earlier contract-once hoist of the same
idea measured 1.14–1.62× slower. D1's difference is indexing Cartesian
functions by total L instead of a per-axis cube.

### Inside the kernels, the cost is per primitive quartet

The HGP block kernel's VRR runs at ~3 GFLOP/s: a few hundred elements per
primitive quartet (183–305 ket elements, 4.5–7.5 KB of table), 290–650 ns
each in large blocks, 1.1–2.3 µs for a block with one primitive quartet. That
is fixed overhead, not arithmetic and not memory. What paid:

- **Several primitive quartets per pass (2b):** 1.38× on cc-pVTZ, 1.21–1.23×
  on cc-pVDZ and 6-31G\*, 1.05–1.07× on sp. Four lanes beat two and eight.
- **Fewer Boys calls (H1, R2):** one vector call instead of MMAX+1 scalar
  calls (1.10–1.12×), then one batched call per block (1.01–1.11× at one
  thread, 1.03–1.11× at four). Boys is still 14–23 % of sp and cc-pVDZ builds,
  nearly all the core's evaluation: batching removed call overhead, not work.
- **Dropping negligible primitive pairs (P1):** |c_a c_b N_a N_b (π/ζ)^{3/2}
  K_ab| < 1e-24, 1.04–1.15× with G byte-identical.
- **Grow-only scratch:** per-block `vector::resize` was out of line and
  re-zeroed regrown tails (2.4–4.7 % of samples); removing it gave 1.03–1.04×
  where HGP runs and nothing elsewhere.

What did not pay:

- **A cheaper recurrence (D1c, the electron-transfer recurrence):** 1.04–1.06×
  against a predicted 1.1–2.3×, and only after choosing the transfer direction
  per primitive quartet (a fixed direction lost 2.8e-9 on cc-pVTZ). The ket
  phase halved; the bra VRR grew 3.3–3.5×.
- **Batching blocks by class (5b):** 93.9–99.8 % of primitive quartets already
  ran in within-block lanes, so pooling could buy at most ~1.15×, and
  enumerating the blocks cost 2–12 %.
- **Sharing work across shells with shared exponents (D2a, B0–B2, R3).** cc
  general contractions and Pople SP shells share exponents, so merging them
  cuts primitive passes 2.1–3.7× (Pople, across L) or 4.8–5.1× (cc). Two-sided
  family passes gain 1.17–1.78× on the closed forms, but they need the blocks
  visited in family order, which costs +3.5–7.4 % on Pople bases (±1 % on cc).
  Sharing on the ket side only keeps today's order but loses most of the merge
  (cc-pVDZ 5.91× → 2.31× fewer passes) and measured 0.90–1.13×, ~1.07× of a
  cc-pVDZ build at best. Only a traversal chosen by basis would pay
  (≤ ~1.29× on cc-pVDZ), and it was not taken up. Lanes had already taken the
  HGP half (a scalar HGP family pass was 0.48–1.03×).
- **Precomputing the closed-form readout in `prepare` (R1):** bitwise, but
  0.92–1.00× in production. The larger `prepare` inlined into the hot loop
  (`noinline` recovers 2–3 %), and it filled components the loop never reads
  (85 % read on 8 H2O).

### Outside the kernels, the cost was digesting one component at a time

While every integral left its kernel through `value()` and was digested
alone, nothing outside the kernels could get cheaper. Digesting each block
densely (K1) gave 1.11–1.22×; under the symmetry skeleton (K1.4) 1.32×. The
per-component traversal's own overheads went one by one: the quadratic search
in `distinct_eri_orbit` (L2, 1.09–1.20×), ket blocks below the bra (L1,
1.06×), the full orbit built for non-representatives (L3a, 1.70× D2h),
per-quartet orbit allocation (L3b, 1.23×), blocks an operation sends lower
(L3c, 1.07×), per-element replication of every write (S1, 1.33×).

### Threads: the schedule, then the partition (W1, partition)

The dense traversal is a triangle. Contiguous chunks handed the first of four
threads 1 − (3/4)² = 44 % of it, a 2.29× ceiling. Round-robin (W1) gave
1.10–1.97× at four threads, 3.4–3.7× scaling from one thread.

Partials were keyed by thread, so G at one thread was not G at four (CO2 /
6-31G\* direct differed in the DIIS residual digits; all 10 Hartree digits
identical). The four-thread idle that remained (naphthalene 11 %) was the
barrier closing the parallel loop: three threads waited 10–17 %, one 0.6 %,
because under S1 the work per bra is uneven. Keying partials by bin fixed both
at once, at four threads:

| fixture | 16 bins | 32 bins | 64 bins | 128 bins |
|---|---|---|---|---|
| 8 H2O / 6-31G | 0.950× | 0.969× | 0.982–0.990× | 0.972× |
| 1 H2O / cc-pVTZ | 0.985× | 1.010× | 1.021× | 1.037× |
| naphthalene / 6-31G | 0.987× | 1.026× | 1.058–1.064× | 1.075× |

At one thread 16–64 bins are within ±1.5 %; 128 costs 1.4–2.2 %, so 64.

### The full point-group path (F1, F2, F4)

SAO-blocked HF with direct SCF fills a dense n_b⁴ skeleton in its own loop,
persisted when n_b ≤ `threshold` (100). Above that it was rebuilt every
iteration and was slower than no symmetry; it now takes the fused build with
the geometry's D2h-subgroup operations, which suffice because an SAO density
is symmetric under any subgroup (F1: benzene / 6-31G\* 17.7 → 4.08 s,
naphthalene / 6-31G 66.6 → 6.35 s per SCF). Auto resolves to HGP there (F2,
1.07–1.12× per skeleton). `guess sad` aborted these runs because making the
SAD density idempotent is ill-posed when a degenerate set straddles the
cutoff; the guess is now the raw density scaled to the electron count (F4;
`docs/PLANCK_TEACHING_GUIDE.md` §6).

On the fused path symmetry cuts integral values and accumulations 7× on
naphthalene (planar, so the orbit caps at 4 and the plane's mirror shows up as
forbidden integrals), but per-block work falls less, which is why the build
gained less than 7×.

### Declined by requirement

- **Primitive screening (P2):** skipping primitive quartets below ε = 1e-12
  bought 1.18–1.37× but moved G by 3.3–4.8e-10. G must stay accurate to
  1e-10; at ε = 1e-14 (≤ 5e-12) it bought only 1.07–1.15×.
- **Block kernels above total L = 12:** g-shell bases are not routine work.

## Where the time goes now

`sample` self time over 10 s of production builds, one thread (four threads in
brackets where different), before grow-only scratch and the partition:

| phase | 8 H2O 6-31G | 4 H2O 6-31G\* | 4 H2O cc-pVDZ | 1 H2O cc-pVTZ | naphthalene D2h |
|---|---|---|---|---|---|
| closed-form kernel | 39 % | 25 % | 39 % | 13 % | 40 % (35 %) |
| Boys | 22 % | 14 % | 23 % | 7 % | 22 % (20 %) |
| closed-form readout | 13 % | 9 % | 6 % | 2 % | 7 % |
| digestion loop | 20 % | 19 % | 11 % | 21 % | 20 % (17 %) |
| HGP VRR + finish + prepare | 2 % | 29 % | 19 % | 52 % (50 %) | 5 % |
| scratch zero + resize (since removed) | 3.5 % | 4.7 % | 2.8 % | 4.0 % | 2.4 % |
| skeleton weight | — | — | — | — | 3 % |
| idle at four threads (since mostly recovered) | 1 % | 5 % | 3 % | 6 % | 11 % |

## Not done

- **R1b:** the closed-form readout (2–13 %) from a per-AO table built once
  per build, with `value()` kept lazy, which avoids both of R1's costs.
  Bitwise; land only at ≥ 1.03× on 8 H2O, timed on the production loop.
- **C5:** screened kernels in both block kernels, so range-separated hybrids
  leave per-component HGP. Needs `hgp_vrr`'s scaling (T × `boys_scale`,
  prefactor × `prefactor_scale`, W−P/W−Q × `screen.rho / rho`, the straddling
  coupling × `boys_scale`: the `inv_2_delta` gotcha); gate against OS for
  LongRange and ShortRange on a mixed bra/ket fixture.
- **F3:** feed the persisted full-group skeleton from the block kernels; a
  one-time build (0.017–0.054 s small, 1.3–3.4 s near n_b = 100), worth it
  only if `threshold` is raised routinely.
- **Outside this work:** D3 (RI-JK recomputes density-independent pair
  factors, ~61 % of the RI Fock build; `docs/ERI_FOCK_BLAS_SCOPE.md`); K2
  (batched same-class ket blocks, the only lever left on HGP, if a census
  shows per-block fixed costs worth it); C4b (a four-raise closed form, 10–16×
  over per-component HGP but competing with the block kernel at 4–9 % of sp
  builds).

## What invariants matter

### 1. The unit of work is the primitive quartet, not the integral

Below the block kernel, a few hundred elements per primitive quartet is
overhead-bound. A cheaper recurrence (D1c) or a better layout barely moved it;
fewer primitive passes, or several per pass, did.

Design rule:

- Before optimizing a kernel's arithmetic, count its primitive quartets per
  pass. A lever that does not change that count or batch it will not pay.

### 2. The generation/digestion contract caps what batching can buy

While each integral leaves its kernel through `value()` and is digested
alone, batching blocks or sharing work across them changes nothing outside the
kernels.

Design rule:

- Before batching inside the kernels, check that the consumer can take a batch.
  Dense per-block digestion (K1) is what made batching worthwhile.

### 3. The two traversals must never be mixed within a build

Dense and per-component visit different block sets; mixing them double-counts
or drops quartets. Dense is exact only for a symmetric density, which is why
it checks the density first.

Design rule:

- Choose the traversal once per build, and add any new kernel through
  `covers()` so the choice stays a single predicate.

### 4. Determinism needs a fixed partition of the work, not just a fixed sum order

Keying partials by thread made G depend on the thread count, even with a fixed
merge order. Keying them by bin fixes which terms each partial sums and in
what order, whoever runs it, which also frees the schedule to be dynamic.

Design rule:

- Accumulate into partials keyed by a function of the work (and of nb for the
  bin count), never by thread index or completion order. The same rule holds
  in the FCI sigma build and the DFT grid layer.

### 5. Price a change on the production loop

`--k1`'s timed dense arm prepares every block up front and reads every
component unmasked; production prepares lazily inside the masked loop. R1
looked like a net gain there (+12 ms prepare, −16 ms digestion) and was 2–8 %
slower in production. Moving work earlier is not removing it.

Design rule:

- Decide on the production time, best of three or more interleaved, and treat
  a 2–3 % difference at best of three as possible noise.

### 6. A profile share is neither a ceiling nor a promise

The `ShellPair` rebuild was 8.4 % of samples and worth 1.16–1.26×. Boys was
9–32 % and a 1.3–2.3× faster Boys realized 1.03–1.13×. Scratch zeroing was
3.5 % of 8 H2O and removing it bought nothing there. Idle time blamed on
serial functions (S2) was a barrier wait.

Design rule:

- Write the prediction down before measuring, and attribute with a call tree
  (who calls the hot leaf, which thread waits where) before acting on a share.

### 7. Restructured code is not bitwise under g++'s defaults

g++ compiles with `-ffp-contract=fast`, so whether `a + b·c` fuses depends on
the surrounding loop: R2 changed 583 of 67 032 closed-form outputs by
≤ 1.1e-16 with byte-identical T and Boys values.

Design rule:

- Gate a restructure on tolerance plus the suites; to call a kernel change
  bitwise, compile old and new into one test and compare call by call.

### 8. Screening must never reach the Schwarz table

The first primitive-screening experiment gave max|ΔG| = 3e-6 because
`schwarz_table_hgp` builds Q with the same kernel.

Design rule:

- Any approximation added to a kernel must be switched off when that kernel
  builds the Schwarz bounds.

## Validation strategy that should remain in place

At close (2026-10-24, after the partition): smoke 43/45 and extended
135/139, the failures being the four known `rccsdt` routing cases;
`planck-fock-accumulate`, `planck-hgp-engine-smoke` and `planck-boys-vec`
pass.

- `planck-fock-accumulate` (`tests/fock_accumulate.cpp`):
  `check_dense_digest` (dense against per-component in every `FusedTerm`
  mode, RHF/UHF, four bases, a counting kernel that proves the dense path
  ran, NaN-safe `max_output_diff`), the skeleton checks under water's real
  C2v with the operation count asserted, and `check_thread_count_invariance`
  (G bitwise identical at 1/2/4/8 threads on the per-component path, dense
  RHF and dense UHF; fails on all three if the bin count follows the thread
  count).
- `planck-hgp-engine-smoke`: the closed forms against HGP on every total-L
  ≤ 3 layout, and the HGP block kernel against per-component HGP up to
  (ff|ff).
- `planck-boys-vec`: `boys_batch` bitwise against `boys_vec`, and `boys_vec`
  against looped `boys`.
- `planck-eri-class-share`: the 28 G fingerprints; a kernel change that is
  not bitwise must say why.
- `nh3_c3v_fullsym_631gd_sad_direct` and the `water_triplet_*` cases for the
  SAD guess on the full point-group path; the f/g/h spherical gates for
  buffer sizes at high L.

## Step index

Commits, tests and code comments cite these names.

| step | commit | what it did | outcome |
|---|---|---|---|
| C0 | Boys v1.1.0 | libm calls in one branch of the Boys core | 1.37× / 1.13× |
| C1 | `c40a016c` | stop rebuilding a `ShellPair` per component quartet | 1.16–1.26× |
| C2–C3.5 | `3c028544`, `4bbb2724`, `f0f7ceae` | closed forms through total L ≤ 3 | 1.22–1.87× on 8 H2O |
| C4a / C4b | — | four-raise closed form | measured 10–16×; not wired |
| D0 | — | shared-exponent census | 2.1–5.1× fewer passes possible |
| D1 | `52698142` | HGP block kernel, total L 4–12 | 1.5–12.9× |
| D1c | — | electron-transfer recurrence | declined, 1.04–1.06× |
| D2a, B0, B1 | — | family probes and family-order loop (B1 a git stash) | not landed |
| B2.1 | `66669cd5`, deleted `1e047aa2` | family-order traversal, off | a cost; deleted |
| B2.2 / B2.3 / R3 | — | closed-form / HGP / ket-only family sharing | parked / dropped / declined |
| H1 | `46dadc6a` | one vector Boys call; VRR zero-fill | 1.10–1.12× × 1.04–1.06× |
| fix | `fd51d29d` | H1's Boys buffer overflow on (ff\|ff) | correctness |
| L1, L2, L3a–c | `de72ca2c`, `659883ab`, `f88991fe`, `a1f99d7a`, `15c532e4` | per-component traversal overheads | 1.06–1.70× each |
| P1 | `15c532e4` | drop negligible primitive pairs | 1.04–1.15×, G byte-identical |
| P2 | — | primitive-quartet screening | declined (accuracy) |
| S1 | `7ec71009` | skeleton accumulation under symmetry | 1.33× (D2h) |
| S2 | — | thread the serial symmetric phases | refuted (idle was imbalance) |
| T1 | `92838e03` | kernel scratch out of `thread_local` | neutral |
| 2b | `bbe705fc` | four VRR lanes per pass | 1.05–1.38× |
| 5b | — | class-batched scheduler | declined after the census |
| K1, K1.4 | `5fd08496`, `f2849647` | dense digestion; dense under S1 | 1.11–1.22×; 1.32× |
| F1, F2, F4 | `92838e03`, `17cf5adf`, `d0224110`/`0c8a9fc7` | full point-group path | 4.3–10.5× per SCF; 1.07–1.12×; correctness |
| W1 | `ffc44caf` | round-robin schedule | 1.10–1.97× at four threads |
| core v1.2.0, R2 | `aa44c44` (core), `102f1d01` | batched Boys entry and callers | 1.01–1.11× |
| loader | `13c60826` | core v1.2.0 required at load | — |
| R1 / R1b | — | closed-form readout in `prepare` / from a table | declined / not done |
| scratch | `0e3e074d` | grow-only kernel scratch | 1.03–1.04× where HGP runs |
| partition | `ea353a83` | partials keyed by bin, dynamic bins | G thread-count invariant; 1.06× at four threads (D2h) |
| C5, F3 | — | screened kernels; skeleton from block kernels | not done |

Measuring: `planck-eri-class-share` with no arguments prints per-class times
and the fingerprints; `--k1 nwater basis` the headline production time;
`--profile N nwater basis` builds for `sample`; `--family nwater basis
[within] [bra]` and `--census nwater basis` the family and lane probes
(`nwater` 0 is naphthalene). Run one binary at a time, build A/B baselines as
whole targets (a binary mixing old and new objects of an inline class crashed
in `--k1`), and write shell loops in bash, since zsh does not word-split an
unquoted variable.
