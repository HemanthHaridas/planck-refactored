# Closed-Form Low-Angular-Momentum ERI Kernels — Scope

Status: **in flight.** Landed: the class-share probe (C0), the Boys-core fix
it found (Boys library v1.1.0), ShellPair reuse in the fused loop (C1),
closed-form kernels for total L ≤ 1 (C2), a profile-driven hotspot round on
the HGP kernel (H1), and closed forms for total L ≤ 2 (C3) and L = 3 (C3.5).
L2 (closed-form orbit dedup) landed too. Open: the rest of the fused-loop
overhead (L1, L3, L4, scoped below), the C4 decision and C5 (screened
kernels). This file stays scope-shaped until C3–C5
land or are abandoned; then rewrite it as an answer.

**Question:** Would specialized closed-form kernels for low-angular-momentum
shell quartets ((ss|ss), (ps|ss), …) speed up the direct Fock build, and by
how much?

## Short answer

Yes, but only at **shell** granularity, and how much depends on the basis.
A closed form that returns one Cartesian component at a time is only about
1.7× faster than HGP on its class. One that returns every component of a
shell quartet from a single primitive pass is about 5× faster. C2, C3 and
C3.5 shipped the second kind through total L ≤ 3.

Each step is a ratio over the build *with all previous steps*, so the
columns multiply along a row. The last column is the product.

| fixture | C1 | C2 | H1 | C3 | C3.5 | cumulative |
|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | 1.22–1.28× | 1.16× | 1.61× | **1.87×** | **~5.4×** |
| 1 H2O / cc-pVDZ | 1.07× | 1.10× | — | 1.23× | 1.43× | — |
| 4 H2O / 6-31G\* | 1.16× | 1.02–1.05× | 1.16× | 1.13× | 1.26× | ~2.0× |

(H1 was not measured on cc-pVDZ, so that row has no cumulative figure.
In absolute terms 8 H2O / 6-31G went from 1.50 s before C1 to 0.28 s.)

(L2 then took 8 H2O / 6-31G from 0.278 s to 0.232 s, so ~6.5× since C1.)

Pure sp bases gain the most, because low-L quartets are most of their
build. What remains above L = 2 needs a compile-time-specialized kernel
family (C4), which is a different project. **Against the ~200× libcint gap,
this is a 2–3× step, not a fix.**

## Where the build time goes

Apple M3, `build-full` (g++-15 `-O3`), 1 thread, measured by
`planck-eri-class-share` (`tests/eri_class_share_benchmark.cpp`, not a
ctest). The probe drives the production `Integrals::fused_fock_build`, times
the HGP per-quartet kernel, and buckets it by (L_AB, L_CD). It prints a
bitwise FNV-1a fingerprint of G, and also the closed-form build time with
max|ΔG| against HGP. `--profile N` repeats the plain build of
4 H2O / 6-31G\* for `sample`.

Shares of the **whole build** after C1 (projected speedup at ~5× on the
covered classes, then the ceiling):

| coverage | 4 H2O / 6-31G\* | 1 H2O / cc-pVDZ | 8 H2O / 6-31G |
|---|---|---|---|
| L_tot ≤ 1 | 6.1 % → 1.05× (1.07×) | 11.5 % → 1.10× (1.13×) | 23.0 % → 1.23× (1.30×) |
| L_tot ≤ 2 | 18.2 % → 1.17× (1.22×) | 30.0 % → 1.32× (1.43×) | 54.5 % → 1.77× (2.20×) |
| L_tot ≤ 4 | 58.3 % → 1.88× (2.40×) | 75.0 % → 2.50× (4.0×) | 87.9 % → 3.4× (8.3×) |

After C1 the HGP kernel is about 89 % of the build on every fixture. The
rest is the canonical filter, the Schwarz tests and accumulation.

**Profile after C2 and H1** (`sample` of `--profile 30`, self time as a
share of the build; parsed with a depth-stack call-tree parser, since the
naive regex misattributes inclusive counts):

| frame | 8 H2O / 6-31G | 4 H2O / 6-31G\* |
|---|---|---|
| `hgp_vrr` (per-primitive VRR, L ≥ 2 classes) | 51.0 % | 57.9 % |
| Boys (`boys_vec`, blob + dispatcher) | 17.8 % | 13.3 % |
| fused loop + inlined closed-form kernel | 13.7 % | 9.5 % |
| `hgp_contract_a0c0` accumulation | 11.2 % | 10.3 % |
| HRR + scratch resize/memset | ~3.4 % | ~6.9 % |

`hgp_vrr` dominated because HGP reruns the whole primitive VRR for every
Cartesian component. That is what C3 removed for L = 2.

**After C3.** Classes above L = 2 still run through HGP, so their absolute
time is unchanged and only the build shrank. Re-based on the post-C3 build:

| | 8 H2O / 6-31G (0.516 s) | 4 H2O / 6-31G\* (0.726 s) | 1 H2O / cc-pVDZ (0.043 s) |
|---|---|---|---|
| HGP, total L = 3 | **47 %** (all (1,2)) | 22 % | 31 % |
| HGP, total L = 4 | 15 % | 26 % | 28 % |
| HGP, total L ≥ 5 | — | ~36 % | — |
| loop + closed-form kernels + Boys | ~38 % | ~16 % | ~41 % |

(The cc-pVDZ column was first written as 50 % / 10 %, which put the C3.5
projection at ~1.7×. Recomputed from the class table it is 31 % / 28 %,
and the projection is ~1.34×.)

Profile self times: `hgp_vrr` 45 % / 60 % (8 H2O / 4 H2O), the fused loop
with the closed-form kernels inlined 25 % / 12 %, Boys ~16 % / ~12 %. The
single largest target is now total L = 3, (ps|pp)-type on sp bases.

**After C3.5** (kernel marked `noinline` in a profiling-only build, so its
time is separable; self time as a share of the build):

| frame | 8 H2O / 6-31G (0.294 s) | 4 H2O / 6-31G\* (0.601 s) |
|---|---|---|
| fused loop: traversal, filters, Schwarz, orbit, accumulate | **35.8 %** | 13.3 % |
| `hgp_vrr` (remaining L ≥ 4 classes) | 23.0 % | **58.9 %** |
| closed-form kernel `low_l_shell_eri` | 18.1 % | 3.5 % |
| Boys (`boys_vec` + `boys`) | ~19 % | ~10 % |
| HGP accumulate + HRR + scratch | ~3 % | ~11 % |

The two bases now want different work. On sp bases the loop itself is the
largest item, and no kernel change touches it (L1–L4). On d bases HGP at
L ≥ 4 still dominates (C4).

**Measuring on this machine.** Timings are bimodal under load (P/E-core
scheduling): one binary alternated between 1.0 s and 1.22 s. So compare
best-of runs of interleaved binaries, or the same binary with a flag off and
on. A single run is never evidence.

## Findings that shaped the plan

**The floor is the primitive loop, not the recurrence.** The (ss|ss) closed
form `Σ c K_p K_q · 2√(ρ/π) F0(ρ|PQ|²)` has no recurrence left to remove,
yet a per-component version measured only 1.66–1.75× faster than HGP. Each
primitive quartet costs one Boys call and one prefactor product. An O 1s in
6-31G\* has 6 primitives, so one (1s 1s|1s 1s) is 1296 primitive quartets.
The per-component interface repeats that loop 3× for (ps|ss), 9× for
(pp|ss) and 81× for (pp|pp). The shell-level prototype ran it once and
measured 4.9×.

**This is not the hoist that already failed.** `docs/ERI_PERFORMANCE_SCOPE.md`
records a contract-once-per-shell-quartet hoist as 1.14–1.62× *slower*: it
copied an (a0|c0) snapshot, gathered sub-boxes and ran HRR per component, and
that overhead beat the savings on sp bases. A closed-form kernel has no
buffer, no gather and no HRR, and writes components from registers. The risk
carried forward to C4 is that a generic template family can drift back into
that shape.

**One Boys-core branch was 17.7 % of the build (C0).** A `sample` profile of
4 H2O / 6-31G\* attributed 84.1 % of the build to the HGP kernel, 31.0 % to
Boys, and 17.7 % to libm calls in a single branch of the out-of-tree Boys
core that water chains hit constantly. Fixed there and shipped in Boys
library v1.1.0. Through the real blob: 4 H2O / 6-31G\* ~1.13×,
8 H2O / 6-31G ~1.37×. Linked directly it measured 1.21× / 1.64×; the
difference is plausibly the loader's function-pointer dispatch, which cannot
inline. `water_rhf_hgp_sto3g` (11 → 10 SCF iterations) and
`ethylene_rhf_ccpvdz_auto` (16 → 15) moved at the convergence noise floor,
with energies unchanged, and were re-pinned.

**A profile share underestimates what removing the work is worth.** The
`ShellPair` rebuild was 8.4 % of the profile. Removing it bought 1.16–1.26×,
not the 1.09× the share implied, because its allocator and cache traffic were
charged to other frames.

## Landed

### C0 — class-share probe (2026-10-17)

`planck-eri-class-share`, described above. Its in-tree class shares
reproduced the earlier scratch bench to within 1–2 points.

### C1 — reuse prebuilt ShellPairs (2026-10-18, `c40a016c`)

`fused_fock_build` rebuilt a `ShellPair` (`pow`, `exp`, heap per primitive
pair) for every bra component pair and every surviving component quartet. Yet
every caller already passes the same pairs, built by the same constructor,
in `shell_pairs`. It now looks them up through an `nb × nb` pointer table
keyed on `_index`, constructing a pair only on a miss, which no caller
reaches. That helps every engine.

- **Bitwise:** the G fingerprint is identical on all four fixtures.
- **Speed:** 4 H2O / 6-31G\* 1.17 → 1.01 s (1.16×), 8 H2O / 6-31G
  1.50 → 1.19 s (1.26×), cc-pVDZ 1.07×.
- **Found on the way:** `planck-hgp-engine-smoke` and `planck-compute-2e`
  segfaulted under ctest because they never called `require_boys_library()`.
  Fixed.

### C2 — closed forms for total L ≤ 1 (2026-10-18, `3c028544`)

`low_l_shell_eri` (`src/integrals/low_l_eri.h`) does one primitive pass per
shell quartet. Per primitive quartet, with θ_m = c K_AB K_CD 2√(ρ/π) F_m(T):

    (ss|ss)    = θ0
    (p_i s|ss) = PA_i θ0 + WP_i θ1      (B: PB_i, same WP)
    (ss|p_i s) = QC_i θ0 + WQ_i θ1      (D: QD_i, same WQ)

It is fed the first-component pair of each shell; every component of an s or
p shell shares the primitive data and the norm. `fused_fock_build` calls it
once per block, on the first component quartet that survives the filters,
behind an opt-in `closed_form_low_l` flag. Only the Auto engine (the default)
sets the flag. Explicit `engine os/hgp/rys` stay pure, so the cross-engine
comparators still compare what they say. The canonical filter, Schwarz,
symmetry orbit and accumulation all run per component on the output,
unchanged. Coulomb only.

- **Speed (flag off vs on, same binary):** 8 H2O / 6-31G 1.22–1.28×
  (predicted 1.23×), cc-pVDZ 1.10–1.11×, 4 H2O / 6-31G\* 1.02–1.05×.
  max|ΔG| against HGP ≤ 4e-15.
- **Gate:** `planck-hgp-engine-smoke` sweeps all 3136 total-L ≤ 1 component
  quartets of water/6-31G and 6-31G\* against HGP. Max relative difference
  is 3.4e-16, with every p position (none, A, B, C, D) required.
  Mutation-verified: a flipped W−P sign fails at 0.26, a swapped y/z readout
  at 0.77.
- Core 79/79. Output is identical at 1 and 4 threads for full-symmetry
  RHF/UHF direct and ethylene/cc-pVDZ Auto.

**Found, not caused:** `water_rhf_spherical_direct_631gd` differs between 1
and 4 threads in the last digits of its converged residuals (1.621e-12 vs
1.619e-12), with the energy identical to all 10 digits. The binary from
before C2 does the same, so the cause is on the spherical direct path,
outside the fused loop. Not chased.

### H1 — hotspot round on the HGP kernel (2026-10-18)

Profiled the production-shaped build (closed forms on) and took the three
bitwise-safe items it exposed. The G fingerprint is identical before and
after each one.

- **One `boys_vec` call instead of MMAX+1 `boys` calls** in `hgp_vrr` and
  in both RI seeds (`src/post_hf/ri/ri_eri.cpp`). OS had already made this
  change; HGP and RI never got it. Each scalar call goes through the
  loader's indirect call, and `Lookup::boys` alone showed 6.8 % *self*.
  `boys_vec` is bitwise-identical to the loop by its contract. Gain: 1.12×
  on 8 H2O / 6-31G, 1.10× on 4 H2O / 6-31G\*.
- **Accumulate the m=0 slice straight from the VRR buffer.** The old path
  copied the slice into `hrr_data` and made a second pass to add it; the
  helper `hgp_eri_primitive_vrr_only` is gone. Same arithmetic.
- **No zero-fill of the VRR buffer.** MMAX is the sum of the per-axis
  maxima, so every cell's m range is written before any read. That is the
  policy OS already follows.

The last two together: 1.04× on 8 H2O / 6-31G, 1.06× on 4 H2O / 6-31G\*.
H1 total: 1.16× on both.

**A stale-binary trap, again.** After an A/B that restored `hgp.cpp`
within the same second the binary was linked, `make` skipped the rebuild,
and the next profile showed no change. The A/B numbers came from copied
binaries and were valid. The profile had to be retaken after a forced
rebuild. Check a symbol (`nm`) or a fingerprint, not a timestamp.

### C3 — closed forms for total L = 2 (2026-10-18)

`low_l_shell_eri` now takes 0, 1 or 2 raised centers and fills a 3×3
output. Per primitive quartet (θ_m from `boys_vec`, m = 0..2):

    R_i(m) = d1_i θm + w1_i θ(m+1)
    V_ij   = d2_j R_i(0) + w2_j R_i(1) + δ_ij κ

d is PA, PB, QC or QD for the raised center; w is W−P on the bra and W−Q on
the ket. κ = (θ0 − (ρ/ζ)θ1)/2ζ with both raises on the bra, the same with
η on the ket, and θ1/2(ζ+η) when they straddle. This covers the ten
layouts: d on one center (4), two p on one side (2), one p on each side (4).

- **Interface.** The flag is now `int closed_form_max_l` (0 off, 1 = C2,
  2 = C3), and Auto passes 2. The loop reads the axes from each
  component's `_cartesian`, walking centers in order and expanding each unit
  of x, y, z, so the readout is one rule for every layout.
- **Norms.** The kernel's pair data carries the first component's norm
  (d xx: 1/√3). Each output is scaled by a per-AO
  `norm_ratio = _component_norm / first component's norm`, which is exactly
  1.0 for s and p.
- **Speed (same binary, max_l 1 vs 2), against predictions:**

  | fixture | predicted | measured |
  |---|---|---|
  | 8 H2O / 6-31G | ~1.42× | **1.61×** (0.830 → 0.516 s) |
  | 1 H2O / cc-pVDZ | ~1.20× | 1.23× |
  | 4 H2O / 6-31G\* | ~1.12× | 1.13× |

  On 8 H2O / 6-31G the build is now 2.0× faster than pure HGP in the same
  loop. The stop rule passes. max|ΔG| against HGP is ≤ 3.8e-15 on every
  fixture.
- **Gate.** The `planck-hgp-engine-smoke` sweep now covers total L ≤ 2:
  6076 quartets on water/6-31G and 8428 on 6-31G\*. Max relative difference
  against HGP is 3.4e-16, and all 15 layouts are required (the four d layouts
  on 6-31G\* only). Mutation-verified three ways, each red: dropping κ
  (rel 1.0), using the bra κ for a straddling pair (1.12), and dropping the
  norm ratio (0.16, on 6-31G\* only, as predicted).
- **A bug the fingerprint caught.** The first C3 build tested
  `sum_L <= low_max_l` without requiring `low_max_l > 0`. So (ss|ss) blocks
  (0 ≤ 0) took the closed form with the flag *off*, which would also have
  quietly changed explicit `engine os/hgp/rys`. The flag-off G fingerprint
  moved on every fixture, and that is how it was found. Fixed with a
  `low_max_l > 0` guard; the fingerprints are back to the originals.
- **Not bitwise against C2 for L ≤ 1.** The L ≤ 1 results move by about an
  ulp (for example max|ΔG| on 1 H2O / cc-pVDZ went from 4.0e-15 to
  3.1e-15), because the generalized kernel's loops compile to different
  instruction sequences. That is within every gate, and the explicit
  engines are bitwise unchanged.
- **Suites.** Core 79/79, `planck-compute-2e` OK. Smoke 43/45; the two
  failures are the known rccsdt routing cases. Output is identical at 1 and
  4 threads for full-symmetry RHF/UHF direct (including NH3/6-31G\*, which
  has d shells) and ethylene/cc-pVDZ Auto.

### C3.5 — closed forms for total L = 3 (2026-10-18)

The same kernel with a third raise, applied in center order (c1 ≤ c2 ≤ c3,
axes i, j, k). With θ0..θ3 from one `boys_vec` call:

    A_i(m)   = d1_i θm + w1_i θ(m+1)                                  m = 0..2
    A2_j(m)  = d2_j θm + w2_j θ(m+1)        (second raise alone)      m = 0..1
    B_ij(m)  = d2_j A_i(m) + w2_j A_i(m+1) + [i=j] cpl(1,2; θ)        m = 0..1
    C_ijk    = d3_k B_ij(0) + w3_k B_ij(1) + [k=i] cpl(1,3; A2_j) + [k=j] cpl(2,3; A_i)

cpl(y,x; X) is the Obara–Saika coupling of the new raise x to the earlier
raise y it lowers: (X(m) − (ρ/ζ)X(m+1))/2ζ when both are on the bra, the
same with η on the ket, X(m+1)/2(ζ+η) when they straddle. It is C3's κ,
one level up. The 27 outputs stay in registers. The 0–2 raise paths are
unchanged, and Auto now passes `closed_form_max_l = 3`.

- **Speed (same binary, max_l 2 vs 3):**

  | fixture | predicted | measured |
  |---|---|---|
  | 8 H2O / 6-31G | ~1.6× | **1.87×** (0.517 → 0.277 s) |
  | 1 H2O / cc-pVDZ | ~1.34× | 1.43× |
  | 4 H2O / 6-31G\* | ~1.2× | 1.26× |

  On 8 H2O / 6-31G the build is now 3.7× faster than pure HGP in the same
  loop. max|ΔG| against HGP is ≤ 4.4e-15, and the flag-off fingerprints are
  unchanged. **The raise chain still beats its ~5× model at three raises**,
  which answers C4's question in its favour.
- **Gate.** The sweep covers total L ≤ 3. Water/6-31G sweeps 7840
  quartets, 6-31G\* sweeps 15736, and water/cc-pVDZ (new) sweeps 43456.
  Max relative difference is 6.1e-16. Every reachable raise layout is
  required: a sorted center tuple of length 0–3, with a center raised twice
  only on a d basis and never three times (no f). 15 layouts on the sp
  fixtures, 31 on cc-pVDZ.
- **Why cc-pVDZ joined the gate.** The first L ≤ 3 sweep failed on 6-31G\*
  with (AAB) and (CCD) never exercised. Those layouts need a d on A and a p
  on B with the d's AO index first. In water/6-31G\* the only d is O's, and
  nothing with p follows it (H has no p). cc-pVDZ's H p shells come after
  the O d.
- **Mutation-verified:** dropping the `[k=i]` coupling fails at 0.0096 /
  0.079 (6-31G / 6-31G\*); dropping the `[k=j]` coupling fails at 0.24.
- **Suites.** Core 79/79, `planck-compute-2e` OK. Smoke 43/45; the two
  failures are the known rccsdt routing cases. Output is identical at 1 and
  4 threads for full-symmetry RHF/UHF direct, NH3/6-31G\* and
  ethylene/cc-pVDZ Auto.

## Open

### L — fused-loop overhead

A reading of `fused_fock_build` and `fock_accumulate.h` against the profile
found four places where the loop does avoidable work per component quartet.
All four keep the result **bitwise-identical**: the same quartets are
accumulated, in the same order, with the same values. The flag-off and
flag-on G fingerprints are the gate for every step.

**L0 — split the 35.8 % first.** `fused_fock_build`'s self time is one
number covering traversal, filters, orbit enumeration and the `G` writes.
Two cheap measurements split it before anything is changed:
- *Counters*, in a probe-only build: blocks visited, blocks containing at
  least one canonical quartet, component iterations, accepted quartets. That
  gives the wasted-traversal fraction exactly, not from a timer.
- *Attribution*: a profiling build with `distinct_eri_orbit` and
  `accumulate_term` marked `noinline`, the same trick that separated the
  kernel above.

Each step below gets its prediction from these numbers.

*L0 results (2026-10-18).* Counters, per build:

| | 8 H2O / 6-31G | 4 H2O / 6-31G\* | 1 H2O / cc-pVDZ |
|---|---|---|---|
| ket-loop iterations | 6.91 M | 672 k | 6.1 k |
| … surviving block Schwarz | 987 k (14 %) | 293 k | 6.1 k |
| … of which L1-skippable | **486 k (49 %)** | 142 k (49 %) | 2.7 k (45 %) |
| component iterations | 4.36 M | 3.86 M | 114 k |
| … passing the canonical filter | 2.13 M (49 %) | 1.84 M (48 %) | 53 k (46 %) |
| accepted quartets | 2.00 M | 1.64 M | 53 k |
| … via closed form | 95 % | 46 % | 41 % |

The consistency check holds: surviving blocks minus skippable (500,935)
≥ blocks with any canonical quartet (495,844). The difference is
same-group blocks where every component fails. (A first version of the
counters overcounted because a block counter landed inside the `ca` loop;
the inconsistency exposed it.)

Attribution, with `distinct_eri_orbit`, `accumulate_term` and the kernel
marked `noinline` (self time as a share of the build):

| frame | 8 H2O / 6-31G | 4 H2O / 6-31G\* |
|---|---|---|
| `hgp_vrr` | 23.2 % | 59.3 % |
| closed-form kernel | 17.8 % | 3.2 % |
| **`distinct_eri_orbit`** | **17.3 %** | 7.9 % |
| fused loop traversal and filters | 13.9 % | 4.1 % |
| Boys | ~19 % | ~10 % |
| `accumulate_term` (the `G` writes) | 4.7 % | 1.8 % |

(The traversal figure includes the counters, so it is slightly high.)

What it means for each step:
- **L2 first.** The orbit dedup costs about 3.7× the `G` writes it feeds.
  Predicted ~1.15–1.2× on 8 H2O / 6-31G, ~1.07× on 6-31G\*.
- **L1 next.** About half the traversal is skippable blocks. Predicted
  ~1.05–1.07× on 8 H2O / 6-31G, ~1.02× on 6-31G\*.
- **The `G` writes are cheap (4.7 %).** That settles the
  local-accumulation-buffer idea below: not worth its non-bitwise cost.
- **L4 has no visible cost of its own.** Deprioritized.
- **L3: measured separately, below.** The water chains run without
  symmetry.

**L1 — skip blocks that cannot hold a canonical quartet.** The ket loop
runs over all `ngp` shell-group pairs, and canonical order (k ≥ i, then
l ≥ j) is enforced only per component. Groups are AO-contiguous and
`group_pairs` is lexicographic, so every ket block with
`group_pairs[ket].a < group_pairs[bra].a` has k < i for *every* component
and contributes nothing. Yet each one still runs the block Schwarz test,
the low-L setup, the pair lookups and four nested component loops. Start
the ket loop at the first block whose `.a` equals `bra.a` (one precomputed
offset per group). Those blocks only ever fed nothing, so the result is
bitwise-identical, and the per-thread work split (static over bra) is
unchanged. Roughly half of all block visits disappear; what fraction of the
loop's time they were is L0's question.

**L2 — closed-form orbit dedup.** `distinct_eri_orbit` builds all 8 tuples
of `(ij|kl)`'s permutational orbit and compares each against every earlier
keeper. That is up to 28 four-index comparisons per accepted quartet, and
twice per quartet in `RawCoulombExchange`. Which tuples are duplicates
follows from three equalities. With `X = {k,l} == {i,j}` as multisets,
tuple *n* in the fixed order (ij|kl), (ji|kl), (ij|lk), (ji|lk), (kl|ij),
(lk|ij), (kl|ji), (lk|ji) is kept iff:

| n | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| keep iff | always | i≠j | k≠l | i≠j ∧ k≠l | ¬X | ¬X ∧ k≠l | ¬X ∧ i≠j | ¬X ∧ i≠j ∧ k≠l |

This holds for any (i,j,k,l), not only canonical ones; each entry was
derived by checking tuple n against every earlier tuple. It keeps the
same tuples in the same order as the current first-occurrence dedup, so
the result is bitwise-identical. Gate: an exhaustive comparison against the
current enumeration over all index tuples in 0..4 (625 cases, including
non-canonical), added to `tests/fock_accumulate.cpp`.

*L2 results (2026-10-18).* Landed as specified: `distinct_eri_orbit` emits
the kept tuples straight from the three equalities, and the comment
records the rule.
- **Gate.** `planck-fock-accumulate` keeps the old pairwise search
  verbatim as an oracle and compares on all 1296 tuples over 0..5, order
  included. Mutation-verified: weakening the fourth tuple's condition
  (`bra && ket` → `bra || ket`) fails at (0,0,0,1). All the existing
  accumulation checks still pass. The flag-off and flag-on G fingerprints
  are identical on all five probe fixtures, including naphthalene with
  `sym_ops`, so the change is bitwise.
- **Speed** (best of three interleaved runs, production configuration):

  | fixture | before | L2 | gain | predicted |
  |---|---|---|---|---|
  | 8 H2O / 6-31G | 0.278 s | 0.232 s | **1.20×** | 1.15–1.2× |
  | 4 H2O / 6-31G\* | 0.584 s | 0.535 s | 1.09× | ~1.07× |
  | naphthalene / 6-31G, D2h | 1.702 s | 1.532 s | 1.11× | — |

  Both predictions held, which is what L0's attribution was for.
- **Suites.** Core 79/79. Smoke 43/45; the two failures are the known
  rccsdt routing cases.

**L3 — the symmetry path.** With integral symmetry on, the fused loop
receives `sym_ops`: the D2h-subgroup signed AO permutations from
`update_integral_symmetry`, at most 8. For every canonical component
quartet it calls `build_quartet_orbit`, which allocates a
`std::vector<QuartetOrbitElem>`, applies every operation, deduplicates,
and sorts. The ERI is then computed only if the quartet is the orbit's
representative (the lexicographic minimum); otherwise the orbit is
thrown away.

*Measured (2026-10-18)* on a new probe fixture: axis-aligned naphthalene /
6-31G (nb = 106, D2h by construction, all 8 operations found).
`Fixture{0, "6-31g"}` in the probe; `--profile N 0 6-31g` profiles it.

- **With `sym_ops`, the build is only 1.90× faster than the same build
  without them** (1.739 vs 3.308 s), although symmetry cuts the evaluated
  ERIs roughly 8×. The bookkeeping eats most of the saving.
- **Only about a quarter of the symmetric build computes integrals.**
  Inclusive shares, with `build_quartet_orbit`, `distinct_eri_orbit` and
  `accumulate_term` marked `noinline`:

  | | share of the D2h build |
  |---|---|
  | `build_quartet_orbit` (including its allocation and sort) | **35.7 %** |
  | heap: `free` 13.8 % (5.3 % of it `mach_absolute_time` inside macOS's zone allocator) + `operator new` 11.5 % | **~25–30 %** |
  | `accumulate_term`, including `distinct_eri_orbit` 13.2 % | 16.2 % |
  | ERI work: HGP 13.2 % + closed form 11.7 % | ~25 % |

Two fixes follow, both bitwise:

- **L3a — reject non-representatives early.** A quartet is not its orbit's
  representative as soon as one operation maps it, after
  canonicalization, to a lexicographically smaller quartet. A pre-check
  loop can return at the first such image, with no allocation, dedup or
  sort. Most canonical quartets are non-representatives (about 7 in 8
  under D2h), and they are skipped today too, so the set of accumulated
  quartets is unchanged. A quartet that is symmetry-forbidden *and* a
  non-representative is skipped either way. A forbidden representative is
  still caught by the full orbit build.
- **L3b — no heap for representatives.** The orbit has at most
  `sym_ops.size()` ≤ 8 elements. Build it in a fixed-capacity stack array
  in the same order, then sort it the same way, so representatives see an
  identical orbit. Keep the vector-returning `build_quartet_orbit` for
  any other caller.

Predicted: the heap share (~25–30 %) goes away entirely, and most of
`build_quartet_orbit`'s remaining ~24 % goes with the non-representatives,
for roughly **1.6–2×** on the symmetric build. L2 then takes
`distinct_eri_orbit`'s 13.2 % here as well. Gate: the naphthalene G
fingerprint (sym on) is unchanged before and after, plus the existing
symmetry regression cases.

**L4 — precompute the low-L readout per block.** Today each component
recomputes its axis slot (nested loops over the Cartesian exponents) and
multiplies four norm ratios. Both depend only on the component indices
within the block, so they can be a small table per block. This is probably
the smallest of the four; do it only if L0 attributes visible time to it.

**Order:** L0, the L3 measurement and L2 are done. Next L1 (helps every
build), then L3a/L3b (symmetric builds, which they could speed up by more
than any other single step). L4 only if a later profile shows it. A step whose measured gain is noise
still lands if it is bitwise and simpler. L2 qualifies on that ground
alone, since it replaces a quadratic search with a table.

**Out of scope here:** restructuring the accumulation itself, for example
accumulating each block into a small local buffer and flushing it to `G`.
That changes summation order (so it is not bitwise) and interacts with the
threading determinism rules. Revisit only if L0 shows the `G` writes
dominate after L1–L3.

### C4 — decision point: a specialized family through total L ≤ 4

After C3.5 the HGP remainder is total L = 4 (all of it on sp bases, since
6-31G has nothing higher) and L ≥ 5 on d bases. Four raises would mean 81
outputs and one more intermediate level. That is still the same chain,
generated rather than hand-written. A post-C3.5 profile should come
first. It needs `template <int LA, int LB, int LC, int
LD>` shell kernels: an OS/MD-style VRR with constexpr-unrolled loops and no
runtime scratch. It **must not** reintroduce the hoist's buffer or gather.
(pp|pp) and (ds|ps) will need intermediates; size them on the stack per
instantiation. C3.5 settled the generalization question: three raises still beat the ~5×
model. So L = 4 can be the same chain (code-generated, since four nested
levels by hand invite exactly the coupling bugs the mutation tests
catch), not a separate template engine. The post-C3.5 profile confirms the
split: on sp bases the loop is the larger item (L series first); on d
bases HGP at L ≥ 4 is (C4).

### C5 — screened kernels

The closed forms take the same screened scaling as `hgp_vrr`: T ×
`boys_scale`, prefactor × `prefactor_scale`, and W−P/W−Q × `screen.rho / rho`.
C3's straddling κ is exactly the `inv_2_delta` term behind the HGP
screened gotcha (`vault/Gotchas/HGP Screened inv_2_delta`), so it also takes
× `boys_scale`. Until a screened sweep against OS passes on a mixed
bra/ket-AM fixture, screened kernels stay on HGP.

## Out of scope

- **Gradients.** Derivative ERIs are a separate kernel family (Open Work
  Phase D).
- **Primitive screening.** Skipping primitive quartets with negligible
  `K_ab K_cd` is the other lever on the primitive-loop floor, and it is
  independent of closed forms. Deep contractions are the target: the
  cc-pVDZ (ss|ss) class ran about 5 µs per quartet before C2.
- **Spherical.** The direct spherical path back-projects to Cartesian, so it
  goes through the same fused loop and needs no separate work.
