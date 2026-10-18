# Closed-Form Low-Angular-Momentum ERI Kernels — Scope

Status: **in flight.** Landed: the class-share probe (C0), the Boys-core fix
it found (Boys library v1.1.0), ShellPair reuse in the fused loop (C1),
closed-form kernels for total L ≤ 1 (C2), a profile-driven hotspot round on
the HGP kernel (H1), and closed forms for total L ≤ 2 (C3). Open: the C4
decision and C5 (screened kernels). This file stays scope-shaped until C3–C5
land or are abandoned; then rewrite it as an answer.

**Question:** Would specialized closed-form kernels for low-angular-momentum
shell quartets ((ss|ss), (ps|ss), …) speed up the direct Fock build, and by
how much?

## Short answer

Yes, but only at **shell** granularity, and how much depends on the basis.
A closed form that returns one Cartesian component at a time is only about
1.7× faster than HGP on its class. One that returns every component of a
shell quartet from a single primitive pass is about 5× faster. C2 and C3
shipped the second kind through total L ≤ 2.

Each step is a ratio over the build *with all previous steps*, so the
columns multiply along a row. The last column is the product.

| fixture | C1 | C2 | H1 | C3 | cumulative |
|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | 1.22–1.28× | 1.16× | **1.61×** | **~2.9×** |
| 1 H2O / cc-pVDZ | 1.07× | 1.10× | — | 1.23× | — |
| 4 H2O / 6-31G\* | 1.16× | 1.02–1.05× | 1.16× | 1.13× | ~1.55× |

(H1 was not measured on cc-pVDZ, so that row has no cumulative figure.
In absolute terms 8 H2O / 6-31G went from 1.50 s before C1 to about
0.52 s.)

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
Cartesian component. That is what C3 removed for L = 2. A profile after C3
has not been taken yet; it is the input to the C4 decision.

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

## Open

### C4 — decision point: a specialized family through total L ≤ 4

What is left above L = 2 is mostly `hgp_vrr` on L = 3–4 classes. Their
share of the post-C3 build has not been measured; take a profile before
deciding. This is a different project. It needs `template <int LA, int LB, int LC, int
LD>` shell kernels: an OS/MD-style VRR with constexpr-unrolled loops and no
runtime scratch. It **must not** reintroduce the hoist's buffer or gather.
(pp|pp) and (ds|ps) will need intermediates; size them on the stack per
instantiation. C3 beat its prediction, so the go/no-go now rests on the post-C3 profile
and on whether someone is actually blocked on Fock time in sp-rich bases.

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
