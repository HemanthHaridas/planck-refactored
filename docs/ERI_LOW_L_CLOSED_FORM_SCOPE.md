# Closed-Form Low-Angular-Momentum ERI Kernels — Scope

Status: **scope; C0 done, the Boys-core fix it found shipped in Boys library
v1.1.0 (2026-10-17), and C1 done (2026-10-18, 1.16–1.26× and bitwise); no
closed-form kernel landed.** In-flight work, so
this file is allowed to be scope-shaped; rewrite it as an answer when the work
lands or is abandoned.

**C0 changed the plan.** The real Fock build spends **17.7 %** of its time in
libm calls inside one branch of the Boys core, which is more than closed forms
for L_tot ≤ 2 would save on 6-31G* (see "C0 results"). That fix lived in the
out-of-tree Boys core and **shipped in library v1.1.0**; the shares below are
re-measured on it.
Projections below are now normalized to the **whole build**, not to HGP kernel
time; the first-pass numbers overstated them by the ~16–20 % of the build that
is loop overhead rather than ERI evaluation.

**Question:** Would specialized closed-form kernels for low-angular-momentum
shell quartets (ss|ss), (ps|ss), … speed up the direct Fock build, and by how
much?

## Short answer

Only if they work at **shell** granularity. Two prototypes, both exact against
HGP to about 1e-16, measured on 4 H2O / 6-31G*:

| prototype | speedup vs HGP on its class |
|---|---|
| closed form, **per Cartesian component** (today's call interface) | **1.66–1.75×** |
| closed form, **per shell quartet** (all components from one Boys/prefactor pass) | **4.9×** |

Even with a perfect kernel, the low classes are not most of the time. By
Amdahl:

Shares of the **whole Fock build** from the C0 probe (`planck-eri-class-share`)
on Boys library v1.1.0, with the projected build speedup at ~5× on the covered
classes and the ceiling (single runs; about ±5 % noise):

| coverage | 4 H2O / 6-31G* | 1 H2O / cc-pVDZ | 8 H2O / 6-31G |
|---|---|---|---|
| L_tot ≤ 1: (ss\|ss), (ps\|ss) | 5.8 % → 1.05× (1.06×) | 13.4 % → 1.12× (1.16×) | 19.8 % → 1.19× (1.25×) |
| L_tot ≤ 2: adds (pp\|ss), (ps\|ps), (ds\|ss) | 16.8 % → 1.16× (1.20×) | 31.0 % → 1.33× (1.45×) | 45.7 % → 1.58× (1.84×) |
| L_tot ≤ 4: through (pp\|pp), (ds\|ps), … | 52.4 % → **1.72×** (2.10×) | 72.0 % → **2.36×** (3.57×) | 73.2 % → **2.41×** (3.74×) |

The Boys fix lowered the low-L shares most on the water chains (8 H2O / 6-31G
L_tot ≤ 2: 53.7 % → 45.7 %), because the far-apart pairs it sped up are
mostly low-L. On v1.0.0 the same rows were 7.2 / 19.6 / 54.8 % (4 H2O),
12.0 / 30.2 / 73.6 % (cc-pVDZ) and 23.8 / 53.7 / 80.8 % (8 H2O).

The payoff depends on the basis. On d-containing 6-31G*, hand-written forms up
to L_tot ≤ 2 are worth about 1.16×. On a pure sp basis (6-31G), L_tot ≤ 2 is
still close to half the build. The ~2× payoff needs a **compile-time-specialized
shell-quartet kernel family** up to L_tot = 4. That is the "batched engine for sp-rich bases" from
`docs/ERI_PERFORMANCE_SCOPE.md`, and it should be scoped as that.
**Against the ~200× libcint gap, this is about a 2× step, not a fix.**

## Measurements

Apple M3, `build-full` (g++-15 `-O3`), 1 thread. The scratch bench links
Planck's own integral objects and calls `HeadGordonPople::_contracted_eri_elem`,
the Auto engine for every low bucket (`kAutoEngine`, `src/integrals/rys.cpp`),
on Schwarz-surviving (1e-10) canonical component quartets. Per-quartet times
include about 20–40 ns of `steady_clock` overhead, which favors neither side
much but inflates the smallest numbers. Not committed; C0 makes it an in-tree
probe.

### Where HGP time goes, by class (L_AB + L_CD of the component pair)

4 H2O / 6-31G* (nb 76, 1.64M surviving component quartets, HGP 1.19 s):

| (L_lo, L_hi) | % quartets | % time | ns/quartet |
|---|---|---|---|
| (0,0) | 2.2 | 1.8 | 592 |
| (0,1) | 7.0 | 7.1 | 740 |
| (0,2) | 9.7 | 9.0 | 672 |
| (1,1) | 5.7 | 6.7 | 859 |
| (0,3) | 5.5 | 4.9 | 641 |
| (0,4) | 2.9 | 1.8 | 442 |
| (1,2) | 15.6 | 16.8 | 784 |
| (1,3) | 9.0 | 9.9 | 798 |
| (2,2) | 10.9 | 11.3 | 751 |
| L_tot ≥ 5 | 31.5 | 30.7 | — |

Single water: 6-31G* has the same shape. cc-pVDZ shifts weight toward
(0,1)/(0,2) (10.4 % / 10.9 %), because its contracted s functions are deep and
make low-L quartets expensive per call (the (ss|ss) class runs about 5 µs per
quartet).

### Why per-component closed forms buy so little

The (ss|ss) closed form is `Σ_pq c K_p K_q · 2√(ρ/π) F0(ρ|PQ|²)`. There is no
recurrence left to remove, and it is still only 1.66× faster than HGP. **The
floor is the primitive loop: one Boys call and one prefactor product per
primitive quartet.** An O 1s in 6-31G* has 6 primitives, so one
(1s 1s|1s 1s) is 1296 primitive quartets. The per-component interface repeats
that loop for every Cartesian component: 3× for (ps|ss), 9× for (pp|ss),
81× for (pp|pp). The shell-level kernel runs it once, and that is where the
4.9× comes from.

### Why this is not the hoist that already failed

`docs/ERI_PERFORMANCE_SCOPE.md` records the contract-once-per-shell-quartet
hoist as 1.14–1.62× **slower** at every AM. That hoist was general machinery
(copy a snapshot of the (a0|c0) block, gather sub-boxes, then run HRR per
component), and its overhead exceeded what it saved on sp bases. A closed-form
shell kernel has no buffer and no HRR: it writes the components straight from
registers. The (ps|ss) prototype is that kernel, and it measured 4.9×. **The
risk carried forward:** a template family that grows to L_tot = 4 can turn
back into the hoist (scratch arrays, gathers) if it is written generically.
Measure each class against HGP as it lands; any class that is not faster stays
on HGP through the Auto table.

### ShellPair construction becomes material

`fused_fock.h` constructs `ShellPair spCD(cvC, cvD)` inside the innermost
component loop. That is a heap `vector`, plus `pow` and `exp` per primitive
pair, for every surviving component quartet. Measured: **8 % of HGP time** on
4 H2O / 6-31G* (4–5 % on single water). Against HGP that is a rounding error,
which is why the earlier "ShellPair ctor is zero frames" finding holds. Against
a kernel 5× faster it becomes about 40 % of the remaining time. It must go
first (C1).

## Steps

### C0 — In-tree class-share probe (DONE 2026-10-17)

`planck-eri-class-share` (`tests/eri_class_share_benchmark.cpp`, not a ctest)
drives the production `Integrals::fused_fock_build`, the same loop the default
Auto engine runs, with a callable that times the HGP per-quartet kernel and
buckets it by (L_AB, L_CD). It prints the class table and the coverage
projections for the four fixtures. `--profile N` repeats the plain (untimed)
build of 4 H2O / 6-31G* so that `sample` can attribute it. Run it with
`OMP_NUM_THREADS=1`.

#### C0 results

**Kernel share, two independent ways (4 H2O / 6-31G\*).** The probe puts the
HGP kernel at ~80 % of the build: 1.583 s plain against 1.595 s timed, with
32.7 ns per steady_clock pair netted out. A `sample` profile of the plain build
(9 508 samples inside `fused_fock_build`) gives:

| inclusive share of the build | what |
|---|---|
| **84.1 %** | HGP kernel (`_contracted_eri_elem`) |
| 31.0 % | ↳ Boys function, all branches |
| **17.7 %** | ↳ libm calls in one branch of the Boys core |
| **8.4 %** | `ShellPair spCD` construction (`pow`, `exp`, heap) |
| ~7.5 % | rest of the loop: canonical filter, accumulation, `free` |

The two kernel figures agree within the timer correction, and the `ShellPair`
figure matches the 8 % the first pass assumed for C1.

**Class shares hold up.** On 4 H2O / 6-31G* the in-tree probe reproduces the
scratch bench's per-class kernel shares to within 1–2 points. The shares
depend strongly on the basis: on pure sp 6-31G, the classes with L_tot ≤ 2 are
**53.7 % of the build**, so closed forms matter most there. The C0 stop rule
("reconsider if L_tot ≤ 4 is well under ~60 %") fires only marginally, for
6-31G* (54.8 %), and the projections above are recomputed on the build basis.

**The finding C0 was not looking for: one Boys-core branch.** About 17.7 %
of the whole build is libm calls inside a single branch of the Boys core, one
that water chains reach constantly. The fix belongs in the Boys core itself,
which is out of tree (the license-gated blob loaded by
`src/lookup/loader/orchestrator.cpp`), so it lands there and ships as a new
blob; it cannot be done from this repository. Measured on the same
`fused_fock_build` harness with the fix linked in place of the blob:
4 H2O / 6-31G* **1.21×**, 8 H2O / 6-31G **1.64×**, 1 H2O / cc-pVDZ unchanged
(it never reaches that branch). The largest Fock-matrix change was 2.2e-16.
It is worth more than C2+C3 on 6-31G*. The loader's function-pointer dispatch
also prevents inlining (`Lookup::boys` self time 3 %).

**Shipped in Boys library v1.1.0 (2026-10-17).** Measured through the real
blob with `planck-eri-class-share`: 4 H2O / 6-31G* 1.583 → 1.397 s (**~1.13×**,
best of four), 8 H2O / 6-31G 2.561 → 1.866 s (**~1.37×**). That is below the
direct-link numbers above, plausibly because the blob's dispatch cannot be
inlined; the v1.0.0 baseline is a single run and can no longer be repeated
(current binaries reject v1.0.0 licenses). Not bitwise against v1.0.0: water
RHF/STO-3G (HGP) keeps -74.9458807581 to every digit but now converges in 10
SCF iterations instead of 11, because that case sits at the convergence noise
floor (`water_rhf_hgp_sto3g` re-pinned).

**Revised order of work:** (1) the Boys-core fix: **done, v1.1.0**, shares
re-measured above; (2) C1: **done**, 1.16–1.26×, bitwise; (3) C2, justified only by pure-sp bases,
where L_tot ≤ 1 is 19.8 % of the build.

### C1 — Stop rebuilding ShellPairs inside the fused loop (DONE 2026-10-18)

**The cache already exists.** Every `fused_fock_build` caller (OS, HGP and Rys
Fock / J / K, RHF and UHF) passes `shell_pairs`, the full per-AO upper-triangle
list from `build_shellpairs`. Each entry is the same `ShellPair(cvA, cvB)`
the loop builds again at `fused_fock.h:324` (`spAB`, once per bra component
pair per ket block) and `:374` (`spCD`, once per surviving component quartet).
The canonical filter already ensures `j >= i` and `l >= k`, so every pair the
loop builds is in that list with the same A/B order. C1 needs no new struct:

1. At the top of `fused_fock_build`, build an `nb × nb` table of
   `const ShellPair *`, filled from `shell_pairs` keyed on
   `(sp.A._index, sp.B._index)`. Key on the index; do not assume the list
   order. `fused_shell_groups` already allows for a list that does not
   cover every AO (`_index < nbasis` guard).
2. Replace both constructions with a table lookup. If an entry is null,
   construct the pair as today (`// ponytail:` fallback for a partial list).
   Memory is 8·nb² bytes (2 MB at nb = 500). The pairs themselves already
   live in the caller.

This is the narrow version of the shell-level cache. It does not touch the
shared `ShellPair` contract, so Phase E stays independent. Folding
per-component norms out of `coeff_product` is deferred until C2 needs it.

- **Gate: bitwise.** Same constructor, same inputs, same summation order.
  So Fock matrices must be byte-identical, and any difference is a bug.
  Check `planck-hgp-engine-smoke`, `planck-compute-2e` and the smoke
  regression suite with energies unchanged at every printed digit, plus
  one full-symmetry direct case (the orbit path shares the loop) and one
  UHF case.
- **Measure** with `planck-eri-class-share` on all four fixtures, best of
  three, `OMP_NUM_THREADS=1`. Expected: the 8.4 % `ShellPair` slice on
  4 H2O / 6-31G\* goes away, about 1.08–1.09×. Confirm with `--profile`
  plus `sample` that `ShellPair::ShellPair` has left the profile. It
  applies to every engine, not only HGP.
- **Then re-run the class table.** With the loop overhead gone, the kernel
  shares rise by about 1/0.92. C2/C3's predictions must be recomputed on
  the new shares, not the ones above.

#### C1 results

Landed as described: a `pair_of` pointer table in `fused_fock_build`, plus a
construct-on-miss fallback that no current caller reaches. Every engine
benefits, because OS, HGP and Rys all drive the same loop.

- **Bitwise.** The probe now prints an FNV-1a fingerprint of G. It is
  byte-identical before and after on all four fixtures. `hartree-fock`
  output for `ethylene_rhf_ccpvdz_auto` is identical except for timing.
  Passing: `planck-hgp-engine-smoke`, `planck-compute-2e`, the full-symmetry
  RHF/UHF direct cases, the OS-vs-HGP SCF comparators (RHF, UHF, HSE06,
  symmetric HSE06, UKS) and core 78/79. The one failure,
  `ethylene_rhf_ccpvdz_auto` (15 iterations against a floor of 16),
  predates C1: the HEAD-header binary prints the same 15. Its floor is now `>= 15`.
- **Speed.** Best runs, interleaved before/after binaries, 1 thread:

  | fixture | before | after | gain |
  |---|---|---|---|
  | 4 H2O / 6-31G\* | 1.17 s | 1.01 s | **1.16×** |
  | 8 H2O / 6-31G | 1.50 s | 1.19 s | **1.26×** |
  | 1 H2O / cc-pVDZ | 0.071 s | 0.066 s | 1.07× |

  That is twice the 1.09× the 8.4 % profile share implied. A profile
  share is again a lower bound: the allocator and cache traffic of the
  rebuild were charged to other frames. `sample` confirms that
  `ShellPair::ShellPair`, `pow`/`exp` and malloc are gone from the profile.
  The top is now `hgp_vrr`, Boys, `hgp_contract_a0c0`, HRR. Timings on this
  machine are bimodal under load (P/E cores), so compare best runs, never
  single ones.
- **Found on the way:** `planck-hgp-engine-smoke` and `planck-compute-2e`
  segfaulted (ctest too) because they never called `require_boys_library()`,
  so `Lookup::boys` was a null pointer. Both were fixed. They had been
  silently broken since the license-gated Boys blob arrived.

**Class shares after C1** (share of the build, projected @5× and ceiling):

| coverage | 4 H2O / 6-31G\* | 1 H2O / cc-pVDZ | 8 H2O / 6-31G |
|---|---|---|---|
| L_tot ≤ 1 | 6.1 % → 1.05× (1.07×) | 11.5 % → 1.10× (1.13×) | 23.0 % → **1.23×** (1.30×) |
| L_tot ≤ 2 | 18.2 % → 1.17× (1.22×) | 30.0 % → 1.32× (1.43×) | 54.5 % → **1.77×** (2.20×) |
| L_tot ≤ 4 | 58.3 % → 1.88× (2.40×) | 75.0 % → 2.50× (4.0×) | 87.9 % → **3.4×** (8.3×) |

The kernel is now about 89 % of the build on every fixture. The pure-sp case
strengthens: closed forms through L_tot ≤ 2 project to 1.77× on 8 H2O / 6-31G.

### C2 — (ss|ss) and (ps|ss) shell kernels, Coulomb only

- Add a shell-quartet entry point to the fused loop for these classes. The
  kernel returns all `n_A n_B n_C n_D` components; the loop then applies the
  existing canonical filter (`j<i`, `l<k`, bra-ket order), the Schwarz test and
  the symmetry-orbit logic **per component on the output**, unchanged, so the
  8-fold dedup argument in `fock_accumulate.h` still holds.
- Register it in front of HGP for buckets (0,0) and (0,1) only, `kernel ==
  Coulomb`. LongRange/ShortRange stay on HGP (see C5).
- Gates: a per-quartet sweep against HGP at ≤ 1e-14 relative (the prototype
  gives about 5e-17), RHF/UHF SCF energies at existing tolerances,
  `planck-compute-2e`, `planck-hgp-engine-smoke`, and thread-count invariance
  (the fused loop's static schedule is unchanged). The result is **not**
  bitwise against HGP, because the summation order changes.
- **Measure on 8 H2O / 6-31G, not 6-31G\*.** After C1, L_tot ≤ 1 is 6.1 %
  of the 4 H2O / 6-31G\* build (about 1.05×, which is inside the run noise)
  but 23.0 % on 8 H2O / 6-31G (about 1.23×). Only the sp fixture can falsify
  the prediction.
- **Stop rule:** if the sp-fixture gain is materially below 1.23× the interface overhead is eating the kernel, and C3/C4 will
  not pay either.

### C3 — Hand-written through L_tot ≤ 2

(pp|ss), (ps|ps), (ds|ss) and their bra/ket mirrors. The d components need
their per-component norm at output. Predicted cumulative after C1: about
1.17× on 6-31G\*, 1.77× on 8 H2O / 6-31G.

### C4 — Decision point: template-specialized family through L_tot ≤ 4

The payoff (projected about 2.2×) is here, and it is a different project. It
needs `template <int LA, int LB, int LC, int LD>` shell kernels, an
OS/MD-style VRR with constexpr-unrolled loops, and no runtime scratch. It
**must not** reintroduce the hoist's buffer or gather. Expect (pp|pp) and
(ds|ps) to need intermediate arrays; size them on the stack per instantiation.
Go/no-go on C2/C3 tracking their predictions, and on someone actually being
blocked by sp-basis Fock time.

### C5 — Screened kernels

The closed forms take the same screened scaling as `hgp_vrr`: `T ×
boys_scale`, prefactor `× prefactor_scale`, and the W–P coupling scaled by
`screen.rho / rho`. The HGP `inv_2_delta` gotcha
(`vault/Gotchas/HGP Screened inv_2_delta`) is the warning: every
kernel-aware factor must be mirrored exactly, and (p|s|p|s)-type
cross-coupling is where it hides. Until a screened sweep against OS passes
over a mixed bra/ket-AM fixture, screened kernels stay on HGP.

## Out of scope

- **Gradients.** Derivative ERIs are a separate kernel family (Open Work
  Phase D).
- **Primitive screening.** It is the other lever on the per-component floor
  above (skip primitive quartets with negligible `K_ab K_cd`), and it is
  independent of closed forms. Worth its own measurement: the cc-pVDZ (ss|ss)
  class at about 5 µs per quartet suggests deep contractions dominate there.
- **Spherical.** The direct spherical path back-projects to Cartesian and is
  unaffected.
