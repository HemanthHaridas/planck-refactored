# Shell-pair granularity (H-10) — architecture

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers one architecture question:

**How does ERI work iterate at shell-quartet granularity, what amortization that
buys is actually live, and what rules must new integral code follow to stay
correct?**

The branch is `perf/shellpair-granularity`. The chronological step-by-step log
(steps 1, 2a–2c, A1–A4-3, OS-A4-0/1) has been folded into the architecture
description below; §4 keeps the one design blocker that shaped the final shape,
and §6 lists what remains out of scope.

## Core idea

The integral engines historically iterated ERI work at **per-Cartesian-AO-
component** granularity: `build_shellpairs` produced one `ShellPair` per ordered
pair of AO components, and each engine computed a 1×1 contribution per component
quartet. A single `(dd|dd)` quartet therefore re-ran the shared per-primitive
seed work (Gaussian product center, prefactor, Boys, VRR) `6⁴ = 1296` times
instead of once.

H-10 moves iteration to **per-shell-quartet** granularity, so the shared per-
primitive work is amortized across a shell quartet's Cartesian components. Every
sub-step was independently bitwise-gated and reversible, and the shared per-AO
`ShellPair` list stays in the tree throughout (via a no-op adapter), so any
engine can still be reverted to per-AO.

## What is live now

- **OS, HGP, and the default Auto path all iterate ERI work at shell-quartet
  granularity.** The shared phase-1 (per-primitive seed + VRR contraction) is
  built once per shell quartet, not once per component.
- **HGP and OS amortize the per-primitive VRR + `(a0|c0)` contraction across a
  shell quartet's components**, then run the two HRR passes once per quartet —
  the actual primitive-work saving (the Head-Gordon-Pople rearrangement). For
  HGP this is live on `_compute_2e`, the Fock builds that delegate to it, and the
  default Auto path's HGP-chosen quartets. For OS it is live in `_contracted_eri`
  (OS-A4).
- **Auto dispatch is three-way (OS / HGP / Rys), data-derived.** A per-bucket
  `(L_AB, L_CD)` calibration picks the empirically fastest engine; OS re-entered
  the menu in a high-L corner once it got the same HRR hoist HGP had.
- **Rys-chosen Auto quartets stay per-component.** Under the three-way table Rys
  is now selected only at the extreme high-L tail `(7,8)` and `(8,8)`; those
  quartets are still evaluated one Cartesian component at a time. This is the one
  remaining un-amortized engine on the Auto path — a Rys hoist *is* possible and
  is now a non-trivial win there (see Phase B, §6), contrary to earlier notes.

## The transform model — how the hoist works

For one shell quartet the engines compute the contracted half-transformed block
`(a0|c0)` (bra and ket reduced to a single center each), then transfer angular
momentum to the other centers with two HRR passes:

- **Phase 1 (per primitive pair, contracted):** run VRR for each primitive pair,
  accumulate the `(a0|c0; m=0)` slice into a quartet-level accumulator scaled by
  the primitive coefficient product. HGP: `hgp_contract_a0c0` → `a0c0_accum`.
  OS: `_os_contract_a0c0` → `EriScratch::a0c0_accum`, reusing `hrr_data` as the
  per-pair VRR scratch.
- **Phase 2 (once per shell quartet):** copy the contracted accumulator into the
  HRR buffer and run A→B then C→D HRR. HGP: `hgp_hrr_finalize`. OS:
  `_os_eri_hrr_to_eri`.

Screened kernels are handled by linearity of HRR in the `(a0|c0)` block: Coulomb
and LongRange each run a single hoisted pass; `ShortRange = Coulomb − LongRange`
runs two and subtracts. This is bitwise-exact against the per-kernel per-pair
form.

## Where the implementation lives

- `src/integrals/shellpair.{h,cpp}`: `ShellGroup`, `build_shell_groups`,
  `expand_shell_groups_to_ao_pairs`; `build_shellpairs` routes through the no-op
  adapter
- `src/integrals/hgp.{h,cpp}`: `hgp_contract_a0c0`, `hgp_hrr_finalize`,
  `HoistedQuartet`/`MaxBoxLayout`, `_contracted_eri_block_hoisted{,_views}`,
  `EriScratch::a0c0_accum`
- `src/integrals/os.{h,cpp}`: `_os_eri_build_a0c0`, `_os_eri_hrr_to_eri`,
  `_os_contract_a0c0`, `_contracted_eri_base`, `EriScratch::a0c0_accum`
- `src/integrals/rys.cpp`: `_compute_2e_auto` (the default Auto tensor + Fock
  builds), the three-way `kAutoEngine[L_AB][L_CD]` table and `_auto_engine`
  selector, `_auto_contracted_eri`
- `tests/auto_dispatch_benchmark.cpp` + `scripts/fit_auto_dispatch.py`: the
  per-bucket calibration and the fitter that derives the dispatch table
- `docs/auto_dispatch_fit.json`: the fitted `region_table` and the generated C++
  that is pasted verbatim into `kAutoEngine`

## Architecture invariants

### 1. `(k,l) >=_lex (i,j)` is exactly the old flat-pair `q >= p`

The per-AO `build_shellpairs` list is the row-major AO upper triangle, so its
flat pair index increases lexically in `(i,j)`. The shell-quartet rewrites visit
**every** ket shell pair (`ket = 0..ngp`, **not** pruned by flat index — the lex
order of component AO indices is not monotonic in the shell-pair index, so
pruning ket by index drops valid quartets) and filter per component with the lex
check. This reproduces the canonical quartet set exactly. The 8-fold scatter is
store-only, so the tensor is independent of visitation order and the result is
bitwise-identical to the per-AO build.

### 2. `_component_norm` is folded per component before contraction — a hoisted contraction must be norm-free and apply the norm at readout

The per-AO path folds each component's `_component_norm` into its primitive
`coeff_product` *before* contraction. A path that builds **one** contraction per
shell quartet cannot carry per-component norms in the shared block: it must
contract norm-free (component-0 views with `_component_norm` forced to 1) and
multiply each component's readout by `normA·normB·normC·normD` *after* HRR. This
makes the hoisted path **not bitwise** against the per-AO build for d-shell
components (norm ≠ 1): `Σ(wᵢ·n)·vᵢ` and `(Σwᵢ·vᵢ)·n` are mathematically equal but
round differently at the last FP bit. The cross-validation bar is therefore
relative `1e-13` (observed worst case ~9.6e-16 on 6-31G\*); cases where all norms
are 1 (sto-3g) stay bitwise.

### 3. The `(a0|c0)` contraction is box-size invariant; the HRR readout is not

`hgp_vrr` is strictly bottom-up over a dense rectangular `EriScratch` cube, so
contracting at the max-AM box (`lAB = L_A+L_B`, `lCD = L_C+L_D` per axis) gives,
at every component's `(a0|c0)` sub-block, **bitwise** the same value as a
contraction sized to that component — the extra high-AM cube corners are never
read by a lower coordinate. The readout must HRR **only that component's sub-box**
(`0..lABx_comp`, `0..lCDx_comp`), never the full max-AM cube: sweeping the
unreachable corners is what produced NaN in the first hoist attempt (§4).

### 4. Per-quartet screening is preserved by deferring the contraction

Schwarz screening and the symmetry orbit-front check are per component. The
`_compute_2e` hoist builds the shared contraction lazily, on the first component
that survives both screens (`HoistedQuartet::prepare()` on first `readout()`), so
a fully screened quartet costs nothing. The Auto path (`_compute_2e_auto`) fills
the whole block on the first survivor instead — the expensive contraction still
runs once; only the cheap per-component HRR is "wasted" on later-screened
components, which is equivalence-preserving (store-only scatter ignores
never-written entries). A shell-quartet-level Schwarz *prescreen* would change the
screened set and is therefore **not** equivalence-preserving — out of scope.

### 5. Auto dispatch is derived from data, not hand-coded inequalities

`kAutoEngine[L_AB][L_CD]` is a dense constexpr table copied verbatim from the
fitter's `rule_in_code` in `docs/auto_dispatch_fit.json`, which assigns each
bucket to the engine with the lowest cross-case median per-quartet time. The
OS/HGP/Rys region boundaries are irregular and move when an engine is optimized,
so the table is kept verbatim — not reduced to inequalities — to avoid drift. To
re-adopt after future engine work: re-run `planck-auto-dispatch-benchmark`,
re-run `scripts/fit_auto_dispatch.py`, paste the regenerated table.

## What a contributor should check before changing integral code

1. Does the change touch the quartet visitation order? If yes, it stays correct
   only because the scatter is store-only — keep it store-only, and keep the
   `(k,l) >=_lex (i,j)` per-component filter.
2. Does it add a hoisted/shared contraction? If yes, contract norm-free and apply
   `normA·normB·normC·normD` at readout (invariant 2), and HRR only each
   component's sub-box (invariant 3).
3. Does it screen? Preserve per-component Schwarz/orbit screening by deferring the
   contraction; do not introduce a quartet-level prescreen (invariant 4).
4. Does it change an engine's per-quartet cost? If yes, re-benchmark and re-fit
   the Auto table (invariant 5) — do not hand-edit `kAutoEngine`.
5. Does it add a new engine entry or kernel? Gate it `==`/`≤1e-13` against the
   per-component path on a d-shell basis (6-31G\*), the gate that caught the §4
   blocker.

---

## §4 — the design blocker that shaped the hoist {#A4}

The first hoist attempt (A4-1) built HGP's `(a0|c0)` block once per shell quartet
at max AM, preserved it, and had each component HRR its readout **from the full
max-AM cube**. The `planck-os-block-kernel` gate caught it: bitwise pass on
STO-3G (s,p only), **NaN on 6-31G\*** (d-shells; ~3000 quartets, `max|diff|=0`
i.e. NaN, not a finite error).

Root cause: `EriScratch` is a dense per-axis rectangular cube, and building "max
AM" as `lABx = lABy = lABz = L_A+L_B` makes `hgp_vrr` fill the diagonal corner
`(L_A+L_B, L_A+L_B, L_A+L_B, …)` that **no single component ever needs** (a
component's total bra AM is distributed *across* the three axes, never `L_A+L_B`
on each). The original A4-1 **HRR readout** swept those unreachable corners — that
is where the NaN came from, **not** the contraction. (It is not a Boys overflow:
`boys(n,x)` supports `n < 66`, and `(dd|dd)`'s `MMAX = 24` is well within range.)

The resolution (invariant 3): the contraction at the max box is bitwise box-size
invariant, so the hoist contracts once at the max box with the *unmodified* dense
`hgp_contract_a0c0`, then HRRs each component's **sub-box** only. No triangular
VRR/`EriScratch` rework was needed — the readout simply must not touch the cube
corners. This is the shape that landed (HGP and OS alike).

---

## §6 — deferred / out of scope

None of these block the landed work.

- **Phase B — Rys shell-quartet hoist.** *(Earlier notes claimed "Rys has no
  VRR/HRR to hoist" and that Auto sends Rys only `L_AB+L_CD ≤ 1` ≤3-component
  quartets. Both were wrong; corrected here with a full scope.)*

  **What is redundant.** In `_rys_eri_primitive` (`src/integrals/rys.cpp`) the
  per-component path recomputes, for every Cartesian component of a shell quartet:
  (a) the primitive-pair geometry `P,Q,W,PA,QC,WP,WQ,T,rho,prefac`; (b) the Rys
  roots/weights and per-root `B00/B10/B01`; (c) the per-root 1D VRR tables
  `Ix/Iy/Iz`; (d) the 6D outer-product accumulation; then (e) the per-component
  HRR (`_rys_hrr_ab`/`_rys_hrr_cd`). Only (e) is genuinely per-component. Auto now
  routes Rys only `(7,8)`/`(8,8)`, the highest-component quartets — a `(g,g|g,g)`
  shell quartet is `15⁴ = 50 625` component quartets, so (a)–(d) are rebuilt
  50 625× when they could be built far fewer times.

  **Why this is NOT the HGP/OS A4 shape (the load-bearing caveat).** The "build
  one max-AM box, read every component's sub-box" trick (invariant 3) **does not
  transfer to Rys.** The Rys root count is `n = L/2 + 1` with `L` the
  *per-component* total angular momentum, and `rys_roots_weights` returns
  Gauss–Stieltjes–Jacobi points that are **not nested across `n`** (the `n=4`
  rule's points differ entirely from the `n=5` rule's). So a 1D VRR table built at
  the max-box root count is on the *wrong quadrature points* for any lower-L
  component. The shared phase can only be shared **within a fixed root count `n`**,
  not across the whole quartet.

  **The simplification that makes Option A clean (no by-`n` grouping).** The
  earlier worry — that non-nested roots force grouping components by their root
  count `n` — turned out to be unnecessary. Two facts collapse it to a *single*
  shared build per quartet:
  1. **1D VRR is bottom-up, so it is box-size invariant.** `_rys_vrr_1d` built at
     the max axis box (`lAB = L_A+L_B`, `lCD = L_C+L_D` per axis) contains every
     lower-AM component's `Ix[a][c]` sub-block **bitwise** (verified: a max box
     `(8,8)` vs a component box `(3,4)` agree to `max|diff| = 0`).
  2. **Gauss over-integration is exact.** A component cell of total degree `d`
     needs `⌈(d+1)/2⌉` roots; evaluating it with the quartet's *max* root count
     `n_max = (L_AB+L_CD)/2+1 ≥ n_comp` is still *mathematically* exact (an
     `n`-point Gauss rule integrates any polynomial of degree `≤ 2n−1` exactly).
     So using `n_max` roots for every component reproduces each component's value
     — the non-nestedness is irrelevant once everyone uses the *largest* count.
     **Crucially `n_max` is the quartet quadrature degree, NOT the summed per-axis
     box** (that would give n=25 for a g max-box, past `RYS_MAX_ROOTS=11`); the
     build takes `n_roots` explicitly (B-1).

  Therefore the shared phase is built **once** per shell quartet at the max-axis
  box with `n_max` roots, and every component reads its 6D `sum` sub-block out of
  it. This is **tight-tolerance, not bitwise** wherever `n_max > n_comp`: summing
  `n_max` weighted roots vs `n_comp` rounds at the last FP bit (the Rys analogue
  of the HGP/OS norm-reorder drift). B-1 measured rel ≤ 4e-16 on d-shells and
  exactly 0 where `n_max==n_comp` — well inside the 1e-13 ERI bar. The cost paid
  is that low-L components carry a few extra roots — negligible against the
  per-component rebuild it removes.

  **Option B (minimal, rejected):** hoist only the primitive geometry (a). That is
  the cheap part; the dominant cost (roots + 1D VRR + 6D sweep) stays per
  component. Not worth its own effort — Option A subsumes it.

  **Recommendation: defer unless g/h-basis throughput is an explicit target.**
  The win is large per quartet but confined to cc-pVQZ/pV5Z-class work; routine
  ≤ 6-31G(d,p) never routes to Rys (zero impact there). If pursued, do Option A.
  **Prerequisite — LANDED.** The high-L Rys path was previously unguarded (the
  suite topped out at d). A g-shell gate now covers the `(7,8)`/`(8,8)` buckets:
  `ne_rhf_ccpvqz_highL_pyscf` (Ne/cc-pVQZ Cartesian RHF, PySCF-anchored to
  -128.5435344972 Eh) and `ne_rhf_ccpvqz_highL_engine_os_vs_rys` (OS==HGP==Rys==
  Auto to 0.0e+00 on the same input). Both `extended`-tagged; reproducer is
  `tests/pyscf/ne_rhf_ccpvqz_highL.py`. Any Rys kernel change (Phase B) must keep
  these green.

  **Option A — small, verifiable, reversible steps.** Each step is independently
  gated and revertible; the per-component `_rys_eri_primitive` stays in the tree
  as the fallback until the last step, mirroring the OS/HGP A4 sequencing. The
  gate at every step is `ne_rhf_ccpvqz_highL_engine_os_vs_rys` (OS==Rys==Auto on
  g shells) plus `planck-compute-2e` (Rys/Rys-Auto-vs-OS), and the
  `ne_rhf_ccpvqz_highL_pyscf` absolute anchor.

  - **B-0 — extract the seam (no behavior change). LANDED (uncommitted).** Split
    `_rys_eri_primitive` into three statics (anonymous namespace in
    `src/integrals/rys.cpp`), called in sequence so the result is
    bitwise-identical: `_rys_eri_prep` (per-primitive-pair geometry →
    `RysPrimGeom` struct, component-independent), `_rys_eri_build_sum` (roots +
    per-root 1D VRR + 6D accumulate + prefactor scale at a given box → fills
    `RysScratch`), and `_rys_eri_hrr_to_eri` (AB-HRR + CD-HRR readout → scalar).
    Mirrors OS-A4-0 (`_os_eri_build_a0c0`/`_os_eri_hrr_to_eri`). Gated: `engine`
    comparator OS==Rys==Auto to 0.000e+00 on water/sto-3g and Ne/cc-pVQZ (g-shell,
    the high-L Rys path), and `planck-compute-2e` (golden checksum + Rys-vs-OS
    7.78e-14 unchanged + Rys-Auto-vs-OS 0). Revert: inline the three back.

  - **B-1 — box-invariance gate. LANDED (commit on branch).** Test hook
    `RysQuad::_build_sum_native_test` (`src/integrals/rys.{h,cpp}`) fills the full
    6D `sum` buffer for a fixed primitive pair at a caller-given `(box, n_roots)`,
    before HRR. Unit test `planck-rys-box-invariance`
    (`tests/rys_box_invariance.cpp`, ctest #11) asserts, for every shell quartet,
    that the max-box build (n_max roots) equals the per-component-box build
    (n_comp roots) at every component coordinate, within the 1e-13 ERI bar.

    **Correction to the scope's "bitwise" claim:** it is **tight-tolerance, not
    bitwise**, wherever `n_max > n_comp`. The two builds sum a different number of
    weighted roots (Gauss over-integration: mathematically equal, but the term
    count rounds differently at the last FP bit), exactly analogous to the
    HGP/OS norm-reorder drift (invariant 2). Observed: water/6-31g\*
    (d-shells, n_max>n_comp for most components) **rel ≤ 3.96e-16** over 1.99M
    coords/kernel (Coulomb/LongRange/ShortRange); Ne/cc-pVQZ (g-shells, Lq≥7 —
    the (7,8)/(8,8) buckets, where the checked components have n_max==n_comp)
    **exactly 0.0** over 4.77e9 coords/kernel. So: bitwise where n_max==n_comp,
    ≤4e-16 where n_max>n_comp; either way well inside the 1e-13 gate. The key
    discovery from the crash that preceded this: the root count must be the
    quartet quadrature degree `(L_AB+L_CD)/2+1`, NOT derived from the summed
    per-axis box (which gives n=25 for a g max-box, past RYS_MAX_ROOTS=11) —
    `_rys_eri_build_sum` now takes `n_roots` explicitly. Revert: delete hook +
    test + CMake hunk; inline the build's `n` derivation.

  - **B-2 — `RysHoistedQuartet` (standalone, off the hot path).** Mirror HGP's
    `HoistedQuartet`/A4-2: a struct that contracts the 6D `sum` **once per shell
    quartet** at `(max box, n_max roots)`, snapshots it, and reads each Cartesian
    component out via a per-component HRR. Not wired into any entry point in B-2
    — validated only against `_rys_contracted_eri`. B-2a folded into B-1;
    B-2.5a/b/c, B-2b, B-2c, and B-2d are ALL LANDED. The entire off-hot-path Rys
    hoist is built and gated; only production wiring remains.

    **Execution order: B-2.5 was done FIRST** (independent of the hoist,
    testable against current code, highest-risk). Remaining sequence:
    **B-3 → B-4** (B-2.5 is listed after B-2 here only to keep the
    integral-side sub-steps grouped).

    - **B-2a — FOLDED INTO B-1; the "bitwise refactor" framing was wrong.**
      B-2a was scoped as a bitwise no-op that factors out `_rys_contract_sum`
      ("contract all pairs, then HRR once") and routes `_rys_contracted_eri`
      through it. **That is not a no-op.** The original `_rys_contracted_eri`
      sums *per-pair HRR scalars* (`Σ coeff·HRR(sum_pair)`); contract-then-HRR
      computes `HRR(Σ coeff·sum_pair)`. HRR is linear so these are mathematically
      equal, but moving the sum across HRR **reorders the floating-point
      accumulation** — there is no bitwise way to hoist HRR. An attempt confirmed
      it: integral-level gates were untouched (`planck-compute-2e` Rys-vs-OS
      *unchanged* at 7.78e-14; all SCF-energy comparators 0.000e+00 on
      sto-3g/6-31g\*/Ne/HSE06), but the ~1e-16 reorder flipped
      `water_casscf_sa2_sto3g_sad_guess_uphill` (`engine rys`) from its expected
      SA-2 basin (−74.7877864784) into the other valid SA stationary point
      (−74.7751377977) — a 0.013 Eh jump from chaotic amplification in a
      deliberately basin-sensitive uphill SA-CASSCF optimization. Reverting the
      reorder makes it pass; restoring it fails. So:
      - the bitwise/non-bitwise boundary is **not** between B-2a and B-3 — it is
        the moment production adopts the hoisted order, i.e. B-3;
      - B-2a's only genuine, behavior-preserving content (the `_rys_eri_build_sum`
        `n_roots` seam) **already landed in B-1**;
      - the contraction/readout helpers are introduced in B-2c/B-2d with their
        off-hot-path test as first consumer (no orphan unused statics).
      Net: B-2a contributes nothing new; it is closed as subsumed by B-1.

    Each remaining sub-step lands its new static(s) **together with the test arm
    that first calls them** — never an orphan unused static (the B-2a lesson) —
    and **production `_rys_contracted_eri` stays on the bitwise per-pair path
    throughout B-2** (the hoisted order is adopted only at B-3/B-4). The shared
    test driver is `tests/rys_box_invariance.cpp` (already builds water/6-31g\*
    and Ne/cc-pVQZ Lq≥7 calculators and iterates shell quartets). Validation is
    **tight tolerance ≤1e-13, not bitwise** wherever `n_max > n_comp` (B-1
    root-count drift); arms that compare two *per-component-box* builds are
    bitwise.

    - **B-2b — `_rys_contract_sum` (contract over primitives at a fixed box).
      LANDED.** Test hook `RysQuad::_contract_sum_native_test` (`src/integrals/
      rys.{h,cpp}`): loop primitive pairs, build each pair's kernel-resolved 6D
      `sum` (`_rys_build_pair_value_block` — Coulomb/LongRange a single build,
      ShortRange = Coulomb − LongRange per pair, mirroring `_rys_contracted_eri`'s
      composition), accumulate weighted by `coeff_product` into `acc` (Rys analog
      of `hgp_contract_a0c0`). Norm handling deferred to B-2c — the views still
      carry folded `_component_norm`, so per-component shell pairs are used.
      Companion hook `_hrr_block_native_test` HRRs a contracted block to a scalar
      (reuses the anon-namespace `_rys_eri_hrr_to_eri`).

      **Gate (≤1e-13, NOT bitwise — scope correction):** the test arm
      (`tests/rys_box_invariance.cpp`, `check_contract`) contracts at the
      *component box / n_comp* over the full primitive-pair set and HRRs the
      result, comparing each component against `_rys_contracted_eri`. The scope's
      "bitwise at a single primitive pair" was **wrong**: even at one pair, the
      hoisted contract-then-HRR is `HRR(coeff·build)` while production is
      `coeff·HRR(build)` — production applies `coeff_product` *once after* HRR,
      the hoist folds it into the block so it rides through every HRR add
      (`coeff·(x+CD·y)` vs `coeff·x+CD·(coeff·y)` round differently). That
      coeff-placement reorder, plus the cross-pair sum moving across HRR, both
      round at the last bit, so the correct bar is the 1e-13 ERI bar (same as B-1
      and B-2c/d), exactly as the B-2a finding already established for any HRR
      hoist. Measured: water/6-31g\* (d-shells, all three kernels)
      `max|Δ| ≤ 1.6e-15`, `rel ≤ 5.4e-15`; Ne/cc-pVQZ Lq≥7 (g-shells, n_max==n_comp)
      **exactly 0.0** over 21.5M components. (The reported `rel` floors near-zero
      ERIs at `adiff > 1e-15`, matching the failure condition, so a near-zero ERI
      can't inflate the reported relative diff.) Revert: delete the two hooks +
      `_rys_build_pair_value_block` + the `check_contract` arm.

    - **B-2c — `RysMaxBoxLayout` + norm-free max-box readout. LANDED.**
      `RysMaxBoxLayout(maxAB..,maxCD..)` holds a snapshot's 6-axis strides (same
      cz-fastest convention as `RysScratch`, independent of `g_rys_scratch`'s
      later per-component resizes; Rys analog of HGP's `MaxBoxLayout`).
      `rys_hoist_readout_component(layout, snapshot, comp_box, …)` gathers a
      component's sub-box from an n_max snapshot into a component-sized
      `RysScratch`, HRRs it; the caller applies `normA·normB·normC·normD`
      (invariant 2: the shared norm-free contraction can't carry per-component
      norms — the norm-free views are made inline, `_component_norm = 1`). Two
      test hooks (`src/integrals/rys.{h,cpp}`): `_contract_maxbox_snapshot_native_test`
      builds the norm-free snapshot ONCE per quartet, `_maxbox_readout_native_test`
      reads each component out — split this way so the test contracts once per
      quartet, exactly as the production hoist (B-3) will (the single all-in-one
      form re-contracted per component, ~25× slower on g-shells).

      **Gate (≤1e-13):** `check_maxbox_readout` (`tests/rys_box_invariance.cpp`) —
      build the snapshot once, read out + norm each component, compare against
      `_rys_contracted_eri` (norm folded per pair, per-component build). This is
      the first step exercising the n_max-over-component reorder *and* the norm-
      after-HRR reorder together. Measured: water/6-31g\* (d-shells, all 3 kernels)
      `max|Δ| ≤ 1.1e-15`, `rel ≤ 5.4e-15`; Ne/cc-pVQZ Lq≥7 (g-shells)
      `max|Δ| ≤ 1.1e-15`, `rel ≤ 6.3e-16`. Note the g-shell case is now *nonzero*
      where B-2b was exactly 0.0: that is precisely the norm-after-HRR reorder
      (production folds `_component_norm` into the primitive coeff before HRR, the
      hoist applies it after) — last-bit, well inside the bar. Revert: delete
      layout + readout + the two hooks + arm.

    - **B-2d — assemble `RysHoistedQuartet` + pointer-array entry. LANDED.**
      `RysHoistedQuartet` (anon namespace in `src/integrals/rys.cpp`, Rys analog of
      HGP's `HoistedQuartet`): ctor takes the four component-0 views + kernel/omega,
      makes norm-free copies (`rys_normfree_view`), records `maxAB/maxCD` and the
      quadrature degree `n_roots`; lazy `prepare()` runs the factored-out
      `_rys_contract_sum` from norm-free views at `(max box, n_max)` into
      `sum_primary` (and, for ShortRange, `sum_secondary` from the LongRange
      kernel); `readout(component, norm)` calls `rys_hoist_readout_component` on
      each snapshot, combines (`primary − secondary` for ShortRange), and scales by
      `norm`. Public entries `_contracted_eri_block_hoisted_views(vA,nA,…)`
      (pointer-array form, so B-3's non-contiguous `ao_views` feed it directly) and
      `_contracted_eri_block_hoisted(basis, gA..gD, …)` (ShellGroup wrapper),
      mirroring HGP one-for-one. **Gate (≤1e-13):** `check_block` in
      `tests/rys_box_invariance.cpp` fills the block via
      `_contracted_eri_block_hoisted` for every shell quartet of water/6-31g\* and
      Ne/cc-pVQZ Lq≥7 (all kernels) and compares each component against
      `_rys_contracted_eri`. Measured: water/6-31g\* `max|Δ| ≤ 2.3e-15`,
      `rel ≤ 1.1e-14`; Ne/cc-pVQZ Lq≥7 `max|Δ| ≤ 1.3e-15`, `rel ≤ 1.4e-15`. Still
      no production wiring (that is B-3/B-4). Revert: delete struct + the two
      entries + `rys_normfree_view` + the `check_block` arm.

    Net B-2: **B-2b/c/d all LANDED.** The hoisted path exists end-to-end
    (`RysHoistedQuartet` + the two block entries) and is proven equal to the
    per-component path to ≤1e-13, but nothing in production calls it yet.
    ShortRange = two snapshots combined per component, as in OS/HGP. **B-2.5
    (CASSCF hardening) landed before this**, and B-2.5c measured the reorder
    robustness the wiring relies on — so B-3/B-4 are unblocked.

  - **B-2.5 — harden the SA-2 CASSCF plateau-escape (PREREQUISITE for B-3 AND
    B-4). a+b+c ALL DONE (c run at B-2d time).** Done *before* any B-2 integral
    work, as scoped — it is pure
    CASSCF-convergence logic, independent of the hoist, fully testable against the
    current per-pair production code, and it is the highest-uncertainty piece
    (touches a basin-sensitive optimizer with 11 PySCF gates), so front-loading it
    surfaces any "this basin isn't cleanly stationary" surprise before building
    B-2b/c/d.

    **What the code actually does (read, not assumed).** The plateau-exit is
    `casscf.cpp:1238` — `stagnation_streak ≥ 2 && small_energy_change &&
    accepted_micro_step_plateau && reported_gnorm < 100·tol_mcscf_grad`.
    `reported_gnorm = st_current.g_orb.cwiseAbs().maxCoeff()` and
    `g_orb = build_weighted_root_orbital_gradient(...)`, so **`reported_gnorm` is
    already `‖sa_g‖∞`** — the screen gates on the right quantity, but the
    threshold `100·tol_mcscf_grad ≈ 1e-3` (default tol 1e-5) is loose and
    implicit. Measured on the uphill case: at the plateau exit `sa_g = 3.45e-10`
    (descending 1.9e-2 → 1.6e-2 → … → 1e-10 over the last macros), while
    `max_root_g` plateaus at 7.69e-2 (per-root screens never vanish under SA — the
    documented reason this exit exists). So the loose screen passes *trivially*;
    the basin flip happened during the descent, not at the exit.

    **Why tightening makes it rounding-robust.** Anchor the plateau exit to a
    regime where `sa_g` is many orders below the threshold: require
    `sa_g ≤ 1e-6` (or a small multiple of `tol_mcscf_grad`), comfortably above the
    observed 3.45e-10 but ~1000× below the current 1e-3. At `sa_g ≈ 1e-10` a
    1e-16 ERI perturbation cannot move `sa_g` across 1e-6, so the exit decision is
    deterministic w.r.t. integral rounding. (This does not, and cannot, make the
    *energy* bitwise across orders — see "What B-2 cannot do" — it makes the
    *gate* insensitive to the ~1e-16 reorder, which is what's needed.)

    Three sub-steps:

    - **B-2.5a — tighten the plateau-exit `sa_g` bound (the load-bearing fix).
      LANDED.** The `reported_gnorm < 100·tol_mcscf_grad` term at the plateau exit
      is replaced with the explicit stationarity bound
      `reported_gnorm <= max(1e-6, tol_mcscf_grad)` (default tol 1e-5 → bound
      1e-5, ~1e4 above the uphill case's observed 3.45e-10 exit `sa_g` and ~100×
      below the old 1e-3). Sets `converged_via_plateau = true` on this exit only.
      Behavior-preserving on the current per-pair order: all 10 CASSCF regression
      cases green, uphill still lands −74.7877864784. Revert: restore the
      `100·tol` term and drop the flag set.

    - **B-2.5b — `casscf_converged_via_plateau` diagnostic. LANDED.** The
      converged-exit block emits a parseable
      `casscf_converged_via_plateau=true|false` line (driven by the
      `converged_via_plateau` flag — `true` only at the plateau exit). The runner
      parses it as a string metric (`run_regressions.py` METRIC_PATTERNS +
      string-passthrough alongside `point_group`). `metric_eq` asserts `false` for
      the three normal SA-2 cases (they exit through the standard `sa_g < tol`
      gate) and `true` only for `..._sad_guess_uphill` — matching observed output.
      Gate: the four SA-2 cases assert the expected flag; all 10 CASSCF cases
      green. Revert: delete the emit + metric + asserts.

    - **B-2.5c — prove rounding-robustness with a throwaway reorder probe. DONE
      (run at B-2d time; probe reverted, uncommitted).** A `#ifdef
      RYS_REORDER_PROBE` block in `_rys_contracted_eri` applied a ~1e-15 *relative*
      perturbation `eri *= 1 + sign·1e-15` (sign from the value's low mantissa bit,
      so it is a reorder-like jitter, not a bias). Built `hartree-fock` with
      `-DRYS_REORDER_PROBE` and ran the uphill SA-2 case (`engine rys`, so it hits
      `_rys_contracted_eri`): the perturbed integrals followed the *same* descent
      (`sa_g` 1.6e-2 → … → 4.2e-6 → 1.35e-9 → **3.54e-10**) to the **same basin**,
      **landing −74.7877864784** and reporting `casscf_converged_via_plateau=true`.
      The plateau exit fired at `sa_g=3.54e-10`, ~3000× below the 1e-6 bound — so
      the gate is empirically robust to the reorder, not just robust by argument.
      The probe was reverted before committing B-2d; it ships nothing. This is the
      robustness claim B-3/B-4 rely on, now measured.

    Net B-2.5: **a+b LANDED** — the plateau exit now fires at a genuinely
    stationary `sa_g` (≤ max(1e-6, tol)), and which exit each SA-2 case takes is a
    regression-checked fact. B-2.5c (the throwaway reorder probe) is intentionally
    deferred to whenever B-2b actually starts: it is a local, reverted-before-
    commit experiment with no committed artifact, and the committed
    `sa_g≈3.4e-10` exit margin already makes the gate's rounding-robustness
    self-evident. So B-3/B-4 are no longer blocked by the CASSCF side — only by
    the B-2b→d integral hoist still being unbuilt.

  - **B-3 — wire into the Auto path (the actual win). LANDED.** In
    `_compute_2e_auto`, the per-quartet engine choice (`_auto_engine(L_AB, L_CD)`)
    now distinguishes HGP / Rys / OS. Both HGP- and Rys-chosen quartets route
    through a single lazy hoisted block (`ensure_hoist_block`, filled on the first
    surviving component): HGP via `HeadGordonPople::_contracted_eri_block_hoisted_views`
    (A4-3, unchanged), Rys via the new `RysQuad::_contracted_eri_block_hoisted_views`
    (B-2d). OS-chosen quartets stay on the per-component `_auto_contracted_eri`
    path (OS-A4 hoists inside `_contracted_eri`, not at this seam). Per-component
    Schwarz/orbit/scatter and the `(k,l) >=_lex (i,j)` ordering are unchanged; the
    block is store-only scatter, so the tensor is independent of visitation order.
    Single change point covers tensor + RHF/UHF Fock Auto builds (the Fock builds
    delegate to `_compute_2e_auto`). **Adopts the hoisted (reordered) order on the
    Auto path**, so it is gated at the 1e-13 ERI bar, not bitwise.

    **Why the basin-robustness gate matters here.** `engine auto` dispatches
    through `base.h` → `_compute_2e_auto` (the function B-3 changes), and the
    three-way `kAutoEngine` table routes the `(7,8)`/`(8,8)` buckets to Rys. Any
    quartet reaching those needs g functions (`L_AB ≥ 7` ⇒ g+g; first at
    cc-pVQZ). So an `engine auto` SA-CASSCF on a g-basis runs the hoisted Rys path
    on a basin-sensitive optimizer. The existing SA-2 gates use explicit
    `engine rys`, so they would *not* catch a B-3 Auto-path regression — a test
    gap. B-3 closes it with a new **`engine auto` SA-CASSCF gate on cc-pVQZ**:
    `water_casscf_sa2_ccpvqz_engine_{os,auto}` (water CAS(4,4) SA-2), where the
    auto case asserts (`metric_close_case`, atol 1e-8) that it lands the **same
    SA-2 basin** as the os twin. This directly tests that the Auto-Rys hoist's
    reorder does not flip the basin, with no external reference needed (and is
    consistent with B-2.5c's measured robustness). Other gates unchanged:
    `ne_rhf_ccpvqz_highL_engine_os_vs_rys` (now exercises the Auto-Rys hoist:
    OS==RYS==AUTO SCF energy on g-shells), `ne_rhf_ccpvqz_highL_pyscf`,
    `planck-compute-2e` Rys-Auto (≤1e-12). Revert: restore the single
    HGP-only `quartet_uses_hgp` block + per-component Rys call.

  - **B-4 (optional) — explicit `engine rys` (`_compute_2e`). Blocked by B-2.5.**
    Same wiring in the non-Auto Rys tensor/Fock builds. Lower value (Auto is the
    default; explicit `engine rys` is rare), so defer unless explicitly wanted.
    The *existing* SA-2 gate runs `engine rys` → `_compute_2e`, so B-4 is the step
    that trips that specific committed gate (B-3 trips only a new Auto-g-basis
    case). Both adopt the same hoisted order and both need B-2.5. Gate:
    `planck-hgp-engine-smoke`-style Rys self-consistency on g shells + the
    now-robust SA-2 uphill case under `engine rys`.

  B-1 established the readout property B-2/B-3 rely on: the max-box build equals
  the per-component build at every component coordinate to ≤4e-16 (bitwise where
  n_max==n_comp, last-bit under Gauss over-integration otherwise). So B-2/B-3 are
  gated at the `1e-13` ERI bar against the per-component path — like the HGP/OS
  hoist, not stricter (the earlier "exact 0 on g shells" expectation was wrong:
  Rys carries no per-component *norm* to reorder, but the differing *root count*
  reorders the sum instead). `RysScratch` growing the max-box variant is the
  natural place to fold in the shared `SpatialQuartetLayout` (below).
- **Phase C — symmetry direct-SCF skeletons** (`os_symm`/`hgp_symm`/`rys_symm`):
  still per-AO. Convert to shell-quartet carrying the signed-AO orbit per
  component (same pattern as 2b's `sym_ops` branch). Needs the block kernels.
- **Phase D — gradient derivative ERIs** (`compute_eri_deriv_dispatch` + loop):
  still per-AO. Gradients under Auto fall back to OS by design
  (`compute_eri_deriv_dispatch` only recognizes explicit `HeadGordonPople`); Rys
  has no derivative kernel, so true per-quartet Auto gradients need one written
  first. The OS fallback was a deliberate decision. OS-A4 covered energy/Fock
  only; `_compute_eri_deriv_elem` is unchanged.
- **Phase E — retire the adapter.** Change `build_shellpairs` to emit shell-level
  `ShellPair` objects and drop `expand_shell_groups_to_ao_pairs`. This is the only
  step that changes the shared `ShellPair` contract; it must come last and resolve
  the per-component `_component_norm` folding (invariant 2).
- **Shared `SpatialQuartetLayout`** (vault Open Work): OS's `EriScratch`, HGP's
  `EriScratch`/`g_hgp_scratch`, and Rys's `RysScratch` carry near-duplicate
  per-quartet scratch. Extract the shared 6-axis layout; the
  triangular-vs-rectangular distinction from §4 should inform the interface.

---

## §7 — how to verify the current state

```
# from build/
./planck-compute-2e               # golden checksum + 8-fold symmetry + Rys/Auto-vs-OS
./planck-os-block-kernel          # OS/HGP block == per-component (exact); hoisted HGP block ≤1e-13 rel
./planck-hgp-engine-smoke         # OS <-> HGP to 1e-12
./planck-hgp-triangular-contract  # max-AM (a0|c0) == per-component, bitwise (d-shells)
# from repo root
python3 tests/run_regressions.py --suite all   # all green
# engine equality incl. OS-A4 and three-way Auto (OS == HGP == Rys == Auto):
#   water/sto-3g, water/6-31g*, water RKS+UKS HSE06, He2/cc-pV5Z (reaches OS corner + Rys tail)
python3 tests/engine_scf_energy_compare.py <input>
# high-L Rys guard (g shells -> (7,8)/(8,8) Rys buckets), extended-tagged:
python3 tests/run_regressions.py --suite all --case ne_rhf_ccpvqz_highL_pyscf
python3 tests/run_regressions.py --suite all --case ne_rhf_ccpvqz_highL_engine_os_vs_rys
tests/pyscf/.venv/bin/python tests/pyscf/ne_rhf_ccpvqz_highL.py   # reproduce PySCF ref
# Auto dispatch re-fit (0 median disagreements):
python3 scripts/fit_auto_dispatch.py
```
