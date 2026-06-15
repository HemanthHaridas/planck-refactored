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

### 2. `_component_norm` is folded per component before contraction — a hoisted
### contraction must be norm-free and apply the norm at readout

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
    — validated only against `_rys_contracted_eri`. Four sub-steps, each gated
    and revertible:

    - **B-2a — contract-over-primitives at a fixed box.** Add a static
      `_rys_contract_sum(spAB, spCD, box, n_roots, kernel, omega, RysScratch&)`
      that loops primitive pairs calling `_rys_eri_prep` + `_rys_eri_build_sum`
      and **accumulates** each pair's 6D `sum` (weighted by
      `coeff_product`) into one buffer. This is the Rys analog of HGP's
      `hgp_contract_a0c0`. Refactor `_rys_contracted_eri` to call it (per-
      component box, n_comp) then `_rys_eri_hrr_to_eri` — bitwise no-op, since it
      is the same per-pair build + HRR, only the accumulation is factored out.
      Gate: `planck-compute-2e`, engine equality on sto-3g/6-31g\*/Ne; bitwise.
      Revert: inline the loop back.

    - **B-2b — norm-free contraction + per-component norm.** Like HGP (invariant
      2), the single shared contraction can't carry per-component norms. Add a
      Rys `normfree_view` (sets `_component_norm = 1`), contract once from the
      component-0 norm-free views, and apply `normA·normB·normC·normD` at the
      readout. Validate via a `_rys_contract_sum`-from-normfree + per-component
      HRR + norm against `_rys_contracted_eri` for one quartet — tight tolerance
      (B-1: ≤4e-16, the root-count drift), not bitwise. No production wiring.
      Revert: delete the normfree path.

    - **B-2c — snapshot layout + per-component readout.** Add a small max-box
      stride helper (the 6-axis `idx` from `tests/rys_box_invariance.cpp`,
      promoted to a `RysMaxBoxLayout`) so a component's sub-box can be gathered
      from the n_max snapshot into a component-sized `RysScratch` and HRR'd,
      independent of the shared scratch's later resizes. Rys analog of HGP's
      `MaxBoxLayout` + `hgp_hoist_readout_component`. Gate: a unit-test arm
      gathering+HRR a known component matches `_rys_eri_hrr_to_eri` on a fresh
      per-component build. Revert: delete helper + test arm.

    - **B-2d — assemble `RysHoistedQuartet` + standalone block entry.** Compose
      B-2a–c into the struct (ctor takes the four component-0 views + kernel/
      omega; lazy `prepare()` contracts both kernel snapshots for ShortRange —
      Coulomb and LongRange — into `a0c0_primary`/`a0c0_secondary`;
      `readout(component, norm)` HRRs each and combines). Add
      `_rys_contracted_eri_block_hoisted_views` (pointer-array form, so B-3's
      non-contiguous `ao_views` can feed it). Gate: extend
      `planck-rys-box-invariance` (or a sibling) to compare the struct's
      per-component readouts against `_rys_contracted_eri` for every component of
      water/6-31g\* and Ne/cc-pVQZ Lq≥7, Coulomb/LongRange/ShortRange, at the
      1e-13 bar. Revert: delete struct + entry + test arm.

    Net B-2: the hoisted path exists and is proven equal to the per-component
    path, but nothing in production calls it yet (that is B-3). ShortRange = two
    snapshots combined per component, as in OS/HGP.

  - **B-3 — wire into the Auto path (the actual win).** In `_compute_2e_auto`'s
    Rys-chosen branch, build a `RysHoistedQuartet` once per `(gA,gB,gC,gD)` shell
    quartet, lazily on the first surviving Rys component (same lazy-`prepare`
    discipline as A4-2's `HoistedQuartet`, so screened quartets pay nothing), and
    have surviving components read out of it instead of calling
    `_auto_contracted_eri` per component. Per-component Schwarz/orbit/scatter
    unchanged. Gate: `ne_rhf_ccpvqz_highL_engine_os_vs_rys` (0.0e+00),
    `ne_rhf_ccpvqz_highL_pyscf`, `planck-compute-2e` Rys-Auto, smoke/core/extended.
    Revert: restore the per-component `_auto_contracted_eri` call.

  - **B-4 (optional) — explicit `engine rys` (`_compute_2e`).** Same wiring in the
    non-Auto Rys tensor/Fock builds. Lower value (Auto is the default; explicit
    `engine rys` is rare), so defer unless explicitly wanted. Gate:
    `planck-hgp-engine-smoke`-style Rys self-consistency on g shells.

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
