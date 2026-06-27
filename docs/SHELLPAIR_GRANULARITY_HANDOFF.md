# Shell-pair Granularity (H-10) Architecture Note

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

Remaining and deferred work for this effort lives in:

- `docs/SHELLPAIR_GRANULARITY_OPEN_WORK.md`

This file answers a narrower architecture question:

**How does ERI work iterate at shell-quartet granularity, what amortization does
that buy, and what invariants must future integral changes preserve?**

## Short answer

The integral engines historically iterated ERI work at per-Cartesian-AO-component
granularity, so a `(dd|dd)` quartet re-ran the shared per-primitive seed work
(Gaussian product center, prefactor, Boys, VRR) `6⁴ = 1296` times instead of once.
H-10 moves iteration to per-shell-quartet granularity and amortizes the shared
per-primitive contraction across a quartet's Cartesian components. The win is real,
but only because a set of coupled invariants — visitation order, norm placement,
box-size invariance, screening deferral, and data-derived dispatch — are kept
consistent with each other. Break one in isolation and the result is a silent
last-bit drift, a NaN, or a changed screened set.

## Where the logic lives

- `src/integrals/shellpair.{h,cpp}` — `ShellGroup`, `build_shell_groups`,
  `expand_shell_groups_to_ao_pairs`; `build_shellpairs` routes through a no-op
  per-AO adapter, so any engine can still be reverted to per-AO.
- `src/integrals/hgp.{h,cpp}` — `hgp_contract_a0c0`, `hgp_hrr_finalize`,
  `HoistedQuartet` / `MaxBoxLayout`, `_contracted_eri_block_hoisted{,_views}`.
- `src/integrals/os.{h,cpp}` — `_os_eri_build_a0c0`, `_os_eri_hrr_to_eri`,
  `_os_contract_a0c0`, `_contracted_eri_base`.
- `src/integrals/rys.cpp` — `_compute_2e_auto` (default Auto tensor + Fock builds),
  the three-way `kAutoEngine[L_AB][L_CD]` table + `_auto_engine` selector, the Rys
  hoist (`RysHoistedQuartet`, `_contracted_eri_block_hoisted{,_views}`).
- `tests/auto_dispatch_benchmark.cpp` + `scripts/fit_auto_dispatch.py` +
  `docs/auto_dispatch_fit.json` — the per-bucket calibration, the fitter, and the
  generated table pasted verbatim into `kAutoEngine`.
- `tests/rys_box_invariance.cpp`, `planck-os-block-kernel`,
  `planck-hgp-engine-smoke`, `planck-compute-2e` — the equivalence gates.

## The transform model

For one shell quartet the engines compute the contracted half-transformed block
`(a0|c0)` (bra and ket each reduced to a single center), then transfer angular
momentum with two HRR passes:

- **Phase 1 (per primitive pair, contracted):** run VRR per primitive pair and
  accumulate the `(a0|c0; m=0)` slice into a quartet-level accumulator scaled by
  the primitive coefficient product.
- **Phase 2 (once per shell quartet):** copy the accumulator into the HRR buffer
  and run A→B then C→D HRR.

Screened kernels exploit HRR linearity in the `(a0|c0)` block: Coulomb and
LongRange each run a single hoisted pass; `ShortRange = Coulomb − LongRange` runs
two and subtracts.

## What invariants matter

### 1. Quartet visitation order is free only because the scatter is store-only

The per-AO `build_shellpairs` list is the row-major AO upper triangle, so its flat
pair index increases lexically in `(i,j)`. The shell-quartet rewrites visit **every**
ket shell pair (not pruned by flat index — the lex order of component AO indices is
not monotonic in the shell-pair index) and filter per component with the
`(k,l) >=_lex (i,j)` check. The 8-fold scatter is store-only, so the tensor is
independent of visitation order and bitwise-identical to the per-AO build.

Design rule: keep the scatter store-only and keep the per-component lex filter. Do
not reconstruct `(ii,jj)` from a flat index — use `sp.A._index` / `sp.B._index`.

### 2. A hoisted contraction must be norm-free and apply the norm at readout

The per-AO path folds each component's `_component_norm` into its primitive
`coeff_product` *before* contraction. A single shared contraction cannot carry
per-component norms: it must contract norm-free (component-0 views, `_component_norm`
forced to 1) and multiply each readout by `normA·normB·normC·normD` *after* HRR.
This makes the hoisted path **not bitwise** for d-shell components (`Σ(wᵢ·n)·vᵢ`
vs `(Σwᵢ·vᵢ)·n` round differently at the last bit); the cross-validation bar is
relative `1e-13`. sto-3g (all norms 1) stays bitwise.

Design rule: a new hoisted/shared contraction contracts norm-free and applies the
four norms at readout, and is gated at `≤1e-13` against the per-component path on a
d-shell basis (6-31G\*).

### 3. The contraction is box-size invariant; the HRR readout is not

`hgp_vrr` is strictly bottom-up over a dense rectangular cube, so contracting at the
max-AM box gives, at every component's `(a0|c0)` sub-block, **bitwise** the same
value as a contraction sized to that component. But the readout must HRR **only that
component's sub-box**, never the full max-AM cube — sweeping the unreachable diagonal
corners is what produced NaN in the first hoist attempt (see "What was fixed").

Design rule: contract at the max box, HRR each component's sub-box only.

### 4. Per-quartet screening is preserved by deferring the contraction

Schwarz screening and the symmetry orbit-front check are per component. The hoist
builds the shared contraction lazily, on the first component that survives both
screens (`HoistedQuartet::prepare()` on first `readout()`), so a fully screened
quartet costs nothing. A shell-quartet-level Schwarz *prescreen* would change the
screened set and is **not** equivalence-preserving.

Design rule: defer the contraction behind per-component screening; do not introduce
a quartet-level prescreen.

### 5. Auto dispatch is derived from data, not hand-coded inequalities

`kAutoEngine[L_AB][L_CD]` is a dense constexpr table copied verbatim from the
fitter's output in `docs/auto_dispatch_fit.json`, assigning each bucket to the engine
with the lowest cross-case median per-quartet time. The OS/HGP/Rys region boundaries
are irregular and move when an engine is optimized.

Design rule: when an engine's per-quartet cost changes, re-run
`planck-auto-dispatch-benchmark`, re-run `scripts/fit_auto_dispatch.py`, and paste
the regenerated table. Do not hand-edit `kAutoEngine`.

## What is live now

- **OS, HGP, and the default Auto path** iterate ERI work at shell-quartet
  granularity; the shared phase-1 is built once per quartet, not per component.
- **HGP and OS** amortize the per-primitive VRR + `(a0|c0)` contraction across a
  quartet's components, then run the two HRR passes once per quartet (HGP on
  `_compute_2e` + its Fock builds + the Auto HGP-chosen quartets; OS in
  `_contracted_eri`).
- **Rys** has the same hoist (`RysHoistedQuartet`), wired into the Auto path's
  Rys-chosen `(7,8)`/`(8,8)` g-shell buckets. Both HGP- and Rys-chosen Auto quartets
  read from one lazy hoisted block; OS-chosen quartets stay per-component (OS hoists
  inside `_contracted_eri`, not at the Auto seam).
- **Auto dispatch is three-way (OS / HGP / Rys)**, data-derived per `(L_AB, L_CD)`
  bucket.

## What was fixed (the design blocker that shaped the hoist)

The first hoist attempt built HGP's `(a0|c0)` block once at max AM and had each
component HRR its readout **from the full max-AM cube**. `planck-os-block-kernel`
caught it: bitwise on STO-3G (s,p only), **NaN on 6-31G\*** (d-shells).

Root cause: `EriScratch` is a dense per-axis rectangular cube, and building "max AM"
as `lABx = lABy = lABz = L_A+L_B` makes `hgp_vrr` fill the diagonal corner
`(L_A+L_B, L_A+L_B, L_A+L_B, …)` that **no single component ever needs** (a
component's total bra AM is distributed *across* the three axes). The HRR readout
swept those unreachable corners — that is where the NaN came from, not the
contraction.

Resolution (invariant 3): the contraction at the max box is bitwise box-size
invariant, so contract once at the max box with the unmodified dense
`hgp_contract_a0c0`, then HRR each component's **sub-box** only. No triangular VRR
rework was needed.

## Validation strategy that should remain in place

- `planck-compute-2e` — golden checksum + 8-fold permutational symmetry +
  Rys/Auto-vs-OS cross-check.
- `planck-os-block-kernel` — OS/HGP block == per-component (exact); hoisted HGP
  block ≤1e-13 rel.
- `planck-hgp-engine-smoke` — OS ↔ HGP to 1e-12.
- `planck-rys-box-invariance` — Rys max-box build == per-component, plus the Rys
  hoist block ≤1e-13 (water/6-31g\*, Ne/cc-pVQZ Lq≥7).
- `tests/engine_scf_energy_compare.py` — OS == HGP == Rys == Auto end-to-end SCF
  energy.
- `ne_rhf_ccpvqz_highL_*` — high-L g-shell guard (the Rys `(7,8)`/`(8,8)` buckets).

## Related but separate outcome: CASSCF inactive-Fock speedup

Profiling the (unrunnable) g-basis CASSCF gate surfaced that MCSCF rebuilds the
inactive/active Fock by contracting the full materialized AO ERI tensor per
candidate orbital step. The inactive Fock was moved to a direct (tensor-free)
shell-pair build in Cartesian mode (Phase A0/A1). This is a CASSCF-engine speedup,
not a shell-pair-granularity change, and its design + the dropped active-Fock
attempt (A2) live in the open-work document.

## Remaining architecture concern

The Rys hoist is wired into the Auto path but not into explicit `engine rys`, and
several downstream subsystems (symmetry direct-SCF skeletons, gradient derivative
ERIs) are still per-AO. Those, plus the per-AO adapter retirement and the CASSCF
speedups, are tracked in `docs/SHELLPAIR_GRANULARITY_OPEN_WORK.md`, not here.
