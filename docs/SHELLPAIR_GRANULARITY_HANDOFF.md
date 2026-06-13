# Shell-pair granularity (H-10) — handoff

Status as of this note: **steps 1, 2a–2c, A1–A3, A4-0 landed and gated; A4-1
attempted and reverted (design blocker, see §A4); A4-pre and A4-1′ landed and
gated (see §5).** Branch `perf/shellpair-granularity`.

This note is the source of truth for the H-10 refactor ("ERI shell pairs are
built per Cartesian AO, not per shell"). It records what landed, the A4
blocker found empirically, and the re-scoped path for the remaining work.

---

## 1. What H-10 is

The integral engines historically iterated ERI work at **per-Cartesian-AO-
component** granularity: `build_shellpairs` produced one `ShellPair` per ordered
pair of AO components, and each engine computed a 1×1 contribution per component
quartet. A single d-shell quartet `(dd|dd)` therefore re-ran the shared per-
primitive seed work (Gaussian product center, prefactor, Boys, VRR) `6⁴ = 1296`
times instead of once.

H-10 moves iteration to **per-shell-quartet** granularity, so the shared per-
primitive work is amortized across a shell quartet's Cartesian components. The
program was sequenced so every sub-step is independently bitwise-gated and
reversible; the shared per-AO `ShellPair` list (via the no-op adapter from
step 1) stays in the tree throughout, so any engine can be reverted to per-AO.

---

## 2. Landed work (gated, on branch `perf/shellpair-granularity`)

| Step | What | Engine / file | Gate |
|---|---|---|---|
| 1 | Shell-granular layer: `ShellGroup`, `build_shell_groups`, `expand_shell_groups_to_ao_pairs`; `build_shellpairs` routes through it (no-op adapter, re-expands to the identical per-AO list) | `src/integrals/shellpair.{h,cpp}` | `planck-compute-2e` bitwise |
| 2a | OS shell-quartet block kernel `_contracted_eri_block` (loops components, delegates to per-component `_contracted_eri_elem`) | `src/integrals/os.{cpp,h}` | `planck-os-block-kernel` (`max\|diff\|=0`) |
| 2b | OS `_compute_2e` iterates shell-quartets; per-component Schwarz/orbit/scatter nested inside; `(k,l) >=_lex (i,j)` reproduces old flat-pair `q>=p` | `src/integrals/os.cpp` | `planck-compute-2e`, symmetry case |
| 2c | OS extract shared phase-1 `build_eri_tensor_shellwise`; `_compute_2e` + both `_compute_2e_fock{,_uhf}` route through it | `src/integrals/os.cpp` | 5× `*_scf_energy_engine_os_vs_hgp` |
| — | Default engine changed OS → **Auto** | `src/base/types.h` | smoke/core/extended |
| A1 | HGP block kernel `_contracted_eri_block` (same shape as 2a) | `src/integrals/hgp.{cpp,h}` | `planck-os-block-kernel` (OS+HGP) |
| A2 | HGP `_compute_2e` iterates shell-quartets (same conversion as 2b). HGP's own `_compute_2e_fock{,_uhf}` already delegate phase-1 to `_compute_2e`, so they came along for free | `src/integrals/hgp.cpp` | `planck-hgp-engine-smoke`, cross-engine |
| A3 | `RysQuad::_compute_2e_auto` (the **default** Auto path's tensor + Fock builds) iterates shell-quartets; per-quartet Rys/HGP dispatch unchanged | `src/integrals/rys.cpp` | `planck-compute-2e` Rys-Auto, suites |
| A4-0 | Extract HGP's per-primitive VRR-contract accumulation into shared `hgp_contract_a0c0`; per-component path routes through it (bitwise no-op) | `src/integrals/hgp.cpp` | `planck-os-block-kernel`, smoke/core |

**Net structural result:** OS, HGP, and the default Auto path all iterate ERI
work at shell-quartet granularity. **No speedup yet** — every path still calls
the per-component kernel once per component (the per-primitive seed work is *not*
yet amortized). A4 is the step that converts the iteration shape into a
performance win, and it is **not done** (see §A4).

Commits (top of branch first):
```
A4-0 refactor              (hgp_contract_a0c0 extraction)   — uncommitted at time of note
A3  shell-quartet Auto path  (rys.cpp _compute_2e_auto)
A2/A1 HGP block + _compute_2e
default engine -> Auto
1+2a-2c OS shell-quartet
RI-MP2 single-point gate (H-15, unrelated)
```

---

## 3. The two non-obvious invariants every step preserved

1. **`(k,l) >=_lex (i,j)` ⟺ old `q >= p`.** The per-AO `build_shellpairs` list is
   the row-major AO upper triangle, so its flat pair index increases lexically in
   `(i,j)`. The shell-quartet rewrites visit every ket shell pair (`ket = 0..ngp`,
   **not** pruned by flat index — the lex order of component AO indices is not
   monotonic in the shell-pair index; pruning ket by index drops valid quartets,
   which is the bug that first failed 2b's gate) and filter per component with the
   lex check. This reproduces the canonical quartet set exactly. Store-only
   scatter makes the tensor independent of visitation order, so the result is
   bitwise-identical.

2. **`_component_norm` is folded into `ShellPair::primitive_pairs` per component.**
   The block kernels (2a/A1) construct a per-component `ShellPair` from the real
   `ContractedView`, so the norm is identical to `build_shellpairs`. Any hoisted
   path that builds **one** contraction per shell quartet cannot carry per-
   component norms in the shared contraction — it must build norm-free and apply
   `normA·normB·normC·normD` at the per-component readout. (This was handled in
   the reverted A4-1; keep it in mind for the re-scope.)

---

## 4. A4 — the blocker, and the re-scope {#A4}

### What A4-1 tried
Build HGP's contracted `(a0|c0)` block **once per shell quartet at max AM**
(`L_A+L_B` per bra axis, `L_C+L_D` per ket axis), preserve it, then have each
Cartesian component do only its HRR readout from that preserved block. This is
the standard Head-Gordon-Pople amortization and would make the now-default
Auto/HGP path actually faster.

### What the gate caught
A new `planck-os-block-kernel` check compared the hoisted block against the A1
block (itself gated `==` per-component `_contracted_eri_elem`). Result:

- **STO-3G (s,p only): bitwise pass.**
- **6-31g\* (has d-shells): NaN.** `~3000 quartets mismatched, max |diff| = 0`
  — i.e. NaN (NaN ≠ NaN registers as mismatch, but `abs(NaN − x)` never exceeds
  the recorded max), **not** a finite value error.

### Root cause: the max-AM region is a rectangular cube, not the triangle the
### recurrence actually needs
`EriScratch` is a **dense per-axis rectangular cube** indexed by `ax_dim ×
ay_dim × az_dim × cx_dim × cy_dim × cz_dim`, and `hgp_vrr` fills it with per-axis
rectangular loops (`for ax in 0..lABx`, independent per axis). Building "max AM"
as `lABx = lABy = lABz = L_A+L_B` makes the VRR compute the full cube, including
the diagonal corner `(L_A+L_B, L_A+L_B, L_A+L_B, …)` that **no single Cartesian
component ever needs** (a component's total bra AM is `L_A+L_B` distributed
*across* the three axes, never `L_A+L_B` on *each*). The dense rectangular sweep
over those unreachable high-total-AM cells is what produces the garbage/NaN for
d-shells; for s/p the cube and the triangle coincide, so it passed.

> Note on Boys: this is **not** a Boys-table overflow. `boys(n,x)` supports
> `n < TABLE_COLS = 66` (`src/lookup/boys.cpp`), and the cube's `MMAX = 24` for
> (dd|dd) is well within range. The exact uninitialized/garbage cell was not
> pinned before the revert; it does not need to be, because the re-scope removes
> the dense-cube build entirely. If a future debugger wants it, instrument
> `hgp_vrr` / the gather in the reverted A4-1 patch and find the first NaN cell —
> but the structural fix below makes it moot.

### Why this is bigger than the original A4 scope
The original A4 assumed "build the contraction once, read per component" was a
pure call-reorganization on top of the existing `hgp_vrr`. It is not: the
existing VRR/`EriScratch` are **dense rectangular**, and a correct max-AM build
needs a **triangular** region (`ax+ay+az ≤ L_A+L_B`, `cx+cy+cz ≤ L_C+L_D`). That
is a rework of `hgp_vrr` and `EriScratch`'s indexing — both shared with the
validated per-component path — which is materially larger and riskier than the
reorganization A4 was scoped as.

---

## 5. Re-scoped A4 (small, verifiable, reversible)

Prerequisite **A4-pre** is new; A4-1′/A4-2/A4-3 are the original A4-1/A4-2/A4-3
rebased onto it.

**A4-pre — box-size invariance of the (a0|c0) contraction, validated bitwise.
LANDED (commit on branch).**

The §4 conclusion that A4 needs a *triangular* VRR/scratch rework turned out to
be stronger than necessary for the contraction itself. The A4-pre gate
established empirically that the existing dense-rectangular `hgp_contract_a0c0`
is **already box-size invariant**: contracting the `(a0|c0)` block once at the
max AM box (`lAB = L_A+L_B`, `lCD = L_C+L_D` per axis) gives, at every Cartesian
component's `(a0|c0)` sub-block, **bitwise** the same value as a contraction
sized to exactly that component's AM. This holds because `hgp_vrr` is strictly
bottom-up — a larger box only *adds* higher-AM cells (the unreachable cube
corners), and those extra cells are never read by any lower coordinate, so they
cannot corrupt the cells a real component needs. The dense-cube corners being
garbage/zero is therefore harmless *for the contraction*; the §4 NaN came from
A4-1's **HRR readout** sweeping those corner cells, not from the contraction.

So A4-pre did **not** need a triangular variant. What landed:
- a test-only hook `HeadGordonPople::_contract_a0c0_at_native_test(spAB, spCD,
  lABx..lCDz, ax..cz, kernel, omega)` (`src/integrals/hgp.{cpp,h}`) that runs
  `hgp_contract_a0c0` at a caller-given AM box and returns the accumulated
  `(a0|c0)` value at a caller-given logical coordinate. ShortRange returns
  `Coulomb − LongRange`, matching production. No production code path changed.
- `tests/hgp_triangular_contract.cpp` → `planck-hgp-triangular-contract`. For
  every shell quartet of water it holds the ShellPair fixed at the component-0
  views (so the folded `_component_norm` is constant and cannot confound the
  comparison — see §3.2), then asserts the max-AM build equals the
  per-component-AM build, **bitwise**, at every coordinate inside each
  component's box. Green on 6-31g\* (d-shells; the (dd|dd) quartets that NaN'd
  A4-1) for Coulomb / LongRange / ShortRange (1,990,921 coords each) and sto-3g
  (9,409 coords). Revert = delete the hook + test + CMake hunk.

**Implication for A4-1′:** the hoisted block can contract once at the max AM box
using the *unmodified* `hgp_contract_a0c0`, then read each component's
`(a0|c0)` sub-block out of `a0c0_data` via `spatial_index(...)` and HRR **only
that component's box** (`0..lABx_comp`, `0..lCDx_comp`). The triangular VRR
rework from the original §4/§5 is **not** required, because the readout never
needs to touch the cube corners — it HRRs a per-component sub-box, not the full
max-AM cube (that was A4-1's mistake). A norm-free contraction + per-component
`normA·normB·normC·normD` at readout (§3.2) is still required, since the single
shared contraction cannot carry per-component norms.

**A4-1′ — hoisted block. LANDED (commit on branch).**
`HeadGordonPople::_contracted_eri_block_hoisted` (`src/integrals/hgp.{cpp,h}`):
contracts the `(a0|c0)` block **once** per shell quartet at the max AM box using
the *unmodified* dense `hgp_contract_a0c0` (no triangular rework needed — A4-pre
showed the dense build is already box-size invariant), snapshots the
accumulator, then for each Cartesian component gathers its `(a0|c0)` sub-box into
a second thread-local scratch (`g_hgp_hoist_comp_scratch`) and HRRs **only that
component's box** (`hgp_hoist_readout_component`). The HRR never touches the
unreachable max-AM cube corners that NaN'd the original A4-1 — that was A4-1's
actual bug, not the contraction. ShortRange is the `Coulomb − LongRange` split
(two snapshots, combined per component). Output layout matches
`_contracted_eri_block` (`[a][b][c][d]`, d fastest).

The contraction is **norm-free** (component-0 views with `_component_norm`
forced to 1; helper `normfree_view`) and each readout is multiplied by
`normA·normB·normC·normD` — §3.2's required factoring, since one shared
contraction serves components with different norms.

Gate: `planck-os-block-kernel` extended with a hoisted-vs-per-component check on
6-31g\* (Coulomb/LongRange/ShortRange) + sto-3g — the exact gate that caught
A4-1.

> **Bitwise → tight-tolerance, deliberately.** The original A4-1′ spec said
> *bitwise*. That is **not achievable for a correct norm-free hoist** and the
> spec was wrong on this point: the per-component path folds `_component_norm`
> into each primitive's `coeff_product` *before* contraction, while the hoist
> applies the norm *after* HRR. `Σ(wᵢ·n)·vᵢ` vs `(Σwᵢ·vᵢ)·n` are mathematically
> equal but round differently, so d-shell components (norm ≠ 1) drift at the
> last FP bit. The contraction *itself* is bitwise box-invariant (A4-pre), and
> sto-3g (all norms = 1) is bitwise here too; only the post-HRR norm multiply
> introduces the drift. The gate therefore checks the hoisted path at relative
> tolerance **1e-13** (the standard ERI cross-validation bar; cf.
> `planck-compute-2e` ~1e-12). Observed worst case on 6-31g\*: **~9.6e-16**
> (Coulomb 7.4e-16, LongRange 4.4e-16, ShortRange 9.6e-16). The per-component
> OS/HGP blocks still gate at exact 0. Revert = delete routine + test hunk.

**A4-2 — wire into HGP `_compute_2e`.** Replace the per-component
`_contracted_eri_elem` call in the A2 inner loops with one
`_contracted_eri_block_hoisted` per shell quartet; run the existing
Schwarz/orbit/scatter per component reading `block[component]`. Gate:
`planck-compute-2e` — note this becomes a **tight-tolerance** cross-check, not
golden-exact, for the same norm-scaling reason as A4-1′ (the golden checksum
asserts exact equality today; once HGP `_compute_2e` routes through the hoist it
will differ at ~1e-15, so the HGP arm of the comparison needs a ~1e-12 tol while
OS/Rys stay exact). Plus 5× `*_scf_energy_engine_os_vs_hgp` (already ≤5e-9 Eh,
unaffected) and smoke/core/extended. Revert = restore per-component call.

**A4-3 — wire into the Auto path.** Route HGP-chosen quartets in
`_compute_2e_auto` through the hoisted block (Rys-chosen quartets stay per-
component; Rys hoisting is Phase B, out of A4). This delivers the win on the
**default** engine. Gate: `planck-compute-2e` Rys-Auto, suites. Revert = restore
per-component `_auto_contracted_eri`.

### Open screening subtlety (applies to A4-2/A4-3, same as OS-2d would have had)
Schwarz screening and the symmetry orbit-front check are **per component** and
may skip individual components. Computing the full block first wastes work on
skipped components. This is acceptable (correctness-first) and stays bitwise. A
shell-quartet-level Schwarz prescreen (skip the whole block when the max bound is
below tol) is a **screening-policy** change — it alters the screened set, so it
is **not** bitwise and is out of A4's scope.

---

## 6. Deferred / out of scope (the rest of H-10)

These were scoped earlier and remain open; none are blocking A4.

- **OS-2d.** The OS analog of A4 (hoist OS's per-component VRR/HRR). Demoted:
  OS is no longer the default (Auto is), so it only helps explicit `engine os` on
  d-shells. Same triangular-region issue would apply to OS's `EriScratch`.
- **Phase B — Rys shell-quartet hoist.** Rys has no VRR/HRR to hoist (angular
  dependence comes from quadrature roots), so there is no A4-equivalent. Iterating
  Rys at shell-quartet granularity (root reuse across components) is a smaller,
  consistency-level win; low priority because Auto only sends `L_AB+L_CD ≤ 1`
  quartets to Rys (≤3 components).
- **Phase C — symmetry direct-SCF skeletons** (`os_symm`/`hgp_symm`/`rys_symm`):
  still per-AO. Convert to shell-quartet carrying the signed-AO orbit per
  component (same pattern as 2b's `sym_ops` branch). Needs the block kernels from
  A/B.
- **Phase D — gradient derivative ERIs** (`compute_eri_deriv_dispatch` + loop):
  still per-AO. Note: gradients under Auto fall back to OS by design
  (`compute_eri_deriv_dispatch` only recognizes explicit `HeadGordonPople`); Rys
  has **no** derivative kernel, so true per-quartet Auto gradients are not
  possible without writing one. Leaving the OS fallback was a deliberate decision.
- **Phase E — retire the adapter.** Change `build_shellpairs` to emit shell-level
  `ShellPair` objects and drop `expand_shell_groups_to_ao_pairs`. This is the only
  step that changes the shared `ShellPair` contract and must come last, after
  A–D, and after resolving the per-component `_component_norm` folding (see §3.2).
- **Shared `SpatialQuartetLayout`** (already in vault Open Work): OS's
  `EriScratch`, HGP's `EriScratch`/`g_hgp_scratch`, and Rys's `RysScratch` carry
  near-duplicate per-quartet scratch. After A/B make all three shell-quartet-
  native (and A4-pre adds the triangular variant), extract the shared 6-axis
  layout. The triangular-vs-rectangular distinction from §4 should inform that
  interface.

---

## 7. How to verify the current state

```
# from build/
./planck-compute-2e               # golden checksum + 8-fold symmetry + Rys/Auto-vs-OS
./planck-os-block-kernel          # OS/HGP block == per-component (exact); A4-1′
                                  #   hoisted HGP block == per-component (≤1e-13 rel)
./planck-hgp-engine-smoke         # OS <-> HGP to 1e-12
./planck-hgp-triangular-contract  # A4-pre: max-AM (a0|c0) == per-component, bitwise
# from repo root
python3 tests/run_regressions.py --suite smoke     # 35/35
python3 tests/run_regressions.py --suite core      # 64/64
python3 tests/run_regressions.py --suite extended  # 82/82
```

All of the above are green at the landed state in §2.
