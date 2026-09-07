# DFT SOSCF for hybrid and range-separated functionals — scope

In-flight scope. Delete or fold into `docs/SOSCF_DFT.md` when it lands
(per `docs/docs_answer_one_question.md`: a scope doc expires the instant
the work does).

**Question this work answers:** what does it take to lift the
pure-functional-only restriction on DFT SOSCF, so B3LYP / PBE0 / HSE06 /
CAM-B3LYP runs get the same second-order acceleration RKS/UKS pure
functionals already have?

## Short answer

The KS orbital Hessian's `h_op` is composed inline as
`diag_term ⊙ x + J_packed + xc_packed` (`docs/SOSCF_DFT.md`). A hybrid's
KS Fock carries one more term, `+ K_scaled`, and `K` is **linear in the
density** exactly like `J`. So the Hessian gains one more term of the same
shape:

```
h_op(x) = diag_term ⊙ x  +  J_packed  +  xc_packed  +  K_packed
```

`K_packed` is built with `_compute_2e_k_direct` / `_compute_2e_k_uhf_direct`
— **already in the tree, already screened-kernel-aware** — on the trial
`δP`, with the same coefficient/sign decomposition the KS potential build
uses (`src/dft/driver.cpp:3936-4022`). No new integral code, no new CPHF
matrix, no new scale convention.

The only real work is verification and deleting the gate.

## What is already done and not touched

- `solve_augmented_hessian`, `apply_orbital_rotation`, per-spin
  semicanonicalization, `kSoscfMaxRot`, the `scf_soscf_*` window logic —
  functional-agnostic, reused unchanged.
- The RKS 4× / UKS 2× total-energy scale conventions
  (`docs/SOSCF_DFT.md` invariant 2). `K` enters the AO Fock exactly as `J`
  does — both linear in `P`, both in the same `F` the SOSCF gradient and
  Hessian are measured against — so adding a `K` term to `h_op` does not
  change the relationship between `h_op` and `d²E_total/dκ²`. **No new
  PySCF cross-check of the constant is needed**; the existing
  FD-of-total-energy verification covers the assembled callback.
- PCM and SAO stay excluded, with their existing warnings.

## Where the code changes

Two `h_op` lambdas and one gate, all in `src/dft/driver.cpp`:

- RKS `h_op` — `src/dft/driver.cpp:2035`
- UKS `h_op` — `src/dft/driver.cpp:2433`
- `soscf_dft_hybrid_blocked` / `soscf_uks_hybrid_blocked` — `:1873`, `:2255`
- the one-time warning blocks that name the hybrid reason — `:1883`, and
  the UKS equivalent near `:2265`

The K-response coefficients come off the `xc_grid` already in scope at
each SOSCF site (result of `evaluate_current_density_and_xc`,
`:1914` / `:2290`): `xc_grid->full_range_exchange_coefficient`,
`xc_grid->short_range_exchange_coefficient`,
`xc_grid->range_separation_omega` (`src/dft/xc_grid.h:81-83`).

## The K-response term, exactly

The RKS KS potential build adds this to `F` (`src/dft/driver.cpp:3992-4018`):

```
exchange   = c_fr · K_Coulomb[P_alpha]  +  c_sr · K_ShortRange[P_alpha, omega]
F         += -0.5 · exchange                       // closed-shell prefactor
```

so the RKS Hessian term is `δ` of that on the trial density:

```cpp
// in the RKS h_op, after xc_packed, before the return:
const double c_fr = xc_grid->full_range_exchange_coefficient;
const double c_sr = xc_grid->short_range_exchange_coefficient;
Eigen::VectorXd K_packed = Eigen::VectorXd::Zero(x.size());
if (c_fr != 0.0 || c_sr != 0.0)
{
    Eigen::MatrixXd dK = Eigen::MatrixXd::Zero(nbasis, nbasis);
    if (c_fr != 0.0)
        dK.noalias() += c_fr * _compute_2e_k_direct(
            prepared.shell_pairs, dP, calculator._shells.nbasis(),
            calculator._integral._engine, HartreeFock::ERIKernel::Coulomb, 0.0,
            calculator._integral._tol_eri,
            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
    if (c_sr != 0.0)
        dK.noalias() += c_sr * _compute_2e_k_direct(
            prepared.shell_pairs, dP, calculator._shells.nbasis(),
            calculator._integral._engine, HartreeFock::ERIKernel::ShortRange,
            xc_grid->range_separation_omega, calculator._integral._tol_eri,
            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
    K_packed = DFT::Driver::pack_hessian_vector_product_cphf_order(
        -0.5 * dK, C_occ_prev, C_virt_prev);
}
return diag_term.cwiseProduct(x) + J_packed + xc_packed + K_packed;
```

Global hybrids (B3LYP, PBE0): `c_sr == 0`, `c_fr == exact-exchange
coefficient`, only the `Coulomb`-kernel branch fires.
Pure short-range screened hybrids (HSE06): `c_fr == 0`, only the
`ShortRange` branch. CAM-B3LYP: both nonzero.

UKS is the spin-resolved analogue (`src/dft/driver.cpp:3950-3988`):
`_compute_2e_k_uhf_direct(shell_pairs, dPa, dPb, …)` gives `(dKa, dKb)`
from one sweep per kernel, prefactor `-1.0` (not `-0.5`), pack `dKa` into
`out.head(nova)` and `dKb` into `out.tail(novb)`. Same `c_fr` /`c_sr` /
`omega` branching.

## Enabling the path

Drop `!soscf_dft_hybrid_blocked` (RKS) / `!soscf_uks_hybrid_blocked`
(UKS) from the `soscf_enabled` conjunction, and remove the
`soscf_*_hybrid_blocked` variable plus its `reason` branch in the warning
block. PCM and SAO branches stay. SOSCF-off runs are unaffected — the
`c_fr/c_sr` guard is dead code when the SOSCF branch is inactive, and a
non-hybrid functional has both coefficients zero.

---

## Steps

Each step is independently verifiable. Order matters: the RKS point-level
check (H2) gates wiring the RKS branch (H3); same for UKS.

### H0 — global-hybrid RKS: wire the `Coulomb`-kernel K term only

Smallest possible first cut. In the RKS `h_op`, add only the `c_fr != 0`
branch (global hybrids). Leave `soscf_dft_hybrid_blocked` in place — the
term is present but unreachable, so the tree is byte-identical.

**Verify:** builds; `h2_dft_b3lyp_sto3g` and the full DFT ctest unchanged
(the new code does not execute). `grep` confirms the `_compute_2e_k_direct`
call compiles against the in-scope names.

### H1 — global-hybrid RKS: env-gated point check of the composed `h_op`

Add a `PLANCK_SOSCF_HYBRID_CHECK`-style probe (same discipline as
`docs/SOSCF_DFT.md`'s "What was measured but is not kept" — env-gated,
run once, reverted) that, at the first SOSCF iteration on a hybrid input,
central-differences the **true total energy** `E(κ)` along 3 `(a,i)`
directions at `h = {1e-2, 1e-3, 1e-4}` and compares `4 · h_bare` (RKS
constant) against it.

**Verify:** on B3LYP water/6-31G, ratio → `1.000000` at all three
directions as `h` shrinks. If it misses, the K prefactor or sign is
wrong — fix before H2. This is the load-bearing correctness gate; it is
the same probe `docs/SOSCF_DFT.md` invariant 2 mandates for the base
composition, extended to the hybrid term.

### H2 — global-hybrid RKS: flip the gate, add the permanent regression

Remove `soscf_dft_hybrid_blocked` from `soscf_enabled` and the warning
branch. Add one regression case: RKS B3LYP (or PBE0) small system,
`scf_soscf_start` set, asserting SOSCF energy agrees with
fully-converged plain DIIS to 10 digits, alongside `h2_dft_b3lyp_sto3g`.

**Verify:** the new case passes. `h2_dft_b3lyp_sto3g` (SOSCF off)
byte-identical to pre-change. Full smoke + DFT ctest green. Warning no
longer emitted for hybrid + `scf_soscf_start`.

### H3 — range-separated RKS: add the `ShortRange`-kernel branch

Add the `c_sr != 0` branch to the RKS `h_op` (now reachable). Re-run the
H1 probe on an **HSE06** input (pure short-range) and a **CAM-B3LYP**
input (both branches).

**Verify:** H1 probe ratio → `1.000000` on HSE06 and CAM-B3LYP
water/6-31G. Then add a permanent regression: RKS HSE06 SOSCF-vs-DIIS
10-digit agreement (pairs with the existing `water_*_hse06_*` cases).

### H4 — UKS global hybrid: mirror H0–H2 in the UKS `h_op`

Spin-resolved K via `_compute_2e_k_uhf_direct`, prefactor `-1.0`, pack
`dKa`/`dKb` into the α/β blocks. Env-gated point check uses the **UKS 2×
constant** (`docs/SOSCF_DFT.md` invariant 2) against FD of the total
energy — `2 · h_bare_polarized`.

**Verify:** point-check ratio → `1.000000` on triplet UKS B3LYP
water/6-31G, 3 directions. Flip `soscf_uks_hybrid_blocked`. Permanent
regression: UKS hybrid SOSCF-vs-DIIS 10 digits (pairs with the existing
`h_dft_uks_b3lyp_sto3g`).

### H5 — UKS range-separated: `ShortRange` branch in the UKS `h_op`

As H3 but UKS. Re-run the H4 point check on UKS HSE06 triplet.

**Verify:** ratio → `1.000000`. Permanent regression: UKS HSE06
SOSCF-vs-DIIS 10 digits (pairs with `water_uks_hse06_*`).

### H6 — revert the probes, update `docs/SOSCF_DFT.md`

Delete the `PLANCK_SOSCF_HYBRID_CHECK` probe. Update `docs/SOSCF_DFT.md`:

- invariant 5 / "Not done": remove hybrid & range-separated from the
  rejected list; PCM and SAO stay.
- the `[WRN] DFT SOSCF :` block description: only PCM / SAO now trigger it.
- optionally add hybrid rows to the wall-clock table (see below).

**Verify:** `docs/SOSCF_DFT.md` no longer claims hybrids are rejected.
The hook regenerates `CLAUDE.md`; `vault/Status/Completion.md` and
`vault/Status/Open Work.md` updated (canonical per
`docs/vault_status_is_canonical.md`).

## Not in scope

- **Wall-clock re-measurement.** The K build adds one direct sweep per
  Krylov iteration — same cost class as the `δJ` sweep already there.
  `docs/SOSCF_DFT.md`'s table can get hybrid rows if someone wants them;
  not blocking, and the correctness gates do not depend on it.
- **PCM, SAO/symmetry** for DFT SOSCF — unchanged, still rejected with a
  warning. Those need the reaction-field / block-diagonal response, a
  separate piece.
- **Double hybrids.** Single-point only in the tree; the perturbative
  correlation term is not in the SCF Fock at all, so there is nothing for
  SOSCF to accelerate there.
- **`wick`-style alternate K path.** There is one direct K builder; use it.

## Risks / where it could go sideways

- **Prefactor / sign.** The RKS `-0.5` vs UKS `-1.0` and the `c_fr`/`c_sr`
  split must match `src/dft/driver.cpp:3936-4022` exactly. The H1/H4
  point checks catch a mismatch as a ratio ≠ 1 — do not skip them.
- **`omega` threading.** `range_separation_omega` must reach the
  `ShortRange` call unchanged; it is read off the same `xc_grid` the KS
  build uses, so a mismatch would mean the KS build and the Hessian
  disagree, which the SOSCF-vs-DIIS 10-digit gate would expose.
- **Screened-kernel engine support.** `_compute_2e_k_direct` with
  `ShortRange` already serves the KS build for HSE06 today (regression
  `water_uks_hse06_*`), so the path is exercised — the SOSCF site is just
  another caller.
