# Analytic XC Hessian double-counts correlation for combined-XC functionals — scope

**Status: C1–C6 all landed.** The fix and its gates are in; the finding
is folded into `docs/DFT_ANALYTIC_FXC_HESSIAN.md` invariant 3a. C5's
end-to-end gate is `water_rks_b3lyp_soscf_631g` (added by
`SOSCF_DFT_HYBRID_SCOPE` H2, which unblocked RKS global-hybrid DFT
SOSCF) — it asserts B3LYP RKS SOSCF converges superlinearly, which fails
if either this guard or the RKS Hessian-scale fix regresses. This file
can be deleted; the substance lives in invariant 3a.

In-flight scope. Found while verifying the RKS Hessian-scale fix
(`docs/SOSCF_DFT_RKS_HESSIAN_SCALE_SCOPE.md`) against B3LYP / PBE0.

**Question this work answers:** `compute_analytic_xc_hessian_vector_product`
and `_polarized` always sum `fxc[x_functional] + fxc[c_functional]`. For a
functional whose exchange slot is a *combined* exchange-correlation libxc
entry (B3LYP, PBE0, HSE06 — every named hybrid, and any input using one of
those names), `x_functional` already carries the whole XC, so adding
`fxc[c_functional]` double-counts a correlation `fxc`. What does it take to
skip the correlation term in exactly the case the KS-matrix build already
skips it?

## Short answer

The KS-matrix build guards this:

```cpp
// src/dft/driver.cpp ~3160
if (exchange->is_combined_exchange_correlation())
    // "Using {} as a combined XC functional; configured correlation is ignored"
```

`compute_analytic_xc_hessian_vector_product` has **no such guard**. Both
the LDA and GGA branches unconditionally do:

```cpp
auto fxc_x = exchange_functional.evaluate_*_fxc(...);   // for B3LYP: the WHOLE XC
auto fxc_c = correlation_functional.evaluate_*_fxc(...); // spurious: PBE_C etc.
... v2rho2_total = v2rho2_x[pi] + v2rho2_c[pi];          // double-counts correlation
```

Fix: when `exchange_functional.is_combined_exchange_correlation()`, zero
(or skip) the `_c` contributions — `v2rho2_c`, `v2rhosigma_c`,
`v2sigma2_c`, and the GGA path's `vsigma_c` (used in the T3
`2·vsigma·δ∇ρ` term). Same one-line predicate the KS build uses.

## Evidence

RKS Hessian-scale probe (`PLANCK_SOSCF_HYBRID_CHECK`, water/6-31G,
DIIS-converged reference, composed `h_op` = `4·diag + 8·(J + V_xc + K)`
vs central-FD of the true total energy, three `(a,i)` directions):

| functional | XC base | ratio, as shipped | ratio, `_c` skipped for combined |
|---|---|---|---|
| Slater/VWN5 (LDA)   | not combined | 1.000000 | 1.000000 (unchanged) |
| PBE_X + PBE_C       | not combined | 1.000000 | 1.000000 (unchanged) |
| B88 + LYP           | not combined | 1.000000 | 1.000000 (unchanged) |
| PBE_X + LYP         | not combined | 1.000000 | 1.000000 (unchanged) |
| **B3LYP** (`c_fr=0.20`) | combined | 0.9934 – 1.0022 | **1.000000** |
| **PBE0** (`c_fr=0.25`)  | combined | 0.9934 – 1.0056 | **1.000000** |

The residual is **not** the K term (verified: the K response is correct —
pure-GGA hybrids' GGA base is exact, and the residual is identical on
PBE0 and B3LYP despite different `c_fr`), **not** a T3 algebra gap (B88,
LYP, PBE all exact as separate x/c functionals), and **not** grid
integration (grid-independent across normal/fine/ultrafine). It is
exactly the extra `fxc[c_functional]`: zeroing it lands every hybrid
direction on `1.000000`.

## Where the code changes

`src/dft/analytic_hessian.cpp`:

- **RKS `compute_analytic_xc_hessian_vector_product`**, LDA branch
  (`~34-58`) and GGA branch (`~64-134`) — after the `evaluate_*_fxc`
  calls, before the per-point accumulation.
- **UKS `compute_analytic_xc_hessian_vector_product_polarized`**, LDA
  branch (`~180-232`) and GGA branch (`~234-380`) — same.

The predicate `exchange_functional.is_combined_exchange_correlation()` is
already on `DFT::XC::Functional` (`src/dft/base/wrapper.h:299`).

## Steps

### C1 — reproduce with a permanent assertion in the S1 probe  **[DONE]**

The RKS scale probe (`PLANCK_SOSCF_HYBRID_CHECK`,
`SOSCF_DFT_RKS_HESSIAN_SCALE_SCOPE` S1) currently hard-asserts only on
LDA. Extend it: also hard-assert `|ratio - 1| < 1e-4` when
`x_functional.is_combined_exchange_correlation()` **and** the GGA base is
one this scope has shown to be individually exact — i.e. assert on
B3LYP / PBE0 after the fix. Before the fix this new assertion fails
(0.9934 on `(a=nv-1, i=no-1)`); after it passes.

**Verify:** on the unpatched tree the probe aborts on B3LYP naming the
direction and ratio; patched, it passes B3LYP and PBE0 silently.

### C2 — add the guard, RKS LDA + GGA  **[DONE]**

In both branches, after `fxc_x` / `fxc_c` are evaluated:

```cpp
if (exchange_functional.is_combined_exchange_correlation())
{
    std::fill(v2rho2_c.begin(), v2rho2_c.end(), 0.0);
    // GGA branch also:
    std::fill(v2rhosigma_c.begin(), v2rhosigma_c.end(), 0.0);
    std::fill(v2sigma2_c.begin(), v2sigma2_c.end(), 0.0);
    std::fill(vsigma_c.begin(), vsigma_c.end(), 0.0);
}
```

(Or skip the `evaluate_*_fxc` / `evaluate_*_exc_vxc` calls on
`correlation_functional` entirely under the predicate — cheaper, and it
avoids a libxc call on a functional whose result is discarded. Pick one;
the fill form is smaller diff.)

**Verify:** C1's probe passes B3LYP / PBE0 at `1.000000`. The four
not-combined rows (LDA, PBE, B88+LYP, PBE_X+LYP) are **byte-identical**
to before (the predicate is false for them). `ctest -R dft` 12/12.

### C3 — mirror the guard in the UKS polarized path  **[DONE]**

`compute_analytic_xc_hessian_vector_product_polarized`, LDA and GGA
branches. The polarized `_c` arrays are wider (`v2rho2_c` is 3-wide,
`v2rhosigma_c` / `v2sigma2_c` 6-wide, `vsigma_c` 3-wide) — `std::fill`
still covers them.

**Verify:** a UKS hybrid Hessian-scale probe (triplet water/6-31G, B3LYP
or PBE0, the `_polarized` analogue of the S1 probe with the UKS `2×`
constant) lands on `1.000000` after the fix and `~0.99` before. Not-combined
UKS functionals byte-identical.

### C4 — point-level gate for the guard  **[DONE — landed alongside C3]**

Landed in `tests/dft_analytic_hessian_polarized_production.cpp` (not
`dft_gga_hessian_selfcheck.cpp` as first sketched) —
`check_combined_no_double_count`, covering **both** the RKS entry and the
polarized entry from one place, because that file already carries the
`require_functional` + synthetic-grid fixtures for both. Run on
`hyb_gga_xc_b3lyp` and `hyb_gga_xc_pbeh`. Mutation-verified: disabling
`drop_correlation_if_combined` fails all six assertions by 0.012–0.018.

Add a case that:
1. builds a combined-XC functional (`functional_id("hyb_gga_xc_b3lyp")`
   or `"pbe0"`), confirms `is_combined_exchange_correlation()` is true;
2. calls `compute_analytic_xc_hessian_vector_product` with a **non-null**
   `c_functional` (e.g. PBE_C) and again with `c_functional == x_functional`
   (a combined functional passed in both slots, the real driver shape);
3. asserts the two results are **equal** — i.e. the `c_functional`
   argument has no effect when `x_functional` is combined.

Mutation-verify: removing the guard makes the two results differ.

**Verify:** the new ctest passes; reverting C2 fails it with a specific
numeric gap.

### C5 — end-to-end regression once H2 unblocks hybrid DFT SOSCF  **[DONE — gate `water_rks_b3lyp_soscf_631g`, added by SOSCF_DFT_HYBRID_SCOPE H2]**

This bug is currently *latent for production* — DFT SOSCF rejects hybrids
(`soscf_dft_hybrid_blocked`), so no shipped SOSCF run reaches the combined-XC
analytic Hessian. It becomes live the moment `SOSCF_DFT_HYBRID_SCOPE` H2
flips that gate. So: after H2, the hybrid SOSCF-vs-DIIS regression case
(RKS B3LYP or PBE0, 10-digit energy agreement) is the end-to-end gate —
and it should be **added to this scope's verification list**, not just
H2's, because a regression of C2/C3 would show as a hybrid SOSCF
convergence slowdown (linear instead of superlinear), the same signature
the scale fix had.

**Verify:** with C2/C3 in, the H2 hybrid regression converges
superlinearly (`dft_soscf_last_gradient` small); with C2/C3 reverted it
converges linearly (larger `dft_soscf_last_gradient`) but to the same
energy — exactly the `water_rks_lda_soscf_631g` non-vacuity pattern.

### C6 — update `docs/DFT_ANALYTIC_FXC_HESSIAN.md`  **[DONE]**

Add an invariant: **the analytic XC Hessian must apply the same
`is_combined_exchange_correlation()` guard the KS-matrix build applies** —
a combined-XC functional carries its correlation in the exchange slot, so
summing a second `fxc[c_functional]` double-counts. Note it was latent
(hybrid SOSCF gated off) and surfaced via the RKS scale probe on B3LYP /
PBE0.

## Not in scope

- **The K response** — verified correct in
  `SOSCF_DFT_RKS_HESSIAN_SCALE_SCOPE`; the residual chased here was never
  the K term.
- **The FD-kernel oracle** (`build_{closed_shell,unrestricted}_xc_kernel_blocks`)
  — check whether it has the same double-count. It reads
  `functionals` the same way the KS build does, so it likely already
  guards; confirm during C4 but do not expand scope if it does not
  (it is verification-only, not a production Hessian).
- **Double hybrids** — the PT2 term is not in the SCF Fock; unaffected.
- **Range-separated** (`is_range_separated()` true, `is_combined_...`
  may also be true for CAM-B3LYP-style entries) — the guard is on
  `is_combined_exchange_correlation()`, which is orthogonal to range
  separation. If a range-separated functional is *also* combined XC, the
  same guard applies and C2/C3 already cover it; if it is not combined
  (separate `x`/`c` slots), nothing changes. No extra work.

## Risk

- **Over-zeroing.** If some combined-XC functional legitimately needs a
  separate correlation `fxc` added (none known — the KS build's own guard
  is the precedent), C2 would under-count. Mitigated by C4 asserting the
  `c_functional` argument is a no-op for combined XC, which is the exact
  contract the KS build already relies on.
- **UKS array widths.** The polarized `_c` arrays must be fully zeroed,
  not just their first element — `std::fill` on the whole vector, verified
  by C3's probe landing on `1.000000` (a partial zero would leave a
  residual).
