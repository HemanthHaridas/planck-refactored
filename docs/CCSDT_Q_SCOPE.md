# Which perturbative higher-order CC corrections can ccgen generate, and what does each need?

Scope, opened 2026-11-21. Nothing here is built. This doc expires when the
first correction lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md`. Sibling scopes: `docs/CCSD_T_SCOPE.md` (the
(T) this generalizes) and `docs/CCSD_T_SIZE_SCOPE.md` (the CCSD cost).

## Why it matters

CCSD(T) is done (restricted and UHF, equal to PySCF). The next rung for
high-accuracy thermochemistry (HEAT, W4-style composites) is a
perturbative quadruples correction on CCSDT: CCSDT(Q). The same family
continues one rank up (CCSDTQ(P)) and has Λ-based variants that behave
better away from equilibrium.

## The family

The definitions are in Kállay and Gauss, JCP 123, 214105 (2005), for
arbitrary rank n, and in Bomble, Stanton, Kállay and Gauss, JCP 123, 054101
(2005), for CCSDT(Q). The Λ variants are in Kállay and Gauss, JCP 129,
144101 (2008). **Transcribe each term list from the papers before
building; do not rebuild it from memory.**

Each non-iterative correction to CC(n−1) has the same shape as (T):

    E = Σ L_n · R_n / D_n

Here R_n is the lowest-order estimate of T_n from the converged CC(n−1)
amplitudes, a projection onto the n-fold manifold, and L_n is a left
vector. The variants differ only in L_n and in which terms enter R_n:

| Correction | Left vector | Needs Λ |
|---|---|---|
| CCSD[T] | T2† (no singles term) | no |
| CCSD(T) (done) | T2† plus the T1 disconnected term | no |
| CCSDT[Q], CCSDT(Q) | T3† (and T2† terms, per Bomble 2005) | no |
| CCSDTQ[P], CCSDTQ(P) | T4† (and lower) | no |
| ΛCCSD(T), CCSDT(Q)_Λ, CR-CC(2,3) | Λ from the left CC equations | yes |

The iterative approximations (CC3, CCSDT-1a/1b, CCSDT-n) are a different
kind of method: truncations of the T3 residual by perturbation order. They
are covered below.

## What ccgen can do now (measured 2026-11-21)

`ccgen.project.nfold_projector` projects onto any rank, and the (T)
generator (`python/ccgen/methods/ccsd_t.py`) is two bare-product
projections. The same calls one rank up work unchanged:

| Projection (connected) | Terms | Max summed indices | Generation time |
|---|---|---|---|
| ⟨T\|(V T2)\|0⟩, (T)'s W | 18 | 1 | 0.0 s |
| ⟨Q\|(V T3)\|0⟩ | 48 | 1 | 0.5 s |
| ⟨Q\|(V T2²)\|0⟩ | 216 | 2 | 2.4 s |
| ⟨P\|(V T4)\|0⟩ (manifold name `quintuples`) | 100 | 1 | 18.7 s |

Closed-shell spin adaptation (`spin_adapt_equations`) of ⟨Q|(V T3)|0⟩
returns the two independent Sz sectors the rank-4 kernels already use:
48 terms on the aabb block and 72 on aaab. ⟨Q|(V T2²)|0⟩ was not
spin-adapted in the probe.

**What ccgen cannot do: Λ.** It has no de-excitation operators, no
left-hand projection ⟨0|(1+Λ) H̄|μ⟩, and no Λ solver. Every Λ-based row
in the table needs all three.

## What each step needs

**Steps 0–3 wait for MRCC access** (user decision, 2026-11-26). Each gate
needs a CFOUR or MRCC value, and neither is available. Step 4 (Λ) does not.

### 0. CCSD[T]: one term, no new kernels

[T] is (T) without V: `(wx + v)·wx/D` becomes `wx·wx/D`
(`src/post_hf/cc/ccsd_t.cpp`). Print it alongside (T) as a diagnostic. The
gate: CCSD[T] from CFOUR or MRCC on one water input. PySCF does not print
it.

### 1. CCSDT[Q] and CCSDT(Q), restricted

- **Amplitudes.** (Q) needs converged spatial t1/t2/t3. Only the generated
  rank-3 route (`ccsdt_gen`, `run_rccgen`) produces them as
  `ArbitraryOrderRCCAmplitudes`. The hand-written `run_rccsdt` mostly goes
  to the determinant backstop and keeps no dense amplitudes (C3 in
  `docs/CC_AMPLITUDE_CHECKPOINT.md`). That route is correct, and its cost
  (`docs/CCGEN_GENERATED_CC_COST.md`) **decides how
  large a (Q) can run**.
- **Generation.** Write `python/ccgen/methods/ccsdt_q.py` on the pattern of
  `ccsd_t.py`. It takes R4 from ⟨Q|(V T3 + ½ V T2²)_C|0⟩ and L4 from the
  paper's left vector. It emits the full-tensor oracle and the
  per-virtual-quadruple batched kernel, as (T) does.
- **Restricted energy weights.** By counting, as for (T):
  E = (2 S_aaaa + 32 S_aaab + 36 S_aabb) / 576. The weights sum to
  70 = C(8,4). If the all-alpha block can be written from aaab, as aaa is
  from aab in (T), it does not need its own kernel. **Verify both
  numerically before relying on them** (the (T) identity was measured to
  4e-14).
- **Cost.** Terms with one summed index are o⁴v⁵ over the o⁴v⁴ target.
  Without intermediates, the two-summed-index ⟨Q|V T2²|0⟩ terms can reach
  o⁴v⁶. List each term's loop shape before emitting. If any term is above
  o⁴v⁵, factor it through a (V T2) intermediate. ccgen's factorizer has no
  numeric gate (`docs/CCGEN_TWO_DRESSING_ROUTES.md`), so a factored term
  must be checked against the unfactored oracle.
- **Gates.** Each one has to fail when the code is broken:
  - **Generic rank.** The same generator at n = 3 must reproduce
    CCSD[T] and the existing CCSD(T) to every printed digit.
  - **Oracle vs. batched.** The batched kernel must match the
    full-tensor oracle to ≤ 1e-14 and be bitwise identical across
    thread counts, as for (T).
  - **Brute-force R4.** Compute ⟨Q|V_N T3|0⟩ as a determinant-space CI
    vector, using the FCI machinery's Slater–Condon elements. It should
    be all connected, because normal-ordered V_N has no pure single
    excitation and f_ov = 0. Check that claim before using it.
  - **External reference.** Compare with CFOUR or MRCC CCSDT(Q) on
    water/STO-3G and one other small molecule, or with the Bomble 2005
    tables if they contain a reproducible case. PySCF has no CCSDT(Q).

### 2. UHF CCSDT(Q)

This mirrors the UCCSD(T) step. It needs the generated `ucc3` amplitudes and
the per-block UCC lowering (`ucc_adapt_equations`). Leave it until step 1
passes.

### 3. CCSDTQ[P] and CCSDTQ(P)

The machinery is the same one rank up, and rank-5 projections generate
(table above). It needs converged generated `cc4` amplitudes (they exist)
and the paper's term list. The costs are o⁵v⁶ and up. This step is useful
only on small benchmark molecules; take it if a user asks for it.

### 4. Λ-based corrections

Scoped separately: `docs/CC_LAMBDA_SCOPE.md` (Λ operators, the Λ
connectivity rule, spin adaptation, the solver, densities, and the
consumers: Λ-based corrections, properties, gradients).

### Iterative approximations (CC3, CCSDT-n)

Scoped separately: `docs/CC_APPROXIMATE_MODELS_SCOPE.md` (CC2, CC3, CCn,
CCSDT-n, and the `ccN` keyword collision).

## Recommended order

1. CCSD[T] (step 0). It is a few lines.
2. Step 1. The cost condition it waited on is met: generated rank 2 is
   48–59× faster than hand-written (`docs/CCSD_T_SIZE_SCOPE.md`), and
   generated rank 3 is faster wall-clock at four threads on every ladder
   rung but CH4 (`docs/CCGEN_GENERATED_CC_COST.md`).
3. Steps 2 and 3 only on request. Step 4 as its own scope, together with
   CC gradients.

## Not in this scope

RI or frozen-core variants, ROHF-based corrections, MPI distribution, and
EOM/LR-CC excited states.
