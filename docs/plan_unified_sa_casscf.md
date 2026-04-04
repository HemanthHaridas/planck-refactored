# Unified SA-CASSCF Plan (Merged Implementation + Theory)

This document merges:
1. Equation-heavy SA-CASSCF theory plan
2. Current repository status and remaining work

---

## Core Problem

Current implementation mixes:
- state-averaged objective
- root-resolved solves
- averaged orbital updates
- rootwise convergence criteria

This causes nonzero gradient plateau for nroots > 1.

---

## Correct Target

Minimize:

E_SA = Σ w_I E_I

Stationarity condition:

g_SA = Σ w_I g_I = 0

NOT:

g_I = 0 for all I

---

## Phase 1: Fix Convergence (Critical)

Replace stopping condition with:

||g_SA|| < tol

Remove dependence on max_root_gnorm.

Keep rootwise gradients only for diagnostics.

---

## Phase 2: True SA Coupled Solve

Solve:

(H - E_I) c1_I = -Q σ_I

and

R_kappa = g_SA + H_kappa kappa + Σ w_I G_kappa_c c1_I = 0

Use one shared kappa.

---

## Phase 3: Orbital Hessian Upgrade

Replace diagonal model:

(HR)_pq = Δ_pq R_pq

with:

HR = δg_SA[R]

Include:
- core response
- active density response
- Q-matrix derivative
- commutator terms

---

## Phase 4: CI RHS Consistency

Build:

σ = (∂H/∂κ) κ c0

Include:
- one-body derivative
- two-body derivative

Remove commutator-only path from production.

---

## Phase 5: Simplify Optimizer

Keep:
- SA coupled step (primary)
- gradient fallback

Remove:
- excessive rescue heuristics
- multiple candidate families

---

## Phase 6: Integrate with Existing Code

Preserve:
- root-resolved data structures
- coupled response blocks
- CI solver and Davidson
- active integral cache

Modify:
- driver logic
- convergence checks
- step construction

---

## Phase 7: Testing

Add:
- SA plateau regression test
- Hessian finite difference test
- CI RHS finite difference test
- multi-root convergence test

Keep:
- existing casscf_tests gate

---

## Acceptance Criteria

- nroots=1 unchanged
- nroots>1 converges without plateau
- SA gradient used for stopping
- shared orbital solve implemented
- true Hessian action in production

---

## Key Insight

The issue is NOT just the Hessian.

It is the mismatch:

SA objective
+ rootwise solves
+ averaged steps
+ rootwise convergence

Fixing this resolves the plateau.
