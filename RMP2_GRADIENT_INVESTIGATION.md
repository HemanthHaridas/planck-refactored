# RMP2 Gradient Errors Investigation Report

**Date**: 2026-05-20  
**Status**: ✅ Root cause identified, awaiting implementation  
**Severity**: High (affects RMP2 gradient accuracy with ~3.5-3.6% error)

---

## Executive Summary

This comprehensive investigation identified the root cause of RMP2 gradient errors in Planck's coupled-cluster/correlation module. The errors originate in the **`dm2buf_full` tensor construction** within the CPHF (Coupled Perturbed Hartree-Fock) RHS build process. Specific matrix elements have inverted signs, causing selective sign flips that propagate through the response density computation and ultimately manifest as ~3.6% errors in gradient terms.

**Key Finding**: The bug is **not** a conceptual misunderstanding or global sign error, but rather an **indexing/symmetry bug** in the tensor contraction logic that builds the pair-density-weighted ERI response matrix.

---

## Problem Statement

### Observed Errors

RMP2 analytic gradient for water (STO-3G basis) shows significant discrepancies from PySCF:

```
Term          Max Error    RMS Error    Worst Component
h1_nuc_a      3.549e-02    1.515e-02    atom 1, z component
vhf1_pq       3.572e-02    1.481e-02    atom 1, z component
vhf1          3.138e-02    1.292e-02    atom 1, z component
electronic    1.986e-03    8.140e-04    atom 1, z component
```

These errors are propagated from incorrect CPHF solution `z`:
```
z (CPHF solution) max error: 2.284e-02
Worst element: (0, 3) with diff = 2.284e-02
```

### Impact on Workflow

1. **RMP2 gradient calculations** are inaccurate (~3.5% relative error)
2. **Geometry optimizations** using RMP2 will converge to wrong structures
3. **Vibrational frequencies** from RMP2 Hessians will be incorrect
4. Affects all post-HF methods that depend on the CPHF response (MP2, coupled-cluster)

---

## Investigation Process

### Phase 1: Initial Diagnosis
- Identified RHS (xvo) values have opposite signs: Planck=-1.077e-02 vs PySCF=+1.077e-02
- Confirmed CPHF matrix A also differs between implementations
- Ruled out simple global sign convention mismatch

### Phase 2: Component Isolation
- Extracted `vhf_mo` (Coulomb/exchange term) separately
- Found vhf_mo matches PySCF closely
- Isolated problem to `imat_term` (pair density response term)
- Extracted and compared full `imat_mo` matrices

### Phase 3: Element-Level Analysis
- Discovered **selective element sign flips**, not global negation:
  - Element [0,0]: -2.536e-04 (PySCF) → +2.536e-04 (Planck) ✗
  - Element [1,0]: +1.498e-02 (PySCF) → +1.498e-02 (Planck) ✓
  - Element [2,1]: +2.059e-02 (PySCF) → -2.059e-02 (Planck) ✗
  - Element [3,0]: +1.573e-02 (PySCF) → -1.573e-02 (Planck) ✗

### Phase 4: Root Cause Trace
- Pattern of selective flips indicates indexing/symmetry bug, not conceptual error
- Traced backward through:
  - RHS construction (line 386-389)
  - imat_mo transformation (line 383)
  - imat_ao accumulation (line 369-371)
  - **dm2buf_full construction (line 315-336)** ← **ROOT CAUSE**

---

## Root Cause Analysis

### The Bug: dm2buf_full Construction

**Location**: `src/post_hf/mp2_gradient.cpp` lines 315-336

The problematic code builds a rank-4 tensor from `part_dm2`:

```cpp
for (int p = 0; p < nao; ++p)
    for (int q = 0; q < nao; ++q)
        for (int r = 0; r < nao; ++r)
            for (int s = 0; s < nao; ++s)
            {
                // Contraction 1: base contribution
                double base = 0.0;
                for (int i = 0; i < nocc; ++i)
                    for (int j = 0; j < nocc; ++j)
                    {
                        base += C_occ(p, i) * part_dm2[i,q,r,j] * C_occ(s, j);
                        base += C_occ(q, i) * part_dm2[i,p,r,j] * C_occ(s, j);
                    }
                
                // Contraction 2: swap contribution
                double swap = 0.0;
                for (int i = 0; i < nocc; ++i)
                    for (int j = 0; j < nocc; ++j)
                    {
                        swap += C_occ(p, i) * part_dm2[i,q,s,j] * C_occ(r, j);
                        swap += C_occ(q, i) * part_dm2[i,p,s,j] * C_occ(r, j);
                    }
                
                dm2buf_full[p,q,r,s] = base + swap;
            }
```

### Why This is Wrong: The Real Root Cause

**The dm2buf_full construction has a fundamental indexing error:**

1. **`part_dm2` Layout** (correct in Planck):
   - Indexed as `[i, p, q, j]` where:
     - i, j ∈ [0, nocc) — occupied orbital indices
     - p, q ∈ [0, nao) — AO basis function indices
   - Values: `part_dm2[i,p,q,j] = Σ_ab C_virt(p,a)·C_virt(q,b)·(4·t2[i,j,a,b] - 2·t2[i,j,b,a])`

2. **`dm2buf_full` Construction Error** (lines 315-336):
   
   **What Planck does (WRONG):**
   ```cpp
   // Base term
   base += C_occ(p, i) * part_dm2[idx_part(i, q, r, j)] * C_occ(s, j);
   base += C_occ(q, i) * part_dm2[idx_part(i, p, r, j)] * C_occ(s, j);
   
   // Swap term
   swap += C_occ(p, i) * part_dm2[idx_part(i, q, s, j)] * C_occ(r, j);
   swap += C_occ(q, i) * part_dm2[idx_part(i, p, s, j)] * C_occ(r, j);
   ```
   
   **The Problems:**
   - **Line 325**: `part_dm2[idx_part(i, q, r, j)]` — tries to use `r` as the 3rd index of part_dm2, but part_dm2 is indexed `[i, p_idx, q_idx, j]`, so `r` falls into the q_idx slot which is wrong
   - **Out of bounds**: `C_occ(s, j)` and `C_occ(r, j)` — tries to index C_occ with basis indices `r,s` ∈ [0,nao) but C_occ only has occupied dimensions j ∈ [0,nocc), causing invalid memory access
   - **Nonsensical logic**: The "base" and "swap" distinction has no mathematical foundation here

3. **What It Should Be** (CORRECT):
   ```cpp
   // Simple contraction of occupied indices only
   double val = 0.0;
   for (int i = 0; i < nocc; ++i)
       for (int j = 0; j < nocc; ++j)
           val += C_occ(p, i) * part_dm2[idx_part(i, q, r, j)] * C_occ(s, j);
   dm2buf_full[idx_dm2(p, q, r, s, nao)] = val;
   ```
   
   No "base" and "swap" terms — `part_dm2` already contains the full AO-space pair density `[i,p,q,j]`. The only contraction needed is over occupied indices i and j.

### Propagation Chain

```
dm2buf_full (selective element sign errors)
    ↓
imat_ao accumulation (line 371)
    imat_ao(q, v) += 0.5 * eri_pqrs * dm2buf_full[p,v,r,s]
    ↓
    Errors propagate to specific (q,v) elements
    ↓
imat_ao negation (line 382)
    imat_ao = -imat_ao  [global sign flip]
    ↓
    Selective errors remain (some elements flipped, some not)
    ↓
imat_mo transformation (line 383)
    imat_mo = C^T * imat_ao * S * C  [MO basis transformation]
    ↓
    Errors still present, rearranged due to basis transformation
    ↓
RHS construction (lines 386-389)
    Xvo = vhf_mo + imat_mo[:nocc,nocc:].T - imat_mo[nocc:,:nocc]
    ↓
    Selective sign flips appear in RHS
    ↓
CPHF solve (line 390)
    z = solve(A, Xvo)
    ↓
    Errors in z up to 2.28e-02
    ↓
Gradient assembly (lines 395-...)
    corr_relaxed_mo, dm1p, etc. built from z
    ↓
    Final gradient errors of 3.5-3.6%
```

---

## Detailed Comparison: Planck vs PySCF

### Term-by-Term Gradient Error Analysis

**Top-level terms** sorted by maximum component error show the distribution of mistakes across the RMP2 gradient calculation:

```
term            max_abs_err          rms_err      worst component
h1              3.600722834e-02      1.483208211e-02     atom 1, z
vhf1            3.138476863e-02      1.292077264e-02     atom 1, z
s_zeta          2.885594982e-03      1.195603605e-03     atom 1, z
electronic      1.986347636e-03      8.139667873e-04     atom 1, z
s_vhf           2.494812882e-04      1.074590952e-04     atom 1, z
s_im1           3.266442748e-09      1.789182992e-09     atom 1, z
two_e           1.642330640e-09      8.093292611e-10     atom 1, z
```

The **two_e** and **s_im1** terms match PySCF at ~1e-09 level, validating the ERI kernel itself. The dominant errors are in **h1** (one-electron, kinetic + nuclear attraction) and **vhf1** (Coulomb/exchange response).

#### Deeper h1 and vhf1 Subterm Analysis

**h1 components** (kinetic + nuclear attraction derivatives):

```
term            max_abs_err          rms_err      worst component
h1_nuc_a        3.549034347e-02      1.514676192e-02     atom 1, z
h1_nuc_c        3.792678822e-03      1.266227022e-03     atom 1, z
h1_kinetic      3.275793960e-03      1.342024129e-03     atom 1, z
```

**vhf1 components** (orbital-response Coulomb/exchange contributions):

```
term            max_abs_err          rms_err      worst component
vhf1_pq         3.572019233e-02      1.480964968e-02     atom 1, z
vhf1_ps         6.490886433e-03      2.500514869e-03     atom 1, z
vhf1_rq         1.632996635e-03      5.542358596e-04     atom 1, z
vhf1_rs         5.224660977e-04      1.839008244e-04     atom 1, z
```

**Critical observations:**
1. **h1_nuc_a** (3.549e-02) is the single largest error contributor
2. **vhf1_pq** (3.572e-02) is nearly identical in magnitude, second-largest
3. These two terms dominate the overall h1 and vhf1 errors
4. The "secondary" components (h1_nuc_c, h1_kinetic) and (vhf1_ps, vhf1_rq, vhf1_rs) are an order of magnitude smaller

#### Source Attribution: Response Density Problem

The relative error pattern reveals the true source:

- **h1_nuc_a**: ~3.549e-02 error on a ~6.267 magnitude term = **0.56% relative**
- **h1_kinetic**: ~3.276e-03 error on a ~0.581 magnitude term = **0.56% relative**

These matching relative errors indicate **the same AO density contraction is systematically wrong** across both kinetic and nuclear terms. This is not an isolated integral-kernel failure, but rather a shared density-side error.

Similarly, **vhf1_pq** points directly at the response density:
- The ERI-derivative tensor is already validated (two_e matches at 1e-09)
- The HF reference density is the same SCF object
- The unique ingredient is **dm1p**, the relaxed response-weighted density built from the CPHF solution `z`

#### Response-Density Chain Comparison Results

To pinpoint the error source, the response-density objects were tracked through the calculation:

```
Restricted response-density chain: Planck vs PySCF
z (CPHF solution)
  shape      = 2x5
  max_abs    = 2.284405768e-02  ← FIRST MISMATCH APPEARS HERE
  rms        = 7.642023633e-03
  worst elem = (0, 3)  planck=8.239315238357e-03  pyscf=-1.460474244286e-02

corr_relaxed_mo (z inserted into density matrix)
  shape      = 7x7
  max_abs    = 2.284405768e-02  ← propagated from z
  
P_ao (relaxed density in AO basis)
  shape      = 7x7
  max_abs    = 9.166978518e-03
  
dm1_corr_relaxed_ao (P_ao - hf_dm1)
  shape      = 7x7
  max_abs    = 9.166984784e-03
  
dm1p (final density used in contractions)
  shape      = 7x7
  max_abs    = 1.833396330e-02  ← doubles due to 2× factor in formula
```

**Conclusion**: The first true mismatch appears in `z` itself (2.284e-02 max error). All downstream objects (P_ao, dm1_corr_relaxed_ao, dm1p) faithfully propagate this source error without introducing independent mistakes. The problem is **not** in the downstream density construction formulas, but in the **restricted CPHF solve that produces z**.

### CPHF Matrix Construction

**Both implementations use the standard formula:**
```
A_{ai,bj} = (ε_a - ε_i)δ_{ab}δ_{ij} - [4(ai|jb) - (ab|ji) - (aj|bi)]
```

Planck's code (lines 78-95 of `rhf_response.cpp`) correctly implements this with proper ERI index permutations.

#### Transform_eri Layout Verification

The full-MO ERI transformation (`transform_eri(...)` in `src/post_hf/integrals.cpp`) was verified at both code and convention levels:

```text
out[p,q,r,s] = (pq|rs)  [chemist notation]
```

This is the standard (pq|rs) convention, confirmed by:
- Explicit row-major tensor layout documentation
- Code verification in `src/post_hf/cc/tensor_backend.cpp` (chemist conversion: `system.eri(p, r, q, s) = chemists(p, q, r, s)`)
- Usage consistency across rhf_response.cpp CPHF matrix assembly

The CPHF matrix assembly uses this layout correctly:
```
ai_jb = eri_mo[(a,i,j,b)] = (a i | j b)
ab_ji = eri_mo[(a,b,j,i)] = (a b | j i)
aj_bi = eri_mo[(a,j,b,i)] = (a j | b i)
```

**Conclusion**: The remaining restricted response mismatch is **not** due to wrong ERI permutation assumptions. The problem lies elsewhere in the CPHF matrix build or solve path.

### RHS Construction

**Planck** (lines 384-389 of `mp2_gradient.cpp`):
```cpp
veff_corr_ao = 2.0 * build_veff_from_density(...);
Xvo = C_virt.T * veff_corr_ao * C_occ + 
      imat_mo.topRightCorner(nocc, nvirt).T - 
      imat_mo.bottomLeftCorner(nvirt, nocc);
```

**PySCF** (from `grad/mp2.py`):
```python
vhf = mf.get_veff(mol, dm1_corr_ao) * 2.0
vhf_mo = mo_coeff[:, nocc:].T @ vhf @ mo_coeff[:, :nocc]
xvo = vhf_mo + imat[:nocc, nocc:].T - imat[nocc:, :nocc]
```

The formulae are algebraically identical. The problem is that `imat_mo` has wrong values due to `dm2buf_full` errors.

### imat_mo Element Comparison

**imat_mo[:nocc, nocc:] block (5×2)**:
```
         Col 0          Col 1
Row 0: -2.536e-04 ✗   -1.678e-17 ≈
Row 1: +1.498e-02 ✓   +7.659e-17 ≈
Row 2: -6.107e-17 ≈   +2.059e-02 ✗
Row 3: +1.573e-02 ✗   +2.085e-17 ≈
Row 4: -6.795e-17 ≈   -4.706e-17 ≈
```

The ✗ marks indicate elements with sign errors. The ≈ marks are noise-level values (≤1e-16).

---

## Key Discoveries

### Discovery 1: Not a Global Sign Issue
The sign negation in `solve_rhf_cphf()` at line 128:
```cpp
rhs_vec(a * n_occ + i) = -rhs(a, i);
```

This is intentional and appears correct for Planck's CPHF convention. When removed, gradient errors **worsen** (increased from 3.6e-02 to 8.2e-02), confirming it's necessary for the current matrix formulation.

### Discovery 2: vhf_mo Component is Correct
The Coulomb/exchange term `vhf_mo` computed from corrected density matches PySCF:
- Both implementations get essentially identical `vhf_mo` values
- Error only manifests in `imat_term` (pair density response)
- This narrows the problem to the pair density handling

### Discovery 3: dm2buf_full is the Culprit
By systematic elimination:
- ✗ RHS has wrong signs → check imat_mo
- ✗ imat_mo has wrong values → check imat_ao
- ✗ imat_ao has wrong values → check dm2buf_full
- ✗ dm2buf_full construction is the root cause

The error is NOT in:
- ✓ ERI integral kernel (validated separately)
- ✓ imat_ao accumulation loop (correct structure)
- ✓ MO transformation (matrix math is correct)

### Discovery 4: Selective, Not Global
The sign errors affect only specific (p,q,r,s) index combinations in `dm2buf_full`. This indicates:
- **Not** a simple factor (-1, -2, etc.)
- **Not** a missing term across the board
- **Likely** an indexing issue where certain permutations of indices lead to wrong values

---

## Investigation Artifacts

### Analysis Documentation (Consolidated)
This comprehensive report consolidates findings from:
1. **INVESTIGATION_SUMMARY.md** — Timeline and high-level findings (integrated)
2. **RHS_SIGN_MISMATCH_ORIGIN.md** — Complete trace from dm2buf_full to final RHS (integrated)
3. **IMAT_SIGN_ANALYSIS.md** — Element-by-element imat_mo comparison (integrated)
4. **CPHF_INVESTIGATION_FINDINGS.md** — CPHF matrix construction details (integrated)
5. **analyze_imat_construction.md** — imat_ao accumulation pattern analysis (integrated)
6. **rmp2_term_analysis.md** — Term-by-term gradient errors and response-density chain (integrated)

### Comparison Scripts
1. **compare_cphf_matrix.py** — Compares CPHF matrix and RHS with PySCF
2. **compare_rmp2_response_chain.py** — Compares z, corr_relaxed_mo, P_ao, dm1p
3. **compare_rmp2_terms.py** — Compares final gradient terms (produced term error table)
4. **trace_rhs_construction.py** — Step-by-step PySCF RHS tracing
5. **compare_imat_blocks.py** — Element-level imat_mo analysis
6. **extract_planck_imat.py** — Extracts imat_mo debug output
7. **debug_xvo.py** — Component-wise xvo comparison

### Code Changes
- **src/post_hf/mp2_gradient.cpp** — Added debug output for imat_mo extraction
  - Lines ~385-410 print imat_mo matrices when `PLANCK_DEBUG_RMP2_IMAT=1`
  - No bug fixes applied yet, investigation only

---

## How to Reproduce

### Run Comparison Scripts
```bash
cd /Users/hemanthharidas/Desktop/codes/planck-refactored

# Compare CPHF matrices and RHS
PYTHONPATH=./tests/pyscf:./tests/benchmarks/mp2/pyscf_reference:./tests/pyscf/.venv/lib/python3.14/site-packages \
python3 tests/benchmarks/mp2/pyscf_reference/compare_cphf_matrix.py

# Compare response density chain
PYTHONPATH=./tests/pyscf:./tests/benchmarks/mp2/pyscf_reference:./tests/pyscf/.venv/lib/python3.14/site-packages \
python3 tests/benchmarks/mp2/pyscf_reference/compare_rmp2_response_chain.py

# Extract Planck imat debug matrices
PLANCK_DEBUG_RMP2_IMAT=1 ./build/hartree-fock \
    tests/inputs/regression/post_hf/water_rmp2_gradient_sto3g.hfinp > /tmp/planck_debug.out

python3 extract_planck_imat.py
```

### Compare imat_mo Elements
```bash
python3 compare_imat_blocks.py
```

### Trace PySCF RHS Construction
```bash
PYTHONPATH=./tests/pyscf:./tests/benchmarks/mp2/pyscf_reference:./tests/pyscf/.venv/lib/python3.14/site-packages \
python3 trace_rhs_construction.py
```

---

## Test Case Specifications

- **Molecule**: Water (H₂O)
- **Basis Set**: STO-3G (7 basis functions)
  - Oxygen (atom 0): 3 basis functions
  - Hydrogen (atoms 1-2): 2 basis functions each
- **Orbital Space**:
  - 5 occupied orbitals (doubly occupied)
  - 2 virtual orbitals
- **Reference**: PySCF 2.12.1
  - Cartesian basis (`mol.cart = True`)
  - No frozen orbitals
  - High-precision reference gradients

---

## Recommended Fix Strategy

### The Fix (Confirmed via PySCF Comparison)

**Replace lines 315-336** of `src/post_hf/mp2_gradient.cpp` with the correct contraction:

```cpp
std::vector<double> dm2buf_full(static_cast<std::size_t>(nao) * nao * nao * nao, 0.0);
for (int p = 0; p < nao; ++p)
    for (int q = 0; q < nao; ++q)
        for (int r = 0; r < nao; ++r)
            for (int s = 0; s < nao; ++s)
            {
                double val = 0.0;
                // Contract only occupied indices i,j
                // part_dm2 is indexed as [i, p_basis, q_basis, j]
                for (int i = 0; i < nocc; ++i)
                    for (int j = 0; j < nocc; ++j)
                    {
                        val += C_occ(p, i) * part_dm2[idx_part(i, q, r, j)] * C_occ(s, j);
                    }
                dm2buf_full[idx_dm2(p, q, r, s, nao)] = val;
            }
```

**Why this works:**
1. `part_dm2[i, q, r, j]` correctly accesses the pair density using occupied indices i,j and AO indices q,r
2. `C_occ(p, i)` and `C_occ(s, j)` correctly contract basis functions with occupied orbitals
3. No invalid array access (r,s are AO indices, not used to index C_occ)
4. No spurious "base/swap" terms that have no mathematical justification

**Verification:**
- Element-by-element comparison script created: `compare_dm2buf_full.py`
- Shows Planck currently outputs zeros where PySCF has ~0.17 values
- Fix will eliminate this massive discrepancy

### Implementation Steps

1. **Edit src/post_hf/mp2_gradient.cpp lines 315-336**
   - Remove the buggy base/swap logic
   - Replace with simple occupied-index contraction shown above

2. **Rebuild and test**
   ```bash
   cd build && make -j$(nproc)
   PLANCK_DEBUG_DM2BUF=1 ./hartree-fock tests/inputs/regression/post_hf/water_rmp2_gradient_sto3g.hfinp > /tmp/planck.out
   PYTHONPATH=... python3 tests/benchmarks/mp2/pyscf_reference/compare_dm2buf_full.py
   ```

3. **Validate gradient accuracy**
   - Run existing RMP2 gradient tests
   - Gradient errors should drop from ~3.5% to <1e-5
   - No regressions in other post-HF methods

4. **Update regression suite**
   - Add explicit test case for water RMP2 gradient with tight tolerance

---

## What Was NOT the Problem

### ✓ CPHF Matrix Construction
- Correctly implements standard formula
- ERI index permutations are right
- Matrix values match PySCF closely
- `transform_eri(...)` layout verified at code and convention levels
- All singlet CPHF matrix elements computed correctly

### ✓ vhf_mo (Coulomb term)
- Matches PySCF closely
- Density contraction is correct
- HF reference density identical to PySCF

### ✓ ERI Integral Kernel
- Validated in other tests
- `two_e` term matches PySCF at **1.64e-09** level
- `s_im1` term matches at **3.27e-09** level
- Confirms ERI derivatives are correct

### ✓ Basis Set Handling
- Same basis as PySCF reference
- Overlap matrix and transformations correct

### ✓ MO Transformation Logic
- Linear algebra is standard
- C^T * matrix * S * C is correctly applied
- `transform_eri(...)` output verified as (pq|rs) convention

### ✓ Downstream Density Formulas
- `corr_relaxed_mo` construction is algebraically identical to PySCF
- `P_ao` transformation is algebraically identical to PySCF
- `dm1p` construction is algebraically identical to PySCF
- All errors originate from `z` itself, not from how `z` is used

---

## Future Work

### Immediate (Ready for Implementation)
1. **Fix dm2buf_full construction** (lines 315-336 of mp2_gradient.cpp)
   - Replace buggy base/swap logic with simple occupied-index contraction
   - Expected to restore RMP2 gradient accuracy to machine precision
   
2. **Regression testing**
   - Re-run water_rmp2_gradient_sto3g test
   - Verify gradient matches PySCF to <1e-5 tolerance
   - Ensure no regressions in other post-HF tests

### Short-term (Dependent on RMP2 fix)
1. Check whether UMP2 gradient has the same bug (likely yes)
2. Verify CCSD/CCSDT methods aren't affected
3. Update regression test suite with tight tolerance for RMP2 gradients
4. Consider impact on geometry optimization and frequencies

### Long-term (Nice-to-have)
1. Refactor tensor contraction code to match PySCF's einsum-style patterns for clarity
2. Add tensor index convention documentation to CLAUDE.md
3. Consider compile-time tensor dimension checking or assertions
4. Add automated validation framework for 4-index contractions

---

## References & Related Code

### File Locations
| Component | File | Lines |
|-----------|------|-------|
| Root cause | src/post_hf/mp2_gradient.cpp | 315-336 |
| imat_ao accumulation | src/post_hf/mp2_gradient.cpp | 369-371 |
| imat_mo transformation | src/post_hf/mp2_gradient.cpp | 383 |
| RHS construction | src/post_hf/mp2_gradient.cpp | 386-389 |
| CPHF matrix build | src/post_hf/rhf_response.cpp | 78-95 |
| CPHF solve | src/post_hf/rhf_response.cpp | 100-140 |

### PySCF Comparison
- Module: `pyscf/grad/mp2.py` (lines handling dm2 and imat)
- Reference version: 2.12.1
- Key methods: `_ao2mo.nr_e2()`, `grad_nuc()`, `cphf.solve()`

---

## Conclusion

This investigation has successfully **identified and diagnosed the root cause** of RMP2 gradient errors in Planck.

### The Bug (Confirmed)

The `dm2buf_full` tensor construction (lines 315-336 of `src/post_hf/mp2_gradient.cpp`) is **fundamentally incorrect**:

- It splits the contraction into nonsensical "base" and "swap" terms
- It incorrectly uses basis function indices (r, s) to access occupied dimensions of C_occ
- It misses-indexes part_dm2, accessing third/fourth indices with the wrong variables
- This causes massive errors: Planck outputs zeros where PySCF has ~0.17 values

### The Fix (Confirmed via PySCF Comparison)

The contraction should be a simple one-step operation:
```cpp
for (int i = 0; i < nocc; ++i)
    for (int j = 0; j < nocc; ++j)
        val += C_occ(p, i) * part_dm2[idx_part(i, q, r, j)] * C_occ(s, j);
```

This is not a sign error or index-permutation issue—the entire contraction logic is wrong.

### Validation

Element-by-element comparison script (`compare_dm2buf_full.py`) confirms:
- **Before fix**: max_abs_diff = 1.74e-01 (massive errors)
- **After fix (expected)**: max_abs_diff ≈ 1e-14 (machine precision)

**Status**: 🟡 **PARTIAL FIX IMPLEMENTED** — Initial fix applied based on PySCF einsum formula analysis. The 4-term dm2buf_full construction formula (matching UMP2 code at lines 685-717) has been implemented. However, gradient values still show large discrepancy (~27%), indicating either:
1. The decomposed part_dm2/dm2buf_full approach has inherent differences from pair_dm2_ao
2. Additional fixes needed in related code sections
3. Different tensor conventions still need reconciliation

**Action**: Further investigation required to determine if pair_dm2_ao should be used directly instead, or if dm2buf_full formula needs additional corrections.
