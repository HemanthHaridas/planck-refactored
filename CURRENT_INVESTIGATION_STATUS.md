# RMP2 Gradient Issue - Current Investigation Status

**Date**: 2026-05-20  
**Status**: Root cause identified but fix incomplete  
**Problem**: RMP2 analytic gradient differs from PySCF by ~27%

## Key Finding: RHS Sign Error

The CPHF comparison script (`compare_cphf_matrix.py`) reveals:

```
RHS Comparison:
  worst elem = (0, 3)  
  planck=-1.077347782669e-02  
  pyscf=+1.077347987910e-02  
  diff=-2.154695770579e-02
```

**The RHS has opposite signs in Planck vs PySCF.** This propagates to the CPHF solution `z`, causing all downstream gradient calculations to be wrong.

## Root Cause: dm2buf_full Construction

The RHS includes the `imat_term` which comes from `imat_mo`. The `imat_mo` is built from `imat_ao` which accumulates contributions from `dm2buf_full`.

### Current Investigation Results

1. **part_dm2 is correct**: Verified that part_dm2 computed in Planck matches PySCF's formula at machine precision (max diff ~1e-17)

2. **dm2buf_full formula ambiguity**:
   - Investigation document suggests using **1-term formula** (just `val += C_occ(p,i) * part_dm2[i,q,r,j] * C_occ(s,j)`)
   - Comparison script uses **4-term formula** (with additional permutations of indices)
   - PySCF computation with 4-term formula gives ~0.1742 for sample element
   - Planck 1-term formula gives ~0.0436 for same element
   - Planck 4-term formula currently gives ~0 (zeros where should be non-zero)

3. **Geometry mismatch in comparison script**: 
   - Test file uses: `O 0,0,0; H 0.758602,0,0.504284; H -0.758602,0,0.504284` (Angstrom)
   - Comparison script uses: `O 0,0,0; H 0,1,1; H 1,0,0` (Bohr)
   - These are completely different molecules, so comparison is invalid

## Hypothesis

The 4-term formula in the comparison script might be incorrect. The correct formula should give the right magnitude when using the right geometry. The investigation document's 1-term formula might actually be correct, but we need to verify with the correct test case geometry.

## Next Steps Required

1. **Modify comparison script** to use the correct geometry (from test file in Angstrom)

2. **Verify dm2buf_full formula** with correct geometry:
   - Recompute expected dm2buf values using correct water geometry
   - Check if 1-term or 4-term formula is correct for this geometry

3. **Debug imat_mo sign** once dm2buf_full is confirmed correct:
   - The `imat_ao = -imat_ao` negation at line 394 might be contributing to the sign flip
   - Check MO transformation formula

4. **Verify the fix** with gradient comparison

## Files Modified

- `src/post_hf/mp2_gradient.cpp`: Debugging infrastructure added, but formula changes incomplete
- Changes use environment variables for debug output: `PLANCK_DEBUG_DM2BUF`, `PLANCK_DEBUG_RMP2_IMAT`, etc.

## Current Code State

The code currently uses the **4-term formula** for dm2buf_full (lines 302-318), which is the same formula used in the UMP2 gradient code. This suggests the formula is likely correct for Planck's architecture.

The real issue is likely NOT the formula, but rather:
1. A sign convention mismatch in the RHS assembly
2. The negation of imat_ao affecting the final RHS  
3. Some other part of the CPHF or response density pipeline
