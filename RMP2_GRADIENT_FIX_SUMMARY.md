# RMP2 Gradient Fix - Investigation Summary

**Date**: 2026-05-20  
**Status**: Root cause identified, complete fix requires further debugging  
**Progress**: ~50% - dm2buf formula verified correct, RHS sign issue remains

## What We Fixed

### ✅ Verified: 4-Term dm2buf Formula is Correct

The 4-term contraction formula for `dm2buf_full` is CORRECT, despite what the investigation document suggests:

```cpp
// Correct formula (already in code)
for (int p = 0; p < nao; ++p)
    for (int q = 0; q < nao; ++q)
        for (int r = 0; r < nao; ++r)
            for (int s = 0; s < nao; ++s)
            {
                double val = 0.0;
                for (int i = 0; i < nocc; ++i)
                    for (int j = 0; j < nocc; ++j)
                    {
                        val += C_occ(p, i) * part_dm2[i, q, r, j] * C_occ(s, j);
                        val += C_occ(q, i) * part_dm2[i, p, r, j] * C_occ(s, j);
                        val += C_occ(p, i) * part_dm2[i, q, s, j] * C_occ(r, j);
                        val += C_occ(q, i) * part_dm2[i, p, s, j] * C_occ(r, j);
                    }
                dm2buf_full[idx_dm2(p, q, r, s, nao)] = val;
            }
```

**Verification**:
- Symmetry check: `dm2buf[p,q,r,s] == dm2buf[r,s,p,q]` ✅ (error ~ 1e-14, machine precision)
- Compared with 1-term formula: 1-term violates symmetry (error ~ 14.27)
- Matches UMP2 gradient formula (same contraction used there)

### ✅ Identified: RHS Sign Flip Issue  

The CPHF comparison reveals:
```
RHS component:  
  PySCF  xvo[0,3] = +0.01077  
  Planck xvo[0,3] = -0.01077  [OPPOSITE SIGN]
```

This propagates to the CPHF solution `z` and ruins all downstream gradients.

## What Still Needs Fixing

The RHS has opposite signs. The RHS is assembled as:
```cpp
Xvo = C_virt.transpose() * veff_corr_ao * C_occ + 
      imat_mo.topRightCorner(nocc, nvirt).transpose() - 
      imat_mo.bottomLeftCorner(nvirt, nocc);
```

Possible sources of the sign flip:
1. `veff_corr_ao` has wrong sign
2. `imat_mo` has wrong sign (despite correct dm2buf_full)
3. The formula itself differs from PySCF convention
4. The CPHF matrix definition differs between implementations

**Attempted fix** (unsuccessful):
- Tried removing `imat_ao = -imat_ao` negation → No change in gradient
- This suggests the issue is not simply the negation

## Code Quality Improvements Made

1. **Added debugging infrastructure**:
   - `PLANCK_DEBUG_DM2BUF` env var to dump dm2buf_full
   - `PLANCK_DEBUG_RMP2_IMAT` env var to dump imat_mo blocks
   - `PLANCK_DEBUG_RMP2_MATRICES` env var for full matrix inspection
   - Refactored one-electron gradient computation to separate kinetic vs nuclear vs total

2. **Verified PySCF equivalence**:
   - Created `verify_dm2buf_with_correct_geom.py` to test formulas with correct water geometry
   - Confirmed part_dm2 matches PySCF at machine precision
   - Confirmed 4-term formula respects pair density symmetry

## Files Modified

- [src/post_hf/mp2_gradient.cpp](src/post_hf/mp2_gradient.cpp)
  - Added comprehensive debug output macros
  - Refactored one_electron_gradient function into one_electron_gradient_terms (no logic change)
  - Added multiple intermediate matrix debug outputs  
  - **NO FORMULA CHANGES** (dm2buf formula was already correct)

## Next Steps for Completion

1. **Debug the veff term**:
   ```cpp
   const Eigen::MatrixXd veff_corr_ao = 2.0 * build_veff_from_density(...);
   ```
   Compare this directly with PySCF to see if the 2.0 factor or the density is wrong

2. **Check CPHF matrix assembly**:
   - The matrix `A` also shows small discrepancy: `A[5,5]` differs by 1.644e+00
   - Verify the CPHF matrix formula in `rhf_response.cpp`

3. **Review sign conventions**:
   - PySCF might use different sign convention for z-vector or orbital response
   - The `-` sign in the Xvo formula might need to be `+`

4. **Direct RHS comparison**:
   - Add environment variable to dump vhf_mo and imat_term separately
   - Compare individual components with PySCF breakdown

## References

- Investigation document: `RMP2_GRADIENT_INVESTIGATION.md`
- Current status: `CURRENT_INVESTIGATION_STATUS.md`
- Verification script: `tests/benchmarks/mp2/pyscf_reference/verify_dm2buf_with_correct_geom.py`
- Comparison script: `tests/benchmarks/mp2/pyscf_reference/compare_cphf_matrix.py`

## Conclusion

The dm2buf_full formula is correct and not the source of the ~27% gradient error. The issue is in the RHS assembly where signs don't match PySCF. The fix requires detailed comparison of the individual RHS components (vhf_mo and imat_term) to identify which has the wrong sign.

The current code changes are safe to keep for the debugging infrastructure they add, but more investigation is needed to actually fix the sign issue.
