# Dressed Intermediates in CCGEN

## Current State

The emitters in `ccgen` (C++, einsum, pretty) emit **flat equations**: each residual term is computed directly from base tensors (`F`, `V`, `T1`, `T2`, `T3`) with explicit nested loops. No intermediate tensors are built or reused.

**Example (CCSD doubles)**:
```cpp
// Term 42: many loops for t1*t1*v products
for (int i = 0; i < no; ++i)
for (int a = 0; a < nv; ++a)
{
    double acc = 0.0;
    for (int k = 0; k < no; ++k)
    for (int c = 0; c < nv; ++c)
        acc += T1(a, i) * T1(c, k) * V(a, c, i, k);
    R2(i, a) += 0.125 * acc;
}

// Term 43: identical inner loop structure
for (int i = 0; i < no; ++i)
for (int a = 0; a < nv; ++a)
{
    double acc = 0.0;
    for (int k = 0; k < no; ++k)
    for (int c = 0; c < nv; ++c)
        acc += T1(b, i) * T1(c, k) * V(b, c, i, k);  // ← same loop, different T1 index
    R2(i, a) += 0.25 * acc;
}
```

## Motivation

Building dressed intermediates would:

1. **Eliminate redundant contractions**: Many terms share the same sub-contraction (e.g., all `T1*T1*V(occ,vir,occ,vir)` products)
2. **Reduce memory traffic**: Intermediate tensors are reused across multiple residual terms
3. **Improve code readability**: Equations become more modular and match textbook presentations
4. **Enable caching strategies**: Intermediates can be computed once and cached across multiple uses
5. **Facilitate parallelization**: Each intermediate build can be parallelized independently

## Intermediate Taxonomy

### Energy-Weighted (Dressed) vs. Plain

**Plain intermediates** (standard in CC):
```
W_oovv(i,j,a,b) = V(i,j,a,b) + P(ab) T1(a,k) V(i,j,k,b) + ...
```
These are non-linear in amplitudes; they appear in residual equations.

**Energy-weighted (effective) intermediates** (MRCC-style):
```
W_oovv^eff(i,j,a,b) = W_oovv(i,j,a,b) / (ε_i + ε_j - ε_a - ε_b)
```
Less common in spin-orbital CC, but used in some formulations.

### Index Space Naming

Standard CC intermediate notation:
- `Xoooo` — occ-occ-occ-occ (e.g., derivative of H_eff w.r.t. occupied orbitals)
- `Xooov` — occ-occ-occ-vir
- `Xoovv` — occ-occ-vir-vir
- `Xovvv` — occ-vir-vir-vir
- `Xvvvv` — vir-vir-vir-vir

For CCSD, common intermediates:
- `F_oo(i,j)` — one-electron occupied-occupied effective Fock
- `F_vv(a,b)` — one-electron virtual-virtual effective Fock
- `F_ov(i,a)` — one-electron occupied-virtual effective Fock
- `W_oovv(i,j,a,b)` — two-electron intermediate
- `W_ovov(i,a,j,b)` — direct/exchange intermediate

For CCSDT, additionally:
- `Vint_oooo`, `Vint_vvvv` — two-electron integrals transformed by doubles
- `X_ooov`, `X_ovvv` — three-body effective vertices

## Implementation Strategy

### Stage 1: Intermediate Detection (Symbolic)

Analyze the generated equations to identify common sub-expressions:

```python
def detect_intermediates(equations: dict[str, list[AlgebraTerm]]) -> list[IntermediateSpec]:
    """Find sub-expressions that appear in multiple terms."""
    subexpr_to_terms: dict[SubexprSignature, list[tuple[str, AlgebraTerm]]] = {}
    
    for target, terms in equations.items():
        for term in terms:
            # Extract all contraction sub-trees
            for subexpr in extract_subexpressions(term):
                sig = signature(subexpr)
                if sig not in subexpr_to_terms:
                    subexpr_to_terms[sig] = []
                subexpr_to_terms[sig].append((target, term))
    
    # Keep sub-expressions that appear multiple times
    intermediates = []
    for sig, occurrences in subexpr_to_terms.items():
        if len(occurrences) >= 2:  # Heuristic: worth extracting if ≥2 uses
            intermediates.append(IntermediateSpec(
                name=suggest_name(sig),
                definition=sig,
                usage_count=len(occurrences),
            ))
    
    return intermediates
```

### Stage 2: Intermediate Definitions

Generate new equations for intermediate tensors:

```python
@dataclass(frozen=True)
class IntermediateSpec:
    """Specification for a dressed intermediate tensor."""
    
    name: str                  # "W_oovv", "F_ov", etc.
    indices: tuple[Index, ...] # (i, j, a, b)
    definition: AlgebraTerm    # the contraction that computes it
    usage_count: int           # number of residual terms that use it
    
    def build_equation(self) -> str:
        """Generate the computation of this intermediate."""
        # Returns code like: W_oovv(i,j,a,b) += V(...) + T1(...)*V(...)
        ...
```

### Stage 3: Residual Rewriting

Replace sub-expressions in residual equations with intermediate references:

**Before**:
```cpp
for (int i = 0; i < no; ++i)
for (int j = 0; j < no; ++j)
for (int a = 0; a < nv; ++a)
for (int b = 0; b < nv; ++b)
{
    double acc = 0.0;
    for (int k = 0; k < no; ++k)
    for (int c = 0; c < nv; ++c)
        acc += V(i,j,a,b) + T1(c,k)*V(i,j,k,b);
    R2(i, a) += 0.25 * acc;
}
```

**After**:
```cpp
// Intermediate definition (computed once, before residuals)
for (int i = 0; i < no; ++i)
for (int j = 0; j < no; ++j)
for (int a = 0; a < nv; ++a)
for (int b = 0; b < nv; ++b)
{
    W_oovv(i, j, a, b) = V(i, j, a, b);
    double acc = 0.0;
    for (int k = 0; k < no; ++k)
    for (int c = 0; c < nv; ++c)
        acc += T1(c, k) * V(i, j, k, b);
    W_oovv(i, j, a, b) += acc;
}

// Residual uses intermediate
for (int i = 0; i < no; ++i)
for (int j = 0; j < no; ++j)
for (int a = 0; a < nv; ++a)
for (int b = 0; b < nv; ++b)
{
    R2(i, j, a, b) += 0.25 * W_oovv(i, j, a, b);
}
```

### Stage 4: Code Generation

Emit three sections:

1. **Intermediate definitions** (top of function)
2. **Residual computations** (using intermediates)
3. **Accessor functions** (for memory layout, zeroing, etc.)

```cpp
void compute_cc_residuals_with_intermediates(
    const int no, const int nv,
    /* accessors: F, V, T1, T2, T3 */
    /* intermediates: W_oovv, F_ov, F_oo, F_vv */
) {
    // ── Allocate intermediates ──
    double* W_oovv = malloc(no*no*nv*nv * sizeof(double));
    double* F_ov = malloc(no*nv * sizeof(double));
    // ...
    
    // ── Build intermediates ──
    build_W_oovv(no, nv, W_oovv, F, V, T1, T2);
    build_F_ov(no, nv, F_ov, F, V, T1);
    // ...
    
    // ── Compute residuals using intermediates ──
    for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
    for (int a = 0; a < nv; ++a)
    for (int b = 0; b < nv; ++b)
        R2(i, j, a, b) += 0.25 * W_oovv[i*no*nv*nv + j*nv*nv + a*nv + b];
    
    // ── Free intermediates ──
    free(W_oovv); free(F_ov);
}
```

## CCSD Example: Intermediate Building

For CCSD, the standard set is:

| Intermediate | Definition | Terms |
|--------------|-----------|-------|
| `F_oo(i,j)` | `f(i,j) + Σ_k f(i,k) T1(a,k) + ...` | 4–6 terms |
| `F_vv(a,b)` | `f(a,b) + Σ_c T1(a,c) f(c,b) + ...` | 4–6 terms |
| `F_ov(i,a)` | `f(i,a) + Σ_jk T1(i,j) V(i,a,j,k) + ...` | 3–4 terms |
| `W_oovv(ijab)` | `V(i,j,a,b) + Σ_k T1(a,k) V(i,j,k,b) + ...` | 6–8 terms |
| `W_ovov(iajb)` | `V(i,a,j,b) - P(ij) V(i,a,k,b) T1(k,j) + ...` | 4–6 terms |

**Impact**: Instead of computing all 320–350 CCSD doubles terms from base tensors, you'd compute ~20–30 intermediate terms once, then use them to build all residuals. For large problems (no=100, nv=200), this could save 50–70% of the contractions.

## API Design

```python
# In generate.py
def generate_cc_residuals_with_intermediates(
    method: str,
    targets: list[str] | None = None,
    connected_only: bool = True,
    use_intermediates: bool = True,  # NEW
) -> tuple[dict[str, list[AlgebraTerm]], list[IntermediateSpec]]:
    """Generate equations and optionally extract intermediates."""
    eqs = generate_cc_equations(method, targets, connected_only)
    
    if use_intermediates:
        intermediates = detect_intermediates(eqs)
        eqs_rewritten = rewrite_with_intermediates(eqs, intermediates)
        return eqs_rewritten, intermediates
    else:
        return eqs, []

# In emit/cpp_loops.py
def emit_with_intermediates(
    equations: dict[str, list[AlgebraTerm]],
    intermediates: list[IntermediateSpec],
) -> str:
    """Emit C++ with intermediate tensor definitions."""
    lines = [...]
    
    # Build intermediate definitions
    for inter in intermediates:
        lines.append(emit_intermediate_definition(inter))
    
    # Build residuals using intermediates
    for target, terms in equations.items():
        lines.append(emit_residual_using_intermediates(target, terms, intermediates))
    
    return "\n".join(lines)
```

## Integration Points

1. **Canonicalization**: Intermediate detection should happen *after* merging like terms
2. **Emission**: Both C++ and einsum backends need support
3. **Memory management**: Discuss allocation strategy (stack vs. malloc vs. external)
4. **Testing**: Compare unoptimized vs. intermediate versions; ensure numerical equivalence

## Performance Expectations

For CCSD with no=50, nv=100:
- **Without intermediates**: 320 full term evaluations, many with redundant inner loops
- **With intermediates**: ~25 intermediate builds + 320 residual lookups (no additional loops)
- **Expected speedup**: 2–3× for tensor contraction phase

For CCSDT: larger speedups (5–10×) due to cubic scaling of term count.

## Open Questions

1. Should intermediate allocation be on the stack (small tensors), heap (larger), or passed in externally?
2. How aggressively should we extract intermediates? (e.g., minimum reuse threshold)
3. Should we support selective intermediate extraction (user picks which ones)?
4. How to handle memory layout and BLAS integration (e.g., W_oovv in batched GEMM)?

