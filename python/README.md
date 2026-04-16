# ccgen

A symbolic coupled-cluster equation generator that derives spin-orbital CC residual equations at arbitrary truncation order. Starting from the normal-ordered Hamiltonian and a cluster operator, ccgen performs the Baker-Campbell-Hausdorff (BCH) expansion, applies Wick's theorem to evaluate Fermi-vacuum matrix elements, canonicalizes the resulting tensor expressions, and emits production-ready code in multiple backend formats.

## Features

- **Arbitrary truncation**: CCD, CCSD, CCSDT, CCSDTQ, and beyond (up to CC6)
- **Multiple emission backends**: human-readable equations, NumPy einsum, and C++ loop nests
- **Algebraic optimizations**: orbital energy denominator collection, permutation-based term grouping, implicit antisymmetry exploitation
- **Intermediate tensor detection**: automatic extraction of reusable sub-contractions (W_oovv, F_ov, etc.) with memory layout and blocking hints
- **Extended contraction IR**: backend-neutral IR with BLAS/GEMM pattern detection, FLOP estimates, and tiling hints
- **C++ code generation tiers**: naive loops, tiled loops with OpenMP, and BLAS/GEMM-lowered output
- **Pipeline instrumentation**: term count and timing statistics at every stage via `PipelineStats`
- **Benchmarking**: built-in `bench` module for profiling generation with and without optimizations

## Theory

ccgen implements the standard CC equation derivation pipeline:

1. **Normal-ordered Hamiltonian** -- constructs `H_N = F_N + V_N` in second quantization, where `F_N` is the one-body Fock operator and `V_N` is the antisymmetrized two-electron interaction.

2. **Cluster operator** -- builds `T = T1 + T2 + ... + Tn` for any truncation level. Each `Tn` carries the correct `(1/n!)^2` prefactor for antisymmetric summation over `n` occupied and `n` virtual dummy indices.

3. **BCH expansion** -- computes the similarity-transformed Hamiltonian via the finite BCH series:

   ```
   H-bar = H + [H,T] + 1/2![[H,T],T] + 1/3![[[H,T],T],T] + 1/4![[[[H,T],T],T],T]
   ```

   The series terminates at fourth order because `H` is a two-body operator and each commutator with `T` reduces the operator rank.

4. **Wick contraction** -- evaluates `<Phi_0| ... |Phi_0>` and `<Phi_{ij...}^{ab...}| ... |Phi_0>` matrix elements by enumerating all fully-contracted Wick pairings, respecting Fermi statistics (sign tracking from operator transpositions) and the normal-ordered vacuum.

5. **Projection** -- projects `H-bar` onto the reference (`<Phi_0|` for energy) and excited determinants (`<Phi_i^a|` for singles, `<Phi_{ij}^{ab}|` for doubles, etc.) to obtain the CC energy and residual equations.

6. **Canonicalization** -- normalizes each term by sorting tensor indices within antisymmetry groups, relabeling dummy indices to a canonical first-appearance convention, and merging terms with identical tensor structure.

7. **Connectivity filtering** -- uses union-find on the contraction graph to identify and discard disconnected terms from amplitude equations (the linked-cluster theorem).

8. **Algebraic optimization** (optional) -- post-canonicalization passes that reduce term count:
   - **Orbital energy denominator collection**: collects diagonal Fock terms like `-f(i,i)*t2 + f(a,a)*t2` into denominator tensors `D(i,a)*t2`
   - **Permutation-based term grouping**: merges terms differing only by index permutations and sign
   - **Implicit antisymmetry exploitation**: emits only unique index orderings using `v(p,q,r,s) = -v(q,p,r,s)` (experimental)

9. **Intermediate extraction** (optional) -- detects sub-contractions appearing in multiple residual terms and extracts them as reusable intermediate tensors with memory layout and blocking hints.

## Installation

```bash
cd python
pip install -e .
```

Requires Python 3.10+. No external dependencies for the core package. Optional extras:

```bash
pip install -e ".[optimize]"   # adds opt_einsum for contraction path optimization
pip install -e ".[test]"       # adds numpy + opt_einsum for regression tests
```

## Usage

### Generate equations

```python
from ccgen import generate_cc_equations

# CCSD energy + singles + doubles residuals
eqs = generate_cc_equations("ccsd")

for manifold, terms in eqs.items():
    print(f"{manifold}: {len(terms)} terms")
# energy: 3 terms
# singles: 9 terms
# doubles: 17 terms
```

The `method` argument accepts any standard CC level string:

| Input | Cluster operator | Manifolds |
|-------|-----------------|-----------|
| `"ccd"` | T2 | energy, doubles |
| `"ccsd"` | T1 + T2 | energy, singles, doubles |
| `"ccsdt"` | T1 + T2 + T3 | energy, singles, doubles, triples |
| `"ccsdtq"` | T1 + T2 + T3 + T4 | energy, singles, doubles, triples, quadruples |
| `"cc6"` | T1 + ... + T6 | energy, singles, ..., sextuples |

### Algebraic optimizations

```python
# Collect diagonal Fock terms into energy denominators
eqs = generate_cc_equations("ccsd", collect_denominators=True)

# Apply permutation-based term grouping
eqs = generate_cc_equations("ccsd", permutation_grouping=True)

# Experimental: exploit implicit antisymmetry
eqs = generate_cc_equations("ccsd", exploit_symmetry=True)

# Enable pipeline instrumentation (prints to stderr, stores in last_stats)
eqs = generate_cc_equations("ccsd", debug=True)
```

### Pretty-print

```python
from ccgen.generate import print_equations

print(print_equations("ccsd"))
```

Output:

```
E_CC() =
  + sum(i,a) f(i,a) t1(a,i)
  + 1/4 sum(i,j,a,b) t2(a,b,i,j) v(i,j,a,b)
  + 1/2 sum(i,j,a,b) t1(a,i) t1(b,j) v(i,j,a,b)

R1(i,a) =
  ...
```

### Pretty-print with intermediates

```python
from ccgen.generate import print_equations_full

# Includes: intermediate definitions, index legend, section headers, statistics
print(print_equations_full("ccsd", intermediate_threshold=5))
```

### NumPy einsum output

```python
from ccgen.generate import print_einsum

print(print_einsum("ccsd"))

# With opt_einsum contraction path optimization (requires opt_einsum)
print(print_einsum("ccsd", use_opt_einsum=True))
```

Output:

```python
E_CC = 0.0
E_CC += np.einsum('ia,ai->',F, T1)
E_CC += 1/4 * np.einsum('ijab,abij->',V, T2)
...
```

### C++ loop-nest output

Three tiers of C++ emission are available:

```python
from ccgen.generate import print_cpp, print_cpp_optimized, print_cpp_blas

# Tier 1: Naive loop nests (readable, no optimization)
print(print_cpp("ccsd"))

# Tier 2: Tiled loops with OpenMP parallelization
print(print_cpp_optimized("ccsd", tile_occ=16, tile_vir=16, use_openmp=True))

# Tier 3: BLAS/GEMM lowering (emits cblas_dgemm calls where patterns match)
print(print_cpp_blas("ccsd", use_blas=True, use_openmp=True))
```

Tier 1 output:

```cpp
// Auto-generated by ccgen
// Spin-orbital coupled-cluster residual equations

#include <cstddef>

void compute_cc_residuals(
    const int no, const int nv,
    /* tensor accessors: F, V, E_CC, R1, R2 */
) {

// E_CC residual (3 terms)
// Free indices:

// Term 1
for (int i = 0; i < no; ++i)
for (int a = 0; a < nv; ++a)
{
    double acc = 0.0;
        acc += F(i,a) * T1(a,i);
    E_CC() += acc;
}
...
}
```

### Contraction IR

For building custom backends, lower the equations to an intermediate representation that exposes tensor slot metadata:

```python
from ccgen import generate_cc_contractions

ir = generate_cc_contractions("ccsd")
for term in ir["energy"]:
    print(term.lhs_name, term.coefficient, term.rhs_factors)
    for c in term.contractions:
        print(f"  {c.index.name} ({c.index.space}): "
              f"{'free' if c.is_free else 'summed'}, "
              f"{len(c.slots)} slots")
```

### Extended contraction IR with optimization hints

```python
from ccgen.generate import generate_cc_contractions_ex

ir_ex = generate_cc_contractions_ex("ccsd", detect_blas=True, tile_occ=16, tile_vir=16)
for term in ir_ex["doubles"]:
    print(f"  flops={term.estimated_flops}, blas={term.blas_hint}")
    print(f"  layout={term.memory_layout}, blocking={term.blocking_hint}")
```

### Selective targets

Generate equations for specific manifolds only:

```python
eqs = generate_cc_equations("ccsdt", targets=["energy", "doubles"])
```

### Pipeline instrumentation

```python
import ccgen.generate as gen

eqs = gen.generate_cc_equations("ccsd", debug=True)
print(gen.last_stats.summary())
```

Output:

```
CCSD generation:
  After BCH expansion:      ... terms  (0.XXs)
  energy:
    After Wick projection:  ... terms  (0.XXs)
    After canonicalization: ... terms  (0.XXs)
    After merge_like_terms: ... terms
  singles:
    ...
```

### Benchmarking

```bash
cd python
python -m ccgen.bench                          # default: CCD, CCSD
python -m ccgen.bench --methods ccsd ccsdt     # specific methods
python -m ccgen.bench --methods ccsd --timing --json   # JSON output
```

### CLI script

A convenience script is provided for quick code generation:

```bash
cd python
python generate_ccsdt_cpp.py ccsdt                 # CCSDT C++ to stdout
python generate_ccsdt_cpp.py ccsd -o ccsd.cpp       # CCSD C++ to file
python generate_ccsdt_cpp.py ccsdt --pretty          # human-readable form
python generate_ccsdt_cpp.py ccd --einsum            # numpy einsum form
python generate_ccsdt_cpp.py ccsd --ir               # tensor contraction IR
```

## Package structure

```
ccgen/
  __init__.py          Public API: generate_cc_equations, generate_cc_contractions
  generate.py          Top-level driver: BCH -> Wick -> canonicalize -> optimize -> emit
  hamiltonian.py       Normal-ordered Hamiltonian builder (F_N + V_N)
  cluster.py           Cluster operator builder (T1..Tn) and CC level parser
  algebra.py           Symbolic multiplication, commutator, BCH expansion
  expr.py              OpTerm / Expr data types (coefficient + tensors + SQ ops)
  sqops.py             Second-quantized creation/annihilation operators
  indices.py           Typed orbital indices (occ/vir/gen), canonical relabeling
  tensors.py           Tensor symbols with antisymmetry metadata (f, v, t1, t2, ...)
  wick.py              Wick contraction engine (fully-contracted pairings)
  project.py           Projection onto excitation manifolds (energy, singles, ...)
  connectivity.py      Union-find connected-component analysis
  canonicalize.py      Antisymmetry normalization, dummy relabeling, term merging,
                       orbital energy denominator collection (collect_fock_diagonals)
  tensor_ir.py         Backend-neutral contraction IR: BackendTerm (basic) and
                       BackendTermEx (extended with BLASHint, FLOP estimates, tiling)
  bench.py             Performance benchmarking (python -m ccgen.bench)
  emit/
    pretty.py          Human-readable equation formatter (with intermediate support)
    einsum.py          NumPy einsum code emitter (with opt_einsum integration)
    cpp_loops.py       C++ emitter: naive, tiled+OpenMP, and BLAS/GEMM-lowered
  optimization/
    __init__.py        Optimization pass registry
    intermediates.py   Intermediate tensor detection, rewriting, and layout hints
    subexpression.py   Common subexpression elimination (CSE)
    permutation.py     Permutation-based term grouping
    symmetry.py        Implicit antisymmetry exploitation (experimental)
  methods/
    ccd.py             CCD convenience driver
    ccsd.py            CCSD convenience driver
    ccsdt.py           CCSDT convenience driver
  tests/
    test_regressions.py    Equation stability, numerical validation, slot tracking
    test_optimizations.py  Algebraic equivalence tests for optimization passes
```

## Data flow

```
build_hamiltonian()          build_cluster("ccsd")
        |                            |
     H = F_N + V_N              T = T1 + T2
        |                            |
        +------- bch_expand ---------+
                     |
               H-bar (Expr)
                     |
              project(H-bar, manifold)
                     |
            Wick contraction + delta resolution
                     |
              [AlgebraTerm, ...]
                     |
         canonicalize_term + merge_like_terms
                     |
           canonical equations
                     |
         ┌───────────┼───────────────────────┐
         │    optional optimization passes    │
         │                                    │
         │  collect_fock_diagonals            │
         │  apply_permutation_grouping        │
         │  exploit_antisymmetry              │
         │  detect_intermediates              │
         └───────────┼────────────────────────┘
                     │
                optimized equations
               /     |     \        \
         pretty   einsum  cpp_loops  cpp_blas    (emitters)
                     |
              lower_equations           (basic contraction IR)
              lower_equations_ex        (extended IR: BLAS, FLOP, tiling)
```

## Testing

```bash
cd python
python -m pytest ccgen/tests/ -v
```

The test suite validates:

- **Equation stability**: repeated generation produces identical term counts
- **Numerical correctness**: generated CCSD energy evaluated with random tensors matches the analytic reference formula (and optionally PySCF's `gccsd.energy`)
- **Contraction lowering**: tensor slot tracking preserves index metadata
- **Antisymmetry**: no terms with repeated indices in antisymmetric slots
- **Dummy relabeling**: canonical relabeling avoids free-index name collisions
- **Optimization equivalence**: each algebraic optimization pass preserves numerical equivalence against the unoptimized baseline

## Integration with Planck

ccgen is designed to generate optimized C++ code targeting Planck's coupled-cluster tensor infrastructure (`Tensor2D`/`Tensor4D`/`Tensor6D` in `src/post_hf/cc/`). The planned integration adds a `TensorOptimized` solver path alongside the existing teaching and production backends. See `CCGEN_DEVELOPMENT_PLAN.md` for the full roadmap.

## License

MIT
