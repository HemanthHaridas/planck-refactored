### CCGEN Teaching Guide

A complete theory-to-code walkthrough of the `python/ccgen` coupled-cluster
equation generator. Intended for students learning symbolic second quantization,
contributors extending the generator, and developers who need to audit how a
named method like CCD or CCSD turns into explicit tensor contractions.

### 1. What `ccgen` Is

`ccgen` is a symbolic algebra package that derives spin-orbital
coupled-cluster equations from second-quantized operators. It lives in
`python/ccgen` and is intentionally separate from Planck's production C++
solvers. The package is meant to be:

- easy to inspect term by term
- mathematically close to textbook CC derivations
- deterministic enough to regression test
- simple enough that new methods can be added without touching the whole stack

The package handles the following workflow:

1. build the normal-ordered Hamiltonian \(H_N = F_N + V_N\)
2. build a cluster operator \(T = T_1 + T_2 + \cdots\)
3. expand the similarity-transformed Hamiltonian
   \(\bar H = e^{-T} H e^T\) through the finite BCH series
4. project \(\bar H\) onto the reference or excitation manifolds
5. apply Wick contractions with Hartree-Fock vacuum rules
6. canonicalize and merge equivalent tensor terms
7. optionally apply algebraic optimization passes (denominator collection,
   permutation grouping, antisymmetry exploitation)
8. optionally detect and extract intermediate tensors (W_oovv, F_ov, etc.)
9. lower the result into an explicit contraction IR (basic or extended with
   BLAS hints, FLOP estimates, and tiling)
10. emit readable text, `einsum` (with optional `opt_einsum` path optimization),
    or C++ loop nests (naive, tiled+OpenMP, or BLAS/GEMM-lowered)

The symbolic derivation layer is intentionally pedagogical. The optimization
and code generation layers are designed to produce production-quality output.

### 2. Architecture Overview

### Data Flow

```text
Named method string ("ccd", "ccsd", "ccsdt")
  → parse excitation ranks (cluster.py)
  → build H and T operator expressions (hamiltonian.py, cluster.py)
  → formal BCH expansion (algebra.py)
  → projector selection (project.py)
  → Wick contraction + delta substitution (wick.py)
  → connectivity filtering (connectivity.py)
  → canonicalization + duplicate merge (canonicalize.py)
  → optional optimization passes:
      → orbital energy denominator collection (canonicalize.py)
      → permutation-based term grouping (optimization/permutation.py)
      → implicit antisymmetry exploitation (optimization/symmetry.py)
  → algebraic equations (generate.py)
  → optional intermediate detection (optimization/intermediates.py)
  → contraction IR lowering:
      → basic: BackendTerm (tensor_ir.py)
      → extended: BackendTermEx with BLASHint, FLOP estimates (tensor_ir.py)
  → emitters:
      → readable text with intermediates (emit/pretty.py)
      → einsum with opt_einsum paths (emit/einsum.py)
      → C++ naive / tiled+OpenMP / BLAS-lowered (emit/cpp_loops.py)
```

### Directory Summary

| File / Directory | Purpose |
|---|---|
| `python/ccgen/indices.py` | typed orbital indices, naming pools, canonical dummy relabeling |
| `python/ccgen/tensors.py` | tensor symbols and antisymmetry metadata |
| `python/ccgen/sqops.py` | creation / annihilation operators |
| `python/ccgen/expr.py` | formal operator monomials and sums |
| `python/ccgen/hamiltonian.py` | normal-ordered \(F_N\) and \(V_N\) |
| `python/ccgen/cluster.py` | builds \(T_1, T_2, T_3, \dots\) from method names |
| `python/ccgen/algebra.py` | multiplication, commutators, BCH expansion |
| `python/ccgen/wick.py` | Wick pairing topology and Kronecker-delta substitution |
| `python/ccgen/project.py` | bra projectors and projection onto energy / residual manifolds |
| `python/ccgen/connectivity.py` | connected-diagram filtering |
| `python/ccgen/canonicalize.py` | tensor antisymmetry normalization, dummy relabeling, duplicate merge, orbital energy denominator collection |
| `python/ccgen/tensor_ir.py` | contraction IR: `BackendTerm` (basic) and `BackendTermEx` (extended with `BLASHint`, FLOP estimates, memory layout, tiling hints) |
| `python/ccgen/generate.py` | top-level user API and `PipelineStats` instrumentation |
| `python/ccgen/bench.py` | performance benchmarking (`python -m ccgen.bench`) |
| `python/ccgen/emit/` | emitters: pretty (with intermediates), einsum (with opt_einsum), C++ (naive / tiled+OpenMP / BLAS-lowered) |
| `python/ccgen/optimization/` | post-canonicalization passes: intermediate detection, CSE, permutation grouping, antisymmetry exploitation |
| `python/ccgen/methods/` | small wrappers like `build_ccsd_equations()` |
| `python/ccgen/tests/` | regression tests (stability, numerical, lowering) and optimization equivalence tests |

### 3. Core Mathematical Model

The package works entirely in spin-orbital second quantization.

### Normal-Ordered Hamiltonian

The starting point is the normal-ordered Hamiltonian with respect to the HF
reference:

\[
H = F_N + V_N
\]

with

\[
F_N = \sum_{pq} f_p^q \{ a_p^\dagger a_q \}
\]

and

\[
V_N = \frac{1}{4}\sum_{pqrs} \langle pq || rs \rangle
\{ a_r^\dagger a_s^\dagger a_q a_p \}
\]

The two-electron object `v(p,q,r,s)` is treated as an antisymmetrized integral
\(\langle pq || rs \rangle\), so antisymmetry is attached directly to the tensor
symbol rather than being expanded into explicit permutation operators later.

### Cluster Operator

For a named method such as CCSD, the cluster operator is built from the
requested excitation ranks:

\[
T = T_1 + T_2
\]

with

\[
T_n = \frac{1}{(n!)^2}
\sum_{i_1\dots i_n, a_1\dots a_n}
t_{i_1\dots i_n}^{a_1\dots a_n}
a_{a_1}^\dagger \cdots a_{a_n}^\dagger
a_{i_n}\cdots a_{i_1}
\]

The reverse ordering of the annihilators is deliberate: it matches the usual
antisymmetric convention and keeps the prefactor bookkeeping simple.

### BCH Expansion

For a two-body Hamiltonian, the BCH series terminates at fourth order:

\[
\bar H = H + [H,T] + \frac{1}{2}[[H,T],T]
+ \frac{1}{6}[[[H,T],T],T]
+ \frac{1}{24}[[[[H,T],T],T],T]
\]

`ccgen` forms this as a formal operator expansion first. No contractions happen
in `algebra.py`; the result is just a large sum of operator strings.

### Projection

Once \(\bar H\) is built, the package projects onto:

- the reference for the CC correlation energy
- singles for \(R_i^a\)
- doubles for \(R_{ij}^{ab}\)
- triples for \(R_{ijk}^{abc}\), if requested

The package constructs the bra projectors explicitly as operator strings, then
performs full Wick contractions against each BCH monomial.

### 4. Core Data Structures

The most important idea in `ccgen` is that each stage has its own data model.
The code does not try to use one object for everything.

### `Index`

Defined in `indices.py`:

```python
@dataclass(frozen=True, order=True)
class Index:
    name: str
    space: str     # "occ", "vir", or "gen"
    is_dummy: bool = False
```

An `Index` does two jobs:

- it says which orbital space an index belongs to
- it says whether this occurrence is a summed dummy or a free residual label

This distinction matters because the same literal name may be interpreted
differently before and after projection. The helper constructors
`make_occ`, `make_vir`, and `make_gen` keep that explicit.

The module also owns the canonical naming pools:

- occupied: `i, j, k, l, ...`
- virtual: `a, b, c, d, ...`
- general: `p, q, r, s, ...`

The `relabel_dummies()` routine is one of the most important correctness
utilities in the package. It renames summed indices deterministically while
avoiding collisions with any free residual labels already present in the term.

### `Tensor`

Defined in `tensors.py`:

```python
@dataclass(frozen=True)
class Tensor:
    name: str
    indices: tuple[Index, ...]
    antisym_groups: tuple[tuple[int, ...], ...] = ()
```

This is the symbolic tensor factor used everywhere after operator building.
Examples:

- `f(p,q)`
- `v(p,q,r,s)`
- `t1(a,i)`
- `t2(a,b,i,j)`

The `antisym_groups` field tells canonicalization which slots may be permuted
at the cost of a sign. For `t2(a,b,i,j)`, the virtual pair `(a,b)` and occupied
pair `(i,j)` are each antisymmetric groups.

### `SQOp`

Defined in `sqops.py`:

```python
@dataclass(frozen=True, order=True)
class SQOp:
    kind: str     # "create" or "annihilate"
    index: Index
```

This is the raw ladder-operator object used before contraction. It is kept very
small on purpose. The interesting physics is not stored in the operator itself,
but in how the Wick engine allows it to contract.

### `OpTerm` and `Expr`

Defined in `expr.py`:

```python
@dataclass(frozen=True)
class OpTerm:
    coeff: Fraction
    tensors: tuple[Tensor, ...]
    sqops: tuple[SQOp, ...]
    origin: tuple[object, ...] = ()
```

```python
@dataclass
class Expr:
    terms: list[OpTerm]
```

`OpTerm` is the formal monomial used in the symbolic BCH stage. It still has
the original second-quantized operators attached. `Expr` is just a list of such
terms with lightweight arithmetic.

### `AlgebraTerm`

Defined in `project.py`:

```python
@dataclass(frozen=True)
class AlgebraTerm:
    coeff: Fraction
    factors: tuple[Tensor, ...]
    free_indices: tuple[Index, ...]
    summed_indices: tuple[Index, ...]
    connected: bool
    provenance: tuple[object, ...] = ()
```

This is the main symbolic output of the projection stage. Once a term becomes
an `AlgebraTerm`, the ladder operators are gone. What remains is:

- a coefficient
- tensor factors
- which labels are free residual indices
- which labels are summed contraction indices
- whether the contraction graph is connected

### `BackendTerm`

Defined in `tensor_ir.py`:

```python
@dataclass(frozen=True)
class BackendTerm:
    lhs_name: str
    lhs_indices: tuple[Index, ...]
    coefficient: Fraction
    rhs_factors: tuple[Tensor, ...]
    contractions: tuple[IndexContraction, ...]
    summed_indices: tuple[Index, ...]
    free_indices: tuple[Index, ...]
    connected: bool
```

This is the contraction-aware IR used by backends. The key addition is
`contractions`, which groups all tensor slots participating in a common index.
That makes the term backend-neutral: a later emitter can interpret the same IR
as loops, `einsum`, a graph, or a tensor-network plan.

### 5. Operator Construction Layer

### `hamiltonian.py`

This file builds the normal-ordered one- and two-body Hamiltonian components.

`build_fock_operator()` constructs the single term

\[
f(p,q)\, a_p^\dagger a_q
\]

using general indices because the occupied/virtual specialization is only
determined later by contractions.

`build_two_body_operator()` constructs

\[
\frac{1}{4} v(p,q,r,s)\,
a_r^\dagger a_s^\dagger a_q a_p
\]

and stores its origin as `("H", "V_N")`. The `origin` tuple is not used in the
algebra itself; it exists so that emitted or debugged terms can still be traced
back to their source blocks.

### `cluster.py`

This file has two jobs.

First, it builds \(T_n\) explicitly with `build_tn(n)`. The tensor is created by
`tn()` in `tensors.py`, then paired with the matching operator string

\[
a_{a_1}^\dagger \cdots a_{a_n}^\dagger a_{i_n}\cdots a_{i_1}
\]

Second, it parses method names. `parse_cc_level()` accepts:

- compact names like `ccd`, `ccsd`, `ccsdt`
- extended names like `ccsdtq5`
- shorthand like `cc4`

The output is always a sorted list of ranks. `build_cluster("ccsd")` therefore
just builds \(T_1 + T_2\).

### 6. Formal Algebra Layer

### `algebra.py`

This module is intentionally formal. It does not know anything about the HF
vacuum and does not do any contractions.

### Multiplication

`multiply_terms(a, b)` concatenates:

- the coefficients
- the tensor lists
- the operator strings

Before concatenation, it renames any dummy indices in `b` that would collide
with indices already present in `a`. This is essential because BCH nesting
otherwise reuses the same `i`, `a`, `p`, and `q` labels in logically unrelated
subexpressions.

### Commutator

`commutator(a, b)` is exactly:

\[
[A,B] = AB - BA
\]

with no further simplification.

### BCH

`bch_expand(H, T, max_order=4)` repeatedly nests the commutator and applies the
factor \(1/n!\) at each level. The output is still an `Expr`, not a tensor
equation.

This separation is a deliberate design choice. It keeps the package readable:

- `algebra.py` only knows formal operator algebra
- `wick.py` only knows contractions
- `canonicalize.py` only knows tensor normalization

### 7. Wick Contraction Layer

### HF Vacuum Rules

The contraction rules in `wick.py` encode the Hartree-Fock vacuum explicitly.
The package uses the following logic:

- a creator can contract with an annihilator only in occupied/general space
- an annihilator can contract with a creator only in virtual/general space

This is the operator-level expression of the reference determinant:

- occupied orbitals are filled
- virtual orbitals are empty

So a contraction like \(a_i^\dagger a_j\) may survive, while
\(a_a^\dagger a_b\) does not.

### Structural Pairing Cache

The expensive part of symbolic Wick expansion is not actually the delta
substitution; it is repeatedly enumerating the same pairing topology for many
operator strings of identical shape. `ccgen` therefore caches pairing patterns
by a structural signature:

```text
(position, op kind, index space, block id)
```

This means the recursive pairing search is paid once per operator-pattern class
rather than once per concrete BCH term.

### `WickResult`

For each complete contraction pattern, the Wick engine returns:

- the fermionic sign
- the list of Kronecker-delta identifications
- the original tensor factors
- the graph edges connecting source blocks

The actual tensor substitution happens later in `apply_deltas()`.

### Delta Substitution

`apply_deltas()` runs a small union-find over identified indices. It enforces:

- space consistency
- replacement of general indices by occupied/virtual ones where possible
- deterministic root choice inside a shared index class

If a delta would force an occupied index equal to a virtual one, the term is
discarded as zero.

### 8. Projection Layer

### Bra Projectors

`project.py` defines projectors for:

- reference: `()`
- singles: \(\langle \Phi_i^a | = \langle \Phi_0 | a_i^\dagger a_a\)
- doubles: \(\langle \Phi_{ij}^{ab} | = \langle \Phi_0 | a_i^\dagger a_j^\dagger a_b a_a\)
- triples: analogous

The general helper `nfold_projector(n)` constructs the rank-\(n\) case.

### Block IDs and Connectivity

Projection does not merely combine operators. It also assigns each operator to
a block so the later connectivity analysis can ask whether a contraction is
connected across:

- the bra projector block
- each tensor/operator factor block from the original term

In practice this matters because amplitude residuals should keep only connected
terms, while energy expressions may include disconnected products like
\(f_{ia} t_i^a\).

### Free-Index Restoration

After delta substitution, a projected free label may temporarily appear as a
dummy-like occupied or virtual index with the same slot name. `_restore_free_indices()`
maps these back onto the canonical projector labels so the residual signature is
stable and later merge logic does not mistake a free index for a summed one.

### `project()`

The projection routine:

1. selects the correct bra projector
2. rejects operator strings with unmatched creation/annihilation counts
3. asks `wick_contract()` for all admissible pairings
4. applies deltas
5. checks connectedness
6. separates free and summed indices
7. emits `AlgebraTerm` objects

At this point the terms are mathematically correct but not yet canonical.

### 9. Canonicalization Layer

This module is where structurally equivalent algebraic terms are forced into a
single normal form.

### Tensor Antisymmetry

`canonicalize_tensor()` inspects each antisymmetry group. Two things happen:

1. if an antisymmetric group repeats an index, the tensor is zero
2. otherwise the indices are permuted into lexicographically smallest order,
   with the sign updated by the parity of the permutation

This is how terms like `t2(a,a,i,j)` are eliminated automatically.

### Dummy Relabeling

After tensor-local normalization, `relabel_term_dummies()` walks the term in
first-appearance order and assigns canonical names to summed indices.

This solves the standard symbolic-equivalence problem:

\[
\sum_{kc} v_{ak}^{ic} t_{kj}^{cb}
\equiv
\sum_{ld} v_{al}^{id} t_{lj}^{db}
\]

### Sorting and Merge

`canonicalize_term()` does the full pipeline:

1. canonicalize each tensor
2. zero out any term containing an antisymmetric repeat
3. relabel dummies once before sorting so temporary fresh suffixes cannot affect order
4. sort factors by tensor name and index tuple
5. relabel dummies again so the final term uses first-appearance ordering

`merge_like_terms()` then buckets by tensor structure and free-index signature,
adding coefficients when the terms are identical.

### 10. Connectivity Analysis

`connectivity.py` is small but conceptually important.

Each Wick contraction adds an edge between the two blocks participating in that
contraction. The package then runs a union-find connectivity test. For residual
equations, the bra projector block is ignored and the remaining contraction
graph must be connected. This matches the standard connected-cluster working
equations.

### 11. Algebraic Optimization Layer

After canonicalization and merging, several optional optimization passes can
reduce term count or restructure equations for more efficient code generation.
These live in `python/ccgen/optimization/` and are wired into
`generate_cc_equations()` via keyword flags.

### Orbital Energy Denominator Collection

`collect_fock_diagonals()` in `canonicalize.py` is a post-canonicalization pass
that recognizes diagonal Fock terms and collects them into energy denominator
tensors:

```text
Before:  -1/4 f(i,i) * t2(a,b,i,j)  +  1/4 f(a,a) * t2(a,b,i,j)  + ...
After:   D(i,j,a,b) * t2(a,b,i,j)    where D = ε_a + ε_b - ε_i - ε_j
```

The C++ emitter recognizes `D` tensors and emits `(eps(a) + eps(b) - eps(i) - eps(j))`
inline. This typically reduces CCSD term count by ~10-20%.

### Permutation-Based Term Grouping

`apply_permutation_grouping()` in `optimization/permutation.py` detects groups
of terms that differ only by index permutations (possibly with sign changes from
antisymmetry) and merges them. For example, two terms differing only by `a↔b`
in an antisymmetric integral are combined with a factor of 2. This reduces loop
nests by 30-40% in typical CCSD/CCSDT equations.

### Implicit Antisymmetry Exploitation

`exploit_antisymmetry()` in `optimization/symmetry.py` is an experimental pass
that exploits `v(p,q,r,s) = -v(q,p,r,s)` before canonicalization to emit only
unique index orderings, potentially reducing pre-canonicalization term count by
30-50%.

### Intermediate Tensor Detection

`detect_intermediates()` in `optimization/intermediates.py` scans all residual
equations for sub-contractions that appear in multiple terms and extracts them
as reusable intermediate tensors (e.g., W_oovv, F_ov). Each intermediate is
described by an `IntermediateSpec` with:

- name and index signature
- definition terms (the sub-contraction)
- usage count across residual equations
- memory layout, blocking hints, and allocation strategy (for code generation)

`annotate_layout_hints()` enriches intermediates with memory layout and blocking
metadata based on access pattern analysis.

### Common Subexpression Elimination

`optimization/subexpression.py` provides CSE detection via `CSESpec` objects.
This factors out common tensor products that appear with different coefficients.

### 12. Contraction IR Lowering

The symbolic equation objects are convenient for mathematics, but backends often
need explicit knowledge of which tensor slots share an index. That is the job
of `tensor_ir.py`.

### `TensorSlot`

One `TensorSlot` means:

- which factor in the RHS we are talking about
- which axis inside that factor
- what tensor name that factor has
- what symbolic index labels that axis

### `IndexContraction`

This groups all slots tied together by one symbolic index. For example, in the
energy term

\[
f(i,a)\, t_1(a,i)
\]

there are two `IndexContraction` objects:

- one for `i`, tying `f` axis 0 to `t1` axis 1
- one for `a`, tying `f` axis 1 to `t1` axis 0

### `BackendTerm`

This lowered form is the bridge between symbolic derivation and code generation.
It still stores the original tensor factors, but it now also says exactly how
their slots connect. This is what makes later backends possible:

- loop emitters
- `einsum` emitters
- contraction planners
- intermediate/factorization passes

Use `lower_equations()` to produce basic `BackendTerm` objects.

### `BackendTermEx`

The extended IR (`BackendTermEx`) inherits all `BackendTerm` fields and adds
optimization metadata for backend emitters:

- `memory_layout`: per-tensor storage order hints (e.g., `{"T1": "row_major"}`)
- `blocking_hint`: suggested tile sizes per index (e.g., `{"i": 16, "a": 16}`)
- `reuse_key`: identifies shared sub-contractions across terms
- `computation_order`: recommended summation index ordering
- `blas_hint`: a `BLASHint` object when the term matches a GEMM pattern (e.g.,
  `gemm_nn`, `gemm_nt`), with fields for the A/B operands, contraction indices,
  and output dimensions
- `estimated_flops`: estimated floating-point operation count

Use `lower_equations_ex()` (or `generate_cc_contractions_ex()` from the
top-level API) to produce these extended terms. The BLAS pattern detector
recognizes two-factor contractions of the form
\(\sum_k A_{ik} B_{kj}\) and emits the appropriate transpose/no-transpose hint.

### `BLASHint`

```python
@dataclass(frozen=True)
class BLASHint:
    pattern: str              # "gemm_nn" | "gemm_nt" | "gemm_tn" | "gemm_tt"
    a_tensor: str             # name of A operand
    b_tensor: str             # name of B operand
    a_indices: tuple[Index, ...]
    b_indices: tuple[Index, ...]
    contraction_indices: tuple[Index, ...]  # shared summation (k-dimension)
    m_indices: tuple[Index, ...]            # output row dimension
    n_indices: tuple[Index, ...]            # output column dimension
```

The C++ BLAS backend (`emit_blas_translation_unit`) uses these hints to emit
`cblas_dgemm` calls instead of loop nests for matching terms.

### 13. Top-Level API

`generate.py` exposes the package entry points.

### `targets_for_method()`

Turns a method name like `ccsd` into the default manifolds:

```python
["energy", "singles", "doubles"]
```

### `generate_cc_equations()`

This is the main symbolic driver. It:

1. builds `H`
2. builds `T`
3. expands `Hbar`
4. projects onto the requested targets
5. optionally filters disconnected residual terms
6. canonicalizes and merges
7. optionally applies post-canonicalization optimization passes

It returns:

```python
dict[str, list[AlgebraTerm]]
```

The optimization flags control post-canonicalization passes:

- `collect_denominators=True`: collect diagonal Fock terms into denominator tensors
- `permutation_grouping=True`: merge terms related by index permutations
- `exploit_symmetry=True`: exploit implicit antisymmetry (experimental)
- `debug=True`: print term counts and timing to stderr; store in `last_stats`

### `generate_cc_contractions()`

Runs `generate_cc_equations()` and lowers the result into basic `BackendTerm`s
via `lower_equations()`.

### `generate_cc_contractions_ex()`

Runs `generate_cc_equations()` and lowers the result into extended
`BackendTermEx` objects via `lower_equations_ex()`. Accepts additional
parameters:

- `detect_blas=True`: enable BLAS/GEMM pattern detection
- `tile_occ`, `tile_vir`: tile sizes for blocking hints

### `PipelineStats`

When `debug=True` is passed to `generate_cc_equations()`, a `PipelineStats`
object is stored in `ccgen.generate.last_stats`. It records term counts and
timing at every pipeline stage (BCH expansion, Wick projection,
canonicalization, merge, and each optimization pass). Call `.summary()` for
a human-readable report.

### Formatting Helpers

- `print_equations()` uses `emit/pretty.py` for basic readable output
- `print_equations_full()` uses `emit/pretty.py` with intermediate definitions,
  index/tensor legend, section headers, and summary statistics
- `print_einsum()` uses `emit/einsum.py`, with optional `use_opt_einsum=True`
  for contraction path optimization
- `print_cpp()` uses `emit/cpp_loops.py` for naive loop nests
- `print_cpp_optimized()` uses `emit/cpp_loops.py` for tiled loops with OpenMP
- `print_cpp_blas()` uses `emit/cpp_loops.py` for BLAS/GEMM-lowered output

### 14. Emitters

### `emit/pretty.py`

This is the human-facing printer. It emits expressions such as:

```text
R1(i,a) =
  + f(a,i)
  + 1/4 sum(j,b) f(j,b) t2(a,b,i,j)
```

This is the best format for inspecting algebraic structure while debugging.

The module also provides `format_equations_with_intermediates()`, which adds:

- intermediate tensor definitions listed separately before the residuals
- an index space legend (i, j, k = occupied; a, b, c = virtual)
- manifold section headers with descriptions
- summary statistics (term counts per manifold)

### `emit/einsum.py`

This backend turns each `AlgebraTerm` into a NumPy-style `einsum` call:

```python
R1 += 1/4 * np.einsum('jb,abij->ia', F, T2)
```

When `opt_einsum` is installed and `use_opt_einsum=True` is passed, the emitter
uses `opt_einsum.contract()` with optimized contraction paths instead of
vanilla `np.einsum()`.

The module also supports intermediate tensor emission via
`format_equations_with_intermediates_einsum()`.

### `emit/cpp_loops.py`

Three tiers of C++ emission are available in a single module:

**Tier 1: Naive loops** (`emit_translation_unit`). Straightforward nested
loops that preserve the algebra in a form a C++ developer can read line by
line. Also handles orbital energy denominator terms (emits `eps(i)` syntax).
Supports intermediate tensor builds via `emit_translation_unit_with_intermediates`.

**Tier 2: Tiled + OpenMP** (`emit_optimized_translation_unit`). Emits blocked
loops with configurable tile sizes (default 16, tuned for L1 cache ~32KB of
doubles) and `#pragma omp parallel for collapse(N)` annotations.

**Tier 3: BLAS/GEMM-lowered** (`emit_blas_translation_unit`). Detects GEMM
patterns in two-factor contractions (using `BackendTermEx.blas_hint`) and emits
`cblas_dgemm` calls. Falls back to tiled loops for non-GEMM terms. Includes
`#include <cblas.h>` and buffer allocation/deallocation.

### 15. Method Wrappers

The files in `python/ccgen/methods/` are intentionally thin.

- `ccd.py` returns energy + doubles
- `ccsd.py` returns energy + singles + doubles
- `ccsdt.py` returns energy + singles + doubles + triples

These wrappers exist mainly so users can import a named builder without needing
to know the generic driver API.

### 16. Worked Example: `generate_cc_equations("ccsd")`

Calling

```python
from ccgen.generate import generate_cc_equations
eqs = generate_cc_equations("ccsd")
```

does the following:

1. `parse_cc_level("ccsd")` returns `[1, 2]`
2. `build_hamiltonian()` returns `F_N + V_N`
3. `build_cluster("ccsd")` returns `T_1 + T_2`
4. `bch_expand()` builds the formal \(\bar H\)
5. `project(..., "energy")` generates the energy terms
6. `project(..., "singles")` generates `R1`
7. `project(..., "doubles")` generates `R2`
8. each manifold is canonicalized and merged

The current CCSD energy reduces to the familiar three-term expression:

\[
E_{CCSD} =
f_i^a t_i^a
+ \frac{1}{4}\langle ab || ij \rangle t_{ij}^{ab}
+ \frac{1}{2}\langle ab || ij \rangle t_i^a t_j^b
\]

and the test suite cross-checks that formula numerically against local PySCF.

### 17. Testing and Regression Strategy

The test suite lives in `python/ccgen/tests/` and is split across two modules.

**`test_regressions.py`** covers core correctness:

- dummy relabeling must not collide with free residual labels
- the generated CCSD energy must match the textbook / PySCF formula
- repeated calls must produce stable term counts
- no canonical term may keep repeated indices in an antisymmetric slot
- contraction lowering must preserve tensor-slot connectivity

**`test_optimizations.py`** covers algebraic equivalence for each optimization
pass:

- orbital energy denominator collection preserves numerical equivalence
- permutation grouping preserves numerical equivalence
- intermediate detection does not change residual values
- each pass is tested by generating equations with and without the
  optimization, evaluating both on small random tensors (no=4, nv=4), and
  asserting agreement to machine precision (< 1e-12)

This two-tier strategy is important because symbolic code can be "almost right"
in ways that are very hard to notice by eye. Tiny errors in sign, relabeling, or
free-index restoration can completely change a residual while still producing
something that looks plausible.

### 18. Current Limitations

- It works in spin-orbital form only (no spin-adapted or spatial-orbital formulation).
- The BCH layer still builds the full formal expansion before projection.
- Intermediate detection is heuristic-based (threshold on reuse count); it does
  not yet perform global optimal factorization.
- Implicit antisymmetry exploitation is experimental and requires extensive
  validation before production use.
- There is no direct integration with Planck's production C++ coupled-cluster
  solvers yet (the `TensorOptimized` backend is planned; see
  `CCGEN_DEVELOPMENT_PLAN.md` Phase 4).

### 19. How to Extend the Package

### To Add a New Method

1. make sure `parse_cc_level()` can express the needed excitation ranks
2. add a wrapper in `python/ccgen/methods/`
3. call `generate_cc_equations()` or `generate_cc_contractions()`
4. add regression tests for the expected manifolds

### To Add a New Backend

1. decide whether the backend wants `AlgebraTerm`, `BackendTerm`, or
   `BackendTermEx` (the extended IR with BLAS hints and tiling metadata)
2. if it needs explicit slot connectivity, consume `tensor_ir.py`
3. if it needs BLAS/GEMM pattern information, use `lower_equations_ex()`
4. add the new emitter under `python/ccgen/emit/`
5. add tests that compare emitted structure against known small cases

### To Add a New Optimization Pass

1. create a module under `python/ccgen/optimization/`
2. the pass receives `list[AlgebraTerm]` and returns `list[AlgebraTerm]`
3. wire it into `generate_cc_equations()` in `generate.py` with a keyword flag
4. update `PipelineStats` to track the pass's effect on term count
5. add equivalence tests in `tests/test_optimizations.py`: generate equations
   with and without the pass, evaluate numerically, assert equivalence to 1e-12

### To Add Intermediate Tensor Support to a Backend

1. use `detect_intermediates()` from `optimization/intermediates.py` to find
   reusable sub-contractions
2. optionally call `annotate_layout_hints()` for memory layout and blocking metadata
3. emit intermediate build code before residual equations
4. see `emit/cpp_loops.py` (`emit_translation_unit_with_intermediates`) and
   `emit/einsum.py` (`format_equations_with_intermediates_einsum`) for examples

### 20. Minimal Usage Examples

### Print CCSD Equations

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import print_equations

print(print_equations("ccsd"))
```

### Print Equations with Intermediates

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import print_equations_full

print(print_equations_full("ccsd", intermediate_threshold=5))
```

### Generate with Optimization Passes

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import generate_cc_equations

# Apply denominator collection + pipeline instrumentation
eqs = generate_cc_equations("ccsd", collect_denominators=True, debug=True)
```

### Generate Explicit Contractions

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import generate_cc_contractions

eqs = generate_cc_contractions("ccsd")
term = eqs["energy"][0]

print(term.lhs_name)
print(term.coefficient)
for contraction in term.contractions:
    print(contraction.index, contraction.slots)
```

### Generate Extended IR with BLAS Hints

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import generate_cc_contractions_ex

eqs = generate_cc_contractions_ex("ccsd", detect_blas=True)
for term in eqs["doubles"]:
    if term.blas_hint:
        print(f"GEMM: {term.blas_hint.pattern} "
              f"{term.blas_hint.a_tensor} x {term.blas_hint.b_tensor}")
    print(f"  estimated FLOPs: {term.estimated_flops}")
```

### Emit Debug `einsum`

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import print_einsum

print(print_einsum("ccd"))
```

### Emit Tiled C++ with OpenMP

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import print_cpp_optimized

print(print_cpp_optimized("ccsd", tile_occ=16, tile_vir=16, use_openmp=True))
```

### Emit C++ with BLAS/GEMM Lowering

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import print_cpp_blas

print(print_cpp_blas("ccsd", use_blas=True))
```

### 21. Mental Model for Contributors

The easiest way to understand `ccgen` is to think of it as six stacked layers:

1. operator algebra (hamiltonian, cluster, algebra)
2. Wick contraction (wick, project, connectivity)
3. tensor canonicalization (canonicalize)
4. algebraic optimization (optimization/)
5. contraction IR lowering (tensor_ir)
6. backend emission (emit/)

When debugging, always identify which layer a bug belongs to. A wrong term can
come from:

- incorrect operator construction (layer 1)
- incorrect vacuum contraction rules (layer 2)
- incorrect delta substitution (layer 2)
- incorrect free/dummy classification (layer 2)
- incorrect antisymmetry sign handling (layer 3)
- incorrect duplicate merging (layer 3)
- incorrect denominator collection or permutation grouping (layer 4)
- incorrect intermediate detection or rewriting (layer 4)
- incorrect BLAS pattern detection (layer 5)
- incorrect loop tiling or GEMM emission (layer 6)

Treating these as separate layers is the main architectural idea of the package.
Each optimization pass in layer 4 is independently toggleable and independently
tested for algebraic equivalence, so a bug in one pass does not affect the
others.
