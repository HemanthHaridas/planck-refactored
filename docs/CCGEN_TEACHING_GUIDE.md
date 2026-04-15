### CCGEN Teaching Guide

A complete theory-to-code walkthrough of the `python/ccgen` coupled-cluster
equation generator. Intended for students learning symbolic second quantization,
contributors extending the generator, and developers who need to audit how a
named method like CCD or CCSD turns into explicit tensor contractions.

### 1. What `ccgen` Is

`ccgen` is a small symbolic algebra package that derives spin-orbital
coupled-cluster equations from second-quantized operators. It lives in
`python/ccgen` and is intentionally separate from Planck's production C++
solvers. The package is meant to be:

- easy to inspect term by term
- mathematically close to textbook CC derivations
- deterministic enough to regression test
- simple enough that new methods can be added without touching the whole stack

The package currently handles the following workflow:

1. build the normal-ordered Hamiltonian \(H_N = F_N + V_N\)
2. build a cluster operator \(T = T_1 + T_2 + \cdots\)
3. expand the similarity-transformed Hamiltonian
   \(\bar H = e^{-T} H e^T\) through the finite BCH series
4. project \(\bar H\) onto the reference or excitation manifolds
5. apply Wick contractions with Hartree-Fock vacuum rules
6. canonicalize and merge equivalent tensor terms
7. optionally lower the result into an explicit contraction IR
8. emit readable text, `einsum`, or C++ loop nests

This is not a full tensor compiler. It is a symbolic front-end plus a very
simple lowering layer.

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
  → algebraic equations (generate.py)
  → optional contraction IR lowering (tensor_ir.py)
  → emitters (emit/pretty.py, emit/einsum.py, emit/cpp_loops.py)
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
| `python/ccgen/canonicalize.py` | tensor antisymmetry normalization, dummy relabeling, duplicate merge |
| `python/ccgen/tensor_ir.py` | explicit contraction metadata for backend use |
| `python/ccgen/generate.py` | top-level user API |
| `python/ccgen/emit/` | readable text, `einsum`, and naive C++ emitters |
| `python/ccgen/methods/` | small wrappers like `build_ccsd_equations()` |
| `python/ccgen/tests/` | regression tests for energy formulas, stability, and lowering |

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

### 11. Contraction IR Lowering

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
- a future contraction planner
- a future intermediate/factorization pass

### 12. Top-Level API

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

It returns:

```python
dict[str, list[AlgebraTerm]]
```

### `generate_cc_contractions()`

This runs `generate_cc_equations()` and then lowers the result into
`BackendTerm`s via `lower_equations()`.

### Formatting Helpers

- `print_equations()` uses `emit/pretty.py`
- `print_einsum()` uses `emit/einsum.py`
- `print_cpp()` uses `emit/cpp_loops.py`

### 13. Emitters

### `emit/pretty.py`

This is the human-facing printer. It emits expressions such as:

```text
R1(i,a) =
  + f(a,i)
  + 1/4 sum(j,b) f(j,b) t2(a,b,i,j)
```

This is the best format for inspecting algebraic structure while debugging.

### `emit/einsum.py`

This backend turns each `AlgebraTerm` into a NumPy-style `einsum` call. It is
mostly for validation and prototyping, because it makes the contraction pattern
visually obvious:

```python
R1 += 1/4 * np.einsum('jb,abij->ia', F, T2)
```

### `emit/cpp_loops.py`

This backend emits very naive loop nests. It does not optimize contraction
ordering or generate intermediates. Its purpose is to preserve the algebra in a
form that a C++ developer can read line by line.

### 14. Method Wrappers

The files in `python/ccgen/methods/` are intentionally thin.

- `ccd.py` returns energy + doubles
- `ccsd.py` returns energy + singles + doubles
- `ccsdt.py` returns energy + singles + doubles + triples

These wrappers exist mainly so users can import a named builder without needing
to know the generic driver API.

### 15. Worked Example: `generate_cc_equations("ccsd")`

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

### 16. Testing and Regression Strategy

The regression tests in `python/ccgen/tests/test_regressions.py` currently cover:

- dummy relabeling must not collide with free residual labels
- the generated CCSD energy must match the textbook / PySCF formula
- repeated calls must produce stable term counts
- no canonical term may keep repeated indices in an antisymmetric slot
- contraction lowering must preserve tensor-slot connectivity

This is important because symbolic code can be "almost right" in ways that are
very hard to notice by eye. Tiny errors in sign, relabeling, or free-index
restoration can completely change a residual while still producing something
that looks plausible.

### 17. Current Limitations

The package is intentionally educational, and that shows in a few places.

- It works in spin-orbital form only.
- The emitters are inspectable, not optimized.
- There is no automated intermediate generation or tensor factorization yet.
- The C++ emitter is a loop printer, not a production code generator.
- The BCH layer still builds the full formal expansion before projection.
- There is no direct integration with Planck's production C++ coupled-cluster solvers.

### 18. How to Extend the Package

### To Add a New Method

1. make sure `parse_cc_level()` can express the needed excitation ranks
2. add a wrapper in `python/ccgen/methods/`
3. call `generate_cc_equations()` or `generate_cc_contractions()`
4. add regression tests for the expected manifolds

### To Add a New Backend

1. decide whether the backend wants `AlgebraTerm` or `BackendTerm`
2. if it needs explicit slot connectivity, consume `tensor_ir.py`
3. add the new emitter under `python/ccgen/emit/`
4. add tests that compare emitted structure against known small cases

### To Add More Aggressive Simplification

The right place is after canonicalization and before backend emission. That is
where common-subexpression detection, intermediate introduction, or contraction
ordering logic should live.

### 19. Minimal Usage Examples

### Print CCSD Equations

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import print_equations

print(print_equations("ccsd"))
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

### Emit Debug `einsum`

```python
import sys
sys.path.insert(0, "python")

from ccgen.generate import print_einsum

print(print_einsum("ccd"))
```

### 20. Mental Model for Contributors

The easiest way to understand `ccgen` is to think of it as four stacked layers:

1. operator algebra
2. Wick contraction
3. tensor canonicalization
4. backend lowering / printing

When debugging, always identify which layer a bug belongs to. A wrong term can
come from:

- incorrect operator construction
- incorrect vacuum contraction rules
- incorrect delta substitution
- incorrect free/dummy classification
- incorrect antisymmetry sign handling
- incorrect duplicate merging
- incorrect backend lowering

Treating these as separate layers is the main architectural idea of the package.
