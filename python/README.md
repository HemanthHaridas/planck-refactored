# ccgen

A symbolic coupled-cluster equation generator that derives spin-orbital CC residual equations at arbitrary truncation order, then emits production-shaped code in several backend formats.

ccgen has **two generation engines** that produce the same equations:

- **`wick`** (default) — the textbook path: Baker-Campbell-Hausdorff (BCH) expansion of the similarity-transformed Hamiltonian, Wick's theorem for the Fermi-vacuum matrix elements, then canonicalization.
- **`diagram`** — enumerate canonical Kállay–Surján diagrams and assign each a solve-free topological weight. Canonical by construction, so no term explosion; far cheaper at high rank (CCSDTQ generation ~3 s vs ~600 s), and validated to produce the same residual as the wick engine.

The generated equations are validated against PySCF and full-CI **through CCSDTQ** (see [Validation](#validation)). They are not yet compiled into any Planck binary — the shipping CC solver is the hand-written `src/post_hf/cc/ccsd.cpp`; ccgen is the algebraic derivation + code-generation layer feeding the arbitrary-order solver and the diagram work.

## Features

- **Two engines, same equations**: `engine="wick"` (default, textbook) or `engine="diagram"` (canonical-by-construction, ~205× faster generation at CCSDTQ, residual-equal to wick)
- **Arbitrary truncation**: CCD, CCSD, CCSDT, CCSDTQ, and beyond (up to CC6)
- **Multiple emission backends**: human-readable equations, NumPy einsum, and C++ loop nests (naive / tiled+OpenMP / BLAS-lowered / Planck-native)
- **Solve-free diagram weight**: each diagram's sign × magnitude derived from its topology alone — no per-method fit, no external oracle at generation time
- **Algebraic optimizations**: orbital-energy denominator collection, permutation-based term grouping, implicit antisymmetry exploitation, canonical-Fock reduction
- **Intermediate tensor detection**: automatic extraction of reusable sub-contractions with memory-layout and blocking hints
- **Extended contraction IR**: backend-neutral IR with BLAS/GEMM pattern detection, FLOP estimates, and tiling hints
- **Restricted closed-shell lowering**: spin-orbital → spatial-orbital IR with occ/vir block signatures and source-slot permutations
- **Pipeline instrumentation & caching**: `PipelineStats`, per-manifold and BCH-prefix on-disk caching (`--cache-dir`)
- **Optional C++ Wick accelerator** (`ccgen._wickaccel`, built on install when a C++ compiler is available; pure-Python fallback)

## Theory

### The wick engine (default)

The standard CC derivation pipeline:

1. **Normal-ordered Hamiltonian** — `H_N = F_N + V_N`.
2. **Cluster operator** — `T = T1 + ... + Tn`, each `Tn` carrying the `(1/n!)^2` prefactor.
3. **BCH expansion** — `H̄ = H + [H,T] + ½![[H,T],T] + ⅓![[[H,T],T],T] + ¼![[[[H,T],T],T],T]`, terminating at fourth order because `H` is two-body.
4. **Wick contraction** — enumerate the fully-contracted pairings of `⟨Φ|…|Φ₀⟩`, tracking Fermi sign.
5. **Projection** — onto `⟨Φ₀|` (energy) and `⟨Φ_{ij…}^{ab…}|` (residuals).
6. **Canonicalization** — sort antisymmetry groups, relabel dummies canonically, merge identical terms.
7. **Connectivity filtering** — union-find on the contraction graph discards disconnected terms (linked-cluster theorem).
8. **Optimization / intermediate extraction** (optional).

This path is pedagogically clear but enumerates every algebraic term and dedups afterwards, so the work grows fast with rank (78× more terms projected than survive at CCSDT; ~10 min to generate CCSDTQ).

### The diagram engine

Instead of enumerating labeled terms, enumerate **diagrams** — the Kállay–Surján integer-triplet encoding, one triplet `(μ₁, μ₂, μ₃)` per cluster operator (its excitation level; internal lines to the Hamiltonian; particle internal lines). Diagrams are canonical by construction, so duplicates are never generated. Each diagram gets a **weight** derived from its topology:

- **magnitude** = `equivalent_vertex_factor · ∏(1/n_ext!) / 2^(equivalent_line_pairs + vertex_pairs)`
- **sign** = `(-1)^bra_level · external_pairing_parity · (-1)^loops · (-1 if the Fock line contracts a hole)`

The diagram is then expanded into the same `AlgebraTerm`s the rest of the pipeline consumes. The two engines emit different term *multisets* (the diagram path merges repeated-factor exchange pairs the wick path keeps split) but the same *residual tensor* — so they are compared by residual equality, and the diagram path emits slightly fewer terms.

The design and full validation are in `docs/CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md` and `docs/CCGEN_GENERATION_AND_VALIDATION.md`; a teaching walkthrough of both engines is `docs/CCGEN_TEACHING_GUIDE.md`.

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

When a C++ compiler is available, installation also builds the optional `ccgen._wickaccel` extension (packed-int Wick recursion); it falls back to pure Python otherwise.

For Google Colab, see [COLAB.md](COLAB.md).

## Usage

### Generate equations

```python
from ccgen import generate_cc_equations

eqs = generate_cc_equations("ccsd")            # wick engine (default)
for manifold, terms in eqs.items():
    print(f"{manifold}: {len(terms)} terms")
# energy: 3 terms
# singles: 16 terms
# doubles: 70 terms

eqs_d = generate_cc_equations("ccsd", engine="diagram")   # same equations, diagram engine
```

The `method` argument accepts any standard CC level string:

| Input | Cluster operator | Manifolds |
|-------|-----------------|-----------|
| `"ccd"` | T2 | energy, doubles |
| `"ccsd"` | T1 + T2 | energy, singles, doubles |
| `"ccsdt"` | T1 + T2 + T3 | energy, singles, doubles, triples |
| `"ccsdtq"` | T1 + ... + T4 | energy, ..., quadruples |
| `"cc6"` | T1 + ... + T6 | energy, ..., sextuples |

### Choosing an engine

```python
# Default: the textbook BCH + Wick path.
eqs = generate_cc_equations("ccsdt")                      # engine="wick"

# Canonical-by-construction diagram path — much faster at high rank, residual-equal.
eqs = generate_cc_equations("ccsdtq", engine="diagram")   # ~3 s vs ~600 s for wick
```

The default is `wick` (the reference); `diagram` is opt-in and validated through CCSDTQ.

### Algebraic options

```python
generate_cc_equations("ccsd", collect_denominators=True)  # collect diagonal Fock -> denominators
generate_cc_equations("ccsd", permutation_grouping=True)  # merge index-permutation-related terms
generate_cc_equations("ccsd", exploit_symmetry=True)      # implicit antisymmetry (experimental)
generate_cc_equations("ccsd", canonical_fock=True)        # drop f_ov/f_vo (zero for canonical HF)
generate_cc_equations("ccsd", debug=True)                 # pipeline timing + counts to stderr
```

### Caching (wick engine)

```python
# Persist BCH expansion + per-manifold canonical equations; higher-order methods
# reuse a stored lower-order BCH expansion as a prefix (e.g. CCSD -> CCSDT).
generate_cc_equations("ccsdt", cache_dir="~/.cache/ccgen")
```

### Pretty-print

```python
from ccgen.generate import print_equations, print_equations_full

print(print_equations("ccsd"))
print(print_equations_full("ccsd", intermediate_threshold=5))   # + intermediates, legend, stats
```

```
E_CC() =
  + sum(i,a) f(i,a) t1(a,i)
  + 1/4 sum(i,j,a,b) t2(a,b,i,j) v(i,j,a,b)
  + 1/2 sum(i,j,a,b) t1(a,i) t1(b,j) v(i,j,a,b)
R1(i,a) = ...
```

### NumPy einsum output

```python
from ccgen.generate import print_einsum

print(print_einsum("ccsd"))
print(print_einsum("ccsd", use_opt_einsum=True))          # opt_einsum contraction paths
print(print_einsum("ccsdtq", engine="diagram"))           # engine threads through every emitter
```

### C++ loop-nest output

```python
from ccgen.generate import print_cpp, print_cpp_optimized, print_cpp_blas, print_cpp_planck

print(print_cpp("ccsd"))                                          # naive loop nests
print(print_cpp_optimized("ccsd", tile_occ=16, tile_vir=16, use_openmp=True))
print(print_cpp_blas("ccsd", use_blas=True, use_openmp=True))     # cblas_dgemm where patterns match
print(print_cpp_planck("ccsd", include_intermediates=True))       # Planck-native tensor kernels
```

The Planck emitter targets the concrete tensor types in `src/post_hf/cc/` (`CanonicalRHFCCReference`, `MOBlockCache`, `DenominatorCache`, `RCCSDAmplitudes`/`RCCSDTAmplitudes`), mapping abstract names (F, V, T1, T2, …) to the right accessors and handling ERI block symmetry.

### Contraction IR and restricted lowering

```python
from ccgen import generate_cc_contractions, generate_cc_equations_lowered
from ccgen.generate import generate_cc_contractions_ex

ir  = generate_cc_contractions("ccsd")                    # basic backend IR
irx = generate_cc_contractions_ex("ccsd", detect_blas=True, tile_occ=16, tile_vir=16)  # + BLAS/FLOP/tiling
low = generate_cc_equations_lowered("ccsd")               # spatial-orbital layout (restricted_closed_shell)
```

Each lowered factor carries a `block_signature` (occ/vir/gen space tuple), a `source_permutation` (algebra slots → spatial layout), and ERI symmetry phase tracking.

### Selective targets & instrumentation

```python
generate_cc_equations("ccsdt", targets=["energy", "doubles"])

import ccgen.generate as gen
gen.generate_cc_equations("ccsd", debug=True)
print(gen.last_stats.summary())
```

### `ccgen` console script

```bash
ccgen ccsd                                  # per-manifold term counts
ccgen ccsd --format pretty                  # human-readable equations
ccgen ccsd --format einsum --opt-einsum
ccgen ccsdt --format cpp-optimized --tile-occ 32 --tile-vir 24
ccgen ccsd --format cpp-planck --include-intermediates
ccgen ccsdt --targets energy doubles triples
ccgen ccsdt --cache-dir ~/.cache/ccgen
ccgen ccsd --debug
```

### CLI scripts

```bash
cd python
python generate_ccsdt_cpp.py ccsdt                 # CCSDT naive C++ to stdout
python generate_ccsdt_cpp.py ccsd --einsum         # numpy einsum
python generate_ccsdt_cpp.py ccsd --planck         # Planck-native kernels
python generate_planck_cc_kernels.py --output-dir build/ --methods ccsd ccsdt
python generate_spinorbital_ccsd_warm_start.py --output ccsd_warm_start.inc
```

## Validation

There is no per-term oracle past CCSD (PySCF ships no spin-orbital `gccsdt`), so ccgen is validated the way production arbitrary-order codes are — reduction + FCI limit + converged energy — with the discipline that **no generation gate is ever pinned to ccgen's own output**:

| check | result |
|---|---|
| full CCSD doubles residual vs PySCF `gccsd.update_amps` | ~1e-16 |
| CCSD energy at PySCF's converged amplitudes | ~1e-15 |
| CCSDT singles/doubles with T3=0 reduce to CCSD | exact |
| generated CCSDT solved to convergence vs FCI (3 e⁻) | ~1e-12 |
| generated CCSDTQ solved to convergence vs FCI (4 e⁻) | ~1e-12 |
| diagram engine vs wick engine, per-manifold residual | ~1e-13 |

The PySCF/FCI gates live in `ccgen/tests/test_reference_vs_pyscf.py` and run under a pyscf virtualenv (the CCSDT/CCSDTQ FCI solves are slow).

> The earlier "CCSD raw-generation weight bug" was **disproven** — it was an off-shell dressed-reference comparison artifact, not a defect. See `docs/CCGEN_GENERATION_AND_VALIDATION.md`.

## Package structure

```
ccgen/
  __init__.py          Public API
  cli.py               `ccgen` console-script entry point
  generate.py          Top-level driver + engine dispatch (wick / diagram),
                       caching, parallel manifold generation
  hamiltonian.py       Normal-ordered Hamiltonian (F_N + V_N)
  cluster.py           Cluster operator builder + CC level parser
  algebra.py           Multiplication, commutator, BCH expansion     (wick engine)
  wick.py              Wick contraction engine                        (wick engine)
  project.py           Projection onto manifolds + shared AlgebraTerm / manifold maps
  connectivity.py      Union-find connected-component analysis
  diagram.py           Diagram engine: Kállay–Surján enumeration, assembly,
                       solve-free weight, AlgebraTerm orbit expansion  (diagram engine)
  canonicalize.py      Antisymmetry normalization, dummy relabeling, term merging,
                       orbital-energy denominator collection
  tensor_ir.py         Backend-neutral contraction IR (basic + extended)
  bench.py             Performance benchmarking (python -m ccgen.bench)
  _wickaccel.cpp       Optional C++ extension (packed-int Wick recursion)
  emit/
    pretty.py                  Human-readable formatter (with intermediates)
    einsum.py                  NumPy einsum emitter (opt_einsum integration)
    cpp_loops.py               C++: naive / tiled+OpenMP / BLAS-lowered
    planck_tensor_cpp.py       Planck-native C++ emitter (Tensor2D/4D/6D)
    planck_rccsd_warm_start.py Planck RCCSD warm-start emitter
  lowering/
    restricted_closed_shell.py Spin-orbital -> spatial-orbital IR
  optimization/
    intermediates.py, subexpression.py, permutation.py, symmetry.py
    tau.py, dressing.py, dressed_equation.py   (retired exact-cover dressing route; see note)
  methods/
    ccd.py, ccsd.py, ccsdt.py                  Convenience drivers
  tests/                                       10 modules (see Testing)
```

> `optimization/{tau,dressing,dressed_equation}.py` implement the term-algebra
> dressed-intermediate route (recognizing `Wmnij`/`Wabef`/τ by index-binding +
> exact cover over the flat term list). That route is **retired** — with the
> diagram representation, dressed operators are identifiable subgraphs, so
> recognition becomes topological. The code is kept as the record of the
> abandoned route.

## Testing

```bash
cd python
python -m pytest ccgen/tests/ -v                          # fast suite (no pyscf)
tests/pyscf/.venv/bin/python -m pytest ccgen/tests/test_reference_vs_pyscf.py   # PySCF/FCI gates
```

Ten test modules. The core ones:

- **`test_regressions.py`** — equation stability, numerical CCSD-energy correctness, slot tracking, cross-rank reduction, and the `engine="wick"` vs `engine="diagram"` residual-equality gate.
- **`test_optimizations.py`** — algebraic equivalence of each optimization pass.
- **`test_diagram.py`** — the diagram engine: enumeration, assembly, the solve-free weight, and the orbit expansion.
- **`test_reference_vs_pyscf.py`** — the PySCF/FCI validation ladder above (pyscf venv).

## Integration with Planck

The Planck emitter (`print_cpp_planck` / `emit/planck_tensor_cpp.py`) maps abstract tensors to Planck's `src/post_hf/cc/` types:

| Abstract tensor | Planck accessor |
|----------------|----------------|
| `F` (Fock blocks) | `reference.f_ov()`, `f_oo()`, … |
| `V` (ERI blocks) | `mo_blocks.oovv()`, `ovov()`, … |
| `T1`, `T2`, `T3` | `amplitudes.t1()`, `t2()`, `t3()` |
| `D` (denominators) | `denominators.d_ov()`, `d_oovv()`, … |

Two code-generation workflows exist — kernel generation (`generate_planck_cc_kernels.py`) and warm-start inclusion (`generate_spinorbital_ccsd_warm_start.py`). Note that generated code is **not compiled into the default build**: the shipping CCSD warm-start calls the hand-written `src/post_hf/cc/ccsd.cpp`. Wiring the (validated) generated path into the build is a separate, deferred step.

## License

MIT
