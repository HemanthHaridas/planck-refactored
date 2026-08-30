# How does ccgen generate coupled-cluster equations, and how do we know they are correct?

Deeper design of the diagram front end lives in
`docs/CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md`.

## The two generation engines

`generate_cc_equations(method, engine="wick"|"diagram")` returns
`{manifold: [AlgebraTerm]}`. Both engines produce the same equations; `wick` is
the default.

- **wick** — BCH-expand the similarity-transformed Hamiltonian, project onto each
  excitation manifold, canonicalize, merge. The textbook term-algebra path.
- **diagram** — enumerate canonical Kállay–Surján diagrams, give each a
  solve-free weight, expand each into its signed term orbit, add the bare
  Hamiltonian term, then run the same canonicalize/merge as wick. No BCH, no Wick
  contraction.

Both feed the unchanged downstream (canonicalize, IR, lowering, emitters). The
shared types they both depend on live in `project.py`.

Files:

- `python/ccgen/generate.py`
- `python/ccgen/algebra.py`, `python/ccgen/wick.py`, `python/ccgen/project.py` (wick path)
- `python/ccgen/diagram.py` (diagram path)

## Why the two engines are compared by residual, not by term multiset

The engines emit different term multisets — wick keeps `t1·t1·v` as two `±½`
terms, diagram merges them to one — but the same residual tensor. The only
difference is how repeated-factor terms split, an exchange-symmetry
representational choice that lowers and emits to the same runtime accumulation.

So equivalence is checked at the residual level (per-manifold arrays agree) and
at the emitted-kernel level (the einsum-emitted `E`/`R1`/`R2` agree in value and
in index layout), never by comparing term lists. Making the diagram terms
`free_indices` occ-first is what aligns the emitted layout with wick.

## Why validation is reduction + FCI + energy, not a per-term oracle

There is no per-term CCSDT oracle to diff against: PySCF ships no spin-orbital
`gccsdt`, only spin-adapted `rccsdt`/`uccsdt` and the perturbative `(T)`. So
ccgen is validated the way MRCC/CFOUR validate arbitrary order:

- the full CCSD residual matches PySCF `gccsd.update_amps`,
- CCSDT singles/doubles with `T3 = 0` reduce exactly to the validated CCSD,
- generated CCSDT solved to convergence reaches FCI on a 3-electron system,
- generated CCSDTQ solved to convergence reaches FCI on a 4-electron system
  (the diagram engine, `1.1e-12`),
- the diagram engine reproduces both the wick residual and the FCI energy.

The load-bearing discipline: no generation gate is pinned to
`generate_cc_equations` itself. The generator is never its own oracle — gates
pin to PySCF, to FCI, or to a lower rank already pinned to PySCF.

Files:

- `python/ccgen/tests/test_reference_vs_pyscf.py` (pyscf venv)
- `python/ccgen/tests/test_regressions.py`

## Why the "raw-generation weight bug" was retracted

Earlier notes documented a ~2–3% error in ccgen's CCSD `t1·t2`-mixing doubles
terms. That was an artifact, not a defect. The old gate compared ccgen to a
hand-transcribed *dressed* (Stanton–Gauss) reference on random *off-shell*
amplitudes, where the raw projection and the dressed form need not agree
term-by-term. On real amplitudes both match PySCF, and the full ccgen residual
matches `update_amps` to machine precision.

Two genuine fixes from that investigation stand: canonical-Fock mode (drops the
identically-zero `f_ov` terms) and adding `is_dummy` to the canonicalization key
(recovered falsely-zeroed terms). The `w0`/`w1`/`w2` test scaffolding predates
the retraction and analyzes the non-bug; its "over-count" framing is historical.

## Why the default is still wick

The diagram engine is validated and opt-in, but flipping the default changes what
the real codegen consumers emit, and only then can the term-path enumeration
(`wick.py`, the BCH path, the projection helpers) be deleted. `project.py` stays
regardless — it holds the shared types both engines use. The flip is a separate,
higher-risk decision, now backed by the kernel-equivalence evidence above.

## Why a generator issue is not (yet) a wrong-energy-in-production issue

**Today** the default build compiles neither generated path into a binary — the
shipping CCSD warm-start calls the hand-written residual in
`src/post_hf/cc/ccsd.cpp`, so a generator defect gates trusting the algebraic
path (the arbitrary-order solver, the diagram front end), not any production
energy.

**That is the current state, not the end state.** The generated kernels are
intended to replace the hand-written solvers in production. Two of the three
things that used to gate that have moved: **spin adaptation landed** (spatial RCC
kernels, validated to FCI at rank 4 — `CCGEN_CCSDTQ_MULTISECTOR.md`), and the
**dressing route was retired**, so FLOP scaling is now owned by the contraction-order
work rather than by dressed operators (`CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`,
`CCGEN_KERNEL_SCALING_SCOPE.md`). What remains is cost, and it is smaller and better
understood than the carried "~500x slower at rank 3" implies: that figure is a
ratio across a **solver boundary** (different amplitude storage, 40 vs 16
iterations on CH4), not a defect size. Profiling the generated path against itself
found and fixed two thirds of its kernel time (redundant per-chunk operator
rebuilds, 1.76x) and identified the largest remaining lever as CC having **no
OpenMP at all** (modelled 3.86x). See `CCGEN_ARBITRARY_HARNESS_COST.md`. Generator correctness — which this
document establishes through CCSDTQ — is therefore already load-bearing for the
generated production route, which is why the validation here is built to the
standard it is.
