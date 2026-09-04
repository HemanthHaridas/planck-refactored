# ccgen Equation Generation and Validation

Deeper design of the diagram front end lives in
`docs/CCGEN_DIAGRAM_REPRESENTATION.md`.

This file answers a narrower architecture question:

**How does ccgen generate coupled-cluster equations, and how do we know they are correct?**

## Short answer

ccgen has two independent generation engines — `wick` (BCH expansion + Wick contraction, the
default) and `diagram` (canonical Kállay–Surján diagrams with solve-free weights) — that produce
the same residual tensors from different term-level representations. Correctness is established
not against a per-term oracle (none exists for arbitrary-order spin-orbital CC) but through a
reduction + FCI + energy chain: CCSD matches PySCF `gccsd.update_amps` to machine precision, CCSDT
and CCSDTQ solved to convergence reach FCI, and the two engines agree with each other at the
residual and emitted-kernel level.

## Where the logic lives

- `python/ccgen/generate.py`
- `python/ccgen/algebra.py`, `python/ccgen/wick.py`, `python/ccgen/project.py` (wick path)
- `python/ccgen/diagram.py` (diagram path)
- `python/ccgen/tests/test_reference_vs_pyscf.py` (pyscf venv)
- `python/ccgen/tests/test_regressions.py`

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

## What invariants matter

### 1. The two engines are compared by residual, not by term multiset

The engines emit different term multisets — wick keeps `t1·t1·v` as two `±½`
terms, diagram merges them to one — but the same residual tensor. The only
difference is how repeated-factor terms split, an exchange-symmetry
representational choice that lowers and emits to the same runtime accumulation.

So equivalence is checked at the residual level (per-manifold arrays agree) and
at the emitted-kernel level (the einsum-emitted `E`/`R1`/`R2` agree in value and
in index layout), never by comparing term lists. Making the diagram terms
`free_indices` occ-first is what aligns the emitted layout with wick.

Design rule:

- Never compare the two engines' term lists directly. Compare residual tensors and emitted-kernel
  output.

### 2. No generation gate is pinned to `generate_cc_equations` itself

There is no per-term CCSDT oracle to diff against: PySCF ships no spin-orbital
`gccsdt`, only spin-adapted `rccsdt`/`uccsdt` and the perturbative `(T)`. So
ccgen is validated the way MRCC/CFOUR validate arbitrary order:

- the full CCSD residual matches PySCF `gccsd.update_amps`,
- CCSDT singles/doubles with `T3 = 0` reduce exactly to the validated CCSD,
- generated CCSDT solved to convergence reaches FCI on a 3-electron system,
- generated CCSDTQ solved to convergence reaches FCI on a 4-electron system
  (the diagram engine, `1.1e-12`),
- the diagram engine reproduces both the wick residual and the FCI energy.

Design rule:

- The generator is never its own oracle. Every gate pins to PySCF, to FCI, or to a lower rank
  already pinned to PySCF.

### 3. The default engine stays `wick` until the term-path code can be deleted

The diagram engine is validated and opt-in, but flipping the default changes what
the real codegen consumers emit, and only then can the term-path enumeration
(`wick.py`, the BCH path, the projection helpers) be deleted. `project.py` stays
regardless — it holds the shared types both engines use. The flip is a separate,
higher-risk decision, now backed by the kernel-equivalence evidence above.

## What was found

1. **The "raw-generation weight bug" was retracted.** Earlier notes documented a ~2–3% error in
   ccgen's CCSD `t1·t2`-mixing doubles terms. That was an artifact, not a defect. The old gate
   compared ccgen to a hand-transcribed *dressed* (Stanton–Gauss) reference on random *off-shell*
   amplitudes, where the raw projection and the dressed form need not agree term-by-term. On real
   amplitudes both match PySCF, and the full ccgen residual matches `update_amps` to machine
   precision. Two genuine fixes from that investigation stand: canonical-Fock mode (drops the
   identically-zero `f_ov` terms) and adding `is_dummy` to the canonicalization key (recovered
   falsely-zeroed terms). The `w0`/`w1`/`w2` test scaffolding predates the retraction and analyzes
   the non-bug; its "over-count" framing is historical.
2. **A generator issue is not (yet) a wrong-energy-in-production issue, but that is changing.**
   **Today** the default build compiles neither generated path into a binary — the shipping CCSD
   warm-start calls the hand-written residual in `src/post_hf/cc/ccsd.cpp`, so a generator defect
   gates trusting the algebraic path (the arbitrary-order solver, the diagram front end), not any
   production energy. **That is the current state, not the end state.** The generated kernels are
   intended to replace the hand-written solvers in production. Two of the three things that used
   to gate that have moved: **spin adaptation landed** (spatial RCC kernels, validated to FCI at
   rank 4 — `CCGEN_CCSDTQ_MULTISECTOR.md`), and the **dressing route was retired**, so FLOP scaling
   is now owned by the contraction-order work rather than by dressed operators
   (`CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`, `CCGEN_KERNEL_SCALING_SCOPE.md`). What remains is cost,
   and it is smaller and better understood than the carried "~500x slower at rank 3" implies: that
   figure is a ratio across a **solver boundary** (different amplitude storage, 40 vs 16 iterations
   on CH4), not a defect size. Profiling the generated path against itself found and fixed two
   thirds of its kernel time (redundant per-chunk operator rebuilds, 1.76x) and identified the
   largest remaining lever as CC having **no OpenMP at all** (modelled 3.86x). See
   `CCGEN_ARBITRARY_HARNESS_COST.md`. Generator correctness — which this document establishes
   through CCSDTQ — is therefore already load-bearing for the generated production route, which is
   why the validation here is built to the standard it is.

## Validation strategy that should remain in place

- `python/ccgen/tests/test_reference_vs_pyscf.py` — CCSD residual vs PySCF `gccsd.update_amps`
- `python/ccgen/tests/test_regressions.py`
- CCSDT singles/doubles with `T3 = 0` reducing exactly to validated CCSD
- Generated CCSDT solved to convergence reaching FCI on a 3-electron system
- Generated CCSDTQ solved to convergence reaching FCI on a 4-electron system (diagram engine,
  `1.1e-12`)
- Cross-checking the diagram engine against both the wick residual and the FCI energy
