# The generated rank-3 CCSDT triples kernel is wrong — handoff

**Read this first if you are picking up this work cold.** In-flight scope; rewrite it as an
architecture answer once the defect is fixed and verified.

**One line:** `compute_ccsdt_triples_residual` (ccgen-generated, rank 3) computes a substantially
different function from the hand-written triples residual — ~100 % of magnitude at identical
amplitudes. One cause is fixed (a missing physicist rebind); at least one remains, and it is
upstream of codegen.

---

## How to reproduce in one command

Three binaries exist, configured to differ only as noted. **They live in `/tmp` and will not
survive a reboot** — rebuild commands are at the end.

```bash
export BASIS_PATH=$PWD/basis-sets
IN=tests/inputs/regression/post_hf/bh3_rccsdt_sto3g.hfinp

# the decisive probe: both residuals from identical amplitudes, ONE evaluation
PLANCK_CC_T3_DIFF=1 /tmp/claude-501/dressed2/hartree-fock $IN 2>&1 | grep T3-DIFF | head -2
```

Current output:

```
raw (no restore): max|gen-hand|=2.044633e-02 rms=1.222017e-03 max|hand|=2.033739e-02
after restore   : max|gen-hand|=1.865172e-03 rms=1.575882e-04 max|hand|=2.779055e-02
```

| build | configuration | role |
|---|---|---|
| `build/` | stock, `SPIN_ADAPT=ON MAXORDER=4` | hand-written reference |
| `/tmp/claude-501/dressed2` | `+DRESS_OPERATORS=ON` | generated + dressed |
| `/tmp/claude-501/undressed_gen` | same minus dressing | generated, undressed (needs `PLANCK_RCCSDT_BACKEND=optimized`) |

---

## What is established — all measured, do not re-derive

### The numbers

`bh3_rccsdt_sto3g`; hand-written reference `E_corr = −0.0533629199` (26 iters, 7.1 s):

| state | E_corr | note |
|---|---|---|
| generated, before any fix | −0.0531812197 | 1.82e-4 off, 1261 s |
| generated + dressed, before any fix | −0.0531789160 | 1283 s |
| generated + dressed, **after T1b** | ≈ −0.05333 | 2.7e-5 off, **oscillates** |

After T1b the solve no longer converges: `dE` alternates ±2e-6 and `rms(R3)` **grows**
(2.6e-4 → 7.4e-4 across iterations 33–38).

### The residual comparison — the key result

At identical amplitudes the **raw** generated and hand-written residuals differ by ~100 % of the
hand-written magnitude. Not a scale factor, not a convention, not a permutation.

**`restore_restricted_t3_structure` masks the error ~11×** (100 % → 6.7 %). That explains the whole
earlier picture: the energy looked nearly right while the residual was wholly wrong, and the
surviving 6.7 % is what makes the solve oscillate instead of settling on a clean wrong number.

> **Any gate on this kernel must compare the RAW residual.** A post-`restore` comparison
> understates the error by an order of magnitude and would call a badly wrong kernel nearly right.

### Fixed: the missing physicist rebind (`eb1c611`)

ccgen emits against physicist `<pq|rs>`; `state.mo_blocks` holds chemists' `(pq|rs)`. The
arbitrary-order path always rebound before invoking a generated kernel — which is why the
CCSDTQ==FCI gate passes — and the plain rank-3 path never did.

`rebind_physicist` is now exposed from `generated_arbitrary_runtime.h` (not copied — the
`oovv`↔`ovov` sources cross), rebinds into a **local** cache (the shared `state.mo_blocks` must stay
chemists' for `build_spin_orbital_blocks` and the hand-written branch), and is hoisted **outside**
the iteration loop.

Effect: 1.82e-4 → 2.7e-5 Eh. Most of the error, not all of it.

### Why this was never caught

`compute_ccsdt_triples_residual` **had no caller**. `choose_rccsdt_backend` returned only
`DeterminantPrototype` or `TensorProduction`; the single call site is guarded by
`use_generated_triples_kernel`, and both `run_tensor_rccsdt_impl` callers passed `false`. Generated,
compiled, linked, never executed. Wiring fixed in `64d0074`.

**The CCSDTQ==FCI gate does not cover this kernel.** That exercises
`compute_ccsdtq_triples_residual` — a *different function* in a different TU (arbitrary-order
runtime). Same generator, separately emitted code.

### Ruled out by measurement — do not re-run these

| hypothesis | verdict |
|---|---|
| Spin-adaptation config mismatch | **No.** A build matched on `SPIN_ADAPT=ON MAXORDER=4 ENGINE=diagram` gave the identical wrong energy. |
| Dressing causes the error | **No.** Dressed and raw rank-3 residuals agree to **8.4e-13** symbolically, and the *undressed* generated build is wrong too. |
| `restore_restricted_t3_structure` is generated-only | **No.** The hand-written branch calls it at `tensor_backend.cpp:2360, 2650, 2657, 2674` and converges correctly. |
| Its non-idempotent ×6 sum is a bug | **No.** Verified 6× on an already-symmetric tensor, but compensated by the repeated-index pre-scaling (`1/6`, `1/2`) at ~line 2010. Self-consistent. |
| `_ERI_SYMMETRY_PERMUTATIONS` has invalid −1 perms | **No.** Already the corrected +1-only form from B5. |
| The emit is unfaithful to the generator | **No.** A fresh `print_cpp_planck("ccsdt", spin_adapt=True, dress_operators=True)` reproduces the built TU's amplitude-read counts exactly (t1 465, t2 935, t3 217). **The defect is upstream of codegen.** |

### The honest blocker

**There is no decisive oracle yet.** Two comparisons were tried; neither localizes the error:

- GCC vs spin-adapted residuals are different objects (spin-orbital vs spatial) — not comparable.
- Arbitrary-order vs plain rank-3 emits are similar in size (485 411 vs 503 227 bytes) and term
  counts (t3 reads 313 vs 310).

Getting an oracle is R0.

---

## Remaining work, in small verifiable steps

### R0 — get an oracle: equations, or the plain TU's path? (~S, do first)

The arbitrary-order rank-3 kernel is **validated** — the CCSDTQ==FCI gate runs it as a warm start.
The plain rank-3 TU is the one that never had a caller. Both come from the same equations.

```bash
cmake -B /tmp/rank3arb -S . -DCMAKE_BUILD_TYPE=Release \
  -DPLANCK_CC_SPIN_ADAPT=ON -DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON
make -C /tmp/rank3arb hartree-fock -j8
```

Then run the `PLANCK_CC_T3_DIFF=1` probe on it.

- **Arbitrary agrees with hand-written** → the equations are correct; the defect is specific to the
  plain TU's lowering/emit path.
- **Arbitrary also differs** → the defect is in the shared rank-3 equations, and the CCSDTQ gate
  misses it because CCSDTQ emits its own triples function.

**Why first:** it halves the search space with one build, and every later step depends on which half.

*Gate:* the two probe lines recorded for the arbitrary-order build.

### R1 — localize by term class (~M, after R0)

Extend the probe to report **where** the raw difference concentrates: by occ/vir index pattern, by
whether the element has repeated indices, and by which amplitude rank dominates the terms writing
there.

Suspects not yet eliminated, in order:

1. **T3 storage/slice convention** — whether the emitted kernel writes the layout the tensor solver
   reads. The natural candidate now that the emit is known faithful to the generator.
2. **Lowering of the spatial (spin-adapted) triples terms** — the `2·direct − exchange` structure at
   rank 3, exercised nowhere else at this rank.
3. **The `restore_restricted_t3_structure` interaction** — not a bug on its own, but the generated
   residual may not arrive in the pre-scaled form the convention assumes.

*Gate:* a named term class and a numeric before/after at fixed amplitudes.

### R2 — fix, and gate on the RAW residual (~M)

Whatever R1 identifies. The gate compares raw residuals at fixed amplitudes — not converged
energies, not post-`restore` values.

*Gate:* `max|gen−hand|` on the raw residual at or below solver tolerance, plus bh3 converging to
−0.0533629199 in a comparable iteration count.

### R3 — re-check the dressing delta (~S, after R2)

2.3e-6 Eh between dressed and undressed generated. Symbolically the two agree to 8.4e-13, so this is
likely downstream of the main defect. **Re-measure after R2** rather than investigating now.

*Gate:* dressed == undressed generated to solver tolerance, or a named emit-layer cause.

### R4 — the ~180× slowdown (~M, last)

7 s → ~1270 s on both generated builds. Correctness first: R2 may change the kernel's shape.
Candidates are intermediates rebuilt inside loops, and the absence of CSE (dressing forces
`include_intermediates` off).

*Gate:* a profile naming the dominant cost, not a guess.

### R5 — correct the false claims (~S, INDEPENDENT — do whenever)

`vault/Status/Completion.md`, `docs/CCGEN_DRESSED_KERNEL_PIPELINE.md`, and the
`dressed_kernel_equivalence_rccsdt` regression case all assert a **verified dressed == undressed
equivalence at rank 3**. That verification never happened — both builds ran hand-written code — and
the real comparison now fails.

These are actively misleading and should be corrected independently of the fix. The gate itself is
already correct: it asserts `RCCSDT[OPT]` and `kernels=ccgen-generated` before believing any number,
and currently reports FAIL, which is the honest state.

---

## Sequencing

```
R0 (oracle: arbitrary vs plain)   ~S  ← halves the search space; do first
 └→ R1 (localize by term class)   ~M
      └→ R2 (fix; gate on RAW)    ~M
           └→ R3 (dressing delta) ~S  ← may vanish
                └→ R4 (slowdown)  ~M
R5 (correct the false docs)       ~S  ← independent, do whenever
```

## What NOT to do

- **Do not gate on converged energy or post-`restore` residuals.** Both understate the error ~11× or
  more. Raw residual at fixed amplitudes.
- **Do not assume the CCSDTQ==FCI result covers this kernel.** Different function, different TU.
- **Do not debug through the solve.** The probe is one evaluation; a solve is ~21 minutes and
  conflates kernel error with convergence path.
- **Do not re-investigate the ruled-out list.** Six hypotheses are already disproven above.
- **Do not attribute the residue to dressing without re-measuring after R2.**
- **Do not revert the backend wiring to make things green.** Reaching this kernel is what exposed a
  real defect; hiding it restores a false pass.

## Key code locations

| what | where |
|---|---|
| generated-vs-hand branch | `src/post_hf/cc/tensor_backend.cpp:2321` |
| T0 probe (`PLANCK_CC_T3_DIFF`) | same file, inside the generated branch |
| `rebind_physicist` | `generated_arbitrary_prepare.cpp`, declared in `generated_arbitrary_runtime.h` |
| backend selection | `choose_rccsdt_backend`, `tensor_backend.cpp:~2740` |
| `restore_restricted_t3_structure` | `tensor_backend.cpp:1976` |
| repeated-index pre-scaling | `tensor_backend.cpp:~2010` |
| equivalence gate | `tests/dressed_kernel_equivalence.py` |

Commits: `64d0074` (wiring), `eb1c611` (physicist rebind), `38946ee` (probe), `89ab544` / `bf1d206`
(investigation history).

## Rebuild commands

```bash
cmake -B /tmp/dressed2 -S . -DCMAKE_BUILD_TYPE=Release \
  -DPLANCK_CC_DRESS_OPERATORS=ON -DPLANCK_CC_SPIN_ADAPT=ON -DPLANCK_CC_MAXORDER=4
cmake -B /tmp/undressed_gen -S . -DCMAKE_BUILD_TYPE=Release \
  -DPLANCK_CC_SPIN_ADAPT=ON -DPLANCK_CC_MAXORDER=4
make -C /tmp/dressed2 hartree-fock -j8
```

Kernel generation is ~1–2 min at `MAXORDER=4` with dressing; the full build is longer.
`dressed2` selects the generated backend automatically via `PLANCK_CC_DRESS_OPERATORS`;
`undressed_gen` needs `PLANCK_RCCSDT_BACKEND=optimized`.
