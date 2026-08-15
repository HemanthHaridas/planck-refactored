# Why dressing and spin adaptation do not compose

Answers one question: **can ccgen's dressed-operator factorization be combined with spin
adaptation, and is it worth doing?**

Short answer: **no, and no.** Each transform is correct alone and the composition is wrong in
either order, for a structural reason. The measured FLOP payoff (~1.2–1.5× actual on
spin-orbital, bounded ~2.8–4× spatial) does not justify the research needed to fix it.

**Decision: dressing stays opt-in and OFF, and the spin-adapted dressed path is unsupported.**
`-DPLANCK_CC_DRESS_OPERATORS=ON` generates and compiles, but its RCC kernels are wrong; the
default build is unaffected and byte-identical.

---

## What dressing is, and where it is correct

Dressing rewrites a CC residual to reference recognized intermediates (the Stanton-Gauss
`Wmnij`/`Wabef`/`Wmbej` plus the `tau`/`tau_c` pseudo-amplitudes) instead of their expanded
contractions, so a repeated sub-expression is computed once.

**On spin-orbital (GCC) input it is a valid rewrite.** `verify_dressed_equation` reports 0
mismatches; recognition changes 2 operator-free terms' coefficients and those changes are exactly
compensated by what the operators absorb. That is what a correct factorization looks like.

## Why the composition fails

Recognition works by matching a sub-expression and **subtracting what the operator absorbs from
the remaining terms**. That subtraction is only valid against the term set it was computed for.

Spin adaptation changes the term set — splitting terms into spin cases, merging others. So:

- **`dress → adapt`** (what production does): the subtraction was computed against GCC terms that
  no longer exist after adaptation.
- **`adapt → dress`**: recognition matches against spatial terms, but the operator *definitions*
  were derived in the spin-orbital basis and were never valid there.

Measured on `ccsd` doubles:

| path | operator-free terms with changed coefficients | compensated? |
|---|---|---|
| GCC (`dress` only) | 2 | **yes** — valid rewrite |
| `adapt → dress` | 8 | **no** — ‖diff‖ = 8.06e+02 against ‖R‖ = 983.79 |
| `dress → adapt` (production) | — | **no** — Be/STO-3G CCSDTQ 52 % short |

The end-to-end symptom: dressed `E_corr` = −0.0247182895 against an exact −0.0517746319.

### The deeper reason: the operators are built on an antisymmetry that spatial terms lack

The seeded definitions carry `P(ij)` antisymmetrizers and `1/4` weights that are artifacts of
`<pq||rs> = <pq|rs> − <pq|sr>`. A spatial residual has only the four `+1` symmetries of `<pq|rs>`.
Consequently the spatial residual has **more distinct contraction topologies** — 13 `v` index-space
patterns vs 9 in GCC, with `ovvo`/`ovov`/`voov`/`vovo` all appearing separately where antisymmetry
makes them one object.

So a spatial `Wmbej` is not a relabeled GCC `Wmbej`; it is several different operators. **The
spatial factorization is a different object, and deriving it is research, not porting.**

## Why it is not worth deriving

Measured with `factorize.py`'s contraction-tree cost model (`contraction_tree_cost`, which accounts
for contraction order — unlike `IntermediateSpec.estimated_build_flops`, which counts elements and
would overstate the benefit):

| quantity | (n_occ, n_vir) = (10,50) | (30,100) | (3,4) |
|---|---|---|---|
| GCC dressed saving, **actual** | 1.20× | 1.50× | 1.59× |
| spatial saving, **upper bound** | 2.78× | 4.00× | 5.27× |
| bound looseness, calibrated on GCC | ÷1.5 | ÷1.4 | ÷3.3 |
| spatial saving, **expected** | ~1.9× | ~2.8× | — |

Note the GCC saving *shrinks* as `n_vir/n_occ` grows, i.e. it pays least in the production regime.

Against a ~2× constant factor, the work required is:

- derive a spatial operator set (research; the literature factorization may not match ccgen's terms)
- build a spatial-capable matcher (the current one assumes antisymmetric factors)
- maintain and validate a second equation path indefinitely

**A larger lever is untouched:** the generated kernels run ~180× slower than hand-written ones
(7 s → ~1270 s on `bh3` RCCSDT), attributed to intermediates rebuilt inside loops and CSE being
disabled under dressing. That is a bigger win than any factorization and does not require new
equations.

## Why this took five failed attempts to find

Every attempt passed its gate and made the energy **worse**:

| attempt | change | `E_corr` |
|---|---|---|
| — | dressed, no fix | −0.0247182895 |
| exempt intermediates from the bra/ket sort | | −0.0145063653 |
| make the builder follow the residual's layout | | −0.0118486498 |
| — | undressed reference (exact) | **−0.0517746319** |

Two reasons, both worth carrying forward:

**1. The only equivalence check was blind to the defect.** `_eri_canonical` folds `v`'s bra↔ket
exchange symmetry — valid for `<pq||rs>`, invalid for a spatial `<pq|rs>`. On spin-adapted input it
reported **0 mismatches** on a case numerically wrong by 8.06e+02. Fixed: it now takes
`spatial=True`, which restricts the fold to the four parity-`+1` relations and correctly reports 30
mismatches.

**2. Structural gates were treated as correctness gates.** A layout-agreement gate passed 5/5 on a
kernel that was 72 % short. Layout self-consistency is necessary but does not constrain values. A
secondary tell was ignored: iteration count dropping 32 → 7 means terms went *missing*, not that
indexing improved.

## What is kept

| artifact | why |
|---|---|
| `spatial=True` on `verify_dressed_equation` / `_eri_canonical` | closes the blind spot; valuable independent of dressing |
| `tests/test_dressed_numeric_oracle.py` | intermediates must equal their primitive expansion |
| `tests/test_dressed_spatial_equivalence.py` | recognition must be value-preserving on spatial terms |
| `GeneratedKernelsAreReachableTests` | a kernel with no caller passes every other check |

The numeric gates matter beyond dressing: they catch **value** defects that structural gates cannot,
and the still-open rank-3 triples defect needs exactly that instrument.

**Test-system note, load-bearing:** every numeric gate uses `no=3, nv=4`. On Be/STO-3G
(`nv == no == 4`) a wrongly-ordered tensor read stays in bounds and returns a wrong number silently;
asymmetric extents make it raise instead.

## If dressing is revisited

**For UCC, the mechanism predicts it would work.** UCC keeps per-spin-block tensors
(`ucc_independent_blocks(4)` → `aaaa`, `abab`, `bbbb` as separate arrays) rather than folding to one
spatial tensor, so each block stays close to the spin-orbital form where recognition is correct.
Untested — `ucc_adapt_equations` does not exist yet and U1 is unstarted.

Two things to settle first, cheaply, while U1 is being scoped:

- **U1.1 is the same shape as this defect.** Its stated problem — "the bridge drops the block" — is
  information a downstream consumer needs being discarded at an adaptation boundary, exactly as the
  spin-case identity was here. Solving it correctly may be the same work as making dressing
  block-aware.
- **`spatial=True` likely applies to UCC too.** A `v_abab` block does not have the full 8-fold
  antisymmetry, so UCC verification could inherit the same blind spot. Check before it costs
  another investigation.

## Not fixed by this decision

The **undressed rank-3 triples residual is independently wrong** — it writes ~45 % of the elements
it should, with unrelated values on those. That defect predates dressing, is unaffected by turning it
off, and blocks a path that is in production. See `CCGEN_RANK3_TRIPLES_DEFECT.md`.

## Key code locations

| what | where |
|---|---|
| seeded operator definitions | `ccgen/optimization/dressing.py`, `seeded_operators()` |
| recognition / assembly | `assemble_dressed_equation`, same file |
| basis-aware ERI fold | `_eri_canonical(..., spatial=)`, `_ERI_PERMUTATIONS_SPATIAL`, same file |
| symbolic verifier | `optimization/dressed_equation.py`, `verify_dressed_equation(..., spatial=)` |
| spin adaptation | `ccgen/spin.py`, `spin_adapt_equations` |
| spatial ERI symmetry contract | `ccgen/emit/planck_tensor_cpp.py`, `_ERI_SYMMETRY_PERMUTATIONS` |
| contraction cost model | `ccgen/optimization/factorize.py`, `contraction_tree_cost` |
