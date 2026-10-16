# ccgen Recognition Dressing vs. Spin Adaptation

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Why does ccgen's *recognition* dressing route (hand-seeded Stanton-Gauss
fingerprints) give wrong kernels once the equations are spin-adapted, and why
was it retired rather than fixed?**

It is about one of ccgen's two dressing routes. The other, *derivation*
(`factorize.py`, operators taken from each term's own contraction tree), does
compose with spin adaptation and is the production dressed path
(`--dressing derived`; 3.12× on LiH and 3.61× on CH4 end to end, with energies
matching the undressed baseline). Read `CCGEN_TWO_DRESSING_ROUTES.md` for that
route and for why its existence overturned this document's original, broader
conclusion that "dressing and spin adaptation do not compose".

## Short answer

Recognition subtracts what each matched operator absorbs from the remaining
terms. That subtraction is valid only against the exact term set it was computed
for, and spin adaptation changes the term set. Each transform is correct alone;
the composition is wrong in either order. The seeded operators also lean on the
antisymmetry of `<pq||rs>`, which spatial integrals lack, so a spatial operator
set would have to be re-derived, not relabeled. Five fix attempts each passed
their own gate and made the energy worse, and on Be/STO-3G CCSDTQ the result
stayed 52 % short.

**Decision:** recognition stays retired. `-DPLANCK_CC_DRESSING=recognized`
still generates and compiles, but its spin-adapted RCC kernels are wrong.
`-DPLANCK_CC_DRESSING=derived` is the supported dressed path.

## Where the logic lives

- `ccgen/optimization/dressing.py`, `seeded_operators()` — seeded operator definitions
- `assemble_dressed_equation`, same file — recognition / assembly
- `_eri_canonical(..., spatial=)`, `_ERI_PERMUTATIONS_SPATIAL`, same file — basis-aware ERI fold
- `optimization/dressed_equation.py`, `verify_dressed_equation(..., spatial=)` — symbolic verifier
- `ccgen/spin.py`, `spin_adapt_equations` — spin adaptation
- `ccgen/emit/planck_tensor_cpp.py`, `_ERI_SYMMETRY_PERMUTATIONS` — spatial ERI symmetry contract
- `ccgen/optimization/factorize.py`, `contraction_tree_cost` — contraction cost model

## What invariants matter

### 1. Dressing is a valid rewrite on spin-orbital (GCC) input

Dressing rewrites a CC residual to reference recognized intermediates (the Stanton-Gauss `Wmnij`/`Wabef`/`Wmbej` plus the `tau`/`tau_c` pseudo-amplitudes) instead of their expanded contractions, so a repeated sub-expression is computed once. On spin-orbital (GCC) input it is a valid rewrite: `verify_dressed_equation` reports 0 mismatches; recognition changes 2 operator-free terms' coefficients and those changes are exactly compensated by what the operators absorb. That is what a correct factorization looks like.

### 2. Recognition's subtraction is only valid against the exact term set it was computed for

Recognition works by matching a sub-expression and subtracting what the operator absorbs from the remaining terms. That subtraction is only valid against the term set it was computed for. Spin adaptation changes the term set — splitting terms into spin cases, merging others. So:

- **`dress → adapt`** (what production does): the subtraction was computed against GCC terms that no longer exist after adaptation.
- **`adapt → dress`**: recognition matches against spatial terms, but the operator *definitions* were derived in the spin-orbital basis and were never valid there.

Measured on `ccsd` doubles:

| path | operator-free terms with changed coefficients | compensated? |
|---|---|---|
| GCC (`dress` only) | 2 | **yes** — valid rewrite |
| `adapt → dress` | 8 | **no** — ‖diff‖ = 8.06e+02 against ‖R‖ = 983.79 |
| `dress → adapt` (production) | — | **no** — Be/STO-3G CCSDTQ 52 % short |

The end-to-end symptom: dressed `E_corr` = −0.0247182895 against an exact −0.0517746319.

Design rule:

- Never assume a factorization derived against one term set remains valid after that term set is transformed by a separate rewrite (here, spin adaptation) — check the compensation identity again against the new term set, not just the old one.

### 3. The seeded operators are built on an antisymmetry that spatial terms lack

The seeded definitions carry `P(ij)` antisymmetrizers and `1/4` weights that are artifacts of `<pq||rs> = <pq|rs> − <pq|sr>`. A spatial residual has only the four `+1` symmetries of `<pq|rs>`. Consequently the spatial residual has more distinct contraction topologies — 13 `v` index-space patterns vs 9 in GCC, with `ovvo`/`ovov`/`voov`/`vovo` all appearing separately where antisymmetry makes them one object. So a spatial `Wmbej` is not a relabeled GCC `Wmbej`; it is several different operators. The spatial factorization is a different object, and deriving it is research, not porting.

Design rule:

- Treat any operator seeded/verified under an antisymmetric (`<pq||rs>`) convention as invalid on a spatial (`<pq|rs>`) residual until re-derived; do not assume a relabeling suffices.

### 4. A structural gate proving self-consistency is not a correctness gate

Every fix attempt below passed its own gate and made the energy worse. Two reasons, both worth carrying forward:

1. **The only equivalence check was blind to the defect.** `_eri_canonical` folds `v`'s bra↔ket exchange symmetry — valid for `<pq||rs>`, invalid for a spatial `<pq|rs>`. On spin-adapted input it reported 0 mismatches on a case numerically wrong by 8.06e+02. Fixed: it now takes `spatial=True`, which restricts the fold to the four parity-`+1` relations and correctly reports 30 mismatches.
2. **Structural gates were treated as correctness gates.** A layout-agreement gate passed 5/5 on a kernel that was 72 % short. Layout self-consistency is necessary but does not constrain values. A secondary tell was ignored: iteration count dropping 32 → 7 means terms went missing, not that indexing improved.

Design rule:

- A layout/structural agreement gate is necessary but not sufficient for correctness — always pair it with a numeric oracle. If an iteration count drops sharply after a "structural" change, treat that as a value-loss signal, not an efficiency win, until proven otherwise.
- Make an ERI-symmetry fold explicit about which convention it assumes (`spatial=True` vs antisymmetric) — a fold silently valid only for `<pq||rs>` will report false equivalence on spatial input.

## What was found

**Why the composition fails is not worth deriving a fix for.** Measured with `factorize.py`'s contraction-tree cost model (`contraction_tree_cost`, which accounts for contraction order — unlike `IntermediateSpec.estimated_build_flops`, which counts elements and would overstate the benefit):

| quantity | (n_occ, n_vir) = (10,50) | (30,100) | (3,4) |
|---|---|---|---|
| GCC dressed saving, **actual** | 1.20× | 1.50× | 1.59× |
| spatial saving, **upper bound** | 2.78× | 4.00× | 5.27× |
| bound looseness, calibrated on GCC | ÷1.5 | ÷1.4 | ÷3.3 |
| spatial saving, **expected** | ~1.9× | ~2.8× | — |

Note the GCC saving *shrinks* as `n_vir/n_occ` grows, i.e. it pays least in the production regime.

Against a ~2× constant factor, the work required is: derive a spatial operator set (research; the literature factorization may not match ccgen's terms), build a spatial-capable matcher (the current one assumes antisymmetric factors), and maintain and validate a second equation path indefinitely.

At the time, a larger lever was untouched: the generated kernels' own cost. (A
"~180×" figure cited here then never recorded its dimensions and did not
reproduce.) That lever has since been worked: inlined tensor accessors (up to
206×), operator-build hoisting, transpose merging, the derivation route, and
OpenMP. See `CCGEN_WHY_GENERATED_IS_SLOW.md` and `CCGEN_KERNEL_PERFORMANCE.md`.

**Five fix attempts, all falsified**, each passing its own gate and making the energy worse:

| attempt | change | `E_corr` |
|---|---|---|
| — | dressed, no fix | −0.0247182895 |
| exempt intermediates from the bra/ket sort | | −0.0145063653 |
| make the builder follow the residual's layout | | −0.0118486498 |
| — | undressed reference (exact) | **−0.0517746319** |

## What was kept

| artifact | why |
|---|---|
| `spatial=True` on `verify_dressed_equation` / `_eri_canonical` | closes the blind spot; valuable independent of dressing |
| `tests/test_dressed_numeric_oracle.py` | intermediates must equal their primitive expansion |
| `tests/test_dressed_spatial_equivalence.py` | recognition must be value-preserving on spatial terms |
| `GeneratedKernelsAreReachableTests` | a kernel with no caller passes every other check |

The numeric gates matter beyond dressing: they catch value defects that structural gates cannot.

**Test-system note, load-bearing:** every numeric gate uses `no=3, nv=4`. On Be/STO-3G (`nv == no == 4`) a wrongly-ordered tensor read stays in bounds and returns a wrong number silently; asymmetric extents make it raise instead.

## Validation strategy that should remain in place

- `tests/test_dressed_numeric_oracle.py` and `tests/test_dressed_spatial_equivalence.py` — numeric gates on a `no=3, nv=4` fixture, chosen so an index-layout error raises instead of silently returning a wrong number
- `GeneratedKernelsAreReachableTests`, to catch a kernel with no production caller
- `_eri_canonical(..., spatial=True)` used on any spatial-residual equivalence check

## Related outcome: recognition does not carry over to UCC either

This document once predicted that recognition would work for UCC, since UCC
keeps per-spin-block tensors close to the spin-orbital form. Measured, it finds
**zero** operators on UCC terms. UCC factors are spin-tagged (`v_abab`,
`t2_aaaa`) while the fingerprints key on bare names, and the obvious tag-blind
match is unsound: it collapses 12 distinct spin-tagged contractions onto one
`Wmbej`. See invariant 3 of `CCGEN_TWO_DRESSING_ROUTES.md` and O6 in
`CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`. The basis-aware ERI fold
(`spatial=True`) remains the right guard for any UCC equivalence check, because
a `v_abab` block lacks the full 8-fold antisymmetry.

## Remaining architecture concern

None for this route: it is retired. The "undressed rank-3 triples residual is
independently wrong" concern once listed here was resolved. The kernel was
correct throughout; the wrong numbers came from the `PLANCK_CC_SPIN_ADAPT=OFF`
default and from a solver-representation mismatch (`CCGEN_SPIN_ADAPT_DEFAULT.md`,
`CCGEN_RANK3_KERNEL_AND_SOLVER.md`). The seven `expectedFailure` gates on this
route stay as a tripwire: an unexpected pass would mean the composition was
genuinely fixed.
