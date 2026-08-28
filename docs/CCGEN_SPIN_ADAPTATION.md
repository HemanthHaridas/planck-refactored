# How does a spin-orbital CC equation become a spatial one?

ccgen derives everything in **spin-orbital (GCC)** form: `indices.Index` carries
a space (occ/vir/gen) and no spin. Production CC needs **spatial** equations —
restricted (RCC) for an RHF reference, unrestricted (UCC) for a UHF one. This
answers how the layer between them works, why it exists, and the traps it took
to get right.

## Why the layer exists: cost, not correctness

This is the fact most worth keeping straight, because it is the opposite of the
intuition.

**The GCC equations already give the right closed-shell answer.** Evaluated on
spin-orbitals built from an RHF reference, ccgen's GCC energy matches PySCF
RCCSD `e_corr` to 1e-8 and PySCF GCCSD to 7e-9. GHF-CC and RCC are
energy-equivalent for a closed shell, because GHF reduces to RHF at the
closed-shell minimum.

So RCC/UCC are **not needed for the right number**. They exist to exploit spin
symmetry for efficiency: the spin-orbital representation is ~16x the `t2`
storage and ~64x the doubles-contraction FLOPs of the spatial one (water/STO-3G;
the ratio is `2^(2·rank)` and grows with system size).

That makes this layer a **performance prerequisite** for replacing the
hand-written `src/post_hf/cc` solvers, not a correctness one. If the goal were
only "get the energy", GCC-on-RHF suffices and none of this is needed.

## Where it sits

```
generate (GCC AlgebraTerms) -> [spin adaptation] -> spatial AlgebraTerms -> lowering -> emit
```

An insertable stage. Generation (both the wick and diagram engines),
canonicalization, lowering and the emitters are untouched, and it is
engine-agnostic by construction because it operates on the `AlgebraTerm`s either
engine produces.

**Not a spin field on `Index`.** `Index`'s identity `(name, space, is_dummy)` is
baked into every canonicalize/wick/diagram hash and equality; adding a spin field
would perturb the validated GCC path. The layer wraps a spatial `Index` in a
lightweight `SpinIndex` (spatial base + spin) instead.

`lowering/restricted_closed_shell.py` does **not** do this. It re-lays-out
spin-orbital terms into spatial blocks and explicitly does not spin-integrate —
a distinction worth holding onto, because that module's own ERI symmetry table
later turned out to be wrong for exactly the spatial blocks it produces
(`CCGEN_WIRING_THE_DERIVATION_ROUTE.md`).

## The two directions

Each spin-orbital index `p` is (spatial `p̄`, spin σ). A GCC term is a sum over
spin-orbital indices; adaptation performs the spin summation.

**UCC keeps the blocks resolved.** One residual per stored block
(`doubles_aaaa`, `doubles_abab`, `doubles_bbbb`). The raw GCC coefficients come
out unchanged — this half is bookkeeping.

**RCC collapses them under α ≡ β.** This is where the real derivation lives, and
the only place coefficients genuinely change: the familiar `2J − K` structure
appears out of the merge. The exchange term is not an extra input — it is what
the collapse produces.

## Four traps, each of which passed a gate first

Every one of these was found *after* something green, which is why they are
recorded rather than summarised.

**1. A synthetic `v` hid the exchange entirely.** The `block_exists` filter
looked like a harmless optimization. Every gate up to that point used a
spin-conserving-per-line synthetic `v` whose forbidden blocks are *zero*, so the
filter only ever dropped zero terms. On a real antisymmetric `<pq||rs>` those
blocks are nonzero — **they are the exchange** — and the filter silently
discarded them (residual off by ~0.06).

The fix, `ucc_integrate_term_antisym`, re-expresses each forbidden factor into
its allowed block via bra/ket swaps carrying `−1`. That `−1` *is* the `−K`.

**2. Rank-4 hardcoding silently dropped valid terms at higher rank.**
`_antisym_to_allowed` had only rank-2/rank-4 candidate swaps; on a rank-6/8
factor it fell through and returned `None` ("genuinely zero") for cases actually
reachable by antisymmetry. Generalised to rank-2n: a factor maps to an allowed
block iff `sorted(bra_spins) == sorted(ket_spins)`, via a within-group
permutation whose parity product gives the sign. `None` now means a genuine
multiset mismatch.

**3. Line-swap antisymmetry is not axis antisymmetry.** The `aab`/`abb` `t3`
blocks are antisymmetric under the occ-pair swap alone and under the vir-pair
swap alone, but **symmetric under the joint swap**. A physical line-swap of two
same-spin lines *is* the joint swap — so the block returns the same value while
the spin-orbital `t3` needs a `−1`.

The value must be `sign(P) × blk[canonical-line-order]`, where `P` permutes the
three **lines**, not axes. The block's own single-axis antisymmetry must not be
used for this. Pinned by a failing example (`t3so[1,3,0, 7,13,8]`) so it cannot
regress silently.

**4. Rank 4 has two independent Sz sectors.** `t4` needs `aabb` *and* `aaab`;
`aaab` is not reducible to `aabb`, proven and not merely assumed. A solver
storing one silently loses the entire T4 contribution and CCSDTQ collapses onto
CCSDT — converged, self-consistent, and 3e-6 short. Answered in full in
`CCGEN_CCSDTQ_MULTISECTOR.md`.

**The pattern across all four:** a fixture with more symmetry than the real
object cannot see a defect that abuses symmetry. That recurs — it is also why
the value gate missed the 41/288 ERI defect, whose fixture antisymmetrizes `v`.

## Status

Complete and validated end to end. RCC reproduces the GCC residual on real
integrals to ~1e-16 and vanishes at PySCF's converged RCCSD amplitudes; UCC is
validated against PySCF UCCSD to ~6e-16 across all five blocks
(`CCGEN_UCC_NUMERIC_VALIDATION.md`).

`PLANCK_CC_SPIN_ADAPT` **defaults ON** as of 2026-08-26. It was OFF for
byte-compatibility with a historical emit that is ~4x wrong, which cost a full
investigation before anyone checked the flag — see
`CCGEN_SPIN_ADAPT_DEFAULT.md`.

## Related

| doc | question |
|---|---|
| `CCGEN_GCC_TO_UCC_BRIDGE.md` | how the adapted terms reach the C++ runtime |
| `CCGEN_CCSDTQ_MULTISECTOR.md` | the rank-4 multi-sector half |
| `CCGEN_UCC_NUMERIC_VALIDATION.md` | how UCC was validated against PySCF |
| `CCGEN_SPIN_ADAPTER_CONTRACT.md` | what `ucc_integrate_term_antisym` guarantees |
| `CCGEN_SPIN_ADAPT_DEFAULT.md` | why the flag was OFF, and what that cost |

## Key code locations

| what | where |
|---|---|
| the layer | `python/ccgen/spin.py` |
| the antisymmetry re-expression (trap 1) | `ucc_integrate_term_antisym`, same file |
| the rank-2n generalization (trap 2) | `_antisym_to_allowed`, same file |
| the spatial representative block | `spin.py:577` — rank 2 `abab`, rank 3 `aabaab`, **not** all-alpha |
| re-layout only, no spin integration | `python/ccgen/lowering/restricted_closed_shell.py` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
