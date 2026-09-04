# ccgen Spin-Orbital to Spatial Adaptation

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**How does a spin-orbital CC equation become a spatial one, and why does that layer exist at all?**

## Short answer

ccgen derives everything in spin-orbital (GCC) form: `indices.Index` carries a space (occ/vir/gen) and no spin. Production CC needs spatial equations — restricted (RCC) for an RHF reference, unrestricted (UCC) for a UHF one. A spin-adaptation layer sits as an insertable stage between generation and lowering (`generate (GCC AlgebraTerms) -> [spin adaptation] -> spatial AlgebraTerms -> lowering -> emit`), wrapping each spatial `Index` in a lightweight `SpinIndex` rather than adding a spin field to `Index` itself, so it does not perturb the validated GCC generation/canonicalization path. Generation (both the wick and diagram engines), canonicalization, lowering and the emitters are all untouched by it, and it is engine-agnostic because it operates on the `AlgebraTerm`s either engine produces.

The layer exists for cost, not correctness — the opposite of the usual intuition. The GCC equations already give the right closed-shell answer: evaluated on spin-orbitals built from an RHF reference, ccgen's GCC energy matches PySCF RCCSD `e_corr` to 1e-8 and PySCF GCCSD to 7e-9 (GHF-CC and RCC are energy-equivalent for a closed shell, because GHF reduces to RHF at the closed-shell minimum). So RCC/UCC are not needed for the right number — they exist to exploit spin symmetry for efficiency, since the spin-orbital representation is ~16x the `t2` storage and ~64x the doubles-contraction FLOPs of the spatial one (water/STO-3G; the ratio is `2^(2·rank)` and grows with system size). That makes this layer a performance prerequisite for replacing the hand-written `src/post_hf/cc` solvers, not a correctness one — if the goal were only "get the energy", GCC-on-RHF suffices.

Each spin-orbital index `p` is (spatial `p̄`, spin σ); a GCC term is a sum over spin-orbital indices, and adaptation performs the spin summation. UCC keeps the blocks resolved — one residual per stored block (`doubles_aaaa`, `doubles_abab`, `doubles_bbbb`), with the raw GCC coefficients unchanged (bookkeeping only). RCC collapses them under α ≡ β — the only place coefficients genuinely change, where the familiar `2J − K` structure appears out of the merge; the exchange term is not an extra input, it is what the collapse produces.

`lowering/restricted_closed_shell.py` does **not** perform spin adaptation. It re-lays-out spin-orbital terms into spatial blocks and explicitly does not spin-integrate — a distinction worth holding onto, because that module's own ERI symmetry table later turned out to be wrong for exactly the spatial blocks it produces (see `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`).

## Where the logic lives

- `python/ccgen/spin.py` — the adaptation layer
- `ucc_integrate_term_antisym`, `python/ccgen/spin.py` — the antisymmetry re-expression (trap 1)
- `_antisym_to_allowed`, `python/ccgen/spin.py` — the rank-2n generalization (trap 2)
- `spin.py:577` — the spatial representative block (rank 2 `abab`, rank 3 `aabaab`, **not** all-alpha)
- `python/ccgen/lowering/restricted_closed_shell.py` — re-layout only, no spin integration

## What invariants matter

### 1. `Index` identity must not carry spin

`Index`'s identity `(name, space, is_dummy)` is baked into every canonicalize/wick/diagram hash and equality. Adding a spin field would perturb the validated GCC path.

Design rule:

- Represent spin by wrapping a spatial `Index` in a `SpinIndex` (spatial base + spin) at the adaptation layer, never by extending `Index` itself.

### 2. A fixture with more symmetry than the real object cannot see a defect that abuses symmetry

This is the pattern behind all four traps below, and it recurs elsewhere in the project (it is also why a separate value gate missed the 41/288 ERI defect described in `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`, whose fixture antisymmetrizes `v`).

Design rule:

- When a gate uses a synthetic tensor (`v`, `t2`, `t3`, ...), check which physical symmetries it does or does not carry before trusting a green result. A fixture that is too symmetric (zero in the forbidden blocks) or too asymmetric (missing a real physical symmetry) can each hide or manufacture a defect.

### 3. Line-swap antisymmetry is not the same as single-axis antisymmetry

The `aab`/`abb` `t3` blocks are antisymmetric under the occ-pair swap alone and under the vir-pair swap alone, but symmetric under the joint swap. A physical line-swap of two same-spin lines *is* the joint swap — so the block returns the same value while the spin-orbital `t3` needs a `−1`.

Design rule:

- The value must be `sign(P) × blk[canonical-line-order]`, where `P` permutes the three **lines**, not axes. Never substitute a block's own single-axis antisymmetry for a line-swap sign. Pinned by a failing example (`t3so[1,3,0, 7,13,8]`) so it cannot regress silently.

## What was found (four traps, each of which passed a gate first)

Every one of these was found *after* something green, which is why they are recorded rather than summarised.

1. **A synthetic `v` hid the exchange entirely.** The `block_exists` filter looked like a harmless optimization. Every gate up to that point used a spin-conserving-per-line synthetic `v` whose forbidden blocks are zero, so the filter only ever dropped zero terms. On a real antisymmetric `<pq||rs>` those blocks are nonzero — they are the exchange — and the filter silently discarded them (residual off by ~0.06). Fixed by `ucc_integrate_term_antisym`, which re-expresses each forbidden factor into its allowed block via bra/ket swaps carrying `−1`; that `−1` *is* the `−K`.
2. **Rank-4 hardcoding silently dropped valid terms at higher rank.** `_antisym_to_allowed` had only rank-2/rank-4 candidate swaps; on a rank-6/8 factor it fell through and returned `None` ("genuinely zero") for cases actually reachable by antisymmetry. Generalised to rank-2n: a factor maps to an allowed block iff `sorted(bra_spins) == sorted(ket_spins)`, via a within-group permutation whose parity product gives the sign. `None` now means a genuine multiset mismatch.
3. **Line-swap antisymmetry is not axis antisymmetry** — see invariant 3 above.
4. **Rank 4 has two independent Sz sectors.** `t4` needs `aabb` *and* `aaab`; `aaab` is not reducible to `aabb`, proven and not merely assumed. A solver storing one silently loses the entire T4 contribution and CCSDTQ collapses onto CCSDT — converged, self-consistent, and 3e-6 short. Answered in full in `docs/CCGEN_CCSDTQ_MULTISECTOR.md`.

## Validation strategy that should remain in place

- RCC reproducing the GCC residual on real integrals to ~1e-16, vanishing at PySCF's converged RCCSD amplitudes
- UCC validated against PySCF UCCSD to ~6e-16 across all five blocks (`docs/CCGEN_UCC_NUMERIC_VALIDATION.md`)
- `PLANCK_CC_SPIN_ADAPT` kept defaulted ON (as of 2026-08-26) — it was OFF for byte-compatibility with a historical emit that is ~4x wrong, which cost a full investigation before anyone checked the flag; see `docs/CCGEN_SPIN_ADAPT_DEFAULT.md`

Status: complete and validated end to end.

## Related docs

| doc | question |
|---|---|
| `docs/CCGEN_GCC_TO_UCC_BRIDGE.md` | how the adapted terms reach the C++ runtime |
| `docs/CCGEN_CCSDTQ_MULTISECTOR.md` | the rank-4 multi-sector half |
| `docs/CCGEN_UCC_NUMERIC_VALIDATION.md` | how UCC was validated against PySCF |
| `docs/CCGEN_SPIN_ADAPTER_CONTRACT.md` | what `ucc_integrate_term_antisym` guarantees |
| `docs/CCGEN_SPIN_ADAPT_DEFAULT.md` | why the flag was OFF, and what that cost |
