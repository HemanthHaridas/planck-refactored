# How does a generated CCSDTQ kernel become a correct spatial energy?

Spin-adapting a spin-orbital CC equation to a spatial (restricted) one is exact at ranks 1–3 with a
single amplitude block per rank. **At rank 4 it is not**: `t4` has *two* independent Sz sectors, and
a solver that stores one of them silently loses the entire T4 contribution — CCSDTQ collapses onto
CCSDT.

This answers what that structure is, why one block is not enough, and how the chain
(generation → spin adaptation → bridge → C++ runtime) is proven correct end to end.

## The structure: rank 2n has ⌊n/2⌋+1 independent Sz sectors

A spatial amplitude block is an Sz sector of the spin-orbital manifold. Through rank 3 there is
exactly one independent sector per rank, so the spatial residual is a single tensor and the
adaptation is a relabelling. At rank 4:

| block | Sz sector | independent? |
|---|---|---|
| `t4` (reference) | `aabbaabb` | yes |
| `t4_aaabaaab` | `aaabaaab` | **yes — not reducible to the reference** |
| `abbb…` | — | folds onto `aaab` by spin flip |

`aaab` is **not** derivable from `aabb` — proven, not assumed, and not even via a shared `tau`. So
the spatial CCSDTQ residual is a *set* of blocks, and every layer downstream has to carry the set:
the bridge must name the sector, the equation generator must emit a residual per stored block, and
the solver must allocate and drive each block from its own residual.

The generalisation is `floor(n/2)` sectors for rank `2n`, so this is the first rank where the
single-block assumption breaks, not a rank-4 special case.

## The two bridge mechanisms

`spinterm_to_algebraterm` failed on 718/859 terms at rank 6 via **exactly two** mechanisms, found by
partitioning the failures rather than by inspection:

1. **Spin** — the bridge dropped per-index spin. Fixed by a β-majority `t3` flip (718 → 52 → 0 with
   the layout half).
2. **Layout** — `free_indices` were ordered by first appearance, which transposes the residual.
   Fixed by fixing the between-space order.

**Contract, load-bearing:** the free-index between-space order is **occ-first**, matching the C++
runtime's `rank_dims`. Both halves landed; the bridge is exact per-term and whole-residual at rank 6,
and at rank 8 once the sector is named (`t4_aaabaaab`).

## How we know the chain is correct

**Be/STO-3G has 4 electrons, so CCSDTQ ≡ FCI exactly.** That makes it a true oracle rather than a
self-consistency check, and it is the numeric gate whose *absence* let the spin-orbital-vs-spatial
defect ship in the first place (the arbitrary-solver unit test used a toy energy kernel).

| layer | gate | result |
|---|---|---|
| bridge algebra (rank 6 + rank 8) | `test_rank8_bridge_solve_path`, `test_rank6_*` (`test_spin.py`) | **12 passed** |
| Python spatial solver | `GeneratedCcsdtqFciGate.test_ccsdtq_spin_adapted_reaches_fci` (`CCGEN_SLOW_TESTS`, ~11 min) | `E_corr = -0.0517746318` vs FCI `-0.0517746319`, **gap 6.4e-11** |
| C++ runtime end-to-end | `be_rccsdtq_sto3g` regression case | **PASS** — `-14.4036551081`, and PySCF `rccsdtq` independently gives `-14.4036551082` |

The recovered T4 contribution is **−4.4e-6** — precisely what the single-sector solver missed, when
adapted CCSDTQ equalled adapted CCSDT and sat ~3e-6 short.

Two independent implementations (Python damped-Jacobi, C++ block-keyed runtime) reach the same
number, and a third code (PySCF) agrees to 1e-10.

## What was wrong before, and why it was invisible

- **The algebra** referenced one block. The residual terms emit `t4_aaabaaab` factor reads, but the
  amplitude dict supplied no such key, so even the *reference* residual was evaluated against a
  missing second sector.
- **The solver** iterated a fixed `targets` list (`{singles, doubles, triples, quadruples}`), so
  `quadruples_aaabaaab` was never read and its block stayed zero.

Both are the same defect at different layers: the algebra names two blocks, so storage and the
update loop must carry two. The fix is block-keyed throughout — `spin_adapted_solve_blocks` maps
each residual key to `(key, rank, tensor_name, sector_tag)` and drives the loop off the actual keys.

**A correction worth keeping:** the sector denominator is *identical* to the reference rank
denominator, not built from the sector's Sz layout. For an RHF reference the orbital energies are
spin-free, so `Σε_occ − Σε_vir` over the spatial slots is the same for `aabb` and `aaab`. Denominators
are keyed by rank alone.

The failure was invisible because CCSDTQ-with-no-T4 is a *converged, self-consistent* CCSDT answer.
Nothing crashes and nothing fails to converge; the energy is simply 3e-6 short, which looks like a
tolerance question rather than a missing manifold.

## Constraints for anyone extending this

- **Do not iterate on Be.** The CCSDTQ solve is ~11 minutes (4557-term quadruples adaptation + t4
  Jacobi). Develop against `test_rank8_bridge_solve_path` (~30 s), which already proves the residual
  is exact given both blocks, and reserve Be for the final oracle.
- **PySCF lives in `tests/pyscf/.venv`**, not the system Python. Running these gates with the wrong
  interpreter reports `SKIPPED [1] pyscf not importable`, which is easy to misread as the
  slow-test guard.
- **Rank ≥ 10 has no numeric oracle.** No small system makes CCSDTQP exact and PySCF has no
  quad/pentuple amplitudes; the arbitrary-order algebra is gated structurally instead.
- **A 4-electron closed shell is always an oracle.** Any of them gives CCSDTQ ≡ FCI, so a cheaper
  one than Be is a legitimate substitute (H4 needs damping ≥ 0.5 and a non-degenerate geometry —
  equal spacing gave NaN under plain Jacobi).

## Key code locations

| what | where |
|---|---|
| sector enumeration per rank | `independent_spin_blocks`, `python/ccgen/spin.py` |
| the bridge | `spinterm_to_algebraterm`, `python/ccgen/spin.py` |
| residual-per-block emission | `spin_adapt_equations`, same file |
| Python block-keyed solver | `spin_adapted_solve_blocks`, `solve_spin_adapted_spatial` (`test_reference_vs_pyscf.py`) |
| C++ sector carry + drive | `ensure_amplitude_sectors`, `evaluate_generated_arbitrary_order_residuals` (`generated_arbitrary_runtime.cpp`) |
| codegen switch | `--spin-adapt` → `print_cpp_planck(spin_adapt=...)`, `python/generate_planck_cc_kernels.py` |
| end-to-end gate | `be_rccsdtq_sto3g` in `tests/regression_cases.json` |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`, which are canonical.
