# CCSDTQ = FCI verification for Be — scope

**The one question.** Be has 4 electrons, so CCSDTQ is exact: CCSDTQ ≡ FCI. Does
the spin-adapted (spatial) generated CCSDTQ, solved to convergence, reproduce the
FCI correlation energy to ~1e-8? This is the numeric oracle whose **absence** let
the spin-orbital-vs-spatial defect ship (the arbitrary-solver unit test uses a
toy energy kernel). Passing it is the proof that the whole ccgen CCSDTQ chain —
generation → spin adaptation → multi-Sz-sector bridge → solve — is correct.

**Status: RED (`GeneratedCcsdtqFciGate.test_ccsdtq_spin_adapted_reaches_fci`,
xfail, `CCGEN_SLOW_TESTS`).** Today the adapted CCSDTQ equals adapted CCSDT
(T4 ≈ 0), leaving CCSDTQ − FCI ≈ 3e-6. The R3.1.3 work (see
`CCGEN_R3_HIGHER_RANK_BRIDGE_SCOPE.md`) fixed the **algebra** — the residual is
now exact and multi-sector — but the **solver** this gate runs
(`solve_spin_adapted_spatial`) still under-drives T4. This doc scopes closing
that gap. It is coupled to `CCGEN_KERNEL_WIRING_MULTISECTOR_SCOPE.md`: the Python
verification solver and the C++ runtime need the **same** multi-sector storage +
update logic, so the two land together.

---

## Why it is still RED after R3.1.3

`solve_spin_adapted_spatial` (`python/ccgen/tests/test_reference_vs_pyscf.py`) is
a damped-Jacobi RCC solver. Two concrete gaps, both a direct consequence of t4
having **two** independent Sz sectors (`t4` reference `aabbaabb` + `t4_aaabaaab`,
see the R3.1.3 doc):

1. **It iterates only the reference residual.** The loop runs
   `for m in targets` over `{singles, doubles, triples, quadruples}` — it never
   reads `adapted["quadruples_aaabaaab"]`, so the second t4 sector is never
   updated and stays zero.
2. **It stores one t4 tensor.** `amps["t4"]` is the reference block; there is no
   `amps["t4_aaabaaab"]`. The residual terms now emit `t4_aaabaaab` factor reads
   (R3.1.3c), which the `tn` dict does not supply, so even the reference residual
   is evaluated against a missing/zero second sector.

Both are the *solver* analog of the bridge fix: the algebra references two
blocks; the solver must store and update two blocks.

## The fix (Python verification solver)

Small, verifiable steps against the FCI oracle. Iterate on a **cheap** system
first (see "Fast inner loop" below) — never the ~10 min Be CCSDTQ solve.

- **V1 — enumerate the residual/amplitude blocks the solver must carry. LANDED.**
  `spin_adapted_solve_blocks(adapted.keys())` maps each residual key to
  `(key, rank, tensor_name, sector_tag)`: a bare manifold (`quadruples`) → `t4`
  (tag None), a tagged sector (`quadruples_aaabaaab`) → `t4_aaabaaab`. Drives the
  solve loop off the actual keys, not a fixed `targets` list. *Gates:*
  `test_v1_solve_block_enumeration` — CCSDTQ yields exactly
  `{singles→t1, doubles→t2, triples→t3, quadruples→t4,
  quadruples_aaabaaab→t4_aaabaaab}` with distinct tensor names, energy excluded;
  `test_v1_blocks_backward_compatible_for_ccsdt` — CCSDT is `t1/t2/t3`, no tags
  (the multi-block enumeration is a no-op below rank 4).

- **V2 — allocate + zero-init every amplitude block, keyed by tensor name.**
  `amps["t4"]` and `amps["t4_aaabaaab"]` each get their own array and their own
  denominator. The sector denominator uses the **sector's** external Sz layout
  (3α1β per half for `aaabaaab`), not the reference's — the orbital-energy
  denominator `Σε_occ − Σε_vir` is layout-dependent only through which
  occ/vir indices sit in which slot, so build it from the sector's external
  block template (`_representative_block_for_sector`). *Gate:* the sector
  denominator has the right shape and its diagonal matches a hand-built
  3α1β/3α1β denominator on a 2-orbital toy.

- **V3 — the residual reads all blocks; the update writes each block from its
  own residual.** Each iteration: build `tn` carrying **every** amplitude tensor
  by name (`t1, t2, t3, t4, t4_aaabaaab, v, f`); for each block key evaluate its
  residual `adapted[key]` and Jacobi-update its own amplitude
  `amps[name] += damping · R / D[name]`. The energy reads the converged blocks.
  *Gate:* the fast-system CCSDTQ energy now moves OFF the CCSDT value (T4 no
  longer ~0) — the direct symptom that the second sector is being driven.

- **V4 — the Be oracle.** With V1–V3, `test_ccsdtq_spin_adapted_reaches_fci`
  reaches FCI to 1e-8. Remove the `@expectedFailure`. Run once, not in the loop.

## Fast inner loop (do not iterate on Be)

Be CCSDTQ is ~10 min (large quadruples manifold + t4 Jacobi). Two faster proxies
to develop V1–V3 against, reserving Be for V4:

- **The rank-8 bridge gate** `test_rank8_bridge_solve_path` (~30 s) already proves
  the *residual* is exact given both t4 blocks. V1–V3 are about the *solver*
  around it, so a minimal solver harness on the same N2/sto-3g fixture (perturbed
  amps, one iteration, check both residual blocks are consumed) is the seconds-
  scale development gate.
- **A smaller 4-electron closed shell than Be** if one converges faster (e.g. a
  short H4 at a non-degenerate geometry — note H4 equal-spacing gave NaN under
  plain Jacobi historically; use damping ≥ 0.5 and a non-degenerate geometry).
  Any 4-electron closed shell is CCSDTQ ≡ FCI, so it is an oracle too.

## The coupling to the C++ wiring

V2/V3 are the **specification** for the C++ multi-sector runtime
(`CCGEN_KERNEL_WIRING_MULTISECTOR_SCOPE.md`): the same block enumeration, the
same per-sector denominator, the same per-block Jacobi/DIIS update. Land the
Python solver first — it is the executable reference the C++ path must match
block-for-block — then port. The Be FCI number is the shared acceptance test for
both: the Python solver hits it in `solve_spin_adapted_spatial`, the C++ runtime
hits it end-to-end in the `hartree-fock` binary on a `correlation rccsdtq` Be
input.

## Acceptance

- `test_ccsdtq_spin_adapted_reaches_fci` (Python) green, `@expectedFailure`
  removed: adapted CCSDTQ = FCI to 1e-8 on Be/sto-3g.
- The CCSDT gate (`test_ccsdt_spin_adapted_solves_between_ccsd_and_fci`) stays
  green (no regression from the multi-block loop; CCSDT has one sector so its
  path is unchanged).
- A note in `generated_ccsdtq_energy_wrong` flipping the blocker from "algebra"
  to "done" once V4 passes.

## What this does NOT cover

- The C++ end-to-end Be number and the codegen `spin_adapt` switch — that is
  `CCGEN_KERNEL_WIRING_MULTISECTOR_SCOPE.md` (the coupled doc).
- Rank ≥ 10 (t5+) oracles — no small system makes CCSDTQP exact, and PySCF has no
  quad/pentuple amplitudes. The arbitrary-order *algebra* is already gated
  structurally (R3.1.4); a numeric t5 oracle is out of scope for both docs.
