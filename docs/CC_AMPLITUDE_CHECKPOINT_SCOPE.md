# Restarting a CC run from persisted amplitudes (`.ccamp`)

**Answer.** A converged CC run can persist its amplitudes to a `<stem>.ccamp`
sidecar and a later run can seed its iteration from them instead of
cold-starting — for the generated arbitrary-order path (any rank, including
cross-rank: `ccsdt_gen` seeding a `cc4` run) and, since this doc's own
follow-on landed, for the hand-written rank-2 `run_rccsd` solver too. X0–X4.1
and X5.0/X5.1 are all landed; what remains (the hand-written rank-3 solver, a
UCC sidecar, and one still-unverified cross-rank gate) is tracked in
`CC_AMPLITUDE_CHECKPOINT_REMAINING_SCOPE.md`, not here.

It is the persistence sibling of W6, which warm-starts amplitudes **in
memory** within one run. Both share the same seed injection point,
`seed_arbitrary_order_amplitudes` — a restart from disk lands there
identically to a W6 in-memory seed.

## What's landed

| Piece | Where | What it does |
|---|---|---|
| Format + writer/loader | `src/post_hf/cc/cc_amplitude_checkpoint.{h,cpp}` | Little-endian, magic+version framed, rank-generic. Now version 3 (two follow-on fixes: the version-2 sector-drop defect, and version-3's `n_by_rank`/UHF-count extension for UCC — see below) |
| Write on a converged generated solve | `rccgen.cpp` (`run_rccgen`) | Gated on `_save_checkpoint && !_checkpoint_path.empty()`; write failure is a warning, never a run failure |
| Read + seed on restart | `rccgen.cpp` (`try_restart_from_sidecar`) | `load_cc_amplitudes` → `seed_arbitrary_order_amplitudes`; any error falls through to cold/W6, never fails the run |
| Rank-3 arbitrary emit | `PLANCK_CC_ARBITRARY_LOWER_RANKS` (CMake option) | Lowers the generated-kernel registry floor from 4 to 3, so `ccsdt_gen` can write a spatial rank-3 sidecar the same way `cc4`+ does |
| Cross-rank restart (`ccsdt_gen`→`cc4`) | Same write/read sites, rank-generic by construction | A `cc4` run reads a rank-3 sidecar and seeds T1/T2/T3, leaving T4 to iterate from cold or from W6's in-memory recursion |
| Spin-orbital→spatial projection | `project_rccsd_amplitudes_to_spatial`, `amplitudes.{h,cpp}` | Lets the hand-written `run_rccsd` (rank 2) participate — see "The hand-written solvers" below |
| Write from `run_rccsd` | `ccsd.cpp` | Same gating/degradation policy as the generated write site |

**Why a separate sidecar, not a bumped `.hfchk` version**, unchanged from the
original design: amplitudes are `O(o^n v^n)`, orders of magnitude larger than
the SCF matrices, and appending them to `.hfchk` would bloat every run that
never touches CC. They are also method/rank-specific and reference-specific
metadata that belongs next to the amplitudes, not in the SCF header. Keeping
them out of `.hfchk` means this feature cannot regress SCF restart —
`ponytail:` one file, one concern. Sidecar path is always derived:
`<stem>.ccamp` next to the SCF checkpoint.

**Degradation policy, load-bearing throughout**: a restart is an
optimization, never a correctness gate. A missing, stale, corrupt, or
mismatched sidecar degrades to a cold start (or W6's in-memory recursion)
with a logged warning — it never fails the run. Every check added to this
area since, including the two described next, follows this same rule.

## Two defects, and one capability gap, found and fixed after the spine landed

Documented in full, with the exact failure mode and how each was verified,
in `CC_AMPLITUDE_CHECKPOINT_REMAINING_SCOPE.md` (C0/C1) and
`CC_AMPLITUDE_CHECKPOINT_UCC_SCOPE.md` (U0/U1) — summarized here only so
this file's own history is honest about what "landed" originally meant
versus what it means now:

- **The original format silently dropped higher Sz sectors** for rank-≥4
  amplitude sets (the independent sector blocks a CCSDTQ+ solve carries
  beyond the balanced `by_rank` reference sector). Fixed by bumping the
  format to version 2 and fixing both the write/read codec and the seed
  hook's application logic — dropping the sector was a bug on both sides
  independently, and fixing only one would have left the other silently
  inert.
- **The sidecar was never validated against the live run it was about to
  seed** — only a per-rank *shape* check existed, not a *meaning* check, so a
  same-shaped sidecar from a different basis or molecule could silently seed
  a wrong basin. Fixed by comparing `basis_name` and `n_occ`/`n_virt` at the
  read site before seeding.
- **The format could not represent a sectors-only amplitude set at all** —
  not a defect in existing data, but a capability gap that blocked a UCC
  sidecar from being written under any metadata scheme, since the reader's
  `by_rank` loop trip count was coupled to `max_rank` with no independent
  count. Fixed by bumping to version 3 and adding `n_by_rank`, alongside
  UHF's four occupation counts the same bump needed anyway. A version-check
  off-by-one (rejecting every version strictly between 1 and current) was
  found and fixed in the same pass — harmless until the bump made it real,
  caught by a version-2 compatibility test, not by inspection.

## The hand-written solvers

The generated arbitrary-order path and the sidecar/seed-hook machinery are
both **spatial RCC** amplitudes; the hand-written CC solvers converge in
**spin-orbital** form (`ccsd.cpp`: `so.n_occ = 2·reference.n_occ`). A byte
copy between the two representations is silently wrong — this is the
"layout, not plumbing" barrier this doc originally scoped as X5, and it
required deriving the actual closed-shell spin-integration relation, not
just moving bytes around.

**`run_rccsd` (rank 2) is done.** `project_rccsd_amplitudes_to_spatial`
converts its converged spin-orbital `t1`/`t2` to spatial form. The relation
— `t1` is either spin channel (they're identical at closed shell, cross-spin
blocks are exactly zero) and `t2` is the opposite-spin spin-orbital block
(the same-spin blocks are the *dependent* combination
`t2_aa = t2_ab - t2_ab.swap(a,b)`, not independent information) — was derived
and then **numerically verified against this codebase's own converged
BH3/STO-3G amplitudes**, not trusted from a textbook citation alone. `run_rccsd`
was worth doing now because it has no generated-arbitrary-order sibling at all
(the generated registry's floor is rank 3), so this genuinely fills a gap
rather than duplicating an existing route.

**`run_rccsdt` (rank 3, hand-written) remains open, and is deliberately
deferred.** Unlike `run_rccsd`, it has a generated sibling that already
reaches rank 3 through the arbitrary runtime (`ccsdt_gen`, landed as X4.0/X4.1
above) and already writes a spatial sidecar — so the gap `run_rccsdt`'s own
sidecar participation would close is already closed by a different route. The
project's own stated intent (generated kernels are meant to replace the
hand-written solvers in production, long-term) means investing further in the
hand-written rank-3 path's persistence is investing in a path being retired.
Full detail — including the extra complication that `run_rccsdt` selects
among three backends at runtime and only one holds dense amplitudes — is in
the REMAINING doc's C3 section.

## What's still open

Tracked in `CC_AMPLITUDE_CHECKPOINT_REMAINING_SCOPE.md`:

- **C2 is done.** Verified end-to-end on Be/STO-3G: a rank-3 `ccsdt_gen`
  sidecar correctly seeds a `cc4` restart (1 iteration vs 6 cold, same
  basin), closing X4 formally.
- **C3** — `run_rccsdt`'s hand-written participation, deferred as above.
- **C4's format blocker is fixed.** `CC_AMPLITUDE_CHECKPOINT_UCC_SCOPE.md`'s
  U0/U1 landed: the sidecar is now version 3, carrying an `n_by_rank` field
  independent of `max_rank` (so a sectors-only amplitude set — UCC's shape —
  is representable and no longer rejected outright) and UHF's four
  occupation counts, always-present and defaulting to 0 for RHF. What
  remains is U2/U3 — a UCC write site and restart site in `run_uccgen`
  itself, which still has zero references to the checkpoint machinery.
  Mechanical now that the format exists, mirroring C0/C1's own
  write-site/read-site pattern for a second caller.

## What this reuses

- `seed_arbitrary_order_amplitudes` — the one and only seed injection point,
  shared with W6's in-memory warm start.
- `solve_generated_rcc` (`rccgen.cpp`) — the CC entry point; the restart
  read/write sit at the same two points W6 already used.
- The W6/X4.0 shared unblock: emitting a rank-3 arbitrary-order kernel and
  lowering the registry floor, which benefited both features from one piece
  of work.

See `ccgen_kernel_wiring_and_benchmark_scope.md` (W6, the in-memory warm
start this persists) and `src/io/checkpoint.h` (the SCF checkpoint format
this sidecar deliberately sits beside rather than inside).
