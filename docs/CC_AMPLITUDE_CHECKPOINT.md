# CC Amplitude Checkpoint (`.ccamp`) Architecture Note

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**How does a CC run persist and restart its amplitudes across processes, and what invariants should future changes to the format or its call sites preserve?**

## Short answer

A converged CC run can persist its amplitudes to a `<stem>.ccamp` sidecar and
a later run can seed its iteration from them instead of cold-starting — for
the generated arbitrary-order path (any rank, including cross-rank:
`ccsdt_gen` seeding a `cc4` run, and UCC via `run_uccgen`) and for the
hand-written rank-2 `run_rccsd` solver. It is the persistence sibling of W6,
which warm-starts amplitudes in memory within one run; both share the same
seed injection point, `seed_arbitrary_order_amplitudes`.

The format went through three versions because two real defects and one
capability gap were found after the spine shipped — a dropped-sector bug, a
missing meaning-check on restart, and an inability to represent UCC's
sectors-only amplitude shape at all. All three are fixed. The one item
investigated and deliberately declined, not merely deferred, is the
hand-written rank-3 `run_rccsdt` solver's own sidecar participation.

## Where the logic lives

- `src/post_hf/cc/cc_amplitude_checkpoint.{h,cpp}` — format, writer, loader
- `src/post_hf/cc/rccgen.cpp` — RCC write site (`run_rccgen`) and restart site (`try_restart_from_sidecar`)
- `src/post_hf/cc/uccgen.cpp` — UCC write site (`run_uccgen`) and restart site (`try_restart_ucc_from_sidecar`)
- `src/post_hf/cc/ccsd.cpp` — hand-written `run_rccsd` write site
- `src/post_hf/cc/amplitudes.{h,cpp}` — `project_rccsd_amplitudes_to_spatial`
- `src/post_hf/cc/generated_arbitrary_prepare.cpp` — `seed_arbitrary_order_amplitudes`
- `tests/cc_amplitude_checkpoint.cpp`, `tests/cc_spatial_amplitude_projection.cpp`

## What invariants matter

### 1. A restart is an optimization, never a correctness gate

A missing, stale, corrupt, or mismatched sidecar must degrade to a cold
start (or W6's in-memory recursion) with a logged warning — it must never
fail the run. Every restart path (RCC, UCC) validates the sidecar's
basis/occupation/reference-kind against the live run *before* seeding,
because two different bases (or molecules, or reference kinds) can share
identical occupation counts, and a same-shaped stale sidecar would otherwise
silently seed a wrong basin.

Design rule:

- Any new check added to a restart path must degrade the same way: warn,
  cold-start, continue. Never propagate the failure upward.

### 2. The writer and the reader must be verified together, not independently

A change to the byte layout is only safe if both `save_cc_amplitudes` and
`load_cc_amplitudes` agree on it. A writer-only fix compiled clean and its
write call succeeded while the reader silently read garbage past the point
where the old coupling (`by_rank` trip count inferred from `max_rank`)
broke — corrupting on the very next field, several bytes past the actual
defect.

Design rule:

- Never trust "the write call succeeded" as evidence the file can be read
  back. Every format change needs an actual save-then-load round-trip test
  on the exact shape the change is meant to support, not just a shape that
  happens to already work.

### 3. Version compatibility must be tiered explicitly, not inferred

Two coupled hazards live here permanently: (a) distinguishing "the stream
ended here because there is nothing more" from "the stream ended here
because it was truncated," and (b) choosing the correct default for a field
that didn't exist in an older version. Getting (b) wrong for `n_by_rank`
would silently discard every existing version-2 sidecar's `by_rank` data on
next load, not just fail to read a new one.

Design rule:

- Any future format bump must default a missing field to the value every
  prior writer actually upheld implicitly (here: `n_by_rank` defaults to
  that file's own `max_rank`, never to 0), and must extend the version
  range check inclusively rather than special-casing only the newest
  version.

### 4. UHF's fields are additive, never conditionally encoded

The four UCC occupation counts (`n_occ_alpha`, `n_occ_beta`, `n_virt_alpha`,
`n_virt_beta`) are always present in the byte stream and always read,
defaulting to 0 and unread for `reference_type == RHF`. This mirrors an
existing precedent in this codebase (`CanonicalRHFCCReference`, which added
the same four counts alongside `orbital_partition` rather than repurposing
it) specifically so RCC's existing reads stay byte-identical.

Design rule:

- Do not encode a reference-kind-specific field conditionally in the byte
  stream. One format, one code path; `load_cc_amplitudes` stays the single
  entry point rather than forking into a UCC-specific loader.

### 5. Spin-orbital and spatial amplitude layouts are not interchangeable by copy

The generated arbitrary-order path and the sidecar/seed-hook machinery both
use **spatial RCC** amplitudes; the hand-written CC solvers converge in
**spin-orbital** form. A byte copy between the two is silently wrong
(roughly 2x too large in each dimension, and mixes same-spin/opposite-spin
contributions that must first be combined) — this is a layout barrier, not
a plumbing one.

Design rule:

- Any new hand-written solver that wants to participate in the sidecar must
  derive its own closed-shell spin-integration relation and verify it
  numerically against a real converged run of that codebase's own solver,
  not assume the rank-2 relation generalizes by pattern-matching.

## What was fixed

The repaired implementation enforces these concrete choices:

1. `seed_arbitrary_order_amplitudes` applies sectors by `(rank, tag)` key,
   not just `by_rank` — the original format silently dropped every
   independent higher-Sz sector on write, and a `cc4` restart always seeded
   that sector at zero (silently, since Jacobi/DIIS still converges from
   zero, so no energy-only gate ever caught it).
2. `try_restart_from_sidecar` compares `basis_name` and `n_occ`/`n_virt`
   against the live reference before seeding, degrading to a warned
   cold-start on mismatch, instead of relying on a per-rank shape check
   alone.
3. `save_cc_amplitudes`'s emptiness check is `by_rank.empty() &&
   sectors.empty()`, not derived from `by_rank.size()` alone — a
   sectors-only amplitude set (UCC's shape) is no longer rejected outright.
4. The format carries an explicit `n_by_rank` field, independent of
   `max_rank`, so the reader's `by_rank` read-loop trip count cannot exceed
   what was actually written.
5. The version check accepts the full `1..CCAMP_VERSION` range rather than
   only exact matches on 1 or the current version — the prior check
   silently rejected every version strictly between 1 and current the
   moment a third version was introduced.
6. `project_rccsd_amplitudes_to_spatial` converts `run_rccsd`'s converged
   spin-orbital `t1`/`t2` to the spatial layout using
   `t1_spatial(i,a) = t1_so(2i, 2a)` and
   `t2_spatial(i,j,a,b) = t2_so(2i, 2j+1, 2a, 2b+1)` — verified numerically
   against this codebase's own converged BH3/STO-3G amplitudes, not assumed
   from a textbook citation alone.
7. `run_uccgen` gained a write site (`meta.reference_type = UHF`, the four
   counts from `state.reference`, `by_rank` left empty) and a restart site
   (`try_restart_ucc_from_sidecar`) that rejects a `reference_type`
   mismatch *before* checking occupation counts, wired in after
   `ensure_amplitude_sectors` so the live sectors are already allocated
   when seeding runs.

## Validation strategy that should remain in place

- Byte-for-byte format round-trip tests (`tests/cc_amplitude_checkpoint.cpp`),
  mutation-verified: reverting the emptiness-check fix, the version-range
  fix, or the `n_by_rank` version-2 default each independently makes the
  test built to catch it fail.
- Three-tier version compatibility (1, 2, 3) exercised in one test binary,
  including a hand-built version-2 file with no `n_by_rank`/UHF counts in
  the byte stream.
- Cross-rank restart verified end-to-end on Be/STO-3G: a `cc4` restart from
  a `ccsdt_gen` rank-3 sidecar converges in 1 iteration versus 6 cold, both
  landing in the same basin as the tight-tolerance reference.
- UCC write + restart verified end-to-end on B/STO-3G (doublet): a `ucc2`
  restart converges in 1 iteration versus 12 cold, identical energy; three
  negative cases (corrupted basis, corrupted occupation count, flipped
  reference type) each cold-start with a warning naming the specific
  mismatch rather than failing the run.
- `run_rccsd` end-to-end on BH3/STO-3G: same energy, sidecar values read
  back bit-for-bit correct.

## Related but separate outcome: `run_rccsdt` (rank 3, hand-written)

`run_rccsdt` was checked for the same kind of sidecar participation because
it superficially resembles `run_rccsd`. It was investigated and
**deliberately declined**, not deferred by inertia: it selects among three
backends at runtime, and only the tensor-production backend ever holds a
dense `RCCSDTAmplitudes` — and even then only for systems large enough to
skip the determinant backstop (in-tree, only `ch4_rccsdt_sto3g` qualifies).
Everywhere else, convergence happens through spin-orbital determinant-space
CC, which returns only the converged energy, with no dense amplitudes
surviving to project. Even the one qualifying case's amplitudes are
spin-orbital and local to a function that never returns them to a caller.
The important takeaway is that surface similarity to `run_rccsd` is not
enough to infer the same fix applies: the backend-selection structure and
the amplitude lifetime differ in ways that change the whole shape of the
work. `ccsdt_gen` already reaches rank 3 through the generated arbitrary
runtime and already writes a spatial sidecar, which is the route this
project's stated long-term intent favors, so the value of building this
anyway is genuinely narrow.

## Remaining architecture concern

None. If `run_rccsdt`'s hand-written backends stay in production materially
longer than `ccsdt_gen`'s adoption, revisit starting from
`run_staged_tensor_triples_iterations`'s local `triples.amplitudes` — the
one place a converged dense spin-orbital T1/T2/T3 genuinely exists — rather
than attempting to reconstruct one from the determinant-space backstop. This
is a future-trigger note, not open work.
