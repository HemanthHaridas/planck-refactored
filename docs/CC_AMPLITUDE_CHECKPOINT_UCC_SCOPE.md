# `.ccamp` support for UCC (C4)

Scope for in-flight work. Not started. Follow-on to
`CC_AMPLITUDE_CHECKPOINT_REMAINING_SCOPE.md`'s C4, which found the real
blocker (not the premise that section originally stated) but did not build
anything, deliberately, per the earlier decision to document rather than
build a version-3 header change without its own pass.

Everything below was re-verified against the current tree while writing this
scope, not carried over from memory — one of the two blockers this doc
describes was confirmed by actually running code, not by reading it.

---

## The two independent blockers, confirmed in-tree

**Blocker 1 — `save_cc_amplitudes` rejects a sectors-only amplitude set
outright, before any format question even applies.** Verified by compiling
and running a standalone probe against the real function: an
`ArbitraryOrderRCCAmplitudes` with an empty `by_rank` and one populated
`sectors` entry — exactly what `prepare_generated_ucc_state`'s own comment
says a UCC state looks like ("No amplitudes at all: `by_rank` stays empty...
the sectors are filled by `ensure_amplitude_sectors`") — is rejected with
`"save_cc_amplitudes: amplitudes are empty."` The check
(`cc_amplitude_checkpoint.cpp:132-134`) derives `max_rank` from
`amplitudes.by_rank.size()`, not from the caller-supplied
`meta.max_rank` field, which is silently ignored on write regardless. **This
blocks a UCC write with ANY metadata format**, independent of Blocker 2.

**Blocker 2 — even past Blocker 1, the metadata cannot describe a UHF
partition.** `CCAmplitudeCheckpointMeta` carries one `n_occ`/`n_virt` pair.
`UHFReference` (and `CanonicalRHFCCReference`, the state struct RCC and UCC
both actually build against) carry four independent counts:
`n_occ_alpha`, `n_occ_beta`, `n_virt_alpha`, `n_virt_beta`. This codebase has
already solved the identically-shaped problem once, at the state-struct
level: `CanonicalRHFCCReference` added the four counts **alongside**
`orbital_partition` rather than repurposing it, specifically so RCC's
existing reads (`orbital_partition.n_occ`/`.n_virt`, measured at 6 call
sites in the rank-3 TU) stay byte-identical and untouched
(`tensor_backend.h:42-59`). The checkpoint format should follow the exact
same additive pattern, not invent a new one.

**No other blocker was found.** UCC's sector tags (`ucc_amplitude_blocks`,
strings like `"aaaa"`/`"abab"`/`"bbbb"`) already fit C0's `(rank, tag)` →
`TensorND` keying with no change. The read-side validation shape C1 built
(compare metadata against the live reference, warn and cold-start on
mismatch, never fail the run) generalizes directly — a UCC restart site
needs the same policy, applied to different fields.

---

## Design decision, made once, up front

**One format, discriminated by `reference_type`, not two file formats.**
`CCReferenceType` (`RHF = 0, UHF = 1`) already exists in the version-2
header. The version-3 extension adds the four UHF counts as **new,
unconditionally-present fields** (not a variant/union encoded differently
per reference type) — an RHF file writes them as zero and a reader never
looks at them for `reference_type == RHF`, mirroring exactly how
`CanonicalRHFCCReference` leaves the four counts at 0 for an RCC reference
and nothing reads them (`tensor_backend.h:52-55`, "on an RCC reference the
four stay 0 and nothing reads them"). Rejected alternative: encoding
UHF-only fields conditionally on `reference_type` inside the byte stream
(fewer bytes for RHF files) — adds a branch to every reader for a few bytes
of saving on a payload that is otherwise `O(o^n v^n)` doubles; not worth the
format complexity `ponytail:` four extra `u64`s is noise next to the tensor
data that follows them.

**`by_rank` stays `std::vector<TensorND>`, possibly empty.** No new type, no
`std::optional`. A UCC write passes an amplitude set whose `by_rank` is
empty and whose `sectors` carries the real data — exactly the shape
`prepare_generated_ucc_state` already produces. The fix is to the *check*
that currently rejects this shape, not to the data model.

---

## Steps

### U0 — fix Blocker 1: let `save_cc_amplitudes` accept a sectors-only amplitude set (~S)

Change the emptiness check from `by_rank.size() < 1` to
`by_rank.size() + sectors.size() < 1` (or equivalently, reject only when
*both* are empty) — the file is meaningless with literally nothing in it,
but a `by_rank`-empty/`sectors`-populated set is real data, not an
accidental empty call.

`max_rank`'s write-time source needs a decision at the same time, since it
currently comes from `by_rank.size()` which will now legitimately be 0 for a
UCC file: switch it to `meta.max_rank` (the field that is already threaded
through every call site today, just silently ignored on write) rather than
inferring it from either container. This also fixes the pre-existing dead
field — `meta.max_rank` stops being ignored on write, for every caller, not
just UCC's.

*Verify:* extend `tests/cc_amplitude_checkpoint.cpp` with a case
constructing an `ArbitraryOrderRCCAmplitudes` with empty `by_rank` and one
`sectors` entry, `meta.max_rank` set explicitly (not derivable from
`by_rank`), and assert `save_cc_amplitudes` succeeds and the round-tripped
`chk.meta.max_rank` matches what was supplied. Re-run the existing
RCC-shaped round-trip case unchanged — `meta.max_rank` there already equals
`by_rank.size()` by construction in every existing caller, so switching the
source must be bitwise-inert for every RCC write site
(`rccgen.cpp`'s two write sites, `ccsd.cpp`'s X5.1 write site) — verify by
diffing a real RCC `.ccamp` byte-for-byte before/after this change, the same
way X5.1's own verification worked, not just by re-running the unit test.

**Stop condition:** if any existing RCC call site's `meta.max_rank` does NOT
already equal its `by_rank.size()` (i.e., the switch is not actually inert),
stop and re-scope — that would mean some caller relies on the old
derived-from-`by_rank` behavior, and this step needs a compatibility shim
before proceeding, not a silent behavior change.

### U1 — the version-3 header: four UHF counts, additive (~S given U0)

Bump `CCAMP_VERSION` to 3. Append four `u64` fields
(`n_occ_alpha`, `n_occ_beta`, `n_virt_alpha`, `n_virt_beta`) to
`CCAmplitudeCheckpointMeta`, written unconditionally right after the
existing `reference_type` byte (matching the "additive, always-present"
decision above). `CCAmplitudeCheckpointMeta` itself gains the four fields as
plain `std::uint64_t = 0` members, same style as the existing `n_occ`/
`n_virt`.

Loader must accept versions 1, 2, and 3 — version 1 has no `reference_type`
byte and no sector block (already handled); version 2 has `reference_type`
but no UHF counts (new: must default the four counts to 0, matching what an
RHF-only version-2 file always implicitly meant); version 3 has everything.
This is the same "read what's there, default the rest" contract U0's
predecessor (C0) already established for the version-1→2 jump — extend the
existing `if (version >= 2)` branching pattern with one more tier rather
than inventing a new one.

*Verify:* extend `tests/cc_amplitude_checkpoint.cpp` with (a) a round-trip of
a version-3 file with non-zero UHF counts, asserting bytewise-equal
metadata; (b) a hand-built version-2 file (no UHF counts in the byte
stream — construct it the same way the existing version-1-compat test
hand-builds a version-1 file) loads with the four counts defaulted to 0 and
`reference_type` whatever the file specified; (c) the existing version-1
compat test still passes unmodified — proving three-tier compatibility, not
just the new tier in isolation.

**Do not skip the version-2 compatibility branch.** The version-1→2 jump
already proved this trap is real (a version-2 file with zero sectors and a
version-1 file both end the stream at the same logical point, and getting
that distinction wrong was the actual defect C0 fixed, not a hypothetical
risk) — the version-2→3 jump has an analogous edge (a version-2 file
correctly has no UHF-count bytes at all, which must read as "0, valid"
rather than "truncated, error").

### U2 — a UCC write site (~S given U0/U1)

Add the write call inside `run_uccgen` (`uccgen.cpp`), mirroring
`run_rccgen`'s write site in `rccgen.cpp` exactly: same
`_save_checkpoint && !_checkpoint_path.empty()` gate, same
`<stem>.ccamp` path derivation, same warn-not-fail policy on write error.
`meta.reference_type = CCReferenceType::UHF` and the four counts come from
`state.reference` (`CanonicalRHFCCReference.n_occ_alpha` etc. — already
populated for a UCC state per `build_ucc_fock_blocks`, no new plumbing
needed to obtain them). `meta.max_rank = rank` (the function parameter,
already available, same as every existing write site already does — this is
what U0 makes safe to do even when `by_rank` is empty).

This directly contradicts `uccgen.cpp`'s own current comment
("`.ccamp` persistence" deliberately omitted) — updating or removing that
comment is part of this step, not an afterthought, since a stale comment
claiming a deliberate omission that no longer holds is worse than no
comment.

*Verify:* a real end-to-end run — `correlation ucc2` (or `ucc3`/`ucc4`,
whichever is cheapest to converge) with `save_checkpoint .true.` on a small
open-shell system (the existing `b_ucc{2,3,4}_sto3g` regression inputs are
the obvious fixture, already gated and already exercising an open-shell
reference) produces a `.ccamp` file; load it back with `load_cc_amplitudes`
directly (a small standalone probe, the same pattern used to verify X5.1)
and assert `reference_type == UHF`, the four counts match the run's actual
occupation, and the sector data is present and non-zero.

### U3 — a UCC restart site (~S given U2)

Add a `try_restart_from_sidecar`-equivalent inside `run_uccgen`, following
C1's validation shape exactly: compare `basis_name` and, since this is UHF,
all four occupation counts (not the two-field RCC comparison) against
`state.reference`; **additionally** reject a `reference_type` mismatch
before even checking counts (an RHF sidecar has zero meaning for a UHF
run and vice versa — checking counts first on a wrong-reference-type file
risks a coincidental match the way C1's own motivating case was a
same-shape-different-basis coincidence). Same degradation policy throughout:
warn, cold-start, never fail the run.

Seeding through `seed_arbitrary_order_amplitudes` needs no change — it
already operates purely on `state.amplitudes.by_rank`/`.sectors`, generic to
either reference kind (confirmed: nothing in that function reads
`state.reference` at all). The restart site's job is purely the
load-and-validate step before calling it, matching C1's own scope exactly.

*Verify:* two-run test on the `b_ucc{2,3,4}_sto3g` fixture — write, then
restart, assert fewer iterations and the same converged energy (mirroring
X0-X3's own original validation shape). Then the negative cases, mirroring
C1's own verification exactly: a hand-corrupted `basis_name` → cold-start
with a warning; a hand-corrupted occupation count (any of the four) →
cold-start with a warning; a **hand-flipped `reference_type`** (an RHF file
fed to a UCC restart, or vice versa) → cold-start with a warning naming the
reference-type mismatch specifically, not a downstream shape error.

**Stop condition:** if seeding via `seed_arbitrary_order_amplitudes` does
*not* work unchanged for a UCC state (i.e., it turns out to implicitly
assume something RCC-specific this scope's grep did not catch), stop and
re-scope that function's contract before adding U3's call site — do not
patch around a wrong assumption in the shared seed hook to make one caller
work.

---

## Sequencing and risk

U0 is the only step with a real inertness risk (a write-time behavior change
for every existing RCC caller, not just UCC) and must be verified byte-for-
byte before anything downstream is trusted. U1 is pure format-and-loader
work, independently testable with hand-built fixtures, no real UCC run
required. U2 and U3 are mechanical once U0/U1 land — they are the same
write-site/read-site pattern C0/C1/C2 already established for RCC, applied
to a second caller. Do U0 and U1 fully (including their own gates) before
starting U2; do not interleave, for the same reason the earlier C0 work
kept its reordering-sensitive and reordering-safe pieces in separate,
individually-verified commits.

## What NOT to do

- **Do not encode UHF fields conditionally in the byte stream based on
  `reference_type`.** Always present, always read, defaulting to 0 for RHF
  — one format, one code path, matching `CanonicalRHFCCReference`'s own
  precedent exactly.
- **Do not add a second file format, a second magic number, or a
  UCC-specific loader function.** `load_cc_amplitudes` stays the single
  entry point; version 3 is a superset read, not a fork.
- **Do not skip the version-2 compatibility branch in U1.** Argued above;
  the version-1→2 jump already proved this exact trap is real, not
  hypothetical.
- **Do not change `seed_arbitrary_order_amplitudes` to accommodate U3**
  unless U3's own verification actually finds it necessary — grep confirms
  it is already reference-kind-agnostic; adding UCC-specific branches to a
  function that does not need them is exactly the kind of speculative
  generalization this codebase avoids elsewhere.
- **Do not fail a run on a missing, stale, or corrupt UCC sidecar.**
  Restart is an optimization for UCC exactly as it is for RCC — U3 must
  degrade the same way U0-U3 of the original scope and C0-C2 all did.
- **Do not build U2/U3 before U0/U1 are individually verified.** U0's own
  stop condition exists because a write-time behavior change that turns out
  not to be inert would silently corrupt every existing RCC caller's
  checkpoints, not just fail to help UCC.

## Key locations

| what | where |
|---|---|
| Blocker 1 (empty check) | `save_cc_amplitudes`, `cc_amplitude_checkpoint.cpp:132-134` |
| Blocker 2 (metadata shape) | `CCAmplitudeCheckpointMeta`, `cc_amplitude_checkpoint.h:67-75` |
| The precedent for the additive-fields pattern | `CanonicalRHFCCReference`, `tensor_backend.h:35-83`, and its own comment explaining exactly why it's additive |
| Where UCC's real occupation counts already live | `UHFReference`, `common.h`; threaded into `CanonicalRHFCCReference` via `build_ucc_fock_blocks`, `ucc_blocks.cpp:194-` |
| UCC's amplitude state (confirms `by_rank` is empty) | `prepare_generated_ucc_state`, `generated_arbitrary_prepare.cpp:106-176` |
| UCC's sector tags (confirmed already compatible) | `ucc_amplitude_blocks`, `amplitudes.cpp:308-331` |
| The entry point needing U2/U3's call sites | `run_uccgen`, `uccgen.cpp` |
| The RCC pattern to mirror for U2 | `run_rccgen`'s write site, `rccgen.cpp:415-445` |
| The RCC pattern to mirror for U3 | `try_restart_from_sidecar`, `rccgen.cpp:203-250` |
| The seed hook (unchanged, verified reference-kind-agnostic) | `seed_arbitrary_order_amplitudes`, `generated_arbitrary_prepare.cpp` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
