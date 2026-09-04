# `.ccamp` support for UCC (C4)

Scope for in-flight work. Not started. Follow-on to
`CC_AMPLITUDE_CHECKPOINT_REMAINING_SCOPE.md`'s C4, which found the real
blocker (not the premise that section originally stated) but did not build
anything, deliberately, per the earlier decision to document rather than
build a version-3 header change without its own pass.

**Revised once already, mid-attempt at U0, before this scope had shipped
anything.** The original version planned U0 (accept an empty `by_rank`) and
U1 (add UHF's four counts) as two separately-landable steps. Starting U0
found — by building it and testing a real round-trip, not by re-reading the
plan — that it does not work in isolation: the reader's `by_rank` loop trip
count was still silently coupled to `max_rank`, a coupling that held for
every caller before this scope and breaks the moment it doesn't. U0 and U1
are now one merged step; the "Steps" section below reflects that, and the
finding itself is kept inline rather than edited away, since it is the
concrete demonstration of the doc's own closing rule: verify the reader,
not just the writer, every time either side's assumptions change.

Everything below was re-verified against the current tree while writing
this scope, not carried over from memory — several of the claims here were
confirmed by actually running code, not by reading it alone.

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

### U0/U1 merged — the version-3 header: `by_rank` count + four UHF counts, both additive (~S)

**U0 and U1 were originally scoped as separate, independently-landable
steps. They are not.** Found by actually building U0 in isolation and
testing round-trip on a real sectors-only amplitude set, not by inspection:
a writer-only fix (accept an empty `by_rank`, switch `max_rank`'s source to
`meta.max_rank`) compiles clean and *appears* to work (the write call
succeeds), but the **reader** still derives its `by_rank` read-loop trip
count from `max_rank` (`cc_amplitude_checkpoint.cpp:239-240` as of the
version-2 format: `for (int rank = 1; rank <= max_rank; ++rank)`), with no
independent count of how many `by_rank` tensors were actually written. For
every caller before this scope, `max_rank == by_rank.size()` held by
construction, so this coupling was invisible. The moment `max_rank` can
legitimately exceed `by_rank.size()` (a UCC file: `max_rank = 2`,
`by_rank.size() = 0`), the reader tries to read tensors that were never
written and corrupts on the very next field it touches — reproduced with a
save-then-load probe: `save_cc_amplitudes` succeeds, `load_cc_amplitudes`
fails with `"rank-1 count 7016996763659665412 disagrees with dims product
2"` (garbage read from bytes that actually belong to `n_sectors`).

**So the actual fix needs a new field, not a new interpretation of an old
one**: an explicit `by_rank` count, independent of `max_rank`. Since that is
a byte-layout change requiring a version bump, and U1's own UHF-count
addition already needed the identical bump, **the two are merged into one
version-3 step** rather than landed as two separate bumps in a row — a
second bump immediately after the first would mean the version-2→3
compatibility work gets partially superseded before anyone reads a
version-3 file in practice.

**What version 3 adds, both unconditionally present (matching the
already-decided "additive, always-present, no per-`reference_type`
encoding" design above):**

- `[4] n_by_rank i32` — the actual number of `by_rank` tensors that follow,
  written right after `max_rank` (which stays as the informational
  "highest excitation rank represented" field it always was — `read_tensor`'s
  bounds-check argument still wants it, and `chk.meta.max_rank` is still a
  meaningful field to round-trip). For every existing RCC caller
  `n_by_rank == max_rank` by construction, so this is bitwise-inert for them
  in the same sense the original U0 write reasoned about `meta.max_rank` —
  now verify it for the actual new field, not the old proxy.
- The four UHF counts (`n_occ_alpha`, `n_occ_beta`, `n_virt_alpha`,
  `n_virt_beta`), unconditionally present as in the original U1 plan,
  unchanged.

The `save_cc_amplitudes` emptiness check moves to
`by_rank.empty() && sectors.empty()` as originally planned in U0 — that part
of the original U0 reasoning was correct, only the "switch `max_rank`'s
source and stop there" part was incomplete.

Loader must accept versions 1, 2, and 3, extending the existing
`if (version >= 2)` tiering with one more branch — version 1 has no
`reference_type` and no sector block (already handled); version 2 has
`reference_type` and sectors but no `n_by_rank`/UHF counts (**for a
version-2 file, `n_by_rank` must default to `max_rank`** — this is the one
place version 2's old coupling has to be reconstructed explicitly, since
every version-2 file on disk was written under the old assumption); version
3 has everything explicit. This is the same "read what's there, default the
rest" contract C0 established for the version-1→2 jump, and the version-2→3
default is not a guess — it is recovering exactly the invariant every
version-2 writer actually upheld.

*Verify, in order:*

1. **Byte-for-byte inertness for every existing RCC caller.** Diff a real
   RCC `.ccamp` (from `rccgen.cpp`'s or `ccsd.cpp`'s write sites) before and
   after this change — the emptiness-check and `max_rank`-source changes
   from the original U0 plan, plus this step's new `n_by_rank` field, must
   not perturb a single byte an RCC caller produces beyond the version
   number and the new field itself. This is the check the original U0 scope
   called for; it still applies, now against the merged step.
2. **The exact failure this step exists to fix, round-tripped for real.**
   Extend `tests/cc_amplitude_checkpoint.cpp` with a case constructing an
   `ArbitraryOrderRCCAmplitudes` with **empty `by_rank` and one populated
   `sectors` entry** (not a synthetic non-empty `by_rank` — the actual UCC
   shape), `meta.max_rank` set explicitly, save, load, and assert the
   round-tripped `by_rank` is empty, `sectors` has the one entry with
   bytewise-equal `dims`/`data`, and `meta.max_rank` matches. This is the
   test that would have caught U0's own incompleteness had it existed
   first — write it before trusting the fix this time.
3. **UHF counts round-trip** (the original U1 gate, unchanged): a
   version-3 file with non-zero UHF counts round-trips bytewise-equal
   metadata.
4. **Version-2 compatibility, both fields.** A hand-built version-2 file
   (no `n_by_rank`, no UHF counts in the byte stream — construct the same
   way the existing version-1-compat test hand-builds a version-1 file)
   loads with `n_by_rank` defaulted to that file's own `max_rank` (not to
   0 — 0 would silently discard every version-2 RCC file's `by_rank` data)
   and the four UHF counts defaulted to 0.
5. **Version-1 compatibility still holds**, unmodified, proving three-tier
   compatibility rather than just the newest tier in isolation.

**Stop condition, restated for the merged step:** if any existing RCC call
site's `by_rank.size()` does NOT already equal what becomes `n_by_rank`
(i.e., step 1's byte-diff finds a real behavior change), stop and re-scope
— exactly the original U0 stop condition, now checked against the field
that actually drives the read loop instead of the field that only looked
like it did.

**Do not skip the version-2 compatibility branch, and do not default
`n_by_rank` to anything but that file's own `max_rank` for a version-2
read.** The version-1→2 jump already proved the general trap is real (a
version-2 file with zero sectors and a version-1 file both end the stream
at the same logical point, and getting that distinction wrong was the
actual defect C0 fixed). The version-2→3 jump's specific edge is sharper:
defaulting `n_by_rank` to 0 instead of `max_rank` would not just misread a
file, it would silently discard every existing version-2 sidecar's
`by_rank` amplitudes on the next load — a correctness regression on
already-shipped data, not merely a new format's own bug.

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

U0/U1 (merged) is the one step with a real inertness risk — a byte-layout
and write-time behavior change for every existing RCC caller, not just
UCC — and must be verified byte-for-byte before anything downstream is
trusted; its own history (planned as two separately-landable steps, found
by testing to be one) is the reason to trust its "merge, don't split
further" conclusion rather than re-attempt the split. U2 and U3 are
mechanical once U0/U1 lands — they are the same write-site/read-site
pattern C0/C1/C2 already established for RCC, applied to a second caller,
and need no real UCC run to verify until U2 itself. Do not start U2 before
U0/U1's own gates (including the version-2 default-`n_by_rank` case, which
is the sharpest of the bunch) are green.

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
- **Do not build U2/U3 before U0/U1's own gates are green.** A byte-layout
  change that turns out not to be inert would silently corrupt every
  existing RCC caller's checkpoints, not just fail to help UCC.
- **Do not trust a writer-only fix without testing the full round-trip
  against the reader.** This is not a hypothetical — the original two-step
  U0/U1 plan looked correct on inspection (the writer compiled, the write
  call succeeded) and was only found broken by actually saving and loading
  a UCC-shaped file. `save_cc_amplitudes` succeeding is not evidence that
  `load_cc_amplitudes` can read what it wrote; the two functions were
  written together but must be *verified* together, every time either one's
  assumptions about the byte layout change.

## Key locations

| what | where |
|---|---|
| Blocker 1 (empty check) | `save_cc_amplitudes`, `cc_amplitude_checkpoint.cpp:132-134` |
| Blocker 1's reader-side twin (the `by_rank` read-loop trip count, coupled to `max_rank` with no independent `n_by_rank`) | `load_cc_amplitudes`, `cc_amplitude_checkpoint.cpp:239-240` |
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
