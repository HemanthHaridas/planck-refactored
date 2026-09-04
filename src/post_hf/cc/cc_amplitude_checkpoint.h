#ifndef HF_POSTHF_CC_AMPLITUDE_CHECKPOINT_H
#define HF_POSTHF_CC_AMPLITUDE_CHECKPOINT_H

#include <cstdint>
#include <expected>
#include <string>

#include "post_hf/cc/amplitudes.h"

// ─── Generated-CC amplitude sidecar (.ccamp) ─────────────────────────────────
//
// Persists the converged amplitudes of a generated arbitrary-order RCC solve so
// a later run can seed its iteration from them (via
// seed_arbitrary_order_amplitudes) instead of cold-starting. This is a SEPARATE
// file from the SCF checkpoint (.hfchk): amplitudes are O(o^n v^n) and
// method/rank-specific, and keeping them out of .hfchk means CC persistence can
// never regress SCF restart.
//
// Binary layout, little-endian:
//   [8]      magic  "PLNKCCA\0"
//   [4]      version u32 = 2
//   [4]      max_rank i32
//   [4+len]  method tag string (u32 length + chars, e.g. "cc4")
//   [4+len]  basis name string
//   [8]      n_occ  u64
//   [8]      n_virt u64
//   [1]      reference_type u8 (0 = RHF, 1 = UHF; see ReferenceType below --
//            C4: folded in with the version-2 bump so an eventual UCC sidecar
//            needs no version 3, "one spare byte in the header beats a second
//            version bump")
//   For rank r in 1..=max_rank (amplitudes.by_rank[r-1]):
//     [4]        order i32  (= 2r, the TensorND order)
//     [order×4]  dims  i32[]
//     [8]        count u64  (product of dims == TensorND.data.size())
//     [count×8]  data  f64[] (TensorND.data, native storage order)
//   [4]      n_sectors i32  (C0: the higher independent Sz sectors carried by
//            amplitudes.sectors -- e.g. (4, "aaabaaab") for CCSDTQ. These are
//            NOT a signed-permutation combination of by_rank and are real,
//            independent converged data; the version-1 format silently
//            dropped them on write, so a restart from a version-1-era sidecar
//            was always partially seeded for rank >= 4)
//   For each sector:
//     [4+len]    excitation_rank i32, tag string (the (rank, tag) key)
//     [tensor]   same order/dims/count/data body as a by_rank entry
//
// VERSION 1 COMPATIBILITY: a version-1 file has no reference_type byte and no
// trailing sector block. `load_cc_amplitudes` accepts version 1, defaults
// `reference_type` to RHF (the only kind version 1 ever wrote), and treats a
// stream that ends immediately after `by_rank` as "zero sectors" rather than
// an error -- version 1 is a strict prefix of version 2's layout by
// construction, which is the whole point of reading it that way instead of
// branching on version number field-by-field.

namespace HartreeFock::Correlation::CC
{
    // C4: which HF reference the amplitudes were converged against. Only RHF
    // is ever written today (no UCC write path exists yet); the field exists
    // so a future UHF/UCC sidecar cannot be silently seeded into an RCC run
    // or vice versa without the read site having to invent a new format
    // version to carry the distinction.
    enum class CCReferenceType : std::uint8_t
    {
        RHF = 0,
        UHF = 1,
    };

    struct CCAmplitudeCheckpointMeta
    {
        int max_rank = 0;
        std::string method;     // e.g. "cc4"
        std::string basis_name; // basis the amplitudes were solved in
        std::uint64_t n_occ = 0;
        std::uint64_t n_virt = 0;
        CCReferenceType reference_type = CCReferenceType::RHF;
    };

    struct CCAmplitudeCheckpoint
    {
        CCAmplitudeCheckpointMeta meta;
        ArbitraryOrderRCCAmplitudes amplitudes;
    };

    // Write amplitudes + metadata to `path`, including any higher Sz sectors
    // in `amplitudes.sectors`. Overwrites any existing file.
    std::expected<void, std::string> save_cc_amplitudes(
        const std::string &path,
        const ArbitraryOrderRCCAmplitudes &amplitudes,
        const CCAmplitudeCheckpointMeta &meta);

    // Read a sidecar back, including any sector block (empty for a version-1
    // file or a version-2 file with no sectors -- both are valid and are not
    // errors). Errors (never crashes) on a missing file, bad magic/version, or
    // truncation. Does NOT validate against a live reference — the caller (or
    // seed_arbitrary_order_amplitudes / sector_tensor) does the dim check.
    std::expected<CCAmplitudeCheckpoint, std::string> load_cc_amplitudes(
        const std::string &path);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_AMPLITUDE_CHECKPOINT_H
