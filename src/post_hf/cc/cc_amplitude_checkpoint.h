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
//   [4]      version u32 = 3
//   [4]      max_rank i32       -- highest excitation rank represented
//                                  (informational + read_tensor's bounds
//                                  check; NOT the by_rank trip count, see
//                                  n_by_rank below)
//   [4+len]  method tag string (u32 length + chars, e.g. "cc4")
//   [4+len]  basis name string
//   [8]      n_occ  u64          -- RHF: the occ/virt pair; UHF: unused,
//   [8]      n_virt u64             always written as 0 (see the four
//                                    UHF fields below)
//   [1]      reference_type u8 (0 = RHF, 1 = UHF; see CCReferenceType below)
//   [4]      n_by_rank i32       -- U0/U1: the ACTUAL number of by_rank
//            tensors that follow, independent of max_rank. RCC files always
//            have n_by_rank == max_rank; a UCC file legitimately has
//            n_by_rank == 0 (UCC carries no privileged reference sector --
//            see prepare_generated_ucc_state -- all its data lives in the
//            sector block below). Splitting this from max_rank is what
//            makes a sectors-only amplitude set representable at all: a
//            reader that inferred the by_rank count from max_rank could not
//            tell "2 tensors written" from "0 tensors written, rank still
//            meaningfully 2" -- confirmed by an actual save/load probe
//            during scoping, not assumed.
//   [8]      n_occ_alpha  u64 )  -- U0/U1: UCC's four independent
//   [8]      n_occ_beta   u64 )     occupation counts. Always present,
//   [8]      n_virt_alpha u64 )     always written; an RHF file writes all
//   [8]      n_virt_beta  u64 )     four as 0 and nothing reads them for
//                                    reference_type == RHF, mirroring
//                                    CanonicalRHFCCReference's own precedent
//                                    for the identical problem
//                                    (tensor_backend.h).
//   For i in 1..=n_by_rank (amplitudes.by_rank[i-1]):
//     [4]        order i32  (= 2r, the TensorND order)
//     [order×4]  dims  i32[]
//     [8]        count u64  (product of dims == TensorND.data.size())
//     [count×8]  data  f64[] (TensorND.data, native storage order)
//   [4]      n_sectors i32  (C0: the higher independent Sz sectors carried by
//            amplitudes.sectors -- e.g. (4, "aaabaaab") for CCSDTQ, or, for
//            UCC, every spin block via ucc_amplitude_blocks. These are NOT a
//            signed-permutation combination of by_rank and are real,
//            independent converged data; the version-1 format silently
//            dropped them on write, so a restart from a version-1-era sidecar
//            was always partially seeded for rank >= 4)
//   For each sector:
//     [4+len]    excitation_rank i32, tag string (the (rank, tag) key)
//     [tensor]   same order/dims/count/data body as a by_rank entry
//
// VERSION 1 COMPATIBILITY: a version-1 file has no reference_type byte, no
// n_by_rank, no UHF counts, and no trailing sector block -- its by_rank
// count IS max_rank (the only interpretation version 1 ever had). Treats a
// stream that ends immediately after by_rank as "zero sectors" rather than
// an error.
//
// VERSION 2 COMPATIBILITY: a version-2 file has reference_type and sectors
// but no n_by_rank and no UHF counts. `n_by_rank` for such a file MUST
// default to that file's own `max_rank`, not to 0 -- every version-2 writer
// upheld n_by_rank == max_rank implicitly (it is the invariant this format
// revision makes explicit), so defaulting to 0 would silently discard every
// existing version-2 RCC sidecar's by_rank data on the next load. The four
// UHF counts default to 0 (every version-2 file is RHF; version 2 predates
// any UCC write path).

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
        // Highest excitation rank represented. Since U0/U1, this is NOT the
        // by_rank tensor count (that is amplitudes.by_rank.size() itself,
        // written to the file as n_by_rank) -- max_rank stays meaningful
        // even for a UCC file whose by_rank is empty.
        int max_rank = 0;
        std::string method;     // e.g. "cc4"
        std::string basis_name; // basis the amplitudes were solved in
        std::uint64_t n_occ = 0;
        std::uint64_t n_virt = 0;
        CCReferenceType reference_type = CCReferenceType::RHF;

        // U0/U1: UCC's four independent occupation counts. Left at 0 (and
        // ignored on write/read) for reference_type == RHF -- an RHF caller
        // sets only n_occ/n_virt above, exactly as before this field
        // existed. Mirrors CanonicalRHFCCReference's own precedent for the
        // identical restricted-vs-unrestricted split.
        std::uint64_t n_occ_alpha = 0;
        std::uint64_t n_occ_beta = 0;
        std::uint64_t n_virt_alpha = 0;
        std::uint64_t n_virt_beta = 0;
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
