#ifndef HF_POSTHF_CC_AMPLITUDE_CHECKPOINT_H
#define HF_POSTHF_CC_AMPLITUDE_CHECKPOINT_H

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
//   [4]      version u32 = 1
//   [4]      max_rank i32
//   [4+len]  method tag string (u32 length + chars, e.g. "cc4")
//   [4+len]  basis name string
//   [8]      n_occ  u64
//   [8]      n_virt u64
//   For rank r in 1..=max_rank (amplitudes.by_rank[r-1]):
//     [4]        order i32  (= 2r, the TensorND order)
//     [order×4]  dims  i32[]
//     [8]        count u64  (product of dims == TensorND.data.size())
//     [count×8]  data  f64[] (TensorND.data, native storage order)

namespace HartreeFock::Correlation::CC
{
    struct CCAmplitudeCheckpointMeta
    {
        int max_rank = 0;
        std::string method;     // e.g. "cc4"
        std::string basis_name; // basis the amplitudes were solved in
        std::uint64_t n_occ = 0;
        std::uint64_t n_virt = 0;
    };

    struct CCAmplitudeCheckpoint
    {
        CCAmplitudeCheckpointMeta meta;
        ArbitraryOrderRCCAmplitudes amplitudes;
    };

    // Write amplitudes + metadata to `path`. Overwrites any existing file.
    std::expected<void, std::string> save_cc_amplitudes(
        const std::string &path,
        const ArbitraryOrderRCCAmplitudes &amplitudes,
        const CCAmplitudeCheckpointMeta &meta);

    // Read a sidecar back. Errors (never crashes) on a missing file, bad
    // magic/version, or truncation. Does NOT validate against a live reference —
    // the caller (or seed_arbitrary_order_amplitudes) does the dim check.
    std::expected<CCAmplitudeCheckpoint, std::string> load_cc_amplitudes(
        const std::string &path);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_AMPLITUDE_CHECKPOINT_H
