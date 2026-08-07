#ifndef HF_POSTHF_CC_AMPLITUDES_H
#define HF_POSTHF_CC_AMPLITUDES_H

#include <expected>
#include <string>
#include <vector>

#include "post_hf/cc/common.h"

namespace HartreeFock::Correlation::CC
{
    // Canonical energy denominators are reused by multiple update schemes.
    // The tensors follow the same occupied/virtual index order as the
    // amplitudes so the update loops read almost like the algebra.
    struct DenominatorCache
    {
        Tensor2D d1; // eps_i - eps_a
        Tensor4D d2; // eps_i + eps_j - eps_a - eps_b
        Tensor6D d3; // eps_i + eps_j + eps_k - eps_a - eps_b - eps_c

        [[nodiscard]] int max_rank() const noexcept;
        [[nodiscard]] bool has_rank(int excitation_rank) const noexcept;
        [[nodiscard]] std::expected<DenseTensorView, std::string> tensor(int excitation_rank);
        [[nodiscard]] std::expected<ConstDenseTensorView, std::string> tensor(int excitation_rank) const;
    };

    struct RCCSDAmplitudes
    {
        Tensor2D t1; // t_i^a stored as (i,a)
        Tensor4D t2; // t_ij^ab stored as (i,j,a,b)

        [[nodiscard]] int max_rank() const noexcept;
        [[nodiscard]] bool has_rank(int excitation_rank) const noexcept;
        [[nodiscard]] std::expected<DenseTensorView, std::string> tensor(int excitation_rank);
        [[nodiscard]] std::expected<ConstDenseTensorView, std::string> tensor(int excitation_rank) const;
    };

    struct RCCSDTAmplitudes
    {
        Tensor2D t1; // t_i^a stored as (i,a)
        Tensor4D t2; // t_ij^ab stored as (i,j,a,b)
        Tensor6D t3; // t_ijk^abc stored as (i,j,k,a,b,c)

        [[nodiscard]] int max_rank() const noexcept;
        [[nodiscard]] bool has_rank(int excitation_rank) const noexcept;
        [[nodiscard]] std::expected<DenseTensorView, std::string> tensor(int excitation_rank);
        [[nodiscard]] std::expected<ConstDenseTensorView, std::string> tensor(int excitation_rank) const;
    };

    struct ArbitraryOrderDenominatorCache
    {
        std::vector<TensorND> by_rank; // rank r stored at by_rank[r-1]

        [[nodiscard]] int max_rank() const noexcept;
        [[nodiscard]] bool has_rank(int excitation_rank) const noexcept;
        [[nodiscard]] std::expected<DenseTensorView, std::string> tensor(int excitation_rank);
        [[nodiscard]] std::expected<ConstDenseTensorView, std::string> tensor(int excitation_rank) const;
    };

    struct ArbitraryOrderRCCAmplitudes
    {
        std::vector<TensorND> by_rank; // rank r stored at by_rank[r-1]

        // R3.1.3d: higher independent Sz sectors of a rank-2n amplitude
        // (n >= 4). `by_rank` holds the reference (balanced) sector; a rank-2n
        // amplitude also has floor(n/2) sectors, keyed here by (rank, tag) where
        // tag is the alpha-before-beta block string, e.g. {4, "aaabaaab"}. The
        // spin-adapted generated kernels read a sector via `sector_tensor`. Empty
        // for methods <= CCSDT, so the default arbitrary-order path is unaffected.
        std::vector<std::pair<std::pair<int, std::string>, TensorND>> sectors;

        [[nodiscard]] int max_rank() const noexcept;
        [[nodiscard]] bool has_rank(int excitation_rank) const noexcept;
        [[nodiscard]] std::expected<DenseTensorView, std::string> tensor(int excitation_rank);
        [[nodiscard]] std::expected<ConstDenseTensorView, std::string> tensor(int excitation_rank) const;

        // A higher Sz sector's dense view (R3.1.3d). Errors if the sector was not
        // allocated (the multi-sector solver populates it; a CCSDT/reference-only
        // run never asks). The generated kernels bind this once per kernel.
        [[nodiscard]] std::expected<DenseTensorView, std::string>
        sector_tensor(int excitation_rank, const std::string &tag);
        [[nodiscard]] std::expected<ConstDenseTensorView, std::string>
        sector_tensor(int excitation_rank, const std::string &tag) const;
    };

    // `include_triples=false` is useful for the current teaching code paths that
    // need only singles/doubles denominators and want to avoid an unnecessary
    // O(o^3 v^3) allocation.
    std::expected<DenominatorCache, std::string> build_denominator_cache(
        const RHFReference &reference,
        bool include_triples = true);

    std::expected<ArbitraryOrderDenominatorCache, std::string> build_arbitrary_order_denominator_cache(
        const RHFReference &reference,
        int max_excitation_rank);

    RCCSDAmplitudes make_zero_rccsd_amplitudes(const RHFReference &reference);

    // The dense T3 container is kept for the future tensor-based CCSDT path. The
    // current determinant-space prototype does not allocate it eagerly inside the
    // top-level solver because that would dominate memory before any iterations.
    RCCSDTAmplitudes make_zero_rccsdt_amplitudes(const RHFReference &reference);

    ArbitraryOrderRCCAmplitudes make_zero_rcc_amplitudes(
        const RHFReference &reference,
        int max_excitation_rank);

    // R3.1.3d / Gap B1: zero-init amplitudes carrying, in addition to the
    // per-rank reference blocks (`by_rank`), the higher independent Sz sectors
    // listed in `sectors` as (excitation_rank, tag) -- e.g. {{4, "aaabaaab"}}
    // for CCSDTQ. Each sector block has the same occ/vir dims as its rank's
    // reference block (only the spin projection it represents differs), so it is
    // `rank_dims(rank)`-shaped and zero-initialized. The multi-sector solver
    // reads/updates each via `ArbitraryOrderRCCAmplitudes::sector_tensor`. The
    // no-sector overload above delegates here with an empty list (unchanged for
    // <= CCSDT). The sector list is supplied by the generated kernel bundle
    // (Gap B3); this allocator does not re-derive spin algebra.
    ArbitraryOrderRCCAmplitudes make_zero_rcc_amplitudes(
        const RHFReference &reference,
        int max_excitation_rank,
        const std::vector<std::pair<int, std::string>> &sectors);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_AMPLITUDES_H
