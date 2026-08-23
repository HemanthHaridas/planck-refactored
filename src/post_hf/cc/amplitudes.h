#ifndef HF_POSTHF_CC_AMPLITUDES_H
#define HF_POSTHF_CC_AMPLITUDES_H

#include <expected>
#include <string>
#include <utility>
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

        // U2.2: per-block denominators, keyed (rank, tag) exactly like
        // ArbitraryOrderRCCAmplitudes::sectors. Populated only on the UCC path,
        // where alpha and beta orbital energies differ and a block therefore
        // CANNOT reuse its rank's reference denominator.
        //
        // The RHF path leaves this empty and keeps reading `by_rank`, so its
        // behavior is byte-identical. That fallback is deliberate and not merely
        // a convenience: for an RHF reference the orbital energies really are
        // spin-free, so every Sz sector of a rank shares one denominator -- which
        // is exactly why the B4 sector update could reuse `tensor(rank)` and why
        // that reuse is wrong the moment the reference is unrestricted.
        std::vector<std::pair<std::pair<int, std::string>, TensorND>> sectors;

        [[nodiscard]] int max_rank() const noexcept;
        [[nodiscard]] bool has_rank(int excitation_rank) const noexcept;
        [[nodiscard]] std::expected<DenseTensorView, std::string> tensor(int excitation_rank);
        [[nodiscard]] std::expected<ConstDenseTensorView, std::string> tensor(int excitation_rank) const;

        // The denominator a (rank, tag) block must be divided by. Returns the
        // block's own entry when one is stored (UCC), otherwise falls back to the
        // rank's reference denominator (RHF, where they coincide). Callers use
        // this rather than `tensor(rank)` so the RHF and UCC paths stay ONE code
        // path -- a second sector-update loop would be a second thing to keep in
        // sync with ensure_amplitude_sectors.
        [[nodiscard]] std::expected<ConstDenseTensorView, std::string>
        sector_tensor(int excitation_rank, const std::string &tag) const;
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

    // U2: the UCC (spin-blocked) denominator for one amplitude block.
    //
    // `block_tag` is the per-slot spin string in the SAME order as the tensor's
    // indices, which is occ-first then vir (`rank_dims`): a rank-2n block tagged
    // "aab" + "aab" has occ slots (a,a,b) and vir slots (a,a,b). This mirrors
    // ccgen's UCC tags, whose halves are bra(vir)-then-ket(occ) -- so the caller
    // converts, and the tag handed here is always occ-half-first.
    //
    // Each slot draws its orbital energy from its own spin's set, so a mixed
    // block's denominator is genuinely spin-resolved rather than a relabeling.
    // U5.1b: the UCC amplitude blocks of one excitation rank, as occ-half-first
    // spin tags. A rank-n block has n+1 independent alpha-count sectors and no
    // a<->b fold is available (alpha and beta are different orbitals), so rank 1
    // gives {aa, bb} and rank 2 gives {aaaa, abab, bbbb}.
    //
    // The C++ mirror of ccgen's `ucc_independent_blocks`, and derivable from the
    // rank alone -- which is what lets `prepare_generated_ucc_state` build the
    // denominators BEFORE the kernel bundle is known. That ordering is forced:
    // `ensure_amplitude_sectors` sizes each amplitude block from its own
    // denominator (U2.2), so the denominators must already be there.
    [[nodiscard]] std::vector<std::string> ucc_amplitude_blocks(int excitation_rank);

    // U2.2: the whole UCC denominator cache, one entry per (rank, tag) block the
    // kernel bundle declares. Drives build_ucc_block_denominator over `blocks`.
    //
    // `blocks` is taken from the bundle's `sector_tags` rather than enumerated
    // here, so the cache carries exactly the blocks the generated kernels will
    // ask for -- one vocabulary, defined in ccgen, not two that can drift.
    //
    // TAG ORDER, and it is the thing a caller gets wrong: the tag handed to this
    // function is per-slot spin in the TENSOR's index order, occ-half first
    // (`rank_dims`). ccgen's UCC tags are bra(vir)-half-then-ket(occ), so a
    // caller passing ccgen tags straight through gets a silently transposed
    // denominator whenever the two halves differ (`abba` vs `baab`). Convert at
    // the boundary; see build_ucc_block_denominator's own note.
    //
    // `by_rank` is left EMPTY: a UCC method has no privileged reference sector
    // (every target is block-tagged), so there is no rank whose denominator is
    // the unqualified one. Callers must reach blocks through `sector_tensor`.
    std::expected<ArbitraryOrderDenominatorCache, std::string>
    build_ucc_denominator_cache(
        const UHFReference &reference,
        const std::vector<std::pair<int, std::string>> &blocks);

    std::expected<TensorND, std::string> build_ucc_block_denominator(
        const UHFReference &reference,
        const std::string &block_tag);

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
