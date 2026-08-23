#ifndef HF_POSTHF_CC_GENERATED_ARBITRARY_RUNTIME_H
#define HF_POSTHF_CC_GENERATED_ARBITRARY_RUNTIME_H

#include <expected>
#include <functional>
#include <string>
#include <vector>

#include "post_hf/cc/solver_arbitrary.h"
#include "post_hf/cc/tensor_backend.h"

namespace HartreeFock::Correlation::CC
{
    struct ArbitraryOrderTensorCCState
    {
        CanonicalRHFCCReference reference;
        TensorCCBlockCache mo_blocks;
        ArbitraryOrderDenominatorCache denominators;
        ArbitraryOrderRCCAmplitudes amplitudes;
        int max_excitation_rank = 0;
    };

    struct GeneratedArbitraryOrderKernels
    {
        using EnergyKernel = std::function<double(
            const CanonicalRHFCCReference &,
            const TensorCCBlockCache &,
            const ArbitraryOrderDenominatorCache &,
            const ArbitraryOrderRCCAmplitudes &)>;
        using ResidualKernel = std::function<TensorND(
            const CanonicalRHFCCReference &,
            const TensorCCBlockCache &,
            const ArbitraryOrderDenominatorCache &,
            const ArbitraryOrderRCCAmplitudes &)>;

        // R3.1.3d / Gap B3: a residual kernel for a higher independent Sz sector
        // of a rank-2n amplitude (n >= 4). `residuals_by_rank` holds the reference
        // sector per rank; `sector_residuals` holds the extra sectors, each tagged
        // (excitation_rank, tag) so B4 updates the matching amplitude block
        // (`ArbitraryOrderRCCAmplitudes::sector_tensor`). Empty for <= CCSDT.
        struct SectorResidual
        {
            int excitation_rank = 0;
            std::string tag;
            ResidualKernel kernel;
        };

        int max_excitation_rank = 0;
        EnergyKernel energy;

        // rank r at [r-1] -- the per-rank REFERENCE residual (RCC / CCSDTQ).
        //
        // U4.0: may be EMPTY, which declares an ALL-SECTORS bundle: every residual
        // this method carries is block-tagged and lives in `sector_residuals`. That
        // is the UCC case, where there is no privileged reference sector to occupy
        // this slot -- `ucc_adapt_equations` tags every target (`doubles_aaaa`,
        // never a bare `doubles`), so the emitter pushes nothing here.
        //
        // Promoting one block per rank into this vector was considered and does
        // NOT work: the slot is sized by `rank_dims`, which yields one shape per
        // rank, while UCC blocks of a single rank have DIFFERENT shapes under UHF
        // (`aaaa` is (noa,noa,nva,nva), `abab` is (noa,nob,nva,nvb)). Promoting
        // one would silently mis-size the others.
        //
        // A partially-filled vector is rejected: either every rank has a reference
        // residual or none does. Half-filled means a bundle lost a kernel, which
        // would otherwise evaluate as a silent zero contribution.
        std::vector<ResidualKernel> residuals_by_rank;

        // True when this bundle carries no per-rank reference residuals and drives
        // every excitation through `sector_residuals` instead.
        [[nodiscard]] bool is_all_sectors() const noexcept
        {
            return residuals_by_rank.empty();
        }

        // The independent Sz sectors this method carries, (excitation_rank, tag).
        // Feeds make_zero_rcc_amplitudes so the state allocates the sector blocks
        // (Gap B1) that `sector_residuals` evaluate and B4 updates.
        std::vector<std::pair<int, std::string>> sector_tags;
        std::vector<SectorResidual> sector_residuals;
    };

    struct GeneratedArbitraryOrderSolveResult
    {
        ArbitraryOrderTensorCCState state;
        double correlation_energy = 0.0;
        double energy_change = 0.0;
        unsigned int iterations = 0;
        bool converged = false;
        ArbitraryOrderIterationMetrics metrics;
    };

    [[nodiscard]] TensorND to_tensor_nd(const Tensor2D &tensor);
    [[nodiscard]] TensorND to_tensor_nd(const Tensor4D &tensor);
    [[nodiscard]] TensorND to_tensor_nd(const Tensor6D &tensor);
    [[nodiscard]] TensorND to_tensor_nd(const TensorND &tensor);

    // Rebind a chemists' (pq|rs) block cache to the physicist <pq|rs> convention that
    // ccgen-generated kernels index. EVERY consumer of a generated kernel needs this: the
    // arbitrary-order path has always called it, and the plain rank-3 path did not, which
    // is why `compute_ccsdt_triples_residual` read the wrong integrals the first time it
    // was ever executed. Exposed rather than duplicated so there is one definition of the
    // convention -- the oovv<->ovov sources cross, so a re-derivation is easy to get wrong.
    //
    // Returns a NEW cache; the caller's chemists' cache is left untouched, because the
    // hand-written RCCSDT[TENSOR] path reads it and expects chemists' order.
    [[nodiscard]] TensorCCBlockCache rebind_physicist(TensorCCBlockCache chem);

    std::expected<ArbitraryOrderTensorCCState, std::string>
    prepare_generated_arbitrary_order_state(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        int max_excitation_rank,
        const std::string &tag = "CC[GENERATED] :");

    // U5.1b: prepare an ALL-SECTORS (UCC) state from an unrestricted reference.
    //
    // A SIBLING of prepare_generated_arbitrary_order_state, not a flag on it:
    // every one of its four steps differs (UHF reference, spin-blocked ERIs,
    // per-block denominators, sector-only amplitudes), so sharing one function
    // would mean a branch at every line.
    //
    // NO BLOCK VOCABULARY IS PASSED, matching how RCC does it:
    // build_tensor_cc_block_cache takes no block list either -- its set IS the
    // struct's seven named members, built unconditionally, and it over-builds
    // (measured: ccsd and ccsdt read 6 of the 7, `ovvo` never touched). Nothing
    // is negotiated with the emitter, so nothing can drift. The UCC ERI set comes
    // from ucc_canonical_blocks() (24 arrays, U5.1a) and the amplitude/denominator
    // tags from ucc_amplitude_blocks(rank) -- both derived, both gated against
    // ccgen.
    //
    // U5.2c: the ERI cache is REBOUND to the physicist <pq|rs| the generated
    // kernels index, so the state is consumable by a solve. (It was deliberately
    // left in chemists order between U5.1b and U5.2b, because the RCC
    // `rebind_physicist` cannot be reused here -- it builds a fresh cache from the
    // seven named members and never copies `spin_blocks`, so it would silently
    // discard all 24 UCC blocks.)
    //
    // The returned state has `by_rank` EMPTY on amplitudes and denominators (a
    // UCC method has no privileged reference sector) and NO amplitude sectors:
    // like the RCC path, prepare runs before the kernel bundle is known, so
    // `ensure_amplitude_sectors` fills those afterwards. It can, because the
    // denominators ARE already populated here and it sizes each amplitude block
    // from its own denominator (U2.2) -- that ordering is forced, not incidental.
    std::expected<ArbitraryOrderTensorCCState, std::string>
    prepare_generated_ucc_state(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        int max_excitation_rank,
        const std::string &tag = "UCC[GENERATED] :");

    // Warm-start seed (W6.0): overwrite the lowest `seed.by_rank.size()` ranks of
    // the state's zero amplitudes with converged lower-rank amplitudes. Ranks not
    // covered by the seed stay zero. Each seeded rank's dims must match the state's
    // slot for that rank (occ/virt shape); a mismatch is an error, not silent.
    std::expected<void, std::string>
    seed_arbitrary_order_amplitudes(
        ArbitraryOrderTensorCCState &state,
        const ArbitraryOrderRCCAmplitudes &seed);

    // Gap B4: allocate the higher Sz sector amplitude blocks the kernel bundle
    // declares (`kernels.sector_tags`) onto an already-prepared state. `prepare`
    // runs before the bundle is known, so the state starts with no sectors; this
    // reconciles them (zero-init) so evaluate/update can drive each sector. A
    // no-op when the bundle has no sectors (<= CCSDT).
    void ensure_amplitude_sectors(
        ArbitraryOrderTensorCCState &state,
        const GeneratedArbitraryOrderKernels &kernels);

    std::expected<ArbitraryOrderResiduals, std::string>
    evaluate_generated_arbitrary_order_residuals(
        const ArbitraryOrderTensorCCState &state,
        const GeneratedArbitraryOrderKernels &kernels);

    std::expected<GeneratedArbitraryOrderSolveResult, std::string>
    run_generated_arbitrary_order_iterations(
        ArbitraryOrderTensorCCState state,
        const GeneratedArbitraryOrderKernels &kernels,
        unsigned int max_iterations,
        double tol_energy,
        double tol_residual,
        double damping,
        bool use_diis,
        int diis_dim,
        const std::string &log_tag = "CC[GENERATED] :");
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_GENERATED_ARBITRARY_RUNTIME_H
