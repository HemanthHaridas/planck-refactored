#ifndef HF_POSTHF_CC_TENSOR_BACKEND_H
#define HF_POSTHF_CC_TENSOR_BACKEND_H

#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include "integrals/shellpair.h"
#include "post_hf/cc/amplitudes.h"
#include "post_hf/cc/common.h"

namespace HartreeFock::Correlation::CC
{
    enum class RCCSDTBackend
    {
        DeterminantPrototype,
        TensorProduction,
        TensorOptimized
    };

    struct TensorMemoryBlock
    {
        std::string label;
        std::size_t elements = 0;
        std::size_t bytes = 0;
    };

    // The production tensor path should not depend directly on the more
    // pedagogical RHF reference slices. This wrapper makes the canonical orbital
    // partition explicit and also snapshots the MO-basis Fock blocks that later
    // tensor kernels will reuse heavily.
    struct CanonicalRHFCCReference
    {
        RHFReference orbital_partition;
        Tensor2D f_oo;
        Tensor2D f_ov;
        Tensor2D f_vv;
    };

    // Unlike the teaching cache in mo_blocks.*, the production cache avoids the
    // full `(pq|rs)` tensor and stores only the blocks needed by tensor-based
    // CCSDT contractions. The memory report is kept with the cache so the
    // solver can print a clear pre-flight allocation summary.
    struct TensorCCBlockCache
    {
        Tensor4D oooo;
        Tensor4D ooov;
        Tensor4D oovv;
        Tensor4D ovov;
        Tensor4D ovvo;
        Tensor4D ovvv;
        Tensor4D vvvv;

        std::vector<TensorMemoryBlock> memory_report;
        std::size_t total_bytes = 0;
    };

    struct TensorTriplesWorkspace
    {
        RCCSDTAmplitudes amplitudes;
        Tensor6D r3;
        bool allocated = false;
        std::size_t storage_bytes = 0;
    };

    struct TensorRCCSDTState
    {
        CanonicalRHFCCReference reference;
        TensorCCBlockCache mo_blocks;
        DenominatorCache denominators;
        std::size_t estimated_t3_elements = 0;
        std::size_t estimated_t3_bytes = 0;
        TensorTriplesWorkspace triples;
        double warm_start_correlation_energy = 0.0;
        unsigned int warm_start_iterations = 0;
    };

    std::expected<CanonicalRHFCCReference, std::string> build_canonical_rhf_cc_reference(
        HartreeFock::Calculator &calculator);

    std::expected<TensorCCBlockCache, std::string> build_tensor_cc_block_cache(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const CanonicalRHFCCReference &reference,
        const std::string &tag);

    [[nodiscard]] RCCSDTBackend choose_rccsdt_backend(
        const RHFReference &reference) noexcept;

    [[nodiscard]] std::string format_tensor_memory_summary(
        const TensorRCCSDTState &state);

    std::expected<void, std::string> allocate_dense_triples_workspace(
        TensorRCCSDTState &state);

    std::expected<TensorRCCSDTState, std::string> prepare_tensor_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<void, std::string> run_tensor_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<void, std::string> run_tensor_optimized_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_TENSOR_BACKEND_H
