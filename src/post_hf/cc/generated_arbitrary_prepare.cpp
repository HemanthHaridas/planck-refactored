#include "post_hf/cc/generated_arbitrary_runtime.h"

#include <exception>

namespace HartreeFock::Correlation::CC
{
    std::expected<ArbitraryOrderTensorCCState, std::string>
    prepare_generated_arbitrary_order_state(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        int max_excitation_rank,
        const std::string &tag)
    {
        if (max_excitation_rank < 1)
        {
            return std::unexpected(
                "prepare_generated_arbitrary_order_state: max_excitation_rank must be at least 1.");
        }

        auto ref_res = build_canonical_rhf_cc_reference(calculator);
        if (!ref_res)
            return std::unexpected(ref_res.error());

        auto blocks_res = build_tensor_cc_block_cache(
            calculator, shell_pairs, *ref_res, tag);
        if (!blocks_res)
            return std::unexpected(blocks_res.error());

        auto denom_res = build_arbitrary_order_denominator_cache(
            ref_res->orbital_partition,
            max_excitation_rank);
        if (!denom_res)
            return std::unexpected(denom_res.error());

        try
        {
            const RHFReference partition = ref_res->orbital_partition;
            ArbitraryOrderTensorCCState state{
                .reference = std::move(*ref_res),
                .mo_blocks = std::move(*blocks_res),
                .denominators = std::move(*denom_res),
                .amplitudes = make_zero_rcc_amplitudes(
                    partition,
                    max_excitation_rank),
                .max_excitation_rank = max_excitation_rank,
            };
            return state;
        }
        catch (const std::exception &ex)
        {
            return std::unexpected(
                "prepare_generated_arbitrary_order_state: " + std::string(ex.what()));
        }
    }
} // namespace HartreeFock::Correlation::CC
