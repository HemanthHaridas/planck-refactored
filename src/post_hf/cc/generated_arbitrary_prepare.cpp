#include "post_hf/cc/generated_arbitrary_runtime.h"

#include <exception>
#include <format>

namespace HartreeFock::Correlation::CC
{
    namespace
    {
        // The generated spin-adapted kernels index every ERI block as PHYSICIST
        // <pq|rs> (ccgen derives in v=<pq||rs>), but build_tensor_cc_block_cache
        // returns CHEMISTS (pq|rs) -- the convention the hand-written tensor
        // backend consumes. The exact bridge is <pq|rs> = (pr|qs): a physicist
        // block is the chemists tensor with its middle two axes swapped,
        //   phys_block(p,q,r,s) = chem_source(p,r,q,s).
        // Swapping o/v labels means each PHYSICIST block's data lives in a
        // DIFFERENT chemists block: physicist oovv <ij|ab> = chemists (ia|jb) =
        // the chemists OVOV block (and vice versa); oooo/ooov/ovvv/vvvv are
        // self-sourced. So this is not an in-place transpose per name -- the
        // oovv<->ovov sources cross. (The shared chemists cache is left untouched
        // so the hand-written RCCSDT[TENSOR] path is unaffected.) All seven blocks
        // are rebound -- ovvo IS referenced once the emit uses only the valid +1
        // symmetries (the -1 antisym perms that used to route around ovvo are gone).
        Tensor4D swap_mid_axes(const Tensor4D &c)
        {
            // out(p,q,r,s) = c(p,r,q,s); out dims = (d1, d3, d2, d4).
            Tensor4D out(c.dim1, c.dim3, c.dim2, c.dim4, 0.0);
            for (int p = 0; p < c.dim1; ++p)
                for (int q = 0; q < c.dim3; ++q)
                    for (int r = 0; r < c.dim2; ++r)
                        for (int s = 0; s < c.dim4; ++s)
                            out(p, q, r, s) = c(p, r, q, s);
            return out;
        }

    } // namespace (anonymous)

    // Declared in generated_arbitrary_runtime.h: every generated-kernel consumer needs this,
    // not just the arbitrary-order path (V1.3/T1b).
    TensorCCBlockCache rebind_physicist(TensorCCBlockCache chem)
    {
            TensorCCBlockCache phys;
            phys.oooo = swap_mid_axes(chem.oooo); // <ij|kl> = (ik|jl)
            phys.ooov = swap_mid_axes(chem.ooov); // <ij|ka> = (ik|ja)
            phys.oovv = swap_mid_axes(chem.ovov); // <ij|ab> = (ia|jb)  <- OVOV
            phys.ovov = swap_mid_axes(chem.oovv); // <ia|jb> = (ij|ab)  <- OOVV
            phys.ovvo = swap_mid_axes(chem.ovvo); // <ia|bj> = (ib|aj)
            phys.ovvv = swap_mid_axes(chem.ovvv); // <ia|bc> = (ib|ac)
            phys.vvvv = swap_mid_axes(chem.vvvv); // <ab|cd> = (ac|bd)
            phys.memory_report = std::move(chem.memory_report);
        phys.total_bytes = chem.total_bytes;
        return phys;
    }

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
                .mo_blocks = rebind_physicist(std::move(*blocks_res)),
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

    std::expected<ArbitraryOrderTensorCCState, std::string>
    prepare_generated_ucc_state(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        int max_excitation_rank,
        const std::string &tag)
    {
        if (max_excitation_rank < 1)
        {
            return std::unexpected(
                "prepare_generated_ucc_state: max_excitation_rank must be at least 1.");
        }

        auto uhf_res = build_uhf_reference(calculator);
        if (!uhf_res)
            return std::unexpected(uhf_res.error());
        const UHFReference &uhf = *uhf_res;

        // The MO-basis spin Fock matrices. Both AO Fock matrices are persisted by
        // the SCF (`_info._scf.{alpha,beta}.fock`), so this is a transform, not a
        // rebuild. Each spin uses ITS OWN coefficients: for a UHF reference the
        // two MO bases differ, which is the whole reason the blocks are split.
        const Eigen::MatrixXd fock_alpha_mo =
            uhf.C_alpha.transpose() * calculator._info._scf.alpha.fock * uhf.C_alpha;
        const Eigen::MatrixXd fock_beta_mo =
            uhf.C_beta.transpose() * calculator._info._scf.beta.fock * uhf.C_beta;

        auto ref_res = build_ucc_fock_blocks(uhf, fock_alpha_mo, fock_beta_mo);
        if (!ref_res)
            return std::unexpected(ref_res.error());

        auto blocks_res = build_ucc_spin_block_cache(
            calculator, shell_pairs, uhf, ucc_canonical_blocks(), tag);
        if (!blocks_res)
            return std::unexpected(blocks_res.error());

        // Denominators for every (rank, tag) up to max_excitation_rank. Built
        // HERE and not later because ensure_amplitude_sectors sizes each amplitude
        // block from its own denominator (U2.2); with these missing it would find
        // no reference rank to fall back to either and skip the block silently.
        std::vector<std::pair<int, std::string>> denominator_blocks;
        for (int rank = 1; rank <= max_excitation_rank; ++rank)
            for (const auto &block_tag : ucc_amplitude_blocks(rank))
                denominator_blocks.push_back({rank, block_tag});

        auto denom_res = build_ucc_denominator_cache(uhf, denominator_blocks);
        if (!denom_res)
            return std::unexpected(denom_res.error());

        try
        {
            ArbitraryOrderTensorCCState state{
                .reference = std::move(*ref_res),
                // Rebound to the physicist <pq|rs| the generated kernels index.
                // Every block is self-sourced: swap_mid_axes on the block stored
                // under its own key. See rebind_physicist_ucc.
                .mo_blocks = rebind_physicist_ucc(std::move(*blocks_res)),
                .denominators = std::move(*denom_res),
                // No amplitudes at all: `by_rank` stays empty (no privileged
                // reference sector) and the sectors are filled by
                // ensure_amplitude_sectors once the bundle is known.
                .amplitudes = ArbitraryOrderRCCAmplitudes{},
                .max_excitation_rank = max_excitation_rank,
            };
            return state;
        }
        catch (const std::exception &ex)
        {
            return std::unexpected(
                "prepare_generated_ucc_state: " + std::string(ex.what()));
        }
    }

    std::expected<void, std::string>
    seed_arbitrary_order_amplitudes(
        ArbitraryOrderTensorCCState &state,
        const ArbitraryOrderRCCAmplitudes &seed)
    {
        if (seed.by_rank.size() > state.amplitudes.by_rank.size())
            return std::unexpected(std::format(
                "seed_arbitrary_order_amplitudes: seed has {} ranks but state holds {}.",
                seed.by_rank.size(), state.amplitudes.by_rank.size()));

        for (std::size_t r = 0; r < seed.by_rank.size(); ++r)
        {
            if (seed.by_rank[r].dims != state.amplitudes.by_rank[r].dims)
                return std::unexpected(std::format(
                    "seed_arbitrary_order_amplitudes: rank-{} dim mismatch.", r + 1));
            state.amplitudes.by_rank[r] = seed.by_rank[r];
        }
        return {};
    }
} // namespace HartreeFock::Correlation::CC
