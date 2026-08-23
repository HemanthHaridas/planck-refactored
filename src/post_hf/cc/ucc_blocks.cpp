// U3.1: the spin-blocked (UCC) MO ERI cache.
//
// Deliberately its OWN translation unit, and not part of tensor_backend_state.cpp,
// so the block/transform logic links without `ensure_eri` -- which drags in the
// Calculator, the AO integral engine, RI, basis parsing and symmetry. That whole
// chain is irrelevant to the question this code gets wrong (which coefficient
// matrix belongs in which transform slot), and pulling it into a unit test would
// have made the gate cost minutes and depend on parsing a basis set.
//
// The thin `build_ucc_spin_block_cache` wrapper that DOES acquire the ERI stays in
// tensor_backend_state.cpp alongside its RCC sibling.

#include "post_hf/cc/tensor_backend.h"
#include "post_hf/cc/tensor_backend_internal.h"
#include "post_hf/integrals.h"

#include <format>
#include <string>
#include <utility>
#include <vector>

namespace HartreeFock::Correlation::CC
{
    std::expected<const Tensor4D *, std::string> TensorCCBlockCache::spin_block(
        const std::string &space,
        const std::string &tag) const
    {
        for (const auto &entry : spin_blocks)
            if (entry.first.first == space && entry.first.second == tag)
                return &entry.second;
        // Deliberately no fallback to the untagged members: on a UCC run they are
        // empty, and on an RHF run they hold the spin-free integral, which for a
        // mixed tag is a different quantity. Erring is the whole point.
        return std::unexpected(
            "TensorCCBlockCache::spin_block: no stored block for (" + space +
            ", " + tag + ")");
    }

    std::expected<const Tensor2D *, std::string> CanonicalRHFCCReference::spin_block(
        const std::string &space,
        const std::string &tag) const
    {
        for (const auto &entry : spin_blocks)
            if (entry.first.first == space && entry.first.second == tag)
                return &entry.second;
        return std::unexpected(
            "CanonicalRHFCCReference::spin_block: no stored Fock block for (" +
            space + ", " + tag + ")");
    }

    std::expected<CanonicalRHFCCReference, std::string> build_ucc_fock_blocks(
        const UHFReference &reference,
        const Eigen::MatrixXd &fock_alpha_mo,
        const Eigen::MatrixXd &fock_beta_mo)
    {
        CanonicalRHFCCReference out;

        struct SpinSpec
        {
            const char *tag;
            int n_occ;
            int n_virt;
            const Eigen::MatrixXd *fock;
        };
        const SpinSpec spins[2] = {
            {"aa", reference.n_occ_alpha, reference.n_virt_alpha, &fock_alpha_mo},
            {"bb", reference.n_occ_beta, reference.n_virt_beta, &fock_beta_mo},
        };

        for (const auto &spin : spins)
        {
            const int no = spin.n_occ;
            const int nv = spin.n_virt;
            if (no <= 0 || nv <= 0)
                return std::unexpected(std::format(
                    "build_ucc_fock_blocks: spin '{}' has an empty space "
                    "(n_occ={} n_virt={}).", spin.tag, no, nv));
            if (spin.fock->rows() < no + nv || spin.fock->cols() < no + nv)
                return std::unexpected(std::format(
                    "build_ucc_fock_blocks: the MO Fock matrix for spin '{}' is "
                    "{}x{} but the partition needs at least {}x{}.",
                    spin.tag, spin.fock->rows(), spin.fock->cols(),
                    no + nv, no + nv));

            Tensor2D f_oo(no, no, 0.0);
            for (int i = 0; i < no; ++i)
                for (int j = 0; j < no; ++j)
                    f_oo(i, j) = (*spin.fock)(i, j);

            Tensor2D f_ov(no, nv, 0.0);
            for (int i = 0; i < no; ++i)
                for (int a = 0; a < nv; ++a)
                    f_ov(i, a) = (*spin.fock)(i, no + a);

            Tensor2D f_vv(nv, nv, 0.0);
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    f_vv(a, b) = (*spin.fock)(no + a, no + b);

            out.spin_blocks.push_back({{"oo", spin.tag}, std::move(f_oo)});
            out.spin_blocks.push_back({{"ov", spin.tag}, std::move(f_ov)});
            out.spin_blocks.push_back({{"vv", spin.tag}, std::move(f_vv)});
        }

        return out;
    }

    std::expected<TensorCCBlockCache, std::string> build_ucc_spin_block_cache_from_eri(
        const std::vector<double> &eri,
        std::size_t nb,
        const UHFReference &reference,
        const std::vector<std::pair<std::string, std::string>> &blocks)
    {
        // Slice the alpha/beta occupied and virtual columns once. UHFReference
        // stores whole C matrices (the determinant-space solvers interleave spin
        // orbitals themselves), so the occ/virt split happens here.
        const int noa = reference.n_occ_alpha;
        const int nob = reference.n_occ_beta;
        const int nva = reference.n_virt_alpha;
        const int nvb = reference.n_virt_beta;

        if (noa <= 0 || nob <= 0 || nva <= 0 || nvb <= 0)
            return std::unexpected(std::format(
                "build_ucc_spin_block_cache_from_eri: the reference has an empty space "
                "(noa={} nob={} nva={} nvb={}); UCC needs all four occupied.",
                noa, nob, nva, nvb));
        if (reference.C_alpha.cols() < noa + nva || reference.C_beta.cols() < nob + nvb)
            return std::unexpected(
                "build_ucc_spin_block_cache_from_eri: a coefficient matrix has fewer columns "
                "than its partition requires.");

        const Eigen::MatrixXd Coa = reference.C_alpha.leftCols(noa);
        const Eigen::MatrixXd Cva = reference.C_alpha.middleCols(noa, nva);
        const Eigen::MatrixXd Cob = reference.C_beta.leftCols(nob);
        const Eigen::MatrixXd Cvb = reference.C_beta.middleCols(nob, nvb);

        // Pick the coefficient matrix for one physicist slot: its space ('o'/'v')
        // and its spin ('a'/'b') are independent choices, which is exactly what
        // makes a mixed block a mixed TRANSFORM rather than a relabeled RCC one.
        const auto column_block =
            [&](char space, char spin) -> const Eigen::MatrixXd & {
            if (spin == 'a')
                return space == 'o' ? Coa : Cva;
            return space == 'o' ? Cob : Cvb;
        };
        const auto extent = [&](char space, char spin) {
            if (spin == 'a')
                return space == 'o' ? noa : nva;
            return space == 'o' ? nob : nvb;
        };

        TensorCCBlockCache cache;
        try
        {
            for (const auto &[space, spin] : blocks)
            {
                if (space.size() != 4 || spin.size() != 4)
                    return std::unexpected(std::format(
                        "build_ucc_spin_block_cache_from_eri: block ({}, {}) must name four "
                        "spaces and four spins.", space, spin));
                for (std::size_t k = 0; k < 4; ++k)
                {
                    if (space[k] != 'o' && space[k] != 'v')
                        return std::unexpected(std::format(
                            "build_ucc_spin_block_cache_from_eri: space pattern '{}' has a slot "
                            "that is neither 'o' nor 'v'.", space));
                    if (spin[k] != 'a' && spin[k] != 'b')
                        return std::unexpected(std::format(
                            "build_ucc_spin_block_cache_from_eri: spin tag '{}' has a slot that "
                            "is neither 'a' nor 'b'.", spin));
                }

                // Physicist <pq|rs> = chemists (pr|qs): the transform pairs slot 0
                // with slot 2 and slot 1 with slot 3. Getting this pairing wrong is
                // the defect class that produced a plausible-but-wrong energy
                // before, so it is written once, here.
                const Eigen::MatrixXd &c_p = column_block(space[0], spin[0]);
                const Eigen::MatrixXd &c_q = column_block(space[1], spin[1]);
                const Eigen::MatrixXd &c_r = column_block(space[2], spin[2]);
                const Eigen::MatrixXd &c_s = column_block(space[3], spin[3]);

                // Stored in CHEMISTS order (p r | q s); rebind_physicist transposes
                // the middle axes, exactly as it does for the RCC blocks.
                Tensor4D block(
                    extent(space[0], spin[0]), extent(space[2], spin[2]),
                    extent(space[1], spin[1]), extent(space[3], spin[3]),
                    HartreeFock::Correlation::transform_eri(
                        eri, nb, c_p, c_r, c_q, c_s));

                if (auto mem_res = detail::append_block_memory(
                        cache.memory_report, cache.total_bytes,
                        space + "_" + spin,
                        {extent(space[0], spin[0]), extent(space[2], spin[2]),
                         extent(space[1], spin[1]), extent(space[3], spin[3])});
                    !mem_res)
                {
                    return std::unexpected("build_ucc_spin_block_cache_from_eri: " + mem_res.error());
                }

                cache.spin_blocks.push_back({{space, spin}, std::move(block)});
            }
        }
        catch (const std::exception &ex)
        {
            return std::unexpected(
                "build_ucc_spin_block_cache_from_eri: " + std::string(ex.what()));
        }

        return cache;
    }

} // namespace HartreeFock::Correlation::CC
