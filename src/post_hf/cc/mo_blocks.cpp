#include "post_hf/cc/mo_blocks.h"

#include <exception>

#include "post_hf/integrals.h"

namespace HartreeFock::Correlation::CC
{
    std::expected<MOBlockCache, std::string> build_mo_block_cache(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RHFReference &reference,
        const std::string &tag)
    {
        std::vector<double> eri_local;
        // Reuse the AO ERI cache when possible so repeated post-HF runs do not
        // silently recompute the expensive four-index tensor.
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calculator, shell_pairs, eri_local, tag);

        const std::size_t nb = static_cast<std::size_t>(reference.n_ao);

        MOBlockCache blocks;
        try
        {
            Eigen::MatrixXd C_full(reference.n_ao, reference.n_mo);
            C_full.leftCols(reference.n_occ) = reference.C_occ;
            C_full.rightCols(reference.n_virt) = reference.C_virt;

            // The full `(pq|rs)` tensor is kept because both the pedagogical
            // spin-orbital CCSD code and the determinant-space CCSDT prototype
            // rebuild other quantities from it.
            blocks.full = Tensor4D(
                reference.n_mo, reference.n_mo, reference.n_mo, reference.n_mo,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    C_full, C_full, C_full, C_full));
            blocks.oooo = Tensor4D(
                reference.n_occ, reference.n_occ, reference.n_occ, reference.n_occ,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_occ, reference.C_occ,
                    reference.C_occ, reference.C_occ));
            blocks.ooov = Tensor4D(
                reference.n_occ, reference.n_occ, reference.n_occ, reference.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_occ, reference.C_occ,
                    reference.C_occ, reference.C_virt));
            blocks.oovv = Tensor4D(
                reference.n_occ, reference.n_occ, reference.n_virt, reference.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_occ, reference.C_occ,
                    reference.C_virt, reference.C_virt));
            blocks.ovov = Tensor4D(
                reference.n_occ, reference.n_virt, reference.n_occ, reference.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_occ, reference.C_virt,
                    reference.C_occ, reference.C_virt));
            blocks.ovvo = Tensor4D(
                reference.n_occ, reference.n_virt, reference.n_virt, reference.n_occ,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_occ, reference.C_virt,
                    reference.C_virt, reference.C_occ));
            blocks.ovvv = Tensor4D(
                reference.n_occ, reference.n_virt, reference.n_virt, reference.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_occ, reference.C_virt,
                    reference.C_virt, reference.C_virt));
            blocks.vvvv = Tensor4D(
                reference.n_virt, reference.n_virt, reference.n_virt, reference.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_virt, reference.C_virt,
                    reference.C_virt, reference.C_virt));
        }
        catch (const std::exception &ex)
        {
            return std::unexpected("build_mo_block_cache: " + std::string(ex.what()));
        }

        return blocks;
    }
} // namespace HartreeFock::Correlation::CC
