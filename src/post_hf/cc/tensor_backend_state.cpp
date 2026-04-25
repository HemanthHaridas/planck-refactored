#include "post_hf/cc/tensor_backend_internal.h"

#include <exception>
#include <format>
#include <limits>
#include <sstream>

#include "post_hf/integrals.h"

namespace HartreeFock::Correlation::CC::detail
{
    std::expected<std::size_t, std::string> checked_product(
        std::initializer_list<int> dims)
    {
        std::size_t total = 1;
        for (const int dim : dims)
        {
            if (dim < 0)
                return std::unexpected("checked_product: negative tensor dimension");

            const std::size_t dim_size = static_cast<std::size_t>(dim);
            if (dim_size != 0 &&
                total > std::numeric_limits<std::size_t>::max() / dim_size)
            {
                return std::unexpected("checked_product: tensor size overflow");
            }

            total *= dim_size;
        }
        return total;
    }

    std::expected<std::size_t, std::string> bytes_for_tensor(
        std::initializer_list<int> dims)
    {
        auto elements = checked_product(dims);
        if (!elements)
            return std::unexpected(elements.error());
        return (*elements) * sizeof(double);
    }

    std::string format_bytes(std::size_t bytes)
    {
        constexpr double kib = 1024.0;
        constexpr double mib = 1024.0 * 1024.0;
        constexpr double gib = 1024.0 * 1024.0 * 1024.0;

        const double value = static_cast<double>(bytes);
        if (value >= gib)
            return std::format("{:.2f} GiB", value / gib);
        if (value >= mib)
            return std::format("{:.2f} MiB", value / mib);
        if (value >= kib)
            return std::format("{:.2f} KiB", value / kib);
        return std::format("{} B", bytes);
    }

    std::expected<void, std::string> append_block_memory(
        std::vector<TensorMemoryBlock> &report,
        std::size_t &total_bytes,
        const std::string &label,
        std::initializer_list<int> dims)
    {
        auto elements = checked_product(dims);
        if (!elements)
            return std::unexpected(elements.error());
        const std::size_t bytes = (*elements) * sizeof(double);
        report.push_back(TensorMemoryBlock{
            .label = label,
            .elements = *elements,
            .bytes = bytes,
        });
        total_bytes += bytes;
        return {};
    }
} // namespace HartreeFock::Correlation::CC::detail

namespace HartreeFock::Correlation::CC
{
    std::expected<CanonicalRHFCCReference, std::string> build_canonical_rhf_cc_reference(
        HartreeFock::Calculator &calculator)
    {
        auto ref_res = build_rhf_reference(calculator);
        if (!ref_res)
            return std::unexpected(ref_res.error());

        const RHFReference &base = *ref_res;

        Eigen::MatrixXd C_full(base.n_ao, base.n_mo);
        C_full.leftCols(base.n_occ) = base.C_occ;
        C_full.rightCols(base.n_virt) = base.C_virt;

        const Eigen::MatrixXd fock_mo =
            C_full.transpose() * calculator._info._scf.alpha.fock * C_full;

        // Keep both views of the RHF reference: the original occupied/virtual
        // partition plus explicit oo/ov/vv Fock blocks that the tensor CC
        // kernels consume directly without repeated MO slicing.
        CanonicalRHFCCReference reference{
            .orbital_partition = std::move(*ref_res),
            .f_oo = Tensor2D(base.n_occ, base.n_occ, 0.0),
            .f_ov = Tensor2D(base.n_occ, base.n_virt, 0.0),
            .f_vv = Tensor2D(base.n_virt, base.n_virt, 0.0),
        };

        for (int i = 0; i < base.n_occ; ++i)
            for (int j = 0; j < base.n_occ; ++j)
                reference.f_oo(i, j) = fock_mo(i, j);

        for (int i = 0; i < base.n_occ; ++i)
            for (int a = 0; a < base.n_virt; ++a)
                reference.f_ov(i, a) = fock_mo(i, base.n_occ + a);

        for (int a = 0; a < base.n_virt; ++a)
            for (int b = 0; b < base.n_virt; ++b)
                reference.f_vv(a, b) = fock_mo(base.n_occ + a, base.n_occ + b);

        return reference;
    }

    std::expected<TensorCCBlockCache, std::string> build_tensor_cc_block_cache(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const CanonicalRHFCCReference &reference,
        const std::string &tag)
    {
        std::vector<double> eri_local;
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calculator, shell_pairs, eri_local, tag);

        const RHFReference &partition = reference.orbital_partition;
        const std::size_t nb = static_cast<std::size_t>(partition.n_ao);

        TensorCCBlockCache blocks;
        try
        {
            // Materialize the standard RHF MO-ERI blocks once up front.  The
            // tensor RCCSD(T) paths pay this memory cost in exchange for much
            // simpler and faster residual kernels later on.
            blocks.oooo = Tensor4D(
                partition.n_occ, partition.n_occ, partition.n_occ, partition.n_occ,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_occ,
                    partition.C_occ, partition.C_occ));
            if (auto mem_res = detail::append_block_memory(
                    blocks.memory_report,
                    blocks.total_bytes,
                    "oooo",
                    {partition.n_occ, partition.n_occ, partition.n_occ, partition.n_occ});
                !mem_res)
            {
                return std::unexpected("build_tensor_cc_block_cache: " + mem_res.error());
            }

            blocks.ooov = Tensor4D(
                partition.n_occ, partition.n_occ, partition.n_occ, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_occ,
                    partition.C_occ, partition.C_virt));
            if (auto mem_res = detail::append_block_memory(
                    blocks.memory_report,
                    blocks.total_bytes,
                    "ooov",
                    {partition.n_occ, partition.n_occ, partition.n_occ, partition.n_virt});
                !mem_res)
            {
                return std::unexpected("build_tensor_cc_block_cache: " + mem_res.error());
            }

            blocks.oovv = Tensor4D(
                partition.n_occ, partition.n_occ, partition.n_virt, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_occ,
                    partition.C_virt, partition.C_virt));
            if (auto mem_res = detail::append_block_memory(
                    blocks.memory_report,
                    blocks.total_bytes,
                    "oovv",
                    {partition.n_occ, partition.n_occ, partition.n_virt, partition.n_virt});
                !mem_res)
            {
                return std::unexpected("build_tensor_cc_block_cache: " + mem_res.error());
            }

            blocks.ovov = Tensor4D(
                partition.n_occ, partition.n_virt, partition.n_occ, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_virt,
                    partition.C_occ, partition.C_virt));
            if (auto mem_res = detail::append_block_memory(
                    blocks.memory_report,
                    blocks.total_bytes,
                    "ovov",
                    {partition.n_occ, partition.n_virt, partition.n_occ, partition.n_virt});
                !mem_res)
            {
                return std::unexpected("build_tensor_cc_block_cache: " + mem_res.error());
            }

            blocks.ovvo = Tensor4D(
                partition.n_occ, partition.n_virt, partition.n_virt, partition.n_occ,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_virt,
                    partition.C_virt, partition.C_occ));
            if (auto mem_res = detail::append_block_memory(
                    blocks.memory_report,
                    blocks.total_bytes,
                    "ovvo",
                    {partition.n_occ, partition.n_virt, partition.n_virt, partition.n_occ});
                !mem_res)
            {
                return std::unexpected("build_tensor_cc_block_cache: " + mem_res.error());
            }

            blocks.ovvv = Tensor4D(
                partition.n_occ, partition.n_virt, partition.n_virt, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_virt,
                    partition.C_virt, partition.C_virt));
            if (auto mem_res = detail::append_block_memory(
                    blocks.memory_report,
                    blocks.total_bytes,
                    "ovvv",
                    {partition.n_occ, partition.n_virt, partition.n_virt, partition.n_virt});
                !mem_res)
            {
                return std::unexpected("build_tensor_cc_block_cache: " + mem_res.error());
            }

            blocks.vvvv = Tensor4D(
                partition.n_virt, partition.n_virt, partition.n_virt, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_virt, partition.C_virt,
                    partition.C_virt, partition.C_virt));
            if (auto mem_res = detail::append_block_memory(
                    blocks.memory_report,
                    blocks.total_bytes,
                    "vvvv",
                    {partition.n_virt, partition.n_virt, partition.n_virt, partition.n_virt});
                !mem_res)
            {
                return std::unexpected("build_tensor_cc_block_cache: " + mem_res.error());
            }
        }
        catch (const std::exception &ex)
        {
            return std::unexpected("build_tensor_cc_block_cache: " + std::string(ex.what()));
        }

        return blocks;
    }

    std::string format_tensor_memory_summary(
        const TensorRCCSDTState &state)
    {
        // Report per-block sizes instead of only a grand total so users can see
        // which ERI block dominates memory when a tensor calculation becomes
        // impractical for a given basis.
        std::ostringstream out;
        out << "Tensor RCCSDT memory estimate:";
        for (const TensorMemoryBlock &block : state.mo_blocks.memory_report)
            out << std::format(" {}={}", block.label, detail::format_bytes(block.bytes));
        out << std::format(" T3~={} total_integrals={}",
                           detail::format_bytes(state.estimated_t3_bytes),
                           detail::format_bytes(state.mo_blocks.total_bytes));
        if (state.triples.allocated)
            out << std::format(" triples_workspace={}", detail::format_bytes(state.triples.storage_bytes));
        return out.str();
    }

    std::expected<void, std::string> allocate_dense_triples_workspace(
        TensorRCCSDTState &state)
    {
        if (state.triples.allocated)
            return {};

        constexpr std::size_t kDenseTriplesSoftCapBytes =
            static_cast<std::size_t>(1024) * 1024 * 1024;
        if (2 * state.estimated_t3_bytes > kDenseTriplesSoftCapBytes)
        {
            return std::unexpected(
                std::format(
                    "allocate_dense_triples_workspace: dense T3/R3 workspace would require about {}; current phase-1 tensor path is capped at {}.",
                    detail::format_bytes(2 * state.estimated_t3_bytes),
                    detail::format_bytes(kDenseTriplesSoftCapBytes)));
        }

        try
        {
            const RHFReference &partition = state.reference.orbital_partition;
            const int nocc_so = 2 * partition.n_occ;
            const int nvirt_so = 2 * partition.n_virt;
            state.triples.amplitudes = RCCSDTAmplitudes{
                .t1 = Tensor2D(nocc_so, nvirt_so, 0.0),
                .t2 = Tensor4D(nocc_so, nocc_so, nvirt_so, nvirt_so, 0.0),
                .t3 = Tensor6D(nocc_so, nocc_so, nocc_so,
                               nvirt_so, nvirt_so, nvirt_so, 0.0),
            };
            state.triples.r3 = Tensor6D(
                nocc_so, nocc_so, nocc_so,
                nvirt_so, nvirt_so, nvirt_so, 0.0);
            state.triples.allocated = true;

            auto t1_bytes = detail::bytes_for_tensor({nocc_so, nvirt_so});
            if (!t1_bytes)
                return std::unexpected("allocate_dense_triples_workspace: " + t1_bytes.error());
            auto t2_bytes = detail::bytes_for_tensor({nocc_so, nocc_so, nvirt_so, nvirt_so});
            if (!t2_bytes)
                return std::unexpected("allocate_dense_triples_workspace: " + t2_bytes.error());
            auto t3_bytes = detail::bytes_for_tensor({nocc_so, nocc_so, nocc_so,
                                                      nvirt_so, nvirt_so, nvirt_so});
            if (!t3_bytes)
                return std::unexpected("allocate_dense_triples_workspace: " + t3_bytes.error());
            state.triples.storage_bytes = *t1_bytes + *t2_bytes + 2 * (*t3_bytes);
        }
        catch (const std::exception &ex)
        {
            return std::unexpected("allocate_dense_triples_workspace: " + std::string(ex.what()));
        }

        return {};
    }

    std::expected<TensorRCCSDTState, std::string> prepare_tensor_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
        {
            return std::unexpected(
                "prepare_tensor_rccsdt: tensor RCCSDT is currently available only for single-point calculations.");
        }

        auto ref_res = build_canonical_rhf_cc_reference(calculator);
        if (!ref_res)
            return std::unexpected(ref_res.error());

        auto blocks_res = build_tensor_cc_block_cache(
            calculator, shell_pairs, *ref_res, "RCCSDT[TENSOR] :");
        if (!blocks_res)
            return std::unexpected(blocks_res.error());

        auto denom_res = build_denominator_cache(ref_res->orbital_partition, false);
        if (!denom_res)
            return std::unexpected(denom_res.error());

        TensorRCCSDTState state{
            .reference = std::move(*ref_res),
            .mo_blocks = std::move(*blocks_res),
            .denominators = std::move(*denom_res),
            .estimated_t3_elements = 0,
            .estimated_t3_bytes = 0,
        };

        const RHFReference &partition = state.reference.orbital_partition;
        const int nocc_so = 2 * partition.n_occ;
        const int nvirt_so = 2 * partition.n_virt;
        auto t3_elements = detail::checked_product(
            {nocc_so, nocc_so, nocc_so,
             nvirt_so, nvirt_so, nvirt_so});
        if (!t3_elements)
            return std::unexpected("prepare_tensor_rccsdt: " + t3_elements.error());
        state.estimated_t3_elements = *t3_elements;
        state.estimated_t3_bytes = state.estimated_t3_elements * sizeof(double);

        auto triples_res = allocate_dense_triples_workspace(state);
        if (!triples_res)
            return std::unexpected(triples_res.error());

        return state;
    }
} // namespace HartreeFock::Correlation::CC
