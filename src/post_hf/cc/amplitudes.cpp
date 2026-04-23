#include "post_hf/cc/amplitudes.h"

#include <vector>

namespace HartreeFock::Correlation::CC
{
    namespace
    {
        std::expected<DenseTensorView, std::string> checked_amplitude_tensor(
            Tensor2D &t1,
            Tensor4D &t2,
            Tensor6D *t3,
            int excitation_rank)
        {
            switch (excitation_rank)
            {
                case 1:
                    return make_tensor_view(t1);
                case 2:
                    return make_tensor_view(t2);
                case 3:
                    if (t3 != nullptr && t3->size() > 0)
                        return make_tensor_view(*t3);
                    break;
                default:
                    break;
            }
            return std::unexpected("Requested excitation rank is not available in this tensor pack");
        }

        std::expected<ConstDenseTensorView, std::string> checked_amplitude_tensor(
            const Tensor2D &t1,
            const Tensor4D &t2,
            const Tensor6D *t3,
            int excitation_rank)
        {
            switch (excitation_rank)
            {
                case 1:
                    return make_tensor_view(t1);
                case 2:
                    return make_tensor_view(t2);
                case 3:
                    if (t3 != nullptr && t3->size() > 0)
                        return make_tensor_view(*t3);
                    break;
                default:
                    break;
            }
            return std::unexpected("Requested excitation rank is not available in this tensor pack");
        }

        std::vector<int> rank_dims(
            const RHFReference &reference,
            int excitation_rank)
        {
            std::vector<int> dims;
            dims.reserve(static_cast<std::size_t>(2 * excitation_rank));
            for (int i = 0; i < excitation_rank; ++i)
                dims.push_back(reference.n_occ);
            for (int a = 0; a < excitation_rank; ++a)
                dims.push_back(reference.n_virt);
            return dims;
        }
    } // namespace

    int DenominatorCache::max_rank() const noexcept
    {
        return d3.size() > 0 ? 3 : 2;
    }

    bool DenominatorCache::has_rank(int excitation_rank) const noexcept
    {
        return excitation_rank >= 1 && excitation_rank <= max_rank();
    }

    std::expected<DenseTensorView, std::string> DenominatorCache::tensor(int excitation_rank)
    {
        return checked_amplitude_tensor(d1, d2, &d3, excitation_rank);
    }

    std::expected<ConstDenseTensorView, std::string> DenominatorCache::tensor(int excitation_rank) const
    {
        return checked_amplitude_tensor(d1, d2, &d3, excitation_rank);
    }

    int RCCSDAmplitudes::max_rank() const noexcept
    {
        return 2;
    }

    bool RCCSDAmplitudes::has_rank(int excitation_rank) const noexcept
    {
        return excitation_rank >= 1 && excitation_rank <= 2;
    }

    std::expected<DenseTensorView, std::string> RCCSDAmplitudes::tensor(int excitation_rank)
    {
        return checked_amplitude_tensor(t1, t2, nullptr, excitation_rank);
    }

    std::expected<ConstDenseTensorView, std::string> RCCSDAmplitudes::tensor(int excitation_rank) const
    {
        return checked_amplitude_tensor(t1, t2, nullptr, excitation_rank);
    }

    int RCCSDTAmplitudes::max_rank() const noexcept
    {
        return 3;
    }

    bool RCCSDTAmplitudes::has_rank(int excitation_rank) const noexcept
    {
        return excitation_rank >= 1 && excitation_rank <= 3;
    }

    std::expected<DenseTensorView, std::string> RCCSDTAmplitudes::tensor(int excitation_rank)
    {
        return checked_amplitude_tensor(t1, t2, &t3, excitation_rank);
    }

    std::expected<ConstDenseTensorView, std::string> RCCSDTAmplitudes::tensor(int excitation_rank) const
    {
        return checked_amplitude_tensor(t1, t2, &t3, excitation_rank);
    }

    int ArbitraryOrderDenominatorCache::max_rank() const noexcept
    {
        return static_cast<int>(by_rank.size());
    }

    bool ArbitraryOrderDenominatorCache::has_rank(int excitation_rank) const noexcept
    {
        return excitation_rank >= 1 && excitation_rank <= max_rank();
    }

    std::expected<DenseTensorView, std::string> ArbitraryOrderDenominatorCache::tensor(int excitation_rank)
    {
        if (!has_rank(excitation_rank))
            return std::unexpected("Requested denominator rank is not available");
        return make_tensor_view(by_rank[static_cast<std::size_t>(excitation_rank - 1)]);
    }

    std::expected<ConstDenseTensorView, std::string> ArbitraryOrderDenominatorCache::tensor(int excitation_rank) const
    {
        if (!has_rank(excitation_rank))
            return std::unexpected("Requested denominator rank is not available");
        return make_tensor_view(by_rank[static_cast<std::size_t>(excitation_rank - 1)]);
    }

    int ArbitraryOrderRCCAmplitudes::max_rank() const noexcept
    {
        return static_cast<int>(by_rank.size());
    }

    bool ArbitraryOrderRCCAmplitudes::has_rank(int excitation_rank) const noexcept
    {
        return excitation_rank >= 1 && excitation_rank <= max_rank();
    }

    std::expected<DenseTensorView, std::string> ArbitraryOrderRCCAmplitudes::tensor(int excitation_rank)
    {
        if (!has_rank(excitation_rank))
            return std::unexpected("Requested amplitude rank is not available");
        return make_tensor_view(by_rank[static_cast<std::size_t>(excitation_rank - 1)]);
    }

    std::expected<ConstDenseTensorView, std::string> ArbitraryOrderRCCAmplitudes::tensor(int excitation_rank) const
    {
        if (!has_rank(excitation_rank))
            return std::unexpected("Requested amplitude rank is not available");
        return make_tensor_view(by_rank[static_cast<std::size_t>(excitation_rank - 1)]);
    }

    std::expected<DenominatorCache, std::string> build_denominator_cache(
        const RHFReference &reference,
        bool include_triples)
    {
        try
        {
            DenominatorCache denoms;
            denoms.d1 = Tensor2D(reference.n_occ, reference.n_virt, 0.0);
            denoms.d2 = Tensor4D(reference.n_occ, reference.n_occ, reference.n_virt, reference.n_virt, 0.0);
            if (include_triples)
                denoms.d3 = Tensor6D(reference.n_occ, reference.n_occ, reference.n_occ,
                                     reference.n_virt, reference.n_virt, reference.n_virt, 0.0);

            for (int i = 0; i < reference.n_occ; ++i)
                for (int a = 0; a < reference.n_virt; ++a)
                    denoms.d1(i, a) = reference.eps_occ(i) - reference.eps_virt(a);

            for (int i = 0; i < reference.n_occ; ++i)
                for (int j = 0; j < reference.n_occ; ++j)
                    for (int a = 0; a < reference.n_virt; ++a)
                        for (int b = 0; b < reference.n_virt; ++b)
                            denoms.d2(i, j, a, b) =
                                reference.eps_occ(i) + reference.eps_occ(j) -
                                reference.eps_virt(a) - reference.eps_virt(b);

            if (include_triples)
            {
                for (int i = 0; i < reference.n_occ; ++i)
                    for (int j = 0; j < reference.n_occ; ++j)
                        for (int k = 0; k < reference.n_occ; ++k)
                            for (int a = 0; a < reference.n_virt; ++a)
                                for (int b = 0; b < reference.n_virt; ++b)
                                    for (int c = 0; c < reference.n_virt; ++c)
                                        denoms.d3(i, j, k, a, b, c) =
                                            reference.eps_occ(i) + reference.eps_occ(j) + reference.eps_occ(k) -
                                            reference.eps_virt(a) - reference.eps_virt(b) - reference.eps_virt(c);
            }

            return denoms;
        }
        catch (const std::exception &ex)
        {
            return std::unexpected("build_denominator_cache: " + std::string(ex.what()));
        }
    }

    std::expected<ArbitraryOrderDenominatorCache, std::string> build_arbitrary_order_denominator_cache(
        const RHFReference &reference,
        int max_excitation_rank)
    {
        if (max_excitation_rank < 1)
            return std::unexpected("build_arbitrary_order_denominator_cache: max_excitation_rank must be at least 1.");

        try
        {
            ArbitraryOrderDenominatorCache denoms;
            denoms.by_rank.reserve(static_cast<std::size_t>(max_excitation_rank));

            for (int rank = 1; rank <= max_excitation_rank; ++rank)
            {
                TensorND tensor(rank_dims(reference, rank), 0.0);
                std::vector<int> indices(static_cast<std::size_t>(2 * rank), 0);

                const std::size_t total = tensor.size();
                for (std::size_t linear = 0; linear < total; ++linear)
                {
                    std::size_t cursor = linear;
                    for (int pos = 2 * rank - 1; pos >= 0; --pos)
                    {
                        const int dim = tensor.dims[static_cast<std::size_t>(pos)];
                        indices[static_cast<std::size_t>(pos)] = static_cast<int>(cursor % static_cast<std::size_t>(dim));
                        cursor /= static_cast<std::size_t>(dim);
                    }

                    double denom = 0.0;
                    for (int occ = 0; occ < rank; ++occ)
                        denom += reference.eps_occ(indices[static_cast<std::size_t>(occ)]);
                    for (int vir = 0; vir < rank; ++vir)
                        denom -= reference.eps_virt(indices[static_cast<std::size_t>(rank + vir)]);

                    tensor(indices) = denom;
                }

                denoms.by_rank.push_back(std::move(tensor));
            }

            return denoms;
        }
        catch (const std::exception &ex)
        {
            return std::unexpected("build_arbitrary_order_denominator_cache: " + std::string(ex.what()));
        }
    }

    RCCSDAmplitudes make_zero_rccsd_amplitudes(const RHFReference &reference)
    {
        // The tensor dimensions follow the conventional CC index order `(i,j,a,b)`
        // so later algebra can mirror the equations closely.
        return RCCSDAmplitudes{
            .t1 = Tensor2D(reference.n_occ, reference.n_virt, 0.0),
            .t2 = Tensor4D(reference.n_occ, reference.n_occ, reference.n_virt, reference.n_virt, 0.0),
        };
    }

    RCCSDTAmplitudes make_zero_rccsdt_amplitudes(const RHFReference &reference)
    {
        return RCCSDTAmplitudes{
            .t1 = Tensor2D(reference.n_occ, reference.n_virt, 0.0),
            .t2 = Tensor4D(reference.n_occ, reference.n_occ, reference.n_virt, reference.n_virt, 0.0),
            .t3 = Tensor6D(reference.n_occ, reference.n_occ, reference.n_occ,
                           reference.n_virt, reference.n_virt, reference.n_virt, 0.0),
        };
    }

    ArbitraryOrderRCCAmplitudes make_zero_rcc_amplitudes(
        const RHFReference &reference,
        int max_excitation_rank)
    {
        ArbitraryOrderRCCAmplitudes amps;
        amps.by_rank.reserve(static_cast<std::size_t>(max_excitation_rank));
        for (int rank = 1; rank <= max_excitation_rank; ++rank)
            amps.by_rank.emplace_back(rank_dims(reference, rank), 0.0);
        return amps;
    }
} // namespace HartreeFock::Correlation::CC
