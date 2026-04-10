#include "post_hf/cc/amplitudes.h"

#include <exception>

namespace HartreeFock::Correlation::CC
{
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
} // namespace HartreeFock::Correlation::CC
