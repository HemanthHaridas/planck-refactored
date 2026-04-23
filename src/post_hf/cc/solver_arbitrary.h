#ifndef HF_POSTHF_CC_SOLVER_ARBITRARY_H
#define HF_POSTHF_CC_SOLVER_ARBITRARY_H

#include <expected>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "post_hf/cc/amplitudes.h"
#include "post_hf/cc/diis.h"

namespace HartreeFock::Correlation::CC
{
    // Generated arbitrary-order kernels naturally produce one residual tensor
    // per excitation rank, so the solver mirrors the amplitude/denominator
    // packs with the same rank-indexed layout.
    struct ArbitraryOrderResiduals
    {
        std::vector<TensorND> by_rank; // rank r stored at by_rank[r-1]

        [[nodiscard]] int max_rank() const noexcept;
        [[nodiscard]] bool has_rank(int excitation_rank) const noexcept;
        [[nodiscard]] std::expected<DenseTensorView, std::string> tensor(int excitation_rank);
        [[nodiscard]] std::expected<ConstDenseTensorView, std::string> tensor(int excitation_rank) const;
    };

    struct ArbitraryOrderIterationMetrics
    {
        double residual_rms = 0.0;
        double update_rms = 0.0;
        std::vector<double> residual_rms_by_rank;
        std::vector<double> step_rms_by_rank;
    };

    std::expected<ArbitraryOrderResiduals, std::string> make_zero_rcc_residuals(
        const RHFReference &reference,
        int max_excitation_rank);

    Eigen::VectorXd pack_amplitudes(const ArbitraryOrderRCCAmplitudes &amps);
    std::expected<void, std::string> unpack_amplitudes(
        const Eigen::VectorXd &packed,
        ArbitraryOrderRCCAmplitudes &amps);

    Eigen::VectorXd pack_residuals(const ArbitraryOrderResiduals &residuals);

    [[nodiscard]] double tensor_rms(ConstDenseTensorView tensor);
    [[nodiscard]] double residual_rms(const ArbitraryOrderResiduals &residuals);

    std::expected<ArbitraryOrderIterationMetrics, std::string>
    update_amplitudes_with_jacobi_diis(
        ArbitraryOrderRCCAmplitudes &amps,
        const ArbitraryOrderResiduals &residuals,
        const ArbitraryOrderDenominatorCache &denominators,
        AmplitudeDIIS &diis,
        double damping,
        bool use_diis);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_SOLVER_ARBITRARY_H
