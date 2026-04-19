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

        int max_excitation_rank = 0;
        EnergyKernel energy;
        std::vector<ResidualKernel> residuals_by_rank; // rank r at [r-1]
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

    std::expected<ArbitraryOrderTensorCCState, std::string>
    prepare_generated_arbitrary_order_state(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        int max_excitation_rank,
        const std::string &tag = "CC[GENERATED] :");

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
        int diis_dim);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_GENERATED_ARBITRARY_RUNTIME_H
