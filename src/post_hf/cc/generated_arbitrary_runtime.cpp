#include "post_hf/cc/generated_arbitrary_runtime.h"

#include <cmath>
#include <exception>

namespace HartreeFock::Correlation::CC
{
    namespace
    {
        std::expected<void, std::string> validate_kernel_bundle(
            const ArbitraryOrderTensorCCState &state,
            const GeneratedArbitraryOrderKernels &kernels)
        {
            if (state.max_excitation_rank < 1)
                return std::unexpected("validate_kernel_bundle: state max_excitation_rank must be at least 1.");
            if (kernels.max_excitation_rank != state.max_excitation_rank)
                return std::unexpected(
                    "validate_kernel_bundle: kernel bundle and state must agree on max_excitation_rank.");
            if (!kernels.energy)
                return std::unexpected("validate_kernel_bundle: energy kernel is missing.");
            if (static_cast<int>(kernels.residuals_by_rank.size()) != kernels.max_excitation_rank)
                return std::unexpected(
                    "validate_kernel_bundle: residual kernel count must match max_excitation_rank.");
            if (state.denominators.max_rank() != state.max_excitation_rank ||
                state.amplitudes.max_rank() != state.max_excitation_rank)
            {
                return std::unexpected(
                    "validate_kernel_bundle: amplitudes and denominators must be allocated through max_excitation_rank.");
            }
            for (int rank = 1; rank <= kernels.max_excitation_rank; ++rank)
            {
                if (!kernels.residuals_by_rank[static_cast<std::size_t>(rank - 1)])
                {
                    return std::unexpected(
                        "validate_kernel_bundle: residual kernel missing for excitation rank " +
                        std::to_string(rank) + ".");
                }
            }
            return {};
        }

    } // namespace

    TensorND to_tensor_nd(const Tensor2D &tensor)
    {
        return TensorND({tensor.dim1, tensor.dim2}, tensor.data);
    }

    TensorND to_tensor_nd(const Tensor4D &tensor)
    {
        return TensorND({tensor.dim1, tensor.dim2, tensor.dim3, tensor.dim4}, tensor.data);
    }

    TensorND to_tensor_nd(const Tensor6D &tensor)
    {
        return TensorND({tensor.dim1, tensor.dim2, tensor.dim3, tensor.dim4, tensor.dim5, tensor.dim6}, tensor.data);
    }

    TensorND to_tensor_nd(const TensorND &tensor)
    {
        return tensor;
    }

    std::expected<ArbitraryOrderResiduals, std::string>
    evaluate_generated_arbitrary_order_residuals(
        const ArbitraryOrderTensorCCState &state,
        const GeneratedArbitraryOrderKernels &kernels)
    {
        auto valid = validate_kernel_bundle(state, kernels);
        if (!valid)
            return std::unexpected(valid.error());

        try
        {
            ArbitraryOrderResiduals residuals;
            residuals.by_rank.reserve(static_cast<std::size_t>(state.max_excitation_rank));
            for (int rank = 1; rank <= state.max_excitation_rank; ++rank)
            {
                const auto &kernel = kernels.residuals_by_rank[static_cast<std::size_t>(rank - 1)];
                TensorND tensor = kernel(
                    state.reference,
                    state.mo_blocks,
                    state.denominators,
                    state.amplitudes);
                if (tensor.dims != state.denominators.tensor(rank).dims)
                {
                    return std::unexpected(
                        "evaluate_generated_arbitrary_order_residuals: residual tensor shape mismatch at rank " +
                        std::to_string(rank) + ".");
                }
                residuals.by_rank.push_back(std::move(tensor));
            }
            return residuals;
        }
        catch (const std::exception &ex)
        {
            return std::unexpected(
                "evaluate_generated_arbitrary_order_residuals: " + std::string(ex.what()));
        }
    }

    std::expected<GeneratedArbitraryOrderSolveResult, std::string>
    run_generated_arbitrary_order_iterations(
        ArbitraryOrderTensorCCState state,
        const GeneratedArbitraryOrderKernels &kernels,
        unsigned int max_iterations,
        double tol_energy,
        double tol_residual,
        double damping,
        bool use_diis,
        int diis_dim)
    {
        if (max_iterations == 0)
            return std::unexpected("run_generated_arbitrary_order_iterations: max_iterations must be positive.");

        auto valid = validate_kernel_bundle(state, kernels);
        if (!valid)
            return std::unexpected(valid.error());

        try
        {
            AmplitudeDIIS diis(diis_dim);
            double previous_energy = kernels.energy(
                state.reference, state.mo_blocks, state.denominators, state.amplitudes);

            GeneratedArbitraryOrderSolveResult result{
                .state = state,
                .correlation_energy = previous_energy,
            };

            for (unsigned int iter = 1; iter <= max_iterations; ++iter)
            {
                auto residuals_res =
                    evaluate_generated_arbitrary_order_residuals(result.state, kernels);
                if (!residuals_res)
                    return std::unexpected(residuals_res.error());

                auto metrics_res = update_amplitudes_with_jacobi_diis(
                    result.state.amplitudes,
                    *residuals_res,
                    result.state.denominators,
                    diis,
                    damping,
                    use_diis);
                if (!metrics_res)
                    return std::unexpected(metrics_res.error());

                const double energy = kernels.energy(
                    result.state.reference,
                    result.state.mo_blocks,
                    result.state.denominators,
                    result.state.amplitudes);

                result.iterations = iter;
                result.energy_change = energy - previous_energy;
                result.correlation_energy = energy;
                result.metrics = std::move(*metrics_res);

                previous_energy = energy;

                if (std::abs(result.energy_change) < tol_energy &&
                    result.metrics.residual_rms < tol_residual)
                {
                    result.converged = true;
                    return result;
                }
            }

            return result;
        }
        catch (const std::exception &ex)
        {
            return std::unexpected(
                "run_generated_arbitrary_order_iterations: " + std::string(ex.what()));
        }
    }
} // namespace HartreeFock::Correlation::CC
