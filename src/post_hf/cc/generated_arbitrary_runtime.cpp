#include "post_hf/cc/generated_arbitrary_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>

#include "io/logging.h"
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
            // Gap B4: the state must carry an amplitude block for every sector the
            // bundle declares a residual for (ensure_amplitude_sectors reconciles
            // this after prepare). A missing sector block would silently drop the
            // sector's contribution, so fail loudly instead.
            for (const auto &sr : kernels.sector_residuals)
            {
                if (!sr.kernel)
                    return std::unexpected(std::format(
                        "validate_kernel_bundle: sector residual kernel missing for (rank {}, tag {}).",
                        sr.excitation_rank, sr.tag));
                const bool have = std::any_of(
                    state.amplitudes.sectors.begin(), state.amplitudes.sectors.end(),
                    [&](const auto &entry)
                    {
                        return entry.first.first == sr.excitation_rank &&
                               entry.first.second == sr.tag;
                    });
                if (!have)
                    return std::unexpected(std::format(
                        "validate_kernel_bundle: state has no amplitude block for sector (rank {}, tag {}); "
                        "call ensure_amplitude_sectors before solving.",
                        sr.excitation_rank, sr.tag));
            }
            return {};
        }

        std::expected<double, std::string> evaluate_generated_arbitrary_order_energy(
            const ArbitraryOrderTensorCCState &state,
            const GeneratedArbitraryOrderKernels &kernels,
            [[maybe_unused]] const char *context)
        {
            return kernels.energy(
                state.reference,
                state.mo_blocks,
                state.denominators,
                state.amplitudes);
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

    void ensure_amplitude_sectors(
        ArbitraryOrderTensorCCState &state,
        const GeneratedArbitraryOrderKernels &kernels)
    {
        for (const auto &[rank, tag] : kernels.sector_tags)
        {
            // idempotent: a restart may already have seeded this sector.
            const bool have = std::any_of(
                state.amplitudes.sectors.begin(), state.amplitudes.sectors.end(),
                [&](const auto &entry)
                { return entry.first.first == rank && entry.first.second == tag; });
            if (have)
                continue;
            // a sector block has the same occ/vir dims as its rank's reference
            // block (the spin projection lives in the algebra, not the shape).
            const auto &ref_dims =
                state.amplitudes.by_rank[static_cast<std::size_t>(rank - 1)].dims;
            state.amplitudes.sectors.push_back(
                {{rank, tag}, TensorND(ref_dims, 0.0)});
        }
    }

    std::expected<ArbitraryOrderResiduals, std::string>
    evaluate_generated_arbitrary_order_residuals(
        const ArbitraryOrderTensorCCState &state,
        const GeneratedArbitraryOrderKernels &kernels)
    {
        auto valid = validate_kernel_bundle(state, kernels);
        if (!valid)
            return std::unexpected(valid.error());

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
            auto denominator = state.denominators.tensor(rank);
            if (!denominator)
            {
                return std::unexpected(
                    "evaluate_generated_arbitrary_order_residuals: " + denominator.error());
            }
            if (tensor.dims != denominator->dims)
            {
                return std::unexpected(
                    "evaluate_generated_arbitrary_order_residuals: residual tensor shape mismatch at rank " +
                    std::to_string(rank) + ".");
            }
            residuals.by_rank.push_back(std::move(tensor));
        }

        // Gap B4: evaluate each higher Sz sector residual, keyed (rank, tag) in
        // the bundle's order (which matches the state's amplitude sectors), so the
        // Jacobi/DIIS update lines them up index-for-index. The sector residual's
        // shape must match its rank's reference (a sector is a rank-2n amplitude
        // of the same occ/vir dims), so it is validated against that denominator.
        residuals.sectors.reserve(kernels.sector_residuals.size());
        for (const auto &sr : kernels.sector_residuals)
        {
            TensorND tensor = sr.kernel(
                state.reference,
                state.mo_blocks,
                state.denominators,
                state.amplitudes);
            auto denominator = state.denominators.tensor(sr.excitation_rank);
            if (!denominator)
                return std::unexpected(
                    "evaluate_generated_arbitrary_order_residuals: " + denominator.error());
            if (tensor.dims != denominator->dims)
                return std::unexpected(std::format(
                    "evaluate_generated_arbitrary_order_residuals: sector residual shape mismatch at (rank {}, tag {}).",
                    sr.excitation_rank, sr.tag));
            residuals.sectors.push_back(
                {{sr.excitation_rank, sr.tag}, std::move(tensor)});
        }
        return residuals;
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
        int diis_dim,
        const std::string &log_tag)
    {
        if (max_iterations == 0)
            return std::unexpected("run_generated_arbitrary_order_iterations: max_iterations must be positive.");

        auto valid = validate_kernel_bundle(state, kernels);
        if (!valid)
            return std::unexpected(valid.error());

        AmplitudeDIIS diis(diis_dim);
        auto initial_energy_res = evaluate_generated_arbitrary_order_energy(
            state,
            kernels,
            "run_generated_arbitrary_order_iterations");
        if (!initial_energy_res)
            return std::unexpected(initial_energy_res.error());
        double previous_energy = *initial_energy_res;

        GeneratedArbitraryOrderSolveResult result{
            .state = state,
            .correlation_energy = previous_energy,
        };

        for (unsigned int iter = 1; iter <= max_iterations; ++iter)
        {
            const auto iter_start = std::chrono::steady_clock::now();

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

            auto energy_res = evaluate_generated_arbitrary_order_energy(
                result.state,
                kernels,
                "run_generated_arbitrary_order_iterations");
            if (!energy_res)
                return std::unexpected(energy_res.error());
            const double energy = *energy_res;

            result.iterations = iter;
            result.energy_change = energy - previous_energy;
            result.correlation_energy = energy;
            result.metrics = std::move(*metrics_res);

            previous_energy = energy;

            const double time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count();
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                log_tag + " Iter :",
                std::format(
                    "{:3d}  E_corr={:.10f}  dE={:+.3e}  rms(res)={:.3e}  rms(step)={:.3e}  diis={}  t={:.3f}s",
                    iter, energy, result.energy_change, result.metrics.residual_rms,
                    result.metrics.update_rms, diis.size(), time_sec));

            if (std::abs(result.energy_change) < tol_energy &&
                result.metrics.residual_rms < tol_residual)
            {
                result.converged = true;
                return result;
            }
        }

        return std::unexpected(
            "run_generated_arbitrary_order_iterations: generated tensor iterations did not converge within " +
            std::to_string(max_iterations) + " iterations.");
    }
} // namespace HartreeFock::Correlation::CC
