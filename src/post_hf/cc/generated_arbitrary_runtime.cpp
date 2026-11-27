#include "post_hf/cc/generated_arbitrary_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
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
            // U4.0: an ALL-SECTORS bundle (UCC) carries no per-rank reference
            // residuals -- every target is block-tagged. Accept an empty vector,
            // but still require a FULL one otherwise: a partially-filled
            // residuals_by_rank means a bundle lost a kernel, and that rank would
            // otherwise contribute a silent zero.
            if (!kernels.is_all_sectors() &&
                static_cast<int>(kernels.residuals_by_rank.size()) != kernels.max_excitation_rank)
            {
                return std::unexpected(
                    "validate_kernel_bundle: residual kernel count must match max_excitation_rank "
                    "(or be empty for an all-sectors bundle).");
            }
            // An all-sectors bundle with no sector residuals evaluates nothing at
            // all and would converge instantly at the reference energy.
            if (kernels.is_all_sectors() && kernels.sector_residuals.empty())
            {
                return std::unexpected(
                    "validate_kernel_bundle: bundle declares no residual kernels of either kind "
                    "(empty residuals_by_rank and empty sector_residuals).");
            }
            // U4.0: `max_rank()` counts the per-rank REFERENCE blocks, so an
            // all-sectors state reports 0 by construction -- its excitations all
            // live in `sectors`. Requiring coverage through max_excitation_rank
            // is therefore an RCC-only invariant. The all-sectors equivalent (a
            // stored amplitude block per declared sector) is enforced by the Gap
            // B4 loop below, which is stricter: it checks per (rank, tag) rather
            // than counting.
            if (!kernels.is_all_sectors() &&
                (state.denominators.max_rank() != state.max_excitation_rank ||
                 state.amplitudes.max_rank() != state.max_excitation_rank))
            {
                return std::unexpected(
                    "validate_kernel_bundle: amplitudes and denominators must be allocated through max_excitation_rank.");
            }
            for (int rank = 1; !kernels.is_all_sectors() && rank <= kernels.max_excitation_rank; ++rank)
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
            // U2.2: take the block's shape from its own denominator when one is
            // stored. For an RHF Sz sector the spin projection lives in the algebra
            // and not the shape, so a sector really does share its rank's reference
            // dims -- but for a UCC block it does not: with noa != nob, `abab` is
            // (noa,nob,nva,nvb) while `aaaa` is (noa,noa,nva,nva). The denominator
            // is already built per block (build_ucc_block_denominator), so it is
            // the one place that shape is known; `sector_tensor` falls back to the
            // rank's reference denominator on the RHF path, which reproduces the
            // old `ref_dims` exactly.
            //
            // U4.2: the fallback indexes `by_rank[rank-1]`, which does not exist
            // on an ALL-SECTORS state (no reference blocks at all). Reaching it
            // there would be an out-of-bounds read returning plausible garbage
            // dims, so the block is skipped instead and left unallocated -- which
            // `validate_kernel_bundle`'s Gap B4 loop then rejects by name. A
            // missing denominator on the UCC path is a build bug, and failing in
            // the validator with the (rank, tag) in the message beats crashing or
            // silently allocating the wrong shape here.
            auto denom = state.denominators.sector_tensor(rank, tag);
            const bool have_reference_rank =
                rank >= 1 &&
                static_cast<std::size_t>(rank) <= state.amplitudes.by_rank.size();
            if (!denom && !have_reference_rank)
                continue;
            const auto &block_dims =
                denom ? denom->dims
                      : state.amplitudes.by_rank[static_cast<std::size_t>(rank - 1)].dims;
            state.amplitudes.sectors.push_back(
                {{rank, tag}, TensorND(block_dims, 0.0)});
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
        // U4.0: skipped entirely for an all-sectors bundle, which leaves
        // `residuals.by_rank` empty to match the amplitudes it will be paired
        // against. The sector loop below is then the whole evaluation.
        residuals.by_rank.reserve(
            kernels.is_all_sectors()
                ? 0u
                : static_cast<std::size_t>(state.max_excitation_rank));
        // H0b (docs/CCGEN_GENERATED_CC_COST.md_SCOPE.md): per-rank attribution.
        // H0a showed the residual evaluation is ~100% of the iteration; this says
        // WHICH rank, which is what separates H1 (the rank-3 kernel dominates) from
        // H2 (the harness pays for lower ranks it need not recompute).
        //
        // Behind PLANCK_CC_RANK_TIME because unlike H0a's three clock reads this
        // one logs per rank per iteration -- useful while profiling, noise in a
        // production log. Inert when unset.
        const bool time_ranks = [] {
            const char *v = std::getenv("PLANCK_CC_RANK_TIME");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();

        // U3a: one call that shares the intermediates across every kernel, in
        // the order the two loops below consume (by rank, then sectors).
        std::vector<TensorND> all;
        std::size_t next = 0;
        if (kernels.all_residuals)
        {
            all = kernels.all_residuals(
                state.reference, state.mo_blocks, state.denominators, state.amplitudes);
            const std::size_t expected =
                (kernels.is_all_sectors() ? 0u : static_cast<std::size_t>(state.max_excitation_rank))
                + kernels.sector_residuals.size();
            if (all.size() != expected)
                return std::unexpected(std::format(
                    "evaluate_generated_arbitrary_order_residuals: all_residuals returned {} "
                    "tensors, the bundle declares {}.", all.size(), expected));
        }
        const auto residual = [&](const GeneratedArbitraryOrderKernels::ResidualKernel &kernel) {
            if (kernels.all_residuals)
                return std::move(all[next++]);
            return kernel(state.reference, state.mo_blocks, state.denominators, state.amplitudes);
        };

        for (int rank = 1; !kernels.is_all_sectors() && rank <= state.max_excitation_rank; ++rank)
        {
            const auto rank_start = std::chrono::steady_clock::now();
            const auto &kernel = kernels.residuals_by_rank[static_cast<std::size_t>(rank - 1)];
            TensorND tensor = residual(kernel);
            if (time_ranks)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info, "CC[RANK-TIME] :",
                    std::format(
                        "rank {} kernel t={:.4f}s elems={}",
                        rank,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - rank_start).count(),
                        tensor.data.size()));
            }
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
            TensorND tensor = residual(sr.kernel);
            // U2.2: validate against the BLOCK's denominator, not the rank's
            // reference one -- under an unrestricted reference they have different
            // shapes (see ensure_amplitude_sectors). Falls back to the rank's on
            // the RHF path, where they are the same tensor.
            auto denominator =
                state.denominators.sector_tensor(sr.excitation_rank, sr.tag);
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

    std::expected<GeneratedLambdaSolveResult, std::string>
    run_generated_lambda_iterations(
        const ArbitraryOrderTensorCCState &state,
        const GeneratedLambdaKernels &kernels,
        unsigned int max_iterations,
        double tol_residual,
        double damping,
        bool use_diis,
        int diis_dim,
        const std::string &log_tag)
    {
        if (max_iterations == 0)
            return std::unexpected("run_generated_lambda_iterations: max_iterations must be positive.");
        const std::size_t n_ranked = state.amplitudes.by_rank.size();
        const std::size_t n_out = n_ranked + kernels.sector_tags.size();
        if (kernels.max_excitation_rank != state.max_excitation_rank ||
            kernels.diagonal_scale.size() != n_out ||
            state.amplitudes.sectors.size() != kernels.sector_tags.size())
            return std::unexpected(std::format(
                "run_generated_lambda_iterations: the Λ bundle (rank {}, {} outputs, {} sectors) "
                "does not match the T state (rank {}, {} ranked, {} sectors).",
                kernels.max_excitation_rank, kernels.diagonal_scale.size(), kernels.sector_tags.size(),
                state.max_excitation_rank, n_ranked, state.amplitudes.sectors.size()));
        for (std::size_t s = 0; s < kernels.sector_tags.size(); ++s)
            if (state.amplitudes.sectors[s].first != kernels.sector_tags[s])
                return std::unexpected(
                    "run_generated_lambda_iterations: Λ and T sector orders differ.");

        // λ has t's shape; start at zero.
        GeneratedLambdaSolveResult result{.lambdas = state.amplitudes};
        for (auto &t : result.lambdas.by_rank)
            std::fill(t.data.begin(), t.data.end(), 0.0);
        for (auto &[key, t] : result.lambdas.sectors)
            std::fill(t.data.begin(), t.data.end(), 0.0);

        // Denominators scaled by c, per output. A sector gets its own entry,
        // since its c can differ from its rank's (RHF sectors otherwise fall
        // back to the rank's denominator).
        ArbitraryOrderDenominatorCache scaled;
        scaled.by_rank = state.denominators.by_rank;
        for (std::size_t r = 0; r < n_ranked; ++r)
            for (double &d : scaled.by_rank[r].data)
                d *= kernels.diagonal_scale[r];
        for (std::size_t s = 0; s < kernels.sector_tags.size(); ++s)
        {
            const auto &[rank, tag] = kernels.sector_tags[s];
            auto denom = state.denominators.sector_tensor(rank, tag);
            if (!denom)
                return std::unexpected("run_generated_lambda_iterations: " + denom.error());
            TensorND block;
            block.dims = denom->dims;
            block.data.assign(denom->data, denom->data + denom->size());
            for (double &d : block.data)
                d *= kernels.diagonal_scale[n_ranked + s];
            scaled.sectors.push_back({{rank, tag}, std::move(block)});
        }

        AmplitudeDIIS diis(diis_dim);
        for (unsigned int iter = 1; iter <= max_iterations; ++iter)
        {
            const auto iter_start = std::chrono::steady_clock::now();
            std::vector<TensorND> out = kernels.residuals(
                state.reference, state.mo_blocks, state.denominators, state.amplitudes,
                result.lambdas);
            if (out.size() != n_out)
                return std::unexpected(std::format(
                    "run_generated_lambda_iterations: the Λ kernels returned {} tensors, expected {}.",
                    out.size(), n_out));
            ArbitraryOrderResiduals residuals;
            for (std::size_t k = 0; k < n_ranked; ++k)
                residuals.by_rank.push_back(std::move(out[k]));
            for (std::size_t s = 0; s < kernels.sector_tags.size(); ++s)
                residuals.sectors.push_back({kernels.sector_tags[s], std::move(out[n_ranked + s])});

            auto metrics = update_amplitudes_with_jacobi_diis(
                result.lambdas, residuals, scaled, diis, damping, use_diis);
            if (!metrics)
                return std::unexpected("run_generated_lambda_iterations: " + metrics.error());
            result.metrics = std::move(*metrics);
            result.iterations = iter;

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, log_tag + " Iter :",
                std::format("{:3d}  rms(res)={:.3e}  rms(step)={:.3e}  diis={}  t={:.3f}s",
                            iter, result.metrics.residual_rms, result.metrics.update_rms, diis.size(),
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - iter_start).count()));

            // The residual is evaluated before the step, so it measures the
            // previous λ; that λ has already moved by one more step here.
            if (result.metrics.residual_rms < tol_residual)
            {
                result.converged = true;
                return result;
            }
        }
        return result;
    }

    std::expected<GeneratedLambdaSolveResult, std::string>
    solve_lambda_after_t(
        HartreeFock::Calculator &calculator,
        const ArbitraryOrderTensorCCState &state,
        const std::expected<GeneratedLambdaKernels, std::string> &kernels,
        const std::string &tag)
    {
        if (!kernels)
            return std::unexpected(kernels.error());
        const unsigned int max_iter =
            std::max(calculator._scf.get_max_cycles(calculator._shells.nbasis()), 100u);
        auto res = run_generated_lambda_iterations(
            state, *kernels, max_iter, calculator._scf._tol_density, calculator._scf._cc_damping,
            calculator._scf._use_DIIS, static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)),
            tag + "[LAMBDA]");
        if (!res)
            return std::unexpected(res.error());
        if (!res->converged)
            return std::unexpected(std::format(
                "Λ did not converge within {} iterations (rms(res)={:.3e}).",
                res->iterations, res->metrics.residual_rms));
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, tag,
            std::format("Λ converged in {} iterations, rms(res)={:.3e}.",
                        res->iterations, res->metrics.residual_rms));
        for (std::size_t r = 0; r < res->lambdas.by_rank.size(); ++r)
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, tag,
                std::format("Lambda rms l{} = {:.10e}", r + 1,
                            tensor_rms(ConstDenseTensorView{res->lambdas.by_rank[r].dims, res->lambdas.by_rank[r].data.data()})));
        for (const auto &[key, block] : res->lambdas.sectors)
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, tag,
                std::format("Lambda rms l{}_{} = {:.10e}", key.first, key.second,
                            tensor_rms(ConstDenseTensorView{block.dims, block.data.data()})));
        return res;
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

        // Moved, not copied: the state holds every integral block (vvvv is
        // 0.96 GB on the water dimer in cc-pVTZ), and `state` is not used
        // again below.
        GeneratedArbitraryOrderSolveResult result{
            .state = std::move(state),
            .correlation_energy = previous_energy,
        };

        for (unsigned int iter = 1; iter <= max_iterations; ++iter)
        {
            const auto iter_start = std::chrono::steady_clock::now();

            // H0a (docs/CCGEN_GENERATED_CC_COST.md_SCOPE.md): attribute the
            // iteration to its three phases. The loop already had exactly these
            // three delimited calls and already reported the total as `t=`; these
            // brackets only split what was being measured as one number.
            //
            // Always on, not behind an env var: three clock reads per iteration
            // against a residual evaluation measured in SECONDS is unmeasurable
            // overhead, and a profile that is present in every log is worth more
            // than one that has to be re-enabled to answer the next question.
            auto residuals_res =
                evaluate_generated_arbitrary_order_residuals(result.state, kernels);
            if (!residuals_res)
                return std::unexpected(residuals_res.error());
            const auto after_residuals = std::chrono::steady_clock::now();

            auto metrics_res = update_amplitudes_with_jacobi_diis(
                result.state.amplitudes,
                *residuals_res,
                result.state.denominators,
                diis,
                damping,
                use_diis);
            if (!metrics_res)
                return std::unexpected(metrics_res.error());
            const auto after_update = std::chrono::steady_clock::now();

            auto energy_res = evaluate_generated_arbitrary_order_energy(
                result.state,
                kernels,
                "run_generated_arbitrary_order_iterations");
            if (!energy_res)
                return std::unexpected(energy_res.error());
            const auto after_energy = std::chrono::steady_clock::now();
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
                    "{:3d}  E_corr={:.10f}  dE={:+.3e}  rms(res)={:.3e}  rms(step)={:.3e}  diis={}  "
                    "t={:.3f}s  t_res={:.3f}s t_upd={:.3f}s t_ene={:.3f}s",
                    iter, energy, result.energy_change, result.metrics.residual_rms,
                    result.metrics.update_rms, diis.size(), time_sec,
                    std::chrono::duration<double>(after_residuals - iter_start).count(),
                    std::chrono::duration<double>(after_update - after_residuals).count(),
                    std::chrono::duration<double>(after_energy - after_update).count()));

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
