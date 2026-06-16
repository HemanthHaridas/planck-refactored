#include "post_hf/cc/solver_arbitrary.h"

#include <cmath>

namespace HartreeFock::Correlation::CC
{
    namespace
    {
        std::vector<int> rank_dims(
            const RHFReference &reference,
            int excitation_rank)
        {
            // Rank-r amplitudes/residuals are stored as dense tensors with
            // occupied indices first and virtual indices second:
            //   t_r(i1, ..., ir, a1, ..., ar)
            // This helper produces the shape expected by every rank-specific
            // buffer so allocation and validation stay consistent.
            std::vector<int> dims;
            dims.reserve(static_cast<std::size_t>(2 * excitation_rank));
            for (int i = 0; i < excitation_rank; ++i)
                dims.push_back(reference.n_occ);
            for (int a = 0; a < excitation_rank; ++a)
                dims.push_back(reference.n_virt);
            return dims;
        }

        std::expected<void, std::string> require_same_layout(
            ConstDenseTensorView lhs,
            ConstDenseTensorView rhs,
            const char *label)
        {
            if (lhs.dims != rhs.dims)
                return std::unexpected(std::string(label) + ": tensor dimensions do not match");
            return {};
        }

        std::expected<DenseTensorView, std::string> checked_residual_tensor(
            ArbitraryOrderResiduals &residuals,
            int excitation_rank)
        {
            if (!residuals.has_rank(excitation_rank))
                return std::unexpected("Requested residual rank is not available");
            return make_tensor_view(residuals.by_rank[static_cast<std::size_t>(excitation_rank - 1)]);
        }

        std::expected<ConstDenseTensorView, std::string> checked_residual_tensor(
            const ArbitraryOrderResiduals &residuals,
            int excitation_rank)
        {
            if (!residuals.has_rank(excitation_rank))
                return std::unexpected("Requested residual rank is not available");
            return make_tensor_view(residuals.by_rank[static_cast<std::size_t>(excitation_rank - 1)]);
        }

        double rms_norm(const Eigen::VectorXd &vector)
        {
            if (vector.size() == 0)
                return 0.0;
            return std::sqrt(vector.squaredNorm() / static_cast<double>(vector.size()));
        }
    } // namespace

    int ArbitraryOrderResiduals::max_rank() const noexcept
    {
        return static_cast<int>(by_rank.size());
    }

    bool ArbitraryOrderResiduals::has_rank(int excitation_rank) const noexcept
    {
        return excitation_rank >= 1 && excitation_rank <= max_rank();
    }

    std::expected<DenseTensorView, std::string> ArbitraryOrderResiduals::tensor(int excitation_rank)
    {
        return checked_residual_tensor(*this, excitation_rank);
    }

    std::expected<ConstDenseTensorView, std::string> ArbitraryOrderResiduals::tensor(int excitation_rank) const
    {
        return checked_residual_tensor(*this, excitation_rank);
    }

    std::expected<ArbitraryOrderResiduals, std::string> make_zero_rcc_residuals(
        const RHFReference &reference,
        int max_excitation_rank)
    {
        if (max_excitation_rank < 1)
        {
            return std::unexpected("make_zero_rcc_residuals: max_excitation_rank must be at least 1");
        }

        ArbitraryOrderResiduals residuals;
        residuals.by_rank.reserve(static_cast<std::size_t>(max_excitation_rank));
        // Allocate one zero-filled tensor per excitation manifold so the
        // solver can treat CCSD, CCSDT, ... with the same update code.
        for (int rank = 1; rank <= max_excitation_rank; ++rank)
            residuals.by_rank.emplace_back(rank_dims(reference, rank), 0.0);
        return residuals;
    }

    Eigen::VectorXd pack_amplitudes(const ArbitraryOrderRCCAmplitudes &amps)
    {
        std::size_t total_size = 0;
        for (const auto &tensor : amps.by_rank)
            total_size += tensor.size();

        // DIIS operates on flat vectors rather than ragged tensor collections.
        // We therefore serialize the rank-1, rank-2, ... amplitude blocks into
        // one contiguous vector while preserving their existing in-memory order.
        Eigen::VectorXd packed(static_cast<Eigen::Index>(total_size));
        Eigen::Index offset = 0;
        for (const auto &tensor : amps.by_rank)
            for (const double value : tensor.data)
                packed(offset++) = value;
        return packed;
    }

    std::expected<void, std::string> unpack_amplitudes(const Eigen::VectorXd &packed, ArbitraryOrderRCCAmplitudes &amps)
    {
        const Eigen::VectorXd::Index expected_size =
            static_cast<Eigen::Index>(pack_amplitudes(amps).size());
        if (packed.size() != expected_size)
        {
            return std::unexpected("unpack_amplitudes: packed vector size does not match amplitude storage");
        }

        Eigen::Index offset = 0;
        // The inverse of pack_amplitudes(): restore the solver state back into
        // the per-rank tensors after Jacobi updates and optional DIIS mixing.
        for (auto &tensor : amps.by_rank)
            for (double &value : tensor.data)
                value = packed(offset++);
        return {};
    }

    Eigen::VectorXd pack_residuals(const ArbitraryOrderResiduals &residuals)
    {
        std::size_t total_size = 0;
        for (const auto &tensor : residuals.by_rank)
            total_size += tensor.size();

        Eigen::VectorXd packed(static_cast<Eigen::Index>(total_size));
        Eigen::Index offset = 0;
        for (const auto &tensor : residuals.by_rank)
            for (const double value : tensor.data)
                packed(offset++) = value;
        return packed;
    }

    double tensor_rms(ConstDenseTensorView tensor)
    {
        if (tensor.size() == 0)
            return 0.0;

        double sum_sq = 0.0;
        for (std::size_t idx = 0; idx < tensor.size(); ++idx)
            sum_sq += tensor.data[idx] * tensor.data[idx];
        return std::sqrt(sum_sq / static_cast<double>(tensor.size()));
    }

    double residual_rms(const ArbitraryOrderResiduals &residuals)
    {
        return rms_norm(pack_residuals(residuals));
    }

    std::expected<ArbitraryOrderIterationMetrics, std::string>
    update_amplitudes_with_jacobi_diis(
        ArbitraryOrderRCCAmplitudes &amps,
        const ArbitraryOrderResiduals &residuals,
        const ArbitraryOrderDenominatorCache &denominators,
        AmplitudeDIIS &diis,
        double damping,
        bool use_diis)
    {
        if (amps.max_rank() != residuals.max_rank() ||
            amps.max_rank() != denominators.max_rank())
        {
            return std::unexpected(
                "update_amplitudes_with_jacobi_diis: amplitudes, residuals, and denominators must cover the same excitation ranks.");
        }

        const Eigen::VectorXd current = pack_amplitudes(amps);
        Eigen::VectorXd updated = current;
        const Eigen::VectorXd residual_vec = pack_residuals(residuals);

        ArbitraryOrderIterationMetrics metrics;
        metrics.residual_rms = rms_norm(residual_vec);
        metrics.residual_rms_by_rank.reserve(static_cast<std::size_t>(amps.max_rank()));
        metrics.step_rms_by_rank.reserve(static_cast<std::size_t>(amps.max_rank()));

        Eigen::Index offset = 0;
        for (int rank = 1; rank <= amps.max_rank(); ++rank)
        {
            // Pull rank-matched tensor views so each excitation manifold is
            // updated against its own residual and denominator tensor.  Layout
            // checks make the flattening logic below safe even if a caller
            // constructed one of the storage objects incorrectly.
            auto amp = static_cast<const ArbitraryOrderRCCAmplitudes &>(amps).tensor(rank);
            if (!amp)
                return std::unexpected("update_amplitudes_with_jacobi_diis: " + amp.error());
            auto residual = checked_residual_tensor(residuals, rank);
            if (!residual)
                return std::unexpected("update_amplitudes_with_jacobi_diis: " + residual.error());
            auto denom = denominators.tensor(rank);
            if (!denom)
                return std::unexpected("update_amplitudes_with_jacobi_diis: " + denom.error());

            auto amp_residual_layout =
                require_same_layout(*amp, *residual, "update_amplitudes_with_jacobi_diis");
            if (!amp_residual_layout)
                return std::unexpected(amp_residual_layout.error());
            auto amp_denom_layout =
                require_same_layout(*amp, *denom, "update_amplitudes_with_jacobi_diis");
            if (!amp_denom_layout)
                return std::unexpected(amp_denom_layout.error());

            metrics.residual_rms_by_rank.push_back(tensor_rms(*residual));

            double step_sum_sq = 0.0;
            for (std::size_t idx = 0; idx < amp->size(); ++idx)
            {
                // Plain Jacobi step:
                //   t <- t + damping * R / D
                // Small denominators are skipped instead of amplified to avoid
                // exploding updates from nearly singular orbital gaps.
                double delta = 0.0;
                if (std::abs(denom->data[idx]) >= 1e-12)
                    delta = damping * residual->data[idx] / denom->data[idx];
                updated(offset + static_cast<Eigen::Index>(idx)) += delta;
                step_sum_sq += delta * delta;
            }

            const double rank_rms =
                amp->size() == 0 ? 0.0 : std::sqrt(step_sum_sq / static_cast<double>(amp->size()));
            metrics.step_rms_by_rank.push_back(rank_rms);
            offset += static_cast<Eigen::Index>(amp->size());
        }

        // DIIS sees the fully packed amplitude/residual vectors so it can mix
        // information across all excitation ranks at once instead of solving
        // separate extrapolation problems for T1, T2, T3, ...
        diis.push(updated, residual_vec);
        if (use_diis && diis.ready())
        {
            auto diis_res = diis.extrapolate();
            if (!diis_res)
                return std::unexpected(diis_res.error());
            updated = std::move(*diis_res);
        }

        auto unpacked = unpack_amplitudes(updated, amps);
        if (!unpacked)
            return std::unexpected(unpacked.error());

        const Eigen::VectorXd update_delta = updated - current;
        metrics.update_rms = rms_norm(update_delta);

        if (use_diis && diis.size() >= 2)
        {
            // Once DIIS changes the step, the per-rank RMS values computed from
            // the raw Jacobi updates are no longer accurate.  Recompute them
            // from the final extrapolated vector so diagnostics reflect the
            // actual move accepted by the solver.
            metrics.step_rms_by_rank.clear();
            Eigen::Index rank_offset = 0;
            for (int rank = 1; rank <= amps.max_rank(); ++rank)
            {
                auto amp = amps.tensor(rank);
                if (!amp)
                    return std::unexpected("update_amplitudes_with_jacobi_diis: " + amp.error());
                const std::size_t rank_size = amp->size();
                double step_sum_sq = 0.0;
                for (std::size_t idx = 0; idx < rank_size; ++idx)
                {
                    const double delta =
                        update_delta(rank_offset + static_cast<Eigen::Index>(idx));
                    step_sum_sq += delta * delta;
                }
                const double rank_rms =
                    rank_size == 0 ? 0.0 : std::sqrt(step_sum_sq / static_cast<double>(rank_size));
                metrics.step_rms_by_rank.push_back(rank_rms);
                rank_offset += static_cast<Eigen::Index>(rank_size);
            }
        }

        return metrics;
    }
} // namespace HartreeFock::Correlation::CC
