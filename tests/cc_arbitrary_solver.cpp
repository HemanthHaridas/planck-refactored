#include "post_hf/cc/generated_arbitrary_runtime.h"
#include "post_hf/cc/solver_arbitrary.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    using namespace HartreeFock::Correlation::CC;

    bool expect(bool condition, const std::string &message)
    {
        if (condition)
            return true;
        std::cerr << message << '\n';
        return false;
    }

    bool expect_close(double value, double expected, double tol, const std::string &message)
    {
        return expect(std::abs(value - expected) <= tol,
                      message + " (got " + std::to_string(value) +
                          ", expected " + std::to_string(expected) + ")");
    }

    bool test_pack_round_trip()
    {
        ArbitraryOrderRCCAmplitudes amps;
        amps.by_rank = {
            TensorND({2, 1}, std::vector<double>{1.0, 2.0}),
            TensorND({1, 2, 1, 2}, std::vector<double>{3.0, 4.0, 5.0, 6.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{7.0}),
        };

        const Eigen::VectorXd packed = pack_amplitudes(amps);
        if (!expect(packed.size() == 7, "pack_amplitudes should flatten all stored ranks"))
            return false;

        ArbitraryOrderRCCAmplitudes restored = amps;
        for (auto &tensor : restored.by_rank)
            std::fill(tensor.data.begin(), tensor.data.end(), -99.0);
        unpack_amplitudes(packed, restored);

        return expect(restored.by_rank[0].data == amps.by_rank[0].data, "Rank-1 unpack mismatch") &&
               expect(restored.by_rank[1].data == amps.by_rank[1].data, "Rank-2 unpack mismatch") &&
               expect(restored.by_rank[2].data == amps.by_rank[2].data, "Rank-3 unpack mismatch");
    }

    bool test_make_zero_residuals()
    {
        const RHFReference ref{
            .n_ao = 0,
            .n_mo = 0,
            .n_occ = 2,
            .n_virt = 1,
        };
        const ArbitraryOrderResiduals residuals = make_zero_rcc_residuals(ref, 4);
        return expect(residuals.by_rank.size() == 4, "Expected ranks 1..4 residual storage") &&
               expect(residuals.by_rank[0].dims == std::vector<int>({2, 1}), "Rank-1 residual dims mismatch") &&
               expect(residuals.by_rank[1].dims == std::vector<int>({2, 2, 1, 1}), "Rank-2 residual dims mismatch") &&
               expect(residuals.by_rank[3].dims == std::vector<int>({2, 2, 2, 2, 1, 1, 1, 1}), "Rank-4 residual dims mismatch");
    }

    bool test_jacobi_update_across_ranks()
    {
        ArbitraryOrderRCCAmplitudes amps;
        amps.by_rank = {
            TensorND({2, 1}, std::vector<double>{0.5, -1.0}),
            TensorND({1, 2, 1, 2}, std::vector<double>{0.0, 1.0, 2.0, 3.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{0.25}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{-0.5}),
        };

        ArbitraryOrderResiduals residuals;
        residuals.by_rank = {
            TensorND({2, 1}, std::vector<double>{4.0, -8.0}),
            TensorND({1, 2, 1, 2}, std::vector<double>{1.0, 2.0, 4.0, 8.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{3.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{-6.0}),
        };

        ArbitraryOrderDenominatorCache denoms;
        denoms.by_rank = {
            TensorND({2, 1}, std::vector<double>{2.0, 4.0}),
            TensorND({1, 2, 1, 2}, std::vector<double>{1.0, 2.0, 4.0, 8.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{3.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{-3.0}),
        };

        AmplitudeDIIS diis(4);
        auto metrics_res = update_amplitudes_with_jacobi_diis(
            amps, residuals, denoms, diis, 0.5, false);
        if (!metrics_res.has_value())
            return expect(false, metrics_res.error());
        if (!expect(metrics_res.has_value(), "Missing update metrics"))
            return false;

        const auto &metrics = *metrics_res;
        return expect_close(amps.by_rank[0].data[0], 1.5, 1e-12, "Rank-1 Jacobi update mismatch") &&
               expect_close(amps.by_rank[0].data[1], -2.0, 1e-12, "Rank-1 Jacobi update mismatch") &&
               expect_close(amps.by_rank[1].data[0], 0.5, 1e-12, "Rank-2 Jacobi update mismatch") &&
               expect_close(amps.by_rank[1].data[3], 3.5, 1e-12, "Rank-2 Jacobi update mismatch") &&
               expect_close(amps.by_rank[2].data[0], 0.75, 1e-12, "Rank-3 Jacobi update mismatch") &&
               expect_close(amps.by_rank[3].data[0], 0.5, 1e-12, "Rank-4 Jacobi update mismatch") &&
               expect(metrics.step_rms_by_rank.size() == 4, "Expected per-rank step RMS metrics") &&
               expect(metrics.residual_rms_by_rank.size() == 4, "Expected per-rank residual RMS metrics") &&
               expect(metrics.update_rms > 0.0, "Overall update RMS should be positive") &&
               expect(diis.size() == 1, "Jacobi update should push one DIIS vector");
    }

    bool test_diis_path_runs_for_arbitrary_rank()
    {
        ArbitraryOrderRCCAmplitudes amps;
        amps.by_rank = {
            TensorND({1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{0.0}),
        };

        ArbitraryOrderResiduals residuals;
        residuals.by_rank = {
            TensorND({1, 1}, std::vector<double>{1.0}),
            TensorND({1, 1, 1, 1}, std::vector<double>{2.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{3.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{4.0}),
        };

        ArbitraryOrderDenominatorCache denoms;
        denoms.by_rank = {
            TensorND({1, 1}, std::vector<double>{1.0}),
            TensorND({1, 1, 1, 1}, std::vector<double>{1.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{1.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{1.0}),
        };

        AmplitudeDIIS diis(4);
        auto step1 = update_amplitudes_with_jacobi_diis(
            amps, residuals, denoms, diis, 0.25, true);
        if (!step1.has_value())
            return expect(false, step1.error());
        if (!expect(step1.has_value(), "Missing first DIIS step metrics"))
            return false;

        residuals.by_rank[0].data[0] = 0.5;
        residuals.by_rank[1].data[0] = 1.0;
        residuals.by_rank[2].data[0] = 1.5;
        residuals.by_rank[3].data[0] = 2.0;
        auto step2 = update_amplitudes_with_jacobi_diis(
            amps, residuals, denoms, diis, 0.25, true);
        if (!step2.has_value())
            return expect(false, step2.error());
        if (!expect(step2.has_value(), "Missing second DIIS step metrics"))
            return false;

        return expect(diis.size() == 2, "DIIS should retain arbitrary-rank history vectors") &&
               expect(std::isfinite(step2->update_rms), "DIIS step RMS should be finite") &&
               expect(step2->step_rms_by_rank.size() == 4, "DIIS path should retain per-rank step metrics");
    }

    bool test_layout_mismatch_is_reported()
    {
        ArbitraryOrderRCCAmplitudes amps;
        amps.by_rank = {TensorND({1, 1}, std::vector<double>{0.0})};

        ArbitraryOrderResiduals residuals;
        residuals.by_rank = {TensorND({2, 1}, std::vector<double>{0.0, 0.0})};

        ArbitraryOrderDenominatorCache denoms;
        denoms.by_rank = {TensorND({1, 1}, std::vector<double>{1.0})};

        AmplitudeDIIS diis(2);
        auto metrics = update_amplitudes_with_jacobi_diis(
            amps, residuals, denoms, diis, 0.5, false);
        return expect(!metrics.has_value(), "Layout mismatch should return an error");
    }

    bool test_generated_runtime_driver_converges_with_mock_kernels()
    {
        ArbitraryOrderTensorCCState state;
        state.max_excitation_rank = 4;
        state.denominators.by_rank = {
            TensorND({1, 1}, std::vector<double>{2.0}),
            TensorND({1, 1, 1, 1}, std::vector<double>{4.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{8.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{16.0}),
        };
        state.amplitudes.by_rank = {
            TensorND({1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{0.0}),
        };

        const std::vector<double> targets{1.0, -2.0, 4.0, -8.0};

        GeneratedArbitraryOrderKernels kernels;
        kernels.max_excitation_rank = 4;
        kernels.energy =
            [](const CanonicalRHFCCReference &,
               const TensorCCBlockCache &,
               const ArbitraryOrderDenominatorCache &,
               const ArbitraryOrderRCCAmplitudes &amps) -> double
        {
            double energy = 0.0;
            for (const auto &tensor : amps.by_rank)
                for (double value : tensor.data)
                    energy += value;
            return energy;
        };

        for (int rank = 1; rank <= 4; ++rank)
        {
            kernels.residuals_by_rank.push_back(
                [rank, targets](const CanonicalRHFCCReference &,
                                const TensorCCBlockCache &,
                                const ArbitraryOrderDenominatorCache &denoms,
                                const ArbitraryOrderRCCAmplitudes &amps) -> TensorND
                {
                    const auto denom = denoms.tensor(rank);
                    const auto amp = amps.tensor(rank);
                    TensorND residual(denom.dims, 0.0);
                    for (std::size_t idx = 0; idx < denom.size(); ++idx)
                        residual.data[idx] = targets[static_cast<std::size_t>(rank - 1)] -
                                             denom.data[idx] * amp.data[idx];
                    return residual;
                });
        }

        auto solve_res = run_generated_arbitrary_order_iterations(
            state,
            kernels,
            4,
            1e-12,
            1e-12,
            1.0,
            false,
            4);
        if (!solve_res.has_value())
            return expect(false, solve_res.error());

        const auto &solve = *solve_res;
        return expect(solve.converged, "Mock generated arbitrary-order solve should converge") &&
               expect(solve.iterations <= 2, "Mock generated arbitrary-order solve should converge quickly") &&
               expect_close(solve.state.amplitudes.by_rank[0].data[0], 0.5, 1e-12, "Rank-1 final amplitude mismatch") &&
               expect_close(solve.state.amplitudes.by_rank[1].data[0], -0.5, 1e-12, "Rank-2 final amplitude mismatch") &&
               expect_close(solve.state.amplitudes.by_rank[2].data[0], 0.5, 1e-12, "Rank-3 final amplitude mismatch") &&
               expect_close(solve.state.amplitudes.by_rank[3].data[0], -0.5, 1e-12, "Rank-4 final amplitude mismatch");
    }
} // namespace

int main()
{
    bool ok = true;
    ok = test_pack_round_trip() && ok;
    ok = test_make_zero_residuals() && ok;
    ok = test_jacobi_update_across_ranks() && ok;
    ok = test_diis_path_runs_for_arbitrary_rank() && ok;
    ok = test_layout_mismatch_is_reported() && ok;
    ok = test_generated_runtime_driver_converges_with_mock_kernels() && ok;
    return ok ? 0 : 1;
}
