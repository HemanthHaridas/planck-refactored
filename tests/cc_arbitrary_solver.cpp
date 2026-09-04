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
        auto unpacked = unpack_amplitudes(packed, restored);
        if (!unpacked.has_value())
            return expect(false, unpacked.error());

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
        auto residuals = make_zero_rcc_residuals(ref, 4);
        if (!residuals.has_value())
            return expect(false, residuals.error());
        return expect(residuals->by_rank.size() == 4, "Expected ranks 1..4 residual storage") &&
               expect(residuals->by_rank[0].dims == std::vector<int>({2, 1}), "Rank-1 residual dims mismatch") &&
               expect(residuals->by_rank[1].dims == std::vector<int>({2, 2, 1, 1}), "Rank-2 residual dims mismatch") &&
               expect(residuals->by_rank[3].dims == std::vector<int>({2, 2, 2, 2, 1, 1, 1, 1}), "Rank-4 residual dims mismatch");
    }

    bool test_make_zero_amplitudes_allocates_sectors()
    {
        // Gap B1: the sector-aware allocator zero-inits both the per-rank
        // reference blocks and each higher Sz sector, keyed (rank, tag). The
        // sector block has the same occ/vir dims as its rank's reference.
        const RHFReference ref{
            .n_ao = 0,
            .n_mo = 0,
            .n_occ = 2,
            .n_virt = 1,
        };
        auto amps = make_zero_rcc_amplitudes(
            ref, 4, std::vector<std::pair<int, std::string>>{{4, "aaabaaab"}});

        const std::vector<int> rank4_dims{2, 2, 2, 2, 1, 1, 1, 1};
        // reference blocks still present + zero
        if (!expect(amps.by_rank.size() == 4, "Expected ranks 1..4 reference storage"))
            return false;
        if (!expect(amps.by_rank[3].dims == rank4_dims, "Rank-4 reference dims mismatch"))
            return false;
        // the second Sz sector is allocated, correctly shaped, and zero
        auto sec = amps.sector_tensor(4, "aaabaaab");
        if (!sec.has_value())
            return expect(false, "sector_tensor(4, aaabaaab) should be allocated: " + sec.error());
        if (!expect(sec->dims == rank4_dims, "Sector block dims must match rank-4 reference"))
            return false;
        double maxabs = 0.0;
        for (std::size_t k = 0; k < sec->size(); ++k)
            maxabs = std::max(maxabs, std::abs(sec->data[k]));
        if (!expect(maxabs == 0.0, "Sector block must be zero-initialized"))
            return false;
        // a sector that was NOT requested errors (does not silently zero-fill)
        auto missing = amps.sector_tensor(4, "aabbaabb");
        return expect(!missing.has_value(),
                      "Unrequested sector must error, not return a view");
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

    bool test_jacobi_update_drives_sector_block()
    {
        // Gap B4: a higher Sz sector amplitude block is Jacobi-updated from its
        // OWN residual, using its rank's reference denominator (B2). Start the
        // sector at zero; one step should move it to damping * R / D_rank.
        ArbitraryOrderRCCAmplitudes amps;
        amps.by_rank = {
            TensorND({1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{0.0}),
        };
        // one rank-4 sector, zero-initialized
        amps.sectors.push_back(
            {{4, "aaabaaab"}, TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{0.0})});

        ArbitraryOrderResiduals residuals;
        residuals.by_rank = {
            TensorND({1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{0.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{0.0}),
        };
        // nonzero sector residual -> the sector must move off zero
        residuals.sectors.push_back(
            {{4, "aaabaaab"}, TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{9.0})});

        ArbitraryOrderDenominatorCache denoms;
        denoms.by_rank = {
            TensorND({1, 1}, std::vector<double>{1.0}),
            TensorND({1, 1, 1, 1}, std::vector<double>{1.0}),
            TensorND({1, 1, 1, 1, 1, 1}, std::vector<double>{1.0}),
            TensorND({1, 1, 1, 1, 1, 1, 1, 1}, std::vector<double>{3.0}),  // D_rank4
        };

        AmplitudeDIIS diis(4);
        auto metrics_res = update_amplitudes_with_jacobi_diis(
            amps, residuals, denoms, diis, 0.5, /*use_diis=*/false);
        if (!metrics_res.has_value())
            return expect(false, metrics_res.error());

        // sector step = damping * R / D_rank4 = 0.5 * 9.0 / 3.0 = 1.5
        auto sec = static_cast<const ArbitraryOrderRCCAmplitudes &>(amps)
                       .sector_tensor(4, "aaabaaab");
        if (!sec.has_value())
            return expect(false, "sector view missing after update: " + sec.error());
        if (!expect_close(sec->data[0], 1.5, 1e-12,
                          "Sector block must be Jacobi-updated from its own residual"))
            return false;
        // the reference rank-4 block, with a zero residual, stays put
        return expect_close(amps.by_rank[3].data[0], 0.0, 1e-12,
                            "Reference rank-4 block should not move on a zero residual");
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

    bool test_bundle_carries_sector_residual()
    {
        // Gap B3: the bundle holds sector residuals keyed (rank, tag) alongside
        // the per-rank reference residuals, and the matching sector_tags entry
        // that feeds the allocator (B1). The sector kernel is invokable and its
        // (rank, tag) route it to the correct amplitude block for B4.
        GeneratedArbitraryOrderKernels kernels;
        kernels.max_excitation_rank = 4;
        kernels.sector_tags.push_back({4, "aaabaaab"});
        kernels.sector_residuals.push_back(
            {4, "aaabaaab",
             [](const CanonicalRHFCCReference &,
                const TensorCCBlockCache &,
                const ArbitraryOrderDenominatorCache &,
                const ArbitraryOrderRCCAmplitudes &) -> TensorND
             {
                 return TensorND({1, 1, 1, 1, 1, 1, 1, 1},
                                 std::vector<double>{0.5});
             }});

        if (!expect(kernels.sector_tags.size() == 1 &&
                        kernels.sector_tags[0] == std::pair<int, std::string>{4, "aaabaaab"},
                    "Bundle must carry the (4, aaabaaab) sector tag"))
            return false;
        if (!expect(kernels.sector_residuals.size() == 1, "Bundle must carry one sector residual"))
            return false;
        const auto &sr = kernels.sector_residuals[0];
        if (!expect(sr.excitation_rank == 4 && sr.tag == "aaabaaab",
                    "Sector residual must be keyed (4, aaabaaab)"))
            return false;
        // the kernel is callable and returns the sector-shaped residual
        CanonicalRHFCCReference ref;
        TensorCCBlockCache blocks;
        ArbitraryOrderDenominatorCache denoms;
        ArbitraryOrderRCCAmplitudes amps;
        TensorND out = sr.kernel(ref, blocks, denoms, amps);
        return expect(out.dims == std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1}) &&
                          out.data.size() == 1 && out.data[0] == 0.5,
                      "Sector residual kernel must return its rank-4 residual");
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
                    const TensorND &denom = denoms.by_rank[static_cast<std::size_t>(rank - 1)];
                    const TensorND &amp = amps.by_rank[static_cast<std::size_t>(rank - 1)];
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

    bool test_generated_driver_solves_a_sector_block()
    {
        // Gap B4 end-to-end: a bundle with a rank-4 SECTOR residual, driven
        // through ensure_amplitude_sectors -> validate -> the full solve loop.
        // The sector amplitude must converge to its own fixed point (residual 0),
        // proving evaluate + update + DIIS all handle the sector block.
        ArbitraryOrderTensorCCState state;
        state.max_excitation_rank = 4;
        for (int rank = 1; rank <= 4; ++rank)
        {
            std::vector<int> dims(static_cast<std::size_t>(2 * rank), 1);
            state.denominators.by_rank.emplace_back(
                dims, std::vector<double>{static_cast<double>(1 << rank)});
            state.amplitudes.by_rank.emplace_back(dims, std::vector<double>{0.0});
        }

        GeneratedArbitraryOrderKernels kernels;
        kernels.max_excitation_rank = 4;
        kernels.energy =
            [](const CanonicalRHFCCReference &, const TensorCCBlockCache &,
               const ArbitraryOrderDenominatorCache &,
               const ArbitraryOrderRCCAmplitudes &amps) -> double
        {
            double e = 0.0;
            for (const auto &t : amps.by_rank)
                for (double v : t.data)
                    e += v;
            for (const auto &s : amps.sectors)
                for (double v : s.second.data)
                    e += v;
            return e;
        };
        // reference residuals: drive each rank to 0 amplitude (R = -D*t)
        for (int rank = 1; rank <= 4; ++rank)
            kernels.residuals_by_rank.push_back(
                [rank](const CanonicalRHFCCReference &, const TensorCCBlockCache &,
                       const ArbitraryOrderDenominatorCache &denoms,
                       const ArbitraryOrderRCCAmplitudes &amps) -> TensorND
                {
                    const TensorND &d = denoms.by_rank[static_cast<std::size_t>(rank - 1)];
                    const TensorND &a = amps.by_rank[static_cast<std::size_t>(rank - 1)];
                    TensorND r(d.dims, 0.0);
                    for (std::size_t i = 0; i < d.size(); ++i)
                        r.data[i] = -d.data[i] * a.data[i];
                    return r;
                });
        // sector residual: fixed point at t_sector = target / D_rank4.
        // R = target - D*t  =>  converges to t = target/D = 5.0 / 16.0 = 0.3125.
        kernels.sector_tags.push_back({4, "aaabaaab"});
        kernels.sector_residuals.push_back(
            {4, "aaabaaab",
             [](const CanonicalRHFCCReference &, const TensorCCBlockCache &,
                const ArbitraryOrderDenominatorCache &denoms,
                const ArbitraryOrderRCCAmplitudes &amps) -> TensorND
             {
                 const TensorND &d = denoms.by_rank[3];
                 auto sec = amps.sector_tensor(4, "aaabaaab").value();
                 TensorND r(d.dims, 0.0);
                 for (std::size_t i = 0; i < d.size(); ++i)
                     r.data[i] = 5.0 - d.data[i] * sec.data[i];
                 return r;
             }});

        ensure_amplitude_sectors(state, kernels);
        if (!expect(state.amplitudes.sectors.size() == 1,
                    "ensure_amplitude_sectors should allocate the bundle's sector"))
            return false;

        auto solve_res = run_generated_arbitrary_order_iterations(
            state, kernels, 50, 1e-13, 1e-13, 1.0, false, 4);
        if (!solve_res.has_value())
            return expect(false, solve_res.error());
        const auto &solve = *solve_res;
        auto sec = static_cast<const ArbitraryOrderRCCAmplitudes &>(solve.state.amplitudes)
                       .sector_tensor(4, "aaabaaab");
        if (!sec.has_value())
            return expect(false, "sector view missing after solve: " + sec.error());
        return expect(solve.converged, "sector-carrying solve should converge") &&
               expect_close(sec->data[0], 5.0 / 16.0, 1e-10,
                            "Sector block must converge to target/D_rank4");
    }
} // namespace

int main()
{
    bool ok = true;
    ok = test_pack_round_trip() && ok;
    ok = test_make_zero_residuals() && ok;
    ok = test_make_zero_amplitudes_allocates_sectors() && ok;
    ok = test_jacobi_update_across_ranks() && ok;
    ok = test_jacobi_update_drives_sector_block() && ok;
    ok = test_diis_path_runs_for_arbitrary_rank() && ok;
    ok = test_layout_mismatch_is_reported() && ok;
    ok = test_bundle_carries_sector_residual() && ok;
    ok = test_generated_runtime_driver_converges_with_mock_kernels() && ok;
    ok = test_generated_driver_solves_a_sector_block() && ok;
    return ok ? 0 : 1;
}
