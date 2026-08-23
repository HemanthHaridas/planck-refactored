// U5.3b: the generated UNRESTRICTED CC entry point.
//
// The UCC sibling of run_rccsdtq, and deliberately a separate file rather than a
// branch inside it: the two share the solver loop but nothing else. This one
// prepares an all-sectors state from a UHF reference (prepare_generated_ucc_state,
// U5.1b), reconciles the amplitude blocks the bundle declares, and drives the same
// Jacobi/DIIS iteration.
//
// Deliberately NOT carried over from the RCC path:
//   - warm start. It recurses to rank-1 through the RCC registry, which returns
//     restricted bundles; seeding an unrestricted solve from them would be a
//     silent reference mismatch. Revisit once a UCC rank ladder exists.
//   - .ccamp persistence. The sidecar's meta carries a single (n_occ, n_virt)
//     pair, which cannot describe a spin-resolved amplitude set. Writing one
//     would produce a file that reloads into the wrong shape.
// Both are omissions with a reason, not oversights; see the checks below.

#include "post_hf/cc/generated_arbitrary_runtime.h"
#include "post_hf/cc/generated_kernel_registry.h"
#include "post_hf/cc/solver_arbitrary.h"
#include "io/logging.h"

#include <algorithm>
#include <format>

namespace HartreeFock::Correlation::CC
{
    std::expected<void, std::string> run_uccgen(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected(
                "run_uccgen: the generated UCC path is currently available only for "
                "single-point calculations.");

        const int rank = calculator._scf._cc_generated_rank;
        if (rank < 2)
            return std::unexpected(std::format(
                "run_uccgen: the generated UCC path requires rank >= 2, got {}.", rank));

        // Fail here rather than at the registry, so the message names the build
        // switch instead of an internal lookup.
        if (!generated_ucc_kernels_available())
            return std::unexpected(
                "run_uccgen: this build carries no UCC kernels. Reconfigure with "
                "-DPLANCK_CC_UCC=ON.");

        auto kernels_res = make_generated_ucc_kernels(rank);
        if (!kernels_res)
            return std::unexpected("run_uccgen: " + kernels_res.error());

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "UCC :",
            std::format("Preparing generated unrestricted CC infrastructure (rank={}).", rank));

        auto state_res = prepare_generated_ucc_state(
            calculator, shell_pairs, rank, "UCC[GENERATED] :");
        if (!state_res)
            return std::unexpected("run_uccgen: " + state_res.error());

        // prepare runs before the bundle is known (as on the RCC path), so the
        // amplitude blocks are allocated here, sized from their own denominators.
        ensure_amplitude_sectors(*state_res, *kernels_res);

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "UCC :",
            std::format("Running generated UCC tensor kernels (rank={}, {} spin blocks).",
                        rank, state_res->amplitudes.sectors.size()));

        // Same solver knobs the RCC path uses, validated the same way -- a
        // non-positive tolerance or an out-of-range damping would otherwise
        // surface as a non-convergence rather than a configuration error.
        const double tol_energy = calculator._scf._tol_energy;
        const double tol_residual = calculator._scf._tol_density;
        const double damping = calculator._scf._cc_damping;
        if (tol_energy <= 0.0)
            return std::unexpected("run_uccgen: tol_energy must be positive.");
        if (tol_residual <= 0.0)
            return std::unexpected("run_uccgen: tol_density must be positive.");
        if (damping < 0.0 || damping > 1.0)
            return std::unexpected("run_uccgen: cc_damping must be between 0 and 1.");

        auto solve_res = run_generated_arbitrary_order_iterations(
            std::move(*state_res), *kernels_res,
            std::max(calculator._scf.get_max_cycles(calculator._shells.nbasis()), 100u),
            tol_energy,
            tol_residual,
            damping,
            calculator._scf._use_DIIS,
            static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)),
            std::format("UCC[GENERATED] :(rank {})", rank));
        if (!solve_res)
            return std::unexpected("run_uccgen: " + solve_res.error());

        HartreeFock::Logger::blank();
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "UCC :",
            std::format(
                "Generated UCC iterations ran {} steps, E_corr={:.10f}, dE={:+.3e}, "
                "rms(res)={:.3e}, rms(step)={:.3e}.",
                solve_res->iterations, solve_res->correlation_energy,
                solve_res->energy_change, solve_res->metrics.residual_rms,
                solve_res->metrics.update_rms));

        if (!solve_res->converged)
            return std::unexpected(std::format(
                "Generated UCC kernels did not converge within {} iterations "
                "(last dE={:+.3e}, rms(res)={:.3e}).",
                solve_res->iterations, solve_res->energy_change,
                solve_res->metrics.residual_rms));

        calculator._correlation_energy = solve_res->correlation_energy;
        return {};
    }
} // namespace HartreeFock::Correlation::CC
