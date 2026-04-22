#include "post_hf/cc/ccsdtq.h"

#include <algorithm>
#include <format>

#include "io/logging.h"
#include "post_hf/cc/generated_arbitrary_runtime.h"
#include "post_hf/cc/generated_kernel_registry.h"

namespace HartreeFock::Correlation::CC
{
    std::expected<void, std::string> run_rccsdtq(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected("run_rccsdtq: RCCSDTQ is currently available only for single-point calculations.");

        calculator._have_ccsd_reference_energy = false;
        calculator._ccsd_reference_correlation_energy = 0.0;

        auto state_res = prepare_generated_arbitrary_order_state(
            calculator,
            shell_pairs,
            4,
            "RCCSDTQ[TENSOR] :");
        if (!state_res)
            return std::unexpected("run_rccsdtq: " + state_res.error());

        auto kernels_res = make_generated_rccsdtq_kernels();
        if (!kernels_res)
            return std::unexpected("run_rccsdtq: " + kernels_res.error());

        const unsigned int max_iter =
            calculator._scf.get_max_cycles(calculator._shells.nbasis());
        const double tol_energy = calculator._scf._tol_energy;
        const double tol_residual = calculator._scf._tol_density;
        const double damping = calculator._scf._cc_damping;
        if (tol_energy <= 0.0)
            return std::unexpected("run_rccsdtq: tol_energy must be positive.");
        if (tol_residual <= 0.0)
            return std::unexpected("run_rccsdtq: tol_density must be positive.");
        if (damping < 0.0 || damping > 1.0)
            return std::unexpected("run_rccsdtq: cc_damping must be between 0 and 1.");

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDTQ :",
            "Running generated arbitrary-order RCCSDTQ tensor kernels.");
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDTQ :",
            std::format(
                "Prepared generated tensor state with nocc={} nvirt={} rank={}.",
                state_res->reference.orbital_partition.n_occ,
                state_res->reference.orbital_partition.n_virt,
                state_res->max_excitation_rank));

        auto solve_res = run_generated_arbitrary_order_iterations(
            std::move(*state_res),
            *kernels_res,
            max_iter,
            tol_energy,
            tol_residual,
            damping,
            calculator._scf._use_DIIS,
            static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));
        if (!solve_res)
            return std::unexpected("run_rccsdtq: " + solve_res.error());

        HartreeFock::Logger::blank();
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDTQ :",
            std::format(
                "Generated RCCSDTQ iterations ran {} steps, E_corr={:.10f}, dE={:+.3e}, rms(res)={:.3e}, rms(step)={:.3e}.",
                solve_res->iterations,
                solve_res->correlation_energy,
                solve_res->energy_change,
                solve_res->metrics.residual_rms,
                solve_res->metrics.update_rms));

        if (!solve_res->converged)
        {
            return std::unexpected(
                std::format(
                    "Generated RCCSDTQ kernels did not converge within {} iterations (last dE={:+.3e}, rms(res)={:.3e}).",
                    solve_res->iterations,
                    solve_res->energy_change,
                    solve_res->metrics.residual_rms));
        }

        calculator._correlation_energy = solve_res->correlation_energy;
        return {};
    }
} // namespace HartreeFock::Correlation::CC
