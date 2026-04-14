#include "post_hf/cc/ccsdt.h"

#include <format>

#include "io/logging.h"
#include "post_hf/cc/determinant_space.h"
#include "post_hf/cc/tensor_backend.h"

namespace HartreeFock::Correlation::CC
{
    std::expected<RCCSDTState, std::string> prepare_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected("prepare_rccsdt: RCCSDT is currently available only for single-point calculations.");

        auto reference_res = build_rhf_reference(calculator);
        if (!reference_res)
            return std::unexpected(reference_res.error());

        auto block_res = build_mo_block_cache(calculator, shell_pairs, *reference_res, "RCCSDT :");
        if (!block_res)
            return std::unexpected(block_res.error());

        // The current determinant-space solver uses excitation-specific energy
        // denominators instead of a dense spatial T3 tensor, so the setup phase
        // avoids allocating one here.
        auto denom_res = build_denominator_cache(*reference_res, false);
        if (!denom_res)
            return std::unexpected(denom_res.error());

        return RCCSDTState{
            .reference = std::move(*reference_res),
            .mo_blocks = std::move(*block_res),
            .denominators = std::move(*denom_res),
            .amplitudes = {},
        };
    }

    std::expected<void, std::string> run_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        calculator._have_ccsd_reference_energy = false;
        calculator._ccsd_reference_correlation_energy = 0.0;

        auto selector_ref = build_rhf_reference(calculator);
        if (!selector_ref)
            return std::unexpected(selector_ref.error());

        const RCCSDTBackend backend = choose_rccsdt_backend(*selector_ref);
        if (backend == RCCSDTBackend::TensorProduction)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT :",
                "Selecting staged tensor backend because the system is larger than the determinant-space prototype limit.");
            return run_tensor_rccsdt(calculator, shell_pairs);
        }

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDT :",
            "Selecting determinant-space prototype backend for a small teaching-sized system.");

        auto state_res = prepare_rccsdt(calculator, shell_pairs);
        if (!state_res)
            return std::unexpected(state_res.error());

        auto system_res = build_rhf_spin_orbital_system(
            calculator, state_res->reference, state_res->mo_blocks);
        if (!system_res)
            return std::unexpected(system_res.error());

        auto ccsd_res = solve_determinant_cc(
            calculator, *system_res, 2, "RCCSDT[CCSD-REF] :");
        if (!ccsd_res)
            return std::unexpected("run_rccsdt: failed to build CCSD reference energy: " + ccsd_res.error());
        calculator._ccsd_reference_correlation_energy = *ccsd_res;
        calculator._have_ccsd_reference_energy = true;

        auto corr_res = solve_determinant_cc(
            calculator, *system_res, 3, "RCCSDT :");
        if (!corr_res)
            return std::unexpected("run_rccsdt: " + corr_res.error());

        calculator._correlation_energy = *corr_res;
        return {};
    }

    std::expected<UCCSDTState, std::string> prepare_uccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        (void)shell_pairs;
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected("prepare_uccsdt: UCCSDT is currently available only for single-point calculations.");

        auto reference_res = build_uhf_reference(calculator);
        if (!reference_res)
            return std::unexpected(reference_res.error());

        return UCCSDTState{
            .reference = std::move(*reference_res),
        };
    }

    std::expected<void, std::string> run_uccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        calculator._have_ccsd_reference_energy = false;
        calculator._ccsd_reference_correlation_energy = 0.0;

        auto state_res = prepare_uccsdt(calculator, shell_pairs);
        if (!state_res)
            return std::unexpected(state_res.error());

        auto system_res = build_uhf_spin_orbital_system(
            calculator, shell_pairs, state_res->reference, "UCCSDT :");
        if (!system_res)
            return std::unexpected(system_res.error());

        auto ccsd_res = solve_determinant_cc(
            calculator, *system_res, 2, "UCCSDT[CCSD-REF] :");
        if (!ccsd_res)
            return std::unexpected("run_uccsdt: failed to build UCCSD reference energy: " + ccsd_res.error());
        calculator._ccsd_reference_correlation_energy = *ccsd_res;
        calculator._have_ccsd_reference_energy = true;

        auto corr_res = solve_determinant_cc(
            calculator, *system_res, 3, "UCCSDT :");
        if (!corr_res)
            return std::unexpected("run_uccsdt: " + corr_res.error());

        calculator._correlation_energy = *corr_res;
        return {};
    }
} // namespace HartreeFock::Correlation::CC
