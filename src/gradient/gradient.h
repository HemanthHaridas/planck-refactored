#ifndef HF_GRADIENT_H
#define HF_GRADIENT_H

#include <Eigen/Core>
#include <array>
#include <optional>

#include "base/types.h"

namespace HartreeFock
{
    namespace Gradient
    {
        struct ExchangeGradientKernel
        {
            // Shared descriptor for the HF-like exchange part of KS gradients.
            // Global hybrids populate only the full-range coefficient; range-
            // separated hybrids additionally use the short-range coefficient
            // and omega.
            double full_range_exchange_coefficient = 0.0;
            double short_range_exchange_coefficient = 0.0;
            double range_separation_omega = 0.0;
        };

        struct WavefunctionGradientBreakdown
        {
            Eigen::MatrixXd core_pulay;
            Eigen::MatrixXd coulomb_two_electron;
            Eigen::MatrixXd exchange_full_range;
            Eigen::MatrixXd exchange_long_range_correction;
            Eigen::MatrixXd exchange_two_electron;
            Eigen::MatrixXd two_electron;
            Eigen::MatrixXd nuclear_repulsion;
            Eigen::MatrixXd total;
        };

        const std::optional<WavefunctionGradientBreakdown> &last_wavefunction_gradient_breakdown();

        // Engine-agnostic 12-component ERI derivative for a shell-pair quartet.
        // Routes to the HGP derivative kernel when the selected integral engine
        // is HeadGordonPople, otherwise to Obara-Saika. Shared by the SCF/KS
        // gradient assembly and the MP2/UMP2 gradient response intermediates so
        // both honor the user's engine selection. kernel/omega default to the
        // plain Coulomb operator.
        std::array<double, 12> compute_eri_deriv_dispatch(
            const HartreeFock::Calculator &calc,
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0);

        // Analytic RHF nuclear gradient.
        // Returns natoms×3 matrix in Ha/Bohr.
        // Requires a converged RHF wavefunction in calc._info._scf.
        std::expected<Eigen::MatrixXd, std::string> compute_rhf_gradient(
            const HartreeFock::Calculator &calc,
            const std::vector<HartreeFock::ShellPair> &shell_pairs);

        // Analytic UHF nuclear gradient.
        // Returns natoms×3 matrix in Ha/Bohr.
        std::expected<Eigen::MatrixXd, std::string> compute_uhf_gradient(
            const HartreeFock::Calculator &calc,
            const std::vector<HartreeFock::ShellPair> &shell_pairs);

        // Restricted Kohn-Sham (RKS): same Pulay/Coulomb/exchange skeleton as
        // RHF, but with hybrid/range-separated exchange scaling carried by the
        // exchange-kernel descriptor.
        std::expected<Eigen::MatrixXd, std::string> compute_rks_gradient(
            const HartreeFock::Calculator &calc,
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const ExchangeGradientKernel &exchange_kernel);

        // Unrestricted Kohn-Sham (UKS): UHF-like Coulomb/exchange derivatives
        // with hybrid/range-separated scaling applied to the same-spin
        // exchange contractions.
        std::expected<Eigen::MatrixXd, std::string> compute_uks_gradient(
            const HartreeFock::Calculator &calc,
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const ExchangeGradientKernel &exchange_kernel);

        // Analytic RMP2 nuclear gradient from the relaxed MP2 density and
        // Z-vector response.
        // Returns natoms×3 matrix in Ha/Bohr.
        // Requires a converged RHF reference and correlation = RMP2.
        std::expected<Eigen::MatrixXd, std::string> compute_rmp2_gradient(
            HartreeFock::Calculator &calc,
            const std::vector<HartreeFock::ShellPair> &shell_pairs);

        // Analytic UMP2 nuclear gradient from spin-resolved UMP2 density and
        // pair-density intermediates.
        // Returns natoms×3 matrix in Ha/Bohr.
        // Requires a converged UHF reference and correlation = UMP2.
        std::expected<Eigen::MatrixXd, std::string> compute_ump2_gradient(
            HartreeFock::Calculator &calc,
            const std::vector<HartreeFock::ShellPair> &shell_pairs);
    } // namespace Gradient
} // namespace HartreeFock

#endif // !HF_GRADIENT_H
