#ifndef HF_GRADIENT_H
#define HF_GRADIENT_H

#include <Eigen/Core>
#include <optional>

#include "base/types.h"

namespace HartreeFock
{
    namespace Gradient
    {
        struct ExchangeGradientKernel
        {
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

        // Restricted Kohn-Sham (RKS): same Pulay/Coulomb/exchange skeleton as RHF.
        // Step 1 carries only the full-range exchange coefficient through this
        // descriptor; short-range exchange and omega are staged for the
        // range-separated follow-up and are currently ignored here.
        std::expected<Eigen::MatrixXd, std::string> compute_rks_gradient(
            const HartreeFock::Calculator &calc,
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const ExchangeGradientKernel &exchange_kernel);

        // Unrestricted Kohn-Sham (UKS): UHF-like Coulomb/exchange derivatives
        // with hybrid scaling on same-spin exchange contractions. As in the RKS
        // path, only the full-range coefficient is consumed in Step 1.
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
