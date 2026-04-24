#ifndef HF_GRADIENT_H
#define HF_GRADIENT_H

#include "base/types.h"
#include <Eigen/Core>

namespace HartreeFock
{
    namespace Gradient
    {
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
