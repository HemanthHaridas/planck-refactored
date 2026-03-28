#ifndef HF_GRADIENT_H
#define HF_GRADIENT_H

#include <Eigen/Core>
#include "base/types.h"

namespace HartreeFock
{
    namespace Gradient
    {
        // Analytic RHF nuclear gradient.
        // Returns natoms×3 matrix in Ha/Bohr.
        // Requires a converged RHF wavefunction in calc._info._scf.
        Eigen::MatrixXd compute_rhf_gradient(const HartreeFock::Calculator& calc,
                                             const std::vector<HartreeFock::ShellPair>& shell_pairs);

        // Analytic UHF nuclear gradient.
        // Returns natoms×3 matrix in Ha/Bohr.
        Eigen::MatrixXd compute_uhf_gradient(const HartreeFock::Calculator& calc,
                                             const std::vector<HartreeFock::ShellPair>& shell_pairs);

        // Analytic RMP2 nuclear gradient from the relaxed MP2 density and
        // Z-vector response.
        // Returns natoms×3 matrix in Ha/Bohr.
        // Requires a converged RHF reference and correlation = RMP2.
        Eigen::MatrixXd compute_rmp2_gradient(HartreeFock::Calculator& calc,
                                             const std::vector<HartreeFock::ShellPair>& shell_pairs);
    }
}

#endif // !HF_GRADIENT_H
