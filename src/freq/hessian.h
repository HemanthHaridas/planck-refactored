#ifndef HF_HESSIAN_H
#define HF_HESSIAN_H

#include "base/types.h"
#include <Eigen/Dense>
#include <functional>
#include <string>
#include <vector>

namespace HartreeFock
{
    namespace Freq
    {
        using GradientMatrixRunner = std::function<Eigen::MatrixXd(HartreeFock::Calculator &)>;

        struct HessianResult
        {
            Eigen::MatrixXd hessian;                // 3N×3N Cartesian Hessian, Ha/Bohr²
            Eigen::VectorXd frequencies;            // n_vib, cm⁻¹ (negative = imaginary)
            Eigen::MatrixXd normal_modes;           // 3N × n_vib mass-unweighted, column-normalised
            std::vector<std::string> mode_symmetry; // n_vib Mulliken labels when symmetry analysis succeeds
            double zpe;                             // zero-point energy, Ha
            int n_imaginary;                        // count of imaginary frequencies
            bool is_linear;                         // molecule linearity flag
            int n_vib;                              // number of vibrational modes (3N-5 or 3N-6)
        };

        // Compute the semi-numerical Hessian via central finite differences of
        // analytic gradients:  H[:,j] = (g(x+h·ê_j) - g(x-h·ê_j)) / (2h).
        // Requires a converged SCF in calc.  After building H, calls vibrational_analysis().
        HessianResult compute_hessian(HartreeFock::Calculator &calc);
        HessianResult compute_hessian(HartreeFock::Calculator &calc,
                                      const GradientMatrixRunner &gradient_runner);

        // Mass-weight H, project out translations and rotations (Eckart conditions),
        // diagonalise, and convert eigenvalues to cm⁻¹.
        // Fills result.frequencies, result.normal_modes, result.zpe, result.n_imaginary.
        void vibrational_analysis(HessianResult &result,
                                  const HartreeFock::Calculator &calc);

    } // namespace Freq
} // namespace HartreeFock

#endif // HF_HESSIAN_H
