#ifndef HF_HESSIAN_H
#define HF_HESSIAN_H

#include "base/types.h"
#include <Eigen/Dense>

namespace HartreeFock
{
    namespace Freq
    {
        struct HessianResult
        {
            Eigen::MatrixXd hessian;        // 3N×3N Cartesian Hessian, Ha/Bohr²
            Eigen::VectorXd frequencies;    // n_vib, cm⁻¹ (negative = imaginary)
            Eigen::MatrixXd normal_modes;   // 3N × n_vib mass-unweighted, column-normalised
            double          zpe;            // zero-point energy, Ha
            int             n_imaginary;    // count of imaginary frequencies
            bool            is_linear;      // molecule linearity flag
            int             n_vib;          // number of vibrational modes (3N-5 or 3N-6)
        };

        // Compute the semi-numerical Hessian via central finite differences of
        // analytic gradients:  H[:,j] = (g(x+h·ê_j) - g(x-h·ê_j)) / (2h).
        // Requires a converged SCF in calc.  After building H, calls vibrational_analysis().
        HessianResult compute_hessian(HartreeFock::Calculator& calc);

        // Mass-weight H, project out translations and rotations (Eckart conditions),
        // diagonalise, and convert eigenvalues to cm⁻¹.
        // Fills result.frequencies, result.normal_modes, result.zpe, result.n_imaginary.
        void vibrational_analysis(HessianResult& result,
                                  const HartreeFock::Calculator& calc);

    }
} // namespace HartreeFock::Freq

#endif // HF_HESSIAN_H
