#ifndef HF_GEOMOPT_H
#define HF_GEOMOPT_H

#include "base/types.h"
#include <Eigen/Core>
#include <expected>
#include <functional>
#include <string>
#include <vector>

namespace HartreeFock
{
    namespace Opt
    {
        using GradientRunner = std::function<std::expected<Eigen::VectorXd, std::string>(HartreeFock::Calculator &)>;

        // SCF (+ MP2 when requested) and analytic gradient at the geometry in
        // calc._molecule._standard: rebuilds basis, working state and PCM cavity.
        // Returns natoms × 3 (Ha/Bohr) and stores it in calc._gradient.
        // warm_start seeds the SCF with the previous converged density.
        std::expected<Eigen::MatrixXd, std::string> run_scf_gradient(HartreeFock::Calculator &calc,
                                                                     bool warm_start);

        struct GeomOptResult
        {
            bool converged = false;
            double energy = 0.0;
            double grad_max = 0.0;
            double grad_rms = 0.0;
            int iterations = 0;
            Eigen::MatrixXd final_coords; // natoms × 3, Bohr
            Eigen::MatrixXd gradient;     // natoms × 3, Ha/Bohr at final geometry
            std::vector<double> energies; // per-step total energies
        };

        // Run geometry optimization starting from the initial geometry in calc.
        // Uses two-loop L-BFGS with Armijo backtracking line search.
        // On return, calc._molecule._standard holds the optimized geometry,
        // calc._gradient holds the gradient at the optimized geometry, and
        // calc._total_energy holds the final energy.
        std::expected<GeomOptResult, std::string> run_geomopt(HartreeFock::Calculator &calc);
        std::expected<GeomOptResult, std::string> run_geomopt(HartreeFock::Calculator &calc,
                                                              const GradientRunner &gradient_runner);

        // Run geometry optimization in generalized internal coordinates (GIC).
        // Uses BFGS Hessian update in redundant IC space with iterative
        // Cartesian back-transform.  Same return semantics as run_geomopt.
        std::expected<GeomOptResult, std::string> run_geomopt_ic(HartreeFock::Calculator &calc);
        std::expected<GeomOptResult, std::string> run_geomopt_ic(HartreeFock::Calculator &calc,
                                                                 const GradientRunner &gradient_runner);
    } // namespace Opt
} // namespace HartreeFock

#endif // !HF_GEOMOPT_H
