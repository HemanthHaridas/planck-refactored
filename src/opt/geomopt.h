#ifndef HF_GEOMOPT_H
#define HF_GEOMOPT_H

#include <functional>
#include <vector>
#include <string>
#include <Eigen/Core>
#include "base/types.h"

namespace HartreeFock
{
    namespace Opt
    {
        using GradientRunner = std::function<Eigen::VectorXd(HartreeFock::Calculator&)>;

        struct GeomOptResult
        {
            bool   converged   = false;
            double energy      = 0.0;
            double grad_max    = 0.0;
            double grad_rms    = 0.0;
            int    iterations  = 0;
            Eigen::MatrixXd final_coords;   // natoms × 3, Bohr
            Eigen::MatrixXd gradient;       // natoms × 3, Ha/Bohr at final geometry
            std::vector<double> energies;   // per-step total energies
        };

        // Run geometry optimization starting from the initial geometry in calc.
        // Uses two-loop L-BFGS with Armijo backtracking line search.
        // On return, calc._molecule._standard holds the optimized geometry,
        // calc._gradient holds the gradient at the optimized geometry, and
        // calc._total_energy holds the final energy.
        GeomOptResult run_geomopt(HartreeFock::Calculator& calc);
        GeomOptResult run_geomopt(HartreeFock::Calculator& calc,
                                  const GradientRunner& gradient_runner);

        // Run geometry optimization in generalized internal coordinates (GIC).
        // Uses BFGS Hessian update in redundant IC space with iterative
        // Cartesian back-transform.  Same return semantics as run_geomopt.
        GeomOptResult run_geomopt_ic(HartreeFock::Calculator& calc);
        GeomOptResult run_geomopt_ic(HartreeFock::Calculator& calc,
                                     const GradientRunner& gradient_runner);
    }
}

#endif // !HF_GEOMOPT_H
