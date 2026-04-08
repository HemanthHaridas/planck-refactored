#ifndef HF_RYS_ROOTS_H
#define HF_RYS_ROOTS_H

#include <cmath>

namespace HartreeFock
{
    namespace Rys
    {

        // Maximum Rys roots for MAX_L=6 (H shells): n = floor(4*5/2)+1 = 11.
        static constexpr int RYS_MAX_ROOTS = 11;

        // T thresholds for solver tier selection.
        static constexpr double RYS_T_ZERO = 1.0e-14; // use Gauss-Legendre limit

        // Compute n Rys roots (t_r^2 in (0,1)) and weights (w_r > 0) for argument T.
        // Precondition: 1 <= n <= RYS_MAX_ROOTS, T >= 0.
        void rys_roots_weights(int n, double T,
                               double *__restrict__ roots,
                               double *__restrict__ weights) noexcept;

        // Exact 1-point rule for the Rys/Boys measure:
        //   w_1   = F_0(T)
        //   t_1^2 = F_1(T) / F_0(T)
        // where F_m(T) = integral_0^1 t^(2m) exp(-T t^2) dt.
        inline void rys_1pt(double T, double &root, double &weight) noexcept
        {
            if (T < RYS_T_ZERO)
            {
                root = 1.0 / 3.0;
                weight = 1.0;
            }
            else
            {
                const double sqrtT = std::sqrt(T);
                const double F0 = 0.5 * std::sqrt(M_PI / T) * std::erf(sqrtT);
                const double F1 = (F0 - std::exp(-T)) / (2.0 * T);
                root = F1 / F0;
                weight = F0;
            }
        }

    } // namespace Rys
} // namespace HartreeFock

#endif // HF_RYS_ROOTS_H
