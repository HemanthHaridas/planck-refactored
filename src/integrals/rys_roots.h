#ifndef HF_RYS_ROOTS_H
#define HF_RYS_ROOTS_H

#include <cmath>
#include <numbers>

namespace HartreeFock
{
    namespace Rys
    {

        // Maximum Rys roots for MAX_L=6 (H shells): n = floor(4*5/2)+1 = 11.
        static constexpr int RYS_MAX_ROOTS = 11;

        // Below this T the one-root rule takes its T = 0 values.
        static constexpr double RYS_T_ZERO = 1.0e-14;

        // Compute n Rys roots (t_r^2 in (0,1)) and weights (w_r > 0) for argument T.
        // Precondition: 1 <= n <= RYS_MAX_ROOTS, T >= 0.
        // Fast: segment fits below 35 + 7n, the Laguerre asymptote above
        // (rys_roots.cpp). The first call for each n builds its table.
        void rys_roots_weights(int n, double T,
                               double *__restrict__ roots,
                               double *__restrict__ weights) noexcept;

        // The same rule by discretized Stieltjes in long double: accurate
        // node by node (~1e-14), ~30 us at n = 11. Builds the fits; tests.
        void rys_roots_weights_reference(int n, double T,
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
            else if (T < 1.0)
            {
                // F_1 = (F_0 - e^-T) / (2T) cancels at small T (1.3e-10
                // relative at T = 2e-6). The positive series
                //   F_m = e^-T sum_n (2T)^n / ((2m+1)(2m+3)...(2m+2n+1))
                // cancels nothing; below T = 1 it needs about 20 terms.
                double t0 = 1.0, t1 = 1.0 / 3.0, s0 = t0, s1 = t1;
                for (int n = 1; n < 64 && t0 > 1e-18 * s0; ++n)
                {
                    t0 *= 2.0 * T / (2 * n + 1);
                    t1 *= 2.0 * T / (2 * n + 3);
                    s0 += t0, s1 += t1;
                }
                root = s1 / s0;
                weight = std::exp(-T) * s0;
            }
            else
            {
                const double sqrtT = std::sqrt(T);
                const double F0 = 0.5 * std::sqrt(std::numbers::pi / T) * std::erf(sqrtT);
                const double F1 = (F0 - std::exp(-T)) / (2.0 * T);
                root = F1 / F0;
                weight = F0;
            }
        }

    } // namespace Rys
} // namespace HartreeFock

#endif // HF_RYS_ROOTS_H
