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
    static constexpr double RYS_T_ZERO = 1.0e-15;  // use Gauss-Legendre limit
    static constexpr double RYS_T_MAX  = 33.0;     // use Golub-Welsch above this

    // Compute n Rys roots (t_r^2 in (0,1)) and weights (w_r > 0) for argument T.
    // Precondition: 1 <= n <= RYS_MAX_ROOTS, T >= 0.
    void rys_roots_weights(int n, double T,
                           double* __restrict__ roots,
                           double* __restrict__ weights) noexcept;

    // Exact closed-form for n=1:
    //   t_1^2 = 1 - T*exp(-T)/(1-exp(-T))   (T > 0)
    //   w_1   = (1 - exp(-T)) / T            (T > 0)
    //   t_1^2 -> 1/2,  w_1 -> 1             (T -> 0)
    inline void rys_1pt(double T, double& root, double& weight) noexcept
    {
        if (T < RYS_T_ZERO) {
            root   = 0.5;
            weight = 1.0;
        } else {
            const double e  = std::exp(-T);
            const double mu0 = (1.0 - e) / T;
            root   = 1.0 - T * e / (1.0 - e);
            weight = mu0;
        }
    }

}} // namespace HartreeFock::Rys

#endif // HF_RYS_ROOTS_H
