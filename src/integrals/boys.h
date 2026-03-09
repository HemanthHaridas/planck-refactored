#ifndef HF_BOYS_H
#define HF_BOYS_H

#include <cmath>

namespace HartreeFock
{
    namespace ObaraSaika
    {
        // F_n(x) = integral_0^1 t^(2n) exp(-x*t^2) dt
        //
        // Used for nuclear attraction and two-electron integrals.
        // Three regions:
        //   x < 1e-7   : F_n(0) = 1/(2n+1)  (Taylor limit)
        //   x > 20.0   : asymptotic Γ(n+1/2) / (2 * x^(n+1/2))
        //   otherwise  : F_0 via erf, then upward recurrence
        inline double _boys(const int n, const double x)
        {
            if (x < 1e-7)
                return 1.0 / (2 * n + 1);

            if (x > 20.0)
                return std::tgamma(n + 0.5) / (2.0 * std::pow(x, n + 0.5));

            // F_0(x) = (1/2) * sqrt(pi/x) * erf(sqrt(x))
            const double sqrt_x = std::sqrt(x);
            const double ex     = std::exp(-x);
            double f = 0.5 * std::sqrt(M_PI / x) * std::erf(sqrt_x);

            // Upward recurrence: F_{m+1}(x) = ((2m+1)*F_m(x) - e^{-x}) / (2x)
            // Stable for moderate x (< 20) and small n
            for (int m = 0; m < n; m++)
                f = ((2 * m + 1) * f - ex) / (2.0 * x);

            return f;
        }
    }
}

#endif // !HF_BOYS_H
