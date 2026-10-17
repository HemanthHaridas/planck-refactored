#ifndef HF_LOOKUP_BOYS_H
#define HF_LOOKUP_BOYS_H

#include <span>

namespace HartreeFock
{
    namespace Lookup
    {

        // Boys function F_n(x) = integral_0^1 t^(2n) exp(-x*t^2) dt
        double boys(int n, double x) noexcept;

        // Vector form: fill out[0..nmax] = F_0(x)..F_nmax(x) for a single x.
        // Bitwise-identical to looping boys(n, x).
        void boys_vec(double x, std::span<double> out) noexcept;

    } // namespace Lookup
} // namespace HartreeFock

#endif // HF_LOOKUP_BOYS_H
