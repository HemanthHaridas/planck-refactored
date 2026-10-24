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

        // Batched form: out[i * (nmax + 1) + n] = F_n(x[i]) for 0 <= n <= nmax.
        // Bitwise-identical to boys_vec on each x[i].
        void boys_batch(std::span<const double> x, int nmax, std::span<double> out) noexcept;

    } // namespace Lookup
} // namespace HartreeFock

#endif // HF_LOOKUP_BOYS_H
