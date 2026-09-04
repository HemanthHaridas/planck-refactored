#ifndef HF_LOOKUP_BOYS_H
#define HF_LOOKUP_BOYS_H

#include <span>

namespace HartreeFock
{
    namespace Lookup
    {

        // Boys function F_n(x) = integral_0^1 t^(2n) exp(-x*t^2) dt
        //
        // Evaluated using a precomputed table (step 0.1, n = 0..65) with a
        // 6-term Taylor series for interpolation within each interval.
        // For x beyond the table range the asymptotic formula is used:
        //   F_n(x) = Gamma(n + 1/2) / (2 * x^(n + 1/2))
        double boys(int n, double x) noexcept;

        // Vector Boys: fill out[0..nmax] = F_0(x)..F_nmax(x) for a single x,
        // where nmax = out.size() - 1. VRR seeds need the whole F_0..F_mmax
        // column for one argument, and calling boys(n,x) in a loop redoes the
        // table-index / delta / bounds setup for every n (profiled at ~26% of
        // the OS Fock build). This shares that setup across all orders.
        // Bitwise-identical to looping boys(n,x): same table, same Taylor terms,
        // same asymptotic tail — only the per-order setup is hoisted.
        void boys_vec(double x, std::span<double> out) noexcept;

    } // namespace Lookup
} // namespace HartreeFock

#endif // HF_LOOKUP_BOYS_H
