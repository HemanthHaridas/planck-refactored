#ifndef HF_LOOKUP_BOYS_H
#define HF_LOOKUP_BOYS_H

namespace HartreeFock { namespace Lookup {

    // Boys function F_n(x) = integral_0^1 t^(2n) exp(-x*t^2) dt
    //
    // Evaluated using a precomputed table (step 0.1, n = 0..65) with a
    // 6-term Taylor series for interpolation within each interval.
    // For x beyond the table range the asymptotic formula is used:
    //   F_n(x) = Gamma(n + 1/2) / (2 * x^(n + 1/2))
    double boys(int n, double x) noexcept;

}} // namespace HartreeFock::Lookup

#endif // HF_LOOKUP_BOYS_H
