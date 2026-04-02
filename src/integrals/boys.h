#ifndef HF_BOYS_H
#define HF_BOYS_H

#include "lookup/boys.h"

namespace HartreeFock
{
    namespace ObaraSaika
    {
        // F_n(x) = integral_0^1 t^(2n) exp(-x*t^2) dt
        //
        // Thin wrapper — implementation lives in src/lookup/boys.cpp:
        // precomputed table (step 0.1, n = 0..65) with 6-term Taylor interpolation,
        // asymptotic formula for x beyond table range.
        inline double _boys(const int n, const double x) noexcept
        {
            return HartreeFock::Lookup::boys(n, x);
        }
    } // namespace ObaraSaika
} // namespace HartreeFock

#endif // HF_BOYS_H
