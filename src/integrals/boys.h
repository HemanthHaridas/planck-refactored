#ifndef HF_BOYS_H
#define HF_BOYS_H

#include "lookup/boys.h"

namespace HartreeFock
{
    namespace ObaraSaika
    {
        // F_n(x) = integral_0^1 t^(2n) exp(-x*t^2) dt
        //
        // Thin wrapper over HartreeFock::Lookup::boys (license-gated loader).
        inline double _boys(const int n, const double x) noexcept
        {
            return HartreeFock::Lookup::boys(n, x);
        }
    } // namespace ObaraSaika
} // namespace HartreeFock

#endif // HF_BOYS_H
