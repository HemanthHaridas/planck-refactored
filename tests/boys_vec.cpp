// boys_vec must be bitwise-identical to looping the scalar boys(n,x): it only
// hoists the per-order table/delta setup, same table and Taylor terms. Covers
// the in-table path, the interval boundaries, and the asymptotic tail (x>=x_max).
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

#include "lookup/boys.h"

int main()
{
    bool ok = true;
    const double xs[] = {0.0, 0.05, 0.1, 1.0, 7.35, 12.9, 157.0, 200.0, 1e4};
    const int nmaxs[] = {0, 1, 5, 14, 40, 65};

    for (double x : xs)
        for (int nmax : nmaxs)
        {
            std::vector<double> vec(nmax + 1);
            HartreeFock::Lookup::boys_vec(x, std::span<double>(vec));
            for (int n = 0; n <= nmax; ++n)
            {
                const double s = HartreeFock::Lookup::boys(n, x);
                const double v = vec[n];
                // Both NaN (n>=TABLE_COLS) counts as agreement.
                const bool both_nan = std::isnan(s) && std::isnan(v);
                if (!both_nan && s != v)
                {
                    std::printf("MISMATCH x=%g n=%d scalar=%.17g vec=%.17g\n",
                                x, n, s, v);
                    ok = false;
                }
            }
        }

    if (!ok)
    {
        std::puts("FAILED: boys_vec deviates from scalar boys");
        return 1;
    }
    std::puts("boys_vec: OK (bitwise-identical to scalar boys)");
    return 0;
}
