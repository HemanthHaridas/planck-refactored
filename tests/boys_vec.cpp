// boys_vec must be bitwise-identical to looping the scalar boys(n,x), and
// boys_batch to boys_vec on each x. Covers the in-table path, the interval
// boundaries, the asymptotic tail (x>=x_max) and orders past the table (NaN).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#include "lookup/boys.h"
#include "require_boys_library.h"

int main()
{
    if (!require_boys_library())
        return 1;

    bool ok = true;
    const double xs[] = {0.0, 0.05, 0.1, 1.0, 7.35, 12.9, 157.0, 200.0, 1e4};
    const int nmaxs[] = {0, 1, 3, 5, 14, 40, 65};

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

    // boys_batch over all xs at once, byte-compared with boys_vec per x.
    for (int nmax : nmaxs)
    {
        const std::size_t width = static_cast<std::size_t>(nmax) + 1;
        std::vector<double> batch(std::size(xs) * width), vec(width);
        HartreeFock::Lookup::boys_batch(std::span<const double>(xs), nmax, std::span<double>(batch));
        for (std::size_t i = 0; i < std::size(xs); ++i)
        {
            HartreeFock::Lookup::boys_vec(xs[i], std::span<double>(vec));
            if (std::memcmp(vec.data(), batch.data() + i * width, width * sizeof(double)) != 0)
            {
                std::printf("MISMATCH boys_batch x=%g nmax=%d\n", xs[i], nmax);
                ok = false;
            }
        }
    }

    if (!ok)
    {
        std::puts("FAILED: boys_vec / boys_batch deviate");
        return 1;
    }
    std::puts("boys_vec: OK (bitwise-identical to scalar boys); boys_batch: OK (bitwise-identical to boys_vec)");
    return 0;
}
