// planck-rys-roots: the Rys rule against an independent reference.
//
// An n-point Rys rule for argument T must reproduce the Boys moments
//   F_k(T) = int_0^1 t^{2k} e^{-T t^2} dt,   k = 0 .. 2n-1,
// as sum_i w_i r_i^k (r_i = t_i^2). The moments here come from composite
// Gauss-Legendre quadrature in long double, which is stable at any T, so the
// check shares no arithmetic with the solver in rys_roots.cpp.
//
// Why it exists: until 2026-10-25 the solver built its moments by upward
// recursion from T = 1, which loses every digit at T in [1, ~5] for the
// F_k up to k = 21 that n >= 7 roots need. Those rules had weights that did
// not sum to F_0. No test called rys_roots_weights directly; the Rys
// regression cases never reach n >= 7, and the high-L Rys case is a single
// atom (T = 0 for every quartet), so nothing exercised the window.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

#include "integrals/rys_roots.h"

namespace
{
    constexpr int kNMax = HartreeFock::Rys::RYS_MAX_ROOTS;
    constexpr int kGL = 20;

    struct GaussLegendre
    {
        long double x[kGL], w[kGL];
    };

    GaussLegendre make_gauss_legendre()
    {
        GaussLegendre g;
        for (int i = 0; i < kGL; ++i)
        {
            long double x = std::cos(std::numbers::pi_v<long double> * (i + 0.75L) / (kGL + 0.5L)), dp = 0;
            for (int it = 0; it < 100; ++it)
            {
                long double p0 = 1, p1 = x;
                for (int k = 2; k <= kGL; ++k)
                {
                    const long double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                    p0 = p1, p1 = p2;
                }
                dp = kGL * (x * p1 - p0) / (x * x - 1);
                const long double dx = p1 / dp;
                x -= dx;
                if (std::abs(dx) < 1e-30L)
                    break;
            }
            g.x[i] = x, g.w[i] = 2 / ((1 - x * x) * dp * dp);
        }
        return g;
    }

    // F_0 .. F_{2 kNMax - 1} at T, by quadrature over [0, min(1, 14/sqrt(T))]
    // (beyond that the integrand is below e^-196).
    std::vector<long double> moments(long double T, const GaussLegendre &g)
    {
        std::vector<long double> F(2 * kNMax, 0.0L);
        const long double top = std::min(1.0L, 14.0L / std::sqrt(std::max(T, 1e-30L)));
        constexpr int kPanels = 400;
        for (int p = 0; p < kPanels; ++p)
        {
            const long double a = top * p / kPanels, b = top * (p + 1) / kPanels;
            const long double mid = 0.5L * (a + b), half = 0.5L * (b - a);
            for (int i = 0; i < kGL; ++i)
            {
                const long double t = mid + half * g.x[i], u = t * t;
                long double term = half * g.w[i] * std::exp(-T * u);
                for (int k = 0; k < 2 * kNMax; ++k, term *= u)
                    F[k] += term;
            }
        }
        return F;
    }
}

int main()
{
    const GaussLegendre g = make_gauss_legendre();

    // A log grid to 2000, made dense over [0.5, 6], where the old recursion
    // failed, and including T = 0.
    std::vector<double> Ts{0.0};
    for (double T = 1e-6; T < 2000.0; T *= 1.15)
        Ts.push_back(T);
    for (double T = 0.5; T <= 6.0; T += 0.01)
        Ts.push_back(T);

    int failures = 0;
    double worst[kNMax + 1] = {};
    double r[kNMax], w[kNMax];
    for (double T : Ts)
    {
        const std::vector<long double> F = moments(T, g);
        for (int n = 1; n <= kNMax; ++n)
        {
            HartreeFock::Rys::rys_roots_weights(n, T, r, w);
            bool ok = true;
            for (int i = 0; i < n; ++i)
                ok = ok && w[i] > 0.0 && r[i] > 0.0 && r[i] < 1.0 && (i == 0 || r[i] > r[i - 1]);
            double err = 0.0;
            for (int k = 0; k < 2 * n; ++k)
            {
                long double q = 0.0L;
                for (int i = 0; i < n; ++i)
                    q += static_cast<long double>(w[i]) * std::pow(static_cast<long double>(r[i]), k);
                err = std::max(err, static_cast<double>(std::abs(q - F[k]) / F[k]));
            }
            worst[n] = std::max(worst[n], err);
            // n <= 6: the double-precision eigensolve's floor (4-5e-14
            // measured). Above, the moment (Hankel) problem's own
            // conditioning, about 4x per root: measured 1.7e-13 (n = 7) to
            // 4.8e-11 (n = 11). The bound keeps a 2-3x margin. The old
            // recursion failed it from n = 4 (5e-12) to n = 11 (1.6e2).
            const double bound = 1e-13 * std::pow(4.0, std::max(0, n - 6));
            if (!ok || !(err <= bound))
            {
                if (++failures <= 10)
                    std::printf("FAIL n=%d T=%.6g: %s, moment error %.2e (bound %.0e)\n", n, T,
                                ok ? "rule valid" : "rule degenerate", err, bound);
            }
        }
    }
    for (int n = 1; n <= kNMax; ++n)
        std::printf("n=%2d  worst moment error %.2e over %zu T values\n", n, worst[n], Ts.size());
    std::printf("%s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
