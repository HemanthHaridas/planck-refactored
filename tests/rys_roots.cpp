// planck-rys-roots: the Rys rule (rys_roots.cpp) checked three ways.
//
//   moments     an n-point rule for argument T must reproduce the Boys
//               moments F_k(T) = int_0^1 t^{2k} e^{-T t^2} dt, k < 2n, as
//               sum_i w_i r_i^k. They come from composite Gauss-Legendre
//               quadrature (400 panels). The reference generator also
//               discretizes with Gauss-Legendre (60 panels), so this is not
//               independent of it, but it is of the fits and of any
//               moment-based solver.
//   scaling     above the asymptote cutoff the rule is exactly roots / T and
//               weights / sqrt(T); the reference must obey it.
//   fast        the fitted rule against the reference at every segment
//               boundary and both sides of the cutoff (which also checks the
//               fast path's Laguerre nodes).
//
// Why it exists: until 2026-10-25 the solver built its moments by upward
// recursion from T = 1, which loses every digit at T in [1, ~5] for the
// F_k up to k = 21 that n >= 7 roots need. Those rules had weights that did
// not sum to F_0. No test called rys_roots_weights directly; the Rys
// regression cases never reach n >= 7, and the high-L Rys case is a single
// atom (T = 0 for every quartet), so nothing exercised the window.
//
// "long double" below is double on Apple silicon; the bounds are set from
// measurements there.
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
            // Measured 4.0e-14 (n = 1) to 1.0e-13 (n = 11); the floor is
            // this file's quadrature reference, which runs in double on
            // Apple silicon (long double is double there). The moment-based
            // solver this replaced reached 4.8e-11 at n = 11, and its
            // pre-2026-10-25 recursion failed from n = 1 to 1.6e2 at n = 11.
            const double bound = 3e-13;
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

    // What an integral sees: max_i |dr_i| and max_i |dw_i| / F_0.
    const auto rule_error = [](int n, const double *a, const double *b, const double *A, const double *B)
    {
        double F0 = 0.0, e = 0.0;
        for (int i = 0; i < n; ++i)
            F0 += B[i];
        for (int i = 0; i < n; ++i)
            e = std::max({e, std::abs(a[i] - A[i]), std::abs(b[i] - B[i]) / F0});
        return e;
    };
    double R[kNMax], W[kNMax];

    // Above the cutoff the rule scales exactly: roots as 1/T, weights as
    // 1/sqrt(T). Check the reference obeys it, anchored at T = 1e4. (The
    // fast path's own Laguerre nodes are checked below, against the
    // reference, on both sides of the cutoff.)
    double worst_asym = 0.0;
    for (int n = 2; n <= kNMax; ++n)
    {
        double u[kNMax], lam[kNMax];
        HartreeFock::Rys::rys_roots_weights_reference(n, 1.0e4, u, lam);
        for (int i = 0; i < n; ++i)
            u[i] *= 1.0e4, lam[i] *= 2.0 * std::sqrt(1.0e4);
        for (double T : {35.0 + 7.0 * n, 150.0, 400.0, 1000.0})
        {
            HartreeFock::Rys::rys_roots_weights_reference(n, T, R, W);
            for (int i = 0; i < n; ++i)
                r[i] = u[i] / T, w[i] = lam[i] / (2.0 * std::sqrt(T));
            worst_asym = std::max(worst_asym, rule_error(n, r, w, R, W));
        }
    }
    std::printf("reference vs exact large-T scaling: %.2e\n", worst_asym);
    if (!(worst_asym <= 1e-14))
        ++failures, std::printf("FAIL: reference breaks the large-T scaling\n");

    // The fast rule against the reference, on every segment boundary, segment
    // midpoints, and both sides of the asymptote cutoff. Measured: <= 6.3e-15
    // for n <= 5 (what planck-rys-box-invariance needs on water 6-31G*),
    // rising to 4.2e-14 at n = 11, which is the reference's own rounding,
    // not the fits (degree 13, 16 and 19 all give the same numbers). An
    // off-by-one segment, degree 8 or a cutoff of 25 + 5n give 9.7e-6,
    // 6.6e-13 and 1.7e-12.
    for (int n = 1; n <= kNMax; ++n)
    {
        const double cut = 35.0 + 7.0 * n;
        std::vector<double> Tf{0.0, 1e-12, 1e-6, cut - 1e-9, cut, cut + 1e-9, 2.0 * cut};
        for (double T = 0.0; T < cut + 2.0; T += 0.5)
            Tf.push_back(T), Tf.push_back(T + 0.123);
        double worst_fast = 0.0;
        for (double T : Tf)
        {
            HartreeFock::Rys::rys_roots_weights(n, T, r, w);
            HartreeFock::Rys::rys_roots_weights_reference(n, T, R, W);
            worst_fast = std::max(worst_fast, rule_error(n, r, w, R, W));
        }
        const double bound = n <= 5 ? 1e-14 : 1e-13;
        std::printf("n=%2d  fast vs reference %.2e (bound %.0e)\n", n, worst_fast, bound);
        if (!(worst_fast <= bound))
            ++failures, std::printf("FAIL n=%d: fast rule departs from the reference\n", n);
    }

    std::printf("%s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
