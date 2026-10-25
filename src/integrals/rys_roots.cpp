#include "rys_roots.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <numbers>
#include <vector>

#include <Eigen/Eigenvalues>

// Rys roots and weights (docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md, K2.1a).
//
// The rule for n roots at argument T is the Gauss rule of the measure
// e^{-T t^2} dt on [0, 1], in u = t^2. Three pieces:
//
//   reference   discretized Stieltjes (Gautschi): the measure is replaced by
//               a Gauss-Legendre grid in t (60 panels x 20 points), the
//               three-term recurrence is built on that discrete measure with
//               compensated sums, and the rule follows from the Jacobi
//               matrix's eigenvalues and the Christoffel function. No
//               moments, so no Hankel ill-conditioning: the moment-based
//               solver it replaced reproduced moments, but its individual
//               nodes were off by up to 8e-3 at n = 11. The grid is
//               converged to <= 1.4e-15 (n <= 5) and 5e-15 (n = 9).
//               "long double" here is double on Apple silicon, so nothing
//               relies on extra precision. Slow (~30 us at n = 11); it only
//               builds tables and backs planck-rys-roots.
//   asymptote   for T >= 35 + 7n, generalized Gauss-Laguerre (alpha = -1/2)
//               in u T: roots u_i / T, weights lambda_i / (2 sqrt T). The
//               cutoff is where it meets the reference's ~2e-15 floor.
//   fits        below the cutoff, Chebyshev interpolants of degree 13 on unit
//               segments of T, built from the reference on first use of each
//               n (0.01-0.04 s, 21-270 KB). Error against the reference, in
//               max_i |dr_i| and max_i |dw_i| / F_0: <= 6.3e-15 for n <= 5,
//               4.2e-14 at n = 11 (the reference's rounding; degree 16 and 19
//               give the same). 23-44 ns per call for n = 2..11, against
//               0.46-3.0 us for the moment solver.

namespace
{
    using LD = long double;

    constexpr int kNMax = HartreeFock::Rys::RYS_MAX_ROOTS;
    constexpr int kDegree = 13;
    constexpr int kCoeffs = kDegree + 1;
    constexpr double kSegment = 1.0;
    constexpr int kPanels = 60;
    constexpr int kGaussPoints = 20;

    double asymptote_cutoff(int n) { return 35.0 + 7.0 * n; }

    struct GaussLegendre
    {
        std::array<LD, kGaussPoints> x, w;
    };

    const GaussLegendre &gauss_legendre()
    {
        static const GaussLegendre g = []
        {
            GaussLegendre out;
            for (int i = 0; i < kGaussPoints; ++i)
            {
                LD x = std::cos(std::numbers::pi_v<LD> * (i + 0.75L) / (kGaussPoints + 0.5L)), dp = 0;
                for (int it = 0; it < 100; ++it)
                {
                    LD p0 = 1, p1 = x;
                    for (int k = 2; k <= kGaussPoints; ++k)
                    {
                        const LD p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                        p0 = p1, p1 = p2;
                    }
                    dp = kGaussPoints * (x * p1 - p0) / (x * x - 1);
                    const LD dx = p1 / dp;
                    x -= dx;
                    if (std::abs(dx) < 1e-30L)
                        break;
                }
                out.x[i] = x, out.w[i] = 2 / ((1 - x * x) * dp * dp);
            }
            return out;
        }();
        return g;
    }

    // Gauss rule from a three-term recurrence (beta[0] = mu_0, the total
    // mass): roots are the Jacobi matrix's eigenvalues; weights come from
    // the Christoffel function w = 1 / sum_k P_k(x)^2 over the orthonormal
    // polynomials, not from eigenvector components, whose error grows as
    // eps / (eigenvalue gap) when the roots crowd together.
    void gauss_rule(int n, const LD *alpha, const LD *beta, double *roots, double *weights)
    {
        Eigen::Matrix<LD, Eigen::Dynamic, Eigen::Dynamic> J =
            Eigen::Matrix<LD, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, n);
        for (int k = 0; k < n; ++k)
        {
            J(k, k) = alpha[k];
            if (k + 1 < n)
                J(k, k + 1) = J(k + 1, k) = std::sqrt(beta[k + 1]);
        }
        const Eigen::SelfAdjointEigenSolver<decltype(J)> s(J, Eigen::EigenvaluesOnly);
        for (int i = 0; i < n; ++i)
        {
            const LD x = s.eigenvalues()[i];
            LD p_prev = 0.0L, p = 1.0L / std::sqrt(beta[0]), sum = p * p;
            for (int k = 0; k + 1 < n; ++k)
            {
                const LD p_next = ((x - alpha[k]) * p - (k ? std::sqrt(beta[k]) : 0.0L) * p_prev) /
                                  std::sqrt(beta[k + 1]);
                p_prev = p, p = p_next, sum += p * p;
            }
            roots[i] = static_cast<double>(x);
            weights[i] = static_cast<double>(1.0L / sum);
        }
    }

    // Discretized Stieltjes on the measure e^{-T t^2} dt, t in [0, top], in u = t^2.
    void discretized_stieltjes(int n, double T, double *roots, double *weights)
    {
        const GaussLegendre &g = gauss_legendre();
        const LD top = std::min(1.0L, 14.0L / std::sqrt(std::max<LD>(T, 1e-30L)));
        std::vector<LD> u, W;
        u.reserve(kPanels * kGaussPoints), W.reserve(kPanels * kGaussPoints);
        for (int p = 0; p < kPanels; ++p)
        {
            const LD a = top * p / kPanels, b = top * (p + 1) / kPanels;
            const LD mid = 0.5L * (a + b), half = 0.5L * (b - a);
            for (int i = 0; i < kGaussPoints; ++i)
            {
                const LD t = mid + half * g.x[i];
                u.push_back(t * t);
                W.push_back(half * g.w[i] * std::exp(-static_cast<LD>(T) * t * t));
            }
        }
        // Stieltjes with each p_k rescaled to unit norm, so beta_k = ||p_k||^2
        // of the unnormalized step. The inner products are Neumaier-compensated:
        // summed naively over N = 1200 points they carry ~sqrt(N) eps = 1e-14
        // of rounding noise, which the fits would then reproduce.
        const auto add = [](LD &sum, LD &comp, LD x)
        {
            const LD t = sum + x;
            comp += std::abs(sum) >= std::abs(x) ? (sum - t) + x : (x - t) + sum;
            sum = t;
        };
        const std::size_t N = u.size();
        std::vector<LD> pm(N, 0.0L), pk(N, 1.0L), pn(N);
        LD alpha[kNMax], beta[kNMax];
        for (int k = 0; k < n; ++k)
        {
            LD nk = 0, mk = 0, cn = 0, cm = 0;
            for (std::size_t j = 0; j < N; ++j)
            {
                const LD q = W[j] * pk[j] * pk[j];
                add(nk, cn, q), add(mk, cm, u[j] * q);
            }
            nk += cn, mk += cm;
            alpha[k] = mk / nk, beta[k] = nk;
            for (std::size_t j = 0; j < N; ++j)
                pn[j] = (u[j] - alpha[k]) * pk[j] - (k ? beta[k] : 0.0L) * pm[j];
            const LD s = 1 / std::sqrt(nk);
            for (std::size_t j = 0; j < N; ++j)
                pm[j] = pk[j] * s, pk[j] = pn[j] * s;
        }
        gauss_rule(n, alpha, beta, roots, weights);
    }

    // Per n: Laguerre(-1/2) nodes for the asymptote, and the segment fits laid
    // out [segment][coefficient][function] so one Clenshaw pass runs every
    // root and weight together.
    struct Table
    {
        double cutoff = 0;
        int segments = 0;
        std::array<double, kNMax> lag_root{}, lag_weight{};
        std::vector<double> coeff;
    };

    void build(int n, Table &t)
    {
        // Laguerre(alpha = -1/2): a_k = 2k + 1/2, b_k = k (k - 1/2), mass Gamma(1/2).
        LD alpha[kNMax], beta[kNMax];
        for (int k = 0; k < n; ++k)
            alpha[k] = 2 * k + 0.5L, beta[k] = k * (k - 0.5L);
        beta[0] = std::sqrt(std::numbers::pi_v<LD>);
        gauss_rule(n, alpha, beta, t.lag_root.data(), t.lag_weight.data());

        t.cutoff = asymptote_cutoff(n);
        t.segments = static_cast<int>(std::ceil(t.cutoff / kSegment));
        const int nf = 2 * n;
        t.coeff.assign(static_cast<std::size_t>(t.segments) * kCoeffs * nf, 0.0);
        std::vector<double> vals(static_cast<std::size_t>(kCoeffs) * nf);
        double r[kNMax], w[kNMax];
        for (int s = 0; s < t.segments; ++s)
        {
            for (int j = 0; j < kCoeffs; ++j)
            {
                const double x = std::cos(std::numbers::pi * (j + 0.5) / kCoeffs);
                discretized_stieltjes(n, kSegment * (s + 0.5 * (x + 1)), r, w);
                for (int i = 0; i < n; ++i)
                    vals[j * nf + i] = r[i], vals[j * nf + n + i] = w[i];
            }
            double *c = &t.coeff[static_cast<std::size_t>(s) * kCoeffs * nf];
            for (int k = 0; k < kCoeffs; ++k)
                for (int f = 0; f < nf; ++f)
                {
                    double sum = 0.0;
                    for (int j = 0; j < kCoeffs; ++j)
                        sum += vals[j * nf + f] * std::cos(std::numbers::pi * k * (j + 0.5) / kCoeffs);
                    c[k * nf + f] = sum * (k ? 2.0 : 1.0) / kCoeffs;
                }
        }
    }

    const Table &table(int n)
    {
        static std::array<Table, kNMax + 1> tables;
        static std::array<std::once_flag, kNMax + 1> once;
        std::call_once(once[n], [n] { build(n, tables[n]); });
        return tables[n];
    }
}

void HartreeFock::Rys::rys_roots_weights_reference(int n, double T,
                                                   double *__restrict__ roots,
                                                   double *__restrict__ weights) noexcept
{
    discretized_stieltjes(n, T, roots, weights);
}

void HartreeFock::Rys::rys_roots_weights(int n, double T,
                                         double *__restrict__ roots,
                                         double *__restrict__ weights) noexcept
{
    if (n == 1)
    {
        HartreeFock::Rys::rys_1pt(T, roots[0], weights[0]);
        return;
    }
    const Table &t = table(n);
    if (T >= t.cutoff)
    {
        const double inv_T = 1.0 / T, half_inv_sqrt_T = 0.5 / std::sqrt(T);
        for (int i = 0; i < n; ++i)
            roots[i] = t.lag_root[i] * inv_T, weights[i] = t.lag_weight[i] * half_inv_sqrt_T;
        return;
    }
    // Clenshaw for all 2n functions at once.
    const int nf = 2 * n;
    const int s = std::min(t.segments - 1, static_cast<int>(T / kSegment));
    const double x = 2.0 * (T - s * kSegment) / kSegment - 1.0, x2 = 2.0 * x;
    const double *c = &t.coeff[static_cast<std::size_t>(s) * kCoeffs * nf];
    double b1[2 * kNMax] = {}, b2[2 * kNMax] = {};
    for (int k = kDegree; k >= 1; --k)
        for (int f = 0; f < nf; ++f)
        {
            const double b0 = c[k * nf + f] + x2 * b1[f] - b2[f];
            b2[f] = b1[f], b1[f] = b0;
        }
    for (int i = 0; i < n; ++i)
    {
        roots[i] = c[i] + x * b1[i] - b2[i];
        weights[i] = c[n + i] + x * b1[n + i] - b2[n + i];
    }
}
