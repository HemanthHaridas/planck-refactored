#include "rys_roots.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>

#include <Eigen/Eigenvalues>

// ─── Numerical fallback nodes/weights on [0,1] ───────────────────────────────
//
// These tables are retained only as a last-resort safety net if the Jacobi
// build fails. The normal small-T path goes through exact moments in
// _boys_moment() and should not use these values.

static const double gl_roots[HartreeFock::Rys::RYS_MAX_ROOTS]
                            [HartreeFock::Rys::RYS_MAX_ROOTS] = {
                                // n=1
                                {0.5},
                                // n=2
                                {0.21132486540518713, 0.78867513459481287},
                                // n=3
                                {0.11270166537925831, 0.5, 0.88729833462074169},
                                // n=4
                                {0.069431844202973713, 0.33000947820757187, 0.66999052179242813, 0.93056815579702629},
                                // n=5
                                {0.046910077030668004, 0.23076534494715846, 0.5, 0.76923465505284154, 0.95308992296933200},
                                // n=6
                                {0.033765242898423990, 0.16939530676686775, 0.38069040695840156,
                                 0.61930959304159844, 0.83060469323313225, 0.96623475710157601},
                                // n=7
                                {0.025447292528756196, 0.12923440720030278, 0.29707742431130141, 0.5,
                                 0.70292257568869859, 0.87076559279969722, 0.97455270747124380},
                                // n=8
                                {0.019855071751231884, 0.10166676129318664, 0.23723379504183550, 0.40828267875217510,
                                 0.59171732124782490, 0.76276620495816450, 0.89833323870681336, 0.98014492824876812},
                                // n=9
                                {0.015919880246186956, 0.081984446336682102, 0.19331428364970481, 0.33787328829809554,
                                 0.5, 0.66212671170190446, 0.80668571635029519, 0.91801555366331790,
                                 0.98408011975381304},
                                // n=10
                                {0.013046735741414139, 0.067234002440879327, 0.16029521585048779, 0.28330230293537640,
                                 0.42556283050918437, 0.57443716949081563, 0.71669769706462360, 0.83970478414951221,
                                 0.93276599755912067, 0.98695326425858586},
                                // n=11
                                {0.010885670926971503, 0.056468700115952352, 0.13492399721297531, 0.24045193539659023,
                                 0.36522842202982773, 0.5, 0.63477157797017227, 0.75954806460340977,
                                 0.86507600278702469, 0.94353129988404765, 0.98911432907302850},
};

static const double gl_weights[HartreeFock::Rys::RYS_MAX_ROOTS]
                              [HartreeFock::Rys::RYS_MAX_ROOTS] = {
                                  // n=1
                                  {1.0},
                                  // n=2
                                  {0.5, 0.5},
                                  // n=3
                                  {0.27777777777777778, 0.44444444444444444, 0.27777777777777778},
                                  // n=4
                                  {0.17392742256872693, 0.32607257743127307, 0.32607257743127307, 0.17392742256872693},
                                  // n=5
                                  {0.11846344252809454, 0.23931433524968324, 0.28444444444444444, 0.23931433524968324,
                                   0.11846344252809454},
                                  // n=6
                                  {0.085662246189585178, 0.18038078652406930, 0.23395696728634552,
                                   0.23395696728634552, 0.18038078652406930, 0.085662246189585178},
                                  // n=7
                                  {0.064742483084434847, 0.13985269574463833, 0.19091502525255947, 0.20897959183673469,
                                   0.19091502525255947, 0.13985269574463833, 0.064742483084434847},
                                  // n=8
                                  {0.050614268145188129, 0.11111905172522298, 0.15685332293894364, 0.18134189168918099,
                                   0.18134189168918099, 0.15685332293894364, 0.11111905172522298, 0.050614268145188129},
                                  // n=9
                                  {0.040637194180787205, 0.090324080347428702, 0.13030534820146773, 0.15617353852000142,
                                   0.16511967750062988, 0.15617353852000142, 0.13030534820146773, 0.090324080347428702,
                                   0.040637194180787205},
                                  // n=10
                                  {0.033335672154344069, 0.074725674575290296, 0.10954318125799103, 0.13463335965499817,
                                   0.14776211235737643, 0.14776211235737643, 0.13463335965499817, 0.10954318125799103,
                                   0.074725674575290296, 0.033335672154344069},
                                  // n=11
                                  {0.027834283558086833, 0.062790184732452390, 0.093145105463867014, 0.11659688229599524,
                                   0.13140227225512332, 0.13646254338895032, 0.13140227225512332, 0.11659688229599524,
                                   0.093145105463867014, 0.062790184732452390, 0.027834283558086833},
};

static long double _boys_moment(int m, long double T) noexcept
{
    if (T < static_cast<long double>(HartreeFock::Rys::RYS_T_ZERO))
        return 1.0L / static_cast<long double>(2 * m + 1);

    // The upward Boys recursion loses many digits for small-but-nonzero T and
    // can drive the moment sequence negative enough to break the Jacobi build.
    // Use the convergent power series there instead:
    //   F_m(T) = sum_{n>=0} (-T)^n / (n! (2m + 2n + 1)).
    if (T < 1.0L)
    {
        long double sum = 0.0L;
        long double coeff = 1.0L;
        for (int n = 0; n < 256; ++n)
        {
            const long double term =
                coeff / static_cast<long double>(2 * m + 2 * n + 1);
            sum += term;
            if (std::abs(term) < 1.0e-28L * std::abs(sum))
                break;
            coeff *= -T / static_cast<long double>(n + 1);
        }
        return sum;
    }

    const long double sqrtT = std::sqrt(T);
    long double F = 0.5L * std::sqrt(std::numbers::pi_v<long double> / T) * std::erfl(sqrtT);
    if (m == 0)
        return F;

    const long double eT = std::exp(-T);
    for (int k = 1; k <= m; ++k)
        F = ((2 * k - 1) * F - eT) / (2.0L * T);
    return F;
}

static long double _poly_inner(const long double *a, int da,
                               const long double *b, int db,
                               const long double *moments) noexcept
{
    long double s = 0.0L;
    for (int i = 0; i <= da; ++i)
        for (int j = 0; j <= db; ++j)
            s += a[i] * b[j] * moments[i + j];
    return s;
}

static void _stieltjes_jacobi(int n, double T,
                              double *__restrict__ roots,
                              double *__restrict__ weights) noexcept
{
    constexpr int max_n = HartreeFock::Rys::RYS_MAX_ROOTS;
    long double moments[2 * max_n + 1] = {};
    for (int k = 0; k <= 2 * n; ++k)
        moments[k] = _boys_moment(k, static_cast<long double>(T));

    long double polys[max_n + 1][max_n + 1] = {};
    long double alphas[max_n] = {};
    long double betas[max_n] = {};

    polys[0][0] = 1.0L / std::sqrt(moments[0]);

    for (int k = 0; k < n; ++k)
    {
        long double xpk[max_n + 1] = {};
        for (int i = 0; i <= k; ++i)
            xpk[i + 1] = polys[k][i];

        alphas[k] = _poly_inner(xpk, k + 1, polys[k], k, moments);

        if (k == n - 1)
            break;

        long double q[max_n + 1] = {};
        for (int i = 0; i <= k + 1; ++i)
            q[i] += xpk[i];
        for (int i = 0; i <= k; ++i)
            q[i] -= alphas[k] * polys[k][i];
        if (k > 0)
            for (int i = 0; i <= k - 1; ++i)
                q[i] -= betas[k - 1] * polys[k - 1][i];

        long double norm2 = _poly_inner(q, k + 1, q, k + 1, moments);
        if (norm2 < 0.0L && std::abs(norm2) < 1.0e-24L)
            norm2 = 0.0L;
        betas[k] = std::sqrt(norm2);
        if (!(betas[k] > 0.0L))
        {
            for (int r = 0; r < n; ++r)
            {
                roots[r] = gl_roots[n - 1][r];
                weights[r] = gl_weights[n - 1][r];
            }
            return;
        }

        for (int i = 0; i <= k + 1; ++i)
            polys[k + 1][i] = q[i] / betas[k];
    }

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i)
    {
        J(i, i) = static_cast<double>(alphas[i]);
        if (i + 1 < n)
        {
            const double b = static_cast<double>(betas[i]);
            J(i, i + 1) = b;
            J(i + 1, i) = b;
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(J);
    if (solver.info() != Eigen::Success)
    {
        for (int r = 0; r < n; ++r)
        {
            roots[r] = gl_roots[n - 1][r];
            weights[r] = gl_weights[n - 1][r];
        }
        return;
    }

    const auto &evals = solver.eigenvalues();
    const auto &evecs = solver.eigenvectors();
    const double m0 = static_cast<double>(moments[0]);
    for (int i = 0; i < n; ++i)
    {
        roots[i] = std::clamp(evals[i], 0.0, 1.0);
        weights[i] = std::max(0.0, m0 * evecs(0, i) * evecs(0, i));
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

void HartreeFock::Rys::rys_roots_weights(int n, double T,
                                         double *__restrict__ roots,
                                         double *__restrict__ weights) noexcept
{
    // n=1: always use exact closed form.
    if (n == 1)
    {
        HartreeFock::Rys::rys_1pt(T, roots[0], weights[0]);
        return;
    }

    _stieltjes_jacobi(n, T, roots, weights);
}
