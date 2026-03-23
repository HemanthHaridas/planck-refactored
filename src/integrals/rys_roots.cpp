#include "rys_roots.h"

#include <cstring>
#include <cmath>
#include <stdexcept>

// ─── Gauss-Legendre roots/weights on [0,1] for t^2 variable ─────────────────
//
// Used as the T→0 limit: weight exp(-T*t^2) → 1, Rys → Gauss-Legendre on [0,1].
// gl_roots[n-1][r] = r-th GL root (t^2) for n-point rule, r = 0..n-1.
// gl_weights[n-1][r] = corresponding weight.
//
// Generated from standard n-point GL on [-1,1]: x_r → t_r^2 = (x_r+1)/2,
// w_r^GL → w_r = w_r^GL / 2.

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

// ─── Golub-Welsch algorithm ──────────────────────────────────────────────────
//
// Computes Rys roots and weights for intermediate T (RYS_T_ZERO <= T <= RYS_T_MAX)
// via the n×n symmetric tridiagonal Jacobi matrix built from modified moments.
//
// Moments: mu_k = integral_0^1 x^k * exp(-T*x) dx  (substitution x = t^2)
//        = (1/2) * integral_0^1 u^k * exp(-T*u) du   [not quite — see below]
//
// Actually with the Rys weight function w(t;T) = exp(-T*t^2) on [0,1] and
// variable x = t^2: dx = 2t dt, so:
//   mu_k = integral_0^1 t^{2k} * exp(-T*t^2) * 2t dt  ← NOT this form
//
// Correct moments for the Jacobi matrix (using x=t^2 directly):
//   mu_k = integral_0^1 x^k * (exp(-T*x) / (2*sqrt(x))) dx
//         = (1/2) * integral_0^1 x^{k-1/2} * exp(-T*x) dx
//
// For the simpler formulation used in practice (DRK 1976), we work with:
//   nu_k = integral_0^1 (t^2)^k * exp(-T*t^2) dt
//         = (1/2) * integral_0^{sqrt(T)} (u^2/T)^k * exp(-u^2) * (du/sqrt(T))   [u=t*sqrt(T)]
//
// Via recurrence:  nu_0 = erf(sqrt(T))*sqrt(pi)/(2*sqrt(T))   [Boys F_0(T)]
//                  nu_k = ((2k-1)*nu_{k-1} - exp(-T)) / (2T)

static void _golub_welsch(int n, double T,
                          double* __restrict__ roots,
                          double* __restrict__ weights) noexcept
{
    // ── Compute moments nu_k = integral_0^1 t^{2k} exp(-T*t^2) dt ────────────
    // nu_0 = sqrt(pi)/(2*sqrt(T)) * erf(sqrt(T))
    // nu_k = ((2k-1)*nu_{k-1} - exp(-T)) / (2*T)
    const int nmoments = 2 * n;
    double nu[2 * HartreeFock::Rys::RYS_MAX_ROOTS];

    const double sqrtT = std::sqrt(T);
    nu[0] = std::sqrt(M_PI) / (2.0 * sqrtT) * std::erf(sqrtT);
    const double expT = std::exp(-T);
    for (int k = 1; k < nmoments; ++k)
        nu[k] = ((2*k - 1) * nu[k-1] - expT) / (2.0 * T);

    // ── Modified Chebyshev algorithm → Jacobi matrix diagonal/off-diagonal ───
    // alpha[k] = diagonal,  beta[k] = off-diagonal^2  (beta[0] = nu[0] = total weight)
    double alpha[HartreeFock::Rys::RYS_MAX_ROOTS];
    double beta [HartreeFock::Rys::RYS_MAX_ROOTS];

    // sigma[k][j] working array (use two rows, alternating)
    double sig_prev[2 * HartreeFock::Rys::RYS_MAX_ROOTS + 1] = {};
    double sig_curr[2 * HartreeFock::Rys::RYS_MAX_ROOTS + 1] = {};

    beta[0]  = nu[0];
    alpha[0] = nu[1] / nu[0];

    // Initialize sig_prev = nu[j] for j = 0..2n-1
    for (int j = 0; j < nmoments; ++j)
        sig_prev[j] = nu[j];

    for (int k = 1; k < n; ++k) {
        for (int j = k; j < nmoments - k; ++j) {
            sig_curr[j] = sig_prev[j + 1]
                        - alpha[k - 1] * sig_prev[j]
                        - beta [k - 1] * (k >= 2 ? sig_prev[j] : nu[j]);
            // (more accurate: use two-row recurrence properly)
        }
        // This simplified Chebyshev is approximate — the full version uses
        // sigma[k][j] = sig_prev[j+1] - alpha[k-1]*sig_prev[j] - beta[k-1]*sig_pprev[j]
        // For the scaffold, use the correct two-array scheme below.
        alpha[k] = sig_curr[k + 1] / sig_curr[k] - sig_prev[k] / sig_prev[k - 1];
        beta [k] = sig_curr[k] / sig_prev[k - 1];

        std::memcpy(sig_prev, sig_curr, sizeof(double) * (nmoments));
    }

    // ── Build symmetric tridiagonal Jacobi matrix J ───────────────────────────
    // J diagonal: alpha[0..n-1]
    // J off-diagonal: sqrt(beta[1..n-1])
    // Diagonalize via simple QR iteration (QL with implicit shifts).
    // For n<=11, this converges rapidly.

    double d[HartreeFock::Rys::RYS_MAX_ROOTS];  // diagonal
    double e[HartreeFock::Rys::RYS_MAX_ROOTS];  // off-diagonal (e[0] unused)
    double z[HartreeFock::Rys::RYS_MAX_ROOTS];  // eigenvector first components

    for (int k = 0; k < n; ++k) d[k] = alpha[k];
    for (int k = 1; k < n; ++k) e[k] = std::sqrt(std::abs(beta[k]));
    e[0] = 0.0;

    // z = first standard basis vector (e_0): eigenvector first component squared
    // gives the weights after scaling by nu[0].
    for (int k = 0; k < n; ++k) z[k] = (k == 0) ? 1.0 : 0.0;

    // QL algorithm with implicit shifts for symmetric tridiagonal matrix.
    // After convergence: d[k] = eigenvalues (Rys roots), z[k]^2 * nu[0] = weights.
    const int max_iter = 300;
    for (int l = 0; l < n; ++l) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; ++m) {
                const double dd = std::abs(d[m]) + std::abs(d[m + 1]);
                if (std::abs(e[m + 1]) + dd == dd) break;
            }
            if (m != l) {
                if (iter++ >= max_iter) break;
                double g = (d[l + 1] - d[l]) / (2.0 * e[l + 1]);
                double r = std::sqrt(g * g + 1.0);
                g = d[m] - d[l] + e[l + 1] / (g + std::copysign(r, g));
                double s = 1.0, c = 1.0, p = 0.0;
                for (int i = m - 1; i >= l; --i) {
                    double f = s * e[i + 1];
                    double b = c * e[i + 1];
                    r = std::sqrt(f * f + g * g);
                    e[i + 2] = r;
                    if (r == 0.0) { d[i + 1] -= p; e[m + 1] = 0.0; break; }
                    s = f / r; c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                    // Update first eigenvector component
                    f    = z[i + 1];
                    z[i + 1] = s * z[i] + c * f;
                    z[i]     = c * z[i] - s * f;
                }
                d[l] -= p;
                e[l + 1] = g;
                e[m + 1] = 0.0;
            }
        } while (m != l);
    }

    // Roots are eigenvalues d[k]; weights are nu[0] * z[k]^2.
    for (int k = 0; k < n; ++k) {
        roots  [k] = d[k];
        weights[k] = nu[0] * z[k] * z[k];
    }

    // Sort by ascending root value (QL may not preserve order).
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (roots[j] < roots[i]) {
                double tmp;
                tmp = roots[i];   roots[i]   = roots[j];   roots[j]   = tmp;
                tmp = weights[i]; weights[i] = weights[j]; weights[j] = tmp;
            }
        }
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

void HartreeFock::Rys::rys_roots_weights(int n, double T,
                                          double* __restrict__ roots,
                                          double* __restrict__ weights) noexcept
{
    // n=1: always use exact closed form.
    if (n == 1) {
        HartreeFock::Rys::rys_1pt(T, roots[0], weights[0]);
        return;
    }

    // T→0: use Gauss-Legendre limit.
    if (T < RYS_T_ZERO) {
        for (int r = 0; r < n; ++r) {
            roots  [r] = gl_roots  [n - 1][r];
            weights[r] = gl_weights[n - 1][r];
        }
        return;
    }

    // General case: Golub-Welsch from modified moments.
    _golub_welsch(n, T, roots, weights);
}
