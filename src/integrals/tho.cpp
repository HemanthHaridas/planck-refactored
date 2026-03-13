#include <cmath>
#include <stdexcept>

#include "tho.h"

// ─── Internal helpers ─────────────────────────────────────────────────────────

// Binomial coefficient C(n, k)
static int _binom(int n, int k)
{
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k; // symmetry
    int result = 1;
    for (int i = 0; i < k; ++i)
    {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

// Integer power: x^n for n >= 0
static double _ipow(double x, int n)
{
    if (n == 0) return 1.0;
    double r = 1.0;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
}

// Double factorial: n!! = n*(n-2)*...  with (-1)!! = 1, 0!! = 1
static double _dfact(int n)
{
    if (n <= -2) return 0.0;
    if (n <=  1) return 1.0;
    double r = 1.0;
    for (int i = n; i > 1; i -= 2) r *= i;
    return r;
}

// Normalised Gaussian moment (no sqrt(pi/zeta) factor):
//   M_norm(n, zeta) = (n-1)!! / (2*zeta)^(n/2)   for even n
//                   = 0                             for odd n
static double _M_norm(int n, double zeta)
{
    if (n % 2 != 0) return 0.0;
    return _dfact(n - 1) / _ipow(2.0 * zeta, n / 2);
}

// ─── Public helpers ───────────────────────────────────────────────────────────

// Full Gaussian moment: M(n, zeta) = M_norm(n, zeta) * sqrt(pi/zeta)
double HartreeFock::Huzinaga::_gaussian_moment(int n, double zeta)
{
    if (n % 2 != 0) return 0.0;
    return _M_norm(n, zeta) * std::sqrt(M_PI / zeta);
}

// 1D overlap via explicit binomial expansion.
// Returns the *reduced* integral (S(0,0) = 1), the prefactor K_AB * (pi/zeta)^1.5
// is already in pp.prefactor — consistent with the OS convention.
//
//   result = sum_{i=0}^{lA} sum_{j=0}^{lB}
//              C(lA,i) * PA^(lA-i) * C(lB,j) * PB^(lB-j) * M_norm(i+j, zeta)
double HartreeFock::Huzinaga::_tho_1d_overlap(int lA, int lB,
                                               double PA, double PB,
                                               double zeta)
{
    double s = 0.0;
    for (int i = 0; i <= lA; ++i)
    {
        const double ci = _binom(lA, i) * _ipow(PA, lA - i);
        for (int j = 0; j <= lB; ++j)
        {
            const int n = i + j;
            if (n % 2 != 0) continue; // M_norm = 0 for odd n
            const double cj = _binom(lB, j) * _ipow(PB, lB - j);
            s += ci * cj * _M_norm(n, zeta);
        }
    }
    return s;
}

// ─── Internal: 1D kinetic integral ────────────────────────────────────────────

// Kinetic 1D contribution from axis q:
//   T_q = beta*(2*lB+1)*S(lA,lB) - 2*beta^2*S(lA,lB+2) - 0.5*lB*(lB-1)*S(lA,lB-2)
static double _tho_1d_kinetic(int lA, int lB, double PA, double PB,
                               double zeta, double beta)
{
    using HartreeFock::Huzinaga::_tho_1d_overlap;

    const double S0  = _tho_1d_overlap(lA, lB,     PA, PB, zeta);
    const double Sp2 = _tho_1d_overlap(lA, lB + 2, PA, PB, zeta);
    const double Sm2 = (lB >= 2) ? _tho_1d_overlap(lA, lB - 2, PA, PB, zeta) : 0.0;

    return beta * (2 * lB + 1) * S0
         - 2.0 * beta * beta * Sp2
         - 0.5 * lB * (lB - 1) * Sm2;
}

// ─── Internal: 3D primitive overlap + kinetic ─────────────────────────────────

static std::pair<double, double> _tho_primitive_ST(
    const HartreeFock::PrimitivePair& pp,
    int lAx, int lAy, int lAz,
    int lBx, int lBy, int lBz)
{
    using HartreeFock::Huzinaga::_tho_1d_overlap;

    const double zeta = pp.zeta;
    const double beta = pp.beta;
    const double scale = pp.prefactor * pp.coeff_product;

    const double PAx = pp.pA[0], PAy = pp.pA[1], PAz = pp.pA[2];
    const double PBx = pp.pB[0], PBy = pp.pB[1], PBz = pp.pB[2];

    // 1D overlaps
    const double Sx = _tho_1d_overlap(lAx, lBx, PAx, PBx, zeta);
    const double Sy = _tho_1d_overlap(lAy, lBy, PAy, PBy, zeta);
    const double Sz = _tho_1d_overlap(lAz, lBz, PAz, PBz, zeta);

    // 1D kinetic contributions
    const double Tx = _tho_1d_kinetic(lAx, lBx, PAx, PBx, zeta, beta);
    const double Ty = _tho_1d_kinetic(lAy, lBy, PAy, PBy, zeta, beta);
    const double Tz = _tho_1d_kinetic(lAz, lBz, PAz, PBz, zeta, beta);

    const double S = scale * Sx * Sy * Sz;
    const double T = scale * (Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz);

    return {S, T};
}

// ─── Phase 1 + 2: 1-electron integrals ───────────────────────────────────────

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
HartreeFock::Huzinaga::_compute_1e(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    std::size_t nbasis)
{
    const std::size_t npairs = shell_pairs.size();
    Eigen::MatrixXd overlap = Eigen::MatrixXd::Zero(nbasis, nbasis);
    Eigen::MatrixXd kinetic = Eigen::MatrixXd::Zero(nbasis, nbasis);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t p = 0; p < npairs; ++p)
    {
        const HartreeFock::ShellPair& sp = shell_pairs[p];
        const std::size_t ii = sp.A._index;
        const std::size_t jj = sp.B._index;

        const int lAx = sp.A._cartesian[0];
        const int lAy = sp.A._cartesian[1];
        const int lAz = sp.A._cartesian[2];
        const int lBx = sp.B._cartesian[0];
        const int lBy = sp.B._cartesian[1];
        const int lBz = sp.B._cartesian[2];

        double S = 0.0, T = 0.0;
        for (const auto& pp : sp.primitive_pairs)
        {
            auto [s, t] = _tho_primitive_ST(pp, lAx, lAy, lAz, lBx, lBy, lBz);
            S += s;
            T += t;
        }

#pragma omp critical
        {
            overlap(ii, jj) = S;
            overlap(jj, ii) = S;
            kinetic(ii, jj) = T;
            kinetic(jj, ii) = T;
        }
    }

    return {overlap, kinetic};
}

// ─── Phase 3 stub: nuclear attraction ────────────────────────────────────────

Eigen::MatrixXd
HartreeFock::Huzinaga::_compute_nuclear_attraction(
    const std::vector<HartreeFock::ShellPair>& /*shell_pairs*/,
    std::size_t /*nbasis*/,
    const HartreeFock::Molecule& /*molecule*/)
{
    throw std::runtime_error(
        "THO nuclear attraction not yet implemented (Phase 3)");
}

// ─── Phase 4 stub: 2e Fock ───────────────────────────────────────────────────

Eigen::MatrixXd
HartreeFock::Huzinaga::_compute_2e_fock(
    const std::vector<HartreeFock::ShellPair>& /*shell_pairs*/,
    const Eigen::MatrixXd& /*density*/,
    std::size_t /*nbasis*/)
{
    throw std::runtime_error(
        "THO two-electron Fock not yet implemented (Phase 4)");
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
HartreeFock::Huzinaga::_compute_2e_fock_uhf(
    const std::vector<HartreeFock::ShellPair>& /*shell_pairs*/,
    const Eigen::MatrixXd& /*Pa*/,
    const Eigen::MatrixXd& /*Pb*/,
    std::size_t /*nbasis*/)
{
    throw std::runtime_error(
        "THO UHF two-electron Fock not yet implemented (Phase 4)");
}
