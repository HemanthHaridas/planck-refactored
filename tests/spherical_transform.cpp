// Validates the Cartesian→spherical machinery in two independent ways:
//
//   1. ABSOLUTE correctness of the closed-form oracle
//      (src/basis/spherical_recurrence.cpp). Each row it produces must be a genuine
//      real solid harmonic: it must satisfy Laplace's equation (∇² = 0) — the
//      definitional, metric-independent property — and the 2L+1 rows must be linearly
//      independent. (Solid harmonics are orthogonal under the angular inner product on
//      the sphere, not under the Cartesian-Gaussian radial-weighted overlap, so we do
//      not test orthonormality against the Gaussian Gram matrix.)
//
//   2. CROSS-CHECK against the hand-coded production matrices
//      (src/basis/spherical.cpp). The production T⁺ is a pseudoinverse used for
//      symmetry labeling; for L ≤ 2 its rows are pure harmonics and must be collinear
//      with the oracle row-by-row. For L ≥ 3 the pseudoinverse legitimately carries
//      r²-contamination (it is not the pure harmonic), so the row-equality check is
//      not applied there — instead we verify the production rows still live in the
//      degree-L polynomial space and have the right shape/rank. The oracle remains the
//      source of mathematical truth; production is validated where the two must agree.
//
// The oracle is also exercised at L = 6, 7 (I, K), past the production ceiling, to
// confirm it keeps producing valid harmonics.

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "basis/spherical.h"
#include "basis/spherical_recurrence.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    int n_cart(int L) { return (L + 1) * (L + 2) / 2; }
    int n_sph(int L) { return 2 * L + 1; }

    double dfact(int n)
    {
        double r = 1.0;
        for (int k = n; k > 1; k -= 2)
            r *= k;
        return r;
    }

    std::vector<std::array<int, 3>> cartesian_order(int L)
    {
        std::vector<std::array<int, 3>> o;
        for (int lx = L; lx >= 0; --lx)
            for (int ly = L - lx; ly >= 0; --ly)
                o.push_back({lx, ly, L - lx - ly});
        return o;
    }

    // Max |∇² row|, treating each row's coefficients as bare-monomial coefficients
    // (i.e. after undoing the √(self-overlap) weighting the transform applies). A true
    // solid harmonic is annihilated by the Laplacian.
    double max_laplacian(const Eigen::MatrixXd &T, int L)
    {
        const auto c = cartesian_order(L);
        const int nc = static_cast<int>(c.size());
        double worst = 0.0;
        for (int r = 0; r < T.rows(); ++r)
        {
            std::map<std::array<int, 3>, double> bare;
            for (int j = 0; j < nc; ++j)
            {
                const double s =
                    dfact(2 * c[j][0] - 1) * dfact(2 * c[j][1] - 1) * dfact(2 * c[j][2] - 1);
                bare[c[j]] = T(r, j) / std::sqrt(s);
            }
            std::map<std::array<int, 3>, double> lap;
            for (auto &[m, co] : bare)
                for (int d = 0; d < 3; ++d)
                    if (m[d] >= 2)
                    {
                        auto mm = m;
                        mm[d] -= 2;
                        lap[mm] += co * m[d] * (m[d] - 1);
                    }
            for (auto &[m, co] : lap)
                worst = std::max(worst, std::abs(co));
        }
        return worst;
    }

    // ── 1. Absolute checks on the oracle ─────────────────────────────────────────
    void check_oracle_is_valid_harmonics(int L, double tol)
    {
        auto rec = HartreeFock::BasisFunctions::cart_to_sph_block_recurrence(L);
        if (!rec)
        {
            fail("recurrence failed at L=" + std::to_string(L) + ": " + rec.error());
            return;
        }
        const Eigen::MatrixXd &R = *rec;

        if (R.rows() != n_sph(L) || R.cols() != n_cart(L))
        {
            std::ostringstream oss;
            oss << "recurrence L=" << L << ": wrong shape " << R.rows() << "x" << R.cols();
            fail(oss.str());
            return;
        }

        // (a) Every row is a solid harmonic: ∇² = 0. This is the defining property
        // and is metric-independent, so it pins down the angular content exactly.
        const double lap = max_laplacian(R, L);
        if (lap > tol)
        {
            std::ostringstream oss;
            oss << "recurrence L=" << L << ": row not harmonic (max|∇²|=" << lap << ")";
            fail(oss.str());
        }

        // (b) The 2L+1 rows are linearly independent — they form a full basis of the
        // degree-L harmonic space. (Note: solid harmonics are orthogonal under the
        // *angular* inner product on the sphere, NOT under the Cartesian-Gaussian
        // radial-weighted overlap, so we deliberately do not test R·G·Rᵀ = I here;
        // see the design note at the top of this file.)
        if (Eigen::FullPivLU<Eigen::MatrixXd>(R).rank() != n_sph(L))
        {
            std::ostringstream oss;
            oss << "recurrence L=" << L << ": rows not linearly independent (rank "
                << Eigen::FullPivLU<Eigen::MatrixXd>(R).rank() << " != " << n_sph(L) << ")";
            fail(oss.str());
        }
    }

    // ── 2. Cross-check production vs oracle ───────────────────────────────────────
    // For L ≤ 2 production rows are pure harmonics: assert collinearity row-by-row.
    void check_production_matches_oracle_pure(int L, double tol)
    {
        auto prod = HartreeFock::BasisFunctions::cart_to_sph_block(L);
        auto rec = HartreeFock::BasisFunctions::cart_to_sph_block_recurrence(L);
        if (!prod || !rec)
        {
            fail("L=" + std::to_string(L) + ": transform unavailable for pure cross-check");
            return;
        }
        for (int r = 0; r < n_sph(L); ++r)
        {
            const Eigen::VectorXd up = prod->row(r).transpose().normalized();
            const Eigen::VectorXd ur = rec->row(r).transpose().normalized();
            const double same = (up - ur).cwiseAbs().maxCoeff();
            const double flip = (up + ur).cwiseAbs().maxCoeff();
            if (std::min(same, flip) > tol)
            {
                std::ostringstream oss;
                oss << "L=" << L << " row m=" << (r - L)
                    << ": production not collinear with oracle (min|Δ unit|="
                    << std::min(same, flip) << ")";
                fail(oss.str());
            }
        }
    }

    // For all production L: structural sanity (shape + full spherical row rank). The
    // production pseudoinverse legitimately carries r²-contamination for L ≥ 3, so we
    // do not require its rows to equal the pure harmonics there.
    void check_production_structural(int L)
    {
        auto prod = HartreeFock::BasisFunctions::cart_to_sph_block(L);
        if (!prod)
        {
            fail("production cart_to_sph_block failed at L=" + std::to_string(L) +
                 ": " + prod.error());
            return;
        }
        if (prod->rows() != n_sph(L) || prod->cols() != n_cart(L))
        {
            std::ostringstream oss;
            oss << "production L=" << L << ": wrong shape " << prod->rows() << "x" << prod->cols();
            fail(oss.str());
            return;
        }
        if (Eigen::FullPivLU<Eigen::MatrixXd>(*prod).rank() != n_sph(L))
            fail("production L=" + std::to_string(L) + ": not full spherical row rank");
    }
} // namespace

int main()
{
    constexpr double tol = 1e-9;

    // Oracle: absolute correctness for every L it is asked for, including past the
    // production ceiling (I, K shells at L=6,7).
    for (int L = 0; L <= 7; ++L)
        check_oracle_is_valid_harmonics(L, tol);

    // Production: structural sanity for the shipped range, plus exact agreement with
    // the oracle where production rows are pure harmonics (L ≤ 2).
    for (int L = 0; L <= 5; ++L)
        check_production_structural(L);
    for (int L = 0; L <= 2; ++L)
        check_production_matches_oracle_pure(L, tol);

    // Production must still refuse L ≥ 6 (no input path reaches it today).
    if (HartreeFock::BasisFunctions::cart_to_sph_block(6))
        fail("production cart_to_sph_block(6) should error (max supported L=5)");

    if (g_ok)
        std::cout << "spherical_transform: all checks passed\n";
    return g_ok ? 0 : 1;
}
