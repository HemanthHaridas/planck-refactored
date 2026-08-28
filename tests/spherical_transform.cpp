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
//      (src/basis/spherical.cpp). For L ≤ 2 the production rows are collinear with the
//      oracle row-by-row. For L ≥ 3 they are a DIFFERENT VALID BASIS of the same space
//      (raw matrices differ by ~0.75, an m-ordering/scaling convention), so row-equality
//      is not applied there — but two convention-invariant properties are, and they are
//      the energy-relevant ones (FU3, section 3 below): the rows must be pure harmonics
//      in the physical (unit-normalized-component) basis, and they must span the same
//      subspace as the oracle. Those two are what the L ≥ 3 normalization defect
//      violated; the older shape/rank check passed throughout while the energy was
//      2.14e-5 wrong for water/cc-pVTZ.
//
//      Historical note: this file previously stated that the L ≥ 3 pseudoinverse
//      "legitimately carries r²-contamination". That described the DEFECT, not a
//      property of the transform — normalized_pseudoinverse removed it, and the
//      checks below now pin its absence.
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

    // ── 3. Energy-relevant checks on the PRODUCTION transform (FU3) ──────────────
    //
    // These are the two checks whose absence let the L ≥ 3 normalization defect ship
    // (docs/SPHERICAL_F_SHELL_ACCURACY_SCOPE.md). Both are convention-invariant, so
    // they do not false-fail on the intra-span basis/m-ordering difference between
    // production and oracle — whose raw matrices legitimately differ by ~0.75.
    //
    // (a) Harmonic purity in the PHYSICAL monomials. max_laplacian undoes the
    //     unit-normalization weighting before differentiating, so it asks the
    //     energy-relevant question: are the functions the integral engine actually
    //     builds pure ℓ-harmonics? Pre-fix this read 2.2e-1 (L=3), 1.8e-1 (L=4) and
    //     8.4e-2 (L=5), and 0 at L ≤ 2 and L = 6 — exactly the shells the fix touched.
    void check_production_is_harmonic_in_physical_monomials(int L, double tol)
    {
        auto prod = HartreeFock::BasisFunctions::cart_to_sph_block(L);
        if (!prod)
        {
            fail("production cart_to_sph_block failed at L=" + std::to_string(L));
            return;
        }
        const double lap = max_laplacian(*prod, L);
        if (lap > tol)
        {
            std::ostringstream oss;
            oss << "production L=" << L
                << ": rows are not pure harmonics in the physical (unit-normalized"
                   " component) basis (max|∇²|=" << lap
                << ") — the L>=3 normalization defect";
            fail(oss.str());
        }
    }

    // (b) Span equality with the oracle. The row-space projector is invariant under
    //     any change of basis WITHIN the span, so it compares the only thing that is
    //     energy-relevant: which subspace the transform selects. Do NOT compare the
    //     matrices element-wise — they are different valid bases of the same space.
    //     Pre-fix this read ~3e-1 for L = 3,4,5.
    void check_production_spans_oracle_space(int L, double tol)
    {
        auto prod = HartreeFock::BasisFunctions::cart_to_sph_block(L);
        auto rec = HartreeFock::BasisFunctions::cart_to_sph_block_recurrence(L);
        if (!prod || !rec)
        {
            fail("L=" + std::to_string(L) + ": transform unavailable for span check");
            return;
        }
        // Row-space projector P = U Uᵀ with U an orthonormal basis of the rows.
        auto row_projector = [](const Eigen::MatrixXd &M) {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(M.transpose(), Eigen::ComputeThinU);
            return Eigen::MatrixXd(svd.matrixU() * svd.matrixU().transpose());
        };
        const double diff =
            (row_projector(*prod) - row_projector(*rec)).cwiseAbs().maxCoeff();
        if (diff > tol)
        {
            std::ostringstream oss;
            oss << "production L=" << L
                << ": row space differs from the oracle's (max|ΔP|=" << diff
                << ") — production selects a different ℓ-subspace";
            fail(oss.str());
        }
    }

    // For all production L: structural sanity (shape + full spherical row rank).
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

    // Production: structural sanity for the shipped range (now through L = 6, I
    // shells), plus exact agreement with the oracle where production rows are pure
    // harmonics (L ≤ 2).
    for (int L = 0; L <= 6; ++L)
        check_production_structural(L);
    for (int L = 0; L <= 2; ++L)
        check_production_matches_oracle_pure(L, tol);

    // FU3: the energy-relevant checks. Post-fix these hold for the whole production
    // range, INCLUDING L ≥ 3 — the pseudoinverse no longer carries r²-contamination
    // once normalized_pseudoinverse scales each Cartesian row. Both are falsifiable:
    // re-introducing the defect (dropping that row scaling) makes L = 3,4,5 fail with
    // max|∇²| ~ 1e-1 and max|ΔP| ~ 3e-1 while L ≤ 2 and L = 6 stay green.
    for (int L = 0; L <= 6; ++L)
    {
        check_production_is_harmonic_in_physical_monomials(L, tol);
        check_production_spans_oracle_space(L, tol);
    }

    // L = 6 production delegates to the recurrence oracle, so the two must be
    // byte-for-byte identical (not merely collinear).
    {
        auto prod = HartreeFock::BasisFunctions::cart_to_sph_block(6);
        auto rec = HartreeFock::BasisFunctions::cart_to_sph_block_recurrence(6);
        if (!prod || !rec)
            fail("L=6: production or oracle transform failed to build");
        else if ((*prod - *rec).cwiseAbs().maxCoeff() > tol)
            fail("L=6: production (delegated) does not match the recurrence oracle");
    }

    // Production must still refuse L ≥ 7 (MAX_L = 6 bounds the integral buffers).
    if (HartreeFock::BasisFunctions::cart_to_sph_block(7))
        fail("production cart_to_sph_block(7) should error (max supported L=6)");

    if (g_ok)
        std::cout << "spherical_transform: all checks passed\n";
    return g_ok ? 0 : 1;
}
