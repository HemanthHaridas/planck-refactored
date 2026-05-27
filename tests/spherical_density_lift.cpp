// Verifies HartreeFock::BasisFunctions::lift_density_sph_to_cart on the one
// property the gradient skin relies on: energy invariance under the basis change.
//
// The lift is M_cart = Cᵀ · M_sph · C, where C : n_sph × n_cart is the
// block-diagonal Cartesian→spherical transform. For any Cartesian operator X_cart
// (one of the derivative integral blocks emitted by the integral engine) with
// spherical lowering X_sph = C · X_cart · Cᵀ:
//
//     tr(M_sph · X_sph) == tr(M_cart · X_cart).
//
// That is the identity gradient kernels need: they compute the right-hand side
// from Cartesian derivative blocks and Cartesian-lifted densities, and it must
// equal the energy expectation in the spherical basis. This test pins it.
//
// We exercise it at L = 0, 1, 2, 3 so the r²-contamination dropping (L ≥ 2) is
// covered. We also sanity-check that the lift preserves symmetry, since the
// SCF density and W are both symmetric and downstream code expects to keep that.

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#include "basis/spherical.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    // Build a deterministic symmetric matrix of given size. Seeded so the same
    // matrix is reproduced on every run — no RNG, no hidden state.
    Eigen::MatrixXd make_symmetric(Eigen::Index n, double seed)
    {
        Eigen::MatrixXd A(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                A(i, j) = std::sin(seed + 0.31 * i + 0.71 * j) + std::cos(seed - 1.13 * i + 0.27 * j);
        return 0.5 * (A + A.transpose());
    }

    void check_for_L(int L)
    {
        auto C_res = HartreeFock::BasisFunctions::cart_to_sph_block(L);
        if (!C_res)
        {
            fail("cart_to_sph_block(L=" + std::to_string(L) + ") failed: " + C_res.error());
            return;
        }
        const Eigen::MatrixXd C = *C_res;
        const Eigen::Index n_sph = C.rows();
        const Eigen::Index n_cart = C.cols();

        // M_sph plays the role of a density-like quantity in the spherical basis.
        const Eigen::MatrixXd M_sph = make_symmetric(n_sph, 0.5 + L);
        // X_cart plays the role of a Cartesian operator block from the integral engine.
        const Eigen::MatrixXd X_cart = make_symmetric(n_cart, 2.3 + L);

        auto M_cart_res = HartreeFock::BasisFunctions::lift_density_sph_to_cart(M_sph, C);
        if (!M_cart_res)
        {
            fail("lift_density_sph_to_cart(L=" + std::to_string(L) + ") failed: " + M_cart_res.error());
            return;
        }
        const Eigen::MatrixXd M_cart = *M_cart_res;

        if (M_cart.rows() != n_cart || M_cart.cols() != n_cart)
        {
            fail("L=" + std::to_string(L) + ": lifted matrix has wrong shape " +
                 std::to_string(M_cart.rows()) + "×" + std::to_string(M_cart.cols()) +
                 " (expected " + std::to_string(n_cart) + "²)");
            return;
        }

        // Symmetry preservation.
        const double sym_err = (M_cart - M_cart.transpose()).cwiseAbs().maxCoeff();
        if (sym_err > 1e-12)
            fail("L=" + std::to_string(L) + ": lifted matrix is not symmetric (max asym = " +
                 std::to_string(sym_err) + ")");

        // The contract: energy invariance.
        const Eigen::MatrixXd X_sph = C * X_cart * C.transpose();
        const double lhs = (M_sph * X_sph).trace();
        const double rhs = (M_cart * X_cart).trace();
        const double err = std::abs(lhs - rhs);
        const double scale = std::max(1.0, std::abs(lhs));
        if (err > 1e-12 * scale)
        {
            fail("L=" + std::to_string(L) + ": energy-invariance contract violated, " +
                 "tr(M_sph·X_sph)=" + std::to_string(lhs) +
                 " vs tr(M_cart·X_cart)=" + std::to_string(rhs) +
                 " (|Δ|=" + std::to_string(err) + ")");
        }
    }

    void check_dimension_mismatch_errors()
    {
        // A 3×3 C (corresponds to L=1) paired with a 5×5 M_sph must error cleanly.
        auto C_res = HartreeFock::BasisFunctions::cart_to_sph_block(1);
        if (!C_res)
        {
            fail("cart_to_sph_block(L=1) failed: " + C_res.error());
            return;
        }
        const Eigen::MatrixXd C = *C_res;
        const Eigen::MatrixXd bad = Eigen::MatrixXd::Identity(5, 5);
        auto res = HartreeFock::BasisFunctions::lift_density_sph_to_cart(bad, C);
        if (res)
            fail("expected dimension-mismatch error, got success");
    }
} // namespace

int main()
{
    for (int L = 0; L <= 4; ++L)
        check_for_L(L);
    check_dimension_mismatch_errors();

    if (g_ok)
    {
        std::cout << "spherical_density_lift: OK\n";
        return 0;
    }
    std::cout << "spherical_density_lift: FAIL\n";
    return 1;
}
