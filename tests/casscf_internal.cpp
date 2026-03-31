#include "post_hf/casscf_internal.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

using namespace HartreeFock::Correlation::CASSCFInternal;

bool expect(bool condition, const std::string& message)
{
    if (condition) return true;
    std::cerr << message << '\n';
    return false;
}

} // namespace

int main()
{
    bool ok = true;

    {
        ok &= expect(single_bit_mask(-1) == 0,
                     "single_bit_mask should guard negative bit indices");
        ok &= expect(single_bit_mask(kCIStringBits) == 0,
                     "single_bit_mask should guard out-of-range bit indices");
        ok &= expect(low_bit_mask(kCIStringBits) == std::numeric_limits<CIString>::max(),
                     "low_bit_mask should saturate instead of performing an undefined full-width shift");
    }

    {
        RASParams ras{1, 1, 1, 1, 2, true};
        const CIString alpha = single_bit_mask(1) | single_bit_mask(2);
        const CIString beta  = single_bit_mask(1) | single_bit_mask(2);

        ok &= expect(ras1_holes(alpha, beta, ras) == 2,
                     "combined RAS1 holes should count alpha and beta together");
        ok &= expect(!admissible_ras_pair(alpha, beta, ras),
                     "pair-level RAS filtering should reject determinants with too many total RAS1 holes");
    }

    {
        RASParams ras{1, 1, 1, 2, 1, true};
        const CIString alpha = single_bit_mask(0) | single_bit_mask(2);
        const CIString beta  = single_bit_mask(1) | single_bit_mask(2);

        ok &= expect(ras3_electrons(alpha, beta, ras) == 2,
                     "combined RAS3 electrons should count alpha and beta together");
        ok &= expect(!admissible_ras_pair(alpha, beta, ras),
                     "pair-level RAS filtering should reject determinants with too many total RAS3 electrons");
    }

    {
        SymmetryContext sym;
        sym.names = {"B1", "A1", "B2", "A2"};
        sym.product = {
            {1, 0, 3, 2},
            {0, 1, 2, 3},
            {3, 2, 1, 0},
            {2, 3, 0, 1},
        };
        sym.abelian_1d_only = true;
        sym.totally_symmetric_irrep = 1;
        const std::vector<int> irr_act = {0, 2};

        ok &= expect(determinant_symmetry(single_bit_mask(0), single_bit_mask(1), irr_act, sym) == 3,
                     "determinant symmetry should use the explicit product table");
        ok &= expect(determinant_symmetry(single_bit_mask(0), single_bit_mask(0), irr_act, sym) == 1,
                     "occupying the same irrep in alpha and beta should multiply back to the totally symmetric irrep");
    }

    {
        const std::vector<double> puvw = {2.0};
        const std::vector<double> gamma = {3.0};
        const Eigen::MatrixXd q = contract_q_matrix(puvw, gamma, 1, 1);

        ok &= expect(q.rows() == 1 && q.cols() == 1 && std::abs(q(0, 0) - 6.0) < 1e-12,
                     "cached Q contraction should reproduce the expected scalar contraction");
    }

    {
        Eigen::MatrixXd c_old = Eigen::MatrixXd::Zero(3, 2);
        c_old(0, 0) = 1.0;
        c_old(1, 1) = 1.0;

        Eigen::MatrixXd c_new = Eigen::MatrixXd::Zero(3, 2);
        c_new(1, 0) = 1.0;
        c_new(0, 1) = -1.0;

        const Eigen::MatrixXd overlaps = compute_root_overlap(c_old, c_new);
        const std::vector<int> match = match_roots_by_max_overlap(overlaps);

        ok &= expect(overlaps.rows() == 2 && overlaps.cols() == 2,
                     "root overlap matrix should have one row/column per tracked root");
        ok &= expect(match.size() == 2 && match[0] == 1 && match[1] == 0,
                     "root matching should preserve state identity across swapped eigenpairs");
        ok &= expect(std::abs(overlaps(0, 1)) > 0.999 && std::abs(overlaps(1, 0)) > 0.999,
                     "root overlaps should be computed from CI-vector inner products");
    }

    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 3);
        H(1, 1) = 2.0;
        H(1, 2) = 0.2;
        H(2, 1) = 0.2;
        H(2, 2) = 4.0;

        Eigen::VectorXd c0 = Eigen::VectorXd::Zero(3);
        c0(0) = 1.0;
        const double E0 = 0.0;
        const Eigen::VectorXd H_diag = H.diagonal();
        Eigen::VectorXd sigma(3);
        sigma << 0.7, 0.3, -0.2;

        const CIResponseResult response =
            solve_ci_response_iterative(H, c0, E0, H_diag, sigma, 1e-12, 16, 1e-6);

        const Eigen::Vector2d rhs = -project_orthogonal(sigma, c0).tail<2>();
        const Eigen::Matrix2d subblock = H.bottomRightCorner<2, 2>();
        Eigen::VectorXd exact = Eigen::VectorXd::Zero(3);
        exact.tail<2>() = subblock.colPivHouseholderQr().solve(rhs);

        ok &= expect(response.converged,
                     "iterative CI response should converge on a small projected linear problem");
        ok &= expect(response.iterations > 0,
                     "iterative CI response should report at least one iteration when solving a nonzero response");
        ok &= expect(response.residual_norm < 1e-10,
                     "iterative CI response should drive the projected residual norm small");
        ok &= expect(std::abs(c0.dot(response.c1)) < 1e-12,
                     "iterative CI response should preserve the orthogonality gauge");
        ok &= expect((response.c1 - exact).norm() < 1e-10,
                     "iterative CI response should match the dense projected solution on a small test problem");
    }

    {
        Eigen::MatrixXd gamma(2, 2);
        gamma << 1.2, 0.3,
                 0.3, 0.8;
        const NaturalOrbitalData natural = diagonalize_natural_orbitals(gamma);
        const Eigen::MatrixXd rebuilt =
            natural.rotation * natural.occupations.asDiagonal() * natural.rotation.transpose();

        ok &= expect(natural.occupations(0) >= natural.occupations(1),
                     "natural occupations should be returned in descending order");
        ok &= expect(std::abs(natural.occupations.sum() - gamma.trace()) < 1e-12,
                     "natural occupations should preserve the active electron count");
        ok &= expect((rebuilt - gamma).norm() < 1e-12,
                     "natural-orbital rotation should diagonalize the 1-RDM without changing it on reconstruction");
    }

    return ok ? 0 : 1;
}
