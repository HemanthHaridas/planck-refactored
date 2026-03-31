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

    return ok ? 0 : 1;
}
