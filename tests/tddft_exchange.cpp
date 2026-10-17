#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "dft/tddft_exchange.h"

namespace
{
    [[nodiscard]] bool close(double lhs, double rhs)
    {
        return std::abs(lhs - rhs) < 1.0e-14;
    }
}

int main()
{
    // Rectangular occupied/virtual spaces are intentional: a square fixture
    // can make the two B-index permutations look equivalent by accident.
    constexpr int no = 2;
    constexpr int nv = 3;
    constexpr int offset = 1;
    constexpr double coefficient = 0.37;
    const int nov = no * nv;
    const int dimension = offset + nov + 2;

    std::vector<double> k_a(static_cast<std::size_t>(no * no * nv * nv));
    std::vector<double> k_b(static_cast<std::size_t>(no * nv * nv * no));
    for (std::size_t index = 0; index < k_a.size(); ++index)
        k_a[index] = 10.0 + static_cast<double>(index);
    for (std::size_t index = 0; index < k_b.size(); ++index)
        k_b[index] = 100.0 + static_cast<double>(index);

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dimension, dimension);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(dimension, dimension);
    DFT::LinearResponse::add_same_spin_exact_exchange(
        A, B, offset, no, nv, coefficient, k_a, k_b);

    bool ok = true;
    for (int i = 0; i < no; ++i)
        for (int a = 0; a < nv; ++a)
            for (int j = 0; j < no; ++j)
                for (int b = 0; b < nv; ++b)
                {
                    const int ia = offset + i * nv + a;
                    const int jb = offset + j * nv + b;
                    const double expected_a = -coefficient * k_a[
                        DFT::LinearResponse::tddft_oovv_index(i, j, a, b, no, nv)];
                    const double expected_b = -coefficient * k_b[
                        DFT::LinearResponse::tddft_ovvo_index(i, b, a, j, no, nv)];
                    if (!close(A(ia, jb), expected_a) || !close(B(ia, jb), expected_b))
                        ok = false;
                }

    // The old (ia|bj) lookup must differ on at least one rectangular entry;
    // otherwise this test would not detect restoring the historical defect.
    bool mutant_detected = false;
    for (int i = 0; i < no; ++i)
        for (int a = 0; a < nv; ++a)
            for (int j = 0; j < no; ++j)
                for (int b = 0; b < nv; ++b)
                {
                    const double correct = B(offset + i * nv + a, offset + j * nv + b);
                    const double wrong = -coefficient * k_b[
                        DFT::LinearResponse::tddft_ovvo_index(i, a, b, j, no, nv)];
                    mutant_detected = mutant_detected || !close(correct, wrong);
                }

    if (!ok || !mutant_detected)
    {
        std::cerr << "FAIL: TDDFT same-spin exchange B-block convention\n";
        return 1;
    }

    std::cout << "TDDFT exchange B-block uses (ib|ja) in shared RKS/UKS helper PASS\n";
    return 0;
}
