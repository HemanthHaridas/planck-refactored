#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace DFT::LinearResponse
{
    // Add one same-spin exact-exchange term (full-range or short-range) of a TDDFT response
    // block.  The transformed blocks use these explicit layouts:
    //
    //   K_A(i,j,a,b) = (ij|ab),     shape (o,o,v,v)
    //   K_B(i,a,b,j) = (ia|bj),     shape (o,v,v,o)
    //
    // For B(ia,jb), however, exchange is (ib|ja), so K_B must be addressed
    // as K_B(i,b,a,j).  Keeping this in one helper prevents the RKS and UKS
    // Casida builders from silently drifting into different conventions.
    [[nodiscard]] inline std::size_t tddft_oovv_index(
        int i, int j, int a, int b, int n_occ, int n_virt)
    {
        return ((static_cast<std::size_t>(i) * n_occ + j) * n_virt + a) * n_virt + b;
    }

    [[nodiscard]] inline std::size_t tddft_ovvo_index(
        int i, int a, int b, int j, int n_occ, int n_virt)
    {
        return ((static_cast<std::size_t>(i) * n_virt + a) * n_virt + b) * n_occ + j;
    }

    inline void add_same_spin_exact_exchange(
        Eigen::Ref<Eigen::MatrixXd> A,
        Eigen::Ref<Eigen::MatrixXd> B,
        int mo_offset,
        int n_occ,
        int n_virt,
        double coefficient,
        const std::vector<double> &k_a,
        const std::vector<double> &k_b)
    {
        if (coefficient == 0.0)
            return;

        for (int i = 0; i < n_occ; ++i)
            for (int a = 0; a < n_virt; ++a)
            {
                const int ia = mo_offset + i * n_virt + a;
                for (int j = 0; j < n_occ; ++j)
                    for (int b = 0; b < n_virt; ++b)
                    {
                        const int jb = mo_offset + j * n_virt + b;
                        A(ia, jb) -= coefficient * k_a[tddft_oovv_index(i, j, a, b, n_occ, n_virt)];
                        B(ia, jb) -= coefficient * k_b[tddft_ovvo_index(i, b, a, j, n_occ, n_virt)];
                    }
            }
    }
}
