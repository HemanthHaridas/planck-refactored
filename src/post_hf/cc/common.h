#ifndef HF_POSTHF_CC_COMMON_H
#define HF_POSTHF_CC_COMMON_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <vector>

#include "base/types.h"

namespace HartreeFock::Correlation::CC
{
    // The CC module intentionally uses small explicit tensor wrappers rather than
    // a heavily abstracted tensor library. Students can see the dimensions and
    // indexing rules directly, while the underlying storage still remains
    // contiguous and cache-friendly.
    struct Tensor2D
    {
        int dim1 = 0;
        int dim2 = 0;
        std::vector<double> data;

        Tensor2D() = default;
        Tensor2D(int d1, int d2, double value = 0.0);
        Tensor2D(int d1, int d2, std::vector<double> values);

        [[nodiscard]] std::size_t size() const noexcept;

        double &operator()(int i, int j) noexcept;
        const double &operator()(int i, int j) const noexcept;
    };

    struct Tensor4D
    {
        int dim1 = 0;
        int dim2 = 0;
        int dim3 = 0;
        int dim4 = 0;
        std::vector<double> data;

        Tensor4D() = default;
        Tensor4D(int d1, int d2, int d3, int d4, double value = 0.0);
        Tensor4D(int d1, int d2, int d3, int d4, std::vector<double> values);

        [[nodiscard]] std::size_t size() const noexcept;

        double &operator()(int i, int j, int k, int l) noexcept;
        const double &operator()(int i, int j, int k, int l) const noexcept;
    };

    struct Tensor6D
    {
        int dim1 = 0;
        int dim2 = 0;
        int dim3 = 0;
        int dim4 = 0;
        int dim5 = 0;
        int dim6 = 0;
        std::vector<double> data;

        Tensor6D() = default;
        Tensor6D(int d1, int d2, int d3, int d4, int d5, int d6, double value = 0.0);
        Tensor6D(int d1, int d2, int d3, int d4, int d5, int d6, std::vector<double> values);

        [[nodiscard]] std::size_t size() const noexcept;

        double &operator()(int i, int j, int k, int l, int m, int n) noexcept;
        const double &operator()(int i, int j, int k, int l, int m, int n) const noexcept;
    };

    struct RHFReference
    {
        int n_ao = 0;
        int n_mo = 0;
        int n_occ = 0;
        int n_virt = 0;

        // The occupied/virtual partition is stored explicitly so every post-HF
        // routine can reuse the same canonical RHF bookkeeping.
        Eigen::MatrixXd C_occ;
        Eigen::MatrixXd C_virt;
        Eigen::VectorXd eps_occ;
        Eigen::VectorXd eps_virt;
    };

    // Build the canonical RHF occupied/virtual partition once so all CC methods
    // share the same validation and indexing conventions.
    std::expected<RHFReference, std::string> build_rhf_reference(
        HartreeFock::Calculator &calculator);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_COMMON_H
