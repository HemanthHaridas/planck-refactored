#ifndef HF_POSTHF_CC_COMMON_H
#define HF_POSTHF_CC_COMMON_H

#include <Eigen/Core>
#include <cassert>
#include <cstddef>
#include <expected>
#include <initializer_list>
#include <string>
#include <vector>

#include "base/types.h"

namespace HartreeFock::Correlation::CC
{
    namespace detail
    {
        // Fixed-rank element access is the innermost operation of every CC kernel, so it
        // is inlined here rather than left out-of-line in common.cpp. The out-of-line form
        // built two std::vector<int> per access (dims + indices) and returned a
        // std::expected; with no LTO configured it could not be inlined away, costing two
        // heap allocations per element read. Measured at 13.5x on the rank-3 triples
        // kernel -- the dominant term in the generated-vs-hand-written gap.
        // See docs/CCGEN_TENSOR_ACCESSOR_FIX_SCOPE.md.
        //
        // The debug assert keeps BOTH conditions the old checked_fixed_rank_index enforced:
        // per-index range, and offset < data_size. The storage half is not redundant with
        // the constructors' size validation -- `data` is a public member that call sites
        // assign directly after construction (e.g. tensor_backend.cpp:197-198), so the
        // size invariant is breakable post-construction by design.
        [[nodiscard]] inline bool fixed_rank_index_valid(
            std::initializer_list<int> dims,
            std::initializer_list<int> indices,
            std::size_t data_size) noexcept
        {
            std::size_t offset = 0;
            const int *dim = dims.begin();
            const int *idx = indices.begin();
            for (std::size_t pos = 0; pos < dims.size(); ++pos)
            {
                if (idx[pos] < 0 || idx[pos] >= dim[pos])
                    return false;
                offset *= static_cast<std::size_t>(dim[pos]);
                offset += static_cast<std::size_t>(idx[pos]);
            }
            return offset < data_size;
        }

        // Same, for the runtime-rank tensors whose dims live in a std::vector.
        [[nodiscard]] inline bool nd_index_valid(
            const std::vector<int> &dims,
            std::initializer_list<int> indices,
            std::size_t data_size) noexcept
        {
            if (dims.size() != indices.size())
                return false;
            std::size_t offset = 0;
            const int *idx = indices.begin();
            for (std::size_t pos = 0; pos < dims.size(); ++pos)
            {
                if (idx[pos] < 0 || idx[pos] >= dims[pos])
                    return false;
                offset *= static_cast<std::size_t>(dims[pos]);
                offset += static_cast<std::size_t>(idx[pos]);
            }
            return offset < data_size;
        }

        // Row-major flat offset for a runtime-rank tensor, computed straight off the
        // initializer_list. The out-of-line overload copies the list into a std::vector
        // (`to_vector`) before calling flatten_index, i.e. one heap allocation per element
        // access -- and the rank-4 generated kernel performs 23338 such accesses per
        // residual evaluation, which is why the fixed-rank fix alone left rank 4 unchanged.
        [[nodiscard]] inline std::size_t nd_flat_index(
            const std::vector<int> &dims,
            std::initializer_list<int> indices) noexcept
        {
            std::size_t offset = 0;
            const int *idx = indices.begin();
            for (std::size_t pos = 0; pos < dims.size(); ++pos)
            {
                offset *= static_cast<std::size_t>(dims[pos]);
                offset += static_cast<std::size_t>(idx[pos]);
            }
            return offset;
        }
    } // namespace detail

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

        double &operator()(int i, int j) noexcept
        {
            assert(detail::fixed_rank_index_valid({dim1, dim2}, {i, j}, data.size()) &&
                   "Tensor2D index validation failed");
            return data[static_cast<std::size_t>(i) * static_cast<std::size_t>(dim2) +
                        static_cast<std::size_t>(j)];
        }

        const double &operator()(int i, int j) const noexcept
        {
            assert(detail::fixed_rank_index_valid({dim1, dim2}, {i, j}, data.size()) &&
                   "Tensor2D index validation failed");
            return data[static_cast<std::size_t>(i) * static_cast<std::size_t>(dim2) +
                        static_cast<std::size_t>(j)];
        }
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

        double &operator()(int i, int j, int k, int l) noexcept
        {
            assert(detail::fixed_rank_index_valid(
                       {dim1, dim2, dim3, dim4}, {i, j, k, l}, data.size()) &&
                   "Tensor4D index validation failed");
            return data[((static_cast<std::size_t>(i) * static_cast<std::size_t>(dim2) +
                          static_cast<std::size_t>(j)) *
                             static_cast<std::size_t>(dim3) +
                         static_cast<std::size_t>(k)) *
                            static_cast<std::size_t>(dim4) +
                        static_cast<std::size_t>(l)];
        }

        const double &operator()(int i, int j, int k, int l) const noexcept
        {
            assert(detail::fixed_rank_index_valid(
                       {dim1, dim2, dim3, dim4}, {i, j, k, l}, data.size()) &&
                   "Tensor4D index validation failed");
            return data[((static_cast<std::size_t>(i) * static_cast<std::size_t>(dim2) +
                          static_cast<std::size_t>(j)) *
                             static_cast<std::size_t>(dim3) +
                         static_cast<std::size_t>(k)) *
                            static_cast<std::size_t>(dim4) +
                        static_cast<std::size_t>(l)];
        }
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

        double &operator()(int i, int j, int k, int l, int m, int n) noexcept
        {
            assert(detail::fixed_rank_index_valid(
                       {dim1, dim2, dim3, dim4, dim5, dim6}, {i, j, k, l, m, n}, data.size()) &&
                   "Tensor6D index validation failed");
            return data[flat_index(i, j, k, l, m, n)];
        }

        const double &operator()(int i, int j, int k, int l, int m, int n) const noexcept
        {
            assert(detail::fixed_rank_index_valid(
                       {dim1, dim2, dim3, dim4, dim5, dim6}, {i, j, k, l, m, n}, data.size()) &&
                   "Tensor6D index validation failed");
            return data[flat_index(i, j, k, l, m, n)];
        }

        // Not private: Tensor6D is aggregate/designated-initialized at several call sites,
        // and an access specifier would make it a non-aggregate.
        [[nodiscard]] std::size_t flat_index(
            int i, int j, int k, int l, int m, int n) const noexcept
        {
            return ((((static_cast<std::size_t>(i) * static_cast<std::size_t>(dim2) +
                       static_cast<std::size_t>(j)) *
                          static_cast<std::size_t>(dim3) +
                      static_cast<std::size_t>(k)) *
                         static_cast<std::size_t>(dim4) +
                     static_cast<std::size_t>(l)) *
                        static_cast<std::size_t>(dim5) +
                    static_cast<std::size_t>(m)) *
                       static_cast<std::size_t>(dim6) +
                   static_cast<std::size_t>(n);
        }
    };

    struct TensorND
    {
        std::vector<int> dims;
        std::vector<double> data;

        TensorND() = default;
        explicit TensorND(std::vector<int> dims, double value = 0.0);
        TensorND(std::vector<int> dims, std::vector<double> values);

        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] int order() const noexcept;

        // The initializer_list overloads are inlined (hot path: the generated arbitrary-order
        // kernels index exclusively through braced lists). The vector<int> overloads stay
        // out-of-line for the handful of non-hot callers that already hold a vector.
        double &operator()(std::initializer_list<int> indices) noexcept
        {
            assert(detail::nd_index_valid(dims, indices, data.size()) &&
                   "TensorND index validation failed");
            return data[detail::nd_flat_index(dims, indices)];
        }

        const double &operator()(std::initializer_list<int> indices) const noexcept
        {
            assert(detail::nd_index_valid(dims, indices, data.size()) &&
                   "TensorND index validation failed");
            return data[detail::nd_flat_index(dims, indices)];
        }

        double &operator()(const std::vector<int> &indices);
        const double &operator()(const std::vector<int> &indices) const;
    };

    struct DenseTensorView
    {
        std::vector<int> dims;
        double *data = nullptr;

        [[nodiscard]] std::size_t size() const;
        [[nodiscard]] int order() const noexcept;

        double &operator()(std::initializer_list<int> indices) noexcept
        {
            assert(detail::nd_index_valid(dims, indices, size()) &&
                   "DenseTensorView index validation failed");
            return data[detail::nd_flat_index(dims, indices)];
        }

        const double &operator()(std::initializer_list<int> indices) const noexcept
        {
            assert(detail::nd_index_valid(dims, indices, size()) &&
                   "DenseTensorView index validation failed");
            return data[detail::nd_flat_index(dims, indices)];
        }

        double &operator()(const std::vector<int> &indices);
        const double &operator()(const std::vector<int> &indices) const;
    };

    struct ConstDenseTensorView
    {
        std::vector<int> dims;
        const double *data = nullptr;

        [[nodiscard]] std::size_t size() const;
        [[nodiscard]] int order() const noexcept;

        const double &operator()(std::initializer_list<int> indices) const noexcept
        {
            assert(detail::nd_index_valid(dims, indices, size()) &&
                   "ConstDenseTensorView index validation failed");
            return data[detail::nd_flat_index(dims, indices)];
        }

        const double &operator()(const std::vector<int> &indices) const;
    };

    [[nodiscard]] DenseTensorView make_tensor_view(Tensor2D &tensor);
    [[nodiscard]] DenseTensorView make_tensor_view(Tensor4D &tensor);
    [[nodiscard]] DenseTensorView make_tensor_view(Tensor6D &tensor);
    [[nodiscard]] DenseTensorView make_tensor_view(TensorND &tensor);

    [[nodiscard]] ConstDenseTensorView make_tensor_view(const Tensor2D &tensor);
    [[nodiscard]] ConstDenseTensorView make_tensor_view(const Tensor4D &tensor);
    [[nodiscard]] ConstDenseTensorView make_tensor_view(const Tensor6D &tensor);
    [[nodiscard]] ConstDenseTensorView make_tensor_view(const TensorND &tensor);

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

    struct UHFReference
    {
        int n_ao = 0;
        int n_mo = 0;
        int n_occ_alpha = 0;
        int n_occ_beta = 0;
        int n_virt_alpha = 0;
        int n_virt_beta = 0;

        // The unrestricted determinant-space teaching solvers need access to the
        // full alpha and beta canonical MO spaces because occupied and virtual
        // spin orbitals are interleaved manually into a single reference state.
        Eigen::MatrixXd C_alpha;
        Eigen::MatrixXd C_beta;
        Eigen::VectorXd eps_alpha;
        Eigen::VectorXd eps_beta;
    };

    // Build the canonical RHF occupied/virtual partition once so all CC methods
    // share the same validation and indexing conventions.
    std::expected<RHFReference, std::string> build_rhf_reference(
        HartreeFock::Calculator &calculator);

    // The unrestricted CC prototypes start from the canonical UHF alpha/beta
    // orbitals exactly as they come out of the SCF code. The reference builder
    // centralizes the occupation counting and dimension checks so the solvers
    // themselves can stay focused on the coupled-cluster algebra.
    std::expected<UHFReference, std::string> build_uhf_reference(
        HartreeFock::Calculator &calculator);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_COMMON_H
