#ifndef HF_POSTHF_CC_GEMM_CONTRACT_H
#define HF_POSTHF_CC_GEMM_CONTRACT_H

// One binary tensor contraction as a matrix product, for the generated CC
// kernels (docs/CCSD_T_SIZE_SCOPE.md, R3):
//
//     out(...) += coeff * sum_k A(m..., k...) B(k..., n...)
//
// with the index groups m, k, n given as label strings in the order the
// matrices use. Each operand is used in place when its storage already is
// that matrix (either orientation), else copied into one. The emitter only
// lowers a contraction when every copied operand is small (o^2 v^2 at most).
//
// `pair_half` (the spatial doubles kernels, which mirror j < i afterwards):
// a group holding both i and j keeps only j >= i, which halves the ladder.
// Where i and j are in different groups the half is not a block, and the
// whole product is computed; the mirror overwrites j < i either way.
//
// The product runs in fixed 64-wide blocks of the larger output dimension,
// so the summation order of every element, and the result, are independent
// of the thread count.

#include <Eigen/Dense>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string_view>
#include <vector>

#include "post_hf/cc/common.h"

namespace HartreeFock::Correlation::CC
{
    struct CCOperand
    {
        const double *data = nullptr;
        std::vector<int> dims;
    };

    inline CCOperand cc_operand(const Tensor2D &t) { return {t.data.data(), {t.dim1, t.dim2}}; }
    inline CCOperand cc_operand(const Tensor4D &t)
    {
        return {t.data.data(), {t.dim1, t.dim2, t.dim3, t.dim4}};
    }
    inline CCOperand cc_operand(const Tensor6D &t)
    {
        return {t.data.data(), {t.dim1, t.dim2, t.dim3, t.dim4, t.dim5, t.dim6}};
    }
    inline CCOperand cc_operand(const TensorND &t) { return {t.data.data(), t.dims}; }
    template <int N> CCOperand cc_operand(const ConstFixedRankView<N> &v)
    {
        return {v.data, std::vector<int>(v.dims, v.dims + N)};
    }

    namespace detail
    {
        // Storage offset of every combination of `group` labels, row-major
        // over the group in the order given, for a tensor stored as `labels`
        // with `dims`. With `pair_half`, a group holding both `i` and `j`
        // keeps only combinations with j >= i (the doubles half range); every
        // tensor sharing the group drops the same combinations, so rows stay
        // aligned.
        inline std::vector<std::ptrdiff_t> cc_group_offsets(std::string_view labels,
                                                            const std::vector<int> &dims,
                                                            std::string_view group,
                                                            bool pair_half = false)
        {
            std::vector<std::ptrdiff_t> stride(labels.size());
            std::ptrdiff_t s = 1;
            for (std::size_t p = labels.size(); p-- > 0;) {
                stride[p] = s;
                s *= dims[p];
            }
            const bool filter = pair_half && group.find('i') != std::string_view::npos &&
                                group.find('j') != std::string_view::npos;
            std::vector<std::ptrdiff_t> out{0};
            std::vector<int> vi{0}, vj{0};  // values of i and j per combination
            for (char c : group) {
                const std::size_t p = labels.find(c);
                assert(p != std::string_view::npos && "cc_gemm_contract: label not in tensor");
                std::vector<std::ptrdiff_t> next;
                std::vector<int> ni, nj;
                next.reserve(out.size() * static_cast<std::size_t>(dims[p]));
                for (std::size_t q = 0; q < out.size(); ++q)
                    for (int x = 0; x < dims[p]; ++x) {
                        next.push_back(out[q] + x * stride[p]);
                        ni.push_back(c == 'i' ? x : vi[q]);
                        nj.push_back(c == 'j' ? x : vj[q]);
                    }
                out = std::move(next);
                vi = std::move(ni);
                vj = std::move(nj);
            }
            if (filter) {
                std::vector<std::ptrdiff_t> kept;
                for (std::size_t q = 0; q < out.size(); ++q)
                    if (vj[q] >= vi[q])
                        kept.push_back(out[q]);
                out = std::move(kept);
            }
            return out;
        }

        inline bool cc_is_range(const std::vector<std::ptrdiff_t> &o, std::ptrdiff_t step)
        {
            for (std::size_t i = 0; i < o.size(); ++i)
                if (o[i] != static_cast<std::ptrdiff_t>(i) * step)
                    return false;
            return true;
        }

        using CCRowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using CCMap = Eigen::Map<const CCRowMat>;

        // Operand as an (rows x cols) matrix over groups (r, c): a map when the
        // storage already is that matrix or its transpose, else a copy.
        struct CCMatrix
        {
            CCRowMat copy;
            const double *data = nullptr;
            bool transposed = false;
            Eigen::Index rows = 0, cols = 0;
        };

        inline CCMatrix cc_matrix(const CCOperand &x, std::string_view labels,
                                  std::string_view r, std::string_view c, bool pair_half)
        {
            const auto ro = cc_group_offsets(labels, x.dims, r, pair_half);
            const auto co = cc_group_offsets(labels, x.dims, c, pair_half);
            CCMatrix m;
            m.rows = static_cast<Eigen::Index>(ro.size());
            m.cols = static_cast<Eigen::Index>(co.size());
            if (cc_is_range(co, 1) && cc_is_range(ro, m.cols)) {
                m.data = x.data;
            } else if (cc_is_range(ro, 1) && cc_is_range(co, m.rows)) {
                m.data = x.data;
                m.transposed = true;
            } else {
                m.copy.resize(m.rows, m.cols);
#pragma omp parallel for schedule(static)
                for (Eigen::Index i = 0; i < m.rows; ++i)
                    for (Eigen::Index j = 0; j < m.cols; ++j)
                        m.copy(i, j) = x.data[ro[i] + co[j]];
                m.data = m.copy.data();
            }
            return m;
        }
    } // namespace detail

    template <class Out>
    void cc_gemm_contract(Out &out_tensor, std::string_view out_labels, double coeff,
                                 const CCOperand &a, std::string_view a_labels,
                                 const CCOperand &b, std::string_view b_labels,
                                 std::string_view m, std::string_view k, std::string_view n,
                                 bool pair_half = false)
    {
        using namespace detail;
        double *const out = out_tensor.data.data();
        const std::vector<int> out_dims = cc_operand(out_tensor).dims;
        const CCMatrix am = cc_matrix(a, a_labels, m, k, pair_half);
        const CCMatrix bm = cc_matrix(b, b_labels, k, n, pair_half);
        const Eigen::Index M = am.rows, K = am.cols, N = bm.cols;

        // Every A and B is an Eigen expression of one of two kinds; one lambda
        // per combination keeps the product a single noalias() GEMM.
        auto amat = [&] { return CCMap(am.data, am.transposed ? K : M, am.transposed ? M : K); };
        auto bmat = [&] { return CCMap(bm.data, bm.transposed ? N : K, bm.transposed ? K : N); };

        CCRowMat prod(M, N);
        constexpr Eigen::Index block = 64;
        const bool by_rows = M >= N;
        const Eigen::Index extent = by_rows ? M : N;
        const Eigen::Index n_blocks = (extent + block - 1) / block;
#pragma omp parallel for schedule(static)
        for (Eigen::Index t = 0; t < n_blocks; ++t) {
            const Eigen::Index s = t * block, w = std::min(block, extent - s);
            const auto A = amat();
            const auto B = bmat();
            if (by_rows) {
                if (am.transposed && bm.transposed)
                    prod.middleRows(s, w).noalias() = A.transpose().middleRows(s, w) * B.transpose();
                else if (am.transposed)
                    prod.middleRows(s, w).noalias() = A.transpose().middleRows(s, w) * B;
                else if (bm.transposed)
                    prod.middleRows(s, w).noalias() = A.middleRows(s, w) * B.transpose();
                else
                    prod.middleRows(s, w).noalias() = A.middleRows(s, w) * B;
            } else {
                if (am.transposed && bm.transposed)
                    prod.middleCols(s, w).noalias() = A.transpose() * B.transpose().middleCols(s, w);
                else if (am.transposed)
                    prod.middleCols(s, w).noalias() = A.transpose() * B.middleCols(s, w);
                else if (bm.transposed)
                    prod.middleCols(s, w).noalias() = A * B.transpose().middleCols(s, w);
                else
                    prod.middleCols(s, w).noalias() = A * B.middleCols(s, w);
            }
        }

        const auto mo = cc_group_offsets(out_labels, out_dims, m, pair_half);
        const auto no = cc_group_offsets(out_labels, out_dims, n, pair_half);
#pragma omp parallel for schedule(static)
        for (Eigen::Index i = 0; i < M; ++i)
            for (Eigen::Index j = 0; j < N; ++j)
                out[mo[i] + no[j]] += coeff * prod(i, j);
    }
} // namespace HartreeFock::Correlation::CC

#endif
