// Pins the inlined fixed-rank Tensor{2,4,6}D::operator() index arithmetic against an
// independent row-major reference.
//
// The accessors were moved out of common.cpp into common.h as flat-index computations
// (docs/CCGEN_TENSOR_ACCESSOR_FIX_SCOPE.md); this asserts the flat form reproduces the
// original `flatten_index` row-major layout exactly.
//
// NON-SQUARE DIMENSIONS ARE THE POINT. With every extent equal (the bh3/STO-3G case is
// no == nv == 4) a transposed or wrongly-ordered index still lands in bounds and reads a
// plausible number, so a square fixture cannot catch a layout regression. Every shape
// below has pairwise-distinct extents.

#include "post_hf/cc/common.h"

#include <cstdio>
#include <vector>

using HartreeFock::Correlation::CC::Tensor2D;
using HartreeFock::Correlation::CC::Tensor4D;
using HartreeFock::Correlation::CC::Tensor6D;

namespace
{
    int failures = 0;

    void check(bool ok, const char *what)
    {
        if (!ok)
        {
            std::printf("[FAIL] %s\n", what);
            ++failures;
        }
    }

    // Independent row-major reference: last index varies fastest.
    std::size_t reference_index(const std::vector<int> &dims, const std::vector<int> &idx)
    {
        std::size_t offset = 0;
        for (std::size_t p = 0; p < dims.size(); ++p)
        {
            offset *= static_cast<std::size_t>(dims[p]);
            offset += static_cast<std::size_t>(idx[p]);
        }
        return offset;
    }

    void test_tensor2d_layout()
    {
        const int d1 = 3, d2 = 5;
        Tensor2D t(d1, d2, 0.0);
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d2; ++j)
                t(i, j) = static_cast<double>(reference_index({d1, d2}, {i, j}));

        bool ok = true;
        for (std::size_t n = 0; n < t.data.size(); ++n)
            ok = ok && (t.data[n] == static_cast<double>(n));
        check(ok, "Tensor2D flat layout matches row-major reference (3x5)");
    }

    void test_tensor4d_layout()
    {
        const int d1 = 2, d2 = 3, d3 = 4, d4 = 5;
        Tensor4D t(d1, d2, d3, d4, 0.0);
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d2; ++j)
                for (int k = 0; k < d3; ++k)
                    for (int l = 0; l < d4; ++l)
                        t(i, j, k, l) =
                            static_cast<double>(reference_index({d1, d2, d3, d4}, {i, j, k, l}));

        bool ok = true;
        for (std::size_t n = 0; n < t.data.size(); ++n)
            ok = ok && (t.data[n] == static_cast<double>(n));
        check(ok, "Tensor4D flat layout matches row-major reference (2x3x4x5)");
    }

    void test_tensor6d_layout()
    {
        // Distinct extents in every axis, and n_occ != n_virt, mirroring a real
        // (o,o,o,v,v,v) triples block on a non-square system.
        const int d1 = 2, d2 = 3, d3 = 4, d4 = 5, d5 = 6, d6 = 7;
        Tensor6D t(d1, d2, d3, d4, d5, d6, 0.0);
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d2; ++j)
                for (int k = 0; k < d3; ++k)
                    for (int l = 0; l < d4; ++l)
                        for (int m = 0; m < d5; ++m)
                            for (int n = 0; n < d6; ++n)
                                t(i, j, k, l, m, n) = static_cast<double>(reference_index(
                                    {d1, d2, d3, d4, d5, d6}, {i, j, k, l, m, n}));

        bool ok = true;
        for (std::size_t n = 0; n < t.data.size(); ++n)
            ok = ok && (t.data[n] == static_cast<double>(n));
        check(ok, "Tensor6D flat layout matches row-major reference (2x3x4x5x6x7)");
    }

    // A transposed read must land somewhere different. If the two innermost extents were
    // ever swapped in the flat form, this is what would catch it.
    void test_tensor6d_axis_order_is_observable()
    {
        Tensor6D t(2, 3, 4, 5, 6, 7, 0.0);
        t(0, 0, 0, 0, 1, 0) = 1.0;
        t(0, 0, 0, 0, 0, 1) = 2.0;
        check(t(0, 0, 0, 0, 1, 0) == 1.0 && t(0, 0, 0, 0, 0, 1) == 2.0,
              "Tensor6D distinguishes adjacent axes (no silent transpose)");
    }

    // The runtime-rank types are what the generated arbitrary-order (rank >= 4) kernels
    // index through, exclusively via braced lists. Their inlined initializer_list overload
    // must agree with the out-of-line vector<int> overload, which still routes through
    // flatten_index -- so this cross-checks the two implementations against each other.
    void test_nd_initializer_list_matches_vector_overload()
    {
        const std::vector<int> dims{2, 3, 4, 5};
        HartreeFock::Correlation::CC::TensorND t(dims, 0.0);
        for (int i = 0; i < dims[0]; ++i)
            for (int j = 0; j < dims[1]; ++j)
                for (int k = 0; k < dims[2]; ++k)
                    for (int l = 0; l < dims[3]; ++l)
                        t({i, j, k, l}) =
                            static_cast<double>(reference_index(dims, {i, j, k, l}));

        bool flat_ok = true;
        for (std::size_t n = 0; n < t.data.size(); ++n)
            flat_ok = flat_ok && (t.data[n] == static_cast<double>(n));
        check(flat_ok, "TensorND braced-index layout matches row-major reference (2x3x4x5)");

        bool overloads_agree = true;
        for (int i = 0; i < dims[0]; ++i)
            for (int j = 0; j < dims[1]; ++j)
                for (int k = 0; k < dims[2]; ++k)
                    for (int l = 0; l < dims[3]; ++l)
                    {
                        const std::vector<int> idx{i, j, k, l};
                        overloads_agree = overloads_agree && (t({i, j, k, l}) == t(idx));
                    }
        check(overloads_agree,
              "TensorND initializer_list overload agrees with vector<int> overload");
    }

    void test_dense_view_matches_owner()
    {
        const std::vector<int> dims{2, 3, 4, 5};
        HartreeFock::Correlation::CC::TensorND t(dims, 0.0);
        for (int i = 0; i < dims[0]; ++i)
            for (int j = 0; j < dims[1]; ++j)
                for (int k = 0; k < dims[2]; ++k)
                    for (int l = 0; l < dims[3]; ++l)
                        t({i, j, k, l}) =
                            static_cast<double>(reference_index(dims, {i, j, k, l}));

        auto view = HartreeFock::Correlation::CC::make_tensor_view(t);
        const auto const_view =
            HartreeFock::Correlation::CC::make_tensor_view(static_cast<const decltype(t) &>(t));

        bool ok = true;
        for (int i = 0; i < dims[0]; ++i)
            for (int j = 0; j < dims[1]; ++j)
                for (int k = 0; k < dims[2]; ++k)
                    for (int l = 0; l < dims[3]; ++l)
                    {
                        const double want = t({i, j, k, l});
                        ok = ok && view({i, j, k, l}) == want &&
                             const_view({i, j, k, l}) == want;
                    }
        check(ok, "Dense/ConstDenseTensorView braced-index matches owning TensorND");
    }

    // const and non-const overloads must agree.
    void test_const_overload_agrees()
    {
        Tensor6D t(2, 3, 4, 5, 6, 7, 0.0);
        t(1, 2, 3, 4, 5, 6) = 42.0;
        const Tensor6D &ct = t;
        check(ct(1, 2, 3, 4, 5, 6) == 42.0, "Tensor6D const overload matches non-const");
    }
} // namespace

int main()
{
    test_tensor2d_layout();
    test_tensor4d_layout();
    test_tensor6d_layout();
    test_tensor6d_axis_order_is_observable();
    test_nd_initializer_list_matches_vector_overload();
    test_dense_view_matches_owner();
    test_const_overload_agrees();

    if (failures == 0)
        std::printf("[PASS] cc tensor index layout\n");
    return failures == 0 ? 0 : 1;
}
