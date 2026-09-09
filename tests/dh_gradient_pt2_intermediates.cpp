// N1 check for DFT::Gradient::build_pt2_mo_intermediates.
//
// build_pt2_mo_intermediates is a thin named entry point over
// HartreeFock::Correlation::build_rmp2_lagrangian (N1.3.4), whose value
// correctness is gated end to end by the RMP2 gradient / geomopt regression
// suite. build_rmp2_lagrangian needs a real basis (it calls _compute_2e_fock
// / ensure_eri), so a synthetic RMP2Result can only exercise the input
// guards that return before any integral work. The value comparison against
// build_rmp2_gradient_intermediates lives in N1.4 (a real water/STO-3G run).
//
// Explicit CHECK, not assert(): this tree builds Release with -DNDEBUG.

#include <cstdio>
#include <vector>

#include <Eigen/Dense>

#include "dft/dh_gradient.h"
#include "post_hf/mp2.h"

using HartreeFock::Correlation::RMP2Result;

namespace
{
    int g_failures = 0;

    void check_impl(bool ok, const char *expr, int line)
    {
        if (!ok)
        {
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, line, expr);
            ++g_failures;
        }
    }
}

#define CHECK(cond) check_impl((cond), #cond, __LINE__)

int main()
{
    HartreeFock::Calculator calc{};
    const std::vector<HartreeFock::ShellPair> no_pairs;

    // Empty T2 -> rejected before any orbital / integral access.
    {
        RMP2Result r;
        r.n_occ = 1;
        r.n_virt = 2;
        auto res = DFT::Gradient::build_pt2_mo_intermediates(calc, no_pairs, r, 1.0);
        CHECK(!res);
        // scale argument must not change the guard behaviour
        auto res0 = DFT::Gradient::build_pt2_mo_intermediates(calc, no_pairs, r, 0.0);
        CHECK(!res0);
    }

    if (g_failures)
    {
        std::fprintf(stderr, "dh_gradient_pt2_intermediates: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("dh_gradient_pt2_intermediates: OK");
    return 0;
}
