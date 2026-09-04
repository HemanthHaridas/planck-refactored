// X5.0 gate: the hand-written RCCSD solver's spin-orbital amplitudes ->
// spatial RCC projection (project_rccsd_amplitudes_to_spatial,
// src/post_hf/cc/amplitudes.cpp).
//
// The relation this pins was derived by numerically inspecting this
// codebase's own converged BH3/STO-3G RCCSD amplitudes (see
// docs/CC_AMPLITUDE_CHECKPOINT_SCOPE.md X5.0), not assumed from theory alone:
//   t1_spatial(i,a)     = t1_so(2i,   2a)
//   t2_spatial(i,j,a,b) = t2_so(2i, 2j+1, 2a, 2b+1)
// with the closed-shell identities t1_alpha == t1_beta, cross-spin t1 blocks
// exactly zero, and t2_aa = t2_ab - t2_ab.swap(a,b) all holding at
// interleaved spin-orbital indexing (spatial index p/2, spin p%2).
//
// This test builds a SYNTHETIC spin-orbital amplitude set that respects
// those closed-shell identities by construction (not a real converged CC
// solve -- that is what the end-to-end regression case, once X5.1 wires the
// write path, checks against a live run) and verifies the projection reads
// back the exact spatial values used to build it. It also checks the error
// path on malformed (odd) spin-orbital dims.

#include <cassert>
#include <cmath>
#include <iostream>

#include "post_hf/cc/amplitudes.h"

using namespace HartreeFock::Correlation::CC;

namespace
{
    constexpr double kTol = 1e-14;

    bool close(double a, double b, double tol = kTol)
    {
        return std::abs(a - b) <= tol;
    }

    // Build a spin-orbital RCCSDAmplitudes at n_occ spatial occupied / n_virt
    // spatial virtual, from arbitrary "true" spatial t1/t2 values, enforcing
    // the closed-shell relation exactly -- this is what a genuinely converged
    // RHF-reference CCSD solve produces, and is the fixture the projection
    // must invert.
    RCCSDAmplitudes make_closed_shell_so_amplitudes(
        int n_occ, int n_virt,
        const std::vector<double> &t1_spatial_flat, // [n_occ * n_virt]
        const std::vector<double> &t2_spatial_flat  // [n_occ^2 * n_virt^2], (i,j,a,b)
    )
    {
        const int n_occ_so = 2 * n_occ;
        const int n_virt_so = 2 * n_virt;

        auto t1_spatial = [&](int i, int a) {
            return t1_spatial_flat[static_cast<std::size_t>(i * n_virt + a)];
        };
        auto t2_spatial = [&](int i, int j, int a, int b) {
            return t2_spatial_flat[static_cast<std::size_t>(
                ((i * n_occ + j) * n_virt + a) * n_virt + b)];
        };

        RCCSDAmplitudes so;
        so.t1 = Tensor2D(n_occ_so, n_virt_so, 0.0);
        so.t2 = Tensor4D(n_occ_so, n_occ_so, n_virt_so, n_virt_so, 0.0);

        // t1: both spin channels equal the spatial value; cross-spin blocks
        // stay zero (spin selection rule).
        for (int i = 0; i < n_occ; ++i)
            for (int a = 0; a < n_virt; ++a)
            {
                const double v = t1_spatial(i, a);
                so.t1(2 * i, 2 * a) = v;         // alpha-alpha
                so.t1(2 * i + 1, 2 * a + 1) = v; // beta-beta
            }

        // t2: the opposite-spin (ab) block IS the spatial tensor. `(i,j,a,b)`
        // ranges over the full domain, so `(2i, 2j+1, 2a, 2b+1)` already
        // covers every opposite-spin spin-orbital index exactly once -- no
        // separate write for the "ba" partner is needed, and adding one is a
        // bug: a caller building this fixture must supply a `t2_spatial`
        // that ALREADY satisfies t2(i,j,a,b) = t2(j,i,b,a) (a real property
        // of the spatial RCC t2, verified on the codebase's own converged
        // BH3/STO-3G amplitudes), or the two independent passes over
        // (i,j,a,b) and (j,i,b,a) disagree and whichever runs last wins --
        // caught by asserts-enabled testing (Release's -DNDEBUG silently
        // hides it), not by inspection. The same-spin blocks are the
        // DEPENDENT combination t2_aa(i,j,a,b) = t2_ab(i,j,a,b) -
        // t2_ab(i,j,b,a), and t2_bb = t2_aa.
        for (int i = 0; i < n_occ; ++i)
            for (int j = 0; j < n_occ; ++j)
                for (int a = 0; a < n_virt; ++a)
                    for (int b = 0; b < n_virt; ++b)
                        so.t2(2 * i, 2 * j + 1, 2 * a, 2 * b + 1) = t2_spatial(i, j, a, b);
        for (int i = 0; i < n_occ; ++i)
            for (int j = 0; j < n_occ; ++j)
                for (int a = 0; a < n_virt; ++a)
                    for (int b = 0; b < n_virt; ++b)
                    {
                        const double v_aa = t2_spatial(i, j, a, b) - t2_spatial(i, j, b, a);
                        so.t2(2 * i, 2 * j, 2 * a, 2 * b) = v_aa;             // aaaa
                        so.t2(2 * i + 1, 2 * j + 1, 2 * a + 1, 2 * b + 1) = v_aa; // bbbb
                    }
        return so;
    }
} // namespace

int main()
{
    // n_occ = 2, n_virt = 3 -- big enough to exercise every off-diagonal
    // combination, matching the checkpoint round-trip gate's own fixture
    // size for consistency.
    const int n_occ = 2, n_virt = 3;

    std::vector<double> t1_flat(static_cast<std::size_t>(n_occ * n_virt));
    for (std::size_t i = 0; i < t1_flat.size(); ++i)
        t1_flat[i] = 0.01 * static_cast<double>(i) - 0.02;

    // t2(i,j,a,b) = t2(j,i,b,a) is a real property of the spatial RCC t2
    // (verified on this codebase's own converged BH3/STO-3G amplitudes to
    // ~1e-17), not an arbitrary choice -- an unconstrained fill here would
    // make the fixture itself unphysical and mask the earlier bug where the
    // spin-orbital builder's now-removed redundant "ba" write depended on
    // exactly this symmetry holding.
    auto t2_index = [&](int i, int j, int a, int b) {
        return static_cast<std::size_t>(((i * n_occ + j) * n_virt + a) * n_virt + b);
    };
    std::vector<double> t2_flat(
        static_cast<std::size_t>(n_occ * n_occ * n_virt * n_virt));
    for (int i = 0; i < n_occ; ++i)
        for (int j = 0; j < n_occ; ++j)
            for (int a = 0; a < n_virt; ++a)
                for (int b = 0; b < n_virt; ++b)
                {
                    const double v = -0.003 * static_cast<double>(t2_index(i, j, a, b)) + 0.05;
                    t2_flat[t2_index(i, j, a, b)] = v;
                    t2_flat[t2_index(j, i, b, a)] = v;
                }

    const RCCSDAmplitudes so_amps =
        make_closed_shell_so_amplitudes(n_occ, n_virt, t1_flat, t2_flat);

    auto projected = project_rccsd_amplitudes_to_spatial(so_amps);
    assert(projected && "projection should succeed on well-formed input");

    assert(projected->by_rank.size() == 2);
    assert(projected->by_rank[0].dims == (std::vector<int>{n_occ, n_virt}));
    assert(projected->by_rank[1].dims == (std::vector<int>{n_occ, n_occ, n_virt, n_virt}));

    // t1: exact recovery of the seeded spatial values.
    for (int i = 0; i < n_occ; ++i)
        for (int a = 0; a < n_virt; ++a)
        {
            const double expected = t1_flat[static_cast<std::size_t>(i * n_virt + a)];
            const double actual = projected->by_rank[0]({i, a});
            assert(close(actual, expected) && "t1 projection mismatch");
        }

    // t2: exact recovery of the seeded spatial values.
    for (int i = 0; i < n_occ; ++i)
        for (int j = 0; j < n_occ; ++j)
            for (int a = 0; a < n_virt; ++a)
                for (int b = 0; b < n_virt; ++b)
                {
                    const double expected = t2_flat[static_cast<std::size_t>(
                        ((i * n_occ + j) * n_virt + a) * n_virt + b)];
                    const double actual = projected->by_rank[1]({i, j, a, b});
                    assert(close(actual, expected) && "t2 projection mismatch");
                }

    // Error path: odd spin-orbital dims must be rejected, not silently
    // truncated -- a real caller passing a malformed reference should see a
    // message naming the problem, not a wrong-shaped result.
    {
        RCCSDAmplitudes bad;
        bad.t1 = Tensor2D(3, 4, 0.0); // odd n_occ_so
        bad.t2 = Tensor4D(3, 3, 4, 4, 0.0);
        auto bad_res = project_rccsd_amplitudes_to_spatial(bad);
        assert(!bad_res && "odd spin-orbital dims must be rejected");
    }

    // Error path: t1/t2 dims disagreeing with each other must also be
    // rejected.
    {
        RCCSDAmplitudes mismatched;
        mismatched.t1 = Tensor2D(4, 6, 0.0);
        mismatched.t2 = Tensor4D(2, 2, 6, 6, 0.0); // dim1/dim2 do not match t1's n_occ_so
        auto mismatched_res = project_rccsd_amplitudes_to_spatial(mismatched);
        assert(!mismatched_res && "t1/t2 dim disagreement must be rejected");
    }

    std::cout << "cc_spatial_amplitude_projection: all checks passed\n";
    return 0;
}
