// Unit tests for the shared DIIS coefficient solve (HartreeFock::solve_diis_coefficients).
//
// The guard must be a no-op on well-behaved subspaces (returning the same
// coefficients the bare bordered solve produces) and must recover gracefully
// when the linear solve would otherwise produce non-finite or explosively
// large coefficients.

#include <cstdio>
#include <vector>

#include <Eigen/Core>

#include "base/types.h"

namespace
{
    int g_failures = 0;

    bool expect(bool cond, const char *msg)
    {
        if (!cond)
        {
            std::printf("  [FAIL] %s\n", msg);
            ++g_failures;
        }
        return cond;
    }

    bool near(double a, double b, double tol = 1e-10)
    {
        return std::abs(a - b) <= tol;
    }

    // Reference bordered solve with no guard, matching the original code path.
    Eigen::VectorXd bare_solve(const std::vector<Eigen::MatrixXd> &errs)
    {
        const Eigen::Index m = static_cast<Eigen::Index>(errs.size());
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(m + 1, m + 1);
        for (Eigen::Index i = 0; i < m; ++i)
        {
            for (Eigen::Index j = i; j < m; ++j)
            {
                const double bij = (errs[i].array() * errs[j].array()).sum();
                B(i, j) = bij;
                B(j, i) = bij;
            }
            B(i, m) = -1.0;
            B(m, i) = -1.0;
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m + 1);
        rhs(m) = -1.0;
        return B.colPivHouseholderQr().solve(rhs).head(m);
    }
}

int main()
{
    // 1) Well-conditioned subspace: guard must return the same coefficients as
    //    the bare solve, and they must sum to 1.
    {
        std::vector<Eigen::MatrixXd> errs;
        Eigen::MatrixXd e0(2, 2);
        e0 << 0.30, -0.10, -0.10, 0.20;
        Eigen::MatrixXd e1(2, 2);
        e1 << -0.05, 0.04, 0.04, -0.03;
        Eigen::MatrixXd e2(2, 2);
        e2 << 0.012, -0.009, -0.009, 0.008;
        errs = {e0, e1, e2};

        std::vector<const Eigen::MatrixXd *> ptrs;
        for (const auto &e : errs)
            ptrs.push_back(&e);

        const Eigen::VectorXd c = HartreeFock::solve_diis_coefficients(ptrs);
        const Eigen::VectorXd ref = bare_solve(errs);

        expect(c.size() == 3, "well-conditioned: one coefficient per vector");
        expect(near(c.sum(), 1.0), "well-conditioned: DIIS coefficients sum to 1");
        bool same = true;
        for (Eigen::Index i = 0; i < c.size(); ++i)
            same = same && near(c(i), ref(i), 1e-9);
        expect(same, "well-conditioned: guard matches the bare bordered solve exactly");
    }

    // 2) Near-converged parallel errors must still be minimized, not merely
    //    passed through the old unscaled QR (which loses the objective's rank).
    {
        Eigen::MatrixXd base(2, 2);
        base << 1.0, 0.5, 0.5, 1.0;
        std::vector<Eigen::MatrixXd> errs = {
            1e-10 * base, 9.9e-11 * base, 9.8e-11 * base};
        std::vector<const Eigen::MatrixXd *> ptrs;
        for (const auto &e : errs)
            ptrs.push_back(&e);

        const Eigen::VectorXd c = HartreeFock::solve_diis_coefficients(ptrs);
        expect(c.allFinite(), "near-converged: coefficients are finite");
        expect(near(c.sum(), 1.0, 1e-6), "near-converged: coefficients sum to 1");
        const double residual = c(0) + .99*c(1) + .98*c(2);
        expect(std::abs(residual) < 1e-8,
               "near-converged: cancel the normalized residual, not just satisfy sum(c)=1");
    }

    // 3) Duplicated vector creates an exactly singular Gram block. The bare
    //    solve is not unique; any finite bounded minimizing coefficient set
    //    satisfying the constraint is acceptable.
    {
        Eigen::MatrixXd e0(2, 2);
        e0 << 0.20, 0.00, 0.00, 0.10;
        Eigen::MatrixXd e1(2, 2);
        e1 << 0.05, 0.00, 0.00, 0.04;
        // e2 is an exact copy of e1 -> two identical rows/cols in G.
        Eigen::MatrixXd e2 = e1;
        std::vector<Eigen::MatrixXd> errs = {e0, e1, e2};
        std::vector<const Eigen::MatrixXd *> ptrs;
        for (const auto &e : errs)
            ptrs.push_back(&e);

        const Eigen::VectorXd c = HartreeFock::solve_diis_coefficients(ptrs);
        expect(c.size() == 3, "singular: one coefficient per input vector (some may be zero)");
        expect(c.allFinite(), "singular: coefficients remain finite despite singular Gram block");
        expect(near(c.sum(), 1.0, 1e-8), "singular: coefficients still sum to 1");
        expect(c.cwiseAbs().maxCoeff() < 1e8, "singular: no explosive coefficients");
    }

    // 4) Exact two-vector oracle: min ||2*c0*e + c1*e|| with c0+c1=1
    //    has the unique solution (-1,2), for every nonzero common scale.
    //    Includes scales whose unnormalized Gram underflows/overflows.
    for (double scale : {1.0, 1e-8, 1e-10, 1e-100, 1e-200, 1e100, 1e200})
    {
        Eigen::MatrixXd e0 = Eigen::MatrixXd::Constant(1, 1, 2*scale);
        Eigen::MatrixXd e1 = Eigen::MatrixXd::Constant(1, 1, scale);
        const auto c = HartreeFock::solve_diis_coefficients({&e0, &e1});
        expect(near(c(0), -1., 1e-11) && near(c(1), 2., 1e-11),
               "scale invariance: two-vector cancellation gives (-1,2)");
        expect(std::abs(2*c(0)+c(1)) < 1e-11, "scale invariance: residual is minimized");
        if (scale == 1e-10)
        {
            const auto old = bare_solve({e0, e1});
            std::printf("tiny-error oracle: old normalized residual %.12e, fixed %.12e\n",
                        std::abs(2*old(0)+old(1)), std::abs(2*c(0)+c(1)));
            expect(std::abs(2*old(0)+old(1)) > .1,
                   "regression reproducer: unscaled QR loses the tiny objective");
        }
    }

    // 5) A coupled spin oracle with an analytic minimizer. For c0=t,c1=1-t,
    //    ||e_a||^2+||e_b||^2 = (1+t)^2+(-1+2t)^2, minimized at t=1/5.
    //    Independent spin solves instead give t=-1 and t=1/2.
    for (double scale : {1., 1e-10})
    {
        auto scalar = [](double x) { return Eigen::MatrixXd::Constant(1, 1, x).eval(); };
        HartreeFock::UHFDIISState spin, swapped, empty_beta;
        spin.push(scalar(10), scalar(30), scalar(2*scale), scalar(scale));
        spin.push(scalar(20), scalar(50), scalar(scale), scalar(-scale));
        const auto [fa, fb] = spin.extrapolate();
        expect(near(fa(0,0), 18) && near(fb(0,0), 46),
               "coupled spin: one minimizing coefficient vector for both Focks");
        swapped.push(scalar(30), scalar(10), scalar(scale), scalar(2*scale));
        swapped.push(scalar(50), scalar(20), scalar(-scale), scalar(scale));
        const auto [sa, sb] = swapped.extrapolate();
        expect(near(sa(0,0), fb(0,0)) && near(sb(0,0), fa(0,0)),
               "coupled spin: alpha/beta exchange invariance");
        empty_beta.push(scalar(10), scalar(30), scalar(2*scale), scalar(0));
        empty_beta.push(scalar(20), scalar(50), scalar(scale), scalar(0));
        const auto [za, zb] = empty_beta.extrapolate();
        expect(near(za(0,0), 30) && near(zb(0,0), 70),
               "coupled spin: zero-beta residual reduces to alpha minimization");
        spin.clear();
        expect(spin.size() == 0 && !spin.ready(), "coupled spin: reset clears both histories");
    }

    // 6) Zero residuals and final-subspace breakdown have deterministic,
    //    finite fallbacks. The coefficient magnitude guard also applies at k=2.
    {
        Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(2, 2);
        const auto c = HartreeFock::solve_diis_coefficients({&zero, &zero});
        expect(c(0) == 0 && c(1) == 1, "zero errors: use the newest Fock");
        Eigen::MatrixXd e0 = Eigen::MatrixXd::Constant(1, 1, 1.001);
        Eigen::MatrixXd e1 = Eigen::MatrixXd::Constant(1, 1, 1.0);
        const auto guarded = HartreeFock::solve_diis_coefficients({&e0, &e1}, 10.);
        expect(guarded(0) == 0 && guarded(1) == 1,
               "two-vector breakdown: do not return explosive coefficients");
    }

    if (g_failures == 0)
        std::printf("[PASS] diis-conditioning: all checks passed\n");
    else
        std::printf("[FAIL] diis-conditioning: %d check(s) failed\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
