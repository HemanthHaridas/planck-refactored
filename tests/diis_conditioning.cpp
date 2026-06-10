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

    // 2) Near-converged subspace (tiny, nearly parallel errors): the Gram block
    //    is extremely ill-conditioned but still positive — the guard must NOT
    //    drop vectors here; it should pass the bare solve through unchanged.
    {
        Eigen::MatrixXd base(2, 2);
        base << 1.0, 0.5, 0.5, 1.0;
        std::vector<Eigen::MatrixXd> errs = {
            1e-10 * base, 9.9e-11 * base, 9.8e-11 * base};
        std::vector<const Eigen::MatrixXd *> ptrs;
        for (const auto &e : errs)
            ptrs.push_back(&e);

        const Eigen::VectorXd c = HartreeFock::solve_diis_coefficients(ptrs);
        const Eigen::VectorXd ref = bare_solve(errs);
        expect(c.allFinite(), "near-converged: coefficients are finite");
        expect(near(c.sum(), 1.0, 1e-6), "near-converged: coefficients sum to 1");
        bool same = true;
        for (Eigen::Index i = 0; i < c.size(); ++i)
            same = same && near(c(i), ref(i), 1e-6);
        expect(same, "near-converged: benign ill-conditioning is passed through unchanged");
    }

    // 3) Duplicated vector creates an exactly singular Gram block. The bare
    //    solve would be unreliable; the guard must drop the oldest vector and
    //    still return a finite, sum-to-1 coefficient set.
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

    if (g_failures == 0)
        std::printf("[PASS] diis-conditioning: all checks passed\n");
    else
        std::printf("[FAIL] diis-conditioning: %d check(s) failed\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
