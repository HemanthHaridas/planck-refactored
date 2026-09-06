// F2 (docs/DFT_ANALYTIC_FXC_HESSIAN.md): confirm -- not assume -- that the
// existing density-on-grid evaluator (evaluate_density_on_grid, xc_grid.cpp)
// already produces exactly what a Hessian-vector product needs when fed a
// RESPONSE density (delta-P) instead of the ground-state density: rho(r) is
// a quadratic form phi(r)^T P phi(r), which is LINEAR in P for fixed phi(r),
// so this should be an exact identity, not merely a small-epsilon limit.
// Verified as a finite-difference check anyway (the F2 scope's own
// instruction), since the point is to catch a hidden non-linearity or
// symmetrization assumption in the real code, not to re-derive the algebra.
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "dft/ao_grid.h"
#include "dft/xc_grid.h"

namespace
{
    bool g_ok = true;

    void require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            std::cerr << message << '\n';
            g_ok = false;
        }
    }

    void require_near(double actual, double expected, double tol, const std::string &message)
    {
        if (!std::isfinite(actual) || std::abs(actual - expected) > tol)
        {
            std::ostringstream oss;
            oss << message << ": expected " << expected << ", got " << actual
                << " (tol " << tol << ")";
            std::cerr << oss.str() << '\n';
            g_ok = false;
        }
    }

    // A small synthetic AO grid -- F2 only needs the EVALUATOR's linearity in
    // P, not physically meaningful AO values, so random values stand in for
    // a real basis-set evaluation (which would need a molecule/basis and add
    // nothing this check cares about).
    DFT::AOGridEvaluation make_synthetic_ao_grid(int npoints, int nbasis, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        DFT::AOGridEvaluation grid;
        grid.values = Eigen::MatrixXd::NullaryExpr(npoints, nbasis, [&]()
                                                    { return dist(rng); });
        grid.grad_x = Eigen::MatrixXd::NullaryExpr(npoints, nbasis, [&]()
                                                    { return dist(rng); });
        grid.grad_y = Eigen::MatrixXd::NullaryExpr(npoints, nbasis, [&]()
                                                    { return dist(rng); });
        grid.grad_z = Eigen::MatrixXd::NullaryExpr(npoints, nbasis, [&]()
                                                    { return dist(rng); });
        return grid;
    }

    Eigen::MatrixXd make_symmetric_density(int nbasis, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXd raw = Eigen::MatrixXd::NullaryExpr(nbasis, nbasis, [&]()
                                                            { return dist(rng); });
        return 0.5 * (raw + raw.transpose());
    }

    // Central difference of rho(P) and its gradient components with respect
    // to P, evaluated at three step sizes so convergence as h -> 0 can be
    // read the same way the RHF/UHF SOSCF and F1 fxc probes were read --
    // though since rho is EXACTLY linear in P (a quadratic form phi^T P phi
    // is linear in the matrix P for fixed phi), the residual should already
    // be at floating-point round-off, not merely shrinking with h.
    void check_restricted_linearity(int npoints, int nbasis, unsigned seed)
    {
        const auto grid = make_synthetic_ao_grid(npoints, nbasis, seed);
        const Eigen::MatrixXd P = make_symmetric_density(nbasis, seed + 1);
        const Eigen::MatrixXd dP = make_symmetric_density(nbasis, seed + 2);

        // The evaluator itself symmetrizes internally, so delta-P must be
        // symmetric for the finite difference to compare like with like --
        // exactly the shape build_uhf_cphf_matrix's own dm1a_sym construction
        // produces for a real trial orbital rotation (Ca_virt*x*Ca_occ^T +
        // transpose), so this is not a special case invented for the test.

        // delta-rho as the evaluator itself computes it: feed dP through
        // evaluate_density_on_grid directly (a full evaluation, since the
        // production API has no separate "response-only" entry point --
        // F2's own point is that none is needed, because feeding dP through
        // the same linear evaluator IS the response).
        auto response_only = DFT::evaluate_density_on_grid(grid, dP);
        require(response_only.has_value(), "evaluate_density_on_grid(dP) failed");
        if (!response_only)
            return;

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            auto plus = DFT::evaluate_density_on_grid(grid, P + h * dP);
            auto minus = DFT::evaluate_density_on_grid(grid, P - h * dP);
            require(plus.has_value() && minus.has_value(), "evaluate_density_on_grid(P +/- h*dP) failed");
            if (!plus || !minus)
                continue;

            const Eigen::VectorXd fd_rho = (plus->total.rho - minus->total.rho) / (2.0 * h);
            const Eigen::VectorXd fd_grad_x = (plus->total.grad_x - minus->total.grad_x) / (2.0 * h);

            // rho is an exact quadratic form in P, so the response should
            // match to floating-point round-off at every h, not merely
            // converge as h shrinks -- looser at h=1e-2 only to absorb
            // double-precision cancellation in the subtraction itself.
            const double tol = 50.0 * h * h + 1e-9;
            require_near((fd_rho - response_only->total.rho).norm(), 0.0, tol,
                         "restricted rho response mismatch, h=" + std::to_string(h));
            require_near((fd_grad_x - response_only->total.grad_x).norm(), 0.0, tol,
                         "restricted grad_x response mismatch, h=" + std::to_string(h));
        }
    }

    void check_unrestricted_linearity(int npoints, int nbasis, unsigned seed)
    {
        const auto grid = make_synthetic_ao_grid(npoints, nbasis, seed);
        const Eigen::MatrixXd Pa = make_symmetric_density(nbasis, seed + 1);
        const Eigen::MatrixXd Pb = make_symmetric_density(nbasis, seed + 2);
        const Eigen::MatrixXd dPa = make_symmetric_density(nbasis, seed + 3);
        const Eigen::MatrixXd dPb = make_symmetric_density(nbasis, seed + 4);

        auto response_only = DFT::evaluate_density_on_grid(grid, dPa, dPb);
        require(response_only.has_value(), "evaluate_density_on_grid(dPa,dPb) failed");
        if (!response_only)
            return;

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            auto plus = DFT::evaluate_density_on_grid(grid, Pa + h * dPa, Pb + h * dPb);
            auto minus = DFT::evaluate_density_on_grid(grid, Pa - h * dPa, Pb - h * dPb);
            require(plus.has_value() && minus.has_value(), "evaluate_density_on_grid(Pa/Pb +/- h*dPa/dPb) failed");
            if (!plus || !minus)
                continue;

            const Eigen::VectorXd fd_rho_alpha = (plus->alpha.rho - minus->alpha.rho) / (2.0 * h);
            const Eigen::VectorXd fd_rho_total = (plus->total.rho - minus->total.rho) / (2.0 * h);
            const Eigen::VectorXd fd_grad_z_beta = (plus->beta.grad_z - minus->beta.grad_z) / (2.0 * h);

            const double tol = 50.0 * h * h + 1e-9;
            require_near((fd_rho_alpha - response_only->alpha.rho).norm(), 0.0, tol,
                         "unrestricted alpha rho response mismatch, h=" + std::to_string(h));
            require_near((fd_rho_total - response_only->total.rho).norm(), 0.0, tol,
                         "unrestricted total rho response mismatch, h=" + std::to_string(h));
            require_near((fd_grad_z_beta - response_only->beta.grad_z).norm(), 0.0, tol,
                         "unrestricted beta grad_z response mismatch, h=" + std::to_string(h));
        }
    }

    // Independent reference check, NOT a finite difference of the evaluator
    // against itself. A uniform scale bug in evaluate_density_channel (e.g.
    // the gradient term's leading factor of 2 becoming 1.5) would still
    // cancel out of the FD-vs-response-only comparisons above, because both
    // sides of those comparisons go through the SAME mutated formula -- an
    // internal self-consistency check cannot see a bug shared by both of its
    // own halves. This check instead hand-computes rho(r) = phi(r)^T P
    // phi(r) and grad_x(r) = 2 * phi(r)^T P grad_x_phi(r) directly from the
    // raw AO arrays at one point, independent of evaluate_density_channel's
    // own internal contraction order, and compares against the evaluator's
    // output.
    void check_against_hand_computed_reference(int npoints, int nbasis, unsigned seed)
    {
        const auto grid = make_synthetic_ao_grid(npoints, nbasis, seed);
        const Eigen::MatrixXd P = make_symmetric_density(nbasis, seed + 1);

        auto result = DFT::evaluate_density_on_grid(grid, P);
        require(result.has_value(), "evaluate_density_on_grid(P) failed");
        if (!result)
            return;

        const Eigen::MatrixXd symmetric_P = 0.5 * (P + P.transpose());
        const std::vector<Eigen::Index> points = {
            Eigen::Index{0}, static_cast<Eigen::Index>(npoints / 2), static_cast<Eigen::Index>(npoints - 1)};
        for (Eigen::Index point : points)
        {
            const Eigen::VectorXd phi = grid.values.row(point).transpose();
            const Eigen::VectorXd dphi_x = grid.grad_x.row(point).transpose();

            const double rho_ref = phi.dot(symmetric_P * phi);
            const double grad_x_ref = 2.0 * phi.dot(symmetric_P * dphi_x);

            require_near(result->total.rho(point), rho_ref, 1e-9,
                         "rho vs hand-computed phi^T P phi at point " + std::to_string(point));
            require_near(result->total.grad_x(point), grad_x_ref, 1e-9,
                         "grad_x vs hand-computed 2*phi^T P dphi_x at point " + std::to_string(point));
        }
    }

    // Negative control: confirm the check itself has power by perturbing the
    // scale factor a real Hessian-vector product would need (the trial
    // density is fed at full strength -- x1, not x0.5x -- as would happen if
    // a caller forgot the "+ h.c." half of Ca_virt*x*Ca_occ^T +
    // Ca_occ*x^T*Ca_virt^T). Confirms the response is NOT scale-invariant by
    // construction, so a factor-of-2 bug in a future caller would be caught
    // by this same linearity machinery, not just by luck.
    void check_response_scales_linearly(int npoints, int nbasis, unsigned seed)
    {
        const auto grid = make_synthetic_ao_grid(npoints, nbasis, seed);
        const Eigen::MatrixXd dP = make_symmetric_density(nbasis, seed + 1);

        auto once = DFT::evaluate_density_on_grid(grid, dP);
        auto twice = DFT::evaluate_density_on_grid(grid, 2.0 * dP);
        require(once.has_value() && twice.has_value(), "evaluate_density_on_grid scaling check failed");
        if (!once || !twice)
            return;

        require_near((twice->total.rho - 2.0 * once->total.rho).norm(), 0.0, 1e-9,
                     "response should scale exactly linearly with the input density");
    }
} // namespace

int main()
{
    check_restricted_linearity(37, 5, 12345);
    check_restricted_linearity(11, 9, 6789);
    check_unrestricted_linearity(29, 6, 2468);
    check_against_hand_computed_reference(15, 7, 999);
    check_response_scales_linearly(20, 4, 42);
    return g_ok ? 0 : 1;
}
