// D2.0 (docs/SOSCF_UHF_DFT_SCOPE.md): promotes F3's point-level-verified
// T1(+T2+T3) algebra into a real production function,
// compute_analytic_xc_hessian_vector_product (src/dft/analytic_hessian.{h,cpp}).
//
// This is NOT a re-derivation of F3's formulas (already verified in
// tests/dft_fxc_selfcheck.cpp and tests/dft_gga_hessian_selfcheck.cpp) --
// it is a check that the NEW function's actual call signature (taking a
// MolecularGrid + AOGridEvaluation + AO-basis density matrices, not bare
// scalars) reproduces those same formulas once real AO/grid machinery is
// involved. A transcription bug in argument order, indexing, or a dropped
// functional contribution would not be visible to F3's own scalar-level
// tests, since they never called a function shaped like this one.
//
// Uses a SYNTHETIC single-point "grid" with hand-picked AO values/gradients
// (no real basis set or molecule) -- following F2's own precedent
// (tests/dft_density_response_linearity.cpp) that a synthetic AO grid is
// enough to test a grid-consuming function's CONTRACTION correctness,
// independent of whether the AO values themselves come from a real basis.
//
// FIXTURE DESIGN NOTE, recorded because it cost a real debugging pass:
// the first version of this file tried to SOLVE for a density matrix P
// that would hit pre-chosen (rho, grad_rho) target values at one point.
// That is over-constrained and fails in general: at a single point,
// rho = phi^T*P*phi and grad_rho_k = 2*(grad_phi_k)^T*P*phi both depend on
// P ONLY through the 3-vector v = P*phi, so the map from P to
// (rho, gx, gy, gz) has rank <= 3 structurally, regardless of how large P
// is (verified directly: a 3x3 symmetric P, 6 free parameters, still gives
// a 4-constraint linear system of rank 3, confirmed both algebraically and
// by checking the rank numerically on multiple random AO gradient choices).
// The fix used here is simpler and avoids the issue entirely: pick P and
// dP FREELY (arbitrary, non-degenerate symmetric matrices), then read off
// whatever (rho, grad_rho, drho, dgrad_rho) actually result from the real
// AO contraction, and feed THOSE into the reference formula -- P/dP are
// themselves legitimate arbitrary inputs, so this is not a weaker check,
// only a differently-parameterized one.
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dft/analytic_hessian.h"

namespace
{
    bool g_ok = true;

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

    DFT::XC::Functional require_functional(const std::string &name)
    {
        auto id = DFT::XC::functional_id(name);
        if (!id)
        {
            std::cerr << "functional_id(" << name << ") failed: " << id.error() << '\n';
            g_ok = false;
            return DFT::XC::Functional::create(1, DFT::XC::Spin::Unpolarized).value();
        }
        auto functional = DFT::XC::Functional::create(*id, DFT::XC::Spin::Unpolarized);
        if (!functional)
        {
            std::cerr << "Functional::create(" << name << ") failed: " << functional.error() << '\n';
            g_ok = false;
            return DFT::XC::Functional::create(1, DFT::XC::Spin::Unpolarized).value();
        }
        return std::move(*functional);
    }

    // A single-point "grid" with 3 synthetic AOs (nbasis=3, so P/dP have
    // enough freedom to be genuinely non-degenerate 3x3 symmetric
    // matrices, not because 3 is otherwise required -- see the file header
    // for why hitting a PRE-CHOSEN (rho,grad_rho) needs no fixed nbasis at
    // all, since that approach was abandoned).
    struct SyntheticPoint
    {
        double weight;
        Eigen::Vector3d phi;
        Eigen::Vector3d gphix, gphiy, gphiz;
    };

    DFT::MolecularGrid make_grid(const SyntheticPoint &p)
    {
        DFT::MolecularGrid grid;
        grid.points = Eigen::MatrixXd(1, 4);
        grid.points << 0.0, 0.0, 0.0, p.weight;
        return grid;
    }

    DFT::AOGridEvaluation make_ao_grid(const SyntheticPoint &p)
    {
        DFT::AOGridEvaluation ao;
        ao.values = Eigen::MatrixXd(1, 3);
        ao.values << p.phi(0), p.phi(1), p.phi(2);
        ao.grad_x = Eigen::MatrixXd(1, 3);
        ao.grad_x << p.gphix(0), p.gphix(1), p.gphix(2);
        ao.grad_y = Eigen::MatrixXd(1, 3);
        ao.grad_y << p.gphiy(0), p.gphiy(1), p.gphiy(2);
        ao.grad_z = Eigen::MatrixXd(1, 3);
        ao.grad_z << p.gphiz(0), p.gphiz(1), p.gphiz(2);
        return ao;
    }

    struct DensityResponse
    {
        double rho, gx, gy, gz;
    };

    DensityResponse evaluate_density_response(const SyntheticPoint &p, const Eigen::Matrix3d &P)
    {
        DensityResponse d;
        d.rho = p.phi.transpose() * P * p.phi;
        d.gx = 2.0 * p.phi.transpose() * P * p.gphix;
        d.gy = 2.0 * p.phi.transpose() * P * p.gphiy;
        d.gz = 2.0 * p.phi.transpose() * P * p.gphiz;
        return d;
    }

    // F3.3.3's own reference formula, evaluated directly from the ACTUAL
    // (rho, grad_rho, drho, dgrad_rho) this fixture's P/dP produce -- NOT
    // calling the production function, so a shared bug cannot hide.
    double reference_delta_vxc_projected(
        const DFT::XC::Functional &f, const DensityResponse &ground, const DensityResponse &trial, double AA,
        double AGx, double AGy, double AGz)
    {
        const double sigma = ground.gx * ground.gx + ground.gy * ground.gy + ground.gz * ground.gz;
        std::vector<double> v2rho2, v2rhosigma, v2sigma2;
        f.evaluate_gga_fxc({ground.rho}, {sigma}, 1, v2rho2, v2rhosigma, v2sigma2);
        std::vector<double> exc0, vrho0, vsigma0;
        f.evaluate_gga_exc_vxc({ground.rho}, {sigma}, 1, exc0, vrho0, vsigma0);

        const double g_dot_dg = ground.gx * trial.gx + ground.gy * trial.gy + ground.gz * trial.gz;
        const double g_dot_AG = ground.gx * AGx + ground.gy * AGy + ground.gz * AGz;
        const double dg_dot_AG = trial.gx * AGx + trial.gy * AGy + trial.gz * AGz;

        const double T1 = (v2rho2[0] * trial.rho + 2.0 * v2rhosigma[0] * g_dot_dg) * AA;
        const double T2 = 2.0 * (v2rhosigma[0] * trial.rho + 2.0 * v2sigma2[0] * g_dot_dg) * g_dot_AG;
        const double T3 = 2.0 * vsigma0[0] * dg_dot_AG;
        return T1 + T2 + T3;
    }

    void check_gga_point(const std::string &name, const Eigen::Matrix3d &P, const Eigen::Matrix3d &dP)
    {
        auto f = require_functional(name);

        // Non-degenerate AO values/gradients: no zero components, no
        // proportionality between phi and its gradients, no axis-aligned
        // gradients -- so a row/column transposition or an axis swap in
        // the production code would not accidentally cancel out.
        SyntheticPoint sp{
            /*weight=*/1.0,
            Eigen::Vector3d(1.3, 0.7, -0.4),
            Eigen::Vector3d(0.4, -0.3, 0.2),
            Eigen::Vector3d(-0.2, 0.5, 0.1),
            Eigen::Vector3d(0.1, -0.1, 0.3)};

        const DFT::MolecularGrid grid = make_grid(sp);
        const DFT::AOGridEvaluation ao = make_ao_grid(sp);

        auto delta_v_xc = DFT::Driver::compute_analytic_xc_hessian_vector_product(grid, ao, P, dP, f, f);
        if (!delta_v_xc)
        {
            std::cerr << name << ": compute_analytic_xc_hessian_vector_product failed: " << delta_v_xc.error()
                      << '\n';
            g_ok = false;
            return;
        }

        // Check element (0,1) -- off-diagonal, so a transposition or a
        // basis-index swap in the production AO update is visible.
        const double production_01 = (*delta_v_xc)(0, 1);
        const double production_10 = (*delta_v_xc)(1, 0);
        require_near(production_01, production_10, 1e-12, name + ": delta_V_xc must be symmetric");

        const DensityResponse ground = evaluate_density_response(sp, P);
        const DensityResponse trial = evaluate_density_response(sp, dP);

        const double AA = sp.phi(0) * sp.phi(1);
        const double AGx = sp.phi(0) * sp.gphix(1) + sp.gphix(0) * sp.phi(1);
        const double AGy = sp.phi(0) * sp.gphiy(1) + sp.gphiy(0) * sp.phi(1);
        const double AGz = sp.phi(0) * sp.gphiz(1) + sp.gphiz(0) * sp.phi(1);

        // Production uses the SAME functional for exchange and
        // correlation, so its additive x+c convention doubles every
        // libxc-derived quantity relative to a single evaluate_gga_fxc
        // call -- matched here with an explicit factor of 2.
        const double reference = 2.0 * reference_delta_vxc_projected(f, ground, trial, AA, AGx, AGy, AGz);

        require_near(production_01, reference, 1e-8,
                     name + ": compute_analytic_xc_hessian_vector_product(0,1) vs F3.3.3 reference formula");
    }

    void check_lda_point(const std::string &name, const Eigen::Matrix3d &P, const Eigen::Matrix3d &dP)
    {
        auto f = require_functional(name);
        SyntheticPoint sp{
            /*weight=*/1.0, Eigen::Vector3d(1.3, 0.7, -0.4), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero()};
        const DFT::MolecularGrid grid = make_grid(sp);
        const DFT::AOGridEvaluation ao = make_ao_grid(sp);

        auto delta_v_xc = DFT::Driver::compute_analytic_xc_hessian_vector_product(grid, ao, P, dP, f, f);
        if (!delta_v_xc)
        {
            std::cerr << name << ": compute_analytic_xc_hessian_vector_product failed: " << delta_v_xc.error()
                      << '\n';
            g_ok = false;
            return;
        }

        const double rho0 = sp.phi.transpose() * P * sp.phi;
        const double trial_rho = sp.phi.transpose() * dP * sp.phi;

        std::vector<double> v2rho2;
        f.evaluate_lda_fxc({rho0}, 1, v2rho2);
        const double AA01 = sp.phi(0) * sp.phi(1);
        // Doubled for the x==c convention, same as GGA.
        const double reference = 2.0 * v2rho2[0] * trial_rho * AA01;
        require_near((*delta_v_xc)(0, 1), reference, 1e-10,
                     name + ": compute_analytic_xc_hessian_vector_product(0,1) vs F3.1 reference formula (LDA)");
    }

    // D2.2.0 (docs/SOSCF_UHF_DFT_SCOPE.md): orbital_energy_difference_diagonal
    // is the RKS analogue of the single line
    // `A(ai,ai) += eps(n_occ+a) - eps(i)` inside
    // HartreeFock::Correlation::build_rhf_cphf_matrix
    // (src/post_hf/rhf_response.cpp) -- pure orbital-energy bookkeeping, no
    // XC dependence, so it is correct for RHF and RKS alike given the same
    // eps/n_occ. Verified against an independent hand-written loop (not the
    // same code shape as the production one-liner) in the SAME virtual-major
    // idx(a,i) = a*n_occ + i convention pack_hessian_vector_product_cphf_order
    // already uses -- non-square (n_occ != n_virt) so a row/column swap in
    // the index formula cannot hide.
    void check_orbital_energy_diagonal(int nbasis, int n_occ)
    {
        Eigen::VectorXd eps(nbasis);
        for (int p = 0; p < nbasis; ++p)
            eps(p) = -5.0 + 0.37 * p; // arbitrary, strictly increasing, no accidental degeneracies

        const Eigen::VectorXd diag = DFT::Driver::orbital_energy_difference_diagonal(eps, n_occ);

        const int n_virt = nbasis - n_occ;
        require_near(static_cast<double>(diag.size()), static_cast<double>(n_virt * n_occ), 0.0,
                     "orbital_energy_difference_diagonal: wrong size for nbasis=" + std::to_string(nbasis) +
                         " n_occ=" + std::to_string(n_occ));

        for (int a = 0; a < n_virt; ++a)
        {
            for (int i = 0; i < n_occ; ++i)
            {
                const double expected = eps(n_occ + a) - eps(i);
                const int flat = a * n_occ + i; // virtual-major, matching pack_hessian_vector_product_cphf_order
                require_near(diag(flat), expected, 1e-14,
                             "orbital_energy_difference_diagonal mismatch at (a=" + std::to_string(a) +
                                 ",i=" + std::to_string(i) + ") nbasis=" + std::to_string(nbasis) +
                                 " n_occ=" + std::to_string(n_occ));
            }
        }
    }
} // namespace

int main()
{
    // Fixed, non-degenerate symmetric P/dP -- arbitrary but not sparse
    // (every entry nonzero, P != dP, neither diagonal nor a multiple of
    // the other), so no term in the algebra can vanish by an accidental
    // choice.
    Eigen::Matrix3d P;
    P << 0.9, 0.15, -0.05, 0.15, 0.6, 0.1, -0.05, 0.1, 0.4;
    Eigen::Matrix3d dP;
    dP << 0.02, 0.01, -0.015, 0.01, -0.03, 0.005, -0.015, 0.005, 0.02;

    Eigen::Matrix3d P2;
    P2 << 1.5, -0.2, 0.3, -0.2, 0.8, -0.1, 0.3, -0.1, 1.1;
    Eigen::Matrix3d dP2;
    dP2 << -0.04, 0.02, 0.01, 0.02, 0.03, -0.02, 0.01, -0.02, -0.015;

    check_lda_point("lda_x", P, dP);
    check_lda_point("lda_x", P2, dP2);

    check_gga_point("pbe", P, dP);
    check_gga_point("pbe", P2, dP2);

    // Non-square (n_occ != n_virt) in every case, per this codebase's own
    // packing-test discipline (see tests/dft_hessian_vector_packing.cpp).
    check_orbital_energy_diagonal(/*nbasis=*/7, /*n_occ=*/2);
    check_orbital_energy_diagonal(/*nbasis=*/10, /*n_occ=*/6);
    check_orbital_energy_diagonal(/*nbasis=*/6, /*n_occ=*/1);
    check_orbital_energy_diagonal(/*nbasis=*/12, /*n_occ=*/4);

    return g_ok ? 0 : 1;
}
