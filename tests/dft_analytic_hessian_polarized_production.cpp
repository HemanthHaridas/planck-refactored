// D3.0 (docs/SOSCF_DFT.md): promotes
// tests/dft_gga_polarized_hessian_selfcheck.cpp's own verified T1..T5
// point-level algebra (F3.4.2-F3.4.4, docs/DFT_ANALYTIC_FXC_HESSIAN.md)
// into a real production function,
// compute_analytic_xc_hessian_vector_product_polarized
// (src/dft/analytic_hessian.{h,cpp}).
//
// This is NOT a re-derivation of F3.4's formulas -- it is a check that the
// NEW function's actual call signature (MolecularGrid + AOGridEvaluation +
// FOUR AO-basis density matrices: ground alpha/beta, trial alpha/beta) still
// reproduces those formulas once real AO/grid machinery and the polarized
// libxc packing (F3.4.1's own confirmed v2rhosigma/v2sigma2 ordering) are
// both involved. A transcription bug reading the wrong slot of the 6-
// component arrays, or swapping which channel's gradient enters the "self"
// vs "cross" term, would not be visible to F3.4's own scalar-level tests,
// since they never called a function shaped like this one.
//
// Same synthetic single-point-grid fixture discipline D2.0's own test
// (tests/dft_analytic_hessian_production.cpp) uses: pick (P^a, P^b, dP^a,
// dP^b) FREELY (arbitrary, non-degenerate, non-diagonal symmetric
// matrices), read off whatever (rho, grad_rho, drho, dgrad_rho) actually
// result from the real AO contraction on both channels, and feed THOSE into
// an INDEPENDENTLY-WRITTEN reference formula (not calling the production
// function) -- so a shared bug in the contraction cannot hide.
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
            return DFT::XC::Functional::create(1, DFT::XC::Spin::Polarized).value();
        }
        auto functional = DFT::XC::Functional::create(*id, DFT::XC::Spin::Polarized);
        if (!functional)
        {
            std::cerr << "Functional::create(" << name << ") failed: " << functional.error() << '\n';
            g_ok = false;
            return DFT::XC::Functional::create(1, DFT::XC::Spin::Polarized).value();
        }
        return std::move(*functional);
    }

    DFT::XC::Functional require_functional_unpolarized(const std::string &name)
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

    // Single-point synthetic "grid", 3 AOs -- same rank-3 fixture-design
    // reasoning D2.0's own test used (P/dP freely chosen 3x3 symmetric
    // matrices, no attempt to hit pre-chosen density targets).
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

    double dot3(double ax, double ay, double az, double bx, double by, double bz)
    {
        return ax * bx + ay * by + az * bz;
    }

    // Independent reference: F3.4's own check_mixed/check_beta formulas,
    // reproduced from tests/dft_gga_polarized_hessian_selfcheck.cpp (NOT
    // calling the production function) -- returns (delta[V_xc^a](0,1),
    // delta[V_xc^b](0,1)), the off-diagonal AO element (so a transposition
    // or channel swap is visible).
    std::pair<double, double> reference_delta_vxc_01(
        const DFT::XC::Functional &f, const DensityResponse &ga, const DensityResponse &gb,
        const DensityResponse &ta, const DensityResponse &tb, double AA, double AGx, double AGy, double AGz)
    {
        const double sigma_aa0 = dot3(ga.gx, ga.gy, ga.gz, ga.gx, ga.gy, ga.gz);
        const double sigma_ab0 = dot3(ga.gx, ga.gy, ga.gz, gb.gx, gb.gy, gb.gz);
        const double sigma_bb0 = dot3(gb.gx, gb.gy, gb.gz, gb.gx, gb.gy, gb.gz);

        std::vector<double> v2rho2, v2rhosigma, v2sigma2;
        f.evaluate_gga_fxc({ga.rho, gb.rho}, {sigma_aa0, sigma_ab0, sigma_bb0}, 1, v2rho2, v2rhosigma, v2sigma2);
        std::vector<double> exc0, vrho0, vsigma0;
        f.evaluate_gga_exc_vxc({ga.rho, gb.rho}, {sigma_aa0, sigma_ab0, sigma_bb0}, 1, exc0, vrho0, vsigma0);
        const double vsigma_aa0 = vsigma0[0], vsigma_ab0 = vsigma0[1], vsigma_bb0 = vsigma0[2];

        const double g_a_dot_AG = dot3(ga.gx, ga.gy, ga.gz, AGx, AGy, AGz);
        const double g_b_dot_AG = dot3(gb.gx, gb.gy, gb.gz, AGx, AGy, AGz);
        const double dg_a_dot_AG = dot3(ta.gx, ta.gy, ta.gz, AGx, AGy, AGz);
        const double dg_b_dot_AG = dot3(tb.gx, tb.gy, tb.gz, AGx, AGy, AGz);

        const double dsigma_aa = 2.0 * dot3(ga.gx, ga.gy, ga.gz, ta.gx, ta.gy, ta.gz);
        const double dsigma_ab = dot3(gb.gx, gb.gy, gb.gz, ta.gx, ta.gy, ta.gz) +
                                  dot3(ga.gx, ga.gy, ga.gz, tb.gx, tb.gy, tb.gz);
        const double dsigma_bb = 2.0 * dot3(gb.gx, gb.gy, gb.gz, tb.gx, tb.gy, tb.gz);

        // Alpha channel (check_mixed).
        const double delta_vrho_a = v2rho2[0] * ta.rho + v2rho2[1] * tb.rho + v2rhosigma[0] * dsigma_aa +
                                     v2rhosigma[1] * dsigma_ab + v2rhosigma[2] * dsigma_bb;
        const double delta_vsigma_aa = v2rhosigma[0] * ta.rho + v2rhosigma[3] * tb.rho + v2sigma2[0] * dsigma_aa +
                                        v2sigma2[1] * dsigma_ab + v2sigma2[2] * dsigma_bb;
        const double delta_vsigma_ab_a = v2rhosigma[1] * ta.rho + v2rhosigma[4] * tb.rho + v2sigma2[1] * dsigma_aa +
                                          v2sigma2[3] * dsigma_ab + v2sigma2[4] * dsigma_bb;
        const double T1 = delta_vrho_a * AA;
        const double T2 = 2.0 * delta_vsigma_aa * g_a_dot_AG;
        const double T3 = 2.0 * vsigma_aa0 * dg_a_dot_AG;
        const double T4 = delta_vsigma_ab_a * g_b_dot_AG;
        const double T5 = vsigma_ab0 * dg_b_dot_AG;
        const double delta_vxc_a = T1 + T2 + T3 + T4 + T5;

        // Beta channel (check_beta) -- independently re-derived, not
        // alpha-with-labels-swapped.
        const double delta_vrho_b = v2rho2[2] * tb.rho + v2rho2[1] * ta.rho + v2rhosigma[5] * dsigma_bb +
                                     v2rhosigma[4] * dsigma_ab + v2rhosigma[3] * dsigma_aa;
        const double delta_vsigma_bb = v2rhosigma[5] * tb.rho + v2rhosigma[2] * ta.rho + v2sigma2[5] * dsigma_bb +
                                        v2sigma2[4] * dsigma_ab + v2sigma2[2] * dsigma_aa;
        const double delta_vsigma_ab_b = v2rhosigma[4] * tb.rho + v2rhosigma[1] * ta.rho + v2sigma2[4] * dsigma_bb +
                                          v2sigma2[3] * dsigma_ab + v2sigma2[1] * dsigma_aa;
        const double T1p = delta_vrho_b * AA;
        const double T2p = 2.0 * delta_vsigma_bb * g_b_dot_AG;
        const double T3p = 2.0 * vsigma_bb0 * dg_b_dot_AG;
        const double T4p = delta_vsigma_ab_b * g_a_dot_AG;
        const double T5p = vsigma_ab0 * dg_a_dot_AG;
        const double delta_vxc_b = T1p + T2p + T3p + T4p + T5p;

        return {delta_vxc_a, delta_vxc_b};
    }

    // LDA-polarized reference: V_xc^sigma = vrho_sigma alone (no gradient
    // terms), so the whole chain rule is
    //   delta[vrho_a] = v2rho2_aa*drho_a + v2rho2_ab*drho_b
    //   delta[vrho_b] = v2rho2_ab*drho_a + v2rho2_bb*drho_b
    // (v2rho2 = [aa,ab,bb], confirmed by tests/dft_fxc_selfcheck.cpp's own
    // check_lda_v2rho2_polarized, including the aa<->bb symmetric ab cross
    // term). Returns (delta[V_xc^a](0,1), delta[V_xc^b](0,1)).
    std::pair<double, double> reference_delta_vxc_lda_01(
        const DFT::XC::Functional &f, double rho_a0, double rho_b0, double drho_a, double drho_b, double AA)
    {
        std::vector<double> v2rho2;
        f.evaluate_lda_fxc({rho_a0, rho_b0}, 1, v2rho2);
        const double delta_vrho_a = v2rho2[0] * drho_a + v2rho2[1] * drho_b;
        const double delta_vrho_b = v2rho2[1] * drho_a + v2rho2[2] * drho_b;
        return {delta_vrho_a * AA, delta_vrho_b * AA};
    }

    void check_point_lda(
        const std::string &name, const Eigen::Matrix3d &Pa, const Eigen::Matrix3d &Pb, const Eigen::Matrix3d &dPa,
        const Eigen::Matrix3d &dPb)
    {
        auto f = require_functional(name);

        SyntheticPoint sp{
            /*weight=*/1.0,
            Eigen::Vector3d(1.3, 0.7, -0.4),
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};

        const DFT::MolecularGrid grid = make_grid(sp);
        const DFT::AOGridEvaluation ao = make_ao_grid(sp);

        auto result = DFT::Driver::compute_analytic_xc_hessian_vector_product_polarized(
            grid, ao, Pa, Pb, dPa, dPb, f, f);
        if (!result)
        {
            std::cerr << name << ": compute_analytic_xc_hessian_vector_product_polarized (LDA) failed: "
                      << result.error() << '\n';
            g_ok = false;
            return;
        }
        const auto &[delta_v_xc_a, delta_v_xc_b] = *result;

        require_near((delta_v_xc_a)(0, 1), (delta_v_xc_a)(1, 0), 1e-12,
                     name + ": LDA delta_V_xc^a must be symmetric");
        require_near((delta_v_xc_b)(0, 1), (delta_v_xc_b)(1, 0), 1e-12,
                     name + ": LDA delta_V_xc^b must be symmetric");

        const double rho_a0 = sp.phi.transpose() * Pa * sp.phi;
        const double rho_b0 = sp.phi.transpose() * Pb * sp.phi;
        const double drho_a = sp.phi.transpose() * dPa * sp.phi;
        const double drho_b = sp.phi.transpose() * dPb * sp.phi;
        const double AA = sp.phi(0) * sp.phi(1);

        const auto [ref_a_raw, ref_b_raw] = reference_delta_vxc_lda_01(f, rho_a0, rho_b0, drho_a, drho_b, AA);
        // Same x+c doubling convention as the GGA checks above.
        const double ref_a = 2.0 * ref_a_raw;
        const double ref_b = 2.0 * ref_b_raw;

        require_near((delta_v_xc_a)(0, 1), ref_a, 1e-10,
                     name + ": LDA delta_V_xc^a(0,1) vs reference formula");
        require_near((delta_v_xc_b)(0, 1), ref_b, 1e-10,
                     name + ": LDA delta_V_xc^b(0,1) vs reference formula");
    }

    void check_point(
        const std::string &name, const Eigen::Matrix3d &Pa, const Eigen::Matrix3d &Pb, const Eigen::Matrix3d &dPa,
        const Eigen::Matrix3d &dPb)
    {
        auto f = require_functional(name);

        // Non-degenerate AO values/gradients: no zero components, no
        // proportionality between phi and its gradients.
        SyntheticPoint sp{
            /*weight=*/1.0,
            Eigen::Vector3d(1.3, 0.7, -0.4),
            Eigen::Vector3d(0.4, -0.3, 0.2),
            Eigen::Vector3d(-0.2, 0.5, 0.1),
            Eigen::Vector3d(0.1, -0.1, 0.3)};

        const DFT::MolecularGrid grid = make_grid(sp);
        const DFT::AOGridEvaluation ao = make_ao_grid(sp);

        auto result = DFT::Driver::compute_analytic_xc_hessian_vector_product_polarized(
            grid, ao, Pa, Pb, dPa, dPb, f, f);
        if (!result)
        {
            std::cerr << name << ": compute_analytic_xc_hessian_vector_product_polarized failed: "
                      << result.error() << '\n';
            g_ok = false;
            return;
        }
        const auto &[delta_v_xc_a, delta_v_xc_b] = *result;

        require_near((delta_v_xc_a)(0, 1), (delta_v_xc_a)(1, 0), 1e-12,
                     name + ": delta_V_xc^a must be symmetric");
        require_near((delta_v_xc_b)(0, 1), (delta_v_xc_b)(1, 0), 1e-12,
                     name + ": delta_V_xc^b must be symmetric");

        const DensityResponse ga = evaluate_density_response(sp, Pa);
        const DensityResponse gb = evaluate_density_response(sp, Pb);
        const DensityResponse ta = evaluate_density_response(sp, dPa);
        const DensityResponse tb = evaluate_density_response(sp, dPb);

        const double AA = sp.phi(0) * sp.phi(1);
        const double AGx = sp.phi(0) * sp.gphix(1) + sp.gphix(0) * sp.phi(1);
        const double AGy = sp.phi(0) * sp.gphiy(1) + sp.gphiy(0) * sp.phi(1);
        const double AGz = sp.phi(0) * sp.gphiz(1) + sp.gphiz(0) * sp.phi(1);

        const auto [ref_a_raw, ref_b_raw] = reference_delta_vxc_01(f, ga, gb, ta, tb, AA, AGx, AGy, AGz);
        // Production sums exchange_functional + correlation_functional
        // (both passed as f here), so its output is 2x a reference built
        // from ONE evaluate_gga_fxc call -- same x+c doubling convention
        // D2.0's own restricted test corrects for explicitly.
        const double ref_a = 2.0 * ref_a_raw;
        const double ref_b = 2.0 * ref_b_raw;

        require_near((delta_v_xc_a)(0, 1), ref_a, 1e-8,
                     name + ": delta_V_xc^a(0,1) vs F3.4 reference formula (check_mixed)");
        require_near((delta_v_xc_b)(0, 1), ref_b, 1e-8,
                     name + ": delta_V_xc^b(0,1) vs F3.4 reference formula (check_beta)");
    }

    // DFT_ANALYTIC_FXC_COMBINED_XC_SCOPE C4: for a combined exchange-
    // correlation functional the exchange slot already carries the whole XC,
    // so compute_analytic_xc_hessian_vector_product{,_polarized} must treat
    // the correlation_functional argument as inert -- otherwise fxc[c] is
    // double-counted. Assert both entries give the SAME result whether the
    // second arg is an unrelated correlation functional or the combined
    // functional itself. Mutation check: reverting the
    // is_combined_exchange_correlation() guard makes these differ.
    void check_combined_no_double_count(const std::string &combined_name)
    {
        auto combined = require_functional_unpolarized(combined_name);
        if (!combined.is_combined_exchange_correlation())
        {
            std::cerr << combined_name << ": expected a combined XC functional\n";
            g_ok = false;
            return;
        }
        auto unrelated_c = require_functional_unpolarized("gga_c_pbe");

        SyntheticPoint sp{
            /*weight=*/1.0,
            Eigen::Vector3d(1.3, 0.7, -0.4),
            Eigen::Vector3d(0.4, -0.3, 0.2),
            Eigen::Vector3d(-0.2, 0.5, 0.1),
            Eigen::Vector3d(0.1, -0.1, 0.3)};
        const DFT::MolecularGrid grid = make_grid(sp);
        const DFT::AOGridEvaluation ao = make_ao_grid(sp);

        Eigen::Matrix3d P, dP;
        P << 0.9, 0.15, -0.05, 0.15, 0.6, 0.1, -0.05, 0.1, 0.4;
        dP << 0.02, 0.01, -0.015, 0.01, -0.03, 0.005, -0.015, 0.005, 0.02;

        // RKS entry.
        auto r_unrelated = DFT::Driver::compute_analytic_xc_hessian_vector_product(
            grid, ao, P, dP, combined, unrelated_c);
        auto r_self = DFT::Driver::compute_analytic_xc_hessian_vector_product(
            grid, ao, P, dP, combined, combined);
        if (!r_unrelated || !r_self)
        {
            std::cerr << combined_name << ": RKS combined call failed: "
                      << (r_unrelated ? r_self.error() : r_unrelated.error()) << '\n';
            g_ok = false;
            return;
        }
        require_near((*r_unrelated - *r_self).cwiseAbs().maxCoeff(), 0.0, 1e-12,
                     combined_name + ": RKS correlation_functional arg must be inert for combined XC");

        // Polarized entry.
        Eigen::Matrix3d Pa, Pb, dPa, dPb;
        Pa << 0.9, 0.15, -0.05, 0.15, 0.6, 0.1, -0.05, 0.1, 0.4;
        Pb << 0.5, -0.1, 0.08, -0.1, 0.7, -0.05, 0.08, -0.05, 0.3;
        dPa << 0.02, 0.01, -0.015, 0.01, -0.03, 0.005, -0.015, 0.005, 0.02;
        dPb << -0.01, 0.02, 0.005, 0.02, 0.015, -0.01, 0.005, -0.01, -0.02;
        auto combined_pol = require_functional(combined_name);
        auto unrelated_c_pol = require_functional("gga_c_pbe");
        auto p_unrelated = DFT::Driver::compute_analytic_xc_hessian_vector_product_polarized(
            grid, ao, Pa, Pb, dPa, dPb, combined_pol, unrelated_c_pol);
        auto p_self = DFT::Driver::compute_analytic_xc_hessian_vector_product_polarized(
            grid, ao, Pa, Pb, dPa, dPb, combined_pol, combined_pol);
        if (!p_unrelated || !p_self)
        {
            std::cerr << combined_name << ": polarized combined call failed\n";
            g_ok = false;
            return;
        }
        require_near((p_unrelated->first - p_self->first).cwiseAbs().maxCoeff(), 0.0, 1e-12,
                     combined_name + ": polarized alpha correlation arg must be inert for combined XC");
        require_near((p_unrelated->second - p_self->second).cwiseAbs().maxCoeff(), 0.0, 1e-12,
                     combined_name + ": polarized beta correlation arg must be inert for combined XC");
    }
} // namespace

int main()
{
    // gga_c_pbe: PBE correlation alone, per F3.4.1's own finding that PBE
    // EXCHANGE has near-zero cross-spin coupling (which would leave the
    // T4/T4' cross terms untested).
    Eigen::Matrix3d Pa, Pb, dPa, dPb;
    Pa << 0.9, 0.15, -0.05, 0.15, 0.6, 0.1, -0.05, 0.1, 0.4;
    Pb << 0.5, -0.1, 0.08, -0.1, 0.7, -0.05, 0.08, -0.05, 0.3;
    dPa << 0.02, 0.01, -0.015, 0.01, -0.03, 0.005, -0.015, 0.005, 0.02;
    dPb << -0.01, 0.02, 0.005, 0.02, 0.015, -0.01, 0.005, -0.01, -0.02;
    check_point("gga_c_pbe", Pa, Pb, dPa, dPb);

    // Second independent point/perturbation set, same discipline as D2.0's
    // own test (P2/dP2).
    Eigen::Matrix3d Pa2, Pb2, dPa2, dPb2;
    Pa2 << 1.5, -0.2, 0.3, -0.2, 0.8, -0.1, 0.3, -0.1, 1.1;
    Pb2 << 0.6, 0.12, -0.2, 0.12, 0.9, 0.08, -0.2, 0.08, 0.7;
    dPa2 << -0.04, 0.02, 0.01, 0.02, 0.03, -0.02, 0.01, -0.02, -0.015;
    dPb2 << 0.03, -0.015, 0.02, -0.015, -0.01, 0.008, 0.02, 0.008, 0.025;
    check_point("gga_c_pbe", Pa2, Pb2, dPa2, dPb2);

    // Same-spin-only trial (dPb = 0): exercises T4/T4' (cross coefficient
    // driven by the UNCHANGED grad_rho_b) without T5/T5' contamination on
    // one side, mirroring F3.4.2's own alpha-only decomposition.
    check_point("gga_c_pbe", Pa, Pb, dPa, Eigen::Matrix3d::Zero());
    // Beta-only trial (dPa = 0): mirrors F3.4.4's own beta-only case (T5'
    // vanishes identically there).
    check_point("gga_c_pbe", Pa, Pb, Eigen::Matrix3d::Zero(), dPb);

    // LDA-polarized: lda_c_pw (PW correlation), the same functional
    // tests/dft_fxc_selfcheck.cpp uses to exercise a genuinely nonzero
    // v2rho2[ab] cross-spin term (lda_x/Slater exchange has NO cross-spin
    // coupling -- v2rho2_ab is exactly zero there, which would leave the
    // cross term untested).
    check_point_lda("lda_c_pw", Pa, Pb, dPa, dPb);
    check_point_lda("lda_c_pw", Pa2, Pb2, dPa2, dPb2);
    check_point_lda("lda_c_pw", Pa, Pb, dPa, Eigen::Matrix3d::Zero());
    check_point_lda("lda_c_pw", Pa, Pb, Eigen::Matrix3d::Zero(), dPb);

    // DFT_ANALYTIC_FXC_COMBINED_XC_SCOPE C4: combined XC must not double-count
    // correlation. B3LYP is GGA-combined; PBE0 (pbeh) too.
    check_combined_no_double_count("hyb_gga_xc_b3lyp");
    check_combined_no_double_count("hyb_gga_xc_pbeh");

    return g_ok ? 0 : 1;
}
