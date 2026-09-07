// F3.3 (docs/DFT_ANALYTIC_FXC_HESSIAN.md): the GGA unpolarized analytic
// Hessian-vector product, broken into F3.3.1 (T1), F3.3.2 (T2), F3.3.3
// (T1+T2+T3), verified as an isolated point-level check against libxc's own
// finite difference -- the same pattern F1's dft_fxc_selfcheck.cpp used,
// extended from LDA to GGA.
//
// Notation (matches the scope doc exactly):
//   AA  = phi_mu * phi_nu                          (plain AO product)
//   AG  = phi_mu*grad_phi_nu + grad_phi_mu*phi_nu   (existing gradient-coupling vector)
//   g   = grad_rho (a 3-vector at the point), sigma = g . g
//   dg  = delta_grad_rho, drho = delta_rho
//
// delta_V_xc = T1 + T2 + T3, where:
//   T1 = [v2rho2*drho + 2*v2rhosigma*(g.dg)] * AA
//   T2 = 2*[v2rhosigma*drho + 2*v2sigma2*(g.dg)] * (g.AG)
//   T3 = 2*vsigma * (dg.AG)
//
// AA and AG are point-local AO factors, independent of the density -- for
// an isolated point check they are just fixed numbers/vectors, not tied to
// any real basis set or molecule.
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dft/base/wrapper.h"

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

    struct Point
    {
        double rho;
        double gx, gy, gz; // grad_rho
    };

    struct Perturbation
    {
        double drho;
        double dgx, dgy, dgz; // delta_grad_rho
    };

    double dot(double ax, double ay, double az, double bx, double by, double bz)
    {
        return ax * bx + ay * by + az * bz;
    }

    // libxc's own second derivative, evaluated at a single point.
    struct Fxc
    {
        double v2rho2 = 0.0;
        double v2rhosigma = 0.0;
        double v2sigma2 = 0.0;
    };

    Fxc evaluate_fxc_at(const DFT::XC::Functional &f, const Point &p)
    {
        const double sigma = dot(p.gx, p.gy, p.gz, p.gx, p.gy, p.gz);
        std::vector<double> v2rho2, v2rhosigma, v2sigma2;
        auto fxc = f.evaluate_gga_fxc({p.rho}, {sigma}, 1, v2rho2, v2rhosigma, v2sigma2);
        if (!fxc)
        {
            std::cerr << "evaluate_gga_fxc failed: " << fxc.error() << '\n';
            g_ok = false;
            return {};
        }
        return {v2rho2[0], v2rhosigma[0], v2sigma2[0]};
    }

    // Raw finite difference of vrho under a joint (drho, dsigma) perturbation,
    // matching delta_sigma = 2*g.dg exactly (sigma is quadratic in grad_rho,
    // so its own perturbation is NOT independent of dg -- substituted here,
    // not assumed).
    double fd_delta_vrho(const DFT::XC::Functional &f, const Point &p, const Perturbation &d, double h)
    {
        const double g_dot_dg = dot(p.gx, p.gy, p.gz, d.dgx, d.dgy, d.dgz);
        const double dsigma = 2.0 * g_dot_dg;
        const double sigma0 = dot(p.gx, p.gy, p.gz, p.gx, p.gy, p.gz);

        std::vector<double> exc_p, vrho_p, vsigma_p, exc_m, vrho_m, vsigma_m;
        f.evaluate_gga_exc_vxc({p.rho + h * d.drho}, {sigma0 + h * dsigma}, 1, exc_p, vrho_p, vsigma_p);
        f.evaluate_gga_exc_vxc({p.rho - h * d.drho}, {sigma0 - h * dsigma}, 1, exc_m, vrho_m, vsigma_m);
        return (vrho_p[0] - vrho_m[0]) / (2.0 * h);
    }

    double fd_delta_vsigma(const DFT::XC::Functional &f, const Point &p, const Perturbation &d, double h)
    {
        const double g_dot_dg = dot(p.gx, p.gy, p.gz, d.dgx, d.dgy, d.dgz);
        const double dsigma = 2.0 * g_dot_dg;
        const double sigma0 = dot(p.gx, p.gy, p.gz, p.gx, p.gy, p.gz);

        std::vector<double> exc_p, vrho_p, vsigma_p, exc_m, vrho_m, vsigma_m;
        f.evaluate_gga_exc_vxc({p.rho + h * d.drho}, {sigma0 + h * dsigma}, 1, exc_p, vrho_p, vsigma_p);
        f.evaluate_gga_exc_vxc({p.rho - h * d.drho}, {sigma0 - h * dsigma}, 1, exc_m, vrho_m, vsigma_m);
        return (vsigma_p[0] - vsigma_m[0]) / (2.0 * h);
    }

    // F3.3.1 -- T1 alone: [v2rho2*drho + 2*v2rhosigma*(g.dg)] * AA.
    // Verified against a raw FD of vrho (delta[vrho], the coefficient T1
    // multiplies AA by) -- this is a pure libxc/point-level check, the same
    // shape F1's own selfcheck used, extended to include the g.dg term GGA
    // adds on top of LDA's plain v2rho2*drho.
    void check_T1(const std::string &name, const Point &p, const Perturbation &d)
    {
        auto f = require_functional(name);
        const Fxc fxc = evaluate_fxc_at(f, p);

        const double g_dot_dg = dot(p.gx, p.gy, p.gz, d.dgx, d.dgy, d.dgz);
        const double delta_vrho_analytic = fxc.v2rho2 * d.drho + 2.0 * fxc.v2rhosigma * g_dot_dg;

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            const double delta_vrho_fd = fd_delta_vrho(f, p, d, h);
            const double tol = 50.0 * h * h + 1e-6;
            require_near(delta_vrho_analytic, delta_vrho_fd, tol,
                         name + " T1 coefficient (delta[vrho]) vs FD, h=" + std::to_string(h) +
                             (p.gx == 0.0 && p.gy == 0.0 && p.gz == 0.0 ? " [grad_rho=0 point]" : ""));
        }
    }

    // Cross-check: at a point with grad_rho = 0 exactly, the g.dg term
    // vanishes identically regardless of dg, so T1's coefficient there must
    // reduce EXACTLY to v2rho2*drho -- the same formula F3.1 already
    // verified for LDA. This is the "T1 reduces toward F3.1's LDA term at a
    // negligible-gradient point" sanity check the scope doc asks for, made
    // exact (grad_rho=0 precisely) rather than approximate.
    void check_T1_reduces_to_lda_at_zero_gradient(const std::string &name, double rho0)
    {
        auto f = require_functional(name);
        const Point p{rho0, 0.0, 0.0, 0.0};
        const Perturbation d{0.01, 0.3, -0.2, 0.1}; // dg is nonzero; g.dg must still be 0 since g=0
        const Fxc fxc = evaluate_fxc_at(f, p);

        const double g_dot_dg = dot(p.gx, p.gy, p.gz, d.dgx, d.dgy, d.dgz);
        require_near(g_dot_dg, 0.0, 1e-15, name + ": g.dg must be exactly 0 when g=0");

        const double delta_vrho_analytic = fxc.v2rho2 * d.drho + 2.0 * fxc.v2rhosigma * g_dot_dg;
        const double delta_vrho_lda_only = fxc.v2rho2 * d.drho;
        require_near(delta_vrho_analytic, delta_vrho_lda_only, 1e-15,
                     name + ": T1 coefficient must reduce EXACTLY to v2rho2*drho when grad_rho=0");
    }

    // F3.3.2 -- T2's coefficient alone: 2*[v2rhosigma*drho + 2*v2sigma2*(g.dg)].
    // T2 = that coefficient times the *existing*, unchanged (g.AG) gradient-
    // coupling factor -- so isolating T2 at the point level means verifying
    // the coefficient itself, i.e. delta[vsigma], against a raw FD of vsigma
    // under the same joint (drho, dsigma=2*g.dg) perturbation T1 used for
    // vrho. This is the mixed-partial sibling of T1's check: T1 checked
    // d(vrho)/d(rho,sigma), T2 checks d(vsigma)/d(rho,sigma).
    void check_T2(const std::string &name, const Point &p, const Perturbation &d)
    {
        auto f = require_functional(name);
        const Fxc fxc = evaluate_fxc_at(f, p);

        const double g_dot_dg = dot(p.gx, p.gy, p.gz, d.dgx, d.dgy, d.dgz);
        const double delta_vsigma_analytic = fxc.v2rhosigma * d.drho + 2.0 * fxc.v2sigma2 * g_dot_dg;

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            const double delta_vsigma_fd = fd_delta_vsigma(f, p, d, h);
            const double tol = 50.0 * h * h + 1e-6;
            require_near(delta_vsigma_analytic, delta_vsigma_fd, tol,
                         name + " T2 coefficient (delta[vsigma]) vs FD, h=" + std::to_string(h) +
                             (p.gx == 0.0 && p.gy == 0.0 && p.gz == 0.0 ? " [grad_rho=0 point]" : ""));
        }
    }

    // At grad_rho=0 the g.dg term vanishes identically (same argument as
    // T1's zero-gradient reduction), so T2's coefficient there must reduce
    // EXACTLY to the plain v2rhosigma*drho piece -- no v2sigma2 contribution
    // survives regardless of dg's own direction or size.
    void check_T2_reduces_at_zero_gradient(const std::string &name, double rho0)
    {
        auto f = require_functional(name);
        const Point p{rho0, 0.0, 0.0, 0.0};
        const Perturbation d{0.01, 0.3, -0.2, 0.1};
        const Fxc fxc = evaluate_fxc_at(f, p);

        const double g_dot_dg = dot(p.gx, p.gy, p.gz, d.dgx, d.dgy, d.dgz);
        require_near(g_dot_dg, 0.0, 1e-15, name + ": g.dg must be exactly 0 when g=0");

        const double delta_vsigma_analytic = fxc.v2rhosigma * d.drho + 2.0 * fxc.v2sigma2 * g_dot_dg;
        const double delta_vsigma_lda_only = fxc.v2rhosigma * d.drho;
        require_near(delta_vsigma_analytic, delta_vsigma_lda_only, 1e-15,
                     name + ": T2 coefficient must reduce EXACTLY to v2rhosigma*drho when grad_rho=0");
    }

    // F3.3.3 -- add T3 = 2*vsigma*(dg.AG), the full T1+T2+T3 sum.
    //
    // Unlike T1/T2 (pure coefficients, checkable against a scalar FD of
    // vrho/vsigma alone), T3 involves the point-local AO vector AG and the
    // *existing*, unchanged ground-state vsigma -- so isolating it means
    // building the actual V_xc = vrho*AA + 2*vsigma*(g.AG) scalar and
    // finite-differencing THAT (not just its vrho/vsigma coefficients)
    // under a joint (drho, dg) perturbation, exactly matching what the real
    // FD-kernel oracle does at the grid level (perturb the density, rebuild
    // the full first-derivative V_xc from scratch). AA/AG are fixed
    // synthetic numbers/vectors here, per the file's own point-check
    // convention -- no real basis set needed to test the algebra.
    struct AOFactors
    {
        double AA;
        double AGx, AGy, AGz;
    };

    double eval_vxc_scalar(const DFT::XC::Functional &f, double rho, double gx, double gy, double gz,
                            const AOFactors &ao)
    {
        const double sigma = dot(gx, gy, gz, gx, gy, gz);
        std::vector<double> exc, vrho, vsigma;
        f.evaluate_gga_exc_vxc({rho}, {sigma}, 1, exc, vrho, vsigma);
        const double g_dot_AG = dot(gx, gy, gz, ao.AGx, ao.AGy, ao.AGz);
        return vrho[0] * ao.AA + 2.0 * vsigma[0] * g_dot_AG;
    }

    // FD of the full V_xc scalar under the joint (drho, dg) perturbation --
    // the direct analogue of what build_closed_shell_xc_kernel_blocks does
    // at the grid level, done here at a single synthetic point.
    double fd_delta_vxc(const DFT::XC::Functional &f, const Point &p, const Perturbation &d,
                         const AOFactors &ao, double h)
    {
        const double vp = eval_vxc_scalar(f, p.rho + h * d.drho, p.gx + h * d.dgx, p.gy + h * d.dgy,
                                           p.gz + h * d.dgz, ao);
        const double vm = eval_vxc_scalar(f, p.rho - h * d.drho, p.gx - h * d.dgx, p.gy - h * d.dgy,
                                           p.gz - h * d.dgz, ao);
        return (vp - vm) / (2.0 * h);
    }

    void check_T1_T2_T3(const std::string &name, const Point &p, const Perturbation &d, const AOFactors &ao)
    {
        auto f = require_functional(name);
        const Fxc fxc = evaluate_fxc_at(f, p);

        std::vector<double> exc0, vrho0, vsigma0;
        const double sigma0 = dot(p.gx, p.gy, p.gz, p.gx, p.gy, p.gz);
        f.evaluate_gga_exc_vxc({p.rho}, {sigma0}, 1, exc0, vrho0, vsigma0);

        const double g_dot_dg = dot(p.gx, p.gy, p.gz, d.dgx, d.dgy, d.dgz);
        const double g_dot_AG = dot(p.gx, p.gy, p.gz, ao.AGx, ao.AGy, ao.AGz);
        const double dg_dot_AG = dot(d.dgx, d.dgy, d.dgz, ao.AGx, ao.AGy, ao.AGz);

        const double T1 = (fxc.v2rho2 * d.drho + 2.0 * fxc.v2rhosigma * g_dot_dg) * ao.AA;
        const double T2 = 2.0 * (fxc.v2rhosigma * d.drho + 2.0 * fxc.v2sigma2 * g_dot_dg) * g_dot_AG;
        const double T3 = 2.0 * vsigma0[0] * dg_dot_AG;
        const double delta_vxc_analytic = T1 + T2 + T3;

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            const double delta_vxc_fd = fd_delta_vxc(f, p, d, ao, h);
            const double tol = 50.0 * h * h + 1e-6;
            require_near(delta_vxc_analytic, delta_vxc_fd, tol,
                         name + " T1+T2+T3 (delta[V_xc]) vs FD, h=" + std::to_string(h));
        }
    }
} // namespace

int main()
{
    // PBE, matching F1's own GGA choice and the scope doc's stated plan.
    // A handful of (rho, grad_rho, drho, dgrad_rho) points, including one
    // exactly at grad_rho=0 and several with genuinely non-uniform gradients
    // (not aligned with delta_grad_rho, so g.dg is a real nonzero number,
    // not accidentally zero by a lucky choice of direction).
    check_T1("pbe", {0.3, 0.1, 0.05, -0.02}, {0.01, 0.02, -0.01, 0.005});
    check_T1("pbe", {1.0, 0.5, -0.3, 0.2}, {0.05, -0.1, 0.08, -0.04});
    check_T1("pbe", {0.15, 0.0, 0.0, 0.0}, {0.02, 0.3, -0.2, 0.1}); // grad_rho=0 point
    check_T1("pbe", {0.6, 0.2, 0.2, 0.2}, {-0.03, 0.1, 0.1, 0.1});  // dg parallel to g

    check_T1_reduces_to_lda_at_zero_gradient("pbe", 0.3);
    check_T1_reduces_to_lda_at_zero_gradient("pbe", 1.2);

    // F3.3.2: same four points, checking T2's coefficient (delta[vsigma])
    // instead of T1's (delta[vrho]).
    check_T2("pbe", {0.3, 0.1, 0.05, -0.02}, {0.01, 0.02, -0.01, 0.005});
    check_T2("pbe", {1.0, 0.5, -0.3, 0.2}, {0.05, -0.1, 0.08, -0.04});
    check_T2("pbe", {0.15, 0.0, 0.0, 0.0}, {0.02, 0.3, -0.2, 0.1}); // grad_rho=0 point
    check_T2("pbe", {0.6, 0.2, 0.2, 0.2}, {-0.03, 0.1, 0.1, 0.1});  // dg parallel to g

    check_T2_reduces_at_zero_gradient("pbe", 0.3);
    check_T2_reduces_at_zero_gradient("pbe", 1.2);

    // F3.3.3: full T1+T2+T3 sum. AA/AG are fixed synthetic point-local AO
    // factors (not tied to a real basis) chosen so dg.AG is genuinely
    // nonzero at every point -- otherwise T3 would vanish and this would
    // silently degrade into re-testing F3.3.2 with an extra unused term.
    const AOFactors ao1{0.7, 0.4, -0.3, 0.2};
    check_T1_T2_T3("pbe", {0.3, 0.1, 0.05, -0.02}, {0.01, 0.02, -0.01, 0.005}, ao1);
    check_T1_T2_T3("pbe", {1.0, 0.5, -0.3, 0.2}, {0.05, -0.1, 0.08, -0.04}, ao1);
    check_T1_T2_T3("pbe", {0.15, 0.0, 0.0, 0.0}, {0.02, 0.3, -0.2, 0.1}, ao1); // grad_rho=0 point
    check_T1_T2_T3("pbe", {0.6, 0.2, 0.2, 0.2}, {-0.03, 0.1, 0.1, 0.1}, ao1);  // dg parallel to g

    // Second AO-factor choice, since AA/AG are point-local and a single
    // fixed choice could accidentally hide a sign/index error in how T1's
    // AA-weighting and T2/T3's AG-weighting combine.
    const AOFactors ao2{-0.2, -0.1, 0.5, -0.4};
    check_T1_T2_T3("pbe", {0.3, 0.1, 0.05, -0.02}, {0.01, 0.02, -0.01, 0.005}, ao2);
    check_T1_T2_T3("pbe", {0.6, 0.2, 0.2, 0.2}, {-0.03, 0.1, 0.1, 0.1}, ao2);

    // B88 exchange and LYP correlation -- B3LYP's GGA components. Previously
    // only "pbe" was exercised here; adding these closes the coverage gap
    // flagged during SOSCF_DFT_HYBRID_SCOPE H4/H5 (the polarized twin adds
    // the same). B88/LYP match the FD of delta[V_xc] to the same 50*h^2
    // tolerance PBE does.
    check_T1_T2_T3("gga_x_b88", {0.3, 0.1, 0.05, -0.02}, {0.01, 0.02, -0.01, 0.005}, ao1);
    check_T1_T2_T3("gga_x_b88", {0.6, 0.2, 0.2, 0.2}, {-0.03, 0.1, 0.1, 0.1}, ao2);
    check_T1_T2_T3("gga_c_lyp", {0.3, 0.1, 0.05, -0.02}, {0.01, 0.02, -0.01, 0.005}, ao1);
    check_T1_T2_T3("gga_c_lyp", {0.6, 0.2, 0.2, 0.2}, {-0.03, 0.1, 0.1, 0.1}, ao2);

    return g_ok ? 0 : 1;
}
