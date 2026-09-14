// S1 (docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md, N3.5.7.7): isolated
// check that libxc's own analytic THIRD derivative (xc_lda_kxc /
// xc_gga_kxc, via DFT::XC::Functional::evaluate_{lda,gga}_kxc) agrees with
// a finite difference of libxc's own SECOND derivative (v2rho2 / v2rhosigma
// / v2sigma2, via the existing evaluate_{lda,gga}_fxc). Exactly the same
// idea as dft_fxc_selfcheck.cpp, one derivative order up: if this fails,
// the bug is in libxc or in how the wrapper reads its kxc output arrays,
// not in anything the double-hybrid gradient derives on top of it.
//
// Eq. 33's Term 1 of the double-hybrid PT2 gradient differentiates the
// response operator R(D') (which already carries f^(2)), so it needs
// f^(3) -- v3rho3 (LDA) and v3rho3/v3rho2sigma/v3rhosigma2/v3sigma3 (GGA).
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

    DFT::XC::Functional require_functional(const std::string &name, DFT::XC::Spin spin)
    {
        auto id = DFT::XC::functional_id(name);
        if (!id)
        {
            std::cerr << "functional_id(" << name << ") failed: " << id.error() << '\n';
            g_ok = false;
            return DFT::XC::Functional::create(1, spin).value();
        }

        auto functional = DFT::XC::Functional::create(*id, spin);
        if (!functional)
        {
            std::cerr << "Functional::create(" << name << ") failed: " << functional.error() << '\n';
            g_ok = false;
            return DFT::XC::Functional::create(1, spin).value();
        }
        return std::move(*functional);
    }

    // Central-difference d(v2rho2)/d(rho) vs v3rho3 at a single unpolarized
    // point. Three step sizes (1e-2, 1e-3, 1e-4), tol loosened O(h^2) with
    // h -- same convergence-reading convention as dft_fxc_selfcheck.cpp.
    void check_lda_v3rho3_unpolarized(const std::string &name, double rho0)
    {
        auto functional = require_functional(name, DFT::XC::Spin::Unpolarized);

        std::vector<double> v3rho3;
        auto kxc = functional.evaluate_lda_kxc({rho0}, 1, v3rho3);
        require(kxc.has_value(), name + ": evaluate_lda_kxc failed: " + (kxc ? "" : kxc.error()));
        if (!kxc)
            return;
        require(v3rho3.size() == 1, name + ": unpolarized v3rho3 must have exactly 1 component");

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            std::vector<double> v2rho2_p, v2rho2_m;
            auto plus = functional.evaluate_lda_fxc({rho0 + h}, 1, v2rho2_p);
            auto minus = functional.evaluate_lda_fxc({rho0 - h}, 1, v2rho2_m);
            require(plus.has_value() && minus.has_value(), name + ": evaluate_lda_fxc failed during FD");
            if (!plus || !minus)
                continue;

            const double fd = (v2rho2_p[0] - v2rho2_m[0]) / (2.0 * h);
            const double tol = 500.0 * h * h + 1e-5;
            require_near(fd, v3rho3[0], tol,
                         name + " v3rho3 vs FD(v2rho2), h=" + std::to_string(h));
        }
    }

    // GGA unpolarized: each third-derivative block against the FD of the
    // corresponding second-derivative block. Cross-checks where two FD
    // routes reach the same kxc component (mixed partials) are verified,
    // not assumed -- same discipline as the fxc self-check's v2rhosigma
    // double route.
    void check_gga_v3_unpolarized(const std::string &name, double rho0, double sigma0)
    {
        auto functional = require_functional(name, DFT::XC::Spin::Unpolarized);

        std::vector<double> v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3;
        auto kxc = functional.evaluate_gga_kxc({rho0}, {sigma0}, 1,
                                               v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);
        require(kxc.has_value(), name + ": evaluate_gga_kxc failed: " + (kxc ? "" : kxc.error()));
        if (!kxc)
            return;
        require(v3rho3.size() == 1, name + ": unpolarized v3rho3 must have 1 component");
        require(v3rho2sigma.size() == 1, name + ": unpolarized v3rho2sigma must have 1 component");
        require(v3rhosigma2.size() == 1, name + ": unpolarized v3rhosigma2 must have 1 component");
        require(v3sigma3.size() == 1, name + ": unpolarized v3sigma3 must have 1 component");

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            // GGA v3 blocks at small sigma have strong curvature, so the
            // O(h^2) FD truncation floor at h=1e-2 runs ~2% (measured on
            // PBE v3rhosigma2). The h=1e-3/1e-4 steps of every block pin
            // the value tightly; the h=1e-2 tol only needs to admit that
            // truncation, and the FD converging as h shrinks is the real
            // signal (same reading as dft_fxc_selfcheck.cpp).
            const double tol = 2000.0 * h * h + 1e-4;

            // v3rho3 vs FD(v2rho2; d rho), sigma fixed.
            {
                std::vector<double> a, b, c, d, e, f;
                functional.evaluate_gga_fxc({rho0 + h}, {sigma0}, 1, a, b, c);
                functional.evaluate_gga_fxc({rho0 - h}, {sigma0}, 1, d, e, f);
                const double fd = (a[0] - d[0]) / (2.0 * h);
                require_near(fd, v3rho3[0], tol, name + " v3rho3 vs FD(v2rho2;d rho), h=" + std::to_string(h));
            }

            // v3rho2sigma: FD(v2rho2; d sigma) == FD(v2rhosigma; d rho).
            {
                std::vector<double> a, b, c, d, e, f;
                functional.evaluate_gga_fxc({rho0}, {sigma0 + h}, 1, a, b, c);
                functional.evaluate_gga_fxc({rho0}, {sigma0 - h}, 1, d, e, f);
                const double fd_from_v2rho2 = (a[0] - d[0]) / (2.0 * h);
                require_near(fd_from_v2rho2, v3rho2sigma[0], tol,
                             name + " v3rho2sigma vs FD(v2rho2;d sigma), h=" + std::to_string(h));
            }
            {
                std::vector<double> a, b, c, d, e, f;
                functional.evaluate_gga_fxc({rho0 + h}, {sigma0}, 1, a, b, c);
                functional.evaluate_gga_fxc({rho0 - h}, {sigma0}, 1, d, e, f);
                const double fd_from_v2rhosigma = (b[0] - e[0]) / (2.0 * h);
                require_near(fd_from_v2rhosigma, v3rho2sigma[0], tol,
                             name + " v3rho2sigma vs FD(v2rhosigma;d rho), h=" + std::to_string(h));
            }

            // v3rhosigma2: FD(v2rhosigma; d sigma) == FD(v2sigma2; d rho).
            {
                std::vector<double> a, b, c, d, e, f;
                functional.evaluate_gga_fxc({rho0}, {sigma0 + h}, 1, a, b, c);
                functional.evaluate_gga_fxc({rho0}, {sigma0 - h}, 1, d, e, f);
                const double fd_from_v2rhosigma = (b[0] - e[0]) / (2.0 * h);
                require_near(fd_from_v2rhosigma, v3rhosigma2[0], tol,
                             name + " v3rhosigma2 vs FD(v2rhosigma;d sigma), h=" + std::to_string(h));
            }
            {
                std::vector<double> a, b, c, d, e, f;
                functional.evaluate_gga_fxc({rho0 + h}, {sigma0}, 1, a, b, c);
                functional.evaluate_gga_fxc({rho0 - h}, {sigma0}, 1, d, e, f);
                const double fd_from_v2sigma2 = (c[0] - f[0]) / (2.0 * h);
                require_near(fd_from_v2sigma2, v3rhosigma2[0], tol,
                             name + " v3rhosigma2 vs FD(v2sigma2;d rho), h=" + std::to_string(h));
            }

            // v3sigma3 vs FD(v2sigma2; d sigma), rho fixed.
            {
                std::vector<double> a, b, c, d, e, f;
                functional.evaluate_gga_fxc({rho0}, {sigma0 + h}, 1, a, b, c);
                functional.evaluate_gga_fxc({rho0}, {sigma0 - h}, 1, d, e, f);
                const double fd = (c[0] - f[0]) / (2.0 * h);
                require_near(fd, v3sigma3[0], tol, name + " v3sigma3 vs FD(v2sigma2;d sigma), h=" + std::to_string(h));
            }
        }
    }

    // Same family guards as evaluate_{lda,gga}_fxc -- must hold for kxc too.
    void check_family_guards()
    {
        auto gga = require_functional("pbe", DFT::XC::Spin::Unpolarized);
        std::vector<double> v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3;
        auto bad_lda_call = gga.evaluate_lda_kxc({0.1}, 1, v3rho3);
        require(!bad_lda_call.has_value(), "evaluate_lda_kxc on a GGA functional must fail");

        auto lda = require_functional("lda_x", DFT::XC::Spin::Unpolarized);
        auto bad_gga_call = lda.evaluate_gga_kxc({0.1}, {0.01}, 1, v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);
        require(!bad_gga_call.has_value(), "evaluate_gga_kxc on an LDA functional must fail");
    }
} // namespace

int main()
{
    // LDA exchange + LDA correlation (real rho-dependence in v3rho3),
    // and PBE exchange + PBE correlation + B88/LYP -- the ingredients of
    // B2PLYP, all carrying XC_FLAGS_HAVE_KXC.
    check_lda_v3rho3_unpolarized("lda_x", 0.15);
    check_lda_v3rho3_unpolarized("lda_x", 1.0);
    check_lda_v3rho3_unpolarized("lda_c_pw", 0.15);

    check_gga_v3_unpolarized("pbe", 0.15, 0.02);
    check_gga_v3_unpolarized("pbe", 1.0, 0.5);
    check_gga_v3_unpolarized("gga_x_b88", 0.3, 0.05);
    check_gga_v3_unpolarized("gga_c_lyp", 0.3, 0.05);

    check_family_guards();

    return g_ok ? 0 : 1;
}
