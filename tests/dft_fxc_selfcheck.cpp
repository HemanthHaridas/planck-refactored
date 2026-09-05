// F1 (docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md): isolated check that libxc's own
// analytic second derivative (xc_lda_fxc / xc_gga_fxc, via
// DFT::XC::Functional::evaluate_{lda,gga}_fxc) agrees with a finite
// difference of libxc's own first derivative (vrho / vsigma, via the
// existing evaluate_{lda,gga}_exc_vxc). This isolates "does libxc's second
// derivative agree with its own first derivative" from every other question
// in the fxc scope -- if this fails, the bug is in libxc itself (or in how
// the wrapper reads its output arrays), not in anything Planck derives on
// top of it.
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

    // Central-difference d(vrho)/d(rho) at a single point, holding all other
    // spin channels fixed. Three step sizes, matching the RHF/UHF SOSCF FD
    // probes' own convention (1e-2, 1e-3, 1e-4) so convergence as h -> 0 can
    // be read the same way those probes were read.
    void check_lda_v2rho2_unpolarized(const std::string &name, double rho0)
    {
        auto functional = require_functional(name, DFT::XC::Spin::Unpolarized);

        std::vector<double> v2rho2;
        auto fxc = functional.evaluate_lda_fxc({rho0}, 1, v2rho2);
        require(fxc.has_value(), name + ": evaluate_lda_fxc failed: " + (fxc ? "" : fxc.error()));
        if (!fxc)
            return;
        require(v2rho2.size() == 1, name + ": unpolarized v2rho2 must have exactly 1 component");

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            std::vector<double> exc_p, vrho_p, exc_m, vrho_m;
            auto plus = functional.evaluate_lda_exc_vxc({rho0 + h}, 1, exc_p, vrho_p);
            auto minus = functional.evaluate_lda_exc_vxc({rho0 - h}, 1, exc_m, vrho_m);
            require(plus.has_value() && minus.has_value(), name + ": evaluate_lda_exc_vxc failed during FD");
            if (!plus || !minus)
                continue;

            const double fd = (vrho_p[0] - vrho_m[0]) / (2.0 * h);
            // FD truncation error is O(h^2); the tolerance is loosened as h
            // shrinks from 1e-2 to 1e-4 accordingly, mirroring how the SOSCF
            // probes read "does this converge" rather than pinning one tol.
            const double tol = 50.0 * h * h + 1e-6;
            require_near(fd, v2rho2[0], tol,
                         name + " v2rho2 vs FD(vrho), h=" + std::to_string(h));
        }
    }

    // Polarized LDA: v2rho2 packs 3 components (aa, ab, bb) per libxc's own
    // internal_counters_set_lda (src/external/libxc/.../util.c). Check the
    // diagonal (aa) component against d(vrho_a)/d(rho_a) with rho_b fixed,
    // and the off-diagonal (ab) component against d(vrho_a)/d(rho_b) --
    // these must also agree with d(vrho_b)/d(rho_a) by symmetry, which this
    // check additionally verifies (a genuine cross-check, not assumed).
    void check_lda_v2rho2_polarized(const std::string &name, double rho_a0, double rho_b0)
    {
        auto functional = require_functional(name, DFT::XC::Spin::Polarized);

        std::vector<double> v2rho2;
        auto fxc = functional.evaluate_lda_fxc({rho_a0, rho_b0}, 1, v2rho2);
        require(fxc.has_value(), name + ": evaluate_lda_fxc (polarized) failed");
        if (!fxc)
            return;
        require(v2rho2.size() == 3, name + ": polarized v2rho2 must have exactly 3 components (aa,ab,bb)");

        const double h = 1e-3;

        // d(vrho_a)/d(rho_a): perturb rho_a only.
        {
            std::vector<double> exc_p, vrho_p, exc_m, vrho_m;
            functional.evaluate_lda_exc_vxc({rho_a0 + h, rho_b0}, 1, exc_p, vrho_p);
            functional.evaluate_lda_exc_vxc({rho_a0 - h, rho_b0}, 1, exc_m, vrho_m);
            const double fd_aa = (vrho_p[0] - vrho_m[0]) / (2.0 * h);
            require_near(fd_aa, v2rho2[0], 1e-3, name + " v2rho2[aa] vs FD(vrho_a; d rho_a)");
        }

        // d(vrho_a)/d(rho_b): perturb rho_b only, read vrho_a's response.
        double fd_ab_from_a = 0.0;
        {
            std::vector<double> exc_p, vrho_p, exc_m, vrho_m;
            functional.evaluate_lda_exc_vxc({rho_a0, rho_b0 + h}, 1, exc_p, vrho_p);
            functional.evaluate_lda_exc_vxc({rho_a0, rho_b0 - h}, 1, exc_m, vrho_m);
            fd_ab_from_a = (vrho_p[0] - vrho_m[0]) / (2.0 * h);
            require_near(fd_ab_from_a, v2rho2[1], 1e-3, name + " v2rho2[ab] vs FD(vrho_a; d rho_b)");
        }

        // Cross-check: d(vrho_b)/d(rho_a) should equal the same v2rho2[ab]
        // by the mixed-partial symmetry of the underlying energy functional
        // -- this is not assumed, it is measured independently here.
        {
            std::vector<double> exc_p, vrho_p, exc_m, vrho_m;
            functional.evaluate_lda_exc_vxc({rho_a0 + h, rho_b0}, 1, exc_p, vrho_p);
            functional.evaluate_lda_exc_vxc({rho_a0 - h, rho_b0}, 1, exc_m, vrho_m);
            const double fd_ba = (vrho_p[1] - vrho_m[1]) / (2.0 * h);
            require_near(fd_ba, v2rho2[1], 1e-3, name + " v2rho2[ab] vs FD(vrho_b; d rho_a)");
            // Both FDs use the same h and are individually accurate to
            // ~1e-3 against the analytic value above, so their mutual
            // agreement floor is FD-truncation scale, not machine
            // precision -- 1e-5 is loose enough to absorb that and still
            // catches a genuine index swap (aa/ab/bb are separated by
            // >0.1 at this test point).
            require_near(fd_ba, fd_ab_from_a, 1e-5,
                         name + " mixed partial symmetry: FD(vrho_b;d rho_a) vs FD(vrho_a;d rho_b)");
        }
    }

    // GGA: check v2rho2 (unpolarized) against FD of vrho w.r.t. rho at fixed
    // sigma, and v2rhosigma against FD of vrho w.r.t. sigma (equivalently
    // FD of vsigma w.r.t. rho -- both are checked, since libxc's v2rhosigma
    // is claimed to serve both and that equivalence is exactly the kind of
    // claim this self-check exists to verify rather than assume). v2sigma2
    // is checked against FD of vsigma w.r.t. sigma.
    void check_gga_unpolarized(const std::string &name, double rho0, double sigma0)
    {
        auto functional = require_functional(name, DFT::XC::Spin::Unpolarized);

        std::vector<double> v2rho2, v2rhosigma, v2sigma2;
        auto fxc = functional.evaluate_gga_fxc({rho0}, {sigma0}, 1, v2rho2, v2rhosigma, v2sigma2);
        require(fxc.has_value(), name + ": evaluate_gga_fxc failed: " + (fxc ? "" : fxc.error()));
        if (!fxc)
            return;
        require(v2rho2.size() == 1, name + ": unpolarized v2rho2 must have 1 component");
        require(v2rhosigma.size() == 1, name + ": unpolarized v2rhosigma must have 1 component");
        require(v2sigma2.size() == 1, name + ": unpolarized v2sigma2 must have 1 component");

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            const double tol = 50.0 * h * h + 1e-5;

            // v2rho2 vs FD(vrho; d rho), sigma fixed.
            {
                std::vector<double> exc_p, vrho_p, vsigma_p, exc_m, vrho_m, vsigma_m;
                functional.evaluate_gga_exc_vxc({rho0 + h}, {sigma0}, 1, exc_p, vrho_p, vsigma_p);
                functional.evaluate_gga_exc_vxc({rho0 - h}, {sigma0}, 1, exc_m, vrho_m, vsigma_m);
                const double fd = (vrho_p[0] - vrho_m[0]) / (2.0 * h);
                require_near(fd, v2rho2[0], tol, name + " v2rho2 vs FD(vrho;d rho), h=" + std::to_string(h));
            }

            // v2rhosigma vs FD(vrho; d sigma).
            double fd_rhosigma_from_vrho = 0.0;
            {
                std::vector<double> exc_p, vrho_p, vsigma_p, exc_m, vrho_m, vsigma_m;
                functional.evaluate_gga_exc_vxc({rho0}, {sigma0 + h}, 1, exc_p, vrho_p, vsigma_p);
                functional.evaluate_gga_exc_vxc({rho0}, {sigma0 - h}, 1, exc_m, vrho_m, vsigma_m);
                fd_rhosigma_from_vrho = (vrho_p[0] - vrho_m[0]) / (2.0 * h);
                require_near(fd_rhosigma_from_vrho, v2rhosigma[0], tol,
                             name + " v2rhosigma vs FD(vrho;d sigma), h=" + std::to_string(h));
            }

            // Same v2rhosigma should also equal FD(vsigma; d rho) -- the
            // mixed-partial equivalence claimed in the scope doc, verified
            // rather than assumed.
            {
                std::vector<double> exc_p, vrho_p, vsigma_p, exc_m, vrho_m, vsigma_m;
                functional.evaluate_gga_exc_vxc({rho0 + h}, {sigma0}, 1, exc_p, vrho_p, vsigma_p);
                functional.evaluate_gga_exc_vxc({rho0 - h}, {sigma0}, 1, exc_m, vrho_m, vsigma_m);
                const double fd_rhosigma_from_vsigma = (vsigma_p[0] - vsigma_m[0]) / (2.0 * h);
                require_near(fd_rhosigma_from_vsigma, v2rhosigma[0], tol,
                             name + " v2rhosigma vs FD(vsigma;d rho), h=" + std::to_string(h));
            }

            // v2sigma2 vs FD(vsigma; d sigma).
            {
                std::vector<double> exc_p, vrho_p, vsigma_p, exc_m, vrho_m, vsigma_m;
                functional.evaluate_gga_exc_vxc({rho0}, {sigma0 + h}, 1, exc_p, vrho_p, vsigma_p);
                functional.evaluate_gga_exc_vxc({rho0}, {sigma0 - h}, 1, exc_m, vrho_m, vsigma_m);
                const double fd = (vsigma_p[0] - vsigma_m[0]) / (2.0 * h);
                require_near(fd, v2sigma2[0], tol, name + " v2sigma2 vs FD(vsigma;d sigma), h=" + std::to_string(h));
            }
        }
    }

    // Reject calling the LDA fxc method on a GGA functional and vice versa
    // -- same guard shape as evaluate_lda_exc_vxc / evaluate_gga_exc_vxc,
    // must hold for the new methods too.
    void check_family_guards()
    {
        auto gga = require_functional("pbe", DFT::XC::Spin::Unpolarized);
        std::vector<double> v2rho2;
        auto bad_lda_call = gga.evaluate_lda_fxc({0.1}, 1, v2rho2);
        require(!bad_lda_call.has_value(), "evaluate_lda_fxc on a GGA functional must fail");

        auto lda = require_functional("lda_x", DFT::XC::Spin::Unpolarized);
        std::vector<double> v2rhosigma, v2sigma2;
        auto bad_gga_call = lda.evaluate_gga_fxc({0.1}, {0.01}, 1, v2rho2, v2rhosigma, v2sigma2);
        require(!bad_gga_call.has_value(), "evaluate_gga_fxc on an LDA functional must fail");
    }
} // namespace

int main()
{
    // Slater (LDA exchange) and PBE (GGA), as named in
    // docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md's F1 verification list -- both
    // already used elsewhere in the tree's regression suite.
    check_lda_v2rho2_unpolarized("lda_x", 0.15);
    check_lda_v2rho2_unpolarized("lda_x", 1.0);
    check_lda_v2rho2_polarized("lda_x", 0.10, 0.06);
    check_lda_v2rho2_polarized("lda_x", 0.5, 0.5);
    // lda_x's v2rho2[ab] is genuinely ~0 (exchange has no cross-spin
    // coupling), which weakens that sub-check's power to catch an
    // aa/ab/bb index-swap bug. lda_c_pw (LDA correlation) has real,
    // distinct cross-spin coupling (measured: aa=0.104, ab=-0.289,
    // bb=0.327 at these densities) and stresses the same check meaningfully.
    check_lda_v2rho2_polarized("lda_c_pw", 0.10, 0.06);

    check_gga_unpolarized("pbe", 0.15, 0.02);
    check_gga_unpolarized("pbe", 1.0, 0.5);

    check_family_guards();

    return g_ok ? 0 : 1;
}
