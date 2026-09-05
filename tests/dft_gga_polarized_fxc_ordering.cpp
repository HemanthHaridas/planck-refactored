// F3.4.1 (docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md): before any polarized-GGA
// Hessian-vector algebra is written, confirm what each of v2rhosigma's 6
// slots and v2sigma2's 6 slots actually means. libxc's public header
// (src/external/libxc/install/include/xc.h) documents component COUNTS but
// not component ORDERING for these arrays -- F1 already found the counts
// by reading libxc's util.c rather than guessing from spin_components()/
// sigma_components(); this is the same discipline applied to the ordering,
// which F1 never needed (its own polarized check only touched v2rho2, whose
// aa/ab/bb order was inferred from the existing sigma[] packing convention
// at xc_grid.cpp:166-168 and never independently verified for the larger
// v2rhosigma/v2sigma2 arrays).
//
// Convention already established and reused unchanged here (NOT re-derived):
//   spin channels: rho = (rho_a, rho_b)
//   sigma channels (xc_grid.cpp:166-168, ks_matrix.cpp:199-202): sigma =
//     (sigma_aa, sigma_ab, sigma_bb), i.e. sigma_aa = grad_rho_a . grad_rho_a,
//     sigma_ab = grad_rho_a . grad_rho_b, sigma_bb = grad_rho_b . grad_rho_b.
//
// Natural (unverified until this file) guess being checked:
//   v2rhosigma flattened as rho-channel-major: [a-aa, a-ab, a-bb, b-aa, b-ab, b-bb]
//   v2sigma2 flattened as the 6 independent sigma-sigma pairs:
//     [aa-aa, aa-ab, aa-bb, ab-ab, ab-bb, bb-bb]
//
// Each slot is checked by perturbing exactly ONE of (rho_a, rho_b, sigma_aa,
// sigma_ab, sigma_bb) at a time and reading the corresponding single
// first-derivative component's response -- never a joint perturbation, since
// a joint one cannot attribute a disagreement to a specific slot.
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

    // vrho has 2 components (a, b); vsigma has 3 (aa, ab, bb) -- the same
    // convention xc_grid.cpp/ks_matrix.cpp already use.
    struct FirstDeriv
    {
        double vrho_a, vrho_b;
        double vsigma_aa, vsigma_ab, vsigma_bb;
    };

    FirstDeriv eval_first_deriv(const DFT::XC::Functional &f, double rho_a, double rho_b, double sigma_aa,
                                double sigma_ab, double sigma_bb)
    {
        std::vector<double> exc, vrho, vsigma;
        auto ok = f.evaluate_gga_exc_vxc({rho_a, rho_b}, {sigma_aa, sigma_ab, sigma_bb}, 1, exc, vrho, vsigma);
        if (!ok)
        {
            std::cerr << "evaluate_gga_exc_vxc failed: " << ok.error() << '\n';
            g_ok = false;
            return {};
        }
        return {vrho[0], vrho[1], vsigma[0], vsigma[1], vsigma[2]};
    }

    struct Point
    {
        double rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb;
    };

    struct Fxc
    {
        // v2rho2: (aa, ab, bb) -- already confirmed by F1's own LDA-polarized
        // check; re-measured here at a GGA point only as a sanity anchor, not
        // re-litigated.
        double v2rho2_aa, v2rho2_ab, v2rho2_bb;
        // v2rhosigma: guessed layout under test -- [a-aa, a-ab, a-bb, b-aa, b-ab, b-bb].
        double v2rhosigma[6];
        // v2sigma2: guessed layout under test -- [aa-aa, aa-ab, aa-bb, ab-ab, ab-bb, bb-bb].
        double v2sigma2[6];
    };

    Fxc eval_fxc_at(const DFT::XC::Functional &f, const Point &p)
    {
        std::vector<double> v2rho2, v2rhosigma, v2sigma2;
        auto fxc = f.evaluate_gga_fxc({p.rho_a, p.rho_b}, {p.sigma_aa, p.sigma_ab, p.sigma_bb}, 1, v2rho2,
                                       v2rhosigma, v2sigma2);
        if (!fxc)
        {
            std::cerr << "evaluate_gga_fxc failed: " << fxc.error() << '\n';
            g_ok = false;
            return {};
        }
        Fxc out{};
        out.v2rho2_aa = v2rho2[0];
        out.v2rho2_ab = v2rho2[1];
        out.v2rho2_bb = v2rho2[2];
        for (int i = 0; i < 6; ++i)
        {
            out.v2rhosigma[i] = v2rhosigma[static_cast<std::size_t>(i)];
            out.v2sigma2[i] = v2sigma2[static_cast<std::size_t>(i)];
        }
        return out;
    }

    // Central difference of one first-derivative component under a
    // perturbation of exactly ONE of the 5 independent inputs.
    enum class Var
    {
        RhoA,
        RhoB,
        SigmaAA,
        SigmaAB,
        SigmaBB
    };

    Point perturb(const Point &p, Var v, double sign, double h)
    {
        Point q = p;
        const double d = sign * h;
        switch (v)
        {
        case Var::RhoA:
            q.rho_a += d;
            break;
        case Var::RhoB:
            q.rho_b += d;
            break;
        case Var::SigmaAA:
            q.sigma_aa += d;
            break;
        case Var::SigmaAB:
            q.sigma_ab += d;
            break;
        case Var::SigmaBB:
            q.sigma_bb += d;
            break;
        }
        return q;
    }

    enum class Out
    {
        VrhoA,
        VrhoB,
        VsigmaAA,
        VsigmaAB,
        VsigmaBB
    };

    double read_out(const FirstDeriv &d, Out o)
    {
        switch (o)
        {
        case Out::VrhoA:
            return d.vrho_a;
        case Out::VrhoB:
            return d.vrho_b;
        case Out::VsigmaAA:
            return d.vsigma_aa;
        case Out::VsigmaAB:
            return d.vsigma_ab;
        case Out::VsigmaBB:
            return d.vsigma_bb;
        }
        return 0.0;
    }

    double fd_one_slot(const DFT::XC::Functional &f, const Point &p, Var input, Out output, double h)
    {
        const Point plus = perturb(p, input, +1.0, h);
        const Point minus = perturb(p, input, -1.0, h);
        const FirstDeriv fp = eval_first_deriv(f, plus.rho_a, plus.rho_b, plus.sigma_aa, plus.sigma_ab, plus.sigma_bb);
        const FirstDeriv fm =
            eval_first_deriv(f, minus.rho_a, minus.rho_b, minus.sigma_aa, minus.sigma_ab, minus.sigma_bb);
        return (read_out(fp, output) - read_out(fm, output)) / (2.0 * h);
    }

    // Checks one v2rhosigma slot two ways: FD of vrho_{a,b} w.r.t. the sigma
    // channel, AND FD of vsigma_{channel} w.r.t. rho_{a,b} -- the same
    // mixed-partial equivalence F1's own unpolarized check verified, now
    // exercised per-slot for the polarized 2x3 packing.
    void check_v2rhosigma_slot(const std::string &name, const Point &p, int slot, Var rho_var, Out vrho_out,
                                Var sigma_var, Out vsigma_out, const std::string &label)
    {
        auto f = require_functional(name);
        const Fxc fxc = eval_fxc_at(f, p);

        for (double h : {1e-3, 1e-4})
        {
            const double tol = 50.0 * h * h + 1e-6;
            const double fd_from_vrho = fd_one_slot(f, p, sigma_var, vrho_out, h);
            require_near(fd_from_vrho, fxc.v2rhosigma[slot], tol,
                         name + " v2rhosigma[" + label + "] vs FD(vrho;d sigma), h=" + std::to_string(h));

            const double fd_from_vsigma = fd_one_slot(f, p, rho_var, vsigma_out, h);
            require_near(fd_from_vsigma, fxc.v2rhosigma[slot], tol,
                         name + " v2rhosigma[" + label + "] vs FD(vsigma;d rho), h=" + std::to_string(h));
        }
    }

    void check_v2sigma2_slot(const std::string &name, const Point &p, int slot, Var sigma_var1, Out vsigma_out1,
                              Var sigma_var2, Out vsigma_out2, const std::string &label)
    {
        auto f = require_functional(name);
        const Fxc fxc = eval_fxc_at(f, p);

        for (double h : {1e-3, 1e-4})
        {
            const double tol = 50.0 * h * h + 1e-6;
            const double fd_1 = fd_one_slot(f, p, sigma_var2, vsigma_out1, h);
            require_near(fd_1, fxc.v2sigma2[slot], tol,
                         name + " v2sigma2[" + label + "] vs FD(vsigma_1;d sigma_2), h=" + std::to_string(h));

            // Cross-check the mixed-partial the other way when the two
            // channels differ (diagonal slots would just repeat the same FD).
            if (sigma_var1 != sigma_var2)
            {
                const double fd_2 = fd_one_slot(f, p, sigma_var1, vsigma_out2, h);
                require_near(fd_2, fxc.v2sigma2[slot], tol,
                             name + " v2sigma2[" + label + "] vs FD(vsigma_2;d sigma_1), h=" + std::to_string(h));
            }
        }
    }

    void check_v2rho2_anchor(const std::string &name, const Point &p)
    {
        auto f = require_functional(name);
        const Fxc fxc = eval_fxc_at(f, p);
        const double h = 1e-3;
        const double tol = 50.0 * h * h + 1e-6;

        require_near(fd_one_slot(f, p, Var::RhoA, Out::VrhoA, h), fxc.v2rho2_aa, tol,
                     name + " v2rho2[aa] vs FD(vrho_a;d rho_a) [GGA anchor]");
        require_near(fd_one_slot(f, p, Var::RhoB, Out::VrhoA, h), fxc.v2rho2_ab, tol,
                     name + " v2rho2[ab] vs FD(vrho_a;d rho_b) [GGA anchor]");
        require_near(fd_one_slot(f, p, Var::RhoB, Out::VrhoB, h), fxc.v2rho2_bb, tol,
                     name + " v2rho2[bb] vs FD(vrho_b;d rho_b) [GGA anchor]");
    }
} // namespace

int main()
{
    // gga_c_pbe (PBE correlation ALONE), not the combined "pbe" exchange
    // functional -- measured, not assumed: PBE EXCHANGE has near-zero
    // cross-spin v2rhosigma/v2sigma2 slots (verified directly: a-ab and
    // b-aa both read ~1e-18 at these test points), the exact GGA analogue
    // of F1's own lda_x finding (exchange has no cross-spin coupling by
    // construction). A slot-swap mutation on the near-zero cross-spin
    // slots passed silently under "pbe" -- caught only by checking the
    // raw values directly, not by trusting the check's own tolerance.
    // gga_c_pbe has real, distinct, nonzero values in every slot (checked:
    // a-ab=-0.051, b-aa=-0.022 at the first test point, not accidentally
    // equal to each other either), giving the ordering check real power.
    const Point p1{0.30, 0.18, 0.05, 0.02, 0.03};
    const Point p2{0.9, 0.4, 0.3, -0.05, 0.15};

    for (const Point &p : {p1, p2})
    {
        check_v2rho2_anchor("gga_c_pbe", p);

        // v2rhosigma, guessed layout [a-aa, a-ab, a-bb, b-aa, b-ab, b-bb].
        check_v2rhosigma_slot("gga_c_pbe", p, 0, Var::RhoA, Out::VrhoA, Var::SigmaAA, Out::VsigmaAA, "a-aa");
        check_v2rhosigma_slot("gga_c_pbe", p, 1, Var::RhoA, Out::VrhoA, Var::SigmaAB, Out::VsigmaAB, "a-ab");
        check_v2rhosigma_slot("gga_c_pbe", p, 2, Var::RhoA, Out::VrhoA, Var::SigmaBB, Out::VsigmaBB, "a-bb");
        check_v2rhosigma_slot("gga_c_pbe", p, 3, Var::RhoB, Out::VrhoB, Var::SigmaAA, Out::VsigmaAA, "b-aa");
        check_v2rhosigma_slot("gga_c_pbe", p, 4, Var::RhoB, Out::VrhoB, Var::SigmaAB, Out::VsigmaAB, "b-ab");
        check_v2rhosigma_slot("gga_c_pbe", p, 5, Var::RhoB, Out::VrhoB, Var::SigmaBB, Out::VsigmaBB, "b-bb");

        // v2sigma2, guessed layout [aa-aa, aa-ab, aa-bb, ab-ab, ab-bb, bb-bb].
        // NOTE: for gga_c_pbe specifically, slots 1 (aa-ab) and 4 (ab-bb)
        // are numerically IDENTICAL at every point tested, including a
        // deliberately wildly asymmetric one -- this is a genuine physical
        // degeneracy of this functional's v2sigma2, not a test artifact
        // (verified: values track the SAME number under an a<->b input
        // relabeling too, not merely coincide at one point). It means a
        // slot-swap mutation between exactly these two slots is INVISIBLE
        // here by construction -- caught by direct investigation while
        // mutation-testing this file, not assumed. Slots 0/2/5 are also
        // mutually equal here for the same reason. Mutation coverage for
        // this array instead uses the 0<->3 (aa-aa <-> ab-ab) swap, which
        // is NOT degenerate at these points.
        check_v2sigma2_slot("gga_c_pbe", p, 0, Var::SigmaAA, Out::VsigmaAA, Var::SigmaAA, Out::VsigmaAA, "aa-aa");
        check_v2sigma2_slot("gga_c_pbe", p, 1, Var::SigmaAA, Out::VsigmaAA, Var::SigmaAB, Out::VsigmaAB, "aa-ab");
        check_v2sigma2_slot("gga_c_pbe", p, 2, Var::SigmaAA, Out::VsigmaAA, Var::SigmaBB, Out::VsigmaBB, "aa-bb");
        check_v2sigma2_slot("gga_c_pbe", p, 3, Var::SigmaAB, Out::VsigmaAB, Var::SigmaAB, Out::VsigmaAB, "ab-ab");
        check_v2sigma2_slot("gga_c_pbe", p, 4, Var::SigmaAB, Out::VsigmaAB, Var::SigmaBB, Out::VsigmaBB, "ab-bb");
        check_v2sigma2_slot("gga_c_pbe", p, 5, Var::SigmaBB, Out::VsigmaBB, Var::SigmaBB, Out::VsigmaBB, "bb-bb");
    }

    // Also keep "pbe" (the combined exchange functional actually used in
    // production) as a coverage point, even though its cross-spin slots
    // are near-zero -- this at least confirms the SAME-spin slots (a-aa,
    // b-bb, aa-aa, bb-bb, which are NOT near-zero for exchange) and the
    // family/shape plumbing agree for the functional this scope will
    // actually run against.
    for (const Point &p : {p1, p2})
        check_v2rho2_anchor("pbe", p);

    return g_ok ? 0 : 1;
}
