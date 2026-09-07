// F3.4 (docs/DFT_ANALYTIC_FXC_HESSIAN.md): the GGA polarized analytic
// Hessian-vector product, broken into F3.4.2 (same-spin alpha-only x),
// F3.4.3 (add cross-spin coupling), F3.4.4 (beta channel). Point-level
// checks against libxc's own finite difference, extending F3.3's unpolarized
// T1/T2/T3 pattern (dft_gga_hessian_selfcheck.cpp) to the polarized case.
//
// Ground-state first-derivative potential (matches ks_matrix.cpp:194-206
// exactly, "coefficient_alpha"/"coefficient_beta"):
//   V_xc^a = vrho_a*AA + [2*vsigma_aa*grad_rho_a + vsigma_ab*grad_rho_b] . AG
//   V_xc^b = vrho_b*AA + [vsigma_ab*grad_rho_a + 2*vsigma_bb*grad_rho_b] . AG
//
// F3.4.2 differentiates V_xc^a under an ALPHA-ONLY perturbation
// (drho_b = dgrad_rho_b = 0 identically). Per the scope doc's own resolved
// question: this still includes the response of the CROSS coefficient
// vsigma_ab (which depends on rho_a too) contracted against the UNCHANGED
// grad_rho_b -- split by INPUT direction, not by which coefficient slot is
// touched. That gives four structurally distinct terms:
//
//   T1 = delta[vrho_a] * AA
//   T2 = 2 * delta[vsigma_aa] * (grad_rho_a . AG)
//   T3 = 2 * vsigma_aa * (delta_grad_rho_a . AG)
//   T4 = delta[vsigma_ab] * (grad_rho_b . AG)   -- NEW: same-spin (alpha)
//        INPUT driving the CROSS coefficient, contracted against the
//        unchanged ground-state grad_rho_b. Present even for a pure
//        alpha-only x, since vsigma_ab = vsigma_ab(rho_a, rho_b, sigma_aa,
//        sigma_ab, sigma_bb) depends on rho_a/sigma_aa too.
//
// THE SUBTLETY THAT COST A DEBUGGING PASS: "alpha-only x" means
// drho_b = dgrad_rho_b = 0, but this does NOT mean dsigma_ab = 0.
// sigma_ab = grad_rho_a . grad_rho_b is LINEAR in grad_rho_a alone, so it
// still responds to delta_grad_rho_a even with grad_rho_b held fixed:
//   dsigma_aa = 2 * (grad_rho_a . delta_grad_rho_a)   (sigma_aa is quadratic in grad_rho_a)
//   dsigma_ab = delta_grad_rho_a . grad_rho_b          (sigma_ab is LINEAR in grad_rho_a -- no factor of 2)
//   dsigma_bb = 0                                       (sigma_bb depends only on grad_rho_b)
// An initial version of this file dropped the dsigma_ab contribution
// entirely (treating "alpha-only x" as if it implied dsigma_ab=0 the same
// way it implies dsigma_bb=0), and disagreed with the FD oracle by a
// small but real, non-shrinking-with-h amount (~5e-4 out of ~4e-4,
// i.e. off by more than 100%) at h=1e-3/1e-4 -- caught by isolating
// delta[vsigma_ab]'s OWN coefficient against a direct FD before trusting
// the combined T1+T2+T3+T4 sum, the same "isolate before combining"
// discipline F3.3 already used. Every coefficient that reads a
// sigma_ab-rooted v2rhosigma/v2sigma2 slot needs the dsigma_ab term:
//
//   delta[vrho_a]    = v2rho2_aa * drho_a
//                       + v2rhosigma[a-aa] * dsigma_aa + v2rhosigma[a-ab] * dsigma_ab
//   delta[vsigma_aa] = v2rhosigma[a-aa] * drho_a
//                       + v2sigma2[aa-aa] * dsigma_aa + v2sigma2[aa-ab] * dsigma_ab
//   delta[vsigma_ab] = v2rhosigma[a-ab] * drho_a
//                       + v2sigma2[aa-ab] * dsigma_aa + v2sigma2[ab-ab] * dsigma_ab
//
// using F3.4.1's own confirmed v2rhosigma=[a-aa,a-ab,a-bb,b-aa,b-ab,b-bb]
// and v2sigma2=[aa-aa,aa-ab,aa-bb,ab-ab,ab-bb,bb-bb] layout. Verified
// EXACT (to ~1e-13, floating-point noise) against a raw FD of the full
// V_xc^a scalar at every step size once this correction was made --
// confirming the earlier disagreement was exactly this missing term, not
// a deeper formula error.
//
// F3.4.3 adds a nonzero drho_b/dgrad_rho_b (a MIXED alpha/beta trial x).
// Per the scope doc's own carry-forward note, the general
// dsigma_ab = grad_rho_b.delta_grad_rho_a + grad_rho_a.delta_grad_rho_b
// now has BOTH halves nonzero (F3.4.2 only exercised the first). Every
// F3.4.2 coefficient formula gains drho_b and dsigma_bb terms:
//
//   delta[vrho_a]    = v2rho2_aa*drho_a + v2rho2_ab*drho_b
//                       + v2rhosigma[a-aa]*dsigma_aa + v2rhosigma[a-ab]*dsigma_ab + v2rhosigma[a-bb]*dsigma_bb
//   delta[vsigma_aa] = v2rhosigma[a-aa]*drho_a + v2rhosigma[b-aa]*drho_b
//                       + v2sigma2[aa-aa]*dsigma_aa + v2sigma2[aa-ab]*dsigma_ab + v2sigma2[aa-bb]*dsigma_bb
//   delta[vsigma_ab] = v2rhosigma[a-ab]*drho_a + v2rhosigma[b-ab]*drho_b
//                       + v2sigma2[aa-ab]*dsigma_aa + v2sigma2[ab-ab]*dsigma_ab + v2sigma2[ab-bb]*dsigma_bb
//
// plus a genuinely NEW fifth term, T5, from differentiating the ARGUMENT
// grad_rho_b inside coefficient_alpha's existing vsigma_ab*grad_rho_b
// piece (the cross-spin sibling of F3.3.3's T3, exactly as the doc
// predicted):
//   T5 = vsigma_ab * (delta_grad_rho_b . AG)   -- NO factor of 2 (unlike
//        T3's 2*vsigma_aa*(dgrad_rho_a.AG) for the SELF term), matching
//        coefficient_alpha's own asymmetric weighting
//        (2*vsigma_aa*grad_rho_a + vsigma_ab*grad_rho_b, ks_matrix.cpp:200).
//
// Verified EXACT (to ~1e-11, floating-point/FD-truncation noise) against a
// raw FD of the full V_xc^a scalar on the FIRST attempt this time --
// F3.4.2's dsigma_ab lesson carried forward correctly rather than being
// rediscovered. T5 was also confirmed non-negligible before trusting the
// check (measured: 67% of the total delta[V_xc^a] at the test point below,
// not a symmetry-suppressed direction).
//
// F3.4.4 mirrors F3.4.2/F3.4.3 for the BETA channel, checked
// INDEPENDENTLY rather than assumed symmetric and copy-pasted --
// coefficient_beta's own asymmetric weighting (vsigma_ab*grad_rho_a +
// 2*vsigma_bb*grad_rho_b, ks_matrix.cpp:201-202) is the alpha<->beta,
// aa<->bb mirror of coefficient_alpha, with the SAME shared "ab" slot
// (not mirrored -- there is only one cross-spin sigma channel). Mirroring
// systematically gives:
//   T1' = delta[vrho_b] * AA
//   T2' = 2 * delta[vsigma_bb] * (grad_rho_b . AG)
//   T3' = 2 * vsigma_bb * (delta_grad_rho_b . AG)
//   T4' = delta[vsigma_ab] * (grad_rho_a . AG)
//   T5' = vsigma_ab * (delta_grad_rho_a . AG)
// where every coefficient rooted at "aa" in the alpha formulas becomes
// "bb" here (v2rho2_bb instead of v2rho2_aa, v2rhosigma[b-bb]/[b-ab]/[b-aa]
// instead of [a-aa]/[a-ab]/[a-bb], v2sigma2[bb-bb]/[ab-bb]/[aa-bb] instead
// of [aa-aa]/[aa-ab]/[aa-bb]) while the "ab" cross-term itself is READ
// FROM THE SAME v2rhosigma[a-ab]/[b-ab] and v2sigma2[ab-ab] slots the
// alpha formulas already used -- not a second independent set. Verified
// EXACT (to ~1e-11) against a raw FD of the full V_xc^b scalar on the
// FIRST attempt, for both a beta-only trial (T5'=0 identically, since
// delta_grad_rho_a=0) and a mixed trial (both T4'/T5' nonzero).
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dft/base/wrapper.h"

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

    double dot(double ax, double ay, double az, double bx, double by, double bz)
    {
        return ax * bx + ay * by + az * bz;
    }

    struct Point
    {
        double rho_a, rho_b;
        double gax, gay, gaz; // grad_rho_a
        double gbx, gby, gbz; // grad_rho_b
    };

    struct Perturbation
    {
        // Alpha-only: drho_b = dgrad_rho_b = 0 always for F3.4.2.
        double drho_a;
        double dgax, dgay, dgaz; // delta_grad_rho_a
    };

    // F3.4.3: a general (mixed) perturbation with both alpha and beta
    // components nonzero.
    struct MixedPerturbation
    {
        double drho_a, drho_b;
        double dgax, dgay, dgaz; // delta_grad_rho_a
        double dgbx, dgby, dgbz; // delta_grad_rho_b
    };

    struct Fxc
    {
        double v2rho2_aa, v2rho2_ab, v2rho2_bb;
        double v2rhosigma[6]; // [a-aa,a-ab,a-bb,b-aa,b-ab,b-bb]
        double v2sigma2[6];   // [aa-aa,aa-ab,aa-bb,ab-ab,ab-bb,bb-bb]
    };

    Fxc eval_fxc_at(const DFT::XC::Functional &f, const Point &p)
    {
        const double sigma_aa = dot(p.gax, p.gay, p.gaz, p.gax, p.gay, p.gaz);
        const double sigma_ab = dot(p.gax, p.gay, p.gaz, p.gbx, p.gby, p.gbz);
        const double sigma_bb = dot(p.gbx, p.gby, p.gbz, p.gbx, p.gby, p.gbz);
        std::vector<double> v2rho2, v2rhosigma, v2sigma2;
        auto fxc = f.evaluate_gga_fxc({p.rho_a, p.rho_b}, {sigma_aa, sigma_ab, sigma_bb}, 1, v2rho2, v2rhosigma,
                                       v2sigma2);
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

    struct AOFactors
    {
        double AA;
        double AGx, AGy, AGz;
    };

    // The full ground-state V_xc^a scalar, matching ks_matrix.cpp's
    // coefficient_alpha exactly.
    double eval_vxc_alpha_scalar(const DFT::XC::Functional &f, double rho_a, double rho_b, double gax, double gay,
                                  double gaz, double gbx, double gby, double gbz, const AOFactors &ao)
    {
        const double sigma_aa = dot(gax, gay, gaz, gax, gay, gaz);
        const double sigma_ab = dot(gax, gay, gaz, gbx, gby, gbz);
        const double sigma_bb = dot(gbx, gby, gbz, gbx, gby, gbz);
        std::vector<double> exc, vrho, vsigma;
        f.evaluate_gga_exc_vxc({rho_a, rho_b}, {sigma_aa, sigma_ab, sigma_bb}, 1, exc, vrho, vsigma);

        const double g_a_dot_AG = dot(gax, gay, gaz, ao.AGx, ao.AGy, ao.AGz);
        const double g_b_dot_AG = dot(gbx, gby, gbz, ao.AGx, ao.AGy, ao.AGz);
        // coefficient_alpha = 2*vsigma_aa*grad_rho_a + vsigma_ab*grad_rho_b
        return vrho[0] * ao.AA + 2.0 * vsigma[0] * g_a_dot_AG + vsigma[1] * g_b_dot_AG;
    }

    double fd_delta_vxc_alpha(const DFT::XC::Functional &f, const Point &p, const Perturbation &d,
                               const AOFactors &ao, double h)
    {
        const double vp = eval_vxc_alpha_scalar(f, p.rho_a + h * d.drho_a, p.rho_b, p.gax + h * d.dgax,
                                                 p.gay + h * d.dgay, p.gaz + h * d.dgaz, p.gbx, p.gby, p.gbz, ao);
        const double vm = eval_vxc_alpha_scalar(f, p.rho_a - h * d.drho_a, p.rho_b, p.gax - h * d.dgax,
                                                 p.gay - h * d.dgay, p.gaz - h * d.dgaz, p.gbx, p.gby, p.gbz, ao);
        return (vp - vm) / (2.0 * h);
    }

    void check_alpha_only(const std::string &name, const Point &p, const Perturbation &d, const AOFactors &ao)
    {
        auto f = require_functional(name);
        const Fxc fxc = eval_fxc_at(f, p);

        std::vector<double> exc0, vrho0, vsigma0;
        const double sigma_aa0 = dot(p.gax, p.gay, p.gaz, p.gax, p.gay, p.gaz);
        const double sigma_ab0 = dot(p.gax, p.gay, p.gaz, p.gbx, p.gby, p.gbz);
        const double sigma_bb0 = dot(p.gbx, p.gby, p.gbz, p.gbx, p.gby, p.gbz);
        f.evaluate_gga_exc_vxc({p.rho_a, p.rho_b}, {sigma_aa0, sigma_ab0, sigma_bb0}, 1, exc0, vrho0, vsigma0);
        const double vsigma_aa0 = vsigma0[0];

        const double g_a_dot_AG = dot(p.gax, p.gay, p.gaz, ao.AGx, ao.AGy, ao.AGz);
        const double g_b_dot_AG = dot(p.gbx, p.gby, p.gbz, ao.AGx, ao.AGy, ao.AGz);
        const double dg_a_dot_AG = dot(d.dgax, d.dgay, d.dgaz, ao.AGx, ao.AGy, ao.AGz);

        // dsigma_aa = 2*(grad_rho_a . delta_grad_rho_a) -- sigma_aa is
        // quadratic in grad_rho_a. dsigma_ab = delta_grad_rho_a . grad_rho_b
        // -- sigma_ab is LINEAR in grad_rho_a, no factor of 2, and this is
        // the term an "alpha-only x implies dsigma_ab=0" assumption misses
        // (see the file header comment for the debugging story).
        const double dsigma_aa = 2.0 * dot(p.gax, p.gay, p.gaz, d.dgax, d.dgay, d.dgaz);
        const double dsigma_ab = dot(d.dgax, d.dgay, d.dgaz, p.gbx, p.gby, p.gbz);

        const double delta_vrho_a =
            fxc.v2rho2_aa * d.drho_a + fxc.v2rhosigma[0] * dsigma_aa + fxc.v2rhosigma[1] * dsigma_ab;
        const double delta_vsigma_aa =
            fxc.v2rhosigma[0] * d.drho_a + fxc.v2sigma2[0] * dsigma_aa + fxc.v2sigma2[1] * dsigma_ab;
        const double delta_vsigma_ab =
            fxc.v2rhosigma[1] * d.drho_a + fxc.v2sigma2[1] * dsigma_aa + fxc.v2sigma2[3] * dsigma_ab;

        const double T1 = delta_vrho_a * ao.AA;
        const double T2 = 2.0 * delta_vsigma_aa * g_a_dot_AG;
        const double T3 = 2.0 * vsigma_aa0 * dg_a_dot_AG;
        const double T4 = delta_vsigma_ab * g_b_dot_AG;
        const double delta_vxc_a_analytic = T1 + T2 + T3 + T4;

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            const double delta_vxc_a_fd = fd_delta_vxc_alpha(f, p, d, ao, h);
            const double tol = 50.0 * h * h + 1e-6;
            require_near(delta_vxc_a_analytic, delta_vxc_a_fd, tol,
                         name + " T1+T2+T3+T4 (delta[V_xc^a], alpha-only x) vs FD, h=" + std::to_string(h));
        }
    }

    double fd_delta_vxc_alpha_mixed(const DFT::XC::Functional &f, const Point &p, const MixedPerturbation &d,
                                     const AOFactors &ao, double h)
    {
        const double vp = eval_vxc_alpha_scalar(f, p.rho_a + h * d.drho_a, p.rho_b + h * d.drho_b,
                                                 p.gax + h * d.dgax, p.gay + h * d.dgay, p.gaz + h * d.dgaz,
                                                 p.gbx + h * d.dgbx, p.gby + h * d.dgby, p.gbz + h * d.dgbz, ao);
        const double vm = eval_vxc_alpha_scalar(f, p.rho_a - h * d.drho_a, p.rho_b - h * d.drho_b,
                                                 p.gax - h * d.dgax, p.gay - h * d.dgay, p.gaz - h * d.dgaz,
                                                 p.gbx - h * d.dgbx, p.gby - h * d.dgby, p.gbz - h * d.dgbz, ao);
        return (vp - vm) / (2.0 * h);
    }

    // F3.4.3: full T1+T2+T3+T4+T5 sum, mixed alpha/beta trial x.
    void check_mixed(const std::string &name, const Point &p, const MixedPerturbation &d, const AOFactors &ao)
    {
        auto f = require_functional(name);
        const Fxc fxc = eval_fxc_at(f, p);

        std::vector<double> exc0, vrho0, vsigma0;
        const double sigma_aa0 = dot(p.gax, p.gay, p.gaz, p.gax, p.gay, p.gaz);
        const double sigma_ab0 = dot(p.gax, p.gay, p.gaz, p.gbx, p.gby, p.gbz);
        const double sigma_bb0 = dot(p.gbx, p.gby, p.gbz, p.gbx, p.gby, p.gbz);
        f.evaluate_gga_exc_vxc({p.rho_a, p.rho_b}, {sigma_aa0, sigma_ab0, sigma_bb0}, 1, exc0, vrho0, vsigma0);
        const double vsigma_aa0 = vsigma0[0];
        const double vsigma_ab0 = vsigma0[1];

        const double g_a_dot_AG = dot(p.gax, p.gay, p.gaz, ao.AGx, ao.AGy, ao.AGz);
        const double g_b_dot_AG = dot(p.gbx, p.gby, p.gbz, ao.AGx, ao.AGy, ao.AGz);
        const double dg_a_dot_AG = dot(d.dgax, d.dgay, d.dgaz, ao.AGx, ao.AGy, ao.AGz);
        const double dg_b_dot_AG = dot(d.dgbx, d.dgby, d.dgbz, ao.AGx, ao.AGy, ao.AGz);

        // General dsigma_ab = grad_rho_b.delta_grad_rho_a + grad_rho_a.delta_grad_rho_b
        // -- BOTH halves nonzero now, unlike F3.4.2's alpha-only case where
        // only the first half survived (delta_grad_rho_b=0 there).
        const double dsigma_aa = 2.0 * dot(p.gax, p.gay, p.gaz, d.dgax, d.dgay, d.dgaz);
        const double dsigma_ab = dot(p.gbx, p.gby, p.gbz, d.dgax, d.dgay, d.dgaz) +
                                  dot(p.gax, p.gay, p.gaz, d.dgbx, d.dgby, d.dgbz);
        const double dsigma_bb = 2.0 * dot(p.gbx, p.gby, p.gbz, d.dgbx, d.dgby, d.dgbz);

        const double delta_vrho_a = fxc.v2rho2_aa * d.drho_a + fxc.v2rho2_ab * d.drho_b +
                                     fxc.v2rhosigma[0] * dsigma_aa + fxc.v2rhosigma[1] * dsigma_ab +
                                     fxc.v2rhosigma[2] * dsigma_bb;
        const double delta_vsigma_aa = fxc.v2rhosigma[0] * d.drho_a + fxc.v2rhosigma[3] * d.drho_b +
                                        fxc.v2sigma2[0] * dsigma_aa + fxc.v2sigma2[1] * dsigma_ab +
                                        fxc.v2sigma2[2] * dsigma_bb;
        const double delta_vsigma_ab = fxc.v2rhosigma[1] * d.drho_a + fxc.v2rhosigma[4] * d.drho_b +
                                        fxc.v2sigma2[1] * dsigma_aa + fxc.v2sigma2[3] * dsigma_ab +
                                        fxc.v2sigma2[4] * dsigma_bb;

        const double T1 = delta_vrho_a * ao.AA;
        const double T2 = 2.0 * delta_vsigma_aa * g_a_dot_AG;
        const double T3 = 2.0 * vsigma_aa0 * dg_a_dot_AG;
        const double T4 = delta_vsigma_ab * g_b_dot_AG;
        // T5: differentiating the ARGUMENT grad_rho_b inside
        // coefficient_alpha's existing vsigma_ab*grad_rho_b piece -- the
        // cross-spin sibling of F3.3.3's T3. No factor of 2 (matches
        // coefficient_alpha's own asymmetric weighting).
        const double T5 = vsigma_ab0 * dg_b_dot_AG;
        const double delta_vxc_a_analytic = T1 + T2 + T3 + T4 + T5;

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            const double delta_vxc_a_fd = fd_delta_vxc_alpha_mixed(f, p, d, ao, h);
            const double tol = 50.0 * h * h + 1e-6;
            require_near(delta_vxc_a_analytic, delta_vxc_a_fd, tol,
                         name + " T1+T2+T3+T4+T5 (delta[V_xc^a], mixed x) vs FD, h=" + std::to_string(h));
        }
    }

    // F3.4.4: the beta-channel mirror. The full ground-state V_xc^b scalar,
    // matching ks_matrix.cpp's coefficient_beta exactly (asymmetric: the
    // "bb" self-term carries 2.0, the "ab" cross term does not, mirroring
    // coefficient_alpha's own weighting with aa<->bb swapped).
    double eval_vxc_beta_scalar(const DFT::XC::Functional &f, double rho_a, double rho_b, double gax, double gay,
                                 double gaz, double gbx, double gby, double gbz, const AOFactors &ao)
    {
        const double sigma_aa = dot(gax, gay, gaz, gax, gay, gaz);
        const double sigma_ab = dot(gax, gay, gaz, gbx, gby, gbz);
        const double sigma_bb = dot(gbx, gby, gbz, gbx, gby, gbz);
        std::vector<double> exc, vrho, vsigma;
        f.evaluate_gga_exc_vxc({rho_a, rho_b}, {sigma_aa, sigma_ab, sigma_bb}, 1, exc, vrho, vsigma);

        const double g_a_dot_AG = dot(gax, gay, gaz, ao.AGx, ao.AGy, ao.AGz);
        const double g_b_dot_AG = dot(gbx, gby, gbz, ao.AGx, ao.AGy, ao.AGz);
        // coefficient_beta = vsigma_ab*grad_rho_a + 2*vsigma_bb*grad_rho_b
        return vrho[1] * ao.AA + vsigma[1] * g_a_dot_AG + 2.0 * vsigma[2] * g_b_dot_AG;
    }

    double fd_delta_vxc_beta(const DFT::XC::Functional &f, const Point &p, const MixedPerturbation &d,
                              const AOFactors &ao, double h)
    {
        const double vp = eval_vxc_beta_scalar(f, p.rho_a + h * d.drho_a, p.rho_b + h * d.drho_b, p.gax + h * d.dgax,
                                                p.gay + h * d.dgay, p.gaz + h * d.dgaz, p.gbx + h * d.dgbx,
                                                p.gby + h * d.dgby, p.gbz + h * d.dgbz, ao);
        const double vm = eval_vxc_beta_scalar(f, p.rho_a - h * d.drho_a, p.rho_b - h * d.drho_b, p.gax - h * d.dgax,
                                                p.gay - h * d.dgay, p.gaz - h * d.dgaz, p.gbx - h * d.dgbx,
                                                p.gby - h * d.dgby, p.gbz - h * d.dgbz, ao);
        return (vp - vm) / (2.0 * h);
    }

    // General beta-channel check, taking a MixedPerturbation so both the
    // beta-only case (drho_a=dgax=dgay=dgaz=0) and the mixed case (both
    // nonzero) reuse the same formula and code path -- deliberately NOT
    // copy-pasted from check_alpha_only/check_mixed with names swapped;
    // every coefficient is re-read from its own (mirrored) slot below.
    void check_beta(const std::string &name, const Point &p, const MixedPerturbation &d, const AOFactors &ao,
                     const std::string &label)
    {
        auto f = require_functional(name);
        const Fxc fxc = eval_fxc_at(f, p);

        std::vector<double> exc0, vrho0, vsigma0;
        const double sigma_aa0 = dot(p.gax, p.gay, p.gaz, p.gax, p.gay, p.gaz);
        const double sigma_ab0 = dot(p.gax, p.gay, p.gaz, p.gbx, p.gby, p.gbz);
        const double sigma_bb0 = dot(p.gbx, p.gby, p.gbz, p.gbx, p.gby, p.gbz);
        f.evaluate_gga_exc_vxc({p.rho_a, p.rho_b}, {sigma_aa0, sigma_ab0, sigma_bb0}, 1, exc0, vrho0, vsigma0);
        const double vsigma_bb0 = vsigma0[2];
        const double vsigma_ab0 = vsigma0[1];

        const double g_a_dot_AG = dot(p.gax, p.gay, p.gaz, ao.AGx, ao.AGy, ao.AGz);
        const double g_b_dot_AG = dot(p.gbx, p.gby, p.gbz, ao.AGx, ao.AGy, ao.AGz);
        const double dg_a_dot_AG = dot(d.dgax, d.dgay, d.dgaz, ao.AGx, ao.AGy, ao.AGz);
        const double dg_b_dot_AG = dot(d.dgbx, d.dgby, d.dgbz, ao.AGx, ao.AGy, ao.AGz);

        const double dsigma_aa = 2.0 * dot(p.gax, p.gay, p.gaz, d.dgax, d.dgay, d.dgaz);
        const double dsigma_ab = dot(p.gbx, p.gby, p.gbz, d.dgax, d.dgay, d.dgaz) +
                                  dot(p.gax, p.gay, p.gaz, d.dgbx, d.dgby, d.dgbz);
        const double dsigma_bb = 2.0 * dot(p.gbx, p.gby, p.gbz, d.dgbx, d.dgby, d.dgbz);

        // Every "aa"-rooted slot in the alpha formulas becomes "bb" here;
        // the shared "ab" cross slots are read unchanged (there is only
        // one cross-spin sigma channel, not a mirrored pair).
        const double delta_vrho_b = fxc.v2rho2_bb * d.drho_b + fxc.v2rho2_ab * d.drho_a +
                                     fxc.v2rhosigma[5] * dsigma_bb + fxc.v2rhosigma[4] * dsigma_ab +
                                     fxc.v2rhosigma[3] * dsigma_aa;
        const double delta_vsigma_bb = fxc.v2rhosigma[5] * d.drho_b + fxc.v2rhosigma[2] * d.drho_a +
                                        fxc.v2sigma2[5] * dsigma_bb + fxc.v2sigma2[4] * dsigma_ab +
                                        fxc.v2sigma2[2] * dsigma_aa;
        const double delta_vsigma_ab = fxc.v2rhosigma[4] * d.drho_b + fxc.v2rhosigma[1] * d.drho_a +
                                        fxc.v2sigma2[4] * dsigma_bb + fxc.v2sigma2[3] * dsigma_ab +
                                        fxc.v2sigma2[1] * dsigma_aa;

        const double T1 = delta_vrho_b * ao.AA;
        const double T2 = 2.0 * delta_vsigma_bb * g_b_dot_AG;
        const double T3 = 2.0 * vsigma_bb0 * dg_b_dot_AG;
        const double T4 = delta_vsigma_ab * g_a_dot_AG;
        const double T5 = vsigma_ab0 * dg_a_dot_AG;
        const double delta_vxc_b_analytic = T1 + T2 + T3 + T4 + T5;

        for (double h : {1e-2, 1e-3, 1e-4})
        {
            const double delta_vxc_b_fd = fd_delta_vxc_beta(f, p, d, ao, h);
            const double tol = 50.0 * h * h + 1e-6;
            require_near(delta_vxc_b_analytic, delta_vxc_b_fd, tol,
                         name + " T1'+T2'+T3'+T4'+T5' (delta[V_xc^b], " + label + ") vs FD, h=" + std::to_string(h));
        }
    }
} // namespace

int main()
{
    // gga_c_pbe: PBE correlation alone, per F3.4.1's own finding that PBE
    // EXCHANGE has near-zero cross-spin coupling (which would leave T4
    // untested). Two open-shell-style points with genuinely non-uniform,
    // non-parallel alpha/beta gradients so no term vanishes by an
    // unlucky choice of direction.
    const AOFactors ao1{0.7, 0.4, -0.3, 0.2};
    const AOFactors ao2{-0.2, -0.1, 0.5, -0.4};

    const Point p1{0.30, 0.18, 0.1, 0.05, -0.02, 0.06, -0.04, 0.03};
    const Point p2{0.9, 0.4, 0.5, -0.3, 0.2, 0.15, 0.1, -0.05};

    const Perturbation d1{0.01, 0.02, -0.01, 0.005};
    const Perturbation d2{0.05, -0.1, 0.08, -0.04};

    check_alpha_only("gga_c_pbe", p1, d1, ao1);
    check_alpha_only("gga_c_pbe", p2, d2, ao1);
    check_alpha_only("gga_c_pbe", p1, d1, ao2);
    check_alpha_only("gga_c_pbe", p2, d2, ao2);

    // F3.4.3: mixed alpha/beta trial x, matching F3.2's own "an alpha-only
    // x cannot by itself catch a bug reading the cross-spin slot" lesson.
    // Directions chosen genuinely non-parallel/non-degenerate -- checked
    // directly (not assumed) that the cross-spin T4/T5 contributions are
    // non-negligible at these points before trusting the check (measured:
    // T5 alone is 67% of the total at md1/p1, not a symmetry-suppressed
    // direction the way F3.2's first (i=0,a=0) probe turned out to be).
    const MixedPerturbation md1{0.01, 0.008, 0.02, -0.01, 0.005, -0.015, 0.01, -0.006};
    const MixedPerturbation md2{0.05, -0.03, -0.1, 0.08, -0.04, 0.06, -0.05, 0.02};

    check_mixed("gga_c_pbe", p1, md1, ao1);
    check_mixed("gga_c_pbe", p2, md2, ao1);
    check_mixed("gga_c_pbe", p1, md1, ao2);
    check_mixed("gga_c_pbe", p2, md2, ao2);

    // F3.4.4: beta channel, checked independently rather than assumed
    // symmetric to alpha. Beta-only trials (drho_a=dgrad_rho_a=0, so T5'
    // vanishes identically -- exercises T1'-T4' without T5') and the same
    // mixed perturbations reused from F3.4.3 (both T4'/T5' nonzero there).
    const MixedPerturbation bo1{0.0, 0.008, 0.0, 0.0, 0.0, -0.015, 0.01, -0.006};
    const MixedPerturbation bo2{0.0, -0.03, 0.0, 0.0, 0.0, 0.06, -0.05, 0.02};

    check_beta("gga_c_pbe", p1, bo1, ao1, "beta-only x");
    check_beta("gga_c_pbe", p2, bo2, ao1, "beta-only x");
    check_beta("gga_c_pbe", p1, bo1, ao2, "beta-only x");
    check_beta("gga_c_pbe", p2, bo2, ao2, "beta-only x");

    check_beta("gga_c_pbe", p1, md1, ao1, "mixed x");
    check_beta("gga_c_pbe", p2, md2, ao1, "mixed x");
    check_beta("gga_c_pbe", p1, md1, ao2, "mixed x");
    check_beta("gga_c_pbe", p2, md2, ao2, "mixed x");

    // B88 exchange and LYP correlation -- the components of B3LYP. These
    // were previously unexercised at the point level (the whole suite only
    // ran gga_c_pbe), which left a gap flagged during the DFT SOSCF hybrid work (docs/SOSCF_DFT.md)
    // H4/H5: a whole-molecule FD probe on B3LYP UKS showed ~8e-4 scatter, and
    // the question was whether the polarized fxc formula is wrong for
    // B88/LYP or the whole-molecule FD is just noisy. It is the FD -- B88
    // and LYP match the grid-level FD of V_xc to ~1e-10, same as PBE.
    check_mixed("gga_x_b88", p1, md1, ao1);
    check_mixed("gga_x_b88", p2, md2, ao2);
    check_beta("gga_x_b88", p1, md1, ao1, "mixed x");
    check_beta("gga_x_b88", p2, md2, ao2, "mixed x");
    check_mixed("gga_c_lyp", p1, md1, ao1);
    check_mixed("gga_c_lyp", p2, md2, ao2);
    check_beta("gga_c_lyp", p1, md1, ao1, "mixed x");
    check_beta("gga_c_lyp", p2, md2, ao2, "mixed x");

    return g_ok ? 0 : 1;
}
