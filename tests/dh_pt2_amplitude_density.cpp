#include <cmath>
#include <cstdio>
#include <limits>
#include <source_location>
#include "dft/dh_pt2_gradient.h"

namespace
{
    bool near(double x, double y,
              std::source_location where = std::source_location::current())
    {
        const bool passed = std::abs(x - y) < 1e-13;
        if (!passed)
            std::fprintf(stderr, "%s:%u: actual %.17g, expected %.17g, delta %.3e\n",
                where.file_name(), where.line(), x, y, x - y);
        return passed;
    }
    struct LocatedCheck
    {
        bool passed;
        std::source_location where;
        LocatedCheck(bool value, std::source_location location = std::source_location::current())
            : passed(value), where(location) {}
    };
    struct Checks
    {
        bool passed;
        Checks(bool value) : passed(value) {}
        operator bool() const { return passed; }
        Checks &operator&=(LocatedCheck check)
        {
            if (!check.passed)
                std::fprintf(stderr, "%s:%u: DH invariant failed\n",
                    check.where.file_name(), check.where.line());
            passed &= check.passed;
            return *this;
        }
    };
    std::size_t idx(int i, int j, int a, int b)
    { return ((static_cast<std::size_t>(i) * 1 + j) * 2 + a) * 2 + b; }
}

int main()
{
    // Step 3: each Eq. (41) channel is supplied independently.  The maps
    // below are unrelated linear operators, so the total/prefactor check is
    // not a restatement of the implementation expression.
    Eigen::Matrix2d a; a << 1.0, 2.0, 2.0, -1.0;
    Eigen::Matrix2d b; b << 0.5, -1.0, 3.0, 2.0;
    Eigen::Matrix2d c; c << -2.0, 4.0, 1.0, 0.25;
    auto response = DFT::Gradient::make_dh_eq41_response_operator(
        0.6,
        [a](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return a * d * a.transpose(); },
        [b](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return b * d * b.transpose(); },
        [c](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return c * d * c.transpose(); });
    Eigen::Matrix2d x; x << 0.3, -0.2, -0.2, 0.7;
    Eigen::Matrix2d y; y << -0.4, 0.1, 0.1, 0.9;
    Checks ok = response.has_value();
    const auto no_exchange = DFT::Gradient::make_dh_eq41_response_operator(
        0.0,
        [a](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return a * d * a.transpose(); },
        {},
        [c](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return c * d * c.transpose(); });
    const auto missing_exchange = DFT::Gradient::make_dh_eq41_response_operator(
        0.6,
        [a](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return a * d * a.transpose(); },
        {},
        [c](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return c * d * c.transpose(); });
    ok &= no_exchange.has_value() && !missing_exchange.has_value();
    const auto zero_channel = [](const Eigen::Ref<const Eigen::MatrixXd> &d)
        -> std::expected<Eigen::MatrixXd, std::string> {
            return Eigen::MatrixXd::Zero(d.rows(), d.cols());
        };
    const auto j_only = DFT::Gradient::make_dh_eq41_response_operator(0.0,
        [a](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return a * d * a.transpose(); },
        {}, zero_channel);
    const auto k_only = DFT::Gradient::make_dh_eq41_response_operator(0.6,
        zero_channel,
        [b](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return b * d * b.transpose(); }, zero_channel);
    const auto xc_only = DFT::Gradient::make_dh_eq41_response_operator(0.0,
        zero_channel, {},
        [c](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return c * d * c.transpose(); });
    if (j_only && k_only && xc_only)
    {
        const auto jx = j_only->apply(x);
        const auto kx = k_only->apply(x);
        const auto xcx = xc_only->apply(x);
        ok &= jx && kx && xcx;
        if (jx && kx && xcx)
        {
            ok &= ((*jx - 4.0*a*x*a.transpose()).norm() < 1e-13);
            ok &= ((*kx + 1.2*b*x*b.transpose()).norm() < 1e-13);
            ok &= ((*xcx - 4.0*c*x*c.transpose()).norm() < 1e-13);
        }
    }
    else ok = false;
    if (response)
    {
        auto rx = response->apply(x);
        auto ry = response->apply(y);
        auto rxy = response->apply(x + y);
        const Eigen::Matrix2d expected = 4.0*a*x*a.transpose() - 1.2*b*x*b.transpose() + 4.0*c*x*c.transpose();
        ok &= rx && ry && rxy;
        if (rx && ry && rxy)
        {
            ok &= ((*rx - expected).norm() < 1e-13);
            ok &= ((*rxy - (*rx + *ry)).norm() < 1e-13);
        }
    }
    HartreeFock::Correlation::RMP2Result r;
    r.n_occ = 1; r.n_virt = 2;
    r.t2 = {1.0, 2.0, 2.0, 4.0}; // symmetric physical i=j amplitude block
    auto one = DFT::Gradient::build_dh_pt2_amplitude_density(r, 1.0);
    auto scaled = DFT::Gradient::build_dh_pt2_amplitude_density(r, 0.27);
    if (!one || !scaled) return 1;
    Eigen::Matrix3d mo_coeff = Eigen::Matrix3d::Identity();
    mo_coeff(0,0) = 0.8; mo_coeff(0,1) = -0.6;
    mo_coeff(1,0) = 0.6; mo_coeff(1,1) = 0.8;
    if (response)
    {
        const auto response3 = DFT::Gradient::make_dh_eq41_response_operator(
            0.6,
            [](const Eigen::Ref<const Eigen::MatrixXd> &d)
                -> std::expected<Eigen::MatrixXd, std::string> { return d; },
            [](const Eigen::Ref<const Eigen::MatrixXd> &d)
                -> std::expected<Eigen::MatrixXd, std::string> { return 2.0 * d; },
            [](const Eigen::Ref<const Eigen::MatrixXd> &d)
                -> std::expected<Eigen::MatrixXd, std::string> { return -0.5 * d; });
        const Eigen::Matrix3d expected_dprime = mo_coeff * one->dprime_mo * mo_coeff.transpose();
        if (!response3) ok = false;
        else
        {
            const auto eq41 = DFT::Gradient::build_dh_eq41_response_density(
                *one, mo_coeff, *response3);
            if (!eq41) ok = false;
            else
            {
                const auto direct = response3->apply(expected_dprime);
                ok &= direct.has_value();
                ok &= ((eq41->dprime_ao - expected_dprime).norm() < 1e-13);
                if (direct)
                    ok &= ((eq41->response_ao - *direct).norm() < 1e-13);
            }
        }
    }
    ok &= true;
    // Eq. (39): for i=j and symmetric t, t_tilde=t.
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            ok &= near(one->t_tilde[idx(0,0,a,b)], r.t2[idx(0,0,a,b)]);
    // Eqs. (37)-(38), evaluated by hand for t=[[1,2],[2,4]].
    ok &= near(one->dprime_oo(0,0), -50.0);
    ok &= near(one->dprime_vv(0,0), 10.0);
    ok &= near(one->dprime_vv(0,1), 20.0);
    ok &= near(one->dprime_vv(1,0), 20.0);
    ok &= near(one->dprime_vv(1,1), 40.0);
    ok &= near(one->dprime_mo.trace(), 0.0);
    ok &= (one->dprime_mo - one->dprime_mo.transpose()).norm() < 1e-13;
    ok &= (scaled->dprime_mo - 0.27 * one->dprime_mo).norm() < 1e-13;

    auto zero = DFT::Gradient::build_dh_pt2_amplitude_density(r, 0.0);
    ok &= zero && zero->dprime_mo.norm() == 0.0 && zero->t_tilde[0] == 0.0;

    // Eq. (40): a sparse, hand-evaluated 1occ/2virt fixture.  The ERI slots
    // deliberately do not obey permutation symmetry: this makes each printed
    // exchange ordering observable instead of allowing swapped labels to hide.
    HartreeFock::Correlation::RMP2Result r40;
    r40.n_occ = 1; r40.n_virt = 2;
    r40.t2 = {1.0, 2.0, 3.0, 4.0};
    constexpr int nmo40 = 3;
    auto eidx = [](int p, int q, int rr, int s)
    { return ((p * nmo40 + q) * nmo40 + rr) * nmo40 + s; };
    std::vector<double> eri40(nmo40 * nmo40 * nmo40 * nmo40, 0.0);
    eri40[eidx(1,1,0,2)] = 13.0; eri40[eidx(1,2,0,1)] = 5.0;
    eri40[eidx(2,1,0,2)] = 17.0; eri40[eidx(2,2,0,1)] = 7.0;
    eri40[eidx(0,0,0,2)] = 19.0; eri40[eidx(0,2,0,0)] = 3.0;
    auto d40 = DFT::Gradient::build_dh_pt2_amplitude_density(r40, 0.5);
    auto l40 = d40 ? DFT::Gradient::build_dh_eq40_amplitude_rhs(r40, *d40, eri40, 0.5, true)
                   : std::expected<DFT::Gradient::DHEq40AmplitudeRHS, std::string>(
                         std::unexpected("Eq. 40 amplitude fixture setup failed"));
    ok &= l40.has_value();
    if (l40)
    {
        // The literal four-coefficient derivative gives external blocks
        // (2,-43), while the separately visible internal spin channels
        // The literal same-spin/exchange and opposite-spin/direct channels
        // are (-19,+3) and (-38,+6); their raw-t sums are (-16,-32).
        ok &= near(l40->three_external(0,0), 2.0);
        ok &= near(l40->three_external(1,0), -43.0);
        ok &= near(l40->internal_same_spin_exchange(0,0), -19.0);
        ok &= near(l40->internal_opposite_spin_direct(0,0), 3.0);
        ok &= near(l40->three_internal(0,0), -16.0);
        ok &= near(l40->internal_same_spin_exchange(1,0), -38.0);
        ok &= near(l40->internal_opposite_spin_direct(1,0), 6.0);
        ok &= near(l40->three_internal(1,0), -32.0);
        ok &= (l40->three_internal - l40->internal_same_spin_exchange -
               l40->internal_opposite_spin_direct).norm() < 1e-13;
        ok &= near(l40->total(0,0), -14.0);
        ok &= near(l40->total(1,0), -75.0);
    }
    auto d40_one = DFT::Gradient::build_dh_pt2_amplitude_density(r40, 1.0);
    auto l40_one = d40_one ? DFT::Gradient::build_dh_eq40_amplitude_rhs(r40, *d40_one, eri40, 1.0, true)
                            : std::expected<DFT::Gradient::DHEq40AmplitudeRHS, std::string>(
                                  std::unexpected("Eq. 40 unit-scale fixture setup failed"));
    ok &= l40 && l40_one;
    if (l40 && l40_one)
        ok &= (l40->total - 0.5 * l40_one->total).norm() < 1e-13;
    if (d40 && l40)
    {
        const auto production = DFT::Gradient::build_dh_eq40_amplitude_rhs(r40, *d40, eri40, 0.5);
        ok &= production.has_value();
        if (production)
        {
            ok &= (production->three_external - l40->three_external).norm() < 1e-13;
            ok &= production->three_internal.norm() == 0.0;
            ok &= production->internal_same_spin_exchange.norm() == 0.0;
            ok &= production->internal_opposite_spin_direct.norm() == 0.0;
            ok &= (production->total - l40->three_external).norm() < 1e-13;
        }
    }

    // Eq. (40) full RHS: independently contract the AO Eq. (41) response
    // into virtual-major MO space, then add the named amplitude blocks.
    if (l40)
    {
        Eigen::Matrix3d rao;
        rao << 2.0, -1.0, 0.5,
                -1.0, 3.0, 1.0,
                0.5, 1.0, -2.0;
        Eigen::Matrix<double, 3, 1> co;
        co << 0.7, -0.2, 0.5;
        Eigen::Matrix<double, 3, 2> cv;
        cv << 0.6, -0.1,
              0.3,  0.8,
             -0.4,  0.2;
        DFT::Gradient::DHEq41ResponseDensity eq41_fixture;
        eq41_fixture.response_ao = rao;
        const auto rhs = DFT::Gradient::build_dh_lagrangian_rhs(
            eq41_fixture, co, cv, *l40, DFT::Gradient::DHRHSConvention::LegacyExternalPlusInternal);
        ok &= rhs.has_value();
        if (rhs)
        {
            Eigen::Matrix<double, 2, 1> response_ref = Eigen::Matrix<double, 2, 1>::Zero();
            for (int av = 0; av < 2; ++av)
                for (int io = 0; io < 1; ++io)
                    for (int mu = 0; mu < 3; ++mu)
                        for (int nu = 0; nu < 3; ++nu)
                            response_ref(av, io) += cv(mu, av) * rao(mu, nu) * co(nu, io);
            ok &= (rhs->response_ai - response_ref).norm() < 1e-13;
            ok &= (rhs->three_external_ai - l40->three_external).norm() < 1e-13;
            ok &= (rhs->internal_same_spin_exchange_ai - l40->internal_same_spin_exchange).norm() < 1e-13;
            ok &= (rhs->internal_opposite_spin_direct_ai - l40->internal_opposite_spin_direct).norm() < 1e-13;
            ok &= (rhs->three_internal_ai - l40->three_internal).norm() < 1e-13;
            ok &= (rhs->amplitude_ai - l40->total).norm() < 1e-13;
            ok &= (rhs->total_ai - response_ref - l40->total).norm() < 1e-13;
            const auto literal = DFT::Gradient::build_dh_lagrangian_rhs(
                eq41_fixture, co, cv, *l40); // Validated production default.
            ok &= literal.has_value();
            if (literal)
            {
                // The hand-evaluated pair derivative is (2,-43); the raw
                // (-16,-32) internal bracket is retained, but not added twice.
                ok &= near(literal->amplitude_ai(0,0), 2.0);
                ok &= near(literal->amplitude_ai(1,0), -43.0);
                ok &= literal->included_internal_ai.norm() == 0.0;
                ok &= (literal->three_internal_ai - l40->three_internal).norm() < 1e-13;
                ok &= (literal->total_ai - response_ref - l40->three_external).norm() < 1e-13;
                ok &= (rhs->total_ai - literal->total_ai - l40->three_internal).norm() < 1e-13;
            }
        }
    }

    // Eq. (28): Z_ai belongs only to the raw virtual-occupied block.  Its
    // symmetric adapter splits that contribution evenly for symmetric traces.
    Eigen::Matrix<double, 2, 1> z_ai;
    z_ai << 1.25, -0.75;
    const auto relaxed = DFT::Gradient::build_dh_relaxed_difference_density(*one, z_ai);
    ok &= relaxed.has_value();
    if (relaxed)
    {
        ok &= (relaxed->z_ai - z_ai).norm() < 1e-13;
        ok &= near(relaxed->raw_mo(1,0), 1.25);
        ok &= near(relaxed->raw_mo(2,0), -0.75);
        ok &= near(relaxed->raw_mo(0,1), 0.0);
        ok &= near(relaxed->raw_mo(0,2), 0.0);
        ok &= near(relaxed->symmetric_mo(1,0), 0.625);
        ok &= near(relaxed->symmetric_mo(2,0), -0.375);
        ok &= near(relaxed->symmetric_mo(0,1), 0.625);
        ok &= near(relaxed->symmetric_mo(0,2), -0.375);
        ok &= (relaxed->symmetric_mo - relaxed->symmetric_mo.transpose()).norm() < 1e-13;
        ok &= near(relaxed->symmetric_mo.trace(), one->dprime_mo.trace());
        Eigen::Matrix3d symmetric_operator;
        symmetric_operator << 2.0, -1.0, 0.3,
                              -1.0, 1.5, -0.8,
                               0.3, -0.8, -0.4;
        const double raw_contract = (relaxed->raw_mo.array() * symmetric_operator.array()).sum();
        const double symmetric_contract =
            (relaxed->symmetric_mo.array() * symmetric_operator.array()).sum();
        ok &= near(raw_contract, symmetric_contract);
    }

    // Eq. (27): raw Z_ai response, with orbital-energy, J, K, and XC pieces
    // checked separately against independent AO/MO nested-loop contractions.
    Eigen::Matrix<double, 3, 1> co27;
    co27 << 0.5, -0.3, 0.7;
    Eigen::Matrix<double, 3, 2> cv27;
    cv27 << 0.2, -0.6,
            0.8,  0.1,
           -0.4,  0.5;
    Eigen::Matrix<double, 2, 1> z27;
    z27 << 0.4, -0.9;
    Eigen::Vector3d eps27;
    eps27 << -0.8, 0.2, 1.1;
    Eigen::Matrix3d ja, kb, xc_map;
    ja << 1.0, 0.2, -0.3, 0.2, 0.7, 0.4, -0.3, 0.4, 1.2;
    kb << 0.5, -0.1, 0.6, -0.1, 1.1, 0.3, 0.6, 0.3, -0.4;
    xc_map << -0.2, 0.7, 0.1, 0.7, 0.4, -0.5, 0.1, -0.5, 0.9;
    const auto response27 = DFT::Gradient::make_dh_eq41_response_operator(
        0.6,
        [ja](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return ja*d*ja.transpose(); },
        [kb](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return kb*d*kb.transpose(); },
        [xc_map](const Eigen::Ref<const Eigen::MatrixXd> &d)
            -> std::expected<Eigen::MatrixXd, std::string> { return xc_map*d*xc_map.transpose(); });
    const auto h27 = response27 ? DFT::Gradient::apply_dh_eq27_hessian(
        z27, co27, cv27, eps27, *response27)
        : std::expected<DFT::Gradient::DHEq27HessianAction, std::string>(
            std::unexpected("Eq. 27 response fixture setup failed"));
    ok &= h27.has_value();
    if (h27)
    {
        Eigen::Matrix3d raw_z_ref = Eigen::Matrix3d::Zero();
        for (int mu = 0; mu < 3; ++mu)
            for (int nu = 0; nu < 3; ++nu)
                for (int av = 0; av < 2; ++av)
                    raw_z_ref(mu, nu) += cv27(mu,av) * z27(av,0) * co27(nu,0);
        const auto project_ai = [&co27, &cv27](const Eigen::Matrix3d &ao)
        {
            Eigen::Matrix<double, 2, 1> result = Eigen::Matrix<double, 2, 1>::Zero();
            for (int av = 0; av < 2; ++av)
                for (int mu = 0; mu < 3; ++mu)
                    for (int nu = 0; nu < 3; ++nu)
                        result(av,0) += cv27(mu,av) * ao(mu,nu) * co27(nu,0);
            return result;
        };
        const Eigen::Matrix3d delta_p27 = 2.0 * (raw_z_ref + raw_z_ref.transpose());
        const Eigen::Matrix3d j27 = ja*delta_p27*ja.transpose();
        const Eigen::Matrix3d k27 = kb*delta_p27*kb.transpose();
        const Eigen::Matrix3d x27 = xc_map*delta_p27*xc_map.transpose();
        const Eigen::Matrix<double, 2, 1> coulomb_ref = project_ai(j27);
        const Eigen::Matrix<double, 2, 1> exchange_ref = project_ai(-0.5*0.6*k27);
        const Eigen::Matrix<double, 2, 1> xc_ref = project_ai(x27);
        Eigen::Matrix<double, 2, 1> orbital_ref;
        orbital_ref(0,0) = (eps27(1)-eps27(0))*z27(0,0);
        orbital_ref(1,0) = (eps27(2)-eps27(0))*z27(1,0);
        ok &= (h27->raw_z_ao - raw_z_ref).norm() < 1e-13;
        ok &= (h27->orbital_energy_ai - orbital_ref).norm() < 1e-13;
        ok &= (h27->coulomb_ai - coulomb_ref).norm() < 1e-13;
        ok &= (h27->exchange_ai - exchange_ref).norm() < 1e-13;
        ok &= (h27->xc_ai - xc_ref).norm() < 1e-13;
        ok &= (h27->response_ai - coulomb_ref - exchange_ref - xc_ref).norm() < 1e-13;
        ok &= (h27->total_ai - orbital_ref - coulomb_ref - exchange_ref - xc_ref).norm() < 1e-13;
    }

    // Eqs. (42)-(43): hand-evaluated diagonal W blocks.  Identity MO
    // coefficients make the Eq. (42) response-density transformation visible
    // while retaining an independently specified linear XC response.
    if (d40)
    {
        Eigen::Matrix<double, 2, 1> z_zero = Eigen::Matrix<double, 2, 1>::Zero();
        const auto relaxed40 = DFT::Gradient::build_dh_relaxed_difference_density(*d40, z_zero);
        eri40[eidx(0,1,0,1)] = 11.0; eri40[eidx(0,1,0,2)] = 13.0;
        eri40[eidx(0,2,0,1)] = 17.0; eri40[eidx(0,2,0,2)] = 19.0;
        const auto zero_w = [](const Eigen::Ref<const Eigen::MatrixXd> &q)
            -> std::expected<Eigen::MatrixXd, std::string>
            { return Eigen::MatrixXd::Zero(q.rows(), q.cols()); };
        const auto response_w = DFT::Gradient::make_dh_eq41_response_operator(
            0.0, zero_w, {},
            [](const Eigen::Ref<const Eigen::MatrixXd> &q)
                -> std::expected<Eigen::MatrixXd, std::string> { return 0.25*q; });
        Eigen::Vector3d eps_w;
        eps_w << -1.0, 0.3, 0.9;
        const auto w = (relaxed40 && response_w)
            ? DFT::Gradient::build_dh_eq42_43_energy_weighted_density(
                *relaxed40, r40, *d40, Eigen::Matrix3d::Identity(), eps_w, eri40, *response_w)
            : std::expected<DFT::Gradient::DHEq42_43EnergyWeightedDensity, std::string>(
                std::unexpected("Eq. 42-43 fixture setup failed"));
        ok &= w.has_value();
        if (w)
        {
            ok &= near(w->response_oo(0,0), 14.0);
            ok &= near(w->orbital_oo(0,0), -28.0);
            ok &= near(w->amplitude_oo(0,0), -42.0);
            ok &= near(w->w_oo(0,0), -56.0);
            ok &= near(w->orbital_vv(0,0), -2.4);
            ok &= near(w->orbital_vv(0,1), -7.5);
            ok &= near(w->orbital_vv(1,0), -7.5);
            ok &= near(w->orbital_vv(1,1), -20.7);
            ok &= near(w->amplitude_vv(0,0), -6.0);
            ok &= near(w->amplitude_vv(0,1), -24.0);
            ok &= near(w->amplitude_vv(1,0), -9.0);
            ok &= near(w->amplitude_vv(1,1), -36.0);
            ok &= near(w->w_vv(0,0), -8.4);
            ok &= near(w->w_vv(0,1), -31.5);
            ok &= near(w->w_vv(1,0), -16.5);
            ok &= near(w->w_vv(1,1), -56.7);
        }

        // Eq. (42): the XC callback H(X)=A X A (A=A^T) enters the
        // closed-shell Eq. (41) operator as R(X)=4 H(X). Consequently
        // Q(X)=2 <X,H(X)> has derivative R(X), and W_response=-R/2.
        // Differentiate Q independently of the response/W builders.
        Eigen::Matrix3d response_kernel;
        response_kernel << 1.1, -0.2, 0.4,
                          -0.2, 0.8, 0.3,
                           0.4, 0.3, -0.5;
        const auto self_adjoint_response = DFT::Gradient::make_dh_eq41_response_operator(
            0.0, zero_w, {},
            [response_kernel](const Eigen::Ref<const Eigen::MatrixXd> &q)
                -> std::expected<Eigen::MatrixXd, std::string>
                { return response_kernel * q * response_kernel; });
        const auto w_self_adjoint = (relaxed40 && self_adjoint_response)
            ? DFT::Gradient::build_dh_eq42_43_energy_weighted_density(
                *relaxed40, r40, *d40, Eigen::Matrix3d::Identity(), eps_w, eri40,
                *self_adjoint_response)
            : std::expected<DFT::Gradient::DHEq42_43EnergyWeightedDensity, std::string>(
                std::unexpected("Eq. 42 self-adjoint fixture setup failed"));
        ok &= w_self_adjoint.has_value();
        if (w_self_adjoint)
        {
            const Eigen::Matrix3d d_raw = relaxed40->symmetric_mo;
            Eigen::Matrix3d probe = Eigen::Matrix3d::Zero();
            probe(0,0) = 1.0;
            Eigen::Matrix3d adjoint_probe;
            adjoint_probe << 0.2, -0.6, 0.1,
                             -0.4, 0.5, 0.7,
                              0.3, 0.8, -0.9;
            const auto response_map = [&response_kernel](const Eigen::Matrix3d &q)
                { return response_kernel * q * response_kernel; };
            const double self_adjoint_lhs = (adjoint_probe.array() * response_map(d_raw).array()).sum();
            const double self_adjoint_rhs = (response_map(adjoint_probe).array() * d_raw.array()).sum();
            ok &= std::abs(self_adjoint_lhs - self_adjoint_rhs) < 1e-13;
            // This fixture gives H(D)_00=-31.88 and W_response_00=63.76.
            ok &= near(w_self_adjoint->response_oo(0,0), 63.76);
            const auto quadratic_energy = [&response_kernel](const Eigen::Matrix3d &q)
            {
                // Use extended precision where the platform provides it;
                // the FD assertion also accounts for scalar roundoff below.
                const Eigen::Matrix<long double, 3, 3> q_long = q.cast<long double>();
                const Eigen::Matrix<long double, 3, 3> a_long = response_kernel.cast<long double>();
                const Eigen::Matrix<long double, 3, 3> h_long = a_long * q_long * a_long;
                return 2.0L * (q_long.array() * h_long.array()).sum();
            };
            for (const double step : {1e-2, 1e-3, 1e-4})
            {
                const long double plus_energy = quadratic_energy(d_raw + step*probe);
                const long double minus_energy = quadratic_energy(d_raw - step*probe);
                const long double finite_difference =
                    (plus_energy - minus_energy) / (2.0L*step);
                const long double roundoff = 8.0L * std::numeric_limits<long double>::epsilon() *
                    (std::abs(plus_energy) + std::abs(minus_energy)) / (2.0L*step);
                ok &= std::abs(w_self_adjoint->response_oo(0,0) + 0.5L*finite_difference) < 1e-9L + roundoff;
            }
        }

        // Eqs. (44)-(45): W_ia and W_ai use opposite raw orientations.
        // A private ERI copy isolates this fixture from the Eq. (42)-(43)
        // exchange matrices above.
        std::vector<double> eri44 = eri40;
        eri44[eidx(0,0,0,1)] = 7.0;
        eri44[eidx(0,0,0,2)] = 11.0;
        const auto relaxed44 = DFT::Gradient::build_dh_relaxed_difference_density(*d40, z_ai);
        const auto w44_45 = relaxed44
            ? DFT::Gradient::build_dh_eq44_45_energy_weighted_density(
                *relaxed44, r40, *d40, eps_w, eri44)
            : std::expected<DFT::Gradient::DHEq44_45EnergyWeightedDensity, std::string>(
                std::unexpected("Eq. 44-45 fixture setup failed"));
        ok &= w44_45.has_value();
        if (w44_45)
        {
            ok &= w44_45->w_ia.rows() == 1 && w44_45->w_ia.cols() == 2;
            ok &= w44_45->w_ai.rows() == 2 && w44_45->w_ai.cols() == 1;
            ok &= near(w44_45->w_ia(0,0), -9.0);
            ok &= near(w44_45->w_ia(0,1), -36.0);
            ok &= near(w44_45->w_ai(0,0), 1.25);
            ok &= near(w44_45->w_ai(1,0), -0.75);
        }

        // Literal Eq. (47) metric-pair W blocks: compare their contraction
        // with an independent four-coefficient derivative of
        // sum_ijab (1+delta_ij)t~_ij^ab(ia|jb), in the paper gauge
        // U_ia=-S_ia and U_ai=0.  This is deliberately not a restatement of
        // the builder's oo/vv/ov block loops.
        std::vector<double> eri47_metric(nmo40 * nmo40 * nmo40 * nmo40, 0.0);
        for (int p = 0; p < nmo40; ++p)
            for (int q = 0; q < nmo40; ++q)
                for (int r = 0; r < nmo40; ++r)
                    for (int s = 0; s < nmo40; ++s)
                        eri47_metric[eidx(p,q,r,s)] = 0.01 * (1 + eidx(p,q,r,s));
        const auto pair_metric = d40
            ? DFT::Gradient::build_dh_eq47_pair_metric_overlap_density(r40, *d40, eri47_metric)
            : std::expected<DFT::Gradient::DHEq47PairMetricOverlapDensity, std::string>(
                std::unexpected("Eq. 47 metric-pair fixture setup failed"));
        ok &= pair_metric.has_value();
        if (pair_metric)
        {
            ok &= pair_metric->w_oo.rows() == 1 && pair_metric->w_oo.cols() == 1;
            ok &= pair_metric->w_vv.rows() == 2 && pair_metric->w_vv.cols() == 2;
            ok &= pair_metric->w_ia.rows() == 1 && pair_metric->w_ia.cols() == 2;
            ok &= pair_metric->w_ai.cwiseAbs().maxCoeff() < 1e-14;
            Eigen::Matrix3d metric_probe;
            metric_probe << 0.4, -0.3, 0.2,
                            -0.3, 0.7, -0.5,
                             0.2, -0.5, 0.9;
            const double block_contraction =
                (pair_metric->w_oo.array() * metric_probe.topLeftCorner(1,1).array()).sum() +
                (pair_metric->w_vv.array() * metric_probe.bottomRightCorner(2,2).array()).sum() +
                (pair_metric->w_ia.array() * metric_probe.topRightCorner(1,2).array()).sum();
            Eigen::Matrix3d u = -0.5 * metric_probe;
            u.topRightCorner(1,2) = -metric_probe.topRightCorner(1,2);
            u.bottomLeftCorner(2,1).setZero();
            double direct_contraction = 0.0;
            for (int i = 0; i < 1; ++i)
                for (int j = 0; j < 1; ++j)
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b)
                        {
                            const double theta = static_cast<double>(1 + (i == j)) * d40->t_tilde[idx(i,j,a,b)];
                            const int ap = 1 + a, bp = 1 + b;
                            for (int p = 0; p < 3; ++p)
                                direct_contraction += theta * (
                                    u(p,i)  * eri47_metric[eidx(p,ap,j,bp)] +
                                    u(p,ap) * eri47_metric[eidx(i,p,j,bp)] +
                                    u(p,j)  * eri47_metric[eidx(i,ap,p,bp)] +
                                    u(p,bp) * eri47_metric[eidx(i,ap,j,p)]);
                        }
            ok &= std::abs(block_contraction - direct_contraction) < 1e-12;
        }
        if (w && w44_45)
        {
            const auto overlap = DFT::Gradient::build_dh_overlap_density_adapter(*w, *w44_45);
            ok &= overlap.has_value();
            if (overlap)
            {
                ok &= (overlap->raw_mo.topLeftCorner(1,1) - w->w_oo).norm() < 1e-13;
                ok &= (overlap->raw_mo.bottomRightCorner(2,2) - w->w_vv).norm() < 1e-13;
                ok &= (overlap->raw_mo.topRightCorner(1,2) - w44_45->w_ia).norm() < 1e-13;
                ok &= (overlap->raw_mo.bottomLeftCorner(2,1) - w44_45->w_ai).norm() < 1e-13;
                ok &= (overlap->symmetric_mo - overlap->symmetric_mo.transpose()).norm() < 1e-13;
                ok &= near(overlap->symmetric_mo(0,1),
                           0.5 * (w44_45->w_ia(0,0) + w44_45->w_ai(0,0)));
                ok &= near(overlap->symmetric_mo(0,2),
                           0.5 * (w44_45->w_ia(0,1) + w44_45->w_ai(1,0)));
                Eigen::Matrix3d overlap_derivative;
                overlap_derivative << 0.4, -1.2, 0.7,
                                      -1.2, 2.3, -0.6,
                                       0.7, -0.6, 1.1;
                const double raw_overlap =
                    (overlap->raw_mo.array() * overlap_derivative.array()).sum();
                const double symmetric_overlap =
                    (overlap->symmetric_mo.array() * overlap_derivative.array()).sum();
                ok &= near(raw_overlap, symmetric_overlap);
            }
        }

        // Eqs. (46)-(47): retain the correction-only separable D P - a_x/2 D P
        // tensor and the direct t-tilde backtransformation as separate AO
        // objects. Identity coefficients make their paper-index elements
        // directly inspectable, while the final check exercises the distinct
        // eightfold ERI-symmetry adapter.
        Eigen::Matrix3d ground_density;
        ground_density << 1.7, -0.2, 0.4,
                         -0.2, 0.9, 0.3,
                          0.4, 0.3, 0.6;
        const auto gamma46_47 = relaxed44
            ? DFT::Gradient::build_dh_eq46_47_two_particle_density(
                *relaxed44, *d40, Eigen::Matrix3d::Identity(), ground_density, 1.0)
            : std::expected<DFT::Gradient::DHEq46_47TwoParticleDensity, std::string>(
                std::unexpected("Eq. 46-47 fixture setup failed"));
        ok &= gamma46_47.has_value();
        if (gamma46_47)
        {
            // Required a_x distinguishes the KS separable complement from
            // the PT2 pair term. Check every raw element, including the zero
            // exchange limit; scaling Eq. (47) here would double-scale PT2.
            for (double ax : {0.0, 0.53, 1.0})
            {
                const auto hybrid = DFT::Gradient::build_dh_eq46_47_two_particle_density(
                    *relaxed44, *d40, Eigen::Matrix3d::Identity(), ground_density, ax);
                ok &= hybrid.has_value();
                if (!hybrid) continue;
                ok &= hybrid->nonseparable_raw_ao == gamma46_47->nonseparable_raw_ao;
                ok &= hybrid->nonseparable_symmetric_ao == gamma46_47->nonseparable_symmetric_ao;
                for (int mu = 0; mu < 3; ++mu)
                    for (int nu = 0; nu < 3; ++nu)
                        for (int ka = 0; ka < 3; ++ka)
                            for (int ta = 0; ta < 3; ++ta)
                                ok &= near(hybrid->separable_raw_ao[eidx(mu,nu,ka,ta)],
                                    relaxed44->symmetric_mo(mu,nu) * ground_density(ka,ta) -
                                    0.5 * ax * relaxed44->symmetric_mo(mu,ka) * ground_density(nu,ta));
            }
            for (double invalid_ax : {-0.1, std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::infinity()})
                ok &= !DFT::Gradient::build_dh_eq46_47_two_particle_density(
                    *relaxed44, *d40, Eigen::Matrix3d::Identity(), ground_density, invalid_ax);
            ok &= gamma46_47->n_ao == 3;
            ok &= (gamma46_47->relaxed_difference_ao - relaxed44->symmetric_mo).norm() < 1e-13;
            for (int mu = 0; mu < 3; ++mu)
                for (int nu = 0; nu < 3; ++nu)
                    for (int ka = 0; ka < 3; ++ka)
                        for (int ta = 0; ta < 3; ++ta)
                        {
                            const double separable_ref =
                                relaxed44->symmetric_mo(mu,nu) * ground_density(ka,ta) -
                                0.5 * relaxed44->symmetric_mo(mu,ka) * ground_density(nu,ta);
                            ok &= near(gamma46_47->separable_raw_ao[eidx(mu,nu,ka,ta)], separable_ref);
                            ok &= near(gamma46_47->total_raw_ao[eidx(mu,nu,ka,ta)],
                                gamma46_47->separable_raw_ao[eidx(mu,nu,ka,ta)] +
                                gamma46_47->nonseparable_raw_ao[eidx(mu,nu,ka,ta)]);
                        }
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    ok &= near(gamma46_47->nonseparable_raw_ao[eidx(0,1+a,0,1+b)],
                        2.0 * d40->t_tilde[static_cast<std::size_t>(a)*2 + b]);

            std::vector<double> derivative_probe(81, 0.0), symmetric_probe(81, 0.0);
            for (std::size_t p = 0; p < derivative_probe.size(); ++p)
                derivative_probe[p] = 0.1 * static_cast<double>(p + 1);
            for (int mu = 0; mu < 3; ++mu)
                for (int nu = 0; nu < 3; ++nu)
                    for (int ka = 0; ka < 3; ++ka)
                        for (int ta = 0; ta < 3; ++ta)
                            symmetric_probe[eidx(mu,nu,ka,ta)] = 0.125 * (
                                derivative_probe[eidx(mu,nu,ka,ta)] + derivative_probe[eidx(nu,mu,ka,ta)] +
                                derivative_probe[eidx(mu,nu,ta,ka)] + derivative_probe[eidx(nu,mu,ta,ka)] +
                                derivative_probe[eidx(ka,ta,mu,nu)] + derivative_probe[eidx(ta,ka,mu,nu)] +
                                derivative_probe[eidx(ka,ta,nu,mu)] + derivative_probe[eidx(ta,ka,nu,mu)]);
            double raw_contraction = 0.0, symmetric_contraction = 0.0;
            for (std::size_t p = 0; p < derivative_probe.size(); ++p)
            {
                raw_contraction += gamma46_47->total_raw_ao[p] * symmetric_probe[p];
                symmetric_contraction += gamma46_47->total_symmetric_ao[p] * derivative_probe[p];
                ok &= near(gamma46_47->total_symmetric_ao[p],
                    gamma46_47->separable_symmetric_ao[p] + gamma46_47->nonseparable_symmetric_ao[p]);
            }
            ok &= std::abs(raw_contraction - symmetric_contraction) < 1e-10;

            // Eq. (33), excluding its explicit XC nuclear term: combine only
            // <D h^(x)>, <W S^(x)>, and Gamma_PT2:(ERI)^(x). Two opposite
            // coordinate derivatives model a rigid translation, so every
            // named channel must cancel separately rather than only in total.
            const auto overlap33 = (w && w44_45)
                ? DFT::Gradient::build_dh_overlap_density_adapter(*w, *w44_45)
                : std::expected<DFT::Gradient::DHOverlapDensity, std::string>(
                    std::unexpected("Eq. 33 overlap fixture setup failed"));
            DFT::Gradient::DHEq33NonXCDerivatives derivatives33;
            Eigen::Matrix3d h33, s33;
            h33 << 0.2, -0.1, 0.4,
                  -0.1, 0.5, -0.3,
                   0.4, -0.3, 0.7;
            s33 << -0.6, 0.2, 0.1,
                   0.2, 0.3, -0.4,
                   0.1, -0.4, 0.8;
            derivatives33.hamiltonian_ao = {h33, -h33};
            derivatives33.overlap_ao = {s33, -s33};
            derivatives33.eri_ao.resize(2, std::vector<double>(81, 0.0));
            for (std::size_t p = 0; p < 81; ++p)
            {
                derivatives33.eri_ao[0][p] = 0.01;
                derivatives33.eri_ao[1][p] = -0.01;
            }
            const auto eq33 = overlap33
                ? DFT::Gradient::build_dh_eq33_non_xc_gradient(
                    *relaxed44, *overlap33, *gamma46_47, Eigen::Matrix3d::Identity(), derivatives33)
                : std::expected<DFT::Gradient::DHEq33NonXCGradient, std::string>(
                    std::unexpected("Eq. 33 fixture setup failed"));
            ok &= eq33.has_value();
            if (eq33)
            {
                const Eigen::Matrix3d d33 = relaxed44->symmetric_mo;
                const Eigen::Matrix3d w33 = overlap33->symmetric_mo;
                const double h_ref = (d33.array() * h33.array()).sum();
                const double s_ref = (w33.array() * s33.array()).sum();
                double sep_ref = 0.0, ns_ref = 0.0;
                for (std::size_t p = 0; p < 81; ++p)
                {
                    sep_ref += 0.01 * gamma46_47->separable_symmetric_ao[p];
                    ns_ref += 0.01 * gamma46_47->nonseparable_symmetric_ao[p];
                }
                ok &= near(eq33->one_electron(0), h_ref);
                ok &= near(eq33->overlap(0), s_ref);
                ok &= near(eq33->two_electron_separable(0), sep_ref);
                ok &= near(eq33->two_electron_nonseparable(0), ns_ref);
                ok &= near(eq33->two_electron(0), sep_ref + ns_ref);
                ok &= near(eq33->total(0), h_ref + s_ref + sep_ref + ns_ref);
                ok &= eq33->one_electron.sum() == 0.0;
                ok &= eq33->overlap.sum() == 0.0;
                ok &= eq33->two_electron_separable.sum() == 0.0;
                ok &= eq33->two_electron_nonseparable.sum() == 0.0;
                ok &= eq33->total.sum() == 0.0;
            }

            const auto zero_amplitudes33 = DFT::Gradient::build_dh_pt2_amplitude_density(r40, 0.0);
            const auto zero_relaxed33 = zero_amplitudes33
                ? DFT::Gradient::build_dh_relaxed_difference_density(*zero_amplitudes33, z_zero)
                : std::expected<DFT::Gradient::DHRelaxedDifferenceDensity, std::string>(
                    std::unexpected("Eq. 33 zero-scale relaxed-density setup failed"));
            const auto zero_gamma33 = (zero_amplitudes33 && zero_relaxed33)
                ? DFT::Gradient::build_dh_eq46_47_two_particle_density(
                    *zero_relaxed33, *zero_amplitudes33, Eigen::Matrix3d::Identity(), ground_density, 1.0)
                : std::expected<DFT::Gradient::DHEq46_47TwoParticleDensity, std::string>(
                    std::unexpected("Eq. 33 zero-scale two-particle setup failed"));
            DFT::Gradient::DHOverlapDensity zero_overlap33;
            zero_overlap33.raw_mo = Eigen::Matrix3d::Zero();
            zero_overlap33.symmetric_mo = Eigen::Matrix3d::Zero();
            const auto eq33_zero = zero_gamma33
                ? DFT::Gradient::build_dh_eq33_non_xc_gradient(
                    *zero_relaxed33, zero_overlap33, *zero_gamma33,
                    Eigen::Matrix3d::Identity(), derivatives33)
                : std::expected<DFT::Gradient::DHEq33NonXCGradient, std::string>(
                    std::unexpected("Eq. 33 zero-scale fixture setup failed"));
            ok &= eq33_zero.has_value();
            if (eq33_zero)
                ok &= eq33_zero->one_electron.norm() == 0.0 &&
                      eq33_zero->overlap.norm() == 0.0 &&
                      eq33_zero->two_electron.norm() == 0.0 &&
                      eq33_zero->total.norm() == 0.0;
        }
    }
    if (!ok) std::fprintf(stderr, "DH paper-object invariant failure (see checks above)\n");
    return ok ? 0 : 1;
}
