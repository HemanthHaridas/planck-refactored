#include "dh_pt2_gradient.h"

#include <algorithm>
#include <cmath>

#include "analytic_hessian.h"
#include "dft_gradient.h"
#include "integrals/base.h"

namespace
{
    std::size_t idx(int i, int j, int a, int b, int no, int nv)
    {
        return ((static_cast<std::size_t>(i) * no + j) * nv + a) * nv + b;
    }

    std::size_t eri_idx(int p, int q, int r, int s, int nmo)
    {
        return ((static_cast<std::size_t>(p) * nmo + q) * nmo + r) * nmo + s;
    }

    std::vector<double> eri_eightfold_symmetrize(const std::vector<double> &raw, int n)
    {
        std::vector<double> symmetric(raw.size(), 0.0);
        for (int mu = 0; mu < n; ++mu)
            for (int nu = 0; nu < n; ++nu)
                for (int ka = 0; ka < n; ++ka)
                    for (int ta = 0; ta < n; ++ta)
                    {
                        const auto get = [&raw, n](int p, int q, int r, int s)
                        { return raw[eri_idx(p, q, r, s, n)]; };
                        symmetric[eri_idx(mu, nu, ka, ta, n)] = 0.125 * (
                            get(mu, nu, ka, ta) + get(nu, mu, ka, ta) +
                            get(mu, nu, ta, ka) + get(nu, mu, ta, ka) +
                            get(ka, ta, mu, nu) + get(ta, ka, mu, nu) +
                            get(ka, ta, nu, mu) + get(ta, ka, nu, mu));
                    }
        return symmetric;
    }
}

namespace DFT::Gradient
{
    std::expected<DHEq41ResponseOperator, std::string>
    make_dh_eq41_response_operator(
        double exact_exchange,
        DHResponseFn coulomb,
        DHResponseFn exchange,
        DHResponseFn xc)
    {
        if (!std::isfinite(exact_exchange) || exact_exchange < 0.0 ||
            !coulomb || !xc || (exact_exchange != 0.0 && !exchange))
            return std::unexpected("DH Eq. 41 response: invalid channel configuration.");
        return DHEq41ResponseOperator{exact_exchange, std::move(coulomb),
                                    std::move(exchange), std::move(xc)};
    }

    std::expected<DHEq41ResponseOperator, std::string>
    make_dh_eq41_direct_eri_response_operator(const DHEq41DirectERIInputs &inputs)
    {
        if (inputs.shell_pairs == nullptr || inputs.nbasis == 0 ||
            !std::isfinite(inputs.tol_eri) || inputs.tol_eri < 0.0)
            return std::unexpected("DH Eq. 41 response: invalid direct-ERI inputs.");

        const auto direct = inputs;
        DHResponseFn coulomb = [direct](const Eigen::Ref<const Eigen::MatrixXd> &density)
            -> std::expected<Eigen::MatrixXd, std::string>
        {
            if (density.rows() != static_cast<Eigen::Index>(direct.nbasis) ||
                density.cols() != static_cast<Eigen::Index>(direct.nbasis))
                return std::unexpected("DH Eq. 41 Coulomb: density dimensions do not match the AO basis.");
            return _compute_2e_j_direct(*direct.shell_pairs, density, direct.nbasis,
                direct.engine, HartreeFock::ERIKernel::Coulomb, 0.0,
                direct.tol_eri, direct.sym_ops);
        };
        DHResponseFn exchange = [direct](const Eigen::Ref<const Eigen::MatrixXd> &density)
            -> std::expected<Eigen::MatrixXd, std::string>
        {
            if (density.rows() != static_cast<Eigen::Index>(direct.nbasis) ||
                density.cols() != static_cast<Eigen::Index>(direct.nbasis))
                return std::unexpected("DH Eq. 41 exchange: density dimensions do not match the AO basis.");
            return _compute_2e_k_direct(*direct.shell_pairs, density, direct.nbasis,
                direct.engine, HartreeFock::ERIKernel::Coulomb, 0.0,
                direct.tol_eri, direct.sym_ops);
        };
        return make_dh_eq41_response_operator(inputs.exact_exchange,
                                                std::move(coulomb), std::move(exchange),
                                                inputs.xc);
    }

    std::expected<DHResponseFn, std::string>
    make_dh_eq41_xc_response_callback(const DHEq41XCInputs &inputs)
    {
        if (inputs.molecular_grid == nullptr || inputs.ao_grid == nullptr ||
            inputs.exchange_functional == nullptr || inputs.correlation_functional == nullptr ||
            inputs.ground_density.rows() == 0 ||
            inputs.ground_density.rows() != inputs.ground_density.cols() ||
            inputs.ground_density.rows() != inputs.ao_grid->nbasis())
            return std::unexpected("DH Eq. 41 XC response: invalid fixed-geometry inputs.");
        const auto fixed = inputs;
        return DHResponseFn{[fixed](const Eigen::Ref<const Eigen::MatrixXd> &trial_density)
            -> std::expected<Eigen::MatrixXd, std::string>
        {
            if (trial_density.rows() != fixed.ground_density.rows() ||
                trial_density.cols() != fixed.ground_density.cols())
                return std::unexpected("DH Eq. 41 XC response: trial-density dimensions do not match ground density.");
            return DFT::Driver::compute_analytic_xc_hessian_vector_product(
                *fixed.molecular_grid, *fixed.ao_grid, fixed.ground_density, trial_density,
                *fixed.exchange_functional, *fixed.correlation_functional);
        }};
    }

    std::expected<Eigen::MatrixXd, std::string>
    DHEq41ResponseOperator::apply(const Eigen::Ref<const Eigen::MatrixXd> &density) const
    {
        auto channels = apply_channels(density);
        if (!channels) return std::unexpected(channels.error());
        return std::move(channels->total);
    }

    std::expected<DHEq41ResponseOperator::Channels, std::string>
    DHEq41ResponseOperator::apply_channels(const Eigen::Ref<const Eigen::MatrixXd> &density) const
    {
        if (density.rows() != density.cols())
            return std::unexpected("DH Eq. 41 response: density must be square.");
        auto j = coulomb(density);
        auto vxc = xc(density);
        if (!j || !vxc)
            return std::unexpected("DH Eq. 41 response: Coulomb or XC channel failed.");
        if (j->rows() != density.rows() || j->cols() != density.cols() ||
            vxc->rows() != density.rows() || vxc->cols() != density.cols())
            return std::unexpected("DH Eq. 41 response: channel dimensions do not match density.");
        Channels out;
        out.coulomb = 4.0 * *j;
        // `density` is the symmetric correction-density test object D' in
        // R(D').  Its adjoint against the physical closed-shell orbital
        // variation delta P=2(Cv X Co^T+Co X^T Cv^T) carries the same factor
        // four as the Coulomb channel.  The HVP itself is linear in D'; the
        // factor belongs to the closed-shell adjoint projection, not libxc.
        out.xc = 4.0 * *vxc;
        out.exchange = Eigen::MatrixXd::Zero(density.rows(), density.cols());
        if (exact_exchange != 0.0)
        {
            auto k = exchange(density);
            if (!k || k->rows() != density.rows() || k->cols() != density.cols())
                return std::unexpected("DH Eq. 41 response: exchange channel failed or has wrong dimensions.");
            // Eq. (41): -a_x [K[D'] + K[D']^T].  For symmetric D', the
            // two contractions are equal but remain explicit here.
            out.exchange.noalias() -= exact_exchange * (*k + k->transpose());
        }
        out.total = out.coulomb + out.exchange + out.xc;
        return out;
    }

    std::expected<DHEq41ResponseDensity, std::string>
    build_dh_eq41_response_density(
        const DHPT2AmplitudeDensity &dprime,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq41ResponseOperator &response)
    {
        const int nmo = dprime.n_occ + dprime.n_virt;
        if (nmo <= 0 || dprime.dprime_mo.rows() != nmo ||
            dprime.dprime_mo.cols() != nmo || mo_coeff.cols() != nmo)
            return std::unexpected("DH Eq. 41 response: inconsistent D' or MO-coefficient dimensions.");
        DHEq41ResponseDensity out;
        out.dprime_ao.noalias() = mo_coeff * dprime.dprime_mo * mo_coeff.transpose();
        auto applied = response.apply(out.dprime_ao);
        if (!applied) return std::unexpected(applied.error());
        out.response_ao = std::move(*applied);
        return out;
    }

    std::expected<DHPT2AmplitudeDensity, std::string>
    build_dh_pt2_amplitude_density(
        const HartreeFock::Correlation::RMP2Result &result,
        double c_pt2)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        if (no <= 0 || nv <= 0 || !std::isfinite(c_pt2))
            return std::unexpected("DH PT2 amplitudes: invalid dimensions or PT2 scale.");
        const std::size_t need = static_cast<std::size_t>(no) * no * nv * nv;
        if (result.t2.size() != need)
            return std::unexpected("DH PT2 amplitudes: stored T2 dimensions are inconsistent.");

        DHPT2AmplitudeDensity out;
        out.n_occ = no;
        out.n_virt = nv;
        out.t_tilde.assign(need, 0.0);
        out.dprime_oo = Eigen::MatrixXd::Zero(no, no);
        out.dprime_vv = Eigen::MatrixXd::Zero(nv, nv);

        // Eq. (39). c_PT2 belongs to the derivative correction, so scale the
        // amplitude-linear tilde tensor here and all density products below
        // use one scaled and one unscaled amplitude.
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        out.t_tilde[idx(i, j, a, b, no, nv)] = c_pt2 * 2.0 /
                            static_cast<double>(1 + (i == j)) *
                            (2.0 * result.t2[idx(i, j, a, b, no, nv)] -
                             result.t2[idx(i, j, b, a, no, nv)]);

        // Eq. (37): D'_ij = -sum_k (1+delta_ik) tr(t~^(ik) t^(kj)).
        // With the virtual-index trace written explicitly, the second factor
        // is t(k,j,b,a), not t(j,k,b,a).
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int k = 0; k < no; ++k)
                {
                    double trace = 0.0;
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                            trace += out.t_tilde[idx(i, k, a, b, no, nv)] *
                                result.t2[idx(k, j, b, a, no, nv)];
                    out.dprime_oo(i, j) -= (1 + (i == k)) * trace;
                }

        // Eq. (38): D'_ab = sum_{i<=j}(t~_ij t_ij^T + t~_ij^T t_ij)_ab.
        for (int i = 0; i < no; ++i)
            for (int j = i; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        for (int c = 0; c < nv; ++c)
                        {
                            out.dprime_vv(a, b) +=
                                out.t_tilde[idx(i, j, a, c, no, nv)] *
                                result.t2[idx(i, j, b, c, no, nv)] +
                                out.t_tilde[idx(i, j, c, a, no, nv)] *
                                result.t2[idx(i, j, c, b, no, nv)];
                        }

        out.dprime_mo = Eigen::MatrixXd::Zero(no + nv, no + nv);
        out.dprime_mo.topLeftCorner(no, no) = out.dprime_oo;
        out.dprime_mo.bottomRightCorner(nv, nv) = out.dprime_vv;
        return out;
    }

    std::expected<DHEq40AmplitudeRHS, std::string>
    build_dh_eq40_amplitude_rhs(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const std::vector<double> &mo_eri,
        double c_pt2,
        bool include_legacy_internal)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        const int nmo = no + nv;
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        const std::size_t ne = static_cast<std::size_t>(nmo) * nmo * nmo * nmo;
        if (no <= 0 || nv <= 0 || !std::isfinite(c_pt2) || result.t2.size() != nt ||
            amplitudes.n_occ != no || amplitudes.n_virt != nv ||
            amplitudes.t_tilde.size() != nt || mo_eri.size() != ne)
            return std::unexpected("DH Eq. 40 RHS: inconsistent amplitudes or MO ERI dimensions.");

        DHEq40AmplitudeRHS out;
        out.three_external = Eigen::MatrixXd::Zero(nv, no);
        out.internal_same_spin_exchange = Eigen::MatrixXd::Zero(nv, no);
        out.internal_opposite_spin_direct = Eigen::MatrixXd::Zero(nv, no);
        out.three_internal = Eigen::MatrixXd::Zero(nv, no);
        for (int a = 0; a < nv; ++a)
            for (int i = 0; i < no; ++i)
            {
                const int ap = no + a;
                // Literal non-metric derivative of the Eq. (47) pair
                // scalar.  For kappa_ai=1, dC_i=C_a and dC_a=-C_i.
                // Differentiate each of the four MO coefficients in
                // Theta_kl^cb (kc|ld), with Theta=(1+delta_kl)t_tilde.
                // The paper's compressed Eq. (40) first term is only valid
                // after amplitude symmetries that Planck's stored spatial
                // t2 tensor does not provide in that orientation.  Keeping
                // all four legs is the same convention already validated
                // for the literal Eq. (47) metric-overlap builder.
                for (int k = 0; k < no; ++k)
                    for (int l = 0; l < no; ++l)
                        for (int c = 0; c < nv; ++c)
                            for (int b = 0; b < nv; ++b)
                            {
                                const int cp = no + c;
                                const int bp = no + b;
                                const double theta = static_cast<double>(1 + (k == l)) *
                                    amplitudes.t_tilde[idx(k, l, c, b, no, nv)];
                                if (k == i)
                                    out.three_external(a, i) += theta *
                                        mo_eri[eri_idx(ap, cp, l, bp, nmo)];
                                if (c == a)
                                    out.three_external(a, i) -= theta *
                                        mo_eri[eri_idx(k, i, l, bp, nmo)];
                                if (l == i)
                                    out.three_external(a, i) += theta *
                                        mo_eri[eri_idx(k, cp, ap, bp, nmo)];
                                if (b == a)
                                    out.three_external(a, i) -= theta *
                                        mo_eri[eri_idx(k, cp, l, i, nmo)];
                            }
                // The excluded internal bracket is available only for
                // standalone reference tests, never evaluated by production.
                if (!include_legacy_internal) continue;
                for (int k = 0; k < no; ++k)
                    for (int l = 0; l < no; ++l)
                        for (int b = 0; b < nv; ++b)
                        {
                            const int bp = no + b;
                            const double scaled_t = c_pt2 * result.t2[idx(k, l, a, b, no, nv)];
                            // Eq. (22) -> Eq. (40), retained literally:
                            // same-spin exchange and opposite-spin direct.
                            out.internal_same_spin_exchange(a, i) -= scaled_t *
                                mo_eri[eri_idx(k, i, l, bp, nmo)];
                            out.internal_opposite_spin_direct(a, i) += scaled_t *
                                mo_eri[eri_idx(k, bp, l, i, nmo)];
                        }
            }
        out.three_internal = out.internal_same_spin_exchange + out.internal_opposite_spin_direct;
        out.total = out.three_external + out.three_internal;
        return out;
    }

    std::expected<DHLagrangianRHS, std::string>
    build_dh_lagrangian_rhs(
        const DHEq41ResponseDensity &eq41,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt,
        const DHEq40AmplitudeRHS &amplitude,
        DHRHSConvention convention)
    {
        const Eigen::Index nao = c_occ.rows();
        const Eigen::Index no = c_occ.cols();
        const Eigen::Index nv = c_virt.cols();
        if (nao == 0 || c_virt.rows() != nao || eq41.response_ao.rows() != nao ||
            eq41.response_ao.cols() != nao || amplitude.three_external.rows() != nv ||
            amplitude.three_external.cols() != no || amplitude.three_internal.rows() != nv ||
            amplitude.three_internal.cols() != no || amplitude.total.rows() != nv ||
            amplitude.total.cols() != no)
            return std::unexpected("DH Eq. 40 RHS: incompatible AO response, orbitals, or amplitude blocks.");

        DHLagrangianRHS out;
        out.response_ai.noalias() = c_virt.transpose() * eq41.response_ao * c_occ;
        out.three_external_ai = amplitude.three_external;
        out.internal_same_spin_exchange_ai = amplitude.internal_same_spin_exchange;
        out.internal_opposite_spin_direct_ai = amplitude.internal_opposite_spin_direct;
        out.three_internal_ai = amplitude.three_internal;
        out.included_internal_ai = amplitude.three_internal;
        if (convention == DHRHSConvention::LiteralEq47)
            out.included_internal_ai.setZero();
        out.amplitude_ai = out.three_external_ai + out.included_internal_ai;
        out.total_ai = out.response_ai + out.amplitude_ai;
        return out;
    }

    std::expected<DHRelaxedDifferenceDensity, std::string>
    build_dh_relaxed_difference_density(
        const DHPT2AmplitudeDensity &dprime,
        const Eigen::Ref<const Eigen::MatrixXd> &z_ai)
    {
        const int no = dprime.n_occ;
        const int nv = dprime.n_virt;
        const int nmo = no + nv;
        if (no <= 0 || nv <= 0 || dprime.dprime_mo.rows() != nmo ||
            dprime.dprime_mo.cols() != nmo || z_ai.rows() != nv || z_ai.cols() != no)
            return std::unexpected("DH Eq. 28 relaxed density: incompatible D' or Z_ai dimensions.");

        DHRelaxedDifferenceDensity out;
        out.z_ai = z_ai;
        out.raw_mo = dprime.dprime_mo;
        out.raw_mo.bottomLeftCorner(nv, no) = z_ai;

        // For a symmetric M, <Z_vo,M> = <1/2 Z_vo + 1/2 Z_ov,M>.
        // Keep this adapter separate: the raw Eq. (28) matrix is not symmetric.
        out.symmetric_mo = dprime.dprime_mo;
        out.symmetric_mo.bottomLeftCorner(nv, no) = 0.5 * z_ai;
        out.symmetric_mo.topRightCorner(no, nv) = 0.5 * z_ai.transpose();
        return out;
    }

    std::expected<DHEq27HessianAction, std::string>
    apply_dh_eq27_hessian(
        const Eigen::Ref<const Eigen::MatrixXd> &z_ai,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt,
        const Eigen::Ref<const Eigen::VectorXd> &orbital_energies,
        const DHEq41ResponseOperator &response)
    {
        const Eigen::Index nao = c_occ.rows();
        const Eigen::Index no = c_occ.cols();
        const Eigen::Index nv = c_virt.cols();
        if (nao == 0 || c_virt.rows() != nao || z_ai.rows() != nv || z_ai.cols() != no ||
            orbital_energies.size() != no + nv)
            return std::unexpected("DH Eq. 27 Hessian: incompatible Z_ai, orbitals, or orbital energies.");

        DHEq27HessianAction out;
        out.raw_z_ao.noalias() = c_virt * z_ai * c_occ.transpose();
        // Eq. (27) is an orbital Hessian, not the Eq. (41) D' response.
        // A non-metric orbital rotation z_ai changes the *closed-shell
        // total density* by delta P = 2(Cv z Co^T + Co z^T Cv^T).  Feeding
        // the raw Cv z Co^T matrix into R(D') misses this symmetrization and
        // its factor of two, most visibly in the XC Hessian channel.
        const Eigen::MatrixXd delta_density = 2.0 * (out.raw_z_ao + out.raw_z_ao.transpose());
        auto coulomb = response.coulomb(delta_density);
        auto xc = response.xc(delta_density);
        if (!coulomb || !xc)
            return std::unexpected("DH Eq. 27 Hessian: Coulomb or XC orbital-density response failed.");
        if (coulomb->rows() != nao || coulomb->cols() != nao || xc->rows() != nao || xc->cols() != nao)
            return std::unexpected("DH Eq. 27 Hessian: Coulomb or XC orbital-density response has wrong dimensions.");
        const auto transform_ai = [&c_occ, &c_virt](const Eigen::MatrixXd &ao)
        { return c_virt.transpose() * ao * c_occ; };
        out.coulomb_ai = transform_ai(*coulomb);
        out.xc_ai = transform_ai(*xc);
        out.exchange_ai = Eigen::MatrixXd::Zero(nv, no);
        if (response.exact_exchange != 0.0)
        {
            auto exchange = response.exchange(delta_density);
            if (!exchange || exchange->rows() != nao || exchange->cols() != nao)
                return std::unexpected("DH Eq. 27 Hessian: exchange orbital-density response failed.");
            out.exchange_ai.noalias() = -0.5 * response.exact_exchange * transform_ai(*exchange);
        }
        out.response_ai = out.coulomb_ai + out.exchange_ai + out.xc_ai;
        out.orbital_energy_ai = Eigen::MatrixXd::Zero(nv, no);
        for (Eigen::Index a = 0; a < nv; ++a)
            for (Eigen::Index i = 0; i < no; ++i)
                out.orbital_energy_ai(a, i) =
                    (orbital_energies(no + a) - orbital_energies(i)) * z_ai(a, i);
        out.total_ai = out.orbital_energy_ai + out.response_ai;
        return out;
    }

    std::expected<DHEq42_43EnergyWeightedDensity, std::string>
    build_dh_eq42_43_energy_weighted_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const Eigen::Ref<const Eigen::VectorXd> &orbital_energies,
        const std::vector<double> &mo_eri,
        const DHEq41ResponseOperator &response,
        bool include_legacy_pair)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        const int nmo = no + nv;
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        const std::size_t ne = static_cast<std::size_t>(nmo) * nmo * nmo * nmo;
        if (no <= 0 || nv <= 0 || result.t2.size() != nt || amplitudes.n_occ != no ||
            amplitudes.n_virt != nv || amplitudes.t_tilde.size() != nt ||
            relaxed.raw_mo.rows() != nmo || relaxed.raw_mo.cols() != nmo ||
            mo_coeff.cols() != nmo || orbital_energies.size() != nmo || mo_eri.size() != ne)
            return std::unexpected("DH Eq. 42-43 density: incompatible densities, orbitals, amplitudes, or MO ERIs.");

        // After Eq. (28), the paper explicitly symmetrizes D before forming
        // the energy-weighted density.  R(D)_ij in Eq. (42) is therefore the
        // response to D' + 1/2(Z_vo + Z_ov), not to raw D' + Z_vo.
        const Eigen::MatrixXd relaxed_ao =
            mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        auto response_ao = response.apply(relaxed_ao);
        if (!response_ao) return std::unexpected(response_ao.error());
        const Eigen::MatrixXd response_mo = mo_coeff.transpose() * *response_ao * mo_coeff;

        DHEq42_43EnergyWeightedDensity out;
        out.response_oo = -0.5 * response_mo.topLeftCorner(no, no);
        out.orbital_oo = Eigen::MatrixXd::Zero(no, no);
        out.amplitude_oo = Eigen::MatrixXd::Zero(no, no);
        out.orbital_vv = Eigen::MatrixXd::Zero(nv, nv);
        out.amplitude_vv = Eigen::MatrixXd::Zero(nv, nv);
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
            {
                out.orbital_oo(i,j) = -0.5 * relaxed.raw_mo(i,j) *
                    (orbital_energies(i) + orbital_energies(j));
                if (!include_legacy_pair) continue;
                for (int k = 0; k < no; ++k)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                            out.amplitude_oo(i,j) -= 0.5 *
                                amplitudes.t_tilde[idx(j,k,a,b,no,nv)] *
                                mo_eri[eri_idx(i,no+a,k,no+b,nmo)];
            }
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b)
            {
                out.orbital_vv(a,b) = -0.5 * relaxed.raw_mo(no+a,no+b) *
                    (orbital_energies(no+a) + orbital_energies(no+b));
                if (!include_legacy_pair) continue;
                for (int i = 0; i < no; ++i)
                    for (int j = i; j < no; ++j)
                        for (int c = 0; c < nv; ++c)
                            out.amplitude_vv(a,b) -=
                                mo_eri[eri_idx(i,no+a,j,no+c,nmo)] *
                                amplitudes.t_tilde[idx(i,j,b,c,no,nv)] /
                                static_cast<double>(1 + (i == j));
            }
        out.w_oo = out.response_oo + out.orbital_oo + out.amplitude_oo;
        out.w_vv = out.orbital_vv + out.amplitude_vv;
        return out;
    }

    std::expected<DHEq44_45EnergyWeightedDensity, std::string>
    build_dh_eq44_45_energy_weighted_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::VectorXd> &orbital_energies,
        const std::vector<double> &mo_eri,
        bool include_legacy_pair)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        const int nmo = no + nv;
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        const std::size_t ne = static_cast<std::size_t>(nmo) * nmo * nmo * nmo;
        if (no <= 0 || nv <= 0 || result.t2.size() != nt || amplitudes.n_occ != no ||
            amplitudes.n_virt != nv || amplitudes.t_tilde.size() != nt ||
            relaxed.z_ai.rows() != nv || relaxed.z_ai.cols() != no ||
            orbital_energies.size() != nmo || mo_eri.size() != ne)
            return std::unexpected("DH Eq. 44-45 density: incompatible Z_ai, amplitudes, orbital energies, or MO ERIs.");

        DHEq44_45EnergyWeightedDensity out;
        out.w_ia = Eigen::MatrixXd::Zero(no, nv);
        out.w_ai = Eigen::MatrixXd::Zero(nv, no);
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
            {
                out.w_ai(a,i) = -orbital_energies(i) * relaxed.z_ai(a,i);
                if (!include_legacy_pair) continue;
                for (int k = 0; k < no; ++k)
                    for (int j = 0; j < no; ++j)
                        for (int b = 0; b < nv; ++b)
                            out.w_ia(i,a) -= amplitudes.t_tilde[idx(k,j,a,b,no,nv)] *
                                mo_eri[eri_idx(k,i,j,no+b,nmo)];
            }
        return out;
    }

    std::expected<DHEq47PairMetricOverlapDensity, std::string>
    build_dh_eq47_pair_metric_overlap_density(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const std::vector<double> &mo_eri)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        const int nmo = no + nv;
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        const std::size_t ne = static_cast<std::size_t>(nmo) * nmo * nmo * nmo;
        if (no <= 0 || nv <= 0 || amplitudes.n_occ != no || amplitudes.n_virt != nv ||
            amplitudes.t_tilde.size() != nt || mo_eri.size() != ne)
            return std::unexpected("DH Eq. 47 pair metric overlap: incompatible amplitudes or MO ERIs.");

        // Let Theta_ij^ab=(1+delta_ij)t~_ij^ab.  Differentiate the four
        // coefficients in Theta_ij^ab (ia|jb) literally, then insert the
        // paper metric gauge U_oo=U_vv=-S/2, U_ia=-S_ia, U_ai=0.  Retaining
        // both appearances of each occupied/virtual coefficient is essential;
        // only afterwards may pair symmetries compress these sums.
        DHEq47PairMetricOverlapDensity out;
        out.w_oo = Eigen::MatrixXd::Zero(no, no);
        out.w_vv = Eigen::MatrixXd::Zero(nv, nv);
        out.w_ia = Eigen::MatrixXd::Zero(no, nv);
        out.w_ai = Eigen::MatrixXd::Zero(nv, no);
        const auto theta = [&amplitudes, no, nv](int i, int j, int a, int b)
        {
            return static_cast<double>(1 + (i == j)) * amplitudes.t_tilde[idx(i, j, a, b, no, nv)];
        };
        for (int p = 0; p < no; ++p)
            for (int q = 0; q < no; ++q)
            {
                double value = 0.0;
                for (int j = 0; j < no; ++j)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                            value += theta(q, j, a, b) *
                                mo_eri[eri_idx(p, no + a, j, no + b, nmo)];
                for (int i = 0; i < no; ++i)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                            value += theta(i, q, a, b) *
                                mo_eri[eri_idx(i, no + a, p, no + b, nmo)];
                out.w_oo(p, q) = -0.5 * value;
            }
        for (int p = 0; p < nv; ++p)
            for (int q = 0; q < nv; ++q)
            {
                double value = 0.0;
                for (int i = 0; i < no; ++i)
                    for (int j = 0; j < no; ++j)
                        for (int b = 0; b < nv; ++b)
                            value += theta(i, j, q, b) *
                                mo_eri[eri_idx(i, no + p, j, no + b, nmo)];
                for (int i = 0; i < no; ++i)
                    for (int j = 0; j < no; ++j)
                        for (int a = 0; a < nv; ++a)
                            value += theta(i, j, a, q) *
                                mo_eri[eri_idx(i, no + a, j, no + p, nmo)];
                out.w_vv(p, q) = -0.5 * value;
            }
        for (int p = 0; p < no; ++p)
            for (int q = 0; q < nv; ++q)
            {
                double value = 0.0;
                for (int i = 0; i < no; ++i)
                    for (int j = 0; j < no; ++j)
                        for (int b = 0; b < nv; ++b)
                            value += theta(i, j, q, b) *
                                mo_eri[eri_idx(i, p, j, no + b, nmo)];
                for (int i = 0; i < no; ++i)
                    for (int j = 0; j < no; ++j)
                        for (int a = 0; a < nv; ++a)
                            value += theta(i, j, a, q) *
                                mo_eri[eri_idx(i, no + a, j, p, nmo)];
                out.w_ia(p, q) = -value;
            }
        return out;
    }

    std::expected<DHOverlapDensity, std::string>
    build_dh_overlap_density_adapter(
        const DHEq42_43EnergyWeightedDensity &diagonal,
        const DHEq44_45EnergyWeightedDensity &off_diagonal)
    {
        const Eigen::Index no = diagonal.w_oo.rows();
        const Eigen::Index nv = diagonal.w_vv.rows();
        if (no <= 0 || nv <= 0 || diagonal.w_oo.cols() != no || diagonal.w_vv.cols() != nv ||
            off_diagonal.w_ia.rows() != no || off_diagonal.w_ia.cols() != nv ||
            off_diagonal.w_ai.rows() != nv || off_diagonal.w_ai.cols() != no)
            return std::unexpected("DH overlap-density adapter: incompatible Eq. 42-45 block dimensions.");

        DHOverlapDensity out;
        out.raw_mo = Eigen::MatrixXd::Zero(no + nv, no + nv);
        out.raw_mo.topLeftCorner(no, no) = diagonal.w_oo;
        out.raw_mo.topRightCorner(no, nv) = off_diagonal.w_ia;
        out.raw_mo.bottomLeftCorner(nv, no) = off_diagonal.w_ai;
        out.raw_mo.bottomRightCorner(nv, nv) = diagonal.w_vv;
        out.symmetric_mo = 0.5 * (out.raw_mo + out.raw_mo.transpose());
        return out;
    }

    std::expected<DHEq46_47TwoParticleDensity, std::string>
    build_dh_eq46_47_two_particle_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density_ao,
        double exact_exchange)
    {
        const int no = amplitudes.n_occ;
        const int nv = amplitudes.n_virt;
        const int nmo = no + nv;
        const Eigen::Index nao = mo_coeff.rows();
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        if (!std::isfinite(exact_exchange) || exact_exchange < 0.0)
            return std::unexpected("DH Eq. 46-47 density: invalid exact-exchange coefficient.");
        if (no <= 0 || nv <= 0 || nao <= 0 || mo_coeff.cols() != nmo ||
            amplitudes.t_tilde.size() != nt || relaxed.symmetric_mo.rows() != nmo ||
            relaxed.symmetric_mo.cols() != nmo || ground_density_ao.rows() != nao ||
            ground_density_ao.cols() != nao)
            return std::unexpected("DH Eq. 46-47 density: incompatible relaxed density, amplitudes, orbitals, or ground density.");

        DHEq46_47TwoParticleDensity out;
        out.n_ao = static_cast<int>(nao);
        out.relaxed_difference_ao.noalias() =
            mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const std::size_t n4 = static_cast<std::size_t>(nao) * nao * nao * nao;
        out.separable_raw_ao.assign(n4, 0.0);
        out.nonseparable_raw_ao.assign(n4, 0.0);

        // Eq. (46), correction only: its 1/2 P P - 1/4 P P terms are the
        // SCF reference and deliberately do not enter the PT2 correction.
        // The separable term differentiates D:F_KS, so it must retain the
        // hybrid coefficient in F_KS=h+J[P]-a_x K[P]/2+V_XC[P]. This also
        // applies to the Z-dependent part of D. Eq. (47) is unscaled by a_x.
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu)
                for (int ka = 0; ka < nao; ++ka)
                    for (int ta = 0; ta < nao; ++ta)
                        out.separable_raw_ao[eri_idx(mu, nu, ka, ta, nao)] =
                            out.relaxed_difference_ao(mu, nu) * ground_density_ao(ka, ta) -
                            0.5 * exact_exchange * out.relaxed_difference_ao(mu, ka) * ground_density_ao(nu, ta);

        // Eq. (47), direct MO-to-AO backtransformation.  The AO pair order
        // is (mu nu | ka ta), so its closed-shell spatial realization is
        // C_mu,i C_nu,a C_ka,j C_ta,b t~_ab^ij, exactly as printed in the
        // paper.  In particular, this is the MP2 (ia|jb) pairing -- it is
        // not C_mu,i C_nu,j C_ka,a C_ta,b, which would contract t~ with the
        // unrelated (ij|ab) integral and violates Eq. (47).  t_tilde already
        // carries c_PT2; the explicit (1+delta_ij) is paper-defined and must
        // not be folded into the amplitude convention.
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu)
                for (int ka = 0; ka < nao; ++ka)
                    for (int ta = 0; ta < nao; ++ta)
                        for (int i = 0; i < no; ++i)
                            for (int j = 0; j < no; ++j)
                                for (int a = 0; a < nv; ++a)
                                    for (int b = 0; b < nv; ++b)
                                        out.nonseparable_raw_ao[eri_idx(mu, nu, ka, ta, nao)] +=
                                            mo_coeff(mu, i) * mo_coeff(nu, no + a) *
                                            mo_coeff(ka, j) * mo_coeff(ta, no + b) *
                                            static_cast<double>(1 + (i == j)) *
                                            amplitudes.t_tilde[idx(i, j, a, b, no, nv)];

        out.total_raw_ao = out.separable_raw_ao;
        for (std::size_t p = 0; p < n4; ++p)
            out.total_raw_ao[p] += out.nonseparable_raw_ao[p];
        out.separable_symmetric_ao = eri_eightfold_symmetrize(out.separable_raw_ao, out.n_ao);
        out.nonseparable_symmetric_ao = eri_eightfold_symmetrize(out.nonseparable_raw_ao, out.n_ao);
        out.total_symmetric_ao = eri_eightfold_symmetrize(out.total_raw_ao, out.n_ao);
        return out;
    }

    std::expected<DHEq33NonXCGradient, std::string>
    build_dh_eq33_non_xc_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const DHOverlapDensity &overlap,
        const DHEq46_47TwoParticleDensity &two_particle,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33NonXCDerivatives &derivatives)
    {
        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const std::size_t n4 = static_cast<std::size_t>(nao) * nao * nao * nao;
        const std::size_t ncoord = derivatives.hamiltonian_ao.size();
        if (nao <= 0 || nmo <= 0 || relaxed.symmetric_mo.rows() != nmo ||
            relaxed.symmetric_mo.cols() != nmo || overlap.symmetric_mo.rows() != nmo ||
            overlap.symmetric_mo.cols() != nmo || two_particle.n_ao != nao ||
            two_particle.separable_symmetric_ao.size() != n4 ||
            two_particle.nonseparable_symmetric_ao.size() != n4 ||
            two_particle.total_symmetric_ao.size() != n4 ||
            derivatives.overlap_ao.size() != ncoord || derivatives.eri_ao.size() != ncoord)
            return std::unexpected("DH Eq. 33 non-XC gradient: incompatible densities, orbitals, or derivative dimensions.");

        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const Eigen::MatrixXd w_ao = mo_coeff * overlap.symmetric_mo * mo_coeff.transpose();
        DHEq33NonXCGradient out;
        out.one_electron = Eigen::VectorXd::Zero(ncoord);
        out.overlap = Eigen::VectorXd::Zero(ncoord);
        out.two_electron_separable = Eigen::VectorXd::Zero(ncoord);
        out.two_electron_nonseparable = Eigen::VectorXd::Zero(ncoord);
        for (std::size_t x = 0; x < ncoord; ++x)
        {
            const auto &h = derivatives.hamiltonian_ao[x];
            const auto &s = derivatives.overlap_ao[x];
            const auto &eri = derivatives.eri_ao[x];
            if (h.rows() != nao || h.cols() != nao || s.rows() != nao || s.cols() != nao ||
                eri.size() != n4)
                return std::unexpected("DH Eq. 33 non-XC gradient: derivative AO dimensions are inconsistent.");
            out.one_electron(x) = (d_ao.array() * h.array()).sum();
            out.overlap(x) = (w_ao.array() * s.array()).sum();
            for (std::size_t q = 0; q < n4; ++q)
            {
                out.two_electron_separable(x) += two_particle.separable_symmetric_ao[q] * eri[q];
                out.two_electron_nonseparable(x) += two_particle.nonseparable_symmetric_ao[q] * eri[q];
            }
        }
        out.two_electron = out.two_electron_separable + out.two_electron_nonseparable;
        out.total = out.one_electron + out.overlap + out.two_electron;
        return out;
    }

    std::expected<DHEq33XCFixedDensityGradient, std::string>
    build_dh_eq33_xc_fixed_density_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.basis == nullptr || inputs.molecular_grid == nullptr ||
            inputs.ao_grid == nullptr || inputs.ao_hessian == nullptr ||
            inputs.exchange_functional == nullptr || inputs.correlation_functional == nullptr ||
            inputs.exchange_functional->is_lda_like() != inputs.correlation_functional->is_lda_like() ||
            inputs.exchange_functional->is_gga_like() != inputs.correlation_functional->is_gga_like())
            return std::unexpected("DH Eq. 33 XC-II: invalid fixed-density grid or functional inputs.");

        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        const auto &hess = *inputs.ao_hessian;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || grid.points.rows() != ao.npoints() || grid.points.cols() != 4 ||
            hess.nbasis() != nao || hess.npoints() != ao.npoints())
            return std::unexpected("DH Eq. 33 XC-II: inconsistent density, AO grid, or Hessian dimensions.");

        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        auto difference = evaluate_density_on_grid(ao, d_ao);
        if (!ground || !difference)
            return std::unexpected("DH Eq. 33 XC-II: failed to evaluate ground or relaxed density on grid.");
        auto atoms_bf = atom_bf_lists(mol, *inputs.basis);
        if (!atoms_bf) return std::unexpected(atoms_bf.error());

        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
            rho[static_cast<std::size_t>(p)] = ground->total.rho(p);

        const auto zero_correlation_if_combined = [&inputs](std::vector<double> &values)
        {
            if (inputs.exchange_functional->is_combined_exchange_correlation())
                std::fill(values.begin(), values.end(), 0.0);
        };
        std::vector<double> frr_x, frr_c, frs_x, frs_c, fss_x, fss_c, fs_x, fs_c;
        if (inputs.exchange_functional->is_lda_like())
        {
            auto x = inputs.exchange_functional->evaluate_lda_fxc(rho, static_cast<int>(npts), frr_x);
            auto c = inputs.correlation_functional->evaluate_lda_fxc(rho, static_cast<int>(npts), frr_c);
            if (!x || !c) return std::unexpected("DH Eq. 33 XC-II: LDA fxc evaluation failed.");
            zero_correlation_if_combined(frr_c);
        }
        else if (inputs.exchange_functional->is_gga_like())
        {
            std::vector<double> sigma(static_cast<std::size_t>(npts));
            for (Eigen::Index p = 0; p < npts; ++p)
                sigma[static_cast<std::size_t>(p)] = ground->total.gradient_squared()(p);
            std::vector<double> exc;
            std::vector<double> vrho_x, vrho_c, exc_c;
            auto vx = inputs.exchange_functional->evaluate_gga_exc_vxc(
                rho, sigma, static_cast<int>(npts), exc, vrho_x, fs_x);
            auto vc = inputs.correlation_functional->evaluate_gga_exc_vxc(
                rho, sigma, static_cast<int>(npts), exc_c, vrho_c, fs_c);
            auto fx = inputs.exchange_functional->evaluate_gga_fxc(
                rho, sigma, static_cast<int>(npts), frr_x, frs_x, fss_x);
            auto fc = inputs.correlation_functional->evaluate_gga_fxc(
                rho, sigma, static_cast<int>(npts), frr_c, frs_c, fss_c);
            if (!vx || !vc || !fx || !fc)
                return std::unexpected("DH Eq. 33 XC-II: GGA vxc/fxc evaluation failed.");
            if (inputs.exchange_functional->is_combined_exchange_correlation())
            {
                std::fill(frr_c.begin(), frr_c.end(), 0.0);
                std::fill(frs_c.begin(), frs_c.end(), 0.0);
                std::fill(fss_c.begin(), fss_c.end(), 0.0);
                std::fill(fs_c.begin(), fs_c.end(), 0.0);
            }
        }
        else
            return std::unexpected("DH Eq. 33 XC-II: functional is neither LDA-like nor GGA-like.");

        DHEq33XCFixedDensityGradient out;
        out.gradient = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        const bool gga = inputs.exchange_functional->is_gga_like();
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const std::size_t pi = static_cast<std::size_t>(p);
            const Eigen::Vector3d grad_p{ground->total.grad_x(p), ground->total.grad_y(p), ground->total.grad_z(p)};
            const Eigen::Vector3d grad_d{difference->total.grad_x(p), difference->total.grad_y(p), difference->total.grad_z(p)};
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                {
                    const double r = drho_channel(inputs.ground_density_ao, ao, p, static_cast<int>(atom), q, *atoms_bf);
                    if (!gga)
                    {
                        out.gradient(static_cast<Eigen::Index>(atom), q) +=
                            weight * (frr_x[pi] + frr_c[pi]) * r * difference->total.rho(p);
                        continue;
                    }
                    const Eigen::Vector3d g{
                        dg_axis_spin(inputs.ground_density_ao, ao, hess, p, 0, static_cast<int>(atom), q, *atoms_bf),
                        dg_axis_spin(inputs.ground_density_ao, ao, hess, p, 1, static_cast<int>(atom), q, *atoms_bf),
                        dg_axis_spin(inputs.ground_density_ao, ao, hess, p, 2, static_cast<int>(atom), q, *atoms_bf)};
                    const double dot_pg = grad_p.dot(g);
                    const double frr = frr_x[pi] + frr_c[pi];
                    const double frs = frs_x[pi] + frs_c[pi];
                    const double fss = fss_x[pi] + fss_c[pi];
                    const double fs = fs_x[pi] + fs_c[pi];
                    out.gradient(static_cast<Eigen::Index>(atom), q) += weight * (
                        (frr * r + 2.0 * frs * dot_pg) * difference->total.rho(p) +
                        2.0 * (frs * r + 2.0 * fss * dot_pg) * grad_p.dot(grad_d) +
                        2.0 * fs * g.dot(grad_d));
                }
        }
        return out;
    }

    std::expected<DHEq33XCLDAMovingGridGradient, std::string>
    build_dh_eq33_xc_lda_moving_grid_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.molecular_grid == nullptr || inputs.ao_grid == nullptr ||
            inputs.exchange_functional == nullptr || inputs.correlation_functional == nullptr ||
            !inputs.exchange_functional->is_lda_like() || !inputs.correlation_functional->is_lda_like())
            return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: invalid LDA grid or functional inputs.");

        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || grid.points.rows() != ao.npoints() || grid.points.cols() != 4 ||
            grid.owner.size() != ao.npoints() || grid.atomic_weights.size() != ao.npoints())
            return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: inconsistent density or molecular grid dimensions.");

        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        const auto difference = evaluate_density_on_grid(ao, d_ao);
        if (!ground || !difference)
            return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: failed to evaluate density on grid.");

        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
            rho[static_cast<std::size_t>(p)] = ground->total.rho(p);
        std::vector<double> exc_x, exc_c, vrho_x, vrho_c, frr_x, frr_c;
        const auto vx = inputs.exchange_functional->evaluate_lda_exc_vxc(
            rho, static_cast<int>(npts), exc_x, vrho_x);
        const auto vc = inputs.correlation_functional->evaluate_lda_exc_vxc(
            rho, static_cast<int>(npts), exc_c, vrho_c);
        const auto fx = inputs.exchange_functional->evaluate_lda_fxc(
            rho, static_cast<int>(npts), frr_x);
        const auto fc = inputs.correlation_functional->evaluate_lda_fxc(
            rho, static_cast<int>(npts), frr_c);
        if (!vx || !vc || !fx || !fc)
            return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: LDA vxc/fxc evaluation failed.");
        if (inputs.exchange_functional->is_combined_exchange_correlation())
        {
            std::fill(vrho_c.begin(), vrho_c.end(), 0.0);
            std::fill(frr_c.begin(), frr_c.end(), 0.0);
        }

        DHEq33XCLDAMovingGridGradient out;
        out.becke_partition = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        out.point_translation = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const int owner = grid.owner(p);
            if (owner < 0 || owner >= static_cast<int>(mol.natoms))
                return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: grid owner is out of range.");
            const std::size_t pi = static_cast<std::size_t>(p);
            const double vrho = vrho_x[pi] + vrho_c[pi];
            const double frr = frr_x[pi] + frr_c[pi];
            const double rho_d = difference->total.rho(p);
            const double integrand = vrho * rho_d;
            const auto dpartition = becke_partition_owner_derivatives(grid, mol, p);
            if (!dpartition) return std::unexpected(dpartition.error());
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                    out.becke_partition(static_cast<Eigen::Index>(atom), q) +=
                        grid.atomic_weights(p) * (*dpartition)(static_cast<Eigen::Index>(atom), q) * integrand;
            for (int q = 0; q < 3; ++q)
            {
                const double drho_dq = q == 0 ? difference->total.grad_x(p) :
                    (q == 1 ? difference->total.grad_y(p) : difference->total.grad_z(p));
                const double drho_pq = q == 0 ? ground->total.grad_x(p) :
                    (q == 1 ? ground->total.grad_y(p) : ground->total.grad_z(p));
                out.point_translation(owner, q) += weight *
                    (frr * drho_pq * rho_d + vrho * drho_dq);
            }
        }
        out.total = out.becke_partition + out.point_translation;
        return out;
    }

    std::expected<DHEq33XCLDADifferenceAOGradient, std::string>
    build_dh_eq33_xc_lda_difference_ao_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.basis == nullptr || inputs.molecular_grid == nullptr ||
            inputs.ao_grid == nullptr || inputs.exchange_functional == nullptr ||
            inputs.correlation_functional == nullptr || !inputs.exchange_functional->is_lda_like() ||
            !inputs.correlation_functional->is_lda_like())
            return std::unexpected("DH Eq. 33 LDA D-AO XC-II: invalid LDA grid or functional inputs.");
        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || grid.points.rows() != ao.npoints() || grid.points.cols() != 4)
            return std::unexpected("DH Eq. 33 LDA D-AO XC-II: inconsistent density or AO grid dimensions.");
        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        if (!ground)
            return std::unexpected("DH Eq. 33 LDA D-AO XC-II: failed to evaluate ground density on grid.");
        const auto atoms_bf = atom_bf_lists(mol, *inputs.basis);
        if (!atoms_bf) return std::unexpected(atoms_bf.error());
        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts)), exc_x, exc_c, vrho_x, vrho_c;
        for (Eigen::Index p = 0; p < npts; ++p)
            rho[static_cast<std::size_t>(p)] = ground->total.rho(p);
        const auto vx = inputs.exchange_functional->evaluate_lda_exc_vxc(
            rho, static_cast<int>(npts), exc_x, vrho_x);
        const auto vc = inputs.correlation_functional->evaluate_lda_exc_vxc(
            rho, static_cast<int>(npts), exc_c, vrho_c);
        if (!vx || !vc)
            return std::unexpected("DH Eq. 33 LDA D-AO XC-II: LDA vxc evaluation failed.");
        if (inputs.exchange_functional->is_combined_exchange_correlation())
            std::fill(vrho_c.begin(), vrho_c.end(), 0.0);
        DHEq33XCLDADifferenceAOGradient out;
        out.gradient = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const double vrho = vrho_x[static_cast<std::size_t>(p)] + vrho_c[static_cast<std::size_t>(p)];
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                    out.gradient(static_cast<Eigen::Index>(atom), q) += weight * vrho *
                        drho_channel(d_ao, ao, p, static_cast<int>(atom), q, *atoms_bf);
        }
        return out;
    }

    std::expected<DHEq33XCGGAMovingGridGradient, std::string>
    build_dh_eq33_xc_gga_moving_grid_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.molecular_grid == nullptr || inputs.ao_grid == nullptr ||
            inputs.ao_hessian == nullptr || inputs.exchange_functional == nullptr ||
            inputs.correlation_functional == nullptr || !inputs.exchange_functional->is_gga_like() ||
            !inputs.correlation_functional->is_gga_like())
            return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: invalid GGA grid or functional inputs.");

        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        const auto &hess = *inputs.ao_hessian;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || hess.nbasis() != nao || hess.npoints() != ao.npoints() ||
            grid.points.rows() != ao.npoints() || grid.points.cols() != 4 ||
            grid.owner.size() != ao.npoints() || grid.atomic_weights.size() != ao.npoints())
            return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: inconsistent density, AO grid, or Hessian dimensions.");

        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        const auto difference = evaluate_density_on_grid(ao, d_ao);
        if (!ground || !difference)
            return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: failed to evaluate density on grid.");

        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts)), sigma(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            rho[static_cast<std::size_t>(p)] = ground->total.rho(p);
            sigma[static_cast<std::size_t>(p)] = ground->total.gradient_squared()(p);
        }
        std::vector<double> exc_x, exc_c, vrho_x, vrho_c, fs_x, fs_c;
        std::vector<double> frr_x, frr_c, frs_x, frs_c, fss_x, fss_c;
        const auto vx = inputs.exchange_functional->evaluate_gga_exc_vxc(
            rho, sigma, static_cast<int>(npts), exc_x, vrho_x, fs_x);
        const auto vc = inputs.correlation_functional->evaluate_gga_exc_vxc(
            rho, sigma, static_cast<int>(npts), exc_c, vrho_c, fs_c);
        const auto fx = inputs.exchange_functional->evaluate_gga_fxc(
            rho, sigma, static_cast<int>(npts), frr_x, frs_x, fss_x);
        const auto fc = inputs.correlation_functional->evaluate_gga_fxc(
            rho, sigma, static_cast<int>(npts), frr_c, frs_c, fss_c);
        if (!vx || !vc || !fx || !fc)
            return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: GGA vxc/fxc evaluation failed.");
        if (inputs.exchange_functional->is_combined_exchange_correlation())
        {
            std::fill(vrho_c.begin(), vrho_c.end(), 0.0);
            std::fill(fs_c.begin(), fs_c.end(), 0.0);
            std::fill(frr_c.begin(), frr_c.end(), 0.0);
            std::fill(frs_c.begin(), frs_c.end(), 0.0);
            std::fill(fss_c.begin(), fss_c.end(), 0.0);
        }

        const Eigen::MatrixXd p_symmetric = 0.5 * (inputs.ground_density_ao + inputs.ground_density_ao.transpose());
        const Eigen::MatrixXd d_symmetric = 0.5 * (d_ao + d_ao.transpose());
        const auto spatial_hessian_column = [&ao, &hess](const Eigen::MatrixXd &density,
                                                Eigen::Index p, int q) -> Eigen::Vector3d
        {
            const Eigen::VectorXd phi = ao.values.row(p).transpose();
            const std::array<Eigen::VectorXd, 3> grad = {
                ao.grad_x.row(p).transpose(), ao.grad_y.row(p).transpose(), ao.grad_z.row(p).transpose()};
            const std::array<Eigen::VectorXd, 3> h_column = {
                q == 0 ? hess.h_xx.row(p).transpose() : (q == 1 ? hess.h_xy.row(p).transpose() : hess.h_xz.row(p).transpose()),
                q == 0 ? hess.h_xy.row(p).transpose() : (q == 1 ? hess.h_yy.row(p).transpose() : hess.h_yz.row(p).transpose()),
                q == 0 ? hess.h_xz.row(p).transpose() : (q == 1 ? hess.h_yz.row(p).transpose() : hess.h_zz.row(p).transpose())};
            const Eigen::VectorXd density_phi = density * phi;
            const Eigen::VectorXd density_grad_q = density * grad[static_cast<std::size_t>(q)];
            Eigen::Vector3d out;
            for (int axis = 0; axis < 3; ++axis)
                out(axis) = 2.0 * (h_column[static_cast<std::size_t>(axis)].dot(density_phi) +
                                   grad[static_cast<std::size_t>(axis)].dot(density_grad_q));
            return out;
        };

        DHEq33XCGGAMovingGridGradient out;
        out.becke_partition = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        out.point_translation = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const int owner = grid.owner(p);
            if (owner < 0 || owner >= static_cast<int>(mol.natoms))
                return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: grid owner is out of range.");
            const std::size_t pi = static_cast<std::size_t>(p);
            const Eigen::Vector3d grad_p{ground->total.grad_x(p), ground->total.grad_y(p), ground->total.grad_z(p)};
            const Eigen::Vector3d grad_d{difference->total.grad_x(p), difference->total.grad_y(p), difference->total.grad_z(p)};
            const double vrho = vrho_x[pi] + vrho_c[pi];
            const double fs = fs_x[pi] + fs_c[pi];
            const double frr = frr_x[pi] + frr_c[pi];
            const double frs = frs_x[pi] + frs_c[pi];
            const double fss = fss_x[pi] + fss_c[pi];
            const double rho_d = difference->total.rho(p);
            const double dot_pd = grad_p.dot(grad_d);
            const auto dpartition = becke_partition_owner_derivatives(grid, mol, p);
            if (!dpartition) return std::unexpected(dpartition.error());
            const double integrand = vrho * rho_d + 2.0 * fs * dot_pd;
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                    out.becke_partition(static_cast<Eigen::Index>(atom), q) +=
                        grid.atomic_weights(p) * (*dpartition)(static_cast<Eigen::Index>(atom), q) * integrand;
            for (int q = 0; q < 3; ++q)
            {
                const Eigen::Vector3d h_p_q = spatial_hessian_column(p_symmetric, p, q);
                const Eigen::Vector3d h_d_q = spatial_hessian_column(d_symmetric, p, q);
                const double dp_sigma = 2.0 * grad_p.dot(h_p_q);
                const double derivative =
                    (frr * grad_p(q) + frs * dp_sigma) * rho_d + vrho * grad_d(q) +
                    2.0 * (frs * grad_p(q) + fss * dp_sigma) * dot_pd +
                    2.0 * fs * (h_p_q.dot(grad_d) + grad_p.dot(h_d_q));
                out.point_translation(owner, q) += weight * derivative;
            }
        }
        out.total = out.becke_partition + out.point_translation;
        return out;
    }

    std::expected<DHEq33XCGGADifferenceAOGradient, std::string>
    build_dh_eq33_xc_gga_difference_ao_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.basis == nullptr || inputs.molecular_grid == nullptr ||
            inputs.ao_grid == nullptr || inputs.ao_hessian == nullptr || inputs.exchange_functional == nullptr ||
            inputs.correlation_functional == nullptr || !inputs.exchange_functional->is_gga_like() ||
            !inputs.correlation_functional->is_gga_like())
            return std::unexpected("DH Eq. 33 GGA D-AO XC-II: invalid GGA grid or functional inputs.");
        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        const auto &hess = *inputs.ao_hessian;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || hess.nbasis() != nao || hess.npoints() != ao.npoints() ||
            grid.points.rows() != ao.npoints() || grid.points.cols() != 4)
            return std::unexpected("DH Eq. 33 GGA D-AO XC-II: inconsistent density, AO grid, or Hessian dimensions.");
        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        if (!ground)
            return std::unexpected("DH Eq. 33 GGA D-AO XC-II: failed to evaluate ground density on grid.");
        const auto atoms_bf = atom_bf_lists(mol, *inputs.basis);
        if (!atoms_bf) return std::unexpected(atoms_bf.error());
        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts)), sigma(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            rho[static_cast<std::size_t>(p)] = ground->total.rho(p);
            sigma[static_cast<std::size_t>(p)] = ground->total.gradient_squared()(p);
        }
        std::vector<double> exc_x, exc_c, vrho_x, vrho_c, fs_x, fs_c;
        const auto vx = inputs.exchange_functional->evaluate_gga_exc_vxc(
            rho, sigma, static_cast<int>(npts), exc_x, vrho_x, fs_x);
        const auto vc = inputs.correlation_functional->evaluate_gga_exc_vxc(
            rho, sigma, static_cast<int>(npts), exc_c, vrho_c, fs_c);
        if (!vx || !vc)
            return std::unexpected("DH Eq. 33 GGA D-AO XC-II: GGA vxc evaluation failed.");
        if (inputs.exchange_functional->is_combined_exchange_correlation())
        {
            std::fill(vrho_c.begin(), vrho_c.end(), 0.0);
            std::fill(fs_c.begin(), fs_c.end(), 0.0);
        }
        DHEq33XCGGADifferenceAOGradient out;
        out.gradient = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const std::size_t pi = static_cast<std::size_t>(p);
            const double vrho = vrho_x[pi] + vrho_c[pi];
            const double fs = fs_x[pi] + fs_c[pi];
            const Eigen::Vector3d grad_p{ground->total.grad_x(p), ground->total.grad_y(p), ground->total.grad_z(p)};
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                {
                    const Eigen::Vector3d dgrad{
                        dg_axis_spin(d_ao, ao, hess, p, 0, static_cast<int>(atom), q, *atoms_bf),
                        dg_axis_spin(d_ao, ao, hess, p, 1, static_cast<int>(atom), q, *atoms_bf),
                        dg_axis_spin(d_ao, ao, hess, p, 2, static_cast<int>(atom), q, *atoms_bf)};
                    out.gradient(static_cast<Eigen::Index>(atom), q) += weight * (
                        vrho * drho_channel(d_ao, ao, p, static_cast<int>(atom), q, *atoms_bf) +
                        2.0 * fs * grad_p.dot(dgrad));
                }
        }
        return out;
    }

    std::expected<DHEq33CompleteXCIIGradient, std::string>
    build_dh_eq33_complete_xc_ii_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.exchange_functional == nullptr || inputs.correlation_functional == nullptr ||
            inputs.exchange_functional->is_lda_like() != inputs.correlation_functional->is_lda_like() ||
            inputs.exchange_functional->is_gga_like() != inputs.correlation_functional->is_gga_like())
            return std::unexpected("DH Eq. 33 complete XC-II: incompatible exchange/correlation functionals.");

        const auto p_side = build_dh_eq33_xc_fixed_density_gradient(relaxed, mo_coeff, inputs);
        if (!p_side) return std::unexpected(p_side.error());
        DHEq33CompleteXCIIGradient out;
        out.p_side_fixed = p_side->gradient;
        if (inputs.exchange_functional->is_lda_like())
        {
            const auto d_side = build_dh_eq33_xc_lda_difference_ao_gradient(relaxed, mo_coeff, inputs);
            const auto moving = build_dh_eq33_xc_lda_moving_grid_gradient(relaxed, mo_coeff, inputs);
            if (!d_side) return std::unexpected(d_side.error());
            if (!moving) return std::unexpected(moving.error());
            out.d_side_ao = d_side->gradient;
            out.becke_partition = moving->becke_partition;
            out.point_translation = moving->point_translation;
        }
        else if (inputs.exchange_functional->is_gga_like())
        {
            const auto d_side = build_dh_eq33_xc_gga_difference_ao_gradient(relaxed, mo_coeff, inputs);
            const auto moving = build_dh_eq33_xc_gga_moving_grid_gradient(relaxed, mo_coeff, inputs);
            if (!d_side) return std::unexpected(d_side.error());
            if (!moving) return std::unexpected(moving.error());
            out.d_side_ao = d_side->gradient;
            out.becke_partition = moving->becke_partition;
            out.point_translation = moving->point_translation;
        }
        else
            return std::unexpected("DH Eq. 33 complete XC-II: functional is neither LDA-like nor GGA-like.");
        if (out.p_side_fixed.rows() != out.d_side_ao.rows() ||
            out.p_side_fixed.cols() != out.d_side_ao.cols() ||
            out.p_side_fixed.rows() != out.becke_partition.rows() ||
            out.p_side_fixed.cols() != out.becke_partition.cols() ||
            out.p_side_fixed.rows() != out.point_translation.rows() ||
            out.p_side_fixed.cols() != out.point_translation.cols())
            return std::unexpected("DH Eq. 33 complete XC-II: component dimensions do not agree.");
        out.total = out.p_side_fixed + out.d_side_ao + out.becke_partition + out.point_translation;
        return out;
    }

    std::expected<DHKSFixedDensityFockDerivative, std::string>
    build_dh_ks_fixed_density_fock_derivative(
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density_ao,
        double exact_exchange_coefficient,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33NonXCDerivatives &derivatives,
        const DHEq33XCFixedDensityInputs &xc_inputs)
    {
        const Eigen::Index nao = ground_density_ao.rows();
        const std::size_t ncoord = derivatives.hamiltonian_ao.size();
        const std::size_t n4 = static_cast<std::size_t>(nao) * nao * nao * nao;
        if (nao == 0 || ground_density_ao.cols() != nao || mo_coeff.rows() != nao ||
            mo_coeff.cols() != nao || !std::isfinite(exact_exchange_coefficient) ||
            derivatives.overlap_ao.size() != ncoord || derivatives.eri_ao.size() != ncoord ||
            xc_inputs.ground_density_ao.rows() != nao || xc_inputs.ground_density_ao.cols() != nao ||
            (xc_inputs.ground_density_ao - ground_density_ao).cwiseAbs().maxCoeff() > 1e-12)
            return std::unexpected("DH fixed-density KS Fock derivative: inconsistent AO density, MO coefficients, or derivative inputs.");
        if ((ground_density_ao - ground_density_ao.transpose()).cwiseAbs().maxCoeff() > 1e-10)
            return std::unexpected("DH fixed-density KS Fock derivative: total AO density must be symmetric.");

        Eigen::FullPivLU<Eigen::MatrixXd> mo_lu(mo_coeff);
        if (!mo_lu.isInvertible())
            return std::unexpected("DH fixed-density KS Fock derivative: MO coefficient matrix is not invertible.");
        const Eigen::MatrixXd mo_inverse = mo_lu.inverse();

        DHKSFixedDensityFockDerivative out;
        out.hamiltonian.resize(ncoord);
        out.coulomb.resize(ncoord);
        out.exact_exchange.resize(ncoord);
        out.xc_geometry.resize(ncoord);
        out.total.resize(ncoord);
        for (std::size_t x = 0; x < ncoord; ++x)
        {
            if (derivatives.hamiltonian_ao[x].rows() != nao ||
                derivatives.hamiltonian_ao[x].cols() != nao ||
                derivatives.overlap_ao[x].rows() != nao ||
                derivatives.overlap_ao[x].cols() != nao || derivatives.eri_ao[x].size() != n4)
                return std::unexpected("DH fixed-density KS Fock derivative: derivative AO dimensions are inconsistent.");
            out.hamiltonian[x] = derivatives.hamiltonian_ao[x];
            out.coulomb[x] = Eigen::MatrixXd::Zero(nao, nao);
            out.exact_exchange[x] = Eigen::MatrixXd::Zero(nao, nao);
            const auto &eri_x = derivatives.eri_ao[x];
            for (Eigen::Index mu = 0; mu < nao; ++mu)
                for (Eigen::Index nu = 0; nu < nao; ++nu)
                    for (Eigen::Index ka = 0; ka < nao; ++ka)
                        for (Eigen::Index ta = 0; ta < nao; ++ta)
                        {
                            const double p = ground_density_ao(ka, ta);
                            out.coulomb[x](mu, nu) += p * eri_x[eri_idx(
                                static_cast<int>(mu), static_cast<int>(nu),
                                static_cast<int>(ka), static_cast<int>(ta), static_cast<int>(nao))];
                            out.exact_exchange[x](mu, nu) -= 0.5 * exact_exchange_coefficient * p *
                                eri_x[eri_idx(static_cast<int>(mu), static_cast<int>(ka),
                                              static_cast<int>(nu), static_cast<int>(ta), static_cast<int>(nao))];
                        }
            out.xc_geometry[x] = Eigen::MatrixXd::Zero(nao, nao);
        }

        // The complete XC-II scalar is linear in a symmetric test density D:
        // G_D^(x) = D : (V_XC[P])_geom^(x).  Probe its diagonal with E_mm
        // and each off-diagonal with 1/2(E_mn + E_nm), then recover the
        // symmetric AO matrix exactly without omitting the D-side AO or
        // moving-grid channels.
        for (Eigen::Index mu = 0; mu < nao; ++mu)
            for (Eigen::Index nu = 0; nu <= mu; ++nu)
            {
                Eigen::MatrixXd probe_ao = Eigen::MatrixXd::Zero(nao, nao);
                if (mu == nu)
                    probe_ao(mu, nu) = 1.0;
                else
                {
                    probe_ao(mu, nu) = 0.5;
                    probe_ao(nu, mu) = 0.5;
                }
                DHRelaxedDifferenceDensity probe;
                probe.symmetric_mo = mo_inverse * probe_ao * mo_inverse.transpose();
                probe.raw_mo = probe.symmetric_mo;
                if ((mo_coeff * probe.symmetric_mo * mo_coeff.transpose() - probe_ao).cwiseAbs().maxCoeff() > 1e-10)
                    return std::unexpected("DH fixed-density KS Fock derivative: AO probe transformation is not reversible.");
                const auto xc_gradient = build_dh_eq33_complete_xc_ii_gradient(probe, mo_coeff, xc_inputs);
                if (!xc_gradient)
                    return std::unexpected("DH fixed-density KS Fock derivative: complete XC geometry probe failed: " +
                                           xc_gradient.error());
                if (xc_gradient->total.rows() * xc_gradient->total.cols() != static_cast<Eigen::Index>(ncoord))
                    return std::unexpected("DH fixed-density KS Fock derivative: XC coordinate layout does not match derivative arrays.");
                for (std::size_t x = 0; x < ncoord; ++x)
                {
                    const Eigen::Index atom = static_cast<Eigen::Index>(x / 3);
                    const Eigen::Index cartesian = static_cast<Eigen::Index>(x % 3);
                    const double value = xc_gradient->total(atom, cartesian);
                    out.xc_geometry[x](mu, nu) = value;
                    out.xc_geometry[x](nu, mu) = value;
                }
            }
        for (std::size_t x = 0; x < ncoord; ++x)
            out.total[x] = out.hamiltonian[x] + out.coulomb[x] +
                out.exact_exchange[x] + out.xc_geometry[x];
        return out;
    }

    std::expected<DHGradientDriverContract, std::string>
    build_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs)
    {
        if (inputs.pt2_result == nullptr || !std::isfinite(inputs.c_pt2))
            return std::unexpected("DH gradient driver contract: missing PT2 result or invalid PT2 coefficient.");
        const auto &pt2 = *inputs.pt2_result;
        const int nocc = pt2.n_occ;
        const int nvirt = pt2.n_virt;
        const Eigen::Index nmo = static_cast<Eigen::Index>(nocc + nvirt);
        if (nocc <= 0 || nvirt <= 0 || inputs.mo_coeff.cols() != nmo ||
            inputs.mo_coeff.rows() == 0 || inputs.orbital_energies.size() != nmo ||
            inputs.z_ai.rows() != nvirt || inputs.z_ai.cols() != nocc ||
            inputs.mo_eri.size() != static_cast<std::size_t>(nmo) * nmo * nmo * nmo)
            return std::unexpected("DH gradient driver contract: inconsistent PT2, MO, ERI, or Z-vector dimensions.");
        if (inputs.xc_inputs.ground_density_ao.rows() != inputs.mo_coeff.rows() ||
            inputs.xc_inputs.ground_density_ao.cols() != inputs.mo_coeff.rows())
            return std::unexpected("DH gradient driver contract: ground AO density does not match MO coefficients.");

        DHGradientDriverContract out;
        const auto amplitudes = build_dh_pt2_amplitude_density(pt2, inputs.c_pt2);
        if (!amplitudes) return std::unexpected(amplitudes.error());
        out.amplitudes = *amplitudes;
        const auto eq41 = build_dh_eq41_response_density(
            out.amplitudes, inputs.mo_coeff, inputs.response_operator);
        if (!eq41) return std::unexpected(eq41.error());
        out.eq41_response = *eq41;
        const auto eq40 = build_dh_eq40_amplitude_rhs(
            pt2, out.amplitudes, inputs.mo_eri, inputs.c_pt2);
        if (!eq40) return std::unexpected(eq40.error());
        out.eq40_amplitude_rhs = *eq40;
        const Eigen::MatrixXd c_occ = inputs.mo_coeff.leftCols(nocc);
        const Eigen::MatrixXd c_virt = inputs.mo_coeff.rightCols(nvirt);
        const auto rhs = build_dh_lagrangian_rhs(
            out.eq41_response, c_occ, c_virt, out.eq40_amplitude_rhs, DHRHSConvention::LiteralEq47);
        if (!rhs) return std::unexpected(rhs.error());
        out.lagrangian_rhs = *rhs;
        const auto relaxed = build_dh_relaxed_difference_density(out.amplitudes, inputs.z_ai);
        if (!relaxed) return std::unexpected(relaxed.error());
        out.relaxed_density = *relaxed;
        const auto diagonal = build_dh_eq42_43_energy_weighted_density(
            out.relaxed_density, pt2, out.amplitudes, inputs.mo_coeff,
            inputs.orbital_energies, inputs.mo_eri, inputs.response_operator, false);
        if (!diagonal) return std::unexpected(diagonal.error());
        out.eq42_43_overlap = *diagonal;
        const auto off_diagonal = build_dh_eq44_45_energy_weighted_density(
            out.relaxed_density, pt2, out.amplitudes, inputs.orbital_energies, inputs.mo_eri, false);
        if (!off_diagonal) return std::unexpected(off_diagonal.error());
        out.eq44_45_overlap = *off_diagonal;
        // Always use the independently validated four-coefficient pair
        // metric derivative, for production as well as contract callers.
        {
            const auto literal_pair = build_dh_eq47_pair_metric_overlap_density(
                pt2, out.amplitudes, inputs.mo_eri);
            if (!literal_pair) return std::unexpected(literal_pair.error());

            // Preserve the independently derived Eq. (42) response and
            // orbital-energy pieces, then replace only the pair-amplitude
            // pieces.  `w_ai` belongs to Eq. (45), not to the Eq. (47) pair
            // metric derivative, and consequently remains untouched.
            out.eq42_43_overlap.amplitude_oo = literal_pair->w_oo;
            out.eq42_43_overlap.amplitude_vv = literal_pair->w_vv;
            out.eq42_43_overlap.w_oo = out.eq42_43_overlap.response_oo +
                out.eq42_43_overlap.orbital_oo + out.eq42_43_overlap.amplitude_oo;
            out.eq42_43_overlap.w_vv = out.eq42_43_overlap.orbital_vv +
                out.eq42_43_overlap.amplitude_vv;
            out.eq44_45_overlap.w_ia = literal_pair->w_ia;
        }
        const auto overlap = build_dh_overlap_density_adapter(out.eq42_43_overlap, out.eq44_45_overlap);
        if (!overlap) return std::unexpected(overlap.error());
        out.overlap_density = *overlap;
        const auto pair_density = build_dh_eq46_47_two_particle_density(
            out.relaxed_density, out.amplitudes, inputs.mo_coeff, inputs.xc_inputs.ground_density_ao,
            inputs.response_operator.exact_exchange);
        if (!pair_density) return std::unexpected(pair_density.error());
        out.two_particle_density = *pair_density;
        const auto non_xc = build_dh_eq33_non_xc_gradient(
            out.relaxed_density, out.overlap_density, out.two_particle_density,
            inputs.mo_coeff, inputs.non_xc_derivatives);
        if (!non_xc) return std::unexpected(non_xc.error());
        out.non_xc_gradient = *non_xc;
        const auto xc_ii = build_dh_eq33_complete_xc_ii_gradient(
            out.relaxed_density, inputs.mo_coeff, inputs.xc_inputs);
        if (!xc_ii) return std::unexpected(xc_ii.error());
        out.xc_ii_gradient = *xc_ii;
        return out;
    }

    std::expected<DHEq33PT2CorrectionGradient, std::string>
    build_dh_eq33_pt2_correction_gradient(const DHGradientDriverContract &contract)
    {
        const Eigen::MatrixXd &xc = contract.xc_ii_gradient.total;
        const Eigen::VectorXd &non_xc = contract.non_xc_gradient.total;
        if (xc.cols() != 3 || xc.rows() == 0 || non_xc.size() != 3 * xc.rows())
            return std::unexpected("DH Eq. 33 correction: non-XC coordinate vector and XC-II atom rows do not agree.");
        DHEq33PT2CorrectionGradient out;
        out.non_xc = Eigen::MatrixXd::Zero(xc.rows(), 3);
        for (Eigen::Index atom = 0; atom < xc.rows(); ++atom)
            for (int q = 0; q < 3; ++q)
                out.non_xc(atom, q) = non_xc(3 * atom + q);
        out.xc_ii = xc;
        out.total = out.non_xc + out.xc_ii;
        return out;
    }
}
