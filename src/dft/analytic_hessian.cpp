#include "analytic_hessian.h"

namespace DFT::Driver
{
    std::expected<Eigen::MatrixXd, std::string> compute_analytic_xc_hessian_vector_product(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density,
        const Eigen::Ref<const Eigen::MatrixXd> &trial_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional)
    {
        if (exchange_functional.is_lda_like() != correlation_functional.is_lda_like() ||
            exchange_functional.is_gga_like() != correlation_functional.is_gga_like())
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product: exchange and correlation "
                "functionals must be the same family (both LDA-like or both GGA-like)");

        auto ground = evaluate_density_on_grid(ao_grid, ground_density);
        if (!ground)
            return std::unexpected("compute_analytic_xc_hessian_vector_product: " + ground.error());
        auto trial = evaluate_density_on_grid(ao_grid, trial_density);
        if (!trial)
            return std::unexpected("compute_analytic_xc_hessian_vector_product: " + trial.error());

        const Eigen::Index npoints = ao_grid.npoints();
        const Eigen::Index nbasis = ao_grid.nbasis();
        std::vector<double> rho_vec(static_cast<std::size_t>(npoints));
        for (Eigen::Index p = 0; p < npoints; ++p)
            rho_vec[static_cast<std::size_t>(p)] = ground->total.rho(p);

        Eigen::MatrixXd delta_v_xc_ao = Eigen::MatrixXd::Zero(nbasis, nbasis);

        if (exchange_functional.is_lda_like())
        {
            // F3.1's own verified formula: delta_V_xc = v2rho2*drho, projected
            // the same rank-1 way assemble_xc_matrix's LDA-only term is.
            std::vector<double> v2rho2_x, v2rho2_c;
            auto fxc_x = exchange_functional.evaluate_lda_fxc(rho_vec, static_cast<int>(npoints), v2rho2_x);
            if (!fxc_x)
                return std::unexpected("compute_analytic_xc_hessian_vector_product: " + fxc_x.error());
            auto fxc_c = correlation_functional.evaluate_lda_fxc(rho_vec, static_cast<int>(npoints), v2rho2_c);
            if (!fxc_c)
                return std::unexpected("compute_analytic_xc_hessian_vector_product: " + fxc_c.error());

            for (Eigen::Index p = 0; p < npoints; ++p)
            {
                const double weight = molecular_grid.points(p, 3);
                if (weight == 0.0)
                    continue;
                const std::size_t pi = static_cast<std::size_t>(p);
                const double v2rho2_total = v2rho2_x[pi] + v2rho2_c[pi];
                const double delta_vrho = v2rho2_total * trial->total.rho(p);
                const auto phi = ao_grid.values.row(p).transpose();
                delta_v_xc_ao.noalias() += (weight * delta_vrho) * (phi * phi.transpose());
            }
            return delta_v_xc_ao;
        }

        if (!exchange_functional.is_gga_like())
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product: functional is neither LDA-like nor GGA-like");

        // GGA path: F3.3.3's own verified T1+T2+T3 decomposition, ported
        // verbatim from the (now-deleted) F3.3.4 whole-molecule probe that
        // first proved this composes correctly with real AO-projection
        // machinery -- see docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md, F3.3.4 and
        // D2.0's own commit for the record of that verification.
        std::vector<double> sigma_vec(static_cast<std::size_t>(npoints));
        for (Eigen::Index p = 0; p < npoints; ++p)
            sigma_vec[static_cast<std::size_t>(p)] = ground->total.gradient_squared()(p);

        std::vector<double> exc_x, vrho_x, vsigma_x;
        std::vector<double> exc_c, vrho_c, vsigma_c;
        auto vxc_x = exchange_functional.evaluate_gga_exc_vxc(
            rho_vec, sigma_vec, static_cast<int>(npoints), exc_x, vrho_x, vsigma_x);
        if (!vxc_x)
            return std::unexpected("compute_analytic_xc_hessian_vector_product: " + vxc_x.error());
        auto vxc_c = correlation_functional.evaluate_gga_exc_vxc(
            rho_vec, sigma_vec, static_cast<int>(npoints), exc_c, vrho_c, vsigma_c);
        if (!vxc_c)
            return std::unexpected("compute_analytic_xc_hessian_vector_product: " + vxc_c.error());

        std::vector<double> v2rho2_x, v2rhosigma_x, v2sigma2_x;
        std::vector<double> v2rho2_c, v2rhosigma_c, v2sigma2_c;
        auto fxc_x = exchange_functional.evaluate_gga_fxc(
            rho_vec, sigma_vec, static_cast<int>(npoints), v2rho2_x, v2rhosigma_x, v2sigma2_x);
        if (!fxc_x)
            return std::unexpected("compute_analytic_xc_hessian_vector_product: " + fxc_x.error());
        auto fxc_c = correlation_functional.evaluate_gga_fxc(
            rho_vec, sigma_vec, static_cast<int>(npoints), v2rho2_c, v2rhosigma_c, v2sigma2_c);
        if (!fxc_c)
            return std::unexpected("compute_analytic_xc_hessian_vector_product: " + fxc_c.error());

        for (Eigen::Index p = 0; p < npoints; ++p)
        {
            const double weight = molecular_grid.points(p, 3);
            if (weight == 0.0)
                continue;

            const std::size_t pi = static_cast<std::size_t>(p);
            const double g_dot_dg = ground->total.grad_x(p) * trial->total.grad_x(p) +
                                     ground->total.grad_y(p) * trial->total.grad_y(p) +
                                     ground->total.grad_z(p) * trial->total.grad_z(p);
            const double drho = trial->total.rho(p);

            const double v2rho2_total = v2rho2_x[pi] + v2rho2_c[pi];
            const double v2rhosigma_total = v2rhosigma_x[pi] + v2rhosigma_c[pi];
            const double v2sigma2_total = v2sigma2_x[pi] + v2sigma2_c[pi];
            const double vsigma_total = vsigma_x[pi] + vsigma_c[pi];

            const double delta_vrho = v2rho2_total * drho + 2.0 * v2rhosigma_total * g_dot_dg;
            const double delta_vsigma = v2rhosigma_total * drho + 2.0 * v2sigma2_total * g_dot_dg;

            const Eigen::Vector3d grad_rho{ground->total.grad_x(p), ground->total.grad_y(p),
                                           ground->total.grad_z(p)};
            const Eigen::Vector3d delta_grad_rho{trial->total.grad_x(p), trial->total.grad_y(p),
                                                 trial->total.grad_z(p)};

            const Eigen::Vector3d delta_gradient_term =
                2.0 * delta_vsigma * grad_rho + 2.0 * vsigma_total * delta_grad_rho;

            const auto phi = ao_grid.values.row(p).transpose();
            // ks_matrix.cpp's gradient_projection has internal linkage;
            // inlined here rather than exposing it (same formula:
            // coefficient . (grad_x,grad_y,grad_z) row at this point).
            const Eigen::VectorXd projected = delta_gradient_term.x() * ao_grid.grad_x.row(p).transpose() +
                                               delta_gradient_term.y() * ao_grid.grad_y.row(p).transpose() +
                                               delta_gradient_term.z() * ao_grid.grad_z.row(p).transpose();

            delta_v_xc_ao.noalias() += (weight * delta_vrho) * (phi * phi.transpose());
            delta_v_xc_ao.noalias() += weight * (phi * projected.transpose() + projected * phi.transpose());
        }

        return delta_v_xc_ao;
    }
} // namespace DFT::Driver
