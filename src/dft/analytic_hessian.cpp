#include "analytic_hessian.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DFT::Driver
{
    namespace
    {
        // A combined exchange-correlation libxc entry (B3LYP, PBE0, HSE06,
        // ...) already carries its correlation in the exchange slot, so
        // summing a second fxc[correlation_functional] double-counts a
        // correlation fxc -- the same reason the KS-matrix build guards on
        // is_combined_exchange_correlation() ("configured correlation is
        // ignored"). Zero the correlation second-derivative (and, for GGA,
        // vsigma) arrays under that predicate. See
        // docs/DFT_ANALYTIC_FXC_HESSIAN.md.
        template <class... Vecs>
        void drop_correlation_if_combined(const XC::Functional &exchange_functional, Vecs &...vecs)
        {
            if (!exchange_functional.is_combined_exchange_correlation())
                return;
            (std::fill(vecs.begin(), vecs.end(), 0.0), ...);
        }
    } // namespace

    struct RKSXCKernelData
    {
        AOGridEvaluation ao;
        Eigen::VectorXd weights;
        DensityChannelOnGrid ground;
        std::vector<double> rr, rs, ss, vs;
        bool gga = false;
    };

    std::expected<RKSXCKernel,std::string> prepare_rks_xc_kernel(
        const MolecularGrid &grid, const AOGridEvaluation &ao,
        const Eigen::Ref<const Eigen::MatrixXd> &density,
        const XC::Functional &x, const XC::Functional &c)
    {
        const auto n=ao.npoints(),nb=ao.nbasis();
        if (n<0 || n>std::numeric_limits<int>::max() || nb<=0 || grid.points.rows()!=n ||
            grid.points.cols()!=4 || !grid.points.allFinite() || !ao.values.allFinite() ||
            density.rows()!=nb || density.cols()!=nb || !density.allFinite() ||
            x.spin()!=XC::Spin::Unpolarized || c.spin()!=XC::Spin::Unpolarized ||
            x.is_lda_like()!=c.is_lda_like() || x.is_gga_like()!=c.is_gga_like() ||
            (!x.is_lda_like() && !x.is_gga_like()))
            return std::unexpected("RKS XC cache: invalid grid, density, spin or functional family.");
        for (const auto *grad:{&ao.grad_x,&ao.grad_y,&ao.grad_z})
            if (grad->rows()!=n || grad->cols()!=nb || !grad->allFinite())
                return std::unexpected("RKS XC cache: invalid AO gradient dimensions or values.");
        auto ground=evaluate_density_on_grid(ao,density);
        if (!ground) return std::unexpected("RKS XC cache: "+ground.error());
        auto data=std::make_shared<RKSXCKernelData>();
        data->ao=ao; data->weights=grid.points.col(3);
        data->ground=std::move(ground->total); data->gga=x.is_gga_like();
        std::vector<double> rho(static_cast<std::size_t>(n)),sigma;
        for (Eigen::Index p=0;p<n;++p) rho[p]=data->ground.rho(p);
        std::vector<double> rr_c,rs_c,ss_c,vs_c;
        if (!data->gga)
        {
            auto a=x.evaluate_lda_fxc(rho,static_cast<int>(n),data->rr);
            if (!a) return std::unexpected(a.error());
            auto b=c.evaluate_lda_fxc(rho,static_cast<int>(n),rr_c);
            if (!b) return std::unexpected(b.error());
            drop_correlation_if_combined(x,rr_c);
        }
        else
        {
            const Eigen::VectorXd sigma_grid=data->ground.gradient_squared();
            sigma.resize(static_cast<std::size_t>(n));
            for (Eigen::Index p=0;p<n;++p) sigma[p]=sigma_grid(p);
            std::vector<double> exc,vrho;
            auto a=x.evaluate_gga_exc_vxc(rho,sigma,static_cast<int>(n),exc,vrho,data->vs);
            if (!a) return std::unexpected(a.error());
            auto b=c.evaluate_gga_exc_vxc(rho,sigma,static_cast<int>(n),exc,vrho,vs_c);
            if (!b) return std::unexpected(b.error());
            auto f=x.evaluate_gga_fxc(rho,sigma,static_cast<int>(n),data->rr,data->rs,data->ss);
            if (!f) return std::unexpected(f.error());
            auto g=c.evaluate_gga_fxc(rho,sigma,static_cast<int>(n),rr_c,rs_c,ss_c);
            if (!g) return std::unexpected(g.error());
            drop_correlation_if_combined(x,rr_c,rs_c,ss_c,vs_c);
        }
        const auto combine=[n](std::vector<double> &a,const std::vector<double> &b)
        {
            if (a.size()!=static_cast<std::size_t>(n) || b.size()!=a.size()) return false;
            for (std::size_t p=0;p<a.size();++p)
            { a[p]+=b[p]; if (!std::isfinite(a[p])) return false; }
            return true;
        };
        if (!combine(data->rr,rr_c) || (data->gga &&
            (!combine(data->rs,rs_c) || !combine(data->ss,ss_c) || !combine(data->vs,vs_c))))
            return std::unexpected("RKS XC cache: invalid functional derivative arrays.");
        return RKSXCKernel(std::move(data));
    }

    std::size_t RKSXCKernel::storage_bytes() const
    {
        if (!data_) return 0;
        const auto &d=*data_;
        return sizeof(double)*(d.ao.values.size()+d.ao.grad_x.size()+d.ao.grad_y.size()+d.ao.grad_z.size()+
            d.weights.size()+d.ground.rho.size()+d.ground.grad_x.size()+d.ground.grad_y.size()+d.ground.grad_z.size()+
            d.rr.size()+d.rs.size()+d.ss.size()+d.vs.size());
    }

    std::expected<Eigen::MatrixXd,std::string> RKSXCKernel::apply(
        const Eigen::Ref<const Eigen::MatrixXd> &density) const
    {
        if (!data_) return std::unexpected("RKS XC cache: moved-from kernel.");
        const auto &d=*data_;
        const auto &ao=d.ao;
        if (density.rows()!=ao.nbasis() || density.cols()!=ao.nbasis() || !density.allFinite())
            return std::unexpected("RKS XC cache: invalid trial density.");
        auto trial=evaluate_density_on_grid(ao,density);
        if (!trial) return std::unexpected(trial.error());
        Eigen::MatrixXd out=Eigen::MatrixXd::Zero(ao.nbasis(),ao.nbasis());
        for (Eigen::Index p=0;p<ao.npoints();++p)
        {
            const double weight=d.weights(p);
            if (weight==0.0) continue;
            const auto phi=ao.values.row(p).transpose();
            const double drho=trial->total.rho(p);
            if (!d.gga)
            {
                const double delta_vrho=d.rr[p]*drho;
                out.noalias()+=(weight*delta_vrho)*(phi*phi.transpose());
                continue;
            }
            const double dot=d.ground.grad_x(p)*trial->total.grad_x(p)+
                d.ground.grad_y(p)*trial->total.grad_y(p)+d.ground.grad_z(p)*trial->total.grad_z(p);
            const double delta_vrho=d.rr[p]*drho+2.0*d.rs[p]*dot;
            const double delta_vsigma=d.rs[p]*drho+2.0*d.ss[p]*dot;
            const Eigen::Vector3d grad{d.ground.grad_x(p),d.ground.grad_y(p),d.ground.grad_z(p)};
            const Eigen::Vector3d dg{trial->total.grad_x(p),trial->total.grad_y(p),trial->total.grad_z(p)};
            const Eigen::Vector3d term=2.0*delta_vsigma*grad+2.0*d.vs[p]*dg;
            const Eigen::VectorXd projected=term.x()*ao.grad_x.row(p).transpose()+
                term.y()*ao.grad_y.row(p).transpose()+term.z()*ao.grad_z.row(p).transpose();
            out.noalias()+=(weight*delta_vrho)*(phi*phi.transpose());
            out.noalias()+=weight*(phi*projected.transpose()+projected*phi.transpose());
        }
        if (!out.allFinite()) return std::unexpected("RKS XC cache: nonfinite action.");
        return out;
    }

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
            drop_correlation_if_combined(exchange_functional, v2rho2_c);

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
        // machinery -- see docs/DFT_ANALYTIC_FXC_HESSIAN.md, F3.3.4 and
        // D2.0's own commit for the record of that verification.
        std::vector<double> sigma_vec(static_cast<std::size_t>(npoints));
        {
            // gradient_squared() evaluates the entire grid: once, not per point.
            const Eigen::VectorXd ground_sigma = ground->total.gradient_squared();
            for (Eigen::Index p = 0; p < npoints; ++p)
                sigma_vec[static_cast<std::size_t>(p)] = ground_sigma(p);
        }

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
        // vsigma_c feeds the GGA T3 term (2*vsigma*delta_grad_rho), so it must
        // be dropped alongside the fxc_c arrays for a combined XC functional.
        drop_correlation_if_combined(exchange_functional, v2rho2_c, v2rhosigma_c, v2sigma2_c, vsigma_c);

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

    Eigen::VectorXd orbital_energy_difference_diagonal(
        const Eigen::Ref<const Eigen::VectorXd> &eps,
        int n_occ)
    {
        const int nbasis = static_cast<int>(eps.size());
        const int n_virt = nbasis - n_occ;
        Eigen::VectorXd diag(n_virt * n_occ);
        for (int a = 0; a < n_virt; ++a)
            for (int i = 0; i < n_occ; ++i)
                diag(a * n_occ + i) = eps(n_occ + a) - eps(i);
        return diag;
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    compute_analytic_xc_hessian_vector_product_polarized(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_beta_density,
        const Eigen::Ref<const Eigen::MatrixXd> &trial_alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &trial_beta_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional)
    {
        if (exchange_functional.is_lda_like() != correlation_functional.is_lda_like() ||
            exchange_functional.is_gga_like() != correlation_functional.is_gga_like())
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product_polarized: exchange and correlation "
                "functionals must be the same family (both LDA-like or both GGA-like)");

        auto ground = evaluate_density_on_grid(ao_grid, ground_alpha_density, ground_beta_density);
        if (!ground)
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product_polarized: " + ground.error());
        auto trial = evaluate_density_on_grid(ao_grid, trial_alpha_density, trial_beta_density);
        if (!trial)
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product_polarized: " + trial.error());

        const Eigen::Index npoints = ao_grid.npoints();
        const Eigen::Index nbasis = ao_grid.nbasis();

        if (exchange_functional.is_lda_like())
        {
            // Polarized LDA analogue of F3.1's formula: no gradient terms
            // at all (V_xc^sigma = vrho_sigma alone), so the only
            // cross-spin coupling is the single v2rho2_ab slot shared by
            // both channels' formulas.
            std::vector<double> rho_vec_lda(static_cast<std::size_t>(npoints) * 2);
            for (Eigen::Index p = 0; p < npoints; ++p)
            {
                const std::size_t pi = static_cast<std::size_t>(p);
                rho_vec_lda[2 * pi] = ground->alpha.rho(p);
                rho_vec_lda[2 * pi + 1] = ground->beta.rho(p);
            }

            std::vector<double> v2rho2_x, v2rho2_c;
            auto fxc_x = exchange_functional.evaluate_lda_fxc(rho_vec_lda, static_cast<int>(npoints), v2rho2_x);
            if (!fxc_x)
                return std::unexpected(
                    "compute_analytic_xc_hessian_vector_product_polarized: " + fxc_x.error());
            auto fxc_c = correlation_functional.evaluate_lda_fxc(rho_vec_lda, static_cast<int>(npoints), v2rho2_c);
            if (!fxc_c)
                return std::unexpected(
                    "compute_analytic_xc_hessian_vector_product_polarized: " + fxc_c.error());
            drop_correlation_if_combined(exchange_functional, v2rho2_c);

            Eigen::MatrixXd delta_v_xc_a = Eigen::MatrixXd::Zero(nbasis, nbasis);
            Eigen::MatrixXd delta_v_xc_b = Eigen::MatrixXd::Zero(nbasis, nbasis);

            for (Eigen::Index p = 0; p < npoints; ++p)
            {
                const double weight = molecular_grid.points(p, 3);
                if (weight == 0.0)
                    continue;
                const std::size_t pi = static_cast<std::size_t>(p);

                // v2rho2 layout: [aa, ab, bb] per point (F3.4.1's own
                // confirmed ordering, same as the GGA branch below).
                const double v2rho2_aa = v2rho2_x[3 * pi] + v2rho2_c[3 * pi];
                const double v2rho2_ab = v2rho2_x[3 * pi + 1] + v2rho2_c[3 * pi + 1];
                const double v2rho2_bb = v2rho2_x[3 * pi + 2] + v2rho2_c[3 * pi + 2];

                const double drho_a = trial->alpha.rho(p);
                const double drho_b = trial->beta.rho(p);

                const double delta_vrho_a = v2rho2_aa * drho_a + v2rho2_ab * drho_b;
                const double delta_vrho_b = v2rho2_ab * drho_a + v2rho2_bb * drho_b;

                const auto phi = ao_grid.values.row(p).transpose();
                delta_v_xc_a.noalias() += (weight * delta_vrho_a) * (phi * phi.transpose());
                delta_v_xc_b.noalias() += (weight * delta_vrho_b) * (phi * phi.transpose());
            }

            return std::make_pair(delta_v_xc_a, delta_v_xc_b);
        }

        if (!exchange_functional.is_gga_like())
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product_polarized: functional is neither "
                "LDA-like nor GGA-like");

        // Polarized libxc packing: rho = [rho_a, rho_b] per point,
        // sigma = [sigma_aa, sigma_ab, sigma_bb] per point (F3.4.1's own
        // confirmed ordering).
        std::vector<double> rho_vec(static_cast<std::size_t>(npoints) * 2);
        std::vector<double> sigma_vec(static_cast<std::size_t>(npoints) * 3);
        for (Eigen::Index p = 0; p < npoints; ++p)
        {
            const std::size_t pi = static_cast<std::size_t>(p);
            const double gax = ground->alpha.grad_x(p), gay = ground->alpha.grad_y(p), gaz = ground->alpha.grad_z(p);
            const double gbx = ground->beta.grad_x(p), gby = ground->beta.grad_y(p), gbz = ground->beta.grad_z(p);
            rho_vec[2 * pi] = ground->alpha.rho(p);
            rho_vec[2 * pi + 1] = ground->beta.rho(p);
            sigma_vec[3 * pi] = gax * gax + gay * gay + gaz * gaz;
            sigma_vec[3 * pi + 1] = gax * gbx + gay * gby + gaz * gbz;
            sigma_vec[3 * pi + 2] = gbx * gbx + gby * gby + gbz * gbz;
        }

        std::vector<double> exc_x, vrho_x, vsigma_x;
        std::vector<double> exc_c, vrho_c, vsigma_c;
        auto vxc_x = exchange_functional.evaluate_gga_exc_vxc(
            rho_vec, sigma_vec, static_cast<int>(npoints), exc_x, vrho_x, vsigma_x);
        if (!vxc_x)
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product_polarized: " + vxc_x.error());
        auto vxc_c = correlation_functional.evaluate_gga_exc_vxc(
            rho_vec, sigma_vec, static_cast<int>(npoints), exc_c, vrho_c, vsigma_c);
        if (!vxc_c)
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product_polarized: " + vxc_c.error());

        std::vector<double> v2rho2_x, v2rhosigma_x, v2sigma2_x;
        std::vector<double> v2rho2_c, v2rhosigma_c, v2sigma2_c;
        auto fxc_x = exchange_functional.evaluate_gga_fxc(
            rho_vec, sigma_vec, static_cast<int>(npoints), v2rho2_x, v2rhosigma_x, v2sigma2_x);
        if (!fxc_x)
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product_polarized: " + fxc_x.error());
        auto fxc_c = correlation_functional.evaluate_gga_fxc(
            rho_vec, sigma_vec, static_cast<int>(npoints), v2rho2_c, v2rhosigma_c, v2sigma2_c);
        if (!fxc_c)
            return std::unexpected(
                "compute_analytic_xc_hessian_vector_product_polarized: " + fxc_c.error());
        // vsigma_c feeds the polarized T2..T5 gradient-coupling terms, so it is
        // dropped alongside the fxc_c arrays for a combined XC functional.
        drop_correlation_if_combined(exchange_functional, v2rho2_c, v2rhosigma_c, v2sigma2_c, vsigma_c);

        Eigen::MatrixXd delta_v_xc_a = Eigen::MatrixXd::Zero(nbasis, nbasis);
        Eigen::MatrixXd delta_v_xc_b = Eigen::MatrixXd::Zero(nbasis, nbasis);

        for (Eigen::Index p = 0; p < npoints; ++p)
        {
            const double weight = molecular_grid.points(p, 3);
            if (weight == 0.0)
                continue;

            const std::size_t pi = static_cast<std::size_t>(p);

            // vsigma layout: [vsigma_aa, vsigma_ab, vsigma_bb] per point
            // (matches ks_matrix.cpp's own polarized convention and the
            // test file's vsigma0[0]/[1]/[2]).
            const double vsigma_aa0 = vsigma_x[3 * pi] + vsigma_c[3 * pi];
            const double vsigma_ab0 = vsigma_x[3 * pi + 1] + vsigma_c[3 * pi + 1];
            const double vsigma_bb0 = vsigma_x[3 * pi + 2] + vsigma_c[3 * pi + 2];

            // v2rho2: [aa, ab, bb]. v2rhosigma: [a-aa,a-ab,a-bb,b-aa,b-ab,b-bb].
            // v2sigma2: [aa-aa,aa-ab,aa-bb,ab-ab,ab-bb,bb-bb]. F3.4.1's own
            // confirmed layout.
            const double v2rho2_aa = v2rho2_x[3 * pi] + v2rho2_c[3 * pi];
            const double v2rho2_ab = v2rho2_x[3 * pi + 1] + v2rho2_c[3 * pi + 1];
            const double v2rho2_bb = v2rho2_x[3 * pi + 2] + v2rho2_c[3 * pi + 2];
            double v2rhosigma[6], v2sigma2[6];
            for (int k = 0; k < 6; ++k)
            {
                v2rhosigma[k] = v2rhosigma_x[6 * pi + static_cast<std::size_t>(k)] +
                                v2rhosigma_c[6 * pi + static_cast<std::size_t>(k)];
                v2sigma2[k] = v2sigma2_x[6 * pi + static_cast<std::size_t>(k)] +
                              v2sigma2_c[6 * pi + static_cast<std::size_t>(k)];
            }

            const Eigen::Vector3d grad_rho_a{ground->alpha.grad_x(p), ground->alpha.grad_y(p),
                                             ground->alpha.grad_z(p)};
            const Eigen::Vector3d grad_rho_b{ground->beta.grad_x(p), ground->beta.grad_y(p),
                                             ground->beta.grad_z(p)};
            const Eigen::Vector3d delta_grad_rho_a{trial->alpha.grad_x(p), trial->alpha.grad_y(p),
                                                   trial->alpha.grad_z(p)};
            const Eigen::Vector3d delta_grad_rho_b{trial->beta.grad_x(p), trial->beta.grad_y(p),
                                                   trial->beta.grad_z(p)};
            const double drho_a = trial->alpha.rho(p);
            const double drho_b = trial->beta.rho(p);

            const double dsigma_aa = 2.0 * grad_rho_a.dot(delta_grad_rho_a);
            const double dsigma_ab = grad_rho_b.dot(delta_grad_rho_a) + grad_rho_a.dot(delta_grad_rho_b);
            const double dsigma_bb = 2.0 * grad_rho_b.dot(delta_grad_rho_b);

            const auto phi = ao_grid.values.row(p).transpose();
            const auto project = [&](const Eigen::Vector3d &v) -> Eigen::VectorXd
            {
                return v.x() * ao_grid.grad_x.row(p).transpose() + v.y() * ao_grid.grad_y.row(p).transpose() +
                       v.z() * ao_grid.grad_z.row(p).transpose();
            };

            // ---- Alpha channel: T1..T5 (check_mixed's own formula) ----
            {
                const double delta_vrho_a = v2rho2_aa * drho_a + v2rho2_ab * drho_b +
                                             v2rhosigma[0] * dsigma_aa + v2rhosigma[1] * dsigma_ab +
                                             v2rhosigma[2] * dsigma_bb;
                const double delta_vsigma_aa = v2rhosigma[0] * drho_a + v2rhosigma[3] * drho_b +
                                                v2sigma2[0] * dsigma_aa + v2sigma2[1] * dsigma_ab +
                                                v2sigma2[2] * dsigma_bb;
                const double delta_vsigma_ab = v2rhosigma[1] * drho_a + v2rhosigma[4] * drho_b +
                                                v2sigma2[1] * dsigma_aa + v2sigma2[3] * dsigma_ab +
                                                v2sigma2[4] * dsigma_bb;

                // T2+T3 (self, grad_rho_a) + T4+T5 (cross, grad_rho_b).
                const Eigen::Vector3d gradient_term_self =
                    2.0 * delta_vsigma_aa * grad_rho_a + 2.0 * vsigma_aa0 * delta_grad_rho_a;
                const Eigen::Vector3d gradient_term_cross =
                    delta_vsigma_ab * grad_rho_b + vsigma_ab0 * delta_grad_rho_b;
                const Eigen::VectorXd projected = project(gradient_term_self + gradient_term_cross);

                delta_v_xc_a.noalias() += (weight * delta_vrho_a) * (phi * phi.transpose());
                delta_v_xc_a.noalias() += weight * (phi * projected.transpose() + projected * phi.transpose());
            }

            // ---- Beta channel: T1'..T5' (check_beta's own formula) ----
            {
                const double delta_vrho_b = v2rho2_bb * drho_b + v2rho2_ab * drho_a +
                                             v2rhosigma[5] * dsigma_bb + v2rhosigma[4] * dsigma_ab +
                                             v2rhosigma[3] * dsigma_aa;
                const double delta_vsigma_bb = v2rhosigma[5] * drho_b + v2rhosigma[2] * drho_a +
                                                v2sigma2[5] * dsigma_bb + v2sigma2[4] * dsigma_ab +
                                                v2sigma2[2] * dsigma_aa;
                const double delta_vsigma_ab = v2rhosigma[4] * drho_b + v2rhosigma[1] * drho_a +
                                                v2sigma2[4] * dsigma_bb + v2sigma2[3] * dsigma_ab +
                                                v2sigma2[1] * dsigma_aa;

                const Eigen::Vector3d gradient_term_self =
                    2.0 * delta_vsigma_bb * grad_rho_b + 2.0 * vsigma_bb0 * delta_grad_rho_b;
                const Eigen::Vector3d gradient_term_cross =
                    delta_vsigma_ab * grad_rho_a + vsigma_ab0 * delta_grad_rho_a;
                const Eigen::VectorXd projected = project(gradient_term_self + gradient_term_cross);

                delta_v_xc_b.noalias() += (weight * delta_vrho_b) * (phi * phi.transpose());
                delta_v_xc_b.noalias() += weight * (phi * projected.transpose() + projected * phi.transpose());
            }
        }

        return std::make_pair(delta_v_xc_a, delta_v_xc_b);
    }
} // namespace DFT::Driver
