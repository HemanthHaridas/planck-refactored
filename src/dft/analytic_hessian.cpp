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
        // The XC kernel action is called repeatedly by the DH Z-vector solve.
        // Keep its temporary storage O(B*N_AO), rather than materializing a
        // grid-sized weighted AO panel, and reduce point blocks in increasing
        // grid order.  This is algebraically the scalar rank-one projection
        // below, but lets Eigen execute each panel contraction as GEMM.
        constexpr Eigen::Index kPointBlock = 128;
        const Eigen::Index npoints=ao.npoints();
        const Eigen::Index nbasis=ao.nbasis();
        Eigen::MatrixXd out=Eigen::MatrixXd::Zero(nbasis,nbasis);
        for (Eigen::Index first=0;first<npoints;first+=kPointBlock)
        {
            const Eigen::Index count=std::min(kPointBlock,npoints-first);
            const auto phi=ao.values.middleRows(first,count);
            Eigen::VectorXd coefficient=Eigen::VectorXd::Zero(count);
            if (!d.gga)
            {
                for (Eigen::Index local=0;local<count;++local)
                {
                    const Eigen::Index p=first+local;
                    coefficient(local)=d.weights(p)*d.rr[static_cast<std::size_t>(p)]*
                        trial->total.rho(p);
                }
                const Eigen::MatrixXd weighted_phi=phi.array().colwise()*coefficient.array();
                out.noalias()+=phi.transpose()*weighted_phi;
                continue;
            }
            Eigen::VectorXd term_x(count),term_y(count),term_z(count);
            for (Eigen::Index local=0;local<count;++local)
            {
                const Eigen::Index p=first+local;
                const std::size_t index=static_cast<std::size_t>(p);
                const double drho=trial->total.rho(p);
                const double dot=d.ground.grad_x(p)*trial->total.grad_x(p)+
                    d.ground.grad_y(p)*trial->total.grad_y(p)+d.ground.grad_z(p)*trial->total.grad_z(p);
                const double delta_vrho=d.rr[index]*drho+2.0*d.rs[index]*dot;
                const double delta_vsigma=d.rs[index]*drho+2.0*d.ss[index]*dot;
                coefficient(local)=d.weights(p)*delta_vrho;
                term_x(local)=d.weights(p)*(2.0*delta_vsigma*d.ground.grad_x(p)+
                    2.0*d.vs[index]*trial->total.grad_x(p));
                term_y(local)=d.weights(p)*(2.0*delta_vsigma*d.ground.grad_y(p)+
                    2.0*d.vs[index]*trial->total.grad_y(p));
                term_z(local)=d.weights(p)*(2.0*delta_vsigma*d.ground.grad_z(p)+
                    2.0*d.vs[index]*trial->total.grad_z(p));
            }
            const Eigen::MatrixXd weighted_phi=phi.array().colwise()*coefficient.array();
            out.noalias()+=phi.transpose()*weighted_phi;
            const Eigen::MatrixXd projected=
                ao.grad_x.middleRows(first,count).array().colwise()*term_x.array()+
                ao.grad_y.middleRows(first,count).array().colwise()*term_y.array()+
                ao.grad_z.middleRows(first,count).array().colwise()*term_z.array();
            out.noalias()+=phi.transpose()*projected+projected.transpose()*phi;
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

    struct UKSXCKernelData
    {
        AOGridEvaluation ao;
        Eigen::VectorXd weights;
        DensityChannelOnGrid ground_a, ground_b;
        // Per point: rr = v2rho2 [aa,ab,bb]; rs = v2rhosigma [a-aa,a-ab,a-bb,
        // b-aa,b-ab,b-bb]; ss = v2sigma2 [aa-aa,aa-ab,aa-bb,ab-ab,ab-bb,bb-bb];
        // vs = vsigma [aa,ab,bb] (Libxc's polarized layout, x + c summed).
        std::vector<double> rr, rs, ss, vs;
        bool gga = false;
    };

    std::expected<UKSXCKernel,std::string> prepare_uks_xc_kernel(
        const MolecularGrid &grid, const AOGridEvaluation &ao,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_alpha,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_beta,
        const XC::Functional &x, const XC::Functional &c)
    {
        const auto n = ao.npoints(), nb = ao.nbasis();
        if (x.is_lda_like() != c.is_lda_like() || x.is_gga_like() != c.is_gga_like() ||
            (!x.is_lda_like() && !x.is_gga_like()))
            return std::unexpected("UKS XC kernel: exchange and correlation must both be LDA-like or both "
                                   "GGA-like");
        if (n < 0 || n > std::numeric_limits<int>::max() || nb <= 0 || grid.points.rows() != n ||
            ground_alpha.rows() != nb || ground_beta.rows() != nb)
            return std::unexpected("UKS XC kernel: grid, AO and density dimensions disagree.");
        auto ground = evaluate_density_on_grid(ao, ground_alpha, ground_beta);
        if (!ground)
            return std::unexpected("UKS XC kernel: " + ground.error());
        auto data = std::make_shared<UKSXCKernelData>();
        data->ao = ao;
        data->weights = grid.points.col(3);
        data->ground_a = std::move(ground->alpha);
        data->ground_b = std::move(ground->beta);
        data->gga = x.is_gga_like();

        const std::size_t np = static_cast<std::size_t>(n);
        std::vector<double> rho(2 * np), sigma;
        for (std::size_t p = 0; p < np; ++p)
        {
            rho[2 * p] = data->ground_a.rho(static_cast<Eigen::Index>(p));
            rho[2 * p + 1] = data->ground_b.rho(static_cast<Eigen::Index>(p));
        }
        std::vector<double> rr_c, rs_c, ss_c, vs_c;
        if (!data->gga)
        {
            auto fx = x.evaluate_lda_fxc(rho, static_cast<int>(n), data->rr);
            if (!fx) return std::unexpected("UKS XC kernel: " + fx.error());
            auto fc = c.evaluate_lda_fxc(rho, static_cast<int>(n), rr_c);
            if (!fc) return std::unexpected("UKS XC kernel: " + fc.error());
            drop_correlation_if_combined(x, rr_c);
        }
        else
        {
            sigma.resize(3 * np);
            const auto &ga = data->ground_a;
            const auto &gb = data->ground_b;
            for (std::size_t p = 0; p < np; ++p)
            {
                const auto i = static_cast<Eigen::Index>(p);
                sigma[3 * p] = ga.grad_x(i) * ga.grad_x(i) + ga.grad_y(i) * ga.grad_y(i) + ga.grad_z(i) * ga.grad_z(i);
                sigma[3 * p + 1] = ga.grad_x(i) * gb.grad_x(i) + ga.grad_y(i) * gb.grad_y(i) + ga.grad_z(i) * gb.grad_z(i);
                sigma[3 * p + 2] = gb.grad_x(i) * gb.grad_x(i) + gb.grad_y(i) * gb.grad_y(i) + gb.grad_z(i) * gb.grad_z(i);
            }
            std::vector<double> exc, vrho;
            auto vx = x.evaluate_gga_exc_vxc(rho, sigma, static_cast<int>(n), exc, vrho, data->vs);
            if (!vx) return std::unexpected("UKS XC kernel: " + vx.error());
            auto vc = c.evaluate_gga_exc_vxc(rho, sigma, static_cast<int>(n), exc, vrho, vs_c);
            if (!vc) return std::unexpected("UKS XC kernel: " + vc.error());
            auto fx = x.evaluate_gga_fxc(rho, sigma, static_cast<int>(n), data->rr, data->rs, data->ss);
            if (!fx) return std::unexpected("UKS XC kernel: " + fx.error());
            auto fc = c.evaluate_gga_fxc(rho, sigma, static_cast<int>(n), rr_c, rs_c, ss_c);
            if (!fc) return std::unexpected("UKS XC kernel: " + fc.error());
            // vsigma_c feeds the polarized gradient-coupling terms, so it is
            // dropped with the fxc_c arrays for a combined XC functional.
            drop_correlation_if_combined(x, rr_c, rs_c, ss_c, vs_c);
        }
        const auto combine = [](std::vector<double> &a, const std::vector<double> &b)
        {
            if (b.size() != a.size()) return false;
            for (std::size_t k = 0; k < a.size(); ++k)
                a[k] += b[k];
            return true;
        };
        if (data->rr.size() != 3 * np || !combine(data->rr, rr_c) ||
            (data->gga && (data->rs.size() != 6 * np || data->ss.size() != 6 * np || data->vs.size() != 3 * np ||
                           !combine(data->rs, rs_c) || !combine(data->ss, ss_c) || !combine(data->vs, vs_c))))
            return std::unexpected("UKS XC kernel: invalid functional derivative arrays.");
        return UKSXCKernel(std::move(data));
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string> UKSXCKernel::apply(
        const Eigen::Ref<const Eigen::MatrixXd> &trial_alpha, const Eigen::Ref<const Eigen::MatrixXd> &trial_beta) const
    {
        if (!data_) return std::unexpected("UKS XC kernel: moved-from kernel.");
        const auto &d = *data_;
        const auto &ao = d.ao;
        const Eigen::Index nbasis = ao.nbasis(), npoints = ao.npoints();
        if (trial_alpha.rows() != nbasis || trial_beta.rows() != nbasis)
            return std::unexpected("UKS XC kernel: invalid trial density.");
        auto trial = evaluate_density_on_grid(ao, trial_alpha, trial_beta);
        if (!trial) return std::unexpected("UKS XC kernel: " + trial.error());
        const auto &ta = trial->alpha;
        const auto &tb = trial->beta;
        const auto &ga = d.ground_a;
        const auto &gb = d.ground_b;

        // Point blocks contracted as GEMM, reduced in increasing grid order,
        // as RKSXCKernel::apply does.
        constexpr Eigen::Index kPointBlock = 128;
        Eigen::MatrixXd out_a = Eigen::MatrixXd::Zero(nbasis, nbasis);
        Eigen::MatrixXd out_b = Eigen::MatrixXd::Zero(nbasis, nbasis);
        for (Eigen::Index first = 0; first < npoints; first += kPointBlock)
        {
            const Eigen::Index count = std::min(kPointBlock, npoints - first);
            const auto phi = ao.values.middleRows(first, count);
            Eigen::VectorXd ca(count), cb(count);
            Eigen::MatrixXd tva(count, 3), tvb(count, 3); // gradient-coupling vectors per point
            for (Eigen::Index local = 0; local < count; ++local)
            {
                const Eigen::Index p = first + local;
                const std::size_t i = static_cast<std::size_t>(p);
                const double w = d.weights(p);
                const double dra = ta.rho(p), drb = tb.rho(p);
                const double rr_aa = d.rr[3 * i], rr_ab = d.rr[3 * i + 1], rr_bb = d.rr[3 * i + 2];
                if (!d.gga)
                {
                    ca(local) = w * (rr_aa * dra + rr_ab * drb);
                    cb(local) = w * (rr_ab * dra + rr_bb * drb);
                    continue;
                }
                const Eigen::Vector3d gra{ga.grad_x(p), ga.grad_y(p), ga.grad_z(p)};
                const Eigen::Vector3d grb{gb.grad_x(p), gb.grad_y(p), gb.grad_z(p)};
                const Eigen::Vector3d dga{ta.grad_x(p), ta.grad_y(p), ta.grad_z(p)};
                const Eigen::Vector3d dgb{tb.grad_x(p), tb.grad_y(p), tb.grad_z(p)};
                const double *rs = &d.rs[6 * i], *ss = &d.ss[6 * i], *vs = &d.vs[3 * i];
                const double dsaa = 2.0 * gra.dot(dga);
                const double dsab = grb.dot(dga) + gra.dot(dgb);
                const double dsbb = 2.0 * grb.dot(dgb);

                const double dvrho_a = rr_aa * dra + rr_ab * drb + rs[0] * dsaa + rs[1] * dsab + rs[2] * dsbb;
                const double dvsaa = rs[0] * dra + rs[3] * drb + ss[0] * dsaa + ss[1] * dsab + ss[2] * dsbb;
                const double dvsab_a = rs[1] * dra + rs[4] * drb + ss[1] * dsaa + ss[3] * dsab + ss[4] * dsbb;
                ca(local) = w * dvrho_a;
                tva.row(local) = w * (2.0 * dvsaa * gra + 2.0 * vs[0] * dga + dvsab_a * grb + vs[1] * dgb).transpose();

                const double dvrho_b = rr_bb * drb + rr_ab * dra + rs[5] * dsbb + rs[4] * dsab + rs[3] * dsaa;
                const double dvsbb = rs[5] * drb + rs[2] * dra + ss[5] * dsbb + ss[4] * dsab + ss[2] * dsaa;
                const double dvsab_b = rs[4] * drb + rs[1] * dra + ss[4] * dsbb + ss[3] * dsab + ss[1] * dsaa;
                cb(local) = w * dvrho_b;
                tvb.row(local) = w * (2.0 * dvsbb * grb + 2.0 * vs[2] * dgb + dvsab_b * gra + vs[1] * dga).transpose();
            }
            out_a.noalias() += phi.transpose() * (phi.array().colwise() * ca.array()).matrix();
            out_b.noalias() += phi.transpose() * (phi.array().colwise() * cb.array()).matrix();
            if (!d.gga)
                continue;
            const auto project = [&](const Eigen::MatrixXd &t) -> Eigen::MatrixXd
            {
                return ao.grad_x.middleRows(first, count).array().colwise() * t.col(0).array() +
                       ao.grad_y.middleRows(first, count).array().colwise() * t.col(1).array() +
                       ao.grad_z.middleRows(first, count).array().colwise() * t.col(2).array();
            };
            const Eigen::MatrixXd pa = project(tva), pb = project(tvb);
            out_a.noalias() += phi.transpose() * pa + pa.transpose() * phi;
            out_b.noalias() += phi.transpose() * pb + pb.transpose() * phi;
        }
        if (!out_a.allFinite() || !out_b.allFinite())
            return std::unexpected("UKS XC kernel: nonfinite action.");
        return std::make_pair(std::move(out_a), std::move(out_b));
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
        auto kernel = prepare_uks_xc_kernel(molecular_grid, ao_grid, ground_alpha_density, ground_beta_density,
                                            exchange_functional, correlation_functional);
        if (!kernel)
            return std::unexpected("compute_analytic_xc_hessian_vector_product_polarized: " + kernel.error());
        return kernel->apply(trial_alpha_density, trial_beta_density);
    }
} // namespace DFT::Driver
