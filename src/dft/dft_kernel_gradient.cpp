#include "dft_kernel_gradient.h"

#include <algorithm>
#include <array>
#include <vector>

#include "dft_gradient.h"
#include "xc_grid.h"

namespace DFT::Gradient
{
    namespace
    {
        // v2rho2 ~ rho^(-2/3) for LDA exchange; libxc's own dens_threshold
        // clamps far below (~1e-32 for LDA_X). A grid point with rho_P that
        // small contributes nothing physical -- screen at a sane floor so
        // the term is well-defined regardless of libxc's threshold.
        constexpr double kRhoFloor = 1e-8;

        // ---- LDA branch --------------------------------------------------
        //   integral  rho_P^(x)(r) * [ (v2rho2_x + v2rho2_c)(rho_P;r) * rho_D(r) ]  dr
        // = d/dR (via rho_P^(x), the basis-function derivative of the SCF
        //   density at a fixed point) of  sum_munu D_munu <mu|V_xc[rho_P]|nu>'s
        //   (delta f / delta rho) piece.  Second derivative only. Plain grid
        //   integral -- no moving-grid correction.
        std::expected<Eigen::MatrixXd, std::string> lda_branch(
            const HartreeFock::Molecule &mol,
            const MolecularGrid &grid,
            const AOGridEvaluation &ao,
            const std::vector<std::vector<int>> &atoms_bf,
            const Eigen::MatrixXd &P_sym,
            const DFT::DensityOnGrid &ground,
            const DFT::DensityOnGrid &relaxed,
            const XC::Functional &x_func,
            const XC::Functional &c_func)
        {
            const Eigen::Index npts = ao.npoints();
            std::vector<double> rho_vec(static_cast<std::size_t>(npts));
            for (Eigen::Index p = 0; p < npts; ++p)
                rho_vec[static_cast<std::size_t>(p)] = ground.total.rho(p);

            std::vector<double> f2x, f2c;
            if (auto r = x_func.evaluate_lda_fxc(rho_vec, static_cast<int>(npts), f2x); !r)
                return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());
            if (auto r = c_func.evaluate_lda_fxc(rho_vec, static_cast<int>(npts), f2c); !r)
                return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());
            if (x_func.is_combined_exchange_correlation())
                std::fill(f2c.begin(), f2c.end(), 0.0);

            Eigen::MatrixXd grad =
                Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);

            for (Eigen::Index ip = 0; ip < npts; ++ip)
            {
                const double w = grid.points(ip, 3);
                if (w == 0.0 || ground.total.rho(ip) < kRhoFloor)
                    continue;
                const std::size_t pi = static_cast<std::size_t>(ip);
                const double c = w * (f2x[pi] + f2c[pi]) * relaxed.total.rho(ip);
                if (c == 0.0)
                    continue;
                for (int q = 0; q < 3; ++q)
                    for (std::size_t A = 0; A < mol.natoms; ++A)
                        grad(static_cast<Eigen::Index>(A), q) +=
                            c * drho_channel(P_sym, ao, ip, static_cast<int>(A), q, atoms_bf);
            }
            return grad;
        }

        // ---- GGA branch ------------------------------------------------------
        //
        // d/dR of  Phi_XC = sum_munu D_munu <mu|V_xc[rho_P]|nu>
        //                 = integral w * { (df/drho)*rho_D + 2*(df/dgamma)*(grad_rho_P . grad_rho_D) } dr
        // where <mu|V_xc|nu> is the SCF operator's XC part (Eq. 10, FIRST
        // functional derivatives). The response part (rho_P, grad_rho_P move
        // via the basis-function derivative; rho_D, grad_rho_D frozen) needs
        // the SECOND functional derivative:
        //
        //   integrand(A,q) =
        //       [ v2rho2*rx + 2*v2rhosigma*(g.gx) ] * rho_D
        //     + 2*[ v2rhosigma*rx + 2*v2sigma2*(g.gx) ] * (g.grad_rho_D)
        //     + 2*vsigma * (gx . grad_rho_D)
        //
        //   rx = rho_P^(x)          (drho_channel)
        //   gx = grad_rho_P^(x)     (dg_axis_spin, per axis)
        //   g  = grad_rho_P         (SCF gradient, frozen coefficient here)
        //
        // First line: the (df/drho) response (Term 1, rho + gamma branches).
        // Second line: the (df/dgamma) response inside the 2*(df/dgamma)*g
        //   factor. Third line (Term 2): the explicit grad_rho_P factor in
        //   that same 2*(df/dgamma)*grad_rho_P . grad_rho_D piece.
        // All second derivative -- no v3.
        std::expected<Eigen::MatrixXd, std::string> gga_branch(
            const HartreeFock::Molecule &mol,
            const MolecularGrid &grid,
            const AOGridEvaluation &ao,
            const AOGridHessian &hess,
            const std::vector<std::vector<int>> &atoms_bf,
            const Eigen::MatrixXd &P_sym,
            const DFT::DensityOnGrid &ground,
            const DFT::DensityOnGrid &relaxed,
            const XC::Functional &x_func,
            const XC::Functional &c_func)
        {
            const Eigen::Index npts = ao.npoints();
            std::vector<double> rho_vec(static_cast<std::size_t>(npts));
            std::vector<double> sigma_vec(static_cast<std::size_t>(npts));
            for (Eigen::Index p = 0; p < npts; ++p)
            {
                rho_vec[static_cast<std::size_t>(p)] = ground.total.rho(p);
                sigma_vec[static_cast<std::size_t>(p)] =
                    ground.total.grad_x(p) * ground.total.grad_x(p) +
                    ground.total.grad_y(p) * ground.total.grad_y(p) +
                    ground.total.grad_z(p) * ground.total.grad_z(p);
            }

            std::vector<double> exc_x, vrho_x, vsigma_x, exc_c, vrho_c, vsigma_c;
            if (auto r = x_func.evaluate_gga_exc_vxc(rho_vec, sigma_vec, static_cast<int>(npts),
                                                    exc_x, vrho_x, vsigma_x);
                !r)
                return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());
            if (auto r = c_func.evaluate_gga_exc_vxc(rho_vec, sigma_vec, static_cast<int>(npts),
                                                    exc_c, vrho_c, vsigma_c);
                !r)
                return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());

            std::vector<double> v2rho2_x, v2rhosigma_x, v2sigma2_x;
            std::vector<double> v2rho2_c, v2rhosigma_c, v2sigma2_c;
            if (auto r = x_func.evaluate_gga_fxc(rho_vec, sigma_vec, static_cast<int>(npts),
                                                v2rho2_x, v2rhosigma_x, v2sigma2_x);
                !r)
                return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());
            if (auto r = c_func.evaluate_gga_fxc(rho_vec, sigma_vec, static_cast<int>(npts),
                                                v2rho2_c, v2rhosigma_c, v2sigma2_c);
                !r)
                return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());

            if (x_func.is_combined_exchange_correlation())
            {
                for (auto *v : {&vsigma_c, &v2rho2_c, &v2rhosigma_c, &v2sigma2_c})
                    std::fill(v->begin(), v->end(), 0.0);
            }

            Eigen::MatrixXd grad =
                Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);

            for (Eigen::Index ip = 0; ip < npts; ++ip)
            {
                const double w = grid.points(ip, 3);
                if (w == 0.0 || ground.total.rho(ip) < kRhoFloor)
                    continue;
                const std::size_t pi = static_cast<std::size_t>(ip);

                const double v2rho2 = v2rho2_x[pi] + v2rho2_c[pi];
                const double v2rhosigma = v2rhosigma_x[pi] + v2rhosigma_c[pi];
                const double v2sigma2 = v2sigma2_x[pi] + v2sigma2_c[pi];
                const double vsigma = vsigma_x[pi] + vsigma_c[pi];

                const std::array<double, 3> g = {
                    ground.total.grad_x(ip), ground.total.grad_y(ip), ground.total.grad_z(ip)};
                const std::array<double, 3> grad_d = {
                    relaxed.total.grad_x(ip), relaxed.total.grad_y(ip), relaxed.total.grad_z(ip)};
                const double d = relaxed.total.rho(ip);
                const double g_dot_gradd = g[0] * grad_d[0] + g[1] * grad_d[1] + g[2] * grad_d[2];

                for (int q = 0; q < 3; ++q)
                {
                    for (std::size_t A = 0; A < mol.natoms; ++A)
                    {
                        const double rx =
                            drho_channel(P_sym, ao, ip, static_cast<int>(A), q, atoms_bf);
                        const std::array<double, 3> gx = {
                            dg_axis_spin(P_sym, ao, hess, ip, 0, static_cast<int>(A), q, atoms_bf),
                            dg_axis_spin(P_sym, ao, hess, ip, 1, static_cast<int>(A), q, atoms_bf),
                            dg_axis_spin(P_sym, ao, hess, ip, 2, static_cast<int>(A), q, atoms_bf)};
                        const double g_dot_gx = g[0] * gx[0] + g[1] * gx[1] + g[2] * gx[2];
                        const double gx_dot_gradd = gx[0] * grad_d[0] + gx[1] * grad_d[1] + gx[2] * grad_d[2];

                        const double d_dfdrho = v2rho2 * rx + 2.0 * v2rhosigma * g_dot_gx;
                        const double d_dfdgamma = v2rhosigma * rx + 2.0 * v2sigma2 * g_dot_gx;

                        const double integrand =
                            d_dfdrho * d +
                            2.0 * d_dfdgamma * g_dot_gradd +
                            2.0 * vsigma * gx_dot_gradd;

                        grad(static_cast<Eigen::Index>(A), q) += w * integrand;
                    }
                }
            }
            return grad;
        }
    } // namespace

    std::expected<Eigen::MatrixXd, std::string>
    compute_dh_xc_pt2_gradient(
        const HartreeFock::Molecule &mol,
        const HartreeFock::Basis &basis,
        const MolecularGrid &grid,
        const AOGridEvaluation &ao,
        const AOGridHessian &hess,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density_restricted,
        const Eigen::Ref<const Eigen::MatrixXd> &relaxed_density_restricted,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional)
    {
        if (exchange_functional.is_lda_like() != correlation_functional.is_lda_like() ||
            exchange_functional.is_gga_like() != correlation_functional.is_gga_like())
            return std::unexpected(
                "compute_dh_xc_pt2_gradient: exchange and correlation functionals "
                "must be the same family");

        if (hess.npoints() != ao.npoints() || hess.nbasis() != ao.nbasis())
            return std::unexpected("compute_dh_xc_pt2_gradient: AO Hessian dimensions do not match AO grid");
        if (grid.points.rows() != ao.npoints())
            return std::unexpected("compute_dh_xc_pt2_gradient: molecular grid point count mismatch");

        const Eigen::Index nb = ao.nbasis();
        if (ground_density_restricted.rows() != nb || ground_density_restricted.cols() != nb ||
            relaxed_density_restricted.rows() != nb || relaxed_density_restricted.cols() != nb)
            return std::unexpected("compute_dh_xc_pt2_gradient: density dimension mismatch");

        if (!exchange_functional.is_lda_like() && !exchange_functional.is_gga_like())
            return std::unexpected(
                "compute_dh_xc_pt2_gradient: functional is neither LDA-like nor GGA-like");

        auto atoms_bf_res = atom_bf_lists(mol, basis);
        if (!atoms_bf_res)
            return std::unexpected(atoms_bf_res.error());
        const auto &atoms_bf = *atoms_bf_res;

        auto ground = evaluate_density_on_grid(ao, ground_density_restricted);
        if (!ground)
            return std::unexpected("compute_dh_xc_pt2_gradient: " + ground.error());
        auto relaxed = evaluate_density_on_grid(ao, relaxed_density_restricted);
        if (!relaxed)
            return std::unexpected("compute_dh_xc_pt2_gradient: " + relaxed.error());

        const Eigen::MatrixXd P_sym =
            0.5 * (ground_density_restricted + ground_density_restricted.transpose());

        if (exchange_functional.is_lda_like())
            return lda_branch(mol, grid, ao, atoms_bf, P_sym, *ground, *relaxed,
                              exchange_functional, correlation_functional);

        return gga_branch(mol, grid, ao, hess, atoms_bf, P_sym, *ground, *relaxed,
                          exchange_functional, correlation_functional);
    }

} // namespace DFT::Gradient
