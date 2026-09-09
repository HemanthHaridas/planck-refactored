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

        // Per-point XC quantities at the GROUND (SCF) density: first and
        // second functional derivatives, spin-summed / combined-XC-guarded.
        struct XcArrays
        {
            std::vector<double> vrho, vsigma;                       // df/drho, df/dsigma
            std::vector<double> v2rho2, v2rhosigma, v2sigma2;       // second derivatives
            bool is_lda = false;
        };

        std::expected<XcArrays, std::string> evaluate_xc_arrays(
            const DFT::DensityOnGrid &ground,
            const XC::Functional &x_func,
            const XC::Functional &c_func,
            Eigen::Index npts)
        {
            XcArrays a;
            a.is_lda = x_func.is_lda_like();

            std::vector<double> rho_vec(static_cast<std::size_t>(npts));
            for (Eigen::Index p = 0; p < npts; ++p)
                rho_vec[static_cast<std::size_t>(p)] = ground.total.rho(p);

            if (a.is_lda)
            {
                std::vector<double> exc_x, exc_c, vrho_c;
                if (auto r = x_func.evaluate_lda_exc_vxc(rho_vec, static_cast<int>(npts), exc_x, a.vrho); !r)
                    return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());
                if (auto r = c_func.evaluate_lda_exc_vxc(rho_vec, static_cast<int>(npts), exc_c, vrho_c); !r)
                    return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());
                std::vector<double> f2x, f2c;
                if (auto r = x_func.evaluate_lda_fxc(rho_vec, static_cast<int>(npts), f2x); !r)
                    return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());
                if (auto r = c_func.evaluate_lda_fxc(rho_vec, static_cast<int>(npts), f2c); !r)
                    return std::unexpected("compute_dh_xc_pt2_gradient: " + r.error());
                const bool combined = x_func.is_combined_exchange_correlation();
                a.vsigma.assign(static_cast<std::size_t>(npts), 0.0);
                a.v2rhosigma.assign(static_cast<std::size_t>(npts), 0.0);
                a.v2sigma2.assign(static_cast<std::size_t>(npts), 0.0);
                a.v2rho2.resize(static_cast<std::size_t>(npts));
                for (Eigen::Index p = 0; p < npts; ++p)
                {
                    const std::size_t pi = static_cast<std::size_t>(p);
                    if (!combined)
                        a.vrho[pi] += vrho_c[pi];
                    a.v2rho2[pi] = f2x[pi] + (combined ? 0.0 : f2c[pi]);
                }
                return a;
            }

            std::vector<double> sigma_vec(static_cast<std::size_t>(npts));
            for (Eigen::Index p = 0; p < npts; ++p)
                sigma_vec[static_cast<std::size_t>(p)] =
                    ground.total.grad_x(p) * ground.total.grad_x(p) +
                    ground.total.grad_y(p) * ground.total.grad_y(p) +
                    ground.total.grad_z(p) * ground.total.grad_z(p);

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

            const bool combined = x_func.is_combined_exchange_correlation();
            a.vrho.resize(static_cast<std::size_t>(npts));
            a.vsigma.resize(static_cast<std::size_t>(npts));
            a.v2rho2.resize(static_cast<std::size_t>(npts));
            a.v2rhosigma.resize(static_cast<std::size_t>(npts));
            a.v2sigma2.resize(static_cast<std::size_t>(npts));
            for (Eigen::Index p = 0; p < npts; ++p)
            {
                const std::size_t pi = static_cast<std::size_t>(p);
                a.vrho[pi] = vrho_x[pi] + (combined ? 0.0 : vrho_c[pi]);
                a.vsigma[pi] = vsigma_x[pi] + (combined ? 0.0 : vsigma_c[pi]);
                a.v2rho2[pi] = v2rho2_x[pi] + (combined ? 0.0 : v2rho2_c[pi]);
                a.v2rhosigma[pi] = v2rhosigma_x[pi] + (combined ? 0.0 : v2rhosigma_c[pi]);
                a.v2sigma2[pi] = v2sigma2_x[pi] + (combined ? 0.0 : v2sigma2_c[pi]);
            }
            return a;
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
        const Eigen::MatrixXd D_sym =
            0.5 * (relaxed_density_restricted + relaxed_density_restricted.transpose());

        const Eigen::Index npts = ao.npoints();
        auto xc_res = evaluate_xc_arrays(*ground, exchange_functional, correlation_functional, npts);
        if (!xc_res)
            return std::unexpected(xc_res.error());
        const XcArrays &xc = *xc_res;
        const bool gga = !xc.is_lda;
        const auto A_end = static_cast<std::size_t>(mol.natoms);

        Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);

        for (Eigen::Index ip = 0; ip < npts; ++ip)
        {
            const double w = grid.points(ip, 3);
            if (w == 0.0 || ground->total.rho(ip) < kRhoFloor)
                continue;
            const std::size_t pi = static_cast<std::size_t>(ip);

            const double vrho = xc.vrho[pi];
            const double vsigma = gga ? xc.vsigma[pi] : 0.0;
            const double v2rho2 = xc.v2rho2[pi];
            const double v2rhosigma = gga ? xc.v2rhosigma[pi] : 0.0;
            const double v2sigma2 = gga ? xc.v2sigma2[pi] : 0.0;

            const std::array<double, 3> g = {
                ground->total.grad_x(ip), ground->total.grad_y(ip), ground->total.grad_z(ip)};
            const std::array<double, 3> grad_d = {
                relaxed->total.grad_x(ip), relaxed->total.grad_y(ip), relaxed->total.grad_z(ip)};
            const double d = relaxed->total.rho(ip);
            const double g_dot_gradd = g[0] * grad_d[0] + g[1] * grad_d[1] + g[2] * grad_d[2];

            // ---- XC_I + XC_II : per (atom A, Cartesian q) ------------------
            // XC_I  : basis-function derivative of rho_D  (drho_channel(D),
            //         dg_axis_spin(D)) against the FIRST XC derivatives.
            // XC_II : rho_P inside V_xc[rho_P] responds (drho_channel(P),
            //         dg_axis_spin(P)) against the SECOND XC derivatives.
            // Also accumulate the sum over A of each channel -- Sum_A
            // drho_channel = -d/dr_q -- for XC_III's point-translation piece.
            std::array<double, 3> sum_rx = {0.0, 0.0, 0.0}; // -d(rho_P)/dr_q
            std::array<double, 3> sum_dx = {0.0, 0.0, 0.0}; // -d(rho_D)/dr_q
            std::array<std::array<double, 3>, 3> sum_gx{};  // sum_gx[q][a] = -d[grad_rho_P]_a/dr_q
            std::array<std::array<double, 3>, 3> sum_dgx{}; // sum_dgx[q][a] = -d[grad_rho_D]_a/dr_q

            for (int q = 0; q < 3; ++q)
            {
                for (std::size_t A = 0; A < A_end; ++A)
                {
                    const int Ai = static_cast<int>(A);
                    const double rx = drho_channel(P_sym, ao, ip, Ai, q, atoms_bf);
                    const double dx = drho_channel(D_sym, ao, ip, Ai, q, atoms_bf);
                    sum_rx[static_cast<std::size_t>(q)] += rx;
                    sum_dx[static_cast<std::size_t>(q)] += dx;

                    std::array<double, 3> gx = {0.0, 0.0, 0.0};
                    std::array<double, 3> dgx = {0.0, 0.0, 0.0};
                    if (gga)
                        for (int a = 0; a < 3; ++a)
                        {
                            gx[static_cast<std::size_t>(a)] =
                                dg_axis_spin(P_sym, ao, hess, ip, a, Ai, q, atoms_bf);
                            dgx[static_cast<std::size_t>(a)] =
                                dg_axis_spin(D_sym, ao, hess, ip, a, Ai, q, atoms_bf);
                            sum_gx[static_cast<std::size_t>(q)][static_cast<std::size_t>(a)] +=
                                gx[static_cast<std::size_t>(a)];
                            sum_dgx[static_cast<std::size_t>(q)][static_cast<std::size_t>(a)] +=
                                dgx[static_cast<std::size_t>(a)];
                        }

                    const double g_dot_gx = gga ? (g[0] * gx[0] + g[1] * gx[1] + g[2] * gx[2]) : 0.0;
                    const double gx_dot_gradd =
                        gga ? (gx[0] * grad_d[0] + gx[1] * grad_d[1] + gx[2] * grad_d[2]) : 0.0;
                    const double dgx_dot_g =
                        gga ? (dgx[0] * g[0] + dgx[1] * g[1] + dgx[2] * g[2]) : 0.0;

                    // XC_I
                    const double xc1 = vrho * dx + 2.0 * vsigma * dgx_dot_g;
                    // XC_II
                    const double d_dfdrho = v2rho2 * rx + 2.0 * v2rhosigma * g_dot_gx;
                    const double d_dfdgamma = v2rhosigma * rx + 2.0 * v2sigma2 * g_dot_gx;
                    const double xc2 = d_dfdrho * d + 2.0 * d_dfdgamma * g_dot_gradd +
                                       2.0 * vsigma * gx_dot_gradd;

                    grad(static_cast<Eigen::Index>(A), q) += w * (xc1 + xc2);
                }
            }

            // ---- XC_III : grid quadrature moving frame --------------------
            //   I_p = vrho*rho_D + 2*vsigma*(grad_rho_P . grad_rho_D)
            //   (a) Becke partition weight response: [dw_p/dR_{A,q}] * I_p
            //   (b) point translation, owner atom only: w_p * dI_p/dr_q
            //       with d(.)/dr_q = -(sum over A of the channel helper).
            const double I_p = vrho * d + 2.0 * vsigma * g_dot_gradd;

            auto dpartition = becke_partition_owner_derivatives(grid, mol, ip);
            if (!dpartition)
                return std::unexpected(dpartition.error());
            const double w_atomic = grid.atomic_weights(ip);
            for (std::size_t A = 0; A < A_end; ++A)
                for (int q = 0; q < 3; ++q)
                    grad(static_cast<Eigen::Index>(A), q) +=
                        w_atomic * (*dpartition)(static_cast<Eigen::Index>(A), q) * I_p;

            const int owner_atom = grid.owner(ip);
            for (int q = 0; q < 3; ++q)
            {
                const std::size_t qi = static_cast<std::size_t>(q);
                const double drho_P_dr = -sum_rx[qi];
                const double drho_D_dr = -sum_dx[qi];
                double g_dot_dgP_dr = 0.0; // grad_rho_P . d(grad_rho_P)/dr_q
                double dsigma_P_dr = 0.0;
                double dgd_dr = 0.0; // d(grad_rho_P . grad_rho_D)/dr_q
                if (gga)
                {
                    for (int a = 0; a < 3; ++a)
                    {
                        const double dgP_a = -sum_gx[qi][static_cast<std::size_t>(a)];
                        const double dgD_a = -sum_dgx[qi][static_cast<std::size_t>(a)];
                        g_dot_dgP_dr += g[static_cast<std::size_t>(a)] * dgP_a;
                        dgd_dr += dgP_a * grad_d[static_cast<std::size_t>(a)] +
                                  g[static_cast<std::size_t>(a)] * dgD_a;
                    }
                    dsigma_P_dr = 2.0 * g_dot_dgP_dr;
                }
                const double dvrho_dr = v2rho2 * drho_P_dr + v2rhosigma * dsigma_P_dr;
                const double dvsigma_dr =
                    gga ? (v2rhosigma * drho_P_dr + v2sigma2 * dsigma_P_dr) : 0.0;
                const double dI_dr = dvrho_dr * d + vrho * drho_D_dr +
                                     (gga ? (2.0 * dvsigma_dr * g_dot_gradd + 2.0 * vsigma * dgd_dr)
                                          : 0.0);
                grad(static_cast<Eigen::Index>(owner_atom), q) += w * dI_dr;
            }
        }

        return grad;
    }

} // namespace DFT::Gradient
