#include "dft_kernel_gradient.h"

#include <algorithm>
#include <vector>

#include "dft_gradient.h"
#include "xc_grid.h"

namespace DFT::Gradient
{

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

        if (!exchange_functional.is_lda_like())
            return std::unexpected(
                "compute_dh_xc_pt2_gradient: only the LDA branch is implemented (S2); "
                "GGA (Term 1 gamma branch + Term 2) is S3");

        auto atoms_bf_res = atom_bf_lists(mol, basis);
        if (!atoms_bf_res)
            return std::unexpected(atoms_bf_res.error());
        const auto &atoms_bf = *atoms_bf_res;

        // rho_P (SCF/ground) and rho_D (relaxed PT2 difference) on the grid.
        auto ground = evaluate_density_on_grid(ao, ground_density_restricted);
        if (!ground)
            return std::unexpected("compute_dh_xc_pt2_gradient: " + ground.error());
        auto relaxed = evaluate_density_on_grid(ao, relaxed_density_restricted);
        if (!relaxed)
            return std::unexpected("compute_dh_xc_pt2_gradient: " + relaxed.error());

        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho_vec(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
            rho_vec[static_cast<std::size_t>(p)] = ground->total.rho(p);

        // f^{rho rho rho} = d(v2rho2)/d(rho), unpolarized total-density
        // convention -- the SAME convention compute_analytic_xc_hessian_
        // vector_product's LDA branch uses for v2rho2 (S1's kxc selfcheck
        // pinned v3rho3 == d(v2rho2)/d(rho) in exactly that convention).
        std::vector<double> v3rho3_x, v3rho3_c;
        auto kxc_x = exchange_functional.evaluate_lda_kxc(rho_vec, static_cast<int>(npts), v3rho3_x);
        if (!kxc_x)
            return std::unexpected("compute_dh_xc_pt2_gradient: " + kxc_x.error());
        auto kxc_c = correlation_functional.evaluate_lda_kxc(rho_vec, static_cast<int>(npts), v3rho3_c);
        if (!kxc_c)
            return std::unexpected("compute_dh_xc_pt2_gradient: " + kxc_c.error());
        // Combined XC libxc entry (B3LYP/PBE0/...) already carries its
        // correlation in the exchange slot; a second kxc[correlation] would
        // double-count. Same guard as compute_analytic_xc_hessian_vector_product.
        if (exchange_functional.is_combined_exchange_correlation())
            std::fill(v3rho3_c.begin(), v3rho3_c.end(), 0.0);

        const Eigen::MatrixXd P_sym =
            0.5 * (ground_density_restricted + ground_density_restricted.transpose());

        Eigen::MatrixXd grad =
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);

        // v3rho3 diverges as rho^(-5/3) for LDA exchange; libxc's own
        // dens_threshold clamps it to 0 only far below (~1e-32 for LDA_X),
        // but a grid point with rho_P that small contributes nothing to a
        // physical integral anyway. Screen here at a sane floor so the term
        // is well-defined regardless of libxc's threshold setting -- 1e-8
        // is conservative (well above LDA_C_PW's own 1e-12) and drops only
        // the numerically meaningless quadrature tails.
        constexpr double kRhoFloor = 1e-8;

        // Eq. 33 Term 1 is a PLAIN grid integral of  rho_P^(x)(r) * c(r) ,
        // NOT d/dR of an integral -- rho_P^(x) (Eq. 15) is the basis-
        // function derivative of the density at a FIXED spatial point, and
        // is itself the integrand. So there is no moving-grid (Becke-weight
        // / point-translation) correction here: only the basis-function-
        // derivative piece, drho_channel, evaluated at the current geometry.
        for (Eigen::Index ip = 0; ip < npts; ++ip)
        {
            const double w = grid.points(ip, 3);
            if (w == 0.0)
                continue;
            if (ground->total.rho(ip) < kRhoFloor)
                continue;

            const std::size_t pi = static_cast<std::size_t>(ip);
            const double c =
                w * (v3rho3_x[pi] + v3rho3_c[pi]) * relaxed->total.rho(ip);
            if (c == 0.0)
                continue;

            for (int q = 0; q < 3; ++q)
                for (std::size_t atom_A = 0; atom_A < mol.natoms; ++atom_A)
                    grad(static_cast<Eigen::Index>(atom_A), q) +=
                        c * drho_channel(P_sym, ao, ip, static_cast<int>(atom_A), q, atoms_bf);
        }

        return grad;
    }

} // namespace DFT::Gradient
