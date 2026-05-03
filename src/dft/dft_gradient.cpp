#include "dft_gradient.h"

#include <array>
#include <vector>

#include "basis/basis.h"

namespace DFT::Gradient
{

    namespace
    {

        std::expected<std::vector<int>, std::string> build_bf_shell_map(const HartreeFock::Basis &basis)
        {
            const auto &shells = basis._shells;
            const auto &bfs = basis._basis_functions;
            const std::size_t nb = basis.nbasis();
            std::vector<int> bf_shell(nb, -1);
            for (std::size_t s = 0; s < shells.size(); ++s)
                for (std::size_t mu = 0; mu < nb; ++mu)
                    if (bfs[mu]._shell == &shells[s])
                        bf_shell[mu] = static_cast<int>(s);
            for (std::size_t mu = 0; mu < nb; ++mu)
                if (bf_shell[mu] < 0)
                    return std::unexpected("DFT XC gradient: basis function shell mapping failed");
            return bf_shell;
        }

        std::expected<std::vector<int>, std::string> build_shell_atom_map(
            const HartreeFock::Molecule &mol,
            const HartreeFock::Basis &basis)
        {
            const auto &shells = basis._shells;
            std::vector<int> map(shells.size(), -1);
            for (std::size_t s = 0; s < shells.size(); ++s)
            {
                const Eigen::Vector3d &sc = shells[s]._center;
                for (std::size_t a = 0; a < mol.natoms; ++a)
                {
                    const double dx = sc[0] - mol._standard(static_cast<Eigen::Index>(a), 0);
                    const double dy = sc[1] - mol._standard(static_cast<Eigen::Index>(a), 1);
                    const double dz = sc[2] - mol._standard(static_cast<Eigen::Index>(a), 2);
                    if (dx * dx + dy * dy + dz * dz < 1e-10)
                    {
                        map[s] = static_cast<int>(a);
                        break;
                    }
                }
                if (map[s] < 0)
                    return std::unexpected("DFT XC gradient: shell center does not match any atom");
            }
            return map;
        }

        std::vector<std::vector<int>> build_atoms_bf_lists(
            std::size_t natoms,
            const std::vector<int> &bf_shell,
            const std::vector<int> &shell_atom)
        {
            std::vector<std::vector<int>> out(natoms);
            for (std::size_t mu = 0; mu < bf_shell.size(); ++mu)
            {
                const int atom = shell_atom[static_cast<std::size_t>(bf_shell[mu])];
                out[static_cast<std::size_t>(atom)].push_back(static_cast<int>(mu));
            }
            return out;
        }

        inline double h_axis_q(const AOGridHessian &H, Eigen::Index ip, Eigen::Index mu, int spatial_axis, int q)
        {
            switch (spatial_axis)
            {
            case 0:
                switch (q)
                {
                case 0:
                    return H.h_xx(ip, mu);
                case 1:
                    return H.h_xy(ip, mu);
                case 2:
                    return H.h_xz(ip, mu);
                default:
                    return 0.0;
                }
            case 1:
                switch (q)
                {
                case 0:
                    return H.h_xy(ip, mu);
                case 1:
                    return H.h_yy(ip, mu);
                case 2:
                    return H.h_yz(ip, mu);
                default:
                    return 0.0;
                }
            case 2:
                switch (q)
                {
                case 0:
                    return H.h_xz(ip, mu);
                case 1:
                    return H.h_yz(ip, mu);
                case 2:
                    return H.h_zz(ip, mu);
                default:
                    return 0.0;
                }
            default:
                return 0.0;
            }
        }

        inline double g_axis_mu(const AOGridEvaluation &ao, Eigen::Index ip, Eigen::Index mu, int axis)
        {
            switch (axis)
            {
            case 0:
                return ao.grad_x(ip, mu);
            case 1:
                return ao.grad_y(ip, mu);
            case 2:
                return ao.grad_z(ip, mu);
            default:
                return 0.0;
            }
        }

        inline double gq_mu(const AOGridEvaluation &ao, Eigen::Index ip, Eigen::Index mu, int q)
        {
            return g_axis_mu(ao, ip, mu, q);
        }

        // ∂g_axis/dR_{atom_A,q} for spin-resolved density matrix P (symmetric).
        double dg_axis_spin(
            const Eigen::MatrixXd &P,
            const AOGridEvaluation &ao,
            const AOGridHessian &H,
            Eigen::Index ip,
            int axis_g,
            int atom_A,
            int q,
            const std::vector<std::vector<int>> &atoms_bf)
        {
            double sum = 0.0;
            const Eigen::Index nb = P.cols();

            for (int mu : atoms_bf[static_cast<std::size_t>(atom_A)])
            {
                const Eigen::Index imu = static_cast<Eigen::Index>(mu);
                const double gm = g_axis_mu(ao, ip, imu, axis_g);
                const double hm = h_axis_q(H, ip, imu, axis_g, q);
                for (Eigen::Index nu = 0; nu < nb; ++nu)
                    sum += P(imu, nu) * (-hm * ao.values(ip, nu) - gm * gq_mu(ao, ip, nu, q));
            }

            for (int nu : atoms_bf[static_cast<std::size_t>(atom_A)])
            {
                const Eigen::Index inu = static_cast<Eigen::Index>(nu);
                const double hn = h_axis_q(H, ip, inu, axis_g, q);
                for (Eigen::Index mu = 0; mu < nb; ++mu)
                    sum += P(mu, inu) * (-gq_mu(ao, ip, mu, q) * g_axis_mu(ao, ip, inu, axis_g) -
                                         ao.values(ip, mu) * hn);
            }

            return sum;
        }

        double drho_channel(
            const Eigen::MatrixXd &P_sym,
            const AOGridEvaluation &ao,
            Eigen::Index ip,
            int atom_A,
            int q,
            const std::vector<std::vector<int>> &atoms_bf)
        {
            double d = 0.0;
            const Eigen::Index nb = P_sym.cols();
            for (int mu : atoms_bf[static_cast<std::size_t>(atom_A)])
            {
                const Eigen::Index imu = static_cast<Eigen::Index>(mu);
                const double gmq = gq_mu(ao, ip, imu, q);
                for (Eigen::Index nu = 0; nu < nb; ++nu)
                    d -= 2.0 * P_sym(imu, nu) * ao.values(ip, nu) * gmq;
            }
            return d;
        }

        std::expected<Eigen::MatrixXd, std::string> becke_partition_owner_derivatives(
            const MolecularGrid &grid,
            const HartreeFock::Molecule &mol,
            Eigen::Index ip)
        {
            const Eigen::Index natoms = static_cast<Eigen::Index>(mol.natoms);
            if (grid.owner.size() != grid.points.rows())
                return std::unexpected("molecular grid owner array does not match point count");
            if (grid.atomic_weights.size() != grid.points.rows() ||
                grid.partition_weights.size() != grid.points.rows())
            {
                return std::unexpected("molecular grid partition metadata does not match point count");
            }

            const int owner = grid.owner(ip);
            if (owner < 0 || owner >= natoms)
                return std::unexpected("molecular grid owner index is out of range");

            const Eigen::Vector3d point = grid.points.row(ip).head<3>().transpose();
            const Eigen::MatrixXd &coordinates = mol._standard;
            const Eigen::VectorXi &atomic_numbers = mol.atomic_numbers;

            std::vector<double> products(static_cast<std::size_t>(natoms), 1.0);
            std::vector<Eigen::MatrixXd> dproducts(
                static_cast<std::size_t>(natoms),
                Eigen::MatrixXd::Zero(natoms, 3));

            for (Eigen::Index i = 0; i < natoms; ++i)
            {
                const Eigen::Vector3d ri = coordinates.row(i).transpose();
                for (Eigen::Index j = i + 1; j < natoms; ++j)
                {
                    const Eigen::Vector3d rj = coordinates.row(j).transpose();
                    const Eigen::Vector3d rij = ri - rj;
                    const double Rij = rij.norm();
                    if (Rij < 1e-14)
                        return std::unexpected("Becke partition derivative encountered coincident atoms");

                    const Eigen::Vector3d eij = rij / Rij;
                    const Eigen::Vector3d dpi = point - ri;
                    const Eigen::Vector3d dpj = point - rj;
                    const double di = dpi.norm();
                    const double dj = dpj.norm();
                    if (di < 1e-14 || dj < 1e-14)
                        return std::unexpected("Becke partition derivative encountered a grid point on a nucleus");

                    const Eigen::Vector3d ui = dpi / di;
                    const Eigen::Vector3d uj = dpj / dj;
                    const double mu_raw = (di - dj) / Rij;
                    const double aij = DFT::detail::treutler_becke_adjustment(
                        atomic_numbers(i),
                        atomic_numbers(j));
                    const double mu_adjusted = mu_raw + aij * (1.0 - mu_raw * mu_raw);
                    const double mu_clamped = std::clamp(mu_adjusted, -1.0, 1.0);
                    const double sij = DFT::detail::becke_switch(mu_clamped);
                    const double ds_dmu =
                        (std::abs(mu_adjusted - mu_clamped) < 1e-14)
                            ? DFT::detail::becke_switch_derivative(mu_clamped)
                            : 0.0;

                    Eigen::MatrixXd ds = Eigen::MatrixXd::Zero(natoms, 3);
                    for (Eigen::Index atom = 0; atom < natoms; ++atom)
                    {
                        const double delta_owner = (atom == owner) ? 1.0 : 0.0;
                        const double delta_i = (atom == i) ? 1.0 : 0.0;
                        const double delta_j = (atom == j) ? 1.0 : 0.0;

                        const Eigen::Vector3d ddi = (delta_owner - delta_i) * ui;
                        const Eigen::Vector3d ddj = (delta_owner - delta_j) * uj;
                        const Eigen::Vector3d dR = (delta_i - delta_j) * eij;
                        const Eigen::Vector3d dmu_raw =
                            ((ddi - ddj) * Rij - (di - dj) * dR) / (Rij * Rij);
                        const Eigen::Vector3d dmu_adjusted =
                            (1.0 - 2.0 * aij * mu_raw) * dmu_raw;
                        ds.row(atom) = (ds_dmu * dmu_adjusted).transpose();
                    }

                    const double old_pi = products[static_cast<std::size_t>(i)];
                    const double old_pj = products[static_cast<std::size_t>(j)];
                    const Eigen::MatrixXd old_dpi = dproducts[static_cast<std::size_t>(i)];
                    const Eigen::MatrixXd old_dpj = dproducts[static_cast<std::size_t>(j)];

                    products[static_cast<std::size_t>(i)] = old_pi * sij;
                    products[static_cast<std::size_t>(j)] = old_pj * (1.0 - sij);
                    dproducts[static_cast<std::size_t>(i)] = old_dpi * sij + old_pi * ds;
                    dproducts[static_cast<std::size_t>(j)] = old_dpj * (1.0 - sij) - old_pj * ds;
                }
            }

            double sum = 0.0;
            Eigen::MatrixXd dsum = Eigen::MatrixXd::Zero(natoms, 3);
            for (Eigen::Index atom = 0; atom < natoms; ++atom)
            {
                sum += products[static_cast<std::size_t>(atom)];
                dsum += dproducts[static_cast<std::size_t>(atom)];
            }
            if (sum <= 0.0)
                return std::unexpected("Becke partition derivative: partition denominator underflowed");

            const double owner_product = products[static_cast<std::size_t>(owner)];
            Eigen::MatrixXd derivatives = Eigen::MatrixXd::Zero(natoms, 3);
            for (Eigen::Index atom = 0; atom < natoms; ++atom)
            {
                for (int q = 0; q < 3; ++q)
                {
                    derivatives(atom, q) =
                        (dproducts[static_cast<std::size_t>(owner)](atom, q) * sum -
                         owner_product * dsum(atom, q)) /
                        (sum * sum);
                }
            }

            return derivatives;
        }

    } // namespace

    std::expected<Eigen::MatrixXd, std::string>
    compute_xc_nuclear_gradient_rks(
        const HartreeFock::Molecule &mol,
        const HartreeFock::Basis &basis,
        const MolecularGrid &grid,
        const AOGridEvaluation &ao,
        const AOGridHessian &hess,
        const XCGridEvaluation &xc,
        const Eigen::Ref<const Eigen::MatrixXd> &density_restricted)
    {
        if (xc.density.polarized)
            return std::unexpected("compute_xc_nuclear_gradient_rks requires spin-unpolarized XC density");

        if (hess.npoints() != ao.npoints() || hess.nbasis() != ao.nbasis())
            return std::unexpected("AO Hessian dimensions do not match AO grid");

        if (grid.points.rows() != ao.npoints())
            return std::unexpected("molecular grid point count mismatch in XC gradient");

        const Eigen::Index nb = ao.nbasis();
        if (density_restricted.rows() != nb || density_restricted.cols() != nb)
            return std::unexpected("restricted density dimension mismatch for XC gradient");

        auto bf_shell = build_bf_shell_map(basis);
        if (!bf_shell)
            return std::unexpected(bf_shell.error());
        auto shell_atom = build_shell_atom_map(mol, basis);
        if (!shell_atom)
            return std::unexpected(shell_atom.error());
        const auto atoms_bf =
            build_atoms_bf_lists(mol.natoms, *bf_shell, *shell_atom);

        const Eigen::MatrixXd P_sym = 0.5 * (density_restricted + density_restricted.transpose());

        Eigen::MatrixXd grad =
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        std::array<Eigen::MatrixXd, 3> vxc1 = {
            Eigen::MatrixXd::Zero(nb, nb),
            Eigen::MatrixXd::Zero(nb, nb),
            Eigen::MatrixXd::Zero(nb, nb)};

        const bool gga = xc.vsigma.cols() >= 1;
        const Eigen::Index npts = ao.npoints();

        for (Eigen::Index ip = 0; ip < npts; ++ip)
        {
            const double w = grid.points(ip, 3);
            if (w == 0.0)
                continue;

            auto dpartition = becke_partition_owner_derivatives(grid, mol, ip);
            if (!dpartition)
                return std::unexpected(dpartition.error());
            const double w_atomic = grid.atomic_weights(ip);
            const int owner_atom = grid.owner(ip);
            const Eigen::VectorXd phi = ao.values.row(ip).transpose();
            const Eigen::VectorXd gx = ao.grad_x.row(ip).transpose();
            const Eigen::VectorXd gy = ao.grad_y.row(ip).transpose();
            const Eigen::VectorXd gz = ao.grad_z.row(ip).transpose();

            const double vrho = xc.vrho(ip, 0);
            const double exc_density = xc.energy_density(ip);

            std::array<Eigen::MatrixXd, 3> vtmp = {
                Eigen::MatrixXd::Zero(nb, nb),
                Eigen::MatrixXd::Zero(nb, nb),
                Eigen::MatrixXd::Zero(nb, nb)};

            if (!gga)
            {
                const Eigen::VectorXd aow = phi * (w * vrho);
                vtmp[0].noalias() += gx * aow.transpose();
                vtmp[1].noalias() += gy * aow.transpose();
                vtmp[2].noalias() += gz * aow.transpose();
            }
            else
            {
                const double vs = xc.vsigma(ip, 0);
                const Eigen::Vector3d coeff =
                    2.0 * vs * Eigen::Vector3d(
                                   xc.density.total.grad_x(ip),
                                   xc.density.total.grad_y(ip),
                                   xc.density.total.grad_z(ip));
                const double wv0 = w * vrho * 0.5;
                const double wv1 = w * coeff.x();
                const double wv2 = w * coeff.y();
                const double wv3 = w * coeff.z();

                const Eigen::VectorXd aow =
                    phi * wv0 + gx * wv1 + gy * wv2 + gz * wv3;
                const Eigen::VectorXd aow_x =
                    gx * wv0 +
                    hess.h_xx.row(ip).transpose() * wv1 +
                    hess.h_xy.row(ip).transpose() * wv2 +
                    hess.h_xz.row(ip).transpose() * wv3;
                const Eigen::VectorXd aow_y =
                    gy * wv0 +
                    hess.h_xy.row(ip).transpose() * wv1 +
                    hess.h_yy.row(ip).transpose() * wv2 +
                    hess.h_yz.row(ip).transpose() * wv3;
                const Eigen::VectorXd aow_z =
                    gz * wv0 +
                    hess.h_xz.row(ip).transpose() * wv1 +
                    hess.h_yz.row(ip).transpose() * wv2 +
                    hess.h_zz.row(ip).transpose() * wv3;

                vtmp[0].noalias() += gx * aow.transpose() + aow_x * phi.transpose();
                vtmp[1].noalias() += gy * aow.transpose() + aow_y * phi.transpose();
                vtmp[2].noalias() += gz * aow.transpose() + aow_z * phi.transpose();
            }

            for (int q = 0; q < 3; ++q)
            {
                vxc1[static_cast<std::size_t>(q)].noalias() -= vtmp[static_cast<std::size_t>(q)];
                grad(static_cast<Eigen::Index>(owner_atom), q) +=
                    2.0 * (vtmp[static_cast<std::size_t>(q)].cwiseProduct(P_sym)).sum();
            }

            for (std::size_t atom_A = 0; atom_A < mol.natoms; ++atom_A)
                for (int q = 0; q < 3; ++q)
                    grad(static_cast<Eigen::Index>(atom_A), q) +=
                        w_atomic *
                        (*dpartition)(static_cast<Eigen::Index>(atom_A), q) *
                        exc_density;
        }

        for (std::size_t atom_A = 0; atom_A < mol.natoms; ++atom_A)
        {
            for (int mu : atoms_bf[atom_A])
            {
                const Eigen::Index imu = static_cast<Eigen::Index>(mu);
                for (Eigen::Index nu = 0; nu < nb; ++nu)
                {
                    for (int q = 0; q < 3; ++q)
                    {
                        grad(static_cast<Eigen::Index>(atom_A), q) +=
                            2.0 * vxc1[static_cast<std::size_t>(q)](imu, nu) * P_sym(imu, nu);
                    }
                }
            }
        }

        return grad;
    }

    std::expected<Eigen::MatrixXd, std::string>
    compute_xc_nuclear_gradient_uks(
        const HartreeFock::Molecule &mol,
        const HartreeFock::Basis &basis,
        const MolecularGrid &grid,
        const AOGridEvaluation &ao,
        const AOGridHessian &hess,
        const XCGridEvaluation &xc,
        const Eigen::Ref<const Eigen::MatrixXd> &density_alpha,
        const Eigen::Ref<const Eigen::MatrixXd> &density_beta)
    {
        if (!xc.density.polarized)
            return std::unexpected("compute_xc_nuclear_gradient_uks requires polarized XC density");

        if (xc.vrho.cols() != 2)
            return std::unexpected("polarized XC gradient expects vrho with two columns");

        if (hess.npoints() != ao.npoints() || hess.nbasis() != ao.nbasis())
            return std::unexpected("AO Hessian dimensions do not match AO grid");

        const Eigen::Index nb = ao.nbasis();
        if (density_alpha.rows() != nb || density_beta.rows() != nb)
            return std::unexpected("UKS density dimension mismatch for XC gradient");

        auto bf_shell = build_bf_shell_map(basis);
        if (!bf_shell)
            return std::unexpected(bf_shell.error());
        auto shell_atom = build_shell_atom_map(mol, basis);
        if (!shell_atom)
            return std::unexpected(shell_atom.error());
        const auto atoms_bf =
            build_atoms_bf_lists(mol.natoms, *bf_shell, *shell_atom);

        const Eigen::MatrixXd Pa_sym = 0.5 * (density_alpha + density_alpha.transpose());
        const Eigen::MatrixXd Pb_sym = 0.5 * (density_beta + density_beta.transpose());

        Eigen::MatrixXd grad =
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        std::array<Eigen::MatrixXd, 3> vxc1_alpha = {
            Eigen::MatrixXd::Zero(nb, nb),
            Eigen::MatrixXd::Zero(nb, nb),
            Eigen::MatrixXd::Zero(nb, nb)};
        std::array<Eigen::MatrixXd, 3> vxc1_beta = {
            Eigen::MatrixXd::Zero(nb, nb),
            Eigen::MatrixXd::Zero(nb, nb),
            Eigen::MatrixXd::Zero(nb, nb)};

        const bool gga = xc.vsigma.cols() == 3;
        const Eigen::Index npts = ao.npoints();

        for (Eigen::Index ip = 0; ip < npts; ++ip)
        {
            const double w = grid.points(ip, 3);
            if (w == 0.0)
                continue;

            auto dpartition = becke_partition_owner_derivatives(grid, mol, ip);
            if (!dpartition)
                return std::unexpected(dpartition.error());
            const double w_atomic = grid.atomic_weights(ip);
            const int owner_atom = grid.owner(ip);
            const Eigen::VectorXd phi = ao.values.row(ip).transpose();
            const Eigen::VectorXd gx = ao.grad_x.row(ip).transpose();
            const Eigen::VectorXd gy = ao.grad_y.row(ip).transpose();
            const Eigen::VectorXd gz = ao.grad_z.row(ip).transpose();

            const double vrho_a = xc.vrho(ip, 0);
            const double vrho_b = xc.vrho(ip, 1);
            const double exc_density = xc.energy_density(ip);

            std::array<Eigen::MatrixXd, 3> vtmp_alpha = {
                Eigen::MatrixXd::Zero(nb, nb),
                Eigen::MatrixXd::Zero(nb, nb),
                Eigen::MatrixXd::Zero(nb, nb)};
            std::array<Eigen::MatrixXd, 3> vtmp_beta = {
                Eigen::MatrixXd::Zero(nb, nb),
                Eigen::MatrixXd::Zero(nb, nb),
                Eigen::MatrixXd::Zero(nb, nb)};

            if (!gga)
            {
                const Eigen::VectorXd aow_a = phi * (w * vrho_a);
                const Eigen::VectorXd aow_b = phi * (w * vrho_b);
                vtmp_alpha[0].noalias() += gx * aow_a.transpose();
                vtmp_alpha[1].noalias() += gy * aow_a.transpose();
                vtmp_alpha[2].noalias() += gz * aow_a.transpose();
                vtmp_beta[0].noalias() += gx * aow_b.transpose();
                vtmp_beta[1].noalias() += gy * aow_b.transpose();
                vtmp_beta[2].noalias() += gz * aow_b.transpose();
            }
            else
            {
                const Eigen::Vector3d grad_a(
                    xc.density.alpha.grad_x(ip),
                    xc.density.alpha.grad_y(ip),
                    xc.density.alpha.grad_z(ip));
                const Eigen::Vector3d grad_b(
                    xc.density.beta.grad_x(ip),
                    xc.density.beta.grad_y(ip),
                    xc.density.beta.grad_z(ip));
                const double vs_aa = xc.vsigma(ip, 0);
                const double vs_ab = xc.vsigma(ip, 1);
                const double vs_bb = xc.vsigma(ip, 2);
                const Eigen::Vector3d coeff_a =
                    2.0 * vs_aa * grad_a + vs_ab * grad_b;
                const Eigen::Vector3d coeff_b =
                    vs_ab * grad_a + 2.0 * vs_bb * grad_b;

                const auto accumulate_gga_spin =
                    [&](std::array<Eigen::MatrixXd, 3> &vtmp_spin,
                        double vrho_spin,
                        const Eigen::Vector3d &coeff_spin)
                {
                    const double wv0 = w * vrho_spin * 0.5;
                    const double wv1 = w * coeff_spin.x();
                    const double wv2 = w * coeff_spin.y();
                    const double wv3 = w * coeff_spin.z();

                    const Eigen::VectorXd aow =
                        phi * wv0 + gx * wv1 + gy * wv2 + gz * wv3;
                    const Eigen::VectorXd aow_x =
                        gx * wv0 +
                        hess.h_xx.row(ip).transpose() * wv1 +
                        hess.h_xy.row(ip).transpose() * wv2 +
                        hess.h_xz.row(ip).transpose() * wv3;
                    const Eigen::VectorXd aow_y =
                        gy * wv0 +
                        hess.h_xy.row(ip).transpose() * wv1 +
                        hess.h_yy.row(ip).transpose() * wv2 +
                        hess.h_yz.row(ip).transpose() * wv3;
                    const Eigen::VectorXd aow_z =
                        gz * wv0 +
                        hess.h_xz.row(ip).transpose() * wv1 +
                        hess.h_yz.row(ip).transpose() * wv2 +
                        hess.h_zz.row(ip).transpose() * wv3;

                    vtmp_spin[0].noalias() += gx * aow.transpose() + aow_x * phi.transpose();
                    vtmp_spin[1].noalias() += gy * aow.transpose() + aow_y * phi.transpose();
                    vtmp_spin[2].noalias() += gz * aow.transpose() + aow_z * phi.transpose();
                };

                accumulate_gga_spin(vtmp_alpha, vrho_a, coeff_a);
                accumulate_gga_spin(vtmp_beta, vrho_b, coeff_b);
            }

            for (int q = 0; q < 3; ++q)
            {
                vxc1_alpha[static_cast<std::size_t>(q)].noalias() -= vtmp_alpha[static_cast<std::size_t>(q)];
                vxc1_beta[static_cast<std::size_t>(q)].noalias() -= vtmp_beta[static_cast<std::size_t>(q)];
                grad(static_cast<Eigen::Index>(owner_atom), q) +=
                    2.0 * (vtmp_alpha[static_cast<std::size_t>(q)].cwiseProduct(Pa_sym)).sum() +
                    2.0 * (vtmp_beta[static_cast<std::size_t>(q)].cwiseProduct(Pb_sym)).sum();
            }

            for (std::size_t atom_A = 0; atom_A < mol.natoms; ++atom_A)
                for (int q = 0; q < 3; ++q)
                    grad(static_cast<Eigen::Index>(atom_A), q) +=
                        w_atomic *
                        (*dpartition)(static_cast<Eigen::Index>(atom_A), q) *
                        exc_density;
        }

        for (std::size_t atom_A = 0; atom_A < mol.natoms; ++atom_A)
        {
            for (int mu : atoms_bf[atom_A])
            {
                const Eigen::Index imu = static_cast<Eigen::Index>(mu);
                for (Eigen::Index nu = 0; nu < nb; ++nu)
                {
                    for (int q = 0; q < 3; ++q)
                    {
                        grad(static_cast<Eigen::Index>(atom_A), q) +=
                            2.0 * vxc1_alpha[static_cast<std::size_t>(q)](imu, nu) * Pa_sym(imu, nu) +
                            2.0 * vxc1_beta[static_cast<std::size_t>(q)](imu, nu) * Pb_sym(imu, nu);
                    }
                }
            }
        }

        return grad;
    }

} // namespace DFT::Gradient
