#include "dft_gradient.h"

#include <algorithm>
#include <array>
#include <vector>

#include "basis/basis.h"

namespace DFT::Gradient
{

    namespace
    {

        std::expected<std::vector<int>, std::string> build_bf_shell_map(const HartreeFock::Basis &basis)
        {
            // The moving-grid terms are organized atom-by-atom, but the AO
            // objects arrive as a flat basis-function list. Recover the shell
            // membership first so we can then map shells back to atoms.
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

    } // namespace

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

    // ∂[grad_rho_P]_{axis_g} / ∂R_{atom_A,q} at grid point `ip` (basis-
    // function derivative only). `P` symmetrized.
    //
    // grad_rho_P|_ag = sum_munu P(mu,nu) [ g_ag(mu) phi(nu) + phi(mu) g_ag(nu) ],
    //   g_ag(k) = d phi_k / d r_ag.
    // A basis function k on atom A has  d phi_k/dR_{A,q} = -g_q(k)  and
    //   d g_ag(k)/dR_{A,q} = -h_{ag,q}(k); functions off A contribute nothing.
    // The four product-rule terms collapse (by P symmetry, relabelling
    // mu<->nu) to  2 * sum_{mu in A, all nu} P(mu,nu) *
    //   [ -h_{ag,q}(mu) phi(nu)  -  g_q(mu) g_ag(nu) ].
    // Note the cross term is g_q(mu)*g_ag(nu) (the atom-A function carries
    // the q-derivative), NOT g_ag(mu)*g_q(nu).
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
            const double gq = gq_mu(ao, ip, imu, q);
            const double hm = h_axis_q(H, ip, imu, axis_g, q);
            for (Eigen::Index nu = 0; nu < nb; ++nu)
                sum += P(imu, nu) *
                       (-hm * ao.values(ip, nu) - gq * g_axis_mu(ao, ip, nu, axis_g));
        }

        return 2.0 * sum;
    }

    std::expected<std::vector<std::vector<int>>, std::string>
    atom_bf_lists(const HartreeFock::Molecule &mol, const HartreeFock::Basis &basis)
    {
        auto bf_shell = build_bf_shell_map(basis);
        if (!bf_shell)
            return std::unexpected(bf_shell.error());
        auto shell_atom = build_shell_atom_map(mol, basis);
        if (!shell_atom)
            return std::unexpected(shell_atom.error());
        return build_atoms_bf_lists(mol.natoms, *bf_shell, *shell_atom);
    }

    std::expected<Eigen::MatrixXd, std::string> becke_partition_owner_derivatives(
        const MolecularGrid &grid,
        const HartreeFock::Molecule &mol,
        Eigen::Index ip)
    {
        // Differentiate the owner atom's Becke partition weight with
        // respect to every nuclear Cartesian. This is the core moving-grid
        // correction needed for analytic DFT gradients.
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
        // d(product_k)/dR (natoms x 3) for every k, side by side in one buffer.
        Eigen::MatrixXd dproducts = Eigen::MatrixXd::Zero(natoms, 3 * natoms);
        const auto dprod = [&dproducts](Eigen::Index k)
        { return dproducts.middleCols(3 * k, 3); };

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

                const double old_pi = products[static_cast<std::size_t>(i)];
                const double old_pj = products[static_cast<std::size_t>(j)];
                products[static_cast<std::size_t>(i)] = old_pi * sij;
                products[static_cast<std::size_t>(j)] = old_pj * (1.0 - sij);
                dprod(i) *= sij;
                dprod(j) *= (1.0 - sij);

                // d s_ij / dR is nonzero only in the rows of the owner, i and j.
                const Eigen::Index rows[3] = {owner, i, j};
                for (int r = 0; r < 3; ++r)
                {
                    const Eigen::Index atom = rows[r];
                    if (r > 0 && atom == owner)
                        continue;
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
                    const Eigen::RowVector3d ds = (ds_dmu * dmu_adjusted).transpose();
                    dprod(i).row(atom) += old_pi * ds;
                    dprod(j).row(atom) -= old_pj * ds;
                }
            }
        }

        double sum = 0.0;
        Eigen::MatrixXd dsum = Eigen::MatrixXd::Zero(natoms, 3);
        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            sum += products[static_cast<std::size_t>(atom)];
            dsum += dprod(atom);
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
                    (dprod(owner)(atom, q) * sum -
                     owner_product * dsum(atom, q)) /
                    (sum * sum);
            }
        }

        return derivatives;
    }

    namespace
    {
        // One spin channel of the XC nuclear gradient: symmetrized density and
        // the per-point weights of the AO contraction. LDA: aow = wv0 phi.
        // GGA: aow = wv0 phi + sum_k wvg_k g_k and aow_q = wv0 g_q + sum_k wvg_k h_qk.
        struct XCGradientChannel
        {
            Eigen::MatrixXd P;   // nb x nb, symmetric
            Eigen::VectorXd wv0; // npts
            Eigen::MatrixXd wvg; // npts x 3 (GGA only)
        };

        constexpr Eigen::Index kXCGradientBlock = 128;

        // Shared RKS/UKS driver. Per point, vtmp_q = g_q aow^T (+ aow_q phi^T
        // for GGA) is never formed: its sum over a block of points is one GEMM
        // pair into M[q], and its contraction with P (the owner-atom term) is
        // a row dot product against A P and Phi P.
        std::expected<Eigen::MatrixXd, std::string> xc_nuclear_gradient(
            const HartreeFock::Molecule &mol,
            const HartreeFock::Basis &basis,
            const MolecularGrid &grid,
            const AOGridEvaluation &ao,
            const XCGridEvaluation &xc,
            const std::vector<XCGradientChannel> &channels,
            bool gga)
        {
            auto atoms_bf = atom_bf_lists(mol, basis);
            if (!atoms_bf)
                return std::unexpected(atoms_bf.error());

            const Eigen::Index nb = ao.nbasis();
            const Eigen::Index npts = ao.npoints();
            Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
            std::vector<std::array<Eigen::MatrixXd, 3>> M(
                channels.size(),
                {Eigen::MatrixXd::Zero(nb, nb), Eigen::MatrixXd::Zero(nb, nb), Eigen::MatrixXd::Zero(nb, nb)});

            Eigen::MatrixXd phi, A, Aq, AP, phiP, owner_t;
            std::array<Eigen::MatrixXd, 3> g;
            std::array<Eigen::MatrixXd, 6> h; // xx xy xz yy yz zz
            MolecularGrid block_grid;         // GGA: this block's points, for its AO Hessian
            constexpr int hidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

            for (Eigen::Index p0 = 0; p0 < npts; p0 += kXCGradientBlock)
            {
                const Eigen::Index n = std::min(kXCGradientBlock, npts - p0);
                phi = ao.values.middleRows(p0, n);
                g[0] = ao.grad_x.middleRows(p0, n);
                g[1] = ao.grad_y.middleRows(p0, n);
                g[2] = ao.grad_z.middleRows(p0, n);
                if (gga)
                {
                    block_grid.points = grid.points.middleRows(p0, n);
                    auto hess = evaluate_ao_hessian_on_grid(basis, block_grid);
                    if (!hess)
                        return std::unexpected("DFT XC gradient: AO Hessian failed: " + hess.error());
                    h = {std::move(hess->h_xx), std::move(hess->h_xy), std::move(hess->h_xz),
                         std::move(hess->h_yy), std::move(hess->h_yz), std::move(hess->h_zz)};
                }
                owner_t.setZero(n, 3);

                for (std::size_t c = 0; c < channels.size(); ++c)
                {
                    const auto &ch = channels[c];
                    const auto wv0 = ch.wv0.segment(p0, n).asDiagonal();
                    A = wv0 * phi;
                    if (gga)
                        for (int k = 0; k < 3; ++k)
                            A.noalias() += ch.wvg.col(k).segment(p0, n).asDiagonal() * g[k];
                    AP.noalias() = A * ch.P;
                    if (gga)
                        phiP.noalias() = phi * ch.P;

                    for (int q = 0; q < 3; ++q)
                    {
                        M[c][q].noalias() += g[q].transpose() * A;
                        owner_t.col(q) += g[q].cwiseProduct(AP).rowwise().sum();
                        if (!gga)
                            continue;
                        Aq = wv0 * g[q];
                        for (int k = 0; k < 3; ++k)
                            Aq.noalias() += ch.wvg.col(k).segment(p0, n).asDiagonal() * h[hidx[q][k]];
                        M[c][q].noalias() += Aq.transpose() * phi;
                        owner_t.col(q) += Aq.cwiseProduct(phiP).rowwise().sum();
                    }
                }

                for (Eigen::Index i = 0; i < n; ++i)
                {
                    const Eigen::Index ip = p0 + i;
                    if (grid.points(ip, 3) == 0.0)
                        continue;
                    const Eigen::Index owner = grid.owner(ip);
                    grad.row(owner) += 2.0 * owner_t.row(i);

                    auto dpartition = becke_partition_owner_derivatives(grid, mol, ip);
                    if (!dpartition)
                        return std::unexpected(dpartition.error());
                    grad += (grid.atomic_weights(ip) * xc.energy_density(ip)) * (*dpartition);
                }
            }

            // Basis-function-motion term: vxc1_q = -M_q, contracted over the
            // rows of the atom that carries each basis function.
            for (std::size_t atom_A = 0; atom_A < mol.natoms; ++atom_A)
                for (int mu : (*atoms_bf)[atom_A])
                    for (std::size_t c = 0; c < channels.size(); ++c)
                        for (int q = 0; q < 3; ++q)
                            grad(static_cast<Eigen::Index>(atom_A), q) -=
                                2.0 * M[c][q].row(mu).dot(channels[c].P.row(mu));

            return grad;
        }

        // Zero the weights of points the quadrature drops, so a non-finite XC
        // value there cannot reach the block GEMMs (the old loop skipped them).
        void mask_zero_weight(const Eigen::VectorXd &w, Eigen::VectorXd &wv0, Eigen::MatrixXd &wvg)
        {
            for (Eigen::Index p = 0; p < w.size(); ++p)
                if (w(p) == 0.0)
                {
                    wv0(p) = 0.0;
                    if (wvg.size() != 0)
                        wvg.row(p).setZero();
                }
        }
    } // namespace

    std::expected<Eigen::MatrixXd, std::string>
    compute_xc_nuclear_gradient_rks(
        const HartreeFock::Molecule &mol,
        const HartreeFock::Basis &basis,
        const MolecularGrid &grid,
        const AOGridEvaluation &ao,
        const XCGridEvaluation &xc,
        const Eigen::Ref<const Eigen::MatrixXd> &density_restricted)
    {
        if (xc.density.polarized)
            return std::unexpected("compute_xc_nuclear_gradient_rks requires spin-unpolarized XC density");

        if (grid.points.rows() != ao.npoints())
            return std::unexpected("molecular grid point count mismatch in XC gradient");

        const Eigen::Index nb = ao.nbasis();
        if (density_restricted.rows() != nb || density_restricted.cols() != nb)
            return std::unexpected("restricted density dimension mismatch for XC gradient");

        const bool gga = xc.vsigma.cols() >= 1;
        const Eigen::VectorXd w = grid.points.col(3);

        XCGradientChannel ch;
        ch.P = 0.5 * (density_restricted + density_restricted.transpose());
        // LDA's term is g_q aow^T alone; GGA splits vrho over two terms.
        ch.wv0 = w.cwiseProduct(xc.vrho.col(0)) * (gga ? 0.5 : 1.0);
        if (gga)
        {
            const Eigen::VectorXd s = 2.0 * w.cwiseProduct(xc.vsigma.col(0));
            ch.wvg.resize(w.size(), 3);
            ch.wvg.col(0) = s.cwiseProduct(xc.density.total.grad_x);
            ch.wvg.col(1) = s.cwiseProduct(xc.density.total.grad_y);
            ch.wvg.col(2) = s.cwiseProduct(xc.density.total.grad_z);
        }
        mask_zero_weight(w, ch.wv0, ch.wvg);

        return xc_nuclear_gradient(mol, basis, grid, ao, xc, {std::move(ch)}, gga);
    }

    std::expected<Eigen::MatrixXd, std::string>
    compute_xc_nuclear_gradient_uks(
        const HartreeFock::Molecule &mol,
        const HartreeFock::Basis &basis,
        const MolecularGrid &grid,
        const AOGridEvaluation &ao,
        const XCGridEvaluation &xc,
        const Eigen::Ref<const Eigen::MatrixXd> &density_alpha,
        const Eigen::Ref<const Eigen::MatrixXd> &density_beta)
    {
        if (!xc.density.polarized)
            return std::unexpected("compute_xc_nuclear_gradient_uks requires polarized XC density");

        if (xc.vrho.cols() != 2)
            return std::unexpected("polarized XC gradient expects vrho with two columns");

        const Eigen::Index nb = ao.nbasis();
        if (density_alpha.rows() != nb || density_beta.rows() != nb)
            return std::unexpected("UKS density dimension mismatch for XC gradient");

        const bool gga = xc.vsigma.cols() == 3;
        const Eigen::VectorXd w = grid.points.col(3);

        std::vector<XCGradientChannel> channels(2);
        channels[0].P = 0.5 * (density_alpha + density_alpha.transpose());
        channels[1].P = 0.5 * (density_beta + density_beta.transpose());
        for (int s = 0; s < 2; ++s)
            channels[s].wv0 = w.cwiseProduct(xc.vrho.col(s)) * (gga ? 0.5 : 1.0);
        if (gga)
        {
            // coeff_a = 2 vs_aa grad_a + vs_ab grad_b; coeff_b = vs_ab grad_a + 2 vs_bb grad_b.
            const auto &da = xc.density.alpha;
            const auto &db = xc.density.beta;
            const std::array<const Eigen::VectorXd *, 3> ga = {&da.grad_x, &da.grad_y, &da.grad_z};
            const std::array<const Eigen::VectorXd *, 3> gb = {&db.grad_x, &db.grad_y, &db.grad_z};
            const Eigen::VectorXd aa = 2.0 * w.cwiseProduct(xc.vsigma.col(0));
            const Eigen::VectorXd ab = w.cwiseProduct(xc.vsigma.col(1));
            const Eigen::VectorXd bb = 2.0 * w.cwiseProduct(xc.vsigma.col(2));
            channels[0].wvg.resize(w.size(), 3);
            channels[1].wvg.resize(w.size(), 3);
            for (int k = 0; k < 3; ++k)
            {
                channels[0].wvg.col(k) = aa.cwiseProduct(*ga[k]) + ab.cwiseProduct(*gb[k]);
                channels[1].wvg.col(k) = ab.cwiseProduct(*ga[k]) + bb.cwiseProduct(*gb[k]);
            }
        }
        for (auto &ch : channels)
            mask_zero_weight(w, ch.wv0, ch.wvg);

        return xc_nuclear_gradient(mol, basis, grid, ao, xc, channels, gga);
    }

} // namespace DFT::Gradient
